"""Live capture: USB camera frames + ESP32 serial IMU packets -> shared session format.

Writes ``meta.json``, ``frames.csv`` (idx,t_ns,laser_on,file), and camera frames as original
MJPEG ``.jpg`` packets when FFmpeg is used (otherwise lossless grayscale ``.png``), plus
when a serial port is given, ``imu.csv`` (t_ns,qw,qx,qy,qz). With serial capture every timestamp
comes from the MCU clock; the host clock is never mixed in.
"""
from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import queue
from pathlib import Path
import threading
import time
from typing import Callable

import cv2
import numpy as np

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.capture.camera import open_usb_camera, to_gray_u8
from ioscan.pipeline.ridge import RidgeConfig, extract_ridge_subpixel
from ioscan.pipeline.subtraction import classify_laser_frame, subtract_ambient

FORMAT_VERSION = 1


class _AsyncFrameWriter:
    """Persist camera frames off the acquisition thread.

    FFmpeg/MJPG frames are written as their original JPEG byte payloads; other
    capture backends use lossless grayscale PNG. Each frame has a unique file
    name, and the bounded queue fails loudly rather than silently losing frames.
    """

    _STOP = object()

    def __init__(
        self,
        root: Path,
        max_queue: int = 128,
        png_compression: int = 1,
        workers: int = 4,
    ):
        self.root = Path(root)
        self.png_compression = int(np.clip(png_compression, 0, 9))
        self.worker_count = max(1, int(workers))
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(max_queue)))
        self._error: Exception | None = None
        self._error_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._closed = False
        self._frames_written = 0
        self._stored_bytes_written = 0
        self._uncompressed_bytes_written = 0
        self._jpeg_frames_written = 0
        self._png_frames_written = 0
        self._write_seconds = 0.0
        self._max_queue_depth = 0
        self._threads = [
            threading.Thread(target=self._run, name=f"ioscan-frame-writer-{i}", daemon=True)
            for i in range(self.worker_count)
        ]
        for thread in self._threads:
            thread.start()

    def _set_error(self, exc: Exception) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                rel, payload, encoding = item
                with self._error_lock:
                    error = self._error
                if error is not None:
                    continue
                path = self.root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                if encoding == "jpeg":
                    # The FFmpeg camera path supplies the original MJPEG packet;
                    # persist it byte-for-byte without decoding/re-encoding.
                    with path.open("wb") as stream:
                        stream.write(payload)
                    stored_bytes = len(payload)
                    uncompressed_bytes = 0  # source JPEG dimensions are decoded by capture path
                    jpeg_count, png_count = 1, 0
                else:
                    ok = cv2.imwrite(
                        str(path), payload, [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression]
                    )
                    if not ok:
                        raise IOError(f"Failed to write {rel}")
                    stored_bytes = path.stat().st_size if path.is_file() else 0
                    uncompressed_bytes = int(payload.nbytes)
                    jpeg_count, png_count = 0, 1
                elapsed = time.perf_counter() - started
                with self._stats_lock:
                    self._frames_written += 1
                    self._stored_bytes_written += stored_bytes
                    self._uncompressed_bytes_written += uncompressed_bytes
                    self._jpeg_frames_written += jpeg_count
                    self._png_frames_written += png_count
                    self._write_seconds += elapsed
            except Exception as exc:  # keep draining so close() cannot deadlock
                self._set_error(exc)
            finally:
                self._queue.task_done()

    def submit(self, relative_path: str, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Frame writer is already closed")
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Asynchronous frame writer failed: {error}") from error
        try:
            self._queue.put_nowait((str(relative_path), np.asarray(frame).copy(), "png"))
            with self._stats_lock:
                self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
        except queue.Full as exc:
            raise RuntimeError(
                f"Frame storage queue is full ({self._queue.maxsize} pending frames): "
                "parallel PNG encoding/disk writes cannot sustain the camera rate. "
                "Capture stopped to avoid silently dropping or mis-pairing frames. "
                "Save the session to a local SSD outside OneDrive/Dropbox, close other disk-heavy apps, "
                "and check free space and real-time antivirus scanning."
            ) from exc

    def submit_jpeg(self, relative_path: str, jpeg_bytes: bytes) -> None:
        """Queue a complete source MJPEG packet for direct .jpg storage."""
        if self._closed:
            raise RuntimeError("Frame writer is already closed")
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Asynchronous frame writer failed: {error}") from error
        if not isinstance(jpeg_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("jpeg_bytes must be bytes-like")
        payload = bytes(jpeg_bytes)
        if not (payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")):
            raise ValueError("MJPEG frame is not a complete JPEG image")
        try:
            self._queue.put_nowait((str(relative_path), payload, "jpeg"))
            with self._stats_lock:
                self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
        except queue.Full as exc:
            raise RuntimeError(
                f"Frame storage queue is full ({self._queue.maxsize} pending frames): "
                "direct MJPEG file writes cannot sustain the camera rate. "
                "Capture stopped to avoid silently dropping or mis-pairing frames. "
                "Save the session to a local SSD outside cloud sync and check disk activity."
            ) from exc

    def stats(self) -> dict:
        with self._stats_lock:
            return {
                "worker_count": self.worker_count,
                "queue_capacity_frames": self._queue.maxsize,
                "max_queue_depth_frames": self._max_queue_depth,
                "frames_written": self._frames_written,
                "stored_bytes_written": self._stored_bytes_written,
                "jpeg_frames_written": self._jpeg_frames_written,
                "png_frames_written": self._png_frames_written,
                "uncompressed_png_input_bytes_written": self._uncompressed_bytes_written,
                "png_compression": self.png_compression,
                "storage_mode": "mixed: source MJPEG JPEG packets and PNG fallback",
                "cumulative_write_seconds": round(self._write_seconds, 3),
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._threads:
            self._queue.put(self._STOP)
        self._queue.join()
        for thread in self._threads:
            thread.join(timeout=10.0)
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("Frame writer workers did not stop after pending frames were flushed")
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Asynchronous frame writer failed: {error}") from error


@dataclass
class CaptureConfig:
    camera_id: int = 0
    camera_backend: str = "Auto"
    camera_name: str | None = None
    width: int = 1600
    height: int = 1200                # Arducam OV2311: 1600x1200
    fps: int = 50
    exposure: int = -7                # manual UVC exposure (~1.5 ms on Windows DirectShow)
    laser_baseline_mm: float = 18.0   # provisional measured camera/laser setup offset; calibrate before dimensional use
    laser_tilt_deg: float = 10.0
    laser_fan_half_angle_deg: float = 25.0  # recorded setup parameter; not used by single-plane triangulation
    serial_port: str | None = None    # e.g., "COM3" on Windows
    baudrate: int = 115200
    max_frames: int = 300
    show_preview: bool = False
    imu_wait_s: float = 5.0           # maximum wait for a valid ESP32 frame packet
    serial_packet_timeout_s: float = 1.0  # maximum gap between synchronized strobe packets
    live_preview_profile_stride: int = 2  # decode every Nth matched laser-ON/OFF pair during capture


def _crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly=0x1021, init=0xFFFF, xorout=0)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


class SerialIMUReader:
    """Read one ESP32 strobe packet per camera exposure.

    Preferred protocol: ``@timestamp_us,laser_on,qw,qx,qy,qz,frame_seq*CRC16``.
    CRC-16/CCITT-FALSE detects serial corruption; '@' provides a resynchronization
    marker if stray bytes precede a packet. Legacy unframed CSV remains accepted for
    older firmware, but malformed/corrupt lines are discarded and recorded. Packets
    are queued in order and sequence gaps still fail capture to prevent frame mispairing.
    """

    _MICROS_MODULUS = 1 << 32
    _MAX_QUEUE = 256  # short jitter buffer; must not mask a sustained slow capture loop

    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._latest: tuple[int, bool, np.ndarray] | None = None
        self._events: deque[tuple[int, bool, np.ndarray, int | None]] = deque()
        self._packets: list[tuple[int, np.ndarray]] = []
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None
        self.warnings: list[str] = []
        self._last_raw_us: int | None = None
        self._wrap_offset_us = 0
        self._last_frame_seq: int | None = None
        self._max_event_queue_depth = 0
        self._events_received = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _set_error(self, exc: Exception) -> None:
        with self._condition:
            if self.error is None:
                self.error = exc
            self._condition.notify_all()

    def _record_warning(self, message: str) -> None:
        with self._condition:
            self.warnings.append(message)
            # Keep diagnostics bounded during long sessions.
            if len(self.warnings) > 100:
                del self.warnings[:-100]
            self._condition.notify_all()

    def _extend_timestamp(self, raw_us: int) -> int:
        if not 0 <= raw_us < self._MICROS_MODULUS:
            raise ValueError(f"ESP32 timestamp {raw_us} is outside the uint32 micros() range")
        if self._last_raw_us is not None and raw_us < self._last_raw_us:
            backwards = self._last_raw_us - raw_us
            if backwards > (self._MICROS_MODULUS // 2):
                self._wrap_offset_us += self._MICROS_MODULUS
            else:
                raise ValueError("ESP32 timestamps moved backwards without a micros() counter wrap")
        self._last_raw_us = raw_us
        return raw_us + self._wrap_offset_us

    def _run(self) -> None:
        try:
            import serial
        except ImportError as exc:
            self._set_error(RuntimeError("Install pyserial (`pip install -e '.[hardware]'`) to read ESP32 data."))
            return
        try:
            with serial.Serial(self.port, self.baudrate, timeout=0.05) as ser:
                # Discard bytes buffered before the host opened the port; those frame events
                # cannot safely be paired with camera frames read after capture startup.
                reset_input = getattr(ser, "reset_input_buffer", None)
                if callable(reset_input):
                    reset_input()
                while not self._stop.is_set():
                    raw_line = ser.readline()
                    if not raw_line:
                        continue
                    raw_line = raw_line.strip(b"\r\n")
                    if not raw_line:
                        continue

                    # New firmware frames begin with '@'. A damaged UART byte can
                    # erase a newline, causing Serial Monitor/PySerial to return two
                    # or more frames in one readline(). Split at EVERY start marker
                    # and validate each candidate independently. The former parser
                    # used only the first '@' and the last '*', which discarded a
                    # later intact packet whenever a preceding packet was truncated.
                    # This improves resynchronization; CRC and strict sequence checks
                    # remain mandatory, so corrupt/missing events are never guessed.
                    marker_positions = [i for i, byte in enumerate(raw_line) if byte == ord("@")] 
                    if marker_positions:
                        if marker_positions[0] > 0:
                            self._record_warning(
                                f"Discarded {marker_positions[0]} stray serial byte(s) before packet marker"
                            )
                        candidates = []
                        for idx, marker_pos in enumerate(marker_positions):
                            end_pos = marker_positions[idx + 1] if idx + 1 < len(marker_positions) else len(raw_line)
                            packet = raw_line[marker_pos + 1:end_pos].strip(b"\r\n")
                            payload, separator, crc_text = packet.rpartition(b"*")
                            if not separator or len(crc_text) != 4:
                                self._record_warning("Discarded malformed framed ESP32 packet (missing CRC)")
                                continue
                            try:
                                received_crc = int(crc_text, 16)
                            except ValueError:
                                self._record_warning("Discarded ESP32 packet with malformed CRC field")
                                continue
                            if _crc16_ccitt(payload) != received_crc:
                                self._record_warning("Discarded ESP32 packet with CRC mismatch")
                                continue
                            candidates.append(payload)
                    else:
                        candidates = [raw_line]

                    for data in candidates:
                        if b"\x00" in data or any(byte < 32 or byte > 126 for byte in data):
                            self._record_warning("Discarded corrupted legacy ESP32 serial line containing non-ASCII/NUL bytes")
                            continue

                        try:
                            line = data.decode("ascii").strip()
                        except UnicodeDecodeError:
                            self._record_warning("Discarded non-ASCII ESP32 serial line")
                            continue
                        if not line:
                            continue
                        if line.startswith("# ERROR:"):
                            self._set_error(RuntimeError(line.partition(":")[2].strip()))
                            return
                        if line.startswith("# WARN:"):
                            self._record_warning(line.partition(":")[2].strip())
                            continue
                        if line.startswith("#"):
                            continue

                        parts = line.split(",")
                        if len(parts) not in {6, 7}:
                            self._record_warning(f"Discarded ESP32 CSV packet with {len(parts)} fields (expected 6 or 7)")
                            continue
                        try:
                            raw_us = int(parts[0].strip())
                            if parts[1].strip() not in {"0", "1"}:
                                raise ValueError("laser_on must be 0 or 1")
                            lon = parts[1].strip() == "1"
                            q = np.array([float(x) for x in parts[2:6]], dtype=np.float64)
                            frame_seq = int(parts[6].strip()) if len(parts) == 7 and parts[6].strip() else None
                            t_us = self._extend_timestamp(raw_us)
                        except ValueError as exc:
                            self._record_warning(f"Discarded invalid ESP32 packet {line!r}: {exc}")
                            continue

                        n = np.linalg.norm(q)
                        if not np.isfinite(n) or n < 1e-9 or not np.isfinite(q).all():
                            self._set_error(RuntimeError("ESP32 transmitted an invalid IMU quaternion"))
                            return
                        q = q / n

                        with self._condition:
                            if frame_seq is not None and self._last_frame_seq is not None:
                                expected = (self._last_frame_seq + 1) & 0xFFFFFFFF
                                if frame_seq != expected:
                                    self.error = RuntimeError(
                                        f"ESP32 strobe sequence gap: expected frame {expected}, received {frame_seq}. "
                                        "Camera/serial capture may be dropping synchronized frames."
                                    )
                                    self._condition.notify_all()
                                    return
                            if frame_seq is not None:
                                self._last_frame_seq = frame_seq
                            if len(self._events) >= self._MAX_QUEUE:
                                self.error = RuntimeError(
                                    f"Host ESP32 strobe queue overflow: backlog reached {self._MAX_QUEUE} packets. "
                                    "The PC capture loop is consuming camera frames more slowly than the "
                                    "ESP32 produces strobe events. This is a host-side backlog, not proof "
                                    "of ESP32 interrupt-ring overflow."
                                )
                                self._condition.notify_all()
                                return
                            event = (t_us, lon, q.copy(), frame_seq)
                            self._events.append(event)
                            self._events_received += 1
                            self._max_event_queue_depth = max(
                                self._max_event_queue_depth, len(self._events)
                            )
                            self._latest = (t_us, lon, q.copy())
                            self._packets.append((t_us, q.copy()))
                            self._condition.notify_all()
        except Exception as exc:
            self._set_error(exc)

    def latest(self) -> tuple[int, bool, np.ndarray] | None:
        """Return the latest packet for diagnostics only; capture must use get_next()."""
        with self._condition:
            return self._latest

    def get_next(self, timeout: float = 1.0) -> tuple[int, bool, np.ndarray, int | None]:
        """Consume exactly one new strobe event, preserving packet/frame order."""
        deadline = time.monotonic() + max(float(timeout), 0.0)
        with self._condition:
            while not self._events:
                if self.error is not None:
                    raise RuntimeError(f"ESP32 serial reader failed: {self.error}") from self.error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f" Latest firmware warning: {self.warnings[-1]}" if self.warnings else ""
                    raise TimeoutError(
                        f"No new ESP32 camera-strobe packet received within {timeout:.2f} s.{detail}"
                    )
                self._condition.wait(remaining)
            if self.error is not None:
                raise RuntimeError(f"ESP32 serial reader failed: {self.error}") from self.error
            return self._events.popleft()

    def packets(self) -> list[tuple[int, np.ndarray]]:
        with self._condition:
            return [(t, q.copy()) for t, q in self._packets]

    def stats(self) -> dict:
        with self._condition:
            return {
                "events_received": self._events_received,
                "max_event_queue_depth": self._max_event_queue_depth,
                "current_event_queue_depth": len(self._events),
                "event_queue_capacity": self._MAX_QUEUE,
            }

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _build_calibration(cfg: CaptureConfig, camera_cal: dict | None, laser_cal: dict | None):
    nominal = camera_cal is None or laser_cal is None
    if camera_cal:
        cam = CameraIntrinsics.from_dict(camera_cal)
    else:
        # Use the project's explicit nominal camera model instead of the old hard-coded
        # 8 mm lens assumption. The checked-in 2.8 mm estimate is still provisional;
        # a measured checkerboard calibration JSON takes precedence whenever supplied.
        camera_cfg = Path(__file__).resolve().parents[3] / "configs" / "camera.yaml"
        if camera_cfg.is_file():
            cam = CameraIntrinsics.from_config(camera_cfg)
            if (cam.width, cam.height) != (cfg.width, cfg.height):
                raise ValueError(
                    f"Camera calibration config is {cam.width}x{cam.height}, "
                    f"but capture is {cfg.width}x{cfg.height}."
                )
        else:
            # Safe provisional fallback for packaged runs where configs/ is not shipped.
            fx_nominal = 2.8 / (3.0e-3)  # 2.8 mm / 3 µm per pixel = 933.3 px
            cam = CameraIntrinsics(
                width=cfg.width, height=cfg.height, fx=fx_nominal, fy=fx_nominal,
                cx=cfg.width / 2.0, cy=cfg.height / 2.0,
            )
    # Nominal plane uses the simulator convention (laser apex below camera along +y).
    laser = LaserPlane.from_dict(laser_cal) if laser_cal else LaserPlane.from_baseline_and_angle(
        cfg.laser_baseline_mm, cfg.laser_tilt_deg
    )
    return cam, laser, nominal


def record_hardware_session(
    out_dir: str | Path,
    cfg: CaptureConfig | None = None,
    camera_cal_dict: dict | None = None,
    laser_cal_dict: dict | None = None,
    frame_callback: Callable[[np.ndarray, int, bool, int, np.ndarray | None], None] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
) -> Path:
    """Capture camera frames paired in order with ESP32 exposure/strobe packets."""
    cfg = cfg or CaptureConfig()
    out = Path(out_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cam_model, laser_model, nominal = _build_calibration(cfg, camera_cal_dict, laser_cal_dict)
    if (cam_model.width, cam_model.height) != (cfg.width, cfg.height):
        raise ValueError(
            f"Calibration is for {cam_model.width}x{cam_model.height} but capture is "
            f"{cfg.width}x{cfg.height}; set CaptureConfig width/height to match."
        )

    # High-speed acquisition is not allowed to start in the OV2311's 1600x1200
    # YUY2 mode (5 FPS). Verify a real MJPG stream and its measured delivery rate.
    cap, _unused_warmup_frame, camera_open_info = open_usb_camera(
        cfg.camera_id, cfg.width, cfg.height, cfg.fps, exposure=cfg.exposure,
        read_attempts=16, backend_name=cfg.camera_backend,
        required_fourcc="MJPG", min_fps=max(1.0, float(cfg.fps) * 0.85),
        camera_name=cfg.camera_name,
        prefer_ffmpeg=(os.name == "nt" and cfg.camera_backend == "DirectShow"),
    )
    camera_set_results = camera_open_info.get("set_results", {})
    frame_writer = _AsyncFrameWriter(out, max_queue=128, png_compression=1, workers=4)
    imu_reader: SerialIMUReader | None = None
    rows: list[tuple[int, int, int, str]] = []
    actual_camera: dict = {}
    try:
        if cfg.serial_port:
            imu_reader = SerialIMUReader(cfg.serial_port, cfg.baudrate)
            imu_reader.start()

        last_off: np.ndarray | None = None
        # In the FFmpeg path keep OFF-frame JPEG bytes rather than decoding every
        # 1600x1200 image in the synchronized acquisition loop. For live preview,
        # decode only a sampled, matched OFF/ON pair. All JPEGs are still saved and
        # the final offline reconstruction processes every captured frame.
        last_off_jpeg: tuple[bytes, int, int, np.ndarray | None] | None = None
        live_on_profiles = 0
        preview_stride = max(1, int(cfg.live_preview_profile_stride))
        ridge_cfg = RidgeConfig(method="cog", saturation_limit=254.0)  # 8-bit frames
        t0 = time.perf_counter()
        last_packet: tuple[int, bool, np.ndarray, int | None] | None = None

        def decode_jpeg_gray(jpeg: bytes, frame_index: int) -> np.ndarray:
            decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if decoded is None or decoded.size == 0:
                raise RuntimeError(f"Could not decode source MJPEG frame {frame_index}.")
            gray_image = to_gray_u8(decoded)
            if gray_image.shape != (cfg.height, cfg.width):
                raise RuntimeError(
                    f"Decoded camera frame {frame_index} has shape {gray_image.shape}; "
                    f"expected {(cfg.height, cfg.width)}."
                )
            return gray_image

        try:
            for idx in range(cfg.max_frames):
                if stop_event is not None and stop_event.is_set():
                    break

                # Each event is consumed in FIFO order and paired with the next
                # camera frame. Keep this critical path limited to packet/frame
                # acquisition, direct JPEG enqueue and manifest bookkeeping.
                if imu_reader is not None:
                    if last_packet is None:
                        try:
                            last_packet = imu_reader.get_next(timeout=cfg.imu_wait_s if idx == 0 else cfg.serial_packet_timeout_s)
                        except TimeoutError as exc:
                            raise RuntimeError(
                                f"No synchronized ESP32 strobe packet on {cfg.serial_port}. "
                                "Check the camera STROBE wire, BNO086 and firmware. " + str(exc)
                            ) from exc
                    t_us, laser_on, q, frame_seq = last_packet
                    last_packet = None
                    t_ns = int(t_us) * 1000
                else:
                    t_ns = int((time.perf_counter() - t0) * 1e9)
                    laser_on = False
                    q = None
                    frame_seq = None

                source_jpeg = None
                is_ffmpeg = camera_open_info.get("capture_engine") == "ffmpeg" and hasattr(cap, "read_encoded")
                if is_ffmpeg:
                    ok, source_jpeg = cap.read_encoded()
                    if not ok or source_jpeg is None:
                        raise RuntimeError(f"Camera stopped delivering frames at frame {idx}.")
                    rel = f"frames/{idx:06d}.jpg"
                    frame_writer.submit_jpeg(rel, source_jpeg)
                    rows.append((idx, t_ns, int(laser_on), rel))

                    # When there is no ESP32 event stream, image analysis is needed
                    # on every frame to infer laser state. Local debug preview also
                    # opts into per-frame decoding. The production dashboard path
                    # instead samples matched pairs to keep acquisition near 50 FPS.
                    if imu_reader is None or cfg.show_preview:
                        gray = decode_jpeg_gray(source_jpeg, idx)
                        if imu_reader is None:
                            laser_on = classify_laser_frame(gray, p999_threshold=100.0)
                            rows[-1] = (idx, t_ns, int(laser_on), rel)
                        if not laser_on:
                            last_off = gray.copy()
                        if frame_callback is not None and imu_reader is None:
                            frame_callback(gray, idx, bool(laser_on), int(t_ns), None if q is None else q.copy())
                    else:
                        gray = None
                        if not laser_on:
                            last_off_jpeg = (source_jpeg, idx, int(t_ns), q.copy() if q is not None else None)
                        else:
                            live_on_profiles += 1
                            should_preview_pair = (
                                frame_callback is not None
                                and last_off_jpeg is not None
                                and live_on_profiles % preview_stride == 0
                            )
                            if should_preview_pair:
                                off_jpeg, off_idx, off_t_ns, off_q = last_off_jpeg
                                if idx - off_idx <= 2:
                                    off_gray = decode_jpeg_gray(off_jpeg, off_idx)
                                    on_gray = decode_jpeg_gray(source_jpeg, idx)
                                    frame_callback(
                                        off_gray, off_idx, False, off_t_ns,
                                        None if off_q is None else off_q.copy(),
                                    )
                                    frame_callback(
                                        on_gray, idx, True, int(t_ns), None if q is None else q.copy(),
                                    )
                                    gray = on_gray

                    # If the developer enabled OpenCV's local preview, `gray` is
                    # decoded on every frame above. The browser preview is fed from
                    # the sampled pair callbacks and does not block acquisition.
                    if cfg.show_preview and gray is not None:
                        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        if laser_on and last_off is not None:
                            diff = subtract_ambient(gray, last_off)
                            u, v, _ = extract_ridge_subpixel(diff, cfg=ridge_cfg)
                            for u_px, v_px in zip(u[::2], v[::2]):
                                cv2.circle(vis, (int(round(u_px)), int(round(v_px))), 1, (0, 255, 0), -1)
                        status = f"Frame {idx}/{cfg.max_frames} | Laser: {'ON' if laser_on else 'OFF'}"
                        cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        small = cv2.resize(vis, (960, int(960 * gray.shape[0] / gray.shape[1])))
                        cv2.imshow("Intraoral Scanner Live Capture (Press Q to stop)", small)
                        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                            break
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"Camera stopped delivering frames at frame {idx}.")
                    gray = to_gray_u8(frame)
                    if gray.shape != (cfg.height, cfg.width):
                        raise RuntimeError(
                            f"Camera delivered {gray.shape[::-1]}, expected {(cfg.width, cfg.height)}. "
                            "Check the UVC resolution/pixel-format settings and calibration dimensions."
                        )
                    if imu_reader is None:
                        laser_on = classify_laser_frame(gray, p999_threshold=100.0)
                    if not laser_on:
                        last_off = gray.copy()
                    rel = f"frames/{idx:06d}.png"
                    frame_writer.submit(rel, gray)
                    rows.append((idx, t_ns, int(laser_on), rel))
                    if frame_callback is not None:
                        frame_callback(gray, idx, bool(laser_on), int(t_ns), None if q is None else q.copy())

                    if cfg.show_preview:
                        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        if laser_on and last_off is not None:
                            diff = subtract_ambient(gray, last_off)
                            u, v, _ = extract_ridge_subpixel(diff, cfg=ridge_cfg)
                            for u_px, v_px in zip(u[::2], v[::2]):
                                cv2.circle(vis, (int(round(u_px)), int(round(v_px))), 1, (0, 255, 0), -1)
                        status = f"Frame {idx}/{cfg.max_frames} | Laser: {'ON' if laser_on else 'OFF'}"
                        cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        small = cv2.resize(vis, (960, int(960 * gray.shape[0] / gray.shape[1])))
                        cv2.imshow("Intraoral Scanner Live Capture (Press Q to stop)", small)
                        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                            break

                if progress_callback is not None:
                    progress_callback({
                        "stage": "capturing", "frame": idx + 1, "frames_total": cfg.max_frames,
                        "laser_on": bool(laser_on), "progress": 100.0 * (idx + 1) / max(cfg.max_frames, 1),
                        "strobe_sequence": frame_seq,
                    })
        finally:
            actual_camera = {
                "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
                "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                "driver_reported_fps": (
                    None if camera_open_info.get("capture_engine") == "ffmpeg"
                    else float(cap.get(cv2.CAP_PROP_FPS))
                ),
                "exposure": float(cap.get(cv2.CAP_PROP_EXPOSURE)),
                "set_results": camera_set_results,
            }
            cap.release()
            if cfg.show_preview:
                cv2.destroyAllWindows()
            if imu_reader:
                imu_reader.stop()
            # Flush all queued frames before writing the manifest/metadata.
            frame_writer.close()

        if not rows:
            raise RuntimeError("No frames were captured")

        with open(out / "frames.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "t_ns", "laser_on", "file"])
            w.writerows(rows)

        has_imu = False
        serial_warnings: list[str] = []
        if imu_reader is not None:
            pk = imu_reader.packets()
            serial_warnings = list(imu_reader.warnings)
            if pk:
                has_imu = True
                arr = np.column_stack([
                    np.array([int(t) * 1000 for t, _ in pk], dtype=np.int64),
                    np.array([q for _, q in pk]),
                ])
                with open(out / "imu.csv", "w", encoding="utf-8") as f:
                    f.write("t_ns,qw,qx,qy,qz\n")
                    for r in arr:
                        f.write(f"{int(r[0])},{r[1]:.9f},{r[2]:.9f},{r[3]:.9f},{r[4]:.9f}\n")

        elapsed_s = (int(rows[-1][1]) - int(rows[0][1])) / 1e9 if len(rows) > 1 else 0.0
        measured_session_fps = (len(rows) - 1) / elapsed_s if elapsed_s > 0 else 0.0
        frame_intervals_ms = [
            (int(b[1]) - int(a[1])) / 1e6 for a, b in zip(rows, rows[1:])
            if int(b[1]) > int(a[1])
        ]
        meta = {
            "format_version": FORMAT_VERSION,
            "source": "hardware",
            "created": datetime.now(timezone.utc).isoformat(),
            "requested_fps": float(cfg.fps),
            "fps": round(measured_session_fps, 3),
            "median_frame_interval_ms": round(float(np.median(frame_intervals_ms)), 3) if frame_intervals_ms else None,
            "num_frames": len(rows),
            "image": {"width": cfg.width, "height": cfg.height, "dtype": "uint8", "bit_depth": 8,
                      "storage": "source_mjpeg_jpeg" if camera_open_info.get("capture_engine") == "ffmpeg" else "lossless_png"},
            "camera": cam_model.to_dict(),
            "laser_plane": laser_model.to_dict(),
            "camera_runtime": {
                **camera_open_info, **actual_camera,
                "measured_session_fps": round(measured_session_fps, 3),
                "camera_name": cfg.camera_name, "selected_backend": cfg.camera_backend,
            },
            "calibration_is_nominal": nominal,
            "has_imu": has_imu,
            "clock": "mcu" if has_imu else "host",
            "serial_warnings": serial_warnings,
            "serial_reader": (imu_reader.stats() if hasattr(imu_reader, "stats") else None) if imu_reader is not None else None,
            "frame_storage": frame_writer.stats(),
            "capture_complete": len(rows) >= int(cfg.max_frames),
            "config": {"capture": {"camera_id": cfg.camera_id, "camera_backend": cfg.camera_backend, "camera_name": cfg.camera_name, "exposure": cfg.exposure,
                                    "serial_port": cfg.serial_port, "baudrate": cfg.baudrate,
                                    "laser_baseline_mm": cfg.laser_baseline_mm, "laser_tilt_deg": cfg.laser_tilt_deg,
                                    "laser_fan_half_angle_deg": cfg.laser_fan_half_angle_deg,
                                    "live_preview_profile_stride": cfg.live_preview_profile_stride}},
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out
    except Exception as capture_exc:
        cap.release()
        writer_close_error = None
        try:
            frame_writer.close()
        except Exception as exc:
            writer_close_error = exc

        # Preserve all frames that actually reached disk when a scan aborts.
        # Earlier versions left PNGs orphaned because frames.csv/meta.json were
        # only written on successful completion, making a stopped scan opaque.
        persisted_rows = [row for row in rows if (out / row[3]).is_file()]
        if persisted_rows:
            with open(out / "frames.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["idx", "t_ns", "laser_on", "file"])
                writer.writerows(persisted_rows)

            serial_warnings = list(imu_reader.warnings) if imu_reader is not None else []
            if imu_reader is not None:
                packets = imu_reader.packets()
                if packets:
                    with open(out / "imu.csv", "w", encoding="utf-8") as f:
                        f.write("t_ns,qw,qx,qy,qz\n")
                        for timestamp_us, quat in packets:
                            f.write(
                                f"{int(timestamp_us) * 1000},{quat[0]:.9f},{quat[1]:.9f},"
                                f"{quat[2]:.9f},{quat[3]:.9f}\n"
                            )
            elapsed_s = (
                (int(persisted_rows[-1][1]) - int(persisted_rows[0][1])) / 1e9
                if len(persisted_rows) > 1 else 0.0
            )
            partial_fps = (len(persisted_rows) - 1) / elapsed_s if elapsed_s > 0 else 0.0
            partial_meta = {
                "format_version": FORMAT_VERSION,
                "source": "hardware",
                "created": datetime.now(timezone.utc).isoformat(),
                "requested_fps": float(cfg.fps),
                "fps": round(partial_fps, 3),
                "num_frames": len(persisted_rows),
                "image": {"width": cfg.width, "height": cfg.height, "dtype": "uint8", "bit_depth": 8,
                          "storage": "source_mjpeg_jpeg" if camera_open_info.get("capture_engine") == "ffmpeg" else "lossless_png"},
                "camera": cam_model.to_dict(),
                "laser_plane": laser_model.to_dict(),
                "camera_runtime": {
                    **camera_open_info, **actual_camera,
                    "measured_session_fps": round(partial_fps, 3),
                    "camera_name": cfg.camera_name, "selected_backend": cfg.camera_backend,
                },
                "calibration_is_nominal": nominal,
                "has_imu": bool(imu_reader is not None and (out / "imu.csv").is_file()),
                "clock": "mcu" if imu_reader is not None else "host",
                "serial_warnings": serial_warnings,
                "serial_reader": (imu_reader.stats() if hasattr(imu_reader, "stats") else None) if imu_reader is not None else None,
                "frame_storage": frame_writer.stats(),
                "capture_complete": False,
                "capture_error": str(capture_exc),
                "writer_close_error": str(writer_close_error) if writer_close_error else None,
                "config": {"capture": {
                    "camera_id": cfg.camera_id, "camera_backend": cfg.camera_backend,
                    "camera_name": cfg.camera_name, "exposure": cfg.exposure,
                    "serial_port": cfg.serial_port, "baudrate": cfg.baudrate,
                    "laser_baseline_mm": cfg.laser_baseline_mm, "laser_tilt_deg": cfg.laser_tilt_deg,
                    "laser_fan_half_angle_deg": cfg.laser_fan_half_angle_deg,
                }},
            }
            (out / "meta.json").write_text(json.dumps(partial_meta, indent=2), encoding="utf-8")

        if writer_close_error is not None and writer_close_error is not capture_exc:
            raise RuntimeError(
                f"Capture failed: {capture_exc}. Frame writer also reported: {writer_close_error}"
            ) from capture_exc
        raise

