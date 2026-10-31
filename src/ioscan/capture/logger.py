"""Live capture: USB camera frames + ESP32 serial IMU packets -> shared session format.

Writes ``meta.json``, ``frames.csv`` (idx,t_ns,laser_on,file), ``frames/*.png`` (8-bit mono) and,
when a serial port is given, ``imu.csv`` (t_ns,qw,qx,qy,qz). With serial capture every timestamp
comes from the MCU clock; the host clock is never mixed in.
"""
from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Callable

import cv2
import numpy as np

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.pipeline.ridge import RidgeConfig, extract_ridge_subpixel
from ioscan.pipeline.subtraction import classify_laser_frame, subtract_ambient

FORMAT_VERSION = 1


@dataclass
class CaptureConfig:
    camera_id: int = 0
    width: int = 1600
    height: int = 1200                # OV2311: 1600x1200
    fps: int = 50
    exposure: int = -7                # manual UVC exposure (~1.5 ms on Windows DirectShow)
    serial_port: str | None = None    # e.g., "COM3" on Windows
    baudrate: int = 921600
    max_frames: int = 300
    show_preview: bool = False
    imu_wait_s: float = 5.0           # maximum wait for a valid ESP32 frame packet
    serial_packet_timeout_s: float = 1.0  # maximum gap between synchronized strobe packets


class SerialIMUReader:
    """Read one ESP32 strobe packet per camera exposure.

    Protocol: ``timestamp_us,laser_on,qw,qx,qy,qz[,frame_seq]``.
    Packets are queued (not collapsed to the latest value), MCU ``micros()`` wrap is
    extended to a monotonic 64-bit timeline, and sequence gaps are reported as errors.
    Six-field packets remain accepted for backwards compatibility with the original sketch.
    """

    _MICROS_MODULUS = 1 << 32
    _MAX_QUEUE = 128

    def __init__(self, port: str, baudrate: int = 921600):
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

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _set_error(self, exc: Exception) -> None:
        with self._condition:
            if self.error is None:
                self.error = exc
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
                    line = ser.readline().decode("ascii", errors="ignore").strip()
                    if not line:
                        continue
                    if line.startswith("# ERROR:"):
                        self._set_error(RuntimeError(line.partition(":")[2].strip()))
                        return
                    if line.startswith("# WARN:"):
                        with self._condition:
                            self.warnings.append(line.partition(":")[2].strip())
                            self._condition.notify_all()
                        continue
                    if line.startswith("#"):
                        continue

                    parts = line.split(",")
                    if len(parts) < 6:
                        continue
                    try:
                        raw_us = int(parts[0])
                        if parts[1].strip() not in {"0", "1"}:
                            raise ValueError("laser_on must be 0 or 1")
                        lon = parts[1].strip() == "1"
                        q = np.array([float(x) for x in parts[2:6]], dtype=np.float64)
                        frame_seq = int(parts[6]) if len(parts) >= 7 and parts[6].strip() else None
                        t_us = self._extend_timestamp(raw_us)
                    except ValueError as exc:
                        self._set_error(RuntimeError(f"Invalid ESP32 packet {line!r}: {exc}"))
                        return

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
                                "ESP32 strobe queue overflow: camera capture is not keeping up with hardware frames."
                            )
                            self._condition.notify_all()
                            return
                        event = (t_us, lon, q.copy(), frame_seq)
                        self._events.append(event)
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

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _build_calibration(cfg: CaptureConfig, camera_cal: dict | None, laser_cal: dict | None):
    nominal = camera_cal is None or laser_cal is None
    cam = CameraIntrinsics.from_dict(camera_cal) if camera_cal else CameraIntrinsics(
        width=cfg.width, height=cfg.height, fx=2666.7, fy=2666.7, cx=cfg.width / 2.0, cy=cfg.height / 2.0
    )
    # Nominal plane uses the same convention/geometry as the simulator (laser apex +12 mm in +y).
    laser = LaserPlane.from_dict(laser_cal) if laser_cal else LaserPlane.from_baseline_and_angle(12.0, 30.963756532)
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

    # On Windows, use DirectShow consistently with the dashboard camera probe.
    backend = cv2.CAP_DSHOW if __import__("os").name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg.camera_id, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open USB camera index {cfg.camera_id}")

    camera_set_results = {}
    try:
        requested = {
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
            cv2.CAP_PROP_FRAME_WIDTH: cfg.width,
            cv2.CAP_PROP_FRAME_HEIGHT: cfg.height,
            cv2.CAP_PROP_FPS: cfg.fps,
            cv2.CAP_PROP_AUTO_EXPOSURE: 0.25,
            cv2.CAP_PROP_EXPOSURE: cfg.exposure,
            cv2.CAP_PROP_BUFFERSIZE: 1,
        }
        for prop, value in requested.items():
            try:
                camera_set_results[str(int(prop))] = bool(cap.set(prop, value))
            except Exception:
                camera_set_results[str(int(prop))] = False

        imu_reader: SerialIMUReader | None = None
        if cfg.serial_port:
            imu_reader = SerialIMUReader(cfg.serial_port, cfg.baudrate)
            imu_reader.start()

        rows: list[tuple[int, int, int, str]] = []
        actual_camera = {}
        last_off: np.ndarray | None = None
        ridge_cfg = RidgeConfig(method="cog", saturation_limit=254.0)  # 8-bit frames
        t0 = time.perf_counter()
        last_packet: tuple[int, bool, np.ndarray, int | None] | None = None

        try:
            for idx in range(cfg.max_frames):
                if stop_event is not None and stop_event.is_set():
                    break

                # Consume the next strobe event instead of reusing a potentially stale latest().
                # The firmware emits one event at exposure start; read the corresponding streamed
                # image immediately afterwards. CAP_PROP_BUFFERSIZE=1 reduces driver-side lag.
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
                    t_ns = int(t_us) * 1000  # extended MCU micros() -> ns
                else:
                    t_ns = int((time.perf_counter() - t0) * 1e9)  # host clock only
                    laser_on = False
                    q = None
                    frame_seq = None

                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Camera stopped delivering frames at frame {idx}.")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                if gray.ndim != 2:
                    raise RuntimeError(f"Camera returned an unsupported frame shape: {gray.shape}")
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
                if not cv2.imwrite(str(out / rel), gray):
                    raise IOError(f"Failed to write {rel}")
                rows.append((idx, t_ns, int(laser_on), rel))

                if frame_callback is not None:
                    frame_callback(gray, idx, bool(laser_on), int(t_ns), None if q is None else q.copy())
                if progress_callback is not None:
                    progress_callback({
                        "stage": "capturing", "frame": idx + 1, "frames_total": cfg.max_frames,
                        "laser_on": bool(laser_on), "progress": 100.0 * (idx + 1) / max(cfg.max_frames, 1),
                        "strobe_sequence": frame_seq,
                    })

                if cfg.show_preview:
                    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    if laser_on:
                        diff = subtract_ambient(gray, last_off)
                        u, v, _ = extract_ridge_subpixel(diff, cfg=ridge_cfg)
                        for u_px, v_px in zip(u[::2], v[::2]):
                            cv2.circle(vis, (int(round(u_px)), int(round(v_px))), 1, (0, 255, 0), -1)
                    status = f"Frame {idx}/{cfg.max_frames} | Laser: {'ON' if laser_on else 'OFF'}"
                    if q is not None:
                        status += f" | q={np.round(q, 2)}"
                    cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    small = cv2.resize(vis, (960, int(960 * gray.shape[0] / gray.shape[1])))
                    cv2.imshow("Intraoral Scanner Live Capture (Press Q to stop)", small)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
        finally:
            actual_camera = {
                "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
                "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "exposure": float(cap.get(cv2.CAP_PROP_EXPOSURE)),
                "set_results": camera_set_results,
            }
            cap.release()
            if cfg.show_preview:
                cv2.destroyAllWindows()
            if imu_reader:
                imu_reader.stop()

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

        meta = {
            "format_version": FORMAT_VERSION,
            "source": "hardware",
            "created": datetime.now(timezone.utc).isoformat(),
            "fps": float(cfg.fps),
            "num_frames": len(rows),
            "image": {"width": cfg.width, "height": cfg.height, "dtype": "uint8", "bit_depth": 8},
            "camera": cam_model.to_dict(),
            "laser_plane": laser_model.to_dict(),
            "camera_runtime": actual_camera,
            "calibration_is_nominal": nominal,
            "has_imu": has_imu,
            "clock": "mcu" if has_imu else "host",
            "serial_warnings": serial_warnings,
            "config": {"capture": {"camera_id": cfg.camera_id, "exposure": cfg.exposure,
                                    "serial_port": cfg.serial_port, "baudrate": cfg.baudrate}},
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out
    except Exception:
        cap.release()
        raise

