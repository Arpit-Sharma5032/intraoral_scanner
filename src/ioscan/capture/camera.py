"""Windows/Linux UVC camera enumeration and capture helpers."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from typing import Any

import cv2
import numpy as np


_BACKEND_IDS = {
    "DirectShow": cv2.CAP_DSHOW,
    "Media Foundation": cv2.CAP_MSMF,
    "Auto": cv2.CAP_ANY,
}


def enumerate_usb_cameras() -> list[dict[str, Any]]:
    """Return camera names together with the exact OpenCV index/backend pair.

    Numeric OpenCV indices are not stable device identities on Windows. In
    particular, virtual cameras (e.g. NVIDIA/OBS) can sit beside physical UVC
    devices. Enumerate each supported Windows backend and let the operator pick
    the physical camera by its device name.
    """
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError as exc:
        raise RuntimeError(
            "Camera-name enumeration is not installed. Run: "
            "python -m pip install cv2-enumerate-cameras"
        ) from exc

    if os.name == "nt":
        backends = [("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF)]
    else:
        backends = [("Auto", cv2.CAP_ANY)]

    devices: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for backend_name, backend_id in backends:
        try:
            entries = enumerate_cameras(backend_id)
        except Exception as exc:
            # A backend can be present in OpenCV but fail device enumeration.
            # Keep the other backend's results rather than aborting the list.
            continue
        for item in entries:
            try:
                index = int(item.index)
            except (TypeError, ValueError, AttributeError):
                continue
            marker = (backend_name, index)
            if marker in seen:
                continue
            seen.add(marker)
            name = str(getattr(item, "name", "") or f"Camera {index}")
            path = getattr(item, "path", None)
            vid = getattr(item, "vid", None)
            pid = getattr(item, "pid", None)
            vid_value = int(vid) if vid is not None else None
            pid_value = int(pid) if pid is not None else None
            usb_id = f" (VID:{vid_value:04X} PID:{pid_value:04X})" if vid_value is not None and pid_value is not None else ""
            devices.append({
                "camera_id": index,
                "backend": backend_name,
                "name": name,
                "path": str(path) if path else None,
                "vid": vid_value,
                "pid": pid_value,
                "label": f"{name}{usb_id} — {backend_name} (index {index})",
            })

    # Put physical Arducam-labelled devices first, then stable alphabetical order.
    devices.sort(key=lambda d: ("arducam" not in d["name"].lower(), d["backend"], d["name"].lower(), d["camera_id"]))
    return devices


def _backend_candidates(preferred_backend: str | None) -> list[tuple[str, int]]:
    if preferred_backend:
        if preferred_backend not in _BACKEND_IDS:
            raise ValueError(f"Unsupported camera backend: {preferred_backend}")
        return [(preferred_backend, _BACKEND_IDS[preferred_backend])]
    if os.name == "nt":
        return [("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF), ("Auto", cv2.CAP_ANY)]
    return [("Auto", cv2.CAP_ANY)]


def _fourcc_text(value: int) -> str | None:
    if not value:
        return None
    try:
        return "".join(chr((int(value) >> (8 * i)) & 0xFF) for i in range(4)).strip() or None
    except Exception:
        return None



class FFmpegDShowCapture:
    """OpenCV-compatible reader for FFmpeg's DirectShow MJPEG image pipe.

    FFmpeg can be preferred for the OV2311 because the operator verified its
    DirectShow/MJPG stream at ~49 FPS while OpenCV's UVC open/read blocked.
    JPEG packets are decoded after crossing the pipe, avoiding uncompressed
    USB/pipe traffic. A background pipe reader makes read() time-bounded.
    """

    def __init__(self, process: subprocess.Popen, width: int, height: int, fps: int):
        self.process = process
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._buffer = bytearray()
        self._closed = False
        self._chunks: queue.Queue[bytes | None] = queue.Queue(maxsize=8)
        self._fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.exposure_readback: float | None = None
        self.control_info: dict[str, Any] = {}
        self._reader_thread = threading.Thread(
            target=self._pipe_reader, name="ioscan-ffmpeg-stdout", daemon=True
        )
        self._reader_thread.start()

    def _pipe_reader(self) -> None:
        if self.process.stdout is None:
            return
        reader = getattr(self.process.stdout, "read1", self.process.stdout.read)
        try:
            while not self._closed:
                chunk = reader(65536)
                if not chunk:
                    break
                while not self._closed:
                    try:
                        self._chunks.put(chunk, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except Exception:
            pass
        finally:
            while not self._closed:
                try:
                    self._chunks.put(None, timeout=0.2)
                    break
                except queue.Full:
                    continue

    def isOpened(self) -> bool:
        return not self._closed and self.process.poll() is None

    def read_encoded(self):
        """Return the next complete camera MJPEG/JPEG frame without decoding it.

        The scan logger can persist these bytes directly as .jpg. This avoids
        CPU-heavy PNG recompression and preserves the exact image delivered by
        the UVC camera. The caller can separately decode a copy for preview and
        laser-profile extraction.
        """
        if self._closed or self.process.stdout is None:
            return False, None
        while True:
            start = self._buffer.find(b"\xff\xd8")
            if start > 0:
                del self._buffer[:start]
                start = 0
            elif start < 0:
                # Retain a possible first byte of the JPEG start marker.
                if self._buffer and self._buffer[-1] == 0xFF:
                    self._buffer[:] = self._buffer[-1:]
                else:
                    self._buffer.clear()

            if start == 0:
                end = self._buffer.find(b"\xff\xd9", 2)
                if end >= 0:
                    jpeg = bytes(self._buffer[:end + 2])
                    del self._buffer[:end + 2]
                    return True, jpeg

            # The reader thread owns the blocking Windows pipe read. Bound this
            # wait so a camera that is busy/not streaming cannot wedge the
            # dashboard worker indefinitely.
            try:
                chunk = self._chunks.get(timeout=5.0)
            except queue.Empty as exc:
                if self.process.poll() is not None:
                    return False, None
                raise TimeoutError(
                    "No MJPEG frame bytes arrived from FFmpeg/DirectShow within 5 seconds. "
                    "Check camera ownership and UVC mode."
                ) from exc
            if chunk is None:
                return False, None
            self._buffer.extend(chunk)
            if len(self._buffer) > 32 * 1024 * 1024:
                self._buffer.clear()
                raise RuntimeError("FFmpeg MJPEG pipe exceeded 32 MiB without a complete JPEG frame")

    def read(self):
        """OpenCV-compatible decoded grayscale read."""
        ok, jpeg = self.read_encoded()
        if not ok or jpeg is None:
            return False, None
        encoded = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if frame is None or frame.size == 0:
            return False, None
        return True, frame

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            # This is the requested DirectShow mode, not a driver readback.
            return self.fps
        if prop == cv2.CAP_PROP_FOURCC:
            return float(self._fourcc)
        if prop == cv2.CAP_PROP_EXPOSURE and self.exposure_readback is not None:
            return self.exposure_readback
        if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            return float(self.control_info.get("auto_exposure_readback", 0.0) or 0.0)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        # DirectShow controls were applied before FFmpeg opened the device. Report
        # whether the requested state is already in effect; do not claim that a
        # post-open mutation was performed by this wrapper.
        if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            expected = 1.0 if self.control_info.get("requested_exposure") is None else 0.0
            return bool(self.control_info.get("auto_exposure_set")) and float(value) == expected
        if prop == cv2.CAP_PROP_EXPOSURE:
            requested = self.control_info.get("requested_exposure")
            return requested is not None and bool(self.control_info.get("exposure_set")) and float(value) == float(requested)
        return False

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2.0)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        try:
            if self.process.stdout:
                self.process.stdout.close()
        except Exception:
            pass
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.5)


def _prime_directshow_controls(
    camera_id: int, exposure: int | None
) -> dict[str, Any]:
    """Apply UVC exposure controls via OpenCV, then release before FFmpeg opens.

    DirectShow's IAMCameraControl is device-level. Drivers commonly retain these
    settings when the graph is reopened by FFmpeg. Readback is recorded so the
    operator can see whether the driver accepted the settings; it is not treated
    as proof of the resulting optical exposure.
    """
    result: dict[str, Any] = {
        "attempted": False, "requested_exposure": exposure,
        "auto_exposure_set": False, "exposure_set": False,
    }
    cap = None
    try:
        cap = cv2.VideoCapture(int(camera_id), cv2.CAP_DSHOW)
        if not cap.isOpened():
            result["error"] = "OpenCV could not open the camera to configure UVC exposure controls"
            return result
        result["attempted"] = True
        auto_value = 1.0 if exposure is None else 0.0
        result["auto_exposure_set"] = bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value))
        result["auto_exposure_readback"] = float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        if exposure is not None:
            result["exposure_set"] = bool(cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure)))
            result["exposure_readback"] = float(cap.get(cv2.CAP_PROP_EXPOSURE))
        else:
            result["exposure_readback"] = float(cap.get(cv2.CAP_PROP_EXPOSURE))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    return result


def _open_ffmpeg_dshow(
    camera_name: str, width: int, height: int, fps: int,
    read_attempts: int, min_fps: float | None, attempts: list[str],
    camera_id: int, exposure: int | None, *, prime_controls: bool = True,
):
    """Open the exact named DirectShow camera using FFmpeg's dshow demuxer."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        attempts.append("FFmpeg fallback: ffmpeg.exe was not found on PATH")
        return None
    if not camera_name or camera_name.strip().lower() in {"selected camera", "camera", "none"}:
        attempts.append("FFmpeg fallback: exact DirectShow camera name was not supplied")
        return None

    if prime_controls:
        controls = _prime_directshow_controls(camera_id, exposure)
    else:
        controls = {
            "attempted": False,
            "requested_exposure": exposure,
            "auto_exposure_set": False,
            "exposure_set": False,
            "note": (
                "UVC exposure was not changed by IOSCAN because opening a second OpenCV camera handle "
                "can block this driver. Set exposure/auto-exposure in the Arducam camera utility "
                "before scanning; FFmpeg DirectShow capture preserves the device setting."
            ),
        }
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-thread_queue_size", "4",
        "-f", "dshow",
        "-video_size", f"{int(width)}x{int(height)}",
        "-framerate", str(int(fps)),
        "-vcodec", "mjpeg",
        "-i", f"video={camera_name.strip()}",
        "-an", "-c:v", "copy", "-flush_packets", "1",
        "-f", "image2pipe", "pipe:1",
    ]
    process = None
    cap = None
    try:
        kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
            "bufsize": 0,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(command, **kwargs)
        cap = FFmpegDShowCapture(process, width, height, fps)
        sample_times: list[float] = []
        frame = None
        for _ in range(max(3, int(read_attempts))):
            ok, candidate = cap.read()
            if not ok or candidate is None:
                break
            frame = candidate
            sample_times.append(time.perf_counter())
        if frame is None or len(sample_times) < 3:
            raise RuntimeError("FFmpeg started but did not deliver at least three JPEG frames")

        elapsed = sample_times[-1] - sample_times[0]
        measured_fps = (len(sample_times) - 1) / elapsed if elapsed > 0 else 0.0
        intervals = [b - a for a, b in zip(sample_times, sample_times[1:]) if b > a]
        median_fps = 1.0 / float(np.median(intervals)) if intervals else 0.0
        if (frame.shape[1], frame.shape[0]) != (int(width), int(height)):
            raise RuntimeError(
                f"FFmpeg delivered {frame.shape[1]}x{frame.shape[0]}, expected {width}x{height}"
            )
        if min_fps is not None and measured_fps < float(min_fps):
            raise RuntimeError(
                f"FFmpeg DirectShow/MJPG measured {measured_fps:.1f} FPS; "
                f"at least {float(min_fps):.1f} FPS is required"
            )

        cap.exposure_readback = controls.get("exposure_readback")
        cap.control_info = controls
        info = {
            "backend": "FFmpeg/DirectShow",
            "requested_format": "MJPG",
            "actual_fourcc": cv2.VideoWriter_fourcc(*"MJPG"),
            "actual_fourcc_text": "MJPG",
            "width": int(width), "height": int(height),
            "fps": round(measured_fps, 3),
            "measured_fps": round(measured_fps, 3),
            "median_fps": round(median_fps, 3),
            # FFmpeg's DirectShow input does not expose CAP_PROP_FPS readback.
            "driver_reported_fps": None,
            "sample_frames": len(sample_times),
            "format_order": "ffmpeg_dshow_explicit_mjpg",
            "capture_engine": "ffmpeg",
            "ffmpeg_command": command,
            "exposure_control": controls,
            "attempts": list(attempts),
        }
        return cap, frame, info
    except Exception as exc:
        attempts.append(f"FFmpeg/DirectShow/MJPG: {type(exc).__name__}: {exc}")
        if cap is not None:
            cap.release()
        elif process is not None:
            try:
                process.terminate()
                process.wait(timeout=2.0)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        return None

def open_usb_camera(
    camera_id: int,
    width: int = 1600,
    height: int = 1200,
    fps: int = 50,
    exposure: int | None = None,
    read_attempts: int = 8,
    backend_name: str | None = None,
    required_fourcc: str | None = None,
    min_fps: float | None = None,
    camera_name: str | None = None,
    prefer_ffmpeg: bool = False,
) -> tuple[Any, Any, dict]:
    """Open a UVC camera and verify the *delivered* pixel format and frame rate.

    ``CAP_PROP_FPS`` and the return value of ``cap.set`` are driver-reported
    settings, not proof that the requested UVC mode was negotiated.  This
    function therefore checks the FOURCC after mode setup and measures the
    arrival rate of several real frames.  A required FOURCC is useful for the
    OV2311 scan path, where 1600x1200 YUY2 is limited to 5 FPS while MJPG can
    deliver the intended high-rate mode.
    """
    if int(read_attempts) < 3:
        raise ValueError("read_attempts must be at least 3 so the camera mode can be verified")
    requested = (required_fourcc or "").strip().upper()
    formats = [
        ("MJPG", cv2.VideoWriter_fourcc(*"MJPG")),
        ("driver-default", None),
        ("YUY2", cv2.VideoWriter_fourcc(*"YUY2")),
        ("YUYV", cv2.VideoWriter_fourcc(*"YUYV")),
        ("GREY", cv2.VideoWriter_fourcc(*"GREY")),
        ("Y16 ", cv2.VideoWriter_fourcc(*"Y16 ")),
    ]
    if requested:
        match = next((item for item in formats if item[0].strip().upper() == requested), None)
        if match is None:
            raise ValueError(f"Unsupported required camera FOURCC: {required_fourcc!r}")
        formats = [match]

    attempts: list[str] = []

    # The OV2311's FFmpeg DirectShow/MJPG path has been independently measured
    # at ~49 FPS by the operator. Prefer that path for the physical scanner: an
    # OpenCV VideoCapture open/read can block inside some Windows UVC drivers,
    # preventing a later FFmpeg fallback from ever being reached.
    if (prefer_ffmpeg and os.name == "nt" and backend_name == "DirectShow"
            and requested == "MJPG" and camera_name):
        ffmpeg_result = _open_ffmpeg_dshow(
            camera_name, width, height, fps, read_attempts, min_fps, attempts,
            camera_id, exposure, prime_controls=False,
        )
        if ffmpeg_result is not None:
            return ffmpeg_result
        raise RuntimeError(
            f"Preferred FFmpeg/DirectShow/MJPG capture failed for {camera_name!r}. "
            + " | ".join(attempts[-4:])
            + ". OpenCV was deliberately not tried because its UVC open/read may block. "
              "Check that no other application owns the camera and that FFmpeg is on PATH."
        )

    # Some Windows drivers honour FOURCC only after dimensions/FPS are set;
    # others require it first.  Reopen between orderings to avoid mixed state.
    property_orders = ("mode_then_format", "format_then_mode")

    # Camera indices are backend-specific on Windows. Never retry the same
    # numeric index under a different backend: it could silently select another
    # physical/virtual camera. The operator can select the Arducam entry for the
    # other backend from the enumerated device list.
    for actual_backend_name, backend in _backend_candidates(backend_name):
        for format_name, fourcc in formats:
            for order_name in property_orders:
                cap = cv2.VideoCapture(int(camera_id), backend)
                try:
                    if not cap.isOpened():
                        attempts.append(f"{actual_backend_name}/{format_name}/{order_name}: open failed")
                        cap.release()
                        continue

                    mode_props = [
                        (cv2.CAP_PROP_FRAME_WIDTH, int(width)),
                        (cv2.CAP_PROP_FRAME_HEIGHT, int(height)),
                        (cv2.CAP_PROP_FPS, int(fps)),
                    ]
                    format_prop = [] if fourcc is None else [(cv2.CAP_PROP_FOURCC, fourcc)]
                    props = (mode_props + format_prop) if order_name == "mode_then_format" else (format_prop + mode_props)
                    props.append((cv2.CAP_PROP_BUFFERSIZE, 1))
                    set_results: dict[str, bool] = {}
                    for prop, value in props:
                        try:
                            set_results[str(int(prop))] = bool(cap.set(prop, value))
                        except Exception:
                            set_results[str(int(prop))] = False

                    exposure_results = {}
                    if exposure is not None:
                        # Arducam's Windows UVC demo uses AUTO_EXPOSURE=0 for
                        # manual mode before applying the requested exposure.
                        for prop, value in ((cv2.CAP_PROP_AUTO_EXPOSURE, 0.0),
                                            (cv2.CAP_PROP_EXPOSURE, float(exposure))):
                            try:
                                exposure_results[str(int(prop))] = bool(cap.set(prop, value))
                            except Exception:
                                exposure_results[str(int(prop))] = False

                    actual_fourcc = int(round(cap.get(cv2.CAP_PROP_FOURCC)))
                    actual_text = _fourcc_text(actual_fourcc)
                    if fourcc is not None and actual_text and actual_text.upper() != format_name.strip().upper():
                        attempts.append(
                            f"{actual_backend_name}/{format_name}/{order_name}: driver reported {actual_text}"
                        )
                        cap.release()
                        continue
                    if requested and actual_text and actual_text.upper() != requested:
                        attempts.append(
                            f"{actual_backend_name}: required {requested}, driver reported {actual_text}"
                        )
                        cap.release()
                        continue

                    frame = None
                    sample_times: list[float] = []
                    for _ in range(int(read_attempts)):
                        ok, candidate = cap.read()
                        if ok and candidate is not None and getattr(candidate, "size", 0):
                            frame = candidate
                            sample_times.append(time.perf_counter())
                        else:
                            time.sleep(0.03)
                    if frame is None or len(sample_times) < 3:
                        attempts.append(f"{actual_backend_name}/{format_name}/{order_name}: fewer than 3 valid frames")
                        cap.release()
                        continue

                    # Read FOURCC again after real frames; some drivers report it
                    # only once streaming has begun.
                    actual_fourcc = int(round(cap.get(cv2.CAP_PROP_FOURCC)))
                    actual_text = _fourcc_text(actual_fourcc)
                    if fourcc is not None and actual_text != format_name.strip().upper():
                        attempts.append(
                            f"{actual_backend_name}/{format_name}/{order_name}: streamed as "
                            f"{actual_text or 'unknown FOURCC'}"
                        )
                        cap.release()
                        continue
                    if requested and actual_text != requested:
                        attempts.append(
                            f"{actual_backend_name}: required {requested}, streamed as "
                            f"{actual_text or 'unknown FOURCC'}"
                        )
                        cap.release()
                        continue

                    intervals = [b - a for a, b in zip(sample_times, sample_times[1:]) if b > a]
                    measured_fps = ((len(sample_times) - 1) / (sample_times[-1] - sample_times[0])) if len(sample_times) > 1 and sample_times[-1] > sample_times[0] else 0.0
                    median_fps = (1.0 / float(np.median(intervals))) if intervals else 0.0
                    if min_fps is not None and measured_fps < float(min_fps):
                        attempts.append(
                            f"{actual_backend_name}/{format_name}/{order_name}: measured {measured_fps:.1f} FPS, "
                            f"required at least {float(min_fps):.1f} FPS"
                        )
                        cap.release()
                        continue

                    frame_width = int(frame.shape[1])
                    frame_height = int(frame.shape[0])
                    if (frame_width, frame_height) != (int(width), int(height)):
                        attempts.append(
                            f"{actual_backend_name}/{format_name}/{order_name}: streamed {frame_width}x{frame_height}, "
                            f"expected {int(width)}x{int(height)}"
                        )
                        cap.release()
                        continue

                    driver_fps = float(cap.get(cv2.CAP_PROP_FPS))
                    info = {
                        "backend": actual_backend_name,
                        "requested_format": format_name,
                        "actual_fourcc": actual_fourcc,
                        "actual_fourcc_text": actual_text,
                        "width": frame_width,
                        "height": frame_height,
                        # Keep the legacy 'fps' field truthful: it is measured
                        # from real frame arrivals, not CAP_PROP_FPS.
                        "fps": round(measured_fps, 3),
                        "measured_fps": round(measured_fps, 3),
                        "median_fps": round(median_fps, 3),
                        "driver_reported_fps": driver_fps,
                        "sample_frames": len(sample_times),
                        "format_order": order_name,
                        "set_results": set_results,
                        "exposure_results": exposure_results,
                        "attempts": attempts,
                    }
                    return cap, frame, info
                except Exception as exc:
                    attempts.append(f"{actual_backend_name}/{format_name}/{order_name}: {type(exc).__name__}: {exc}")
                    cap.release()

    # FFmpeg's DirectShow path has been verified independently on the OV2311
    # at 1600x1200 MJPG/50 FPS. Use it only after OpenCV's DirectShow attempts
    # fail the real-frame-rate gate, and only when the selected device name is
    # available so we never accidentally open a different camera index.
    if (os.name == "nt" and backend_name == "DirectShow" and requested == "MJPG"
            and camera_name):
        ffmpeg_result = _open_ffmpeg_dshow(
            camera_name, width, height, fps, read_attempts, min_fps, attempts,
            camera_id, exposure,
        )
        if ffmpeg_result is not None:
            return ffmpeg_result

    requirement = f" Required format: {requested}." if requested else ""
    min_rate = f" Minimum measured rate: {float(min_fps):.1f} FPS." if min_fps is not None else ""
    raise RuntimeError(
        f"Could not negotiate a verified {int(width)}x{int(height)} camera stream from index {camera_id}."
        f"{requirement}{min_rate} Tried: " + " | ".join(attempts[-12:])
        + " Try the other Windows backend (Media Foundation/DirectShow), close other camera applications, "
        "and verify the UVC mode in Arducam's camera utility."
    )


def to_gray_u8(frame):
    """Normalize UVC mono, BGR, or BGRA frames to the pipeline's uint8 grayscale."""
    import numpy as np

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        if arr.dtype == np.uint16:
            peak = int(np.percentile(arr, 99.9)) if arr.size else 0
            shift = max(0, int(np.ceil(np.log2(peak + 1))) - 8) if peak > 255 else 0
            arr = (arr >> shift).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    raise RuntimeError(f"Camera returned unsupported frame shape/dtype: {arr.shape}/{arr.dtype}")
