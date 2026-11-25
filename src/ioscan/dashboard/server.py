from __future__ import annotations

from datetime import datetime
import math
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import threading
import time
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ioscan.capture.logger import CaptureConfig, record_hardware_session
from ioscan.capture.camera import enumerate_usb_cameras, open_usb_camera, to_gray_u8
from ioscan.utils.roi import (
    enabled_roi_points, load_roi_config, save_roi_config,
)


SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class SegmentationRequest(BaseModel):
    jaw: Literal["lower", "upper"] = "lower"
    models_dir: str | None = None


class HardwareGeometryRequest(BaseModel):
    baseline_mm: float = Field(default=18.0, gt=0.1, le=100.0)
    laser_tilt_deg: float = Field(default=10.0, gt=1.0, lt=89.0)
    fan_half_angle_deg: float = Field(default=25.0, gt=1.0, lt=89.0)
    exposure: int = Field(default=-7, ge=-13, le=0)
    speed_mm_s: float = Field(default=5.0, gt=0.05, le=50.0)
    depth_min_mm: float = Field(default=75.0, ge=0.1, le=500.0)
    depth_max_mm: float = Field(default=125.0, gt=0.2, le=1000.0)
    # Reconstruction (see scripts/run_pipeline.py and ioscan.pipeline.sweep)
    recon_mode: Literal["sweep", "imu_hybrid"] = "sweep"
    translation: Literal["auto", "markers", "planar", "flow", "length", "speed"] = "auto"
    scan_length_mm: float = Field(default=0.0, ge=0.0, le=500.0)   # 0 = unknown
    auto_trim: bool = True
    auto_depth: bool = True
    secondary_ratio: float = Field(default=0.6, ge=0.1, le=1.0)
    min_prominence: float = Field(default=25.0, ge=3.0, le=250.0)

    def checked(self) -> dict:
        values = self.model_dump() if hasattr(self, "model_dump") else self.dict()
        if values["depth_min_mm"] >= values["depth_max_mm"]:
            raise ValueError("Minimum reconstruction depth must be smaller than maximum depth.")
        values["working_distance_mm"] = round(
            values["baseline_mm"] / math.tan(math.radians(values["laser_tilt_deg"])), 3
        )
        return values

class CaptureRequest(BaseModel):
    camera_id: int = Field(default=0, ge=0, le=64)
    camera_backend: Literal["DirectShow", "Media Foundation", "Auto"] = "Auto"
    camera_name: str | None = Field(default=None, max_length=256)
    serial_port: str = Field(default="COM3", min_length=1, max_length=32)
    frames: int = Field(default=1200, ge=20, le=20000)
    exposure: int = Field(default=-7, ge=-13, le=0)
    baseline_mm: float = Field(default=18.0, gt=0.1, le=100.0)
    laser_tilt_deg: float = Field(default=10.0, gt=1.0, lt=89.0)
    fan_half_angle_deg: float = Field(default=25.0, gt=1.0, lt=89.0)
    depth_min_mm: float = Field(default=75.0, ge=0.1, le=500.0)
    depth_max_mm: float = Field(default=125.0, gt=0.2, le=1000.0)
    session_name: str | None = None
    jaw: Literal["lower", "upper"] = "lower"
    models_dir: str | None = None
    calibration_file: str | None = None
    auto_segment: bool = True
    mesh_method: Literal["bpa", "poisson", "tsdf"] = "bpa"
    speed_mm_s: float = Field(default=5.0, gt=0.05, le=50.0)
    # Reconstruction (see scripts/run_pipeline.py and ioscan.pipeline.sweep)
    recon_mode: Literal["sweep", "imu_hybrid"] = "sweep"
    translation: Literal["auto", "markers", "planar", "flow", "length", "speed"] = "auto"
    scan_length_mm: float = Field(default=0.0, ge=0.0, le=500.0)   # 0 = unknown
    auto_trim: bool = True
    auto_depth: bool = True
    secondary_ratio: float = Field(default=0.6, ge=0.1, le=1.0)
    min_prominence: float = Field(default=25.0, ge=3.0, le=250.0)
    # "calibration": record the bare marker sheet and compute the camera + laser calibration from it
    purpose: Literal["scan", "calibration"] = "scan"
    bar_length_mm: float = Field(default=100.0, ge=90.0, le=110.0)   # measured length of the printed 100 mm bar


class CalibrationRequest(BaseModel):
    """Compute the calibration from an already recorded session of the bare marker sheet."""
    session_name: str | None = None
    bar_length_mm: float = Field(default=100.0, ge=90.0, le=110.0)


class ProcessRequest(BaseModel):
    """Re-run reconstruction on an already captured session with the current settings."""
    session_name: str | None = None          # None = most recent session
    frames: int = 0
    depth_min_mm: float = Field(default=75.0, ge=0.1, le=500.0)
    depth_max_mm: float = Field(default=125.0, gt=0.2, le=1000.0)
    jaw: Literal["lower", "upper"] = "lower"
    models_dir: str | None = None
    auto_segment: bool = False
    mesh_method: Literal["bpa", "poisson", "tsdf"] = "bpa"
    speed_mm_s: float = Field(default=5.0, gt=0.05, le=50.0)
    # Reconstruction (see scripts/run_pipeline.py and ioscan.pipeline.sweep)
    recon_mode: Literal["sweep", "imu_hybrid"] = "sweep"
    translation: Literal["auto", "markers", "planar", "flow", "length", "speed"] = "auto"
    scan_length_mm: float = Field(default=0.0, ge=0.0, le=500.0)   # 0 = unknown
    auto_trim: bool = True
    auto_depth: bool = True
    secondary_ratio: float = Field(default=0.6, ge=0.1, le=1.0)
    min_prominence: float = Field(default=25.0, ge=3.0, le=250.0)
    use_calibration: bool = True             # process with the current calibration file, if there is one


class StitchRequest(BaseModel):
    """Fuse several sweep passes (in scan order) into one full-arch surface."""
    passes: list[str] = []
    name: str | None = None
    keep_base: bool = False
    auto_segment: bool = True
    jaw: Literal["lower", "upper"] = "lower"
    models_dir: str | None = None


class ScannerDashboard:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.roi_config_path = self.project_root / "configs" / "roi.json"
        self.hardware_geometry_path = self.project_root / "configs" / "hardware_geometry.json"
        self.calibration_path = self.project_root / "data" / "calibration" / "calibration.json"
        self.marker_board_path = self.project_root / "configs" / "marker_board.json"
        self._marker_det = None
        self._marker_det_votes = [0, 0]
        self._ready_markers: dict = {"count": None, "at": 0.0}
        self._ready_line: dict | None = None
        self._ready_pub_at = 0.0
        self._active_roi_polygon: list[list[float]] | None = None
        self._lock = threading.RLock()
        self._stop_event: threading.Event | None = None
        self._worker: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._last_frame_at: float | None = None
        self._capture_metric_at: float | None = None
        self._capture_metric_frame: int = 0
        self._idle_preview_stop = threading.Event()
        self._idle_preview_thread: threading.Thread | None = None
        self._idle_preview_ready = threading.Event()
        self._idle_preview_info: dict = {}
        self._idle_preview_error: str | None = None
        self._camera_selection: tuple[int, str, str] | None = None   # (id, backend, name) for preview restarts
        self._state: dict = self._initial_state()
        self.session_dir: Path | None = None
        self.results_dir: Path | None = None
        self._live_camera = None
        self._live_laser = None
        self._live_fuser = None
        self._live_tracker = None
        self._live_last_off: np.ndarray | None = None
        self._live_profiles = 0
        self._live_points = 0
        self._live_last_off_idx: int = -10
        self._live_z_range: tuple[float, float] = (75.0, 125.0)
        self._preview_q: queue.Queue | None = None
        self._preview_stop = threading.Event()
        self._preview_thread: threading.Thread | None = None
        self._last_request: CaptureRequest | None = None
        self._live_sweep = True
        self._line_ema: dict | None = None
        self._line_pub_at = 0.0
        self._line_roi_cache: tuple | None = None
        self._live_ridge: tuple[float, float] = (0.6, 25.0)

    @staticmethod
    def _initial_state() -> dict:
        return {
            "phase": "idle", "stage": "ready", "message": "Connect the scanner to begin.",
            "progress": 0.0, "frame": 0, "processed_frames": 0, "frames_total": 0, "laser_on": False,
            "camera_fps": 0.0, "preview_revision": 0, "model_revision": 0,
            "model_asset": None, "session_dir": None, "results_dir": None,
            "jaw": "lower", "auto_segment": True, "mesh_vertices": 0,
            "mesh_faces": 0, "point_count": 0, "trajectory_mm": 0.0,
            "segmentation": None, "warnings": [], "error": None,
            "recon_mode": None, "translation_source": None,
            "line_stats": None, "ready_check": None, "verdict": None, "calibration_result": None,
            "outputs": {}, "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _update(self, **updates) -> None:
        with self._lock:
            self._state.update(updates)
            self._state["updated_at"] = datetime.now().isoformat(timespec="seconds")

    def snapshot(self) -> dict:
        with self._lock:
            snap = json.loads(json.dumps(self._state))
        t = self._idle_preview_thread
        snap["camera_live"] = bool(t is not None and t.is_alive() and self._idle_preview_ready.is_set()
                                   and not self._idle_preview_error)
        return snap

    def get_hardware_geometry(self) -> dict:
        defaults = HardwareGeometryRequest().checked()
        if not self.hardware_geometry_path.is_file():
            return defaults
        try:
            raw = json.loads(self.hardware_geometry_path.read_text(encoding="utf-8"))
            model = HardwareGeometryRequest(**raw)
            return model.checked()
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(500, f"Saved hardware geometry is invalid: {exc}") from exc

    def save_hardware_geometry(self, payload: HardwareGeometryRequest) -> dict:
        if self.snapshot()["phase"] in {"capturing", "processing"}:
            raise HTTPException(409, "Hardware geometry cannot be changed while a scan is active.")
        try:
            values = payload.checked()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        self.hardware_geometry_path.parent.mkdir(parents=True, exist_ok=True)
        saved = {key: value for key, value in values.items() if key != "working_distance_mm"}
        temp_path = self.hardware_geometry_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        temp_path.replace(self.hardware_geometry_path)
        return values

    def _stop_idle_camera_preview(self) -> bool:
        self._idle_preview_stop.set()
        thread = self._idle_preview_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        # Some Windows UVC drivers can leave cap.read() blocked. Do not silently
        # forget that worker and open a second handle to the same camera.
        if thread is not None and thread.is_alive():
            return False
        self._idle_preview_thread = None
        return True

    def _idle_camera_preview_loop(self, camera_id: int, camera_backend: str = "DirectShow", camera_name: str = "Selected camera") -> None:
        cap = None
        try:
            # Idle preview must NOT inherit the scan's fixed manual exposure.
            # Arducam's Windows UVC guide defines AUTO_EXPOSURE=1 as automatic
            # and 0 as manual (the latter is applied only during synchronized scan capture).
            cap, frame, info = open_usb_camera(
                camera_id, width=1600, height=1200, fps=50, exposure=None,
                read_attempts=16, backend_name=camera_backend,
                required_fourcc="MJPG", min_fps=42.5, camera_name=camera_name,
                prefer_ffmpeg=(os.name == "nt" and camera_backend == "DirectShow"),
            )
            info["camera_name"] = camera_name
            info["selected_backend"] = camera_backend
            try:
                info["auto_exposure_set"] = bool(
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
                )
                info["auto_exposure_value"] = float(
                    cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                )
            except Exception as exc:
                info["auto_exposure_set"] = False
                info["auto_exposure_error"] = str(exc)

            # Let auto exposure settle and reject virtual-camera/invalid streams
            # that report a valid frame shape but contain only zero pixels.
            # This was the hidden failure: an NVIDIA Broadcast virtual camera can
            # open and report 1600x1200 while not returning the physical Arducam feed.
            sample = None
            brightness_p95 = 0.0
            brightness_mean = 0.0
            for _ in range(12):
                if self._idle_preview_stop.is_set():
                    break
                preview_frame = to_gray_u8(frame)
                sample = preview_frame[::8, ::8]
                brightness_mean = float(np.mean(sample)) if sample.size else 0.0
                brightness_p95 = float(np.percentile(sample, 95)) if sample.size else 0.0
                if brightness_p95 >= 2.0:
                    break
                ok, adjusted = cap.read()
                if ok and adjusted is not None and getattr(adjusted, "size", 0):
                    frame = adjusted
                else:
                    time.sleep(0.05)

            if brightness_p95 < 2.0:
                name = camera_name.lower()
                hint = (
                    " The selected device name indicates an NVIDIA/virtual camera; "
                    "select the physical Arducam UVC camera instead."
                    if any(token in name for token in ("nvidia", "broadcast", "virtual", "obs"))
                    else " Select the physical Arducam device by name, not a virtual camera. "
                         "Also disable OV2311 external-trigger/low-brightness-compensation mode "
                         "unless the hardware is actively providing the required trigger pulses."
                )
                raise RuntimeError(
                    f"{camera_name} ({camera_backend}, index {camera_id}) returned an all-black frame "
                    f"(mean {brightness_mean:.2f}, p95 {brightness_p95:.2f}/255)." + hint
                )

            self._idle_preview_info = {
                **info,
                "brightness_mean": round(brightness_mean, 2),
                "brightness_p95": round(brightness_p95, 2),
                "brightness_max": int(np.max(sample)) if sample is not None and sample.size else 0,
            }
            ok_jpg, encoded = cv2.imencode(
                ".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            if not ok_jpg:
                raise RuntimeError("OpenCV received a camera frame but could not encode its preview JPEG.")
            with self._lock:
                self._latest_jpeg = encoded.tobytes()
            self._idle_preview_ready.set()
            last_encode = 0.0
            last_ready = 0.0
            prev_gray = None
            while not self._idle_preview_stop.is_set():
                # Publish the latest real frame after the initial non-black frame passed validation.
                # Encode at most ~10 fps (the page polls at ~11 fps): encoding every 50 fps frame cost
                # about half a CPU core, which now matters because the preview also runs during processing.
                now = time.perf_counter()
                if now - last_encode >= 0.1:
                    last_encode = now
                    preview_frame = to_gray_u8(frame)
                    ok_jpg, encoded = cv2.imencode(
                        ".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 82]
                    )
                    if ok_jpg:
                        with self._lock:
                            self._latest_jpeg = encoded.tobytes()
                ok, next_frame = cap.read()
                if not ok or next_frame is None:
                    if self._idle_preview_stop.wait(0.05):
                        break
                    continue
                if now - last_ready >= 0.33:
                    # Ready check on two consecutive frames (laser on in one of them if it is strobing)
                    try:
                        cur_gray = to_gray_u8(next_frame)
                        if prev_gray is not None and prev_gray.shape == cur_gray.shape:
                            self._idle_ready_check(prev_gray, cur_gray)
                            last_ready = now
                            prev_gray = None
                        else:
                            prev_gray = cur_gray
                    except Exception:
                        prev_gray = None
                frame = next_frame
        except Exception as exc:
            self._idle_preview_error = str(exc)
            self._idle_preview_ready.set()
        finally:
            if cap is not None:
                cap.release()

    def _resume_idle_preview_async(self, delay_s: float = 0.8) -> None:
        """Reopen the live camera preview after a capture released the camera.

        Previously nothing restarted it, so the dashboard showed the last captured frame until the
        next scan (or a server restart). Processing does not use the camera, so the preview can run
        while the scan is reconstructed.
        """
        sel = self._camera_selection
        if sel is None:
            return

        def worker():
            time.sleep(delay_s)                      # let the driver finish releasing the capture handle
            if self.snapshot()["phase"] == "capturing":
                return
            try:
                self.camera_probe(*sel)
            except Exception as exc:  # the preview is a convenience; never disturb processing
                self._idle_preview_error = f"Live preview could not reopen after the scan: {getattr(exc, 'detail', exc)}"

        threading.Thread(target=worker, name="ioscan-preview-resume", daemon=True).start()

    def camera_probe(self, camera_id: int, camera_backend: str = "DirectShow", camera_name: str = "Selected camera") -> dict:
        if self.snapshot()["phase"] == "capturing":
            raise HTTPException(409, "Camera is in use by the active scan.")
        self._camera_selection = (int(camera_id), str(camera_backend), str(camera_name))
        if not self._stop_idle_camera_preview():
            raise HTTPException(
                503,
                "The previous camera preview read is still blocked by the driver. "
                "Restart the dashboard process to release the camera, then retry.",
            )
        self._idle_preview_stop = threading.Event()
        self._idle_preview_ready = threading.Event()
        self._idle_preview_error = None
        self._idle_preview_info = {}
        with self._lock:
            self._latest_jpeg = None
        self._idle_preview_thread = threading.Thread(
            target=self._idle_camera_preview_loop, args=(camera_id, camera_backend, camera_name),
            name="ioscan-idle-camera-preview", daemon=True,
        )
        self._idle_preview_thread.start()
        # Initial UVC negotiation can take several seconds on Windows, especially
        # when a monochrome camera is switching into its native 1600x1200 mode.
        # The worker also owns the first read/encode, so wait longer than a single
        # camera frame interval before declaring the preview unavailable.
        if not self._idle_preview_ready.wait(timeout=20.0):
            alive = bool(self._idle_preview_thread and self._idle_preview_thread.is_alive())
            stopped = self._stop_idle_camera_preview()
            if alive:
                detail = (
                    "OpenCV is still blocked while opening/reading the selected camera. "
                    "Close Windows Camera and other camera apps. "
                    + ("Restart the dashboard process before retrying; the driver has not released the camera. "
                       if not stopped else "Then try another camera index. ")
                    + "The driver-default UVC mode is now tried before forced pixel formats."
                )
            else:
                detail = "The camera worker exited without producing a preview frame."
            raise HTTPException(503, f"No preview frame arrived within 20 seconds. {detail}")
        if self._idle_preview_error:
            error = self._idle_preview_error
            self._stop_idle_camera_preview()
            raise HTTPException(503, error)
        info = self._idle_preview_info
        return {
            "connected": True, "camera_id": camera_id,
            "width": int(info.get("width", 0)), "height": int(info.get("height", 0)),
            "backend": info.get("backend"), "camera_backend": info.get("backend"),
            "preferred_camera_backend": camera_backend,
            "camera_name": camera_name,
            "pixel_format": info.get("actual_fourcc_text"),
            "requested_format": info.get("requested_format"),
            "fps": info.get("measured_fps", info.get("fps")),
            "measured_fps": info.get("measured_fps", info.get("fps")),
            "driver_reported_fps": info.get("driver_reported_fps"),
            "auto_exposure_set": info.get("auto_exposure_set"),
            "brightness_mean": info.get("brightness_mean"),
            "brightness_p95": info.get("brightness_p95"),
            "brightness_max": info.get("brightness_max"),
        }

    def start_capture(self, request: CaptureRequest) -> dict:
        current = self.snapshot()
        if current["phase"] in {"capturing", "processing"}:
            raise HTTPException(409, "A scan is already running.")
        if request.depth_min_mm >= request.depth_max_mm:
            raise HTTPException(422, "Minimum reconstruction depth must be smaller than maximum depth.")

        # Validate the requested jaw checkpoint before acquiring a potentially long scan.
        # The previous dashboard flow only discovered missing/incompatible weights after
        # reconstruction, which wasted capture time and left a partial session behind.
        # Sweep reconstructions are single-pass patches; MeshSegNet is not run automatically on them.
        if request.auto_segment and request.recon_mode != "sweep":
            try:
                from ioscan.pipeline.segment import (
                    SegmentConfig, _import_torch, _load_network, resolve_segmentation_model,
                )
                seg_kwargs = {"jaw": request.jaw, "device": "cpu"}
                if request.models_dir:
                    seg_kwargs["models_dir"] = Path(request.models_dir).expanduser()
                seg_cfg = SegmentConfig(**seg_kwargs)
                _import_torch()
                checkpoint, _selected_jaw = resolve_segmentation_model(seg_cfg)
                net, _inner_checkpoint = _load_network(seg_cfg, checkpoint, "cpu")
                del net
            except Exception as exc:
                raise HTTPException(
                    422,
                    "MeshSegNet preflight failed before capture. Check PyTorch, the selected jaw, "
                    f"and the checkpoint folder. Details: {exc}",
                ) from exc
        prefix = "calib" if request.purpose == "calibration" else "scan"
        session_name = request.session_name or datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
        if not SESSION_RE.fullmatch(session_name):
            raise HTTPException(422, "Session name may contain letters, numbers, underscores and hyphens only.")
        self._camera_selection = (int(request.camera_id), str(request.camera_backend), str(request.camera_name))

        try:
            roi_config = load_roi_config(self.roi_config_path)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        session_dir = self.project_root / "data" / "sessions" / session_name
        results_dir = self.project_root / "data" / "results" / session_name
        if session_dir.exists():
            raise HTTPException(409, f"Session folder already exists: {session_dir}")
        if results_dir.exists():
            raise HTTPException(409, f"Results folder already exists: {results_dir}")

        # Create the session directory itself before writing its per-session
        # ROI snapshot. Creating only session_dir.parent leaves `scan_x/`
        # missing and caused start_capture() to fail with FileNotFoundError.
        session_dir.mkdir(parents=True, exist_ok=False)
        results_dir.mkdir(parents=True, exist_ok=False)
        # Freeze the ROI for this scan so later edits in Setup cannot change
        # reconstruction of an already captured session.
        (session_dir / "roi.json").write_text(
            json.dumps(roi_config, indent=2) + "\n", encoding="utf-8"
        )
        self._active_roi_polygon = enabled_roi_points(roi_config)
        if not self._stop_idle_camera_preview():
            raise HTTPException(
                503,
                "The camera preview thread is still blocked by the UVC driver. "
                "Restart the dashboard process before starting a synchronized scan.",
            )
        self.session_dir, self.results_dir = session_dir, results_dir
        self._last_request = request
        self._stop_event = threading.Event()
        self._capture_metric_at = time.perf_counter()
        self._capture_metric_frame = 0
        self._last_frame_at = None
        self._line_ema = None
        self._line_pub_at = 0.0
        self._live_swap_votes = 0.0
        self._live_flags_swapped = False
        with self._lock:
            self._latest_jpeg = None
        self._state = self._initial_state()
        self._update(
            phase="capturing", stage="capturing", message="Opening camera and recording scan frames…",
            frames_total=request.frames, session_dir=str(session_dir), results_dir=str(results_dir),
            jaw=request.jaw, auto_segment=request.auto_segment,
        )
        self._worker = threading.Thread(
            target=self._capture_worker, args=(request, session_dir, results_dir),
            name="ioscan-dashboard-capture", daemon=True,
        )
        self._worker.start()
        return {"started": True, "session_name": session_name, "session_dir": str(session_dir)}

    def stop_capture(self) -> dict:
        with self._lock:
            if self._state.get("phase") != "capturing" or self._stop_event is None:
                raise HTTPException(409, "There is no active capture to stop.")
            self._stop_event.set()
            self._state["message"] = "Stopping capture and closing the session…"
        return {"stopping": True}

    def _init_live_reconstruction(self, cfg: CaptureConfig, camera_cal: dict | None,
                                  laser_cal: dict | None, speed_mm_s: float,
                                  z_range: tuple[float, float] = (75.0, 125.0),
                                  sweep: bool = True, auto_depth: bool = True) -> None:
        """Prepare a lightweight fusion path so the viewport grows during acquisition."""
        self._live_camera = self._live_laser = self._live_fuser = self._live_tracker = None
        self._live_last_off = None
        self._live_profiles = self._live_points = 0
        self._live_z_range = (float(z_range[0]), float(z_range[1]))
        try:
            from ioscan.capture.logger import _build_calibration
            from ioscan.pipeline.fusion import FusionConfig, ScanFuser
            from ioscan.pipeline.pose import PoseConfig
            from ioscan.pipeline.triangulate import laser_plane_depth_bounds
            camera_model, laser_model, nominal = _build_calibration(cfg, camera_cal, laser_cal)
            self._live_camera, self._live_laser = camera_model, laser_model
            live_warnings = []
            if nominal:
                live_warnings.append("No measured calibration JSON supplied; live preview uses nominal example camera/laser geometry.")
            plane_z_bounds = laser_plane_depth_bounds(camera_model, laser_model)
            if plane_z_bounds is not None and (plane_z_bounds[1] < z_range[0] or plane_z_bounds[0] > z_range[1]):
                live_warnings.append(
                    f"Live point cloud unavailable: laser plane predicts Z={plane_z_bounds[0]:.1f}.."
                    f"{plane_z_bounds[1]:.1f} mm, outside selected depth range {z_range[0]:.1f}..{z_range[1]:.1f} mm. "
                    "Verify baseline/tilt against the physical mount and use measured calibration."
                )
            if self._active_roi_polygon is None:
                live_warnings.append("Camera ROI mask is disabled; chassis/interior pixels are included in laser extraction.")
            else:
                from ioscan.pipeline.triangulate import resolve_depth_window
                from ioscan.pipeline.triangulate import roi_depth_coverage
                cov = roi_depth_coverage(camera_model, laser_model, self._active_roi_polygon, z_range) if auto_depth else {}
                if cov.get("suggested_z_range_mm"):
                    self._live_z_range = tuple(map(float, cov["suggested_z_range_mm"]))
                else:
                    z_used, _depth_info, depth_msg = resolve_depth_window(
                        camera_model, laser_model, self._active_roi_polygon, z_range)
                    if depth_msg:
                        live_warnings.append(depth_msg)
                        self._live_z_range = (float(z_used[0]), float(z_used[1]))
            if live_warnings:
                self._update(warnings=live_warnings)
            self._live_fuser = ScanFuser(camera_model, FusionConfig(voxel_size_mm=0.08, use_tsdf=False))
            self._live_pose_config = PoseConfig(mode="imu_hybrid", nominal_speed_mm_s=speed_mm_s)
            # Sweep mode: place profiles along the scan axis without IMU rotation (a straight pass).
            self._live_sweep = bool(sweep)
        except Exception as exc:
            # Preview is a convenience; the saved-session pipeline remains authoritative.
            self._live_camera = self._live_laser = self._live_fuser = None
            self._update(warnings=[f"Live 3D preview unavailable: {exc}"])

    # Target band for the laser line peak above ambient (8-bit DN). Below it, ridge points drop
    # out near the noise floor; above it, the line saturates and the sub-pixel centre is biased.
    LINE_TARGET_DN = (150.0, 220.0)

    def _line_roi_bounds(self, shape: tuple[int, int]):
        key = (shape, None if self._active_roi_polygon is None else json.dumps(self._active_roi_polygon))
        if self._line_roi_cache is not None and self._line_roi_cache[0] == key:
            return self._line_roi_cache[1]
        if self._active_roi_polygon is not None:
            from ioscan.utils.roi import rasterize_roi
            mask = rasterize_roi(shape, self._active_roi_polygon) > 0
            ys, xs = np.nonzero(mask)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            bounds = (y0, y1, x0, x1, mask[y0:y1, x0:x1])
        else:
            bounds = (0, shape[0], 0, shape[1], None)
        self._line_roi_cache = (key, bounds)
        return bounds

    def _line_sample(self, on: np.ndarray, off: np.ndarray) -> dict:
        """Laser-line statistics of one ON/OFF pair inside the ROI (column peaks of ON-OFF)."""
        y0, y1, x0, x1, mask = self._line_roi_bounds(on.shape)
        on_c = on[y0:y1, x0:x1]
        diff = on_c.astype(np.int16) - off[y0:y1, x0:x1].astype(np.int16)
        if mask is not None:
            diff[~mask] = 0
        rows = np.argmax(diff, axis=0)
        cols = np.arange(diff.shape[1])
        peak = diff[rows, cols].astype(np.float64)
        raw = on_c[rows, cols]
        in_roi = mask.any(axis=0) if mask is not None else np.ones(diff.shape[1], dtype=bool)
        # same idea as the pipeline's adaptive threshold: a few noise sigmas, at most the configured value
        noise = 1.4826 * float(np.median(np.abs(diff[::3, ::3] - np.median(diff[::3, ::3]))))
        min_prom = float(np.clip(6.0 * noise, 7.0, max(float(self._live_ridge[1]), 7.0)))
        found = in_roi & (peak >= min_prom)
        n_roi = max(int(in_roi.sum()), 1)
        return {
            "peak_p50": float(np.median(peak[found])) if found.any() else 0.0,
            "peak_p95": float(np.percentile(peak[found], 95)) if found.any() else 0.0,
            "coverage_pct": 100.0 * float(found.sum()) / n_roi,
            "saturated_pct": 100.0 * float((found & (raw >= 254)).sum()) / n_roi,
            "ambient_p50": float(np.median(off[y0:y1, x0:x1][mask] if mask is not None else off[y0:y1, x0:x1])),
        }

    def _measure_line(self, on: np.ndarray, off: np.ndarray) -> None:
        """Live laser-line brightness readout (column peaks of ON-OFF inside the ROI)."""
        sample = self._line_sample(on, off)
        a = 0.3  # EMA so the number is readable while the hand moves
        if self._line_ema is None:
            self._line_ema = dict(sample, frames=1)
        else:
            for k, v in sample.items():
                self._line_ema[k] = (1 - a) * self._line_ema[k] + a * v
            self._line_ema["frames"] += 1
        self._publish_ready(off, dict(self._line_ema), laser_seen=True, capturing=True)
        now = time.perf_counter()
        if now - self._line_pub_at < 0.25:
            return
        self._line_pub_at = now
        e = self._line_ema
        lo, hi = self.LINE_TARGET_DN
        if e["coverage_pct"] < 20.0:
            verdict, advice = "no_line", "Line not found across the ROI: aim at the model, check the ROI and laser."
        elif e["saturated_pct"] > 3.0 or e["peak_p50"] > hi:
            verdict, advice = "high", "Too bright: lower exposure (or laser power)."
        elif e["peak_p50"] < lo and e["ambient_p50"] > 100.0:
            verdict, advice = "low", (
                f"Line too faint against a bright background (ambient {e['ambient_p50']:.0f} DN): dim the room "
                "light or shade the model, or raise laser power. Raising exposure would clip the background.")
        elif e["peak_p50"] < lo:
            verdict, advice = "low", "Too dim: raise exposure (or laser power)."
        else:
            verdict, advice = "good", "Line brightness is in the target band."
        if e["ambient_p50"] < 15.0 and verdict in ("good", "low"):
            # Motion tracking needs the model visible in the laser-OFF frames.
            advice += (" Background is almost black: turn the room light on until the background reads "
                       "~40-80 DN, otherwise the model's motion cannot be tracked.")
        with self._lock:
            self._state["line_stats"] = {
                "peak_p50": round(e["peak_p50"], 1), "peak_p95": round(e["peak_p95"], 1),
                "coverage_pct": round(e["coverage_pct"], 1), "saturated_pct": round(e["saturated_pct"], 2),
                "ambient_p50": round(e["ambient_p50"], 1), "frames": int(e["frames"]),
                "target_dn": [lo, hi], "min_prominence": float(self._live_ridge[1]),
                "verdict": verdict, "advice": advice,
                "flags_swapped": bool(getattr(self, "_live_flags_swapped", False)),
            }

    def _update_live_geometry(self, gray: np.ndarray, idx: int, laser_on: bool,
                              t_ns: int, quat_wxyz: np.ndarray | None) -> None:
        if not laser_on:
            self._live_last_off = gray
            self._live_last_off_idx = idx
            return
        on_img, off_img = gray, self._live_last_off
        if off_img is not None and idx - self._live_last_off_idx <= 3 and off_img.shape == gray.shape:
            # The recorded ON/OFF flag can be one frame out of phase with the camera (seen on
            # hardware). The laser frame is unmistakable in the images, so pair by content.
            from ioscan.session.laser_phase import laser_pair_polarity
            pol = laser_pair_polarity(gray, off_img)
            self._live_swap_votes = 0.8 * getattr(self, "_live_swap_votes", 0.0) + 0.2 * (
                -1.0 if pol < -10.0 else (1.0 if pol > 10.0 else 0.0))
            if self._live_swap_votes < -0.3:
                on_img, off_img = off_img, gray
            self._live_flags_swapped = self._live_swap_votes < -0.3
            try:
                self._measure_line(on_img, off_img)
            except Exception as exc:  # the readout must never stop the preview
                self._update(warnings=[f"Line brightness readout paused: {exc}"])
        if self._live_fuser is None or self._live_camera is None or self._live_laser is None:
            return
        if getattr(self, "_live_sweep", False):
            quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        if quat_wxyz is None or self._live_last_off is None:
            return
        if idx - self._live_last_off_idx > 3:
            return  # preview frames were dropped; do not subtract a stale ambient image
        try:
            from ioscan.pipeline.pose import PoseTracker
            from ioscan.pipeline.ridge import RidgeConfig
            from ioscan.pipeline.triangulate import triangulate_frame
            timestamp_s = float(t_ns) * 1e-9
            if self._live_tracker is None:
                self._live_tracker = PoseTracker(self._live_pose_config, initial_timestamp_s=timestamp_s)
            pts_cam, uv, _ = triangulate_frame(
                on_img, off_img, self._live_camera, self._live_laser,
                ridge_cfg=RidgeConfig(method="cog", saturation_limit=254.0,
                                      max_secondary_ratio=self._live_ridge[0], min_prominence=self._live_ridge[1]),
                z_range=self._live_z_range, roi_polygon=self._active_roi_polygon,
            )
            transform = self._live_tracker.update(
                frame_index=idx, timestamp_s=timestamp_s, quat_wxyz=quat_wxyz, pts_cam=pts_cam,
            )
            if pts_cam.shape[0]:
                self._live_fuser.integrate(pts_cam, uv, transform)
                self._live_points += int(len(pts_cam))
            self._live_profiles += 1
            if self._live_profiles % 12 == 0 and self.results_dir is not None:
                preview = self._live_fuser.get_preview_point_cloud(voxel_downsample_mm=0.16)
                if len(preview.points):
                    import os
                    import open3d as o3d
                    final_path = self.results_dir / "reconstruction_preview.ply"
                    temp_path = self.results_dir / "reconstruction_preview.tmp.ply"
                    if o3d.io.write_point_cloud(str(temp_path), preview, write_ascii=False):
                        os.replace(temp_path, final_path)
                        with self._lock:
                            self._state.update(
                                model_asset="preview",
                                model_revision=int(self._state.get("model_revision", 0)) + 1,
                                point_count=self._live_points,
                                message="Live 3D point cloud is growing as scan profiles are fused…",
                            )
        except Exception as exc:
            # Keep camera capture alive if the optional live preview encounters a geometry issue.
            self._live_fuser = None
            self._update(warnings=[f"Live 3D preview paused: {exc}"])

    def _start_preview_worker(self) -> None:
        """Run live triangulation/JPEG encoding off the capture thread.

        The capture loop must pair every ESP32 strobe packet with exactly one camera frame at the
        camera rate; any heavy per-frame work there overflows the serial event queue.
        """
        self._preview_stop = threading.Event()
        self._preview_q = queue.Queue(maxsize=8)       # each item holds a full 1600x1200 frame
        self._preview_thread = threading.Thread(
            target=self._preview_loop, name="ioscan-dashboard-preview", daemon=True)
        self._preview_thread.start()

    def _stop_preview_worker(self) -> None:
        self._preview_stop.set()
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=5.0)
        self._preview_thread = None

    def _preview_loop(self) -> None:
        q = self._preview_q
        while True:
            try:
                gray, idx, laser_on, t_ns, quat = q.get(timeout=0.1)
            except queue.Empty:
                if self._preview_stop.is_set():
                    return
                continue
            try:
                self._update_live_geometry(gray, idx, laser_on, t_ns, quat)
                ok, encoded = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 76])
                if ok:
                    with self._lock:
                        self._latest_jpeg = encoded.tobytes()
            except Exception as exc:  # preview must never break capture
                self._update(warnings=[f"Live preview paused: {exc}"])

    def _on_capture_progress(self, event: dict) -> None:
        """Cheap per-frame dashboard metrics; does not decode or copy image data."""
        now = time.perf_counter()
        frame = int(event.get("frame", 0))
        with self._lock:
            fps = float(self._state.get("camera_fps", 0.0))
            last_at = self._capture_metric_at
            last_frame = self._capture_metric_frame
            # Smooth the rate over at least ten frames to avoid 50 Hz display jitter.
            if last_at is None:
                self._capture_metric_at = now
                self._capture_metric_frame = frame
            elif frame - last_frame >= 10 or now - last_at >= 0.5:
                dt = max(now - last_at, 1e-6)
                fps = max(0.0, (frame - last_frame) / dt)
                self._capture_metric_at = now
                self._capture_metric_frame = frame
            self._state.update(
                frame=frame,
                laser_on=bool(event.get("laser_on", False)),
                progress=float(event.get("progress", 0.0)),
                camera_fps=round(fps, 1) if fps < 200 else 0.0,
            )

    def _on_frame(self, gray: np.ndarray, idx: int, laser_on: bool, t_ns: int,
                  quat_wxyz: np.ndarray | None) -> None:
        """Sparse preview callback; capture counters are updated separately per frame."""
        with self._lock:
            self._state.update(
                preview_revision=int(self._state.get("preview_revision", 0)) + 1,
            )
        q = self._preview_q
        if q is None:
            return
        item = (gray.copy(), idx, bool(laser_on), int(t_ns), None if quat_wxyz is None else quat_wxyz.copy())
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()          # drop the oldest preview frame, never block capture
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _capture_worker(self, request: CaptureRequest, session_dir: Path, results_dir: Path) -> None:
        try:
            cfg = CaptureConfig(
                camera_id=request.camera_id, camera_backend=request.camera_backend,
                camera_name=request.camera_name,
                width=1600, height=1200, fps=50,
                exposure=request.exposure, laser_baseline_mm=request.baseline_mm,
                laser_tilt_deg=request.laser_tilt_deg,
                laser_fan_half_angle_deg=request.fan_half_angle_deg,
                serial_port=request.serial_port.strip(), max_frames=request.frames, show_preview=False,
            )
            camera_cal = laser_cal = None
            cal_file = request.calibration_file
            if not cal_file and request.purpose != "calibration" and self._active_calibration_file() is not None:
                cal_file = str(self._active_calibration_file())     # the calibration made with the marker sheet
            if cal_file:
                calibration_path = Path(cal_file).expanduser()
                if not calibration_path.is_file():
                    raise FileNotFoundError(f"Calibration JSON not found: {calibration_path}")
                calibration_data = json.loads(calibration_path.read_text(encoding="utf-8"))
                camera_cal = calibration_data.get("camera")
                laser_cal = calibration_data.get("laser")
                if not camera_cal or not laser_cal:
                    raise ValueError("Calibration JSON must contain both 'camera' and 'laser' objects.")
            self._live_ridge = (float(request.secondary_ratio), float(request.min_prominence))
            self._init_live_reconstruction(
                cfg, camera_cal, laser_cal, request.speed_mm_s,
                (request.depth_min_mm, request.depth_max_mm),
                sweep=request.recon_mode == "sweep", auto_depth=request.auto_depth,
            )
            self._start_preview_worker()
            try:
                record_hardware_session(
                    session_dir, cfg=cfg, camera_cal_dict=camera_cal, laser_cal_dict=laser_cal,
                    frame_callback=self._on_frame, progress_callback=self._on_capture_progress,
                    stop_event=self._stop_event,
                )
            finally:
                self._stop_preview_worker()
                self._resume_idle_preview_async()
            # Preserve jaw and processing settings for later CLI runs or segmentation retries.
            meta_path = session_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["jaw"] = request.jaw
            meta["purpose"] = request.purpose
            meta.setdefault("config", {})["jaw"] = request.jaw
            meta["config"]["dashboard"] = {
                "mesh_method": request.mesh_method, "speed_mm_s": request.speed_mm_s,
                "baseline_mm": request.baseline_mm, "laser_tilt_deg": request.laser_tilt_deg,
                "nominal_working_distance_mm": round(request.baseline_mm / math.tan(math.radians(request.laser_tilt_deg)), 3),
                "fan_half_angle_deg": request.fan_half_angle_deg,
                "depth_min_mm": request.depth_min_mm, "depth_max_mm": request.depth_max_mm,
                "exposure": request.exposure,
                "recon_mode": request.recon_mode, "translation": request.translation,
                "scan_length_mm": request.scan_length_mm, "auto_trim": request.auto_trim,
                "auto_depth": request.auto_depth, "secondary_ratio": request.secondary_ratio,
                "min_prominence": request.min_prominence,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            count = self.snapshot()["frame"]
            stopped = meta.get("capture_stopped_reason")
            self._update(
                phase="captured", stage="captured", progress=100.0,
                message=f"Capture saved · {count} frames recorded.",
                **({"warnings": [stopped + " Close other programs (browser tabs, other apps) or restart the "
                                 "dashboard before long scans."]} if stopped else {}),
            )
            # The live preview has done its job: free its memory before the full reconstruction.
            self._live_fuser = self._live_tracker = None
            self._live_last_off = None
            import gc
            gc.collect()
            if request.purpose == "calibration":
                self._calibration_worker(session_dir, float(request.bar_length_mm))
            else:
                self._process_scan(request, session_dir, results_dir)
        except Exception as exc:
            self._update(phase="error", stage="error", error=str(exc), message="Capture stopped because an error occurred.")

    # ------------------------------------------------------------------ calibration
    def calibration_status(self) -> dict:
        """What the scanner currently uses to turn pixels into millimetres."""
        from ioscan.pipeline.markers import MarkerBoard
        board = MarkerBoard.load(self.marker_board_path)
        out = {"calibrated": False, "path": str(self.calibration_path), "board": board.to_dict(),
               "sheet_pdf": (self.project_root / "docs" / "marker_sheet_A4.pdf").is_file()}
        if not self.calibration_path.is_file():
            return out
        try:
            data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            cam, meta = data["camera"], data.get("metadata") or {}
            out.update(
                calibrated=True, created=meta.get("created"), quality=meta.get("quality"),
                notes=meta.get("notes") or [], method=meta.get("method", "file"),
                focal_px=round(float(cam["fx"]), 1),
                distortion_k1=round(float((cam.get("dist_coeffs") or [0.0])[0]), 4),
                camera_rms_px=meta.get("camera_rms_px"), sheet_residual_mm=meta.get("sheet_residual_mm"),
                laser_plane_rms_mm=meta.get("laser_plane_rms_mm"), mirrored_optics=meta.get("mirrored_optics"),
                focal_change_percent=meta.get("focal_change_percent"),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            out["error"] = f"The calibration file could not be read: {exc}"
        return out

    def _active_calibration_file(self) -> Path | None:
        return self.calibration_path if self.calibration_path.is_file() else None

    def start_calibration(self, request: CalibrationRequest) -> dict:
        if self.snapshot()["phase"] in {"capturing", "processing"}:
            raise HTTPException(409, "A scan or processing job is already running.")
        sessions = [x for x in self.list_sessions() if x["kind"] != "arch"]
        if not sessions:
            raise HTTPException(404, "No recorded sessions found.")
        name = request.session_name or sessions[0]["name"]
        if not SESSION_RE.fullmatch(name) or name not in {x["name"] for x in sessions}:
            raise HTTPException(404, f"Session not found: {name}")
        session_dir = self.project_root / "data" / "sessions" / name
        self.session_dir, self.results_dir = session_dir, None
        self._state = self._initial_state()
        self._update(phase="processing", stage="calibrating", progress=0.0, session_dir=str(session_dir),
                     message=f"Calibrating from {name}…")
        threading.Thread(target=self._calibration_worker, args=(session_dir, float(request.bar_length_mm)),
                         name="ioscan-dashboard-calibrate", daemon=True).start()
        return {"started": True, "session_name": name}

    def _calibration_worker(self, session_dir: Path, bar_length_mm: float) -> None:
        try:
            from dataclasses import replace
            from ioscan.calibration.marker_calibration import (
                CalibrationError, calibrate_from_marker_session, save_marker_calibration,
            )
            from ioscan.pipeline.markers import MarkerBoard
            board = replace(MarkerBoard.load(self.marker_board_path), scale=round(bar_length_mm / 100.0, 5))
            self._update(phase="processing", stage="calibrating", progress=1.0,
                         message="Calibrating from the marker sheet recording…")
            try:
                result = calibrate_from_marker_session(
                    session_dir, board=board,
                    progress=lambda e: self._update(stage="calibrating", progress=float(e.get("progress", 0.0)),
                                                    message=e.get("message", "Calibrating…")))
            except CalibrationError as exc:
                self._update(phase="error", stage="error", error=str(exc),
                             message="Calibration was not saved; the previous calibration is unchanged.")
                return
            if self.calibration_path.is_file():
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.calibration_path.replace(self.calibration_path.with_name(f"calibration_{stamp}.bak.json"))
            board.save(self.marker_board_path)
            save_marker_calibration(result, self.calibration_path)
            st = result["stats"]
            self._update(
                phase="complete", stage="calibrated", progress=100.0, error=None,
                calibration_result=self.calibration_status(),
                warnings=list(st.get("notes") or []),
                message=(f"Calibration saved · focal length {result['camera'].fx:.0f} px "
                         f"({st['focal_change_percent']:+.1f}% vs nominal) · sheet residual "
                         f"{st['sheet_residual_mm']:.2f} mm. New scans and Reprocess now use it."),
            )
        except Exception as exc:
            self._update(phase="error", stage="error", error=str(exc), message="Calibration failed.")

    # ------------------------------------------------------------------ ready check
    def _count_markers(self, gray: np.ndarray) -> int | None:
        """Markers of the printed plate visible in an ambient frame (None if detection is unavailable)."""
        try:
            from ioscan.pipeline.markers import MarkerBoard, MarkerDetector
            if self._marker_det is None:
                board = MarkerBoard.load(self.marker_board_path)
                self._marker_det = [MarkerDetector(board, False), MarkerDetector(board, True)]
            votes = self._marker_det_votes
            decided = abs(votes[0] - votes[1]) >= 12
            order = [int(votes[1] > votes[0])] if decided else [0, 1]
            best = 0
            for k in order:
                n = len(self._marker_det[k].detect(gray)[0])
                votes[k] += n
                best = max(best, n)
            return best
        except Exception:
            return None

    def _publish_ready(self, off: np.ndarray | None, line: dict | None, *, laser_seen: bool, capturing: bool) -> None:
        """Traffic-light readiness shown next to the camera image, before and during a scan."""
        now = time.perf_counter()
        if off is not None and now - self._ready_markers["at"] >= 0.35:
            self._ready_markers = {"count": self._count_markers(off), "at": now}
        if now - self._ready_pub_at < 0.3:
            return
        self._ready_pub_at = now
        items = []
        cal = self._active_calibration_file() is not None
        items.append({"key": "calibration", "label": "Calibration", "state": "ok" if cal else "warn",
                      "value": "Measured" if cal else "Nominal",
                      "hint": "" if cal else "Run Calibrate once with the bare marker sheet for correct sizes and marker-plate scans."})
        n = self._ready_markers["count"]
        if n is None:
            items.append({"key": "markers", "label": "Marker plate", "state": "warn", "value": "n/a",
                          "hint": "Marker detection is unavailable in this OpenCV build."})
        else:
            st = "ok" if n >= 4 else "warn"
            hint = ("" if n >= 4 else "Few markers in view: slide the plate so more of the printed sheet is visible."
                    if n >= 2 else "No marker plate in view. Put the model on the printed marker sheet: without it the "
                                   "scan falls back to image tracking, which is much less dependable.")
            items.append({"key": "markers", "label": "Marker plate", "state": st, "value": f"{n} markers", "hint": hint})
        amb = None
        if line is not None:
            amb = float(line["ambient_p50"])
        elif off is not None:
            y0, y1, x0, x1, mask = self._line_roi_bounds(off.shape)
            crop = off[y0:y1, x0:x1]
            amb = float(np.median(crop[mask] if mask is not None else crop))
            self._ready_amb = amb
        elif getattr(self, "_ready_amb", None) is not None:
            amb = self._ready_amb
        if amb is not None:
            if amb > 120:
                a_state, a_hint = "bad", "Too much light on the model: dim the room or shade the box."
            elif amb > 90:
                a_state, a_hint = "warn", "Bright background: the laser line will be weak on white teeth. Dim the light a little."
            elif amb < 8 and (n or 0) < 2:
                a_state, a_hint = "warn", "Very dark: the markers cannot be read. Add a little steady light."
            else:
                a_state, a_hint = "ok", ""
            items.append({"key": "light", "label": "Background light", "state": a_state, "value": f"{amb:.0f}", "hint": a_hint})
        if line is None:
            items.append({"key": "line", "label": "Laser line", "state": "idle",
                          "value": "waiting" if capturing else "checked when the scan starts", "hint": ""})
        else:
            peak, cov, sat = line["peak_p50"], line["coverage_pct"], line["saturated_pct"]
            if cov < 15:
                l_state, l_hint = "bad", "Laser line not found in the scan area: aim it at the model and check the ROI."
            elif sat > 3:
                l_state, l_hint = "warn", "The line is saturated: lower the exposure or the laser power."
            elif peak < 25:
                l_state, l_hint = "bad", "The line is too faint: reduce the background light or raise the laser power."
            elif peak < 60 or cov < 40:
                l_state, l_hint = "warn", "The line is weak or short: expect holes. Less background light gives a stronger line."
            else:
                l_state, l_hint = "ok", ""
            items.append({"key": "line", "label": "Laser line", "state": l_state,
                          "value": f"{peak:.0f} over background · {cov:.0f}% of width", "hint": l_hint})
        if now - getattr(self, "_ready_mem_at", -1e9) >= 2.0:
            from ioscan.utils.memory import system_memory
            self._ready_mem = system_memory()
            self._ready_mem_at = now
        mem = getattr(self, "_ready_mem", None)
        if mem and mem.get("available_mb") is not None:
            free_gb = min(mem["available_mb"], mem.get("commit_available_mb") or mem["available_mb"]) / 1024.0
            if free_gb < 1.0:
                m_state, m_hint = "bad", ("Very little free memory: a long scan will stop early. Close other programs "
                                          "or restart the dashboard.")
            elif free_gb < 2.0:
                m_state, m_hint = "warn", "Free memory is low: close other programs before a long scan."
            else:
                m_state, m_hint = "ok", ""
            items.append({"key": "memory", "label": "Free memory", "state": m_state, "value": f"{free_gb:.1f} GB",
                          "hint": m_hint})
        states = [i["state"] for i in items if i["key"] != "calibration"]
        overall = "bad" if "bad" in states else ("warn" if "warn" in states or not cal else "ok")
        first = next((i["hint"] for i in items if i["state"] == "bad" and i["hint"]), None) \
            or next((i["hint"] for i in items if i["state"] == "warn" and i["hint"]), "")
        title = {"ok": "Ready to scan", "warn": "Can scan, with limits", "bad": "Not ready"}[overall]
        with self._lock:
            self._state["ready_check"] = {"state": overall, "title": title, "advice": first, "items": items,
                                           "live": bool(capturing)}

    def _idle_ready_check(self, prev: np.ndarray, cur: np.ndarray) -> None:
        """Ready check from two consecutive preview frames (the laser may or may not be strobing)."""
        if self.snapshot()["phase"] in {"capturing", "processing"}:
            return
        from ioscan.session.laser_phase import laser_pair_polarity
        try:
            self._active_roi_polygon = enabled_roi_points(load_roi_config(self.roi_config_path))
        except Exception:
            pass
        y0, y1, x0, x1, _m = self._line_roi_bounds(cur.shape)
        pol = laser_pair_polarity(cur[y0:y1, x0:x1], prev[y0:y1, x0:x1])
        line = None
        off = cur
        if abs(pol) > 10.0:
            on, off = (cur, prev) if pol > 0 else (prev, cur)
            line = self._line_sample(on, off)
        self._publish_ready(off, line, laser_seen=abs(pol) > 10.0, capturing=False)

    def list_sessions(self) -> list[dict]:
        root = self.project_root / "data" / "sessions"
        out = []
        if not root.is_dir():
            return out
        for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
            if not (d / "frames.csv").is_file() or not SESSION_RE.fullmatch(d.name):
                continue
            frames = None
            try:
                with open(d / "frames.csv", "r", encoding="utf-8") as f:
                    frames = max(sum(1 for _ in f) - 1, 0)
            except OSError:
                pass
            res = self.project_root / "data" / "results" / d.name
            verdict = motion = None
            try:
                rep_path = res / "pipeline_report.json"
                if rep_path.is_file():
                    rj = json.loads(rep_path.read_text(encoding="utf-8"))
                    verdict = (rj.get("verdict") or {}).get("status")
                    motion = (rj.get("translation") or {}).get("used")
            except (OSError, ValueError):
                pass
            out.append({
                "name": d.name, "frames": frames,
                "modified": datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds"),
                "has_mesh": (res / "reconstructed_mesh.ply").is_file(),
                "kind": "calibration" if d.name.startswith("calib") else "scan",
                "verdict": verdict, "motion": motion,
            })
        # arches built from several scans have results but no recording of their own
        results_root = self.project_root / "data" / "results"
        known = {o["name"] for o in out}
        if results_root.is_dir():
            for d in results_root.iterdir():
                if (d.is_dir() and d.name not in known and SESSION_RE.fullmatch(d.name)
                        and (d / "reconstructed_mesh.ply").is_file()):
                    out.append({"name": d.name, "frames": None, "has_mesh": True, "kind": "arch",
                                "modified": datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds"),
                                "verdict": None, "motion": None})
        out.sort(key=lambda o: o["modified"], reverse=True)
        return out

    def open_result(self, name: str) -> dict:
        """Show an already processed scan (or built arch) again without reprocessing it."""
        if self.snapshot()["phase"] in {"capturing", "processing"}:
            raise HTTPException(409, "A scan or processing job is already running.")
        if not SESSION_RE.fullmatch(name or ""):
            raise HTTPException(422, "Invalid scan name.")
        res = self.project_root / "data" / "results" / name
        if not (res / "reconstructed_mesh.ply").is_file():
            raise HTTPException(404, f"{name} has no reconstructed surface yet; reprocess it first.")
        rep = {}
        try:
            rep = json.loads((res / "pipeline_report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        session_dir = self.project_root / "data" / "sessions" / name
        self.session_dir = session_dir if session_dir.is_dir() else None
        self.results_dir = res
        outputs = {"mesh": str(res / "reconstructed_mesh.ply")}
        if (res / "fused_cloud.ply").is_file():
            outputs["cloud"] = str(res / "fused_cloud.ply")
        asset = "mesh"
        if (res / "segmented_full_colored.ply").is_file():
            outputs["segmented"] = str(res / "segmented_full_colored.ply")
            asset = "segmented"
        if (res / "segmented_clean_teeth.ply").is_file():
            outputs["teeth"] = str(res / "segmented_clean_teeth.ply")
        seg = None
        try:
            seg_path = res / "segmentation_report.json"
            if seg_path.is_file():
                seg = json.loads(seg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            seg = None
        prev_rev = int(self.snapshot().get("model_revision", 0))
        self._state = self._initial_state()
        self._update(
            phase="complete", stage="complete", progress=100.0, model_asset=asset, model_revision=prev_rev + 1,
            session_dir=str(self.session_dir) if self.session_dir else str(res), results_dir=str(res),
            mesh_vertices=int(rep.get("mesh_vertices") or 0), mesh_faces=int(rep.get("mesh_triangles") or 0),
            point_count=int(rep.get("fused_cloud_points") or 0),
            trajectory_mm=round(float(rep.get("estimated_trajectory_length_mm") or 0.0), 2),
            warnings=list(rep.get("warnings") or []), verdict=rep.get("verdict"), outputs=outputs,
            segmentation=seg if isinstance(seg, dict) and "per_tooth_meshes" in seg else None,
            recon_mode=rep.get("pose_mode"), translation_source=(rep.get("translation") or {}).get("used"),
            auto_segment=False, message=f"Opened {name}.",
        )
        return {"opened": True, "name": name}

    def start_processing(self, request: ProcessRequest) -> dict:
        if self.snapshot()["phase"] in {"capturing", "processing"}:
            raise HTTPException(409, "A scan or processing job is already running.")
        if request.depth_min_mm >= request.depth_max_mm:
            raise HTTPException(422, "Minimum reconstruction depth must be smaller than maximum depth.")
        sessions = [x for x in self.list_sessions() if x["kind"] != "arch"]
        if not sessions:
            raise HTTPException(404, "No captured sessions found in data/sessions.")
        name = request.session_name or sessions[0]["name"]
        if not SESSION_RE.fullmatch(name) or name not in {s["name"] for s in sessions}:
            raise HTTPException(404, f"Session not found: {name}")
        session_dir = self.project_root / "data" / "sessions" / name
        results_dir = self.project_root / "data" / "results" / name
        results_dir.mkdir(parents=True, exist_ok=True)
        roi_path = session_dir / "roi.json"
        try:
            self._active_roi_polygon = enabled_roi_points(load_roi_config(roi_path)) if roi_path.is_file() else None
        except ValueError:
            self._active_roi_polygon = None
        self.session_dir, self.results_dir = session_dir, results_dir
        self._state = self._initial_state()
        self._update(
            phase="processing", stage="reconstructing", progress=0.0,
            message=f"Reprocessing {name} with the current reconstruction settings…",
            session_dir=str(session_dir), results_dir=str(results_dir),
            jaw=request.jaw, auto_segment=request.auto_segment, frames_total=request.frames,
        )
        worker = threading.Thread(
            target=self._process_scan, args=(request, session_dir, results_dir),
            name="ioscan-dashboard-reprocess", daemon=True,
        )
        worker.start()
        return {"started": True, "session_name": name}

    def start_stitch(self, request: StitchRequest) -> dict:
        if self.snapshot()["phase"] in {"capturing", "processing"}:
            raise HTTPException(409, "A scan or processing job is already running.")
        if not 1 <= len(request.passes) <= 12:
            raise HTTPException(422, "Select between 1 and 12 scans.")
        results_root = self.project_root / "data" / "results"
        dirs = []
        motions: list = []
        for name in request.passes:
            if not SESSION_RE.fullmatch(name):
                raise HTTPException(422, f"Invalid pass name: {name}")
            d = results_root / name
            if not (d / "fused_cloud.ply").is_file():
                raise HTTPException(409, f"Pass {name} has no sweep reconstruction yet; reprocess it first.")
            rep = d / "pipeline_report.json"
            motion = None
            if rep.is_file():
                try:
                    rj = json.loads(rep.read_text(encoding="utf-8"))
                    mode, motion = rj.get("pose_mode"), (rj.get("translation") or {}).get("used")
                except (OSError, json.JSONDecodeError):
                    mode = None
                if mode not in (None, "sweep"):
                    raise HTTPException(409, f"Pass {name} was reconstructed in '{mode}' mode; reprocess it in sweep mode.")
            dirs.append(d)
            motions.append(motion)
        if len(dirs) == 1 and motions[0] != "markers":
            raise HTTPException(422, "A single scan can only be finished into an arch if it was made on the marker "
                                     "plate. Scans made without the plate need two or more overlapping passes.")
        if len(set(request.passes)) != len(request.passes):
            raise HTTPException(422, "Each pass can be used once.")
        name = request.name or datetime.now().strftime("arch_%Y%m%d_%H%M%S")
        if not SESSION_RE.fullmatch(name):
            raise HTTPException(422, "Arch name may contain letters, numbers, underscores and hyphens only.")
        out = results_root / name
        if out.exists():
            raise HTTPException(409, f"Results folder already exists: {out}")
        self.session_dir, self.results_dir = None, out
        self._state = self._initial_state()
        self._update(phase="processing", stage="stitching", progress=0.0, results_dir=str(out),
                     jaw=request.jaw, auto_segment=request.auto_segment,
                     message=f"Stitching {len(dirs)} passes into a full arch…")
        threading.Thread(target=self._stitch_worker, args=(request, dirs, out),
                         name="ioscan-dashboard-stitch", daemon=True).start()
        return {"started": True, "name": name, "results_dir": str(out)}

    def _stitch_worker(self, request: StitchRequest, dirs: list[Path], out: Path) -> None:
        try:
            from ioscan.pipeline.stitch import StitchConfig, stitch_to_files
            cfg = StitchConfig()
            if request.keep_base:
                cfg.trim_to_teeth_mm = None
            rep = stitch_to_files(dirs, out, cfg, progress=lambda m, p: self._update(
                stage="stitching", progress=float(p), message=m))
            q = rep["quality"]
            notes = []
            for lk in rep["links"]:
                g = lk.get("icp_global", lk["icp"])
                imu = lk["imu_rotation_vs_previous_deg"]
                notes.append(
                    f"{lk['pass']}: relief match {lk['coarse']['relief_ncc']:.2f}, overlap {100 * g['overlap_frac']:.0f}%, "
                    f"RMS {g['rmse_mm']:.3f} mm, turned {lk['rotation_vs_previous_deg']:.0f} deg"
                    + (f" (IMU {imu:.0f} deg)" if imu is not None else ""))
            self._update(
                stage="reconstructed", progress=100.0, model_asset="mesh",
                model_revision=self.snapshot()["model_revision"] + 1,
                mesh_vertices=int(rep["mesh_vertices"]), mesh_faces=int(rep["mesh_triangles"]),
                point_count=int(rep["fused_cloud_points"]), recon_mode="stitched_sweeps",
                warnings=notes + list(rep["warnings"]),
                outputs={"mesh": rep["paths"]["mesh_ply"], "cloud": rep["paths"]["cloud_ply"]},
                message=(f"Merged {len(dirs)} marker-plate scans · {int(rep['mesh_triangles']):,} faces."
                         if rep.get("merged_on_marker_plate") else
                         f"Full arch stitched from {len(dirs)} passes · {int(rep['mesh_triangles']):,} faces."
                         if not rep.get("unreliable_joints") else
                         f"Stitched, but {rep['unreliable_joints']} of {len(dirs) - 1} joints are unreliable — see notices."),
            )
            if request.auto_segment and rep.get("unreliable_joints"):
                self._update(phase="complete_with_warnings", stage="quality_error", progress=100.0,
                             error="Some joints are unreliable; MeshSegNet was skipped on this arch.")
            elif request.auto_segment and q["segmentation_ready_by_basic_geometry_gate"]:
                self._update(phase="processing", stage="segmenting", progress=0.0,
                             message="Running MeshSegNet on the stitched arch…")
                self._segmentation_worker(SegmentationRequest(jaw=request.jaw, models_dir=request.models_dir))
            elif request.auto_segment:
                self._update(phase="complete_with_warnings", stage="quality_error", progress=100.0,
                             error="Stitched mesh failed the quality gate; MeshSegNet was skipped.",
                             message="Arch stitched, but it failed the quality gate (see notices).")
            else:
                self._update(phase="complete", stage="complete", progress=100.0)
        except Exception as exc:
            self._update(phase="error", stage="error", error=str(exc), message="Stitching failed.")

    def start_segmentation(self, request: SegmentationRequest) -> dict:
        snap = self.snapshot()
        if snap["phase"] in {"capturing", "processing"}:
            raise HTTPException(409, "A processing job is already running.")
        if self.results_dir is None or not (self.results_dir / "reconstructed_mesh.ply").is_file():
            raise HTTPException(409, "Complete a scan and reconstruction before segmentation.")
        if int(snap.get("mesh_faces", 0)) < 5000:
            raise HTTPException(
                409,
                f"The current mesh has only {int(snap.get('mesh_faces', 0)):,} faces. "
                "IOSCAN requires at least 5,000 faces and a dental-arch-sized scan before MeshSegNet. "
                "Recapture with the camera ROI enabled and dense, overlapping laser profiles."
            )
        self._update(phase="processing", stage="segmenting", progress=0.0,
                     error=None, message="Starting MeshSegNet segmentation…")
        worker = threading.Thread(
            target=self._segmentation_worker, args=(request,),
            name="ioscan-dashboard-segmentation", daemon=True,
        )
        worker.start()
        return {"started": True}

    def _segmentation_worker(self, request: SegmentationRequest) -> None:
        try:
            from ioscan.pipeline.segment import SegmentConfig, segment_mesh_and_cloud
            kwargs = {"jaw": request.jaw}
            if request.models_dir:
                kwargs["models_dir"] = Path(request.models_dir).expanduser()
            mesh_path = self.results_dir / "reconstructed_mesh.ply"
            cloud_path = self.results_dir / "fused_cloud.ply"
            seg = segment_mesh_and_cloud(
                mesh_path, self.results_dir, cfg=SegmentConfig(**kwargs),
                cloud_path=cloud_path if cloud_path.is_file() else None,
                session_hint=self.session_dir or "",
                progress_callback=lambda e: self._update(
                    stage="segmenting", progress=float(e.get("progress", 0.0)),
                    message=e.get("message", "Running MeshSegNet inference…"),
                ),
            )
            current_outputs = self.snapshot().get("outputs", {})
            self._update(
                phase="complete", stage="complete", progress=100.0,
                model_asset="segmented", model_revision=self.snapshot()["model_revision"] + 1,
                segmentation=seg,
                outputs={**current_outputs, "segmented": seg["full_colored_ply"], "teeth": seg.get("clean_teeth_ply")},
                warnings=list(self.snapshot().get("warnings", [])) + list(seg["warnings"]),
                error=None,
                message=f"MeshSegNet complete · {len(seg['per_tooth_meshes'])} tooth classes detected.",
            )
        except Exception as exc:
            self._update(phase="complete_with_warnings", stage="segmentation_error",
                         error=str(exc), message="Reconstruction is ready, but segmentation could not complete.")

    def _load_pipeline_script(self):
        script = self.project_root / "scripts" / "run_pipeline.py"
        if not script.is_file():
            raise FileNotFoundError(f"Pipeline script not found: {script}")
        spec = importlib.util.spec_from_file_location("ioscan_dashboard_pipeline", script)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load the reconstruction pipeline module.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _process_scan(self, request: CaptureRequest, session_dir: Path, results_dir: Path) -> None:
        try:
            self._update(phase="processing", stage="reconstructing", progress=0.0,
                         message="Triangulating laser profiles and building the 3D scan…")
            pipeline = self._load_pipeline_script()

            def pipeline_progress(event: dict) -> None:
                updates = {
                    "stage": event.get("stage", "reconstructing"),
                    "progress": float(event.get("progress", 0.0)),
                    "message": event.get("message") or "Reconstructing scan surface…",
                    "processed_frames": event.get("frames_done", self.snapshot()["processed_frames"]),
                    "frames_total": event.get("frames_total", request.frames),
                    "point_count": event.get("triangulated_points", self.snapshot()["point_count"]),
                }
                if event.get("preview_path"):
                    updates["model_asset"] = "preview"
                    updates["model_revision"] = self.snapshot()["model_revision"] + 1
                if event.get("stage") == "mesh_ready":
                    updates["model_asset"] = "mesh"
                    updates["model_revision"] = self.snapshot()["model_revision"] + 1
                self._update(**updates)

            recon_mode = getattr(request, "recon_mode", "sweep")
            stats = pipeline.run_pipeline(
                session_dir=session_dir, out_dir=results_dir, pose_mode=recon_mode,
                mesh_method=request.mesh_method, speed_mm_s=request.speed_mm_s,
                z_range=(request.depth_min_mm, request.depth_max_mm),
                segment=False, progress_callback=pipeline_progress,
                translation=getattr(request, "translation", "auto"),
                scan_length_mm=(float(request.scan_length_mm) or None) if getattr(request, "scan_length_mm", 0) else None,
                auto_trim=bool(getattr(request, "auto_trim", True)),
                auto_depth_from_roi=bool(getattr(request, "auto_depth", True)),
                ridge_secondary_ratio=float(getattr(request, "secondary_ratio", 0.6)),
                ridge_min_prominence=float(getattr(request, "min_prominence", 25.0)),
                # Reprocess applies the current marker-sheet calibration to sessions recorded before it
                calibration_file=(self._active_calibration_file()
                                  if isinstance(request, ProcessRequest) and request.use_calibration else None),
            )
            verdict = stats.get("verdict")
            mesh_path = Path(stats["paths"]["mesh_ply"])
            self._update(
                stage="reconstructed", progress=100.0, model_asset="mesh",
                model_revision=self.snapshot()["model_revision"] + 1,
                mesh_vertices=int(stats["mesh_vertices"]), mesh_faces=int(stats["mesh_triangles"]),
                point_count=int(stats["fused_cloud_points"]), trajectory_mm=round(float(stats["path_mm"]), 2),
                warnings=list(stats["warnings"]),
                outputs={"mesh": str(mesh_path), "cloud": str(stats["paths"]["cloud_ply"])},
                message="3D reconstruction complete.",
                recon_mode=stats.get("pose_mode", recon_mode),
                translation_source=stats.get("translation_source"),
            )
            if stats.get("pose_mode") == "sweep":
                note = (
                    "Sweep reconstruction is a single-pass surface patch. Automatic MeshSegNet was skipped "
                    "because the model expects a full arch; use 'Run / retry MeshSegNet' to try it anyway."
                )
                source = {"markers": "marker plate", "planar": "image tracking", "flow": "image shift",
                          "length": "known length", "speed": "assumed speed"}.get(stats.get("translation_source"), "?")
                done_msg = (f"{(verdict or {}).get('title', '3D surface ready')} · {int(stats['mesh_triangles']):,} faces · "
                            f"motion from {source}.")
                self._update(
                    phase="complete", stage="complete", progress=100.0, error=None, verdict=verdict,
                    warnings=list(stats["warnings"]),
                    message=done_msg,
                )
                return

            if request.auto_segment and not bool(stats.get("quality", {}).get("segmentation_ready_by_basic_geometry_gate", False)):
                quality = stats.get("quality", {})
                reason = (
                    f"Mesh has {int(stats['mesh_triangles']):,} faces and spans "
                    f"{max(quality.get('mesh_extent_mm') or [0.0, 0.0, 0.0]):.2f} mm. "
                    "This is not a usable dental-arch surface, so automatic MeshSegNet was skipped. "
                    "Enable the optical-opening ROI and recapture with overlapping profiles."
                )
                self._update(
                    phase="complete_with_warnings", stage="segmentation_error", progress=100.0,
                    warnings=list(stats["warnings"]) + [reason], error=reason,
                    message="Reconstruction is incomplete; recapture required before tooth segmentation."
                )
            elif request.auto_segment:
                self._update(stage="segmenting", progress=0.0, message="Loading MeshSegNet and segmenting individual teeth…")
                from ioscan.pipeline.segment import SegmentConfig, segment_mesh_and_cloud
                models_dir = Path(request.models_dir).expanduser() if request.models_dir else None
                seg_kwargs = {"jaw": request.jaw}
                if models_dir is not None:
                    seg_kwargs["models_dir"] = models_dir
                seg = segment_mesh_and_cloud(
                    mesh_path, results_dir,
                    cfg=SegmentConfig(**seg_kwargs),
                    cloud_path=Path(stats["paths"]["cloud_ply"]), session_hint=session_dir,
                    progress_callback=lambda e: self._update(
                        stage="segmenting", progress=float(e.get("progress", 0.0)),
                        message=e.get("message", "Running MeshSegNet segmentation…"),
                    ),
                )
                self._update(
                    phase="complete", stage="complete", progress=100.0,
                    model_asset="segmented", model_revision=self.snapshot()["model_revision"] + 1,
                    segmentation=seg,
                    outputs={
                        **self.snapshot()["outputs"],
                        "segmented": seg["full_colored_ply"],
                        "teeth": seg.get("clean_teeth_ply"),
                    },
                    warnings=list(stats["warnings"]) + list(seg["warnings"]),
                    message=f"Scan and tooth segmentation complete · {len(seg['per_tooth_meshes'])} tooth classes detected.",
                )
            elif bool(stats.get("quality", {}).get("segmentation_ready_by_basic_geometry_gate", False)):
                self._update(phase="complete", stage="complete", progress=100.0, message="3D reconstruction complete.")
            else:
                quality = stats.get("quality", {})
                reason = (
                    f"The reconstructed mesh spans only "
                    f"{max(quality.get('mesh_extent_mm') or [0.0, 0.0, 0.0]):.2f} mm and/or has "
                    f"fewer than 5,000 faces. This is a fragment, not a complete dental surface."
                )
                self._update(
                    phase="complete_with_warnings", stage="quality_error", progress=100.0,
                    warnings=list(stats["warnings"]) + [reason], error=reason,
                    message="Reconstruction is incomplete; recapture required."
                )
        except Exception as exc:
            # Keep the reconstruction visible when only the optional segmentation stage fails.
            snap = self.snapshot()
            if snap.get("mesh_faces", 0) > 0 or snap.get("outputs", {}).get("mesh"):
                self._update(phase="complete_with_warnings", stage="segmentation_error",
                             error=str(exc), message="Reconstruction completed, but a later processing stage failed.")
            else:
                self._update(phase="error", stage="error", error=str(exc), message="Reconstruction failed.")


def create_app(project_root: str | Path | None = None) -> FastAPI:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[3]
    controller = ScannerDashboard(root)
    static_dir = Path(__file__).resolve().parent / "static"
    app = FastAPI(title="IOSCAN Operator Console", version="0.1.0")
    app.state.controller = controller
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def dashboard_home():
        # Cache-bust the front-end script: browsers otherwise keep running an old app.js after an
        # update, and new panels in the (always re-read) HTML never come alive.
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        version = int((static_dir / "app.js").stat().st_mtime)
        return html.replace('src="/static/app.js"', f'src="/static/app.js?v={version}"')

    @app.get("/api/status")
    def get_status():
        return controller.snapshot()

    @app.get("/api/camera/devices")
    def camera_devices():
        try:
            return {"devices": enumerate_usb_cameras()}
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/hardware-geometry")
    def get_hardware_geometry():
        return controller.get_hardware_geometry()

    @app.post("/api/hardware-geometry")
    def put_hardware_geometry(payload: HardwareGeometryRequest):
        return controller.save_hardware_geometry(payload)

    @app.get("/api/roi")
    def get_roi():
        try:
            return load_roi_config(controller.roi_config_path)
        except ValueError as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.post("/api/roi")
    def put_roi(payload: dict):
        if controller.snapshot()["phase"] in {"capturing", "processing"}:
            raise HTTPException(409, "ROI cannot be changed while a scan is active.")
        try:
            saved = save_roi_config(controller.roi_config_path, payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return saved

    @app.post("/api/camera/probe")
    def probe_camera(payload: dict):
        try:
            camera_id = int(payload.get("camera_id", 0))
            camera_backend = str(payload.get("camera_backend", "DirectShow"))
            camera_name = str(payload.get("camera_name", "Selected camera"))
        except (TypeError, ValueError):
            raise HTTPException(422, "camera_id must be an integer")
        if camera_backend not in {"DirectShow", "Media Foundation", "Auto"}:
            raise HTTPException(422, "camera_backend must be DirectShow, Media Foundation, or Auto")
        return controller.camera_probe(camera_id, camera_backend, camera_name)

    @app.get("/api/camera/preview.jpg")
    def camera_preview():
        with controller._lock:
            data = controller._latest_jpeg
        if data is None:
            return Response(status_code=204)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store, max-age=0"})

    @app.post("/api/capture/start")
    def start_capture(payload: CaptureRequest):
        return controller.start_capture(payload)

    @app.post("/api/capture/stop")
    def stop_capture():
        return controller.stop_capture()

    @app.get("/api/sessions")
    def list_sessions():
        return {"sessions": controller.list_sessions()}

    @app.post("/api/results/open")
    def open_result(payload: dict):
        return controller.open_result(str(payload.get("name", "")))

    @app.post("/api/process/start")
    def start_processing(payload: ProcessRequest):
        return controller.start_processing(payload)

    @app.get("/api/calibration")
    def get_calibration():
        return controller.calibration_status()

    @app.post("/api/calibration/start")
    def start_calibration(payload: CalibrationRequest):
        return controller.start_calibration(payload)

    @app.get("/api/marker-sheet.pdf")
    def marker_sheet():
        path = controller.project_root / "docs" / "marker_sheet_A4.pdf"
        if not path.is_file():
            raise HTTPException(404, "docs/marker_sheet_A4.pdf is missing; run scripts/make_marker_sheet.py.")
        return FileResponse(path, media_type="application/pdf",
                            headers={"Content-Disposition": 'inline; filename="ioscan_marker_sheet_A4.pdf"'})

    @app.post("/api/stitch/start")
    def start_stitch(payload: StitchRequest):
        return controller.start_stitch(payload)

    @app.post("/api/segmentation/start")
    def start_segmentation(payload: SegmentationRequest):
        return controller.start_segmentation(payload)

    @app.get("/api/model/{asset}")
    def get_model(asset: str):
        names = {
            "preview": "reconstruction_preview.ply",
            "mesh": "reconstructed_mesh.ply",
            "segmented": "segmented_full_colored.ply",
            "teeth": "segmented_clean_teeth.ply",
            "cloud": "fused_cloud.ply",
        }
        filename = names.get(asset)
        if not filename or controller.results_dir is None:
            raise HTTPException(404, "Model is not available.")
        path = controller.results_dir / filename
        if not path.is_file():
            raise HTTPException(404, "Model is not ready yet.")
        return FileResponse(path, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})

    @app.get("/api/model/tooth/{class_id}")
    def get_tooth_model(class_id: int):
        if class_id < 1 or class_id > 14 or controller.results_dir is None:
            raise HTTPException(404, "Tooth class is not available.")
        path = controller.results_dir / "segmented_teeth_by_class" / f"tooth_class_{class_id:02d}.ply"
        if not path.is_file():
            raise HTTPException(404, "This tooth class was not predicted for the current scan.")
        return FileResponse(path, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})

    return app
