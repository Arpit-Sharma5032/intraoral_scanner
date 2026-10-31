from __future__ import annotations

# Prefer this checkout's src/ package over any stale ioscan installation in site-packages.
import sys as _sys
from pathlib import Path as _Path
_project_src = _Path(__file__).resolve().parents[1] / "src"
if _project_src.is_dir():
    _sys.path.insert(0, str(_project_src))
import argparse
import json
from pathlib import Path
from ioscan.capture.logger import CaptureConfig, record_hardware_session


def main() -> None:
    ap = argparse.ArgumentParser(description="Record live OV2311 camera + ESP32 BNO086 IMU session to disk.")
    ap.add_argument("--out", type=Path, required=True, help="Output session directory")
    ap.add_argument("--cam", type=int, default=0, help="USB UVC camera index")
    ap.add_argument("--port", type=str, default=None, help="ESP32 Serial port (e.g. COM3)")
    ap.add_argument("--frames", type=int, default=300, help="Number of frames to capture")
    ap.add_argument("--exposure", type=int, default=-7, help="UVC manual exposure value")
    ap.add_argument("--calibration", type=Path, default=None, help="Optional path to calibration.json")
    ap.add_argument("--preview", action="store_true", help="Show live OpenCV HUD with detected laser ridge")
    args = ap.parse_args()

    cam_cal, las_cal = None, None
    if args.calibration and args.calibration.exists():
        cal_data = json.loads(args.calibration.read_text(encoding="utf-8"))
        cam_cal = cal_data.get("camera")
        las_cal = cal_data.get("laser")

    cfg = CaptureConfig(
        camera_id=args.cam,
        serial_port=args.port,
        max_frames=args.frames,
        exposure=args.exposure,
        show_preview=args.preview,
    )
    out = record_hardware_session(args.out, cfg=cfg, camera_cal_dict=cam_cal, laser_cal_dict=las_cal)
    print(f"Hardware session recorded to: {out}")


if __name__ == "__main__":
    main()