"""Diagnose OpenCV access to USB cameras without hanging the whole script.

Run from the activated project environment after closing Windows Camera and other
apps that may be using the scanner camera:

    python scripts/diagnose_camera.py --max-index 5

Each backend/index test runs in a child process. A blocked Windows camera read is
terminated after the configured timeout and reported rather than freezing the tool.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from typing import Any


def _camera_trial(index: int, backend: int, backend_name: str,
                  width: int, height: int, fps: int, conn: Any) -> None:
    import cv2

    cap = None
    try:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            conn.send({"ok": False, "stage": "open", "detail": "device did not open"})
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        started = time.monotonic()
        ok, frame = cap.read()
        elapsed = time.monotonic() - started
        if not ok or frame is None or frame.size == 0:
            conn.send({"ok": False, "stage": "read", "detail": f"no frame; read took {elapsed:.2f}s"})
            return
        conn.send({
            "ok": True,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "shape": tuple(int(v) for v in frame.shape),
            "dtype": str(frame.dtype),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "fourcc": int(round(cap.get(cv2.CAP_PROP_FOURCC))),
            "read_seconds": round(elapsed, 3),
        })
    except Exception as exc:
        try:
            conn.send({"ok": False, "stage": "exception", "detail": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
    finally:
        if cap is not None:
            cap.release()
        conn.close()


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description="Find which OpenCV backend can read the scanner camera.")
    parser.add_argument("--index", type=int, help="Test only this OpenCV camera index.")
    parser.add_argument("--max-index", type=int, default=5, help="Test indices 0 through this value (default: 5).")
    parser.add_argument("--timeout", type=float, default=7.0, help="Seconds allowed for each open/read test (default: 7).")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.timeout <= 0 or args.max_index < 0:
        parser.error("--timeout must be positive and --max-index cannot be negative")

    if os.name == "nt":
        backends = [("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF), ("Auto", cv2.CAP_ANY)]
    else:
        backends = [("Auto", cv2.CAP_ANY)]
    indices = [args.index] if args.index is not None else list(range(args.max_index + 1))
    ctx = mp.get_context("spawn")

    print(f"OpenCV {cv2.__version__}; requested mode {args.width}x{args.height}@{args.fps}")
    print("Close the Windows Camera app and any other app using the scanner before this test.\n")
    found = False
    for index in indices:
        for backend_name, backend in backends:
            recv_conn, send_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_camera_trial,
                args=(index, backend, backend_name, args.width, args.height, args.fps, send_conn),
            )
            proc.start()
            send_conn.close()
            proc.join(args.timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(1.0)
                result = {"ok": False, "stage": "timeout", "detail": f"no response within {args.timeout:.1f}s"}
            elif recv_conn.poll():
                result = recv_conn.recv()
            else:
                result = {"ok": False, "stage": "process", "detail": f"worker exited with code {proc.exitcode}"}
            recv_conn.close()
            prefix = f"index={index} backend={backend_name}:"
            if result.get("ok"):
                found = True
                print(f"[FRAME] {prefix} {result['width']}x{result['height']} "
                      f"shape={result['shape']} dtype={result['dtype']} "
                      f"fps={result['fps']:.1f} read={result['read_seconds']:.3f}s")
            else:
                print(f"[FAIL]  {prefix} stage={result.get('stage')} {result.get('detail')}")

    print("\n" + ("At least one backend returned a frame. Use the matching index in the dashboard." if found
                    else "No tested index/backend returned a frame. Check the Arducam USB connection/driver and try the vendor camera utility."))
    return 0 if found else 2


if __name__ == "__main__":
    raise SystemExit(main())
