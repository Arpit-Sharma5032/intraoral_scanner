# Camera + laser calibration (creating `calibration.json`)

## What it calibrates

This file contains two measured parts:

- `camera`: image width/height, focal lengths (`fx`, `fy`), optical centre (`cx`, `cy`), and lens-distortion coefficients. These convert image pixels into camera rays.
- `laser`: the laser-sheet plane in camera coordinates, stored as a unit normal and offset `d` in millimetres. Intersecting a camera ray with this plane gives a 3D point.

This is different from the Camera ROI polygon. ROI says *which image pixels to ignore*; camera/laser calibration says *where a retained pixel lies in 3D*. This file does not calibrate IMU-to-camera rotation or scanner translation/speed.

Do not use the nominal values in `configs/camera.yaml` or `configs/laser.yaml` as a real calibration. They are startup/example geometry only.

## Calibration target

The current tool defaults to a checkerboard with **9 × 6 internal corners** (that means 10 × 7 squares) and **2.00 mm per square**. Use a matte, flat, rigid print. Print at 100% / actual size, turn off "fit to page", and measure a square with a ruler or caliper. If your squares are a different measured size, pass that size to `--square-mm` exactly.

Keep the scanner camera at its final 1600 × 1200 resolution, with the final lens and focus. Do not move the lens or camera after calibration.

## Capture a calibration session

1. Connect the camera and laser. Connect the ESP32 serial port if available; this gives the most reliable ON/OFF labels. The calibration math itself does not require IMU orientation, but the capture tool can use the serial strobe events to label paired frames.
2. Place the checkerboard inside the scanner's usable working volume. Make sure the whole 9 × 6 internal-corner pattern is visible and the laser stripe crosses the board clearly.
3. Capture several different board poses: centre, left/right and top/bottom of the usable image; near and far working distances; and a range of tilts/rotations. The board must remain still between each laser-OFF frame and its paired laser-ON frame. Move it between pairs/pose groups, not during a pair.
4. Collect enough varied poses that the laser stripe lies along different 3D lines on the checkerboard. A single pose is not enough to determine a unique laser plane.

Example capture command (replace camera index/backend and COM port with the selected Arducam and your ESP32 port):

```powershell
python scripts/run_capture.py --cam 1 --backend DirectShow --camera-name "Arducam" --port COM3 --frames 600 --out data/sessions/calibration_board
```

If the ESP32 is not connected, omit `--port COM3`; in that case laser ON/OFF is inferred from image intensity, so verify that the saved `frames.csv` alternates sensibly and that the stripe is bright enough. For the most reliable calibration, use the synchronized ESP32 strobe.

## Generate the JSON

From the project root:

```powershell
python scripts/run_calibrate.py --session data/sessions/calibration_board --cols 9 --rows 6 --square-mm 2.0 --out data/calibration/calibration.json
```

The program detects the checkerboard in laser-OFF frames to calculate camera intrinsics, then subtracts paired OFF frames from ON frames, finds the laser ridge on the board, estimates 3D calibration points, and fits the laser plane. It saves `data/calibration/calibration.json` only after both camera and laser calibration succeed.

The calibration run reports camera reprojection RMS in pixels and laser-plane RMS in micrometres. If it reports too few checkerboard detections, no laser points, or nearly collinear points, do not use a calibration file from that run. Capture more clear, varied board poses and retry.

## Use it in the dashboard

Open **Setup → Scan → Calibration JSON** and enter the full path, for example:

```text
C:\path\to\your-project\data\calibration\calibration.json
```

The dashboard reads that file at the start of a scan. The session also stores a copy of the calibration values in its metadata so that later reconstruction remains reproducible.

## Before trusting the result

- Confirm the JSON camera size is 1600 × 1200.
- Confirm `camera_rms_px` is finite and preferably comfortably below 1 pixel; large values suggest poor checkerboard detections, blur, glare, or inaccurate print scale.
- Check `laser_plane_rms_um` and inspect reconstructed dimensions using an independent object of known size.
- Recalibrate after changing the lens, focus, camera mounting, camera resolution, or laser mounting/alignment.
- Calibration is an engineering measurement step, not proof of clinical accuracy.
