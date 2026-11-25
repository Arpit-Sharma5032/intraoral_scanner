# Intraoral Scanner — laser triangulation and 3D reconstruction

This project captures a moving laser line, triangulates each line into camera-frame 3D points,
places those profiles into a shared coordinate frame, and exports a point cloud and surface mesh.
All camera/laser/reconstruction distances are in **millimetres**.

> **Use the marker plate.** Motion is measured from a printed marker sheet under the model (full pose in every
> frame, no drift), and the same sheet calibrates the camera and laser. Read `MARKER_PLATE_GUIDE.md` first.
>
> **Reconstruction default: linear sweep.** A single laser line cannot be placed reliably from IMU orientation
> plus an assumed speed. The default pipeline now measures translation from the laser-OFF frames, trims to the
> on-surface window and meshes a height map. Read `RECONSTRUCTION_APPROACH.md` before scanning. The Open3D TSDF integration
internally converts lengths to metres because that API expects metre-scale depth data.

## Environment

Use Python 3.11 on Windows (Open3D wheels are most reliable there):

```powershell
conda create -n ios_scan python=3.11 -y
conda activate ios_scan
python -m pip install -e ".[dev]"
```

Tooth segmentation is optional and needs PyTorch plus a MeshSegNet checkpoint (not included):

```powershell
python -m pip install -e ".[dev,seg]"
$env:IOSCAN_MESHSEGNET_MODELS = "C:\path\to\MeshSegNet\models"   # or pass --models-dir / --ckpt
```

For live ESP32 capture, also install the serial extra:

```powershell
python -m pip install -e ".[hardware]"
```

Run the tests from the repository root:

```powershell
$env:PYTHONPATH = "src"
pytest -q
```

## Run a reconstruction

For the included Teeth3DS simulator session:

```powershell
python scripts/run_pipeline.py data/sessions/FUYJPUPX_lower `
  --pose-mode imu_hybrid --mesh bpa --init-from-gt `
  --out data/results/FUYJPUPX_hybrid_fixed

python scripts/run_eval.py data/results/FUYJPUPX_hybrid_fixed `
  --session data/sessions/FUYJPUPX_lower --no-icp

python scripts/view_results.py data/results/FUYJPUPX_hybrid_fixed
```

`--init-from-gt` (simulator only) starts the camera at the ground-truth position so that `--no-icp`
evaluation compares absolute pose; omit it for real captures. Evaluation uses the session's
`gt/reference_mesh.ply` when present; without a reference mesh it falls back to a circular
reference built from the scan's own laser samples and says so.

The pipeline writes `fused_cloud.ply`, `reconstructed_mesh.ply`, `reconstructed_mesh.stl`,
`estimated_poses.npz`, and `pipeline_report.json`. The report includes the calibration, trajectory length and
warnings about capture coverage and pose assumptions. Pose orientation and timestamps are read from
`imu.csv` and the `t_ns` column of `frames.csv`; a session without IMU data fails in the IMU pose
modes instead of using a made-up orientation, and a session without calibration fails instead of
falling back to nominal values.

`--no-icp` now means **no rigid alignment at all**: evaluation compares the reconstructed cloud
in its original world coordinates. Without this option, evaluation performs centroid initialization
and then multiscale point-to-plane ICP, which is useful for shape-only comparisons but hides some
absolute pose error.

## Pose modes and their limits

- `imu_hybrid` uses the timestamped IMU orientation and integrates the configured nominal scan
  speed along the camera's local +Y axis. The local axis follows a curved dental arch. This is the
  recommended baseline for the current line scanner.
- `imu_const_vel` integrates a fixed world-space velocity. It is useful only for straight-line
  motion and should not be used for the curved arch trajectory.
- `gt` uses simulator ground-truth camera poses. It is an upper-bound/debug mode, not a mode for
  real captures.
- `submap_icp` is retained as a deprecated command-line alias for `imu_hybrid`. A single laser
  profile is only a 1D curve; unconstrained rigid ICP on each profile does not determine a reliable
  six-degree-of-freedom pose and can fold the accumulated surface.

A gyroscope/IMU orientation stream does **not** measure translation. `imu_hybrid` assumes a speed;
for real hand-held scanning that assumption is not metrologically accurate unless speed is measured
or otherwise constrained. The IMU-to-camera axes must also be mechanically aligned or calibrated.
For accurate scale and geometry, add an encoder or reliable visual odometry and calibrate the IMU
mounting transform.

## Interactive operator dashboard

The project includes a local browser dashboard for the hardware workflow. It provides a live USB camera feed, capture/IMU status, a growing fused point-cloud preview during reconstruction, the final 3D mesh, and the MeshSegNet-labelled model after inference. No synthetic camera footage is shown when hardware is disconnected.

Install the dashboard and hardware extras (MeshSegNet is optional if you want reconstruction only):

```powershell
python -m pip install -e ".[dashboard,hardware]"
# Also install the segmentation extra when using MeshSegNet:
python -m pip install -e ".[seg]"
python scripts/run_dashboard.py
```

Open `http://127.0.0.1:8787` on the scanner workstation. Select the physical Arducam camera by its enumerated device name/backend and the ESP32 serial port (for example `COM3`), choose the jaw, and optionally supply the measured camera/laser `calibration.json` and MeshSegNet model directory. Press **Check camera**, then **Start scan workflow**. On Windows, FFmpeg must also be installed on the dashboard process PATH. If OpenCV DirectShow cannot sustain the required MJPG frame rate, IOSCAN now falls back to FFmpeg DirectShow using the exact selected camera name and checks the actual delivered FPS. See `CAMERA_FPS_FIX.md` for setup and exposure-validation notes. The dashboard records the session under `data/sessions/`, updates a live fused point-cloud preview while profiles are being captured, then runs the full reconstruction into `data/results/`. When enabled, MeshSegNet runs automatically after reconstruction. The **Run / retry MeshSegNet** control can also start segmentation later without repeating capture or reconstruction. The stop button ends the capture early and then processes the captured session.

The camera/laser calibration file is strongly recommended for real scans. Without it, the capture logger uses nominal example intrinsics and laser-plane geometry; output dimensions are not suitable for measurement. A valid ESP32 quaternion stream is required by the current `imu_hybrid` reconstruction mode. The browser 3D viewer loads Three.js modules from jsDelivr, so the workstation needs internet access for the interactive 3D viewport; the scanner capture and reconstruction backend remain local. The server binds to `127.0.0.1` by default and should not be exposed to a network without adding authentication and access controls.

## Tooth segmentation (MeshSegNet)

The integrated model supports the two 15-class reference checkpoints: `Man` for the lower jaw and
`Max` for the upper jaw. Put the supplied archives in the same directory, preserving their names:

- `MeshSegNet_Man_15_classes_72samples_lr1e-2_best.zip`
- `MeshSegNet_Max_15_classes_72samples_lr1e-2_best.zip`

In PowerShell, configure that directory once (use the folder containing the two files, not the
individual files):

```powershell
$env:IOSCAN_MESHSEGNET_MODELS = "C:\Users\arpit\MeshSegNet\models"
```

Run reconstruction and segmentation in one command. The jaw is read from the input session's
`meta.json`, so the lower or upper checkpoint is selected automatically:

```powershell
python -m pip install -e ".[seg]"
python scripts/run_pipeline.py data/sessions/my_scan --segment --out data/results/my_scan
```

To segment an existing reconstruction without rebuilding it:

```powershell
python scripts/run_segment.py data/results/my_scan --session data/sessions/my_scan
```

You can also select the jaw/checkpoint explicitly:

```powershell
python scripts/run_segment.py data/results/my_scan --jaw lower `
  --ckpt "C:\Users\arpit\MeshSegNet\models\MeshSegNet_Man_15_classes_72samples_lr1e-2_best.zip"
```

`.zip` may be a PyTorch ZIP checkpoint or an archive containing one `.tar`, `.pth`, `.pt`, or
`.ckpt` file. The loader uses safe `weights_only` loading and strictly validates the state dictionary
against the 15-class network. The bundled MeshSegNet implementation has the layer/state-dict names
of the public reference implementation; `--repo-dir` can be supplied only when a custom external
reference implementation is actually needed. By default, inference uses CUDA when available and
retries on CPU after a CUDA out-of-memory error. Use `--seg-device cpu` on `run_pipeline.py` or
`--device cpu` on `run_segment.py` to force CPU inference.

Outputs include:

- `segmented_full_colored.ply`: all input mesh faces coloured by the predicted class (0 = gingiva,
  1-14 = individual teeth).
- `segmented_clean_teeth.ply` and `.stl`: all predicted tooth faces combined.
- `segmented_teeth_by_class/tooth_class_XX.ply` and `.stl`: one mesh per predicted tooth class.
- `segmented_cloud.ply`: reconstructed points coloured using the same predictions, when a cloud is
  available.
- `segmentation_labels.npz`: face labels, confidence values, selected checkpoint, and jaw.

**Important geometry limitation:** segmentation labels the surface that the scanner reconstructed;
it does not invent missing teeth or unseen tooth walls. The included `FUYJPUPX_lower` example is a
roughly 20 mm local scan path, not a full arch, and a single occlusal pass cannot produce complete
360-degree tooth surfaces. For a full-mouth result, the input reconstruction must contain overlapping
occlusal, buccal and lingual coverage from additional scan passes registered into one mesh. Also,
MeshSegNet is orientation-sensitive and expects full dental surfaces; the tool reports a warning for
small/partial scans and does not claim that the resulting tooth numbering is clinically validated.

## Real hardware capture

Prepare a calibration JSON with this shape:

```json
{
  "camera": {
    "width": 1600, "height": 1200,
    "fx": 2666.7, "fy": 2666.7, "cx": 800, "cy": 600,
    "dist_coeffs": [0, 0, 0, 0, 0], "pixel_size_um": 3.0
  },
  "laser": { "normal": [0, -0.8575, -0.5144], "d": 10.2903 }
}
```

The numbers above are only an example/nominal geometry, **not a substitute for calibration**.
Use checkerboard-derived camera intrinsics for the actual lens and a fitted laser plane from
measured 3D calibration points. A lens change requires a new camera calibration. Follow
[`CALIBRATION_GUIDE.md`](CALIBRATION_GUIDE.md) to capture the checkerboard session and generate
`data/calibration/calibration.json` with `scripts/run_calibrate.py`. For example:

```powershell
python scripts/run_capture.py --cam 0 --port COM3 --frames 300 `
  --calibration path\to\calibration.json --out data/sessions/my_scan

python scripts/run_pipeline.py data/sessions/my_scan `
  --pose-mode imu_hybrid --mesh bpa --speed-mm-s 5.0 `
  --out data/results/my_scan
```

The logger writes the shared session format (`meta.json`, `frames.csv` with `idx,t_ns,laser_on,file`,
8-bit PNG frames and `imu.csv`) and expects framed ESP32 packets in the format
`@timestamp_us,laser_on,qw,qx,qy,qz,frame_sequence*CRC16` at 115200 baud. CRC-16/CCITT-FALSE
checks each packet before use. Older unframed six/seven-field CSV firmware remains accepted for
backwards compatibility, but malformed lines are discarded and packet sequence gaps stop capture.
With a serial port, all timestamps come from the MCU clock and capture waits for the first valid IMU
packet; without one, only the host clock is used. If no serial port is provided, the logger classifies laser
ON/OFF frames from image intensity, but that does not provide camera orientation and the pipeline
will not run in an IMU pose mode without quaternion data.

## What the current simulator can reconstruct

Simulator v1 executes a **single top-down/occlusal pass** along a short section of the arch. Its own
trajectory documentation states that buccal and lingual walls are not visible in this pass. It
therefore cannot produce a complete, closed 360-degree tooth from this session, regardless of the
meshing algorithm. The expected output is a partial occlusal surface patch. Full tooth-by-tooth
geometry requires additional scans from different approach angles (occlusal, buccal, lingual, and
where possible interproximal) with overlapping geometry and calibrated registration.

The supplied `FUYJPUPX_lower` recording contains 200 frames (100 laser-on frames) and a camera path
of about 19.9 mm. Its trajectory plot shows that this recording covers only a local segment on the
right side of the arch, not the full jaw. The old session metadata did not preserve the command-line
frame cap used when it was generated. To create a full-length **single occlusal pass**, regenerate
without `--max-frames` and write to a new session directory:

```powershell
python scripts/run_sim.py --root "C:\path\to\Teeth3DS" `
  --case FUYJPUPX --jaw lower `
  --out data/sessions/FUYJPUPX_lower_full
```

Newly generated sessions record the number of frames and the actual path length in `meta.json`.
The single-pass limitation still applies: a full-length pass covers the arch from above, but does
not reveal buccal/lingual walls. The sample's line extraction and triangulation are accurate against
the simulator's stored profiles; most of the visible distortion in the previous result came from
the motion/fusion stage. The corrected pose path follows the measured arch tangent using each IMU
orientation rather than one constant world velocity.

## Repository layout

| Path | Purpose |
|---|---|
| `src/ioscan/session` | Common simulator/hardware session format |
| `src/ioscan/sim` | Teeth3DS virtual scanner, laser model, trajectory, and IMU simulation |
| `src/ioscan/calibration` | Camera intrinsics, laser plane model, and calibration fitting |
| `src/ioscan/capture` | Live USB-camera and ESP32 serial logger |
| `src/ioscan/pipeline` | Ambient subtraction, laser ridge, triangulation, pose, fusion, and meshing |
| `src/ioscan/eval` | Cloud-to-mesh deviation and scanned-ROI coverage metrics |
| `configs/` | Camera, laser, simulator, and pipeline defaults |
| `scripts/` | Capture, simulation, reconstruction, evaluation, and viewing entry points |

Teeth3DS is not redistributed. Download it under `data/teeth3ds/` and observe its license.

## ESP32 firmware and hardware bring-up

The ESP32 sketch supplied for this project is included at
`firmware/esp32_scanner/esp32_scanner.ino`. The current sketch uses the BNO086 over I2C
(SDA GPIO21, SCL GPIO22 at 400 kHz); it is not an SPI firmware build. Read
`firmware/esp32_scanner/README.md` before uploading it.

The updated host parser accepts the legacy unframed six/seven-field CSV packet and the new
framed packet:

```text
@timestamp_us,laser_on,qw,qx,qy,qz,frame_sequence*CRC16
```

The firmware and host both use 115200 baud. The CRC is CRC-16/CCITT-FALSE over the CSV payload.
The updated sketch emits the framed form, begins with a laser-OFF exposure so the first
laser-ON frame has an ambient reference, checks that the BNO086 is initialized and recently
updated, and stops emitting scan records when camera STROBE is absent. The Python capture code
consumes serial packets in FIFO order, extends the ESP32 `micros()` counter across uint32 wrap,
and aborts on detected serial packet sequence gaps or queue overrun.

Before a physical scan:

1. Compile and flash the included sketch with the Arduino ESP32 core and Adafruit BNO08x library
   versions selected for the exact board.
2. Confirm the firmware reports BNO086 initialization and that camera STROBE events are present.
   The sketch intentionally will not fabricate free-running frame packets if STROBE is missing.
3. Close Arduino Serial Monitor before launching the dashboard; only one process can own the COM port.
4. Use **Check camera** in Setup. The scan path now requires a verified 1600x1200 MJPG stream and
   at least 85% of the configured acquisition rate (42.5 FPS for a 50 FPS target). It reads back
   the actual FOURCC and measures real frame arrivals; a successful `cap.set()` or `CAP_PROP_FPS`
   value alone is not accepted. If it fails, try the other Windows backend, close other camera apps,
   and verify the UVC mode with the Arducam utility.
5. Use measured camera intrinsics and a measured laser plane in `calibration.json`; example values
   are only for software smoke tests.

The host pairs the next queued strobe event with the next frame returned by OpenCV and requests a
one-frame camera buffer. Frame PNG writes run on a bounded background writer so disk encoding does
not block camera acquisition; if storage cannot keep up, capture stops rather than silently dropping
saved frames. This is an ordered pairing strategy, but OpenCV does not expose the OV2311
sensor's frame counter/timestamp here. A camera/firmware timing test is still required to verify
there is no one-frame offset or driver buffering on the actual Windows camera. If the camera vendor
provides UVC metadata/frame timestamps, integrating that SDK is preferable for strict synchronization.

The current pose estimate still assumes constant hand speed because the BNO086 supplies orientation,
not translation. The IMU-to-camera mounting transform must also be measured or mechanically aligned.
For geometrically reliable scans, add a measured translation source (encoder or validated visual
odometry), calibrate the sensor extrinsics, and validate coverage/registration on real multi-pass scans.
The camera/laser calibration workflow currently uses the documented CLI process in
`CALIBRATION_GUIDE.md`; enter the generated JSON path in the dashboard Setup tab. The Scan Library
items remain informative placeholders and calibration results are stored in the
project's `data/results/` directory.

The firmware has not been compiled/flashed or electrically tested as part of this source-code build.
Hardware strobe polarity, electrical levels, laser safety/interlocks and the board-specific pin map
must be verified by a qualified hardware integrator against the actual schematic and component datasheets.

## High-speed scan and geometry quality gates

The 1600 × 1200 hardware scan path now requires an actual MJPG stream, checked after real camera
frames arrive, and rejects a measured acquisition rate below 85% of the configured target. The
requested rate and actual session rate are stored separately in `meta.json`. Frame PNG encoding is
performed by a bounded background writer; a full queue stops acquisition rather than silently
dropping frames.

`run_pipeline.py` reports measured capture FPS, ON-profile rate, points per usable profile, assumed
profile spacing, mesh extent, ROI state, and a basic segmentation eligibility result. The
segmentation eligibility floor (5,000 faces and at least 35 mm maximum mesh extent) is an IOSCAN
engineering guardrail, not a validated MeshSegNet model requirement. A scan that fails it is not
labelled as a successful dental reconstruction and is not automatically sent to MeshSegNet.

The BNO086 provides orientation; the current `imu_hybrid` pose mode still assumes a configured
constant translation speed. It is suitable only for controlled-motion experiments until an encoder
or validated visual-odometry translation source and the IMU-to-camera extrinsic transform are
calibrated. A single pass observes only the visible laser-swept surface; buccal/lingual coverage and
registered multi-pass fusion remain necessary for a complete tooth/arch model.
