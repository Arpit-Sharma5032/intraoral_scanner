# IOSCAN hardware integration audit

> **Latest acquisition/reconstruction review — 7 November 2026:** see
> [`ENGINEERING_REVIEW_AND_FIXES.md`](ENGINEERING_REVIEW_AND_FIXES.md) for the audited session numbers,
> low-FPS/flat-mesh root causes, code changes, test results and hardware acceptance checklist.
> The previous notes below describe earlier integration work; read the latest review where details differ.

## Overall assessment

The acquisition, reconstruction, and MeshSegNet software stages are connected in the dashboard workflow. The ESP32 CSV protocol is compatible with the Python capture reader. The codebase is **not yet verified as hardware-ready**: no physical OV2311/ESP32/BNO086 capture, timing test, or firmware upload has been performed in this build environment.

## Changes in this update

- Added the provided ESP32 reference sketch under `firmware/esp32_scanner/` and prepared a stricter version for the host protocol.
- Changed serial sampling from `latest()` polling to FIFO event consumption, so a serial event cannot be silently reused for multiple camera images.
- Added extension of ESP32's 32-bit `micros()` counter across wraparound.
- Added support for the seven-field packet with a strobe frame sequence, backwards-compatible with the original six-field CSV packet.
- Added strobe sequence-gap and event-queue-overflow errors.
- Opened the camera before the serial reader, clears stale serial bytes at COM-port open, and requests a one-frame UVC buffer; pairs each queued strobe event with the next OpenCV frame.
- Made the bundled firmware start with an OFF exposure, then alternate OFF/ON; it forces the laser output low when STROBE disappears.
- Disabled the firmware's old synthetic free-running fallback. Missing camera STROBE now causes a warning and no scan packets rather than fake synchronization.
- Made the firmware refuse scan packets if the BNO086 fails to initialize or its orientation report becomes stale; this prevents the host from accepting a default identity quaternion as valid tracking data.
- Added dashboard preflight loading of the selected 15-class checkpoint before capture starts.
- Added mocked tests for serial packet ordering, `micros()` rollover, sequence gaps, firmware errors, the legacy six-field packet, and capture frame/event pairing.

## Verified in software

- `python -m compileall -q src scripts` passes.
- `python -m pytest -q -rs`: **44 passed, 5 skipped**.
- `run_capture.py --help`, `run_pipeline.py --help`, `run_segment.py --help`, and `run_dashboard.py --help` run.
- Dashboard JavaScript syntax check passes with Node.js.

The five skipped tests are three Teeth3DS dataset tests (`IOSCAN_TEETH3DS` is not configured) and two Open3D-dependent geometry tests (Open3D is not installed in this environment).

## Still required before a real scan

### 1. Verify camera-to-strobe frame alignment

The host consumes the next serial strobe event and then reads the next OpenCV frame, while requesting `CAP_PROP_BUFFERSIZE=1`. This preserves event order and is better than the previous latest-value sampling, but OpenCV's current UVC interface does not expose the OV2311 sensor frame ID/timestamp here. The actual Windows driver may still buffer frames. Verify the laser OFF/ON state in captured images against the firmware packet sequence on the real camera. For strict timing, integrate camera-provided hardware timestamps/frame IDs through the vendor SDK or UVC metadata.

### 2. Measure camera and laser calibration

Use actual 1600x1200 camera intrinsics (including distortion) and fit the laser plane with a measured calibration target and multiple poses. The example numbers in `configs/camera.yaml`, `configs/laser.yaml`, and README examples are nominal only. A pipeline run with nominal calibration is a software smoke test, not a geometrically reliable scan.

### 3. Calibrate the IMU mounting transform

The current pose code consumes BNO086 quaternions as if they directly represent camera orientation. Measure or mechanically align the BNO086-to-camera transform and establish the required initial reference frame. Firmware packet format currently sends the raw quaternion, not a calibrated camera-frame quaternion.

### 4. Add/validate translational tracking

The BNO086 orientation stream does not measure translation. `imu_hybrid` assumes a constant configured scan speed along camera +Y. Handheld motion will vary, so this can distort scale and profile placement. For reliable geometry, add an encoder or validated visual-odometry/other translational measurement and integrate it into `PoseTracker`.

### 5. Verify scan coverage and registration

A single laser-line pass cannot observe all tooth surfaces. Full tooth coverage needs overlapping occlusal, buccal and lingual passes with registration into a common frame. The current dashboard records one continuous session; it does not yet provide a validated multi-pass registration/loop-closure workflow.

### 6. Complete dashboard operator tools

The Calibration and Scan Library navigation entries are placeholders/informational controls. Calibration is currently launched from the CLI, and prior sessions are stored on disk but are not yet browsable/restorable from a full in-dashboard library.

### 7. Flash and validate firmware on the actual board

This environment did not have `arduino-cli`/PlatformIO, so the supplied sketch has not been compiled, flashed, or electrically tested. Confirm the actual ESP32 variant, pin mapping, BNO08x library version, camera STROBE polarity/logic levels, exposure timing and laser safety interlocks against the physical schematic and component datasheets before enabling the optical hardware.

## Suggested first hardware acceptance test

Before scanning a person or attempting a full arch, run a short calibration-target capture and check:

- dashboard camera probe reports the expected 1600x1200 image;
- ESP32 reports successful BNO086 initialization;
- strobe packets arrive at the expected camera rate with no sequence gaps;
- saved images alternate OFF, ON, OFF, ON in the same sequence as `laser_on`;
- all frames have monotonically increasing timestamps and valid quaternions;
- laser-line extraction yields a stable ridge in the ON-minus-OFF images;
- triangulated points occupy a physically plausible range under measured calibration;
- the short reconstructed point cloud follows the calibration target without folded/ribbon-like geometry.

Only after this acceptance test passes should you attempt a full dental arch and evaluate the tooth segmentation output.

This remains an engineering development system, not a validated clinical or diagnostic device.
