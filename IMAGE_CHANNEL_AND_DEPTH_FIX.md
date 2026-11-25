# JPEG channel decoding and laser-depth validation fix

## What the uploaded session showed

- 1,200 frames saved at a session average of 49.331 FPS.
- All 1,200 source MJPEG frames were written as JPEG; maximum storage queue depth was one frame.
- The reconstruction error `Expected a single-channel ON image, got shape (1200, 1600, 3)` came from reading the JPEG as BGR with `IMREAD_UNCHANGED` in the offline reconstruction path.
- The scan metadata records nominal geometry with a 25 mm baseline and 6 degree tilt, while the selected depth filter is 10–45 mm. This plane's positive intersections over sampled sensor rays lie outside the configured depth band (minimum approximately 75.8 mm; optical-axis crossing approximately 237.9 mm). No point in the selected 10–45 mm band can be generated from that nominal geometry.
- The supplied ROI configuration is disabled. The chassis/interior is therefore not masked.

## Changes

- Added `read_grayscale_frame()` to load color-decoded JPEGs as grayscale while preserving 16-bit simulator PNGs.
- Updated offline reconstruction, session reader, and benchmark paths to use the grayscale loader.
- Made `triangulate_frame()` defensively accept BGR/BGRA images and convert them to grayscale before ambient subtraction and ridge extraction.
- Added laser-plane/depth-range overlap diagnostics. The offline pipeline now stops early with the predicted Z range if the stored laser plane cannot intersect the configured depth window.
- Added live dashboard warnings for nominal calibration, depth-range mismatch, and disabled ROI.
- Increased the low-priority live-preview queue from 6 to 24 frames to better tolerate short processing spikes; it remains a lossy preview queue and does not affect saved capture frames.

## Important hardware/configuration note

Do not blindly change baseline, tilt, or depth limits to force a point cloud. Measure the actual camera-to-laser baseline, laser-plane tilt, and tooth working distance, then create measured calibration. A nominal geometry warning is not a calibration substitute.
