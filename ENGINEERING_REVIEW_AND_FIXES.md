# IOSCAN engineering review: 5 FPS capture and collapsed 3D mesh

**Review date:** 7 November 2026  
**Scope:** Uploaded `intraoral-scanner-hardware-integration` project ZIP, its saved 350-frame hardware session and result folder, the separate dashboard screenshot (1,200-frame run), firmware, capture, reconstruction, segmentation, and dashboard code.

## Executive assessment

The 5 FPS result is explained by a confirmed camera-mode mismatch: the session requested MJPG but the camera negotiated YUY2 at 1600 x 1200. The saved timestamps show that the camera/strobe events were actually arriving at about 5.2 FPS. The metadata field called `fps` was the requested value (50), not the measured rate.

The collapsed/near-2D mesh is a separate reconstruction failure caused by sparse laser profiles being placed too far apart under an assumed motion model. The supplied BNO086 orientation stream does not measure translation. The pipeline assumes a constant 5 mm/s movement, and the BPA mesher cannot bridge the profile gaps implied by the observed 5 FPS acquisition rate. Only about 31 points were retained per usable laser profile in the saved session. The resulting mesh contains 18 vertices and 22 triangles and is practically coplanar.

Changing I2C to SPI or adding a strobed ring light does not address either confirmed root cause. The IMU bus may be optimized later, but current evidence points to camera format negotiation as the FPS limiter. A ring light cannot create missing camera frames and can reduce laser-line contrast.

The software can now detect and reject the invalid camera mode, protect capture from synchronous image-write stalls, report reconstruction-quality metrics, prevent tiny fragments from being sent to MeshSegNet, and remove stale results. A physically correct handheld 3D scan is **not yet solved by software alone**: translational tracking, measured calibration, actual laser/strobe timing, and scan coverage still need hardware validation or new sensing.


## Follow-up: confirmed 49 FPS in FFmpeg and IOSCAN fallback patch

After the original review, the operator ran the DirectShow camera independently with FFmpeg using 1600x1200, 50 FPS, and MJPG. The reported output was `frame=750`, `time=00:00:15.00`, `fps=49`. This demonstrates that the physical camera/USB/Windows DirectShow stack can deliver approximately 50 FPS through FFmpeg on that workstation. It narrows the 25.2 FPS IOSCAN result to the OpenCV capture/configuration path rather than proving a hardware bandwidth ceiling.

The follow-up patch retains OpenCV as the first capture attempt. When the explicit DirectShow + MJPG path fails the real-frame-rate threshold, and the selected physical camera name is available, IOSCAN falls back to FFmpeg's DirectShow input with the same explicit mode settings as the successful standalone command. It streams JPEG packets over a pipe and decodes them to grayscale for the monochrome processing pipeline. The fallback is wired into both dashboard camera probe/preview and hardware capture. It still enforces the measured 42.5 FPS minimum; it does not silently lower the quality gate.

The requested UVC exposure is pre-applied through OpenCV before FFmpeg opens the camera, and the driver's acceptance/readback is recorded in `camera_runtime.exposure_control`. The development tests cannot prove that this particular camera driver preserves exposure when the capture graph is reopened. Verify the actual laser-line brightness/exposure during a short test capture before a real scan. FFmpeg must be on the dashboard process PATH, the exact camera device name must be selected, and other applications must release the camera. See `CAMERA_FPS_FIX.md`.

Follow-up automated validation: **62 passed, 6 skipped**. The six skips remain environment/data-dependent. This validates software compilation, the pipe decoder, and command construction; it is not a physical Windows hardware test.

## What was actually in the uploaded data

The ZIP contains the 350-frame session `data/sessions/scan_20261107_005822`. Its PNG camera frames were omitted, so the laser ridge and frame-to-frame image alignment could not be inspected directly. The dashboard screenshot is a separate 1,200-frame run and that run's session data was not included in the ZIP.

### Capture measurements from the included 350-frame session

| Measurement | Recorded / derived value |
|---|---:|
| Image size | 1600 x 1200 |
| Requested format | MJPG |
| Actual negotiated FOURCC | **YUY2** |
| Driver-reported FPS | 50 |
| Measured event/frame rate from `frames.csv` | **5.205 FPS** |
| Mean frame interval | 192.106 ms |
| Median frame interval | 192.658 ms |
| Frames | 350 |
| Laser-ON exposures | 175 |
| Median interval between ON profiles | 385.316 ms |
| Laser profile rate | 2.603 profiles/s |
| Firmware serial warnings | 352 |

The exact 192 ms event interval is consistent with a 5 FPS YUY2 stream. `meta.json` had `fps: 50.0`, but the code wrote `cfg.fps`; it did not measure the camera's delivered FPS.

### Reconstruction measurements from the included result folder

| Measurement | Result |
|---|---:|
| ON frames that produced usable points | 145 |
| Raw triangulated points | 4,502 |
| Average points per usable profile | **31.0** |
| Fused point cloud | 1,191 points |
| Reconstructed mesh | **18 vertices / 22 faces** |
| Mesh extent | **0.623 x 0.136 x 0.162 mm** |
| Estimated trajectory length | 333.94 mm |
| ROI | **Disabled** |
| Calibration | **Nominal / not measured** |

The 333.94 mm trajectory is generated by integrating the assumed 5 mm/s speed prior. It is not a measured handheld path.

The separate dashboard screenshot displays 1,200 / 1,200 frames, 4.7 FPS, 2,918 fused points, 11 vertices, 12 faces, and an extent of approximately 0.4 x 0.4 x 0.3 mm. Those screenshot values are not from the ZIP's 350-frame session; the two runs should not be mixed when comparing metrics.

### Stale segmentation files

The included result folder contains `segmented_full_colored.ply`, `segmented_clean_teeth.ply`, `segmentation_labels.npz`, and two per-tooth class meshes, while the current `pipeline_report.json` records `segmentation: null`. Those files therefore cannot be treated as evidence that the mesh in the current report was successfully segmented. They are stale/orphaned outputs or outputs from a different earlier attempt. The updated pipeline removes old segmentation and preview artifacts before starting a new reconstruction.

## Root causes and confidence

### 1. Camera runs in YUY2 instead of MJPG — confirmed

`src/ioscan/capture/logger.py` called `open_usb_camera(..., read_attempts=0)`. That disabled the helper's real-frame verification. `src/ioscan/capture/camera.py` accepted a camera handle without checking that the requested FOURCC was actually negotiated. The session metadata confirms `requested_format: MJPG` and `actual_fourcc_text: YUY2`.

The Arducam OV2311 UVC documentation for the B0322 variant lists 1600 x 1200 MJPG at up to 50 FPS and YUY2 at 5 FPS. Confirm the exact board SKU, but the captured 5.2 FPS matches the slow YUY2 mode. Some documented trigger modes have lower limits, so the camera's trigger/low-brightness-compensation mode must also be checked.

### 2. Synchronous PNG writing — likely next throughput bottleneck, not the current measured root cause

Every grayscale frame was synchronously written using `cv2.imwrite()` from the capture thread. The session directory is under a OneDrive-synced Documents path. This can stall a future 50 FPS stream even after the camera format is corrected. The stable 192 ms interval in the included run is much more consistent with the camera's YUY2 mode than with random disk stalls, but the writer was still a real throughput risk.

### 3. Sparse laser profiles cannot form a connected surface — confirmed by timestamps and geometry

The firmware alternates laser-OFF and laser-ON exposures. At 5.205 FPS, ON profiles arrive about every 0.385 seconds. With the pipeline's assumed speed of 5 mm/s:

`profile_spacing = 5 mm/s x 0.385 s = about 1.93 mm`

The BPA mesher caps its largest ball radius at 0.34 mm. Adjacent profiles are therefore approximately 5.7 times farther apart than that cap. BPA cannot create a connected surface across those gaps. At a real 50 FPS stream with alternating OFF/ON frames, the ON profile rate would be about 25 Hz; at the same assumed 5 mm/s, profile spacing would be about 0.20 mm. That is a much more plausible starting point for the current BPA settings.

### 4. Too few ridge points per profile — measured sparsity; exact image-level cause unverified

The included session retained only 4,502 points across 145 usable ON profiles, approximately 31 points per profile. The ridge extractor currently selects one maximum peak per image column and rejects a column if a second peak outside a six-pixel guard region exceeds 45% of the main peak. Low laser contrast, reflections, multiple visible laser lines, saturation, exposure, or chassis pixels could all cause rejection. The raw PNGs are absent, so it would be unsafe to change the peak-rejection threshold blindly.

### 5. ROI mask is disabled — confirmed

The session `roi.json` contains a four-corner polygon but has `enabled: false`. The point-cloud pipeline therefore processed the full frame, including chassis/interior pixels. The dashboard already has a Camera ROI editor; it must be enabled and saved before the next scan. The ROI excludes pixels outside the chosen polygon. It does not remove chassis surfaces that lie inside the polygon and cannot recover teeth physically occluded by the chassis.

### 6. Translation is assumed, not measured — confirmed design limitation

`PoseTracker` uses the BNO086 quaternion as camera orientation and advances the camera by a constant configured speed along camera +Y. A BNO086 orientation report is not a translation measurement. Actual handheld speed and direction vary, so the generated pose path is only a prior and can stretch, shear, or smear the fused cloud. The IMU-to-camera mounting transform is also not calibrated in the supplied metadata.

For an initial engineering test, use a motorized linear stage with a known speed and travel direction. For handheld scanning, add a translation source such as an encoder, validated visual odometry, or a well-constrained scan-matching method. Do not interpret the integrated 5 mm/s trajectory as measured motion.

### 7. MeshSegNet gate was too permissive — confirmed

The old pipeline attempted segmentation once a mesh had more than 20 triangles. MeshSegNet cannot restore missing surface geometry; a 12- or 22-face fragment is not a tooth arch. The dashboard's warning in the screenshot is therefore correct to reject the 12-triangle mesh, but the earlier 20-face gate was far too weak.

The patch adds an IOSCAN-level engineering gate of at least 5,000 faces and at least 35 mm maximum mesh extent. These are conservative project guardrails, **not formal model-intrinsic MeshSegNet requirements** and not proof that a scan is complete. The model still needs a sufficiently dense, well-covered, correctly oriented dental surface.

### 8. Calibration is nominal — confirmed; exact scale error depends on lens SKU

The session is marked `calibration_is_nominal: true`. The camera intrinsics use `fx = fy = 2666.7 px`, an 8 mm nominal lens assumption, and zero distortion. The laser plane is also created from nominal baseline/angle values. If the physical camera is the Arducam B0322 variant advertised with 75-degree horizontal FOV, that nominal focal length is likely inconsistent with the advertised FOV; however, the session's device name does not prove the exact SKU or lens. Do not replace intrinsics with a catalogue estimate. Calibrate the actual lens at 1600 x 1200 with a checkerboard, then measure the laser plane.

### 9. Exposure and camera/firmware synchronization still need measurement

The capture config comments that exposure `-7` is approximately 1.5 ms. UVC exposure values can be backend/driver-specific; the actual integration time must be measured or confirmed through the exact camera utility. The host pairs the next queued ESP32 strobe packet with the next OpenCV frame. Serial sequence gaps are detected, but OpenCV does not expose the sensor frame ID here; a camera-side dropped frame could still shift event/image pairing. The captured line images must be checked against `laser_on` flags or the vendor SDK's timestamps/frame IDs integrated.

### 10. Laser switching and I2C are not proven FPS bottlenecks

The firmware sets the laser MOSFET gate in the camera-STROBE interrupt, and forces the laser low if STROBE is absent for 100 ms. At 5 FPS the watchdog warning is expected as a consequence of the slow event rate; increasing the timeout would hide a fault rather than speed the camera. The UI's `LASER ON` label reflects the commanded exposure state, not a photodiode measurement of emitted light.

The BNO086 is currently polled over I2C at 400 kHz and configured for a 100 Hz Game Rotation Vector. That is sufficient for this low-bandwidth orientation packet stream in the observed configuration. SPI may reduce bus latency/jitter in a later hardware revision, but the captured frame intervals point to camera UVC format negotiation, not I2C saturation.

## Software changes applied in this patched project

1. **Camera negotiation (`src/ioscan/capture/camera.py`)**
   - Requires at least three actual frames for validation; it no longer accepts `read_attempts=0`.
   - Reads back the actual FOURCC after streaming begins and rejects a known mismatch or an unreported required format.
   - Measures frame arrival rate from real frame timestamps rather than trusting `CAP_PROP_FPS`.
   - Supports required FOURCC and minimum measured FPS. It respects the selected camera/backend pair; it never reuses a DirectShow numeric index under Media Foundation because that could silently select a different physical or virtual camera. Select the corresponding Arducam entry under the other backend from the device list when testing that backend.

2. **Capture and storage (`src/ioscan/capture/logger.py`)**
   - Hardware scan capture requires MJPG and at least 85% of the configured FPS (42.5 FPS for a 50 FPS request).
   - PNG encoding/writing runs in a bounded background queue of 32 frames.
   - If disk writing cannot keep up and the queue fills, capture stops with a clear error instead of silently dropping saved frames.
   - Metadata now separates `requested_fps` from measured session `fps`, and records actual FOURCC and driver-reported FPS separately.

3. **Dashboard preflight (`src/ioscan/dashboard/server.py`, `static/app.js`)**
   - Camera probe is now scan-grade: it requests MJPG at 1600 x 1200, measures the rate, and refuses to report ready unless it reaches the threshold.
   - The probe response and toast report the actual pixel format and measured FPS.
   - Automatic and manual segmentation are blocked for obviously inadequate mesh geometry.
   - A poor scan is shown as incomplete/recapture-required rather than a successful 3D reconstruction.

4. **Reconstruction diagnostics (`scripts/run_pipeline.py`)**
   - Removes stale preview, pose, report, and segmentation artifacts before a new run (after dependencies and session inputs validate).
   - Reports actual capture FPS, laser profile rate, points per usable profile, assumed median profile spacing, mesh extent, and the basic segmentation geometry gate.
   - Warns when the camera format is not MJPG, ROI is disabled, capture FPS is low, ridge-point density is sparse, or the assumed profile spacing exceeds the BPA bridge radius.

5. **Segmentation safeguards (`src/ioscan/pipeline/segment.py`)**
   - Rejects meshes below the 5,000-face project gate or below 35 mm maximum extent before loading PyTorch/checkpoints.
   - Removes stale segmentation files even when a direct segmentation attempt fails its geometry gate.

6. **Firmware/docs/tests**
   - Disabled verbose IMU debug logging in the production sketch (`DEBUG_IMU 0`). The 100 ms STROBE watchdog remains in place intentionally.
   - Corrected docs that incorrectly described the current BNO086 wiring as SPI; the current sketch is I2C at 400 kHz.
   - Added camera-format negotiation tests and stale-output-cleanup tests; updated capture tests for strict MJPG validation and measured FPS metadata.

## Test and replay status

- `python -m compileall -q src scripts tests`: passed.
- `python -m pytest -q -rs`: **60 passed, 6 skipped**.
- The skipped tests are Open3D-dependent geometry tests (Open3D is not installed in this review runtime), one test requiring a missing sample session, and three external Teeth3DS dataset tests (the dataset path is not configured).
- JavaScript syntax is checked separately in the package validation step.

The patch was not run against the physical camera, ESP32, BNO086, MOSFET/laser, or a live scan. Open3D is unavailable in this environment, and the archived PNG frames were omitted, so a full replay of the stored capture through the patched mesher and image-level ridge diagnostics could not be performed. The new 50 FPS mode and a real 3D scan must be confirmed on the target Windows machine.

## Hardware/software action plan for the next test

1. **Extract this patched project to a local SSD folder**, not a OneDrive-synced directory. Install the project's declared dependencies and run the dashboard.
2. **Setup → Check camera:** require 1600 x 1200, `MJPG`, and at least 42.5 measured FPS. If this fails, close Windows Camera/OBS/NVIDIA Broadcast/other camera applications, try Media Foundation versus DirectShow, check the Arducam UVC utility for MJPG mode, verify cable/USB port/hub, and confirm external-trigger/low-brightness-compensation mode is disabled unless the hardware is intentionally configured for external triggering.
3. **Setup → Camera ROI:** adjust the saved polygon to the inside edge of the optical opening, enable **Apply ROI to scan**, then click **Save ROI**. Confirm the outside of the desired optical opening is shaded. The included ROI polygon is saved but disabled; do not assume it matches the current chassis without looking at the live image.
4. **Verify laser/strobe electrically:** use an oscilloscope/logic analyser to check the camera STROBE signal at ESP32 GPIO34 and the MOSFET gate at GPIO27. Use a photodiode or a verified exposure image to confirm the laser actually emits light during ON frames and goes dark during OFF frames. Confirm compatible 3.3 V logic, polarity, grounding, driver current, and interlocks against the schematic. Do not increase the STROBE timeout just to suppress warnings.
5. **Run a short motion-controlled calibration-target scan** on a stage with known travel speed. Check the ON/OFF order, actual measured FPS, number of usable profiles, points/profile, measured profile spacing, point-cloud shape, and mesh extent before trying teeth.
6. **Calibrate geometry:** checkerboard camera intrinsics/distortion, measured camera-to-laser plane, and IMU-to-camera extrinsics. Do not use nominal values for dimensional measurements.
7. **Only after those checks**, attempt overlapping occlusal, buccal and lingual passes. A single laser pass cannot produce a complete closed tooth or arch. Add a real translation sensor/validated visual odometry before treating hand-held scans as metrically accurate.

## Hardware reference links

- [Arducam OV2311 UVC camera product information](https://www.arducam.com/blog/product/arducam-2mp-global-shutter-usb-camera-board-for-computer-50fps-ov2311-monochrome-uvc-webcam-module-with-low-distortion-m12-lens-without-microphones-compatible-with-windows-linux-android-and-mac-os/)
- [Arducam OV2311 UVC documentation](https://docs.arducam.com/UVC-Camera/Appilcation-Note/External-Trigger-Mode/OV2311-Global-Shutter/)

This remains an engineering development scanner, not a validated clinical or diagnostic device.
