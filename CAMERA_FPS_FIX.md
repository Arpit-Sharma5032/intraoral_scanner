# Camera 50 FPS capture fix

## Why this patch was added

The Arducam OV2311 advertised 1600x1200 MJPG at up to 50 FPS. On the same Windows workstation, the independent DirectShow/FFmpeg test delivered 750 frames in 15 seconds (reported `fps=49`). IOSCAN's OpenCV DirectShow path, however, measured only 25.2 FPS when trying MJPG, while the alternate FOURCC order fell back to YUY2.

## What changed

- IOSCAN still tries its existing OpenCV camera path first.
- If the selected backend is DirectShow, MJPG is mandatory, the exact camera name is known, and OpenCV cannot meet the measured FPS threshold, IOSCAN starts FFmpeg's DirectShow input with the same mode options as the working standalone test:
  - `-f dshow`
  - `-video_size 1600x1200`
  - `-framerate 50`
  - `-vcodec mjpeg`
  - `-i video=<selected camera name>`
- FFmpeg sends the MJPEG frames through an image pipe. IOSCAN decodes each JPEG as grayscale, which is the format the monochrome reconstruction pipeline consumes.
- The fallback checks real delivered frame timing. It does not treat the requested 50 FPS as proof of 50 FPS.
- The dashboard camera check/preview and the hardware scan capture use the same fallback.
- UVC exposure controls are attempted through OpenCV before FFmpeg opens the camera. The driver readback is recorded in `meta.json` under `camera_runtime.exposure_control`; verify this on the physical scanner because a particular driver may not preserve settings when the capture graph is reopened.

## Requirements

1. Windows DirectShow backend.
2. FFmpeg installed and `ffmpeg.exe` visible on the same PATH as the Python dashboard process (`where.exe ffmpeg` in PowerShell should find it).
3. The dashboard must pass the exact selected device name, for example `Arducam OV2311 USB Camera`.
4. Close Windows Camera, OBS, Arducam utility preview, and other applications before starting IOSCAN. Only one process can own the camera reliably at a time.

## How to verify

1. Start IOSCAN from the same PowerShell environment in which `ffmpeg` works.
2. Select the physical Arducam device and choose DirectShow.
3. Click **Check camera**. The result should show `MJPG`, `1600 x 1200`, and measured FPS at or above 42.5. If the OpenCV path cannot achieve the threshold, the backend field should show `FFmpeg/DirectShow`.
4. Start a short stationary-target capture and inspect `data/sessions/<session>/meta.json`:
   - `camera_runtime.capture_engine` should be `ffmpeg` when the fallback was used.
   - `camera_runtime.measured_fps` is the measured opening-rate sample.
   - `camera_runtime.measured_session_fps` is the session rate calculated from recorded acquisition timestamps.
   - `camera_runtime.exposure_control` records the pre-open UVC exposure-control result.
5. Compare actual laser visibility and brightness with the known exposure setting. If exposure is not retained after FFmpeg opens the device, do not proceed to a dental scan; the DirectShow exposure control path will need a device-specific adjustment.

## Validation performed in this package

- Python compilation succeeded.
- Automated tests: 62 passed, 6 skipped.
- The automated tests validate JPEG pipe parsing and the exact DirectShow/MJPG FFmpeg argument construction. They do not access a Windows camera or verify optical exposure persistence.
- The 49 FPS physical-camera result is from the operator's independent FFmpeg test, not a test run by this patch in the development container.
