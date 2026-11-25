# IOSCAN camera fix v4 — select the physical Arducam by name

## Apply this update
1. Stop the dashboard server. Close the Windows Camera app and other camera applications while testing so the Arducam is not in use by two applications.
2. Extract this ZIP into the project root (the folder containing `pyproject.toml`) and overwrite the files.
3. Activate the `ios_scan` environment.
4. Install the new camera enumerator dependency in the active environment:
   ```powershell
   python -m pip install cv2-enumerate-cameras
   ```
   The project now uses `cv2-enumerate-cameras` to list camera names, USB VID/PID values, and the matching OpenCV index/backend pair.
5. Restart the dashboard:
   ```powershell
   python scripts/run_dashboard.py
   ```
6. Open **Setup → Devices → Refresh list**. Select the physical **Arducam / OV2311 UVC** entry. Do not select NVIDIA Broadcast, OBS, or another virtual camera. Then click **Check camera**.
7. Confirm that the live preview is visibly showing the camera feed before starting a scan.

## Why this revision is different
- The earlier numeric-only camera selection could select a virtual camera. The diagnostic log contains `VCAMDS` and `NvMxnCltShmConsumer` messages, which are associated with NVIDIA Broadcast's virtual-camera path, not the OV2311 sensor. The dashboard now enumerates Windows camera names separately for DirectShow and Media Foundation and opens the exact backend/index returned by that enumeration.
- The camera probe now rejects all-zero frames instead of reporting a successful camera connection just because OpenCV returned a 1600x1200 buffer. The error identifies the selected device and backend.
- The scan-grade camera probe and capture path require MJPG at 1600x1200 and reject other formats. The selected backend/index pair is preserved; to test another backend, select the physical Arducam entry listed for that backend rather than reusing the same numeric index.
- Idle preview requests automatic exposure. Synchronized scan capture switches to manual exposure before setting the configured exposure value; these settings are kept separate.
- The camera's selected name/backend are passed through to synchronized capture and recorded in session metadata.

## Verification and limitations
- Python compilation and JavaScript syntax checks pass. The test suite passes in this build; these tests use simulated camera frames and do not prove the physical camera is working.
- I cannot directly access the Arducam attached to your Windows PC from this environment. The decisive hardware verification is that the dashboard device list identifies the physical Arducam, and **Check camera** produces a non-black preview.
- If the OV2311 is configured in external-trigger snapshot mode (Arducam's “low-brightness compensation” trigger mode), a preview may stop until trigger pulses arrive. This project expects free-running camera frames with the camera STROBE output observed by the ESP32; disable external-trigger snapshot mode unless the camera is intentionally being triggered by compatible hardware.
- The supplied `fx`/`fy` remain nominal placeholders. Accurate 3D reconstruction still requires measured calibration for this camera/lens at 1600x1200.
- Older saved sessions at 1600x1300 should not be replayed using the new 1600x1200 calibration.


## OV2311 high-rate DirectShow path

On Windows, when the selected physical device is the Arducam OV2311 and the selected backend is DirectShow, the dashboard preview and scan capture now prefer FFmpeg's DirectShow MJPG pipe before attempting OpenCV. This avoids a known failure mode where OpenCV blocks in UVC open/read and prevents a fallback from being reached. The mode is verified from actual decoded frame arrivals and must meet the configured minimum rate.

Because this preferred FFmpeg path does not expose UVC exposure controls through OpenCV, IOSCAN does not claim to change exposure while FFmpeg owns the camera. Set automatic/manual exposure and the exposure value in the Arducam camera utility before starting a scan; inspect the `exposure_control.note` field in session metadata. Do not assume exposure is correct until the laser ON/OFF frame contrast has been checked.
