# IOSCAN host strobe-queue headroom fix

## What the Nov 9 capture error meant

The error `ESP32 strobe queue overflow` was raised by the **Python `SerialIMUReader` host queue** (`_events`), not by the ESP32 firmware ring. The host serial thread was receiving roughly 50 strobe packets/s while the main capture loop consumed only about 39 camera frames/s. The difference accumulated until the 128-event host queue filled.

The host loop was doing avoidable work for every 1600x1200 MJPEG frame: JPEG decode to grayscale, full-frame copies, and a dashboard preview callback. That work is not required to preserve the original camera frame.

## Changes

- The FFmpeg path still saves every original MJPEG/JPEG frame to disk and records every matching strobe event in `frames.csv`.
- The synchronized acquisition loop no longer decodes every FFmpeg frame just to feed the live preview.
- Live preview decodes only matched OFF/ON frame pairs, sampling every second laser-ON profile by default (12.5 preview profiles/s at 50 camera frames/s).
- The final post-capture pipeline still has every saved frame and processes the complete session; preview sampling does not decimate the recorded scan.
- Dashboard frame count/progress/FPS are updated from a lightweight per-frame callback, separate from the sampled image preview callback.
- The host event queue was increased from 128 to 256 as a short jitter buffer, not as a substitute for adequate sustained capture throughput.
- Serial reader queue high-water and event-count diagnostics are written to `meta.json`.
- The dashboard label now identifies the live point count as sampled preview geometry.

## What did not change

- ESP32 firmware, GPIO strobe ISR, laser timing, serial packet format, and CRC validation are unchanged in this patch.
- No packets or camera frames are intentionally dropped from the saved session.
- The live preview may be less dense than the final reconstruction because it samples profiles to protect acquisition throughput.

## Validation

- `pytest -q`: 80 passed, 6 skipped.
- Python compileall: passed.
- Dashboard `app.js` syntax check: passed.
- Hardware test still required: confirm capture sustains the strobe rate and inspect `meta.json` `serial_reader.max_event_queue_depth` after a 2,000-frame scan.

## Acceptance criteria for the next hardware test

1. 2,000/2,000 frames saved.
2. Session FPS close to the hardware strobe rate (target at least 45 FPS; exact rate depends on configured camera mode).
3. `serial_reader.max_event_queue_depth` remains low and does not trend upward through the scan.
4. `capture_complete` is true.
5. The saved `frames.csv` contains one unique camera image filename for each consumed strobe event.
6. Live preview point count is understood to be sampled; final reconstruction uses all frames.
