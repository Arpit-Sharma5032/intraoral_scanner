# Direct MJPEG storage fix

## Why the previous storage change still stopped

The earlier capture path read a camera MJPEG frame, decoded it to grayscale, then encoded that grayscale array as PNG. Four writer threads and a 128-frame queue reduced brief stalls but did not remove the expensive PNG encoding step. The queue-full message at frame 457 indicates that the PNG writer workers still could not keep up with the delivered camera rate.

## What changed

- `FFmpegDShowCapture.read_encoded()` returns each complete JPEG packet directly from the FFmpeg `image2pipe` stream.
- The hardware capture logger uses the original JPEG bytes for storage when `capture_engine == "ffmpeg"`.
- The logger decodes that JPEG once to grayscale in memory for laser detection, callbacks, and preview.
- Source packets are stored byte-for-byte as `frames/000000.jpg`, etc. The logger does not apply a second lossy JPEG encode or PNG encode.
- Other capture backends continue to store lossless grayscale PNG files.
- `frames.csv` already stores the exact relative filename, and the pipeline loads the image by path. OpenCV's grayscale JPEG decode was verified in a test, so JPEG frame files remain single-channel for triangulation.
- `meta.json` records `image.storage` and `frame_storage` counts/bytes by format.
- A bounded queue and stop-on-overflow protection remain in place. This avoids silently losing or mis-pairing frames if the disk still cannot sustain the data rate.

## Expected result and validation

The DirectShow/FFmpeg camera path should no longer consume CPU cycles on PNG compression. On a local SSD this should substantially reduce storage backlog. It does not guarantee 50 FPS on every PC: actual JPEG packet sizes, disk latency, antivirus scanning, and competing disk/CPU use still matter. A short real-hardware test must confirm that `max_queue_depth_frames` stays low and `frames_written` tracks frames captured.

## Tests

- Python compilation: passed.
- Test suite: 74 passed, 6 skipped.
- Unit test verifies JPEG packet bytes are written unchanged and OpenCV reads the result as a 2D grayscale array.
- Camera test verifies interleaved encoded reads and decoded reads from the FFmpeg pipe.

The tests do not exercise the physical camera or SSD on the user's Windows machine.
