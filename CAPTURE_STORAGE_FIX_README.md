# Capture storage throughput update (source MJPEG passthrough)

The Windows FFmpeg/DirectShow/MJPG scan path now writes each complete camera JPEG packet directly to `frames/*.jpg`. It does not decode and recompress the stored frame as PNG. The grayscale image is decoded once in memory for laser extraction and preview. `frames.csv` contains each exact filename, and the existing session loader reads image files by path, so mixed `.jpg`/`.png` sessions are supported.

This removes PNG compression from the 50 FPS acquisition/storage path and preserves the camera's original MJPEG bytes. The JPEG compression is already applied by the camera/firmware mode; IOSCAN does not recompress it. Non-FFmpeg backends continue to store lossless grayscale PNG.

The bounded queue remains a safety mechanism. If direct JPEG writes still cannot keep up, the application stops rather than silently dropping frames. In that case, inspect `meta.json` → `frame_storage`, confirm that the session is on a local SSD outside OneDrive/Dropbox, and check Windows Defender real-time scanning and disk activity for that specific session folder.

`meta.json` now records `image.storage` as `source_mjpeg_jpeg` or `lossless_png`, and `frame_storage.payload_bytes_written` counts actual encoded bytes persisted.
