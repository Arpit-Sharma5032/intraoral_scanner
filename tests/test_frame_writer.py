from __future__ import annotations

import threading

import numpy as np
import pytest

from ioscan.capture import logger
from ioscan.capture.logger import _AsyncFrameWriter


def test_async_frame_writer_uses_multiple_workers_and_flushes(tmp_path):
    writer = _AsyncFrameWriter(tmp_path, max_queue=8, png_compression=1, workers=3)
    images = [np.full((24, 32), i, dtype=np.uint8) for i in range(6)]
    for i, image in enumerate(images):
        writer.submit(f"frames/{i:06d}.png", image)
    writer.close()

    assert all((tmp_path / f"frames/{i:06d}.png").is_file() for i in range(6))
    stats = writer.stats()
    assert stats["worker_count"] == 3
    assert stats["frames_written"] == 6
    assert stats["uncompressed_png_input_bytes_written"] == sum(image.nbytes for image in images)
    assert stats["stored_bytes_written"] > 0


def test_async_frame_writer_reports_sustained_queue_overflow(monkeypatch, tmp_path):
    entered_write = threading.Event()
    allow_write = threading.Event()

    def slow_write(_path, _frame, _params):
        entered_write.set()
        allow_write.wait(timeout=3)
        return True

    monkeypatch.setattr(logger.cv2, "imwrite", slow_write)
    writer = _AsyncFrameWriter(tmp_path, max_queue=1, png_compression=1, workers=1)
    frame = np.zeros((8, 8), dtype=np.uint8)
    writer.submit("frames/000000.png", frame)
    assert entered_write.wait(timeout=1)
    writer.submit("frames/000001.png", frame)
    with pytest.raises(RuntimeError, match="queue is full"):
        writer.submit("frames/000002.png", frame)
    allow_write.set()
    writer.close()


def test_async_frame_writer_stores_source_jpeg_bytes_unchanged(tmp_path):
    ok, encoded = logger.cv2.imencode(".jpg", np.full((24, 32), 127, dtype=np.uint8))
    assert ok
    payload = encoded.tobytes()
    writer = _AsyncFrameWriter(tmp_path, max_queue=4, workers=2)
    writer.submit_jpeg("frames/000000.jpg", payload)
    writer.close()

    saved = (tmp_path / "frames/000000.jpg").read_bytes()
    assert saved == payload
    assert saved.startswith(b"\xff\xd8") and saved.endswith(b"\xff\xd9")
    decoded = logger.cv2.imread(str(tmp_path / "frames/000000.jpg"), logger.cv2.IMREAD_UNCHANGED)
    assert decoded is not None and decoded.ndim == 2 and decoded.shape == (24, 32)
    stats = writer.stats()
    assert stats["frames_written"] == 1
    assert stats["stored_bytes_written"] == len(payload)
    assert stats["jpeg_frames_written"] == 1 and stats["png_frames_written"] == 0
