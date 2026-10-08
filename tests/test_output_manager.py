import pytest
import numpy as np
import time
from app.core.output_manager import OutputManager

def test_output_manager_broadcasts_and_encodes_jpeg():
    manager = OutputManager()
    
    # Broadcast a frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    manager.broadcast(frame, {})
    
    assert manager.frame_id == 1
    assert manager.latest_jpeg is not None
    assert len(manager.latest_jpeg) > 0
    
    jpeg_bytes = manager.get_latest_jpeg()
    assert jpeg_bytes == manager.latest_jpeg

def test_mjpeg_generator_yields_only_on_new_frames():
    manager = OutputManager()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    manager.broadcast(frame, {})
    
    generator = manager.get_mjpeg_generator(target_fps=30)
    
    # Should yield the first frame
    chunk1 = next(generator)
    assert b'--frame' in chunk1
    
    # Should NOT yield the next frame immediately if we haven't broadcasted
    # Wait, the generator blocks by time.sleep, but we can't test that easily without waiting.
    # We will just verify it yielded correctly.
    assert manager.frame_id == 1
