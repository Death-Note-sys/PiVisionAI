import pytest
import numpy as np
from app.core.output_manager import OutputManager

def test_output_manager_ng_queue_bounds():
    manager = OutputManager()
    
    # We shouldn't run the thread because it expects the Container to be loaded.
    # But we can test the enqueueing bounds.
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    
    # Fill the queue beyond its maxlen
    for i in range(manager.ng_queue.maxlen + 10):
        manager.save_ng(frame, {"index": i})
        
    # Verify bounds
    assert len(manager.ng_queue) == manager.ng_queue.maxlen
    
    # Verify the oldest were dropped
    oldest = manager.ng_queue.popleft()
    # If 10 extra were added, the new oldest should be index 10
    assert oldest[1]["index"] == 10
