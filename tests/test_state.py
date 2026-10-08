import pytest
import threading
from app.core.state import StateStore

def test_state_store_thread_safety():
    store = StateStore()
    
    def worker():
        for _ in range(100):
            store.increment_ok()
            store.increment_ng({"error": "test"})
            
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    state = store.get_global_state()
    assert state["ok_count"] == 1000
    assert state["ng_count"] == 1000
    assert state["total_count"] == 2000
    assert state["latest_ng_metadata"] == {"error": "test"}

def test_state_store_module_update():
    store = StateStore()
    
    store.update("mod_1", {"key1": "val1"})
    store.update("mod_1", {"key2": "val2"})
    
    mod_state = store.get_module_state("mod_1")
    assert mod_state == {"key1": "val1", "key2": "val2"}
    
    # Check that another module is empty
    assert store.get_module_state("mod_2") == {}
