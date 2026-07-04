import logging
import asyncio
from typing import Callable, Any, Coroutine, Dict
import threading

logger = logging.getLogger(__name__)

class TaskScheduler:
    """
    Manages non-blocking background tasks (e.g. saving screenshots, encoding video, I/O).
    Runs an asyncio event loop in a dedicated background thread.
    """
    
    def __init__(self, event_bus: Any):
        self.event_bus = event_bus
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True, name="TaskSchedulerThread")
        self._thread.start()
        
        # Keep track of running tasks if needed
        self._tasks: Dict[str, asyncio.Task] = {}

    def _start_loop(self):
        """Run the asyncio event loop indefinitely."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    def submit_task(self, name: str, coro: Coroutine) -> None:
        """
        Submit an asynchronous coroutine to run in the background.
        """
        if not asyncio.iscoroutine(coro):
            logger.error(f"Task {name} must be a coroutine.")
            return
            
        def _done_callback(fut):
            try:
                fut.result()
            except Exception as e:
                logger.error(f"Background task '{name}' failed: {e}", exc_info=True)
            finally:
                self._tasks.pop(name, None)

        task = asyncio.run_coroutine_threadsafe(coro, self._loop)
        task.add_done_callback(_done_callback)
        self._tasks[name] = task
        logger.debug(f"Submitted background task: {name}")

    def shutdown(self):
        """Cleanly stop the background loop."""
        logger.info("Shutting down TaskScheduler...")
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
        logger.info("TaskScheduler stopped.")
