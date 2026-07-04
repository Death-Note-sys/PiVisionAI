import asyncio
import logging
from typing import Callable, Dict, List, Any
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class EventType(str, Enum):
    CAMERA_STARTED = "CameraStarted"
    CAMERA_STOPPED = "CameraStopped"
    CAMERA_CHANGED = "CameraChanged"
    FRAME_CAPTURED = "FrameCaptured"
    FRAME_PROCESSED = "FrameProcessed"
    INFERENCE_COMPLETED = "InferenceCompleted"
    OBJECT_DETECTED = "ObjectDetected"
    FACE_DETECTED = "FaceDetected"
    OCR_DETECTED = "OCRDetected"
    MEASUREMENT_COMPLETED = "MeasurementCompleted"
    COLOR_SELECTED = "ColorSelected"
    RECORDING_STARTED = "RecordingStarted"
    RECORDING_STOPPED = "RecordingStopped"
    SCREENSHOT_TAKEN = "ScreenshotTaken"
    EXPORT_COMPLETED = "ExportCompleted"
    MODULE_LOADED = "ModuleLoaded"
    MODULE_UNLOADED = "ModuleUnloaded"
    SETTINGS_CHANGED = "SettingsChanged"
    BACKEND_CHANGED = "BackendChanged"
    MODEL_LOADED = "ModelLoaded"
    MODEL_UNLOADED = "ModelUnloaded"
    APPLICATION_STARTED = "ApplicationStarted"
    APPLICATION_SHUTDOWN = "ApplicationShutdown"


class EventBus:
    """Centralized asynchronous pub/sub event bus."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: EventType | str, callback: Callable) -> None:
        """Subscribe to an event."""
        event_name = event_type.value if isinstance(event_type, EventType) else event_type
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        if callback not in self._subscribers[event_name]:
            self._subscribers[event_name].append(callback)
            logger.debug(f"Subscribed {callback.__name__} to {event_name}")

    def unsubscribe(self, event_type: EventType | str, callback: Callable) -> None:
        """Unsubscribe from an event."""
        event_name = event_type.value if isinstance(event_type, EventType) else event_type
        if event_name in self._subscribers and callback in self._subscribers[event_name]:
            self._subscribers[event_name].remove(callback)
            logger.debug(f"Unsubscribed {callback.__name__} from {event_name}")

    def publish(self, event_type: EventType | str, payload: Any = None) -> None:
        """
        Publish an event to all subscribers asynchronously.
        If a subscriber is a coroutine function, it will be wrapped in a Task.
        Otherwise, it will be executed directly.
        """
        event_name = event_type.value if isinstance(event_type, EventType) else event_type
        if event_name not in self._subscribers:
            return

        # Ensure payload is serialized correctly if it's a pydantic model
        if isinstance(payload, BaseModel):
            payload = payload.model_dump()
            
        for callback in self._subscribers[event_name]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(payload))
                else:
                    callback(payload)
            except Exception as e:
                logger.error(f"Error executing callback {callback.__name__} for event {event_name}: {e}", exc_info=True)
