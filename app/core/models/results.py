from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ModuleResult(BaseModel):
    """Base class for all framework results, independent of AI/Backend."""
    latency_ms: float = 0.0
    backend_name: Optional[str] = None
    model_name: Optional[str] = None
    frame_number: int = 0
    timestamp: float = 0.0
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)

class DetectionResult(ModuleResult):
    """Standardized result for Object Detection."""
    detections: List[Dict[str, Any]] = Field(default_factory=list) # Dicts with box, label, conf, id
    objects_count: int = 0
    average_confidence: float = 0.0

class MeasurementResult(ModuleResult):
    """Standardized result for Measurements."""
    measurements: List[Dict[str, Any]] = Field(default_factory=list)
    calibration_status: str = "Uncalibrated"
    pixels_per_cm: Optional[float] = None

class OCRResult(ModuleResult):
    """Standardized result for OCR."""
    texts: List[Dict[str, Any]] = Field(default_factory=list)

class ColorResult(ModuleResult):
    """Standardized result for Color Analysis."""
    dominant_colors: List[Dict[str, Any]] = Field(default_factory=list)

class MotionResult(ModuleResult):
    """Standardized result for Motion Detection."""
    motion_areas: List[Dict[str, Any]] = Field(default_factory=list)
    global_motion_score: float = 0.0

class ShapeResult(ModuleResult):
    """Standardized result for Shape Detection."""
    shapes: List[Dict[str, Any]] = Field(default_factory=list)

class PoseResult(ModuleResult):
    """Standardized result for Pose Estimation."""
    poses: List[Dict[str, Any]] = Field(default_factory=list) # Landmarks

class TrackingResult(ModuleResult):
    """Standardized result for Object Tracking."""
    tracks: List[Dict[str, Any]] = Field(default_factory=list)

class SegmentationResult(ModuleResult):
    """Standardized result for Instance/Semantic Segmentation."""
    masks: List[Dict[str, Any]] = Field(default_factory=list)
