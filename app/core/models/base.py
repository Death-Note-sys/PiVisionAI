from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any, Dict

class BoundingBox(BaseModel):
    """Represents a bounding box in an image."""
    model_config = ConfigDict(frozen=True)
    
    x1: int = Field(..., description="Top-left X coordinate")
    y1: int = Field(..., description="Top-left Y coordinate")
    x2: int = Field(..., description="Bottom-right X coordinate")
    y2: int = Field(..., description="Bottom-right Y coordinate")

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)


class Detection(BaseModel):
    """Represents a single detected object or feature."""
    box: Optional[BoundingBox] = None
    label: str = Field(..., description="Class label or name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional module-specific data")


class InferenceResult(BaseModel):
    """Standardized output from any AI model/module."""
    module_id: str
    detections: List[Detection] = Field(default_factory=list)
    inference_time_ms: float = Field(0.0, description="Time taken for inference in milliseconds")
    global_metadata: Dict[str, Any] = Field(default_factory=dict, description="Frame-level metadata (e.g. face count, dominant color)")
