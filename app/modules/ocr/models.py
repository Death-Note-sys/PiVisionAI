from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class UpdateOCRSettingsRequest(BaseModel):
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    show_text: Optional[bool] = None
    show_confidence: Optional[bool] = None
    thickness: Optional[int] = Field(None, ge=1, le=10)
    model_id: Optional[str] = None

    def to_update_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)
