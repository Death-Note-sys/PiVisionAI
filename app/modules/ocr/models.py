import re
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError

class UpdateOCRSettingsRequest(BaseModel):
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    show_text: Optional[bool] = None
    show_confidence: Optional[bool] = None
    thickness: Optional[int] = Field(None, ge=1, le=10)
    model_id: Optional[str] = None
    canvas_size: Optional[int] = Field(None, ge=320, le=2560)
    expected_pattern: Optional[str] = None
    judge_enabled: Optional[bool] = None

    @field_validator("expected_pattern")
    @classmethod
    def _expected_pattern_must_compile(cls, v):
        if v:
            try:
                re.compile(v)
            except re.error as e:
                raise PydanticCustomError("invalid_regex",
                    "invalid regular expression: {detail}", {"detail": str(e)})
        return v

    def to_update_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

