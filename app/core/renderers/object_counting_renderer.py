import cv2
import numpy as np
from typing import Dict, Any
from app.core.renderers.base_renderer import BaseRenderer


class ObjectCountingRenderer(BaseRenderer):
    """Draws bounding boxes, index numbers, and count badges for Similar Object Counting."""

    def render(self, frame: np.ndarray, result: Any, metadata: Dict[str, Any]) -> np.ndarray:
        if not hasattr(result, "count"):
            return frame

        out_frame = frame.copy()
        settings = metadata.get("settings", {})
        show_boxes = settings.get("show_boxes", True)
        show_numbers = settings.get("show_numbers", True)
        thickness = settings.get("thickness", 2)

        # Color based on judging state: cyan (not judged), green (OK), red (NG)
        is_ok = getattr(result, "is_ok", None)
        if is_ok is True:
            color = (62, 207, 142)   # Green OK
            judge_str = " [OK]"
        elif is_ok is False:
            color = (92, 92, 255)    # Red NG
            judge_str = " [NG]"
        else:
            color = (255, 255, 0)    # Cyan (not judged)
            judge_str = ""

        # Draw matches
        matches = getattr(result, "matches", [])
        for i, m in enumerate(matches):
            if isinstance(m, dict):
                x, y, w, h = m.get("x", 0), m.get("y", 0), m.get("w", 0), m.get("h", 0)
            else:
                x, y, w, h = getattr(m, "x", 0), getattr(m, "y", 0), getattr(m, "w", 0), getattr(m, "h", 0)

            if show_boxes:
                cv2.rectangle(out_frame, (x, y), (x + w, y + h), color, thickness)
            if show_numbers:
                cv2.putText(
                    out_frame, str(i + 1), (x + 3, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, max(1, thickness - 1)
                )

        # Large "COUNT: N" label (plus OK/NG when judging)
        count_val = getattr(result, "count", len(matches))
        label = f"COUNT: {count_val}{judge_str}"
        cv2.putText(out_frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, max(2, thickness))

        # Subtitle if untaught
        if getattr(result, "template_status", "Untaught") == "Untaught":
            cv2.putText(
                out_frame, "No template taught", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2
            )

        return out_frame
