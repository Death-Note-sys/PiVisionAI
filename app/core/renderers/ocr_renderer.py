import cv2
import numpy as np
from typing import Dict, Any
from app.core.renderers.base_renderer import BaseRenderer

class OCRRenderer(BaseRenderer):
    """Draws polygon outlines and recognized text for OCR results."""

    def render(self, frame: np.ndarray, result: Any, metadata: Dict[str, Any]) -> np.ndarray:
        if not hasattr(result, "texts"):
            return frame

        out_frame = frame.copy()
        settings = metadata.get("settings", {})
        show_text = settings.get("show_text", True)
        show_conf = settings.get("show_confidence", True)
        thickness = int(settings.get("thickness", 2))

        for t in result.texts:
            points = t.get("points", [])
            if len(points) < 4:
                continue
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out_frame, [pts], isClosed=True, color=(0, 217, 198), thickness=thickness)

            if show_text:
                label = t.get("text", "")
                if show_conf:
                    label += f" ({t.get('confidence', 0):.2f})"
                x, y = points[0]
                cv2.putText(out_frame, label, (x, max(y - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 217, 198), max(1, thickness - 1))

        return out_frame
