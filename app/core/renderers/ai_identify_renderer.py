import cv2
import numpy as np
from typing import Dict, Any
from app.core.renderers.base_renderer import BaseRenderer

class AIIdentifyRenderer(BaseRenderer):
    """Draws teach status, search state, and located Good/Bad/Uncertain classification."""

    def render(self, frame: np.ndarray, result: Any, metadata: Dict[str, Any]) -> np.ndarray:
        if not hasattr(result, "teach_status"):
            return frame

        out_frame = frame.copy()
        settings = metadata.get("settings", {})
        thickness = int(settings.get("thickness", 2))

        if result.teach_status != "Taught":
            cv2.putText(out_frame, f"Teach status: {result.teach_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (92, 92, 255), 2)
            return out_frame

        if not result.located:
            cv2.putText(out_frame, "Searching...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            return out_frame

        bbox = result.bbox or {}
        x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0)

        color_map = {"Good": (62, 207, 142), "Bad": (92, 92, 255), "Uncertain": (0, 200, 255)}
        color = color_map.get(result.classification, (200, 200, 200))

        cv2.rectangle(out_frame, (x, y), (x + w, y + h), color, thickness)

        label = f"{result.classification}"
        if result.good_similarity is not None:
            label += f" (good={result.good_similarity:.2f}"
            if result.bad_similarity is not None:
                label += f", bad={result.bad_similarity:.2f}"
            label += ")"
        cv2.putText(out_frame, label, (x, max(y - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, max(1, thickness - 1))

        return out_frame
