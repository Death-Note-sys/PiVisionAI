import cv2
import numpy as np
from typing import Dict, Any
from app.core.renderers.base_renderer import BaseRenderer

class ColorAnalysisRenderer(BaseRenderer):
    """Draws a dominant-color palette strip and, if taught, a match/mismatch badge."""

    def render(self, frame: np.ndarray, result: Any, metadata: Dict[str, Any]) -> np.ndarray:
        if not hasattr(result, "dominant_colors"):
            return frame

        out_frame = frame.copy()
        settings = metadata.get("settings", {})
        show_colors = settings.get("show_dominant_colors", True)
        show_match = settings.get("show_reference_match", True)

        if show_colors and result.dominant_colors:
            swatch_w = 40
            swatch_h = 40
            x0, y0 = 10, 10
            for i, c in enumerate(result.dominant_colors[:5]):
                r, g, b = c["rgb"]
                x = x0 + i * (swatch_w + 4)
                cv2.rectangle(out_frame, (x, y0), (x + swatch_w, y0 + swatch_h), (b, g, r), -1)
                cv2.rectangle(out_frame, (x, y0), (x + swatch_w, y0 + swatch_h), (255, 255, 255), 1)
                cv2.putText(out_frame, f"{c['percentage']:.0f}%", (x, y0 + swatch_h + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if show_match and result.reference_status == "Taught":
            if result.match is True:
                text = f"MATCH (dE={result.delta_e:.1f})"
                color = (62, 207, 142)
            elif result.match is False:
                text = f"NO MATCH (dE={result.delta_e:.1f})"
                color = (92, 92, 255)
            else:
                text = "Reference taught"
                color = (200, 200, 200)
            cv2.putText(out_frame, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif show_match:
            cv2.putText(out_frame, "No reference taught", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

        return out_frame
