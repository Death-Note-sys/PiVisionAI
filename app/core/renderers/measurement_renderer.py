import cv2
import numpy as np
from typing import Dict, Any
from app.core.renderers.base_renderer import BaseRenderer

class MeasurementRenderer(BaseRenderer):
    """Draws bounding boxes, dimension labels, and calibration status."""

    def render(self, frame: np.ndarray, result: Any, metadata: Dict[str, Any]) -> np.ndarray:
        if not hasattr(result, "measurements"):
            return frame

        out_frame = frame.copy()
        settings = metadata.get("settings", {})
        show_dims = settings.get("show_dimensions", True)
        thickness = int(settings.get("thickness", 2))
        unit = settings.get("unit", "cm")

        for m in result.measurements:
            bbox = m.get("bbox", {})
            x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0)

            cv2.rectangle(out_frame, (x, y), (x + w, y + h), (0, 217, 198), thickness)

            if show_dims:
                if unit == "cm" and m.get("width_cm") is not None:
                    label = f"{m['width_cm']:.1f}x{m['height_cm']:.1f}cm"
                else:
                    label = f"{int(m['width_px'])}x{int(m['height_px'])}px"
                cv2.putText(out_frame, label, (x, max(y - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 217, 198), max(1, thickness - 1))

        status_text = f"Calibration: {result.calibration_status}"
        color = (62, 207, 142) if result.calibration_status == "Calibrated" else (92, 92, 255)
        cv2.putText(out_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return out_frame
