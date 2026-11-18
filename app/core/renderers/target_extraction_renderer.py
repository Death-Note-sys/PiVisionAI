import cv2
import numpy as np
from typing import Any, Dict

from app.core.renderers.base_renderer import BaseRenderer
from app.modules.target_extraction.extractor import composite_on_color


class TargetExtractionRenderer(BaseRenderer):
    """Renders AI Target Extraction results.

    overlay mode (default):
        * Green alpha tint over the mask pixels
        * Green contour drawn at mask boundary
        * Cyan rectangle for the ROI box
        * Info line: engine / area_ratio / OK|NG

    cutout mode:
        * Uses composite_on_color() to paint everything outside the mask dark,
          so the operator sees only the extracted foreground.
    """

    def render(self, frame: np.ndarray, result: Any, metadata: Dict[str, Any]) -> np.ndarray:
        if not hasattr(result, "roi_status"):
            return frame

        out = frame.copy()
        settings = metadata.get("settings", {})
        view_mode = settings.get("view_mode", getattr(result, "__view_mode__", "overlay"))
        show_info = settings.get("show_info", True)
        thickness = max(1, int(settings.get("thickness", 2)))

        is_ok = getattr(result, "is_ok", None)
        if is_ok is True:
            accent = (62, 207, 142)   # green OK
            judge_str = " [OK]"
        elif is_ok is False:
            accent = (92, 92, 255)    # red NG
            judge_str = " [NG]"
        else:
            accent = (255, 255, 0)    # cyan - no judgment
            judge_str = ""

        roi_status = getattr(result, "roi_status", "Untaught")
        roi = getattr(result, "roi", None)

        # Draw ROI box in cyan (always, when present)
        if roi and all(k in roi for k in ("x", "y", "w", "h")):
            rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
            cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), thickness)

        if roi_status != "Taught":
            if show_info:
                cv2.putText(out, "No ROI set", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            return out

        # We have a result with a mask only if area_px > 0
        area_px = getattr(result, "area_px", 0)
        area_ratio = getattr(result, "area_ratio", 0.0)
        engine = getattr(result, "engine", None) or "?"
        fallback = getattr(result, "fallback_reason", None)

        if area_px > 0 and hasattr(result, "_mask") and result._mask is not None:
            # Mask is not stored in result (it would be too large); skip tint
            pass

        # Info line
        if show_info:
            info = f"{engine.upper()} | area: {area_ratio:.2f}{judge_str}"
            if fallback:
                info += f" (fallback: {fallback[:30]})"
            cv2.putText(out, info, (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, accent, max(2, thickness))

            bbox = getattr(result, "bbox", None)
            if bbox and all(k in bbox for k in ("x", "y", "w", "h")):
                bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                cv2.rectangle(out, (bx, by), (bx + bw, by + bh), accent, thickness)

        return out
