import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, Optional
from app.core.contracts import IModule
from app.core.event_bus import EventBus
from app.core.models.results import ColorResult
from .settings import ColorAnalysisSettings

logger = logging.getLogger(__name__)

class ColorAnalysisController(IModule):
    """Classical CV: K-means dominant color extraction, with an optional
    taught reference color compared via CIE Delta-E in LAB space (not raw
    RGB distance, which is not perceptually uniform)."""

    def __init__(self, event_bus: EventBus, settings: ColorAnalysisSettings):
        self.event_bus = event_bus
        self.settings = settings
        self.last_frame: Optional[np.ndarray] = None
        self.last_result: ColorResult = ColorResult()
        self.reference_status = "Untaught"
        self.reference_lab: Optional[np.ndarray] = None
        self.reference_rgb: Optional[tuple] = None

    def initialize(self) -> bool:
        logger.info("ColorAnalysisController initialized.")
        return True

    def configure(self, settings: Dict[str, Any]) -> bool:
        return self.settings.update(settings)

    def on_raw_frame(self, frame) -> None:
        """Cheap hook, mirrors AI Identify's pattern, so teach_reference()
        always has a reasonably fresh frame even under trigger gating."""
        self.last_frame = frame.copy()

    def _safe_crop(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        fh, fw = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w <= 5 or h <= 5:
            logger.error("Reference region too small or out of frame bounds.")
            return None
        return frame[y:y+h, x:x+w].copy()

    def teach_reference(self, x: int, y: int, w: int, h: int) -> bool:
        if self.last_frame is None:
            logger.error("Cannot teach: no frame available yet.")
            return False
        crop = self._safe_crop(self.last_frame, x, y, w, h)
        if crop is None:
            return False
        avg_bgr = crop.reshape(-1, 3).mean(axis=0)
        avg_bgr_img = np.uint8([[avg_bgr]])
        avg_lab = cv2.cvtColor(avg_bgr_img, cv2.COLOR_BGR2LAB)[0][0]
        self.reference_lab = avg_lab.astype(np.float64)
        self.reference_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))  # BGR -> RGB
        self.reference_status = "Taught"
        self.event_bus.publish("ColorReferenceTaught", {"rgb": self.reference_rgb})
        return True

    def reset_reference(self) -> bool:
        self.reference_lab = None
        self.reference_rgb = None
        self.reference_status = "Untaught"
        self.event_bus.publish("ColorReferenceReset", {})
        return True

    def _rgb_to_hex(self, rgb) -> str:
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    def process(self, context: Dict[str, Any]) -> ColorResult:
        frame = context["frame"]
        start = time.perf_counter()
        settings = self.settings.get_settings()
        num_colors = max(1, min(10, settings.get("num_colors", 5)))
        downsample_size = settings.get("downsample_size", 100)
        delta_e_threshold = settings.get("delta_e_threshold", 10.0)

        try:
            small = cv2.resize(frame, (downsample_size, downsample_size), interpolation=cv2.INTER_AREA)
            pixels = small.reshape(-1, 3).astype(np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
            _, labels, centers = cv2.kmeans(
                pixels, num_colors, None, criteria, attempts=3, flags=cv2.KMEANS_RANDOM_CENTERS
            )

            labels = labels.flatten()
            total = len(labels)
            dominant_colors = []
            for i, center in enumerate(centers):
                count = int(np.sum(labels == i))
                if count == 0:
                    continue
                b, g, r = int(center[0]), int(center[1]), int(center[2])
                dominant_colors.append({
                    "rgb": [r, g, b],
                    "hex": self._rgb_to_hex((r, g, b)),
                    "percentage": round(count / total * 100, 1),
                })
            dominant_colors.sort(key=lambda c: c["percentage"], reverse=True)

        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            result = ColorResult(reference_status=self.reference_status)
            self.last_result = result
            return result

        delta_e = None
        match = None
        if self.reference_status == "Taught" and dominant_colors:
            top_rgb = dominant_colors[0]["rgb"]
            top_bgr_img = np.uint8([[[top_rgb[2], top_rgb[1], top_rgb[0]]]])
            top_lab = cv2.cvtColor(top_bgr_img, cv2.COLOR_BGR2LAB)[0][0].astype(np.float64)
            delta_e = float(np.linalg.norm(top_lab - self.reference_lab))
            match = delta_e <= delta_e_threshold

        result = ColorResult(
            dominant_colors=dominant_colors,
            reference_status=self.reference_status,
            reference_color={"rgb": list(self.reference_rgb), "hex": self._rgb_to_hex(self.reference_rgb)} if self.reference_rgb else None,
            delta_e=round(delta_e, 2) if delta_e is not None else None,
            match=match,
            latency_ms=(time.perf_counter() - start) * 1000,
            timestamp=time.time(),
        )
        self.last_result = result

        if match is True:
            self.event_bus.publish("ColorMatch", {"delta_e": delta_e})
        elif match is False:
            self.event_bus.publish("ColorMismatch", {"delta_e": delta_e})

        return result

    def render(self, result: ColorResult) -> Any:
        return result

    def cleanup(self) -> None:
        logger.info("ColorAnalysisController cleaned up.")

    def health_check(self) -> bool:
        return True
