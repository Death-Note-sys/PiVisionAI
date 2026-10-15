import os
from pathlib import Path
import cv2
import numpy as np
import time
import logging
import threading
from typing import Dict, Any, Optional
from app.core.base_module import BaseModuleController
from app.core.event_bus import EventBus
from app.core.models.results import ColorResult
from .settings import ColorAnalysisSettings

logger = logging.getLogger(__name__)

class ColorAnalysisController(BaseModuleController):
    """Classical CV: K-means dominant color extraction, with an optional
    taught reference color compared via CIE Delta-E in LAB space (not raw
    RGB distance, which is not perceptually uniform)."""

    def __init__(self, event_bus: EventBus, settings: ColorAnalysisSettings):
        super().__init__("core-color-analysis", event_bus)
        self.settings = settings
        
        # Load settings
        saved_settings = self.load_settings()
        if saved_settings:
            norm_path = os.path.normpath(os.path.abspath(self.settings_file))
            is_scratch = any(part.lower() == "scratch" for part in Path(norm_path).parts)

            version = saved_settings.get("version", 0)
            if version < 1 and not is_scratch:
                old_thresh = saved_settings.get("delta_e_threshold")
                if old_thresh == 30.5:
                    new_thresh = 25.0
                    saved_settings["delta_e_threshold"] = new_thresh
                    logger.warning(
                        f"ColorAnalysis: Migrated settings file {self.settings_file} "
                        f"(delta_e_threshold: {old_thresh} -> {new_thresh}, version: {version} -> 1)"
                    )
                else:
                    logger.warning(
                        f"ColorAnalysis: Migrated settings file {self.settings_file} "
                        f"(delta_e_threshold kept custom: {old_thresh}, version: {version} -> 1)"
                    )
                saved_settings["version"] = 1
                self.save_settings(saved_settings)
            self.settings.update(saved_settings)
            
        # Load teach data
        saved_data = self.load_data()
        if saved_data and saved_data.get("reference_status") == "Taught":
            self.reference_status = "Taught"
            self.reference_box = tuple(saved_data.get("reference_box", []))
            rgb = saved_data.get("reference_rgb")
            if rgb:
                self.reference_rgb = tuple(rgb)
                bgr = (rgb[2], rgb[1], rgb[0])
                self.reference_lab = self._bgr_mean_to_lab(bgr)
        else:
            self.reference_status = "Untaught"
            self.reference_lab = None
            self.reference_rgb = None
            self.reference_box = None
            
        self.last_frame: Optional[np.ndarray] = None
        self.last_result: ColorResult = ColorResult()
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        logger.info("ColorAnalysisController initialized.")
        return True

    def configure(self, settings: Dict[str, Any]) -> bool:
        if self.settings.update(settings):
            self.save_settings(self.settings.get_settings())
            return True
        return False

    def on_raw_frame(self, frame) -> None:
        """Cheap hook, mirrors AI Identify's pattern, so teach_reference()
        always has a reasonably fresh frame even under trigger gating."""
        with self._lock:
            self.last_frame = frame.copy()

    def _safe_crop(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        fh, fw = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w <= 5 or h <= 5:
            logger.error("Reference region too small or out of frame bounds.")
            return None
        return frame[y:y+h, x:x+w].copy()

    def _bgr_mean_to_lab(self, bgr_mean) -> np.ndarray:
        px = (np.asarray(bgr_mean, np.float32) / 255.0).reshape(1, 1, 3)
        return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)

    def teach_reference(self, x: int, y: int, w: int, h: int) -> bool:
        with self._lock:
            if self.last_frame is None:
                logger.error("Cannot teach: no frame available yet.")
                return False
            last_frame = self.last_frame.copy()

        crop = self._safe_crop(last_frame, x, y, w, h)
        if crop is None:
            return False
        avg_bgr = crop.reshape(-1, 3).mean(axis=0)
        with self._lock:
            self.reference_lab = self._bgr_mean_to_lab(avg_bgr)
            self.reference_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))  # BGR -> RGB
            self.reference_status = "Taught"
            self.reference_box = (x, y, w, h)
            
        self.save_data({
            "reference_status": "Taught",
            "reference_box": self.reference_box,
            "reference_rgb": self.reference_rgb
        })
        self.save_image("reference.png", crop)
        
        self.event_bus.publish("ColorReferenceTaught", {"rgb": self.reference_rgb})
        return True

    def reset_reference(self) -> bool:
        with self._lock:
            self.reference_lab = None
            self.reference_rgb = None
            self.reference_status = "Untaught"
            self.reference_box = None
            
        self.save_data({"reference_status": "Untaught"})
        
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
        
        with self._lock:
            ref_status = self.reference_status
            ref_box = self.reference_box
            ref_lab = self.reference_lab
            ref_rgb = self.reference_rgb
            
        if ref_status == "Taught" and ref_box is not None:
            rx, ry, rw, rh = ref_box
            live_crop = self._safe_crop(frame, rx, ry, rw, rh)
            if live_crop is not None:
                live_avg_bgr = live_crop.reshape(-1, 3).mean(axis=0)
                live_avg_lab = self._bgr_mean_to_lab(live_avg_bgr)
                delta_e = float(np.linalg.norm(live_avg_lab - ref_lab))
                match = delta_e <= delta_e_threshold

        result = ColorResult(
            dominant_colors=dominant_colors,
            reference_status=ref_status,
            reference_color={"rgb": list(ref_rgb), "hex": self._rgb_to_hex(ref_rgb)} if ref_rgb else None,
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
