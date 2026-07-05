import cv2
import math
import time
import logging
from typing import Dict, Any, Optional
from app.core.contracts import IModule
from app.core.event_bus import EventBus
from app.core.models.results import MeasurementResult
from .settings import MeasurementSettings

logger = logging.getLogger(__name__)

class MeasurementController(IModule):
    """Classical CV dimension inspection. No AI runtime involved."""

    def __init__(self, event_bus: EventBus, settings: MeasurementSettings):
        self.event_bus = event_bus
        self.settings = settings
        self.pixels_per_cm: Optional[float] = None
        self.calibration_status: str = "Uncalibrated"
        self.last_result: MeasurementResult = MeasurementResult()

    def initialize(self) -> bool:
        logger.info("MeasurementController initialized.")
        return True

    def configure(self, settings: Dict[str, Any]) -> bool:
        return self.settings.update(settings)

    def calibrate(self, x1: int, y1: int, x2: int, y2: int, real_length_cm: float) -> bool:
        """Set pixels-per-cm from two clicked points and a known real-world distance."""
        pixel_distance = math.hypot(x2 - x1, y2 - y1)
        if pixel_distance <= 0 or real_length_cm <= 0:
            logger.error("Calibration failed: zero-length reference line or invalid real length.")
            return False
        self.pixels_per_cm = pixel_distance / real_length_cm
        self.calibration_status = "Calibrated"
        self.event_bus.publish("MeasurementCalibrated", {"pixels_per_cm": self.pixels_per_cm})
        logger.info(f"Measurement calibrated: {self.pixels_per_cm:.3f} px/cm")
        return True

    def process(self, context: Dict[str, Any]) -> MeasurementResult:
        frame = context["frame"]
        start = time.perf_counter()
        settings = self.settings.get_settings()

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, settings["canny_low"], settings["canny_high"])
            edges = cv2.dilate(edges, None, iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception as e:
            logger.error(f"Contour detection failed: {e}")
            return MeasurementResult(calibration_status=self.calibration_status, pixels_per_cm=self.pixels_per_cm)

        measurements = []
        for c in contours:
            area_px = cv2.contourArea(c)
            if area_px < settings["min_contour_area"]:
                continue

            x, y, w, h = cv2.boundingRect(c)
            perimeter_px = cv2.arcLength(c, True)

            entry = {
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "width_px": float(w),
                "height_px": float(h),
                "perimeter_px": float(perimeter_px),
                "area_px": float(area_px),
                "width_cm": None,
                "height_cm": None,
                "perimeter_cm": None,
                "area_cm2": None,
            }

            if self.pixels_per_cm:
                entry["width_cm"] = round(w / self.pixels_per_cm, 2)
                entry["height_cm"] = round(h / self.pixels_per_cm, 2)
                entry["perimeter_cm"] = round(perimeter_px / self.pixels_per_cm, 2)
                entry["area_cm2"] = round(area_px / (self.pixels_per_cm ** 2), 2)

            measurements.append(entry)

        latency_ms = (time.perf_counter() - start) * 1000

        result = MeasurementResult(
            measurements=measurements,
            calibration_status=self.calibration_status,
            pixels_per_cm=self.pixels_per_cm,
            latency_ms=latency_ms,
            timestamp=time.time(),
        )
        self.last_result = result
        return result

    def render(self, result: MeasurementResult) -> Any:
        return result

    def cleanup(self) -> None:
        logger.info("MeasurementController cleaned up.")

    def health_check(self) -> bool:
        return True
