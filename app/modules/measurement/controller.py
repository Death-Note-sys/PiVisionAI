import cv2
import math
import time
import logging
from typing import Dict, Any, Optional
from app.core.base_module import BaseModuleController
from app.core.event_bus import EventBus
from app.core.models.results import MeasurementResult
from .settings import MeasurementSettings

logger = logging.getLogger(__name__)

class MeasurementController(BaseModuleController):
    """Classical CV dimension inspection. No AI runtime involved."""

    def __init__(self, event_bus: EventBus, settings: MeasurementSettings):
        super().__init__("core-measurement", event_bus)
        self.settings = settings
        
        # Load settings
        saved_settings = self.load_settings()
        if saved_settings:
            self.settings.update(saved_settings)
            
        # Load calibration data
        saved_data = self.load_data()
        if "pixels_per_mm" in saved_data:
            self.pixels_per_mm = saved_data.get("pixels_per_mm")
        elif "pixels_per_cm" in saved_data:
            cm_val = saved_data.get("pixels_per_cm")
            if cm_val is not None and cm_val > 0:
                self.pixels_per_mm = float(cm_val) / 10.0
                self.save_data({"pixels_per_mm": self.pixels_per_mm})
                logger.info(f"Migrated legacy pixels_per_cm {cm_val} -> pixels_per_mm {self.pixels_per_mm}")
            else:
                self.pixels_per_mm = None
        else:
            self.pixels_per_mm = None
        self.calibration_status = "Calibrated" if self.pixels_per_mm else "Uncalibrated"
        
        self.last_result: MeasurementResult = MeasurementResult()

    def initialize(self) -> bool:
        logger.info("MeasurementController initialized.")
        return True

    def configure(self, settings: Dict[str, Any]) -> bool:
        current = self.settings.get_settings()
        merged_judge = settings["judge_enabled"] if "judge_enabled" in settings else current.get("judge_enabled", False)
        merged_tol = settings["tolerance_mm"] if "tolerance_mm" in settings else current.get("tolerance_mm", 0.0)

        if merged_judge:
            if merged_tol is None or merged_tol <= 0:
                logger.error("Measurement configure failed: tolerance_mm must be > 0 when judge_enabled is True.")
                return False
        if self.settings.update(settings):
            self.save_settings(self.settings.get_settings())
            return True
        return False


    def calibrate(self, x1: int, y1: int, x2: int, y2: int, real_length_mm: float) -> bool:
        """Set pixels-per-mm from two clicked points and a known real-world distance."""
        pixel_distance = math.hypot(x2 - x1, y2 - y1)
        if pixel_distance <= 0 or real_length_mm <= 0:
            logger.error("Calibration failed: zero-length reference line or invalid real length.")
            return False
        self.pixels_per_mm = pixel_distance / real_length_mm
        self.calibration_status = "Calibrated"
        
        self.save_data({"pixels_per_mm": self.pixels_per_mm})
        
        self.event_bus.publish("MeasurementCalibrated", {"pixels_per_mm": self.pixels_per_mm})
        logger.info(f"Measurement calibrated: {self.pixels_per_mm:.3f} px/mm")
        return True

    def reset_calibration(self) -> bool:
        """Reset calibration to Uncalibrated and clear saved calibration data."""
        self.pixels_per_mm = None
        self.calibration_status = "Uncalibrated"
        self.save_data({"pixels_per_mm": None})
        self.event_bus.publish("MeasurementCalibrationReset", {})
        logger.info("Measurement calibration reset to Uncalibrated.")
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
            return MeasurementResult(calibration_status=self.calibration_status, pixels_per_mm=self.pixels_per_mm)

        measurements = []
        for c in contours:
            area_px = cv2.contourArea(c)
            if area_px < settings["min_contour_area"]:
                continue

            rect = cv2.minAreaRect(c)
            x, y, _, _ = cv2.boundingRect(c)
            w, h = rect[1]
            perimeter_px = cv2.arcLength(c, True)

            entry = {
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "width_px": float(w),
                "height_px": float(h),
                "perimeter_px": float(perimeter_px),
                "area_px": float(area_px),
                "width_mm": None,
                "height_mm": None,
                "perimeter_mm": None,
                "area_mm2": None,
            }

            if self.pixels_per_mm:
                entry["width_mm"] = round(w / self.pixels_per_mm, 2)
                entry["height_mm"] = round(h / self.pixels_per_mm, 2)
                entry["perimeter_mm"] = round(perimeter_px / self.pixels_per_mm, 2)
                entry["area_mm2"] = round(area_px / (self.pixels_per_mm ** 2), 2)

            measurements.append(entry)

        is_ok = None
        judge_enabled = settings.get("judge_enabled", False)
        target_w = settings.get("target_width_mm", 0.0)
        target_l = settings.get("target_length_mm", 0.0)
        tol = settings.get("tolerance_mm", 0.0)
        roi = settings.get("judgment_roi", {})
        
        if judge_enabled and self.pixels_per_mm and tol > 0 and (target_w > 0 or target_l > 0):
            is_ok = False
            for m in measurements:
                x, y, w, h = m["bbox"]["x"], m["bbox"]["y"], m["bbox"]["w"], m["bbox"]["h"]
                cx, cy = x + w/2, y + h/2
                if roi and roi.get("w", 0) > 0 and roi.get("h", 0) > 0:
                    if not (roi["x"] <= cx <= roi["x"]+roi["w"] and roi["y"] <= cy <= roi["y"]+roi["h"]):
                        continue
                
                w_mm = m["width_mm"]
                h_mm = m["height_mm"]
                if w_mm is not None and h_mm is not None:
                    long_side = max(w_mm, h_mm)
                    short_side = min(w_mm, h_mm)
                    
                    if target_l > 0 and target_w > 0:
                        if abs(long_side - target_l) <= tol and abs(short_side - target_w) <= tol:
                            is_ok = True
                            break
                    elif target_l > 0:
                        if abs(long_side - target_l) <= tol:
                            is_ok = True
                            break
                    elif target_w > 0:
                        if abs(short_side - target_w) <= tol or abs(long_side - target_w) <= tol:
                            is_ok = True
                            break


        latency_ms = (time.perf_counter() - start) * 1000

        result = MeasurementResult(
            measurements=measurements,
            calibration_status=self.calibration_status,
            pixels_per_mm=self.pixels_per_mm,
            pixels_per_cm=self.pixels_per_mm * 10.0 if self.pixels_per_mm else None,
            latency_ms=latency_ms,
            timestamp=time.time(),
            is_ok=is_ok
        )
        self.last_result = result
        return result

    def render(self, result: MeasurementResult) -> Any:
        return result

    def cleanup(self) -> None:
        logger.info("MeasurementController cleaned up.")

    def health_check(self) -> bool:
        return True
