import os
import time
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from app.core.base_module import BaseModuleController
from app.core.event_bus import EventBus
from app.core.models.results import TargetExtractionResult
from app.modules.target_extraction.extractor import (
    clamp_box, extract_target, mask_stats, judge_extraction, build_cutout,
)
from app.modules.target_extraction.settings import TargetExtractionSettings

logger = logging.getLogger(__name__)

EXTRACTION_FILENAME = "last_extraction.png"


class TargetExtractionController(BaseModuleController):
    """AI Target Extraction: taught ROI + SAM/GrabCut segmentation per triggered frame.

    Design decisions
    ----------------
    * Heavy segmentation only on triggered frames (context["triggered"]).
      Non-triggered frames get the last result re-rendered, keeping Continuous
      mode from burning a model call on every camera frame.
    * Segmenter is lazy + injectable: tests pass a fake; production creates
      SamSegmenter on demand only when the SAM engine is actually needed.
    * Atomic write (tmp + os.replace) for last_extraction.png.
    * Persisted state (roi_box, roi_status) survives restart/module switch
      via save_data() / load_data() inherited from BaseModuleController.
    """

    def __init__(self, event_bus: EventBus, settings: TargetExtractionSettings,
                 segmenter=None):
        super().__init__("core-target-extraction", event_bus)
        self.settings = settings
        self._segmenter_override = segmenter   # injectable for tests

        # Load persisted settings with version migration
        saved_settings = self.load_settings()
        if saved_settings:
            norm_path = os.path.normpath(os.path.abspath(self.settings_file)) if self.settings_file else ""
            is_scratch = any(part.lower() == "scratch" for part in Path(norm_path).parts) if norm_path else False

            version = saved_settings.get("version", 0)
            if version < 1 and not is_scratch:
                saved_settings["version"] = 1
                self.save_settings(saved_settings)
            self.settings.update(saved_settings)

        # ROI state (restored from data.json)
        self.roi_status: str = "Untaught"
        self._roi_box: Optional[tuple] = None     # (x, y, w, h) or None

        saved_data = self.load_data()
        if saved_data and saved_data.get("roi_status") == "Taught":
            rb = saved_data.get("roi_box")
            if rb and all(k in rb for k in ("x", "y", "w", "h")):
                self._roi_box = (rb["x"], rb["y"], rb["w"], rb["h"])
                self.roi_status = "Taught"

        # Runtime state
        self.last_frame: Optional[np.ndarray] = None
        self.last_result: TargetExtractionResult = TargetExtractionResult()
        self._lazy_segmenter = None    # created on first SAM use
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        logger.info("TargetExtractionController initialized.")
        return True

    def configure(self, settings: Dict[str, Any]) -> bool:
        """Update settings with merged-state area ratio validation."""
        current = self.settings.get_settings()
        merged_min = settings.get("min_area_ratio", current.get("min_area_ratio", 0.10))
        merged_max = settings.get("max_area_ratio", current.get("max_area_ratio", 0.95))
        if merged_min >= merged_max:
            logger.error("TargetExtraction configure failed: min_area_ratio must be < max_area_ratio.")
            return False
        if self.settings.update(settings):
            self.save_settings(self.settings.get_settings())
            return True
        return False

    def cleanup(self) -> None:
        logger.info("TargetExtractionController cleaned up.")

    def health_check(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Raw-frame hook
    # ------------------------------------------------------------------

    def on_raw_frame(self, frame) -> None:
        """Store a fresh frame so set_roi() always has one available."""
        with self._lock:
            self.last_frame = frame.copy()

    # ------------------------------------------------------------------
    # ROI teach / reset
    # ------------------------------------------------------------------

    def set_roi(self, x: int, y: int, w: int, h: int) -> bool:
        """Validate the box against the latest raw frame and persist."""
        with self._lock:
            if self.last_frame is None:
                logger.error("No frame available yet - start the stream first.")
                return False
            frame = self.last_frame.copy()

        clamped = clamp_box((x, y, w, h), frame.shape)
        if clamped is None:
            logger.error(f"ROI box ({x},{y},{w},{h}) is too small or outside the frame.")
            return False

        cx, cy, cw, ch = clamped
        with self._lock:
            self._roi_box = (cx, cy, cw, ch)
            self.roi_status = "Taught"

        self.save_data({
            "roi_status": "Taught",
            "roi_box": {"x": cx, "y": cy, "w": cw, "h": ch},
        })
        self.event_bus.publish("TargetExtractionROISet", {"x": cx, "y": cy, "w": cw, "h": ch})
        return True

    def reset_roi(self) -> bool:
        """Clear ROI. Counters are not touched."""
        with self._lock:
            self._roi_box = None
            self.roi_status = "Untaught"

        self.save_data({"roi_status": "Untaught", "roi_box": {}})
        self.event_bus.publish("TargetExtractionROIReset", {})
        return True

    def get_extraction_path(self) -> Optional[str]:
        if not self.module_dir:
            return None
        p = os.path.join(self.module_dir, EXTRACTION_FILENAME)
        return p if os.path.exists(p) else None

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _get_segmenter(self, s: Dict[str, Any]):
        """Return the injectable override, or lazily create a SamSegmenter."""
        if self._segmenter_override is not None:
            return self._segmenter_override
        if self._lazy_segmenter is None:
            try:
                from app.modules.target_extraction.extractor import SamSegmenter
                weights = s.get("sam_weights", "models/mobile_sam.pt")
                self._lazy_segmenter = SamSegmenter(weights=weights)
            except Exception as exc:
                logger.warning(f"Could not create SamSegmenter: {exc}")
        return self._lazy_segmenter

    def process(self, context: Dict[str, Any]) -> TargetExtractionResult:
        triggered = context.get("triggered", True)
        s = self.settings.get_settings()
        judge_enabled = s.get("judge_enabled", False)

        with self._lock:
            roi_box = self._roi_box
            roi_status = self.roi_status

        # Non-triggered frame: re-render last result, no segmentation
        if not triggered:
            return self.last_result

        start = time.perf_counter()

        # No ROI configured
        if roi_status != "Taught" or roi_box is None:
            is_ok: Optional[bool] = None
            reason: Optional[str] = None
            if judge_enabled:
                is_ok = False
                reason = "not configured"
            result = TargetExtractionResult(
                roi_status="Untaught",
                is_ok=is_ok,
                reason=reason,
                custom_metadata={"reason": reason} if reason else {},
                latency_ms=(time.perf_counter() - start) * 1000,
                timestamp=time.time(),
            )
            self.last_result = result
            return result

        frame = context["frame"]
        engine = s.get("engine", "auto")
        iters = s.get("grabcut_iterations", 5)

        # Choose segmenter - only needed for SAM
        segmenter = None
        if engine in ("auto", "sam"):
            segmenter = self._get_segmenter(s)

        try:
            extraction = extract_target(
                frame_bgr=frame,
                box=roi_box,
                segmenter=segmenter,
                engine=engine,
                grabcut_iterations=iters,
            )
        except Exception as exc:
            logger.error(f"TargetExtractionController.process error: {exc}")
            result = TargetExtractionResult(
                roi_status="Taught",
                engine=None,
                fallback_reason=str(exc),
                is_ok=False if judge_enabled else None,
                reason=str(exc) if judge_enabled else None,
                custom_metadata={"error": str(exc)},
                latency_ms=(time.perf_counter() - start) * 1000,
                timestamp=time.time(),
            )
            self.last_result = result
            return result

        stats = mask_stats(extraction.mask, roi_box)
        area_ratio = stats["area_ratio"]

        is_ok = None
        reason = None
        if judge_enabled:
            min_r = s.get("min_area_ratio", 0.10)
            max_r = s.get("max_area_ratio", 0.95)
            is_ok = judge_extraction(area_ratio, min_r, max_r)
            if is_ok:
                reason = "area ratio OK"
            elif area_ratio < min_r:
                reason = f"area ratio {area_ratio:.2f} below {min_r:.2f}"
            else:
                reason = f"area ratio {area_ratio:.2f} above {max_r:.2f}"

        # Atomic write of last_extraction.png
        if self.module_dir:
            try:
                os.makedirs(self.module_dir, exist_ok=True)
                out_path = os.path.join(self.module_dir, EXTRACTION_FILENAME)
                tmp_path = out_path + ".tmp"
                # Save the cutout (BGRA) as PNG
                cutout = build_cutout(frame, extraction.mask, crop_to_bbox=True)
                success, buf = cv2.imencode(".png", cutout)
                if success:
                    with open(tmp_path, "wb") as f:
                        f.write(buf.tobytes())
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, out_path)
            except Exception as exc:
                logger.warning(f"Could not write last_extraction.png: {exc}")

        rx, ry, rw, rh = roi_box
        result = TargetExtractionResult(
            roi_status="Taught",
            engine=extraction.engine,
            fallback_reason=extraction.fallback_reason,
            area_px=stats["area_px"],
            area_ratio=area_ratio,
            bbox=stats["bbox"],
            centroid=stats["centroid"],
            roi={"x": rx, "y": ry, "w": rw, "h": rh},
            is_ok=is_ok,
            reason=reason,
            custom_metadata={"reason": reason} if reason else {},
            latency_ms=(time.perf_counter() - start) * 1000,
            timestamp=time.time(),
        )
        self.last_result = result
        self.event_bus.publish("TargetExtractionComplete", {
            "engine": extraction.engine,
            "area_ratio": area_ratio,
            "is_ok": is_ok,
        })
        return result

    def render(self, result: TargetExtractionResult) -> Any:
        return result
