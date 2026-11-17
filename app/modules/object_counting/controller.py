import os
import time
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.core.base_module import BaseModuleController
from app.core.event_bus import EventBus
from app.core.models.results import ObjectCountingResult
from app.modules.object_counting.matcher import find_matches, judge_count, validate_template
from app.modules.object_counting.settings import ObjectCountingSettings

logger = logging.getLogger(__name__)

TEMPLATE_FILENAME = "template.png"


class ObjectCountingController(BaseModuleController):
    """Similar Object Counting: teach one template, count every matching region
    per frame using normalized cross-correlation (see matcher.py for details).

    Limits (by design, documented for operators):
      * Rotated copies are NOT found unless added to the scales list later.
      * Size variation is handled through the allow_size_variation setting.
    """

    def __init__(self, event_bus: EventBus, settings: ObjectCountingSettings):
        super().__init__("core-object-counting", event_bus)
        self.settings = settings

        # Load persisted settings
        saved_settings = self.load_settings()
        if saved_settings:
            norm_path = os.path.normpath(os.path.abspath(self.settings_file)) if self.settings_file else ""
            is_scratch = any(part.lower() == "scratch" for part in Path(norm_path).parts) if norm_path else False

            version = saved_settings.get("version", 0)
            if version < 1 and not is_scratch:
                saved_settings["version"] = 1
                self.save_settings(saved_settings)
            self.settings.update(saved_settings)

        # Template state
        self.template_status: str = "Untaught"
        self.template_bgr: Optional[np.ndarray] = None
        self.taught_frame_width: Optional[int] = None
        self.taught_frame_height: Optional[int] = None
        self.last_teach_error: Optional[str] = None

        # Load persisted template image if previously taught
        saved_data = self.load_data()
        if saved_data and saved_data.get("template_status") == "Taught":
            img = self.load_image(TEMPLATE_FILENAME)
            if img is not None:
                self.template_bgr = img
                self.template_status = "Taught"
                self.taught_frame_width = saved_data.get("frame_width")
                self.taught_frame_height = saved_data.get("frame_height")
            else:
                logger.warning("ObjectCountingController: data.json says Taught but template.png missing; resetting.")
                self.template_status = "Untaught"
                self.taught_frame_width = None
                self.taught_frame_height = None

        self.last_frame: Optional[np.ndarray] = None
        self.last_result: ObjectCountingResult = ObjectCountingResult()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        logger.info("ObjectCountingController initialized.")
        return True

    def configure(self, settings: Dict[str, Any]) -> bool:
        """Update settings with the same merged-state count_min/count_max
        validation used by Object Detection."""
        current = self.settings.get_settings()
        merged_cmin = settings.get("count_min", current.get("count_min", 0))
        merged_cmax = settings.get("count_max", current.get("count_max", 0))
        if (merged_cmin is not None and merged_cmin < 0) or (merged_cmax is not None and merged_cmax < 0):
            logger.error("ObjectCounting configure failed: count_min and count_max must be >= 0.")
            return False
        if merged_cmax > 0 and merged_cmin > merged_cmax:
            logger.error("ObjectCounting configure failed: count_min cannot be greater than count_max.")
            return False
        if self.settings.update(settings):
            self.save_settings(self.settings.get_settings())
            return True
        return False

    def cleanup(self) -> None:
        logger.info("ObjectCountingController cleaned up.")

    def health_check(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Raw-frame hook (mirrors AI Identify / Color Analysis)
    # ------------------------------------------------------------------

    def on_raw_frame(self, frame) -> None:
        """Cheap hook so teach_template() always has a fresh frame even when
        inference is gated off by Single/Interval trigger mode."""
        with self._lock:
            self.last_frame = frame.copy()

    # ------------------------------------------------------------------
    # Teach / Reset
    # ------------------------------------------------------------------

    def _safe_crop(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        fh, fw = frame.shape[:2]
        x, y = max(0, int(x)), max(0, int(y))
        w, h = min(int(w), fw - x), min(int(h), fh - y)
        if w <= 0 or h <= 0:
            return None
        return frame[y:y + h, x:x + w].copy()

    def teach_template(self, x: int, y: int, w: int, h: int) -> bool:
        """Crop the region from the latest raw frame and store it as the template.

        Returns True on success, False on failure (sets self.last_teach_error).
        """
        with self._lock:
            if self.last_frame is None:
                self.last_teach_error = "No frame available yet - start the stream first."
                logger.error(self.last_teach_error)
                return False
            frame = self.last_frame.copy()

        fh, fw = frame.shape[:2]
        if w > fw or h > fh:
            self.last_teach_error = f"Template is larger than frame ({w}x{h} > {fw}x{fh})"
            logger.error(self.last_teach_error)
            return False
        if x < 0 or y < 0 or x >= fw or y >= fh:
            self.last_teach_error = "Region is out of frame bounds."
            logger.error(self.last_teach_error)
            return False

        crop = self._safe_crop(frame, x, y, w, h)
        if crop is None:
            self.last_teach_error = "Region is out of frame bounds or too small."
            logger.error(self.last_teach_error)
            return False

        ok, reason = validate_template(crop)
        if not ok:
            self.last_teach_error = reason
            logger.error(f"Cannot teach template: {reason}")
            return False

        # Atomic write: encode to buffer -> tmp file -> os.replace
        if self.module_dir:
            os.makedirs(self.module_dir, exist_ok=True)
            tpl_path = os.path.join(self.module_dir, TEMPLATE_FILENAME)
            tmp_path = tpl_path + ".tmp"
            success, buf = cv2.imencode(".png", crop)
            if not success:
                self.last_teach_error = "Failed to encode template image."
                return False
            with open(tmp_path, "wb") as f:
                f.write(buf.tobytes())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, tpl_path)

        with self._lock:
            self.template_bgr = crop
            self.template_status = "Taught"
            self.taught_frame_width = fw
            self.taught_frame_height = fh
            self.last_teach_error = None

        self.save_data({
            "template_status": "Taught",
            "frame_width": fw,
            "frame_height": fh,
        })
        self.event_bus.publish("ObjectCountingTemplateTaught", {"w": crop.shape[1], "h": crop.shape[0]})
        return True

    def reset_template(self) -> bool:
        """Clear the in-memory template and the persisted file.
        Counters are NOT touched - that is coordinator/state-store territory."""
        with self._lock:
            self.template_bgr = None
            self.template_status = "Untaught"
            self.taught_frame_width = None
            self.taught_frame_height = None
            self.last_teach_error = None

        self.save_data({"template_status": "Untaught"})

        if self.module_dir:
            tpl_path = os.path.join(self.module_dir, TEMPLATE_FILENAME)
            try:
                if os.path.exists(tpl_path):
                    os.remove(tpl_path)
            except OSError as e:
                logger.warning(f"ObjectCountingController: could not remove template.png: {e}")

        self.event_bus.publish("ObjectCountingTemplateReset", {})
        return True

    def get_template_path(self) -> Optional[str]:
        """Return the absolute path to template.png if it exists."""
        if not self.module_dir:
            return None
        p = os.path.join(self.module_dir, TEMPLATE_FILENAME)
        return p if os.path.exists(p) else None

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, context: Dict[str, Any]) -> ObjectCountingResult:
        frame = context["frame"]
        start = time.perf_counter()
        s = self.settings.get_settings()
        judge_enabled = s.get("judge_enabled", False)

        with self._lock:
            tmpl = self.template_bgr
            tpl_status = self.template_status
            taught_w = self.taught_frame_width
            taught_h = self.taught_frame_height

        if tpl_status != "Taught" or tmpl is None:
            # Untaught: count = 0, no judgment (or NG "not configured" if judging on)
            is_ok: Optional[bool] = None
            reason: Optional[str] = None
            if judge_enabled:
                is_ok = False  # NG - not configured
                reason = "not configured"
            result = ObjectCountingResult(
                template_status="Untaught",
                count=0,
                matches=[],
                is_ok=is_ok,
                reason=reason,
                custom_metadata={"reason": reason} if reason else {},
                latency_ms=(time.perf_counter() - start) * 1000,
                timestamp=time.time(),
            )
            self.last_result = result
            return result

        # Resolution safety check: if taught size is known and live frame size differs, flag re-teach
        fh, fw = frame.shape[:2]
        if taught_w is not None and taught_h is not None and (fw != taught_w or fh != taught_h):
            is_ok = False if judge_enabled else None
            reason = (
                f"template taught at {taught_w}x{taught_h}, frame is {fw}x{fh}"
                if judge_enabled else None
            )
            result = ObjectCountingResult(
                template_status="Re-teach (resolution changed)",
                count=0,
                matches=[],
                is_ok=is_ok,
                reason=reason,
                custom_metadata={"reason": reason} if reason else {},
                latency_ms=(time.perf_counter() - start) * 1000,
                timestamp=time.time(),
            )
            self.last_result = result
            return result

        # Build scales tuple
        if s.get("allow_size_variation", False):
            p = s.get("size_variation_pct", 10) / 100.0
            scales = (round(1.0 - p, 4), 1.0, round(1.0 + p, 4))
        else:
            scales = (1.0,)

        # Build ROI tuple if set
        roi_dict = s.get("search_roi", {})
        roi = None
        if roi_dict and all(k in roi_dict for k in ("x", "y", "w", "h")):
            roi = (roi_dict["x"], roi_dict["y"], roi_dict["w"], roi_dict["h"])

        hits = find_matches(
            frame_bgr=frame,
            template_bgr=tmpl,
            threshold=s.get("match_threshold", 0.7),
            scales=scales,
            suppress_frac=s.get("suppress_frac", 0.5),
            iou_threshold=s.get("iou_threshold", 0.3),
            max_objects=s.get("max_objects", 200),
            roi=roi,
            work_scale=s.get("work_scale", 1.0),
        )

        count = len(hits)
        matches_serialized: List[Dict[str, Any]] = [
            {"x": m.x, "y": m.y, "w": m.w, "h": m.h, "score": round(m.score, 4)}
            for m in hits
        ]

        is_ok = None
        reason = None
        if judge_enabled:
            cmin = s.get("count_min", 0)
            cmax = s.get("count_max", 0)
            is_ok = judge_count(count, cmin, cmax)
            reason = "count OK" if is_ok else f"count {count} outside [{cmin}, {cmax}]"

        result = ObjectCountingResult(
            template_status="Taught",
            count=count,
            matches=matches_serialized,
            is_ok=is_ok,
            reason=reason,
            custom_metadata={"reason": reason} if reason else {},
            latency_ms=(time.perf_counter() - start) * 1000,
            timestamp=time.time(),
        )
        self.last_result = result
        return result

    def render(self, result: ObjectCountingResult) -> Any:
        return result
