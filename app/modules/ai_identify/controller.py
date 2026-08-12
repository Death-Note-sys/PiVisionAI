import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, Optional
from skimage.metrics import structural_similarity as ssim
from app.core.contracts import IModule
from app.core.event_bus import EventBus
from app.core.models.results import IdentifyResult
from .settings import AIIdentifySettings

logger = logging.getLogger(__name__)

class AIIdentifyController(IModule):
    """Classical CV pose-invariant part identification and good/bad
    differentiation, taught from two reference images. No AI model."""

    def __init__(self, event_bus: EventBus, settings: AIIdentifySettings):
        self.event_bus = event_bus
        self.settings = settings
        self.last_frame: Optional[np.ndarray] = None
        self.last_result: IdentifyResult = IdentifyResult()

        self.teach_status = "Untaught"
        self.good_reference: Optional[np.ndarray] = None
        self.bad_reference: Optional[np.ndarray] = None
        self.good_keypoints = None
        self.good_descriptors = None
        self.bad_keypoints = None
        self.bad_descriptors = None

        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def initialize(self) -> bool:
        logger.info("AIIdentifyController initialized.")
        return True

    def configure(self, settings: Dict[str, Any]) -> bool:
        return self.settings.update(settings)

    def _safe_crop(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        fh, fw = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w <= 5 or h <= 5:
            logger.error("Reference region too small or out of frame bounds.")
            return None
        return frame[y:y+h, x:x+w].copy()

    def _update_teach_status(self):
        if self.good_reference is not None and self.bad_reference is not None:
            self.teach_status = "Taught"
        elif self.good_reference is not None or self.bad_reference is not None:
            self.teach_status = "Partial"
        else:
            self.teach_status = "Untaught"

    def teach_good(self, x: int, y: int, w: int, h: int) -> bool:
        if self.last_frame is None:
            logger.error("Cannot teach: no frame available yet.")
            return False
        crop = self._safe_crop(self.last_frame, x, y, w, h)
        if crop is None:
            return False
        kp, des = self.orb.detectAndCompute(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None)
        if des is None or len(kp) < 10:
            logger.error("Good reference has too few distinctive features to track reliably.")
            return False
        self.good_reference = crop
        self.good_keypoints = kp
        self.good_descriptors = des
        self._update_teach_status()
        self.event_bus.publish("AIIdentifyTaughtGood", {})
        return True

    def teach_bad(self, x: int, y: int, w: int, h: int) -> bool:
        if self.last_frame is None:
            logger.error("Cannot teach: no frame available yet.")
            return False
        crop = self._safe_crop(self.last_frame, x, y, w, h)
        if crop is None:
            return False
        kp, des = self.orb.detectAndCompute(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None)
        if des is None or len(kp) < 10:
            logger.error("Bad reference has too few distinctive features to track reliably.")
            return False
        self.bad_reference = crop
        self.bad_keypoints = kp
        self.bad_descriptors = des
        self._update_teach_status()
        self.event_bus.publish("AIIdentifyTaughtBad", {})
        return True

    def reset_teaching(self) -> bool:
        self.good_reference = None
        self.bad_reference = None
        self.good_keypoints = None
        self.good_descriptors = None
        self.bad_keypoints = None
        self.bad_descriptors = None
        self.teach_status = "Untaught"
        self.event_bus.publish("AIIdentifyReset", {})
        return True

    def on_raw_frame(self, frame) -> None:
        """Cheap hook called every captured frame regardless of trigger
        mode, so teach_good()/teach_bad() always have a reasonably fresh
        frame to crop from even while full inference is gated behind
        Single/Interval trigger mode."""
        self.last_frame = frame.copy()

    def _match_against_reference(self, kp_frame, des_frame, frame,
                                  ref_keypoints, ref_descriptors, ref_image,
                                  ratio_thresh, min_match_count):
        """Attempt to locate+align the live frame against ONE reference
        (Good or Bad, called independently for each). Returns a dict with
        match quality and an SSIM score computed in THIS reference's own
        aligned space, or None if no valid geometric match was found."""
        if des_frame is None or ref_descriptors is None:
            return None

        matches = self.matcher.knnMatch(ref_descriptors, des_frame, k=2)
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

        if len(good_matches) < min_match_count:
            return None

        src_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        inlier_ratio = float(mask.sum()) / len(mask) if mask is not None and len(mask) > 0 else 0.0

        ref_h, ref_w = ref_image.shape[:2]
        ref_corners = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)
        frame_corners = cv2.perspectiveTransform(ref_corners, H)
        x_coords = frame_corners[:, 0, 0]
        y_coords = frame_corners[:, 0, 1]
        bbox = {
            "x": int(max(0, x_coords.min())),
            "y": int(max(0, y_coords.min())),
            "w": int(x_coords.max() - x_coords.min()),
            "h": int(y_coords.max() - y_coords.min()),
        }

        H_inv = np.linalg.inv(H)
        warped = cv2.warpPerspective(frame, H_inv, (ref_w, ref_h))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        ssim_score = float(ssim(ref_gray, warped_gray))

        return {
            "inlier_ratio": inlier_ratio,
            "bbox": bbox,
            "ssim_score": ssim_score,
        }

    def process(self, context: Dict[str, Any]) -> IdentifyResult:
        frame = context["frame"]
        self.last_frame = frame.copy()
        start = time.perf_counter()

        if self.teach_status != "Taught":
            result = IdentifyResult(teach_status=self.teach_status)
            self.last_result = result
            return result

        settings = self.settings.get_settings()
        min_match_count = settings.get("min_match_count", 10)
        ratio_thresh = settings.get("match_ratio_threshold", 0.75)
        ssim_margin = settings.get("classification_margin", 0.05)

        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_frame, des_frame = self.orb.detectAndCompute(gray_frame, None)

            good_match = self._match_against_reference(
                kp_frame, des_frame, frame,
                self.good_keypoints, self.good_descriptors, self.good_reference,
                ratio_thresh, min_match_count,
            )
            bad_match = self._match_against_reference(
                kp_frame, des_frame, frame,
                self.bad_keypoints, self.bad_descriptors, self.bad_reference,
                ratio_thresh, min_match_count,
            )

            if good_match is None and bad_match is None:
                result = IdentifyResult(
                    teach_status=self.teach_status, located=False,
                    latency_ms=(time.perf_counter() - start) * 1000, timestamp=time.time(),
                )
                self.last_result = result
                return result

            if good_match and not bad_match:
                classification = "Good"
                chosen = good_match
            elif bad_match and not good_match:
                classification = "Bad"
                chosen = bad_match
            else:
                good_score = good_match["ssim_score"]
                bad_score = bad_match["ssim_score"]
                if abs(good_score - bad_score) < ssim_margin:
                    classification = "Uncertain"
                elif good_score >= bad_score:
                    classification = "Good"
                else:
                    classification = "Bad"
                chosen = good_match if classification != "Bad" else bad_match

            result = IdentifyResult(
                teach_status=self.teach_status,
                located=True,
                bbox=chosen["bbox"],
                classification=classification,
                good_similarity=round(good_match["ssim_score"], 4) if good_match else None,
                bad_similarity=round(bad_match["ssim_score"], 4) if bad_match else None,
                match_confidence=round(chosen["inlier_ratio"], 4),
                latency_ms=(time.perf_counter() - start) * 1000,
                timestamp=time.time(),
            )
            self.last_result = result

            if classification == "Bad":
                self.event_bus.publish("PartRejected", {"good_similarity": result.good_similarity, "bad_similarity": result.bad_similarity})
            elif classification == "Good":
                self.event_bus.publish("PartAccepted", {"good_similarity": result.good_similarity})

            return result

        except Exception as e:
            logger.error(f"AI Identify processing failed: {e}", exc_info=True)
            result = IdentifyResult(teach_status=self.teach_status)
            self.last_result = result
            return result

    def render(self, result: IdentifyResult) -> Any:
        return result

    def cleanup(self) -> None:
        logger.info("AIIdentifyController cleaned up.")

    def health_check(self) -> bool:
        return True
