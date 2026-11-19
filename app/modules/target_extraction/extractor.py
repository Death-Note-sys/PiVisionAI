"""Pure helpers for AI Target Extraction (background removal around a target).

Engines:
  * "sam"      box-prompted, class-agnostic segmentation model (primary).
               Wrapped by SamSegmenter below; NOT exercised by the unit tests
               because it needs the model weights. Verify it with the spike
               script before relying on it.
  * "grabcut"  classical OpenCV GrabCut seeded with the same box (fallback).
  * "auto"     try SAM first, fall back to GrabCut and report why.

Everything else (mask clean-up, statistics, cut-out, fallback logic) is plain
numpy/OpenCV and is unit-tested.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class Extraction:
    mask: np.ndarray                 # uint8, 0 or 255, same size as the frame
    engine: str                      # "sam" or "grabcut"
    fallback_reason: Optional[str] = None


def clamp_box(box: Box, frame_shape) -> Optional[Box]:
    """Clip a box to the frame. Returns None if nothing usable is left."""
    fh, fw = frame_shape[:2]
    x, y, w, h = (int(v) for v in box)
    x, y = max(0, x), max(0, y)
    w, h = min(w, fw - x), min(h, fh - y)
    if w < 8 or h < 8:
        return None
    return x, y, w, h


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected blob (drops speckle and stray islands)."""
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Smooth the edge, close small holes, keep the largest blob."""
    m = (mask > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return largest_component(m)


def grabcut_mask(frame_bgr: np.ndarray, box: Box, iterations: int = 5,
                 margin: Optional[float] = 0.25) -> np.ndarray:
    """Foreground mask (0/255, full frame size) from GrabCut seeded with `box`.

    GrabCut's cost grows with the number of pixels it works on, not with the
    size of the target, so by default it runs on a crop: the box plus `margin`
    (a fraction of the box size) of surrounding background, which is all the
    background it needs to learn from. On a 1000x1333 frame this is roughly
    10x faster with an identical mask. margin=None runs on the whole frame.
    """
    clamped = clamp_box(box, frame_bgr.shape)
    if clamped is None:
        raise ValueError("box is empty or outside the frame")
    x, y, w, h = clamped
    fh, fw = frame_bgr.shape[:2]
    if margin is None:
        x0, y0, x1, y1 = 0, 0, fw, fh
    else:
        px, py = int(w * margin), int(h * margin)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(fw, x + w + px), min(fh, y + h + py)
    crop = frame_bgr[y0:y1, x0:x1]
    cv2.setRNGSeed(0)  # GrabCut's GMM init uses k-means; fix the seed for repeatable masks
    mask = np.zeros(crop.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, (x - x0, y - y0, w, h), bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    fg = np.zeros(frame_bgr.shape[:2], np.uint8)
    fg[y0:y1, x0:x1] = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return fg


def mask_stats(mask: np.ndarray, box: Box) -> Dict[str, Any]:
    """Area, area ratio against the prompt box, bounding box and centroid."""
    binary = (mask > 0).astype(np.uint8)
    area = int(binary.sum())
    box_area = max(1, int(box[2]) * int(box[3]))
    if area == 0:
        return {"area_px": 0, "area_ratio": 0.0, "bbox": None, "centroid": None}
    x, y, w, h = cv2.boundingRect(binary)
    m = cv2.moments(binary, binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    return {
        "area_px": area,
        "area_ratio": round(area / box_area, 4),
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "centroid": {"x": round(float(cx), 1), "y": round(float(cy), 1)},
    }


def judge_extraction(area_ratio: float, min_ratio: float, max_ratio: float) -> bool:
    """OK when the target fills a sensible share of the box.
    A missing part gives a tiny ratio; a leaking mask gives one near/above 1."""
    return min_ratio <= area_ratio <= max_ratio


def build_cutout(frame_bgr: np.ndarray, mask: np.ndarray, crop_to_bbox: bool = True) -> np.ndarray:
    """BGRA image: the target with a transparent background."""
    bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = (mask > 0).astype(np.uint8) * 255
    if crop_to_bbox:
        ys, xs = np.where(mask > 0)
        if len(xs):
            bgra = bgra[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return bgra


def composite_on_color(frame_bgr: np.ndarray, mask: np.ndarray, color=(24, 24, 24)) -> np.ndarray:
    """Frame-sized BGR image with everything outside the mask painted `color`.
    Used by the renderer so the operator sees the background removed."""
    out = np.empty_like(frame_bgr)
    out[:] = color
    keep = mask > 0
    out[keep] = frame_bgr[keep]
    return out


class SamSegmenter:
    """Box-prompted, class-agnostic segmentation through Ultralytics SAM.

    UNVERIFIED against your installed ultralytics version: confirm the class
    name, the weights file name and the result layout with the spike script
    (scripts in the prompt file) before wiring this into the module.
    The weights file must be present locally for offline machines.
    """

    def __init__(self, weights: str = "mobile_sam.pt"):
        self.weights = weights
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import SAM  # imported lazily: heavy, and optional
            self._model = SAM(self.weights)
        return self._model

    def segment(self, frame_bgr: np.ndarray, box: Box) -> np.ndarray:
        clamped = clamp_box(box, frame_bgr.shape)
        if clamped is None:
            raise ValueError("box is empty or outside the frame")
        x, y, w, h = clamped
        results = self._load()(frame_bgr, bboxes=[x, y, x + w, y + h], verbose=False)
        masks = results[0].masks
        if masks is None or len(masks.data) == 0:
            raise RuntimeError("SAM returned no mask for this box")
        m = masks.data[0].cpu().numpy() if hasattr(masks.data[0], "cpu") else np.asarray(masks.data[0])
        m = (m > 0.5).astype(np.uint8) * 255
        if m.shape != frame_bgr.shape[:2]:
            m = cv2.resize(m, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        return m


def extract_target(frame_bgr: np.ndarray, box: Box, segmenter=None,
                   engine: str = "auto", grabcut_iterations: int = 5) -> Extraction:
    """Run the chosen engine and return a cleaned mask plus which engine ran.

    engine="sam"      needs `segmenter`; raises if it is missing or fails.
    engine="grabcut"  never touches the segmenter.
    engine="auto"     SAM when a segmenter is given, otherwise GrabCut; if SAM
                      raises, falls back to GrabCut and records the reason so
                      the fallback is never silent.
    """
    if engine not in ("auto", "sam", "grabcut"):
        raise ValueError("engine must be 'auto', 'sam' or 'grabcut'")
    reason = None
    if engine in ("auto", "sam"):
        if segmenter is None:
            if engine == "sam":
                raise ValueError("engine='sam' requires a segmenter")
            reason = "no SAM segmenter available"
        else:
            try:
                return Extraction(clean_mask(segmenter.segment(frame_bgr, box)), "sam", None)
            except Exception as exc:  # noqa: BLE001 - any SAM failure triggers the fallback
                if engine == "sam":
                    raise
                reason = "SAM failed: %s" % exc
    return Extraction(clean_mask(grabcut_mask(frame_bgr, box, grabcut_iterations)), "grabcut", reason)
