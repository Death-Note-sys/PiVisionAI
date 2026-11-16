"""Pure template-matching helpers for Similar Object Counting.

No framework imports on purpose: everything here is plain numpy/OpenCV, so it
can be unit-tested without a camera, Flask or the Container.

Limits (by design, documented for operators):
  * Objects must share the taught orientation. Rotated copies are NOT found.
  * Size variation is only handled through the optional `scales` list.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Match:
    x: int
    y: int
    w: int
    h: int
    score: float


def validate_template(template_bgr: np.ndarray, min_size: int = 8, min_std: float = 3.0) -> Tuple[bool, str]:
    """A flat or tiny template can match anything; reject it at teach time."""
    if template_bgr is None or template_bgr.ndim < 2:
        return False, "template is empty"
    h, w = template_bgr.shape[:2]
    if h < min_size or w < min_size:
        return False, "template is smaller than %dx%d pixels" % (min_size, min_size)
    gray = template_bgr if template_bgr.ndim == 2 else cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < min_std:
        return False, "template has almost no texture (std %.1f < %.1f)" % (float(gray.std()), min_std)
    return True, ""


def _iou(a: Match, b: Match) -> float:
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(matches: Sequence[Match], iou_threshold: float) -> List[Match]:
    """Keep the best-scoring match among boxes that overlap more than iou_threshold."""
    kept: List[Match] = []
    for m in sorted(matches, key=lambda m: m.score, reverse=True):
        if all(_iou(m, k) <= iou_threshold for k in kept):
            kept.append(m)
    return kept


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _peaks_for_template(gray: np.ndarray, tmpl: np.ndarray, threshold: float,
                        suppress_frac: float, max_objects: int) -> List[Match]:
    """Iterative peak picking: take the best correlation peak, blank out the
    neighbourhood around it, repeat until the best remaining peak is below
    `threshold`. This avoids the 'plateau' of near-identical hits that
    surround every true object."""
    th, tw = tmpl.shape[:2]
    if gray.shape[0] < th or gray.shape[1] < tw:
        return []
    result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
    result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
    rh, rw = result.shape
    dx = max(1, int(tw * suppress_frac))
    dy = max(1, int(th * suppress_frac))
    out: List[Match] = []
    while len(out) < max_objects:
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < threshold:
            break
        x, y = max_loc
        out.append(Match(x=int(x), y=int(y), w=int(tw), h=int(th), score=float(max_val)))
        x0, x1 = max(0, x - dx + 1), min(rw, x + dx)
        y0, y1 = max(0, y - dy + 1), min(rh, y + dy)
        result[y0:y1, x0:x1] = -1.0
    return out


def find_matches(frame_bgr: np.ndarray,
                 template_bgr: np.ndarray,
                 threshold: float = 0.7,
                 scales: Sequence[float] = (1.0,),
                 suppress_frac: float = 0.5,
                 iou_threshold: float = 0.3,
                 max_objects: int = 200,
                 roi: Optional[Tuple[int, int, int, int]] = None,
                 work_scale: float = 1.0) -> List[Match]:
    """Find every region of `frame_bgr` that looks like `template_bgr`.

    threshold      normalized-correlation score needed (0..1, higher = stricter)
    scales         template size factors to try, e.g. (0.9, 1.0, 1.1)
    suppress_frac  two hits closer than this fraction of the template size are
                   treated as the same object (0.5 = half a template)
    iou_threshold  overlap above which hits from different scales are merged
    max_objects    safety cap per scale so a too-low threshold cannot flood
    roi            optional (x, y, w, h) search region in frame coordinates
    work_scale     <1.0 matches on a downscaled copy for speed
    Returned boxes are in full-frame coordinates, sorted in reading order.
    """
    if frame_bgr is None or template_bgr is None:
        return []
    ox = oy = 0
    search = frame_bgr
    if roi is not None:
        fh, fw = frame_bgr.shape[:2]
        rx, ry, rw, rh = roi
        rx, ry = max(0, int(rx)), max(0, int(ry))
        rw, rh = min(int(rw), fw - rx), min(int(rh), fh - ry)
        if rw <= 0 or rh <= 0:
            return []
        search = frame_bgr[ry:ry + rh, rx:rx + rw]
        ox, oy = rx, ry

    ws = float(work_scale) if work_scale and 0.1 <= work_scale <= 1.0 else 1.0
    gray = _to_gray(search)
    tmpl_gray = _to_gray(template_bgr)
    if ws != 1.0:
        gray = cv2.resize(gray, None, fx=ws, fy=ws, interpolation=cv2.INTER_AREA)
        tmpl_gray = cv2.resize(tmpl_gray, None, fx=ws, fy=ws, interpolation=cv2.INTER_AREA)

    all_hits: List[Match] = []
    for s in scales:
        if s <= 0:
            continue
        t = tmpl_gray if abs(s - 1.0) < 1e-6 else cv2.resize(
            tmpl_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
        if t.shape[0] < 4 or t.shape[1] < 4:
            continue
        all_hits.extend(_peaks_for_template(gray, t, threshold, suppress_frac, max_objects))

    merged = non_max_suppression(all_hits, iou_threshold) if len(scales) > 1 else all_hits
    merged = merged[:max_objects]

    inv = 1.0 / ws
    result = [Match(x=int(round(m.x * inv)) + ox, y=int(round(m.y * inv)) + oy,
                    w=int(round(m.w * inv)), h=int(round(m.h * inv)), score=m.score)
              for m in merged]
    result.sort(key=lambda m: (m.y, m.x))
    return result


def judge_count(count: int, count_min: int, count_max: int) -> bool:
    """OK/NG rule for a count.
    count_min = count_max = 0 means 'expect none' (OK only if count == 0).
    Otherwise OK when count_min <= count <= count_max, where count_max = 0
    means 'no upper limit'."""
    if count_min == 0 and count_max == 0:
        return count == 0
    upper = float("inf") if count_max == 0 else count_max
    return count_min <= count <= upper
