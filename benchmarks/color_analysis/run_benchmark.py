"""
Color Analysis Delta-E Benchmark
=================================
Teaches the ColorAnalysisController from a reference image (or a crop of it),
then probes every other image and reports the dominant-color delta_e score.

Goal: establish what delta_e values look like for:
  - Same object, same shot (noise floor — should be ~0)
  - Same object, different background/position (same-object variation)
  - Genuinely different colors (the separation margin)

Run from the project root:
    python benchmarks/color_analysis/run_benchmark.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
from unittest.mock import MagicMock
from app.modules.color_analysis.controller import ColorAnalysisController
from app.modules.color_analysis.settings import ColorAnalysisSettings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
AI_GOOD  = os.path.join(BASE, "../ai_identify/good")
AI_BAD   = os.path.join(BASE, "../ai_identify/bad")
AI_PROBES = os.path.join(BASE, "../ai_identify/probes")
OCR_PROBES = os.path.join(BASE, "../ocr/probes")

# Reference: ₹2 coin — our best-photographed, single-material surface.
# Crop matches the HoughCircles box used in the AI Identify benchmark.
REFERENCE_PATH = os.path.join(AI_GOOD, "rupee2_coin.jpg")
REFERENCE_CROP = (164, 320, 400, 400)   # x, y, w, h  (from HoughCircles)

# Probes: (path, label, expected_relationship)
#   "SAME"    — same object, should score low delta_e
#   "SIMILAR" — similar material/tone, moderate delta_e expected
#   "DIFF"    — clearly different colour, should score high delta_e
PROBES = [
    # ── Same object ─────────────────────────────────────────────────────────
    (REFERENCE_PATH,                                        "rupee2_coin.jpg (reference self)",        "SAME"),
    (os.path.join(AI_PROBES, "probe_rupee2_on_table.jpg"), "probe_rupee2_on_table (same coin, diff bg)", "SAME"),

    # ── Similar material (silver/grey coins) ────────────────────────────────
    (os.path.join(AI_BAD,   "rupee20_coin.jpg"),           "rupee20_coin (bimetallic — gold+silver)", "SIMILAR"),
    (os.path.join(AI_PROBES,"probe_rupee20_on_table.jpg"), "probe_rupee20_on_table (bimetallic, diff bg)", "SIMILAR"),
    (os.path.join(AI_PROBES,"probe_heads_1.jpg"),          "probe_heads_1 (US quarter, silver, white bg)", "SIMILAR"),
    (os.path.join(AI_PROBES,"probe_heads_2.jpg"),          "probe_heads_2 (US quarter, silver, white bg)", "SIMILAR"),
    (os.path.join(AI_PROBES,"probe_tails_1.jpg"),          "probe_tails_1 (US quarter reverse)",      "SIMILAR"),
    (os.path.join(AI_PROBES,"probe_tails_2.jpg"),          "probe_tails_2 (US quarter reverse)",      "SIMILAR"),
    (os.path.join(AI_PROBES,"probe_wrong_coin.png"),       "probe_wrong_coin (copper penny)",         "SIMILAR"),

    # ── Clearly different colours ────────────────────────────────────────────
    (os.path.join(AI_PROBES,"probe_masterchow_lid.jpg"),   "probe_masterchow_lid (RED plastic lid)",  "DIFF"),
    (os.path.join(AI_PROBES,"probe_unrelated.jpg"),        "probe_unrelated (YELLOW banana)",         "DIFF"),
    (os.path.join(OCR_PROBES,"ocr_stop_sign.jpg"),         "ocr_stop_sign (RED sign)",                "DIFF"),
    (os.path.join(OCR_PROBES,"ocr_caution_sign.jpg"),      "ocr_caution_sign (YELLOW sign)",          "DIFF"),
    (os.path.join(OCR_PROBES,"ocr_speed_limit.jpg"),       "ocr_speed_limit (WHITE/BLACK sign)",      "DIFF"),

    # ── Synthetic textures (noise floor sanity) ──────────────────────────────
    (os.path.join(AI_PROBES,"probe_good.png"),             "probe_good.png (synthetic grey texture)", "SIMILAR"),
    (os.path.join(AI_PROBES,"probe_bad.png"),              "probe_bad.png (synthetic grey texture)",  "SIMILAR"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"  [WARN] Could not load: {path}")
    return img


def build_controller():
    event_bus = MagicMock()
    settings = ColorAnalysisSettings()
    ctrl = ColorAnalysisController(event_bus, settings)
    ctrl.initialize()
    return ctrl


def dominant_color_str(colors):
    if not colors:
        return "—"
    top = colors[0]
    return f"rgb({top['rgb'][0]:3d},{top['rgb'][1]:3d},{top['rgb'][2]:3d}) {top['hex']}  {top['percentage']:.0f}%"

def calc_old_delta_e(ref_img, rx, ry, rw, rh, probe_img):
    ref_crop = ref_img[ry:ry+rh, rx:rx+rw]
    ref_avg = ref_crop.reshape(-1, 3).mean(axis=0)
    ref_lab = cv2.cvtColor(np.uint8([[ref_avg]]), cv2.COLOR_BGR2LAB)[0][0].astype(np.float64)
    
    probe_crop = probe_img[ry:ry+rh, rx:rx+rw]
    probe_avg = probe_crop.reshape(-1, 3).mean(axis=0)
    probe_lab = cv2.cvtColor(np.uint8([[probe_avg]]), cv2.COLOR_BGR2LAB)[0][0].astype(np.float64)
    return float(np.linalg.norm(probe_lab - ref_lab))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 72)
    print("Color Analysis Delta-E Benchmark")
    print(f"Reference: {os.path.basename(REFERENCE_PATH)}  crop={REFERENCE_CROP}")
    print("=" * 72)

    ctrl = build_controller()

    # Teach from reference image
    ref_img = load_image(REFERENCE_PATH)
    if ref_img is None:
        print("ERROR: Could not load reference image. Stopping.")
        return

    ctrl.on_raw_frame(ref_img)
    x, y, w, h = REFERENCE_CROP
    ok = ctrl.teach_reference(x, y, w, h)
    if not ok:
        print("ERROR: teach_reference() failed. Stopping.")
        return

    ref_rgb = ctrl.reference_rgb
    print(f"Reference color taught: rgb({ref_rgb[0]},{ref_rgb[1]},{ref_rgb[2]})  "
          f"hex={ctrl._rgb_to_hex(ref_rgb)}")
    print()

    # Column header
    hdr = f"{'Probe':<50} {'Rel':<7} {'Old DE':>7} {'New DE':>7}  {'Dominant color'}"
    print(hdr)
    print("-" * 90)

    results = {"SAME": [], "SIMILAR": [], "DIFF": []}

    for path, label, rel in PROBES:
        img = load_image(path)
        if img is None:
            print(f"  {'MISSING':<50}  —")
            continue

        result = ctrl.process({"frame": img})
        de = result.delta_e
        
        try:
            old_de = calc_old_delta_e(ref_img, x, y, w, h, img)
            old_de_str = f"{old_de:7.2f}"
        except:
            old_de_str = "    N/A"
            
        dom = dominant_color_str(result.dominant_colors)
        de_str = f"{de:7.2f}" if de is not None else "    N/A"
        print(f"{label:<50} {rel:<7} {old_de_str} {de_str}  {dom}")
        if de is not None:
            results[rel].append((label, de))

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)

    for rel, entries in results.items():
        if not entries:
            continue
        values = [v for _, v in entries]
        print(f"\n{rel} ({len(entries)} probes):")
        print(f"  min={min(values):.2f}  max={max(values):.2f}  mean={sum(values)/len(values):.2f}")
        for label, v in entries:
            print(f"  {v:7.2f}  {label}")

    print()
    print("Interpretation guide:")
    print("  SAME max   -> upper bound on noise from position/bg change")
    print("  SIMILAR    -> objects with similar tone; threshold must sit ABOVE these")
    print("  DIFF min   -> lowest 'clearly wrong' score; threshold must sit BELOW this")
    print()

    same_vals  = [v for _, v in results["SAME"]]
    diff_vals  = [v for _, v in results["DIFF"]]

    if same_vals and diff_vals:
        same_max = max(same_vals)
        diff_min = min(diff_vals)
        print(f"  Safe threshold window:  ({same_max:.1f}, {diff_min:.1f})")
        if same_max < diff_min:
            mid = (same_max + diff_min) / 2
            print(f"  Suggested threshold:    ~{mid:.1f}  (midpoint of window)")
            print(f"  Current default (10.0): {'INSIDE' if same_max < 10.0 < diff_min else 'OUTSIDE'} window")
        else:
            print("  WARNING: SAME max >= DIFF min — no clean separation with this reference/crop.")
    print()


if __name__ == "__main__":
    run()
