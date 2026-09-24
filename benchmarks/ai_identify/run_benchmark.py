"""
Internal benchmark for AI Identify — runs the real controller logic
(ORB matching + homography + SSIM) against a folder of your own real
photos, with no camera or live setup required. See README.md for setup.
"""

import sys
import json
import glob
import os
from unittest.mock import MagicMock

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.modules.ai_identify.controller import AIIdentifyController
from app.modules.ai_identify.settings import AIIdentifySettings

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
GOOD_DIR = os.path.join(BENCH_DIR, "good")
BAD_DIR = os.path.join(BENCH_DIR, "bad")
PROBES_DIR = os.path.join(BENCH_DIR, "probes")
MANIFEST_PATH = os.path.join(BENCH_DIR, "probes_manifest.json")
REFERENCES_MANIFEST_PATH = os.path.join(BENCH_DIR, "references_manifest.json")

SETTINGS_OVERRIDE = {
    # "classification_margin": 0.03,
    # "match_ratio_threshold": 0.8,
    # "min_match_count": 8,
}


def load_images(folder):
    paths = sorted(
        glob.glob(os.path.join(folder, "*.jpg"))
        + glob.glob(os.path.join(folder, "*.jpeg"))
        + glob.glob(os.path.join(folder, "*.png"))
    )
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  WARNING: could not read {p}, skipping")
            continue
        images.append((os.path.basename(p), img))
    return images


def build_controller():
    event_bus = MagicMock()
    settings = AIIdentifySettings()
    if SETTINGS_OVERRIDE:
        settings.update(SETTINGS_OVERRIDE)
    return AIIdentifyController(event_bus, settings)


def load_references_manifest():
    if not os.path.exists(REFERENCES_MANIFEST_PATH):
        return {"good": {}, "bad": {}}
    with open(REFERENCES_MANIFEST_PATH) as f:
        data = json.load(f)
    return {"good": data.get("good", {}), "bad": data.get("bad", {})}


def teach_from_folder(controller, folder, teach_fn, label, crop_overrides):
    images = load_images(folder)
    if not images:
        print(f"  No images found in {folder}")
        return 0
    taught = 0
    for name, img in images:
        controller.last_frame = img
        h, w = img.shape[:2]
        box = crop_overrides.get(name, {})
        if box and "x" in box:
            x, y, cw, ch = box["x"], box["y"], box["w"], box["h"]
            crop_note = f" (cropped {cw}x{ch} @ {x},{y})"
        else:
            x, y, cw, ch = 0, 0, w, h
            crop_note = " (full frame — no crop defined in references_manifest.json)"
        ok = teach_fn(x, y, cw, ch)
        status = "OK" if ok else "REJECTED (too few features or gallery full)"
        print(f"  [{label}] {name}: {status}{crop_note}")
        if ok:
            taught += 1
    return taught


def run():
    print("=" * 70)
    print("AI Identify Internal Benchmark")
    print("=" * 70)

    if not os.path.exists(MANIFEST_PATH):
        print(f"\nNo probes_manifest.json found at {MANIFEST_PATH}")
        return

    controller = build_controller()

    references = load_references_manifest()

    print("\nTeaching GOOD references:")
    good_count = teach_from_folder(controller, GOOD_DIR, controller.teach_good, "GOOD", references["good"])

    print("\nTeaching BAD references:")
    bad_count = teach_from_folder(controller, BAD_DIR, controller.teach_bad, "BAD", references["bad"])

    print(f"\nTaught {good_count} good / {bad_count} bad reference(s). "
          f"Teach status: {controller.teach_status}")

    if controller.teach_status != "Taught":
        print("\nNeed at least 1 image in both good/ and bad/ to run probes. Stopping.")
        return

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    probe_images = {name: img for name, img in load_images(PROBES_DIR)}

    print("\n" + "-" * 70)
    print(f"{'Probe':<28} {'Expected':<10} {'Actual':<10} {'Result':<8} {'Conf':<7} {'GoodSim':<8} {'BadSim':<8}")
    print("-" * 70)

    total = 0
    correct = 0
    failures = []

    for probe_name, meta in manifest.items():
        expected = meta.get("expected")
        if probe_name not in probe_images:
            print(f"{probe_name:<28} MISSING FILE in probes/ — skipping")
            continue

        img = probe_images[probe_name]
        result = controller.process({"frame": img})
        actual = result.classification if result.located else "NOT LOCATED"

        total += 1
        is_correct = (actual == expected)
        if is_correct:
            correct += 1
        else:
            failures.append((probe_name, expected, actual, meta.get("note", "")))

        good_sim = f"{result.good_similarity:.3f}" if result.good_similarity is not None else "—"
        bad_sim = f"{result.bad_similarity:.3f}" if result.bad_similarity is not None else "—"
        conf = f"{result.match_confidence:.3f}" if result.match_confidence else "—"
        mark = "PASS" if is_correct else "FAIL"

        print(f"{probe_name:<28} {str(expected):<10} {str(actual):<10} {mark:<8} {conf:<7} {good_sim:<8} {bad_sim:<8}")

    print("-" * 70)
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")

    if failures:
        print("\nFailures:")
        for name, expected, actual, note in failures:
            note_str = f" — {note}" if note else ""
            print(f"  {name}: expected {expected}, got {actual}{note_str}")

    print("\n" + "=" * 70)
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    run()
