"""
Internal benchmark for OCR — runs the real OCRController + EasyOCR against
your own photos, with no camera required. See README.md for setup.
"""

import sys
import json
import glob
import os
import time
from unittest.mock import MagicMock

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.core.container import Container
from app.modules.ocr.controller import OCRController
from app.modules.ocr.settings import OCRSettings

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROBES_DIR = os.path.join(BENCH_DIR, "probes")
MANIFEST_PATH = os.path.join(BENCH_DIR, "probes_manifest.json")

# Edit this list to test different values against your real dataset.
CANVAS_SIZES = [640, 960, 1280]
MIN_CONFIDENCE = 0.3  # held fixed across the sweep so canvas_size is the only variable


def load_probes():
    paths = sorted(
        glob.glob(os.path.join(PROBES_DIR, "*.jpg"))
        + glob.glob(os.path.join(PROBES_DIR, "*.jpeg"))
        + glob.glob(os.path.join(PROBES_DIR, "*.png"))
    )
    images = {}
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  WARNING: could not read {p}, skipping")
            continue
        images[os.path.basename(p)] = img
    return images


def text_matches(recognized_texts, expected, match_type):
    expected_norm = expected.strip().lower()
    for t in recognized_texts:
        found = t.get("text", "").strip().lower()
        if match_type == "exact":
            if found == expected_norm:
                return True
        else:  # "contains"
            if expected_norm in found:
                return True
    return False


def run():
    print("=" * 78)
    print("OCR Internal Benchmark")
    print("=" * 78)

    if not os.path.exists(MANIFEST_PATH):
        print(f"\nNo probes_manifest.json found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    probe_images = load_probes()
    if not probe_images:
        print(f"\nNo images found in {PROBES_DIR}")
        return

    print("\nLoading real AIRuntimeManager via Container (this loads EasyOCR "
          "on the first inference call below — expect a real cold-load delay "
          "on the very first probe)...")
    container = Container.get_instance()
    ai_runtime = container.ai_runtime

    event_bus = MagicMock()
    settings = OCRSettings()
    controller = OCRController(event_bus, ai_runtime, settings)

    summary_rows = []

    for canvas_size in CANVAS_SIZES:
        print("\n" + "-" * 78)
        print(f"canvas_size = {canvas_size}")
        print("-" * 78)
        print(f"{'Probe':<28} {'Expected':<20} {'Match':<7} {'Latency':<10} Recognized")

        controller.settings.update({"canvas_size": canvas_size, "min_confidence": MIN_CONFIDENCE})

        total = 0
        correct = 0
        total_latency = 0.0

        for probe_name, meta in manifest.items():
            if probe_name not in probe_images:
                print(f"{probe_name:<28} MISSING FILE in probes/ — skipping")
                continue

            expected = meta.get("expected", "")
            match_type = meta.get("match_type", "contains")
            img = probe_images[probe_name]

            result = controller.process({"frame": img})

            total += 1
            total_latency += result.latency_ms
            is_match = text_matches(result.texts, expected, match_type)
            if is_match:
                correct += 1

            recognized_summary = ", ".join(t.get("text", "") for t in result.texts) or "(none found)"
            mark = "PASS" if is_match else "FAIL"
            latency_str = f"{result.latency_ms/1000:.1f}s"

            print(f"{probe_name:<28} {expected:<20} {mark:<7} {latency_str:<10} {recognized_summary}")

        accuracy = (correct / total * 100) if total > 0 else 0
        avg_latency = (total_latency / total) if total > 0 else 0
        summary_rows.append((canvas_size, correct, total, accuracy, avg_latency))

    print("\n" + "=" * 78)
    print("SUMMARY — accuracy and latency by canvas_size")
    print("=" * 78)
    print(f"{'canvas_size':<14} {'Accuracy':<16} {'Avg latency':<14}")
    for canvas_size, correct, total, accuracy, avg_latency in summary_rows:
        print(f"{canvas_size:<14} {correct}/{total} ({accuracy:.1f}%){'':<3} {avg_latency/1000:.1f}s")

    print("\nUse this table to pick the smallest canvas_size that doesn't cost")
    print("you accuracy on your actual text — that's your real speed/accuracy line.")
    print("=" * 78)


if __name__ == "__main__":
    run()
