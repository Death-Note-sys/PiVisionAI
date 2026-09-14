# AI Identify Internal Benchmark

Tests the real ORB + homography + SSIM classification pipeline against your
own photos, with no camera or live setup required.

## Setup

1. Put cropped photos of your GOOD feature in `good/` (one or more — this
   is exactly the multi-image gallery, taught from files instead of a live
   freeze). Crop tightly to the actual distinguishing feature, same lesson
   learned from live testing — whole-object framing dilutes the SSIM signal.
2. Put cropped photos of your BAD feature in `bad/`.
3. Put test photos to classify in `probes/`.
4. Edit `probes_manifest.json` to say what each probe SHOULD classify as.
5. Run: `python benchmarks/ai_identify/run_benchmark.py`

## probes_manifest.json format

```json
{
  "probe1.jpg": {"expected": "Good", "note": "same angle as taught"},
  "probe2.jpg": {"expected": "Bad", "note": "different lighting"},
  "probe3.jpg": {"expected": "Good", "note": "45 degree rotation, untaught angle"}
}
```

## Tuning experiments

Edit SETTINGS_OVERRIDE at the top of run_benchmark.py to test different
classification_margin / match_ratio_threshold / min_match_count values
against the same fixed dataset, without touching application code — this
is the fast way to answer "did raising the margin help or hurt?" instead
of re-teaching live each time.
