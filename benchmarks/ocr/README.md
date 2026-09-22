# OCR Internal Benchmark

Tests EasyOCR against your own photos, no camera or live setup required.
Also sweeps canvas_size against a fixed dataset to answer "does a smaller
canvas_size actually hurt accuracy on my real text" with real data.

## Setup

1. Put photos of real text in `probes/` (printed labels, part numbers,
   anything representative of what you'll actually read in production).
2. Edit `probes_manifest.json` — for each probe, say what text you expect
   to find and how strictly to check it.
3. Run: `python benchmarks/ocr/run_benchmark.py`

## probes_manifest.json format

```json
{
  "probe1.jpg": {"expected": "SERIAL123", "match_type": "contains"},
  "probe2.jpg": {"expected": "LOT-4471", "match_type": "exact"}
}
```

`match_type`: "contains" (default — expected text appears as a substring in
at least one recognized text block, case-insensitive) or "exact" (a
recognized block matches the expected text exactly, case-insensitive,
whitespace-trimmed). OCR rarely gets every character perfect, so "contains"
is the more realistic default — use "exact" only for short, clean text.

## canvas_size sweep

Edit CANVAS_SIZES at the top of run_benchmark.py to test different values
against your real dataset in one run. The summary table at the end shows
accuracy and average latency per size — this is how you find the actual
speed/accuracy line for YOUR text, instead of guessing.

Note: the first probe of the first sweep pass includes EasyOCR's real
cold-load time (many seconds) — this is expected and matches live behavior.
Everything after that reflects steady-state latency.
