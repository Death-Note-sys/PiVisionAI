# Pi Vision AI

A modular, browser-based computer vision platform built on **Flask + OpenCV**. Attach a USB or IP camera, open the web UI, and classify parts, read barcodes, measure dimensions, run OCR, detect faces, and more — all without writing code.

> **Note:** Despite the name, this project is a general-purpose desktop computer vision workbench. It is **not** optimised for Raspberry Pi or any ARM/embedded hardware. It runs on any Windows or Linux machine with a compatible webcam and Python 3.10+.

---

## Features

| Module | What it does |
|---|---|
| **Object Detection** | YOLO11n real-time object detection with per-class filtering and confidence thresholds |
| **AI Identify** | ORB + homography + SSIM template-matching for Good/Bad/Uncertain part classification — teach from live camera without a custom model |
| **Measurement** | Pixel-to-real-world calibration with on-screen dimension overlay |
| **OCR** | EasyOCR text extraction with configurable confidence floor and inference resolution (`canvas_size`) |
| **Face Detection** | MediaPipe-based face landmark detection |
| **Barcode / QR** | pyzbar scan with live decode readout |
| **Color Detector** | Dominant color extraction and classification |
| **Edge Detection** | Canny edge overlay |
| **Shape Detection** | Contour-based geometric shape recognition |
| **Motion Detection** | Frame-diff motion alerting |
| **Background Removal** | Real-time background segmentation |

All modules share a single live video stream and are hot-swappable at runtime through the UI.

---

## Architecture

```
Pi Vision AI
├── app/
│   ├── api/v1/             # Flask REST API blueprints (one per module)
│   ├── core/
│   │   ├── adapters/       # AI framework adapters (YOLO, EasyOCR, ONNX, …)
│   │   ├── ai_runtime.py   # Model lifecycle manager
│   │   ├── container.py    # Dependency injection container
│   │   ├── event_bus.py    # Pub/sub event bus
│   │   └── contracts.py    # IModule / IService / ISettingsProvider interfaces
│   ├── modules/            # One sub-package per vision module
│   │   ├── ai_identify/    # ORB + SSIM part classification
│   │   ├── ocr/            # EasyOCR integration
│   │   ├── object_detection/
│   │   ├── measurement/
│   │   └── …
│   └── services/           # Cross-cutting services (trigger, system, …)
├── templates/index.html    # Single-page web UI
├── benchmarks/
│   └── ai_identify/        # Offline benchmark tool (see below)
├── tests/                  # pytest unit tests
├── cli.py                  # Entry point (click CLI)
├── config.py               # Environment configuration
└── requirements.txt
```

The design follows a clean **IModule → Controller → Service → Adapter** layering. Each module has its own settings model (Pydantic), REST blueprint, and controller — no module reaches into another.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Death-Note-sys/PiVisionAI.git
cd PiVisionAI
pip install -r requirements.txt
```

> **OpenCV note:** `requirements.txt` pulls `opencv-python-headless`. If you need the `select_roi.py` benchmark helper (which opens a GUI window), swap it for `opencv-python` in your environment:
> ```bash
> pip uninstall opencv-python-headless
> pip install opencv-python
> ```

### 2. Run the server

```bash
python cli.py run
# or with custom host/port:
python cli.py run --host 0.0.0.0 --port 5000
```

Open **http://127.0.0.1:5000** in your browser.

### 3. Select a module and a camera

Use the toolbar dropdown to pick a module, then select your camera index. The stream starts automatically.

---

## CLI Reference

```bash
python cli.py --help

Commands:
  run      Start the REST API server
  doctor   Run system diagnostics (camera, RAM, CPU)
  models   List discovered AI model files
```

---

## REST API

All routes are versioned under `/api/v1/`. Every active module exposes a standard set of endpoints:

### General

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/system/status` | Server health, camera, performance stats |
| `GET` | `/api/v1/modules` | List all registered modules |
| `POST` | `/api/v1/modules/<id>/activate` | Activate a module |
| `GET` | `/api/v1/camera/stream` | MJPEG video stream |

### Object Detection (`/api/v1/object-detection`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Current detections + telemetry |
| `PATCH` | `/settings` | Update confidence, class filter, overlay options |

### AI Identify (`/api/v1/ai-identify`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Classification result + Good/Bad gallery counts |
| `POST` | `/teach-good` | Teach a Good reference from the current frozen frame (`x, y, w, h`) |
| `POST` | `/teach-bad` | Teach a Bad reference |
| `DELETE` | `/good-reference/<index>` | Remove a Good reference by index |
| `DELETE` | `/bad-reference/<index>` | Remove a Bad reference by index |
| `POST` | `/reset-teaching` | Clear all references |
| `PATCH` | `/settings` | Update `classification_margin`, `min_confident_similarity`, `show_bbox`, etc. |

### OCR (`/api/v1/ocr`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Detected texts, confidence, latency, model name |
| `PATCH` | `/settings` | Update `min_confidence` (0.05–0.95), `canvas_size` (320–2560), overlay toggles |

### Measurement (`/api/v1/measurement`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Measured dimensions |
| `POST` | `/calibrate` | Set pixel-to-mm ratio |

### Trigger (`/api/v1/trigger`)

| Method | Endpoint | Description |
|---|---|---|
| `PATCH` | `/mode` | Set trigger mode: `continuous`, `single`, `interval` |
| `POST` | `/fire` | Fire a single inference manually |

---

## Module Configuration

All settings can be changed at runtime through the REST API or the inspector panel in the web UI. Example — lower OCR canvas size for faster inference on a slow CPU:

```bash
curl -X PATCH http://127.0.0.1:5000/api/v1/ocr/settings \
  -H "Content-Type: application/json" \
  -d '{"canvas_size": 640, "min_confidence": 0.4}'
```

### AI Identify: key settings

| Setting | Default | Description |
|---|---|---|
| `classification_margin` | `0.05` | SSIM delta below which both-match results → Uncertain |
| `min_confident_similarity` | `0.4` | SSIM floor for single-match results — scores below this → Uncertain instead of Good/Bad |
| `min_match_count` | `10` | Minimum ORB inliers required for a valid geometric match |
| `match_ratio_threshold` | `0.75` | Lowe's ratio test threshold |

### OCR: key settings

| Setting | Default | Description |
|---|---|---|
| `min_confidence` | `0.3` | Discard text detections below this confidence |
| `canvas_size` | `960` | EasyOCR internal detection resolution. Lower = faster but less accurate. Range: 320–2560 |

> **OCR performance note:** EasyOCR on CPU takes 10–25 s per frame at default resolution. Use **Single** or **Interval** trigger mode, not Continuous. Reduce `canvas_size` to `640` or `480` for meaningful speedup at some accuracy cost.

---

## AI Identify: Teaching Workflow

AI Identify uses no pretrained model for classification — you teach it directly from the camera:

1. Activate the **AI Identify** module.
2. Click **Add Good Image** → freeze frame → drag a tight box around the distinguishing feature → Confirm.
3. Click **Add Bad Image** → repeat for the Bad class.
4. The system classifies every subsequent frame as **Good**, **Bad**, or **Uncertain** in real time.

**Tips for reliable classification:**
- Draw boxes tightly around the actual distinguishing feature, not the whole part. Background included in the reference dilutes the SSIM signal.
- Teach 2–3 angles per class for better robustness.
- Keep reference and probe lighting consistent — the SSIM metric is sensitive to brightness and contrast changes.
- The gallery holds up to **20 references per class**. Delete underperforming ones with the ✕ button on each thumbnail.

---

## Offline Benchmark Tool

`benchmarks/ai_identify/` is a standalone script for validating the AI Identify pipeline against real photos, with no camera required.

```
benchmarks/ai_identify/
├── good/                   # Reference images for Good class
├── bad/                    # Reference images for Bad class
├── probes/                 # Test images to classify
├── references_manifest.json  # Optional per-image crop overrides {filename: {x,y,w,h}}
├── probes_manifest.json    # Ground truth {filename: {expected: "Good"|"Bad"|"Uncertain"|"NOT LOCATED", …}}
├── run_benchmark.py        # Main runner
└── select_roi.py           # GUI tool to select crop boxes → prints JSON for manifest
```

### Run

```bash
python benchmarks/ai_identify/run_benchmark.py
```

Non-zero exit code on any mismatch — suitable for local pre-commit checks.

### Override settings for tuning

```bash
python benchmarks/ai_identify/run_benchmark.py \
  --settings '{"min_confident_similarity": 0.35, "classification_margin": 0.08}'
```

### Select a reference crop interactively

```bash
python benchmarks/ai_identify/select_roi.py good/my_part.jpg
# Drag a box → press Enter → copy the printed JSON into references_manifest.json
```

---

## Running Tests

```bash
pytest tests/ -v
```

Key test files:

| File | Coverage |
|---|---|
| `tests/test_ai_identify_controller.py` | Controller unit tests: teach, classify, gallery ops, similarity floor |
| `tests/test_ai_identify_api.py` | REST API integration tests |

---

## Project Status

This is an active personal project under continuous development. Modules vary in maturity:

| Module | Status |
|---|---|
| Object Detection | ✅ Stable |
| AI Identify | ✅ Stable |
| Measurement | ✅ Stable |
| OCR | ✅ Working (CPU slow — use non-continuous trigger) |
| Face Detection | ⚙️ Working |
| Barcode / QR | ⚙️ Working |
| Color Detector | ⚙️ Working |
| Edge / Shape / Motion | 🔧 Beta |
| Background Removal | 🔧 Beta |

---

## Dependencies

| Package | Purpose |
|---|---|
| Flask | REST API server |
| OpenCV | Camera capture, image processing |
| Ultralytics (YOLO) | Object detection inference |
| EasyOCR | Text recognition |
| MediaPipe | Face detection landmarks |
| pyzbar | Barcode / QR decoding |
| ONNX Runtime | Generic ONNX model inference |
| Pydantic v2 | Settings validation |
| Flask-CORS | Cross-origin support |
| Flask-Limiter | Rate limiting |

---

## License

MIT
