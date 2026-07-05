# Pi Vision AI

An advanced, highly modular Edge AI computer vision platform designed for industrial and robotics applications (optimized for edge devices like the Raspberry Pi 4/5). Pi Vision AI provides a real-time, responsive single-page dashboard with dynamic camera controls and seamlessly hot-swappable AI vision modules.

## Features

- **Plug-and-Play Module Architecture:** Built around strict `IModule`, `IService`, `ISettingsProvider`, and `IPlugin` contracts. New capabilities can be registered seamlessly without modifying the core pipeline.
- **Hardware Agnostic AI Runtime:** The `AIRuntimeManager` uses abstract adapter patterns to support multiple frameworks (e.g., Ultralytics YOLO, EasyOCR) simultaneously, automatically loading metadata and weights.
- **Real-Time SPA Dashboard:** A sleek, dark-mode native dashboard offering live MJPEG streaming, sub-second telemetry polling, and instantaneous settings application.
- **Dynamic Camera Management:** Swap between built-in cameras or USB webcams on the fly and dynamically change resolutions (e.g., 640x480 to 1080p).
- **Core Modules Included:**
  - **Object Detection:** Powered by YOLOv11 for high-speed bounding box detection.
  - **Measurement:** Classical CV-based dimensional measurement featuring interactive reference calibration.
  - **OCR (Optical Character Recognition):** Integrated EasyOCR for robust text extraction and reading.

## Architecture Structure

- `app/api/v1/` - Versioned REST API endpoints for modules and system controls.
- `app/core/` - The beating heart of the platform. Contains the `Container`, `Pipeline`, `EventBus`, `RendererManager`, and hardware `adapters`.
- `app/modules/` - The self-contained vision plugins (Object Detection, Measurement, OCR).
- `app/services/` - Thin bridge services that wire the REST API layer to the Core Container with strict state-identity safety.
- `ai_models/` - Local registry for AI metadata, model files, and weights.
- `templates/` & `static/` - Frontend assets.

## Setup & Installation

1. **Install dependencies:**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Depending on your hardware, you may need to install specific versions of PyTorch/Torchvision for hardware acceleration or OpenCV headless).*

2. **Run the application:**
   Start the Flask backend (the camera will initialize automatically when requested by the dashboard).
   ```bash
   python main.py
   ```

3. **Access the Dashboard:**
   Open a modern web browser and navigate to:
   ```
   http://localhost:5000
   ```
   *(Or `http://<YOUR_DEVICE_IP>:5000` if accessing remotely over the network).*

## Development

To add a new module to the platform:
1. Create a new directory in `app/modules/<your-module>`.
2. Implement the core contracts: `IModule`, `ISettingsProvider`, and `IPlugin`.
3. Create an API blueprint in `app/api/v1/` and a corresponding bridge service in `app/services/`.
4. Register the plugin's REST blueprint in `app/api/__init__.py`.
5. Update `templates/index.html` to integrate your new module into the frontend UI.
