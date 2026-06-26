# Pi Vision AI

A modular Edge AI computer vision dashboard designed for the Raspberry Pi 4.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Access the dashboard:**
   Open a web browser and navigate to `http://localhost:5000` (or `http://<PI_IP>:5000`).

## Architecture
- `core/`: Contains core system components like camera management and module management.
- `modules/`: Contains custom AI vision modules that inherit from `BaseVisionModule`.
- `app.py`: The Flask application factory and entry point.
