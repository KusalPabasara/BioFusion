# BioFusion Kiosk

**Arduino-Integrated Hospital X-Ray Scanning Station** — A standalone kiosk system for AI-assisted pneumonia detection in hospitals.

Part of the BioFusion project (Team GMora — BioFusion Hackathon 2026).

## Overview

A hospital technician can:
1. Press **START SCAN** on the touchscreen
2. ESP32 turns on the NeoPixel LED strip for illumination
3. Live camera preview appears on screen
4. Press **CAPTURE** to take the X-ray photo
5. AI model (ResNet50) analyzes the image with Grad-CAM explainability
6. Results displayed with a **QR code** — scan with a phone to download the PDF report

## Quick Start

### 1. Install Dependencies

```bash
cd kiosk
pip install -r requirements.txt
```

### 2. (Optional) Set Up ESP32

See [`esp32/README.md`](esp32/README.md) for wiring and firmware upload instructions.

The kiosk works in **software-only mode** without the ESP32 connected.

### 3. Run the Kiosk

```bash
cd server
python app.py
```

Open your browser to: **http://localhost:5000**

### 4. Model Weights

Place `pneumonia_resnet50_best.pth` in `server/model/`. Without it, the app runs in demo mode using ImageNet weights.

## Architecture

```
kiosk/
├── esp32/                  # ESP32 firmware (NeoPixel + Status LED)
│   ├── kiosk_controller/
│   │   └── kiosk_controller.ino
│   └── README.md           # Wiring diagram
├── server/                 # Python Flask backend
│   ├── app.py              # Main application
│   ├── config.py           # Configuration
│   ├── camera.py           # OpenCV webcam controller
│   ├── inference.py        # ResNet50 + Grad-CAM (self-contained)
│   ├── serial_bridge.py    # ESP32 serial communication
│   ├── report.py           # PDF report + QR code generator
│   ├── templates/          # Jinja2 HTML templates
│   └── static/             # CSS, JS, images
├── captures/               # Saved X-ray captures
├── reports/                # Generated PDF reports
└── requirements.txt
```

## Tech Stack

| Component | Technology |
|---|---|
| Microcontroller | ESP32 with Adafruit NeoPixel |
| Backend | Python / Flask |
| Frontend | HTML + HTMX (no JS framework) |
| Camera | USB Webcam via OpenCV |
| AI Model | PyTorch ResNet50 (fine-tuned) |
| Reports | ReportLab PDF + QR code |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KIOSK_SERIAL_PORT` | `/dev/tty.usbserial-0001` | ESP32 serial port |
| `KIOSK_CAMERA_INDEX` | `0` | OpenCV camera index |
| `KIOSK_HOST` | `0.0.0.0` | Server bind address |
| `KIOSK_PORT` | `5000` | Server port |
| `KIOSK_DEBUG` | `false` | Flask debug mode |
