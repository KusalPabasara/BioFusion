# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioFusion is a pneumonia detection clinical decision support system built for the BioFusion Hackathon 2026 (Team GMora). It uses a ResNet50 model fine-tuned on pediatric chest X-rays to classify images as NORMAL or PNEUMONIA, with Grad-CAM explainability overlays.

## Commands

### Run the Streamlit app locally
```bash
cd streamlit_app && streamlit run app.py
```

### Train the model
```bash
python train_model.py
```
Downloads the dataset via `kagglehub` (paultimothymooney/chest-xray-pneumonia). Outputs `pneumonia_resnet50_best.pth`.

### Install dependencies
```bash
pip install -r streamlit_app/requirements.txt
# For CPU-only PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Deploy to VPS
```bash
bash deploy.sh
```
Deploys to `/var/www/biofusion-pneumonia` as a systemd service on port 8502, with nginx reverse proxy.

## Architecture

The app is a multi-page Streamlit application rooted in `streamlit_app/`.

- **`app.py`** — Landing page with metrics and feature overview. Entry point for Streamlit.
- **`pages/1_Live_Prediction.py`** — Upload chest X-ray, run inference, display Grad-CAM overlay and confidence scores.
- **`pages/2_Model_Insights.py`** — Model performance metrics (confusion matrix, ROC curve, training history).
- **`pages/3_Dataset_Explorer.py`** — Dataset statistics and exploration.
- **`utils/model.py`** — Model loading (ResNet50 with modified FC layer for binary classification) and inference.
- **`utils/gradcam.py`** — Grad-CAM implementation using hooks on `model.layer4[-1]`.
- **`utils/preprocessing.py`** — Image preprocessing with ImageNet normalization (224x224).
- **`train_model.py`** (root) — Standalone training script. Freezes ResNet50 backbone, trains only the FC layer. Uses early stopping (patience=3).

### Key design details

- `load_model(weights_path)` falls back to raw ImageNet weights ("demo mode") when `weights_path` is None or the file is missing. **Note:** the Live Prediction page calls `load_model()` with no argument, so the app currently always runs in demo mode even if `pneumonia_resnet50_best.pth` is present — wire the path through `get_model()` to use trained weights.
- The training script (`train_model.py`) freezes the entire ResNet50 backbone and trains only the new 2-class FC layer (`model.fc.parameters()`), so only ~4K params are trainable.
- Preprocessing and training both use ImageNet normalization at 224×224; grayscale X-rays are converted to RGB before inference (`preprocessing.py`).
- Grad-CAM hooks `model.layer4[-1]`; `create_gradcam_visualization` sets `requires_grad=True` and runs a backward pass, so it cannot be used inside a `torch.no_grad()` block (unlike `predict`).
- Pages import utils via `sys.path.insert` to add the parent directory. Public API is re-exported from `utils/__init__.py`.
- Each page duplicates its own CSS and top navigation bar inline (not shared via a component) and calls `st.set_page_config` itself. The model is cached with `@st.cache_resource`.
- UI palette: Sapphire Blue (`#2563eb`) primary, Emerald (`#10b981`) success, Amber (`#f59e0b`) warning. **No red** — this is an explicit design decision.
- Streamlit config lives in `streamlit_app/.streamlit/config.toml`.

### Deployment specifics

`deploy.sh` clones `https://github.com/KusalPabasara/BioFusion.git`, installs CPU-only PyTorch, and runs Streamlit bound to `127.0.0.1:8502` (headless) behind an nginx proxy (`nginx.conf`). The systemd service runs as `www-data` with `WorkingDirectory` set to `streamlit_app/`.
