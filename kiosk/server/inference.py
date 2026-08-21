"""
BioFusion Kiosk — Inference Engine
Self-contained module for pneumonia detection using ResNet50.
Extracted from the main BioFusion project — no cross-imports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


# ─── Model Loading ──────────────────────────────────────────────────────────

def get_device():
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(weights_path=None):
    """
    Load the ResNet50 model for pneumonia detection.
    Falls back to ImageNet weights (demo mode) if weights file is missing.

    Returns:
        model: PyTorch model in eval mode
        device: Device the model is on
    """
    device = get_device()

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    if weights_path and __import__('os').path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded trained weights from {weights_path}")
    else:
        logger.warning("Using ImageNet pretrained weights (demo mode)")

    model = model.to(device)
    model.eval()
    return model, device


# ─── Preprocessing ───────────────────────────────────────────────────────────

def get_transforms():
    """Get preprocessing transforms for inference."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for model inference.
    Converts grayscale to RGB, resizes, normalizes.

    Returns:
        Tensor [1, 3, 224, 224]
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = get_transforms()
    tensor = transform(image)
    return tensor.unsqueeze(0)


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict(model, image_tensor, device):
    """
    Run inference on a preprocessed image tensor.

    Returns:
        predicted_class: 0 (Normal) or 1 (Pneumonia)
        confidence: float 0-1
        probabilities: numpy array [P(Normal), P(Pneumonia)]
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    return (
        predicted_class.item(),
        confidence.item(),
        probabilities[0].cpu().numpy()
    )


def get_prediction_label(class_index):
    """Get the class label string for a prediction index."""
    return CLASS_NAMES[class_index]


# ─── Grad-CAM ───────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM visualization for ResNet50."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or model.layer4[-1]
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap. Returns numpy array [H, W] in [0, 1]."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay_heatmap(self, original_image, heatmap, alpha=0.5):
        """Overlay heatmap on original image. Returns RGB numpy array."""
        img_array = np.array(original_image.convert('RGB'))
        h, w = img_array.shape[:2]

        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlaid = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
        return overlaid


def generate_gradcam(model, image_tensor, original_image, device, target_class=None):
    """
    Generate Grad-CAM visualization.

    NOTE: requires_grad must be True — cannot be used inside torch.no_grad().

    Returns:
        heatmap: raw heatmap array
        overlay: RGB numpy array with heatmap overlay
    """
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad = True

    gradcam = GradCAM(model)
    heatmap = gradcam.generate(image_tensor, target_class)
    overlay = gradcam.overlay_heatmap(original_image, heatmap)

    return heatmap, overlay


# ─── Convenience ─────────────────────────────────────────────────────────────

def analyze_image(model, device, pil_image):
    """
    Full analysis pipeline: preprocess → predict → gradcam.

    Args:
        model: loaded PyTorch model
        device: torch device
        pil_image: PIL Image of the X-ray

    Returns:
        dict with keys: class_index, class_name, confidence,
                       probabilities, heatmap, overlay
    """
    tensor = preprocess_image(pil_image)

    class_index, confidence, probabilities = predict(model, tensor, device)

    heatmap, overlay = generate_gradcam(model, tensor, pil_image, device, class_index)

    return {
        "class_index": class_index,
        "class_name": get_prediction_label(class_index),
        "confidence": confidence,
        "probabilities": {
            "normal": float(probabilities[0]),
            "pneumonia": float(probabilities[1])
        },
        "heatmap": heatmap,
        "overlay": overlay
    }
