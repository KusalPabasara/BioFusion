"""
Model utilities for Pneumonia Detection.
Handles model loading, inference, and predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path

# Class labels
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


def get_device():
    """Get the best available device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(weights_path: str = None):
    """
    Load the ResNet50 model for pneumonia detection.
    
    Args:
        weights_path: Path to saved model weights (.pth file).
                     If None, uses ImageNet pretrained weights.
    
    Returns:
        model: Loaded PyTorch model in eval mode
        device: Device the model is on
    """
    device = get_device()
    
    # Load ResNet50 with ImageNet weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Modify the final fully connected layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Load custom weights if provided
    if weights_path and Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded custom weights from {weights_path}")
    else:
        print("Using ImageNet pretrained weights (demo mode)")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    return model, device


def predict(model, image_tensor: torch.Tensor, device: torch.device):
    """
    Make a prediction on a preprocessed image tensor.
    
    Args:
        model: The loaded PyTorch model
        image_tensor: Preprocessed image tensor [1, 3, 224, 224]
        device: Device to run inference on
    
    Returns:
        predicted_class: Index of predicted class (0=Normal, 1=Pneumonia)
        confidence: Confidence score (0-1)
        probabilities: Full probability distribution [P(Normal), P(Pneumonia)]
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


def get_prediction_label(class_index: int) -> str:
    """Get the class label for a prediction index."""
    return CLASS_NAMES[class_index]


def get_model_info():
    """Get information about the model architecture."""
    return {
        "architecture": "ResNet50",
        "input_size": "224 x 224 x 3",
        "classes": CLASS_NAMES,
        "total_params": "~23.5M",
        "trainable_params": "4,098 (0.02%)",
        "pretrained": "ImageNet"
    }
