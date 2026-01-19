"""
Preprocessing utilities for chest X-ray images.
Handles image loading, resizing, normalization for ResNet50 inference.
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# ImageNet normalization values (used for pretrained ResNet50)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Image size for ResNet50
IMAGE_SIZE = 224


def get_transforms():
    """Get the preprocessing transforms for inference."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for model inference.
    
    Args:
        image: PIL Image (can be grayscale or RGB)
    
    Returns:
        Preprocessed tensor ready for model input [1, 3, 224, 224]
    """
    # Convert to RGB if grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    transform = get_transforms()
    tensor = transform(image)
    
    # Add batch dimension
    return tensor.unsqueeze(0)


def load_image(uploaded_file) -> Image.Image:
    """
    Load an image from Streamlit uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        PIL Image
    """
    return Image.open(uploaded_file)


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor [C, H, W]
    
    Returns:
        Denormalized numpy array [H, W, C] in range [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    tensor = tensor.clone()
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose to [H, W, C]
    return tensor.permute(1, 2, 0).numpy()
