"""
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization.
Generates heatmaps showing which regions of the image influenced the prediction.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for ResNet50.
    Visualizes which parts of the chest X-ray the model focuses on.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model (ResNet50)
            target_layer: Layer to extract gradients from. 
                         Defaults to last conv layer (layer4)
        """
        self.model = model
        self.target_layer = target_layer or model.layer4[-1]
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed image tensor [1, 3, 224, 224]
            target_class: Class index to generate CAM for. 
                         If None, uses predicted class.
        
        Returns:
            heatmap: Numpy array [H, W] with values in [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_heatmap(self, 
                        original_image: Image.Image, 
                        heatmap: np.ndarray, 
                        alpha: float = 0.5,
                        colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            original_image: PIL Image
            heatmap: Grad-CAM heatmap [H, W]
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap to use
        
        Returns:
            overlaid: RGB numpy array with heatmap overlay
        """
        # Resize heatmap to match image size
        img_array = np.array(original_image.convert('RGB'))
        h, w = img_array.shape[:2]
        
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlaid = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlaid


def create_gradcam_visualization(model, 
                                  image_tensor: torch.Tensor, 
                                  original_image: Image.Image,
                                  device: torch.device,
                                  target_class: int = None) -> tuple:
    """
    Create Grad-CAM visualization for a prediction.
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed tensor [1, 3, 224, 224]
        original_image: Original PIL Image
        device: Device
        target_class: Target class for CAM (None = use prediction)
    
    Returns:
        heatmap: Raw heatmap array
        overlay: Image with heatmap overlay
    """
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad = True
    
    gradcam = GradCAM(model)
    heatmap = gradcam.generate(image_tensor, target_class)
    overlay = gradcam.overlay_heatmap(original_image, heatmap)
    
    return heatmap, overlay
