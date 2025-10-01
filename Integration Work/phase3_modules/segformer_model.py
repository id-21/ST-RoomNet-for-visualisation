# File: segformer_model.py
"""
SegFormer model loading and inference wrapper.

Provides a clean interface for loading the SegFormer model and running inference
to get wall confidence maps.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
from typing import Tuple, Optional


# Global model cache to avoid reloading
_cached_model = None
_cached_processor = None


def load_segformer_model(model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
                        use_cache: bool = True) -> Tuple:
    """
    Load SegFormer model and processor.

    Args:
        model_name: HuggingFace model identifier
        use_cache: If True, cache and reuse model (default: True)

    Returns:
        Tuple of (model, processor, device)
    """
    global _cached_model, _cached_processor

    # Return cached model if available
    if use_cache and _cached_model is not None and _cached_processor is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _cached_model, _cached_processor, device

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load processor
    processor = AutoImageProcessor.from_pretrained(model_name)

    # Load model
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    model.eval()  # Set to evaluation mode

    # Cache
    if use_cache:
        _cached_model = model
        _cached_processor = processor

    return model, processor, device


def run_segformer_inference(model,
                            processor,
                            device,
                            image: Image.Image,
                            wall_class_id: int = 0) -> np.ndarray:
    """
    Run SegFormer inference to get wall confidence map.

    Args:
        model: Loaded SegFormer model
        processor: Loaded image processor
        device: Torch device (cpu or cuda)
        image: PIL Image
        wall_class_id: Class ID for walls in ADE20K (default: 0)

    Returns:
        Confidence map as numpy array (H, W) with values 0-1
    """
    # Get original image size
    original_size = (image.width, image.height)

    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract wall confidence
    logits = outputs.logits.squeeze().cpu()  # shape: [num_classes, H, W]
    probs = torch.softmax(logits, dim=0)     # probabilities
    wall_probs = probs[wall_class_id]        # per-pixel probability for wall class
    confidence_map = wall_probs.numpy()      # H x W, floats 0-1

    # Resize confidence map to match original image
    confidence_resized = cv2.resize(
        confidence_map,
        original_size,
        interpolation=cv2.INTER_LINEAR
    )

    return confidence_resized


def inference_from_path(image_path: str,
                       model = None,
                       processor = None,
                       device = None,
                       wall_class_id: int = 0,
                       model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512") -> np.ndarray:
    """
    Complete inference pipeline from image path.

    Args:
        image_path: Path to room image
        model: Optional pre-loaded model
        processor: Optional pre-loaded processor
        device: Optional device
        wall_class_id: Class ID for walls (default: 0)
        model_name: Model identifier (used if model is None)

    Returns:
        Confidence map as numpy array (H, W) with values 0-1
    """
    # Load model if not provided
    if model is None or processor is None or device is None:
        model, processor, device = load_segformer_model(model_name)

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Run inference
    confidence_map = run_segformer_inference(model, processor, device, image, wall_class_id)

    return confidence_map


def inference_from_array(image_array: np.ndarray,
                        model = None,
                        processor = None,
                        device = None,
                        wall_class_id: int = 0,
                        model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512") -> np.ndarray:
    """
    Complete inference pipeline from numpy array.

    Args:
        image_array: Numpy array image (H, W, 3) - RGB or BGR
        model: Optional pre-loaded model
        processor: Optional pre-loaded processor
        device: Optional device
        wall_class_id: Class ID for walls (default: 0)
        model_name: Model identifier (used if model is None)

    Returns:
        Confidence map as numpy array (H, W) with values 0-1
    """
    # Load model if not provided
    if model is None or processor is None or device is None:
        model, processor, device = load_segformer_model(model_name)

    # Convert numpy array to PIL Image
    # Assume RGB if 3 channels
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Check if BGR (OpenCV format) by simple heuristic - if from cv2.imread
        # For safety, we'll assume RGB unless specified
        image = Image.fromarray(image_array.astype(np.uint8), mode='RGB')
    else:
        raise ValueError("Image array must have shape (H, W, 3)")

    # Run inference
    confidence_map = run_segformer_inference(model, processor, device, image, wall_class_id)

    return confidence_map


def inference_from_pil(image: Image.Image,
                      model = None,
                      processor = None,
                      device = None,
                      wall_class_id: int = 0,
                      model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512") -> np.ndarray:
    """
    Complete inference pipeline from PIL Image.

    Args:
        image: PIL Image (RGB)
        model: Optional pre-loaded model
        processor: Optional pre-loaded processor
        device: Optional device
        wall_class_id: Class ID for walls (default: 0)
        model_name: Model identifier (used if model is None)

    Returns:
        Confidence map as numpy array (H, W) with values 0-1
    """
    # Load model if not provided
    if model is None or processor is None or device is None:
        model, processor, device = load_segformer_model(model_name)

    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Run inference
    confidence_map = run_segformer_inference(model, processor, device, image, wall_class_id)

    return confidence_map


def visualize_confidence_map(confidence_map: np.ndarray,
                             colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Create colored visualization of confidence map.

    Args:
        confidence_map: Numpy array (H, W) with values 0-1
        colormap: OpenCV colormap constant (default: COLORMAP_JET)

    Returns:
        Colored confidence map as BGR numpy array (H, W, 3)
    """
    # Convert to 0-255 range
    confidence_uint8 = (confidence_map * 255).astype(np.uint8)

    # Apply colormap
    confidence_colored = cv2.applyColorMap(confidence_uint8, colormap)

    return confidence_colored


def create_binary_mask(confidence_map: np.ndarray,
                      threshold: float = 0.4) -> np.ndarray:
    """
    Create binary wall mask from confidence map.

    Args:
        confidence_map: Numpy array (H, W) with values 0-1
        threshold: Confidence threshold (default: 0.4)

    Returns:
        Binary mask as boolean numpy array (H, W)
    """
    return confidence_map > threshold


def get_confidence_stats(confidence_map: np.ndarray) -> dict:
    """
    Get statistics about the confidence map.

    Args:
        confidence_map: Numpy array (H, W) with values 0-1

    Returns:
        Dictionary with statistics
    """
    return {
        'shape': confidence_map.shape,
        'mean': float(confidence_map.mean()),
        'std': float(confidence_map.std()),
        'min': float(confidence_map.min()),
        'max': float(confidence_map.max()),
        'median': float(np.median(confidence_map))
    }


# Example usage
if __name__ == "__main__":
    import os

    print("Loading SegFormer model...")
    model, processor, device = load_segformer_model()
    print(f"Model loaded successfully on {device}!")

    # Test inference on room1.jpeg
    test_image_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'images',
        'room1.jpeg'
    )

    if os.path.exists(test_image_path):
        print(f"\nRunning inference on {test_image_path}...")
        confidence = inference_from_path(test_image_path, model, processor, device)

        print("\nResults:")
        stats = get_confidence_stats(confidence)
        print(f"  Confidence map shape: {stats['shape']}")
        print(f"  Mean confidence: {stats['mean']:.3f}")
        print(f"  Std confidence: {stats['std']:.3f}")
        print(f"  Min confidence: {stats['min']:.3f}")
        print(f"  Max confidence: {stats['max']:.3f}")

        # Test binary mask creation
        binary_mask = create_binary_mask(confidence, threshold=0.4)
        print(f"\n  Binary mask (threshold=0.4):")
        print(f"    Wall pixels: {np.sum(binary_mask)}")
        print(f"    Total pixels: {binary_mask.size}")
        print(f"    Wall coverage: {100 * np.sum(binary_mask) / binary_mask.size:.1f}%")
    else:
        print(f"\nTest image not found: {test_image_path}")
