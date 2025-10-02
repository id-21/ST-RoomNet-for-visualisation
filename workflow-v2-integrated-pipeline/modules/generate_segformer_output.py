# File: generate_segformer_output.py
"""
Generate SegFormer confidence map from an image using the pretrained model
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../segformer-model'))

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor

def generate_segformer_confidence(image_path, output_path=None, wall_class_id=0):
    """
    Generate SegFormer wall confidence map from an image

    Args:
        image_path: Path to input image
        output_path: Path to save confidence map (.npy). If None, returns array
        wall_class_id: Class ID for walls in ADE20K (default: 0)

    Returns:
        Confidence map as numpy array (H, W) with values 0-1
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    print("Loading SegFormer model...")
    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512"
    ).to(device)

    # Load and prepare image
    print(f"Loading image from {image_path}...")
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Process image
    print("Running inference...")
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract wall confidence
    logits = outputs.logits.squeeze().cpu()  # shape: [num_classes, H, W]
    probs = torch.softmax(logits, dim=0)     # probabilities
    wall_probs = probs[wall_class_id]       # per-pixel probability for wall class
    confidence_map = wall_probs.numpy()      # H x W, floats 0–1

    # Resize confidence map to match original image
    print("Resizing confidence map to match original image...")
    confidence_resized = cv2.resize(
        confidence_map,
        (img_np.shape[1], img_np.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    # Save if output path provided
    if output_path:
        print(f"Saving confidence map to {output_path}...")
        np.save(output_path, confidence_resized)
        print(f"Confidence map shape: {confidence_resized.shape}")
        print(f"Confidence range: [{confidence_resized.min():.3f}, {confidence_resized.max():.3f}]")

    return confidence_resized

def visualize_confidence_map(confidence_map, output_path=None):
    """
    Create visualization of confidence map

    Args:
        confidence_map: Numpy array (H, W) with values 0-1
        output_path: Path to save visualization image

    Returns:
        Colored confidence map as numpy array
    """
    # Convert to 0-255 range
    confidence_uint8 = (confidence_map * 255).astype(np.uint8)

    # Apply JET colormap
    confidence_colored = cv2.applyColorMap(confidence_uint8, cv2.COLORMAP_JET)

    if output_path:
        cv2.imwrite(output_path, confidence_colored)
        print(f"Visualization saved to {output_path}")

    return confidence_colored

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SegFormer confidence map")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output", "-o", help="Output path for confidence map (.npy)")
    parser.add_argument("--visualize", "-v", help="Save visualization image")
    parser.add_argument("--resize", type=int, nargs=2, help="Resize to (width, height)")

    args = parser.parse_args()

    # Generate confidence map
    confidence = generate_segformer_confidence(args.image_path, args.output)

    # Resize if requested
    if args.resize:
        width, height = args.resize
        confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
        print(f"Resized to {width}x{height}")

        # Update output if specified
        if args.output:
            np.save(args.output, confidence)

    # Create visualization if requested
    if args.visualize:
        visualize_confidence_map(confidence, args.visualize)

    print("\nDone!")
    print(f"Final confidence map shape: {confidence.shape}")
    print(f"Confidence statistics:")
    print(f"  Mean: {confidence.mean():.3f}")
    print(f"  Std: {confidence.std():.3f}")
    print(f"  Min: {confidence.min():.3f}")
    print(f"  Max: {confidence.max():.3f}")
