# File: st_roomnet_model.py
"""
ST-RoomNet model loading and inference wrapper.

Provides a clean interface for loading the ST-RoomNet model and running inference
to get both segmentation and theta transformation parameters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.convnext import ConvNeXtTiny, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

from spatial_transformer_fixed import ProjectiveTransformerLayer

ABS_PATH_TO_WEIGHTS = '/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/Weight_ST_RroomNet_ConvNext.h5'
ABS_PATH_TO_IMG = '/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/Integration Work/images/ref_img2.png'

# Global model cache to avoid reloading
_cached_model = None
_cached_ref_img = None


def load_reference_image(ref_img_path: str = ABS_PATH_TO_IMG) -> tf.Tensor:
    """
    Load the reference cuboid image for spatial transformation.

    Args:
        ref_img_path: Path to reference image (default: 'ref_img2.png' in repo root)

    Returns:
        Reference image tensor with shape (1, 400, 400, 1)
    """
    # If path is relative, make it relative to repo root
    if not os.path.isabs(ref_img_path):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ref_img_path = os.path.join(repo_root, ref_img_path)

    if not os.path.exists(ref_img_path):
        raise FileNotFoundError(f"Reference image not found: {ref_img_path}")

    ref_img = tf.io.read_file(ref_img_path)
    ref_img = tf.io.decode_png(ref_img)
    ref_img = tf.cast(ref_img, tf.float32) / 51.0
    ref_img = ref_img[tf.newaxis, ...]

    return ref_img


def load_st_roomnet_model(weights_path: str = ABS_PATH_TO_WEIGHTS,
                          ref_img_path: str = ABS_PATH_TO_IMG,
                          use_cache: bool = True) -> Model:
    """
    Load ST-RoomNet model with dual output (segmentation + theta).

    Args:
        weights_path: Path to model weights file
        ref_img_path: Path to reference cuboid image
        use_cache: If True, cache and reuse model (default: True)

    Returns:
        Keras Model with dual output: [segmentation, theta]
    """
    global _cached_model, _cached_ref_img

    # Return cached model if available
    if use_cache and _cached_model is not None:
        return _cached_model

    # Make paths absolute if relative
    if not os.path.isabs(weights_path):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        weights_path = os.path.join(repo_root, weights_path)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    # Load reference image
    ref_img = load_reference_image(ref_img_path)
    _cached_ref_img = ref_img

    # Build model architecture
    base_model = ConvNeXtTiny(
        include_top=False,
        weights="imagenet",
        input_shape=(400, 400, 3),
        pooling='avg'
    )

    # Theta parameters output
    theta = Dense(8)(base_model.output)

    # Spatial transformation layer
    stl = ProjectiveTransformerLayer(ref_img, (400, 400))(theta)

    # First, create single-output model (matches trained weights structure)
    single_output_model = Model(base_model.input, stl)

    # Load trained weights
    single_output_model.load_weights(weights_path)

    # Now create dual-output model by extracting theta from intermediate layer
    # Find the Dense(8) layer - it should be named 'dense' or 'dense_1'
    theta_layer = None
    for layer in single_output_model.layers:
        if isinstance(layer, Dense):
            # Check output shape - should be (None, 8)
            output_shape = layer.output.shape
            if output_shape[-1] == 8:
                theta_layer = layer
                break

    if theta_layer is None:
        raise RuntimeError("Could not find Dense(8) theta layer in model")

    # Create dual-output model: [segmentation, theta]
    dual_output_model = Model(
        inputs=single_output_model.input,
        outputs=[single_output_model.output, theta_layer.output]
    )

    # Cache model
    if use_cache:
        _cached_model = dual_output_model

    return dual_output_model


def preprocess_image(image_path: str) -> tuple:
    """
    Load and preprocess image for ST-RoomNet inference.

    Args:
        image_path: Path to room image

    Returns:
        Tuple of (preprocessed_batch, original_rgb)
        - preprocessed_batch: Ready for model.predict() with shape (1, 400, 400, 3)
        - original_rgb: Original RGB image for visualization with shape (400, 400, 3)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to 400x400
    img = cv2.resize(img, (400, 400))

    # Keep original for visualization
    original_rgb = np.array(img, copy=True)

    # Preprocess for model
    img_batch = img[tf.newaxis, ...]
    img_preprocessed = preprocess_input(img_batch)

    return img_preprocessed, original_rgb


def preprocess_image_array(image_array: np.ndarray) -> tuple:
    """
    Preprocess numpy array image for ST-RoomNet inference.

    Args:
        image_array: Numpy array image (H, W, 3) in RGB format
                    (Standard for PIL, Gradio, and most numpy arrays)

    Returns:
        Tuple of (preprocessed_batch, original_rgb)
    """
    # Assume input is already RGB (standard for PIL/Gradio/numpy arrays)
    # If you have BGR input from cv2.imread(), use preprocess_image() instead
    img = image_array.copy()

    # Resize to 400x400
    img = cv2.resize(img, (400, 400))

    # Keep original for visualization
    original_rgb = np.array(img, copy=True)

    # Preprocess for model
    img_batch = img[tf.newaxis, ...]
    img_preprocessed = preprocess_input(img_batch)

    return img_preprocessed, original_rgb


def run_st_roomnet_inference(model: Model,
                             image_preprocessed: np.ndarray) -> tuple:
    """
    Run inference on preprocessed image.

    Args:
        model: Loaded ST-RoomNet model
        image_preprocessed: Preprocessed image batch (1, 400, 400, 3)

    Returns:
        Tuple of (segmentation, theta)
        - segmentation: 2D array with shape (400, 400) containing labels 0-4
        - theta: 1D array with shape (8,) containing transformation parameters
    """
    # Run inference
    seg_out, theta_out = model.predict(image_preprocessed, verbose=0)

    # Process segmentation (round to nearest integer label)
    segmentation = np.rint(seg_out[0, :, :, 0]).astype(int)

    # Process theta (extract from batch dimension)
    theta = theta_out[0]

    print("Inference outputs: ")
    print(f" seg_out shape: {seg_out.shape}, theta_out shape: {theta_out.shape}")
    print(f" Segmentation shape: {segmentation.shape}, unique labels: {np.unique(segmentation)}")
    print(f" Theta shape: {theta.shape}, values: {theta}")

    return segmentation, theta


def inference_from_path(image_path: str,
                       model: Model = None,
                       weights_path: str = ABS_PATH_TO_WEIGHTS,
                       ref_img_path: str = 'ref_img2.png') -> tuple:
    """
    Complete inference pipeline from image path.

    Convenience function that handles loading, preprocessing, and inference.

    Args:
        image_path: Path to room image
        model: Optional pre-loaded model (if None, will load from weights_path)
        weights_path: Path to model weights (used if model is None)
        ref_img_path: Path to reference image (used if model is None)

    Returns:
        Tuple of (segmentation, theta, original_rgb)
        - segmentation: 2D array (400, 400) with labels 0-4
        - theta: 1D array (8,) with transformation parameters
        - original_rgb: Original RGB image (400, 400, 3)
    """
    # Load model if not provided
    if model is None:
        model = load_st_roomnet_model(weights_path, ref_img_path)

    # Preprocess image
    img_preprocessed, original_rgb = preprocess_image(image_path)

    # Run inference
    segmentation, theta = run_st_roomnet_inference(model, img_preprocessed)

    return segmentation, theta, original_rgb


def inference_from_array(image_array: np.ndarray,
                        model: Model = None,
                        weights_path: str = ABS_PATH_TO_WEIGHTS,
                        ref_img_path: str = ABS_PATH_TO_IMG) -> tuple:
    """
    Complete inference pipeline from numpy array.

    Args:
        image_array: Numpy array image (H, W, 3)
        model: Optional pre-loaded model
        weights_path: Path to model weights (used if model is None)
        ref_img_path: Path to reference image (used if model is None)

    Returns:
        Tuple of (segmentation, theta, original_rgb)
    """
    # Load model if not provided
    if model is None:
        model = load_st_roomnet_model(weights_path, ref_img_path)

    # Preprocess image
    img_preprocessed, original_rgb = preprocess_image_array(image_array)

    # Run inference
    segmentation, theta = run_st_roomnet_inference(model, img_preprocessed)

    return segmentation, theta, original_rgb


def get_homography_matrix(theta: np.ndarray) -> np.ndarray:
    """
    Convert 8 theta parameters to 3x3 homography matrix.

    Args:
        theta: 8-element array of transformation parameters

    Returns:
        3x3 homography matrix
    """
    theta_with_one = np.append(theta, 1.0)
    H = theta_with_one.reshape(3, 3)
    return H


# Example usage
if __name__ == "__main__":
    print("Loading ST-RoomNet model...")
    model = load_st_roomnet_model()
    print("Model loaded successfully!")

    # Test inference on room1.jpeg
    test_image_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'images',
        'room1.jpeg'
    )
    test_image_path = ABS_PATH_TO_IMG.replace('ref_img2.png', 'room1.jpeg')

    if os.path.exists(test_image_path):
        print(f"\nRunning inference on {test_image_path}...")
        seg, theta, original = inference_from_path(test_image_path, model)

        print("\nResults:")
        print(f"  Segmentation shape: {seg.shape}")
        print(f"  Segmentation unique labels: {np.unique(seg)}")
        print(f"  Theta shape: {theta.shape}")
        print(f"  Theta values: {theta}")
        print(f"\n  Homography matrix:")
        print(get_homography_matrix(theta))
    else:
        print(f"\nTest image not found: {test_image_path}")
