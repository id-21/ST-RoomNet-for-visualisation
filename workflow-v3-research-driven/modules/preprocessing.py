"""
Preprocessing module for converting H×W input images to 400×400 for ST-RoomNet.

Reference: workflow-v3-research-driven/planning/fix-resolution-aspect-ratio.md:31-37

Three preprocessing options:
- Option A: Direct resize (distortion acceptable)
- Option B: Letterbox padding (aspect ratio preserved)
- Option C: Center crop (aspect ratio preserved, content lost)

Each function returns (preprocessed_image, metadata) where metadata
contains transformation parameters needed for reverse postprocessing.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


def preprocess_option_a(img: np.ndarray, target_size: int = 400) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Option A: Direct resize to 400×400 (distortion acceptable).

    Simple resize that may distort aspect ratio. ST-RoomNet sees distorted geometry
    during inference, but no content is lost.

    Args:
        img: Input image (H, W, 3) in BGR format
        target_size: Target dimension (default: 400)

    Returns:
        Tuple of (resized_image, metadata)
        - resized_image: (400, 400, 3) BGR image
        - metadata: {'option': 'A', 'original_shape': (H, W), 'scale_x': float, 'scale_y': float}
    """
    H, W = img.shape[:2]

    # Direct resize to target_size × target_size
    resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Calculate scale factors for postprocessing
    scale_x = W / target_size
    scale_y = H / target_size

    metadata = {
        'option': 'A',
        'original_shape': (H, W),
        'scale_x': scale_x,
        'scale_y': scale_y,
        'target_size': target_size
    }

    return resized, metadata


def preprocess_option_b(img: np.ndarray, target_size: int = 400) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Option B: Letterbox padding (aspect ratio preserved).

    Pads shorter dimension to create square image, then resizes to 400×400.
    Preserves aspect ratio during inference, but ST-RoomNet sees black bars.

    Args:
        img: Input image (H, W, 3) in BGR format
        target_size: Target dimension (default: 400)

    Returns:
        Tuple of (padded_resized_image, metadata)
        - padded_resized_image: (400, 400, 3) BGR image with letterboxing
        - metadata: Contains padding offsets and scale factor
    """
    H, W = img.shape[:2]

    # Determine padding needed to make square
    max_dim = max(H, W)

    # Calculate padding for each side (center the image)
    pad_top = (max_dim - H) // 2
    pad_bottom = max_dim - H - pad_top
    pad_left = (max_dim - W) // 2
    pad_right = max_dim - W - pad_left

    # Pad image to square with black borders
    padded = cv2.copyMakeBorder(
        img,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    # Resize padded square to target size
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Calculate scale factor for postprocessing
    scale = max_dim / target_size

    metadata = {
        'option': 'B',
        'original_shape': (H, W),
        'pad_top': pad_top,
        'pad_bottom': pad_bottom,
        'pad_left': pad_left,
        'pad_right': pad_right,
        'padded_size': max_dim,
        'scale': scale,
        'target_size': target_size
    }

    return resized, metadata


def preprocess_option_c(img: np.ndarray, target_size: int = 400) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Option C: Center crop to square (aspect ratio preserved, content lost).

    Crops to min(H,W) × min(H,W) square from center, then resizes to 400×400.
    No distortion, no artificial padding, but may lose ceiling/floor content.

    Args:
        img: Input image (H, W, 3) in BGR format
        target_size: Target dimension (default: 400)

    Returns:
        Tuple of (cropped_resized_image, metadata)
        - cropped_resized_image: (400, 400, 3) BGR image
        - metadata: Contains crop offsets and scale factor
    """
    H, W = img.shape[:2]

    # Determine crop size (smallest dimension)
    crop_size = min(H, W)

    # Calculate crop offsets (center aligned)
    crop_top = (H - crop_size) // 2
    crop_left = (W - crop_size) // 2

    # Crop to square
    cropped = img[crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]

    # Resize to target size
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Calculate scale factor for postprocessing
    scale = crop_size / target_size

    metadata = {
        'option': 'C',
        'original_shape': (H, W),
        'crop_top': crop_top,
        'crop_left': crop_left,
        'crop_size': crop_size,
        'scale': scale,
        'target_size': target_size
    }

    return resized, metadata


def preprocess_for_st_roomnet(img: np.ndarray, option: str = 'A', target_size: int = 400) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Unified preprocessing interface that dispatches to the selected option.

    Args:
        img: Input image (H, W, 3) in BGR format
        option: Preprocessing option ('A', 'B', or 'C')
        target_size: Target dimension (default: 400)

    Returns:
        Tuple of (preprocessed_image, metadata)

    Raises:
        ValueError: If option is not 'A', 'B', or 'C'
    """
    option = option.upper()

    if option == 'A':
        return preprocess_option_a(img, target_size)
    elif option == 'B':
        return preprocess_option_b(img, target_size)
    elif option == 'C':
        return preprocess_option_c(img, target_size)
    else:
        raise ValueError(f"Invalid option '{option}'. Must be 'A', 'B', or 'C'.")