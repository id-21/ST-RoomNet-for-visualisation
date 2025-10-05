"""
Postprocessing module to upscale ST-RoomNet outputs to original resolution.

Reference: workflow-v3-research-driven/planning/fix-resolution-aspect-ratio.md:39-44

Handles:
1. Upscaling segmentation mask from 400×400 to H×W
2. Scaling theta (homography) coordinates via conjugation: H_scaled = S @ H_400 @ S^-1

Three options corresponding to preprocessing options:
- Option A: Direct upscale (reverse distortion)
- Option B: Remove letterbox padding, then upscale
- Option C: Place in cropped region, zero-fill lost areas
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


def postprocess_option_a(seg_400: np.ndarray,
                         theta_400: np.ndarray,
                         metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Option A: Direct upscale (reverse distortion).

    Upscale 400×400 segmentation to original H×W, reversing the aspect ratio distortion.
    Scale theta coordinates using diagonal scaling matrix.

    Args:
        seg_400: Segmentation mask (400, 400) with labels 0-4
        theta_400: Theta parameters (8,) from ST-RoomNet
        metadata: Preprocessing metadata containing scale factors

    Returns:
        Tuple of (seg_HxW, theta_scaled)
        - seg_HxW: Segmentation mask at original resolution (H, W)
        - theta_scaled: Scaled theta parameters (8,) for original resolution
    """
    H, W = metadata['original_shape']
    scale_x = metadata['scale_x']  # W / 400
    scale_y = metadata['scale_y']  # H / 400

    # Upscale segmentation to original size (nearest neighbor for labels)
    seg_HxW = cv2.resize(seg_400, (W, H), interpolation=cv2.INTER_NEAREST)

    # Scale theta coordinates
    # Theta represents homography H: [h0, h1, h2]
    #                                [h3, h4, h5]
    #                                [h6, h7,  1]
    # Scaling: H_scaled = S @ H_400 @ S^-1
    # where S = diag([scale_x, scale_y, 1])

    # Append 1.0 to theta to make 9 elements, reshape to 3×3
    theta_9 = np.append(theta_400, 1.0)
    H_400 = theta_9.reshape(3, 3)

    # Create scaling matrices
    S = np.diag([scale_x, scale_y, 1.0])
    S_inv = np.diag([1.0/scale_x, 1.0/scale_y, 1.0])

    # Apply conjugation: H_scaled = S @ H_400 @ S_inv
    H_scaled = S @ H_400 @ S_inv

    # Extract first 8 parameters
    theta_scaled = H_scaled.flatten()[:8]

    return seg_HxW, theta_scaled


def postprocess_option_b(seg_400: np.ndarray,
                         theta_400: np.ndarray,
                         metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Option B: Remove letterbox padding, then upscale.

    First upscale to padded square size, then crop out padding regions.

    Args:
        seg_400: Segmentation mask (400, 400) with padding
        theta_400: Theta parameters (8,) from ST-RoomNet
        metadata: Preprocessing metadata with padding info

    Returns:
        Tuple of (seg_HxW, theta_scaled)
        - seg_HxW: Segmentation mask at original resolution (H, W)
        - theta_scaled: Scaled theta parameters (8,) for original resolution
    """
    H, W = metadata['original_shape']
    pad_top = metadata['pad_top']
    pad_left = metadata['pad_left']
    padded_size = metadata['padded_size']
    scale = metadata['scale']  # padded_size / 400

    # Step 1: Upscale to padded square size
    seg_padded = cv2.resize(seg_400, (padded_size, padded_size), interpolation=cv2.INTER_NEAREST)

    # Step 2: Remove padding regions (extract original content area)
    seg_HxW = seg_padded[pad_top:pad_top + H, pad_left:pad_left + W]

    # Step 3: Scale theta coordinates
    # Need to account for:
    # 1. Scale from 400 to padded_size
    # 2. Translation offset from padding removal

    theta_9 = np.append(theta_400, 1.0)
    H_400 = theta_9.reshape(3, 3)

    # Scaling matrix: 400 → padded_size
    S_scale = np.diag([scale, scale, 1.0])
    S_scale_inv = np.diag([1.0/scale, 1.0/scale, 1.0])

    # Translation matrix: offset from padding removal
    T_offset = np.array([
        [1, 0, -pad_left],
        [0, 1, -pad_top],
        [0, 0, 1]
    ], dtype=np.float32)

    T_offset_inv = np.array([
        [1, 0, pad_left],
        [0, 1, pad_top],
        [0, 0, 1]
    ], dtype=np.float32)

    # Combined transformation: H_scaled = T_offset @ S_scale @ H_400 @ S_scale_inv @ T_offset_inv
    H_scaled = T_offset @ S_scale @ H_400 @ S_scale_inv @ T_offset_inv

    # Extract first 8 parameters
    theta_scaled = H_scaled.flatten()[:8]

    return seg_HxW, theta_scaled


def postprocess_option_c(seg_400: np.ndarray,
                         theta_400: np.ndarray,
                         metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Option C: Place in cropped region, zero-fill lost areas.

    Upscale 400×400 to crop_size, then place in original image space
    at the crop offset. Areas outside crop are filled with zeros.

    Args:
        seg_400: Segmentation mask (400, 400) from center crop
        theta_400: Theta parameters (8,) from ST-RoomNet
        metadata: Preprocessing metadata with crop info

    Returns:
        Tuple of (seg_HxW, theta_scaled)
        - seg_HxW: Segmentation mask at original resolution (H, W)
        - theta_scaled: Scaled theta parameters (8,) for original resolution
    """
    H, W = metadata['original_shape']
    crop_top = metadata['crop_top']
    crop_left = metadata['crop_left']
    crop_size = metadata['crop_size']
    scale = metadata['scale']  # crop_size / 400

    # Step 1: Upscale to crop_size
    seg_crop = cv2.resize(seg_400, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

    # Step 2: Create full-size image and place crop in correct position
    seg_HxW = np.zeros((H, W), dtype=seg_400.dtype)
    seg_HxW[crop_top:crop_top + crop_size, crop_left:crop_left + crop_size] = seg_crop

    # Step 3: Scale theta coordinates with offset
    theta_9 = np.append(theta_400, 1.0)
    H_400 = theta_9.reshape(3, 3)

    # Scaling matrix: 400 → crop_size
    S_scale = np.diag([scale, scale, 1.0])
    S_scale_inv = np.diag([1.0/scale, 1.0/scale, 1.0])

    # Translation matrix: offset from crop placement
    T_offset = np.array([
        [1, 0, crop_left],
        [0, 1, crop_top],
        [0, 0, 1]
    ], dtype=np.float32)

    # Combined transformation: H_scaled = T_offset @ S_scale @ H_400 @ S_scale_inv
    H_scaled = T_offset @ S_scale @ H_400 @ S_scale_inv

    # Extract first 8 parameters
    theta_scaled = H_scaled.flatten()[:8]

    return seg_HxW, theta_scaled


def postprocess_st_roomnet_output(seg_400: np.ndarray,
                                   theta_400: np.ndarray,
                                   metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified postprocessing function that dispatches to the correct option.

    Args:
        seg_400: Segmentation mask (400, 400) from ST-RoomNet
        theta_400: Theta parameters (8,) from ST-RoomNet
        metadata: Preprocessing metadata containing 'option' key

    Returns:
        Tuple of (seg_HxW, theta_scaled)

    Raises:
        ValueError: If metadata['option'] is not 'A', 'B', or 'C'
    """
    option = metadata.get('option', 'A').upper()

    if option == 'A':
        return postprocess_option_a(seg_400, theta_400, metadata)
    elif option == 'B':
        return postprocess_option_b(seg_400, theta_400, metadata)
    elif option == 'C':
        return postprocess_option_c(seg_400, theta_400, metadata)
    else:
        raise ValueError(f"Invalid option '{option}' in metadata. Must be 'A', 'B', or 'C'.")
