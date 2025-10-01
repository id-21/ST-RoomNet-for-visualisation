# File: simple_mask_combination.py
import numpy as np
import cv2

def combine_masks_simple(st_segmentation, segformer_confidence,
                         wall_threshold=0.4, wall_labels=[1, 2, 3]):
    """
    Simple AND operation between ST-RoomNet planes and SegFormer walls

    Args:
        st_segmentation: 400x400 array with labels 0-4
            0=ceiling, 1=left wall, 2=front wall, 3=right wall, 4=floor
        segformer_confidence: 400x400 array with wall confidence scores (0-1)
        wall_threshold: Confidence threshold for SegFormer (default: 0.4)
        wall_labels: ST-RoomNet labels for walls (default: [1, 2, 3])

    Returns:
        dict: {label: binary_mask} for each wall
    """
    # Create binary wall mask from SegFormer
    segformer_wall_mask = (segformer_confidence > wall_threshold)

    # Combine for each wall plane
    combined_masks = {}
    for label in wall_labels:
        # Get ST-RoomNet plane mask
        st_plane_mask = (st_segmentation == label)

        # Simple AND operation
        combined_mask = st_plane_mask & segformer_wall_mask

        # Basic cleanup
        combined_mask = cleanup_mask_simple(combined_mask)

        combined_masks[label] = combined_mask

    return combined_masks

def cleanup_mask_simple(mask):
    """
    Basic morphological cleanup

    Args:
        mask: Binary mask (boolean array)

    Returns:
        Cleaned binary mask
    """
    # Convert to uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8) * 255

    # Remove small holes
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    # Remove small isolated regions
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

    return mask_cleaned > 0

def combine_masks_adaptive(st_segmentation, segformer_confidence,
                           base_threshold=0.4, wall_labels=[1, 2, 3]):
    """
    Adaptive thresholding version for future enhancement

    Uses per-wall adaptive thresholds based on confidence distribution

    Args:
        st_segmentation: ST-RoomNet segmentation
        segformer_confidence: SegFormer confidence map
        base_threshold: Base confidence threshold
        wall_labels: Wall labels to process

    Returns:
        dict: {label: binary_mask} for each wall
    """
    combined_masks = {}

    for label in wall_labels:
        st_plane_mask = (st_segmentation == label)

        # Get confidence values for this plane
        plane_confidences = segformer_confidence[st_plane_mask]

        if len(plane_confidences) > 0:
            # Adaptive threshold: mean - std
            adaptive_threshold = max(base_threshold,
                                    np.mean(plane_confidences) - np.std(plane_confidences))
        else:
            adaptive_threshold = base_threshold

        # Apply threshold
        segformer_mask = (segformer_confidence > adaptive_threshold)
        combined_mask = st_plane_mask & segformer_mask
        combined_mask = cleanup_mask_simple(combined_mask)

        combined_masks[label] = combined_mask

    return combined_masks
