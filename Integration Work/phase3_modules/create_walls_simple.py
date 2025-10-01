# File: create_walls_simple.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import cv2
from room_wall_layout import Wall, RoomWallLayout

# Label mappings from ST-RoomNet
LABEL_NAMES = {
    0: "ceiling",
    1: "left_wall",
    2: "front_wall",
    3: "right_wall",
    4: "floor"
}

SURFACE_TYPES = {
    0: "ceiling",
    1: "wall",
    2: "wall",
    3: "wall",
    4: "floor"
}

def create_wall_from_mask(mask, label, theta=None):
    """
    Create Wall object from combined mask

    Args:
        mask: Binary mask (boolean or uint8 array)
        label: ST-RoomNet label (0-4)
        theta: Optional theta parameters for normal calculation (future)

    Returns:
        Wall object
    """
    wall_id = LABEL_NAMES.get(label, f"surface_{label}")
    wall = Wall(wall_id=wall_id)

    # Convert mask to uint8 if needed
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = mask.astype(np.uint8)

    # Extract boundary contour
    contours, _ = cv2.findContours(mask_uint8,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Take largest contour
        largest = max(contours, key=cv2.contourArea)

        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        # Extract corners
        wall.corners_2d = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
    else:
        wall.corners_2d = []

    # Set pixel mask
    wall.pixel_mask = mask > 0 if mask.dtype == bool else mask > 127

    # Set type based on label
    wall.surface_type = SURFACE_TYPES.get(label, "unknown")
    wall.is_visible = np.sum(wall.pixel_mask) > 100  # Min 100 pixels

    # Calculate confidence based on mask density
    if len(contours) > 0:
        total_pixels = np.sum(wall.pixel_mask)
        bounding_area = cv2.contourArea(largest) if cv2.contourArea(largest) > 0 else 1
        wall.confidence = min(1.0, total_pixels / bounding_area)
    else:
        wall.confidence = 0.0

    # Normal vector calculation deferred to future refinement
    wall.normal_vector = None

    return wall

def create_walls_from_masks(combined_masks, theta=None):
    """
    Create multiple Wall objects from mask dictionary

    Args:
        combined_masks: Dict of {label: mask}
        theta: Optional theta parameters

    Returns:
        List of Wall objects
    """
    walls = []

    for label, mask in combined_masks.items():
        wall = create_wall_from_mask(mask, label, theta)
        if wall.is_visible:
            walls.append(wall)

    return walls

def extract_wall_stats(wall):
    """
    Extract statistics from a Wall object for debugging

    Args:
        wall: Wall object

    Returns:
        Dict of statistics
    """
    stats = {
        'wall_id': wall.wall_id,
        'surface_type': wall.surface_type,
        'is_visible': wall.is_visible,
        'confidence': wall.confidence,
        'num_corners': len(wall.corners_2d),
        'pixel_count': np.sum(wall.pixel_mask) if wall.pixel_mask is not None else 0,
        'has_normal': wall.normal_vector is not None
    }

    if wall.pixel_mask is not None:
        # Calculate bounding box
        y_coords, x_coords = np.where(wall.pixel_mask)
        if len(y_coords) > 0:
            stats['bbox'] = {
                'min_x': int(np.min(x_coords)),
                'max_x': int(np.max(x_coords)),
                'min_y': int(np.min(y_coords)),
                'max_y': int(np.max(y_coords))
            }

    return stats
