# File: wallpaper_application.py
"""
Modular wallpaper application using computed normal vectors for perspective transformation.

This module provides a clean interface for applying wallpaper patterns to wall surfaces
with perspective correction based on computed 3D normal vectors.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class WallpaperPattern:
    """
    Container for wallpaper pattern data.

    Attributes:
        image: The wallpaper pattern image (BGR or RGB)
        tile_width: Width of a single tile in pixels (for future tiling)
        tile_height: Height of a single tile in pixels (for future tiling)
        offset_x: Horizontal offset for pattern alignment (for future use)
        offset_y: Vertical offset for pattern alignment (for future use)
    """
    image: np.ndarray
    tile_width: Optional[int] = None
    tile_height: Optional[int] = None
    offset_x: float = 0.0
    offset_y: float = 0.0


def compute_perspective_from_normal(normal_vector: np.ndarray,
                                   wall_corners: list,
                                   image_height: int) -> np.ndarray:
    """
    Compute perspective transformation matrix from 3D normal vector.

    Uses the normal vector to determine the wall's orientation in 3D space
    and creates an appropriate perspective transformation.

    Args:
        normal_vector: 3D unit normal vector [nx, ny, nz]
        wall_corners: List of 2D corner points [(x, y), ...]
        image_height: Height of the image (for perspective calculation)

    Returns:
        3x3 perspective transformation matrix
    """
    if len(wall_corners) < 4:
        # If less than 4 corners, use bounding rectangle
        corners_array = np.array(wall_corners, dtype=np.float32)
        x_min, y_min = corners_array.min(axis=0)
        x_max, y_max = corners_array.max(axis=0)
        src_pts = np.array([
            [x_min, y_max],  # Bottom-left
            [x_max, y_max],  # Bottom-right
            [x_max, y_min],  # Top-right
            [x_min, y_min]   # Top-left
        ], dtype=np.float32)
    else:
        # Use actual corners (take first 4 if more than 4)
        src_pts = np.array(wall_corners[:4], dtype=np.float32)

    # Calculate perspective based on normal vector
    nx, ny, nz = normal_vector

    # Determine wall orientation and perspective offset
    # Horizontal normals (walls) have larger |nx| or |nz|
    # The normal vector indicates the direction the wall faces

    # Calculate perspective offset based on normal direction
    width = np.linalg.norm(src_pts[1] - src_pts[0])
    height = np.linalg.norm(src_pts[3] - src_pts[0])

    # Use normal vector to determine vanishing point offset
    # Walls facing away have stronger perspective distortion
    perspective_strength = abs(nz) * 0.3  # Front-facing walls (nz ≈ -1) have more perspective

    # Calculate offset for trapezoidal transformation
    offset = width * perspective_strength

    # Destination points with perspective correction
    dst_pts = np.array([
        [src_pts[0][0] - offset, src_pts[0][1]],  # Bottom-left (shift left)
        [src_pts[1][0] + offset, src_pts[1][1]],  # Bottom-right (shift right)
        [src_pts[2][0], src_pts[2][1]],           # Top-right (no shift)
        [src_pts[3][0], src_pts[3][1]]            # Top-left (no shift)
    ], dtype=np.float32)

    # If wall is left/right facing, adjust horizontal perspective
    if abs(nx) > abs(nz):
        # Left wall (nx < 0) or right wall (nx > 0)
        horizontal_offset = height * abs(nx) * 0.2

        if nx < 0:  # Left wall
            dst_pts[0][0] += horizontal_offset
            dst_pts[1][0] += horizontal_offset
        else:  # Right wall
            dst_pts[0][0] -= horizontal_offset
            dst_pts[1][0] -= horizontal_offset

    # Compute perspective transformation matrix
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return H


def apply_wallpaper(wallpaper_pattern: WallpaperPattern,
                   wall_object,
                   use_normal_perspective: bool = True) -> np.ndarray:
    """
    Apply wallpaper pattern to wall with perspective transformation.

    This is the main modular function that applies wallpaper to a wall surface
    using either normal vector-based perspective or simple mapping.

    Args:
        wallpaper_pattern: WallpaperPattern object containing the pattern image
        wall_object: Wall object with corners_2d, pixel_mask, and normal_vector
        use_normal_perspective: If True, use normal vector for perspective (default: True)

    Returns:
        Transformed wallpaper pattern as RGBA image (with alpha channel for masking)

    Raises:
        ValueError: If wall_object is missing required attributes
    """
    # Validate wall object
    if not hasattr(wall_object, 'pixel_mask') or wall_object.pixel_mask is None:
        raise ValueError("Wall object must have pixel_mask attribute")

    if not hasattr(wall_object, 'corners_2d') or len(wall_object.corners_2d) == 0:
        raise ValueError("Wall object must have corners_2d attribute with at least one corner")

    # Get wall mask dimensions
    mask_height, mask_width = wall_object.pixel_mask.shape

    # Get bounding box from pixel mask
    y_coords, x_coords = np.where(wall_object.pixel_mask)
    if len(y_coords) == 0:
        # Empty mask, return empty image
        return np.zeros((mask_height, mask_width, 4), dtype=np.uint8)

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    # Prepare wallpaper pattern (tile to cover bounding box)
    pattern = wallpaper_pattern.image
    pattern_h, pattern_w = pattern.shape[:2]

    # Tile pattern to at least cover the bounding box
    tiles_x = max(1, int(np.ceil(bbox_width / pattern_w)))
    tiles_y = max(1, int(np.ceil(bbox_height / pattern_h)))

    tiled_pattern = np.tile(pattern, (tiles_y, tiles_x, 1))
    # Crop to exact bbox size
    tiled_pattern = tiled_pattern[:bbox_height, :bbox_width]

    # Apply perspective transformation if normal vector available
    if use_normal_perspective and hasattr(wall_object, 'normal_vector') and wall_object.normal_vector is not None:
        # Compute perspective transformation from normal vector
        H = compute_perspective_from_normal(
            wall_object.normal_vector,
            wall_object.corners_2d,
            mask_height
        )

        # Apply transformation to tiled pattern
        warped_pattern = cv2.warpPerspective(
            tiled_pattern,
            H,
            (mask_width, mask_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
    else:
        # Simple placement without perspective
        warped_pattern = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
        warped_pattern[y_min:y_min+bbox_height, x_min:x_min+bbox_width] = tiled_pattern

    # Apply wall mask to create alpha channel
    if warped_pattern.shape[2] == 3:
        # Add alpha channel
        alpha = (wall_object.pixel_mask * 255).astype(np.uint8)
        warped_pattern_rgba = np.dstack([warped_pattern, alpha])
    else:
        # Already has alpha, update it
        warped_pattern_rgba = warped_pattern.copy()
        warped_pattern_rgba[:, :, 3] = (wall_object.pixel_mask * 255).astype(np.uint8)

    return warped_pattern_rgba


def overlay_wallpaper_on_image(original_image: np.ndarray,
                               wallpaper_rgba: np.ndarray,
                               blend_mode: str = 'normal') -> np.ndarray:
    """
    Overlay the transformed wallpaper onto the original image.

    Args:
        original_image: Original room image (BGR or RGB)
        wallpaper_rgba: Transformed wallpaper with alpha channel
        blend_mode: Blending mode ('normal', 'multiply', 'screen')

    Returns:
        Composite image with wallpaper applied
    """
    # Ensure dimensions match
    if original_image.shape[:2] != wallpaper_rgba.shape[:2]:
        wallpaper_rgba = cv2.resize(wallpaper_rgba,
                                    (original_image.shape[1], original_image.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)

    # Extract alpha channel and normalize
    alpha = wallpaper_rgba[:, :, 3].astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)  # Shape: (H, W, 1)

    # Extract RGB from wallpaper
    wallpaper_rgb = wallpaper_rgba[:, :, :3].astype(np.float32)
    original_float = original_image.astype(np.float32)

    # Blend based on mode
    if blend_mode == 'normal':
        # Standard alpha blending
        blended = wallpaper_rgb * alpha + original_float * (1 - alpha)
    elif blend_mode == 'multiply':
        # Multiply blend (darker)
        blended = (wallpaper_rgb * original_float / 255.0) * alpha + original_float * (1 - alpha)
    elif blend_mode == 'screen':
        # Screen blend (lighter)
        blended = (255 - (255 - wallpaper_rgb) * (255 - original_float) / 255.0) * alpha + original_float * (1 - alpha)
    else:
        # Default to normal
        blended = wallpaper_rgb * alpha + original_float * (1 - alpha)

    return blended.astype(np.uint8)


def apply_wallpaper_to_multiple_walls(wallpaper_pattern: WallpaperPattern,
                                     walls: list,
                                     original_image: np.ndarray,
                                     wall_filter: Optional[list] = None) -> np.ndarray:
    """
    Apply wallpaper to multiple walls in a single image.

    Args:
        wallpaper_pattern: WallpaperPattern object
        walls: List of Wall objects
        original_image: Original room image
        wall_filter: Optional list of wall_ids to apply wallpaper to (None = all visible walls)

    Returns:
        Image with wallpaper applied to all specified walls
    """
    result = original_image.copy()

    for wall in walls:
        # Skip if wall not visible
        if not wall.is_visible:
            continue

        # Skip if wall not in filter
        if wall_filter is not None and wall.wall_id not in wall_filter:
            continue

        # Skip ceiling and floor
        if wall.surface_type in ['ceiling', 'floor']:
            continue

        try:
            # Apply wallpaper to this wall
            wallpaper_rgba = apply_wallpaper(wallpaper_pattern, wall, use_normal_perspective=True)

            # Overlay on result
            result = overlay_wallpaper_on_image(result, wallpaper_rgba)

        except Exception as e:
            print(f"Warning: Failed to apply wallpaper to {wall.wall_id}: {e}")
            continue

    return result


# Example usage
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    from room_wall_layout import Wall

    # Create test wallpaper pattern (red and white checkerboard)
    pattern = np.zeros((100, 100, 3), dtype=np.uint8)
    pattern[0:50, 0:50] = [200, 50, 50]    # Red
    pattern[0:50, 50:100] = [255, 255, 255]  # White
    pattern[50:100, 0:50] = [255, 255, 255]  # White
    pattern[50:100, 50:100] = [200, 50, 50]  # Red

    wallpaper = WallpaperPattern(image=pattern)

    # Create test wall with normal vector
    test_wall = Wall(wall_id="test_wall")
    test_wall.pixel_mask = np.zeros((400, 400), dtype=bool)
    test_wall.pixel_mask[100:300, 50:200] = True
    test_wall.corners_2d = [(50, 300), (200, 300), (200, 100), (50, 100)]
    test_wall.normal_vector = np.array([0.0, 0.0, -1.0])  # Front-facing wall
    test_wall.is_visible = True
    test_wall.surface_type = "wall"

    print("Testing wallpaper application...")
    result = apply_wallpaper(wallpaper, test_wall, use_normal_perspective=True)
    print(f"Result shape: {result.shape}")
    print(f"Has alpha channel: {result.shape[2] == 4}")
    print(f"Non-zero pixels: {np.sum(result[:, :, 3] > 0)}")
    print("Test complete!")
