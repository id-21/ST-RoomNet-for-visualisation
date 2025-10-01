# File: edge_refinement_simple.py
import numpy as np
import cv2

def refine_edges_with_st_boundaries(combined_mask, st_plane_mask):
    """
    Use ST-RoomNet boundaries to clean up edges

    Args:
        combined_mask: Initial combined mask (boolean array)
        st_plane_mask: Original ST-RoomNet plane (boolean array)

    Returns:
        Refined mask with cleaner edges (boolean array)
    """
    # Convert to uint8 for OpenCV operations
    combined_uint8 = combined_mask.astype(np.uint8) * 255
    st_uint8 = st_plane_mask.astype(np.uint8) * 255

    # Extract ST-RoomNet boundaries
    st_edges = cv2.Canny(st_uint8, 100, 200)

    # Create buffer zone around edges (5 pixels)
    edge_buffer = cv2.dilate(st_edges, np.ones((5, 5), np.uint8))

    # Copy combined mask
    refined = combined_uint8.copy()

    # In edge buffer zone, use ST-RoomNet boundaries
    edge_pixels = edge_buffer > 0
    refined[edge_pixels] = st_uint8[edge_pixels]

    # Final smoothing
    refined = cv2.medianBlur(refined, 3)

    return refined > 0

def refine_edges_bilateral(combined_mask, st_plane_mask, sigma_color=75, sigma_space=75):
    """
    Use bilateral filtering for edge-preserving smoothing

    Args:
        combined_mask: Initial combined mask
        st_plane_mask: Original ST-RoomNet plane
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space

    Returns:
        Refined mask with smooth edges
    """
    combined_uint8 = combined_mask.astype(np.uint8) * 255

    # Apply bilateral filter
    filtered = cv2.bilateralFilter(combined_uint8, d=9,
                                   sigmaColor=sigma_color,
                                   sigmaSpace=sigma_space)

    # Extract ST boundaries as constraints
    st_uint8 = st_plane_mask.astype(np.uint8) * 255
    st_edges = cv2.Canny(st_uint8, 100, 200)
    edge_buffer = cv2.dilate(st_edges, np.ones((3, 3), np.uint8))

    # Enforce ST boundaries in edge regions
    edge_pixels = edge_buffer > 0
    filtered[edge_pixels] = st_uint8[edge_pixels]

    return filtered > 127  # Binary threshold

def smooth_contours(mask, epsilon_factor=0.01):
    """
    Smooth mask by approximating contours

    Args:
        mask: Binary mask
        epsilon_factor: Approximation accuracy factor (smaller = more accurate)

    Returns:
        Smoothed mask
    """
    mask_uint8 = mask.astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask

    # Create blank canvas
    smoothed = np.zeros_like(mask_uint8)

    for contour in contours:
        # Approximate contour
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Draw approximated contour
        cv2.fillPoly(smoothed, [approx], 255)

    return smoothed > 0
