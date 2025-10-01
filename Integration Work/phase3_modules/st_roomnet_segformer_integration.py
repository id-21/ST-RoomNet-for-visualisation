# File: st_roomnet_segformer_integration.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from load_outputs import load_st_roomnet_output, load_segformer_output, resize_to_match
from simple_mask_combination import combine_masks_simple
from edge_refinement_simple import refine_edges_with_st_boundaries
from create_walls_simple import create_walls_from_masks
from room_wall_layout import RoomWallLayout
from normal_computation import compute_normal_vectors

def process_room_image(image_path, st_output_path, segformer_confidence_path,
                      theta_path=None, wall_threshold=0.4, wall_labels=[1, 2, 3]):
    """
    Main function to process room image with both models

    Args:
        image_path: Path to original room image
        st_output_path: Path to ST-RoomNet segmentation output
        segformer_confidence_path: Path to SegFormer confidence map
        theta_path: Path to theta parameters file (optional, for normal computation)
        wall_threshold: Confidence threshold for SegFormer (default: 0.4)
        wall_labels: Labels to process (default: [1,2,3] for walls only)

    Returns:
        RoomWallLayout object with integrated results
    """
    # Load outputs
    print("Loading ST-RoomNet output...")
    st_seg = load_st_roomnet_output(st_output_path)

    print("Loading SegFormer confidence map...")
    seg_conf = load_segformer_output(segformer_confidence_path)

    # Resize SegFormer output to match ST-RoomNet if needed
    seg_conf = resize_to_match(seg_conf, st_seg.shape)

    print(f"ST-RoomNet shape: {st_seg.shape}, SegFormer shape: {seg_conf.shape}")

    # Combine masks
    print(f"Combining masks with threshold={wall_threshold}...")
    combined_masks = combine_masks_simple(st_seg, seg_conf,
                                         wall_threshold=wall_threshold,
                                         wall_labels=wall_labels)

    # Refine edges
    print("Refining edges...")
    refined_masks = {}
    for label, mask in combined_masks.items():
        st_plane = (st_seg == label)
        refined = refine_edges_with_st_boundaries(mask, st_plane)
        refined_masks[label] = refined

    # Compute normal vectors if theta available
    normals_dict = None
    if theta_path is not None:
        print(f"Loading theta parameters from {theta_path}...")
        try:
            theta = np.loadtxt(theta_path, skiprows=2)  # Skip header lines
            if len(theta) == 8:
                print("Computing normal vectors...")
                normals_dict = compute_normal_vectors(theta, wall_labels)
            else:
                print(f"Warning: Expected 8 theta values, got {len(theta)}")
        except Exception as e:
            print(f"Warning: Could not load theta parameters: {e}")

    # Create Wall objects with normals
    print("Creating Wall objects...")
    walls = create_walls_from_masks(refined_masks, normals_dict=normals_dict)

    # Create RoomWallLayout
    layout = RoomWallLayout(image_width=st_seg.shape[1],
                           image_height=st_seg.shape[0])
    layout.walls = walls

    # Update model source to indicate normal vectors
    if normals_dict is not None:
        layout.model_source = "ST-RoomNet + SegFormer (with normal vectors)"
    else:
        layout.model_source = "ST-RoomNet + SegFormer"

    print(f"Created {len(walls)} visible walls")
    if normals_dict is not None:
        print(f"Computed normal vectors for {len(normals_dict)} surfaces")

    return layout, refined_masks, combined_masks

def get_integration_summary(layout):
    """
    Generate summary statistics for the integration

    Args:
        layout: RoomWallLayout object

    Returns:
        Dict of summary statistics
    """
    summary = {
        'model_source': layout.model_source,
        'image_dimensions': layout.image_dimensions,
        'num_walls': len(layout.walls),
        'walls': []
    }

    for wall in layout.walls:
        wall_info = {
            'wall_id': wall.wall_id,
            'surface_type': wall.surface_type,
            'is_visible': wall.is_visible,
            'confidence': wall.confidence,
            'num_corners': len(wall.corners_2d),
            'pixel_count': int(np.sum(wall.pixel_mask)) if wall.pixel_mask is not None else 0
        }
        summary['walls'].append(wall_info)

    return summary
