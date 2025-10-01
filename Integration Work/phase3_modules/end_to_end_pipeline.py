# File: end_to_end_pipeline.py
"""
End-to-end pipeline for ST-RoomNet + SegFormer integration with normal vector computation.

This module provides a complete pipeline that:
1. Runs ST-RoomNet inference (segmentation + theta)
2. Runs SegFormer inference (wall confidence)
3. Combines masks using AND operation
4. Refines edges with ST-RoomNet boundaries
5. Computes normal vectors from theta
6. Creates complete Wall objects with all attributes
7. Returns RoomWallLayout ready for visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from PIL import Image
from typing import Tuple, List, Optional

from room_wall_layout import Wall, RoomWallLayout
from st_roomnet_model import load_st_roomnet_model, inference_from_path as st_inference_path, inference_from_array as st_inference_array
from segformer_model import load_segformer_model, inference_from_path as seg_inference_path, inference_from_pil as seg_inference_pil
from simple_mask_combination import combine_masks_simple
from edge_refinement_simple import refine_edges_with_st_boundaries
from create_walls_simple import create_walls_from_masks
from normal_computation import compute_normal_vectors

ABS_PATH_TO_WEIGHTS = '/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/Weight_ST_RroomNet_ConvNext.h5'
ABS_PATH_TO_IMG = '/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/Integration Work/images/ref_img2.png'

# Global model caches
_st_roomnet_model = None
_segformer_model = None
_segformer_processor = None
_segformer_device = None


def initialize_models(st_weights_path: str = ABS_PATH_TO_WEIGHTS,
                     st_ref_img_path: str = ABS_PATH_TO_IMG,
                     segformer_model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512"):
    """
    Pre-load and cache both models for faster subsequent inference.

    Args:
        st_weights_path: Path to ST-RoomNet weights
        st_ref_img_path: Path to reference cuboid image
        segformer_model_name: HuggingFace model identifier

    Returns:
        Tuple of (st_model, seg_model, seg_processor, seg_device)
    """
    global _st_roomnet_model, _segformer_model, _segformer_processor, _segformer_device

    print("Initializing models...")

    # Load ST-RoomNet
    if _st_roomnet_model is None:
        print("  Loading ST-RoomNet...")
        _st_roomnet_model = load_st_roomnet_model(st_weights_path, st_ref_img_path, use_cache=True)

    # Load SegFormer
    if _segformer_model is None:
        print("  Loading SegFormer...")
        _segformer_model, _segformer_processor, _segformer_device = load_segformer_model(
            segformer_model_name, use_cache=True
        )

    print("Models initialized!")

    return _st_roomnet_model, _segformer_model, _segformer_processor, _segformer_device


def process_room_image_complete(image_path: str,
                                wall_threshold: float = 0.4,
                                wall_labels: List[int] = [1, 2, 3],
                                st_weights_path: str = ABS_PATH_TO_WEIGHTS,
                                st_ref_img_path: str = ABS_PATH_TO_IMG,
                                segformer_model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
                                use_cached_models: bool = True) -> RoomWallLayout:
    """
    Complete end-to-end pipeline from image path to RoomWallLayout with normal vectors.

    Args:
        image_path: Path to room image
        wall_threshold: Confidence threshold for SegFormer (default: 0.4)
        wall_labels: Labels to process (default: [1,2,3] for walls only)
        st_weights_path: Path to ST-RoomNet weights
        st_ref_img_path: Path to reference cuboid image
        segformer_model_name: HuggingFace model identifier
        use_cached_models: Use globally cached models (default: True)

    Returns:
        RoomWallLayout object with complete Wall objects including normal vectors
    """
    # Initialize models
    if use_cached_models:
        st_model, seg_model, seg_processor, seg_device = initialize_models(
            st_weights_path, st_ref_img_path, segformer_model_name
        )
    else:
        st_model = load_st_roomnet_model(st_weights_path, st_ref_img_path, use_cache=False)
        seg_model, seg_processor, seg_device = load_segformer_model(segformer_model_name, use_cache=False)

    # 1. Run ST-RoomNet inference
    print("Running ST-RoomNet inference...")
    st_seg, theta, original_rgb = st_inference_path(image_path, model=st_model)

    # 2. Run SegFormer inference
    print("Running SegFormer inference...")
    # Load image as PIL for SegFormer
    pil_image = Image.open(image_path).convert('RGB')
    seg_conf = seg_inference_pil(pil_image, model=seg_model, processor=seg_processor, device=seg_device)

    # 3. Resize SegFormer output to match ST-RoomNet (400x400)
    import cv2
    if seg_conf.shape != st_seg.shape:
        seg_conf = cv2.resize(seg_conf, (st_seg.shape[1], st_seg.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 4. Combine masks
    print(f"Combining masks with threshold={wall_threshold}...")
    combined_masks = combine_masks_simple(st_seg, seg_conf, wall_threshold=wall_threshold, wall_labels=wall_labels)

    # 5. Refine edges
    print("Refining edges...")
    refined_masks = {}
    for label, mask in combined_masks.items():
        st_plane = (st_seg == label)
        refined = refine_edges_with_st_boundaries(mask, st_plane)
        refined_masks[label] = refined

    # 6. Compute normal vectors from theta
    print("Computing normal vectors...")
    normals_dict = compute_normal_vectors(theta, wall_labels)

    # 7. Create Wall objects with normals
    print("Creating Wall objects with normal vectors...")
    walls = create_walls_from_masks_with_normals(refined_masks, normals_dict)

    # 8. Create RoomWallLayout
    layout = RoomWallLayout(image_width=st_seg.shape[1], image_height=st_seg.shape[0])
    layout.walls = walls
    layout.model_source = "ST-RoomNet + SegFormer (with normal vectors)"

    print(f"Pipeline complete! Created {len(walls)} visible walls with normal vectors.")

    return layout


def process_room_image_from_array(image_array: np.ndarray,
                                  wall_threshold: float = 0.4,
                                  wall_labels: List[int] = [1, 2, 3],
                                  st_weights_path: str = ABS_PATH_TO_WEIGHTS,
                                  st_ref_img_path: str = ABS_PATH_TO_IMG,
                                  segformer_model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
                                  use_cached_models: bool = True) -> RoomWallLayout:
    """
    Complete end-to-end pipeline from numpy array to RoomWallLayout with normal vectors.

    Args:
        image_array: Numpy array image (H, W, 3) - RGB format
        wall_threshold: Confidence threshold for SegFormer (default: 0.4)
        wall_labels: Labels to process (default: [1,2,3] for walls only)
        st_weights_path: Path to ST-RoomNet weights
        st_ref_img_path: Path to reference cuboid image
        segformer_model_name: HuggingFace model identifier
        use_cached_models: Use globally cached models (default: True)

    Returns:
        RoomWallLayout object with complete Wall objects including normal vectors
    """
    # Initialize models
    if use_cached_models:
        st_model, seg_model, seg_processor, seg_device = initialize_models(
            st_weights_path, st_ref_img_path, segformer_model_name
        )
    else:
        st_model = load_st_roomnet_model(st_weights_path, st_ref_img_path, use_cache=False)
        seg_model, seg_processor, seg_device = load_segformer_model(segformer_model_name, use_cache=False)

    # 1. Run ST-RoomNet inference
    print("Running ST-RoomNet inference...")
    st_seg, theta, original_rgb = st_inference_array(image_array, model=st_model)

    # 2. Run SegFormer inference
    print("Running SegFormer inference...")
    # Convert numpy array to PIL
    from segformer_model import inference_from_array as seg_inference_array
    seg_conf = seg_inference_array(image_array, model=seg_model, processor=seg_processor, device=seg_device)

    # 3. Resize SegFormer output to match ST-RoomNet (400x400)
    import cv2
    if seg_conf.shape != st_seg.shape:
        seg_conf = cv2.resize(seg_conf, (st_seg.shape[1], st_seg.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 4. Combine masks
    print(f"Combining masks with threshold={wall_threshold}...")
    combined_masks = combine_masks_simple(st_seg, seg_conf, wall_threshold=wall_threshold, wall_labels=wall_labels)

    # 5. Refine edges
    print("Refining edges...")
    refined_masks = {}
    for label, mask in combined_masks.items():
        st_plane = (st_seg == label)
        refined = refine_edges_with_st_boundaries(mask, st_plane)
        refined_masks[label] = refined

    # 6. Compute normal vectors from theta
    print("Computing normal vectors...")
    normals_dict = compute_normal_vectors(theta, wall_labels)

    # 7. Create Wall objects with normals
    print("Creating Wall objects with normal vectors...")
    walls = create_walls_from_masks_with_normals(refined_masks, normals_dict)

    # 8. Create RoomWallLayout
    layout = RoomWallLayout(image_width=st_seg.shape[1], image_height=st_seg.shape[0])
    layout.walls = walls
    layout.model_source = "ST-RoomNet + SegFormer (with normal vectors)"

    print(f"Pipeline complete! Created {len(walls)} visible walls with normal vectors.")

    return layout


def create_walls_from_masks_with_normals(refined_masks: dict,
                                        normals_dict: dict) -> List[Wall]:
    """
    Create Wall objects from masks and assign normal vectors.

    This is a modified version of create_walls_from_masks that assigns normal vectors.

    Args:
        refined_masks: Dictionary mapping label to refined mask
        normals_dict: Dictionary mapping label to normal vector

    Returns:
        List of Wall objects with normal vectors assigned
    """
    # Create walls using existing function
    walls = create_walls_from_masks(refined_masks)

    # Assign normal vectors
    for wall in walls:
        # Determine label from wall_id
        label_mapping = {
            'ceiling': 0,
            'left_wall': 1,
            'front_wall': 2,
            'right_wall': 3,
            'floor': 4
        }

        label = label_mapping.get(wall.wall_id, None)

        if label is not None and label in normals_dict:
            wall.normal_vector = normals_dict[label]
        else:
            # No normal vector available for this wall
            wall.normal_vector = None

    return walls


def get_summary(layout: RoomWallLayout) -> dict:
    """
    Generate summary statistics for the pipeline output.

    Args:
        layout: RoomWallLayout object

    Returns:
        Dictionary with summary statistics
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
            'pixel_count': int(np.sum(wall.pixel_mask)) if wall.pixel_mask is not None else 0,
            'has_normal': wall.normal_vector is not None
        }

        if wall.normal_vector is not None:
            wall_info['normal_vector'] = wall.normal_vector.tolist()
            wall_info['normal_magnitude'] = float(np.linalg.norm(wall.normal_vector))

        summary['walls'].append(wall_info)

    return summary


# Example usage
if __name__ == "__main__":
    test_image_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'Integration Work',
        'images',
        'room1.jpeg'
    )

    if os.path.exists(test_image_path):
        print("=" * 60)
        print("End-to-End Pipeline Test")
        print("=" * 60)

        # Run complete pipeline
        layout = process_room_image_complete(test_image_path, wall_threshold=0.4)

        # Get summary
        summary = get_summary(layout)

        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"Model: {summary['model_source']}")
        print(f"Image size: {summary['image_dimensions']}")
        print(f"Walls detected: {summary['num_walls']}")

        for wall_info in summary['walls']:
            print(f"\n  {wall_info['wall_id']}:")
            print(f"    Type: {wall_info['surface_type']}")
            print(f"    Visible: {wall_info['is_visible']}")
            print(f"    Confidence: {wall_info['confidence']:.3f}")
            print(f"    Pixels: {wall_info['pixel_count']}")
            print(f"    Corners: {wall_info['num_corners']}")
            print(f"    Has normal: {wall_info['has_normal']}")

            if wall_info['has_normal']:
                normal = wall_info['normal_vector']
                magnitude = wall_info['normal_magnitude']
                print(f"    Normal: [{normal[0]:7.4f}, {normal[1]:7.4f}, {normal[2]:7.4f}]")
                print(f"    Magnitude: {magnitude:.6f}")

        print("\n" + "=" * 60)
        print("Pipeline test complete!")
        print("=" * 60)
    else:
        print(f"Test image not found: {test_image_path}")
