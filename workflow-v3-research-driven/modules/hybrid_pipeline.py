"""
Hybrid dual-resolution pipeline for ST-RoomNet + SegFormer with resolution preservation.

Reference: workflow-v3-research-driven/planning/fix-resolution-aspect-ratio.md:84-95

Key architecture:
- Branch 1: ST-RoomNet at 400×400 (with preprocessing option A/B/C) → upscale to H×W
- Branch 2: SegFormer at original H×W resolution (no preprocessing)
- Combine masks at H×W resolution
- Compute normal vectors from SCALED theta (CRITICAL for wallpaper)
- Create Wall objects with normals
- Apply wallpaper at H×W resolution

This preserves original image resolution while respecting ST-RoomNet's 400×400 requirement.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional

# Import preprocessing/postprocessing
from modules.preprocessing import preprocess_for_st_roomnet
from modules.postprocessing import postprocess_st_roomnet_output

# Import model loaders and inference
from modules.st_roomnet_model import (
    load_st_roomnet_model,
    inference_from_array as st_inference_array
)
from modules.segformer_model import (
    load_segformer_model,
    run_segformer_inference
)

# Import mask processing
from modules.simple_mask_combination import combine_masks_simple
from modules.normal_computation import compute_normal_vectors
from modules.create_walls_simple import create_walls_from_masks

# Import data structures and wallpaper
from modules.room_wall_layout import RoomWallLayout
from modules.wallpaper_application import WallpaperPattern, apply_wallpaper_to_multiple_walls


# Absolute paths (update these for your machine)
ABS_PATH_TO_WEIGHTS = '/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/Weight_ST_RroomNet_ConvNext.h5'
ABS_PATH_TO_IMG = '/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/images/ref_img2.png'

# Global model caches
_st_roomnet_model = None
_segformer_model = None
_segformer_processor = None
_segformer_device = None


def initialize_models_v3(st_weights_path: str = ABS_PATH_TO_WEIGHTS,
                        st_ref_img_path: str = ABS_PATH_TO_IMG,
                        segformer_model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512"):
    """
    Pre-load and cache both models for faster subsequent inference.

    Reference: workflow-v2-integrated-pipeline/modules/end_to_end_pipeline.py:37-69

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


def process_dual_resolution(image_path: str,
                            option: str = 'A',
                            wall_threshold: float = 0.4,
                            wall_labels: List[int] = [1, 2, 3],
                            wallpaper_pattern: Optional[np.ndarray] = None,
                            use_cached_models: bool = True) -> Tuple[RoomWallLayout, dict]:
    """
    Run complete hybrid dual-resolution pipeline with normal vectors.

    Reference: workflow-v3-research-driven/planning/fix-resolution-aspect-ratio.md:84-95

    Args:
        image_path: Path to input room image
        option: Preprocessing option 'A', 'B', or 'C'
        wall_threshold: SegFormer confidence threshold (default: 0.4)
        wall_labels: Labels to process (default: [1,2,3] for walls)
        wallpaper_pattern: Optional wallpaper image to apply (BGR format)
        use_cached_models: Use cached models (default: True)

    Returns:
        Tuple of (layout, debug_info)
        - layout: RoomWallLayout with walls including normal vectors
        - debug_info: Dictionary with intermediate results
    """
    # Load input image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    H, W = img.shape[:2]
    print(f"\n{'='*70}")
    print(f"HYBRID PIPELINE - Option {option}")
    print(f"{'='*70}")
    print(f"Input: {W}×{H} ({W*H:,} pixels)")

    # Initialize models
    if use_cached_models:
        st_model, seg_model, seg_processor, seg_device = initialize_models_v3()
    else:
        st_model = load_st_roomnet_model(ABS_PATH_TO_WEIGHTS, ABS_PATH_TO_IMG, use_cache=False)
        seg_model, seg_processor, seg_device = load_segformer_model(use_cache=False)

    # ==========================================
    # BRANCH 1: ST-RoomNet at 400×400
    # ==========================================
    print(f"\n[Branch 1] ST-RoomNet at 400×400")

    # Step 1.1: Preprocess for ST-RoomNet
    preprocessed_img, metadata = preprocess_for_st_roomnet(img, option=option, target_size=400)
    print(f"  Preprocessed: {preprocessed_img.shape[1]}×{preprocessed_img.shape[0]} (option {option})")

    # Step 1.2: ST-RoomNet inference at 400×400
    # Convert BGR to RGB for ST-RoomNet
    preprocessed_rgb = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB)
    seg_400, theta_400, _ = st_inference_array(preprocessed_rgb, model=st_model)
    print(f"  ST output: seg {seg_400.shape}, theta {theta_400.shape}")

    # Step 1.3: Postprocess to original resolution
    seg_HxW, theta_scaled = postprocess_st_roomnet_output(seg_400, theta_400, metadata)
    print(f"  Postprocessed: seg {seg_HxW.shape}, theta scaled")

    # ==========================================
    # BRANCH 2: SegFormer at original H×W
    # ==========================================
    print(f"\n[Branch 2] SegFormer at {W}×{H}")

    # Convert BGR to PIL RGB for SegFormer
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # SegFormer inference at original resolution
    confidence_HxW = run_segformer_inference(seg_model, seg_processor, seg_device, img_pil, wall_class_id=0)
    print(f"  SegFormer output: {confidence_HxW.shape}")

    # ==========================================
    # COMBINE at H×W resolution
    # ==========================================
    print(f"\n[Combine] Mask combination at {W}×{H}")

    # Combine masks using AND operation
    combined_masks = combine_masks_simple(
        seg_HxW,
        confidence_HxW,
        wall_threshold=wall_threshold,
        wall_labels=wall_labels
    )

    for wall_label, mask in combined_masks.items():
        pixel_count = np.sum(mask)
        print(f"  Wall {wall_label}: {pixel_count:,} pixels")

    # ==========================================
    # COMPUTE NORMAL VECTORS (CRITICAL!)
    # ==========================================
    print(f"\n[Normal Vectors] Computing from scaled theta")

    # CRITICAL: Use theta_scaled (not theta_400) for normal computation
    # Reference: planning doc line 91
    normals_dict = compute_normal_vectors(theta_scaled, wall_labels)

    for label, normal in normals_dict.items():
        magnitude = np.linalg.norm(normal)
        print(f"  Wall {label}: normal={normal}, magnitude={magnitude:.4f}")

    # ==========================================
    # CREATE WALL OBJECTS WITH NORMALS
    # ==========================================
    print(f"\n[Walls] Creating Wall objects with normals")

    # Reference: planning doc line 92, workflow-v2/modules/create_walls_simple.py:87-109
    walls = create_walls_from_masks(combined_masks, normals_dict=normals_dict)

    print(f"  Created {len(walls)} visible walls")
    for wall in walls:
        print(f"    {wall.wall_id}: {np.sum(wall.pixel_mask):,} pixels, normal={wall.normal_vector}")

    # ==========================================
    # CREATE ROOM LAYOUT
    # ==========================================
    layout = RoomWallLayout(image_width=W, image_height=H)
    layout.walls = walls
    layout.model_source = f"ST-RoomNet + SegFormer (Option {option}, H×W resolution)"

    # ==========================================
    # APPLY WALLPAPER (if provided)
    # ==========================================
    output_image = img.copy()

    if wallpaper_pattern is not None:
        print(f"\n[Wallpaper] Applying at {W}×{H}")

        wallpaper = WallpaperPattern(image=wallpaper_pattern)
        output_image = apply_wallpaper_to_multiple_walls(
            wallpaper_pattern=wallpaper,
            walls=walls,
            original_image=img
        )
        print(f"  Wallpaper applied to {len(walls)} walls")
    else:
        print(f"\n[Output] No wallpaper provided, returning layout only")

    # ==========================================
    # DEBUG INFO
    # ==========================================
    debug_info = {
        'option': option,
        'original_shape': (H, W),
        'metadata': metadata,
        'seg_400': seg_400,
        'theta_400': theta_400,
        'seg_HxW': seg_HxW,
        'theta_scaled': theta_scaled,
        'confidence_HxW': confidence_HxW,
        'combined_masks': combined_masks,
        'normals_dict': normals_dict,
        'output_image': output_image,
        'resolution_gain': (W * H) / (400 * 400)
    }

    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"{'='*70}")
    print(f"Resolution gain: {debug_info['resolution_gain']:.1f}x (from 400×400 to {W}×{H})")
    print(f"Walls created: {len(walls)}")

    return layout, debug_info
