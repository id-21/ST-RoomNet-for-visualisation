#!/usr/bin/env python3
"""
Shadow Overlay Experiment - V3 Pipeline Enhancement

Tests shadow overlay on ST-RoomNet V3 wallpaper output using pre-computed
RGB→X irradiance shadow maps. Compares wallpaper with and without shadows.

Following pattern from: experiments/v2-baseline-analysis/run_v2_baseline.py
Research doc: research/04-shadow-overlay-integration-research.md
Planning doc: planning/shadow-overlay-experiment-plan.md
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
try:
    import imageio
except ImportError:
    imageio = None

# Add workflow-v3 root to path to access modules
v3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, v3_root)

from modules.end_to_end_pipeline import process_room_image_complete
from modules.wallpaper_application import WallpaperPattern, apply_wallpaper_to_multiple_walls
from modules.room_wall_layout import RoomWallLayout


def load_shadow_exr(exr_path):
    """
    Load shadow EXR file as linear RGB float array.

    Args:
        exr_path: Path to irradiance shadow EXR file (linear color space)

    Returns:
        np.ndarray: Float32 array (H, W, 3) in linear RGB space

    Raises:
        FileNotFoundError: If EXR file doesn't exist
        ValueError: If loaded data has unexpected format
    """
    if not os.path.exists(exr_path):
        raise FileNotFoundError(f"Shadow EXR not found: {exr_path}")

    # Try imageio first (preferred for EXR)
    if imageio is not None:
        try:
            # Try without format specification first
            shadow_exr = imageio.imread(exr_path)
            # Ensure float32
            if shadow_exr.dtype != np.float32:
                shadow_exr = shadow_exr.astype(np.float32)
        except Exception as e:
            print(f"Warning: imageio failed ({e}), falling back to JPG preview...")
            # Fallback: load JPG preview and convert to linear space (approximation)
            jpg_path = exr_path.replace('.exr', '_preview.jpg').replace('irradiance_shadow', 'jpg_irradiance')
            if os.path.exists(jpg_path):
                print(f"  Loading JPG preview: {jpg_path}")
                shadow_jpg = imageio.imread(jpg_path)
                # Convert from sRGB to linear space
                shadow_exr = (shadow_jpg.astype(np.float32) / 255.0) ** 2.2
                print(f"  Converted JPG preview to linear RGB (approximation)")
            else:
                print(f"  JPG preview not found at {jpg_path}")
                shadow_exr = None
    else:
        shadow_exr = None

    # Fallback to OpenCV
    if shadow_exr is None:
        shadow_exr = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
        if shadow_exr is None:
            raise ValueError(f"Failed to load EXR file: {exr_path}")

    # Verify shape
    if len(shadow_exr.shape) != 3 or shadow_exr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) array, got {shadow_exr.shape}")

    # Verify data type
    if shadow_exr.dtype != np.float32:
        print(f"Warning: EXR dtype is {shadow_exr.dtype}, converting to float32")
        shadow_exr = shadow_exr.astype(np.float32)

    print(f"Loaded shadow EXR: shape={shadow_exr.shape}, dtype={shadow_exr.dtype}")
    print(f"  Value range: [{shadow_exr.min():.3f}, {shadow_exr.max():.3f}]")

    return shadow_exr


def apply_shadow_to_wall(wallpaper_rgb, shadow_exr, wall_mask, ambient_clamp=0.4):
    """
    Apply shadow to wallpaper for a single wall using linear RGB multiplication.

    Implements 5-step shadow extraction strategy from:
    research/04-shadow-overlay-integration-research.md (Shadow Extraction Strategy)

    Args:
        wallpaper_rgb: Wallpaper image (H, W, 3) uint8 in sRGB space
        shadow_exr: Shadow map (H, W, 3) float32 in linear space
        wall_mask: Boolean mask (H, W) defining wall region
        ambient_clamp: Minimum shadow value (default 0.4 for ambient light)

    Returns:
        np.ndarray: Shadowed wallpaper (H, W, 3) uint8 in sRGB space
    """
    # Step 1: Isolate wall region (mask already provided)
    if not np.any(wall_mask):
        print("    Warning: Empty wall mask, skipping shadow application")
        return wallpaper_rgb

    # Step 2: Establish baseline brightness (98th percentile of wall pixels)
    # Extract wall shadow values (average across RGB channels for brightness)
    wall_shadow_pixels = shadow_exr[wall_mask].mean(axis=1)  # Average RGB per pixel
    baseline = np.percentile(wall_shadow_pixels, 98)

    print(f"    Baseline brightness: {baseline:.3f} (98th percentile)")

    # Handle edge case: completely dark wall
    if baseline < 1e-6:
        print(f"    Warning: Baseline too low ({baseline:.3e}), using fallback value 0.1")
        baseline = 0.1

    # Step 3: Create relative shadow mask (normalize by baseline)
    # Work with grayscale shadow for simpler normalization
    shadow_gray = shadow_exr.mean(axis=2)  # Convert to grayscale
    normalized_shadow = shadow_gray / baseline

    # Step 4: Clamp to preserve ambient light
    normalized_shadow = np.clip(normalized_shadow, ambient_clamp, 1.0)

    print(f"    Normalized shadow range: [{normalized_shadow.min():.3f}, {normalized_shadow.max():.3f}]")

    # Step 5: Apply to wallpaper (multiply in linear space)
    # Convert wallpaper from sRGB to linear
    wallpaper_linear = (wallpaper_rgb.astype(np.float32) / 255.0) ** 2.2

    # Apply shadow (broadcast grayscale shadow to RGB)
    shadow_3ch = normalized_shadow[:, :, np.newaxis]  # (H, W, 1)
    shadowed_linear = wallpaper_linear * shadow_3ch

    # Convert back to sRGB
    shadowed_srgb = (shadowed_linear ** (1.0 / 2.2)) * 255.0
    shadowed_srgb = np.clip(shadowed_srgb, 0, 255).astype(np.uint8)

    return shadowed_srgb


def create_comparison_image(original, wallpaper_only, wallpaper_with_shadow):
    """
    Create horizontal 3-panel comparison image with labels.

    Args:
        original: Original room image (H, W, 3)
        wallpaper_only: Wallpaper without shadow (H, W, 3)
        wallpaper_with_shadow: Wallpaper with shadow (H, W, 3)

    Returns:
        np.ndarray: Horizontal concatenation with text labels
    """
    # Ensure all images are same height
    target_height = original.shape[0]

    def resize_to_height(img, height):
        h, w = img.shape[:2]
        aspect = w / h
        new_width = int(height * aspect)
        return cv2.resize(img, (new_width, height), interpolation=cv2.INTER_LINEAR)

    img1 = resize_to_height(original, target_height)
    img2 = resize_to_height(wallpaper_only, target_height)
    img3 = resize_to_height(wallpaper_with_shadow, target_height)

    # Add padding between images
    padding = np.ones((target_height, 10, 3), dtype=np.uint8) * 255

    # Concatenate horizontally
    comparison = np.hstack([img1, padding, img2, padding, img3])

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    labels = ["Original", "Wallpaper Only", "Wallpaper + Shadow"]
    x_positions = [10, img1.shape[1] + 20, img1.shape[1] + img2.shape[1] + 40]

    for label, x in zip(labels, x_positions):
        y = 40
        # Draw background rectangle
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(comparison, (x - 5, y - text_height - 5),
                     (x + text_width + 5, y + 5), bg_color, -1)
        # Draw text
        cv2.putText(comparison, label, (x, y), font, font_scale, color, font_thickness)

    return comparison


def main():
    """
    Main experiment orchestration.

    Pipeline flow:
    1. Run ST-RoomNet V3 → get walls + masks
    2. Apply wallpaper → get wallpaper_only output
    3. Load shadow EXR → resize to match
    4. Apply shadow per wall → get wallpaper_with_shadow output
    5. Save comparison outputs
    """
    print("="*80)
    print("SHADOW OVERLAY EXPERIMENT - ST-ROOMNET V3")
    print("="*80)

    # Define paths
    input_image = '/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/living_simple2.jpg'
    wallpaper_path = '/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/images/graphite-clay-repeated-patterns-coordonne.jpg'
    shadow_exr_path = '/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/irradiance_shadow.exr'
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')

    print(f"\nInput image: {input_image}")
    print(f"Wallpaper pattern: {wallpaper_path}")
    print(f"Shadow EXR: {shadow_exr_path}")
    print(f"Output directory: {output_dir}")

    # Verify files exist
    for path in [input_image, wallpaper_path, shadow_exr_path]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            return

    # Step 1: Run ST-RoomNet V3 pipeline
    print("\n" + "-"*80)
    print("STEP 1: Running ST-RoomNet V3 Pipeline")
    print("-"*80)

    layout, st_seg, seg_conf, refined_masks = process_room_image_complete(
        image_path=input_image,
        wall_threshold=0.4,
        use_cached_models=True
    )

    # Load original image for compositing
    original_img = cv2.imread(input_image)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    print(f"\nPipeline complete:")
    print(f"  Original image shape: {original_img.shape}")
    print(f"  ST-RoomNet segmentation shape: {st_seg.shape}")
    print(f"  Number of walls detected: {len(layout.walls)}")
    print(f"  Wall labels: {list(refined_masks.keys())}")

    # Step 2: Load and apply wallpaper (without shadow)
    print("\n" + "-"*80)
    print("STEP 2: Applying Wallpaper (No Shadow)")
    print("-"*80)

    pattern_img = cv2.imread(wallpaper_path)
    pattern_img = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2RGB)
    print(f"Wallpaper pattern loaded: {pattern_img.shape}")

    wallpaper_pattern = WallpaperPattern(
        image=pattern_img,
        wall_height_inches=108.0,
        pattern_height_inches=21.0
    )

    # Resize original to match ST-RoomNet output (400x400)
    original_resized = cv2.resize(original_img, (400, 400), interpolation=cv2.INTER_LINEAR)

    wallpaper_only = apply_wallpaper_to_multiple_walls(
        wallpaper_pattern=wallpaper_pattern,
        walls=layout.walls,
        original_image=original_resized,
        wall_filter=None  # Apply to all walls
    )

    print(f"Wallpaper applied (no shadow): {wallpaper_only.shape}")

    # Step 3: Load shadow EXR
    print("\n" + "-"*80)
    print("STEP 3: Loading Shadow EXR")
    print("-"*80)

    shadow_exr = load_shadow_exr(shadow_exr_path)

    # Resize shadow to match ST-RoomNet output (400x400)
    print(f"Resizing shadow from {shadow_exr.shape[:2]} to (400, 400)")
    shadow_resized = cv2.resize(shadow_exr, (400, 400), interpolation=cv2.INTER_LINEAR)
    print(f"Shadow resized: {shadow_resized.shape}")

    # Step 4: Apply shadow to wallpaper per wall
    print("\n" + "-"*80)
    print("STEP 4: Applying Shadow to Each Wall")
    print("-"*80)

    # Start with wallpaper_only as base
    wallpaper_with_shadow = wallpaper_only.copy()

    for wall in layout.walls:
        if not wall.is_visible:
            continue

        # wall.wall_id might be a string like "left_wall" or an int like 1
        # Need to find corresponding numeric label in refined_masks
        wall_id_str = wall.wall_id

        # Map string wall IDs to numeric labels (1=left, 2=front, 3=right)
        label_map = {'left_wall': 1, 'front_wall': 2, 'right_wall': 3}

        if isinstance(wall_id_str, str) and wall_id_str in label_map:
            wall_label = label_map[wall_id_str]
            wall_name = wall_id_str
        elif isinstance(wall_id_str, int):
            wall_label = wall_id_str
            wall_name = {1: 'left_wall', 2: 'front_wall', 3: 'right_wall'}.get(wall_label, f'wall_{wall_label}')
        else:
            wall_label = wall_id_str
            wall_name = f'wall_{wall_label}'

        print(f"\n  Processing {wall_name} (wall_id={wall_id_str}, label={wall_label}):")

        # Get refined mask for this wall
        if wall_label not in refined_masks:
            print(f"    Warning: No refined mask for label {wall_label}, skipping")
            print(f"    Available masks: {list(refined_masks.keys())}")
            continue

        wall_mask = refined_masks[wall_label]
        print(f"    Wall mask shape: {wall_mask.shape}, pixels: {np.sum(wall_mask)}")

        # Apply shadow to this wall region only
        shadowed_wall = apply_shadow_to_wall(
            wallpaper_rgb=wallpaper_with_shadow,
            shadow_exr=shadow_resized,
            wall_mask=wall_mask,
            ambient_clamp=0.4
        )

        # Composite shadowed wall back (only replace wall pixels)
        wallpaper_with_shadow[wall_mask] = shadowed_wall[wall_mask]
        print(f"    Shadow applied to {wall_name}")

    print(f"\nShadow application complete: {wallpaper_with_shadow.shape}")

    # Step 5: Save outputs
    print("\n" + "-"*80)
    print("STEP 5: Saving Output Images")
    print("-"*80)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save individual outputs
    cv2.imwrite(
        os.path.join(output_dir, '00_original.jpg'),
        cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)
    )
    print(f"  Saved: 00_original.jpg")

    cv2.imwrite(
        os.path.join(output_dir, '01_wallpaper_only.jpg'),
        cv2.cvtColor(wallpaper_only, cv2.COLOR_RGB2BGR)
    )
    print(f"  Saved: 01_wallpaper_only.jpg")

    cv2.imwrite(
        os.path.join(output_dir, '02_wallpaper_with_shadow.jpg'),
        cv2.cvtColor(wallpaper_with_shadow, cv2.COLOR_RGB2BGR)
    )
    print(f"  Saved: 02_wallpaper_with_shadow.jpg")

    # Create and save comparison
    comparison = create_comparison_image(
        original_resized,
        wallpaper_only,
        wallpaper_with_shadow
    )

    cv2.imwrite(
        os.path.join(output_dir, '03_comparison.jpg'),
        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    )
    print(f"  Saved: 03_comparison.jpg")

    # Save shadow visualization
    shadow_vis = (np.clip(shadow_resized, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(
        os.path.join(output_dir, '04_shadow_exr_visualization.jpg'),
        cv2.cvtColor(shadow_vis, cv2.COLOR_RGB2BGR)
    )
    print(f"  Saved: 04_shadow_exr_visualization.jpg")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"All outputs saved to: {output_dir}")
    print("\nKey outputs:")
    print(f"  - 00_original.jpg: Original room image")
    print(f"  - 01_wallpaper_only.jpg: Wallpaper without shadow")
    print(f"  - 02_wallpaper_with_shadow.jpg: Wallpaper with shadow overlay")
    print(f"  - 03_comparison.jpg: Side-by-side comparison (3 panels)")
    print(f"  - 04_shadow_exr_visualization.jpg: Shadow map visualization")


if __name__ == "__main__":
    main()
