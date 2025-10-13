#!/usr/bin/env python3
"""
Shadow Darkening Visualization - White Canvas Analysis

Applies shadow overlay to a pure WHITE canvas (instead of wallpaper) to isolate
the shadow darkening effect. Tests multiple ambient clamp values to visualize
the exact shadow intensity at different parameter settings.

CORRECTED: Uses fixed inverted shadow application where EXR irradiance has
HIGH values for shadows (not bright areas).

Reference: planning/shadow-darkening-visualization-plan.md
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
from modules.room_wall_layout import RoomWallLayout

# Clamp values to test
CLAMP_VALUES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]


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


def create_white_canvas(shape=(400, 400, 3)):
    """
    Create pure white RGB image for shadow visualization.

    Args:
        shape: Image dimensions (H, W, C), default 400x400 RGB

    Returns:
        np.ndarray: Pure white image (255, 255, 255) uint8
    """
    return np.ones(shape, dtype=np.uint8) * 255


def apply_shadow_to_wall(canvas_rgb, shadow_exr, wall_mask, ambient_clamp=0.4):
    """
    Apply shadow to canvas for a single wall using linear RGB multiplication.

    IMPORTANT: This implementation corrects for INVERTED irradiance values where:
    - High EXR values represent DARK shadows (should darken canvas)
    - Low EXR values represent BRIGHT areas (should keep canvas bright)

    Implements corrected 6-step shadow extraction strategy:
    1. Isolate wall region
    2. Establish min/max irradiance range for normalization
    3. Normalize to 0-1 (0=bright, 1=dark)
    4. INVERT to create shadow multiplier (1=no darkening, low=darken)
    5. Clamp to preserve ambient light
    6. Multiply in linear RGB space

    Args:
        canvas_rgb: Canvas image (H, W, 3) uint8 in sRGB space
        shadow_exr: Shadow map (H, W, 3) float32 - HIGH values = shadows
        wall_mask: Boolean mask (H, W) defining wall region
        ambient_clamp: Minimum shadow multiplier (default 0.4 for ambient light)

    Returns:
        np.ndarray: Shadowed canvas (H, W, 3) uint8 in sRGB space
    """
    # Step 1: Isolate wall region (mask already provided)
    if not np.any(wall_mask):
        print("    Warning: Empty wall mask, skipping shadow application")
        return canvas_rgb

    # Step 2: Establish value range for normalization
    # IMPORTANT: EXR irradiance has INVERTED relationship:
    # - High EXR values = darkest shadows (should darken canvas)
    # - Low EXR values = brightest areas (should keep canvas bright)
    wall_shadow_pixels = shadow_exr[wall_mask].mean(axis=1)  # Average RGB per pixel
    min_val = np.percentile(wall_shadow_pixels, 2)   # Brightest areas (low irradiance values)
    max_val = np.percentile(wall_shadow_pixels, 98)  # Darkest shadows (high irradiance values)

    # Handle edge case: no variation in shadow
    if max_val - min_val < 1e-6:
        print(f"    Warning: No shadow variation detected, skipping shadow application")
        return canvas_rgb

    # Step 3: Normalize to 0-1 range where 0=bright, 1=dark
    shadow_gray = shadow_exr.mean(axis=2)  # Convert to grayscale
    normalized = (shadow_gray - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Step 4: INVERT so that high irradiance (shadows) → low multiplier (darken)
    # and low irradiance (bright) → high multiplier (stay bright)
    shadow_multiplier = 1.0 - normalized

    # Step 5: Clamp to preserve ambient light (don't go below ambient_clamp)
    shadow_multiplier = np.clip(shadow_multiplier, ambient_clamp, 1.0)

    # Step 6: Apply to canvas (multiply in linear space)
    # Convert canvas from sRGB to linear
    canvas_linear = (canvas_rgb.astype(np.float32) / 255.0) ** 2.2

    # Apply shadow (broadcast grayscale shadow to RGB)
    shadow_3ch = shadow_multiplier[:, :, np.newaxis]  # (H, W, 1)
    shadowed_linear = canvas_linear * shadow_3ch

    # Convert back to sRGB
    shadowed_srgb = (shadowed_linear ** (1.0 / 2.2)) * 255.0
    shadowed_srgb = np.clip(shadowed_srgb, 0, 255).astype(np.uint8)

    return shadowed_srgb


def create_clamp_comparison_grid(original, shadow_vis, white_no_shadow, white_results_dict):
    """
    Create grid comparing white canvas results at different clamp values.

    Layout (3 columns):
    Row 1: Original | Shadow Vis | White (no shadow)
    Row 2: Clamp 0.0 | Clamp 0.2 | Clamp 0.3
    Row 3: Clamp 0.4 | Clamp 0.5 | Clamp 0.6
    Row 4: Clamp 0.8 |           |

    Args:
        original: Original room image (H, W, 3)
        shadow_vis: Shadow EXR visualization (H, W, 3)
        white_no_shadow: White canvas without shadow (H, W, 3)
        white_results_dict: {clamp_value: result_image}

    Returns:
        np.ndarray: Grid layout with labels
    """
    # Target size for each cell
    cell_size = 300
    padding = 10

    def resize_and_label(img, label):
        """Resize image and add text label."""
        img_resized = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_LINEAR)
        # Add label at top
        labeled = img_resized.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        color = (255, 255, 255)
        bg_color = (0, 0, 0)

        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        x = 10
        y = 30

        # Draw background rectangle
        cv2.rectangle(labeled, (x - 5, y - text_height - 5),
                     (x + text_width + 5, y + 5), bg_color, -1)
        # Draw text
        cv2.putText(labeled, label, (x, y), font, font_scale, color, font_thickness)

        return labeled

    # Prepare all cells
    cells = []

    # Row 1: Original, Shadow Vis, White (no shadow)
    cells.append([
        resize_and_label(original, "Original"),
        resize_and_label(shadow_vis, "Shadow Map"),
        resize_and_label(white_no_shadow, "White (No Shadow)")
    ])

    # Rows 2-4: Clamp variations
    clamp_images = []
    for clamp in CLAMP_VALUES:
        if clamp in white_results_dict:
            clamp_images.append(
                resize_and_label(white_results_dict[clamp], f"Clamp {clamp:.1f}")
            )

    # Arrange clamp images in rows of 3
    num_clamps = len(clamp_images)
    for i in range(0, num_clamps, 3):
        row = clamp_images[i:i+3]
        # Pad row with black if less than 3
        while len(row) < 3:
            row.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))
        cells.append(row)

    # Create grid
    rows = []
    for row_cells in cells:
        # Add padding between cells
        row_with_padding = []
        for i, cell in enumerate(row_cells):
            row_with_padding.append(cell)
            if i < len(row_cells) - 1:
                row_with_padding.append(np.ones((cell_size, padding, 3), dtype=np.uint8) * 255)
        rows.append(np.hstack(row_with_padding))

    # Add padding between rows
    rows_with_padding = []
    for i, row in enumerate(rows):
        rows_with_padding.append(row)
        if i < len(rows) - 1:
            rows_with_padding.append(np.ones((padding, row.shape[1], 3), dtype=np.uint8) * 255)

    grid = np.vstack(rows_with_padding)

    return grid


def create_per_wall_breakdown(layout, white_canvas, shadow_exr, refined_masks, ambient_clamp=0.4):
    """
    Show shadow effect on each wall individually.

    Creates horizontal layout: Wall 1 | Wall 2 | Wall 3
    Each showing white canvas with shadow applied only to that wall.

    Args:
        layout: RoomWallLayout with detected walls
        white_canvas: Pure white canvas (H, W, 3)
        shadow_exr: Shadow map (H, W, 3) float32
        refined_masks: Dict mapping wall_label to boolean mask
        ambient_clamp: Ambient light clamp value

    Returns:
        np.ndarray: Horizontal concatenation of per-wall results
    """
    wall_results = []

    # Map string wall IDs to numeric labels (1=left, 2=front, 3=right)
    label_map = {'left_wall': 1, 'front_wall': 2, 'right_wall': 3}
    name_map = {1: 'Left Wall', 2: 'Front Wall', 3: 'Right Wall'}

    for wall in layout.walls:
        if not wall.is_visible:
            continue

        # Determine wall label
        wall_id_str = wall.wall_id
        if isinstance(wall_id_str, str) and wall_id_str in label_map:
            wall_label = label_map[wall_id_str]
            wall_name = name_map.get(wall_label, f'Wall {wall_label}')
        elif isinstance(wall_id_str, int):
            wall_label = wall_id_str
            wall_name = name_map.get(wall_label, f'Wall {wall_label}')
        else:
            wall_label = wall_id_str
            wall_name = f'Wall {wall_label}'

        # Get wall mask
        if wall_label not in refined_masks:
            print(f"Warning: No refined mask for {wall_name}, skipping")
            continue

        wall_mask = refined_masks[wall_label]

        # Create white canvas copy and apply shadow only to this wall
        canvas_copy = white_canvas.copy()
        shadowed = apply_shadow_to_wall(
            canvas_rgb=canvas_copy,
            shadow_exr=shadow_exr,
            wall_mask=wall_mask,
            ambient_clamp=ambient_clamp
        )

        # Add label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        color = (255, 255, 255)
        bg_color = (0, 0, 0)

        (text_width, text_height), _ = cv2.getTextSize(wall_name, font, font_scale, font_thickness)
        x = 10
        y = 40

        # Draw background rectangle
        cv2.rectangle(shadowed, (x - 5, y - text_height - 5),
                     (x + text_width + 5, y + 5), bg_color, -1)
        # Draw text
        cv2.putText(shadowed, wall_name, (x, y), font, font_scale, color, font_thickness)

        wall_results.append(shadowed)

    # Concatenate horizontally with padding
    if not wall_results:
        return white_canvas

    padded_results = []
    for i, result in enumerate(wall_results):
        padded_results.append(result)
        if i < len(wall_results) - 1:
            padded_results.append(np.ones((result.shape[0], 10, 3), dtype=np.uint8) * 255)

    breakdown = np.hstack(padded_results)

    return breakdown


def main():
    """
    Main visualization pipeline.

    Pipeline flow:
    1. Run ST-RoomNet V3 → get walls + masks
    2. Create pure white canvas
    3. Load shadow EXR → resize to match
    4. Apply shadow with multiple clamp values
    5. Generate comparison grid and per-wall breakdown
    6. Save all outputs
    """
    print("="*80)
    print("SHADOW DARKENING VISUALIZATION - WHITE CANVAS ANALYSIS")
    print("="*80)

    # Define paths
    input_image = '/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/living_simple2.jpg'
    shadow_exr_path = '/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/irradiance_shadow.exr'
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs-visualization')

    print(f"\nInput image: {input_image}")
    print(f"Shadow EXR: {shadow_exr_path}")
    print(f"Output directory: {output_dir}")
    print(f"Clamp values to test: {CLAMP_VALUES}")

    # Verify files exist
    for path in [input_image, shadow_exr_path]:
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

    # Load original image for reference
    original_img = cv2.imread(input_image)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_resized = cv2.resize(original_img, (400, 400), interpolation=cv2.INTER_LINEAR)

    print(f"\nPipeline complete:")
    print(f"  Original image shape: {original_img.shape}")
    print(f"  Number of walls detected: {len(layout.walls)}")
    print(f"  Wall labels: {list(refined_masks.keys())}")

    # Step 2: Create white canvas
    print("\n" + "-"*80)
    print("STEP 2: Creating White Canvas")
    print("-"*80)

    white_canvas = create_white_canvas(shape=(400, 400, 3))
    print(f"White canvas created: shape={white_canvas.shape}, all values=255")

    # Step 3: Load shadow EXR
    print("\n" + "-"*80)
    print("STEP 3: Loading Shadow EXR")
    print("-"*80)

    shadow_exr = load_shadow_exr(shadow_exr_path)

    # Resize shadow to match ST-RoomNet output (400x400)
    print(f"Resizing shadow from {shadow_exr.shape[:2]} to (400, 400)")
    shadow_resized = cv2.resize(shadow_exr, (400, 400), interpolation=cv2.INTER_LINEAR)
    print(f"Shadow resized: {shadow_resized.shape}")

    # Create shadow visualization
    shadow_vis = (np.clip(shadow_resized, 0, 1) * 255).astype(np.uint8)

    # Step 4: Apply shadow with multiple clamp values
    print("\n" + "-"*80)
    print("STEP 4: Applying Shadow with Multiple Clamp Values")
    print("-"*80)

    white_results = {}

    for clamp in CLAMP_VALUES:
        print(f"\n  Testing clamp value: {clamp:.1f}")
        white_with_shadow = white_canvas.copy()

        for wall in layout.walls:
            if not wall.is_visible:
                continue

            # Determine wall label
            wall_id_str = wall.wall_id
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

            # Get refined mask for this wall
            if wall_label not in refined_masks:
                print(f"    Warning: No refined mask for label {wall_label}, skipping")
                continue

            wall_mask = refined_masks[wall_label]

            # Apply shadow to this wall region only
            shadowed_wall = apply_shadow_to_wall(
                canvas_rgb=white_with_shadow,
                shadow_exr=shadow_resized,
                wall_mask=wall_mask,
                ambient_clamp=clamp
            )

            # Composite shadowed wall back (only replace wall pixels)
            white_with_shadow[wall_mask] = shadowed_wall[wall_mask]

        white_results[clamp] = white_with_shadow
        print(f"  Clamp {clamp:.1f} complete")

    print(f"\nShadow application complete for all {len(CLAMP_VALUES)} clamp values")

    # Step 5: Generate comparison outputs
    print("\n" + "-"*80)
    print("STEP 5: Generating Comparison Visualizations")
    print("-"*80)

    print("  Creating comparison grid...")
    comparison_grid = create_clamp_comparison_grid(
        original_resized,
        shadow_vis,
        white_canvas,
        white_results
    )
    print(f"  Comparison grid created: {comparison_grid.shape}")

    print("  Creating per-wall breakdown...")
    per_wall_breakdown = create_per_wall_breakdown(
        layout,
        white_canvas,
        shadow_resized,
        refined_masks,
        ambient_clamp=0.4
    )
    print(f"  Per-wall breakdown created: {per_wall_breakdown.shape}")

    # Step 6: Save all outputs
    print("\n" + "-"*80)
    print("STEP 6: Saving Output Images")
    print("-"*80)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save reference images
    cv2.imwrite(
        os.path.join(output_dir, '00_original.jpg'),
        cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)
    )
    print(f"  Saved: 00_original.jpg")

    cv2.imwrite(
        os.path.join(output_dir, '01_shadow_visualization.jpg'),
        cv2.cvtColor(shadow_vis, cv2.COLOR_RGB2BGR)
    )
    print(f"  Saved: 01_shadow_visualization.jpg")

    cv2.imwrite(
        os.path.join(output_dir, '02_white_canvas_no_shadow.jpg'),
        cv2.cvtColor(white_canvas, cv2.COLOR_RGB2BGR)
    )
    print(f"  Saved: 02_white_canvas_no_shadow.jpg")

    # Save clamp variations
    for i, clamp in enumerate(CLAMP_VALUES):
        filename = f'{3 + i:02d}_white_canvas_clamp_{clamp:.1f}.jpg'
        cv2.imwrite(
            os.path.join(output_dir, filename),
            cv2.cvtColor(white_results[clamp], cv2.COLOR_RGB2BGR)
        )
        print(f"  Saved: {filename}")

    # Save comparison grid
    cv2.imwrite(
        os.path.join(output_dir, '10_comparison_grid.jpg'),
        cv2.cvtColor(comparison_grid, cv2.COLOR_RGB2BGR)
    )
    print(f"  Saved: 10_comparison_grid.jpg")

    # Save per-wall breakdown
    cv2.imwrite(
        os.path.join(output_dir, '11_per_wall_breakdown.jpg'),
        cv2.cvtColor(per_wall_breakdown, cv2.COLOR_RGB2BGR)
    )
    print(f"  Saved: 11_per_wall_breakdown.jpg")

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"All outputs saved to: {output_dir}")
    print("\nKey outputs:")
    print(f"  - 00_original.jpg: Original room image")
    print(f"  - 01_shadow_visualization.jpg: Shadow map visualization")
    print(f"  - 02_white_canvas_no_shadow.jpg: Pure white reference")
    print(f"  - 03-09_white_canvas_clamp_X.X.jpg: Shadow at different clamp values")
    print(f"  - 10_comparison_grid.jpg: Grid showing all variations")
    print(f"  - 11_per_wall_breakdown.jpg: Per-wall shadow visualization")
    print(f"\nTotal output files: 12")


if __name__ == "__main__":
    main()
