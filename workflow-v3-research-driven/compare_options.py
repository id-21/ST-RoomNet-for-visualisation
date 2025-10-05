#!/usr/bin/env python3
"""
Driver script to compare all three preprocessing options side-by-side.

Reference: workflow-v3-research-driven/planning/fix-resolution-aspect-ratio.md:97-106

Runs the hybrid pipeline with options A, B, and C, then generates
a visual comparison grid for human judgment.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import cv2
import numpy as np
import argparse
from pathlib import Path

from modules.hybrid_pipeline import process_dual_resolution


def create_comparison_grid(images: list, labels: list, output_path: str):
    """
    Create a side-by-side comparison grid.

    Args:
        images: List of images to compare (all must be same size)
        labels: List of labels for each image
        output_path: Path to save comparison grid
    """
    assert len(images) == len(labels), "Number of images and labels must match"
    assert len(images) > 0, "Need at least one image"

    # Get dimensions
    H, W = images[0].shape[:2]

    # Verify all images are same size
    for img in images:
        assert img.shape[:2] == (H, W), f"All images must be same size. Got {img.shape[:2]}, expected {(H, W)}"

    # Create grid: 2 rows × 2 columns
    # Row 1: Input | Option A
    # Row 2: Option B | Option C
    n_cols = 2
    n_rows = (len(images) + n_cols - 1) // n_cols  # Ceiling division

    # Add padding and label space
    label_height = 60
    padding = 20

    # Calculate grid dimensions
    cell_w = W + padding
    cell_h = H + label_height + padding

    grid_w = cell_w * n_cols + padding
    grid_h = cell_h * n_rows + padding

    # Create white canvas
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    # Place images in grid
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // n_cols
        col = idx % n_cols

        # Calculate position
        x = padding + col * cell_w
        y = padding + row * cell_h + label_height

        # Place image
        grid[y:y+H, x:x+W] = img

        # Add label
        label_y = padding + row * cell_h + 40
        cv2.putText(
            grid, label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    # Save grid
    cv2.imwrite(output_path, grid)
    print(f"\n✓ Saved comparison grid: {output_path}")
    print(f"  Grid size: {grid_w}×{grid_h} pixels")


def main():
    """
    Main comparison script.
    """
    parser = argparse.ArgumentParser(description="Compare preprocessing options A/B/C")
    parser.add_argument('--input', type=str, default='images/room1.jpeg',
                       help='Path to input room image')
    parser.add_argument('--wallpaper', type=str,
                       default='images/graphite-clay-repeated-patterns-coordonne.jpg',
                       help='Path to wallpaper pattern image')
    parser.add_argument('--output', type=str,
                       default='workflow-v3-research-driven/outputs/option-comparison',
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.4,
                       help='Wall confidence threshold (default: 0.4)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("HYBRID RESOLUTION PIPELINE - OPTION COMPARISON")
    print("=" * 70)
    print(f"\nTest image: {args.input}")
    print(f"Wallpaper: {args.wallpaper}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output}/")

    # Load input image
    print("\n" + "=" * 70)
    print("Loading input image...")
    print("=" * 70)
    input_img = cv2.imread(args.input)
    if input_img is None:
        raise FileNotFoundError(f"Could not load image: {args.input}")

    H, W = input_img.shape[:2]
    print(f"Loaded: {W}×{H} ({W*H:,} pixels)")

    # Load wallpaper pattern
    wallpaper_pattern = cv2.imread(args.wallpaper)
    if wallpaper_pattern is None:
        print(f"Warning: Could not load wallpaper: {args.wallpaper}")
        print("Proceeding without wallpaper application")
        wallpaper_pattern = None

    # Store results
    outputs = {}
    debug_infos = {}
    labels = ['Input (Original)', 'Option A: Direct Resize', 'Option B: Letterbox', 'Option C: Center Crop']

    # Run each option
    for option in ['A', 'B', 'C']:
        print("\n" + "=" * 70)
        print(f"RUNNING OPTION {option}")
        print("=" * 70)

        try:
            layout, debug_info = process_dual_resolution(
                image_path=args.input,
                option=option,
                wall_threshold=args.threshold,
                wallpaper_pattern=wallpaper_pattern,
                use_cached_models=True
            )

            outputs[option] = debug_info['output_image']
            debug_infos[option] = debug_info

            # Save individual output
            output_path = os.path.join(args.output, f'option_{option}_output.png')
            cv2.imwrite(output_path, debug_info['output_image'])
            print(f"\n✓ Saved: {output_path}")

        except Exception as e:
            print(f"\n✗ Error running option {option}: {e}")
            import traceback
            traceback.print_exc()
            # Use input image as fallback
            outputs[option] = input_img.copy()
            debug_infos[option] = {'error': str(e)}

    # Create comparison grid
    print("\n" + "=" * 70)
    print("CREATING COMPARISON GRID")
    print("=" * 70)

    comparison_images = [input_img, outputs['A'], outputs['B'], outputs['C']]
    comparison_path = os.path.join(args.output, 'comparison.png')

    create_comparison_grid(comparison_images, labels, comparison_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nInput image: {W}×{H} ({W*H:,} pixels)")
    print(f"ST-RoomNet: 400×400 (160,000 pixels)")
    print(f"SegFormer: {W}×{H} ({W*H:,} pixels)")
    print(f"Output: {W}×{H} ({W*H:,} pixels)")
    print(f"\nResolution gain: {(W*H) / (400*400):.1f}x over v2 (400×400)")

    print("\n" + "=" * 70)
    print("OPTION DETAILS")
    print("=" * 70)

    for option in ['A', 'B', 'C']:
        debug = debug_infos[option]
        if 'error' in debug:
            print(f"\nOption {option}: ERROR - {debug['error']}")
        else:
            print(f"\nOption {option}:")
            print(f"  Resolution gain: {debug['resolution_gain']:.1f}x")
            print(f"  Normal vectors computed: {len(debug['normals_dict'])}")
            print(f"  Combined wall pixels:")
            for wall_label, mask in debug['combined_masks'].items():
                wall_names = {1: 'Left', 2: 'Front', 3: 'Right'}
                print(f"    {wall_names.get(wall_label, f'Wall {wall_label}')}: {np.sum(mask):,} pixels")

    print("\n" + "=" * 70)
    print("VISUAL COMPARISON")
    print("=" * 70)
    print(f"\nOpen the comparison grid to visually judge which option produces:")
    print(f"  • Sharpest wallpaper pattern")
    print(f"  • Cleanest edges")
    print(f"  • Best aspect ratio preservation")
    print(f"  • Most realistic perspective")
    print(f"\nComparison grid: {comparison_path}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
