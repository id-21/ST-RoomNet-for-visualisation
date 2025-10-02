# File: compare_pipelines.py
"""
Debugging script to compare old (file-based) vs new (direct inference) pipeline outputs.

This helps identify where the outputs diverge between the two approaches.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import cv2
from PIL import Image

# Old pipeline imports
from load_outputs import load_st_roomnet_output, load_segformer_output, resize_to_match
from simple_mask_combination import combine_masks_simple
from edge_refinement_simple import refine_edges_with_st_boundaries
from create_walls_simple import create_walls_from_masks

# New pipeline imports
from end_to_end_pipeline import process_room_image_from_array, initialize_models


def compare_arrays(arr1, arr2, name):
    """Compare two numpy arrays and print statistics"""
    print(f"\n{name}:")
    print(f"  Shape: {arr1.shape} vs {arr2.shape}")
    print(f"  Dtype: {arr1.dtype} vs {arr2.dtype}")

    if arr1.shape == arr2.shape:
        # Pixel-wise comparison
        diff = arr1.astype(np.float32) - arr2.astype(np.float32)
        abs_diff = np.abs(diff)

        print(f"  Value range OLD: [{arr1.min():.3f}, {arr1.max():.3f}]")
        print(f"  Value range NEW: [{arr2.min():.3f}, {arr2.max():.3f}]")
        print(f"  Max absolute difference: {abs_diff.max():.3f}")
        print(f"  Mean absolute difference: {abs_diff.mean():.3f}")
        print(f"  Identical pixels: {np.sum(diff == 0)} / {diff.size} ({100 * np.sum(diff == 0) / diff.size:.1f}%)")

        if abs_diff.max() > 0:
            print(f"  Different pixels: {np.sum(diff != 0)} ({100 * np.sum(diff != 0) / diff.size:.1f}%)")
    else:
        print(f"  ⚠️ Shapes don't match - cannot compare values")


def compare_masks(masks1, masks2, name):
    """Compare two mask dictionaries"""
    print(f"\n{name}:")
    print(f"  Labels OLD: {sorted(masks1.keys())}")
    print(f"  Labels NEW: {sorted(masks2.keys())}")

    common_labels = set(masks1.keys()) & set(masks2.keys())
    print(f"  Common labels: {sorted(common_labels)}")

    for label in sorted(common_labels):
        mask1 = masks1[label]
        mask2 = masks2[label]

        # Convert to same dtype for comparison
        mask1_bool = mask1 > 0 if mask1.dtype != bool else mask1
        mask2_bool = mask2 > 0 if mask2.dtype != bool else mask2

        pixel_count1 = np.sum(mask1_bool)
        pixel_count2 = np.sum(mask2_bool)

        if mask1_bool.shape == mask2_bool.shape:
            identical = np.sum(mask1_bool == mask2_bool)
            total = mask1_bool.size

            print(f"\n  Label {label}:")
            print(f"    Pixels OLD: {pixel_count1}")
            print(f"    Pixels NEW: {pixel_count2}")
            print(f"    Identical: {identical} / {total} ({100 * identical / total:.1f}%)")
        else:
            print(f"\n  Label {label}:")
            print(f"    ⚠️ Shape mismatch: {mask1_bool.shape} vs {mask2_bool.shape}")


def main():
    print("=" * 70)
    print("Pipeline Comparison: Old (File-based) vs New (Direct Inference)")
    print("=" * 70)

    # Paths
    image_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'Integration Work',
        'images',
        'room1.jpeg'
    )

    st_output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'Integration Work',
        'outputs_for_ref',
        'st_roomnet_output.txt'
    )

    segformer_confidence_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'Integration Work',
        'outputs_for_ref',
        'segformer_confidence_room1.npy'
    )

    threshold = 0.4
    wall_labels = [1, 2, 3]

    # Check files exist
    print("\nChecking files...")
    for path, name in [
        (image_path, "Room image"),
        (st_output_path, "ST-RoomNet output"),
        (segformer_confidence_path, "SegFormer confidence")
    ]:
        if os.path.exists(path):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} - NOT FOUND")
            print(f"    Path: {path}")
            return

    # === OLD PIPELINE (File-based) ===
    print("\n" + "=" * 70)
    print("Running OLD Pipeline (File-based)")
    print("=" * 70)

    # Load pre-computed outputs
    st_seg_old = load_st_roomnet_output(st_output_path)
    seg_conf_old = load_segformer_output(segformer_confidence_path)
    seg_conf_old = resize_to_match(seg_conf_old, st_seg_old.shape)

    # Combine masks
    combined_masks_old = combine_masks_simple(st_seg_old, seg_conf_old,
                                             wall_threshold=threshold,
                                             wall_labels=wall_labels)

    # Refine edges
    refined_masks_old = {}
    for label, mask in combined_masks_old.items():
        st_plane = (st_seg_old == label)
        refined = refine_edges_with_st_boundaries(mask, st_plane)
        refined_masks_old[label] = refined

    # Create walls
    walls_old = create_walls_from_masks(refined_masks_old)

    print(f"\nOLD Pipeline Results:")
    print(f"  ST-RoomNet shape: {st_seg_old.shape}")
    print(f"  SegFormer shape: {seg_conf_old.shape}")
    print(f"  Combined masks: {list(combined_masks_old.keys())}")
    print(f"  Walls created: {len(walls_old)}")
    for wall in walls_old:
        pixel_count = np.sum(wall.pixel_mask) if wall.pixel_mask is not None else 0
        print(f"    - {wall.wall_id}: {pixel_count} pixels")

    # === NEW PIPELINE (Direct Inference) ===
    print("\n" + "=" * 70)
    print("Running NEW Pipeline (Direct Inference)")
    print("=" * 70)

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    # Initialize models
    print("Initializing models...")
    initialize_models()

    # Run new pipeline
    layout, st_seg_new, seg_conf_new, refined_masks_new = process_room_image_from_array(
        image_array,
        wall_threshold=threshold,
        wall_labels=wall_labels,
        use_cached_models=True
    )

    print(f"\nNEW Pipeline Results:")
    print(f"  ST-RoomNet shape: {st_seg_new.shape}")
    print(f"  SegFormer shape: {seg_conf_new.shape}")
    print(f"  Refined masks: {list(refined_masks_new.keys())}")
    print(f"  Walls created: {len(layout.walls)}")
    for wall in layout.walls:
        pixel_count = np.sum(wall.pixel_mask) if wall.pixel_mask is not None else 0
        print(f"    - {wall.wall_id}: {pixel_count} pixels, has_normal: {wall.normal_vector is not None}")

    # === COMPARISON ===
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # Compare ST-RoomNet segmentation
    compare_arrays(st_seg_old, st_seg_new, "ST-RoomNet Segmentation")

    # Compare SegFormer confidence
    compare_arrays(seg_conf_old, seg_conf_new, "SegFormer Confidence Map")

    # Compare refined masks
    compare_masks(refined_masks_old, refined_masks_new, "Refined Masks")

    # Save comparison visualizations
    print("\n" + "=" * 70)
    print("Saving Comparison Visualizations")
    print("=" * 70)

    output_dir = os.path.join(
        os.path.dirname(__file__),
        'comparison_outputs'
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save ST-RoomNet comparison
    cv2.imwrite(os.path.join(output_dir, 'st_seg_old.png'),
                (st_seg_old * 50).astype(np.uint8))  # Scale for visibility
    cv2.imwrite(os.path.join(output_dir, 'st_seg_new.png'),
                (st_seg_new * 50).astype(np.uint8))

    # Save SegFormer comparison
    cv2.imwrite(os.path.join(output_dir, 'segformer_old.png'),
                (seg_conf_old * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, 'segformer_new.png'),
                (seg_conf_new * 255).astype(np.uint8))

    # Save difference maps
    if st_seg_old.shape == st_seg_new.shape:
        st_diff = np.abs(st_seg_old.astype(np.float32) - st_seg_new.astype(np.float32))
        cv2.imwrite(os.path.join(output_dir, 'st_seg_diff.png'),
                    (st_diff * 255).astype(np.uint8))

    if seg_conf_old.shape == seg_conf_new.shape:
        seg_diff = np.abs(seg_conf_old - seg_conf_new)
        cv2.imwrite(os.path.join(output_dir, 'segformer_diff.png'),
                    (seg_diff * 255).astype(np.uint8))

    # Save mask comparisons
    for label in sorted(set(refined_masks_old.keys()) & set(refined_masks_new.keys())):
        mask_old = refined_masks_old[label]
        mask_new = refined_masks_new[label]

        if mask_old.shape == mask_new.shape:
            mask_old_uint8 = (mask_old.astype(np.uint8) * 255)
            mask_new_uint8 = (mask_new.astype(np.uint8) * 255)

            cv2.imwrite(os.path.join(output_dir, f'mask_label{label}_old.png'), mask_old_uint8)
            cv2.imwrite(os.path.join(output_dir, f'mask_label{label}_new.png'), mask_new_uint8)

            # XOR to show differences
            diff = np.logical_xor(mask_old > 0, mask_new > 0).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(output_dir, f'mask_label{label}_diff.png'), diff)

    print(f"\n  Saved comparison images to: {output_dir}")

    print("\n" + "=" * 70)
    print("Comparison Complete!")
    print("=" * 70)

    # Summary
    print("\nSUMMARY:")
    if st_seg_old.shape == st_seg_new.shape:
        st_identical = np.sum(st_seg_old == st_seg_new) / st_seg_old.size
        print(f"  ST-RoomNet: {100 * st_identical:.1f}% identical")

    if seg_conf_old.shape == seg_conf_new.shape:
        seg_diff_mean = np.mean(np.abs(seg_conf_old - seg_conf_new))
        print(f"  SegFormer: Mean abs diff = {seg_diff_mean:.4f}")

    print(f"  Number of walls: {len(walls_old)} (old) vs {len(layout.walls)} (new)")


if __name__ == "__main__":
    main()
