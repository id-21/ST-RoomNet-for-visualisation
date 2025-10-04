# File: test_gradio_fix.py
"""
Quick test to verify the Gradio visualizer fix.

This tests that the pipeline returns proper st_seg and seg_conf arrays.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from PIL import Image

from end_to_end_pipeline import process_room_image_from_array, initialize_models


def main():
    print("=" * 60)
    print("Testing Gradio Visualizer Fix")
    print("=" * 60)

    # Load test image
    image_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'Integration Work',
        'images',
        'room1.jpeg'
    )

    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return

    print(f"\n✓ Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    # Initialize models
    print("\n✓ Initializing models...")
    initialize_models()

    # Run pipeline
    print("\n✓ Running pipeline...")
    layout, st_seg, seg_conf, refined_masks = process_room_image_from_array(
        image_array,
        wall_threshold=0.4,
        wall_labels=[1, 2, 3],
        use_cached_models=True
    )

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Check st_seg
    print("\n1. ST-RoomNet Segmentation (st_seg):")
    if st_seg is not None:
        print(f"   ✓ Not None")
        print(f"   ✓ Shape: {st_seg.shape}")
        print(f"   ✓ Dtype: {st_seg.dtype}")
        print(f"   ✓ Unique values: {np.unique(st_seg)}")
        print(f"   ✓ Value range: [{st_seg.min()}, {st_seg.max()}]")

        # Check if empty (the bug we fixed)
        if np.all(st_seg == 0):
            print("   ❌ EMPTY - All zeros! (This is the bug)")
        else:
            print("   ✓ Contains data - Not empty!")
    else:
        print("   ❌ Is None!")

    # Check seg_conf
    print("\n2. SegFormer Confidence (seg_conf):")
    if seg_conf is not None:
        print(f"   ✓ Not None")
        print(f"   ✓ Shape: {seg_conf.shape}")
        print(f"   ✓ Dtype: {seg_conf.dtype}")
        print(f"   ✓ Value range: [{seg_conf.min():.3f}, {seg_conf.max():.3f}]")
        print(f"   ✓ Mean confidence: {seg_conf.mean():.3f}")
    else:
        print("   ❌ Is None!")

    # Check refined_masks
    print("\n3. Refined Masks:")
    if refined_masks is not None:
        print(f"   ✓ Not None")
        print(f"   ✓ Number of masks: {len(refined_masks)}")
        print(f"   ✓ Labels: {sorted(refined_masks.keys())}")

        for label, mask in sorted(refined_masks.items()):
            pixel_count = np.sum(mask > 0)
            print(f"      - Label {label}: {pixel_count} pixels")
    else:
        print("   ❌ Is None!")

    # Check layout
    print("\n4. RoomWallLayout:")
    if layout is not None:
        print(f"   ✓ Not None")
        print(f"   ✓ Model source: {layout.model_source}")
        print(f"   ✓ Image dimensions: {layout.image_dimensions}")
        print(f"   ✓ Number of walls: {len(layout.walls)}")

        for wall in layout.walls:
            pixel_count = np.sum(wall.pixel_mask) if wall.pixel_mask is not None else 0
            has_normal = wall.normal_vector is not None
            print(f"      - {wall.wall_id}: {pixel_count} pixels, normal: {has_normal}")
    else:
        print("   ❌ Is None!")

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    # Overall status
    all_good = True

    if st_seg is None or np.all(st_seg == 0):
        print("\n❌ FAILED: ST-RoomNet segmentation is empty or None")
        all_good = False
    else:
        print("\n✓ PASSED: ST-RoomNet segmentation contains data")

    if seg_conf is None:
        print("❌ FAILED: SegFormer confidence is None")
        all_good = False
    else:
        print("✓ PASSED: SegFormer confidence map is valid")

    if refined_masks is None or len(refined_masks) == 0:
        print("❌ FAILED: No refined masks")
        all_good = False
    else:
        print("✓ PASSED: Refined masks created")

    if layout is None or len(layout.walls) == 0:
        print("❌ FAILED: No walls created")
        all_good = False
    else:
        print("✓ PASSED: Wall objects created")

    print("\n" + "=" * 60)
    if all_good:
        print("✅ ALL TESTS PASSED - Gradio fix is working!")
    else:
        print("❌ SOME TESTS FAILED - Review output above")
    print("=" * 60)


if __name__ == "__main__":
    main()
