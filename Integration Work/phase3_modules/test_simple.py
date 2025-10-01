# File: test_simple.py
"""
Simple test script for Phase 3 integration
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import cv2
from PIL import Image

from st_roomnet_segformer_integration import process_room_image, get_integration_summary

def test_integration():
    """Test the integration with room1.jpeg"""

    print("=" * 60)
    print("Phase 3: ST-RoomNet + SegFormer Integration Test")
    print("=" * 60)

    # File paths (relative to ST-RoomNet-for-visualisation root)
    image_path = '../images/room1.jpeg'
    st_output_path = '../outputs_for_ref/st_roomnet_output.txt'
    segformer_confidence_path = '../outputs_for_ref/segformer_confidence_room1.npy'

    # Check if files exist
    print("\nChecking input files...")
    for path, name in [
        (image_path, "Room image"),
        (st_output_path, "ST-RoomNet output"),
        (segformer_confidence_path, "SegFormer confidence")
    ]:
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: NOT FOUND at {path}")
            if "segformer" in path.lower():
                print(f"    → Run: python generate_segformer_output.py {image_path}")
            return

    # Process integration with different thresholds
    thresholds = [0.3, 0.4, 0.5]

    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing with threshold = {threshold}")
        print("=" * 60)

        try:
            layout, refined_masks, combined_masks = process_room_image(
                image_path=image_path,
                st_output_path=st_output_path,
                segformer_confidence_path=segformer_confidence_path,
                wall_threshold=threshold
            )

            # Get summary
            summary = get_integration_summary(layout)

            print(f"\n📊 Results:")
            print(f"  Model: {summary['model_source']}")
            print(f"  Image size: {summary['image_dimensions']}")
            print(f"  Walls detected: {summary['num_walls']}")

            for wall_info in summary['walls']:
                print(f"\n  🏠 {wall_info['wall_id']}:")
                print(f"     Type: {wall_info['surface_type']}")
                print(f"     Visible: {wall_info['is_visible']}")
                print(f"     Confidence: {wall_info['confidence']:.3f}")
                print(f"     Pixels: {wall_info['pixel_count']}")
                print(f"     Corners: {wall_info['num_corners']}")

            # Analyze results
            print(f"\n  Analysis:")
            wall_dict = {w['wall_id']: w for w in summary['walls']}

            if 'left_wall' in wall_dict:
                left_pixels = wall_dict['left_wall']['pixel_count']
                if left_pixels < 1000:
                    print(f"    ✓ Left wall (window): {left_pixels} pixels - Likely excluded!")
                else:
                    print(f"    ⚠ Left wall (window): {left_pixels} pixels - May need higher threshold")

            if 'front_wall' in wall_dict:
                front_pixels = wall_dict['front_wall']['pixel_count']
                print(f"    ℹ Front wall: {front_pixels} pixels")

            if 'right_wall' in wall_dict:
                right_pixels = wall_dict['right_wall']['pixel_count']
                print(f"    ℹ Right wall: {right_pixels} pixels")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review the results above")
    print("  2. Launch Gradio interface: python gradio_visualizer.py")
    print("  3. Visually inspect the wall masks")
    print("  4. Adjust threshold as needed")

if __name__ == "__main__":
    test_integration()
