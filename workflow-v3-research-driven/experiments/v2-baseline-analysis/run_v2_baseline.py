#!/usr/bin/env python3
"""
V2 Baseline Analysis Script

Runs the v2 pipeline on test images and captures all intermediate outputs
for detailed analysis of the edge bleeding bug.
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Add repo root to path to access workflow-v2 modules
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(repo_root, 'workflow-v2-integrated-pipeline'))

from modules.end_to_end_pipeline import initialize_models
from modules.st_roomnet_model import inference_from_path as st_inference
from modules.segformer_model import inference_from_pil as seg_inference
from modules.simple_mask_combination import combine_masks_simple
from modules.edge_refinement_simple import refine_edges_with_st_boundaries


def save_visualization(array, filename, title="", cmap='gray', vmin=None, vmax=None):
    """Save array as image with optional colormap"""
    plt.figure(figsize=(8, 8))
    plt.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def analyze_edge_bleeding(combined_mask, refined_mask, st_plane_mask, segformer_conf, threshold=0.4):
    """
    Analyze the edge bleeding bug by comparing masks at different pipeline stages.

    Returns dict with metrics about incorrect pixels.
    """
    # Find pixels where refinement added pixels that SegFormer rejected
    segformer_wall_mask = segformer_conf > threshold

    # Pixels that are in refined but NOT in combined (added by refinement)
    added_by_refinement = refined_mask & ~combined_mask

    # Of those added pixels, which ones contradict SegFormer?
    # (SegFormer said "not wall" but refinement added them anyway)
    contradicts_segformer = added_by_refinement & ~segformer_wall_mask

    # These pixels should be within ST-RoomNet edge buffer
    st_uint8 = st_plane_mask.astype(np.uint8) * 255
    st_edges = cv2.Canny(st_uint8, 100, 200)
    edge_buffer = cv2.dilate(st_edges, np.ones((5, 5), np.uint8))

    # Distance from ST boundaries
    within_edge_buffer = contradicts_segformer & (edge_buffer > 0)

    metrics = {
        'total_combined_pixels': int(np.sum(combined_mask)),
        'total_refined_pixels': int(np.sum(refined_mask)),
        'pixels_added_by_refinement': int(np.sum(added_by_refinement)),
        'pixels_contradicting_segformer': int(np.sum(contradicts_segformer)),
        'percentage_contradicting': float(np.sum(contradicts_segformer)) / max(np.sum(refined_mask), 1) * 100,
        'within_5px_buffer': int(np.sum(within_edge_buffer)),
        'percentage_in_buffer': float(np.sum(within_edge_buffer)) / max(np.sum(contradicts_segformer), 1) * 100,
    }

    return metrics, contradicts_segformer, edge_buffer


def run_v2_baseline(image_path, output_dir, wall_threshold=0.4):
    """
    Run v2 pipeline and save all intermediate outputs for analysis.
    """
    print(f"\n{'='*60}")
    print(f"Running V2 Baseline Analysis")
    print(f"{'='*60}")
    print(f"Input image: {image_path}")
    print(f"Output directory: {output_dir}")
    print(f"Wall threshold: {wall_threshold}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize models
    print("\n[1/7] Initializing models...")
    st_model, seg_model, seg_processor, seg_device = initialize_models()

    # Run ST-RoomNet
    print("\n[2/7] Running ST-RoomNet inference...")
    st_seg, theta, original_rgb = st_inference(image_path, model=st_model)
    print(f"  ST-RoomNet output shape: {st_seg.shape}")
    print(f"  Theta parameters: {theta}")

    # Save ST-RoomNet segmentation
    save_visualization(st_seg, os.path.join(output_dir, '01_st_roomnet_segmentation.png'),
                      "ST-RoomNet Segmentation (0=ceil, 1=left, 2=front, 3=right, 4=floor)",
                      cmap='tab10', vmin=0, vmax=4)

    # Run SegFormer
    print("\n[3/7] Running SegFormer inference...")
    pil_image = Image.open(image_path).convert('RGB')
    seg_conf = seg_inference(pil_image, model=seg_model, processor=seg_processor, device=seg_device)

    # Resize to match ST-RoomNet
    if seg_conf.shape != st_seg.shape:
        seg_conf = cv2.resize(seg_conf, (st_seg.shape[1], st_seg.shape[0]), interpolation=cv2.INTER_LINEAR)

    print(f"  SegFormer output shape: {seg_conf.shape}")
    print(f"  Confidence range: [{seg_conf.min():.3f}, {seg_conf.max():.3f}]")

    # Save SegFormer confidence map
    save_visualization(seg_conf, os.path.join(output_dir, '02_segformer_confidence.png'),
                      f"SegFormer Wall Confidence (threshold={wall_threshold})",
                      cmap='viridis', vmin=0, vmax=1)

    # Save thresholded SegFormer
    seg_wall_mask = seg_conf > wall_threshold
    save_visualization(seg_wall_mask, os.path.join(output_dir, '02b_segformer_thresholded.png'),
                      f"SegFormer Wall Mask (>{wall_threshold})",
                      cmap='gray')

    # Combine masks
    print("\n[4/7] Combining masks (ST-RoomNet AND SegFormer)...")
    combined_masks = combine_masks_simple(st_seg, seg_conf,
                                         wall_threshold=wall_threshold,
                                         wall_labels=[1, 2, 3])

    # Analyze each wall
    print("\n[5/7] Analyzing edge bleeding for each wall...")
    all_metrics = {}

    for label in [1, 2, 3]:
        wall_name = {1: 'left_wall', 2: 'front_wall', 3: 'right_wall'}[label]
        print(f"\n  Analyzing {wall_name} (label {label})...")

        st_plane = (st_seg == label)
        combined = combined_masks[label]

        # Run edge refinement (the buggy step)
        print(f"    Running edge refinement...")
        refined = refine_edges_with_st_boundaries(combined, st_plane)

        # Analyze the bug
        metrics, contradiction_mask, edge_buffer = analyze_edge_bleeding(
            combined, refined, st_plane, seg_conf, wall_threshold
        )

        all_metrics[wall_name] = metrics

        print(f"    Combined mask pixels: {metrics['total_combined_pixels']}")
        print(f"    Refined mask pixels: {metrics['total_refined_pixels']}")
        print(f"    Pixels added by refinement: {metrics['pixels_added_by_refinement']}")
        print(f"    Pixels contradicting SegFormer: {metrics['pixels_contradicting_segformer']} ({metrics['percentage_contradicting']:.2f}%)")
        print(f"    Within 5px edge buffer: {metrics['within_5px_buffer']} ({metrics['percentage_in_buffer']:.1f}%)")

        # Save visualizations
        label_dir = os.path.join(output_dir, f'wall_{label}_{wall_name}')
        os.makedirs(label_dir, exist_ok=True)

        save_visualization(st_plane, os.path.join(label_dir, 'a_st_plane.png'),
                          f"{wall_name}: ST-RoomNet Plane", cmap='gray')
        save_visualization(combined, os.path.join(label_dir, 'b_combined_mask.png'),
                          f"{wall_name}: Combined (ST AND SegFormer)", cmap='gray')
        save_visualization(refined, os.path.join(label_dir, 'c_refined_mask.png'),
                          f"{wall_name}: After Edge Refinement (BUGGY)", cmap='gray')
        save_visualization(contradiction_mask, os.path.join(label_dir, 'd_contradictions.png'),
                          f"{wall_name}: Pixels Contradicting SegFormer", cmap='Reds')
        save_visualization(edge_buffer, os.path.join(label_dir, 'e_edge_buffer.png'),
                          f"{wall_name}: ST-RoomNet Edge Buffer (5px)", cmap='gray')

        # Create diff visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(combined, cmap='gray')
        axes[0].set_title('Combined (Correct)')
        axes[0].axis('off')

        axes[1].imshow(refined, cmap='gray')
        axes[1].set_title('Refined (Buggy)')
        axes[1].axis('off')

        # Highlight contradictions in red
        diff_viz = np.stack([refined.astype(float)] * 3, axis=-1)
        diff_viz[contradiction_mask, 0] = 1.0  # Red channel
        diff_viz[contradiction_mask, 1] = 0.0
        diff_viz[contradiction_mask, 2] = 0.0
        axes[2].imshow(diff_viz)
        axes[2].set_title(f'Contradictions Highlighted ({metrics["pixels_contradicting_segformer"]} px)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(label_dir, 'f_comparison.png'), dpi=150)
        plt.close()
        print(f"    Saved comparison: {label_dir}/f_comparison.png")

    # Save metrics summary
    print("\n[6/7] Saving metrics summary...")
    metrics_file = os.path.join(output_dir, 'metrics_summary.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("V2 BASELINE EDGE BLEEDING ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Wall threshold: {wall_threshold}\n\n")

        for wall_name, metrics in all_metrics.items():
            f.write(f"\n{wall_name.upper()}:\n")
            f.write("-" * 40 + "\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")

    print(f"  Saved: {metrics_file}")

    # Save original image copy
    print("\n[7/7] Saving original image...")
    cv2.imwrite(os.path.join(output_dir, '00_original_input.png'),
                cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))

    print(f"\n{'='*60}")
    print("V2 BASELINE ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"All outputs saved to: {output_dir}")
    print("\nKey findings:")
    for wall_name, metrics in all_metrics.items():
        if metrics['pixels_contradicting_segformer'] > 0:
            print(f"  ⚠️  {wall_name}: {metrics['pixels_contradicting_segformer']} pixels contradict SegFormer ({metrics['percentage_contradicting']:.2f}%)")

    return all_metrics


if __name__ == "__main__":
    # Get paths relative to workflow-v3 root
    v3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Run on test image
    test_image = os.path.join(v3_root, 'test_images', 'room1.jpeg')
    output_dir = os.path.join(v3_root, 'outputs', 'v2-baseline-room1')

    if not os.path.exists(test_image):
        print(f"ERROR: Test image not found: {test_image}")
        sys.exit(1)

    metrics = run_v2_baseline(test_image, output_dir, wall_threshold=0.4)
