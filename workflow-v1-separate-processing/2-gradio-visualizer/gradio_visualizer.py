# File: gradio_visualizer.py
import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io

from load_outputs import load_st_roomnet_output, load_segformer_output, resize_to_match
from simple_mask_combination import combine_masks_simple
from edge_refinement_simple import refine_edges_with_st_boundaries
from create_walls_simple import create_walls_from_masks
from st_roomnet_segformer_integration import process_room_image, get_integration_summary

def fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def visualize_wall_overlay(original_image, walls, alpha=0.5):
    """Overlay wall masks on original image with different colors"""
    overlay = original_image.copy()
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    for idx, wall in enumerate(walls):
        if wall.pixel_mask is not None and wall.is_visible:
            color = colors[idx % len(colors)]
            mask_colored = np.zeros_like(original_image)
            mask_colored[wall.pixel_mask] = color
            overlay = cv2.addWeighted(overlay, 1, mask_colored, alpha, 0)

            # Draw corners if available
            if len(wall.corners_2d) > 0:
                corners_array = np.array(wall.corners_2d, dtype=np.int32)
                cv2.polylines(overlay, [corners_array], True, color, 2)

    return overlay

def create_mask_comparison(st_seg, segformer_mask, combined_masks):
    """Create comparison visualization of masks"""
    fig = Figure(figsize=(12, 8))
    axes = fig.subplots(2, 3)

    # ST-RoomNet segmentation
    axes[0, 0].imshow(st_seg, cmap='tab10', vmin=0, vmax=4)
    axes[0, 0].set_title("ST-RoomNet Planes")
    axes[0, 0].axis('off')

    # SegFormer wall mask
    axes[0, 1].imshow(segformer_mask, cmap='gray')
    axes[0, 1].set_title("SegFormer Walls")
    axes[0, 1].axis('off')

    # All combined walls
    all_combined = np.zeros_like(st_seg, dtype=np.uint8)
    for label, mask in combined_masks.items():
        all_combined[mask] = label
    axes[0, 2].imshow(all_combined, cmap='tab10', vmin=0, vmax=4)
    axes[0, 2].set_title("All Combined Walls")
    axes[0, 2].axis('off')

    # Individual wall masks
    wall_labels = sorted(combined_masks.keys())
    for i, label in enumerate(wall_labels[:3]):
        if i < 3:
            mask = combined_masks[label]
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].set_title(f"Wall {label} (Label: {label})")
            axes[1, i].axis('off')

    fig.tight_layout()
    return fig_to_pil(fig)

def process_integration(st_output_file, segformer_confidence_file,
                       original_image, wall_threshold):
    """
    Main processing function for Gradio interface

    Args:
        st_output_file: Uploaded ST-RoomNet output file
        segformer_confidence_file: Uploaded SegFormer confidence file
        original_image: Original room image
        wall_threshold: Confidence threshold slider value

    Returns:
        Tuple of output images and text summary
    """
    try:
        # Save uploaded files temporarily
        st_path = st_output_file.name if hasattr(st_output_file, 'name') else str(st_output_file)
        seg_path = segformer_confidence_file.name if hasattr(segformer_confidence_file, 'name') else str(segformer_confidence_file)

        # Load outputs
        st_seg = load_st_roomnet_output(st_path)
        seg_conf = load_segformer_output(seg_path)

        # Resize to match
        seg_conf = resize_to_match(seg_conf, st_seg.shape)

        # Convert threshold to float
        threshold = float(wall_threshold)

        # Combine masks
        combined_masks = combine_masks_simple(st_seg, seg_conf,
                                             wall_threshold=threshold,
                                             wall_labels=[1, 2, 3])

        # Refine edges
        refined_masks = {}
        for label, mask in combined_masks.items():
            st_plane = (st_seg == label)
            refined = refine_edges_with_st_boundaries(mask, st_plane)
            refined_masks[label] = refined

        # Create Wall objects
        walls = create_walls_from_masks(refined_masks)

        # Convert original image to numpy if needed
        if isinstance(original_image, Image.Image):
            original_np = np.array(original_image)
        else:
            original_np = original_image

        # Resize original image to 400x400 to match
        if original_np.shape[:2] != (400, 400):
            original_np = cv2.resize(original_np, (400, 400))

        # Create visualizations
        # 1. Wall overlay
        wall_overlay = visualize_wall_overlay(original_np, walls, alpha=0.5)

        # 2. Segformer binary mask
        segformer_binary = (seg_conf > threshold).astype(np.uint8) * 255

        # 3. Mask comparison
        mask_comparison = create_mask_comparison(st_seg, segformer_binary, refined_masks)

        # 4. Individual refined masks
        refined_wall_images = []
        for label in sorted(refined_masks.keys()):
            mask_img = (refined_masks[label].astype(np.uint8) * 255)
            refined_wall_images.append(Image.fromarray(mask_img))

        # Pad with None if less than 3 walls
        while len(refined_wall_images) < 3:
            refined_wall_images.append(None)

        # Generate summary text
        summary_lines = [
            f"### Integration Results",
            f"**Threshold Used:** {threshold}",
            f"**Number of Walls Detected:** {len(walls)}",
            f"",
            f"#### Wall Details:"
        ]

        for wall in walls:
            pixel_count = np.sum(wall.pixel_mask) if wall.pixel_mask is not None else 0
            summary_lines.append(
                f"- **{wall.wall_id}**: {pixel_count} pixels, "
                f"{len(wall.corners_2d)} corners, "
                f"confidence: {wall.confidence:.2f}"
            )

        summary_text = "\n".join(summary_lines)

        return (
            Image.fromarray(original_np),  # Original
            Image.fromarray(wall_overlay),  # Wall overlay
            Image.fromarray(st_seg.astype(np.uint8)),  # ST-RoomNet
            Image.fromarray(segformer_binary),  # SegFormer binary
            mask_comparison,  # Comparison grid
            refined_wall_images[0],  # Wall 1
            refined_wall_images[1],  # Wall 2
            refined_wall_images[2],  # Wall 3
            summary_text
        )

    except Exception as e:
        error_msg = f"### Error\n\n{str(e)}"
        return (None, None, None, None, None, None, None, None, error_msg)

# Create Gradio interface
with gr.Blocks(title="ST-RoomNet + SegFormer Integration Visualizer") as demo:
    gr.Markdown("# ST-RoomNet + SegFormer Integration Visualizer")
    gr.Markdown("""
    Upload ST-RoomNet segmentation output and SegFormer confidence map to visualize
    the integrated wall detection results. Adjust the confidence threshold to see
    how it affects the final wall masks.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Files")
            original_image_input = gr.Image(label="Original Room Image", type="pil")
            st_output_input = gr.File(label="ST-RoomNet Output (.txt)")
            segformer_conf_input = gr.File(label="SegFormer Confidence (.npy or image)")
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                label="Wall Confidence Threshold"
            )
            process_btn = gr.Button("Process Integration", variant="primary")

        with gr.Column():
            gr.Markdown("### Quick View")
            original_output = gr.Image(label="Original Image")
            wall_overlay_output = gr.Image(label="Wall Overlay (Colored)")

    gr.Markdown("## Model Outputs")
    with gr.Row():
        st_output = gr.Image(label="ST-RoomNet Segmentation")
        segformer_output = gr.Image(label="SegFormer Binary Mask")
        comparison_output = gr.Image(label="Mask Comparison Grid")

    gr.Markdown("## Individual Wall Masks (Refined)")
    with gr.Row():
        wall1_output = gr.Image(label="Wall 1 (Left)")
        wall2_output = gr.Image(label="Wall 2 (Front)")
        wall3_output = gr.Image(label="Wall 3 (Right)")

    gr.Markdown("## Integration Summary")
    summary_output = gr.Markdown()

    # Connect button to processing function
    process_btn.click(
        fn=process_integration,
        inputs=[st_output_input, segformer_conf_input, original_image_input, threshold_slider],
        outputs=[
            original_output,
            wall_overlay_output,
            st_output,
            segformer_output,
            comparison_output,
            wall1_output,
            wall2_output,
            wall3_output,
            summary_output
        ]
    )

if __name__ == "__main__":
    demo.launch()
