# File: gradio_visualizer_complete.py
"""
Complete Gradio Interface for ST-RoomNet + SegFormer Integration with Wallpaper Application

Single interface that:
1. Takes room image upload
2. Runs both models directly
3. Computes normal vectors
4. Allows wallpaper upload and application
5. Displays results with perspective-correct wallpaper
"""

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

# Import pipeline modules
from end_to_end_pipeline import process_room_image_from_array, initialize_models, get_summary
from wallpaper_application import WallpaperPattern, apply_wallpaper_to_multiple_walls


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


def visualize_normals(original_image, walls):
    """Visualize normal vectors as arrows on the image"""
    overlay = original_image.copy()

    for wall in walls:
        if wall.normal_vector is None or not wall.is_visible:
            continue

        # Get wall center
        if wall.pixel_mask is not None:
            y_coords, x_coords = np.where(wall.pixel_mask)
            if len(y_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))

                # Project 3D normal to 2D for visualization
                # Simple projection: use x and z components
                nx, ny, nz = wall.normal_vector
                arrow_scale = 50

                # Arrow endpoint (simplified 2D projection)
                end_x = int(center_x + nx * arrow_scale)
                end_y = int(center_y + nz * arrow_scale)

                # Draw arrow
                cv2.arrowedLine(overlay, (center_x, center_y), (end_x, end_y),
                              (0, 255, 255), 3, tipLength=0.3)

                # Draw label
                label = f"n=[{nx:.2f},{ny:.2f},{nz:.2f}]"
                cv2.putText(overlay, label, (center_x + 10, center_y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return overlay


def create_mask_grid(st_seg, refined_masks):
    """Create grid visualization of masks"""
    fig = Figure(figsize=(12, 8))

    # Determine grid size based on number of masks
    n_masks = len(refined_masks) + 1  # +1 for ST-RoomNet
    n_cols = 3
    n_rows = (n_masks + n_cols - 1) // n_cols

    axes = fig.subplots(n_rows, n_cols)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    # ST-RoomNet segmentation
    axes_flat[0].imshow(st_seg, cmap='tab10', vmin=0, vmax=4)
    axes_flat[0].set_title("ST-RoomNet Planes")
    axes_flat[0].axis('off')

    # Individual refined walls
    for idx, (label, mask) in enumerate(sorted(refined_masks.items()), start=1):
        if idx < len(axes_flat):
            axes_flat[idx].imshow(mask, cmap='gray')
            wall_names = {0: "Ceiling", 1: "Left", 2: "Front", 3: "Right", 4: "Floor"}
            axes_flat[idx].set_title(f"{wall_names.get(label, f'Wall {label}')}")
            axes_flat[idx].axis('off')

    # Hide unused subplots
    for idx in range(n_masks, len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.tight_layout()
    return fig_to_pil(fig)


def process_complete_pipeline(room_image, wallpaper_image, wall_threshold,
                              apply_wallpaper_flag, wallpaper_walls_filter,
                              progress=gr.Progress()):
    """
    Complete pipeline: run models, compute normals, apply wallpaper.

    Args:
        room_image: PIL Image of room
        wallpaper_image: PIL Image of wallpaper pattern (optional)
        wall_threshold: Confidence threshold slider value
        apply_wallpaper_flag: Whether to apply wallpaper
        wallpaper_walls_filter: Which walls to apply wallpaper to
        progress: Gradio progress tracker

    Returns:
        Tuple of output images and summary text
    """
    try:
        if room_image is None:
            return (None, None, None, None, None, "Please upload a room image.")

        progress(0.1, desc="Initializing models...")

        # Convert PIL to numpy array (RGB)
        room_array = np.array(room_image)

        progress(0.2, desc="Running ST-RoomNet...")

        # Run complete pipeline
        layout, st_seg, seg_conf, refined_masks = process_room_image_from_array(
            room_array,
            wall_threshold=wall_threshold,
            wall_labels=[1, 2, 3],  # Walls only
            use_cached_models=True
        )

        progress(0.6, desc="Processing results...")

        # Resize original to 400x400 to match model output
        original_resized = cv2.resize(room_array, (400, 400))

        # Create visualizations
        # 1. Wall overlay
        wall_overlay = visualize_wall_overlay(original_resized, layout.walls, alpha=0.5)

        # 2. Normal vectors visualization
        normals_viz = visualize_normals(original_resized, layout.walls)

        # 3. Mask grid (now using actual ST-RoomNet segmentation)
        mask_grid = create_mask_grid(st_seg, refined_masks)

        # 4. SegFormer binary mask (for comparison with old visualizer)
        segformer_binary = (seg_conf > wall_threshold).astype(np.uint8) * 255

        # 5. Apply wallpaper if requested
        wallpaper_result = None
        if apply_wallpaper_flag and wallpaper_image is not None:
            progress(0.7, desc="Applying wallpaper...")

            # Convert wallpaper to numpy array
            wallpaper_array = np.array(wallpaper_image)
            wallpaper_pattern = WallpaperPattern(image=wallpaper_array)

            # Determine which walls to apply to
            wall_filter = None
            if wallpaper_walls_filter != "All Walls":
                wall_filter = [wallpaper_walls_filter.lower().replace(" ", "_")]

            # Apply wallpaper
            wallpaper_result = apply_wallpaper_to_multiple_walls(
                wallpaper_pattern,
                layout.walls,
                original_resized,
                wall_filter=wall_filter
            )

        progress(0.9, desc="Generating summary...")

        # Generate summary
        summary = get_summary(layout)

        summary_lines = [
            "# Integration Results",
            "",
            f"**Model:** {summary['model_source']}",
            f"**Image Size:** {summary['image_dimensions']}",
            f"**Walls Detected:** {summary['num_walls']}",
            "",
            "## Wall Details",
            ""
        ]

        for wall_info in summary['walls']:
            summary_lines.append(f"### {wall_info['wall_id']}")
            summary_lines.append(f"- **Type:** {wall_info['surface_type']}")
            summary_lines.append(f"- **Visible:** {wall_info['is_visible']}")
            summary_lines.append(f"- **Confidence:** {wall_info['confidence']:.3f}")
            summary_lines.append(f"- **Pixels:** {wall_info['pixel_count']}")
            summary_lines.append(f"- **Corners:** {wall_info['num_corners']}")

            if wall_info['has_normal']:
                normal = wall_info['normal_vector']
                magnitude = wall_info['normal_magnitude']
                summary_lines.append(f"- **Normal:** [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
                summary_lines.append(f"- **Magnitude:** {magnitude:.6f}")
            else:
                summary_lines.append(f"- **Normal:** Not computed")

            summary_lines.append("")

        summary_text = "\n".join(summary_lines)

        progress(1.0, desc="Complete!")

        return (
            Image.fromarray(original_resized),
            Image.fromarray(wall_overlay),
            Image.fromarray(normals_viz),
            Image.fromarray(st_seg.astype(np.uint8)),  # ST-RoomNet segmentation
            Image.fromarray(segformer_binary),  # SegFormer binary mask
            mask_grid,
            Image.fromarray(wallpaper_result) if wallpaper_result is not None else None,
            summary_text
        )

    except Exception as e:
        import traceback
        error_msg = f"### Error\n\n```\n{str(e)}\n\n{traceback.format_exc()}\n```"
        return (None, None, None, None, None, None, None, error_msg)


# Pre-initialize models when the app starts
print("Pre-loading models...")
try:
    initialize_models()
    print("Models loaded successfully!")
except Exception as e:
    print(f"Warning: Could not pre-load models: {e}")
    print("Models will be loaded on first inference.")


# Create Gradio interface
with gr.Blocks(title="ST-RoomNet + SegFormer Complete Pipeline") as demo:
    gr.Markdown("# 🏠 ST-RoomNet + SegFormer Complete Pipeline")
    gr.Markdown("""
    Upload a room image to automatically:
    1. Detect wall geometry (ST-RoomNet)
    2. Segment wall surfaces (SegFormer)
    3. Compute 3D normal vectors
    4. Apply wallpaper with perspective correction
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input")
            room_image_input = gr.Image(label="Room Image", type="pil")
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                label="Wall Confidence Threshold"
            )

            gr.Markdown("### Wallpaper Application (Optional)")
            wallpaper_image_input = gr.Image(label="Wallpaper Pattern", type="pil")
            apply_wallpaper_checkbox = gr.Checkbox(label="Apply Wallpaper", value=False)
            wallpaper_walls_dropdown = gr.Dropdown(
                choices=["All Walls", "Left Wall", "Front Wall", "Right Wall"],
                value="All Walls",
                label="Apply To"
            )

            process_btn = gr.Button("🚀 Process Image", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("### Results")
            original_output = gr.Image(label="Original (400x400)")
            wall_overlay_output = gr.Image(label="Wall Segmentation")

    gr.Markdown("## Model Outputs")

    with gr.Row():
        st_output = gr.Image(label="ST-RoomNet Segmentation")
        segformer_output = gr.Image(label="SegFormer Binary Mask")
        normals_output = gr.Image(label="Normal Vectors Visualization")

    gr.Markdown("## Combined Results")

    with gr.Row():
        masks_output = gr.Image(label="Mask Grid")
        wallpaper_output = gr.Image(label="Wallpaper Applied")

    gr.Markdown("## Summary")
    summary_output = gr.Markdown()

    # Connect button
    process_btn.click(
        fn=process_complete_pipeline,
        inputs=[
            room_image_input,
            wallpaper_image_input,
            threshold_slider,
            apply_wallpaper_checkbox,
            wallpaper_walls_dropdown
        ],
        outputs=[
            original_output,
            wall_overlay_output,
            normals_output,
            st_output,
            segformer_output,
            masks_output,
            wallpaper_output,
            summary_output
        ]
    )


if __name__ == "__main__":
    demo.launch()
