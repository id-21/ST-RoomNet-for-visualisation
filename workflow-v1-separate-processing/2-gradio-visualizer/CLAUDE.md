<system_context>
This directory contains a Gradio web interface for visualizing the integration of ST-RoomNet segmentation with SegFormer wall detection. It processes model outputs to create refined wall masks and displays interactive comparisons.
</system_context>

<file_map>
## FILE MAP
- `gradio_visualizer.py` - Main Gradio web app with UI for uploading model outputs and visualizing results
- `st_roomnet_segformer_integration.py` - Core integration pipeline that combines both models into RoomWallLayout
- `load_outputs.py` - Utilities for loading ST-RoomNet segmentation (.txt) and SegFormer confidence maps (.npy/image)
- `simple_mask_combination.py` - AND-based mask combination with SegFormer confidence thresholding
- `edge_refinement_simple.py` - Edge refinement using ST-RoomNet boundaries as constraints
- `create_walls_simple.py` - Converts refined masks to Wall objects with contours and corners
- `normal_computation.py` - Computes 3D normal vectors from ST-RoomNet theta parameters using homography transformation
</file_map>

<patterns>
## PATTERNS
**Integration Pipeline Flow:**
1. Load outputs → 2. Combine masks → 3. Refine edges → 4. Create Wall objects
Example: `st_roomnet_segformer_integration.py:14-89`, function:`process_room_image`

**Mask Processing Pattern:**
- ST-RoomNet provides geometric structure (400x400, labels 0-4)
- SegFormer provides wall confidence (0-1 range)
- Combination uses AND operation: `st_plane & (segformer > threshold)`
Example: `simple_mask_combination.py:5-37`, function:`combine_masks_simple`

**Wall Object Creation:**
All walls follow standardized structure from `room_wall_layout.Wall`:
- `pixel_mask`: Binary mask of wall pixels
- `corners_2d`: 2D polygon corners from contour approximation
- `normal_vector`: 3D normal computed via homography transform
- `confidence`: Ratio of mask pixels to bounding area
Example: `create_walls_simple.py:27-85`, function:`create_wall_from_mask`
</patterns>

<critical_notes>
## CRITICAL NOTES
- **Resolution constraint** - All processing happens at 400x400 (ST-RoomNet output size), images auto-resized
- **Label mapping** - ST-RoomNet labels: 0=ceiling, 1=left_wall, 2=front_wall, 3=right_wall, 4=floor
- **Confidence threshold** - Default 0.4 for SegFormer, adjustable via Gradio slider (0.1-0.9)
- **Edge preservation** - ST-RoomNet boundaries used as constraints during refinement to maintain geometric accuracy
- **Normal vectors** - Computed via inverse-transpose of homography matrix (H^-1)^T * n, requires 8-param theta file
- **Dependency chain** - Must import from parent directory: `room_wall_layout.Wall` and `RoomWallLayout`
- **Minimum visibility** - Walls need >100 pixels to be marked as visible (create_walls_simple.py:72)
</critical_notes>

<paved_path>
## PAVED PATH
**To run the visualizer:**
```bash
python gradio_visualizer.py
# Upload: original image, ST-RoomNet .txt output, SegFormer confidence map
# Adjust threshold slider to refine wall detection
```

**To use integration programmatically:**
```python
from st_roomnet_segformer_integration import process_room_image
layout, refined_masks, combined_masks = process_room_image(
    image_path="room.jpg",
    st_output_path="seg_output.txt",
    segformer_confidence_path="wall_conf.npy",
    theta_path="theta.txt",  # Optional, for normal vectors
    wall_threshold=0.4
)
```

**Adding new refinement methods:**
1. Add function to `edge_refinement_simple.py` following signature: `(combined_mask, st_plane_mask) -> refined_mask`
2. Call in integration pipeline at `st_roomnet_segformer_integration.py:48-54`
3. Existing methods: `refine_edges_with_st_boundaries` (default), `refine_edges_bilateral`, `smooth_contours`
</paved_path>