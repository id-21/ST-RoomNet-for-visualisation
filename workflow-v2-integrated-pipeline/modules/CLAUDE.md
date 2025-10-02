<system_context>
This directory contains modular components for an integrated end-to-end pipeline that combines ST-RoomNet geometric segmentation with SegFormer semantic validation, computes 3D normal vectors, and applies wallpaper with perspective correction. All modules are designed to work together in a complete pipeline from image input to wallpaper application.
</system_context>

<file_map>
## FILE MAP

### Core Pipeline
- `end_to_end_pipeline.py` - Complete pipeline orchestrator: loads models, runs inference, combines masks, computes normals, creates RoomWallLayout
- `st_roomnet_model.py` - ST-RoomNet model wrapper with dual output (segmentation + theta parameters)
- `segformer_model.py` - SegFormer model wrapper for wall confidence map inference

### Data Structures
- `room_wall_layout.py` - Core data classes: Wall (with corners, mask, normal) and RoomWallLayout (container)

### Mask Processing (from workflow-v1)
- `simple_mask_combination.py` - AND-based mask combination: ST-RoomNet planes & SegFormer confidence thresholding
- `edge_refinement_simple.py` - Edge refinement using ST-RoomNet boundaries as constraints
- `create_walls_simple.py` - Creates Wall objects from masks with contours and corners

### Normal Vector Computation
- `normal_computation.py` - Computes 3D normal vectors from theta using inverse-transpose homography transformation
- `spatial_transformer_fixed.py` - ProjectiveTransformerLayer for ST-RoomNet spatial transformation

### Wallpaper Application
- `wallpaper_application.py` - Applies wallpaper patterns with perspective correction based on normal vectors

### Utilities
- `compare_pipelines.py` - Pipeline comparison and validation script
- `generate_segformer_output.py` - Standalone SegFormer inference for generating confidence maps
- `test_simple.py`, `test_gradio_fix.py` - Test scripts for module validation

### Outputs
- `comparison_outputs/` - Visual comparison outputs (old/new/diff for debugging)
- `README.md` - Setup instructions and usage guide
</file_map>

<patterns>
## PATTERNS

**End-to-End Pipeline Flow:**
```python
# From end_to_end_pipeline.py
1. initialize_models() - Cache ST-RoomNet & SegFormer models
2. ST-RoomNet inference → (segmentation, theta, original_rgb)
3. SegFormer inference → wall_confidence
4. combine_masks_simple() → combined_masks
5. refine_edges_with_st_boundaries() → refined_masks
6. compute_normal_vectors(theta) → normals_dict
7. create_walls_from_masks_with_normals() → Wall objects with normals
8. RoomWallLayout() → standardized output
```

**Model Loading Pattern:**
Both models use global caching to avoid reloading:
- `load_st_roomnet_model(use_cache=True)` - Caches TensorFlow model + reference image
- `load_segformer_model(use_cache=True)` - Caches PyTorch model + processor
Example: `st_roomnet_model.py:59-136`, `segformer_model.py:22-56`

**Dual Input Support:**
All inference functions accept both file paths and numpy arrays:
- `inference_from_path(image_path)` - Load from disk
- `inference_from_array(image_array)` - Use in-memory array
- `inference_from_pil(pil_image)` - SegFormer only
Example: `st_roomnet_model.py:230-291`, `segformer_model.py:103-206`

**Normal Vector Transformation:**
Normals transform via inverse-transpose of homography matrix:
- Theta (8 params) → H (3x3 homography)
- Normal: n' = (H^-1)^T * n_reference
- Reference normals: left=[-1,0,0], front=[0,0,-1], right=[1,0,0]
Example: `normal_computation.py:43-104`

**Wallpaper Application:**
Uses normal vectors for perspective-correct warping:
- `compute_perspective_from_normal()` - Derives transformation from normal
- `apply_wallpaper()` - Warps pattern with perspective + alpha masking
- `apply_wallpaper_to_multiple_walls()` - Batch processing for all walls
Example: `wallpaper_application.py:34-107`, `110-195`
</patterns>

<critical_notes>
## CRITICAL NOTES

- **Model caching** - Global variables `_cached_model` used for performance; models loaded once per session
- **Hardcoded paths** - `ABS_PATH_TO_WEIGHTS` and `ABS_PATH_TO_IMG` in st_roomnet_model.py:25-26 must be updated per machine
- **Dual output architecture** - ST-RoomNet model trained with single output but accessed with dual output (segmentation + theta) via layer extraction
- **Resolution constraint** - All processing happens at 400x400 (ST-RoomNet output size); images auto-resized
- **RGB vs BGR** - ST-RoomNet requires RGB input (cv2.cvtColor needed if using cv2.imread), SegFormer handles RGB
- **Label mapping** - ST-RoomNet: 0=ceiling, 1=left_wall, 2=front_wall, 3=right_wall, 4=floor (consistent across all modules)
- **Normal vector validation** - Should be unit vectors (magnitude ≈ 1.0), adjacent walls perpendicular (dot product ≈ 0)
- **Theta reference image** - ref_img2.png must be 400x400, values 0-255, normalized by dividing by 51.0
- **Module imports** - Uses relative imports (from .module) when used as package, absolute when standalone
- **Minimum visibility** - Walls require >100 pixels to be marked as visible (create_walls_simple.py:72)
</critical_notes>

<paved_path>
## PAVED PATH

**Quick Start - Complete Pipeline:**
```python
from modules.end_to_end_pipeline import process_room_image_complete

# Single function call for entire pipeline
layout, st_seg, seg_conf, refined_masks = process_room_image_complete(
    image_path='room.jpg',
    wall_threshold=0.4,
    use_cached_models=True
)

# Access results
for wall in layout.walls:
    print(f"{wall.wall_id}: normal={wall.normal_vector}, pixels={np.sum(wall.pixel_mask)}")
```

**Apply Wallpaper:**
```python
from modules.wallpaper_application import WallpaperPattern, apply_wallpaper_to_multiple_walls

# Load wallpaper pattern
pattern_img = cv2.imread('pattern.jpg')
wallpaper = WallpaperPattern(image=pattern_img)

# Apply to all walls
result = apply_wallpaper_to_multiple_walls(
    wallpaper_pattern=wallpaper,
    walls=layout.walls,
    original_image=original_img,
    wall_filter=['left_wall', 'front_wall']  # Optional: specific walls only
)
```

**Individual Components:**
```python
# Load models once
from modules.end_to_end_pipeline import initialize_models
st_model, seg_model, seg_processor, seg_device = initialize_models()

# Run ST-RoomNet
from modules.st_roomnet_model import inference_from_path
segmentation, theta, original_rgb = inference_from_path('room.jpg', model=st_model)

# Run SegFormer
from modules.segformer_model import inference_from_path as seg_inference
confidence = seg_inference('room.jpg', model=seg_model, processor=seg_processor, device=seg_device)

# Compute normals
from modules.normal_computation import compute_normal_vectors
normals = compute_normal_vectors(theta, wall_labels=[1, 2, 3])
```

**Adding New Models:**
1. Create wrapper in new file (e.g., `new_model.py`)
2. Follow pattern: `load_model()`, `inference_from_path()`, `inference_from_array()`
3. Add to `end_to_end_pipeline.py` integration
4. Update `RoomWallLayout` if new attributes needed
</paved_path>
