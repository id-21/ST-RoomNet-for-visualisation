<!-- Original Path: Integration Work/phase3_modules/CLAUDE.md -->

<system_context>
This directory contains modular Python implementations for Phase 3 dual-model integration. Each module handles one specific step in the pipeline: loading outputs, combining masks, refining edges, computing normals, creating Wall objects, and applying wallpaper. Modules are designed to be composable and testable independently.
</system_context>

<file_map>
## FILE MAP

### Core Pipeline Modules (use in this order)
- `st_roomnet_model.py` - ST-RoomNet model wrapper with caching, outputs segmentation + theta
- `segformer_model.py` - SegFormer model wrapper with caching, outputs wall confidence
- `load_outputs.py` - Load pre-computed outputs from disk (.txt, .npy, images)
- `simple_mask_combination.py` - AND operation between ST planes and SegFormer confidence
- `edge_refinement_simple.py` - Edge cleanup using ST boundaries as constraints
- `normal_computation.py` - Compute 3D normals from theta via inverse transpose
- `create_walls_simple.py` - Convert masks to Wall objects with corners, confidence
- `wallpaper_application.py` - Apply patterns to walls with perspective transform

### Integration & Visualization
- `end_to_end_pipeline.py` - Complete pipeline orchestration, model initialization
- `st_roomnet_segformer_integration.py` - Legacy integration (use end_to_end_pipeline instead)
- `gradio_visualizer.py` - Basic Gradio interface for testing
- `gradio_visualizer_complete.py` - Full Gradio app with normal vectors and wallpaper

### Testing & Comparison
- `generate_segformer_output.py` - CLI tool to generate .npy confidence maps
- `compare_pipelines.py` - Compare different threshold/refinement strategies
- `test_simple.py` - Basic integration test
- `test_gradio_fix.py` - Gradio debugging script

### Reference Data
- `comparison_outputs/` - Saved outputs for validation
- `README.md` - Setup instructions and usage guide
</file_map>

<patterns>
## PATTERNS

### Module Function Signatures
All core modules follow this pattern:
```python
# Input: numpy arrays (400x400)
# Output: numpy arrays or dicts
# Example: simple_mask_combination.py
def combine_masks_simple(st_segmentation, segformer_confidence,
                         wall_threshold=0.4, wall_labels=[1,2,3]):
    return {label: binary_mask}  # Dict of masks
```

### Model Caching Pattern
Every model wrapper uses global caching:
```python
_cached_model = None
def load_model(use_cache=True):
    global _cached_model
    if use_cache and _cached_model is not None:
        return _cached_model
    # Load model...
    _cached_model = model
    return _cached_model
```
Example: `st_roomnet_model.py:29`, `segformer_model.py:18`

### Path Handling Convention
All modules use `sys.path.append()` to access repo root:
```python
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from room_wall_layout import Wall, RoomWallLayout  # From repo root
```

### Mask Type Conversions
Standard pattern for boolean <-> uint8:
```python
# To uint8: mask_uint8 = mask.astype(np.uint8) * 255
# To bool: mask_bool = mask > 0  (from bool) or mask > 127 (from uint8)
```
Example: `create_walls_simple.py:44-47`

### Label Mapping Constants
ST-RoomNet labels defined in `create_walls_simple.py:11-25`:
- 0=ceiling, 1=left_wall, 2=front_wall, 3=right_wall, 4=floor
- Wall labels default: `[1, 2, 3]`
</patterns>

<critical_notes>
## CRITICAL NOTES

- **Pipeline order matters** - Must follow: ST inference → SegFormer inference → combine → refine → compute normals → create walls. Skipping steps breaks assumptions
- **Array shapes must match** - ST outputs 400×400, SegFormer can vary. Use `resize_to_match()` before combining
- **Threshold tuning is critical** - SegFormer threshold (0.2-0.6) dramatically affects results. No single value works for all images. Start with 0.4
- **Normal vector math** - Normals transform by `(H^-1)^T`, NOT by `H` directly. See `normal_computation.py:43-80`
- **Reference normals coordinate system** - Defined in `normal_computation.py:14-20`. Left wall is `[-1,0,0]`, front is `[0,0,-1]`
- **Minimum visibility threshold** - Walls need >100 pixels to be marked visible (`create_walls_simple.py:72`)
- **Contour approximation epsilon** - Set to 2% of perimeter (`create_walls_simple.py:59`). Lower = more corners, higher = simpler polygons
- **Model weights paths** - Hardcoded absolute paths in model files (`ABS_PATH_TO_WEIGHTS`, `ABS_PATH_TO_IMG`). Update for new environments
- **Edge refinement buffer** - Uses 5-pixel dilation around ST edges (`edge_refinement_simple.py:24`). Adjust for sharper/softer transitions
</critical_notes>

<paved_path>
## PAVED PATH

### Quick Start:
```python
from end_to_end_pipeline import initialize_models, process_room_with_models
initialize_models()
layout = process_room_with_models('room.jpg', wall_threshold=0.4, resize_to=(400,400))
```

### Pipeline Steps (see end_to_end_pipeline.py for full example):
1. Load models: `load_st_roomnet_model()`, `load_segformer_model()`
2. Inference: `inference_from_path()` → st_seg, theta, seg_conf
3. Combine: `combine_masks_simple(st_seg, seg_conf, threshold=0.4)` → masks dict
4. Refine: `refine_edges_with_st_boundaries(mask, st_plane)` → refined masks
5. Normals: `compute_normal_vectors(theta, [1,2,3])` → normals dict
6. Walls: `create_walls_from_masks(refined, normals_dict)` → Wall objects
7. Layout: Create `RoomWallLayout`, assign walls, set model_source

### Visualization & Tools:
```bash
python gradio_visualizer_complete.py  # Full UI
python generate_segformer_output.py room.jpg --output conf.npy --resize 400 400
```
</paved_path>

<troubleshooting>
## COMMON ISSUES

**Import errors**: Check `sys.path.append()` in module header. Assumes 3-level nesting: `repo/Integration Work/phase3_modules/`
**"No walls detected"**: Lower threshold or check ST-RoomNet segmentation succeeded. Debug with `load_outputs.py` directly
**Shape mismatch**: Use `resize_to_match()` before combining. ST=400×400, SegFormer varies
**"Singular matrix" in normals**: Theta is degenerate. Check ST-RoomNet inference succeeded
**All masks empty**: Threshold too high or SegFormer confidence too low. Try 0.2-0.3
**Edges too blocky**: Use `refine_edges_bilateral()` instead of `refine_edges_with_st_boundaries()`
**Wrong normal directions**: Check reference normals in `normal_computation.py:14-20` match your coordinate system
</troubleshooting>
