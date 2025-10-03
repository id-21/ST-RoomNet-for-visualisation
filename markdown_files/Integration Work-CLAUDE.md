<!-- Original Path: Integration Work/CLAUDE.md -->

<system_context>
This directory contains the working implementation of a dual-model wallpaper visualization system. It combines ST-RoomNet (geometric plane detection) with SegFormer (semantic pixel classification) to apply wallpaper patterns to room photographs with correct perspective and accurate masking. This is a Phase 3 implementation showing the evolution from initial concept to working integration.
</system_context>

<file_map>
## FILE MAP
- `project-context.md` - Original project vision and approach (September context)
- `updated_project_context.md` - Post-Phase 2 understanding with dual-model strategy
- `Phase 2 Progress.md` - Completed theta extraction and ST-RoomNet dual-output modification
- `st_roomnet_integration_plan.md` - Original integration planning document
- `phase3_simple_implementation.md` - Phase 3 mask combination implementation plan
- `phase3b_normal_vectors.md` - Normal vector computation strategy
- `phase3_modules/` - Working Python modules for complete pipeline
- `segformer-model/` - SegFormer Gradio app (separate git repo)
- `images/` - Test images and reference cuboid (ref_img2.png)
- `outputs_for_ref/` - Reference outputs for validation
</file_map>

<patterns>
## PATTERNS

### Dual-Model Architecture
All processing follows this pattern:
1. ST-RoomNet inference → geometric planes (5 classes) + theta (8 homography params)
2. SegFormer inference → pixel-level semantics (150 ADE20K classes) + confidence
3. Mask combination → `final_mask = st_plane_mask & segformer_wall_mask`
4. Edge refinement → use ST boundaries to smooth SegFormer blockiness
Example: `phase3_modules/end_to_end_pipeline.py`, search:`process_room_with_models`

### Model Caching
Both models use module-level caching to avoid reloading:
```python
_cached_model = None  # Global cache
def load_model(use_cache=True):
    global _cached_model
    if use_cache and _cached_model is not None:
        return _cached_model
```
Example: `phase3_modules/st_roomnet_model.py:28`, `phase3_modules/segformer_model.py:18`

### Wall Object Creation
Standard pattern for creating Wall objects from masks:
- Extract corners via `cv2.findContours()` + `cv2.approxPolyDP()`
- Compute normal vectors from theta using homography decomposition
- Generate pixel masks from combined ST + SegFormer outputs
Example: `phase3_modules/create_walls_simple.py`, search:`create_walls_from_masks`
</patterns>

<critical_notes>
## CRITICAL NOTES

- **Resolution constraint** - Both models process at 400×400. All masks, outputs, and reference images MUST match this size
- **Theta matrix format** - ST-RoomNet outputs 8 params [θ₀...θ₇]. Append 1.0 and reshape to 3×3 for homography: `np.append(theta, 1.0).reshape(3,3)`
- **Label mapping** - ST-RoomNet: 0=ceiling, 1=left wall, 2=front wall, 3=right wall, 4=floor. SegFormer: label 0="wall" in ADE20K (check `processor.config.id2label`)
- **RGB vs BGR** - ST-RoomNet expects RGB after ConvNeXt preprocess. SegFormer PIL expects RGB. OpenCV loads as BGR. ALWAYS verify color space in `cv2.imread()` conversions
- **Absolute paths** - Many modules hardcode absolute paths to weights (`ABS_PATH_TO_WEIGHTS`) and reference image (`ABS_PATH_TO_IMG`). Update these for new environments
- **The key innovation** - No single model can solve both geometric (where are wall planes?) and semantic (which pixels are wall surface?) problems. Combining specialized models achieves better results than either alone
- **Threshold tuning** - SegFormer confidence threshold (0.2-0.6) dramatically affects mask quality. Lower = more coverage, higher = more precision. Tune per image
</critical_notes>

<paved_path>
## PAVED PATH

### To Process a New Room Image:

1. **Prepare models** (one-time setup):
   ```python
   from phase3_modules.end_to_end_pipeline import initialize_models
   initialize_models()  # Caches ST-RoomNet + SegFormer
   ```

2. **Run complete pipeline**:
   ```python
   from phase3_modules.end_to_end_pipeline import process_room_with_models
   layout = process_room_with_models(
       image_path='room.jpg',
       wall_threshold=0.4,  # Tune this
       resize_to=(400, 400)
   )
   ```

3. **Apply wallpaper**:
   ```python
   from phase3_modules.wallpaper_application import apply_wallpaper_to_wall
   result = apply_wallpaper_to_wall(
       original_image=img,
       wall=layout.walls[0],
       wallpaper_pattern=pattern,
       blend_mode='overlay'
   )
   ```

4. **Visualize** (recommended for debugging):
   ```bash
   cd phase3_modules
   python gradio_visualizer_complete.py
   # Open http://localhost:7860
   ```

### To Add a New Model:
1. Create `[model]_model.py` with `load_model()` and `inference()` functions
2. Add model caching pattern (see existing models)
3. Create integration in `end_to_end_pipeline.py`
4. Update `RoomWallLayout.model_source` to track provenance

### Key Files to Read First:
1. `updated_project_context.md` - Understand the dual-model rationale
2. `Phase 2 Progress.md` - See what theta extraction achieved
3. `phase3_modules/README.md` - Learn the Phase 3 module structure
4. `phase3_modules/end_to_end_pipeline.py` - Study the complete flow
</paved_path>

<phase_progression>
## PHASE PROGRESSION

**Phase 1 (Complete)**: Data structure design (`room_wall_layout.py` in repo root)
**Phase 2 (Complete)**: ST-RoomNet dual output extraction (segmentation + theta)
**Phase 3 (Current)**: Dual-model mask combination and integration
**Phase 4 (Planned)**: Wallpaper application with full perspective correction
**Phase 5 (Future)**: Multi-image testing and refinement

Documentation reflects this evolution - earlier docs show single-model thinking, later docs reveal dual-model necessity.
</phase_progression>

<troubleshooting>
## COMMON ISSUES

**"Reference image not found"**: Update `ABS_PATH_TO_IMG` in module files or pass `st_ref_img_path` parameter
**"Model weights not found"**: Update `ABS_PATH_TO_WEIGHTS` or download from repo root
**"Tensor shape mismatch"**: Ensure all images resized to 400×400 before processing
**"No walls detected"**: Lower SegFormer threshold or check if ST-RoomNet segmentation succeeded
**"SSL certificate error"**: ST-RoomNet downloads ConvNeXt weights; workaround in `st_roomnet_model.py:13-14`
**"Import errors"**: Check `sys.path.append()` in module headers - assumes repo root structure
</troubleshooting>
