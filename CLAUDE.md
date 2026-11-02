<system_context>
Root directory of ST-RoomNet room layout visualization system. Two complete workflows (v1: separate processing, v2: integrated pipeline) combine ST-RoomNet geometric plane detection with SegFormer semantic segmentation to apply wallpaper patterns with perspective-correct transformation. Also contains legacy development work in "Integration Work" directory showing the evolution to current implementation.
</system_context>

<file_map>
## FILE MAP

### Active Workflows
- `workflow-v1-separate-processing/` - Two-step workflow: run inference → visualize results separately (see workflow-v1-separate-processing/CLAUDE.md)
- `workflow-v2-integrated-pipeline/` - Single Gradio interface for end-to-end processing in real-time (see workflow-v2-integrated-pipeline/CLAUDE.md)
- `Integration Work/` - Phase 3 development history showing dual-model integration evolution (see Integration Work/CLAUDE.md)

### Shared Resources
- `shared/` - Common modules used across workflows: `room_wall_layout.py` (Wall/RoomWallLayout classes), `spatial_transformer_fixed.py` (TF 2.x STN)
- `images/` - Test room images and wallpaper patterns, includes `ref_img2.png` (400×400 reference cuboid for ST-RoomNet)
- `Weight_ST_RroomNet_ConvNext.h5` - Pre-trained ST-RoomNet model weights (111MB, required by all workflows)

### Utilities
- `spatial_transformer_fixed.py` - Root copy of TF 2.x/Keras 3.x spatial transformer layer (also in shared/)
- `claude_summary.sh` - Script to generate `repository_structure.md` and `markdown_files.zip` for documentation
- `README.md` - Quick start guide and directory structure overview

### Deprecated/Legacy (candidate for deletion)
- Root `.py` files (room_wall_layout.py, test*.py, train.py) - Old single-file prototypes superseded by modular workflows
- Summary scripts (flat_summary.sh, gpt_summary.sh, summary.sh) - Use claude_summary.sh instead
- repo_restructure.sh, repository_structure.md, markdown_files.zip - Generated artifacts from repo reorganization
</file_map>

<patterns>
## PATTERNS

### Dual-Model Architecture
All workflows follow this core pattern:
1. **ST-RoomNet** → 5-class segmentation (0=ceiling, 1-3=walls, 4=floor) + 8-param theta (homography)
2. **SegFormer** → 150-class ADE20K semantics (label 0="wall") + per-pixel confidence
3. **Mask Combination** → `final_wall_mask = st_plane_mask & (segformer_mask > threshold)`
4. **Normal Vectors** → Extracted from theta via homography decomposition for 3D orientation

Example: `workflow-v2-integrated-pipeline/modules/end_to_end_pipeline.py`, search:`process_room_with_models`

### Resolution Standardization
All processing normalized to **400×400 pixels**:
- ST-RoomNet outputs 400×400 by design
- Input images auto-resized to match
- Reference image (`ref_img2.png`) must be 400×400
- All masks, visualizations maintained at this size

### Model Caching Pattern
Workflows cache models globally to avoid reloading:
```python
_cached_model = None
def load_model(use_cache=True):
    global _cached_model
    if use_cache and _cached_model is not None:
        return _cached_model
```
Example: `Integration Work/phase3_modules/st_roomnet_model.py:28`
</patterns>

<critical_notes>
## CRITICAL NOTES

- **Choose your workflow** - v1 for iterative development/testing (save outputs, tweak visualization), v2 for demos/production (real-time end-to-end)
- **Model weights location** - All modules expect `Weight_ST_RroomNet_ConvNext.h5` in repo root. SegFormer auto-downloads from HuggingFace
- **Absolute paths issue** - Many modules hardcode paths (ABS_PATH_TO_WEIGHTS, ABS_PATH_TO_IMG). Update when moving repo
- **RGB vs BGR** - ST-RoomNet expects RGB after preprocessing, OpenCV loads BGR. Use `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
- **Theta format** - ST-RoomNet outputs 8 params [θ₀...θ₇]. Append 1.0 and reshape to 3×3: `np.append(theta, 1.0).reshape(3,3)`
- **Shared modules** - Both `shared/` directory and root contain some duplicate files (room_wall_layout.py, spatial_transformer_fixed.py). Workflows use shared/ versions
- **Legacy code cleanup** - Root directory contains deprecated single-file prototypes from early development. See CLAUDE.md for deletion candidates
- **Phase progression** - "Integration Work" preserves Phase 1-3 development history. Current production code in workflow-v1/v2
</critical_notes>

<paved_path>
## PAVED PATH

### Quick Start (Production Use):
```bash
# Option 1: Integrated Pipeline (easiest)
cd workflow-v2-integrated-pipeline
python gradio_visualizer_complete.py
# Open http://localhost:7860, upload room image, adjust threshold, process

# Option 2: Separate Processing (for experimentation)
cd workflow-v1-separate-processing/1-model-inference
python run_st_roomnet.py && python run_segformer.py
cd ../2-gradio-visualizer
python gradio_visualizer.py  # Upload outputs from step 1
```

### Understanding the Codebase:
1. Read this CLAUDE.md first for overview
2. Choose workflow: v1 (experimentation) or v2 (production)
3. Read chosen workflow's CLAUDE.md for detailed usage
4. Review "Integration Work/CLAUDE.md" to understand dual-model rationale and evolution
5. Check `shared/room_wall_layout.py` for core data structures (Wall, RoomWallLayout)

### Adding New Features:
- New model integration: Follow dual-model pattern in workflow-v2/modules/
- New visualization: Modify Gradio visualizers in either workflow
- New wallpaper application: Edit `wallpaper_application.py` in workflow modules

### Troubleshooting:
- "Weight file not found" → Check `Weight_ST_RroomNet_ConvNext.h5` in repo root
- "Reference image not found" → Verify `images/ref_img2.png` exists (400×400)
- Import errors → Check sys.path.append() statements point to correct parent/shared directories
- "No walls detected" → Lower SegFormer threshold (0.2-0.4) or verify input has visible walls
</paved_path>

<workflow_comparison>
## WORKFLOW V1 VS V2

**Workflow V1** (workflow-v1-separate-processing/):
- ✅ Fast iteration on visualization without re-inference
- ✅ Save/reuse model outputs for experiments
- ✅ Good for development, debugging, parameter tuning
- ❌ Two-step process (manual output transfer)

**Workflow V2** (workflow-v2-integrated-pipeline/):
- ✅ Single-click end-to-end processing
- ✅ Real-time results in one interface
- ✅ Pre-loaded models for faster inference
- ✅ Good for demos, production use
- ❌ Must re-run inference to change visualization

**Integration Work/**:
- Documentation of Phase 1-3 development
- Shows evolution from single-model to dual-model approach
- Reference for understanding architectural decisions
- Not intended for active development
</workflow_comparison>

## Setup / rebuild
`venv/` (~2.6 GB) is disposable; `requirements.txt` here is a `pip freeze` of the working venv (Python 3.13; TensorFlow 2.20, Keras 3.11, PyTorch 2.10). `Integration Work/segformer-model/requirements.txt` is a separate, smaller list.
```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```
Weights are not in git. `Weight_ST_RroomNet_ConvNext.h5` (107 MB, required in the repo root) comes from the original ST-RoomNet authors: https://github.com/lab231/ST-RoomNet (see its `weight file.txt`), which links to Google Drive https://drive.google.com/file/d/1KWHofYVsjb0bRi6sVuZtPpanZWCGyWSr/view?usp=sharing . Download it and save it as `Weight_ST_RroomNet_ConvNext.h5` in the repo root. SegFormer weights auto-download from Hugging Face; other weights go in `weights/`.
