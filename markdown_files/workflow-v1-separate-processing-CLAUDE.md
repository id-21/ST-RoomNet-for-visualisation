<!-- Original Path: workflow-v1-separate-processing/CLAUDE.md -->

<system_context>
Two-step workflow for room wall detection: (1) run model inference to generate outputs, (2) visualize and integrate results via Gradio interface. This separation enables reusing model outputs for iterative visualization development without recomputing inference.
</system_context>

<file_map>
## FILE MAP
- `1-model-inference/` - Standalone inference scripts for ST-RoomNet and SegFormer (see 1-model-inference/CLAUDE.md)
- `2-gradio-visualizer/` - Gradio web interface for visualization and mask integration (see 2-gradio-visualizer/CLAUDE.md)
- `README.md` - Quick start guide for running each workflow step
</file_map>

<patterns>
## PATTERNS

**Two-Step Workflow:**
```
Step 1: Model Inference (1-model-inference/)
  Input: room image
  Output: ST-RoomNet segmentation (.txt) + SegFormer confidence (.npy)

Step 2: Visualization (2-gradio-visualizer/)
  Input: model outputs from Step 1
  Output: refined wall masks, RoomWallLayout object, interactive visualizations
```

**Data Flow:**
1. `1-model-inference/run_st_roomnet.py` → outputs/st_roomnet_output.txt (400x400, labels 0-4)
2. `1-model-inference/run_segformer.py` → outputs/wall_confidence.npy (HxW, values 0-1)
3. Upload both outputs to `2-gradio-visualizer/gradio_visualizer.py` for integration and visualization

**Key Integration:**
ST-RoomNet provides geometric structure (room layout planes), SegFormer provides semantic validation (wall confidence). Combined via AND operation with adjustable threshold to produce refined wall masks.
</patterns>

<critical_notes>
## CRITICAL NOTES
- **Decoupled execution** - Steps 1 and 2 run independently, enabling fast iteration on visualization without re-inference
- **Standard resolution** - All processing normalized to 400x400 (ST-RoomNet output size)
- **Output format** - Step 1 saves: .txt (ST-RoomNet segmentation), .npy (SegFormer confidence), optional .png (theta params)
- **Shared dependencies** - Both steps require RoomWallLayout classes from parent directory
- **Path management** - Scripts use sys.path.append to access shared modules (../../shared, parent dirs)
- **No end-to-end pipeline** - This workflow deliberately separates inference from visualization for modularity
</critical_notes>

<paved_path>
## PAVED PATH

**Quick Start:**
```bash
# Step 1: Run inference (do this once)
cd 1-model-inference
python run_st_roomnet.py              # Generates outputs/st_roomnet_output.txt
python run_segformer.py room.jpg      # Generates wall confidence map

# Step 2: Visualize (iterate here)
cd ../2-gradio-visualizer
python gradio_visualizer.py           # Upload outputs from Step 1
```

**For development:**
- Modify inference: Work in `1-model-inference/`, see subdirectory CLAUDE.md
- Modify visualization/integration: Work in `2-gradio-visualizer/`, see subdirectory CLAUDE.md
- Add new models: Add inference script to Step 1, integrate in Step 2's mask combination logic

**Typical use case:**
Run Step 1 once per image, then experiment with different visualization parameters, thresholds, and refinement methods in Step 2 without recomputing expensive model inference.
</paved_path>