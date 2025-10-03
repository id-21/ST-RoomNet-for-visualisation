<!-- Original Path: workflow-v2-integrated-pipeline/CLAUDE.md -->

<system_context>
Integrated end-to-end workflow providing a single Gradio interface for complete room wall detection and wallpaper application. This workflow combines ST-RoomNet geometry, SegFormer semantics, normal vector computation, and perspective-correct wallpaper rendering in real-time from a single image upload.
</system_context>

<file_map>
## FILE MAP
- `gradio_visualizer_complete.py` - Main Gradio web app: upload image → run models → display results with wallpaper application
- `modules/` - Complete pipeline implementation (see modules/CLAUDE.md for detailed documentation)
- `README.md` - Quick start guide for running the application
</file_map>

<patterns>
## PATTERNS

**Single-Interface Workflow:**
```
User uploads room image in Gradio
    ↓
gradio_visualizer_complete.py calls process_complete_pipeline()
    ↓
modules/end_to_end_pipeline.py orchestrates:
  1. Initialize models (cached globally)
  2. ST-RoomNet inference → segmentation + theta
  3. SegFormer inference → wall confidence
  4. Mask combination & edge refinement
  5. Normal vector computation from theta
  6. Wall object creation with normals
    ↓
Optional: Apply wallpaper with perspective correction
    ↓
Display 8 outputs: original, overlay, normals, ST-seg, SegFormer, mask grid, wallpaper, summary
```

**Pre-loaded Models:**
Models initialize on app startup (line 266-272 in gradio_visualizer_complete.py):
- ST-RoomNet (TensorFlow) + reference image cached
- SegFormer (PyTorch) + processor cached
- First inference faster due to pre-loading

**Real-time Processing:**
- User uploads image → automatic resize to 400x400
- Both models run in <5 seconds on typical hardware
- Normal vectors computed instantly from theta
- Wallpaper application optional, adds ~1-2 seconds
</patterns>

<critical_notes>
## CRITICAL NOTES
- **Entry point** - Run `python gradio_visualizer_complete.py` (not main.py as README suggests)
- **Model pre-loading** - App loads models on startup; wait for "Models loaded successfully!" message
- **Memory usage** - Both TensorFlow and PyTorch models loaded simultaneously (~2-3GB GPU/RAM)
- **Single-step processing** - Unlike workflow-v1 (separate inference + visualization), this runs everything in one step
- **Hardcoded paths** - modules/st_roomnet_model.py contains absolute paths that must be updated per machine
- **Resolution normalization** - All images auto-resized to 400x400 for consistency
- **Wallpaper filter** - Can apply to specific walls (left/front/right) or all visible walls
- **Progress tracking** - Gradio progress bar shows pipeline stages: initializing → ST-RoomNet → processing → wallpaper → complete
</critical_notes>

<paved_path>
## PAVED PATH

**Running the Application:**
```bash
cd workflow-v2-integrated-pipeline
python gradio_visualizer_complete.py
# Wait for "Models loaded successfully!"
# Open browser to http://localhost:7860
```

**Basic Usage:**
1. Upload room image (any resolution, will be resized to 400x400)
2. Adjust "Wall Confidence Threshold" slider (default: 0.4, range: 0.1-0.9)
3. Click "🚀 Process Image"
4. View 8 output panels showing different stages and results

**With Wallpaper:**
1. Follow steps 1-2 above
2. Upload wallpaper pattern image
3. Check "Apply Wallpaper" checkbox
4. Select walls: "All Walls", "Left Wall", "Front Wall", or "Right Wall"
5. Click "🚀 Process Image"
6. View wallpaper applied with perspective correction in "Wallpaper Applied" panel

**Understanding Outputs:**
- **Original (400x400)** - Input image resized to standard resolution
- **Wall Segmentation** - Colored overlay showing detected wall masks
- **Normal Vectors Visualization** - Yellow arrows showing 3D normal directions
- **ST-RoomNet Segmentation** - Raw plane labels (0-4) from geometry model
- **SegFormer Binary Mask** - Wall/non-wall classification at current threshold
- **Mask Grid** - Individual wall masks side-by-side for comparison
- **Wallpaper Applied** - Final result with perspective-correct pattern (if enabled)
- **Summary** - Detailed statistics: pixel counts, confidence scores, normal vectors, corners

**Adjusting Parameters:**
- Lower threshold (0.1-0.3): More aggressive wall detection, may include non-wall surfaces
- Higher threshold (0.5-0.9): More conservative, may miss partial walls
- Default 0.4 works well for most indoor scenes

**Troubleshooting:**
- "Models loading..." takes long → Normal on first run, models download from HuggingFace
- No walls detected → Lower threshold or check if image contains clear wall surfaces
- Wallpaper looks distorted → Normal vectors may be incorrect, check summary for magnitude ≈ 1.0
- Out of memory → Close other applications, models require ~2-3GB

**Programmatic Usage:**
See `modules/CLAUDE.md` for using pipeline components in custom scripts without Gradio interface.
</paved_path>