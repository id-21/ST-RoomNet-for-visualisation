# Shadow Overlay Experiment

**Date:** October 11, 2025
**Research Document:** `research/04-shadow-overlay-integration-research.md`
**Planning Document:** `planning/shadow-overlay-experiment-plan.md`

## Objective

Test integration of RGB→X diffusion model's irradiance (shadow) output with ST-RoomNet V3 wallpaper visualization. Apply realistic shadow overlay to warped wallpaper patterns without combining model inference pipelines.

## Methodology

### Input Data
- **Test Image:** `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/living_simple2.jpg`
- **Wallpaper Pattern:** `graphite-clay-repeated-patterns-coordonne.jpg`
- **Shadow Data:** `irradiance_shadow.exr` (loaded via JPG preview due to OpenCV EXR limitations)

### Pipeline Flow

1. **ST-RoomNet V3 Pipeline**
   - Detected 2 visible walls: left_wall (19,334 pixels), right_wall (36,312 pixels)
   - Generated wall masks with SegFormer semantic validation
   - Computed 3D normal vectors for perspective transformation

2. **Wallpaper Application**
   - Applied wallpaper pattern with perspective correction using normal vectors
   - Output: 400×400 RGB image with wallpaper warped to wall surfaces

3. **Shadow Loading**
   - Attempted EXR loading (OpenCV EXR support disabled)
   - Fallback: Loaded JPG preview and converted to linear RGB via gamma correction (value^2.2)
   - Resized shadow from 682×1024 to 400×400 to match ST-RoomNet output

4. **Shadow Application (Per-Wall)**
   - **Left Wall:**
     - Baseline brightness: 0.383 (98th percentile of wall shadow pixels)
     - Normalized shadow range: [0.400, 1.000] (clamped for ambient light)
   - **Right Wall:**
     - Baseline brightness: 0.305 (98th percentile)
     - Normalized shadow range: [0.400, 1.000]
   - Applied in linear RGB space: `wallpaper_linear × shadow_mask`
   - Converted back to sRGB for display

## Results

### Output Files

1. **`00_original.jpg`** - Original room image (resized to 400×400)
2. **`01_wallpaper_only.jpg`** - Wallpaper applied without shadow overlay
3. **`02_wallpaper_with_shadow.jpg`** - Wallpaper with realistic shadow overlay
4. **`03_comparison.jpg`** - Side-by-side 3-panel comparison
5. **`04_shadow_exr_visualization.jpg`** - Shadow map visualization

### Key Findings

**Successful Integration:**
- ✅ Shadow overlay successfully applied to both detected walls
- ✅ Per-wall baseline normalization preserved relative lighting differences
- ✅ Ambient light clamping (0.4 minimum) prevented unrealistic black shadows
- ✅ Linear RGB processing maintained color fidelity (no visible hue shifts)

**Technical Notes:**
- Used JPG preview as EXR fallback (OpenCV compiled without OpenEXR support)
- JPG → Linear conversion approximation: `(value/255)^2.2`
- Right wall has lower baseline (0.305) → more shadowed overall compared to left wall (0.383)
- Shadow normalization clamps at 0.4-1.0 range for realistic ambient lighting

## Visual Quality Assessment

**Compare `01_wallpaper_only.jpg` vs `02_wallpaper_with_shadow.jpg`:**
- Shadow overlay adds depth and realism to wallpaper
- Darker regions align with shadowed areas in original room photo
- Gradual shadow transitions preserved (no hard edges)
- Wallpaper pattern remains visible in shadowed areas (no pure black)

## Technical Implementation

### Dependencies
- ST-RoomNet V3 pipeline (modules/end_to_end_pipeline.py)
- Wallpaper application (modules/wallpaper_application.py)
- Shadow loading: imageio (fallback to JPG when EXR unavailable)

### Key Functions
- `load_shadow_exr()` - Load shadow with EXR→JPG fallback
- `apply_shadow_to_wall()` - Per-wall shadow normalization and multiplication
- `create_comparison_image()` - 3-panel horizontal comparison generator

### Color Space Handling
```python
# Convert wallpaper to linear RGB
wallpaper_linear = (wallpaper_srgb / 255.0) ** 2.2

# Multiply shadow (already linear from EXR/conversion)
shadowed = wallpaper_linear * normalized_shadow

# Convert back to sRGB for display
result = (shadowed ** (1/2.2)) * 255.0
```

## Running the Experiment

```bash
cd workflow-v3-research-driven/experiments/shadow-overlay-test
source ../../../venv/bin/activate
python test_shadow_overlay.py
```

**Expected Runtime:** ~15-20 seconds (includes ST-RoomNet + SegFormer inference)

## Next Steps

**For Production Integration:**
1. Resolve OpenEXR support (recompile OpenCV with `-DWITH_OPENEXR=ON` or use dedicated EXR library)
2. Add user-adjustable shadow strength parameter (0.5-1.0 multiplier)
3. Implement resolution-independent shadow application (address v3 400×400 limitation)
4. Add shadow visualization controls in Gradio interface

**For Further Experimentation:**
- Test different ambient clamp values (0.3, 0.5, 0.6)
- Compare baseline percentile options (95th, 98th, 99th)
- Evaluate shadow quality on saturated wallpaper colors
- Test with multiple room images and lighting conditions
