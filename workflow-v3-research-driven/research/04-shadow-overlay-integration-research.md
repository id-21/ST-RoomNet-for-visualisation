# Shadow Overlay Integration - Research Analysis

**Research Date:** October 11, 2025
**Research Type:** Feature Implementation / Model Integration
**Test Image/Data:** `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/living_simple2.jpg`
**Researcher:** Claude Code

---

## Executive Summary

Investigate integration of RGB→X diffusion model's irradiance (shadow) output with ST-RoomNet V3 wallpaper visualization pipeline. The RGB→X model has already been run separately, producing shadow maps in both EXR (linear color space) and JPG (display preview) formats. Goal is to create an experimental script in `workflow-v3-research-driven/experiments/` to test shadow overlay on warped wallpaper patterns without combining model inference pipelines.

**Recommended next steps:** Create planning document for shadow overlay experiment implementation after reviewing this research.

---

## CLAUDE.md Analysis

**Files Read:**
- `workflow-v3-research-driven/CLAUDE.md` - Key patterns: Research→Planning→Implementation workflow, modular experiment structure
- `workflow-v3-research-driven/modules/CLAUDE.md` - Key patterns: End-to-end pipeline flow, wallpaper application with normal vectors, model caching
- `/Users/ishan-aiworkspace/Documents/apps/rgb-x/normal_irradiance/CLAUDE.md` - Key patterns: Shadow extraction workflow, linear vs sRGB color space handling, EXR export format
- `/Users/ishan-aiworkspace/Documents/apps/rgb-x/research/understanding-shadow.md` - Key patterns: Irradiance normalization strategy, ambient light preservation, multiplicative shadow application

**Relevant Patterns Identified:**
- **Experiment isolation pattern** from `workflow-v3-research-driven/CLAUDE.md` - Each experiment gets its own subdirectory under `experiments/` with isolated test scripts
- **Wallpaper application flow** from `modules/CLAUDE.md` - Pipeline: load models → inference → combine masks → compute normals → apply wallpaper with perspective
- **Shadow extraction strategy** from `understanding-shadow.md` - Workflow: isolate wall region → establish baseline brightness → create relative shadow mask → clamp ambient light → multiply wallpaper
- **Linear color space requirement** from `normal_irradiance/CLAUDE.md` - Shadow multiplication MUST occur in linear RGB space to prevent color shifts in saturated wallpapers

**Existing Code to Reference:**
- `workflow-v3-research-driven/modules/wallpaper_application.py:114-199` (documented in `modules/CLAUDE.md`) - Current wallpaper warping with perspective transformation
- `workflow-v3-research-driven/modules/end_to_end_pipeline.py` (documented in `modules/CLAUDE.md`) - Complete pipeline orchestration pattern
- `workflow-v3-research-driven/experiments/v2-baseline-analysis/run_v2_baseline.py` (documented in `workflow-v3-research-driven/CLAUDE.md`) - Example experiment script structure

---

## Goal Statement

**Objective:**
Create an experimental script to test if RGB→X model's irradiance shadow maps can be overlaid on ST-RoomNet V3's warped wallpaper patterns to add realistic lighting depth without integrating the model inference pipelines.

**Current State:**
- ST-RoomNet V3 pipeline: Detects walls, computes normals, applies wallpaper with perspective transformation
- Wallpaper output currently appears flat without lighting/depth perception (documented in `research/03-normal-detection-and-shading.md`)
- RGB→X model: Already run separately, produced shadow outputs for test image `living_simple2.jpg`
- Shadow data available in two formats:
  - Linear EXR: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/irradiance_shadow.exr`
  - Preview JPG: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/jpg_irradiance_preview.jpg`
- Models NOT being integrated - this is purely a post-processing experiment

**Integration Points:**
- `workflow-v3-research-driven/modules/wallpaper_application.py:114-199` - Where wallpaper is currently applied to walls
- New experiment script location: `workflow-v3-research-driven/experiments/shadow-overlay-test/`
- Shadow processing will happen AFTER wallpaper warping, as a separate multiplicative layer

**Dependencies:**
- Must leverage existing `end_to_end_pipeline.py` pattern for loading models and generating wallpaper output
- Should follow `experiments/v2-baseline-analysis/run_v2_baseline.py` structure for experiment isolation
- Requires OpenCV EXR reading capability (`cv2.imread(path, cv2.IMREAD_UNCHANGED)`)
- Shadow processing must respect wall mask boundaries from ST-RoomNet/SegFormer combination

---

## Shadow Data Analysis

### Available Shadow Outputs

**Source Image:**
- Original room photo: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/living_simple2.jpg`
- RGB→X model already processed this image separately

**Shadow Output Files (Verified to exist):**
1. **Linear EXR format**: `irradiance_shadow.exr` (2.99 MB)
   - Purpose: Linear color space shadow data for physically-accurate multiplication
   - Color space: Linear RGB (gamma correction reversed via `value^2.2`)
   - Data type: Float32 per channel
   - Reference: `normal_irradiance/CLAUDE.md:72` - "ST-RoomNet V3 expects LINEAR RGB in EXR"

2. **Preview JPG format**: `jpg_irradiance_preview.jpg` (68 KB)
   - Purpose: Visual verification for humans
   - Color space: sRGB (gamma-corrected for display)
   - Not suitable for shadow multiplication operations

### Shadow Extraction Strategy

**From `understanding-shadow.md:113-141` - 5-step process:**

**Step 1: Isolate Wall Region**
- Use ST-RoomNet normal detection + SegFormer semantic mask
- Ensures analyzing shadow patterns only on target wall, not furniture/other surfaces
- Current v3 implementation: `modules/simple_mask_combination.py` + `modules/edge_refinement_simple.py`

**Step 2: Establish Baseline Brightness**
- Find brightest areas of wall (98th percentile to avoid specular highlight outliers)
- Represents "fully lit" reference for relative shadow computation
- Formula: `baseline = np.percentile(irradiance[wall_mask], 98)`

**Step 3: Create Relative Shadow Mask**
- Divide irradiance by baseline to normalize
- Results in values where:
  - 1.0 = fully lit areas (no shadow)
  - 0.5 = half brightness (moderate shadow)
  - 0.2 = heavily shadowed
- Formula: `shadow_mask = irradiance / baseline`

**Step 4: Clamp to Preserve Ambient Light**
- Real rooms have ambient light from bounce lighting
- Prevent pure black shadows (unrealistic)
- Recommended clamp: 0.4-0.5 minimum
- Formula: `shadow_mask = np.clip(shadow_mask, 0.4, 1.0)`
- Reference: `understanding-shadow.md:137` - "maintains realism"

**Step 5: Apply to Wallpaper**
- Multiply warped wallpaper by shadow mask (in LINEAR space)
- Areas with 1.0 remain unchanged
- Areas with 0.5 darken by 50%
- Formula: `result = wallpaper_linear * shadow_mask`

### Color Space Requirements

**Critical: Linear vs sRGB Workflow**

**From `normal_irradiance/CLAUDE.md:54-67` and `understanding-shadow.md:34-98`:**

**Why Linear Space is Required:**
- Shadow multiplication is a physical light operation
- Formula: `final_light = surface_reflectance × incoming_light`
- In sRGB space, this formula produces incorrect results (color shifts)
- Saturated colors (reds, blues) show visible hue shifts if processed in sRGB
- Reference: `understanding-shadow.md:95-98` - "vibrant/saturated colors show color shifts if processed in sRGB"

**Conversion Process:**
```
# sRGB to Linear (for wallpaper input)
wallpaper_linear = wallpaper_srgb ** 2.2

# Multiply in linear space
shadowed = wallpaper_linear * shadow_mask  # shadow_mask already linear from EXR

# Linear to sRGB (for display output)
result_srgb = shadowed ** (1/2.2)
```

**Reference Implementation:**
- `normal_irradiance/CLAUDE.md:61-66` - Documents linear space multiplication pattern
- `understanding-shadow.md:34-58` - Explains gamma correction fundamentals

---

## Current V3 Pipeline Flow

**From `workflow-v3-research-driven/modules/CLAUDE.md:42-52`:**

```
1. initialize_models() - Cache ST-RoomNet & SegFormer models
2. ST-RoomNet inference → (segmentation, theta, original_rgb)
3. SegFormer inference → wall_confidence
4. combine_masks_simple() → combined_masks
5. refine_edges_with_st_boundaries() → refined_masks
6. compute_normal_vectors(theta) → normals_dict
7. create_walls_from_masks_with_normals() → Wall objects with normals
8. RoomWallLayout() → standardized output
9. apply_wallpaper() → perspective-transformed wallpaper
```

**Shadow Overlay Insertion Point:**
- Happens AFTER step 9 (wallpaper application)
- Input: Warped wallpaper RGBA from `wallpaper_application.py:199`
- Process: Load shadow EXR → extract wall-specific shadow → multiply in linear space
- Output: Wallpaper with realistic lighting depth

---

## Integration Points

### Where Shadow Processing Fits

**Not modifying existing pipeline** - Creating separate experiment script:

**Script location:** `workflow-v3-research-driven/experiments/shadow-overlay-test/test_shadow_overlay.py`

**Script structure (following `experiments/v2-baseline-analysis/run_v2_baseline.py` pattern):**
```
1. Load test image (living_simple2.jpg)
2. Run ST-RoomNet V3 pipeline → get wallpaper output + wall masks
3. Load shadow EXR from rgb-x output directory
4. For each wall:
   a. Extract wall-specific shadow region
   b. Compute baseline brightness (98th percentile)
   c. Create normalized shadow mask (clamp 0.4-1.0)
   d. Convert wallpaper to linear RGB
   e. Multiply wallpaper × shadow
   f. Convert result back to sRGB
5. Save comparison outputs:
   - Without shadow (current v3)
   - With shadow (new approach)
   - Shadow mask visualization
6. Generate metrics/visual quality assessment
```

### File Dependencies

**Imports needed:**
```python
import sys
import os
import numpy as np
import cv2

# ST-RoomNet V3 modules
sys.path.insert(0, '../modules')
from end_to_end_pipeline import process_room_image_complete
from wallpaper_application import WallpaperPattern, apply_wallpaper_to_multiple_walls
from room_wall_layout import RoomWallLayout
```

**External file paths:**
- Shadow EXR: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/irradiance_shadow.exr`
- Shadow preview: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/jpg_irradiance_preview.jpg`
- Test image: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/living_simple2.jpg`

---

## Technical Considerations

### Resolution Matching Challenge

**Problem:** Shadow EXR and ST-RoomNet output may have different resolutions

**From `workflow-v3-research-driven/CLAUDE.md:14-16`:**
- ST-RoomNet V3 currently has resolution issues (400×400 constraint)
- Resolution preservation fix is IN PROGRESS
- Shadow overlay experiment must handle resolution mismatch

**Solution approaches:**
1. **Resize shadow to match ST-RoomNet output** (400×400)
   - Pro: Simple, guaranteed alignment
   - Con: Loses shadow detail
   - Use: `cv2.resize(shadow_exr, (400, 400), interpolation=cv2.INTER_LINEAR)`

2. **Resize ST-RoomNet output to match shadow** (original resolution)
   - Pro: Preserves shadow quality
   - Con: Upscaling wallpaper may introduce artifacts
   - Aligns with v3's resolution preservation goal

3. **Process at both resolutions and compare**
   - Pro: Provides data for resolution preservation planning
   - Con: More complex experiment
   - Recommended for thorough research

**Recommendation:** Approach #3 - Test both to inform resolution preservation work

### Wall Mask Alignment

**Challenge:** Shadow map is full-frame, but wallpaper is applied per-wall

**From `modules/CLAUDE.md:135-150`:**
- ST-RoomNet segments image into walls (labels 1=left, 2=front, 3=right)
- Each wall has pixel_mask attribute defining its region
- Wallpaper applied separately to each wall with perspective transform

**Shadow extraction per wall:**
```python
# For each wall in RoomWallLayout
wall_mask = wall.pixel_mask  # Boolean mask (H, W)
wall_shadow = irradiance_shadow[wall_mask]  # Extract wall pixels only

# Compute baseline from THIS wall's shadow data
baseline = np.percentile(wall_shadow, 98)

# Create normalized shadow for this wall
normalized_shadow = np.clip(irradiance_shadow / baseline, 0.4, 1.0)

# Apply only to wall region
wallpaper_with_shadow = wallpaper_linear * normalized_shadow
```

**Critical:** Each wall may have different baseline brightness
- Left wall might be in shadow (low baseline)
- Front wall might be directly lit (high baseline)
- Computing per-wall baselines prevents cross-contamination

### Edge Case Handling

**From `understanding-shadow.md:180-183` and `normal_irradiance/CLAUDE.md:75`:**

**Specular highlights:**
- EXR may contain bright specular reflections (windows, lamps)
- Using 98th percentile instead of max() avoids outlier contamination
- Reference: `understanding-shadow.md:128`

**Empty walls:**
- Some walls may have <100 pixels (marked invisible)
- Check `wall.visible` before processing
- Reference: `modules/CLAUDE.md:94` - "Minimum visibility: >100 pixels"

**Division by zero:**
- If baseline is 0 (completely dark wall), shadow division fails
- Check: `if baseline > 1e-6: shadow = irradiance / baseline`
- Fallback: No shadow application for that wall

**Out-of-bounds shadow values:**
- Clamp after normalization: `np.clip(shadow, 0.0, 1.5)`
- Allow slight over-brightness (>1.0) for highlight effects
- Final clamp before output: `np.clip(result, 0.0, 255.0)`

---

## Experiment Design

### Test Configuration

**Input Image:**
- Path: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/living_simple2.jpg`
- This is the image that RGB→X already processed

**Wallpaper Pattern:**
- Use same test pattern from v3 experiments
- Should be high-contrast to clearly show shadow effects
- Recommended: Pattern with both light and dark areas

**Shadow Parameters to Test:**
1. **Ambient light clamp values:** 0.3, 0.4, 0.5
   - Lower = darker shadows (more dramatic)
   - Higher = softer shadows (more commercial/aspirational)
   - Reference: `understanding-shadow.md:169-171` - "70-80% shadow strength for commercial visualization"

2. **Baseline percentile:** 95th, 98th, 99th
   - Lower = considers more bright areas as "baseline"
   - Higher = only brightest areas as baseline
   - 98th recommended in documentation

3. **Resolution approach:** 400×400 vs original vs both
   - Test impact on visual quality

### Validation Metrics

**Qualitative Assessment:**
1. **Physical plausibility** (`understanding-shadow.md:188-194`)
   - Do shadows align with visible objects (furniture, windows)?
   - Are shadow directions consistent with light sources?
   - Do soft shadows have gradual transitions?

2. **Color accuracy** (`understanding-shadow.md:195-199`)
   - Does wallpaper hue remain consistent in shadows?
   - Are there unexpected color shifts or desaturation?
   - Do saturated colors look muddy or contaminated?

3. **Perceptual realism** (`understanding-shadow.md:200-203`)
   - Are there pure black regions (bad - should have ambient light)?
   - Does shadow intensity feel natural vs overdramatic?
   - Does composite look like single photograph vs layered fake?

**Quantitative Metrics (optional):**
- Mean shadow intensity per wall
- Shadow contrast ratio (brightest / darkest pixels)
- Color shift measurement (hue deviation in CIELAB space)
- Visual comparison: with shadow vs without shadow

### Output Structure

**Following v2-baseline-analysis pattern:**

```
workflow-v3-research-driven/experiments/shadow-overlay-test/
├── test_shadow_overlay.py         # Main experiment script
├── outputs/
│   ├── 00_original_input.png      # living_simple2.jpg
│   ├── 01_shadow_exr_loaded.png   # Visualization of loaded shadow
│   ├── 02_shadow_preview_jpg.png  # Reference preview
│   ├── 03_v3_wallpaper_no_shadow.png  # Current v3 output (baseline)
│   ├── 04_wall_masks.png          # ST-RoomNet wall segmentation
│   ├── wall_1_left/
│   │   ├── a_wall_shadow_region.png       # Shadow extracted for this wall
│   │   ├── b_normalized_shadow_mask.png   # After baseline normalization
│   │   ├── c_wallpaper_no_shadow.png      # Wallpaper without shadow
│   │   ├── d_wallpaper_with_shadow.png    # Wallpaper with shadow
│   │   ├── e_comparison.png               # Side-by-side comparison
│   ├── wall_2_front/
│   │   └── [same structure]
│   ├── wall_3_right/
│   │   └── [same structure]
│   ├── 05_final_composite_no_shadow.png   # All walls without shadow
│   ├── 06_final_composite_with_shadow.png # All walls with shadow
│   └── metrics_summary.txt                # Quantitative measurements
└── README.md                       # Experiment documentation
```

---

## Open Questions

Research phase questions to consider during planning:

- [ ] **Resolution strategy:** Should we resize shadow to 400×400 or upscale v3 output to original? Test both?
- [ ] **Per-wall vs global baseline:** Compute baseline per wall or use global maximum? (Recommendation: per-wall for lighting variation)
- [ ] **Ambient clamp value:** Start with 0.4 or 0.5? Allow user adjustment in experiment?
- [ ] **Color space validation:** How to verify no hue shifts? Visual inspection sufficient or need CIELAB metrics?
- [ ] **Edge artifacts:** Will shadow boundaries align cleanly with wall mask edges? Need edge feathering?
- [ ] **Wallpaper pattern selection:** Which test pattern best demonstrates shadow effects? High-contrast or mid-tone?
- [ ] **Shadow strength control:** Should experiment include adjustable shadow intensity multiplier? (e.g., 70-80% strength for commercial look)

---

## Next Steps

**Planning Phase** (create `planning/shadow-overlay-experiment-plan.md`):
- Design experiment script structure following v2-baseline-analysis pattern
- Define exact shadow processing algorithm with code references
- Specify output visualization formats
- List all parameters to test (ambient clamp, baseline percentile, resolution)
- Define success criteria for "shadow overlay works well"
- Create implementation roadmap with file structure

---

## References

**CLAUDE.md Files:**
- `workflow-v3-research-driven/CLAUDE.md` - Sections: Workflow, Patterns, File Map
- `workflow-v3-research-driven/modules/CLAUDE.md` - Sections: Patterns (wallpaper application, pipeline flow)
- `/Users/ishan-aiworkspace/Documents/apps/rgb-x/normal_irradiance/CLAUDE.md` - Sections: Workflow (shadow extraction), Critical Notes (linear RGB requirement)

**Code Locations:**
- Primary wallpaper application: `workflow-v3-research-driven/modules/wallpaper_application.py:114-199`
- Pipeline orchestration: `workflow-v3-research-driven/modules/end_to_end_pipeline.py`
- Example experiment structure: `workflow-v3-research-driven/experiments/v2-baseline-analysis/run_v2_baseline.py`
- Wall data structures: `workflow-v3-research-driven/modules/room_wall_layout.py`

**External Documentation:**
- Shadow theory: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/research/understanding-shadow.md` (complete document)
- RGB→X workflow: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/normal_irradiance/CLAUDE.md` (complete document)

**Output Artifacts:**
- Shadow data: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/irradiance_shadow.exr` (2.99 MB, linear RGB)
- Shadow preview: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/jpg_irradiance_preview.jpg` (68 KB, sRGB)
- Test image: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/living_simple2.jpg` (105 KB)
- Experiment outputs: `workflow-v3-research-driven/experiments/shadow-overlay-test/outputs/` (to be created)

**Related Research:**
- `workflow-v3-research-driven/research/03-normal-detection-and-shading.md` - Original plan for shading implementation (now superseded by RGB→X integration)
- `workflow-v3-research-driven/research/01-edge-bleeding-analysis.md` - Example research document structure
- `workflow-v3-research-driven/research/02-resolution-aspect-ratio-analysis.md` - Related resolution concerns

---

## Conclusion

Shadow overlay integration is feasible as a separate experiment without combining model pipelines. The RGB→X model has already produced high-quality irradiance shadow maps in linear EXR format, which can be directly multiplied with v3's warped wallpaper output. Key technical requirements are clear: per-wall baseline normalization, ambient light clamping (0.4-0.5), and linear color space processing to prevent hue shifts. The experiment should follow v3's established research pattern with isolated script in `experiments/shadow-overlay-test/` and comprehensive output visualization.

**Recommended path forward:** Create planning document (`planning/shadow-overlay-experiment-plan.md`) detailing the exact implementation steps, parameter values, and validation criteria before proceeding to implementation phase.
