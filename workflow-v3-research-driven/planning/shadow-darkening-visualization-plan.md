# Implementation Plan: Shadow Darkening Visualization

**Planning Date:** October 13, 2025
**Research Reference:** `research/04-shadow-overlay-integration-research.md`
**Target:** Create visualization script to isolate shadow effect on white canvas with parameterized ambient clamp testing

---

## Overview

### Purpose
Create a new experimental script that applies shadow overlay to a **pure white canvas** instead of wallpaper. This isolates the shadow darkening effect and makes it easy to visualize the exact shadow intensity at different ambient clamp values.

### Key Features
- Pure white canvas generation (eliminates wallpaper pattern variables)
- Parameterized ambient clamp values for side-by-side comparison
- Grid layout showing all clamp variations simultaneously
- Per-wall breakdown showing shadow effect on each wall individually
- Visual verification of shadow inversion correctness

### Motivation
The current `test_shadow_overlay.py` applies shadows to patterned wallpaper, making it difficult to see the pure shadow effect. A white canvas:
- Shows ONLY the darkening caused by shadows (no pattern interference)
- Makes it trivial to verify shadow inversion is working (dark areas should darken white)
- Allows easy comparison of different ambient clamp values
- Provides clear visual feedback for choosing optimal parameters

---

## Files to Create

### Primary Script
**File:** `experiments/shadow-overlay-test/visualize_shadow_darkening.py`
**Based on:** `experiments/shadow-overlay-test/test_shadow_overlay.py`
**Purpose:** Run ST-RoomNet V3 pipeline, apply shadow to white canvas with multiple clamp values, generate comparison outputs

---

## Implementation Steps

### Step 1: Create Script Structure
**File:** `experiments/shadow-overlay-test/visualize_shadow_darkening.py`
**Actions:**
1. Copy imports from `test_shadow_overlay.py`:
   - Standard: `sys, os, numpy as np, cv2`
   - PIL: `from PIL import Image`
   - imageio: `import imageio` (for EXR loading)
   - ST-RoomNet modules: `process_room_image_complete, WallpaperPattern, apply_wallpaper_to_multiple_walls, RoomWallLayout`
2. Copy `load_shadow_exr()` function unchanged
3. Create file header with corrected shadow inversion note

**Success criteria:** Script imports successfully, no errors

---

### Step 2: Implement White Canvas Generator
**File:** `experiments/shadow-overlay-test/visualize_shadow_darkening.py`
**Actions:**
1. Create `create_white_canvas(shape=(400, 400, 3))` function:
   ```python
   def create_white_canvas(shape=(400, 400, 3)):
       """
       Create pure white RGB image for shadow visualization.

       Args:
           shape: Image dimensions (H, W, C), default 400x400 RGB

       Returns:
           np.ndarray: Pure white image (255, 255, 255) uint8
       """
       return np.ones(shape, dtype=np.uint8) * 255
   ```
2. Test: Generate 400×400 white canvas, verify all pixels are 255

**Success criteria:** Function returns (400, 400, 3) uint8 array with all values = 255

---

### Step 3: Parameterize Shadow Application
**File:** `experiments/shadow-overlay-test/visualize_shadow_darkening.py`
**Actions:**
1. Copy `apply_shadow_to_wall()` from `test_shadow_overlay.py`
2. Ensure `ambient_clamp` parameter is explicit (already is in corrected version)
3. Function signature: `apply_shadow_to_wall(canvas_rgb, shadow_exr, wall_mask, ambient_clamp)`
4. Define clamp value test range as global constant:
   ```python
   CLAMP_VALUES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
   ```

**Success criteria:** Function accepts any ambient_clamp value, applies shadow correctly

---

### Step 4: Implement Comparison Grid Generator
**File:** `experiments/shadow-overlay-test/visualize_shadow_darkening.py`
**Actions:**
1. Create `create_clamp_comparison_grid(original, shadow_vis, white_no_shadow, white_results_dict)`:
   ```python
   def create_clamp_comparison_grid(original, shadow_vis, white_no_shadow,
                                     white_results_dict):
       """
       Create grid comparing white canvas results at different clamp values.

       Layout (3 columns):
       Row 1: Original | Shadow Vis | White (no shadow)
       Row 2: Clamp 0.0 | Clamp 0.2 | Clamp 0.3
       Row 3: Clamp 0.4 | Clamp 0.5 | Clamp 0.6
       Row 4: Clamp 0.8 |           |

       Args:
           original: Original room image (H, W, 3)
           shadow_vis: Shadow EXR visualization (H, W, 3)
           white_no_shadow: White canvas without shadow (H, W, 3)
           white_results_dict: {clamp_value: result_image}

       Returns:
           np.ndarray: Grid layout with labels
       """
   ```
2. Resize all images to same size (e.g., 300×300)
3. Arrange in grid with padding between cells
4. Add text labels with clamp values using `cv2.putText()`
5. Test: Generate grid from dummy images

**Success criteria:** Grid shows all images with clear labels, proper layout

---

### Step 5: Implement Per-Wall Breakdown
**File:** `experiments/shadow-overlay-test/visualize_shadow_darkening.py`
**Actions:**
1. Create `create_per_wall_breakdown(layout, white_canvas, shadow_exr, refined_masks, ambient_clamp)`:
   ```python
   def create_per_wall_breakdown(layout, white_canvas, shadow_exr,
                                  refined_masks, ambient_clamp=0.4):
       """
       Show shadow effect on each wall individually.

       Creates horizontal layout: Wall 1 | Wall 2 | Wall 3
       Each showing white canvas with shadow applied only to that wall.

       Args:
           layout: RoomWallLayout with detected walls
           white_canvas: Pure white canvas (H, W, 3)
           shadow_exr: Shadow map (H, W, 3) float32
           refined_masks: Dict mapping wall_label to boolean mask
           ambient_clamp: Ambient light clamp value

       Returns:
           np.ndarray: Horizontal concatenation of per-wall results
       """
   ```
2. For each wall:
   - Create white canvas copy
   - Apply shadow only to that wall's mask region
   - Add text label: "Wall {id}" or "Left Wall", "Front Wall", "Right Wall"
3. Concatenate horizontally with padding
4. Test: Generate breakdown for 3 walls

**Success criteria:** Shows each wall separately with shadow applied, clear labels

---

### Step 6: Implement Main Pipeline
**File:** `experiments/shadow-overlay-test/visualize_shadow_darkening.py`
**Actions:**
1. Create `main()` function following pattern from `test_shadow_overlay.py`
2. Pipeline flow:
   ```python
   # Step 1: Run ST-RoomNet V3
   layout, st_seg, seg_conf, refined_masks = process_room_image_complete(
       image_path=input_image,
       wall_threshold=0.4,
       use_cached_models=True
   )

   # Step 2: Load shadow EXR and resize
   shadow_exr = load_shadow_exr(shadow_exr_path)
   shadow_resized = cv2.resize(shadow_exr, (400, 400), interpolation=cv2.INTER_LINEAR)

   # Step 3: Create white canvas (replaces wallpaper loading)
   white_canvas = create_white_canvas(shape=(400, 400, 3))

   # Step 4: Apply shadow with multiple clamp values
   white_results = {}
   for clamp in CLAMP_VALUES:
       white_with_shadow = white_canvas.copy()
       for wall in layout.walls:
           if not wall.is_visible:
               continue
           # Get wall mask and apply shadow
           wall_mask = refined_masks[wall_label]
           shadowed = apply_shadow_to_wall(white_with_shadow, shadow_resized,
                                           wall_mask, ambient_clamp=clamp)
           white_with_shadow[wall_mask] = shadowed[wall_mask]
       white_results[clamp] = white_with_shadow

   # Step 5: Generate comparison outputs
   comparison_grid = create_clamp_comparison_grid(original, shadow_vis,
                                                   white_canvas, white_results)
   per_wall_breakdown = create_per_wall_breakdown(layout, white_canvas,
                                                   shadow_resized, refined_masks,
                                                   ambient_clamp=0.4)

   # Step 6: Save all outputs
   ```
3. Define paths:
   - Input: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/living_simple2.jpg`
   - Shadow EXR: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/irradiance_shadow.exr`
   - Output dir: `experiments/shadow-overlay-test/outputs-visualization/`

**Success criteria:** Pipeline executes without errors, generates all intermediate results

---

### Step 7: Save Output Images
**File:** `experiments/shadow-overlay-test/visualize_shadow_darkening.py`
**Actions:**
1. Create output directory: `outputs-visualization/`
2. Save individual files:
   ```python
   cv2.imwrite('00_original.jpg', cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR))
   cv2.imwrite('01_shadow_visualization.jpg', cv2.cvtColor(shadow_vis, cv2.COLOR_RGB2BGR))
   cv2.imwrite('02_white_canvas_no_shadow.jpg', cv2.cvtColor(white_canvas, cv2.COLOR_RGB2BGR))

   for clamp in CLAMP_VALUES:
       filename = f'0{3 + CLAMP_VALUES.index(clamp)}_white_canvas_clamp_{clamp:.1f}.jpg'
       cv2.imwrite(filename, cv2.cvtColor(white_results[clamp], cv2.COLOR_RGB2BGR))

   cv2.imwrite('10_comparison_grid.jpg', cv2.cvtColor(comparison_grid, cv2.COLOR_RGB2BGR))
   cv2.imwrite('11_per_wall_breakdown.jpg', cv2.cvtColor(per_wall_breakdown, cv2.COLOR_RGB2BGR))
   ```
3. Print summary of saved files

**Success criteria:** All output files exist in `outputs-visualization/` directory

---

## Output Structure

**Directory:** `experiments/shadow-overlay-test/outputs-visualization/`

**Files:**
```
00_original.jpg                        # Original room image (400×400)
01_shadow_visualization.jpg            # Shadow EXR visualized (clipped to 0-255)
02_white_canvas_no_shadow.jpg         # Pure white reference (no shadow applied)
03_white_canvas_clamp_0.0.jpg         # No ambient light (maximum shadow)
04_white_canvas_clamp_0.2.jpg         # 20% minimum brightness
05_white_canvas_clamp_0.3.jpg         # 30% minimum brightness
06_white_canvas_clamp_0.4.jpg         # 40% minimum (current default)
07_white_canvas_clamp_0.5.jpg         # 50% minimum brightness
08_white_canvas_clamp_0.6.jpg         # 60% minimum brightness
09_white_canvas_clamp_0.8.jpg         # 80% minimum (very subtle shadows)
10_comparison_grid.jpg                # Grid showing all clamp variations
11_per_wall_breakdown.jpg             # Each wall visualized separately (clamp=0.4)
```

**Total:** 12 output images

---

## Data Flow

```
living_simple2.jpg → ST-RoomNet V3 pipeline → RoomWallLayout + wall_masks
                                                       ↓
irradiance_shadow.exr → resize(400×400) → load_shadow_exr → shadow_exr (linear float)
                                                       ↓
create_white_canvas() → white_canvas (255, 255, 255) → pure white reference
                                                       ↓
For each clamp in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
    white_canvas + shadow_exr + wall_masks → apply_shadow_to_wall(clamp) → result_clamp
                                                       ↓
[original, shadow_vis, white_canvas, results_dict] → create_clamp_comparison_grid → grid.jpg
                                                       ↓
[layout, white_canvas, shadow_exr, masks] → create_per_wall_breakdown → breakdown.jpg
```

---

## Code Reuse Strategy

### Functions to Reuse (from `test_shadow_overlay.py`)
1. **`load_shadow_exr()`** - unchanged, handles EXR loading with fallback
2. **`apply_shadow_to_wall()`** - unchanged, already has parameterized `ambient_clamp`
3. **ST-RoomNet pipeline setup** - model loading, mask generation code
4. **Wall iteration logic** - mapping wall IDs to numeric labels

### New Functions to Implement
1. **`create_white_canvas()`** - generates pure white RGB array
2. **`create_clamp_comparison_grid()`** - grid layout with multiple clamp results
3. **`create_per_wall_breakdown()`** - horizontal per-wall visualization
4. **`main()`** - orchestrates white canvas pipeline (different from wallpaper pipeline)

---

## Testing Plan

### Unit Tests
1. **White canvas generation:**
   - Verify shape is (400, 400, 3)
   - Verify dtype is uint8
   - Verify all values are 255

2. **Shadow application:**
   - Apply to white canvas with clamp=0.0, verify darkening occurs
   - Apply with clamp=0.8, verify minimal darkening
   - Compare clamp=0.0 vs clamp=0.8, verify 0.0 is darker

3. **Grid generation:**
   - Verify layout has correct number of rows/columns
   - Verify labels are visible and correct
   - Verify images are properly sized and aligned

### Integration Test
1. Run: `cd experiments/shadow-overlay-test && python visualize_shadow_darkening.py`
2. Expected: Script completes without errors
3. Verify outputs:
   - 12 files in `outputs-visualization/` directory
   - `00_original.jpg` matches input image
   - `02_white_canvas_no_shadow.jpg` is pure white
   - `03_white_canvas_clamp_0.0.jpg` shows strong darkening
   - `09_white_canvas_clamp_0.8.jpg` shows subtle darkening
   - `10_comparison_grid.jpg` shows all variations clearly
   - `11_per_wall_breakdown.jpg` shows 2-3 walls horizontally

### Visual Verification
1. **Shadow inversion correctness:**
   - Open `01_shadow_visualization.jpg` - should show bright/dark regions
   - Open `03_white_canvas_clamp_0.0.jpg` - dark regions should match shadow vis
   - Confirms high EXR values correctly darken white canvas

2. **Clamp value effect:**
   - Compare `03_clamp_0.0.jpg` vs `09_clamp_0.8.jpg`
   - Lower clamp = more dramatic shadows
   - Higher clamp = softer, more subtle shadows

3. **Per-wall variation:**
   - Open `11_per_wall_breakdown.jpg`
   - Verify each wall shows different shadow patterns
   - Confirms per-wall baseline normalization is working

---

## Success Criteria

- [ ] Script runs without errors on `living_simple2.jpg`
- [ ] White canvas generated correctly (pure white, 255 RGB)
- [ ] Shadow applied to white canvas shows visible darkening
- [ ] Clamp 0.0 shows stronger darkening than clamp 0.8
- [ ] All 7 clamp values produce distinct results
- [ ] Comparison grid displays all variations with clear labels
- [ ] Per-wall breakdown shows 2-3 walls individually
- [ ] Output files total 12 images in `outputs-visualization/` directory
- [ ] Visual inspection confirms shadow inversion is correct (dark EXR → dark canvas)

---

## Expected Benefits

1. **Pure visualization:** White canvas shows ONLY shadow effect without wallpaper pattern interference
2. **Parameter exploration:** Side-by-side comparison makes choosing optimal clamp value trivial
3. **Verification tool:** Easy to confirm shadow inversion correctness (dark should darken white)
4. **Per-wall insights:** See how shadow varies across different wall orientations
5. **Grid layout:** All variations visible in single image for quick assessment

---

## Running the Experiment

```bash
cd workflow-v3-research-driven/experiments/shadow-overlay-test
source ../../../venv/bin/activate
python visualize_shadow_darkening.py
```

**Expected Runtime:** ~15-20 seconds (includes ST-RoomNet + SegFormer inference)

**Output Location:** `experiments/shadow-overlay-test/outputs-visualization/`

---

## Next Steps After Implementation

1. **Visual analysis:** Open `10_comparison_grid.jpg` to see all clamp variations
2. **Choose optimal clamp:** Based on visual comparison, select best ambient clamp value (likely 0.4-0.5)
3. **Verify correctness:** Confirm dark regions in `01_shadow_visualization.jpg` darken white canvas (not brighten)
4. **Test other images:** Run on different room photos to validate consistency
5. **Document findings:** Update research notes with optimal clamp value and visual examples

---

## Notes

- **No metrics computation:** Intentionally omitted to keep visualization simple and focused
- **White canvas approach:** Eliminates variables from wallpaper patterns/colors
- **Multiple clamp values:** Allows empirical comparison rather than theoretical guessing
- **Grid output:** Single image shows full parameter exploration
- **Reuses corrected shadow logic:** Benefits from inverted irradiance fix in `test_shadow_overlay.py`
