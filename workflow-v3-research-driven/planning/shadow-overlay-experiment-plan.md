# Implementation Plan: Shadow Overlay Experiment

**Planning Date:** October 11, 2025
**Research Reference:** `research/04-shadow-overlay-integration-research.md`
**Target:** Create experiment script to test shadow overlay on ST-RoomNet V3 wallpaper output

---

## Required Reading

**CLAUDE.md Files:**
- `workflow-v3-research-driven/modules/CLAUDE.md` - Section: Patterns (wallpaper application flow)
- `/Users/ishan-aiworkspace/Documents/apps/rgb-x/research/understanding-shadow.md` - Section: Shadow Extraction Strategy (lines 113-141)

**Reference Implementations:**
- `experiments/v2-baseline-analysis/run_v2_baseline.py:77-234` - Experiment script structure
- `modules/wallpaper_application.py:114-199` - Wallpaper application with perspective
- `modules/end_to_end_pipeline.py` - Pipeline orchestration pattern

**Dependencies:**
- Module: `end_to_end_pipeline` - Function: `process_room_image_complete(image_path, wall_threshold, use_cached_models) -> (layout, st_seg, seg_conf, refined_masks)`
- Module: `wallpaper_application` - Function: `apply_wallpaper_to_multiple_walls(wallpaper_pattern, walls, original_image, wall_filter) -> result_image`
- Module: `wallpaper_application` - Class: `WallpaperPattern(image, wall_height_inches, pattern_height_inches)`
- Data structure: `RoomWallLayout.walls` - Fields: `pixel_mask`, `wall_id`, `normal_vector`, `corners_2d`

---

## Files to Create

### New: `experiments/shadow-overlay-test/test_shadow_overlay.py`
**Purpose:** Run ST-RoomNet V3 pipeline, apply wallpaper, overlay shadows, save comparison outputs
**Pattern source:** `experiments/v2-baseline-analysis/run_v2_baseline.py:77-234`
**Functions to implement:**
- `load_shadow_exr(exr_path) -> np.ndarray` - Load linear RGB shadow data
- `apply_shadow_to_wall(wallpaper_rgb, shadow_exr, wall_mask, ambient_clamp=0.4) -> np.ndarray` - Per-wall shadow multiplication in linear space
- `create_comparison_image(original, wallpaper_only, wallpaper_with_shadow) -> np.ndarray` - Horizontal 3-panel comparison
- `main()` - Orchestrate full pipeline

---

## Implementation Steps

### Step 1: Create Experiment Directory Structure
**Files:** `experiments/shadow-overlay-test/`
**Actions:**
1. Create directory: `experiments/shadow-overlay-test/`
2. Create subdirectory: `experiments/shadow-overlay-test/outputs/`
3. Test: Verify directories exist

**Success criteria:** Directories created at specified paths

### Step 2: Implement Shadow Loading Function
**Files:** `experiments/shadow-overlay-test/test_shadow_overlay.py`
**Actions:**
1. Create `load_shadow_exr(exr_path)` function
2. Use `cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)` to load linear RGB float data
3. Verify shape matches expected dimensions
4. Test: Load `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/irradiance_shadow.exr` and print shape

**Success criteria:** Shadow EXR loaded as float32 array with 3 channels

### Step 3: Implement Per-Wall Shadow Application Function
**Files:** `experiments/shadow-overlay-test/test_shadow_overlay.py`
**Actions:**
1. Create `apply_shadow_to_wall(wallpaper_rgb, shadow_exr, wall_mask, ambient_clamp)` function
2. Extract wall shadow region: `wall_shadow = shadow_exr[wall_mask]`
3. Establish value range (CORRECTED): `min_val = np.percentile(wall_shadow, 2); max_val = np.percentile(wall_shadow, 98)`
4. Normalize to 0-1: `normalized = (shadow_exr - min_val) / (max_val - min_val)`
5. INVERT (HIGH EXR = shadows → LOW multiplier): `shadow_multiplier = 1.0 - normalized`
6. Clamp: `shadow_multiplier = np.clip(shadow_multiplier, ambient_clamp, 1.0)`
7. Convert wallpaper to linear: `wallpaper_linear = (wallpaper_rgb / 255.0) ** 2.2`
8. Multiply: `shadowed = wallpaper_linear * shadow_multiplier`
9. Convert back to sRGB: `result = (shadowed ** (1/2.2)) * 255.0`
10. Test: Apply to dummy 100×100 wallpaper with synthetic shadow

**Success criteria:** Function returns uint8 array with shadowed wallpaper, no NaN/inf values, high EXR values darken output

### Step 4: Implement Comparison Image Generator
**Files:** `experiments/shadow-overlay-test/test_shadow_overlay.py`
**Actions:**
1. Create `create_comparison_image(original, wallpaper_only, wallpaper_with_shadow)` function
2. Resize all three images to same height (preserve aspect ratio)
3. Concatenate horizontally: `np.hstack([img1, img2, img3])`
4. Add text labels using `cv2.putText()`
5. Test: Generate comparison from 3 dummy images

**Success criteria:** Single horizontal image with 3 panels and labels

### Step 5: Implement Main Pipeline Orchestration
**Files:** `experiments/shadow-overlay-test/test_shadow_overlay.py`
**Actions:**
1. Create `main()` function following pattern from `experiments/v2-baseline-analysis/run_v2_baseline.py:77-234`
2. Run ST-RoomNet V3 pipeline: `layout, st_seg, seg_conf, refined = process_room_image_complete(image_path, 0.4, True)`
3. Load wallpaper: `pattern_img = cv2.imread('/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/images/graphite-clay-repeated-patterns-coordonne.jpg')`
4. Apply wallpaper: `wallpaper_result = apply_wallpaper_to_multiple_walls(WallpaperPattern(pattern_img), layout.walls, original_img, None)`
5. Load shadow: `shadow_exr = load_shadow_exr('/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/irradiance_shadow.exr')`
6. Resize shadow to match ST-RoomNet output: `shadow_resized = cv2.resize(shadow_exr, (400, 400), cv2.INTER_LINEAR)`
7. Test: Run pipeline and print intermediate shapes

**Success criteria:** Pipeline executes without errors, all intermediate outputs have expected shapes

### Step 6: Apply Shadow to Each Wall
**Files:** `experiments/shadow-overlay-test/test_shadow_overlay.py`
**Actions:**
1. In `main()`, iterate over `layout.walls`
2. For each wall, extract combined mask from `refined_masks[wall.wall_id]`
3. Apply shadow: `shadowed_wall = apply_shadow_to_wall(wallpaper_result, shadow_resized, wall_mask, 0.4)`
4. Composite shadowed walls back onto original image
5. Test: Verify each wall processes without errors

**Success criteria:** All visible walls have shadows applied, output is valid uint8 image

### Step 7: Save Output Images
**Files:** `experiments/shadow-overlay-test/test_shadow_overlay.py`
**Actions:**
1. Save original input: `cv2.imwrite('outputs/00_original.jpg', original_img)`
2. Save wallpaper without shadow: `cv2.imwrite('outputs/01_wallpaper_only.jpg', wallpaper_result)`
3. Save wallpaper with shadow: `cv2.imwrite('outputs/02_wallpaper_with_shadow.jpg', final_result)`
4. Generate and save comparison: `cv2.imwrite('outputs/03_comparison.jpg', create_comparison_image(...))`
5. Test: Verify all 4 output files exist

**Success criteria:** 4 JPG files saved in `experiments/shadow-overlay-test/outputs/`

---

## Data Flow

```
living_simple2.jpg → ST-RoomNet V3 pipeline → RoomWallLayout + original_rgb
                                                        ↓
graphite-clay.jpg → WallpaperPattern → apply_wallpaper_to_multiple_walls → wallpaper_rgb (no shadow)
                                                        ↓
irradiance_shadow.exr → resize(400×400) → load_shadow_exr → shadow_exr (linear float)
                                                        ↓
wallpaper_rgb + shadow_exr + wall_masks → apply_shadow_to_wall (per wall) → wallpaper_rgb (with shadow)
                                                        ↓
[original, wallpaper_only, wallpaper_with_shadow] → create_comparison_image → comparison.jpg
```

---

## Testing Plan

**Per-step tests:**
- Step 2: Load shadow EXR, verify shape is (H, W, 3) and dtype is float32
- Step 3: Apply shadow to synthetic 100×100 test image, verify output range [0, 255]
- Step 4: Create comparison from 3 dummy images, verify concatenation works
- Step 5: Run pipeline on test image, print all intermediate shapes
- Step 6: Verify each wall in layout gets shadow applied
- Step 7: Check all 4 output files exist and are valid JPG images

**Integration test:**
- Run: `cd experiments/shadow-overlay-test && python test_shadow_overlay.py`
- Expected: Script completes without errors
- Outputs: 4 JPG files in `outputs/` directory
- Visual check: Comparison image shows original, wallpaper without shadow, wallpaper with shadow side-by-side

---

## Success Criteria

- [ ] Script executes without errors on `living_simple2.jpg`
- [ ] Wallpaper applied to all visible walls with perspective correction
- [ ] Shadows overlay on wallpaper (darker in shadowed regions)
- [ ] No color shifts or artifacts (hue preserved in shadows)
- [ ] Comparison image shows clear visual difference between with/without shadow
- [ ] 4 output files saved: original, wallpaper-only, wallpaper-with-shadow, comparison
- [ ] Follows experiment isolation pattern from `workflow-v3-research-driven/CLAUDE.md`
