# Implementation Plan: Resolution & Aspect Ratio Preservation

**Planning Date:** Oct 5, 2025
**Research Reference:** `research/02-resolution-aspect-ratio-analysis.md`
**Target:** Eliminate 400×400 constraint, preserve original H×W and aspect ratio

---

## Required Reading

**CLAUDE.md Files:**
- `workflow-v2-integrated-pipeline/modules/CLAUDE.md` - Section: Resolution Constraint (line 88), Pipeline Flow (lines 41-52)
- `workflow-v3-research-driven/CLAUDE.md` - Section: Resolution Preservation Pattern (lines 77-86)

**Reference Implementations:**
- `workflow-v2-integrated-pipeline/modules/end_to_end_pipeline.py:107-145` - Current pipeline pattern (THE BUG at 107-120)
- `workflow-v2-integrated-pipeline/modules/normal_computation.py:43-104` - Theta to normals transformation
- `workflow-v2-integrated-pipeline/modules/create_walls_simple.py:87-109` - Wall creation with normals

**Dependencies:**
- Module: `st_roomnet_model` - Function: `inference_from_array(array, model) -> (seg, theta, rgb)`
- Module: `segformer_model` - Function: `run_segformer_inference(pil_image, model, processor, device) -> confidence`
- Module: `simple_mask_combination` - Function: `combine_masks_simple(st_seg, seg_conf, threshold, labels) -> dict`
- Module: `normal_computation` - Function: `compute_normal_vectors(theta, labels) -> dict`
- Data structure: `Wall` - Constructor: `Wall(wall_id)`, then set: `pixel_mask`, `corners_2d`, `normal_vector`

---

## Files to Create/Modify

### New: `modules/preprocessing.py`
**Purpose:** Convert H×W input to 400×400 for ST-RoomNet (3 options)
**Pattern source:** PIL.Image resize/pad/crop operations
**Functions to implement:**
- `preprocess_option_a(image, target_size=400) -> (PIL.Image, metadata)` - Direct resize
- `preprocess_option_b(image, target_size=400) -> (PIL.Image, metadata)` - Letterbox padding
- `preprocess_option_c(image, target_size=400) -> (PIL.Image, metadata)` - Center crop

### New: `modules/postprocessing.py`
**Purpose:** Upscale 400×400 ST outputs to H×W and scale theta
**Pattern source:** `cv2.resize()` + homography conjugation
**Functions to implement:**
- `upscale_segmentation(seg_400, target_shape, metadata, option) -> seg_HxW` - Reverse preprocessing
- `scale_theta_parameters(theta_400, scale_factor, metadata, option) -> theta_scaled` - H_scaled = S @ H_400 @ S^-1

### New: `modules/hybrid_pipeline.py`
**Purpose:** Dual-resolution orchestrator (ST at 400×400, SegFormer at H×W)
**Pattern source:** `workflow-v2-integrated-pipeline/modules/end_to_end_pipeline.py:107-145`
**Functions to implement:**
- `process_dual_resolution(image_path, option, threshold, use_cached_models) -> (RoomWallLayout, dict)` - Main pipeline
- `initialize_models_v3() -> tuple` - Model caching from v2:37-57

### New: `compare_options.py`
**Purpose:** Driver script to test Options A/B/C side-by-side
**Pattern source:** CLI with argparse
**Functions to implement:**
- `run_comparison(input_path, output_dir, threshold) -> dict` - Run all options, save outputs
- `main()` - CLI entry point

---

## Implementation Steps

### Step 1: Create Preprocessing Module
**Files:** `modules/preprocessing.py`
**Actions:**
1. Implement Option A - `PIL.Image.resize((400, 400))`, return metadata with scale_x/scale_y
2. Implement Option B - Calculate padding to square, apply letterbox, return metadata with pad offsets
3. Implement Option C - Calculate center crop to min(H,W), return metadata with crop offsets
4. Test: Load 1024×1536, verify all return 400×400, save visualizations

**Success criteria:** All functions return 400×400 PIL Images, metadata contains correct transformation parameters

### Step 2: Create Postprocessing Module
**Files:** `modules/postprocessing.py`
**Actions:**
1. Implement `upscale_segmentation()` - Option A: direct resize, B: remove padding, C: place in canvas
2. Implement `scale_theta_parameters()` - Convert theta to 3×3 H, apply S @ H @ S^-1, extract 8 params
3. Test: Use known theta, scale by 2.5×, verify inverse recovers original
4. Test: Upscale 400×400 seg to 1024×1536, verify dimensions

**Success criteria:** Upscaled seg matches H×W, theta scaling preserves geometry (numerical precision <1e-6)

### Step 3: Create Hybrid Pipeline
**Files:** `modules/hybrid_pipeline.py`
**Actions:**
1. Copy `initialize_models_v3()` from `end_to_end_pipeline.py:37-57`
2. Implement `process_dual_resolution()` - Branch 1: preprocess → ST inference → upscale/scale
3. Continue - Branch 2: SegFormer at H×W
4. Combine at H×W using `combine_masks_simple()`
5. Call `compute_normal_vectors(theta_scaled, [1,2,3])` BEFORE creating walls
6. Create Wall objects, set `wall.normal_vector = normals_dict[label]`
7. Test: Run on 1024×1536, verify output dimensions, check normals magnitude ≈ 1.0

**Success criteria:** Output H×W (not 400×400), normal vectors valid, no wallpaper application errors

### Step 4: Create Comparison Script
**Files:** `compare_options.py`
**Actions:**
1. Implement `run_comparison()` - Call `process_dual_resolution()` for options A/B/C
2. Save outputs to `outputs/option-comparison/{option}/`
3. Measure metrics: resolution, aspect ratio, processing time
4. Generate comparison grid image
5. Implement `main()` with argparse (`--input`, `--output`, `--threshold`)

**Success criteria:** Script generates 3 output sets at H×W, comparison grid saved, metrics printed

### Step 5: Test on Diverse Images
**Files:** Execute `compare_options.py`
**Actions:**
1. Collect test images: portrait (1024×1536), landscape (1920×1080), square (1200×1200)
2. Run script on each with threshold=0.4
3. Visually inspect wallpaper sharpness vs v2 baseline
4. Measure edge precision at original vs 400×400
5. Document findings in `outputs/option-comparison/analysis.md`

**Success criteria:** All outputs H×W (not 400×400), wallpaper sharper than v2, aspect ratios preserved

### Step 6: Compute Quality Metrics
**Files:** Add to `compare_options.py`
**Actions:**
1. Add `compute_quality_metrics(v2_output, v3_output)` - SSIM, edge sharpness (Laplacian variance)
2. Run v2 baseline → save 400×400 outputs
3. Run v3 options → save H×W outputs
4. Compare metrics for each option vs v2
5. Generate `outputs/option-comparison/quality-report.md`

**Success criteria:** V3 shows >5× resolution improvement, edge sharpness >2× higher, aspect ratio error = 0%

### Step 7: Select Optimal Option
**Files:** Documentation update
**Actions:**
1. Review visual quality from Step 5
2. Review metrics from Step 6
3. Evaluate edge cases (Option B: black bars, Option C: content loss)
4. Document decision in this file (append)
5. Update `workflow-v3-research-driven/CLAUDE.md` with recommended option

**Success criteria:** Winner identified with evidence, CLAUDE.md updated with pattern

### Step 8: Integration and Validation
**Files:** `modules/end_to_end_pipeline.py`, new validation script
**Actions:**
1. Replace `end_to_end_pipeline.py:107-120` with call to `hybrid_pipeline.process_dual_resolution()`
2. Update function signature to accept `preprocessing_option` parameter
3. Create `test_resolution_preservation.py` - Automated validation (resolution, aspect ratio, normals)
4. Run regression test on 10 diverse images
5. Verify 100% pass rate

**Success criteria:** No 400×400 outputs, validation passes all checks, normal vectors valid

---

## Data Flow

```
Input (H×W) → Split
    ↓                           ↓
ST Branch                    Seg Branch
    ↓                           ↓
Preprocess → 400×400         Keep H×W
    ↓                           ↓
ST inference                 Seg inference
    ↓                           ↓
seg_400 + theta_400          confidence_HxW
    ↓
Upscale → seg_HxW
Scale → theta_HxW
    ↓                           ↓
    └─────── Combine ──────────┘
              ↓
    compute_normal_vectors(theta_HxW)
              ↓
    create walls with normals
              ↓
    RoomWallLayout (H×W)
```

---

## Testing Plan

**Per-step tests:**
- Step 1: Visual inspection, metadata accuracy
- Step 2: Numerical validation, dimension checks
- Step 3: End-to-end on single image, verify H×W
- Step 4: Script execution, file outputs exist
- Step 5: Visual quality, aspect ratio measurement
- Step 6: SSIM/edge sharpness computation
- Step 7: Decision matrix evaluation
- Step 8: Regression suite, automated validation

**Integration test:**
- Run: `python compare_options.py --input images/room1.jpeg --output outputs/option-comparison/`
- Expected: Three subdirectories with H×W outputs (1024×1536 for room1)
- Compare: Wallpaper sharper than v2, aspect ratio preserved, <15s processing

---

## Rollback Plan

If implementation fails:
1. Revert `end_to_end_pipeline.py` to commit 78f3489
2. Document failure in `research/02-resolution-aspect-ratio-analysis.md` Open Questions
3. Alternative: Implement 2048px resolution cap with controlled downscaling

---

## Success Criteria

- [ ] All preprocessing/postprocessing modules created
- [ ] Hybrid pipeline functional
- [ ] Comparison script generates outputs
- [ ] Visual quality improvement confirmed
- [ ] >5× resolution improvement measured
- [ ] Optimal option selected and documented
- [ ] Integration complete, validation passes
- [ ] No regressions
- [ ] CLAUDE.md updated