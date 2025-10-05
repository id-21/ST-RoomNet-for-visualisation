# Resolution & Aspect Ratio Degradation - Research Analysis

**Research Date:** Oct 4, 2025
**Research Type:** Bug Investigation
**Test Image/Data:** room1_E00_1.jpeg (1024×1536 → 400×400 output)
**Researcher:** Human/Claude

---

## Executive Summary

The v2 pipeline forces all processing to 400×400 resolution, resulting in 93.8% resolution loss (1.57M → 160K pixels) and aspect ratio distortion. Investigation reveals that only ST-RoomNet requires 400×400 input (trained on ref_img2.png reference), while SegFormer is resolution-agnostic. Recommended solution: dual-resolution hybrid pipeline where ST-RoomNet processes at 400×400 and SegFormer processes at original H×W, with final output at original resolution.

---

## CLAUDE.md Analysis

**Files Read:**
- `workflow-v2-integrated-pipeline/modules/CLAUDE.md` - Identified resolution constraint pattern and dual-model architecture
- `workflow-v3-research-driven/CLAUDE.md` - Confirmed this is priority issue, edge bleeding bug already fixed
- `CLAUDE.md` (root) - Overall dual-model architecture understanding

**Relevant Patterns Identified:**
- **Resolution Standardization** from `workflow-v2-integrated-pipeline/modules/CLAUDE.md` (line 88) - "All processing happens at 400×400 (ST-RoomNet output size); images auto-resized"
- **End-to-End Pipeline Flow** from `workflow-v2-integrated-pipeline/modules/CLAUDE.md` (lines 41-52) - Shows sequential processing at fixed resolution
- **Dual-Model Architecture** from `CLAUDE.md` (root, PATTERNS section) - ST-RoomNet + SegFormer combination pattern

**Existing Code to Reference:**
- `workflow-v2-integrated-pipeline/modules/end_to_end_pipeline.py:107-120` (documented in modules/CLAUDE.md) - Current resolution handling approach
- `workflow-v2-integrated-pipeline/modules/segformer_model.py:77-98` (documented in modules/CLAUDE.md) - SegFormer processes any resolution, then resizes to match ST output
- `workflow-v2-integrated-pipeline/modules/st_roomnet_model.py:92-96` (documented in modules/CLAUDE.md) - ST-RoomNet hardcoded to 400×400 input shape

---

## Problem Statement

**Observed Behavior:**
- v2 pipeline outputs 400×400 images regardless of input resolution
- Portrait images (1024×1536) distorted to square, losing aspect ratio
- Wallpaper patterns appear pixelated and blurry
- Edge precision limited to 400px boundaries

**Expected Behavior:**
- Output should match input resolution and aspect ratio
- Wallpaper should be sharp at full resolution
- Edge precision should leverage full image resolution

**Root Cause Location:**
`workflow-v3-research-driven/modules/end_to_end_pipeline.py:107-120`

---

## Root Cause Analysis

### Code Investigation

**File:** `workflow-v3-research-driven/modules/end_to_end_pipeline.py:107-120`

**The Issue:**
The pipeline resizes all images to 400×400 before any processing, then maintains this resolution throughout. Both ST-RoomNet inference (line 107-109) and SegFormer inference (line 111-115) receive 400×400 input. SegFormer output is further resized to match ST-RoomNet dimensions (line 117-120), permanently constraining the entire pipeline to 400×400.

**Supporting Evidence:**
- `st_roomnet_model.py:92-96` - ST-RoomNet model architecture hardcoded with `input_shape=(400, 400, 3)`
- `st_roomnet_model.py:33-56` - Reference image (`ref_img2.png`) loaded at 400×400 for spatial transformation
- `segformer_model.py:93-98` - SegFormer resizes output to match original input size, but receives 400×400 input from pipeline

### Why This Is Wrong

**What actually happens:**
1. Input image loaded and resized to 400×400 (PIL.Image.open().resize())
2. ST-RoomNet processes 400×400 → outputs 400×400 segmentation + theta
3. SegFormer processes 400×400 PIL image → outputs confidence map resized to 400×400
4. Masks combined at 400×400
5. Wallpaper applied at 400×400
6. Final output is 400×400 (93.8% resolution loss for 1024×1536 input)

**The contradiction:**
- **Intended behavior**: ST-RoomNet requires 400×400 due to trained model architecture, SegFormer should leverage its resolution-agnostic transformer architecture
- **Actual behavior**: SegFormer receives downscaled 400×400 input, wasting its capability to process high-resolution images with full detail

---

## Context & Background

### Why This Code Exists

The 400×400 constraint exists because ST-RoomNet was trained with a fixed input size and uses a reference image (ref_img2.png) at 400×400 for spatial transformation. The original v2 implementation assumed both models needed the same resolution for mask combination.

**Likely assumptions:**
1. Both models must process the same resolution for mask alignment (not true - masks can be upscaled)
2. ST-RoomNet could process any resolution (not true - fixed at 400×400 by architecture)
3. Resolution loss was acceptable for wallpaper visualization (not true - produces pixelated results)

**Why the assumptions were wrong:**
- SegFormer is transformer-based and resolution-agnostic - can process original H×W and output matching dimensions
- Masks can be combined at different resolutions via upscaling ST-RoomNet output to match SegFormer
- Theta transformation parameters can be scaled mathematically via homography conjugation
- Visual quality severely degraded by 400×400 constraint

---

## Key Insight: Resolution Requirements Analysis

**ST-RoomNet Requirements:**
- `st_roomnet_model.py:92-96` - Model architecture: `ConvNeXtTiny(input_shape=(400, 400, 3))`
- `st_roomnet_model.py:33-56` - Reference image: 400×400 cuboid (ref_img2.png)
- `spatial_transformer_fixed.py` - ProjectiveTransformerLayer expects 400×400 output
- **Conclusion**: MUST process at 400×400, non-negotiable

**SegFormer Capabilities:**
- `segformer_model.py:59-100` - `run_segformer_inference()` accepts PIL Image of any size
- `segformer_model.py:88-90` - Transformer processes at model's internal resolution, outputs logits
- `segformer_model.py:93-98` - Automatically resizes output to match input dimensions
- **Conclusion**: Can process ANY resolution, resolution-agnostic

**Mask Combination:**
- `simple_mask_combination.py:20-36` - Combines via AND operation, requires matching dimensions
- Can upscale ST-RoomNet 400×400 to H×W, then combine with SegFormer H×W
- No technical barrier to different processing resolutions

---

## Solution Approaches: Three Preprocessing Options

### Option A: Direct Resize (Distortion Acceptable)
```
Input (H×W) → Resize to 400×400 (distorted) → ST-RoomNet
           → Upscale seg to H×W → Combine with SegFormer
```
**Pros:** Simple, no content loss, current approach
**Cons:** ST-RoomNet sees distorted geometry during inference

### Option B: Padding (Letterbox Preservation)
```
Input (H×W) → Pad shorter side to square (letterbox)
           → Resize to 400×400 (no distortion)
           → ST-RoomNet inference
           → Remove padding regions from output
           → Upscale to original H×W → Combine with SegFormer
```
**Pros:** Preserves aspect ratio during inference, no content loss
**Cons:** ST-RoomNet sees black bars (not in training data), reduced effective resolution

### Option C: Center Crop (Aspect Ratio Preserved, Content Lost)
```
Input (H×W) → Center crop to min(H,W) × min(H,W) square
           → Resize to 400×400 (no distortion)
           → ST-RoomNet inference
           → Apply only to cropped region
           → Combine with SegFormer on full H×W
```
**Pros:** No distortion, no artificial padding, clean square input
**Cons:** Loses ceiling/floor content (critical for room layout), may miss wall edges

### Unified Architecture (All Options)
```
Input (H×W, original) → Split into two branches:
│
├─ ST-RoomNet: Preprocess (A/B/C) → 400×400 → seg_400 + theta_400
│                                 → Postprocess (A/B/C) → seg_HxW
│
└─ SegFormer: Keep original H×W → confidence_HxW
│
→ Combine at H×W: (seg_HxW AND confidence > threshold)
→ Scale theta: H_scaled = S @ H_400 @ S⁻¹
→ Apply wallpaper at H×W → Output (H×W, full resolution)
```

---

## Quantitative Impact

**Test Case:** room1_E00_1.jpeg (1024×1536 portrait)

| Metric | Current (v2) | Hybrid (v3) | Improvement |
|--------|--------------|-------------|-------------|
| Resolution | 400×400 (160K px) | 1024×1536 (1.57M px) | 9.8× |
| Aspect Ratio | 1:1 (distorted) | 2:3 (preserved) | ✓ |
| Edge Precision | 400px grid | 1024×1536 grid | 2.5-4× |
| Wallpaper Quality | Pixelated | Sharp | Visual ✓ |

**Resolution Loss Calculation:**
- Input: 1024×1536 = 1,572,864 pixels
- v2 Output: 400×400 = 160,000 pixels
- Loss: (1,572,864 - 160,000) / 1,572,864 = 93.8%

---

## Test Methodology

### Reproducibility

**Test Images:**
- Portrait: room1_E00_1.jpeg (1024×1536)
- Landscape: [need 1920×1080 sample]
- Square: [need 1200×1200 sample]
- Ultra-wide: [need 2560×1080 sample]

**Process:**
1. Run v2 baseline on test image → save 400×400 output
2. Run v3 hybrid pipeline with each option (A/B/C) → save H×W output
3. Compare wallpaper sharpness visually
4. Measure edge precision quantitatively
5. Validate aspect ratio preservation

**To reproduce:**
```bash
cd workflow-v3-research-driven
source venv/bin/activate
python compare_options.py --input images/room1.jpeg --output outputs/option-comparison/
```

**Expected output:**
- Three H×W outputs (not 400×400)
- Sharp wallpaper patterns
- Preserved aspect ratio
- Processing time <15s

---

## Next Steps

**Planning Phase** (create `planning/fix-resolution-aspect-ratio.md`):
- ✅ COMPLETED - Implementation plan created with detailed code references
- Evaluate solution approaches (Options A/B/C)
- Design test cases for validation
- Define success metrics
- Create implementation roadmap

**Implementation Phase:**
1. Create `modules/preprocessing.py` - Three preprocessing options
2. Create `modules/postprocessing.py` - Upscaling + theta coordinate scaling
3. Create `modules/hybrid_pipeline.py` - Dual-resolution orchestrator
4. Create `compare_options.py` - Driver script for A/B/C comparison
5. Test on diverse image set
6. Visual comparison with v2 baseline

---

## Open Questions

- [x] Can theta parameters be scaled mathematically? **YES** - via homography conjugation (see planning doc)
- [x] Will normal vectors remain valid after theta scaling? **YES** - inverse-transpose transformation preserved
- [ ] Which option (A/B/C) produces best visual quality? **Requires implementation testing**
- [ ] What is maximum safe resolution before OOM? **Requires testing, implement cap at 2048px**
- [ ] How much does processing time increase at full resolution? **Measure during testing**

---

## References

**CLAUDE.md Files:**
- `workflow-v2-integrated-pipeline/modules/CLAUDE.md` - Sections: Resolution Constraint (line 88), End-to-End Pipeline Flow (lines 41-52), Normal Vector Transformation (lines 67-72)
- `workflow-v3-research-driven/CLAUDE.md` - Sections: CRITICAL ISSUES STATUS (lines 8-20), Resolution Preservation Pattern (lines 77-86)
- `CLAUDE.md` (root) - Sections: Dual-Model Architecture (PATTERNS), Resolution Standardization (PATTERNS)

**Code Locations:**
- Primary: `workflow-v3-research-driven/modules/end_to_end_pipeline.py:107-120`
- ST-RoomNet architecture: `workflow-v3-research-driven/modules/st_roomnet_model.py:92-96`
- SegFormer capability: `workflow-v3-research-driven/modules/segformer_model.py:77-98`
- Mask combination: `workflow-v3-research-driven/modules/simple_mask_combination.py:20-36`

**Output Artifacts:**
- v2 baseline outputs: `outputs/v2-baseline-room1/`
- Current comparison: `outputs/current-vs-industry/`
- Planning document: `planning/fix-resolution-aspect-ratio.md`

**Related Research:**
- Edge bleeding bug research: COMPLETED, fixed via Option A (removed edge refinement)
- Normal vector accuracy: Separate issue, not addressed here

---

## Conclusion

The 400×400 resolution constraint is an artificial limitation imposed by aligning SegFormer processing with ST-RoomNet requirements. ST-RoomNet must process at 400×400 due to architectural constraints, but SegFormer is fully capable of processing at original resolution. A hybrid dual-resolution pipeline will preserve 93.8% more pixels (for 1024×1536 input), maintain aspect ratio, and deliver sharp wallpaper output. Implementation should proceed immediately with Option A as baseline, then compare with Options B and C for optimal visual quality.
