# Edge Bleeding Analysis - V2 Pipeline Bug

**Research Date:** October 4, 2025
**Test Image:** `test_images/room1.jpeg`
**Researcher:** Claude Code (Automated Analysis)

## Executive Summary

The v2 pipeline contains a **critical bug** in edge refinement that causes ST-RoomNet trapezoid boundaries to override SegFormer semantic understanding. This results in **up to 94% of refined pixels** contradicting the semantic segmentation model, especially for walls with small initial masks.

**Impact:** High - Affects 100% of walls, completely undermines the dual-model architecture's purpose.

---

## Problem Statement

**Observed Behavior:**
ST-RoomNet geometric boundaries "bleed through" into the final wall mask even in areas where SegFormer confidently identifies non-wall surfaces (windows, doors, furniture, shadows).

**Expected Behavior:**
Final mask should respect semantic understanding - only include pixels where BOTH models agree a wall exists.

**Root Cause Location:**
`workflow-v2-integrated-pipeline/modules/edge_refinement_simple.py:31`

---

## Visual Evidence

### Pipeline Flow with Bug Location

```
Input Image (room1.jpeg)
    ↓
ST-RoomNet → Geometric Segmentation (trapezoid planes)
    ↓
SegFormer → Semantic Confidence Map (wall vs non-wall)
    ↓
combine_masks_simple() → Combined = ST_plane AND SegFormer > 0.4  ✓ CORRECT
    ↓
refine_edges_with_st_boundaries() → refined[edge_pixels] = st_plane[edge_pixels]  ✗ BUG!
    ↓
Final Output (with bleeding artifacts)
```

### Output Files Location
All visual evidence saved in: `outputs/v2-baseline-room1/`

**Key Visualizations:**
- `00_original_input.png` - Input room image
- `01_st_roomnet_segmentation.png` - ST-RoomNet geometric planes (labels 0-4)
- `02_segformer_confidence.png` - SegFormer wall confidence (heatmap)
- `02b_segformer_thresholded.png` - Binary SegFormer mask (threshold=0.4)
- `wall_X_*/f_comparison.png` - Side-by-side showing bug impact (RED = contradictions)

---

## Quantitative Analysis

### Metrics Summary

| Wall | Combined Pixels | Refined Pixels | Added by Refinement | Contradicts SegFormer | % Contradicting | Within 5px Buffer |
|------|----------------|----------------|---------------------|----------------------|-----------------|-------------------|
| **Left Wall** | 46,508 | 48,229 | 1,751 | **1,749** | **3.6%** | 98.4% |
| **Front Wall** | 186 | 1,408 | 1,242 | **1,227** | **87.1%** | 99.8% |
| **Right Wall** | 10 | 72 | 72 | **68** | **94.4%** | 100.0% |

### Key Findings

1. **All walls affected**: 100% of detected walls show edge bleeding
2. **Severe for small masks**: Front wall (87%) and right wall (94%) have majority of pixels contradicting semantics
3. **Spatially localized**: 98-100% of contradicting pixels are within 5px of ST-RoomNet boundaries (as expected from buffer zone)
4. **Total contradicting pixels**: 3,044 pixels across all walls violate semantic understanding

### Interpretation

- **Left wall** (large mask): 3.6% contamination is "acceptable" but still incorrect
- **Front/Right walls** (tiny masks): Almost entirely composed of bleeding artifacts - the combined AND operation correctly rejected these areas, but refinement re-added them

---

## Root Cause Analysis

### The Buggy Code

**File:** `workflow-v2-integrated-pipeline/modules/edge_refinement_simple.py`

```python
def refine_edges_with_st_boundaries(combined_mask, st_plane_mask):
    """Use ST-RoomNet boundaries to clean up edges"""

    # Convert to uint8 for OpenCV operations
    combined_uint8 = combined_mask.astype(np.uint8) * 255
    st_uint8 = st_plane_mask.astype(np.uint8) * 255

    # Extract ST-RoomNet boundaries
    st_edges = cv2.Canny(st_uint8, 100, 200)  # Line 21

    # Create buffer zone around edges (5 pixels)
    edge_buffer = cv2.dilate(st_edges, np.ones((5, 5), np.uint8))  # Line 24

    # Copy combined mask
    refined = combined_uint8.copy()

    # In edge buffer zone, use ST-RoomNet boundaries
    edge_pixels = edge_buffer > 0  # Line 30
    refined[edge_pixels] = st_uint8[edge_pixels]  # Line 31 ← THE BUG!

    # Final smoothing
    refined = cv2.medianBlur(refined, 3)

    return refined > 0
```

### Why This Is Wrong

**Line 31 explanation:**
```python
refined[edge_pixels] = st_uint8[edge_pixels]
```

This line **directly copies** ST-RoomNet plane values into the refined mask within a 5-pixel buffer around detected edges, **completely overriding** the earlier AND operation.

**What actually happens:**

1. Line 124 in `end_to_end_pipeline.py`: `combined = st_plane & segformer_wall_mask` ✓
   - Correctly combines both models: "wall exists if ST says wall AND SegFormer says wall"

2. Line 131 calls `refine_edges_with_st_boundaries(combined, st_plane)`

3. Line 31 in edge refinement: Within 5px of ST boundaries, **ignore SegFormer entirely**
   - If ST-RoomNet detected a trapezoid edge at pixel (x, y), that pixel becomes "wall" regardless of what SegFormer detected
   - This includes pixels where SegFormer confidence is 0.0 (definitely not wall)

**The contradiction:**
- **Purpose of dual-model:** Use geometry (ST) + semantics (SegFormer) for robust detection
- **Actual behavior:** Geometry overrides semantics at boundaries (where errors are most likely!)

---

## Why Edges Are Where Errors Happen

ST-RoomNet's trapezoid boundaries frequently pass through:
- Window frames (geometry detects plane boundary, semantics says "glass, not wall")
- Door frames (geometry boundary, semantics says "door")
- Furniture near walls (geometry approximation error)
- Shadows and lighting transitions (texture changes that aren't physical boundaries)

**The irony:** Edge zones are exactly where we MOST need semantic validation, but the current code disables it!

---

## Pipeline Stage Comparison

### Stage 1: ST-RoomNet Segmentation
- Left wall: Large trapezoid covering left side of image
- Front wall: Tiny trapezoid (correctly small - SegFormer will reject most of it)
- Right wall: Minimal detection (correctly minimal)

### Stage 2: SegFormer Confidence
- High confidence (>0.8) on actual painted wall surfaces
- Low confidence (<0.2) on windows, furniture, floor
- Medium confidence (0.3-0.6) on textured areas

### Stage 3: Combined Mask (ST AND SegFormer)
- **Left wall:** 46,508 pixels - correctly excludes windows/furniture within ST trapezoid
- **Front wall:** 186 pixels - correctly rejects most of ST's over-detection
- **Right wall:** 10 pixels - correctly identifies minimal visible wall

### Stage 4: Refined Mask (BUGGY)
- **Left wall:** 48,229 pixels (+1,751) - adds back 1,749 pixels SegFormer rejected
- **Front wall:** 1,408 pixels (+1,242) - adds back 1,227 pixels SegFormer rejected (7.5x inflation!)
- **Right wall:** 72 pixels (+72) - adds back 68 pixels SegFormer rejected (7.2x inflation!)

**Conclusion:** Refinement **degrades** mask quality rather than improving it.

---

## Distance Analysis

All contradicting pixels fall within the expected 5-pixel edge buffer:
- Left wall: 98.4% within buffer
- Front wall: 99.8% within buffer
- Right wall: 100% within buffer

**Interpretation:** The bug is behaving exactly as coded - it's not a sporadic error, it's systematic override behavior within the dilated edge zone.

---

## Test Methodology

### Reproducibility

**Script:** `experiments/v2-baseline-analysis/run_v2_baseline.py`

**Process:**
1. Load test image (`room1.jpeg`)
2. Run ST-RoomNet inference (segmentation + theta)
3. Run SegFormer inference (confidence map)
4. Combine masks with AND operation
5. Apply edge refinement (buggy step)
6. Compare refined vs combined masks
7. Identify pixels where: `refined[i] == True AND segformer_confidence[i] < threshold`
8. Measure spatial distribution relative to ST boundaries

**To reproduce:**
```bash
source venv/bin/activate
cd workflow-v3-research-driven/experiments/v2-baseline-analysis
python3 run_v2_baseline.py
```

---

## Hypothesis for Original Intent

**Why was this code written this way?**

Likely assumptions:
1. ST-RoomNet boundaries are more precise than SegFormer edges (not validated)
2. SegFormer might have "ragged" edges that need geometric constraint (may be true)
3. Edge buffer approach would "snap" SegFormer edges to ST boundaries (intended effect)

**Why the assumptions were wrong:**
1. ST-RoomNet trapezoids are **approximations** - boundaries aren't ground truth
2. SegFormer's "ragged" edges often represent **real semantic boundaries** (window edges, door frames)
3. "Snapping" to geometric boundaries loses semantic information (defeats dual-model purpose)

---

## Impact Assessment

### Severity: **CRITICAL**

**Functional Impact:**
- Wallpaper will be applied to non-wall surfaces (windows, doors, furniture)
- Visual quality severely degraded for rooms with small wall visibility
- User trust broken when obvious errors appear in output

**Architectural Impact:**
- Completely undermines the dual-model approach
- Makes SegFormer essentially useless at boundaries (where it's most valuable)
- Creates false confidence in output quality

**Affected Use Cases:**
- ✗ Rooms with large windows (windows will get wallpaper)
- ✗ Rooms with doors in frame (door surfaces misclassified)
- ✗ Rooms with furniture near walls (furniture surfaces included)
- ✓ Empty rooms with uniform walls (bug impact minimal but still present)

---

## Proposed Solutions (Preliminary)

### Option A: Remove Edge Refinement Entirely
**Hypothesis:** Edge refinement may not be necessary.

**Test:** Run pipeline without refinement, compare edge quality.

**Pros:**
- Eliminates bug completely
- Simplifies pipeline
- Maintains semantic integrity

**Cons:**
- May result in slightly jagged edges
- Loses potential benefit of geometric guidance

---

### Option B: Weighted Combination Instead of Override
**Hypothesis:** Blend ST and SegFormer confidence at edges rather than hard override.

**Implementation:**
```python
# Instead of:
refined[edge_pixels] = st_uint8[edge_pixels]

# Use:
segformer_weight = 0.8  # Trust semantics more
st_weight = 0.2         # Use geometry as hint
refined[edge_pixels] = (
    segformer_confidence[edge_pixels] * segformer_weight +
    st_plane_mask[edge_pixels] * st_weight
) > threshold
```

**Pros:**
- Respects both models
- Maintains some geometric guidance
- Adjustable weights for tuning

**Cons:**
- Introduces new hyperparameters
- May still allow some bleeding if weights wrong

---

### Option C: SegFormer Confidence Gating
**Hypothesis:** Only use ST boundaries where SegFormer is uncertain.

**Implementation:**
```python
# Only override where SegFormer confidence is close to threshold
uncertainty_zone = (segformer_confidence > threshold - 0.1) & (segformer_confidence < threshold + 0.1)
edge_and_uncertain = edge_pixels & uncertainty_zone

# Only apply ST guidance in uncertain areas
refined[edge_and_uncertain] = st_uint8[edge_and_uncertain]
```

**Pros:**
- Preserves semantic certainty
- Uses geometry only for ambiguous cases
- Theoretically sound approach

**Cons:**
- Complex logic
- Threshold selection critical
- May have edge cases at confidence boundaries

---

## Next Steps

1. **Planning Phase** (create `planning/fix-edge-bleeding.md`):
   - Evaluate each solution option with detailed analysis
   - Design test cases for each approach
   - Define success metrics (edge smoothness, semantic accuracy, visual quality)
   - Rank solutions by viability

2. **Implementation Phase** (after planning approval):
   - Create git worktree: `experiments/edge-fix-option-[A/B/C]`
   - Implement each solution in isolated branch
   - Run quantitative comparison on 5+ test images
   - Visual inspection and user testing
   - Select best approach based on data

3. **Validation Phase:**
   - Ensure zero pixels contradict SegFormer (except true ambiguous cases)
   - Measure edge smoothness (compare to v2 baseline)
   - Performance testing (ensure no slowdown)
   - Integration testing with wallpaper application

---

## References

**Code Locations:**
- Bug location: `workflow-v2-integrated-pipeline/modules/edge_refinement_simple.py:31`
- Pipeline caller: `workflow-v2-integrated-pipeline/modules/end_to_end_pipeline.py:131`
- Mask combination: `workflow-v2-integrated-pipeline/modules/simple_mask_combination.py:30`

**Output Artifacts:**
- Visual evidence: `workflow-v3-research-driven/outputs/v2-baseline-room1/`
- Metrics: `workflow-v3-research-driven/outputs/v2-baseline-room1/metrics_summary.txt`
- Test script: `workflow-v3-research-driven/experiments/v2-baseline-analysis/run_v2_baseline.py`

**Related Issues:**
- Resolution degradation (separate issue - 400x400 processing)
- Normal vector inaccuracy (separate issue - theta transformation)

---

## Conclusion

The edge bleeding bug is a **systematic architectural flaw** that contradicts the core principle of the dual-model approach. The quantitative evidence shows:

- **3,044 pixels** across one test image violate semantic understanding
- **87-94%** of small walls are contaminated artifacts
- **100%** of walls are affected to some degree

**Recommendation:** Prioritize fix immediately before any other v3 improvements. This bug undermines trust in the entire system and produces visually obvious errors in output.

The fix is straightforward (remove or modify 1 line of code), but requires careful testing to ensure edge quality is maintained while respecting semantic boundaries.
