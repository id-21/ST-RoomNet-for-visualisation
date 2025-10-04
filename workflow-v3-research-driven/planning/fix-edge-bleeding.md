# Edge Bleeding Fix - Implementation Planning

**Planning Date:** October 4, 2025
**Research Reference:** `research/01-edge-bleeding-analysis.md`
**Target:** Fix `modules/edge_refinement_simple.py:31` bug

---

## Problem Summary

**Bug:** ST-RoomNet geometric boundaries override SegFormer semantic understanding at edges.

**Impact:** 3,044 pixels incorrectly classified across test image, up to 94% contamination on small walls.

**Root Cause:** Line 31 in `edge_refinement_simple.py` directly copies ST-RoomNet values within 5px edge buffer, ignoring SegFormer confidence.

---

## Solution Options (Ranked by Viability)

### 🥇 Option A: Remove Edge Refinement Entirely

**Hypothesis:** Edge refinement may be unnecessary - the combined AND mask is already correct.

**Implementation:**
```python
# In end_to_end_pipeline.py, line 131:
# BEFORE (buggy):
refined = refine_edges_with_st_boundaries(mask, st_plane)

# AFTER (test):
refined = mask  # Use combined mask directly, no refinement
```

**Pros:**
- ✅ Simplest fix (1 line change)
- ✅ Eliminates bug completely
- ✅ Maintains semantic integrity 100%
- ✅ Faster processing (skip refinement step)
- ✅ No new hyperparameters

**Cons:**
- ❓ May result in slightly jagged edges (NEEDS TESTING)
- ❓ Loses potential geometric guidance (may not be valuable anyway)

**Test Plan:**
1. Run pipeline on 5 test images without refinement
2. Compare edge quality visually (zoom to pixel level)
3. Measure edge smoothness: compute edge pixel count variance
4. If edges acceptable → this is the solution ✓

**Risk Assessment:** LOW
- Worst case: Edges are jagged → try Option B instead
- No risk of incorrect semantic classification

**Success Metrics:**
- Zero pixels contradict SegFormer (already achieved by combined mask)
- Edge smoothness within 5% of v2 baseline (acceptable trade-off)
- Visual inspection passes quality check

---

### 🥈 Option B: Weighted Combination (Semantic Priority)

**Hypothesis:** Blend ST geometry and SegFormer semantics at edges with semantic priority.

**Implementation:**
```python
def refine_edges_with_weighted_combination(combined_mask, st_plane_mask,
                                           segformer_confidence, threshold=0.4,
                                           semantic_weight=0.8, geometric_weight=0.2):
    """
    Use weighted combination at edges, prioritizing semantic understanding.

    Args:
        combined_mask: Combined ST AND SegFormer mask
        st_plane_mask: ST-RoomNet plane mask
        segformer_confidence: SegFormer confidence map (0-1)
        threshold: SegFormer threshold used
        semantic_weight: Weight for SegFormer (default: 0.8)
        geometric_weight: Weight for ST-RoomNet (default: 0.2)
    """
    combined_uint8 = combined_mask.astype(np.uint8) * 255
    st_uint8 = st_plane_mask.astype(np.uint8) * 255

    # Extract ST-RoomNet boundaries
    st_edges = cv2.Canny(st_uint8, 100, 200)
    edge_buffer = cv2.dilate(st_edges, np.ones((5, 5), np.uint8))

    # Start with combined mask
    refined = combined_uint8.copy()

    # At edges: weighted combination of confidence values
    edge_pixels = edge_buffer > 0

    # Normalize ST plane to 0-1 for fair comparison
    st_confidence = st_plane_mask.astype(np.float32)

    # Weighted combination
    combined_confidence = (
        segformer_confidence[edge_pixels] * semantic_weight +
        st_confidence[edge_pixels] * geometric_weight
    )

    # Apply threshold
    refined[edge_pixels] = (combined_confidence > threshold).astype(np.uint8) * 255

    # Final smoothing
    refined = cv2.medianBlur(refined, 3)

    return refined > 0
```

**Pros:**
- ✅ Respects semantic understanding (80% weight)
- ✅ Maintains some geometric guidance (20% weight)
- ✅ Adjustable weights for fine-tuning
- ✅ Should produce smooth edges

**Cons:**
- ⚠️ Introduces hyperparameters (semantic_weight, geometric_weight)
- ⚠️ More complex than Option A
- ⚠️ May still allow some bleeding if weights suboptimal
- ⚠️ Requires confidence normalization

**Test Plan:**
1. Implement with default weights (0.8 semantic, 0.2 geometric)
2. Test on 5 images, measure contradiction pixels
3. If contradictions remain, adjust weights (try 0.9/0.1, then 0.95/0.05)
4. Compare edge smoothness vs Option A and v2 baseline
5. Visual quality inspection

**Risk Assessment:** MEDIUM
- Risk: Weight selection may be image-dependent
- Mitigation: Use adaptive weights based on confidence variance
- Risk: Still allows some semantic override if SegFormer confidence near threshold
- Mitigation: Add confidence gate (see Option C)

**Success Metrics:**
- <1% pixels contradicting SegFormer across test set
- Edge smoothness equal to or better than v2 baseline
- Visually smooth edges without artifacts

---

### 🥉 Option C: Confidence-Gated Refinement

**Hypothesis:** Only apply ST geometric guidance where SegFormer is genuinely uncertain.

**Implementation:**
```python
def refine_edges_with_confidence_gate(combined_mask, st_plane_mask,
                                     segformer_confidence, threshold=0.4,
                                     uncertainty_margin=0.1):
    """
    Use ST boundaries only in zones where SegFormer confidence is ambiguous.

    Args:
        combined_mask: Combined ST AND SegFormer mask
        st_plane_mask: ST-RoomNet plane mask
        segformer_confidence: SegFormer confidence map (0-1)
        threshold: SegFormer threshold
        uncertainty_margin: Confidence range around threshold to consider "uncertain"
    """
    combined_uint8 = combined_mask.astype(np.uint8) * 255
    st_uint8 = st_plane_mask.astype(np.uint8) * 255

    # Extract ST-RoomNet boundaries
    st_edges = cv2.Canny(st_uint8, 100, 200)
    edge_buffer = cv2.dilate(st_edges, np.ones((5, 5), np.uint8))

    # Define uncertainty zone: confidence close to threshold
    lower_bound = threshold - uncertainty_margin  # e.g., 0.3
    upper_bound = threshold + uncertainty_margin  # e.g., 0.5
    uncertainty_zone = (segformer_confidence >= lower_bound) & (segformer_confidence <= upper_bound)

    # Only apply geometric guidance in uncertain edge pixels
    edge_and_uncertain = (edge_buffer > 0) & uncertainty_zone

    # Start with combined mask
    refined = combined_uint8.copy()

    # Override ONLY where both at edge AND SegFormer uncertain
    refined[edge_and_uncertain] = st_uint8[edge_and_uncertain]

    # Final smoothing
    refined = cv2.medianBlur(refined, 3)

    return refined > 0
```

**Pros:**
- ✅ Theoretically sound: uses geometry only for ambiguous cases
- ✅ Preserves high-confidence SegFormer decisions
- ✅ Should eliminate most contradictions

**Cons:**
- ⚠️ Complex logic with edge cases
- ⚠️ Introduces uncertainty_margin hyperparameter
- ⚠️ May produce discontinuous edges (pixels treated differently based on confidence)
- ⚠️ Threshold selection critical - wrong margin causes problems

**Test Plan:**
1. Implement with default margin (0.1)
2. Visualize uncertainty zones on test images (are they where we expect?)
3. Measure contradictions (should be near zero for high-confidence areas)
4. Test various margins: 0.05, 0.1, 0.15, 0.2
5. Check for edge discontinuities at confidence boundaries

**Risk Assessment:** HIGH
- Risk: Edge discontinuities create visual artifacts
- Mitigation: Add morphological closing to smooth transitions
- Risk: Margin too small → no geometric help; too large → reintroduces bug
- Mitigation: Adaptive margin based on per-wall confidence distribution
- Risk: Complexity makes debugging difficult

**Success Metrics:**
- <0.5% pixels contradict high-confidence SegFormer (>0.6)
- No visible edge discontinuities
- Edge smoothness comparable to v2

---

## Recommended Approach

### Phase 1: Quick Win (Option A)
**Timeline:** 1-2 hours

1. Test removing edge refinement entirely
2. If edges look acceptable → DONE, ship it ✓
3. If edges unacceptable → proceed to Phase 2

**Rationale:** Option A is the simplest fix with zero risk of semantic errors. We should validate whether edge refinement is even necessary before adding complexity.

### Phase 2: Enhanced Refinement (Option B)
**Timeline:** 3-4 hours

If Option A produces jagged edges:
1. Implement weighted combination
2. Start with 0.9 semantic / 0.1 geometric
3. Tune weights based on test results
4. If satisfactory → DONE ✓
5. If still problematic → proceed to Phase 3

**Rationale:** Option B balances semantic accuracy with edge smoothing. Lower risk than Option C while maintaining quality.

### Phase 3: Advanced (Option C) - ONLY IF NEEDED
**Timeline:** 5-6 hours

Only pursue if Option B fails:
1. Implement confidence gating
2. Extensive testing with various margins
3. Add morphological smoothing for edge continuity
4. May require per-wall adaptive parameters

**Rationale:** High complexity, only justified if simpler approaches fail.

---

## Test Cases

### Test Image Set
Use diverse room types to ensure robustness:

1. **room1.jpeg** (current test image) - Baseline reference
2. **Large window room** - Tests window boundary handling
3. **Door frame visible** - Tests door vs wall classification
4. **Furniture near wall** - Tests object rejection
5. **Empty uniform room** - Control case (should work well for all options)

### Quantitative Metrics

For each option and test image, measure:

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Pixels contradicting SegFormer | <1% of wall area | Count: `refined[i] == True AND segformer_conf[i] < threshold - 0.1` |
| Edge smoothness | Within 5% of v2 | Edge pixel count variance: `var(edge_pixel_counts)` |
| Processing time | <10% slower than v2 | Time refinement step |
| Wall area accuracy | ±2% of ground truth | Manual annotation vs output |

### Visual Quality Checklist

For each output, verify:
- [ ] No obvious wallpaper on windows
- [ ] No obvious wallpaper on doors
- [ ] Wall boundaries look smooth (no jagged artifacts)
- [ ] Corners properly defined
- [ ] No floating isolated regions

---

## Implementation Workflow

### Step 1: Create Experiment Branch
```bash
cd /path/to/repo
git worktree add ../experiments/edge-fix-option-a research/v3-edge-bleeding-analysis
cd ../experiments/edge-fix-option-a
```

### Step 2: Implement Fix (Option A first)
```bash
# Edit modules/end_to_end_pipeline.py
# Line 131: refined = mask  # Remove refinement call
```

### Step 3: Run Comparative Testing
```bash
# Create test script: experiments/test_edge_fixes.py
# Run on all 5 test images
# Save outputs to: outputs/option-a-results/

python3 experiments/test_edge_fixes.py --option A --output-dir outputs/option-a-results/
```

### Step 4: Analyze Results
```bash
# Generate comparison report
python3 experiments/compare_fix_options.py --baseline v2-baseline-room1 --test option-a-results
```

### Step 5: Decision Point
- If Option A passes → integrate to main workflow
- If fails → implement Option B, repeat Steps 3-5
- If Option B fails → implement Option C, repeat Steps 3-5

### Step 6: Integration
```bash
# Copy fixed module to workflow-v3/modules/
# Run full pipeline test
# Update documentation
# Commit with results
```

---

## Success Criteria

The fix is considered successful if:

1. **Zero critical contradictions:** No pixels where SegFormer confidence <0.2 but refined mask includes them
2. **Minimal ambiguous contradictions:** <1% pixels in uncertainty zone (0.3-0.5 confidence)
3. **Edge quality maintained:** No visually obvious jagged artifacts
4. **Performance acceptable:** <10% processing time increase
5. **Generalizes well:** Works across all 5 test images

---

## Rollback Plan

If all options fail to meet success criteria:

1. **Document why:** What specific issues prevented success?
2. **Revert to v2 behavior:** Keep edge refinement but add warning in docs
3. **Alternative approach:** Investigate post-processing smoothing instead of edge refinement
4. **Research path:** May need to retrain ST-RoomNet with different architecture

---

## Dependencies

**Code files to modify:**
- `workflow-v3-research-driven/modules/edge_refinement_simple.py` (fix location)
- `workflow-v3-research-driven/modules/end_to_end_pipeline.py` (caller, line 131 & 211)

**Test infrastructure needed:**
- `experiments/test_edge_fixes.py` - Run all options systematically
- `experiments/compare_fix_options.py` - Generate comparison reports
- Test image set (5 diverse room images)

**No external dependencies:** All required libraries already in environment (cv2, numpy).

---

## Timeline Estimate

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Option A implementation | 30 min | Modified pipeline code |
| Option A testing | 1 hour | Test results for 5 images |
| Option A analysis | 30 min | Decision: pass/fail |
| **Option A total** | **2 hours** | **Go/no-go decision** |
| Option B implementation | 1 hour | Weighted combination code |
| Option B testing | 2 hours | Hyperparameter tuning + testing |
| Option B analysis | 1 hour | Decision: pass/fail |
| **Option B total** | **4 hours** | **Refined solution** |
| Documentation | 1 hour | Update CLAUDE.md, code comments |
| **Grand total** | **7 hours max** | **Production-ready fix** |

**Expected outcome:** 80% chance Option A sufficient, 19% chance need Option B, 1% chance need Option C.

---

## Post-Fix Validation

After implementing fix:

1. **Re-run baseline analysis:** Verify contradiction metrics now at 0%
2. **Visual regression test:** Compare outputs to ground truth annotations
3. **User acceptance test:** Show outputs to stakeholders for approval
4. **Performance benchmark:** Ensure no significant slowdown
5. **Documentation update:** Update all CLAUDE.md files with fix details

---

## Future Enhancements (Out of Scope)

Ideas for future improvement (NOT part of this fix):

- Adaptive edge buffer size based on wall size
- Machine learning-based edge refinement (learn optimal combination)
- Multi-scale edge processing (different strategies for different zoom levels)
- Confidence-aware morphological operations

---

## Questions & Uncertainties

**Q: Should we keep the median blur smoothing (line 34) even in Option A?**
A: Test both with and without. If edges acceptable without blur, remove it for speed.

**Q: What if different walls need different strategies?**
A: Start with one strategy for all. If issues, implement per-wall adaptive approach.

**Q: How to handle ceiling/floor edges?**
A: Current focus is walls only (labels 1-3). Ceiling/floor (0,4) not processed by this code.

---

## Approval Checklist

Before proceeding to implementation:

- [x] Research phase complete (01-edge-bleeding-analysis.md)
- [x] Planning phase complete (this document)
- [ ] Test infrastructure ready (test scripts, test images)
- [ ] Human approval received
- [ ] Timeline acceptable to stakeholders
- [ ] Success criteria agreed upon

---

## Notes

- This planning assumes the dual-model AND combination (line 124 in end_to_end_pipeline.py) is correct and should be preserved
- Edge refinement is the ONLY component being fixed in this iteration
- Other v3 issues (resolution, aspect ratio, normal vectors) will be addressed in separate planning docs
- All options maintain backward compatibility with existing Wall and RoomWallLayout data structures

---

**Next Action:** Await human approval, then proceed to implement Option A test.
