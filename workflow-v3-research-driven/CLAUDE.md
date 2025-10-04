# CLAUDE.md - Wallpaper Visualization V3

<system_context>
You are working on workflow-v3 of a wallpaper visualization system that combines ST-RoomNet (geometric plane detection) with SegFormer (semantic segmentation) to apply wallpaper patterns to room photographs. This version focuses on fixing critical quality issues identified in v2 through a systematic research → planning → implementation approach.
</system_context>

<critical_notes>
## CRITICAL ISSUES STATUS

### ✅ FIXED
- **Edge bleeding bug**: FIXED via Option A - removed edge refinement entirely, combined AND mask respects semantic understanding (commit 78f3489)

### 🔄 IN PROGRESS
- **Resolution degradation**: 400x400 constraint destroys image quality - must preserve original resolution in final output
- **Aspect ratio distortion**: Square processing breaks room proportions - maintain original aspect ratio
  - These two issues are LINKED and must be solved together

### ⏳ PENDING
- **Normal vector inaccuracy**: Current normals don't match expected geometry - validate against known references
</critical_notes>

<file_map>
## PROJECT STRUCTURE
- `research/` - Document problems with evidence (screenshots, metrics, analysis)
- `planning/` - Detailed implementation plans based on research findings
- `experiments/` - Isolated test branches for parallel approaches
- `modules/` - Core implementation inherited from workflow-v2
- `test_images/` - Standardized test set for comparison
- `outputs/` - Results from different approaches for A/B testing
</file_map>

<paved_path>
## WORKFLOW (STRICTLY FOLLOW THIS ORDER)

### Phase 1: Research First (ALWAYS)
1. Document the problem with visual evidence
2. Measure the issue quantitatively where possible
3. Identify root causes in the code
4. Write findings in `research/` directory
5. NEVER start coding without research

### Phase 2: Planning (REQUIRED)
1. Read all research findings
2. Create step-by-step fix strategy
3. Identify dependencies and risks
4. Document in `planning/` directory
5. Get human approval before implementation

### Phase 3: Implementation
1. Create git worktree for isolated testing
2. Implement according to plan
3. Test on standardized image set
4. Compare with v2 baseline
5. Document results with metrics
</paved_path>

<patterns>
## CODE PATTERNS

### Edge Refinement (CORRECT)
```python
# GOOD: Weighted combination respecting semantics
edge_pixels = edge_buffer > 0
refined[edge_pixels] = (
    segformer_confidence[edge_pixels] * 0.8 +  # Trust semantics more
    st_plane_mask[edge_pixels] * 0.2           # Geometry as hint only
) > threshold
```

### Edge Refinement (BUGGY - CURRENT V2)
```python
# BAD: Overrides semantics with geometry
edge_pixels = edge_buffer > 0
refined[edge_pixels] = st_plane_mask[edge_pixels]  # THIS IS THE BUG
```

### Resolution Preservation Pattern
```python
# Process at different resolutions
geometry_mask = process_st_roomnet(resize_to_400x400(image))
semantic_mask = process_segformer(image)  # Original resolution
final_mask = combine_at_original_resolution(
    upscale(geometry_mask, original_size),
    semantic_mask
)
```
</patterns>

<common_tasks>
## TASK: Fix Edge Bleeding Bug

1. **Research Phase**
   - Load test image showing the issue
   - Run v2 pipeline and save output
   - Highlight ST-RoomNet boundary pixels that shouldn't be in final mask
   - Document in `research/01-edge-bleeding-analysis.md`

2. **Planning Phase**
   - Review edge_refinement_simple.py
   - Identify exact line causing override
   - Design weighted combination approach
   - Document fix strategy in `planning/fix-edge-bleeding.md`

3. **Implementation Phase**
   - Create worktree: `git worktree add ../experiments/edge-fix`
   - Modify edge refinement function
   - Test on 5 different room images
   - Compare outputs side-by-side
   - Pick best approach

## TASK: Preserve Original Resolution

1. **Research Phase**
   - Measure quality loss from 400x400 processing
   - Test ST-RoomNet at higher resolutions
   - Profile memory usage at different sizes
   - Document in `research/02-resolution-analysis.md`

2. **Planning Phase**
   - Design dual-resolution pipeline
   - Plan upscaling strategy for geometry
   - Define memory constraints
   - Document in `planning/resolution-preservation.md`

3. **Implementation Phase**
   - Separate geometry and semantic processing
   - Implement guided upscaling
   - Test quality metrics (SSIM, PSNR)
   - Validate performance
</common_tasks>

<examples>
## Example Research Document Structure

```markdown
# Edge Bleeding Analysis

## Problem Statement
ST-RoomNet boundaries appear in final mask even where SegFormer correctly identifies non-wall areas.

## Visual Evidence
[Screenshot: Original] [Screenshot: V2 Output] [Screenshot: Expected]

## Root Cause
Line 24 in edge_refinement_simple.py:
`refined[edge_pixels] = st_plane_mask[edge_pixels]`

This directly copies ST-RoomNet values, overriding the AND operation performed earlier.

## Metrics
- Incorrect pixels: 4,832 (2.9% of wall area)
- All located within 5px of ST-RoomNet boundaries
- Affects 100% of test images

## Proposed Solution
Replace with weighted combination...
```
</examples>

<workflow>
## EXECUTION WORKFLOW

1. **Start Research**
   ```bash
   # Create research structure
   mkdir -p workflow-v3/research
   cd workflow-v3
   
   # Run v2 baseline
   python ../workflow-v2/modules/end_to_end_pipeline.py
   
   # Document issues
   # Take screenshots, measure metrics
   ```

2. **Parallel Research Tasks**
   - Use parallel agents to investigate different issues
   - Each agent documents in separate research file
   - Main agent synthesizes findings

3. **Planning Session**
   - Review all research
   - Prioritize fixes by impact
   - Create implementation roadmap
   - Get human approval

4. **Parallel Implementation**
   - Create worktrees for each fix
   - Test approaches simultaneously
   - Compare results quantitatively
</workflow>

<fatal_implications>
## THINGS THAT WILL BREAK EVERYTHING
- Processing final output at 400x400 resolution
- Letting ST-RoomNet override SegFormer semantics  
- Using binary masks instead of soft confidence values
- Ignoring aspect ratio preservation
- Implementing without research phase
</fatal_implications>