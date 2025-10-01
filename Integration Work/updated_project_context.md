# Wallpaper Visualization Project - Updated Context

## Project Overview

### Main Goal
Create a system that automatically applies wallpaper patterns to room walls in photographs with correct perspective and accurate masking. The system detects walls in interior room images, understands their geometry and perspective, and applies wallpaper patterns only to actual wall surfaces while avoiding furniture, windows, doors, and other non-wall elements.

### Why This Project?
This enables virtual interior design visualization - allowing users to see how different wallpaper patterns would look in their actual rooms before making purchasing decisions. It's a practical computer vision application combining geometric understanding, perspective transformation, and semantic segmentation.

---

## Technical Approach

### The Core Challenge
When applying wallpaper to room photographs, we face two distinct problems:
1. **Geometric Problem**: Walls appear as trapezoids due to perspective, requiring correct transformation
2. **Semantic Problem**: Not all pixels in a wall plane are actual wallpaper-applicable surfaces

### The Dual-Model Solution
We discovered that no single model can solve both problems effectively:
- **ST-RoomNet**: Excellent at finding wall plane geometry but includes windows, furniture in its planes
- **SegFormer**: Excellent at pixel-level semantic labeling but lacks geometric understanding

**Solution**: Combine both models to leverage their respective strengths.

---

## Solution Pipeline

### Phase 1: Data Structure Design ✅
Created standardized classes to represent room layout information

### Phase 2: Model Output Extraction ✅
- Modified ST-RoomNet to output both segmentation and transformation parameters
- Extracted theta (8 homography parameters) alongside plane segmentation

### Phase 3: Dual-Model Integration (CURRENT)
1. **ST-RoomNet Processing**:
   - Extract wall plane geometry (clean boundaries)
   - Compute perspective transformation parameters
   - Identify plane labels (left/front/right walls, floor, ceiling)

2. **SegFormer Processing**:
   - Semantic segmentation at pixel level
   - Confidence scores for "wall" classification
   - Excludes windows, furniture, decorations automatically

3. **Mask Combination**:
   - Use ST-RoomNet for geometric boundaries
   - Use SegFormer for semantic filtering
   - Combine: `final_mask = st_plane_mask & segformer_wall_mask`

4. **Wallpaper Application**:
   - Apply patterns using combined masks
   - Transform patterns according to ST-RoomNet geometry
   - Ensure application only on valid wall surfaces

---

## Model Specifications

### ST-RoomNet (CVPRW 2023)
- **Purpose**: Room layout estimation via spatial transformation
- **Output**: 
  - 400×400 segmentation (5 classes: ceiling, left wall, front wall, right wall, floor)
  - 8 theta parameters forming 3×3 homography matrix
- **Strength**: Clean geometric boundaries, perspective understanding
- **Limitation**: Includes all pixels in plane (windows, furniture, etc.)

### SegFormer-B2 (NVIDIA, ADE20K)
- **Purpose**: Pixel-level semantic segmentation
- **Output**:
  - Per-pixel class labels (wall, window, furniture, etc.)
  - Confidence scores for each class
- **Strength**: Accurate semantic understanding
- **Limitation**: Blocky edges at high thresholds, no geometric understanding

---

## Key Innovation

The project's main contribution is recognizing that room layout understanding requires both:
1. **Geometric intelligence** (where are the wall planes?)
2. **Semantic intelligence** (which pixels are actual wall surface?)

By combining specialized models for each aspect, we achieve better results than any single model could provide.

---

## Data Structure Design

### `Wall` Class
Represents a single wall surface with:
- **`wall_id`**: Unique identifier
- **`corners_2d`**: List of (x,y) pixel coordinates from ST-RoomNet boundaries
- **`normal_vector`**: 3D direction the wall faces [x, y, z] from theta parameters
- **`pixel_mask`**: Binary mask from combined ST-RoomNet + SegFormer
- **`surface_type`**: "wall", "floor", or "ceiling"
- **`is_visible`**: Whether this wall is visible in the image
- **`confidence`**: Combined confidence score

### `RoomWallLayout` Class
Container for complete room understanding:
- **`walls`**: List of Wall objects
- **`image_dimensions`**: (width, height) of original image
- **`model_source`**: "ST-RoomNet + SegFormer"

---

## Current Status

### Completed
- ✅ Phase 1: Data structure design
- ✅ Phase 2: ST-RoomNet dual output extraction (segmentation + theta)
- ✅ SegFormer integration for pixel-level semantics

### In Progress
- 🔄 Phase 3: Mask combination algorithm
- 🔄 Edge refinement strategy

### Upcoming
- ⏳ Phase 4: Full integration module
- ⏳ Phase 5: Wallpaper application pipeline
- ⏳ Phase 6: Multi-model comparison (SRW-Net)

---

## Technical Constraints

- **Python-based**: All models and integration in Python
- **Resolution**: Both models process at 400×400
- **Real-time**: Target <1 second per image processing
- **Accuracy**: Prioritize avoiding false positives (wallpaper on non-walls)

---

## Success Criteria

The integration is successful when:
- [x] Extract theta parameters from ST-RoomNet
- [x] Generate plane segmentation masks
- [x] Extract semantic masks from SegFormer
- [ ] Combine masks effectively
- [ ] Extract clean wall boundaries
- [ ] Calculate normal vectors from theta
- [ ] Generate accurate pixel masks excluding furniture/windows
- [ ] Create valid RoomWallLayout objects
- [ ] Successfully apply wallpaper patterns

---

## Key Files

### Core Implementation
- `room_wall_layout.py` - Data structures
- `st_roomnet_integration.py` - ST-RoomNet wrapper
- `segformer_integration.py` - SegFormer wrapper
- `mask_combination.py` - Combination logic

### Documentation
- `Phase 2 Progress.md` - Completed theta extraction
- `Phase 3 Implementation.md` - Current mask combination work
- `Future Refinements.md` - Advanced edge refinement strategies