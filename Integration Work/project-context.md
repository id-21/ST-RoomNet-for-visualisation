# Wallpaper Visualization Project - Complete Context

## Project Overview

### Main Goal
Create a system that can automatically apply wallpaper patterns to room walls in photographs with correct perspective and masking. The system should detect walls in interior room images, understand their geometry and perspective, and apply wallpaper patterns that match the wall's orientation and shape while avoiding furniture, windows, doors, and other non-wall elements.

### Why This Project?
This enables virtual interior design visualization - allowing users to see how different wallpaper patterns would look in their actual rooms before making purchasing decisions. It's a practical computer vision application combining geometric understanding, perspective transformation, and image segmentation.

---

## Technical Approach

### The Problem
When you photograph a room:
1. **Walls appear as trapezoids** due to perspective distortion (not rectangles)
2. **Walls are partially occluded** by furniture, windows, doors, decorations
3. **Multiple walls** may be visible at different angles
4. **Wallpaper patterns** need to be warped to match each wall's perspective

### The Solution Pipeline
1. **Wall Detection**: Use deep learning models to identify walls in the image
2. **Geometric Understanding**: Extract wall boundaries, corners, and orientation
3. **Segmentation**: Get pixel-level masks showing exactly which pixels are wall surfaces
4. **Standardization**: Convert model outputs into a unified data structure
5. **Wallpaper Application**: 
   - Take a wallpaper tile/pattern
   - Apply perspective transformation to match wall geometry
   - Use segmentation masks to apply only to visible wall pixels
   - Composite result back into original image

---

## Data Structure Design

We've designed two Python classes to standardize outputs from different models:

### `Wall` Class
Represents a single wall surface with:
- **`wall_id`**: Unique identifier
- **`corners_2d`**: List of (x,y) pixel coordinates defining wall boundaries in the image
- **`normal_vector`**: 3D direction the wall faces [x, y, z] - crucial for perspective calculations
- **`pixel_mask`**: Binary 2D array showing which pixels belong to this wall (excludes furniture, etc.)
- **`surface_type`**: "wall", "floor", or "ceiling"
- **`is_visible`**: Whether this wall is visible in the image
- **`confidence`**: Model's confidence score for this wall detection

### `RoomWallLayout` Class
Container for the complete room understanding:
- **`walls`**: List of Wall objects
- **`image_dimensions`**: (width, height) of original image
- **`model_source`**: Which model generated this layout (for debugging/comparison)

**Key Design Decision**: This standardized format allows us to:
- Compare outputs from different models fairly
- Swap models easily without changing downstream code
- Build the wallpaper application pipeline once, use with any model

---

## Model Integration Strategy

### Models Being Evaluated

We're testing three research models for room layout understanding:

#### 1. **RoomNet (ICCV 2017)** - ⏸️ ON HOLD
- Estimates keypoints in a room's 2D image
- Defines 10 room layout types for cuboid rooms
- Outputs labeled keypoints + room type classification
- **Status**: No pre-trained weights available - requires full training pipeline
- **Action**: Deferred until other models are working

#### 2. **ST-RoomNet (CVPRW 2023)** - 🔄 CURRENT FOCUS
- Predicts perspective transformation matrix (3x3)
- Transforms reference cuboid to match input room perspective
- **Output**: 400x400 matrix (needs analysis to understand structure)
- **Status**: Successfully ran inference, now need to understand output format
- **Next Step**: Deep dive into model architecture and output interpretation

#### 3. **SRW-Net (ICPR 2022)** - ⏳ PENDING
- Outputs keypoints (labeled as false/proper)
- Outputs edges (labeled as wall/floor/ceiling/door/window)
- Requires reconstructing planes from labeled edges
- **Status**: Not yet downloaded

### Integration Architecture

Each model needs its own integration module:

```
project/
├── models/
│   ├── room_wall_layout.py         # Our standardized classes
│   ├── st_roomnet_integration.py   # ST-RoomNet wrapper
│   ├── srw_net_integration.py      # SRW-Net wrapper  
│   └── roomnet_integration.py      # RoomNet wrapper (future)
├── st_roomnet/                     # Original ST-RoomNet repository
├── srw_net/                        # Original SRW-Net repository
└── main.py                         # Main pipeline
```

**Integration Module Responsibilities**:
1. Import/wrap the original model's inference code
2. Handle model-specific pre-processing
3. Run inference
4. Parse model's raw output format
5. Convert to standardized `RoomWallLayout` object
6. Extract wall corners, normals, and pixel masks

---

## Current Status & Immediate Goals

### Completed
- ✅ Analyzed problem and approach (traditional CV vs. ML methods)
- ✅ Researched and selected three candidate models
- ✅ Designed standardized data structures (`Wall`, `RoomWallLayout`)
- ✅ Cloned and tested ST-RoomNet inference
- ✅ Generated 400x400 output matrix from sample image

### Current Task: ST-RoomNet Deep Dive
**Objective**: Understand ST-RoomNet's output format and create integration code

**What We Need**:
1. **Understand the 400x400 matrix**:
   - What does it represent? (transformation matrix? heatmap? coordinate grid?)
   - How does it relate to the paper's description of "3x3 perspective transformation"?
   - Is it a flattened representation or something else?

2. **Analyze the paper thoroughly**:
   - Model architecture details
   - Training methodology
   - Output format specification
   - How to use the output for room reconstruction

3. **Study the inference code**:
   - How is input processed?
   - What are the model's intermediate outputs?
   - How is the final 400x400 matrix generated?
   - Any post-processing steps in the original code?

4. **Plan the extraction logic**:
   - How to derive wall corners from this output?
   - How to calculate normal vectors?
   - How to generate pixel masks for each wall?
   - How to determine number of walls and their boundaries?

5. **Handle repository integration**:
   - Should we copy ST-RoomNet code into our project?
   - Use it as a git submodule?
   - Extract only necessary inference code?
   - How to manage dependencies without conflicts?

### After ST-RoomNet Integration
- [ ] Document ST-RoomNet in `docs/st_roomnet_analysis.md`
- [ ] Test integration with multiple room images
- [ ] Validate `RoomWallLayout` outputs are correct
- [ ] Move to SRW-Net integration
- [ ] Compare model outputs
- [ ] Implement wallpaper application pipeline
- [ ] Revisit RoomNet if needed

---

## Key Constraints & Principles

### Development Approach
- **Modular design**: Each model is isolated in its own integration module
- **Compartmentalized**: Easy to swap, test, and compare models
- **Standardized output**: All models must produce `RoomWallLayout` objects
- **Claude Code for implementation**: Use terminal-based Claude for file operations
- **This chat for planning**: Use conversational Claude for architecture decisions

### Technical Constraints
- **Python-based**: All models and integration code in Python
- **OpenCV compatibility**: For image processing and perspective transforms
- **NumPy arrays**: For efficient matrix operations
- **No redundant properties**: Keep data structures lean and purposeful

### Quality Standards
- **Thorough understanding before coding**: Read papers, analyze outputs, plan carefully
- **Document everything**: Markdown files explaining each model's workings
- **Test incrementally**: Validate each component before moving forward
- **Maintain checklists**: Track progress systematically

---

## Questions to Answer in This Session

### About ST-RoomNet Output
1. What exactly is the 400x400 matrix structure and meaning?
2. How do we extract wall geometry from it?
3. Does it directly give us wall corners or do we need to reconstruct them?
4. How do we generate pixel masks from this output?

### About Integration Strategy
1. How should we organize the ST-RoomNet code in our project?
2. What's the cleanest way to wrap their inference pipeline?
3. How do we handle model weights and config files?
4. Should we modify their code or keep it separate?

### About Implementation Plan
1. What's the step-by-step process for creating `st_roomnet_integration.py`?
2. What helper functions do we need?
3. How do we validate our integration is correct?
4. What test cases should we use?

---

## Files to Share

For effective analysis, please provide:
1. **ST-RoomNet paper (PDF)** - for understanding methodology
2. **Main inference script** - shows how model is used
3. **Model architecture file** - defines the network
4. **Output matrix file** - the 400x400 result to analyze
5. **Sample input image** - that generated this output
6. **Config/requirements** - for understanding dependencies

---

## Success Criteria

This ST-RoomNet integration is successful when:
- [ ] We fully understand what the 400x400 matrix represents
- [ ] We can extract wall corners reliably
- [ ] We can calculate wall normal vectors
- [ ] We can generate accurate pixel masks
- [ ] `st_roomnet_integration.py` produces valid `RoomWallLayout` objects
- [ ] The integration is well-documented and maintainable
- [ ] We can run it on new room images successfully

---

## Notes for Claude Assistant

**Your Role**:
- Help analyze the ST-RoomNet paper and code thoroughly
- Ask clarifying questions about ambiguous outputs
- Create detailed markdown documentation
- Design the integration architecture
- Suggest implementation strategies (but don't write full code yet)
- Maintain updated checklists
- Push for comprehensive understanding before implementation

**Remember**:
- Always update the progress tracker artifact
- Be detailed in documentation - this is reference material
- Encourage testing at each step
- Ask for sample outputs to validate understanding
- Suggest breaking large tasks into smaller checkpoints