# ST-RoomNet Integration Plan

## Current Understanding
- ✅ Model outputs 400x400 segmentation mask with labels 0-4
- ✅ Each label represents: ceiling(0), left wall(1), front wall(2), right wall(3), floor(4)
- ✅ Visualization working correctly

## Critical Next Steps

### 1. Extract Transformation Parameters (PRIORITY)
**Goal**: Modify model to return the 8 theta parameters alongside segmentation
**Where to look**:
- Find the fully connected layer that predicts 8 parameters
- Look for variable names like `theta`, `transform`, `homography`, or `params`
- Check before the spatial transformation layer is applied
**What to do**:
- Modify the model's forward pass to return both segmentation AND theta
- These 8 parameters define the 3x3 homography matrix (p33=1)

### 2. Create Integration Module Structure
```
st_roomnet_integration.py
├── load_model()
├── preprocess_image()
├── get_room_layout()
│   ├── run_inference() -> segmentation, theta
│   ├── extract_walls_from_segmentation()
│   ├── compute_corners_from_boundaries()
│   ├── calculate_normals_from_theta()
│   └── create_Wall_objects()
└── convert_to_RoomWallLayout()
```

### 3. Wall Extraction Logic
From segmentation mask:
1. **Find boundaries**: Use cv2.findContours() for each label
2. **Extract corners**: Find intersection points of wall boundaries
3. **Generate masks**: Binary mask for each wall (pixels == wall_label)

### 4. Geometry Reconstruction
From theta parameters (3x3 homography):
1. **Decompose homography**: Extract rotation and perspective components
2. **Calculate normals**: Use homography to determine 3D orientation
3. **Validate geometry**: Ensure walls form valid cuboid structure

## Immediate Action Items
1. **Locate theta extraction point** in model code
2. **Test theta values** on sample image
3. **Verify homography math** (reshape 8 params → 3x3 matrix)
4. **Create minimal working example** with one wall

## Files to Examine
- Model definition file (network architecture)
- Training script (loss calculation uses theta)
- Inference script (where spatial transform happens)

## Success Metrics
- [ ] Can extract 8 theta parameters
- [ ] Can reconstruct 3x3 homography matrix
- [ ] Can identify wall corners from segmentation
- [ ] Can calculate normal vectors from homography
- [ ] Creates valid RoomWallLayout object