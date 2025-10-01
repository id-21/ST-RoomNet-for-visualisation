# Phase 2 Progress Report: ST-RoomNet Integration

## Completed: Theta Parameter Extraction ✅

**Date Completed:** September 30, 2024
**Branch:** `feature/integration-module`
**Commit:** `290fc3a`

---

## Summary

Successfully modified ST-RoomNet model to extract both segmentation output and theta transformation parameters. The dual-output model now provides all information needed to reconstruct room geometry and create Wall objects.

---

## Phase 1 Recap: Data Structures ✅

### Created Files
- **`room_wall_layout.py`** - Data structure classes

### Classes Implemented

#### `Wall` Class
```python
class Wall:
    wall_id: str
    corners_2d: List[Tuple[float, float]]  # (x,y) pixel coordinates
    normal_vector: Optional[np.ndarray]    # 3D wall normal [x, y, z]
    pixel_mask: Optional[np.ndarray]       # Binary mask of wall pixels
    surface_type: str                       # "wall", "floor", "ceiling"
    is_visible: bool                        # Visibility flag
    confidence: float                       # Model confidence
```

#### `RoomWallLayout` Class
```python
class RoomWallLayout:
    walls: List[Wall]
    image_dimensions: Tuple[int, int]
    model_source: Optional[str]
```

---

## Phase 2: Model Output Extraction ✅

### Implementation Details

#### 1. Created `test_dual.py`
Modified inference script to output both segmentation and theta:

**Key Changes:**
```python
# Line 52: Dual output model
model = Model(base_model.input, [stl, theta])

# Line 59: Unpack both outputs
seg_out, theta_out = model.predict(img)

# Lines 69-77: Process and display theta
theta_matrix = np.append(theta_out[0], 1.0).reshape(3, 3)
```

#### 2. Theta Parameter Extraction
Successfully extracted 8 transformation parameters that form a 3×3 homography matrix.

**Extracted Values (from room1.jpeg):**
```
θ₀ = 0.304548    θ₁ = 0.006075    θ₂ = -0.017495
θ₃ = 0.011261    θ₄ = 0.424205    θ₅ = 0.141033
θ₆ = 0.145683    θ₇ = -0.013271   θ₈ = 1.0 (fixed)
```

**As 3×3 Homography Matrix:**
```
H = [0.304548   0.006075  -0.017495]
    [0.011261   0.424205   0.141033]
    [0.145683  -0.013271   1.000000]
```

#### 3. Output Files
- **`theta_params.txt`** - 8 transformation parameters (180 bytes)
- **`st_roomnet_output.txt`** - 400×400 segmentation matrix (625 KB)

---

## Current Understanding

### Model Architecture
```
Input Image (400×400×3)
    ↓
ConvNeXtTiny (feature extraction)
    ↓
Dense(8) → theta parameters [θ₀...θ₇]
    ↓
ProjectiveTransformerLayer(ref_img, theta)
    ↓
Segmentation Output (400×400×1)
```

### Segmentation Labels
The model outputs integer labels (0-4) for each pixel:
- **0** = Ceiling
- **1** = Left Wall
- **2** = Front Wall
- **3** = Right Wall
- **4** = Floor

### Reference Image (`ref_img2.png`)
A canonical room layout image with pre-labeled surfaces. The theta parameters define how to warp this reference to match the input image's perspective.

### How ST-RoomNet Works
1. **Input:** Room photograph (400×400)
2. **Feature Extraction:** ConvNeXt extracts visual features
3. **Transformation Prediction:** Dense layer predicts 8 parameters
4. **Spatial Transformation:** ProjectiveTransformerLayer warps reference image using theta
5. **Output:** Segmentation mask aligned with input image perspective

---

## What Theta Represents

### Homography Matrix
The 3×3 matrix `H` defines a projective transformation (homography) that maps points from the reference room coordinate system to the input image coordinate system.

**Mathematical relationship:**
```
[x']   [θ₀  θ₁  θ₂]   [x]
[y'] = [θ₃  θ₄  θ₅] × [y]
[w']   [θ₆  θ₇  1.0]   [1]

x_img = x' / w'
y_img = y' / w'
```

### Uses in Phase 3
1. **Wall Normal Vectors:** Can be computed from homography decomposition
2. **3D Geometry:** Perspective parameters encode camera viewpoint
3. **Validation:** Can verify segmentation matches geometric constraints

---

## Validation Results

### Test Run on `room1.jpeg`
- ✅ Model loads weights successfully
- ✅ Dual outputs extracted without errors
- ✅ Segmentation shape: (1, 400, 400, 1)
- ✅ Theta shape: (1, 8)
- ✅ No NaN or Inf values in theta
- ✅ Visualization displays correctly
- ✅ Both output files saved

### Observations
1. Theta values are small floats (magnitude ~0.01 to ~0.4)
2. Diagonal elements (θ₀, θ₄) are larger than off-diagonal
3. This suggests mild perspective distortion (not extreme angles)
4. Segmentation output matches visual inspection

---

## Technical Challenges Overcome

### Issue 1: TensorFlow 2.x Compatibility
**Problem:** Original `spatial_transformer.py` used TF1.x patterns incompatible with Keras functional API

**Solution:** Created `spatial_transformer_fixed.py` with proper Keras Layer implementation
- Inherited from `tf.keras.layers.Layer`
- Moved grid creation to `build()` method
- Moved transformation to `call()` method

### Issue 2: SSL Certificate Errors
**Problem:** Python 3.13 on macOS couldn't download ConvNeXt weights

**Solution:** Added SSL workaround (temporary, for development)
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### Issue 3: Large File Git Push
**Problem:** Model weights (106 MB) exceeded GitHub's 100 MB limit

**Solution:** Added `*.h5` to `.gitignore` and removed from tracking

---

## Files Modified/Created

### New Files
- `spatial_transformer_fixed.py` - TF2.x compatible spatial transformer
- `room_wall_layout.py` - Data structure classes
- `test_dual.py` - Dual-output inference script
- `theta_params.txt` - Extracted theta parameters
- `Integration Work/Phase 2 Progress.md` - This file

### Modified Files
- `test.py` - Updated for local inference (kept as single-output reference)
- `.gitignore` - Added `*.h5` to exclude model weights

---

## Next Steps: Phase 3 Preview

### Objectives
1. **Extract Wall Boundaries from Segmentation**
   - Use `cv2.findContours()` on each label
   - Extract corner points from contours
   - Create binary pixel masks per wall

2. **Compute Wall Normal Vectors from Theta**
   - Decompose homography matrix
   - Extract rotation/perspective components
   - Calculate 3D normal directions

3. **Create Wall Objects**
   - Populate `corners_2d` from segmentation
   - Populate `normal_vector` from theta
   - Populate `pixel_mask` from segmentation
   - Set `surface_type` based on label

4. **Build RoomWallLayout**
   - Create Wall for each visible surface
   - Set image dimensions
   - Set model source to "ST-RoomNet"

---

## Questions for Phase 3 Planning

### Segmentation Processing
1. How to handle contour approximation? (Douglas-Peucker algorithm?)
2. Should we validate that walls form a closed room?
3. How to handle partially visible walls?
4. What corner ordering convention? (clockwise/counterclockwise?)

### Homography Decomposition
1. Which decomposition method? (OpenCV's `decomposeHomographyMat`?)
2. How to map reference plane normals to actual wall normals?
3. Do we need camera intrinsics? (Assume unit camera?)
4. Validation: how to check if normals make geometric sense?

### Integration Architecture
1. Create `st_roomnet_integration.py` module?
2. Function structure:
   - `load_model(weights_path, ref_img_path)`
   - `preprocess_image(image_path)`
   - `extract_walls_from_segmentation(seg_mask)`
   - `compute_normals_from_theta(theta)`
   - `get_room_layout(image_path)` → `RoomWallLayout`

### Testing Strategy
1. Test with multiple images to ensure generalization
2. Visualize extracted corners overlaid on input
3. Validate normal vectors point "outward" from room
4. Compare against ground truth (if available)

---

## Repository Structure

```
ST-RoomNet-for-visualisation/
├── Integration Work/
│   ├── project-context.md              # Overall project context
│   ├── st_roomnet_integration_plan.md  # Original integration plan
│   └── Phase 2 Progress.md             # This file
├── spatial_transformer.py              # Original (TF1.x)
├── spatial_transformer_fixed.py        # Fixed (TF2.x)
├── room_wall_layout.py                 # Data structures
├── test.py                             # Single-output inference
├── test_dual.py                        # Dual-output inference
├── ref_img2.png                        # Reference room layout
├── room1.jpeg                          # Test input image
├── theta_params.txt                    # Extracted parameters
└── st_roomnet_output.txt              # Segmentation output
```

---

## Git Branches

### `main` Branch
- Clean, working ST-RoomNet inference
- Fixed spatial transformer for TF2.x
- Simple inference for quick testing

### `feature/integration-module` Branch (Current)
- Phase 1: Data structure classes ✅
- Phase 2: Theta extraction ✅
- Phase 3: Wall extraction (next)
- Phase 4: Full integration module (planned)

---

## Success Metrics Achieved

- [x] Can extract 8 theta parameters
- [x] Can reconstruct 3×3 homography matrix
- [ ] Can identify wall corners from segmentation
- [ ] Can calculate normal vectors from homography
- [ ] Creates valid RoomWallLayout object

**2 of 5 complete - 40% progress on core integration**

---

## Notes for Claude Web Discussion

### Context to Provide
1. This progress report (entire file)
2. `project-context.md` for overall goals
3. `st_roomnet_integration_plan.md` for original plan
4. Sample theta values and segmentation output
5. Reference image visualization

### Key Topics to Discuss
1. **Contour extraction strategy** - Best practices for wall boundary detection
2. **Homography decomposition** - Mathematical approach for normal computation
3. **Corner detection** - Algorithm selection and validation
4. **Integration module design** - Function breakdown and error handling
5. **Testing approach** - How to validate correctness

### Questions to Answer
1. Should we use OpenCV's homography decomposition or manual calculation?
2. How to handle the reference image's coordinate system?
3. What's the relationship between theta and actual room geometry?
4. How do we ensure wall normals are correctly oriented?
5. What edge cases should we handle (occluded walls, non-cuboid rooms)?

---

## References

### ST-RoomNet Paper
- **Title:** Spatial Transformer Room Network
- **Conference:** CVPRW 2023
- **Key Contribution:** Uses spatial transformer networks for room layout estimation

### Codebase
- **Original Repository:** [Link to original ST-RoomNet repo]
- **Forked Repository:** `https://github.com/id-21/ST-RoomNet-for-visualisation.git`

### Related Work
- Spatial Transformer Networks (Jaderberg et al., 2015)
- ConvNeXt Architecture (Liu et al., 2022)
- Homography Estimation and Decomposition

---

## Appendix: Code Snippets

### Model Definition (test_dual.py:48-52)
```python
base_model = ConvNeXtTiny(include_top=False, weights="imagenet",
                          input_shape=(400,400,3), pooling='avg')
theta = Dense(8)(base_model.output)
stl = ProjectiveTransformerLayer(ref_img, (400,400))(theta)
model = Model(base_model.input, [stl, theta])  # Dual output
```

### Theta Visualization (test_dual.py:70-77)
```python
print("Theta shape:", theta_out.shape)
print("Theta values:", theta_out[0])
print("\nTheta as 3x3 homography matrix:")
theta_matrix = np.append(theta_out[0], 1.0).reshape(3, 3)
print(theta_matrix)
```

### Data Structure Example
```python
# Creating a Wall object
wall = Wall(wall_id="left_wall")
wall.corners_2d = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
wall.normal_vector = np.array([nx, ny, nz])
wall.pixel_mask = seg_mask == 1  # Binary mask for label 1
wall.surface_type = "wall"
wall.is_visible = True
wall.confidence = 0.95

# Creating RoomWallLayout
layout = RoomWallLayout(image_width=400, image_height=400)
layout.walls.append(wall)
layout.model_source = "ST-RoomNet"
```

---

**End of Phase 2 Progress Report**

*Ready to proceed to Phase 3: Wall Extraction and Normal Computation*