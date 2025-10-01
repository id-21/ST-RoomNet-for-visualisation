# Phase 3B: Normal Vector Computation from Theta Parameters

## Objective
Complete the Wall objects by computing 3D normal vectors from ST-RoomNet's theta transformation parameters.

---

## Understanding the Problem

### What We Have
- 8 theta parameters forming a 3x3 homography matrix H
- Wall plane identifications (left=1, front=2, right=3)
- Reference cuboid with known normal directions

### What We Need
- 3D normal vectors [nx, ny, nz] for each visible wall
- Vectors should point outward from room interior
- Should be unit vectors (normalized)

---

## Mathematical Background

### Homography and Normal Transformation
The theta parameters define how the reference cuboid transforms to match the input image perspective. When transforming normal vectors through a homography:

**Key principle**: Normals transform by the inverse transpose of the transformation matrix
- Points transform: p' = H * p
- Normals transform: n' = (H^-1)^T * n

### Reference Cuboid Normals
In the reference cuboid coordinate system:
- Left wall normal: [-1, 0, 0] (points right)
- Front wall normal: [0, 0, -1] (points toward viewer)  
- Right wall normal: [1, 0, 0] (points left)
- Floor normal: [0, 1, 0] (points up)
- Ceiling normal: [0, -1, 0] (points down)

---

## Implementation Approach

### Function: compute_normal_vectors
**Purpose**: Calculate 3D normal vectors for each wall using theta parameters

**Inputs**:
- theta: 8-element array of transformation parameters
- wall_labels: list of wall labels to process

**Process**:
1. Reshape theta (8 params) + 1.0 into 3x3 homography matrix H
2. Compute inverse transpose: H_inv_T = (H^-1)^T
3. For each wall label:
   - Get reference normal based on label
   - Transform: n_transformed = H_inv_T * n_reference
   - Normalize to unit length
   - Verify orientation (should point outward)

**Output**:
- Dictionary mapping wall_label to normal_vector

### Function: update_walls_with_normals
**Purpose**: Add computed normals to existing Wall objects

**Inputs**:
- walls: list of Wall objects
- theta: transformation parameters

**Process**:
1. Compute normal vectors using theta
2. For each Wall object:
   - Identify wall type from wall_id
   - Assign corresponding normal vector
   - Set to None if computation fails

---

## Pseudocode

```
FUNCTION compute_normal_vectors(theta, wall_labels):
    # Build homography matrix
    H = reshape theta (8 elements) + [1.0] into 3x3 matrix
    
    # Compute transformation matrix for normals
    H_inverse = matrix_inverse(H)
    H_inv_transpose = transpose(H_inverse)
    
    # Define reference normals
    reference_normals = {
        1: [-1, 0, 0],  # left wall
        2: [0, 0, -1],  # front wall  
        3: [1, 0, 0],   # right wall
        4: [0, 1, 0],   # floor
        0: [0, -1, 0]   # ceiling
    }
    
    computed_normals = {}
    
    FOR each label in wall_labels:
        ref_normal = reference_normals[label]
        
        # Transform normal
        transformed = H_inv_transpose * ref_normal
        
        # Normalize to unit length
        magnitude = sqrt(transformed[0]^2 + transformed[1]^2 + transformed[2]^2)
        IF magnitude > 0:
            normalized = transformed / magnitude
        ELSE:
            normalized = ref_normal  # fallback
        
        computed_normals[label] = normalized
    
    RETURN computed_normals
```

---

## Integration with Existing Code

### Modification to st_roomnet_segformer_integration.py
In the `process_room_image` function, after creating walls:

1. Load theta parameters from file
2. Call `compute_normal_vectors` with theta and wall labels
3. Update each Wall object's normal_vector field

### Modification to create_walls_simple.py
Add parameter to accept normal vectors:

1. Modify `create_walls_from_masks` to accept optional normals dictionary
2. When creating each Wall, look up and assign normal if available

---

## Validation Checks

### Geometric Consistency
- Adjacent walls should have approximately perpendicular normals (dot product ≈ 0)
- Floor and ceiling normals should be opposite (dot product ≈ -1)
- All normals should have magnitude = 1.0

### Orientation Check
- Normals should point "outward" from room center
- Can verify by checking sign consistency with reference

---

## Error Handling

### Potential Issues
1. **Singular matrix**: H might not be invertible
   - Fallback: Use reference normals
   
2. **Degenerate transformation**: Very small theta values
   - Check condition number of H
   - Use identity if poorly conditioned

3. **Incorrect orientation**: Normal pointing inward
   - Flip sign if needed
   - Validate against expected direction

---

## Testing Strategy

### Unit Test
1. Create known theta values (identity, simple rotation, etc.)
2. Verify normals match expected transformations
3. Check orthogonality of wall normals

### Integration Test
1. Process room1.jpeg with computed normals
2. Visualize normals as arrows on the image
3. Verify they make geometric sense

---

## Files to Modify

1. **normal_computation.py** (new file)
   - Function: compute_normal_vectors
   - Function: validate_normal_consistency

2. **st_roomnet_segformer_integration.py**
   - Add theta loading
   - Call normal computation
   - Update Wall objects

3. **create_walls_simple.py**
   - Accept normals parameter
   - Assign to Wall objects

---

## Success Criteria

- [x] Wall objects created with all fields except normal_vector
- [ ] Normal vectors computed from theta parameters
- [ ] Normals are unit vectors
- [ ] Adjacent walls have perpendicular normals
- [ ] All Wall objects have complete data