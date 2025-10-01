# Phase 3: Simple Mask Combination Implementation

## Objective
Combine ST-RoomNet geometric planes with SegFormer semantic segmentation to create accurate wall masks that exclude windows, furniture, and other non-wall elements.

---

## Implementation Plan

### Step 1: Load Both Model Outputs
```python
# File: load_outputs.py
import numpy as np
import cv2

def load_st_roomnet_output(seg_file_path):
    """Load ST-RoomNet segmentation (400x400 with labels 0-4)"""
    segmentation = np.loadtxt(seg_file_path).astype(int)
    return segmentation

def load_segformer_output(confidence_file_path):
    """Load SegFormer wall confidence map"""
    # Assuming saved as numpy array or image
    confidence = np.load(confidence_file_path)
    return confidence
```

### Step 2: Simple Mask Combination
```python
# File: simple_mask_combination.py
def combine_masks_simple(st_segmentation, segformer_confidence, 
                         wall_threshold=0.4, wall_labels=[1, 2, 3]):
    """
    Simple AND operation between ST-RoomNet planes and SegFormer walls
    
    Args:
        st_segmentation: 400x400 array with labels 0-4
        segformer_confidence: 400x400 array with wall confidence scores
        wall_threshold: Confidence threshold for SegFormer
        wall_labels: ST-RoomNet labels for walls (1=left, 2=front, 3=right)
    
    Returns:
        dict: {label: binary_mask} for each wall
    """
    # Create binary wall mask from SegFormer
    segformer_wall_mask = (segformer_confidence > wall_threshold)
    
    # Combine for each wall plane
    combined_masks = {}
    for label in wall_labels:
        # Get ST-RoomNet plane mask
        st_plane_mask = (st_segmentation == label)
        
        # Simple AND operation
        combined_mask = st_plane_mask & segformer_wall_mask
        
        # Basic cleanup
        combined_mask = cleanup_mask_simple(combined_mask)
        
        combined_masks[label] = combined_mask
    
    return combined_masks

def cleanup_mask_simple(mask):
    """Basic morphological cleanup"""
    # Convert to uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Remove small holes
    kernel = np.ones((3,3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    # Remove small isolated regions
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
    
    return mask_cleaned > 0
```

### Step 3: Edge Refinement (Simple Version)
```python
# File: edge_refinement_simple.py
def refine_edges_with_st_boundaries(combined_mask, st_plane_mask):
    """
    Use ST-RoomNet boundaries to clean up edges
    
    Args:
        combined_mask: Initial combined mask
        st_plane_mask: Original ST-RoomNet plane
    
    Returns:
        Refined mask with cleaner edges
    """
    # Extract ST-RoomNet boundaries
    st_edges = cv2.Canny(st_plane_mask.astype(np.uint8) * 255, 100, 200)
    
    # Create buffer zone around edges (5 pixels)
    edge_buffer = cv2.dilate(st_edges, np.ones((5,5), np.uint8))
    
    # Copy combined mask
    refined = combined_mask.copy()
    
    # In edge buffer zone, use ST-RoomNet boundaries
    edge_pixels = edge_buffer > 0
    refined[edge_pixels] = st_plane_mask[edge_pixels]
    
    # Final smoothing
    refined = cv2.medianBlur(refined.astype(np.uint8), 3)
    
    return refined > 0
```

### Step 4: Create Wall Objects
```python
# File: create_walls_simple.py
from room_wall_layout import Wall, RoomWallLayout

def create_wall_from_mask(mask, label, theta=None):
    """Create Wall object from combined mask"""
    wall = Wall(wall_id=f"wall_{label}")
    
    # Extract boundary contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Take largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        # Extract corners
        wall.corners_2d = [(pt[0][0], pt[0][1]) for pt in approx]
        
    # Set pixel mask
    wall.pixel_mask = mask
    
    # Set type based on label
    wall.surface_type = "wall"
    wall.is_visible = np.sum(mask) > 100  # Min 100 pixels
    
    # Normal vector calculation deferred to future refinement
    wall.normal_vector = None
    
    return wall
```

### Step 5: Main Integration Function
```python
# File: st_roomnet_segformer_integration.py
def process_room_image(image_path, st_output_path, segformer_confidence_path):
    """
    Main function to process room image with both models
    
    Returns:
        RoomWallLayout object
    """
    # Load outputs
    st_seg = load_st_roomnet_output(st_output_path)
    seg_conf = load_segformer_output(segformer_confidence_path)
    
    # Combine masks
    combined_masks = combine_masks_simple(st_seg, seg_conf)
    
    # Refine edges
    refined_masks = {}
    for label, mask in combined_masks.items():
        st_plane = (st_seg == label)
        refined = refine_edges_with_st_boundaries(mask, st_plane)
        refined_masks[label] = refined
    
    # Create Wall objects
    walls = []
    for label, mask in refined_masks.items():
        wall = create_wall_from_mask(mask, label)
        if wall.is_visible:
            walls.append(wall)
    
    # Create RoomWallLayout
    layout = RoomWallLayout(image_width=400, image_height=400)
    layout.walls = walls
    layout.model_source = "ST-RoomNet + SegFormer"
    
    return layout
```

### Step 6: Visualization
```python
# File: visualize_results.py
import matplotlib.pyplot as plt

def visualize_comparison(original_image, st_seg, segformer_mask, combined_masks):
    """Show comparison of different masks"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")
    
    # ST-RoomNet segmentation
    axes[0, 1].imshow(st_seg, cmap='tab10')
    axes[0, 1].set_title("ST-RoomNet Planes")
    
    # SegFormer wall mask
    axes[0, 2].imshow(segformer_mask, cmap='gray')
    axes[0, 2].set_title("SegFormer Walls")
    
    # Combined masks for each wall
    for i, (label, mask) in enumerate(combined_masks.items()):
        if i < 3:
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].set_title(f"Combined Wall {label}")
    
    plt.tight_layout()
    plt.show()
```

---

## Testing Steps

### 1. Initial Test
```python
# test_simple_combination.py
# Load room1.jpeg outputs
st_seg = load_st_roomnet_output("st_roomnet_output.txt")
seg_conf = load_segformer_output("segformer_confidence.npy")

# Try different thresholds
for threshold in [0.3, 0.4, 0.5]:
    masks = combine_masks_simple(st_seg, seg_conf, threshold)
    visualize_results(masks)
```

### 2. Verify Window Exclusion
- Check that left wall (label=1) has minimal pixels if it's a window
- Confirm furniture is excluded from front wall (label=2)

### 3. Edge Quality Check
- Compare edges before and after refinement
- Ensure smooth boundaries without pixelation

---

## Parameters to Tune

```python
config = {
    'segformer_threshold': 0.4,    # Start with 0.4, adjust based on results
    'min_wall_pixels': 100,        # Minimum pixels to consider wall visible
    'edge_buffer_size': 5,          # Pixels around boundaries
    'morphology_kernel': 3,         # Cleanup kernel size
    'median_blur_size': 3           # Final smoothing
}
```

---

## Expected Output

For `room1.jpeg`:
- **Left wall (window)**: Should be mostly empty after combination
- **Front wall**: Should exclude furniture but include wall surface
- **Right wall**: Should have clean boundaries with desk/items excluded

---

## Next Session Goals

1. Run the simple implementation on multiple test images
2. Compare results with different threshold values
3. Identify cases where simple combination fails
4. Document specific issues for future refinement

---

## Success Metrics

- [ ] Masks exclude windows completely
- [ ] Furniture is removed from wall surfaces
- [ ] Edges are reasonably smooth (not blocky)
- [ ] At least 80% of actual wall surface is preserved
- [ ] Processing time < 500ms per image

---

## Notes

This simple implementation prioritizes:
1. **Correctness**: Better to exclude some wall than include non-wall
2. **Simplicity**: Easy to understand and debug
3. **Speed**: Minimal processing overhead

Future refinements will address:
- Normal vector computation from theta
- Advanced edge smoothing
- Adaptive thresholding
- Handling partial walls