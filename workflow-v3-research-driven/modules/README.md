# Phase 3: ST-RoomNet + SegFormer Integration

This directory contains the Phase 3 implementation for combining ST-RoomNet geometric planes with SegFormer semantic segmentation.

## Files Overview

### Core Modules
1. **`load_outputs.py`** - Load ST-RoomNet and SegFormer outputs
2. **`simple_mask_combination.py`** - Combine masks using AND operation
3. **`edge_refinement_simple.py`** - Refine edges using ST boundaries
4. **`create_walls_simple.py`** - Create Wall objects from masks
5. **`st_roomnet_segformer_integration.py`** - Main integration pipeline

### Utilities
6. **`generate_segformer_output.py`** - Generate SegFormer confidence maps from images
7. **`gradio_visualizer.py`** - Gradio web interface for visualization

## Setup

### 1. Install Dependencies

Make sure you're in the virtual environment:
```bash
source venv/bin/activate  # or activate venv on Windows
```

Install required packages:
```bash
pip install gradio torch transformers opencv-python matplotlib pillow
```

### 2. Generate SegFormer Confidence Map

For `room1.jpeg`, generate the confidence map:

```bash
cd "Integration Work/phase3_modules"
python generate_segformer_output.py ../../room1.jpeg \
    --output ../../segformer_confidence_room1.npy \
    --visualize ../../segformer_confidence_room1_viz.png \
    --resize 400 400
```

This will create:
- `segformer_confidence_room1.npy` - Confidence map (400x400)
- `segformer_confidence_room1_viz.png` - Visualization

## Usage

### Method 1: Command Line Test

Create a simple test script:

```python
# test_simple.py
from load_outputs import load_st_roomnet_output, load_segformer_output
from simple_mask_combination import combine_masks_simple
from st_roomnet_segformer_integration import process_room_image
import cv2

# Process integration
layout, refined_masks, combined_masks = process_room_image(
    image_path='../../room1.jpeg',
    st_output_path='../../st_roomnet_output.txt',
    segformer_confidence_path='../../segformer_confidence_room1.npy',
    wall_threshold=0.4
)

# Print summary
print(f"Detected {len(layout.walls)} walls")
for wall in layout.walls:
    print(f"  {wall.wall_id}: {wall.confidence:.2f} confidence")
```

Run it:
```bash
python test_simple.py
```

### Method 2: Gradio Web Interface (Recommended)

Launch the visual interface:

```bash
python gradio_visualizer.py
```

Then open http://localhost:7860 in your browser.

**Upload:**
1. Original room image (`room1.jpeg`)
2. ST-RoomNet output (`st_roomnet_output.txt`)
3. SegFormer confidence map (`segformer_confidence_room1.npy`)

**Adjust:**
- Use the threshold slider (0.1-0.9) to see different results
- Click "Process Integration"

**View:**
- Wall overlay with colored masks
- Individual wall masks
- Comparison grid
- Integration summary statistics

## Expected Workflow

### Step 1: Generate SegFormer Output
```bash
# From phase3_modules directory
python generate_segformer_output.py ../../room1.jpeg \
    --output ../../segformer_confidence_room1.npy \
    --resize 400 400
```

### Step 2: Launch Gradio Interface
```bash
python gradio_visualizer.py
```

### Step 3: Upload and Process
- Upload `room1.jpeg`
- Upload `st_roomnet_output.txt`
- Upload `segformer_confidence_room1.npy`
- Adjust threshold
- Click "Process Integration"

### Step 4: Analyze Results
- Check if window (left wall) is excluded
- Verify furniture is removed from front wall
- Examine edge quality
- Review pixel counts and confidence scores

## Parameters to Tune

In the Gradio interface or code:

```python
config = {
    'wall_threshold': 0.4,        # SegFormer confidence threshold
    'wall_labels': [1, 2, 3],     # Which walls to process
    'min_wall_pixels': 100,        # Minimum pixels for visibility
    'edge_buffer_size': 5,         # Edge refinement buffer
}
```

## Troubleshooting

### Issue: SegFormer model download fails
**Solution**: Ensure internet connection, or download model manually from Hugging Face

### Issue: Confidence map wrong size
**Solution**: Use `--resize 400 400` when generating

### Issue: Import errors
**Solution**: Check that you're in the correct directory and venv is activated

### Issue: No walls detected
**Solution**: Lower the threshold value (try 0.2-0.3)

## File Locations

After running, you should have:

```
ST-RoomNet-for-visualisation/
├── room1.jpeg
├── st_roomnet_output.txt
├── segformer_confidence_room1.npy
├── segformer_confidence_room1_viz.png
└── Integration Work/
    └── phase3_modules/
        ├── load_outputs.py
        ├── simple_mask_combination.py
        ├── edge_refinement_simple.py
        ├── create_walls_simple.py
        ├── st_roomnet_segformer_integration.py
        ├── generate_segformer_output.py
        ├── gradio_visualizer.py
        └── README.md
```

## Success Metrics

- [ ] Window (left wall) has minimal/no pixels
- [ ] Furniture excluded from front wall
- [ ] Edges are smooth, not pixelated
- [ ] At least 80% of wall surface preserved
- [ ] Processing time < 500ms

## Next Steps

After validating Phase 3:
1. Test on multiple room images
2. Document threshold values for different scenarios
3. Add normal vector computation from theta
4. Implement advanced edge smoothing
5. Handle partial walls and occlusions

## Support

For issues, refer to:
- `phase3_simple_implementation.md` - Implementation plan
- `Phase 2 Progress.md` - Background context
- ST-RoomNet and SegFormer documentation
