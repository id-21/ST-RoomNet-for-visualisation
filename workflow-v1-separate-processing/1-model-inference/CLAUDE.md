<system_context>
This directory contains standalone inference scripts for semantic segmentation models. Each script runs model inference on a single image and saves outputs to the `outputs/` directory.
</system_context>

<file_map>
## FILE MAP
- `run_st_roomnet.py` - ST-RoomNet inference with ConvNeXtTiny backbone, outputs 400x400 segmentation mask
- `run_segformer.py` - SegFormer (nvidia/segformer-b2-finetuned-ade-512-512) inference, outputs wall confidence maps
- `outputs/` - Directory for model inference outputs (.txt, .npy files)
</file_map>

<patterns>
## PATTERNS

### ST-RoomNet Pattern
- Loads image → resize to 400x400 → ConvNeXtTiny preprocessing
- Uses spatial transformer layer with reference image to warp segmentation
- Output: 400x400 pixel classification (1-5 class range)
- Example: `run_st_roomnet.py:39-73`

### SegFormer Pattern
- Loads pretrained HuggingFace model → runs inference → extracts per-class probabilities
- Outputs wall confidence map (0-1 range) resized to original image dimensions
- Supports visualization with JET colormap
- Example: `run_segformer.py:15-71`
</patterns>

<critical_notes>
## CRITICAL NOTES
- **Hardcoded paths** - Both scripts use absolute paths that need updating per machine (lines 35-37 in run_st_roomnet.py)
- **RGB/BGR conversion** - ST-RoomNet requires cv2.COLOR_BGR2RGB conversion (line 40) before preprocessing
- **Reference image dependency** - ST-RoomNet requires ref_img2.png (400x400, values 0-255, divided by 51.0 to normalize)
- **Model weights** - ST-RoomNet expects Weight_ST_RroomNet_ConvNext.h5 file
- **Spatial transformer** - Uses custom ProjectiveTransformerLayer from shared/spatial_transformer_fixed.py
- **Output ranges** - ST-RoomNet: 1-5 (segmentation classes), SegFormer: 0-1 (wall confidence)
- **Shared dependencies** - ST-RoomNet adds ../../shared to sys.path for spatial_transformer_fixed module
</critical_notes>

<paved_path>
## PAVED PATH

**Running ST-RoomNet inference:**
1. Update absolute paths in lines 35-37
2. Ensure shared/spatial_transformer_fixed.py is available
3. Run: `python run_st_roomnet.py`
4. Output saved to: `outputs/st_roomnet_output.txt`

**Running SegFormer inference:**
1. Run: `python run_segformer.py <image_path> -o <output.npy> -v <viz.png>`
2. Optional: `--resize WIDTH HEIGHT` to resize output
3. Output: .npy file (H, W) with wall confidence values 0-1
</paved_path>
