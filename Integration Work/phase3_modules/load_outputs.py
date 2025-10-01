# File: load_outputs.py
import numpy as np
import cv2

def load_st_roomnet_output(seg_file_path):
    """
    Load ST-RoomNet segmentation (400x400 with labels 0-4)

    Args:
        seg_file_path: Path to ST-RoomNet output file

    Returns:
        numpy array with shape (400, 400) containing integer labels 0-4
    """
    segmentation = np.loadtxt(seg_file_path).astype(int)
    return segmentation

def load_segformer_output(confidence_file_path):
    """
    Load SegFormer wall confidence map

    Args:
        confidence_file_path: Path to SegFormer confidence map (.npy or image)

    Returns:
        numpy array with shape (H, W) containing confidence values 0-1
    """
    if confidence_file_path.endswith('.npy'):
        confidence = np.load(confidence_file_path)
    else:
        # Load as image and normalize to 0-1
        confidence = cv2.imread(confidence_file_path, cv2.IMREAD_GRAYSCALE)
        if confidence is not None:
            confidence = confidence.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Could not load confidence map from {confidence_file_path}")

    return confidence

def resize_to_match(segformer_confidence, target_shape=(400, 400)):
    """
    Resize SegFormer confidence map to match ST-RoomNet output size

    Args:
        segformer_confidence: Confidence map array
        target_shape: Target shape (height, width)

    Returns:
        Resized confidence map
    """
    if segformer_confidence.shape[:2] != target_shape:
        resized = cv2.resize(segformer_confidence,
                            (target_shape[1], target_shape[0]),
                            interpolation=cv2.INTER_LINEAR)
        return resized
    return segformer_confidence
