import numpy as np
from typing import List, Tuple, Optional

class Wall:
    """
    Represents a single wall in a room with geometric and visual properties
    """
    def __init__(self, wall_id: str = None):
        # Geometric representation
        self.wall_id: str = wall_id
        self.corners_2d: List[Tuple[float, float]] = []  # (x,y) pixel coordinates defining wall boundaries

        # Perspective properties
        self.normal_vector: Optional[np.ndarray] = None  # Wall normal in 3D space [x, y, z]
        
        # Segmentation mask
        self.pixel_mask: Optional[np.ndarray] = None  # Binary mask indicating wall pixels
        
        # Wall properties
        self.surface_type: str = "wall"  # wall, floor, ceiling
        self.is_visible: bool = True  # Whether wall is visible in image
        self.confidence: float = 1.0  # Model confidence score


class RoomWallLayout:
    """
    Standardized representation of room layout with walls for wallpaper visualization
    """
    def __init__(self, image_width: int, image_height: int):
        self.walls: List[Wall] = []
        self.image_dimensions: Tuple[int, int] = (image_width, image_height)

        self.model_source: Optional[str] = None  # Which model generated this layout
