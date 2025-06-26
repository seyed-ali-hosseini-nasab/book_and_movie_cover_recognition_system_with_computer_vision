from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import cv2

@dataclass
class BookCover:
    image_path: str
    image: Optional[np.ndarray] = None
    keypoints: Optional[List[cv2.KeyPoint]] = None
    descriptors: Optional[np.ndarray] = None
    name: Optional[str] = None
