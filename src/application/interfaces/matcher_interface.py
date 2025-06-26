from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import cv2

class IMatcher(ABC):
    @abstractmethod
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        pass
