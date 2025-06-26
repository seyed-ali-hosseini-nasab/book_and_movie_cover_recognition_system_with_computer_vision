from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np
import cv2

class IFeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        pass
