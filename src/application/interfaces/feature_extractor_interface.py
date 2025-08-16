from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np
import cv2


@dataclass
class ExtractFeatureData:
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray | None


class IFeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, image: np.ndarray) -> ExtractFeatureData:
        pass
