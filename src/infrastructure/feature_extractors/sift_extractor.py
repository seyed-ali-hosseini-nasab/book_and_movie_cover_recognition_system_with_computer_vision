import numpy as np
import cv2
from src.application.interfaces.feature_extractor_interface import IFeatureExtractor, ExtractFeatureData


class SIFTExtractor(IFeatureExtractor):
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def extract_features(self, image: np.ndarray) -> ExtractFeatureData:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return ExtractFeatureData(keypoints, descriptors)
