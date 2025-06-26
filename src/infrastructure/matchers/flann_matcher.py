from typing import List
import numpy as np
import cv2
from src.application.interfaces.matcher_interface import IMatcher


class FLANNMatcher(IMatcher):
    def __init__(self):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []

        matches = self.flann.knnMatch(desc1, desc2, k=2)
        good_matches = []

        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        return good_matches
