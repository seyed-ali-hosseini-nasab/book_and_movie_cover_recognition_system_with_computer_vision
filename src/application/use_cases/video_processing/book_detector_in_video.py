import cv2
import os
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from src.application.use_cases.image_processing.find_matching_book_movie import FindMatchingBookMovieUseCase
from src.application.interfaces.image_repository_interface import IImageRepository


class BookDetectorInVideo:
    """Responsible for detecting the best matching book in video middle frame"""

    def __init__(self, book_matcher: FindMatchingBookMovieUseCase, image_repo: IImageRepository):
        self.book_matcher = book_matcher
        self.image_repo = image_repo

    def detect_best_book(self, cap: cv2.VideoCapture, total_frames: int, min_conf: float) -> Optional[Tuple]:
        """Returns (book_name, book_path, book_image, homography) or None"""

        mid = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, mid_frame = cap.read()
        if not ret:
            return None

        return self._find_best_match(mid_frame, min_conf)

    def _find_best_match(self, frame, min_conf: float) -> Optional[Tuple]:
        tmp_path = "data/temp/mid.jpg"
        Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(tmp_path, frame)

        try:
            best = None
            best_score = 0

            for book in self.image_repo.load_book_movie_images():
                res = self.book_matcher.execute_single_comparison(tmp_path, book.image_path)

                if res.confidence_score >= min_conf and res.confidence_score > best_score:
                    H = self._compute_homography_for_book(frame, book.image)
                    if H is not None:
                        best = (book.name, book.image_path, book.image, H)
                        best_score = res.confidence_score

            return best

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _compute_homography_for_book(self, frame, book_image):
        feature_frame = self.book_matcher.feature_extractor.extract_features(frame)
        feature_book = self.book_matcher.feature_extractor.extract_features(book_image)

        if feature_frame.descriptors is None or feature_book.descriptors is None:
            return None

        matches = self.book_matcher.matcher.match_features(
            feature_frame.descriptors, feature_book.descriptors
        )

        if len(matches) < 4:
            return None

        src = np.float32([feature_book.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([feature_frame.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        return H
