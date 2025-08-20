import cv2
import os
from typing import Optional, Tuple, List
from pathlib import Path
import numpy as np

from src.application.use_cases.image_processing.find_matching_book_movie import FindMatchingBookMovieUseCase
from src.application.interfaces.image_repository_interface import IImageRepository


class BookDetectorInVideo:
    """Responsible for detecting the best matching book in video frames"""

    def __init__(self, book_matcher: FindMatchingBookMovieUseCase, image_repo: IImageRepository):
        self.book_matcher = book_matcher
        self.image_repo = image_repo

    def detect_best_book(self, cap: cv2.VideoCapture, total_frames: int, min_conf: float) -> Optional[Tuple]:
        """
        Returns (book_name, book_path, book_image, homography) or None
        Tests multiple frames (1/4, 2/4, 3/4) and uses weighted average confidence
        """

        # Define test frame positions
        test_frames = [
            total_frames // 4,  # 25%
            total_frames // 2,  # 50%
            total_frames * 3 // 4  # 75%
        ]

        # Extract frames for testing
        frame_data = []
        for frame_idx in test_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_data.append((frame_idx, frame.copy()))
            else:
                print(f"Warning: Could not read frame {frame_idx}")

        if not frame_data:
            print("Error: No valid frames extracted for book detection")
            return None

        return self._find_best_match_multi_frame(frame_data, min_conf)

    def _find_best_match_multi_frame(self, frame_data: List[Tuple[int, np.ndarray]], min_conf: float) -> Optional[
        Tuple]:
        """Find best book match using weighted average confidence from multiple frames"""

        # Save frames temporarily
        tmp_paths = self._save_frames_temporarily(frame_data)

        try:
            # Test each book and find the best match
            best_match = self._evaluate_all_books(tmp_paths, frame_data, min_conf)
            return best_match

        finally:
            # Cleanup temporary files
            self._cleanup_temporary_files(tmp_paths)

    def _save_frames_temporarily(self, frame_data: List[Tuple[int, np.ndarray]]) -> List[str]:
        """Save frames to temporary files and return paths"""
        tmp_dir = Path("data/temp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        tmp_paths = []
        for i, (frame_idx, frame) in enumerate(frame_data):
            tmp_path = tmp_dir / f"test_frame_{i}_{frame_idx}.jpg"
            cv2.imwrite(str(tmp_path), frame)
            tmp_paths.append(str(tmp_path))

        return tmp_paths

    def _evaluate_all_books(self, tmp_paths: List[str], frame_data: List[Tuple[int, np.ndarray]], min_conf: float) -> \
    Optional[Tuple]:
        """Evaluate all books against frames and return best match"""
        best = None
        best_weighted_score = 0.0

        for book in self.image_repo.load_book_movie_images():
            weighted_score = self._calculate_book_weighted_score(book, tmp_paths)

            if weighted_score >= min_conf and weighted_score > best_weighted_score:
                homography = self._get_homography_for_best_match(frame_data, book)

                if homography is not None:
                    best = (book.name, book.image_path, book.image, homography)
                    best_weighted_score = weighted_score
                    print(f"✅ New best match: {book.name} with weighted confidence {weighted_score:.2f}")

        if best:
            print(f"🎯 Final best match: {best[0]} with weighted confidence {best_weighted_score:.2f}")
        else:
            print("❌ No book matches found with sufficient confidence")

        return best

    def _calculate_book_weighted_score(self, book, tmp_paths: List[str]) -> float:
        """Calculate weighted average score for a book against all frames"""
        weights = [0.5, 1.0, 1.5]  # 25%, 50%, 75%
        frame_scores = []

        # Get confidence scores for each frame
        for tmp_path in tmp_paths:
            try:
                res = self.book_matcher.execute_single_comparison(tmp_path, book.image_path)
                frame_scores.append(res.confidence_score)
            except Exception as e:
                print(f"Error matching {book.name} with frame: {e}")
                frame_scores.append(0.0)

        # Calculate weighted average confidence
        if frame_scores and len(frame_scores) == len(weights):
            weighted_sum = sum(score * weight for score, weight in zip(frame_scores, weights))
            total_weight = sum(weights)
            weighted_avg = weighted_sum / total_weight

            print(f"📖 {book.name}:")
            print(f"   Scores: 25%={frame_scores[0]:.2f}, 50%={frame_scores[1]:.2f}, 75%={frame_scores[2]:.2f}")
            print(f"   Weighted average = {weighted_avg:.2f}")

            return weighted_avg

        return 0.0

    def _get_homography_for_best_match(self, frame_data: List[Tuple[int, np.ndarray]], book) -> Optional[np.ndarray]:
        """Get homography matrix using middle frame for the best matching book"""
        # Use the middle frame (index 1) for homography calculation
        middle_frame_idx = len(frame_data) // 2
        middle_frame = frame_data[middle_frame_idx][1]

        return self._compute_homography_for_book(middle_frame, book.image)

    def _cleanup_temporary_files(self, tmp_paths: List[str]):
        """Remove temporary files"""
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {tmp_path}: {e}")

    def _compute_homography_for_book(self, frame, book_image):
        """Compute homography between frame and book image"""
        try:
            feature_frame = self.book_matcher.feature_extractor.extract_features(frame)
            feature_book = self.book_matcher.feature_extractor.extract_features(book_image)

            if feature_frame.descriptors is None or feature_book.descriptors is None:
                return None

            matches = self.book_matcher.matcher.match_features(
                feature_frame.descriptors, feature_book.descriptors
            )

            if len(matches) < 4:
                return None

            import numpy as np
            src = np.float32([feature_book.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([feature_frame.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

            return H

        except Exception as e:
            print(f"Error computing homography: {e}")
            return None
