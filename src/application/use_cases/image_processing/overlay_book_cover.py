import cv2
import numpy as np
from typing import Optional
from src.application.interfaces.feature_extractor_interface import IFeatureExtractor
from src.application.interfaces.matcher_interface import IMatcher
from src.domain.entities.match_result import MatchResult
from src.domain.entities.book_cover import BookCover


class OverlayBookCoverUseCase:
    """
    Use case for overlaying detected book cover onto an image/frame.
    """

    def __init__(
            self,
            feature_extractor: IFeatureExtractor,
            matcher: IMatcher
    ):
        self.feature_extractor = feature_extractor
        self.matcher = matcher

    def overlay_book_on_image(
            self,
            original: np.ndarray,
            book_cover: BookCover,
            match_result: MatchResult,
            min_matches: int = 10
    ) -> Optional[np.ndarray]:
        """
        Compute homography and warp book_cover.image onto original.
        Return blended image or None if insufficient matches.
        """
        if match_result.good_matches_count < min_matches:
            return None

        # Extract matched keypoints
        feature = self.feature_extractor.extract_features(original)
        kp_orig = feature.keypoints
        feature = self.feature_extractor.extract_features(book_cover.image)
        kp_book = feature.keypoints
        src_pts = np.float32([kp_book[m.trainIdx].pt for m in match_result.matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_orig[m.queryIdx].pt for m in match_result.matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is None:
            return None

        h, w = original.shape[:2]
        warped = cv2.warpPerspective(
            book_cover.image, homography, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )

        # Create mask of warped area
        mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
        mask_3c = cv2.merge([mask, mask, mask])
        alpha = 0.7

        # Blend overlay with original
        blended = (original.astype(np.float32) * (1 - alpha * (mask_3c / 255.0)) +
                   warped.astype(np.float32) * (alpha * (mask_3c / 255.0)))
        return blended.astype(np.uint8)
