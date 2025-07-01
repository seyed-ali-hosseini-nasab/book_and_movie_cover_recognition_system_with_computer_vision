import os
import cv2
from src.domain.entities.book_cover import BookCover
from src.domain.entities.match_result import MatchResult
from src.application.interfaces.feature_extractor_interface import IFeatureExtractor
from src.application.interfaces.matcher_interface import IMatcher
from src.application.interfaces.image_repository_interface import IImageRepository


class FindMatchingBookMovieUseCase:
    """
    Use case for comparing an input image against a single book/movie cover.
    """

    def __init__(
        self,
        feature_extractor: IFeatureExtractor,
        matcher: IMatcher,
        image_repository: IImageRepository
    ):
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.image_repository = image_repository

    def execute_single_comparison(
        self,
        input_image_path: str,
        book_image_path: str
    ) -> MatchResult:
        """
        Compare one input image and one book cover.
        Always returns a MatchResult, even on error.
        """
        try:
            src = self._load_cover(input_image_path, is_input=True)
            dst = self._load_cover(book_image_path, is_input=False)

            self._describe_cover(src)
            self._describe_cover(dst)

            matches = self.matcher.match_features(src.descriptors, dst.descriptors)
            score = len(matches)

            return MatchResult(
                source_name=src.name,
                target_name=dst.name,
                matches=matches,
                confidence_score=score,
                good_matches_count=score,
                target_image_path=dst.image_path,
                error_message=None
            )

        except Exception as ex:
            return MatchResult(
                source_name=os.path.splitext(os.path.basename(input_image_path))[0],
                target_name=os.path.splitext(os.path.basename(book_image_path))[0],
                matches=[],
                confidence_score=0.0,
                good_matches_count=0,
                target_image_path=book_image_path,
                error_message=str(ex)
            )

    def _load_cover(self, path: str, is_input: bool) -> BookCover:
        """
        Load image from disk and wrap in BookCover.
        Raises FileNotFoundError if load fails.
        """
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot load {'input' if is_input else 'book'} image: {path}")
        name = os.path.splitext(os.path.basename(path))[0]
        return BookCover(image_path=path, image=image, name=name)

    def _describe_cover(self, cover: BookCover) -> None:
        """
        Extract keypoints and descriptors for a cover.
        """
        kp, desc = self.feature_extractor.extract_features(cover.image)
        cover.keypoints = kp
        cover.descriptors = desc
