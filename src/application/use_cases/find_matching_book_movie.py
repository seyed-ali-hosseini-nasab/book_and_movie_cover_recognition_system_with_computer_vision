from typing import Optional
from src.domain.entities.match_result import MatchResult
from src.application.interfaces.feature_extractor_interface import IFeatureExtractor
from src.application.interfaces.matcher_interface import IMatcher
from src.application.interfaces.image_repository_interface import IImageRepository


class FindMatchingBookMovieUseCase:
    def __init__(self,
                 feature_extractor: IFeatureExtractor,
                 matcher: IMatcher,
                 image_repository: IImageRepository):
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.image_repository = image_repository

    def execute(self, input_image_name: str) -> Optional[MatchResult]:
        input_book = self.image_repository.load_input_image(input_image_name)

        input_book.keypoints, input_book.descriptors = self.feature_extractor.extract_features(input_book.image)

        movie_posters = self.image_repository.load_book_movie_images()

        best_match = None
        best_score = 0

        for poster in movie_posters:
            poster.keypoints, poster.descriptors = self.feature_extractor.extract_features(poster.image)

            matches = self.matcher.match_features(input_book.descriptors, poster.descriptors)

            if matches:
                confidence_score = len(matches)

                if confidence_score > best_score:
                    best_score = confidence_score
                    best_match = MatchResult(
                        source_name=input_book.name,
                        target_name=poster.name,
                        matches=matches,
                        confidence_score=confidence_score,
                        good_matches_count=len(matches)
                    )

        return best_match
