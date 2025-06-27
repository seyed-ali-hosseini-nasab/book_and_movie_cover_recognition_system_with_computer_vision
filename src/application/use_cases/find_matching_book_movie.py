from typing import List, Optional
from src.domain.entities.book_cover import BookCover
from src.domain.entities.match_result import MatchResult
from src.application.interfaces.feature_extractor_interface import IFeatureExtractor
from src.application.interfaces.matcher_interface import IMatcher
from src.application.interfaces.image_repository_interface import IImageRepository
import os
import cv2


class FindMatchingBookMovieUseCase:
    def __init__(self,
                 feature_extractor: IFeatureExtractor,
                 matcher: IMatcher,
                 image_repository: IImageRepository):
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.image_repository = image_repository

    def execute(self, input_image: str, book_image_paths: Optional[List[str]] = None) -> Optional[MatchResult]:
        # بارگذاری تصویر ورودی
        if os.path.isabs(input_image):
            image = cv2.imread(input_image)
            if image is None:
                raise FileNotFoundError(f"Image not found: {input_image}")
            input_book = BookCover(
                image_path=input_image,
                image=image,
                name=os.path.splitext(os.path.basename(input_image))[0]
            )
        else:
            input_book = self.image_repository.load_input_image(input_image)

        # استخراج ویژگی از تصویر ورودی
        input_book.keypoints, input_book.descriptors = self.feature_extractor.extract_features(input_book.image)

        # بارگذاری تصاویر کتاب/فیلم
        if book_image_paths:
            book_movie_images = []
            for path in book_image_paths:
                image = cv2.imread(path)
                if image is not None:
                    book = BookCover(
                        image_path=path,
                        image=image,
                        name=os.path.splitext(os.path.basename(path))[0]
                    )
                    book_movie_images.append(book)
        else:
            book_movie_images = self.image_repository.load_book_movie_images()

        best_match = None
        best_score = 0

        for book_movie in book_movie_images:
            book_movie.keypoints, book_movie.descriptors = self.feature_extractor.extract_features(book_movie.image)
            matches = self.matcher.match_features(input_book.descriptors, book_movie.descriptors)

            if matches:
                confidence_score = len(matches)

                if confidence_score > best_score:
                    best_score = confidence_score
                    best_match = MatchResult(
                        source_name=input_book.name,
                        target_name=book_movie.name,
                        matches=matches,
                        confidence_score=confidence_score,
                        good_matches_count=len(matches),
                        target_image_path=book_movie.image_path
                    )

        return best_match

    def execute_single_comparison(self, input_image_path: str, book_image_path: str) -> MatchResult:
        """مقایسه تصویر ورودی با یک تصویر کتاب خاص؛ حتی در صورت خطا ادامه می‌دهد."""
        try:
            # بارگذاری تصویر ورودی
            input_image = cv2.imread(input_image_path)
            if input_image is None:
                raise FileNotFoundError(f"Cannot load input image: {input_image_path}")

            input_book = BookCover(
                image_path=input_image_path,
                image=input_image,
                name=os.path.splitext(os.path.basename(input_image_path))[0]
            )

            # بارگذاری تصویر کتاب
            book_image = cv2.imread(book_image_path)
            if book_image is None:
                raise FileNotFoundError(f"Cannot load book image: {book_image_path}")

            book_cover = BookCover(
                image_path=book_image_path,
                image=book_image,
                name=os.path.splitext(os.path.basename(book_image_path))[0]
            )

            # استخراج ویژگی‌ها
            input_book.keypoints, input_book.descriptors = self.feature_extractor.extract_features(input_book.image)
            book_cover.keypoints, book_cover.descriptors = self.feature_extractor.extract_features(book_cover.image)

            # تطبیق ویژگی‌ها
            matches = self.matcher.match_features(input_book.descriptors, book_cover.descriptors)
            confidence = len(matches)

            return MatchResult(
                source_name=input_book.name,
                target_name=book_cover.name,
                matches=matches,
                confidence_score=confidence,
                good_matches_count=confidence,
                target_image_path=book_cover.image_path
            )

        except Exception as ex:
            # در صورت هر خطا، پیام خطا را ذخیره می‌کنیم و باز هم ادامه می‌دهیم
            return MatchResult(
                source_name=os.path.splitext(os.path.basename(input_image_path))[0],
                target_name=os.path.splitext(os.path.basename(book_image_path))[0],
                matches=[],
                confidence_score=0.0,
                good_matches_count=0,
                target_image_path=book_image_path,
                error_message=str(ex)
            )
