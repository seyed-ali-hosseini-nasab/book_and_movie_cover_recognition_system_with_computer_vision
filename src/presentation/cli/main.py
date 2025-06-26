from src.application.use_cases.find_matching_book_movie import FindMatchingBookMovieUseCase
from src.infrastructure.feature_extractors.sift_extractor import SIFTExtractor
from src.infrastructure.matchers.flann_matcher import FLANNMatcher
from src.infrastructure.repositories.file_image_repository import FileImageRepository


def main():
    # تنظیم dependency injection
    feature_extractor = SIFTExtractor()
    matcher = FLANNMatcher()
    image_repository = FileImageRepository()

    # ایجاد use case
    find_matching_use_case = FindMatchingBookMovieUseCase(
        feature_extractor=feature_extractor,
        matcher=matcher,
        image_repository=image_repository
    )

    # اجرای تشخیص
    result = find_matching_use_case.execute("Return.jpg")

    if result:
        print(f"بهترین تطبیق: {result.target_name}")
        print(f"امتیاز اطمینان: {result.confidence_score}")
        print(f"تعداد تطبیق‌های خوب: {result.good_matches_count}")
    else:
        print("هیچ تطبیق مناسبی یافت نشد")


if __name__ == "__main__":
    main()
