import cv2
from src.application.use_cases.image_processing.find_matching_book_movie import FindMatchingBookMovieUseCase
from src.infrastructure.feature_extractors.sift_extractor import SIFTExtractor
from src.infrastructure.matchers.flann_matcher import FLANNMatcher
from src.infrastructure.repositories.file_image_repository import FileImageRepository
from tests.utils import setup_test_environment, OverlayTestHelper


def test_overlay_comparison():
    """Test creation of comparison image."""
    print("🔍 Testing Comparison Image Creation...")

    input_image = "data/input_images/Tower.jpg"
    book_image = "data/book_images/The_Lord_Of_The_Rings_Towers_book.png"

    setup_test_environment()

    # Load images
    img_input, img_book = OverlayTestHelper.load_test_images(input_image, book_image)
    if img_input is None or img_book is None:
        print("❌ Cannot load images for comparison test!")
        return False

    # Generate overlay
    use_case = FindMatchingBookMovieUseCase(
        SIFTExtractor(),
        FLANNMatcher(),
        FileImageRepository()
    )
    result = use_case.execute_single_comparison_with_overlay(
        input_image, book_image, enable_overlay=True
    )

    if not result.overlay_image_path:
        print("❌ Cannot generate overlay for comparison test!")
        return False

    img_overlay = cv2.imread(result.overlay_image_path)
    if img_overlay is None:
        print("❌ Cannot load generated overlay!")
        return False

    # Create comparison
    comparison = OverlayTestHelper.create_comparison_image(img_input, img_book, img_overlay)
    comparison_path = OverlayTestHelper.save_comparison_image(comparison)

    print(f"📸 Comparison saved: {comparison_path}")

    # Display if possible
    OverlayTestHelper.display_image(comparison, "Input | Book | Overlay")

    print("✅ Comparison image creation test passed!")
    return True


if __name__ == "__main__":
    test_overlay_comparison()
