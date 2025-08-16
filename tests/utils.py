import os
import sys
import cv2
import time
from pathlib import Path
from typing import Tuple, Optional


def setup_test_environment():
    """Find project root and configure paths"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "main.py").exists() and (parent / "src").exists():
            sys.path.insert(0, str(parent))
            os.chdir(parent)
            return parent
    raise FileNotFoundError("Project root not found")


class OverlayTestHelper:
    """Helper class for overlay generation tests."""

    @staticmethod
    def load_test_images(input_path: str, book_path: str) -> Tuple[Optional[cv2.Mat], Optional[cv2.Mat]]:
        """Load and validate test images."""
        img_input = cv2.imread(input_path)
        img_book = cv2.imread(book_path)

        if img_input is None:
            print(f"❌ Cannot load input image: {input_path}")
        if img_book is None:
            print(f"❌ Cannot load book image: {book_path}")

        return img_input, img_book

    @staticmethod
    def resize_image_for_comparison(image: cv2.Mat, max_height: int = 300) -> cv2.Mat:
        """Resize image maintaining aspect ratio."""
        h, w = image.shape[:2]
        new_width = int(w * max_height / h)
        return cv2.resize(image, (new_width, max_height))

    @staticmethod
    def create_comparison_image(img_input: cv2.Mat, img_book: cv2.Mat, img_overlay: cv2.Mat) -> cv2.Mat:
        """Create side-by-side comparison of three images."""
        # Resize all images
        max_height = 300
        img_input_resized = OverlayTestHelper.resize_image_for_comparison(img_input, max_height)
        img_book_resized = OverlayTestHelper.resize_image_for_comparison(img_book, max_height)
        img_overlay_resized = OverlayTestHelper.resize_image_for_comparison(img_overlay, max_height)

        # Combine horizontally
        comparison = cv2.hconcat([img_input_resized, img_book_resized, img_overlay_resized])

        # Add labels
        OverlayTestHelper._add_labels_to_comparison(comparison, img_input_resized, img_book_resized)

        return comparison

    @staticmethod
    def _add_labels_to_comparison(comparison: cv2.Mat, img_input_resized: cv2.Mat, img_book_resized: cv2.Mat):
        """Add text labels to comparison image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)
        thickness = 2

        cv2.putText(comparison, "Input", (10, 30), font, 1, color, thickness)
        cv2.putText(comparison, "Book", (img_input_resized.shape[1] + 10, 30), font, 1, color, thickness)
        cv2.putText(comparison, "Overlay",
                    (img_input_resized.shape[1] + img_book_resized.shape[1] + 10, 30),
                    font, 1, color, thickness)

    @staticmethod
    def save_comparison_image(comparison: cv2.Mat) -> str:
        """Save comparison image and return path."""
        output_dir = Path("data/tests/overlay_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = output_dir / f"comparison_{int(time.time())}.jpg"
        cv2.imwrite(str(comparison_path), comparison)
        return str(comparison_path)

    @staticmethod
    def display_image(image: cv2.Mat, window_title: str = "Comparison"):
        """Display image in window if GUI is available."""
        try:
            cv2.imshow(window_title, image)
            print("👁️  Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return True
        except:
            print("🖥️  Image saved to file (no GUI available)")
            return False
