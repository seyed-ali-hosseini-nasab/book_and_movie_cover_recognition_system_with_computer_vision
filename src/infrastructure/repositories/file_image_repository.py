import json
from typing import List, Dict
import cv2
import os
from pathlib import Path
from src.application.interfaces.image_repository_interface import IImageRepository
from src.domain.entities.book_cover import BookCover


class FileImageRepository(IImageRepository):
    def __init__(self, input_path: str = "data/input_images", book_movie_path: str = "data/book_images"):
        self.input_path = input_path
        self.book_movie_path = book_movie_path
        self.movie_cover_path = "data/movie_images"
        self.book_movie_mapping_path = Path("data/book_movie_mapping.json")
        self.book_movie_mapping = None

    def load_input_image(self, path: str) -> BookCover:
        full_path = os.path.join(self.input_path, path)
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {full_path}")

        return BookCover(
            image_path=full_path,
            image=image,
            name=Path(path).stem
        )

    def load_book_movie_images(self) -> List[BookCover]:
        books = []
        for filename in os.listdir(self.book_movie_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(self.book_movie_path, filename)
                image = cv2.imread(full_path)
                if image is not None:
                    book = BookCover(
                        image_path=full_path,
                        image=image,
                        name=Path(filename).stem
                    )
                    books.append(book)
        return books

    def get_movie_image_for_book(self, book_name: str) -> str:
        # Load mapping if not cached
        if self.book_movie_mapping is None:
            self.book_movie_mapping = self._load_book_movie_mapping()

        # Check direct mapping first
        if book_name in self.book_movie_mapping:
            movie_image_filename = self.book_movie_mapping[book_name]
            movie_image_path = Path(f"{self.movie_cover_path}/{movie_image_filename}")

            if movie_image_path.exists():
                return str(movie_image_path)
            else:
                print(f"⚠️ Mapped movie not found: {movie_image_path}")
        raise FileNotFoundError()

    def _load_book_movie_mapping(self) -> Dict[str, str]:
        try:
            if self.book_movie_mapping_path.exists():
                with open(self.book_movie_mapping_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                return mapping
            else:

                return {}
        except json.JSONDecodeError as e:
            print(f"❌ Error reading JSON mapping file: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error loading book-movie mapping: {e}")
            return {}
