from typing import List
import cv2
import os
from pathlib import Path
from src.application.interfaces.image_repository_interface import IImageRepository
from src.domain.entities.book_cover import BookCover


class FileImageRepository(IImageRepository):
    def __init__(self, input_path: str = "data/input_images", book_movie_path: str = "data/book_movie_images"):
        self.input_path = input_path
        self.book_movie_path = book_movie_path

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
