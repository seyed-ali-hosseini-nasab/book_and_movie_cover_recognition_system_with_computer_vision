from abc import ABC, abstractmethod
from typing import List
from src.domain.entities.book_cover import BookCover


class IImageRepository(ABC):
    @abstractmethod
    def load_input_image(self, path: str) -> BookCover:
        pass

    @abstractmethod
    def load_book_movie_images(self) -> List[BookCover]:
        pass

    @abstractmethod
    def get_movie_image_for_book(self, path: str) -> BookCover:
        pass
