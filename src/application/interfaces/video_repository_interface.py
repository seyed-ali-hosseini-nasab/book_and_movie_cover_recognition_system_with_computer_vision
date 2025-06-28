from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

class IVideoRepository(ABC):
    @abstractmethod
    def load_input_video(self, path: str) -> Path:
        pass

    @abstractmethod
    def list_trailers(self) -> List[Path]:
        pass
