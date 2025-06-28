import os
from pathlib import Path
from typing import List
from src.application.interfaces.video_repository_interface import IVideoRepository

class FileVideoRepository(IVideoRepository):
    def __init__(self, input_path: str = "data/trailers"):
        self.input_path = Path(input_path)

    def load_input_video(self, filename: str) -> Path:
        full = self.input_path / filename
        if not full.exists():
            raise FileNotFoundError(f"Video not found: {full}")
        return full

    def list_trailers(self) -> List[Path]:
        return [
            self.input_path / f
            for f in os.listdir(self.input_path)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ]
