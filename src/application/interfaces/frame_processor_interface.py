from abc import ABC, abstractmethod
from typing import List, Optional, Callable
import numpy as np


class IFrameProcessor(ABC):
    """Abstract interface for frame processing strategies"""

    @abstractmethod
    def process_frames(
            self,
            video_path: str,
            trailer_frames: List[np.ndarray],
            book_image: np.ndarray,
            base_homography: np.ndarray,
            output_path: str,
            total_frames: int,
            fps: float,
            w: int,
            h: int,
            alpha: float,
            progress_callback: Optional[Callable] = None
    ):
        """Process video frames and return count of replaced frames"""
        pass
