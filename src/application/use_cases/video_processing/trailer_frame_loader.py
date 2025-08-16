import cv2
from typing import List
from pathlib import Path


class TrailerFrameLoader:
    """Responsible for loading all frames from trailer video"""

    def load_trailer_frames(self, trailer_path: str) -> List:
        """Load all frames from trailer video"""
        if not trailer_path or not Path(trailer_path).exists():
            return []

        cap_tr = cv2.VideoCapture(trailer_path)
        trailer_frames = []

        while True:
            ret, frame = cap_tr.read()
            if not ret:
                break
            trailer_frames.append(frame)

        cap_tr.release()
        return trailer_frames
