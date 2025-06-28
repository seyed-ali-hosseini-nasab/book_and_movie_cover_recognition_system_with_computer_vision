import cv2
from pathlib import Path
from typing import Iterator

class FrameExtractor:
    def __init__(self, frame_skip: int = 30):
        self.frame_skip = frame_skip

    def extract_keyframes(self, video_path: Path, hist_thresh: float) -> Iterator[cv2.Mat]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        last_hist = None
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.frame_skip == 0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                if last_hist is None or cv2.compareHist(last_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > hist_thresh:
                    last_hist = hist
                    yield frame
            idx += 1
        cap.release()
