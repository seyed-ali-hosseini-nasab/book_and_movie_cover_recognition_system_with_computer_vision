import cv2
import os
import tempfile
from pathlib import Path
from typing import List
from src.application.interfaces.video_repository_interface import IVideoRepository
from src.application.interfaces.image_repository_interface import IImageRepository
from src.infrastructure.video_processors.frame_extractor import FrameExtractor
from src.application.interfaces.feature_extractor_interface import IFeatureExtractor
from src.application.interfaces.matcher_interface import IMatcher
from src.domain.entities.match_result import MatchResult

class ProcessVideoUseCase:
    def __init__(
        self,
        video_repo: IVideoRepository,
        frame_extractor: FrameExtractor,
        feature_extractor: IFeatureExtractor,
        matcher: IMatcher,
        image_repo: IImageRepository,
        phash_thresh: int = 10,
        hist_thresh: float = 0.3
    ):
        self.phash_algo = cv2.img_hash.PHash_create()
        self.video_repo = video_repo
        self.frame_extractor = frame_extractor
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.image_repo = image_repo
        self.phash_thresh = phash_thresh
        self.hist_thresh = hist_thresh
        self.book_hashes = self._compute_book_hashes()
        self.temp_frame_dir = tempfile.mkdtemp(prefix="video_frames_")

    def _compute_book_hashes(self) -> dict[str, int]:
        hashes = {}
        for book in self.image_repo.load_book_movie_images():
            gray = cv2.cvtColor(book.image, cv2.COLOR_BGR2GRAY)
            mat = self.phash_algo.compute(gray)
            hashes[book.name] = int.from_bytes(mat.tobytes(), byteorder='big')
        return hashes

    def _save_frame(self, frame, video_name: str, frame_idx: int) -> str:
        filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
        path = os.path.join(self.temp_frame_dir, filename)
        cv2.imwrite(path, frame)
        return path

    def execute(self, input_video_name: str) -> List[MatchResult]:
        video_path = self.video_repo.load_input_video(input_video_name)
        results: List[MatchResult] = []
        frame_idx = 0

        for frame in self.frame_extractor.extract_keyframes(Path(video_path), self.hist_thresh):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mat = self.phash_algo.compute(gray)
            fh = int.from_bytes(mat.tobytes(), byteorder='big')

            candidates = [
                name for name, bh in self.book_hashes.items()
                if bin(fh ^ bh).count("1") <= self.phash_thresh
            ]
            if not candidates:
                frame_idx += 1
                continue

            kp_f, desc_f = self.feature_extractor.extract_features(frame)
            best: MatchResult | None = None
            best_score = 0

            for book in self.image_repo.load_book_movie_images():
                if book.name not in candidates:
                    continue
                kp_b, desc_b = self.feature_extractor.extract_features(book.image)
                matches = self.matcher.match_features(desc_f, desc_b)
                score = len(matches)
                if score > best_score:
                    best_score = score
                    frame_path = self._save_frame(frame, os.path.splitext(input_video_name)[0], frame_idx)
                    best = MatchResult(
                        source_name=f"frame_{frame_idx}",
                        target_name=book.name,
                        matches=matches,
                        confidence_score=score,
                        good_matches_count=score,
                        target_image_path=book.image_path,
                        source_frame_path=frame_path
                    )
            if best:
                results.append(best)
            frame_idx += 1

        return results
