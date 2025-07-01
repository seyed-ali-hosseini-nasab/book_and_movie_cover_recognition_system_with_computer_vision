import cv2
import json
from pathlib import Path
from typing import List, Optional
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
        self.video_repo = video_repo
        self.frame_extractor = frame_extractor
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.image_repo = image_repo
        self.phash_algo = cv2.img_hash.PHash_create()
        self.phash_thresh = phash_thresh
        self.hist_thresh = hist_thresh
        self.book_hashes = self._compute_book_hashes()
        self.cache_dir = Path("data/cache/video_frames_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, input_video_name: str) -> List[MatchResult]:
        """
        Process video: for each keyframe, if cached metadata exists, load it;
        otherwise compute match, cache both frame image and metadata.
        """
        video_path = self.video_repo.load_input_video(input_video_name)
        video_name = Path(input_video_name).stem
        results: List[MatchResult] = []

        for idx, frame in enumerate(self.frame_extractor.extract_keyframes(video_path, self.hist_thresh)):
            meta_path = self._get_metadata_path(video_name, idx)
            if meta_path.exists():
                # load cached metadata
                data = json.loads(meta_path.read_text())
                results.append(MatchResult(
                    source_name=data["source_name"],
                    target_name=data["target_name"],
                    matches=[],  # detailed matches not cached
                    confidence_score=data["confidence_score"],
                    good_matches_count=data["good_matches_count"],
                    target_image_path=data["target_image_path"],
                    source_frame_path=data["source_frame_path"]
                ))
                continue

            frame_hash = self._compute_pHash(frame)
            candidates = self._filter_candidates(frame_hash)
            if not candidates:
                continue

            best = self._find_best_match(frame, candidates, video_name, idx)
            if best:
                # cache metadata
                meta = {
                    "source_name": best.source_name,
                    "target_name": best.target_name,
                    "confidence_score": best.confidence_score,
                    "good_matches_count": best.good_matches_count,
                    "target_image_path": best.target_image_path,
                    "source_frame_path": best.source_frame_path
                }
                meta_path.write_text(json.dumps(meta))
                results.append(best)

        return results

    def _compute_book_hashes(self) -> dict[str, int]:
        hashes = {}
        for book in self.image_repo.load_book_movie_images():
            gray = cv2.cvtColor(book.image, cv2.COLOR_BGR2GRAY)
            mat = self.phash_algo.compute(gray)
            hashes[book.name] = int.from_bytes(mat.tobytes(), byteorder='big')
        return hashes

    def _compute_pHash(self, frame) -> int:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mat = self.phash_algo.compute(gray)
        return int.from_bytes(mat.tobytes(), byteorder='big')

    def _filter_candidates(self, frame_hash: int) -> List[str]:
        return [
            name for name, bh in self.book_hashes.items()
            if bin(frame_hash ^ bh).count("1") <= self.phash_thresh
        ]

    def _find_best_match(
            self,
            frame,
            candidates: List[str],
            video_name: str,
            frame_idx: int
    ) -> Optional[MatchResult]:
        kp_f, desc_f = self.feature_extractor.extract_features(frame)
        best_match: Optional[MatchResult] = None
        best_score = 0

        for book in self.image_repo.load_book_movie_images():
            if book.name not in candidates:
                continue
            kp_b, desc_b = self.feature_extractor.extract_features(book.image)
            matches = self.matcher.match_features(desc_f, desc_b)
            score = len(matches)
            if score > best_score:
                best_score = score
                frame_path = self._save_frame_if_not_exists(frame, video_name, frame_idx)
                best_match = MatchResult(
                    source_name=f"frame_{frame_idx}",
                    target_name=book.name,
                    matches=matches,
                    confidence_score=score,
                    good_matches_count=score,
                    target_image_path=book.image_path,
                    source_frame_path=frame_path
                )
        return best_match

    def _get_frame_path(self, video_name: str, frame_idx: int) -> Path:
        filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
        return self.cache_dir / filename

    def _get_metadata_path(self, video_name: str, frame_idx: int) -> Path:
        filename = f"{video_name}_frame_{frame_idx:06d}.json"
        return self.cache_dir / filename

    def _save_frame_if_not_exists(self, frame, video_name: str, frame_idx: int) -> str:
        path = self._get_frame_path(video_name, frame_idx)
        if not path.exists():
            success = cv2.imwrite(str(path), frame)
            if not success:
                raise IOError(f"Failed to write frame to {path}")
        return str(path)

    def clear_cache(self):
        for file in self.cache_dir.glob("*"):
            file.unlink()
