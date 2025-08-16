import cv2
import time
import asyncio
from typing import Optional

from src.application.interfaces.frame_processor_interface import IFrameProcessor
from src.application.use_cases.video_processing.book_detector_in_video import BookDetectorInVideo
from src.application.use_cases.video_processing.trailer_frame_loader import TrailerFrameLoader
from src.application.use_cases.image_processing.find_matching_book_movie import FindMatchingBookMovieUseCase
from src.domain.entities.video_replacement_result import VideoReplacementResult
from src.application.interfaces.image_repository_interface import IImageRepository
from src.application.interfaces.video_repository_interface import IVideoRepository


class ProcessInputVideoUseCase:
    """Orchestrates video processing using injected frame processor"""

    def __init__(
            self,
            feature_extractor,
            matcher,
            image_repository: IImageRepository,
            video_repository: IVideoRepository,
            frame_processor: IFrameProcessor,
            min_conf: float = 10.0
    ):
        book_matcher = FindMatchingBookMovieUseCase(
            feature_extractor=feature_extractor,
            matcher=matcher,
            image_repository=image_repository
        )

        self.book_detector = BookDetectorInVideo(book_matcher, image_repository)
        self.trailer_loader = TrailerFrameLoader()
        self.frame_processor = frame_processor
        self.vid_repo = video_repository
        self.min_conf = min_conf

    def execute(
            self,
            input_video_name: str,
            output_path: Optional[str] = None,
            alpha: float = 0.7,
            progress_callback: Optional[callable] = None
    ) -> VideoReplacementResult:
        """Execute video processing - automatically handles async/sync"""

        start_time = time.time()

        # Load and validate input video
        video_data = self._load_input_video(input_video_name)
        if not video_data:
            return VideoReplacementResult.error(input_video_name, "Cannot open input video", start_time)

        cap_in, fps, w, h, total_frames = video_data

        # Detect book
        if progress_callback:
            progress_callback("Detecting book in video...", 10)

        book_data = self.book_detector.detect_best_book(cap_in, total_frames, self.min_conf)
        if not book_data:
            cap_in.release()
            return VideoReplacementResult.error(input_video_name, "No book detected", start_time)

        book_name, book_path, book_image, base_homography = book_data

        # Load trailer frames
        if progress_callback:
            progress_callback(f"Loading trailer for {book_name}...", 15)

        trailer_path = self.vid_repo.get_trailer_for_book(book_name)
        trailer_frames = self.trailer_loader.load_trailer_frames(trailer_path)
        if not trailer_frames:
            cap_in.release()
            return VideoReplacementResult.error(input_video_name, f"No frames in trailer", start_time)

        # Setup output path
        if not output_path:
            from pathlib import Path
            out_dir = Path("data/output_videos")
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(out_dir / f"{Path(input_video_name).stem}_replaced.mp4")

        # Process frames - automatically handle async/sync
        if progress_callback:
            progress_callback("Processing frames...", 20)

        video_path = str(self.vid_repo.load_input_video(input_video_name))
        cap_in.release()

        # Check if frame processor is async and handle accordingly
        if self._is_async_method(self.frame_processor.process_frames):
            replaced_count = asyncio.run(self.frame_processor.process_frames(
                video_path, trailer_frames, book_image, base_homography,
                output_path, total_frames, fps, w, h, alpha, progress_callback
            ))
        else:
            replaced_count = self.frame_processor.process_frames(
                video_path, trailer_frames, book_image, base_homography,
                output_path, total_frames, fps, w, h, alpha, progress_callback
            )

        # Return result
        result = VideoReplacementResult(
            source_video_name=input_video_name,
            target_book_name=book_name,
            replaced_frames_count=replaced_count,
            total_frames_processed=total_frames,
            output_video_path=output_path,
            success=True,
            processing_time_seconds=time.time() - start_time
        )

        if progress_callback:
            progress_callback("Completed", 100)

        return result

    def _is_async_method(self, method) -> bool:
        """Check if a method is async"""
        return asyncio.iscoroutinefunction(method)

    def _load_input_video(self, input_video_name: str):
        """Load input video and return properties"""
        try:
            in_path = self.vid_repo.load_input_video(input_video_name)
            cap_in = cv2.VideoCapture(str(in_path))

            if not cap_in.isOpened():
                return None

            fps = cap_in.get(cv2.CAP_PROP_FPS)
            w = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))

            return cap_in, fps, w, h, total

        except Exception:
            return None
