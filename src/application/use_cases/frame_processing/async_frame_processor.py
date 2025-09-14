import asyncio
import cv2
import numpy as np
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from src.application.interfaces.frame_processor_interface import IFrameProcessor
from src.application.use_cases.image_processing.find_matching_book_movie import FindMatchingBookMovieUseCase


class AsyncFrameProcessor(IFrameProcessor):
    """Async frame processor using asyncio with thread pool"""

    def __init__(self, book_matcher: FindMatchingBookMovieUseCase, max_workers: int = 4):
        self.book_matcher = book_matcher
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def process_frames(
            self,
            video_path: str,
            trailer_frames: List,
            book_image,
            base_homography,
            output_path: str,
            total_frames: int,
            fps: float,
            w: int,
            h: int,
            alpha: float,
            progress_callback: Optional[Callable] = None
    ) -> int:
        """Process frames asynchronously"""

        # Pre-compute book features
        feature_book = self.book_matcher.feature_extractor.extract_features(book_image)

        # Create output writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Process in chunks
        chunk_size = 50
        replaced_count = 0

        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)

            # Process chunk asynchronously
            chunk_results = await self._process_chunk_async(
                video_path, chunk_start, chunk_end, trailer_frames,
                book_image, feature_book, base_homography, w, h, alpha
            )

            # Write results in order
            for frame in chunk_results:
                if frame is not None:
                    writer.write(frame)
                    replaced_count += 1

            if progress_callback:
                progress = 20 + (chunk_end / total_frames) * 60
                progress_callback(f"Processed {chunk_end}/{total_frames} frames", progress)

        writer.release()
        return replaced_count

    async def _process_chunk_async(
            self,
            video_path: str,
            start: int,
            end: int,
            trailer_frames: List,
            book_image,
            feature_book,
            base_homography,
            w: int,
            h: int,
            alpha: float
    ) -> List[np.ndarray]:
        """Process a chunk of frames asynchronously"""

        loop = asyncio.get_event_loop()

        # Create tasks for each frame in chunk
        tasks = []
        for idx in range(start, end):
            task = loop.run_in_executor(
                self.executor,
                self._process_frame_by_index,
                video_path, idx, trailer_frames, book_image,
                feature_book, base_homography, w, h, alpha
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions and return frames
        processed_frames = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Frame processing error: {result}")
                processed_frames.append(None)
            else:
                processed_frames.append(result)

        return processed_frames

    def _process_frame_by_index(
            self,
            video_path: str,
            frame_idx: int,
            trailer_frames: List,
            book_image,
            feature_book,
            base_homography,
            w: int,
            h: int,
            alpha: float
    ) -> np.ndarray:
        """Process single frame by index (runs in thread pool)"""

        # Extract frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Get trailer frame
        tr_frame = trailer_frames[min(frame_idx, len(trailer_frames) - 1)]
        tr_frame = cv2.rotate(tr_frame, cv2.ROTATE_90_CLOCKWISE)
        # Resize the trailer
        h_book, w_book = book_image.shape[:2]
        tr_resized = cv2.resize(tr_frame, (w_book, h_book), interpolation=cv2.INTER_CUBIC)

        # Compute homography
        current_H = self._compute_homography_for_frame(frame, feature_book, base_homography)

        # Warp and blend
        warped = cv2.warpPerspective(
            tr_resized, current_H, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )

        return (frame * (1 - alpha) + warped * alpha).astype(np.uint8)

    def _compute_homography_for_frame(self, frame, feature_book, base_homography):
        """Compute homography for frame"""
        try:
            feature_frame = self.book_matcher.feature_extractor.extract_features(frame)

            if feature_frame.descriptors is not None and feature_book.descriptors is not None:
                matches = self.book_matcher.matcher.match_features(
                    feature_frame.descriptors, feature_book.descriptors
                )

                if len(matches) >= 4:
                    src = np.float32([feature_book.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst = np.float32([feature_frame.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

                    if H is not None:
                        return H
        except Exception:
            pass

        return base_homography
