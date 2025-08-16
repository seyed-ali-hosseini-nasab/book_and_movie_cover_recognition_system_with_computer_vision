import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Callable, Tuple
import threading

from src.application.interfaces.frame_processor_interface import IFrameProcessor
from src.application.use_cases.image_processing.find_matching_book_movie import FindMatchingBookMovieUseCase


class ParallelFrameProcessor(IFrameProcessor):
    """Process frames in parallel using thread pool"""

    def __init__(self, book_matcher: FindMatchingBookMovieUseCase, max_workers: int = 4):
        self.book_matcher = book_matcher
        self.max_workers = max_workers
        self._lock = threading.Lock()

    def process_frames(
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
            alpha: float = 0.7,
            progress_callback: Optional[Callable] = None
    ) -> int:
        """Process frames in parallel and write sequentially"""

        # Pre-compute book features once
        feature_book = self.book_matcher.feature_extractor.extract_features(book_image)

        # Create output writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Process in batches to manage memory
        batch_size = min(100, total_frames)  # Smaller batches
        replaced_count = 0

        try:
            for batch_start in range(0, total_frames, batch_size):
                batch_end = min(batch_start + batch_size, total_frames)

                # Extract batch of frames with their indices
                batch_data = self._extract_frame_batch(video_path, batch_start, batch_end)

                if not batch_data:
                    # Write empty frames if extraction failed
                    for _ in range(batch_end - batch_start):
                        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
                    continue

                # Process batch in parallel
                processed_frames = self._process_batch_parallel(
                    batch_data, trailer_frames, feature_book, base_homography,
                    w, h, alpha
                )

                # Write frames in order
                for frame in processed_frames:
                    if frame is not None:
                        writer.write(frame)
                        replaced_count += 1
                    else:
                        # Write black frame if processing failed
                        writer.write(np.zeros((h, w, 3), dtype=np.uint8))

                if progress_callback:
                    progress = 20 + (batch_end / total_frames) * 60
                    progress_callback(f"Processed {batch_end}/{total_frames} frames", progress)

        finally:
            writer.release()

        return replaced_count

    def _extract_frame_batch(self, video_path: str, start: int, end: int) -> List[Tuple[int, np.ndarray]]:
        """Extract a batch of frames from video with proper error handling"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return []

        frames = []

        try:
            for idx in range(start, end):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append((idx, frame.copy()))  # Make sure to copy the frame
                else:
                    print(f"Warning: Could not read frame {idx}")
                    # Add placeholder for missing frame
                    frames.append((idx, None))

        finally:
            cap.release()

        return frames

    def _process_batch_parallel(
            self,
            batch_data: List[Tuple[int, np.ndarray]],
            trailer_frames: List,
            feature_book,
            base_homography,
            w: int,
            h: int,
            alpha: float
    ) -> List[np.ndarray]:
        """Process a batch of frames in parallel"""

        results = [None] * len(batch_data)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all frame processing tasks
            future_to_idx = {}

            for i, (frame_idx, frame) in enumerate(batch_data):
                if frame is not None:  # Only process valid frames
                    future = executor.submit(
                        self._process_single_frame_safe,
                        frame,
                        trailer_frames,
                        frame_idx,  # Make sure this is an int
                        feature_book,
                        base_homography,
                        w, h, alpha
                    )
                    future_to_idx[future] = i
                else:
                    # Handle missing frame
                    results[i] = np.zeros((h, w, 3), dtype=np.uint8)

            # Collect results maintaining order
            for future in as_completed(future_to_idx):
                batch_idx = future_to_idx[future]
                try:
                    processed_frame = future.result()
                    results[batch_idx] = processed_frame
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    # Use black frame on error
                    results[batch_idx] = np.zeros((h, w, 3), dtype=np.uint8)

        return results

    def _process_single_frame_safe(
            self,
            frame: np.ndarray,
            trailer_frames: List,
            frame_idx: int,  # Ensure this is int
            feature_book,
            base_homography,
            w: int,
            h: int,
            alpha: float
    ) -> np.ndarray:
        """Thread-safe single frame processing with proper type checking"""
        try:
            # Ensure frame_idx is int
            if not isinstance(frame_idx, int):
                print(f"Warning: frame_idx is {type(frame_idx)}, converting to int")
                frame_idx = int(frame_idx)

            # Ensure trailer_frames is not empty
            if not trailer_frames:
                print("Error: No trailer frames available")
                return frame

            # Get trailer frame with rotation - safe indexing
            trailer_idx = min(frame_idx, len(trailer_frames) - 1)
            tr_frame = trailer_frames[trailer_idx]

            if tr_frame is None:
                print(f"Error: Trailer frame {trailer_idx} is None")
                return frame

            # Rotate trailer frame
            tr_frame = cv2.rotate(tr_frame, cv2.ROTATE_90_CLOCKWISE)

            # Compute homography for current frame
            current_H = self._compute_homography_safe(frame, feature_book, base_homography)

            # Warp and blend
            warped = cv2.warpPerspective(
                tr_frame, current_H, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT
            )

            # Safe blending
            result = (frame.astype(np.float32) * (1 - alpha) +
                      warped.astype(np.float32) * alpha).astype(np.uint8)

            return result

        except Exception as e:
            print(f"Error in frame processing: {e}")
            return frame  # Return original frame on any error

    def _compute_homography_safe(self, frame, feature_book, base_homography):
        """Thread-safe homography computation"""
        try:
            # Use lock for thread safety
            with self._lock:
                feature_frame = self.book_matcher.feature_extractor.extract_features(frame)

            # Check if features are valid
            if (feature_frame and feature_frame.descriptors is not None and
                    feature_book and feature_book.descriptors is not None):

                matches = self.book_matcher.matcher.match_features(
                    feature_frame.descriptors, feature_book.descriptors
                )

                if len(matches) >= 4:
                    src = np.float32([feature_book.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst = np.float32([feature_frame.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

                    if H is not None:
                        return H

        except Exception as e:
            print(f"Error in homography computation: {e}")

        return base_homography
