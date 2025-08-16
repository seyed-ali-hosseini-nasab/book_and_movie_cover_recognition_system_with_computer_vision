import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class VideoReplacementResult:
    """Result of video replacement operation - equivalent to MatchResult for input_videos."""

    # Basic identification
    source_video_name: str
    target_book_name: str

    # Replacement details
    replaced_frames_count: int
    total_frames_processed: int
    output_video_path: Optional[str] = None

    # Tracking information
    first_detection_frame: int = -1
    last_detection_frame: int = -1
    tracking_confidence: float = 0.0

    # Technical details
    replacement_regions: List[Tuple[int, Tuple[int, int, int, int]]] = None  # List of (frame_idx, bbox)
    processing_time_seconds: float = 0.0

    # Error handling
    error_message: Optional[str] = None
    success: bool = False

    def __post_init__(self):
        if self.replacement_regions is None:
            self.replacement_regions = []

    @property
    def replacement_percentage(self) -> float:
        """Calculate what percentage of frames had replacements."""
        if self.total_frames_processed == 0:
            return 0.0
        return (self.replaced_frames_count / self.total_frames_processed) * 100

    @property
    def duration_frames(self) -> int:
        """Duration in frames from first to last detection."""
        if self.first_detection_frame == -1 or self.last_detection_frame == -1:
            return 0
        return self.last_detection_frame - self.first_detection_frame + 1

    @classmethod
    def error(
        cls,
        source_video_name: str,
        error_message: str,
        processing_start_time: float
    ) -> "VideoReplacementResult":
        """
        Create an error result with the given message and elapsed time.
        """
        return cls(
            source_video_name=source_video_name,
            target_book_name="error",
            replaced_frames_count=0,
            total_frames_processed=0,
            output_video_path=None,
            first_detection_frame=-1,
            last_detection_frame=-1,
            tracking_confidence=0.0,
            replacement_regions=[],
            processing_time_seconds=time.time() - processing_start_time,
            error_message=error_message,
            success=False
        )
