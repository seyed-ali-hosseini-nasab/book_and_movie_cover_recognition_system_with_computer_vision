from .image_processing import *
from .video_processing import *
from .frame_processing import *

# backward compatibility
from .image_processing.find_matching_book_movie import FindMatchingBookMovieUseCase
from .image_processing.overlay_book_cover import OverlayBookCoverUseCase
from .video_processing.process_input_video import ProcessInputVideoUseCase

__all__ = [
    # Image Processing
    'FindMatchingBookMovieUseCase',
    'OverlayBookCoverUseCase',

    # Video Processing
    'ProcessInputVideoUseCase',
    'BookDetectorInVideo',
    'TrailerFrameLoader',

    # Frame Processing
    'ParallelFrameProcessor',
    'AsyncFrameProcessor'
]
