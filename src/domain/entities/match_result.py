from dataclasses import dataclass
from typing import List, Optional
import cv2


@dataclass
class MatchResult:
    source_name: str
    target_name: str
    matches: List[cv2.DMatch]
    confidence_score: float
    good_matches_count: int
    target_image_path: Optional[str] = None
    error_message: Optional[str] = None
