from dataclasses import dataclass
from typing import List
import cv2

@dataclass
class MatchResult:
    source_name: str
    target_name: str
    matches: List[cv2.DMatch]
    confidence_score: float
    good_matches_count: int
