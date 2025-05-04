from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class SegmentedModelOutput:
    bbox_mask: List[np.ndarray]
    bbox_x1: List[int]
    bbox_y1: List[int]
    bbox_x2: List[int]
    bbox_y2: List[int]
    bbox_conf: List[float]
    bbox_class: List[int]

