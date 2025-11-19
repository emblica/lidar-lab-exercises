from dataclasses import dataclass
import numpy as np

@dataclass
class Frame:
    """Container for a single LiDAR frame.

    Attributes:
        scan_id: Frame identifier from server.
        timestamp: Timestamp string provided by server.
        points: (N,3) float32 array of XYZ positions in meters.
        intensity: (N,) float array of per-point intensity (may be empty).
        original_count: Number of original points before any downsampling.
    """

    scan_id: int
    timestamp: str
    points: np.ndarray
    intensity: np.ndarray
    original_count: int

    @property
    def count(self) -> int:
        return int(self.points.shape[0])