### scripts/utils.py
# ---------------------------------------------------------------------------
# Utility functions for point cloud processing and visualization.
# NOTE: This file is intended to be modified by you as part of the exercises in notebook 02_advanced.ipynb.
#
# How to use: 
#   1. Modify log_rerun() function to implement different visualization techniques.
#       Remember to save your changes!
#   2. Run the main script (`uv run scripts/exercise_client.py ...``) with the appropriate arguments
#       to see the results in Rerun viewer.
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import rerun as rr
from frameclass import Frame

# Helper functions:
def normalize_colors(colors: np.ndarray) -> np.ndarray:
    """Return normalized colors values in range [0,1]. If invalid, returns ones."""
    if colors.size == 0:
        return np.ones(0, dtype=np.float32)
    int_min = float(np.min(colors))
    int_max = float(np.max(colors))
    if int_max <= int_min:
        return np.ones_like(colors, dtype=np.float32)
    return (colors - int_min) / (int_max - int_min)


def map_colors(norm: np.ndarray) -> np.ndarray:
    """Create  RGB colors (uint8) from normalized intensities."""
    if norm.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    cmap = plt.get_cmap('PiYG') # Can be changed to other colormaps, like spring, viridis, PiYG. Check plt.colormaps() for more.
    rgba = cmap(norm)
    rgb = (rgba[:, :3] * 255).astype(np.uint8)
    return rgb

# Feel free to write more functions here as needed!

# ---------------------------------------------------------------------------
# The main function to be modified for Rerun visualization:
# ---------------------------------------------------------------------------
def log_rerun(frame: Frame) -> None:
    """Send frame to Rerun viewer."""
    # Commented sections work as a hint of order of transformations. Some might not be needed.
    # AXIS TRANSFORMATION:

    # (ROTATION:)

    # SET BOUNDING BOX FILTER:

    # COLOR:
    # Distance from the origin
    # distances = TODO
    # Height-based coloring
    # heights = TODO

    colors = frame.intensity # distances # heights
    colors = normalize_colors(colors)

    # Map to RGB using matplotlib colormap
    colors = map_colors(colors)

    rr.log(
        "lidar/points",
        rr.Points3D(
            frame.points,
            colors=colors,
            radii=0.1 # Adjust if needed
            ),
        static=False,
    )


# BACKUP OPTION: The main function to be modified for matplotlib visualization:
def update_matplotlib(ax, frame: Frame) -> None:
    """Update a matplotlib 3D scatter plot with the latest frame."""
    points = frame.points
    intensity = frame.intensity

    colors = intensity # distances # heights
    ax.clear()
    if intensity.size == points.shape[0] and colors.shape[0] > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            cmap="PiYG",
            s=1
            )
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="cyan", s=1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"LiDAR Point Cloud - Frame {frame.scan_id}")
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    ax.set_zlim([-5, 5])

# ---------------------------------------------------------------------------
# Hints and example solutions can be found in examples/utils.py
# ---------------------------------------------------------------------------

