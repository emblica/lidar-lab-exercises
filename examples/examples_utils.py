### examples_utils.py
# ---------------------------------------------------------------------------
# Working examples for point cloud visualization and processing.
# 
# NOTE: These scripts are not runnable from this file directly. The actual file
# clients/python/utils.py has similar structure and is runnable.
# Use this file as hints for your own implementations, or to copy-paste code snippets.
# ---------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import rerun as rr
from rerun.datatypes import Angle, RotationAxisAngle
from typing import Optional, Tuple
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
# Visualization functions
# ---------------------------------------------------------------------------

# The main function to be modified for rerun visualization:
def log_rerun(frame: Frame) -> None:
    """Send frame to Rerun viewer."""
    # ROTATION EXAMPLE: Rotate around z axis by 45 degrees
    # theta = np.radians(45)
    # c, s = np.cos(theta), np.sin(theta)
    # R = np.array([[c, -s, 0],
    #               [s, c, 0],
    #               [0, 0, 1]])
    # frame.points = frame.points @ R.T

    # FILTERING EXAMPLE: Select only points inside a bounding box
    # mask = (
    #     (frame.points[:, 0] >= -6) & (frame.points[:, 0] <= 15) &
    #     (frame.points[:, 1] >= -6) & (frame.points[:, 1] <= 6) &
    #     (frame.points[:, 2] >= -5) & (frame.points[:, 2] <= 5)
    # )
    # frame.points = frame.points[mask]
    # frame.intensity = frame.intensity[mask]

    # TRANSFORMATION EXAMPLE: Flip z axix (needed at least for Pertsa lidar)
    # frame.points = np.column_stack((frame.points[:, 0], frame.points[:, 1], -frame.points[:, 2]))

    # COLOR EXAMPLE: Distance from the origin
    distances = np.linalg.norm(frame.points, axis=1)
    
    # COLOR EXAMPLE: Height-based coloring
    heights = frame.points[:, 2]

    colors = distances # frame.intensity # distances # heights
    colors = normalize_colors(colors)

    # Map to RGB using matplotlib colormap
    colors = map_colors(colors)

    # EXAMPLE of how to do transformations in rerun
    # rr.log(
    #     "lidar",
    #     rr.Transform3D(
    #         rotation=RotationAxisAngle(axis=[0, 0, 1], angle=Angle(rad=np.pi / 4)),
    #         scale=2,
    #     )
    # )

    rr.log(
        "lidar/points",
        rr.Points3D(
            frame.points,
            colors=colors,
            radii=0.3 # Adjust if needed
            ),
        static=False,
    )


# The main function to be modified for matplotlib visualization:
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
# EXAMPLES: Movement detector - Static vs Dynamic points (tasks 4-6)
# ---------------------------------------------------------------------------
# See the different versions below.
# You can replace the log_rerun function with any of logging functions below to test them out.

# ---------------------------------------------------------------------------
# EXAMPLE: Global Nearest-Neighbor Background Model
# ---------------------------------------------------------------------------

from scipy.spatial import KDTree # Efficient nearest-neighbor search
# Initialize the background cloud as an empty NumPy array.
# It should be updated to store static points.
GLOBAL_BACKGROUND_CLOUD = np.empty((0, 3), dtype=np.float32) 
BACKGROUND_UPDATE_RATE = 0.05 # How much new static data contributes to the background
STATIC_THRESHOLD = 0.3 # Max distance to be considered 'static'

# Replase log_rerun function with this one to use simple nearest-neighbor static/dynamic classification
def simple_log_rerun_with_tracking(frame: Frame) -> None:
    """Logs points with two colors: Static (Green) vs. Dynamic (Red), 
       and updates a global background model."""
    global GLOBAL_BACKGROUND_CLOUD
    
    current_points = frame.points
    
    # --- 1. INITIALIZE/UPDATE BACKGROUND ---
    if GLOBAL_BACKGROUND_CLOUD.shape[0] == 0:
        # First frame: Assume all points are static to initialize the background
        GLOBAL_BACKGROUND_CLOUD = current_points
        dynamic_mask = np.zeros(current_points.shape[0], dtype=bool)
    else:
        # Subsequent frames: Check proximity to the existing background
        tree = KDTree(GLOBAL_BACKGROUND_CLOUD)
        # Find distance to the closest point in the background
        distances, _ = tree.query(current_points, k=1, distance_upper_bound=STATIC_THRESHOLD * 2) 

        # A point is dynamic if its closest distance is greater than the threshold
        dynamic_mask = distances > STATIC_THRESHOLD
        
        # Identify static points for background refinement
        static_points = current_points[~dynamic_mask]
        
        # Only update the background with static points, and only a fraction of them (downsampling)
        if static_points.size > 0:
            # Simple probabilistic update: select a few static points to add to the background
            sample_count = int(static_points.shape[0] * BACKGROUND_UPDATE_RATE)
            if sample_count > 0:
                 indices = np.random.choice(static_points.shape[0], sample_count, replace=False)
                 GLOBAL_BACKGROUND_CLOUD = np.vstack([GLOBAL_BACKGROUND_CLOUD, static_points[indices]])
                 # Optional: Limit the size of GLOBAL_BACKGROUND_CLOUD to prevent memory issues
                 # GLOBAL_BACKGROUND_CLOUD = GLOBAL_BACKGROUND_CLOUD[-100000:] 


    # --- 2. MAP TO COLORS (Same as original simple solution) ---
    COLOR_STATIC = np.array([0, 255, 0], dtype=np.uint8)  # Green
    COLOR_DYNAMIC = np.array([255, 0, 0], dtype=np.uint8) # Red

    colors = np.zeros((current_points.shape[0], 3), dtype=np.uint8)
    colors[dynamic_mask] = COLOR_DYNAMIC
    colors[~dynamic_mask] = COLOR_STATIC

    # --- 3. LOG TO RERUN ---
    rr.log(
        "lidar/points",
        rr.Points3D(
            current_points,
            colors=colors,
            radii=0.3
            ),
        static=False,
    )



# ---------------------------------------------------------------------------
# EXAMPLE: Voxel Grid Background Model
# ---------------------------------------------------------------------------

# Stores the (i, j, k) coordinates of voxels that contain static points.
GLOBAL_OCCUPIED_VOXELS = set() 
VOXEL_SIZE = 0.5  # voxel edge length

def point_to_voxel_coords(points: np.ndarray, size: float) -> np.ndarray:
    """Helper function to map (x, y, z) points to (i, j, k) voxel indices."""
    # This efficiently performs the floor(coord / size) operation for all points
    return np.floor(points / size).astype(np.int32)

# Replase log_rerun function with this one to use voxel-based static/dynamic classification
def voxel_log_rerun_with_tracking(frame: Frame) -> None:
    """Logs points using a voxel grid for Static (Green) / Dynamic (Red) classification.
    """
    global GLOBAL_OCCUPIED_VOXELS

    current_points = frame.points
    if current_points.size == 0:
        return

    # --- 1. Map points to voxel indices ---
    voxel_coords_array = point_to_voxel_coords(current_points, VOXEL_SIZE)
    voxel_coords_tuples = [tuple(c) for c in voxel_coords_array]

    # --- 2. First-frame initialization ---
    if not GLOBAL_OCCUPIED_VOXELS:  # Empty background => first frame
        GLOBAL_OCCUPIED_VOXELS.update(voxel_coords_tuples)
        dynamic_mask = np.zeros(len(voxel_coords_tuples), dtype=bool)  # all static
    else:
        # A point is dynamic if its voxel NOT in the background set
        dynamic_mask = np.fromiter(
            (vc not in GLOBAL_OCCUPIED_VOXELS for vc in voxel_coords_tuples),
            count=len(voxel_coords_tuples),
            dtype=bool,
        )
        # Update background with newly observed static voxels
        static_voxels = [vc for vc, is_dyn in zip(voxel_coords_tuples, dynamic_mask) if not is_dyn]
        if static_voxels:
            GLOBAL_OCCUPIED_VOXELS.update(static_voxels)

    # --- 3. Map to colors ---
    COLOR_STATIC = np.array([0, 255, 0], dtype=np.uint8)   # Green
    COLOR_DYNAMIC = np.array([255, 0, 0], dtype=np.uint8)  # Red
    colors = np.zeros((current_points.shape[0], 3), dtype=np.uint8)
    colors[dynamic_mask] = COLOR_DYNAMIC
    colors[~dynamic_mask] = COLOR_STATIC

    # --- 4. Log ---
    rr.log(
        "lidar/points",
        rr.Points3D(
            current_points,
            colors=colors,
            radii=0.3,
        ),
        static=False,
    )


# ---------------------------------------------------------------------------
# EXAMPLE: Advanced movement detector with clustering + simple tracking
# ---------------------------------------------------------------------------
from scipy.spatial import KDTree # For simple tracking distance check
from sklearn.cluster import DBSCAN # For segmentation
from matplotlib import colormaps as cm

# --- GLOBAL STATE VARIABLES ---
# Voxel Grid for Static Background (Concept from previous answer)
GLOBAL_OCCUPIED_VOXELS = set() 
VOXEL_SIZE = 0.5 

# Tracking State
GLOBAL_NEXT_TRACK_ID = 1 # Start IDs from 1 (0 is reserved for static)
# Stores {ID: (x, y, z) centroid} of objects from the *previous* frame
GLOBAL_TRACKED_OBJECTS = {} 
TRACKING_THRESHOLD = .03 # Max distance for re-assigning an old ID

# --- HELPER FUNCTIONS (Need to be defined or imported) ---

def point_to_voxel_coords(points: np.ndarray, size: float) -> np.ndarray:
    """Helper function to map (x, y, z) points to (i, j, k) voxel indices."""
    return np.floor(points / size).astype(np.int32)

def get_unique_colors(max_id: int) -> dict:
    """Generates a mapping from object ID to a deterministic, distinct color.

    Uses the discrete 'tab10' palette for readability and cycles if IDs exceed 10.
    Returns colors as floats in [0,1]. ID 0 is reserved for static (grey).
    """
    tab = cm.get_cmap('tab10')
    colors = {}
    colors[0] = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # static = grey
    if max_id <= 0:
        return colors
    for i in range(1, max_id + 1):
        idx = (i - 1) % 10  # cycle 0..9
        colors[i] = np.array(tab(idx)[0:3], dtype=np.float32)
    return colors


def voxelize_points(points: np.ndarray, size: float) -> List[Tuple[int, int, int]]:
    """Quantize points into voxel-index tuples for hashable set membership."""
    if points.size == 0:
        return []
    vox = point_to_voxel_coords(points, size)
    return [tuple(c) for c in vox]


def classify_dynamic(voxel_tuples: List[Tuple[int, int, int]], occupied: Set[Tuple[int, int, int]]) -> np.ndarray:
    """Return boolean mask of dynamic points (True = dynamic) based on background set."""
    if not occupied:
        # Background not initialized yet
        return np.zeros(len(voxel_tuples), dtype=bool)
    return np.fromiter((vt not in occupied for vt in voxel_tuples), count=len(voxel_tuples), dtype=bool)


def update_background_static(voxel_tuples: List[Tuple[int, int, int]], dynamic_mask: np.ndarray, occupied: Set[Tuple[int, int, int]]) -> None:
    """Add static voxels (where dynamic_mask is False) into the occupied set."""
    for vt, is_dyn in zip(voxel_tuples, dynamic_mask):
        if not is_dyn:
            occupied.add(vt)


def cluster_dynamic_points(dynamic_points: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster dynamic points with DBSCAN and return labels and unique non-noise labels."""
    if dynamic_points.size == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dynamic_points)
    labels = clustering.labels_
    uniq = np.unique(labels[labels != -1])
    return labels, uniq


def assign_ids_for_clusters(
    dynamic_points: np.ndarray,
    db_labels: np.ndarray,
    uniq_labels: np.ndarray,
    prev_objects: Dict[int, np.ndarray],
    next_id: int,
    threshold: float,
) -> Tuple[Dict[int, np.ndarray], Dict[int, int], int]:
    """Assign persistent IDs to clusters based on nearest previous centroids.

    Returns new_objects (id->centroid), label_to_id mapping, and updated next_id.
    """
    new_objects: Dict[int, np.ndarray] = {}
    label_to_id: Dict[int, int] = {}

    if prev_objects:
        prev_ids = list(prev_objects.keys())
        prev_centroids = np.vstack(list(prev_objects.values())).astype(np.float32)
        tree: Optional[KDTree] = KDTree(prev_centroids)
    else:
        prev_ids = []
        prev_centroids = np.empty((0, 3), dtype=np.float32)
        tree = None

    for lbl in uniq_labels:
        mask = db_labels == lbl
        pts = dynamic_points[mask]
        centroid = np.mean(pts, axis=0)

        assigned = -1
        if tree is not None and prev_centroids.shape[0] > 0:
            dist, idx = tree.query(centroid, k=1)
            if dist < threshold:
                assigned = prev_ids[idx]

        if assigned == -1:
            assigned = next_id
            next_id += 1

        new_objects[assigned] = centroid
        label_to_id[lbl] = assigned

    return new_objects, label_to_id, next_id


def ids_to_colors(ids: np.ndarray, max_id: int) -> np.ndarray:
    """Map per-point IDs to uint8 RGB colors using a discrete palette.

    ID 0 (static) -> grey.
    """
    palette = get_unique_colors(max_id)
    colors_float = np.array([palette[i] for i in ids], dtype=np.float32)
    return (colors_float * 255).astype(np.uint8)


def log_points_rr(entity: str, points: np.ndarray, colors: np.ndarray, radii: float = 0.3) -> None:
    """Small wrapper to log a point cloud to Rerun."""
    rr.log(entity, rr.Points3D(points, colors=colors, radii=radii), static=False)


def log_centroids_rr(entity: str, obj_dict: Dict[int, np.ndarray], color_map: Dict[int, np.ndarray]) -> None:
    if not obj_dict:
        return
    centroids = np.array(list(obj_dict.values()))
    ids = list(obj_dict.keys())
    rr.log(entity, rr.Points3D(centroids, radii=0.5, colors=[color_map[i] for i in ids]))

# Replace log_rerun function with this to visualize with clustering + tracking
def segmenter_log_rerun(frame: Frame) -> None:
    """Clean, modular movement visualization with clustering + simple tracking.

    Pipeline:
      1) Voxel background model -> static vs dynamic classification
      2) DBSCAN clusters on dynamic points
      3) Nearest-centroid matching for persistent IDs
      4) Colorize by ID (ID 0 = static grey) and log to Rerun
    """
    global GLOBAL_OCCUPIED_VOXELS, GLOBAL_TRACKED_OBJECTS, GLOBAL_NEXT_TRACK_ID

    pts = frame.points
    if pts.size == 0:
        return

    # 1) Background classification
    vox = voxelize_points(pts, VOXEL_SIZE)
    dyn_mask = classify_dynamic(vox, GLOBAL_OCCUPIED_VOXELS)

    # First-frame init: everything static, fill background and log
    if not GLOBAL_OCCUPIED_VOXELS:
        GLOBAL_OCCUPIED_VOXELS.update(vox)
        ids = np.zeros(pts.shape[0], dtype=np.int32)
        colors = ids_to_colors(ids, max_id=0)
        log_points_rr("lidar/points", pts, colors, radii=0.3)
        return

    # Update background with statics
    update_background_static(vox, dyn_mask, GLOBAL_OCCUPIED_VOXELS)

    # 2) Cluster dynamics
    dyn_pts = pts[dyn_mask]
    ids_full = np.zeros(pts.shape[0], dtype=np.int32)
    if dyn_pts.size == 0:
        # No dynamics: clear trackers, just log statics
        GLOBAL_TRACKED_OBJECTS = {}
        colors = ids_to_colors(ids_full, max_id=0)
        log_points_rr("lidar/points", pts, colors, radii=0.3)
        return

    db_labels, uniq = cluster_dynamic_points(dyn_pts, eps=0.5, min_samples=5)

    # 3) Assign IDs
    new_objs, lbl_to_id, GLOBAL_NEXT_TRACK_ID = assign_ids_for_clusters(
        dyn_pts,
        db_labels,
        uniq,
        GLOBAL_TRACKED_OBJECTS,
        GLOBAL_NEXT_TRACK_ID,
        TRACKING_THRESHOLD,
    )
    GLOBAL_TRACKED_OBJECTS = new_objs

    # Apply IDs to dynamic indices
    dyn_indices = np.where(dyn_mask)[0]
    for lbl in uniq:
        ids_full[dyn_indices[db_labels == lbl]] = lbl_to_id[lbl]

    # 4) Color and log
    max_id = max([0] + list(GLOBAL_TRACKED_OBJECTS.keys()))
    colors = ids_to_colors(ids_full, max_id=max_id)
    log_points_rr("lidar/points", pts, colors, radii=0.3)
    # Also log centroids for debugging
    color_map = get_unique_colors(max_id)
    log_centroids_rr("lidar/centroids", GLOBAL_TRACKED_OBJECTS, color_map)



# ---------------------------------------------------------------------------
# ADVANCED SEGMENTATION (Mask3D model) 
# ---------------------------------------------------------------------------
# Provides: init_segmenter, log_rerun (segmented), and fallback clustering.

try:  # Optional torch import
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:  # Optional Mask3D import (adjust path as needed)
    from mask3d.model import Mask3DModel  # type: ignore
except Exception:  # pragma: no cover
    Mask3DModel = None

_SEGMENTER = None  # Real model or fallback flag
_SEGMENTER_DEVICE = "cpu"
_ANNOTATION_LOGGED = False

def init_segmenter(model_name: str = "mask3d", weights_path: str | None = None, device: str | None = None) -> None:
    """Initialize segmentation model (Mask3D) or set up fallback clustering.

    Call once before streaming. If anything fails, a lightweight KMeans-style
    clustering is used to generate pseudo-segments.
    """
    global _SEGMENTER, _SEGMENTER_DEVICE
    if device is None:
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    _SEGMENTER_DEVICE = device

    if model_name.lower() == "none":
        _SEGMENTER = None
        return

    if model_name.lower() == "mask3d" and Mask3DModel is not None and torch is not None:
        try:
            model = Mask3DModel()  # Placeholder; adapt ctor & weights for real use
            if weights_path:
                ckpt = torch.load(weights_path, map_location=device)
                model.load_state_dict(ckpt.get("state_dict", ckpt))
            model.to(device).eval()
            _SEGMENTER = model
            return
        except Exception as e:  # pragma: no cover
            print(f"[segmentation] Mask3D init failed, using fallback: {e}")

    _SEGMENTER = "kmeans_fallback"

def _kmeans_segments(points: np.ndarray, k: int = 8) -> np.ndarray:
    if points.size == 0:
        return np.zeros(0, dtype=np.int32)
    idx = np.random.choice(points.shape[0], size=min(k, points.shape[0]), replace=False)
    centroids = points[idx]
    for _ in range(5):
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        for ci in range(centroids.shape[0]):
            mask = labels == ci
            if np.any(mask):
                centroids[ci] = points[mask].mean(axis=0)
    return labels.astype(np.int32)

def segment_points(frame: Frame) -> tuple[np.ndarray, dict[int, str]]:
    pts = frame.points
    if pts.size == 0:
        return np.zeros(0, dtype=np.int32), {}
    if _SEGMENTER is not None and _SEGMENTER != "kmeans_fallback" and torch is not None:
        try:
            with torch.no_grad():
                inp = torch.from_numpy(pts).float().to(_SEGMENTER_DEVICE)
                out = _SEGMENTER(inp[None, ...])  # Placeholder forward
                seg = out.squeeze().argmax(dim=-1) if out.dim() == 3 else out.squeeze()
                seg_np = seg.cpu().numpy().astype(np.int32)
        except Exception as e:  # pragma: no cover
            print(f"[segmentation] inference failed, fallback: {e}")
            seg_np = _kmeans_segments(pts, k=8)
    else:
        seg_np = _kmeans_segments(pts, k=8)
    label_map = {int(i): f"segment_{int(i)}" for i in np.unique(seg_np)}
    return seg_np, label_map

def _log_annotation_context(label_map: dict[int, str]) -> None:
    global _ANNOTATION_LOGGED
    if _ANNOTATION_LOGGED or not label_map:
        return
    cmap = plt.get_cmap("tab20")
    desc = []
    for idx in sorted(label_map.keys()):
        color = (np.array(cmap(idx % cmap.N)[:3]) * 255).astype(np.uint8)
        desc.append(
            rr.ClassDescription(
                info=rr.AnnotationInfo(
                    id=int(idx),
                    label=label_map[idx],
                    color=color,
                )
            )
        )
    rr.log("lidar/segments_context", rr.AnnotationContext(desc), static=True)
    _ANNOTATION_LOGGED = True

def log_rerun(frame: Frame, use_segmentation: bool = True) -> None:
    """Advanced visualization: segmentation colors + class_ids.

    Comment out this whole function (and related helpers) for the basic workshop.
    """
    pts = frame.points
    if pts.ndim != 2 or pts.shape[1] != 3:
        print("[log_rerun] invalid point shape; expected (N,3)")
        return
    if pts.shape[0] == 0:
        rr.log("lidar/points_segmented", rr.Points3D(np.zeros((0, 3), dtype=np.float32)), static=False)
        return
    if use_segmentation:
        seg_ids, label_map = segment_points(frame)
        if seg_ids.shape[0] == pts.shape[0]:
            _log_annotation_context(label_map)
            cmap = plt.get_cmap("tab20")
            color_lookup = {int(s): (np.array(cmap(int(s) % cmap.N)[:3]) * 255).astype(np.uint8) for s in np.unique(seg_ids)}
            colors = np.vstack([color_lookup[int(s)] for s in seg_ids])
            rr.log(
                "lidar/points_segmented",
                rr.Points3D(
                    pts,
                    colors=colors,
                    class_ids=seg_ids.astype(np.uint16),
                    radii=0.05,
                ),
                static=False,
            )
            return
        else:
            print("[log_rerun] seg size mismatch; falling back to intensity")
    intensities = frame.intensity if frame.intensity.size == pts.shape[0] else np.linspace(0, 1, pts.shape[0])
    norm = normalize_colors(intensities)
    colors = map_colors(norm)
    rr.log(
        "lidar/points_segmented",
        rr.Points3D(pts, colors=colors, radii=0.3),
        static=False,
    )

# Usage (advanced):
# In the main script we_client.py, initialize the segmenter once:
#   from utils import init_segmenter, log_rerun
#   init_segmenter(model_name="mask3d", weights_path="/path/to/weights.ckpt")
#   log_rerun(frame)


# Write a function for voxel grid downsampling. TIP: Use LLM to help you write it.
def voxel_grid_downsample(frame, voxel_size):
    """
    Input:
        frame: a frame dictionary containing 'points' key with Nx3 numpy array of point coordinates
        voxel_size: float, size of the voxel grid
    Output:
        downsampled_points: Mx3 numpy array of downsampled point coordinates
    """
    points = frame['points']
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    # Create a dictionary to hold voxel to points mapping
    voxel_dict = {}
    for i, voxel in enumerate(map(tuple, voxel_indices)):
        if voxel not in voxel_dict:
            voxel_dict[voxel] = []
        voxel_dict[voxel].append((points[i], intensity[i])) # Tuple of point and intensity
    # Compute the centroid of points in each voxel
    downsampled_frames= []
    for pts, intns in voxel_dict.values():
        centroid = np.mean(pts, axis=0)
        mean_intensity = np.mean(intns, axis=0)
        downsampled_frames.append({'points': centroid, 'intensity': mean_intensity})
    return downsampled_frames