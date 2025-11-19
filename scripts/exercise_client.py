"""LiDAR Workshop Python WebSocket client.
Connects to a LiDAR WebSocket server, receives point cloud frames,
and visualizes them using Rerun (preferred way) or Matplotlib (backup).

Also supports saving received frames to a compressed .npz file,
and offline playback from such files.
"""

import argparse
import asyncio
import time
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import msgpack
import numpy as np
import websockets
import matplotlib.pyplot as plt

from frameclass import Frame
from utils import log_rerun, update_matplotlib

# ---------------------------------------------------------------------------
# Data structures & helpers
# ---------------------------------------------------------------------------


class StatsTracker:
    """Tracks streaming statistics (FPS & bandwidth) and formats periodic output."""

    def __init__(self, print_interval: float = 5.0) -> None:
        self.start_time = time.time()
        self.last_print = self.start_time
        self.frames = 0
        self.bytes = 0
        self.print_interval = print_interval

    def update(self, message_size: int) -> None:
        self.frames += 1
        self.bytes += message_size

    def maybe_print(self, current_scan_id: int, frame: Frame) -> None:
        now = time.time()
        if now - self.last_print < self.print_interval:
            return

        elapsed = now - self.start_time
        fps = self.frames / elapsed if elapsed > 0 else 0.0
        bandwidth_kb_s = (self.bytes / elapsed / 1024) if elapsed > 0 else 0.0
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Frame {current_scan_id} | Points: {frame.count:4d} (from {frame.original_count:4d}) | "
            f"FPS: {fps:.1f} | Bandwidth: {bandwidth_kb_s:.1f} KB/s"
        )
        self.last_print = now

    def summary(self) -> Tuple[float, int, float, float]:
        elapsed = time.time() - self.start_time
        avg_fps = self.frames / elapsed if elapsed > 0 else 0.0
        total_kb = self.bytes / 1024
        avg_bandwidth = (self.bytes / elapsed / 1024) if elapsed > 0 else 0.0
        return elapsed, self.frames, avg_fps, avg_bandwidth


def dequantize_points(points_int: np.ndarray, resolution: float = 0.01) -> np.ndarray:
    """Convert quantized int16 coordinates back to float32 meters."""
    return points_int.astype(np.float32) * resolution


def process_message(data: dict, fallback_scan_id: int) -> Optional[Frame]:
    """Decode a raw parsed MessagePack dictionary into a Frame.

    Handles both list-based and quantized-bytes formats.
    Returns None if format is not recognized.
    """
    if data.get("type") == "welcome":
        return None

    scan_id = int(data.get("scan_id", fallback_scan_id))
    timestamp = str(data.get("timestamp", ""))
    points_data = data.get("points", [])
    intensity_data = data.get("intensity", [])

    # Decode points & intensity according to format
    if isinstance(points_data, (list, tuple)):
        points = np.array(points_data, dtype=np.float32)
        intensity = np.array(intensity_data, dtype=np.float32)
        if points.ndim == 1 and points.size % 3 == 0:  # flat list
            points = points.reshape(-1, 3)
    elif isinstance(points_data, bytes):
        resolution = float(data.get("resolution", 0.01))
        shape = data.get("shape", [0, 3])
        try:
            points_int = np.frombuffer(points_data, dtype=np.int16).reshape(shape)
            points = dequantize_points(points_int, resolution)
            intensity = np.frombuffer(intensity_data, dtype=np.uint8).astype(np.float32)
        except Exception as e:  # Defensive; malformed shape
            print(f"Warning: failed to parse quantized format: {e}")
            return None
    else:
        print(f"Warning: Unknown data format for points: {type(points_data)}")
        return None

    original_count = int(data.get("original_count", points.shape[0]))
    return Frame(scan_id, timestamp, points, intensity, original_count)


def load_frames_from_npz(file_path: str) -> List[Frame]:
    """Load point cloud frames from an ``.npz`` file into a list of ``Frame``.

    Tries a few common layouts:
      1) Our own saved format: frames=<object-array of dicts>
      2) Single frame: points (N,3) and optional intensity (N,)
      3) Multi-frame arrays: points (F,N,3) and optional intensity (F,N)
      4) Indexed keys: points_0, intensity_0, ...
    """
    frames: List[Frame] = []
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load '{file_path}': {e}")
        return frames

    try:
        files = set(getattr(data, 'files', []))

        # 1) Our own saved format
        if 'frames' in files:
            arr = data['frames']
            # Could be an object array of dicts
            for i, item in enumerate(arr):
                if isinstance(item, dict):
                    points = np.array(item.get('points', []), dtype=np.float32)
                    intensity = np.array(item.get('intensity', []), dtype=np.float32)
                    scan_id = int(item.get('scan_id', i))
                    timestamp = str(item.get('timestamp', ''))
                else:
                    # Try to interpret as (points, intensity) tuple-like
                    try:
                        points = np.array(item[0], dtype=np.float32)
                        intensity = np.array(item[1] if len(item) > 1 else [], dtype=np.float32)
                    except Exception:
                        continue
                    scan_id = i
                    timestamp = ''
                frames.append(Frame(scan_id, timestamp, points, intensity, original_count=points.shape[0]))

        # 2) Single or multi-frame arrays under 'points' (+ optional 'intensity')
        elif 'points' in files:
            points = data['points']
            intensity_arr = data['intensity'] if 'intensity' in files else None

            if points.ndim == 2 and points.shape[1] == 3:
                intensity = np.array(intensity_arr, dtype=np.float32) if intensity_arr is not None else np.array([], dtype=np.float32)
                frames.append(Frame(0, '', points.astype(np.float32), intensity, original_count=points.shape[0]))
            elif points.ndim == 3 and points.shape[2] == 3:
                F = points.shape[0]
                for i in range(F):
                    pts = points[i].astype(np.float32)
                    if intensity_arr is not None:
                        try:
                            if intensity_arr.ndim == 2:
                                inten = np.array(intensity_arr[i], dtype=np.float32)
                            elif intensity_arr.ndim == 1 and intensity_arr.shape[0] == pts.shape[0]:
                                inten = np.array(intensity_arr, dtype=np.float32)
                            else:
                                inten = np.array([], dtype=np.float32)
                        except Exception:
                            inten = np.array([], dtype=np.float32)
                    else:
                        inten = np.array([], dtype=np.float32)
                    frames.append(Frame(i, '', pts, inten, original_count=pts.shape[0]))

        # 3) Indexed keys like points_0, intensity_0
        else:
            point_keys = sorted([k for k in files if k.startswith('points')])
            if point_keys:
                for i, k in enumerate(point_keys):
                    pts = np.array(data[k], dtype=np.float32)
                    inten_key = k.replace('points', 'intensity')
                    inten = np.array(data[inten_key], dtype=np.float32) if inten_key in files else np.array([], dtype=np.float32)
                    frames.append(Frame(i, '', pts, inten, original_count=pts.shape[0]))

    finally:
        try:
            data.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    if not frames:
        print("No frames found in file. Supported layouts: 'frames', 'points' (+optional 'intensity'), or indexed keys like 'points_0'.")
    else:
        print(f"Loaded {len(frames)} frame(s) from '{file_path}'.")
    return frames


def visualize_frames_offline(frames: List[Frame], visualize: bool, use_rerun: bool, fps: float = 10.0) -> None:
    """Visualize a list of frames using Rerun or Matplotlib at given FPS."""
    if not frames:
        print("Nothing to visualize: empty frames list.")
        return

    interval = max(1.0 / max(fps, 0.001), 0.001)

    # Prefer Rerun if requested
    if use_rerun:
        try:
            import rerun as rr
            rr.init("lidar_workshop_offline", spawn=True)
            print("✅ Rerun viewer opened for offline playback")
            for f in frames:
                log_rerun(f)
                time.sleep(interval)
            return
        except ImportError:
            print("Warning: rerun-sdk not available. Falling back to Matplotlib if enabled.")
            use_rerun = False

    # Matplotlib path
    if visualize or not use_rerun:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            plt.ion()
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            for f in frames:
                update_matplotlib(ax, f)
                plt.draw(); plt.pause(interval)
            plt.ioff(); plt.show()
        except ImportError:
            print("Matplotlib not available; cannot visualize offline without Rerun.")



async def connect_and_receive(
    server_url: str,
    token: str,
    visualize: bool = False,
    use_rerun: bool = False,
    save_to_file: Optional[str] = None,
    nframes_to_save: int = 0,
) -> Optional[dict]:
    """Connect to the LiDAR WebSocket server and stream frames.

    Args:
        server_url: WebSocket endpoint (e.g. ``ws://localhost:8765``).
        token: Authentication token sent immediately after connection.
        visualize: Enable basic matplotlib visualization.
        use_rerun: Use Rerun SDK instead of matplotlib for visualization.
        save_to_file: Path to ``.npz`` file for saving frames.
        nframes_to_save: Number of frames to persist (0 = none).

    Returns:
        A dictionary with session statistics or ``None`` if auth failed.
    """
    print(f"Connecting to {server_url}...")

    # Rerun setup
    if use_rerun:
        try:
            import rerun as rr  # noqa: F401
            print("Initializing Rerun viewer...")
            rr.init("lidar_workshop_client", spawn=True)
            print("✅ Rerun viewer opened - you should see it in a new window!")
            visualize = False  # Disable matplotlib if Rerun is active
        except ImportError:
            print("Warning: rerun-sdk not available. Install with: pip install rerun-sdk")
            print("Falling back to no visualization")
            use_rerun = False

    # Matplotlib setup
    fig = ax = None
    if visualize and not use_rerun:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            plt.ion()
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title("LiDAR Point Cloud - Live Stream")
            print("Matplotlib visualization enabled")
        except ImportError:
            print("Warning: matplotlib not available, visualization disabled")
            visualize = False

    saved_frames: List[Frame] = []
    stats = StatsTracker()

    async with websockets.connect(server_url) as websocket:
        # Authenticate
        await websocket.send(token)
        print("Sent authentication token")

        welcome_data = await websocket.recv()
        welcome = msgpack.unpackb(welcome_data, raw=False)
        if "error" in welcome:
            print(f"Authentication failed: {welcome['error']}")
            return None

        print(f"Connected! {welcome.get('message', '')}")
        print(f"Data format: {welcome.get('format', 'unknown')}")
        print(f"Resolution: {welcome.get('resolution', 0.01)}m")
        print()
        print("Receiving point clouds... (Press Ctrl+C to stop)")
        print("-" * 60)

        frame_index = 0  # Local sequencing (fallback if scan_id absent)

        try:
            # Keep connection alive by listening for messages with timeout
            while True:
                try:
                    # Wait for message with timeout to allow checking connection state
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                except asyncio.TimeoutError:
                    # No message received in 30s, but connection still alive
                    # Send a ping to keep connection active
                    try:
                        pong = await websocket.ping()
                        await asyncio.wait_for(pong, timeout=10.0)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Keepalive ping successful")
                    except Exception:
                        print("Connection lost (ping failed)")
                        break
                    continue

                stats.update(len(message))
                data = msgpack.unpackb(message, raw=False)
                frame = process_message(data, frame_index)
                if frame is None:
                    continue
                frame_index += 1

                # Periodic stats output
                stats.maybe_print(frame.scan_id, frame)

                # Visualization
                if use_rerun:
                    log_rerun(frame)
                elif visualize and frame_index % 2 == 0 and ax is not None:
                    update_matplotlib(ax, frame)
                    # Only import pyplot if we actually visualize
                    import matplotlib.pyplot as plt
                    plt.draw(); plt.pause(0.001)

                # Saving
                if save_to_file and nframes_to_save > 0:
                    if frame_index <= nframes_to_save:
                        saved_frames.append(frame)
                    elif frame_index == nframes_to_save + 1:
                        print(f"Reached {nframes_to_save} frames to save, stopping further saves.")
                        print(f"\nSaving {len(saved_frames)} frames to {save_to_file}...")
                        # Convert to simple serializable structure
                        serializable = [
                            {
                                "scan_id": f.scan_id,
                                "timestamp": f.timestamp,
                                "points": f.points,
                                "intensity": f.intensity,
                            }
                            for f in saved_frames
                        ]
                        np.savez_compressed(save_to_file, frames=serializable)
                        print("Saved successfully!")
                        # Close the connection after saving
                        print("Closing connection after saving frames.")
                        await websocket.close()
                        break

        except KeyboardInterrupt:
            print("\n\nStopped by user")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"\n\nConnection closed by server: {e.code} - {e.reason}")

    # Final statistics (after context manager closes)
    elapsed, frames, avg_fps, avg_bandwidth = stats.summary()
    total_kb = stats.bytes / 1024
    print("\n" + "=" * 60)
    print("Session Statistics:")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Frames received: {frames}")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Total data: {total_kb:.1f} KB")
    print(f"  Average bandwidth: {avg_bandwidth:.1f} KB/s")
    print("=" * 60)

    if visualize:
        try:
            import matplotlib.pyplot as plt
            plt.ioff(); plt.show()
        except ImportError:
            pass

    return {
        "duration_s": elapsed,
        "frames": frames,
        "avg_fps": avg_fps,
        "total_kb": total_kb,
        "avg_bandwidth_kb_s": avg_bandwidth,
    }
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Python WebSocket Client for LiDAR Workshop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic connection (no visualization)
  python ws_client.py --server ws://INSTRUCTOR_IP:PORT --token TOKEN
  
  # With Rerun visualization (recommended - best quality)
  python ws_client.py --server ws://INSTRUCTOR_IP:PORT --token TOKEN --rerun
  
  # With matplotlib visualization
  python ws_client.py --server ws://INSTRUCTOR_IP:PORT --token TOKEN --visualize
  
  # Save data to file
  python ws_client.py --server ws://INSTRUCTOR_IP:PORT --token TOKEN --save lidar_data.npz
  
  # Offline playback from saved file (Rerun)
  python ws_client.py --play data/pointclouds.npz --rerun

  # Offline playback (matplotlib) at 15 FPS
  python ws_client.py --play data/pointclouds.npz --visualize --fps 15

  # Connect to remote server with Rerun
  python ws_client.py --server ws://workshop.example.com:8765 --token mytoken --rerun

Note: Install visualization dependencies:
  - For Rerun: pip install rerun-sdk
  - For matplotlib: pip install matplotlib
        """
    )
    
    parser.add_argument(
        '--server',
        type=str,
        default='ws://localhost:8765',
        help='WebSocket server URL (default: ws://localhost:8765)'
    )
    
    parser.add_argument(
        '--token',
        type=str,
        help='Authentication token'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable matplotlib 3D visualization (basic)'
    )
    
    parser.add_argument(
        '--rerun',
        action='store_true',
        help='Use Rerun SDK for visualization (recommended - better performance and features)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save received point clouds to file (.npz format)'
    )
    # Add number of frames to save, set default to 0 and maximum to 100
    parser.add_argument(
        '--nframes',
        type=int,
        default=50,
        choices=range(0, 501),
        help='Number of frames to save when using --save (default: 50)'
    )
    # Offline playback options
    parser.add_argument(
        '--play',
        nargs='?',
        const='data/pointclouds.npz',
        type=str,
        default=None,
        help='Visualize a saved .npz file instead of live stream. If used without a value, defaults to data/pointclouds.npz.'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=10.0,
        help='Playback speed for offline visualization (frames per second). Default: 10.0'
    )
    
    args = parser.parse_args()

    try:
        # If --play is specified, run offline visualization and exit
        if args.play:
            frames = load_frames_from_npz(args.play)
            visualize_frames_offline(frames, visualize=args.visualize, use_rerun=args.rerun, fps=args.fps)
            stats = {
                'duration_s': 0.0,
                'frames': len(frames),
                'avg_fps': args.fps if frames else 0.0,
                'total_kb': 0.0,
                'avg_bandwidth_kb_s': 0.0,
            }
        else:
            stats = asyncio.run(
                connect_and_receive(
                    args.server,
                    args.token,
                    args.visualize,
                    args.rerun,
                    args.save,
                    args.nframes,
                )
            )
        if stats:
            # Provide a concise summary line for scripting usage.
            print(
                f"Summary: frames={stats['frames']} avg_fps={stats['avg_fps']:.2f} "
                f"duration={stats['duration_s']:.2f}s"
            )
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

