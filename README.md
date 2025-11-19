# EmbliCats LiDAR Lab exercises

This repository contains the exercises for the EmbliCats LiDAR Lab.

## Setup

```bash
git clone https://github.com/EmbliCats/lidar-lab-exercises.git
cd lidar-lab-exercises
```

After that you can go to the exercise notebooks, starting with `00_getting_started.ipynb`. Some basic information below as well.

## LiDAR Workshop Clients

Workshop attendees can visualize live LiDAR point clouds using two different client options. All connect via WebSocket (no ZMQ installation required).

### Quick Start for Attendees:

### Option 1: Web Viewer (No Installation Required!)

**Simplest option:** Just open the `web/viewer.html` file in your browser.

**Controls:**
- **Left Mouse**: Rotate view
- **Right Mouse**: Pan view
- **Scroll Wheel**: Zoom in/out
- **Color Mode**: Choose how points are colored
- **Point Size**: Adjust point size

### Option 2: Custom Python Client (Requires Python and other Package installations)

For attendees who want to process data programmatically. Setup your environment first!

#### With uv (recommended):
```bash
uv sync
```

#### Pyenv, venv, global environments etc. with pip:

```bash
pip install -r requirements.txt
```



**Usage:**
```bash
# Basic connection (no visualization)
uv run scripts/exercise_client.py --server ws://INSTRUCTOR_IP:PORT --token INSTRUCTOR_TOKEN

# With Rerun visualization (recommended - best quality)
uv run scripts/exercise_client.py --server ws://INSTRUCTOR_IP:PORT --token INSTRUCTOR_TOKEN --rerun

# With matplotlib visualization
uv run scripts/exercise_client.py --server ws://INSTRUCTOR_IP:PORT --token INSTRUCTOR_TOKEN --visualize

# Save data to file
uv run scripts/exercise_client.py --server ws://INSTRUCTOR_IP:PORT --token INSTRUCTOR_TOKEN --save lidar_data.npz
  
# Offline playback from saved file (Rerun)
uv run scripts/exercise_client.py --play data/FILENAME.npz --rerun

# Offline playback (matplotlib) at 15 FPS
uv run scripts/exercise_client.py --play data/FILENAME.npz --visualize --fps 15
```

NOTE: If you are not using `uv`, instead of running with `uv run` use just `python scripts/exercise_client.py ...` 

---

## Troubleshooting

**Connection refused:**
- Verify the server URL and port (e.g. ws://INSTRUCTOR_IP:PORT)
- Check firewall settings

**Authentication failed:**
- Verify the token matches the server configuration
- Check that you're sending the token as the first message

**Visualization issues:**
- If you turned off rerun on window in the notebook, you need to restart the kernel and run the notebook again.
- Web: Check browser console for JavaScript errors
