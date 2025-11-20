# EmbliCats LiDAR Lab exercises

This repository contains the exercises for the EmbliCats LiDAR Lab.

## Setup

```bash
git clone https://github.com/emblica/lidar-lab-exercises.git
cd lidar-lab-exercises
```

After that you can go to the exercise notebooks, starting with `00_getting_started.ipynb`. Some quick start information below as well.

## LiDAR Workshop Clients

Workshop attendees can visualize live LiDAR point clouds using two different client options. All connect via WebSocket (no ZMQ installation required).

### Option 1: Web Viewer (No Installation Required!)

**Simplest option:** Just open the `web/viewer.html` file in your browser, and fill in `IP`, `PORT` and `TOKEN` provided by your workshop host.

**Controls:**
- **Left Mouse**: Rotate view
- **Right Mouse**: Pan view
- **Scroll Wheel**: Zoom in/out
- **Color Mode**: Choose how points are colored
- **Point Size**: Adjust point size

### Option 2: Custom Python Client (Requires Python and other Package installations)

For attendees who want to process data programmatically. Setup your environment first!

NOTE: This project targets Python 3.11 or 3.12.

#### Option A: *With uv (recommended):*
```bash
uv sync
```

If that fails due to python version being >3.12, try this first:

```bash
uv python pin 3.12 # or 3.11
```

#### Option B: *With pyenv, venv, global environments etc. using pip:*

Activate your environment first! Then install requirements with:

```bash
pip install -r requirements.txt
```

-------------------------------------------

## Usage:
```bash
# Basic connection (no visualization)
uv run scripts/exercise_client.py --server ws://INSTRUCTOR_IP:PORT --token INSTRUCTOR_TOKEN

# With Rerun visualization (recommended - best quality)
uv run scripts/exercise_client.py --server ws://INSTRUCTOR_IP:PORT --token INSTRUCTOR_TOKEN --rerun

# With matplotlib visualization
uv run scripts/exercise_client.py --server ws://INSTRUCTOR_IP:PORT --token INSTRUCTOR_TOKEN --visualize

# Save data to file
uv run scripts/exercise_client.py --server ws://INSTRUCTOR_IP:PORT --token INSTRUCTOR_TOKEN --save FILENAME.npz
  
# Offline playback from saved file (Rerun)
uv run scripts/exercise_client.py --play FILENAME.npz --rerun

# Offline playback (matplotlib) at 15 FPS
uv run scripts/exercise_client.py --play FILENAME.npz --visualize --fps 15
```

NOTE: If you are not using `uv`, just run as `python` instead of `uv run` :

```bash
python scripts/exercise_client.py ...
``` 

----------------------------------------------

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
