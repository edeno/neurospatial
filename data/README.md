# Example Data

Example datasets for the `neurospatial` package. The raw files (`*.pkl`,
`*.mp4`, `*.h264`, `*.parquet`) are **not** checked in — they're hosted
externally and downloaded on demand. This README and the
`load_bandit_data.py` helper are the only files tracked here.

## Downloading the J16 bandit dataset

The J16 plus-maze recording used by
[`examples/19_real_data_bandit_task.ipynb`](../examples/19_real_data_bandit_task.ipynb)
is hosted on Zenodo. To run that notebook locally:

```bash
# Replace <ZENODO_DOI> with the published DOI when available.
# Until then, ask the maintainer for a pre-publication download link.
wget https://zenodo.org/record/<RECORD_ID>/files/j1620210710_02_r1_position_info.pkl -P data/
wget https://zenodo.org/record/<RECORD_ID>/files/j1620210710_02_r1_HPC_spike_times.pkl -P data/
wget https://zenodo.org/record/<RECORD_ID>/files/j1620210710_02_r1_track_graph.pkl -P data/
wget https://zenodo.org/record/<RECORD_ID>/files/j1620210710_02_r1_linear_edge_order.pkl -P data/
wget https://zenodo.org/record/<RECORD_ID>/files/j1620210710_02_r1_linear_edge_spacing.pkl -P data/
```

The notebook prints the same URL and exits cleanly when the data files
are missing, so you can browse the rendered notebook on the docs site
without downloading anything.

---

## Arthur Session (2022-03-24)

Real trodestrack data from rat "Arthur" recorded on March 24, 2022.
To be used as example data for `neurospatial` package.

### Arthur Files

- **`arthur20220324_position_info.parquet`** - Camera tracking data (73,450 frames, 40 min)
- **`20220324_arthur_02_r1.mp4`** - Video recording (279 MB)

### Arthur Data Format

#### Timestamps

- **Format**: Unix timestamps (float64) as pandas DataFrame index
- **Example**: 1648163575.123 (seconds since epoch)

#### Camera Data

- **Frames**: 73,450
- **Rate**: 30.4 Hz
- **Duration**: 2416.5 seconds (40.3 minutes)
- **Columns**:
  - `xloc`, `yloc`: LED positions in pixels (raw 16-bit integers) of first LED
  - `xloc2`, `yloc2`: LED positions in pixels (raw 16-bit integers) of second LED
  - `video_frame_ind`: Video frame index
  - `HWframeCount`: Hardware frame count
  - `HWTimestamp`: Hardware timestamp
- **Units**: Pixels (raw 16-bit integers)
- **Conversion to meters**: `position_m = pixel_value × 0.0022` (user-provided)
- **LED separation**: 21.6 pixels = 4.75 cm

### Loading Arthur Data

```python
import pandas as pd

pos_df = pd.read_parquet('arthur20220324_position_info.parquet')
```

---

## J16 Bandit Session (2021-07-10)

Neural recording data from rat "J16" on a bandit task, recorded July 10, 2021.
Used as example data for `neurospatial` place field and linearization examples.

### J16 Files

- **`j1620210710_02_r1_position_info.pkl`** - Position tracking data (709,321 samples, ~24 min)
- **`j1620210710_02_r1_HPC_spike_times.pkl`** - Hippocampal spike times (203 units)
- **`j1620210710_02_r1_track_graph.pkl`** - Plus maze track graph (10 nodes, 9 edges)
- **`j1620210710_02_r1_linear_edge_order.pkl`** - Edge ordering for linearization
- **`j1620210710_02_r1_linear_edge_spacing.pkl`** - Node spacing (15 cm)
- **`20210710_j16_02_r1.1.mp4`** - Video recording (49 MB, 820×780, ~24 min at 30 fps)
- **`load_bandit_data.py`** - Loading utility function

### J16 Data Format

#### Position Info

- **Samples**: 709,321
- **Rate**: 500 Hz
- **Duration**: 1418.6 seconds (~23.6 minutes)
- **Index**: Unix timestamps (float64) as pandas DataFrame index
- **Columns**:
  - `head_position_x`, `head_position_y`: 2D position (cm)
  - `head_orientation`: Head direction (radians)
  - `head_velocity_x`, `head_velocity_y`: Velocity components (cm/s)
  - `head_speed`: Speed magnitude (cm/s)
  - `linear_position`: Linearized track position (0–608 cm)
  - `track_segment_id`: Track segment identifier (0–8)
  - `projected_x_position`, `projected_y_position`: Position projected onto track
  - `patch_id`: Reward patch identifier (1, 2, or 3)
- **Spatial extent**: X: [30, 233] cm, Y: [33, 223] cm

#### Spike Times

- **Units**: 203 hippocampal units
- **Total spikes**: 870,018
- **Format**: List of numpy arrays, one per unit

#### Track Graph (Plus Maze)

- **Nodes**: 10 (including 3 arm endpoints, 3 junction nodes, center)
- **Edges**: 9 (tree structure)
- **Patches**: 3 reward locations at arm endpoints

### Loading J16 Data

```python
from data.load_bandit_data import load_neural_recording_from_files

data = load_neural_recording_from_files('data/', 'j1620210710_02_r1')

position_info = data['position_info']      # pd.DataFrame
spike_times = data['spike_times']          # list[np.ndarray]
track_graph = data['track_graph']          # networkx.Graph
linear_edge_order = data['linear_edge_order']  # list[tuple]
linear_edge_spacing = data['linear_edge_spacing']  # float
```

### Track Graph Annotation Example

The script `track_graph_annotation_example.py` demonstrates how to use
neurospatial's track graph annotation tool with the J16 video.

```bash
# Visualize the existing track graph overlaid on video frame
uv run python data/track_graph_annotation_example.py visualize

# Create a new track graph from scratch (interactive napari)
uv run python data/track_graph_annotation_example.py create

# Edit the existing track graph (interactive napari)
uv run python data/track_graph_annotation_example.py edit

# Load saved track graph and create Environment
uv run python data/track_graph_annotation_example.py load
```

**Keyboard Shortcuts** (for create/edit modes):

- `A` - Add node mode
- `E` - Add edge mode (click two nodes)
- `X` - Delete mode
- `Shift+S` - Set start node
- `Ctrl+Z` - Undo
