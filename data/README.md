# Example Data

The public J16 recording used by the real-data tutorial is downloaded on demand.
Data files are not committed to this repository; this README and
`load_bandit_data.py` are tracked.

## Downloading the J16 bandit dataset

[Tutorial 04: real-data analysis](../examples/04_real_data_analysis.ipynb) uses the
recording from rat J16, July 10, 2021, run epoch `02_r1`, on a spatial-bandit
maze task.

From the repository root, download the published file with:

```bash
uv run python data/load_bandit_data.py
```

This saves `data/j1620210710_02_r1_figure04_inputs.npz` (about 75 MB) and
verifies its SHA-256. A verified cached copy is reused without downloading.
An existing file with a different checksum is left untouched; move it aside
before fetching the published copy. To choose another directory:

```bash
uv run python data/load_bandit_data.py --data-path /path/to/recordings
```

The tutorial fetches and verifies the file when run, so its first run needs an
internet connection. Download it in advance to run offline. Browsing the
rendered documentation does not download data.

| Published input | Value |
| --- | --- |
| Archive | [Zenodo, version 1.0, DOI: 10.5281/zenodo.23020757](https://doi.org/10.5281/zenodo.23020757) |
| File | `j1620210710_02_r1_figure04_inputs.npz` |
| Size | 75,064,686 bytes |
| SHA-256 | `60383b394b597e2900545548ecac7c53a8601038ace9dbeb42d7f9a5fe1c93b3` |
| Data license | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The archive contains numeric and string arrays, loaded with
`np.load(..., allow_pickle=False)`. It replaces the five pickle/joblib files
previously used by these examples; those files are not needed.

### Loading J16 data

```python
from data.load_bandit_data import fetch_bandit_data, load_neural_recording_from_files

fetch_bandit_data("data/")  # Verify the cached file or download it once.
data = load_neural_recording_from_files("data/", "j1620210710_02_r1")

position_info = data["position_info"]          # pandas DataFrame
spike_times = data["spike_times"]              # Per-unit spike-time arrays
track_graph = data["track_graph"]              # NetworkX graph
linear_edge_order = data["linear_edge_order"]  # Ordered track edges
linear_edge_spacing = data["linear_edge_spacing"]  # 15 cm
```

`load_neural_recording_from_files` reads the local NPZ and returns the same
dictionary keys as the earlier loader. Fetching is explicit; importing the
helper does not download anything. Use the same directory in both calls if
you chose a custom download location.

### Recording contents and units

The published file contains 709,321 position samples spanning approximately
1,418.6 seconds (23.6 minutes), sampled at 500 Hz. The position DataFrame has
Unix timestamps in seconds as its index and these columns:

| Columns | Meaning and units |
| --- | --- |
| `head_position_x`, `head_position_y` | Head position in cm |
| `head_orientation` | Head direction in radians |
| `head_velocity_x`, `head_velocity_y`, `head_speed` | Velocity components and speed in cm/s |
| `linear_position` | Position along the linearized track in cm |
| `track_segment_id` | Track segment identifier |
| `projected_x_position`, `projected_y_position` | Head position projected onto the track, in cm |
| `patch_id` | Reward-patch identifier |

The 203 hippocampal spike-time arrays contain 870,018 spikes in total; 21
arrays are empty in this epoch. Spike times use the same clock as position.
The track graph has 10 nodes and 9 edges, with an edge spacing of 15 cm for
linearization. Spike times are clipped to the epoch's first and last position
timestamps; this is an epoch export rather than a full recording session.

### Provenance and attribution

These are derived analysis inputs from the spatial-bandit recordings of
[Comrie et al. (2026)](https://doi.org/10.1016/j.neuron.2026.08.023).
Position comes from LED tracking processed through Spyglass, with smoothing,
upsampling to 500 Hz and track linearization already applied. The spike
arrays were exported from the hippocampal Mountainsort4 sorting after the
source workflow's automatic curation. The public input preserves their
exported order, but does **not** include original unit identities, waveforms
or per-unit curation labels. Unit numbers in the examples identify array
positions and do not establish single-unit quality or cell type.

The related [DANDI dandiset 001942](https://dandiarchive.org/dandiset/001942)
contains raw recording and position data; the exact hippocampal spike sorting
used here is supplied by the Zenodo input rather than that DANDI export.
When reusing these data, attribute the study and the
[versioned Zenodo input](https://doi.org/10.5281/zenodo.23020757) under its
CC BY 4.0 license.

### Optional video

The local J16 video `20210710_j16_02_r1.1.mp4` used by some video-overlay
demonstrations is **not** part of the Zenodo archive. It is not downloaded
by this helper and has no public download provided here. The place-field
and path-progression analyses do not require it.

## Separate local example: Arthur session (2022-03-24)

The following describes optional Trodes tracking data from rat Arthur on
March 24, 2022. These files are separate from the public J16 archive and
are not provided or downloaded by `load_bandit_data.py`.

- `arthur20220324_position_info.parquet`: camera tracking (73,450 frames,
  approximately 40 minutes).
- `20220324_arthur_02_r1.mp4`: video recording (279 MB).

The tracking DataFrame uses Unix timestamps in seconds as its index. Its
columns include `xloc`, `yloc`, `xloc2`, `yloc2` (the two LED positions in
pixels), `video_frame_ind`, `HWframeCount` and `HWTimestamp`. The documented
calibration is `position_m = pixel_value * 0.0022`, with an LED separation
of 21.6 pixels (4.75 cm). If you have this local file, load it with:

```python
import pandas as pd

pos_df = pd.read_parquet("data/arthur20220324_position_info.parquet")
```
