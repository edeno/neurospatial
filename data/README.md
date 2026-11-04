# Example Data

The public J16 recording used by the real-data tutorial is downloaded on demand.
Data files are not committed to this repository; this README is tracked.

## The J16 bandit dataset

[Tutorial 04: real-data analysis](../examples/04_real_data_analysis.ipynb) uses the
recording from rat J16, July 10, 2021, run epoch `02_r1`, on a spatial-bandit
maze task. It is the `comrie_2026` dataset of `neurospatial.datasets`:

```python
from neurospatial.datasets import load

rec = load("comrie_2026")  # Verify the cached file or download it once.
print(rec)  # Attribution, clock, tracking, environment, units, watch-outs.

rec.times          # Unix timestamps in seconds, (n_time,)
rec.positions      # Head position in cm, (n_time, 2)
rec.spike_times    # Per-unit spike-time arrays, on the clock of rec.times
rec.linearized     # Position along the linearized track in cm
rec.env            # The authors' track graph as an Environment, 2 cm bins
track = rec.env.binning.linearization
track.graph, track.edge_order, track.edge_spacing  # NetworkX graph, edges, 15 cm
rec.params["head_orientation"]  # The file's head orientation, radians
```

The first call saves `j1620210710_02_r1_figure04_inputs.npz` (about 75 MB)
into `~/.cache/neurospatial/datasets/comrie_2026/` (or under the folder
`NEUROSPATIAL_DATA_DIR` names) and verifies its SHA-256. A verified cached
copy is reused without downloading, so run the call once in advance to work
offline. An existing file with a different checksum is left untouched; move it
aside before fetching the published copy. Browsing the rendered documentation
does not download data.

| Published input | Value |
| --- | --- |
| Archive | [Zenodo, version 1.0, DOI: 10.5281/zenodo.23020757](https://doi.org/10.5281/zenodo.23020757) |
| File | `j1620210710_02_r1_figure04_inputs.npz` |
| Size | 75,064,686 bytes |
| SHA-256 | `60383b394b597e2900545548ecac7c53a8601038ace9dbeb42d7f9a5fe1c93b3` |
| Data license | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The archive contains numeric and string arrays, loaded with
`np.load(..., allow_pickle=False)`.

`rec.headings` is None: the file's `head_orientation` is kept in
`rec.params["head_orientation"]` instead, because minus the direction of
movement while running it has a circular mean of about +13 degrees, a
constant offset from the LEDs' mounting or the rat's head posture that the
file cannot tell apart (see the watch-outs in `print(rec)`).

### Recording contents and units

The published file contains 709,321 position samples spanning approximately
1,418.6 seconds (23.6 minutes), sampled at 500 Hz, on Unix timestamps in
seconds. Its position arrays, `position/<name>` in the file, are:

| Names | Meaning and units |
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
by `neurospatial.datasets` and has no public download provided here. The place-field
and path-progression analyses do not require it.

## Separate local example: Arthur session (2022-03-24)

The following describes optional Trodes tracking data from rat Arthur on
March 24, 2022. These files are separate from the public J16 archive and
are not provided or downloaded by `neurospatial.datasets`.

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
