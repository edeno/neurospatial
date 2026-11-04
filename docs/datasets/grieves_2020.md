# Grieves et al. (2020): rat CA1 place cells in a 3-D climbing lattice

**The question.** How do hippocampal place cells represent a volume, not a
floor? Rats climbed a cubic lattice maze, 0.97 m a side, either aligned with
gravity or tilted to stand on one corner, while CA1 tetrodes were recorded
wirelessly. Position is 3-D, so the environment is a grid of voxels. The
dataset suits 3-D spatial rate maps, place fields in a volume, and comparing
the aligned and tilted lattices across rats.

| | |
| --- | --- |
| Dataset id | `grieves_2020` |
| Animal and region | rats (13), CA1 |
| Apparatus | cubic climbing lattice, 0.97 m a side (6 x 6 x 6 cubes of 150 mm tubes), aligned or tilted onto a vertex, between square arena sessions |
| Archive | UCL Research Data Repository 9926477, version 1 (CC0), `Summary data.zip`, 12.0 GB; one session's member is 66 to 708 MB |
| Sessions | 57, one per rat and date. One is checked by hand: the default `926_06042018` (24 units, tilted lattice) |
| Needs | `inflate64`, to read the zip's Deflate64 members |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'grieves_2020'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]].head()
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import load

rec = load("grieves_2020")              # reads 431 MB of the 12 GB zip
rates = compute_spatial_rates(
    rec.env, rec.spike_times, rec.times, rec.positions,
    epochs=rec.params["analysis_epochs"], method="binned",
)
```

## What was recorded

Each session is one rat on one day: a square arena, then the lattice maze,
then (in 41 of the 57 sessions) the arena again, recorded on one wireless
headstage (the paper's Supplementary Methods). Nine rats climbed the lattice
aligned (bars horizontal and vertical, `config3.lattice_type` `'normal'` in the
files) and four climbed it tilted, rotated to stand on one vertex
(`'diagonal'`). Five cameras tracked four infrared LEDs on the headstage, and
the head's 3-D path was triangulated from every pair of cameras. The loader
reads the lattice part: its 3-D positions at 50 Hz, the sorted units, and the
head azimuth.

Each session is one MATLAB v7.3 (HDF5) member of the 12 GB zip, read by HTTP
byte range, inflated with `inflate64` (the members are compressed with
Deflate64, which Python's `zipfile` cannot read) and verified against its own
SHA-256.

Sources read: the paper (Grieves, Jedidi-Ayoub, Mishchanchuk, Liu,
Renaudineau & Jeffery 2020, Nature Communications 11:789, main text on Europe
PMC, PMC7005894) and its Supplementary Information (Supplementary Methods and
Table 1), the record's `Summary data readme.m`, and the files.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| positions: `lattice_3d.pox/poy/poz` / 10, mm to cm | inferred | No source states the unit. The stored positions' speed is 10 times the stored velocity `pov` (10.03 on the default session), whose median is a climbing rat's speed in cm/s; `config3.lat_size` is 970 and every bar 970 long, against the 0.97 m lattice. A session whose ratio is not near 10 is refused. |
| times: `lattice_3d.pot` as stored, seconds | stated | The file's `position_srate_hz` is 50 and `pot` steps 0.02 s. The clock joins the session's recordings end to end (below). |
| units: every tetrode/cluster field, cluster 0 of each tetrode left out | inferred | The cluster-0 fields hold only spike times, with none of the per-cell analyses the other clusters have. Without them every rat's sessions hold the number of cells Supplementary Table 1 gives for it (1064 in all). |
| headings: `lattice.poh`, degrees to radians, turned into the positions' frame | inferred | No source read gives `poh`'s unit or frame. It runs from 0 to 360, so degrees, and is taken to be an azimuth in the frame of the 2-D positions `lattice.pox/poy`. The loader fits the map from those 2-D positions to the 3-D x and y (a rotation or reflection, a scale and a shift) and turns the azimuth by its rotation or reflection. In the default session the 2-D positions are the 3-D x, y mirrored in y and shifted, sample by sample, so the heading is `-poh`; the turned azimuth follows the direction of movement (below). |
| epochs: `square1`, `lattice`, `square2` from `part_config`, a row per recording | stated | The file's `part_config` and `session_names`. |
| epochs: rows past the recording (its first to last sample or spike) clipped to it | inferred | In two sessions `square2` ends after the last sorted spike, past the clock check's slack. Each clipped row is in the `Conversion` and under "Watch out". `params["analysis_epochs"]` and the decoding blocks come from the clipped rows. |
| `params["lattice_bars"]`: the lattice's 147 bars, `lattice_3d.lx/ly/lz` / 10, shape (147, 2, 3) | stated | The readme: maze outlines are stored as `lx`, `ly`, `lz`; each bar is 97 cm, the lattice's side. |
| environment: every voxel of the authors' 3-D rate-map grid, 5 cm voxels | inferred | `lattice_3d.dwellmap` gives the grid's shape; `lxrm/lyrm/lzrm` are the bars in its voxel units, `lx / 50 + c` exactly, which places it. Each voxel is taken to span `[i - 1, i]` in those units, where in every session the authors' kernel dwell time matches the stored `dwellmap` best. |

The environment is the authors' analysis volume: it spans the tracked samples
padded by about two voxels, so it is sized from the samples, not from the
apparatus, and is marked inferred. It holds every tracked sample of the default
session. Most of its voxels are air around the bars that a climbing rat never
reaches, and their rates are NaN, not zero.

The stated sampling rate is the files' 50 Hz. The stated position span runs
from 80% of the 0.97 m side to 120% of the tilted lattice's height
(0.97 x √3 = 1.68 m), so the units check catches positions read in mm or m.

The lattice's bars are not drawn by `rec.plot_overview()` or
`rec.animation()`; draw them over a 3-D path yourself:

<!-- docs-test: skip needs a download -->
```python
import matplotlib.pyplot as plt

ax = plt.figure().add_subplot(projection="3d")
for start, stop in rec.params["lattice_bars"]:
    ax.plot(*zip(start, stop), color="0.6", linewidth=0.5)
run = (rec.times >= 2000) & (rec.times < 2120)
ax.plot(*rec.positions[run].T, color="tab:red")
ax.set(xlabel="x (cm)", ylabel="y (cm)", zlabel="z (cm)")
```

## Watch out

- **Only the lattice part is loaded.** The square arena parts are a different
  apparatus, so their positions are not in `rec.positions`. Their spikes are
  in `rec.spike_times`, and the epochs `square1` and `square2` hold them:
  restrict analyses to `params["analysis_epochs"]`, the lattice.
- **A joined clock.** The clock joins the session's recordings end to end; the
  rests of about 10 minutes between parts are not on it. In one session
  (`894_09012018`) the recordings were made on two days. The summary says so
  where the recordings' names give different dates.
- **Azimuths stored as exactly 0.** In the default session 26.7% of `poh`
  samples are exactly 0 deg, most in long runs while the head moves: no
  direction a rat holds that exactly, so most likely samples without an
  azimuth, though the file does not say. They are kept; the summary gives the
  value they have in `rec.headings`, which selects them. The head-direction
  histogram of the overview shows them as a peak at 0.
- **Held azimuths.** 27.8% of the default session's heading samples repeat the
  sample before, in runs of up to 44 s while the head moves along its path.
  The summary counts them.
- **Heading frames not verified.** In 16 of the 56 sessions that load, the
  turned azimuth does not follow the direction of movement (circular mean
  within 0.2 rad of 0 and median absolute deviation under 45 deg). In 8 of
  them the 2-D positions hold still for more than half of their steps, so
  their map to the 3-D frame is undetermined; in the other 8 the mean is
  within 8 deg of 0 but the spread is above 45 deg. The refused session
  `951_13092018`, loaded with `accept_failed_checks=("units",)`, is not
  verified either: its mean is 179 deg. Those sessions keep the azimuth and
  say the frame is not verified.
- **The head beyond the bars.** 24.9% of the default session's samples lie
  outside the lattice's cube, up to 33 cm beyond a face: the tracked head of a
  rat on the outer bars. The environment, the authors' grid, holds all of
  them, in every session; the summary gives the share.
- **Units that stop firing.** In the default session t14c4, t14c5 and t14c6
  stop firing partway through the lattice part, and t15c1 starts late in the
  first arena part; the file does not say whether the clusters were lost.
- **Single precision.** Where `lattice_3d.pot` or the spike times are stored
  as float32 (in every session of the archive both are), one float32 step at the end
  of a session is about 0.5 ms, so the sample intervals vary around 0.02 s.
  The summary says so only for a session whose file stores them as float32.
- **Processed positions.** The Supplementary Methods say the cameras tracked
  the LEDs at 25 Hz, the paths from every pair of cameras were combined, and
  missing stretches were interpolated and smoothed. The file holds the result
  at 50 Hz, so speeds describe that processing as much as the tracking.
- **Square parts that end after the last spike.** In `894_21122017` and
  `926_16042018` the file lists `square2` as 17.5 s and 94.6 s longer than
  the data: the square parts hold no position sample, and the sorted units
  stop firing before the listed part ends. The epoch is clipped to the
  recording (its first to last sample or spike), so a per-epoch rate over
  `epochs["square2"]` covers the data, not the listed part. The summary
  gives the listed and the clipped row.
- **A session refused by the checks.** `951_13092018` fails the units check:
  its positions span 207.7 cm in y, past the stated 201.6 cm. It loads with
  `accept_failed_checks=("units",)`.
- **Unknown configuration.** `896_16012018`'s `config3.lattice_type` is
  `'atrack'`, which no source read describes, and its `pov` holds no value, so
  its mm rests on the lattice's size alone.

## Analysis choices

`rec.params`: `bin_size` 5 cm, the paper's voxel; `min_speed` `None`, since the
paper's rate maps state no speed filter; `analysis_epochs`, the lattice part;
`train_epochs` and `test_epochs`, alternating 60 s blocks of each lattice
recording, for held-out decoding; `dt` 0.25 s, as for the other datasets.

## Worked result

**Status: not possible yet.**

Every result read in the paper pools sessions: the main text over the 57
sessions of 13 rats, and Supplementary Table 1 per rat over all of its sessions
(rat 926: 9 sessions, 295 cells, 232 place cells in the tilted lattice). A
cluster is a place cell by its waveform's peak-to-trough width, its mean rate
and its spatial information in the session with the most spikes (Methods,
"Place cell criteria"); the loader gives one session at a time and reads no
waveforms. The cell counts per rat match Supplementary Table 1 when every
session of the rat is loaded, which checks the units the loader keeps, not a
finding.

## Cite

Grieves RM, Jedidi-Ayoub S, Mishchanchuk K, Liu A, Renaudineau S, Jeffery KJ
(2020). The place-cell representation of volumetric space in rats. Nature
Communications 11:789. <https://doi.org/10.1038/s41467-020-14611-7>

Data: Grieves, Roddy; Jedidi-Ayoub, Selim; Mishchanchuk, Karyna; Liu, Anyi;
Renaudineau, Sophie; Jeffery, Kate (2020). Place-cell representation of
volumetric space in rats. University College London. Dataset.
<https://doi.org/10.5522/04/9926477.v1>
