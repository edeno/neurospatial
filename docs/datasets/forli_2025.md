# Forli et al. (2025): bat hippocampus during free flight

**The question.** What does the hippocampus of a flying bat represent, and does
it replay flight paths at rest? This dataset is the demo session of the paper's
Zenodo record: about 90 minutes of free flight in a room (38 flights) with 289
units from three Neuropixels probes. Positions are 3-D, so the environment is a
grid of voxels. It suits 3-D spatial rate maps and 3-D decoding.

| | |
| --- | --- |
| Dataset id | `forli_2025` |
| Animal and region | Egyptian fruit bat, dorsal hippocampus |
| Apparatus | indoor flight room, 5.6 x 5.2 x 2.5 m, with feeders |
| Archive | Zenodo 15738988 (CC-BY-4.0), `Demo_Session.zip`, 3.3 GB |
| Sessions | 1: `Dataset_3_14543_240419`, checked by hand |
| Needs | `mat-io`, which installs only on Python 3.11 or later |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

catalog().query("dataset == 'forli_2025'")[["session", "n_units", "size_bytes", "verified"]]
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial.datasets import load

rec = load("forli_2025")              # fetches 260 MB of the 3.3 GB zip
print(rec.positions.shape, rec.env.n_bins)
```

## What was recorded

An Egyptian fruit bat flying freely in a room for about 90 minutes (38
flights) while three Neuropixels probes recorded units in dorsal hippocampus.
Position is 3-D, in metres, sampled at 120 Hz: the centroid of three reflective
markers on the neural recording headstage, tracked by 16 motion-capture cameras
(the paper's Methods, for the indoor room). Flights are the authors'
segmentation at a 0.5 m/s velocity threshold.

Only the five members the loader needs are read from the zip, by HTTP byte
range (260 MB): the behavior file, the three Kilosort 4 single-unit tables and
the merge list, each verified against its own SHA-256. The single-unit tables
are MATLAB `table` objects, read with `mat-io`.

Sources read: the paper (Forli, Fan, Qi & Yartsev 2025, Nature,
doi:10.1038/s41586-025-09341-z, accelerated article preview, Methods), the
record's `NP_Bat_Demo.m` (whose header describes every variable) and
`How to use.pdf`, and the files.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| positions: `r` as stored, in metres | inferred | The file states no unit. It is metres because `r_lim` spans 5.8 x 5.2 x 2.3 against the paper's 5.6 x 5.2 x 2.5 m room, and `v_th` is 0.5 against the paper's 0.5 m/s flight threshold. |
| spike times: `spikeTimes_usec` (the global clock) / 1e6 | stated | `NP_Bat_Demo.m`. `localSpikeTimes_usec` is the probe's own clock ("DO NOT USE", the script) and is not used. |
| putative duplicate units in `merges_Dataset_*.mat` pooled under the first cluster's id, sorted | stated | As the script does; the script concatenates without sorting. |
| in every unit, a spike within 1 ms of its predecessor dropped | stated | The script's rule. The `Conversion` gives the number dropped. |
| `epochs["flight"]` from the 1-based sample indices `f_smp` | stated | The file. |
| `epochs["neural_recording"]` from the first to the last spike of any unit; `params["analysis_epochs"]` the flights clipped to it | inferred | The files hold no recording start or stop. |
| environment: every voxel of a grid that tiles `r_lim` exactly (29 x 26 x 11 cells of 0.2 x 0.2 x 0.209 m) | stated | `r_lim`, "boundaries of the room/outdoor arena" (the script); the file's frame, not fitted. |

The grid has the whole number of cells nearest to each side over 0.2 m, a tie
going to fewer, larger cells: the 2.3 m height is 11.5 cells of 0.2 m, and
cells of exactly 0.2 m would make the grid reach past the floor or the ceiling.
`r_lim` is used because it is stated in the positions' frame; the paper gives
only the room's size. The environment holds the air the bat could fly
through, not only the voxels it visited, so unvisited voxels have NaN rates,
not zero.

## Watch out

- **One demo session.** This is the record's one demo session; the paper
  reports 23 sessions from six bats (the Methods of Forli et al. 2025), and the
  record holds no other.
- **Samples outside the room.** 3.7% of samples lie beyond the -y bound of
  `r_lim`, 0.1% of them in the file's flights (`bflying`). They are kept, and
  not set to NaN. The load-time check reports the share of stopped and of
  moving samples among them, split at the 0.5 m/s flight threshold, and the
  room's `Conversion` says where they lie.
- **The behavior starts before the neural recording.** The summary gives the
  behavior's span and the spikes' and says how many flights lie wholly outside
  the neural recording, so analyses use `params["analysis_epochs"]`, the
  flights clipped to it.
- **Interneurons are kept.** The summary counts the units that fire above the
  rate at which `NP_Bat_Demo.m` sets units aside as putative interneurons
  before its place analyses. All are kept here, with label `None`.
- **Processed positions.** Positions are the authors' processed behavior, not
  raw tracking: the script says "Processing mainly includes smoothing of
  position and flight segmentation", and the paper that positions "were
  preprocessed ... to obtain continuous and smooth 3D positional data". Speeds
  and held-sample counts describe that processing as much as the tracking.

## Analysis choices

`rec.params`: `bin_size` 0.2 m (3-D bins about the size of a bat's body);
`min_speed`, the file's `v_th`, the authors' flight threshold; `analysis_epochs`,
the flights during the neural recording, so rate maps describe the bat in
flight; `train_epochs` and `test_epochs`, even and odd of those flights, for
held-out decoding; `dt` 0.25 s, as for the 2-D datasets.

## Worked result

**Status: not possible yet.**

The finding to reproduce: most flight-active units are spatially tuned in
flight. The paper: "a large portion of single units that were active during
flight exhibited robust spatial selectivity (place cells: 990 out of 1,386
flight-active neurons (71%), n = 6 bats" (Results); spatially tuned cells have
"more than one spike per flight, more than 3 Hz firing at the field peak, a
minimum stability of 0.4 and a maximum pks value of 0.5" (Methods,
"Identification of spatially tuned cells"), stability being "the Spearman
correlation between 1D fields for the first versus the second half of the
flights".

**Why it is not possible yet.** The paper's place cells are defined on 1-D
fields of each clustered flight path, and neurospatial has no flight-path
clustering or path fields.

**A 3-D attempt, which is not the paper's analysis.** One 3-D map on the
room's voxels, pooling every path, tested against a criterion fixed before the
run (more than half of the flight-active units tuned), which was not met:

- 31 of 204 flight-active units tuned (15.2%), against the paper's 71%;
- flight-active: more than 37 spikes in the 37 flights of the neural recording
  (204 of 289 units); tuned: peak above 3 Hz and a Spearman
  `compare_spatial_rates` of at least 0.4 between the first 18 and last 19
  flights; maps from 3-D `compute_spatial_rates` (diffusion KDE, bandwidth 0.2
  m, our choice) on `rec.env`'s 0.2 m voxels.

How it differs from the paper:

- the paper's fields are 1-D, one per clustered flight path (0.15 m bins, a
  seven-bin Gaussian window); this is one 3-D map pooling every path, whose
  sparsely visited voxels are too noisy to be stable (every unit peaks above 3
  Hz; the median first-half against second-half Spearman correlation is 0.15
  over a median 183 shared voxels, and even against odd flights 0.13);
- the one-field condition (pks at most 0.5) needs the 1-D fields;
- flight-active is not defined beyond the spike condition; this uses more than
  one spike per flight on average;
- the paper pools 23 sessions of 6 bats; this is one demo session.

Source: Forli et al. 2025, Nature, Results ("Hippocampal replay of flight
trajectories") and Methods ("Replay analysis: spike sequences"), read on
PubMed Central, PMC12460160.

## Cite

Forli A, Fan W, Qi KK, Yartsev MM (2025). Replay and representation dynamics in
the hippocampus of freely flying bats. Nature.
<https://doi.org/10.1038/s41586-025-09341-z>

Data: Forli, Angelo (2025) Demo data for 'Replay and Representation Dynamics in
the Hippocampus of Freely Flying Bats' [Data set]. Zenodo.
<https://doi.org/10.5281/zenodo.15738988>
