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

rec = load("forli_2025")              # fetches 1.1 GB of the 3.3 GB zip
print(rec.positions.shape, rec.env.n_bins)
print(rec.events["swr_probe2"].size)  # the authors' sharp-wave ripples
```

## What was recorded

An Egyptian fruit bat flying freely in a room for about 90 minutes (38
flights) while three Neuropixels probes recorded units in dorsal hippocampus.
Position is 3-D, in metres, sampled at 120 Hz: the centroid of three reflective
markers on the neural recording headstage, tracked by 16 motion-capture cameras
(the paper's Methods, for the indoor room). Flights are the authors'
segmentation at a 0.5 m/s velocity threshold.

Only the nine members the loader needs are read from the zip, by HTTP byte
range (1.1 GB): the behavior file, the three Kilosort 4 single-unit tables, the
merge list, the three ripple files (817 MB, which also hold each probe's
ripple-channel trace and every event's waveforms) and the best-probe choice,
each verified against its own SHA-256. The single-unit and ripple tables are
MATLAB `table` objects, read with `mat-io`.

Sources read: the paper (Forli, Fan, Qi & Yartsev 2025, Nature,
doi:10.1038/s41586-025-09341-z, accelerated article preview, Methods), the
record's `NP_Bat_Demo.m` (whose header describes every variable) and
`How to use.pdf`, the paper's code at github.com/kevin-qi/ripple-bat, commit
6812188d (`NP_RippleDetect_AF_v0.m`, which builds the ripple tables, and
`NP_Spatial_AF_v9_SWRs.m`, which reads them; read, not reused: the repository
states no license), and the files.

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
| `events["ripple_candidates_probe<n>"]`: `RPL_out.table.t` as stored, s; `events["swr_probe<n>"]`: its rows with `corr` at least 0.2 | stated | `NP_RippleDetect_AF_v0.m` takes `t` from `findpeaks` on the ripple power, in seconds on the LFP's time vector, which the script states is "synchronized with the other data streams"; the paper keeps events of "a minimum correlation value of 0.2". On the data, the population rate peaks in the 10 ms before `t`. |
| `params["ripple_probe"]`: `best_prb_ch` of `BPC_Dataset_*.mat`, probe 2 | stated | The script's "Best probe and channel for LFP analysis"; `NP_Spatial_AF_v9_SWRs.m` loads that probe's ripple file. |

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

- **The ripple events are the authors'.** `rec.events` holds each probe's table
  by its peak times: `ripple_candidates_probe<n>` every row, `swr_probe<n>` the
  rows at the paper's minimum correlation of 0.2 with the average event (3,350
  of 4,943 on probe 2, the file's best probe). The table's other columns
  (amplitude, duration, channel) are not kept. neurospatial did not detect
  these events. The files give them at 1 kHz, where the paper states 500 Hz.
  Flight was excluded with the headstage's accelerometer, not the file's
  flights, so a few SWRs lie in `bflying` samples; the summary counts them.
  The first seconds of the recording hold a burst of rows at about 10 Hz, which
  the files do not explain.
- **Not known to be awake.** The events are outside flight, but the files hold
  no sleep scoring, and a resting bat may doze.
- **The LFP is large and has no stated unit.** `rec.lfp` holds 384 channels at
  1250 Hz. One channel is 53 MB in memory and all of them 405 GB, so ask for the
  channels you need. The file gives a "voltage scaling factor" with no unit,
  so values are arbitrary units: do not read them as microvolts. Times are
  `red_out.t_ds` as stored, with 55 steps over 1.5 sampling periods, the longest
  7.6 ms.
- **A 3-D decode here cannot show a trajectory.** See the worked result: the
  paper's replay analysis was 1-D along clustered flight paths, which
  neurospatial cannot build yet, and 3-D fields here barely reproduce between
  halves of the session.

## Analysis choices

`rec.params`: `bin_size` 0.2 m (3-D bins about the size of a bat's body);
`min_speed`, the file's `v_th`, the authors' flight threshold; `analysis_epochs`,
the flights during the neural recording, so rate maps describe the bat in
flight; `train_epochs` and `test_epochs`, even and odd of those flights, for
held-out decoding; `dt` 0.25 s, as for the 2-D datasets; `ripple_probe`, the
probe whose events analyses use.

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

**The 3-D attempt did not reproduce the paper's result (15.2% of
flight-active units tuned, against 71%).** It is not the paper's analysis. One
3-D map on the room's voxels, pooling every path, tested against a criterion fixed before the
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

Source: Forli et al. 2025, Nature, Results and Methods ("Identification of
spatially tuned cells"), read on PubMed Central, PMC12460160.

### Our own 3-D analysis of the ripple events: its test against chance failed

This is neurospatial's own analysis, recorded in
`forli_2025.OWN_3D_ANALYSIS`. **It is not the paper's replay analysis, and its
numbers are not comparable with the paper's.** The paper's replay analysis was
1-D, along clustered flight paths, which neurospatial cannot build yet. Here the
authors' ripple events are decoded in the room's 3-D voxels instead.

The property, fixed before the run: windows around ripple events are judged
trajectories more often than the same windows after circularly shifting the
spikes, above the 95th percentile of each of two nulls (20 draws each, seed
2026).

| | Windows judged trajectories, of 2,535 |
| --- | --- |
| ripple events | 2 |
| every unit shifted by its own offset | 0 to 2, 95th percentile 2 |
| all units shifted by one common offset | 0 to 6, 95th percentile 4.1 |

**The property was not met, and this says nothing about replay in this
session.** A count this small cannot separate the events from their nulls, and
these maps cannot show a trajectory:

- 3-D place fields barely reproduce between halves of the session (median
  Spearman correlation 0.15, above);
- held-out flights decode to a median 2.2 m from the bat, against 2.4 m with the
  positions shuffled (100 ms bins, fields from the even flights, decoding the
  odd ones).

A low count here is not evidence of no replay. We planted replay into 300 of the
windows. Every spike kept its time, and its cell was drawn from these rate maps along a
real flight played 15 times faster. The rule judged 1 of the 300, 0.3%. This is a limit
of the number, not a measurement of the bat: planted events come from the model the
decoder assumes, and compression 15 is not calibrated for bats.

How it was run: `events["swr_probe2"]`, peak ± 0.15 s, overlapping windows
merged, kept inside the neural recording and clear of the file's flights; 134
units with more spikes in the flights than there are flights and at most 2 Hz;
fields over the flights from 3-D `compute_spatial_rates` (diffusion KDE,
bandwidth 0.2 m); 20 ms bins stepped by 5 ms to each window's end; judged by the rule of
`PFEIFFER_FOSTER_2013`, a MAP step under 0.6 m over at least 11 bins and at
least 1.2 m from start to end. The two distances are our choice: the published
20 and 40 cm scaled from a 2 m arena to this 5.8 m room, rounded. A shuffle per
draw takes about a minute and a half, so the test reruns the count and the first
draw of each null.

## Cite

Forli A, Fan W, Qi KK, Yartsev MM (2025). Replay and representation dynamics in
the hippocampus of freely flying bats. Nature.
<https://doi.org/10.1038/s41586-025-09341-z>

Data: Forli, Angelo (2025) Demo data for 'Replay and Representation Dynamics in
the Hippocampus of Freely Flying Bats' [Data set]. Zenodo.
<https://doi.org/10.5281/zenodo.15738988>
