# Shipley et al. (2026): mouse CA1 on an 8-arm radial maze, with rest

**The question.** Do place cells of an Alzheimer's mouse model hold their map
of a radial maze, and is the replay during rest that should stabilize it
intact? This dataset is tetrode recordings in CA1 of App NL-G-F/ChAT-cre mice
and wild-type littermates over 36 days of a radial-maze memory task, each day
with a 45-minute rest recording. It suits place fields on a track graph,
comparing them across a day's trials, and decoding position on the maze.

| | |
| --- | --- |
| Dataset id | `shipley_2026` |
| Animal and region | Male App NL-G-F/ChAT-cre and wild-type mice, CA1 pyramidal layer |
| Apparatus | 8-arm radial maze, 80 x 10 cm arms on an octagonal centre platform 24.7 cm across; a rest box on the platform |
| Archive | UCL Research Data Repository 30902972 (CC-BY-4.0): one 4.4 GB zip, of which each day reads about 47 to 162 MB by byte range |
| Sessions | 36 days of 14 mice (14 more days of the archive are not offered; see below). One is checked by hand: the default `m4005_2020-09-26` (71 units) |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'shipley_2026'")
sessions[["session", "n_units", "verified", "default"]].head()
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import load

rec = load("shipley_2026")
rates = compute_spatial_rates(
    rec.env,
    rec.spike_times,
    rec.times,
    rec.positions,
    epochs=rec.params["analysis_epochs"],
    min_speed=rec.params["min_speed"],
)
```

## What was recorded

Units from 16 tetrodes in the CA1 pyramidal layer, sorted with Kilosort 2.0
and curated in Phy. On each test day four arms were baited. The day began
with an 8-minute exploration trial, then two sessions of five trials (4
minutes at most, or until the mouse had visited the four baited arms),
separated by a 45-minute rest recording in a cylindrical box on the centre
platform; a 2-minute holding-box interval separated the trials (the paper's
Methods). Position is the headstage's LEDs, filmed by an overhead camera at
about 30 Hz.

The default day, `m4005_2020-09-26`, is an App NL-G-F/ChAT-cre mouse: 71 good
units, 11 maze trials (`0`, the exploration, to `10`) and one rest recording
(`S2`). Each mouse's genotype is in the record's `MouseStats.xlsx` and in the
summary.

Each day is read from four members of its `PhyOutput/OldData` folder:
`spike_times.npy`, `spike_clusters.npy`, `cluster_info.tsv` and
`trialData_newMUA.mat`.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| spike times: `spike_times.npy` / 30 kHz for the clusters `cluster_info.tsv` labels `good` | stated | The Methods' 30 kHz sampling; the spikes must equal the curated spike times the file's tetrodes hold, or the day is refused. |
| positions: cm = `pos.xy` (pixels) / `pixels_per_metre` x 100 | stated | The file's header: 274 pixels per metre. The camera frame's handedness (whether pixel y grows up or down the room) is not stated, so the positions keep the file's axes and may be the room's mirror image; the arms `arm_0` to `arm_7` are numbered by angle in that frame. |
| position times: per trial, a straight line from sample index to the times of the spikes the file maps to each sample | inferred | The file holds no sample times. On the default day the fitted rates are 30.047 to 30.064 Hz, not the header's 30 Hz, and the largest residual is 41 ms (about one frame). The camera's own timestamps in the zip (`experiment_1_pos.npy`) agree for the two trials checked. |
| epochs: `exploration` (trial `0`), `trials` (the test trials), `maze` (both) and `rest` (`S1`, `S2`); `rec.trials` holds the maze trials | stated | The Methods' day structure and the file's trial names. |
| environment: a star of eight 80 cm arms from a 24.7 cm platform, 2 cm bins along the track, placed on the maze trials' samples | inferred | Each arm's direction is the principal axis of its samples, and the centre the point nearest those axes; the `Conversion` gives how far the axes pass from the centre and how far the samples reach along each arm. |
| headings | none | `rec.headings` is `None`: the file stores one LED, and `pos.dir` equals the direction of displacement. |

## Watch out

- **The clock is the day's recordings joined together.** The Methods say the
  day's recordings were combined for spike sorting, so the 2-minute
  holding-box intervals are not in the clock: each trial starts within a few
  frames of the previous one's end. Durations within a trial are real;
  times between trials are not.
- **Days not offered.** The archive holds 50 days of 15 mice; 14 days cannot
  be put on one increasing clock, so the catalog does not list them (all three
  days of m4232 among them). In 12 a trial's fitted line starts before the
  previous trial's ends. In the 9 where the overlap is under one frame
  interval (33 ms), the file's spike-to-sample map gives the spikes at the
  boundary to both trials. The other 3 (`m4201_2019-12-17`,
  `m4202_2019-12-16`, `m4020_2020-08-18`, 0.8 to 10.6 s, each where a trial
  follows the rest S1) are more than that can explain, and no source read
  says why. In 2 a trial has no spikes mapped to it, so its samples cannot be
  timed:

  | Day | Why it is not offered |
  | --- | --- |
  | `m4005_2020-09-28` | Trial 8 starts 0.53 ms before trial 7 ends |
  | `m4101_2020-11-03` | Trial 2 starts 0.34 ms before trial 1 ends |
  | `m4230_2020-12-18` | Trial 9 starts 0.42 ms before trial 8 ends |
  | `m4230_2020-12-19` | Trial 5 starts 0.36 ms before trial 4 ends |
  | `m4232_2020-12-17` | Trial 2 starts 0.38 ms before trial 1 ends |
  | `m4232_2020-12-18` | Trial 9 starts 0.11 ms before trial 8 ends |
  | `m4232_2020-12-19` | Trial 5 starts 0.04 ms before trial 4 ends |
  | `m4376_2021-08-16` | Trial 9 starts 0.75 ms before trial 8 ends |
  | `m4604_2022-03-02` | Trial 1 starts 28.95 ms before trial 0 ends |
  | `m4201_2019-12-17` | Trial 6 starts 796.89 ms before rest S1 ends |
  | `m4202_2019-12-16` | Trial 6 starts 1381.64 ms before rest S1 ends |
  | `m4020_2020-08-18` | Trial 6 starts 10583.43 ms before rest S1 ends |
  | `m4020_2020-08-21` | Trials 0 to 3 have no spikes mapped to them |
  | `m4578_2021-10-18` | Trial 0 has no spikes mapped to it |

- **Rebuilt position times.** Within a trial the times are a straight line:
  dropped frames, if any, are not located. The summary gives each day's
  fitted rates and residuals. In the 2021 and 2022 days of some mice every
  fitted rate is exactly 30 Hz, so the file's map was likely built from the
  header's rate and the fit cannot measure the camera.
- **No headings.** The file stores one LED (`n_leds` 1), though the Methods fit
  two, and `pos.dir` equals `dir_disp`, the direction of displacement.
- **Processed positions.** The authors removed position jumps and
  interpolated over them (Methods), so held and fast samples describe that
  processing as much as the tracking. In the default day 20% of samples repeat
  the one before in runs of 11 or more, most of them during rest.
- **Genotype.** Pool days by genotype, not across it; the summary names each
  mouse's.
- **A missing rest.** The default day has one rest recording, `S2`; the first
  (`S1`) is not in the archive.

## Analysis choices

`rec.params`: `bin_size` 2 cm and `min_speed` 2 cm/s, the paper's rate-map
bins and running-speed filter; `analysis_epochs`, the maze trials, so rate
maps leave out the rest box; `train_epochs` and `test_epochs`, the even and
odd maze trials, for held-out decoding at `dt` 0.25 s.

## Worked result

**Status: not possible yet.**

The paper's numbers pool many mice and days (for example the peak firing
rates of 232 wild-type and 222 NL-G-F place cells), and its replay results
need a replay analysis of the rest recordings; none is a number one day can
reproduce.

## Cite

Shipley S, Abrate MP, Hayman R, Chan D, Barry C (2026). Disrupted hippocampal
replay is associated with reduced offline map stabilization in an Alzheimer's
mouse model. Current Biology 36, 859-871.e5.
<https://doi.org/10.1016/j.cub.2025.12.061>

Data: Shipley S, Abrate MP, Hayman R, Chan D, Barry C (2025). Supporting data
for: Disrupted hippocampal replay is associated with reduced offline map
stabilisation in an Alzheimer's mouse model (Version 1) [Data set]. University
College London. <https://doi.org/10.5522/04/30902972.v1>
