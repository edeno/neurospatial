# Huszár et al. (2022): mouse CA1 on a figure-eight maze, with sleep and ripples

**The question.** Which hippocampal sequences do sharp-wave ripples carry while a
mouse pauses on a maze, and in the sleep around it? These mice ran spatial
alternation on a figure-eight maze for water, between hours of rest in the home
cage, with up to 376 CA1 units recorded at once. Each file holds the units, the
position, the trials, the sleep states and the ripples, so it suits place fields
on a maze with two loops, ripple-triggered analyses, awake replay and sleep
reactivation. Yang et al. (2024) reused these data for "Selection of experience
for memory by hippocampal sharp wave ripples".

| | |
| --- | --- |
| Dataset id | `huszar_2022` |
| Animal and region | mouse (11 animals with units here), hippocampus CA1 |
| Apparatus | 79 x 79 cm figure-eight maze raised 61 cm above the ground, between rest in the home cage; some sessions add a second maze |
| Archive | DANDI 000552, version 0.230630.2304 (CC-BY-4.0); NWB files of 41 MB to 2.5 GB |
| Sessions | 65, those with units and position. One is checked by hand: the default `sub-e14-2m2_ses-e14-2m2-201019` (113 units). The other 39 `behavior+ecephys` files hold ripples and sleep states only, and are not offered |
| LFP | 12 sessions, in a sibling `-raw_ecephys` file of 5.6 to 184 GB: read one channel by byte range |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'huszar_2022'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]].head()
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import load

rec = load("huszar_2022")
rates = compute_spatial_rates(
    rec.env,                    # the figure eight, both loops closed
    rec.spike_times,
    rec.times,
    rec.positions,              # camera (x, y), cm
    epochs=rec.params["analysis_epochs"],
    min_speed=rec.params["min_speed"],
)
ripples = rec.events["ripples"]      # (n, 2) [start, stop) in s, the file's table
peaks = rec.events["ripple_peaks"]
nrem = rec.epochs["nrem"]            # the file's sleep scoring
linear = rec.linearized              # the file's 0 to 222 cm loop; see Watch out
```

## What was recorded

Huszár et al. (2022, Nature Neuroscience; read on PubMed Central, PMC10807234)
birthdated CA1 pyramidal cells in utero and recorded them with silicon probes in
freely moving adult mice. Nine mice ran "a spatial alternation task in a
figure-eight maze", "a customized 79 x 79 cm2 figure-eight maze ... raised 61 cm
above the ground". They were "shaped to visit alternate arms between trials to
receive a water reward in the first corner reached after making a correct
left/right turn", with "a 5-s delay in the start area" between trials.

Yang et al. (2024, Science; Supplementary Materials read) add: "animals spent
~120-min in the homecage before running on the maze and another ~120-min in the
homecage afterward", and "head-mounted red LEDs" were tracked "at a frame rate of
30-Hz". Yang et al. kept "sessions with more than 150 cells and >20 trials".

Some files hold a second recording, on another maze: the `behavior` module's
description names them (`alternation, linearMaze`, `alternation, alternation`).

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| positions: `SubjectPosition` as stored, cm | stated | The series' unit is `cm`, conversion 1, reference frame "Arbitrary, camera", so the axes' handedness is not stated. |
| `rec.linearized`: `LinearizedPosition` as stored, cm; 0 and 222 cm are one place | inferred | The file says only "Linearization of the (x,y) coordinates". On the default session the samples within 5 cm of either end lie a median 3.0 cm apart on the camera. |
| environment: the figure eight's six nodes fitted to the samples of the first alternation recording, seven edges, both loops closed, 2 cm bins, `track_width` a placement tolerance | inferred | No source gives the nodes. On the default session the sides fit at about 72 cm against the stated 79 x 79 cm, and the samples span 83 x 83 cm; over the 65 sessions the long edges are 67 to 78 cm. |
| epochs: `maze`, the rows of the file's `epochs` table holding samples, or the tracked span; `alternation` and `linear_maze` by the recordings the file names; `trials`; `awake`, `nrem`, `rem` from `SleepStates` | stated | The file's tables and descriptions. |
| epochs: `off_maze`, the rest of the recording | inferred | Position was tracked only on a maze, and the paper says the animal was in the home cage before and after. |
| trials: the file's table; rows without a time left out and counted | stated | The file's table. Eight sessions have rows with NaN times. |
| events: `ripples`, `ripple_peaks`, `rewards`, `left_rewards`, `right_rewards` | stated | The file's `Ripples` and `RewardEventsEightMazeTrack` tables. Huszár et al. state the detection: `bz_FindRipples` on manually selected channels, 130 to 200 Hz, 5 SD peak, 2 SD edges, 20 to 200 ms. On the default session the table's peak power starts at 5.00 and its durations run 20 to 150 ms, as stated. |
| `units`: `brain_region` CA1; `label` missing; `unit_name`, `clu_id`, `group_id` kept | stated | Both papers record CA1; the file's electrodes say "unknown", and its units table has no cell-type column. |
| `rec.lfp`: samples times the series' conversion, in µV; times from its start and 1,250 Hz | stated | The raw file's `ElectricalSeriesLFP`: conversion 1.95e-07 V, 1,250 Hz. |

The sampling rate (30 Hz) is **stated**. The position span is **inferred**: the
maze is stated, and the 116 cm of the longest second maze comes from the samples,
so a session outside 63 to 139 cm loads with a watch-out.

## Watch out

- **The linear position is a loop.** `rec.linearized` runs along the centre bar,
  around a loop and back, and both loops share its values, so the top and bottom
  arms read the same. Its 0 and 222 cm are one place, and it jumps by 216 cm 62
  times on the default session while the camera moves under a centimetre. A line
  would put neighbours 222 cm apart, so `rec.env` is the 2-D figure eight. Its
  distances run through the junctions.
- **No cell-type labels.** The file does not say which units are pyramidal cells.
  Huszár et al. classified 7,411 putative pyramidal cells and 2,060 putative
  interneurons from waveforms and autocorrelograms, and the waveforms are not in
  these files. In one session 33 units firing above 5 Hz carry 65% of the spikes.
- **Yang et al.'s sessions.** The default session has 113 units, so Yang et al.'s
  analyses did not use it. The summary says so for every session outside their
  rule.
- **Second recordings.** `rec.env` is the figure eight of the first alternation
  recording. A later maze can stand elsewhere or be another shape, and 12% to 54%
  of all samples then lie outside `rec.env`. `params["analysis_epochs"]` holds the
  first recording only.
- **LFP.** Twelve sessions have one, in a raw file of up to 184 GB, so read a
  channel with `rec.lfp.fetch(["ch1"], stream=True)`. No source names the ripple
  channels ("manually selected channels"), so `rec.lfp.ripple_channels` is empty and
  `rec.lfp.fetch()` with no channels raises: name the channels.
  `sub-e16-3m1_ses-e16-3m1-210201` also holds a copy in its own file, whose
  metadata state 30 kHz for a 1,250 Hz series, so the loader reads the raw file's.
  `sub-e16-3m1_ses-e16-3m1-210203`'s raw file holds only the raw voltage.
- **Tracking.** On the default session 13 runs of 11 or more samples repeat, 25 s
  in all, and 1% of samples lie outside the track by construction: its width is
  set from those samples, and a head-mounted LED swings past an elevated track.
- **`is_familiar_maze`.** It changes within one recording on one maze (True for
  the first 2 of the default session's 21 trials), so the loader does not use it.

## Analysis choices

`rec.params`: `bin_size` 2 cm, the loader's choice (Huszár et al. use 1 cm bins
and a 9 cm FWHM Gaussian); `min_speed` 4 cm/s, the loader's choice, which leaves
out stops at the reward corners (Huszár et al. use 1.5 cm/s after a Kalman filter,
Yang et al. "2-5-cm/s"); `analysis_epochs`, the first alternation recording;
`train_epochs` and `test_epochs`, its trials alternating within each visited arm,
since consecutive trials alternate between the arms; `dt` 0.25 s and `replay_dt`
0.02 s.

## Candidate events

Yang et al. defined ripple events as population bursts holding an LFP ripple. The
combined spiking of "all recorded CA1 pyramidal cells" in 1 ms bins, convolved
with "a 15-ms Gaussian kernel", triggers at "3 standard deviations above the mean
of all 1-ms bins within NREM epochs", and each event runs until the rate returns
to that mean. An event must contain a ripple, last 50 to 500 ms, fall in "quite
awake immobility or NREM", and have "at least five distinct pyramidal cells".

`ripple_detection.literature_methods.yang_2024` implements this. Its inputs here
come from the file:

- the ripples are the file's table. **The paper does not state how its LFP
  ripples were detected, and this table may have been made differently.** It says
  only "ripple events were detected from the pyramidal layer LFP"; Huszár et al.'s
  table follows the stated `bz_FindRipples` rule.
- the NREM rows are the file's sleep scoring, which the paper's state scoring
  describes.
- **the pyramidal cells are not in the file**, so the population is every sorted
  unit. No firing-rate cut stands in for the classification.
- the awake candidates are those on the maze while the file says Awake. The paper
  states no speed for "immobility".

`ripple_detection` is not in the `datasets` extra. Install it with
`pip install 'ripple_detection @ git+https://github.com/Eden-Kramer-Lab/ripple_detection@0eddae64a5ebf7b810fb1ec693643b1673650142'`.

<!-- docs-test: skip needs a download and ripple_detection -->
```python
import numpy as np
from ripple_detection.literature_methods import Recording, run_method

from neurospatial.datasets import load

rec = load("huszar_2022", "sub-e14-2m3_ses-e14-2m3-201017")   # 34 units, 5.4 GB of counts
last = max(train[-1] for train in rec.spike_times)
grid = np.arange(int(np.ceil(last * 1000)) + 2) / 1000
# One row per ms: a count belongs to the millisecond it falls in.
multiunit = np.column_stack(
    [
        np.bincount(np.searchsorted(grid, train, side="right") - 1, minlength=grid.size)
        for train in rec.spike_times
    ]
)
maze, awake = rec.epochs["maze"], rec.epochs["awake"]
starts = np.maximum(maze[:, 0, None], awake[:, 0])
stops = np.minimum(maze[:, 1, None], awake[:, 1])
eligible = np.column_stack([starts[stops > starts], stops[stops > starts]])
recording = Recording.from_arrays(
    grid,
    1000.0,
    multiunit=multiunit,
    pyramidal=np.ones(rec.n_units, dtype=bool),          # no labels: every unit
    sleep_intervals=rec.epochs["nrem"],
    external_ripples=np.column_stack(
        [rec.events["ripples"], rec.events["ripple_peaks"]]
    ),
)
events = run_method("yang_2024", recording, behavior_intervals=eligible)
```

`ripple_detection` keeps one float64 per sample per unit and copies the selection,
which is 190 GB for 376 units over 7 hours at 1 kHz, twice that during the copy. For a large session pool the
counts for the rate and count active units from the spike times, as
`tests/datasets/test_huszar_2022.py` does. That test shows the two give the same
events, within rounding, on the session above.

## Agreement with the authors' events

The event file `ripple_detection`'s notes name,
`e13_26m1_210913.rippleHSE.events.mat`, is a pickle-free MATLAB file in the paper's
GitHub repository (it is not in the Zenodo archive). **Its session is not in this
dandiset**, whose four sessions of that mouse are 211019, 211116, 211119 and
211124, so it cannot be compared with these files. The repository does hold that
session's spike times, which test the list itself (below).

The same repository holds Yang et al.'s analysis file for `e15_13f1_220118`, a
session of this dandiset: 20 ms bins with an event id on each ripple bin, in a
pickle-free MATLAB 7.3 file. Its matrix has 363 columns, the 363 units of the NWB
file in the same order (per-unit spike counts correlate above 0.999), so it is on
this clock. Event by event on the whole session:

| | |
| --- | --- |
| Their events | 4,088 |
| Our events | 7,536 |
| Their events overlapped by ours | **94.3%** |
| Our events overlapped by theirs | **50.1%** |

Both causes are named, and nothing was tuned:

- **Why half of ours are not in their file.** Their events end sooner, and the file
  keeps the longer ones. Their code, `find_HSE`, ends an event where the z-score of the
  rate over all bins falls below the mean raw count per ms, 0.75 here. That is a units
  slip: it is not the mean. The code also notes a minimum of about 100 ms for decoding.
  Their matched events last 100 ms against our 178 ms. With our trigger, an end at that
  same rate and 80 ms or longer, we have 4,353 events, and the two shares are 96.0% and
  89.7%.
- **Why 5.7% of theirs are missed.** Of the 232, 145 peak below our trigger of 3 SD,
  and the other 87 lie in a rise above the NREM mean longer than the paper's 500 ms,
  which the tighter bound splits: with it 4.0% are missed. Normalizing over all bins,
  as `find_HSE` does, is worse (88.3%).

**Their list was made from every unit.** On `e13_26m1_210913`, whose spike times and
cell types are in the paper's repository, `find_HSE`'s rule on all 482 units finds
86.8% of their 4,266 stored events to 1.5 ms at both ends, and on the 422 pyramidal
cells 0.4%. So pooling all sorted units here matches what was run, though the Methods
say pyramidal cells.

## Worked result

**Status: a sanity property of our own analysis, level B.** This is neurospatial's
method on Yang et al.'s candidates, **not the paper's analysis**. Yang et al.
judge replay by distance to a UMAP manifold and trajectory length against
shuffles, which states no cutoff or shuffle count and is not a preset in
`neurospatial.replay`.

**What was tested, fixed in writing before any event was decoded:** awake ripple
candidates are judged trajectories more often than the same windows after
circularly shifting the spikes.

- The session, by rule: the most units among those meeting Yang et al.'s inclusion
  with one figure-eight recording, `sub-e15-13f1_ses-e15-13f1-220117` (376 units,
  35 trials, 20 candidates).
- Fields from the 325 units with `has_place_field`, over the first recording at
  `min_speed`, with `compute_spatial_rates` on `rec.env`.
- Decoded in 20 ms bins stepped by 5 ms to each event's end; judged by
  `PFEIFFER_FOSTER_2013` unchanged: a MAP step under 20 cm over at least 11 bins and at least 40 cm from
  start to end, geodesic on the closed loops.
- Two nulls, 20 draws each (seed 2022): every unit shifted by its own offset, and
  all units by one common offset, circularly over the recording, by at least 10 s.

**Result: met.** 5 of the 20 candidates are judged trajectories (25%). Every one of
the 40 shifted draws judges none. Looking at the decoded paths, the 5 run smoothly
along one arm, most from the corner the mouse sits at.

As a check that these maps can show a trajectory, also fixed beforehand, held-out
laps decode to a median 2.0 cm from the mouse, against 69.6 cm with positions
shuffled.

Read this as evidence that neurospatial finds sequences in these candidates, not
as a rate of replay. Twenty candidates in one session is a small count, and since every
null draw is 0, the criterion would pass on 1 of the 20. Yang et al.'s "about 33% of
the awake SPW-Rs were significant replays" measures something else, and is not compared.

**This judge misses most planted replay, so a low count would not be evidence of
absence.** We planted replay into the 20 windows. Every spike kept its time, and its
cell was drawn from these rate maps along a real running path played 15 times faster.
`PFEIFFER_FOSTER_2013` judged 26 of 300, 8.7%, against 0.7% for events with no content
and 0.3% for events that stay in one place. Its 20 and 40 cm are a rat's in a 2 x 2 m
arena, used unchanged on this 79 cm maze. Planted events come from the model the decoder
assumes, and compression 15 is not calibrated for mice: at that speed a mouse's path
often falls short of 40 cm. This is a limit of the number, not a measurement of the mice.

## Cite

Huszár R, Zhang Y, Blockus H, Buzsáki G (2022). Preconfigured dynamics in the
hippocampus are guided by embryonic birthdate and rate of neurogenesis. Nature
Neuroscience 25, 1201-1212. <https://doi.org/10.1038/s41593-022-01138-x>

Data: Huszár, Roman; Zhang, Yunchang; Blockus, Heike; Buzsáki, György (2023)
Preconfigured dynamics in the hippocampus are guided by embryonic birthdate and
rate of neurogenesis (Version 0.230630.2304) [Data set]. DANDI archive.
<https://doi.org/10.48324/dandi.000552/0.230630.2304>

Yang W, Sun C, Huszár R, Hainmueller T, Kiselev K, Buzsáki G (2024). Selection of
experience for memory by hippocampal sharp wave ripples. Science 383, 1478-1483.
<https://doi.org/10.1126/science.adk8261>
