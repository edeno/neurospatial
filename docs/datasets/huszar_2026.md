# Huszár et al. (2026): mouse CA1 and CA3 on a cheeseboard maze

**The question.** How do hippocampal place cells and ripples change while a mouse
learns where two hidden rewards are? Each day the mice learned two new reward
wells on a 90 cm circular cheeseboard maze, between probe sessions and rest in
the home cage, with CA1 and CA3 recorded together. This dataset has learning
trials, reward times and locations, ripples, sleep states and opto-tagged units,
so it suits place fields around goals, reward-related activity, ripple-triggered
analyses and sleep reactivation.

| | |
| --- | --- |
| Dataset id | `huszar_2026` |
| Animal and region | mouse (8 animals), hippocampus CA1 and CA3 |
| Apparatus | 90 cm diameter circular cheeseboard maze with 177 wells and a holding box, between rest sessions in the home cage |
| Archive | DANDI 001880, version 0.261007.1849 (CC0-1.0); NWB files of 66 to 230 MB |
| Sessions | 42, one per recording day. One is checked by hand: the default `sub-iuvE14-cagCre-ef1aDIOchr2-3_ses-iuvE14-cagCre-ef1aDIOchr2-3-221208` (49 units, the smallest file) |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'huszar_2026'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]].head()
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import load

rec = load("huszar_2026")
# The learning trials' maze visits: on_maze rows inside the Learning
# sub-session (no row crosses its edges in any session), so the held
# positions while the mouse waits in the holding box are left out.
learning, on_maze = rec.epochs["learning"], rec.epochs["on_maze"]
inside = (on_maze[:, 0] >= learning[0, 0]) & (on_maze[:, 1] <= learning[-1, 1])
rates = compute_spatial_rates(
    rec.env,
    rec.spike_times,
    rec.times,
    rec.positions,
    epochs=on_maze[inside],
    min_speed=rec.params["min_speed"],
)
wells = rec.env.regions                          # reward_1, reward_2
ripples = rec.events["ripples"]                  # (n, 2) [start, stop) in s
```

## What was recorded

The paper (Huszár, Kirsanov, Henze, Huilgol, Valero, Huang & Buzsáki 2026,
Neuron; the author manuscript on PubMed Central, PMC13573635) records CA1 and
CA3 together while mice whose pyramidal cells of one embryonic birthdate express
channelrhodopsin learn a place-reward association. Its STAR Methods describe:

- the maze: "a flat circular platform (90cm diameter) made of white acrylic
  with 177 equally spaced wells", reached from a holding box through a
  drawbridge;
- the day: Probe 1 (~30 mins), Rest 1 (~1h, home cage), Learning (~1h),
  Rest 2 (~1h), Probe 2 (~30 mins). The files name the three maze sub-sessions
  `PRE`, `Learning` and `POST`;
- a trial: the mouse finds two hidden water rewards; between trials it waits in
  the holding box while the maze is "wiped ... and rotated 45° clockwise", and
  the same two wells "with respect to room cues" are baited again;
- tracking: "a red LED on the head of the animal with an overhead camera ...
  recording at a frame rate of 30 Hz"; "The overhead camera's field of view
  covered the platform but excluded the holding box itself."

Each file holds the units (with their region, putative cell type and opto-tag),
the position, the learning trials with their reward times, the two reward
coordinates, the maze and box intervals, the sub-sessions, sleep states and the
ripples.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| positions: `PositionSeries` as stored, cm | stated | The series' unit is `cm`. The file's `SubjectPosition` also holds a `VelocitySeries`, which is not loaded. The camera frame's handedness (whether y grows up or down the room) is not stated, so the positions may be the room's mirror image, in which case the maze's clockwise turn between trials appears counter-clockwise. |
| epochs: `on_maze`, `open_box`, `pre`, `learning`, `post`, `awake`, `nrem`, `rem` from the file's `onMaze`, `openBox`, `subSessions` and `SleepStates` tables | stated | The file's interval tables and their descriptions. |
| epochs: `rest` (between sub-sessions), `before_tracking`, `after_tracking` (from the last position sample to the latest stop of the sleep states, sub-sessions and light pulses) and `opto_tagging` (first to last light pulse) | inferred | The paper's day plan and its "in the homecage following Probe 2" opto-tagging; the file's sleep scoring covers the whole recording. |
| trials: the `LearningTrials` table; `events["rewards"]`, every reward time found | stated | The file's table; the paper: "The experimenter manually recorded water consumption time points using a footswitch". |
| events: `ripples` ([start, stop) rows) and `ripple_peaks` from the `Ripples` table | stated | The file's table, detected as the paper describes (130-200 Hz band, 5 SD peak, 2 SD edges, 20-200 ms). |
| regions `reward_1`, `reward_2` from `rewardCoordinates`; a trial's `rewardOrder` k names `reward_k` | inferred | The file says only "Order of rewards"; on the default session the tracked position at each of the 90 reward times lies a median 2.3 cm from the well its `rewardOrder` names and 43.5 cm from the other. |
| environment: the stated 90 cm disc, centred midway between the 0.5th and 99.5th percentiles of the samples tracked during `on_maze`, 2.5 cm bins | inferred placement | The file states no origin (reference frame "Arbitrary, camera"). The percentiles keep a few far samples from moving the centre; on the default session it lies 4.3 cm from the samples' min-max midrange (0.1 to 6.9 cm over the 42 sessions). The on-maze samples span 90.7 x 89.6 cm; 1.5% of them lie beyond the disc's radius. |
| `units`: `brain_region` from `location`, `label` from `cell_type`; `CA1_optoTag`, `CA3_optoTag` and the file's other columns kept | stated | The file's units table. |

The sampling rate (30 Hz) and the 90 cm diameter are **stated** by the paper; a
session whose largest tracked span is outside 72 to 108 cm is refused.

## Watch out

- **Off-maze positions are not the animal's.** The camera does not see the
  holding box. On the default session 95% of the tracked samples outside
  `epochs["on_maze"]` repeat the sample before (6% on the maze), a median
  41 cm from the platform's centre: most likely the tracker holding its last
  fix at the edge while the mouse is in the box. `params["analysis_epochs"]`
  is `on_maze`; use it for occupancy and rate maps. The positions are kept as
  stored. The summary gives the share for every session where more than half
  of the off-maze samples are held (89.5 to 99.9% over the 42 sessions).
- **Held tracking.** 29% of the default session's frames repeat in runs of 11
  or more (the longest 82 s), most of them off the maze. On the maze, check a
  stop at a well before reading it as lost tracking.
- **Samples outside the disc.** 2.6% of the default session's tracked samples
  lie outside the declared disc, 3.5% of the stopped ones and 1.0% of the
  moving ones: held fixes at the edge and the head over the platform's rim.
- **Trial stops are not the second reward.** The paper defines a trial's end as
  "the time point of the second water consumption", but the file's trials stop
  when the mouse leaves the maze, 15 to 52 s after the second reward on the
  default session (in every session, every trial with both rewards stops more
  than a frame after the second; the summary says so where one does). Some
  sessions have trials in which only one reward was found (`rewardTime`
  missing, `rewardOrder` 0); those are counted in the summary.
- **The maze turns between trials, the frame does not.** Positions, the
  environment and the reward wells are in the camera's (room) frame; the maze
  under them was turned 45 degrees after every trial.
- **Fewer units than the paper's average.** The default (smallest) file holds 49
  units, 26 CA1 and 23 CA3; the paper's mean is 123.5 CA1 and 88.3 CA3 per
  session. Other sessions hold up to 480.
- **Zero-length sleep-state rows.** Two sessions
  (`...-6-230328`, `...-6-230330`) have `SleepStates` rows that start and stop
  at the same second; they are left out of the epochs, which must start before
  they stop, and counted in the summary.

## Analysis choices

`rec.params`: `bin_size` 2.5 cm, the loader's choice (the paper's rate maps use
1 cm bins and a 4 cm FWHM Gaussian, about 6,400 bins in the disc); `min_speed`
5 cm/s, the paper's running threshold; `analysis_epochs` the `on_maze` rows;
`dt` 0.25 s, and `train_epochs` / `test_epochs` the even and odd `on_maze`
rows.

## Worked result

**Status: not possible yet.**

The paper's text gives numbers pooled over its 42 sessions (for example
"34.8 ± 2.5 trials (mean ± SEM)" per learning session); its per-session results
(memory retrieval, subspace expression) are plotted points without their values
or rest on the authors' contrastive-PCA pipeline. One session's number cannot be
checked against them.

## Cite

Huszár R, Kirsanov A, Henze G, Huilgol D, Valero M, Huang ZJ, Buzsáki G (2026).
Developmentally defined subcircuits in the hippocampus support spatial learning.
Neuron. <https://doi.org/10.1016/j.neuron.2026.07.023>

Data: Huszar, Roman (2026) CA1 and CA3 electrophysiological dataset in mice
performing a spatial memory task (Version 0.261007.1849) [Data set]. DANDI
Archive. <https://doi.org/10.48324/dandi.001880/0.261007.1849>
