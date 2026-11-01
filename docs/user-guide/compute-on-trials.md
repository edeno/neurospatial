# Compute on Trials and Segments

**The question.** A place cell fires on runs toward one end of a track. How
does its firing unfold over time within a run, along the route, and from one
run to the next, and how do you hand those runs to a model? Each answer
starts from the same thing: the recording split into pieces (trials, visits,
laps, running bouts), each with a stable id. This guide computes on those
pieces in five ways:

1. **Restrict** an analysis to some trials (`epochs=`).
2. **Lay trials out by time** since departure or until arrival.
3. **Lay trials out along the route**, one curve per trial.
4. **One row per trial** of spike counts and behavior, joined by id.
5. **Per-sample model inputs**: a trial label, time in the trial, a spatial
   basis, and trial-blocked cross-validation groups.

neurospatial does what needs the environment, the behavior, or the trial ids.
Generic time-series steps (time warping, design matrices, fitting) belong to
[pynapple](https://pynapple.org) and [NeMoS](https://nemos.readthedocs.io);
the recipes below show the hand-off and what each conversion keeps.

## The session

A rat shuttles between a west and an east well on a 100 cm track. It rests
1-2 s at each well, then runs across. Tracking runs at 30 Hz with 40 lost
frames (NaN rows). The cell has a place field at x = 70 cm and fires only on
eastward runs.

```python
import numpy as np
from shapely.geometry import box

from neurospatial import Environment, Regions
from neurospatial.behavior import compute_speed, detect_trials, path_progress
from neurospatial.simulation import generate_poisson_spikes

rng = np.random.default_rng(0)
frame_rate = 30.0  # Hz
pieces = []
for lap in range(16):
    for x0, x1 in [(5.0, 95.0), (95.0, 5.0)]:
        rest = int(rng.uniform(1.0, 2.0) * frame_rate)  # 1-2 s at the well
        pieces.append(np.column_stack([np.full(rest, x0), np.full(rest, 5.0)]))
        run = int(rng.uniform(1.8, 2.6) * frame_rate)  # one run across
        x = x0 + (x1 - x0) * (1 - np.cos(np.linspace(0, np.pi, run))) / 2
        pieces.append(np.column_stack([x, 5.0 + rng.normal(0, 0.5, run)]))
positions = np.concatenate(pieces)  # (n_samples, 2), cm
times = np.arange(len(positions)) / frame_rate  # s
positions[rng.choice(len(positions), 40, replace=False)] = np.nan  # lost frames

wells = Regions()
wells.add("west", point=(5.0, 5.0), radius=8.0)
wells.add("east", point=(95.0, 5.0), radius=8.0)
track = Environment.from_polygon(box(0, 0, 100, 10), bin_size=2.5, units="cm")
track = track.with_regions(wells)

speed = compute_speed(times, positions)  # cm/s
heading_east = np.gradient(np.nan_to_num(positions[:, 0]), times) > 0
in_field = np.exp(-((positions[:, 0] - 70.0) ** 2) / (2 * 8.0**2))
rate = 0.5 + 20.0 * np.nan_to_num(in_field) * heading_east  # Hz
spike_times = [generate_poisson_spikes(rate, times, seed=1)]

trials = detect_trials(track, times, positions, regions=["west", "east"])
east = trials[trials.end_region == "east"]
print(len(trials), "trials,", len(east), "eastward")
print(trials.intervals_table().head(3))
```

```text
32 trials, 16 eastward
             start_time  stop_time start_region end_region  outcome
interval_id
0              2.000000   3.266667         west       east  reached
1              5.166667   6.500000         east       west  reached
2              8.600000   9.900000         west       east  reached
```

A trial runs from departure (leaving a well) to arrival (entering the next).
Your own trials work the same way: a task log through
`Intervals.from_dataframe`, an NWB trials table through `read_intervals`, or
a pynapple `IntervalSet` through `from_pynapple`.

## 1. Restrict to some trials

Every analysis that takes `epochs=` accepts an `Intervals`, a single trial,
or any selection of one. The cell fires on eastward runs and barely on
westward ones:

```python
from neurospatial.encoding import compute_tuning_curve

for direction in ["east", "west"]:
    curve = compute_tuning_curve(
        spike_times[0], times, positions[:, 0], bin_size=5.0,
        value_range=(0.0, 100.0), variable="x", units="cm",
        epochs=trials[trials.end_region == direction],
    )
    peak = np.nanargmax(curve.firing_rate)
    print(f"{direction}: peak {curve.firing_rate[peak]:.1f} Hz at x = {curve.bin_centers[peak]} cm")
```

```text
east: peak 17.2 Hz at x = 72.5 cm
west: peak 1.5 Hz at x = 67.5 cm
```

## 2. Trials laid out by time

Pass the trials themselves as events, and choose `align="start"` (departure)
or `align="stop"` (arrival). Each trial's window ends at its other bound:
aligned at departure, nothing after the trial's arrival is used, so a short
trial never borrows the next trial's data. The side of the alignment point is
used as for any event, so the second before departure (the animal at the
well) stays in.

```python
from neurospatial.events import align_continuous_to_events, peri_event_histogram

behavior = align_continuous_to_events(
    times, speed, east, (-1.0, 2.0), dt=0.1,
    feature_ids=["speed"], feature_units={"speed": "cm/s"},
)
psth = peri_event_histogram(spike_times[0], east, (-1.0, 2.0), bin_size=0.1)
print("time from departure (s):", behavior.relative_times[::5])
print("trials under way:       ", behavior.counts[::5, 0])
print("speed (cm/s):           ", np.round(behavior.means[::5, 0], 1))
print("firing (Hz):            ", np.round(psth.firing_rate[::5], 1))
print(behavior.events_table().head(3))
```

```text
time from departure (s): [-1.  -0.5  0.   0.5  1.   1.5]
trials under way:        [16 16 16 16 16  4]
speed (cm/s):            [ 0.   1.3 37.8 64.7 56.  32.7]
firing (Hz):             [0.  0.  1.9 0.  9.4 nan]
   event_index  event_id    status  overlapping
0            0         0  retained        False
1            1         2  retained        False
2            2         4  retained        False
```

Read the counts with the means. Runs last about 1.3 s, so 1.5 s after
departure only 4 of 16 trials are still under way, and the speed there is
their mean. The PSTH counts a bin for a trial only when the trial lasts
through all of it (`psth.events_per_bin`); no trial lasts through 1.5-1.6 s,
so that bin is NaN, not zero. Late points therefore describe a different
set of trials, the slowest runs, so a rise or drop there can come from which
trials remain rather than from time in the run. Check the counts before
reading the end of the curve; to compare the same trials at every point, use
progress along the route (section 3). The rows of `behavior.values` (trial × time ×
feature) are labelled by `event_ids`, the trials' `interval_id`. Values are sampled by
time, so the lost frames shift nothing; a gap longer than `max_gap` is NaN.

To keep only trials whose whole window lies inside the trial instead,
so every bin averages the same trials, align plain times and restrict: `peri_event_histogram(spikes, east.start,
(0, 1), epochs=east)`.

**The same with pynapple.** `build_tensor` makes a trial × sample array.
It aligns by sample index (so it assumes regular sampling without dropped
frames), includes the sample at each trial's stop (pynapple intervals are
closed), pads with NaN, and returns a plain array without trial ids. Trials
without duration are dropped, so recover the row ids from the trials that
have one:

<!-- docs-test: skip needs the optional pynapple extra -->
```python
import pynapple as nap
from neurospatial.io import to_pynapple

tensor = nap.build_tensor(nap.Tsd(t=times, d=speed), to_pynapple(east), align="start")
row_ids = east[east.durations > 0].interval_ids  # the trial of each row
print(tensor.shape, row_ids[:4])
```

```text
(16, 46) [0 2 4 6]
```

## 3. Trials along the route

Progress from the start well to the goal well (0 to 1) puts every run on one
axis whatever its speed. One tuning curve per trial, on shared bin edges,
gives a trial × progress matrix for a stability plot. Select each trial by
its id:

```python
progress = path_progress(track, times, positions, east, start="west", goal="east")
edges = np.linspace(0.0, 1.0, 11)
per_trial = np.array([
    compute_tuning_curve(
        spike_times[0], times, progress, bin_edges=edges, variable="progress",
        epochs=east.sel(interval_id=trial_id),
    ).firing_rate
    for trial_id in east.interval_ids
])  # (n_trials, n_bins), Hz; NaN where a trial spent no time
print(per_trial.shape)
print("mean over trials (Hz):", np.round(np.nanmean(per_trial, axis=0), 1))
```

```text
(16, 10)
mean over trials (Hz): [ 1.   0.5  0.5  0.5  0.6  2.7 12.8 15.7 13.7  4.7]
```

Each row is noisy: one pass through a bin lasts a fraction of a second. The
plain mean over trials weights every trial equally; weighting each trial by
its time in the bin gives the pooled curve, `compute_tuning_curve(...,
epochs=east)`, exactly.

## 4. One row per trial

`Intervals.spike_counts` counts each unit's spikes in each trial, indexed by
`interval_id`, so it joins the trial table. Behavioral summaries come from
grouping samples by their trial:

```python
import pandas as pd

counts = trials.spike_counts(spike_times)  # (n_trials, n_units)
table = trials.intervals_table().join(
    counts.div(trials.durations, axis=0).add_prefix("rate_")  # Hz
)
sample_trial = trials.sample_interval_ids(times)  # -1 between trials
table["peak_speed"] = pd.Series(speed).groupby(sample_trial).max()
print(table.groupby("end_region")[["rate_0", "peak_speed"]].mean().round(1))
```

```text
            rate_0  peak_speed
end_region
east           5.1        66.4
west           0.5        66.4
```

Both are aligned by `interval_id`, not by row position: the `-1` group of
samples between trials matches no trial, and a trial without samples gets NaN
rather than a shifted row.

## 5. Model inputs and trial-blocked cross-validation

For a model, every input is one value per sample. `sample_values` gives each
sample its trial's field (None between trials), and from the trial bounds the
time since departure and the fraction of the trial elapsed. `basis_at_positions`
evaluates a spatial basis at each position, with NaN rows outside the track:

```python
from sklearn.model_selection import GroupKFold
from neurospatial.ops import basis_at_positions, spatial_basis

direction = trials.sample_values(times, trials.end_region)  # "east"/"west"/None
elapsed = times - trials.sample_values(times, trials.start)  # s since departure
fraction = elapsed / trials.sample_values(times, trials.durations)  # 0 to 1
X_space = basis_at_positions(track, spatial_basis(track, n_features=30, rng=0), positions)

rows = (sample_trial >= 0) & ~np.isnan(X_space).any(axis=1)  # in a trial, tracked
folds = GroupKFold(n_splits=4).split(X_space[rows], groups=sample_trial[rows])
for train, test in folds:
    assert not set(sample_trial[rows][train]) & set(sample_trial[rows][test])
print(X_space.shape, int(rows.sum()), "samples in trials")
```

```text
(3450, 30) 1261 samples in trials
```

Samples within a trial are strongly correlated, so a random split of samples
puts parts of the same trial in training and test sets and overstates how well
a model generalizes. Grouping by `sample_interval_ids` keeps each trial whole.

**The same model in NeMoS.** NeMoS builds bases over one variable (progress,
speed, time, spike history) and fits the GLM; neurospatial supplies the
spatial basis, which respects the track's walls, and the trial-level inputs:

<!-- docs-test: skip needs nemos, which neurospatial does not depend on -->
```python
import nemos as nmo

progress_all = path_progress(
    track, times, positions, trials, start=trials.start_region, goal=trials.end_region
)
X_progress = nmo.basis.BSplineEval(n_basis_funcs=6, bounds=(0, 1)).compute_features(progress_all)
X = np.column_stack([X_space, X_progress, direction == "east"])
spike_counts = np.histogram(spike_times[0], bins=np.r_[times, times[-1] + 1 / frame_rate])[0]
fit_rows = rows & ~np.isnan(X).any(axis=1)
model = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.1)
model.fit(X[fit_rows], spike_counts[fit_rows])
```

## What lives where

| Step | Here | Elsewhere |
| --- | --- | --- |
| Trials, visits, laps with ids and fields | `detect_trials`, `Intervals` | pynapple `IntervalSet` (converts both ways) |
| Restrict any analysis | `epochs=` | pynapple `restrict` |
| Trials × time with exact timing and ids | `align="start"` / `"stop"` | pynapple `build_tensor` (by sample index, no ids) |
| Trials along the route | `path_progress` + `compute_tuning_curve` | — |
| One row per trial | `spike_counts`, `sample_interval_ids` | pandas `groupby` |
| Spatial basis at positions | `basis_at_positions` | — |
| Bases over one variable, design matrices, fitting | — | NeMoS |
| Time warping | — | pynapple `warp_tensor` |
