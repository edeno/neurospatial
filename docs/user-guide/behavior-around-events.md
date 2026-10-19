# Behavior Around Events

**The question.** What do running speed, heading and distance to a wall do
around reward? A cell whose firing drops at reward may respond to the reward,
or it may respond to the animal stopping, turning or reaching a wall that
always comes with the reward. Before you interpret a peri-event response, look
at the behavior on the same time axis. This guide builds the behavioral
counterpart of a PSTH: continuous signals sampled at fixed times around each
event and averaged across events.

Two functions do the work:

- `neurospatial.events.align_continuous_to_events` samples per-frame signals
  (speed, heading, a zone code) at fixed times around each event, only where
  they were observed, and summarizes them across events.
- `neurospatial.ops.sample_spatial_features` reads a map with one value per
  bin (floor texture, a reward zone, the height of a platform) at the animal's
  positions, so it can be aligned like any other per-frame signal.

## The session

A rat shuttles across a 100 × 100 cm box between the west wall and a reward
port on the east wall, one 8 s lap at a time. It arrives at the port
(x = 90 cm, 10 cm from the east wall) 4 s into each lap and gets a reward
there; on each lap its path bends a little north or south. The east half of
the floor is rough, the west half smooth. Tracking runs at 30 Hz, and the
tracker lost the animal for 0.8 s around one reward.

Some choices in the code:

- **Lost frames are NaN rows.** Every function here treats a NaN row as a
  missing sample: it gets NaN, and nothing is computed or interpolated across
  it.
- **Speed** comes from `compute_speed`, which smooths the velocity with a
  0.1 s Gaussian so tracking jitter does not read as running.
- **Heading** comes from `heading_from_velocity`: the direction of movement in
  radians, 0 = East, +π/2 = North. Below `min_speed` the animal is not going
  anywhere, so the direction of movement is not observed: the function
  returns NaN for those frames, and they are already marked unobserved.
- **Wall distance** comes from `nearest_boundary`: the distance from each
  position to the closest wall of the box, in cm.
- **Floor texture** is a map: code 0 (smooth) or 1 (rough) for every bin.
  `sample_spatial_features` gives each position the value of the bin that
  contains it. It never smooths or interpolates, so the texture boundary stays
  a step; a position outside the environment, or a bin whose value is NaN,
  gets NaN with `support` False.

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.behavior import compute_speed
from neurospatial.ops import heading_from_velocity, nearest_boundary, sample_spatial_features

rng = np.random.default_rng(0)

# 120 s of tracking at 30 Hz in a 100 x 100 cm box
times = np.arange(0.0, 120.0, 1 / 30)  # s
lap = (times // 8).astype(int)  # one 8 s lap: west wall -> reward port -> west wall
lateral = rng.uniform(-10.0, 10.0, size=lap.max() + 1)  # cm, each lap's sideways bend
x = 50.0 - 40.0 * np.cos(2 * np.pi * times / 8)
y = 50.0 + lateral[lap] * np.sin(np.pi * times / 8) ** 2
positions = np.column_stack([x, y])  # cm
reward_times = np.arange(4.0, 120.0, 8.0)  # s, each arrival at the port (x = 90 cm)

# The tracker lost the animal for 0.8 s around one reward
lost = (times > 35.6) & (times < 36.4)
positions[lost] = np.nan  # lost frames are missing, not zero
print(f"{lost.sum()} frames lost; {times.size} frames; {reward_times.size} rewards")
assert (lost.sum(), times.size, reward_times.size) == (23, 3600, 15)

env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")

# Behavioral features, one value per frame
speed = compute_speed(times, positions)  # cm/s
heading = heading_from_velocity(times, positions, min_speed=2.0)  # rad, 0 = East
wall_distance, _ = nearest_boundary(env, positions)  # cm
texture_map = np.where(env.bin_centers[:, 0] < 50, 0, 1).astype(float)  # 0 smooth, 1 rough
texture = sample_spatial_features(env, texture_map, positions, feature_ids=["texture"])

around = {}
for t in (11.0, 12.0, 13.0):  # 1 s before, at, and 1 s after the second reward
    i = np.argmin(np.abs(times - t))
    print(
        f"t = {t:4.1f} s: speed {speed[i]:4.1f} cm/s, heading {heading[i]:5.2f} rad, "
        f"wall {wall_distance[i]:4.1f} cm, texture code {texture.values[i, 0]:.0f}"
    )
    around[t] = (speed[i], heading[i], wall_distance[i], texture.values[i, 0])

# Rows: speed, heading, wall distance, texture at 11, 12 and 13 s (atol = 0.6 * 10**-decimals)
speed_12, heading_12, wall_12, texture_12 = np.array(list(around.values())).T
assert np.allclose(speed_12, [22.2, 0.0, 22.2], atol=0.06)
assert np.allclose(heading_12, [-0.06, np.nan, 3.08], atol=0.006, equal_nan=True)
assert np.allclose(wall_12, [21.7, 10.0, 21.7], atol=0.06)
assert (texture_12 == 1).all()
```

The tracker lost 23 of the 3600 frames, and there are 15 rewards. Around the
second reward (12 s) the features look as expected. One second before it the
animal runs at 22.2 cm/s with heading −0.06 rad (East), 21.7 cm from the
nearest wall. At the reward its speed is 0.0 cm/s and it is 10.0 cm from the
east wall. One second after, it runs back at 22.2 cm/s with heading 3.08 rad
(West). It is on rough floor (code 1) at all three times.

The heading printed at the reward is `nan`: the animal is standing at the
port, so its direction of movement is not observed there.

## Align the features to reward

`align_continuous_to_events` takes one column per feature, on the clock of
`times`, and samples each column at `event + relative_times` for every event.
Each choice below is an analysis decision:

- **`window=(-3.0, 3.0)`**, in seconds around each reward: long enough to
  include the run in and the run out. Rewards are 8 s apart, so windows of
  consecutive rewards do not share frames; when they do, `events_table()`
  flags them as `overlapping`.
- **`dt=0.05`**: query times every 50 ms. Tracking is 30 Hz (33 ms frames),
  so a finer grid would only add interpolated points. This *samples* each
  signal at the query times (linearly between the two frames around each
  query); it does not bin or average frames.
- **`value_kinds`**: speed and wall distance are ordinary numbers (`"linear"`,
  the default). Texture is `"categorical"`: its codes are labels, not
  quantities, so the result counts each label instead of averaging codes. A
  categorical feature needs `category_labels` and holds the last observed code
  between frames.
- **`window_coverage="complete"`** (the default): an event is kept only if
  every feature was observed without a break across its whole window. Every
  mean then averages the same events at every time point, so a change over
  time cannot come from a change in which events contribute.

Heading is left out here; a later section explains why.

<!-- docs-test: run setup=docs_behavior_around_reward -->
```python
from neurospatial.events import align_continuous_to_events

result = align_continuous_to_events(
    times,
    np.column_stack([speed, wall_distance, texture.values[:, 0]]),
    reward_times,
    (-3.0, 3.0),  # s around each reward
    dt=0.05,  # s between query times
    feature_ids=["speed", "wall_distance", "texture"],
    feature_units={"speed": "cm/s", "wall_distance": "cm"},
    value_kinds={"texture": "categorical"},
    category_labels={"texture": {0: "smooth", 1: "rough"}},
)

events = result.events_table()
events["reward_time"] = reward_times
print(events[events["status"] != "retained"])
print(f"{result.n_events} rewards kept, {result.n_events_dropped} excluded")
assert events.loc[events["reward_time"] == 36.0, "status"].item() == "incomplete_support"
assert (result.n_events, result.n_events_dropped) == (14, 1)
```

`events_table()` has one row per input event, in input order, and `status`
says whether the event was kept (`"retained"`) or why it was not. 14 rewards
are kept and 1 is excluded: the reward at 36.0 s, with status
`"incomplete_support"`. The tracking dropout (35.6 to 36.4 s) lies inside its
window, so none of the features was observed there. With complete
coverage, the other reasons are:

- `"outside_recording"`: the window starts before the first frame or ends
  after the last one. The function never extrapolates.
- `"outside_epochs"`: you passed `epochs` and the window is not inside one of
  them.

Excluding the event keeps every mean on the same events: if the 36 s reward
were kept, the means during its dropout would rest on one event fewer than the
rest of the window. A later section shows how to keep partly observed events
when you need them.

## What the behavior does around reward

`time_course_table()` has one row per relative time and feature, with the
number of events that contribute (`count`), the `mean`, and a `status` that
says whether the mean is defined. `category_counts_table()` gives, for each
category, the number and `proportion` of events in it.
`result.plot(ax=..., feature_id=...)` draws one feature per axes: linear
features as the mean against time, categorical features as one proportion
line per category, with a dashed line at the event. The block repeats the
alignment above so that it runs on its own.

<!-- docs-test: run setup=docs_behavior_around_reward -->
```python
import matplotlib.pyplot as plt

from neurospatial.events import align_continuous_to_events

result = align_continuous_to_events(  # same alignment as above
    times,
    np.column_stack([speed, wall_distance, texture.values[:, 0]]),
    reward_times,
    (-3.0, 3.0),
    dt=0.05,
    feature_ids=["speed", "wall_distance", "texture"],
    feature_units={"speed": "cm/s", "wall_distance": "cm"},
    value_kinds={"texture": "categorical"},
    category_labels={"texture": {0: "smooth", 1: "rough"}},
)

course = result.time_course_table()
table = course.pivot(index="relative_time", columns="feature_id", values="mean")
categories = result.category_counts_table()
rough = categories[categories["category_label"] == "rough"]
table["rough"] = rough.set_index("relative_time")["proportion"]
shown = table.index.round(2).isin([-3.0, -2.1, -1.9, -1.0, -0.5, 0.0, 0.5, 1.0, 1.9, 2.1])
print(table.loc[shown, ["speed", "wall_distance", "rough"]].round(1))
print(f"events per cell: {result.counts.min()} to {result.counts.max()}")
assert result.counts.min() == result.counts.max() == 14

# The table below: speed, wall distance and proportion on rough floor
expected = [
    [22.2, 21.7, 0.0], [31.3, 46.4, 0.0], [31.3, 46.0, 1.0], [22.2, 21.7, 1.0],
    [12.0, 13.0, 1.0], [0.0, 10.0, 1.0], [12.0, 13.0, 1.0], [22.2, 21.7, 1.0],
    [31.3, 46.0, 1.0], [31.3, 46.4, 0.0],
]  # fmt: skip
assert np.allclose(table.loc[shown, ["speed", "wall_distance", "rough"]], expected, atol=0.06)

at_reward = np.argmin(np.abs(result.relative_times))
speed_mean, wall_mean = result.means[at_reward, :2]
assert speed_mean < 1.0  # cm/s: the animal stops at the port
assert abs(wall_mean - 10.0) < 0.1  # cm: the port is 10 cm from the east wall

fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
for ax, feature_id in zip(axes, ["speed", "wall_distance", "texture"]):
    result.plot(ax=ax, feature_id=feature_id)
    ax.set_title(feature_id)
axes[0].set_xlabel("")
axes[1].set_xlabel("")
fig.tight_layout()
```

Every cell averages the same 14 events (`events per cell: 14 to 14`). The
printed table shows, with time in seconds from reward:

| time (s) | speed (cm/s) | wall distance (cm) | proportion on rough floor |
| --- | --- | --- | --- |
| −3.0 | 22.2 | 21.7 | 0.0 |
| −2.1 | 31.3 | 46.4 | 0.0 |
| −1.9 | 31.3 | 46.0 | 1.0 |
| −1.0 | 22.2 | 21.7 | 1.0 |
| −0.5 | 12.0 | 13.0 | 1.0 |
| 0.0 | 0.0 | 10.0 | 1.0 |
| 0.5 | 12.0 | 13.0 | 1.0 |
| 1.0 | 22.2 | 21.7 | 1.0 |
| 1.9 | 31.3 | 46.0 | 1.0 |
| 2.1 | 31.3 | 46.4 | 0.0 |

- **Speed** is 31.3 cm/s about 2 s before reward, in the middle of the box,
  and falls to 0.0 cm/s at the reward: the animal stops at the port. It speeds
  up again in the same way afterwards.
- **Wall distance** is 46.4 cm 2.1 s before reward, mid-run, and reaches its
  minimum, 10.0 cm, at the reward. A cell that responds near walls will be
  active around every reward here.
- **Texture**: the animal crosses the midline, where the two textures meet, 2 s
  before and 2 s after every reward. All events cross at the same moment, so
  the proportion on rough floor steps from 0.0 at −2.1 s to 1.0 at −1.9 s,
  stays 1.0 through the reward, and steps from 1.0 at 1.9 s back to 0.0 at
  2.1 s.

In the figure, speed (top) traces a "V" with its point at 0 s. Wall distance
(middle) has the same shape: the stop and the closest approach to the wall
happen together. In the bottom panel the "rough" line rises from 0 to 1 about
2 s before reward and falls back about 2 s after, and the "smooth" line
mirrors it. Speed, wall distance and floor texture all change together around
reward, so from this session alone a response at reward cannot be told apart
from a response to any of them.

## Heading, and windows that are only partly observed

Heading needs a different coverage rule. The animal stops at every reward, so
heading is unobserved around 0 s in **every** event. With
`window_coverage="complete"`, no event would have heading observed across its
whole window, so every event would be excluded for `"incomplete_support"` and
the function would raise an error saying so.

`window_coverage="partial"` keeps every event with at least one observed
query time. Each cell then summarizes only the events observed there, and
`count` says how many those are. Heading is declared `"circular"`: its values
are angles in radians, and the mean is the direction of the average unit
vector, not the average of the numbers. The slow frames already carry NaN
heading, so the alignment treats them as unobserved for heading only.

<!-- docs-test: run setup=docs_behavior_around_reward -->
```python
import matplotlib.pyplot as plt

from neurospatial.events import align_continuous_to_events

signals = np.column_stack([speed, heading, wall_distance, texture.values[:, 0]])

partial = align_continuous_to_events(
    times,
    signals,
    reward_times,
    (-3.0, 3.0),
    dt=0.05,
    feature_ids=["speed", "heading", "wall_distance", "texture"],
    feature_units={"speed": "cm/s", "heading": "rad", "wall_distance": "cm"},
    value_kinds={"heading": "circular", "texture": "categorical"},
    category_labels={"texture": {0: "smooth", 1: "rough"}},
    window_coverage="partial",
)
print(partial.events_table()["status"].value_counts().to_string())
assert partial.n_events == 15

course = partial.time_course_table()
shown = course["relative_time"].round(2).isin([-1.0, -0.4, -0.35, 0.0, 0.35, 0.4, 1.0])
columns = ["relative_time", "feature_id", "count", "mean", "resultant_length", "status"]
print(
    course.loc[shown & course["feature_id"].isin(["speed", "heading"]), columns]
    .round(3)
    .to_string(index=False)
)

# Counts, heading means and resultant length quoted below (atol = 0.6 * 10**-decimals)
by_time = course.assign(time=course["relative_time"].round(2)).set_index(["feature_id", "time"])
assert by_time.loc[("speed", -0.4), "count"] == by_time.loc[("speed", 0.4), "count"] == 15
assert by_time.loc[("speed", -0.35), "count"] == by_time.loc[("speed", 0.35), "count"] == 14
assert by_time.loc[("heading", 0.0), "count"] == 0
assert by_time.loc[("heading", 0.0), "status"] == "no_support"
assert np.isclose(by_time.loc[("heading", -1.0), "mean"], 0.008, atol=0.0006)
assert np.isclose(by_time.loc[("heading", -1.0), "resultant_length"], 0.996, atol=0.0006)
after_means = [by_time.loc[("heading", t), "mean"] for t in (0.35, 0.4, 1.0)]
assert np.min(after_means) > -3.1396 and np.max(after_means) < -3.1334  # -3.139 to -3.134

# Averaging angles as plain numbers goes wrong across the +-pi wrap
after = np.argmin(np.abs(partial.relative_times - 1.0))
angles = partial.values[:, after, 1]
print(f"headings 1 s after reward: {np.round(np.sort(angles), 2)}")
print(f"circular mean {partial.means[after, 1]:.2f} rad, arithmetic mean {np.mean(angles):.2f} rad")
below, above = angles[angles < 0], angles[angles > 0]
assert np.allclose([below.min(), below.max()], [-3.13, -3.03], atol=0.006)
assert np.allclose([above.min(), above.max()], [3.02, 3.08], atol=0.006)
assert np.isclose(partial.means[after, 1], -3.13, atol=0.006)
assert np.isclose(np.mean(angles), -1.04, atol=0.006)

at_reward = np.argmin(np.abs(partial.relative_times))
assert partial.mean_status[at_reward, 1] == "no_support"
assert abs(abs(partial.means[after, 1]) - np.pi) < 0.1  # West

fig, ax = plt.subplots(figsize=(6, 3))
partial.plot(ax=ax, feature_id="heading")
ax.set_yticks(
    [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    ["−π (W)", "−π/2 (S)", "0 (E)", "π/2 (N)", "π (W)"],
)
```

All 15 rewards are retained now, including the one at 36 s. The printed time
course shows what that costs:

- **Counts dip at the dropout.** Speed has 15 events at −0.40 s and +0.40 s
  but 14 from −0.35 to +0.35 s: the 36 s reward has no tracking there, so the
  mean at those times rests on one event fewer than its neighbors.
- **Heading is undefined at the stop.** At 0 s heading has `count` 0 and
  `status` `"no_support"`: no event was moving fast enough to have a heading.
  The mean is NaN there, not 0 rad, which would mean East.
- **Heading turns around.** One second before reward the circular mean is
  0.008 rad (East); 0.35, 0.40 and 1.00 s after reward it is −3.139 to
  −3.134 rad (West). The resultant length, 0.996, is near 1: the events agree
  closely on the direction.

The headings 1 s after reward show why heading must be declared circular. The
animal runs west in every event, but its small north or south bend puts
heading just below +π in some events (3.02 to 3.08 rad) and just above −π in
others (−3.13 to −3.03 rad). These are nearly the same direction. The circular
mean is −3.13 rad, West. The arithmetic mean of the same numbers is −1.04 rad,
roughly south-southeast, a direction the animal never ran.

In the figure, `plot` draws the circular mean as points on [−π, π] rather
than a line, so nothing jumps across the wrap. Before reward the points sit on
0 (East); after it they sit on −π (West), the same direction as +π. Around
0 s there are no points: heading was not observed while the animal stood at
the port.

Use `window_coverage="partial"` when a feature is undefined in part of every
window by its nature (heading at a stop, a position-based feature when the
animal leaves the environment), or when losing events would cost more than
having different counts at different times. Each mean then uses only observed
samples, so always read `count` beside `mean`, and treat a dip in count as a
change in which events you are averaging.

## Using your own recording

- **One clock, in seconds.** `times`, `event_times` and any `epochs` must be on
  the same clock. If your reward times come from a behavioral controller and
  your positions from a camera, align the clocks first.
- **Joined recordings.** If you concatenate recordings on one clock, mark the
  first frame of each later recording with `break_before`, and pass it to
  `compute_speed`, `heading_from_velocity`,
  `compute_movement_angular_velocity` and `align_continuous_to_events`.
  Nothing is then smoothed, filled, interpolated or averaged across the join,
  and a jump in position between the recordings is not read as running.
  Passing each recording's `[start, stop)` as `epochs` does not do this:
  windows that touch merge into one.
- **Angles.** Declare every angle feature (heading, head direction, a bearing
  to an object) `"circular"` and give it in radians. A rate of turning in
  rad/s is not an angle: leave it `"linear"`.
- **Features read from maps.** A linear feature read with
  `sample_spatial_features` (the height of a ramp, the value of a reward zone)
  changes in steps at bin edges. Pass `interpolation={"height": "previous"}`
  so the alignment holds the last value between frames instead of drawing a
  ramp between bins. Categorical features always do this.
- **What lies ahead of the animal.** `sample_egocentric_features` reads the
  same kind of map at points fixed in the animal's frame, such as 10 cm ahead
  (`ego_points=[[10.0, 0.0]]`; x is ahead, y is to the left). It places the
  points with the heading. Headings below `min_speed` are already NaN, and
  those frames come back unsupported.
- **Many events or long windows.** `retain_aligned=False` keeps only the
  summaries (`counts`, `means`, `category_counts`), computed in bounded memory,
  and gives the same summaries. Keep the default `True` when you need
  per-event values (`values`, `aligned_table()`).

The function below runs the whole analysis on one session. Here it is applied
to the simulated session as if it were two recordings joined at 60 s.

<!-- docs-test: run setup=docs_behavior_around_reward -->
```python
from neurospatial.behavior import compute_movement_angular_velocity, compute_speed
from neurospatial.events import align_continuous_to_events
from neurospatial.ops import heading_from_velocity, nearest_boundary


def behavior_around_events(env, times, positions, recording_id, event_times, window, dt):
    """Speed, turning, heading and wall distance around events.

    times : (n_frames,) s, one clock for all recordings, strictly increasing
    positions : (n_frames, 2) in env units, lost frames as NaN rows
    recording_id : (n_frames,) int, which recording each frame came from
    event_times : (n_events,) s, on the same clock as times
    """
    new_recording = np.r_[False, np.diff(recording_id) != 0]  # first frame of each later recording

    speed = compute_speed(times, positions, break_before=new_recording)
    turning = compute_movement_angular_velocity(times, positions, break_before=new_recording)
    heading = heading_from_velocity(times, positions, min_speed=2.0, break_before=new_recording)
    wall_distance, _ = nearest_boundary(env, positions)

    signals = np.column_stack([speed, turning, heading, wall_distance])
    return align_continuous_to_events(
        times,
        signals,
        event_times,
        window,
        dt=dt,
        feature_ids=["speed", "turning", "heading", "wall_distance"],
        feature_units={"speed": "cm/s", "turning": "rad/s", "heading": "rad", "wall_distance": "cm"},
        value_kinds={"heading": "circular"},  # turning is a rate in rad/s: linear
        break_before=new_recording,
        window_coverage="partial",
        retain_aligned=False,  # summaries only, for many events
    )


# Two recordings joined on one clock at 60 s
recording_id = (times >= 60.0).astype(int)
summary = behavior_around_events(
    env, times, positions, recording_id, reward_times, (-3.0, 3.0), dt=0.05
)
print(summary.summary())
assert summary.values is None

# Check: if the second recording started 30 cm further north, the jump would
# not be read as running. Heading equals heading of each recording on its own.
second = recording_id == 1
moved = positions + np.where(second[:, None], [0.0, 30.0], [0.0, 0.0])  # cm
new_recording = np.r_[False, np.diff(recording_id) != 0]
joined = heading_from_velocity(times, moved, min_speed=2.0, break_before=new_recording)
separate = np.r_[
    heading_from_velocity(times[~second], moved[~second], min_speed=2.0),
    heading_from_velocity(times[second], moved[second], min_speed=2.0),
]
bridged = heading_from_velocity(times, moved, min_speed=2.0)  # join not declared
near_join = np.abs(times - 60.0) < 0.5  # s
north = np.sin(joined[near_join]) > np.sqrt(0.5)  # within 45 degrees of North
north_bridged = np.sin(bridged[near_join]) > np.sqrt(0.5)
print(
    f"frames within 0.5 s of the join heading North: {north.sum()} with "
    f"break_before, {north_bridged.sum()} without, of {near_join.sum()}"
)
assert np.allclose(joined, separate, equal_nan=True)
assert not north.any()
assert (north.sum(), north_bridged.sum(), near_join.sum()) == (0, 14, 29)
```

`summary()` lists the number of events kept and excluded, the window and the
smallest count over all cells. The check at the end moves the second
recording 30 cm north, as if the animal had been put down elsewhere. With
`break_before`, heading is exactly what each recording gives on its own, and
no frame within 0.5 s of the join points North. Without it, 14 of those 29
frames point North: the jump between the recordings is read as running.

For your data, replace `times`, `positions`, `recording_id` and
`reward_times` with your own arrays, with lost frames as NaN rows.

## What this does not do

`align_continuous_to_events` describes behavior around events: counts, means,
resultant lengths and category proportions. It gives no confidence intervals
and no significance tests, and it does not say whether two conditions differ.
For those, start from the per-event values (`values`, or `aligned_table()` for
one row per event, time and feature) and resample across events, or fit a
model with event as the unit of replication. Events whose windows overlap
reuse the same frames and are not independent; check `overlapping` in
`events_table()` before treating them as separate samples.
