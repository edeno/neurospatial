# Comparing Replay Methods

**The question.** You recorded candidate events (sharp-wave ripples or bursts
of population activity) and decoded them with place fields from running.
Which events are replay? Published methods answer this differently. Each
combines a score for how well the decoded positions follow a path, shuffles
that say how large a score arises by chance, and thresholds. On the same
events, methods often disagree (Tingley & Peyrache 2020), and their
false-positive rates differ even at the same α (Takigawa et al. 2024). Before
you interpret a replay rate, know how much it depends on the method.

This guide applies several published methods to the same decoded events and
asks three questions:

1. How often does each method call replay events, and events without replay,
   "replay"?
2. How often does each call null events "replay", and does that depend on the
   kind of null data and on the event?
3. How do the methods compare at the same false-positive rate?

Detecting the candidate events (ripple power, multi-unit activity) is a
separate step and is not covered here: start from your own event intervals.

## The session

A rat runs 20 laps on a 2 m linear track while 40 place cells are recorded.
Their rate maps, estimated from running, are the decoder's encoding models.
After the run, 24 candidate events occur at rest. In this simulation we know
what each event contains:

- 12 **sweeps**: the decoded position moves along the track at 4 to 8 m/s,
  forward or backward, as replay is usually modeled;
- 6 **stationary** events: the population encodes one position;
- 6 **random** events: the encoded position jumps to a new random place every
  20 ms.

Each event lasts 80 to 200 ms, and cells fire at 4 times their running rate
at the encoded position (`rate_gain=4.0`). The spikes of all events are laid
out on one session clock, as a recording would have them.

<!-- docs-test: run -->
```python
import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurospatial.encoding import compute_spatial_rates
from neurospatial.replay import (
    apply_alpha,
    compare_replay_methods,
    decode_intervals,
    detection_rates,
    event_properties,
    matched_alpha,
    method_agreement,
    null_events,
)
from neurospatial.replay.methods import (
    DAVIDSON_2009,
    DIBA_BUZSAKI_2007,
    SILVA_2015,
    TIROLE_2022,
    WU_2017,
)
from neurospatial.simulation import linear_track_session, simulate_replay_event

# Running: 20 laps on a 200 cm track, 40 place cells
session = linear_track_session(
    duration=240.0, track_length=200.0, bin_size=4.0, n_place_cells=40, seed=0
)
env = session.env
rates = compute_spatial_rates(
    env,
    session.spike_times,
    session.times,
    session.positions,
    bandwidth=5.0,  # cm
    min_occupancy=0.5,  # s per bin
)

# Rest: 24 candidate events, 2 s apart, after the run
rng = np.random.default_rng(1)
kinds = np.array(["sweep"] * 12 + ["stationary"] * 6 + ["random"] * 6)
spikes_by_cell = [[] for _ in range(40)]
intervals = []  # (start, stop) of each event, s
for index, kind in enumerate(kinds):
    speed = {}
    if kind == "sweep":
        speed["speed"] = float(rng.choice([-1, 1]) * rng.uniform(400, 800))  # cm/s
    event = simulate_replay_event(
        env,
        rates,
        kind=kind,
        duration=float(rng.uniform(0.08, 0.2)),
        rate_gain=4.0,
        seed=index,
        **speed,
    )
    start = 300.0 + 2.0 * index
    for cell, train in enumerate(event.spike_times):
        spikes_by_cell[cell].append(np.asarray(train) + start)
    intervals.append((start, start + event.duration))
rest_spikes = [np.concatenate(trains) for trains in spikes_by_cell]
intervals = np.asarray(intervals)
print(f"{len(intervals)} events, {sum(t.size for t in rest_spikes)} spikes")
```

Each event is decoded in non-overlapping 10 ms time bins with the running
rate maps. `decode_intervals` returns one decoded event and one spike-count
matrix per interval. Decode every event, including the null events later,
with the same function and settings, so that differences between events come
from their spikes and not from the decoding.

<!-- docs-test: run -->
```python
dt = 0.01  # s
events, counts = decode_intervals(env, rest_spikes, intervals, rates, dt=dt)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
for ax, index in zip(axes, [0, 12, 18], strict=True):
    events[index].plot(ax=ax)
    ax.set_title(kinds[index])
fig.tight_layout()
```

Each panel shows the posterior over position (4 cm bins) in each 10 ms time
bin. The sweep's posterior moves steadily along the track. The stationary event
stays near one position. The random event jumps between positions. Real
events are less clean: they mix these patterns, end early, or cover only part
of the track.

## The published methods

`neurospatial.replay.methods` holds published methods as recipes. A recipe
records the paper's score, shuffles, α and criteria, and notes where the
paper and its code leave a choice open. This guide compares five that test
in different ways:

| Recipe | Score | How chance is judged |
|---|---|---|
| `DAVIDSON_2009` | Line fit: posterior mass within 15 cm of the best line | Column-cycle, cell-identity and pseudo-event shuffles; every p below 0.01 |
| `WU_2017` | Linear regression (R²) of the most likely position on time, in time bins with spikes | Time-bin permutation, p < 0.05; at least 4 active cells |
| `TIROLE_2022` | Weighted correlation of position and time | Spike-train, place-field and position-bin circular shifts; every p below 0.05; at least 5 active cells |
| `SILVA_2015` | Weighted correlation | No shuffles: \|r\| > 0.6 and no decoded jump over 40% of the track |
| `DIBA_BUZSAKI_2007` | Rank order of the cells' spikes vs. their place fields | Cell-order shuffle, p < 0.1; at least 5 active cells and 30% of the cells |

Methods are compared on the same decoded events, so differences come from
the test alone. A recipe's `decoding` attribute records the paper's
decoding settings, which are not enforced. Criteria counted in time bins
depend on them. For example, `PFEIFFER_FOSTER_2013` requires a trajectory
of 11 time bins in the paper's 20 ms windows stepped by 5 ms, which a
short event decoded in 10 ms bins cannot reach.

The published shuffle counts (1,500 for each of Davidson's three nulls) make
this guide slow to run, so here each method gets 200 shuffles. With 200
shuffles the smallest possible p-value is 1/201, or 2/201 for the two-sided
rank-order test, which is still below every α used here. For your analysis,
keep the published counts.

<!-- docs-test: run -->
```python
methods = [
    dataclasses.replace(method, n_shuffles=200)
    for method in (DAVIDSON_2009, WU_2017, TIROLE_2022, DIBA_BUZSAKI_2007)
] + [SILVA_2015]
for method in methods:
    print(f"{method.name:18s} {method.reference}")
```

## Apply them to the same events

`compare_replay_methods` applies every method to every event and returns one
row per (event, method). Each row holds the verdict `is_replay`, the score,
the p-value, and whether each criterion passed. Shuffles that change the
spikes or the rate maps decode again (Davidson's cell-identity shuffle,
Tirole's spike-train and place-field shifts), so they need the spike counts,
the rate maps and `dt`; rank-order methods also need the spike times. Each
(event, method) pair gets its own random stream from `rng`, so results are
reproducible and do not change when you add or remove methods.

`detection_rates` gives each method's share of events called replay, with a
95% Clopper-Pearson confidence interval. Here we break it down by the kind of
event.

<!-- docs-test: run -->
```python
table = compare_replay_methods(
    env,
    events,
    replay_methods=methods,
    spike_counts=counts,
    spike_times=rest_spikes,
    encoding_models=rates,
    dt=dt,
    rng=0,
)
table["kind"] = kinds[table["event"]]
by_kind = detection_rates(table, groups=["kind"])
print(by_kind.pivot(index="method", columns="kind", values="rate").round(2))
```

Every method except `SILVA_2015` detects 11 of the 12 sweeps; Silva's two
thresholds reject 4 of them (three fail |r| > 0.6 and three the 40% jump
limit). On the events without a path, the methods fail in opposite ways:

- `DAVIDSON_2009` calls five of the six stationary events replay. Its line
  search includes lines of zero speed, and all three of its shuffles spread
  the posterior across the track, so any event whose posterior stays in one
  place beats them. The correlation and regression scores need position to
  change with time: the correlation scores reject every stationary event and
  the regression rejects all but one.
- The rank-order test calls most random events replay, and the correlation
  and regression methods call four of six. Six events give a noisy rate: in
  a simulation of 60 such events, `WU_2017` and `TIROLE_2022` called 12 to
  15% replay, still above their α, and `DIBA_BUZSAKI_2007` 60%. A random
  event holds each position for 20 ms, two 10 ms time bins, so it has about
  half as many independent positions as time bins. Shuffles that move each
  time bin independently assume every bin is independent, so they
  underestimate how often a high correlation arises by chance. The rank-order
  test also treats each cell as independent, although cells with
  overlapping fields fire together. `DAVIDSON_2009` rejects these events,
  which look like its own column-cycle shuffle, and Silva's jump threshold
  rejects them too.

Neither failure is a bug: each test answers a narrower question than "is
this a trajectory?". This is why a false-positive rate must be measured on
several kinds of events without replay, and broken down by event properties.

`method_agreement` counts how often every pair of methods give the same
verdict, and Cohen's kappa, the agreement beyond what their rates predict.

<!-- docs-test: run -->
```python
agreement = method_agreement(table)
print(agreement[["first", "second", "agreement", "kappa"]].round(2).to_string())
```

`WU_2017`, `TIROLE_2022` and `DIBA_BUZSAKI_2007` agree closely (kappa above
0.8), including on the random events they all accept. `DAVIDSON_2009` agrees
little with them, because it accepts the stationary events and rejects the
random ones. Agreement depends on the mix of events: on your recording it
tells you how much the choice of method changes which events you analyze.

## False positives on null data

A false-positive rate needs events known to contain no replay. No single
kind of null event is right. Each keeps some structure of real events and
breaks the rest, and a method looks best on null data that repeats its own
shuffle (Takigawa et al. 2024). `null_events` makes null events from your
own candidate events in four ways:

- `"cell_identity"` reassigns spike trains among cells (Takigawa et al.
  2024). It keeps the population's timing but breaks the link between cells
  and their fields. Davidson's cell-identity shuffle and the rank-order
  cell-order test do the same.
- `"spike_shift"` shifts each cell's spikes circularly within the event. It
  keeps which cells fire and how much, but breaks their order. Tirole's
  spike-train shuffle does the same.
- `"spike_time_uniform"` keeps each cell's spike count and draws new spike
  times, uniformly over the event (the "spike jittering" of Stella et al.
  2019).
- `"rate_poisson"` draws each cell's count again, from its mean rate over
  all the events you pass, and places the spikes uniformly. These are the
  Poisson surrogate data of Silva et al. (2015), Grosmark & Buzsáki (2016)
  and Farooq et al. (2019). Unlike the other kinds, it does not keep an
  event's own spike counts, so every null event carries the session's mean
  activity rather than the event's burst. Grosmark & Buzsáki estimated the
  rates separately for each behavioral epoch and running direction: call
  `null_events` once per group of events.

Here we make two copies of each sweep for the first three kinds, which
keep each event's spike counts. Each kind is decoded like the real events and scored in its own
`compare_replay_methods` call, because the pseudo-event shuffle draws from
every event in a call.

<!-- docs-test: run -->
```python
sweeps = intervals[kinds == "sweep"]
null_tables, decoded_nulls = [], {}
for null_kind in ("cell_identity", "spike_shift", "spike_time_uniform"):
    null = null_events(rest_spikes, sweeps, kind=null_kind, n_copies=2, rng=2)
    null_decoded, null_counts = decode_intervals(
        env, null.spike_times, null.intervals, rates, dt=dt
    )
    decoded_nulls[null_kind] = (null_decoded, null_counts)
    null_table = compare_replay_methods(
        env,
        null_decoded,
        replay_methods=methods,
        spike_counts=null_counts,
        spike_times=null.spike_times,
        encoding_models=rates,
        dt=dt,
        rng=0,
    )
    null_table["null_kind"] = null_kind
    null_tables.append(null_table)
null_table = pd.concat(null_tables, ignore_index=True)

false_positives = detection_rates(null_table, groups=["null_kind"])
print(false_positives.pivot(index="method", columns="null_kind", values="rate").round(2))

fig, ax = plt.subplots(figsize=(7, 3))
names = [method.name for method in methods]
for x, method in enumerate(methods):
    if method is not SILVA_2015:  # Silva has no shuffles, so no α
        ax.hlines(method.alpha, x - 0.4, x + 0.4, color="gray", linestyle=":")
for offset, (null_kind, rows) in enumerate(false_positives.groupby("null_kind")):
    rows = rows.set_index("method").loc[names]
    x = np.arange(len(names)) + 0.25 * (offset - 1)
    ax.errorbar(
        x,
        rows["rate"],
        yerr=[rows["rate"] - rows["ci_low"], rows["ci_high"] - rows["rate"]],
        fmt="o",
        label=null_kind,
    )
ax.set_xticks(np.arange(len(names)), names, rotation=20)
ax.set_ylabel("False-positive rate")
ax.legend(title="null data")
fig.tight_layout()
```

The figure shows each method's false-positive rate on each kind of null data,
with 95% confidence intervals; the dotted lines mark each method's α. The 24
null events of each kind come from 12 sweeps, and copies of one sweep share
its spikes, so they are not independent and these intervals are too narrow.
More real events, not more copies, narrow them.

- `DAVIDSON_2009` calls none of the cell-identity copies replay, which its
  own cell-identity shuffle repeats, but a third to a half of the
  spike-shift and uniform-spike-time copies. Those copies keep which cells
  fire, so their posterior stays in the part of the track the sweep
  covered. As with stationary events, its shuffles test only that the
  posterior is concentrated somewhere.
- `WU_2017` calls 12% of the cell-identity copies replay at α = 0.05, and
  `TIROLE_2022` stays near its α. `DIBA_BUZSAKI_2007` calls 17% of the
  spike-shift and uniform-spike-time copies replay at α = 0.1: those copies
  keep which cells fire, and in a sweep that covers part of the track
  those cells' fields still lie along it.
- `SILVA_2015` has no shuffles, and its thresholds keep its rate low on
  every kind.

## Rates depend on the event

A single false-positive rate hides how it depends on the event: chance
replay is more common in events with few active cells (Tingley & Peyrache
2020). `event_properties` measures each event: active cells, spikes,
duration, posterior sharpness and the path of the decoded position.
`detection_rates(by=...)` breaks rates down by any of them.

<!-- docs-test: run -->
```python
cell_identity = null_table[null_table["null_kind"] == "cell_identity"]
null_decoded, null_counts = decoded_nulls["cell_identity"]
properties = event_properties(env, null_decoded, spike_counts=null_counts, dt=dt)
by_cells = detection_rates(
    cell_identity,
    properties=properties,
    by={"n_active_cells": [0, 20, np.inf]},
)
print(by_cells[["method", "n_active_cells", "n_events", "rate"]].to_string())
```

Here the false positives fall in events with fewer than 20 active cells, but
with 6 events in the other stratum that is weak evidence: compare strata only
with enough events in each. With real data, report rates per stratum of cells
and duration, and compare conditions with similar event properties (van der
Meer et al. 2020).

## Comparing at a matched false-positive rate

Methods with the same α call different shares of null events replay.
Takigawa et al. (2024) therefore compared methods at the α that gives each
the same false-positive rate on cell-identity copies. `matched_alpha` finds,
for each method, the largest α at which at most `target` of the null events
are called replay. `apply_alpha` then gives the real events' verdicts at
those α values, from the p-values already in the table, without running the
shuffles again.

Only methods that compare a p-value with α can be matched. `SILVA_2015` has
no shuffles, so it is left out.

<!-- docs-test: run -->
```python
tested = [method for method in methods if method is not SILVA_2015]
names = [method.name for method in tested]
matched = matched_alpha(
    cell_identity[cell_identity["method"].isin(names)],
    replay_methods=tested,
    target=0.05,
)
print(matched.round(4).to_string())

at_matched = apply_alpha(
    table[table["method"].isin(names)],
    alpha=matched.set_index("method")["alpha"],
    replay_methods=tested,
)
at_matched_rates = detection_rates(at_matched, groups=["kind"])
print(at_matched_rates.pivot(index="method", columns="kind", values="rate").round(2))

# The same alphas on the other kinds of null data
other_nulls = apply_alpha(
    null_table[null_table["method"].isin(names)],
    alpha=matched.set_index("method")["alpha"],
    replay_methods=tested,
    groups=["null_kind"],
)
other_rates = detection_rates(other_nulls, groups=["null_kind"])
print(other_rates.pivot(index="method", columns="null_kind", values="rate").round(2))
```

`published_rate` is each method's false-positive rate at its own α, and
`alpha` and `rate` are the matched α and the rate there. A method whose rate
was above the target gets a smaller α (`WU_2017`, 0.015;
`DIBA_BUZSAKI_2007`, just below its 0.1), and a conservative one a larger α
(`DAVIDSON_2009`, about 0.18).

These α values only illustrate the procedure. With 24 null events and a
target of 0.05, at most one null event may pass, so each matched α is set by
one or two null p-values. `matched_alpha` warns when there are too few null
events for any to pass; aim for `target * n_events` of at least 10, from
many real events.

The second table is the answer to the third question. At the matched α,
`DAVIDSON_2009` detects every sweep but also every stationary event and two
of six random events, while `WU_2017` calls fewer random events replay than
at its own α.

Matching holds the rate on one kind of null data only, and here it was done
on cell-identity copies, which `DAVIDSON_2009`'s own cell-identity shuffle
and the rank-order cell-order test already repeat. Those methods look
conservative on these copies, so they get a larger α. Takigawa et al.
matched on cell-identity copies for methods without such a shuffle. The last
table applies the matched α to the other kinds: `DAVIDSON_2009`, now at
α = 0.18, calls nearly every spike-shift and uniform-spike-time copy
replay. Match on null data that none of the compared methods repeats, and
report the rates on the other kinds and on simulated replay with it. The `alpha` column of
`matched_alpha` is NaN for a method no α can hold to the target, and a
warning says why.

## Data-set-level replay tests

Several papers draw no verdict on single events. They ask whether a data set,
a session or every event of a condition, holds more trajectory events than
data sets without sequences would. `neurospatial.replay.dataset` holds these
tests and records each paper's choices as a `DatasetTestRecipe`
(`SILVA_2015_MATRIX`, `SILVA_2015_RANK_ORDER`, `GROSMARK_2016_SURROGATES`,
`FAROOQ_2019_TIME_BIN`, `FAROOQ_2019_POISSON`, `FAROOQ_2019_SPACE_BIN`,
`FAROOQ_2019_WEIGHTED`, `LIU_2019_TEMPLATES`, `STELLA_2019_JITTER`,
`XU_2019_ROTATION`). Each recipe's `notes` give the calls that run it and
what the paper leaves open. The tests come in three kinds.

**A data-set statistic against null data sets.** Silva et al. (2015) counted
the events with |weighted correlation| above one threshold and maximum jump
below another, for every pair of thresholds on a 10 x 10 grid, and compared
each count with its value in 5,000 null data sets, each made by
column-cycling every event once. `dataset_values` measures the events of the
real data set and of the null data sets with the same method, so both are
scored the same way, and `dataset_test` gives a p-value per grid point,
`(n + 1) / (r + 1)` as Silva et al. computed it. Here we use 199 null data
sets so the guide runs quickly.

<!-- docs-test: run -->
```python
from neurospatial.replay import dataset_test, dataset_values, distribution_test
from neurospatial.replay.dataset import FAROOQ_2019_POISSON, SILVA_2015_MATRIX

real = dataset_values(env, events, replay_method=SILVA_2015, spike_counts=counts)
shuffled = dataset_values(
    env, events, replay_method=SILVA_2015, null="column_cycle", n_datasets=199, rng=0
)
matrix = dataset_test(real, shuffled, criteria=SILVA_2015_MATRIX.criteria)
ax = matrix.plot()
ax.set_title("Silva et al. (2015) significance matrix, 199 data sets")
# Silva's trio: (|r| > 0.6, jump < 0.4), (0.7, 0.4), (0.7, 0.3)
print(matrix.p_value[[6, 7, 7], [3, 3, 2]].round(3))
print(matrix.statistic[[6, 7, 7], [3, 3, 2]].round(2))
```

The matrix shows the p-value of every pair of thresholds: rows are |r|
thresholds from > 0 (top) to > 0.9, columns jump thresholds from < 0.1 to
< 1 of the track, red below 0.05 and blue above. A third to a quarter of
the 24 events pass Silva's three threshold pairs, more than in any of the
199 column-cycled data sets, so p is the smallest 199 data sets can give,
1/200. Silva et al. read trajectory events from the lower-left quadrant
(high |r|, small jumps). With 12 sweeps among 24 events nearly every grid
point is red here. Two corners are not: at the top right, thresholds so
lenient that shuffled events pass them too (any |r| with jumps up to the
whole track), and at the bottom left, a pair no event passes (|r| > 0.9
with jumps below a tenth of the track), where every data set counts 0 and
p = 1. With 5,000 data sets the smallest p is 1/5,001, the 0.0002 of Silva
et al.'s Supplementary Table 1.

**Surrogate data sets made from spikes.** Silva et al. (2015), Grosmark &
Buzsáki (2016) and Farooq et al. (2019) also compared the real events with
data sets of Poisson spikes at each cell's mean rate. `null_events` makes
`n_copies` such copies of every event, and `copy_index` says which copy, so
copy `k` of every event is the `k`-th null data set. Farooq et al. computed
p as one minus the share of null data sets below the real one, without the
`+ 1`; `paper_p_value` applies a recipe's own rule.

<!-- docs-test: run -->
```python
poisson = null_events(rest_spikes, intervals, kind="rate_poisson", n_copies=50, rng=3)
poisson_events, poisson_counts = decode_intervals(
    env, poisson.spike_times, poisson.intervals, rates, dt=dt
)
surrogate = dataset_values(
    env, poisson_events, replay_method=SILVA_2015, spike_counts=poisson_counts
)
surrogate["dataset"] = poisson.copy_index[surrogate["event"]]
against_poisson = dataset_test(real, surrogate, criteria=SILVA_2015_MATRIX.criteria)
print(
    "p at (|r| > 0.6, jump < 0.4):",
    round(against_poisson.p_value[6, 3], 3),
    "library;",
    round(FAROOQ_2019_POISSON.paper_p_value(against_poisson)[6, 3], 3),
    "Farooq's rule",
)
```

No Poisson data set has as many trajectory events as the real one, so the
library's p is 1/51 and Farooq's rule gives 0. The `+ 1` keeps p above zero
and changes it by at most 1/(n + 1); report which rule you use. Poisson
surrogates break every sequence, but also each event's own burst: every
surrogate event fires at the session's mean rate.

**Pooled two-sample tests.** Stella et al. (2019), Xu et al. (2019) and
Grosmark & Buzsáki (2016, Fig. S8) compared a value of every real event with
the pooled values of the null events, by a Kolmogorov-Smirnov or rank-sum
test. `distribution_test` states the side in terms of the values:
`alternative="greater"` asks whether the real events' values are larger.
(SciPy's `ks_2samp` names the side by the distribution functions, the
opposite way.)

<!-- docs-test: run -->
```python
pooled = distribution_test(
    real.assign(abs_score=real["score"].abs()),
    surrogate.assign(abs_score=surrogate["score"].abs()),
    column="abs_score",
    test="ks",
    alternative="greater",
)
print(pooled.summary())
```

The real events' |r| (median 0.61) are larger than those of the 1,200
Poisson events (median 0.19). The 50 copies of one event share its interval
and are not independent; with many copies the pooled null estimates the
null distribution well, but the test's power rests on the 24 real events,
and a small p from tens of thousands of null values can come from a small
shift. Report the medians with the p-value.

**A share of events against chance.** Grosmark & Buzsáki (2016) tested the
share of events with p < 0.05 against 5%. `detection_rates(...,
expected_rate=0.05)` adds an exact binomial p-value to each rate.

<!-- docs-test: run -->
```python
wu = table[table["method"] == "wu_2017"]
share = detection_rates(
    wu.assign(is_replay=wu["p_value"] < 0.05), groups=["kind"], expected_rate=0.05
)
print(share[["kind", "n_events", "n_replay", "rate", "p_value"]].round(4))
```

Eleven of 12 sweeps have p < 0.05 under `WU_2017`, far more than 5%. So do
4 of 6 random events (p = 0.0001), the false positives found above: a share
above chance says that some events beat their shuffles, not that they are
trajectories.

**Template order.** Liu et al. (2019) asked whether the order in which cells
fire in sleep follows their order on the track, without decoding. Each
cell's mean relative rank across sleep frames (`relative_rank_template`) is
correlated with its field position, averaged over animals and running
directions, and compared with shuffles of the running template
(`template_correlation_test`). Our sweeps run both ways, and a backward
sweep reverses the order, so here the frames are the sweeps whose weighted
correlation is positive, for one "animal" and one running direction.

<!-- docs-test: run -->
```python
from neurospatial.replay import relative_rank_template, template_correlation_test

field_peaks = env.bin_centers[np.nanargmax(np.asarray(rates.firing_rates), axis=1), 0]
forward = intervals[(kinds == "sweep") & (real["score"].to_numpy() > 0)]
sleep_order = relative_rank_template(rest_spikes, forward, min_active_cells=4)
templates = template_correlation_test(
    [sleep_order], [field_peaks], n_shuffles=999, rng=0
)
print(templates.summary())
```

The sleep order correlates with the field order (rho = 0.54) better than in
all 999 shuffles. Liu et al. reported the percentile of the real
correlation among the shuffles (97.6 before running) and a one-tailed p of
`1 - percentile / 100`, which `LIU_2019_TEMPLATES.paper_p_value(templates)`
gives.

Paper-scale runs are tractable. For the column-cycle, weighted column-cycle
and time-bin nulls with a weighted-correlation score, `dataset_values`
measures each event once per circular shift of each time bin and then reads
every shuffle from those tables: 1,000 events and 5,000 data sets took about
2 s on a laptop, and the 10 x 10 matrix test as long, so Silva et al.'s 9,818
events take well under a minute (their table holds 49 million rows, about
2 GB). `decode_intervals` decodes disjoint intervals in one pass, about 70
times faster than one interval at a time, which matters for Farooq et al.'s
500 Poisson data sets of 38,820 frames (about 10 minutes to decode). Nested
designs, such as Grosmark & Buzsáki's 100 surrogate data sets each tested
event by event against 200 place-field shuffles (about an hour per class
and epoch), parallelize over data sets with seeds from
`np.random.SeedSequence(seed).spawn(100)`.

## Using your own recording

- **Rate maps.** Estimate them from running only, e.g.
  `compute_spatial_rates(env, spike_times, times, positions, speed=speed,
  min_speed=...)`. Unsupported bins are NaN, which the decoder treats as
  outside the track.
- **Candidate events.** Pass your event intervals in seconds, on the same
  clock as the spike times. `decode_intervals` decodes them in
  non-overlapping time bins with one set of rate maps. Several papers
  decoded overlapping windows (`method.decoding`, e.g. 20 ms windows stepped
  by 10 ms for `WU_2017`). For those, bin each interval at the step,
  `counts, grid = bin_spikes_in_time(spike_times, dt=0.01, t_start=start,
  t_stop=stop)`, and add each pair of neighboring rows to make 20 ms
  windows, `counts[:-1] + counts[1:]`, centered at
  `(grid[:-1] + grid[1:]) / 2`. Decode them with
  `decode_position(env, windows, rates, dt=0.02, times=centers)`. Do the
  same for real and null events.
- **Shuffles.** Keep each recipe's published `n_shuffles`. Give `rng` a
  fixed seed and report it.
- **Two running directions or two tracks.** Decode each event's spike
  counts with one rate-map set per direction
  (`decode_jointly(env, counts, {"outbound": ..., "inbound": ...}, dt=dt,
  times=...)`) and pass the joint result as the event, with the mapping of
  rate maps as `encoding_models`. Recipes such as `MALLORY_2025` and
  `BERNERS_LEE_2022` choose or sum the templates as their papers did.
  `JADHAV_2016`, `TANG_2017`, `SHIN_2019` (the four W-track trajectories)
  and `OLAFSDOTTIR_2016` (two directions decoded separately) test every
  template and keep the one with the smallest p-value
  (`template_choice="smallest_p"`). For other methods, pass `template=` to
  test one, or add that choice with `dataclasses.replace(method,
  template_choice="smallest_p")`. The p-value is not corrected for the
  number of templates, as in those papers, so measure the false-positive
  rate with all your templates. `MUESSIG_2019` decodes each direction
  separately and takes the largest score over the directions, in the
  event and in every shuffle (`template_choice="max_score"`): decode each
  direction with `decode_position` and pass
  `{"cw": cw_result, "ccw": ccw_result}` as the event, with
  `encoding_models={"cw": ..., "ccw": ...}`. Taking the largest score of
  the event but comparing it with one direction's shuffles would call too
  many events replay. A method with the pseudo-event shuffle cannot draw
  from joint or separately decoded templates. Davidson et al.
  (2009) decoded position and direction jointly and summed over direction,
  but `DAVIDSON_2009`'s cell-identity shuffle decodes the shuffled counts
  again and refuses a summed posterior. Either decode on one set of rate
  maps and pass that posterior, or drop that shuffle with
  `dataclasses.replace(DAVIDSON_2009, nulls=("column_cycle",
  "pseudo_event"))` and pass the summed posterior,
  `joint.joint("outbound") + joint.joint("inbound")`. `DIBA_BUZSAKI_2007`
  given both directions' rate maps calls an event replay when either
  template passes, so it calls more events without replay significant than
  with one template: measure its false-positive rate with both templates,
  as you run it on real events.
- **Circular and square tracks.** `WIKENHEISER_REDISH_2013` and
  `MUESSIG_2019` score events on a closed loop: build the track with
  `Environment.from_graph` from a cycle, with an `edge_order` whose last
  segment ends where the first starts, so a sequence that runs past the
  point where the linear position starts again is scored whole.
- **Criteria with a shuffle test.** `WIDLOSKI_FOSTER_2025` finds each
  candidate as `WIDLOSKI_FOSTER_2022` does, then decodes the candidate's
  time bins again with the rate maps permuted across cells. Its mean
  posterior spread and mean center-of-mass jump must each be smaller than
  in at least 96 of 100 shuffles (`tested_quantities`). Pass the 80 ms
  windows stepped by 5 ms as the event, `dt=0.08`, the rate maps as
  `encoding_models` and the animal's speed rule as `valid_time_bins`. The
  table reports each quantity and its p-value, such as `mean_spread` and
  `p_value[mean_spread]`.
- **A narrow line band.** `MABOUDI_2018` fits the line within 3 cm, so it
  searches 35,000 random lines (`score_options["n_lines"]`) rather than the
  dense grid, which is slow at that band. Set `line_seed` in
  `score_options` to repeat the same lines.
- **Spike pairs without decoding.** `GUPTA_2010` scores every pair of
  spikes less than 80 ms apart by whether the two cells fire in the order
  of their place fields. Pass `spike_times=`, and pass one set of rate maps
  per lap direction as a mapping in `encoding_models`, in bins of about
  3.5 cm for the paper's field rule. Each template is tested forward and
  backward, and the smallest p-value is reported without correction, as in
  the paper. `unit_groups=` on `detect_pairwise_order_replay` leaves out
  pairs of cells on the same tetrode.
- **Mazes with arms.** `ArmReplayMethod` recipes (`WU_FOSTER_2014`,
  `BERNERS_LEE_2021`) classify replay per arm: pass `arms=` to
  `compare_replay_methods`.
- **Fernández-Ruiz-lab recipes.** `HARVEY_2023` (weighted correlation) and
  `HARVEY_2023_LINE` or `SOULA_2023` (`regression_line_score`, the
  posterior within 10.5 cm of the line through the most likely positions)
  score the same 20 ms decodes; pass `score_options={"band": 10.5,
  "wrap_band": True}` to wrap the band around the track ends as the
  authors' code does. `CHANG_2025` takes a joint decode of the two T-maze
  trajectories (`decode_jointly`) in 10 ms bins and tests both tails of
  the signed correlation. `LIU_2023_BIAS` needs no decoding: build the
  template from the running theta cycles' spikes,
  `template = mean_pairwise_bias(spike_times, theta_cycles)`, and pass it
  as `compare_replay_methods(..., spike_times=..., template_bias=template,
  dt=dt)`. Liu et al. report the share of significant post-task events
  minus the share in pre-task sleep: run both sets of events and subtract
  their `detection_rates`.
- **Firing order without decoding.** `LEE_WILSON_2002` and
  `JI_WILSON_2007` test whether the cells fire in the order of their
  fields, from the spike times and rate maps alone (pass `spike_times=` as
  for the rank-order methods). Their p-values are exact and take few
  values, so a 1/24 cutoff counts a perfect four-letter word. Both papers
  drew their conclusions from counts of matching events over a data set:
  run `template_match_dataset_test` for those. Lee & Wilson's events are
  words cut from sleep spikes, one template's cells at a time:
  `spike_words(spike_times, max_isi=0.05, max_gap=0.1, cells=template_cells,
  epochs=sws).intervals`. `NADASDY_1999` asks a different question, whether
  spike sequences recur within one session more often than in surrogate
  trains (`recurring_sequences_significance`), and has no per-event
  verdict.
- **Several sessions.** Score each session's events in its own call, add a
  `session` column, and pass it as `groups=` to `detection_rates`,
  `method_agreement` and `matched_alpha`.

## References

- Davidson TJ, Kloosterman F, Wilson MA (2009). Hippocampal replay of
  extended experience. *Neuron* 63:497–507.
- Diba K, Buzsáki G (2007). Forward and reverse hippocampal place-cell
  sequences during ripples. *Nat Neurosci* 10:1241–1242.
- Farooq U, Sibille J, Liu K, Dragoi G (2019). Strengthened temporal
  coordination within pre-existing sequential cell assemblies supports
  trajectory replay. *Neuron* 103:719–733.
- Grosmark AD, Buzsáki G (2016). Diversity in neural firing dynamics supports
  both rigid and learned hippocampal sequences. *Science* 351:1440–1443.
- Liu K, Sibille J, Dragoi G (2019). Preconfigured patterns are the primary
  driver of offline multi-neuronal sequence replay. *Hippocampus*
  29:275–283.
- Silva D, Feng T, Foster DJ (2015). Trajectory events across hippocampal
  place cells require previous experience. *Nat Neurosci* 18:1772–1779.
- Stella F, Baracskay P, O'Neill J, Csicsvari J (2019). Hippocampal
  reactivation of random trajectories resembling Brownian diffusion.
  *Neuron* 102:450–461.
- Takigawa M, Huelin Gorriz M, Tirole M, Bendor D (2024). Evaluating
  hippocampal replay without a ground truth. *eLife* 13:e85635.
- Tingley D, Peyrache A (2020). On the methods for reactivation and replay
  analysis. *Phil Trans R Soc B* 375:20190231.
- Tirole M, Huelin Gorriz M, Takigawa M, Kukovska L, Bendor D (2022).
  Experience-driven rate modulation is reinstated during hippocampal replay.
  *eLife* 11:e79031.
- van der Meer MAA, Kemere C, Diba K (2020). Progress and issues in
  second-order analysis of hippocampal replay. *Phil Trans R Soc B*
  375:20190238.
- Wu X, Foster DJ (2014). Hippocampal replay captures the unique topological
  structure of a novel environment. *J Neurosci* 34:6459–6469.
- Wu CT, Haggerty D, Kemere C, Ji D (2017). Hippocampal awake replay in
  fear memory retrieval. *Nat Neurosci* 20:571–580.
