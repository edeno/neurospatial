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
shuffles the smallest possible p-value is 1/201, which is still below every
α used here. For your analysis, keep the published counts.

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
  change with time, so they reject all but one stationary event.
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
own candidate events in three ways:

- `"cell_identity"` reassigns spike trains among cells (Takigawa et al.
  2024). It keeps the population's timing but breaks the link between cells
  and their fields. Davidson's cell-identity shuffle and the rank-order
  cell-order test do the same.
- `"spike_shift"` shifts each cell's spikes circularly within the event. It
  keeps which cells fire and how much, but breaks their order. Tirole's
  spike-train shuffle does the same.
- `"poisson"` keeps each cell's spike count and draws new spike times.

Here we make two copies of each sweep for each kind. Each kind is decoded
like the real events and scored in its own `compare_replay_methods` call,
because the pseudo-event shuffle draws from every event in a call.

<!-- docs-test: run -->
```python
sweeps = intervals[kinds == "sweep"]
null_tables, decoded_nulls = [], {}
for null_kind in ("cell_identity", "spike_shift", "poisson"):
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
  own cell-identity shuffle repeats, but a third to a half of the spike-shift
  and Poisson copies. Those copies keep which cells fire, so their posterior
  stays in the part of the track the sweep covered. As with stationary
  events, its shuffles test only that the posterior is concentrated
  somewhere.
- `WU_2017` calls 12% of the cell-identity copies replay at α = 0.05, and
  `TIROLE_2022` stays near its α. `DIBA_BUZSAKI_2007` calls 17% of the
  spike-shift and Poisson copies replay at α = 0.1: those copies keep which
  cells fire, and in a sweep that covers part of the track those cells'
  fields still lie along it.
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
α = 0.18, calls nearly every spike-shift and Poisson copy replay. Match on
null data that none of the compared methods repeats, and report the rates
on the other kinds and on simulated replay with it. The `alpha` column of
`matched_alpha` is NaN for a method no α can hold to the target, and a
warning says why.

## Using your own recording

- **Rate maps.** Estimate them from running only, e.g.
  `compute_spatial_rates(env, spike_times, times, positions, speed=speed,
  min_speed=...)`. Unsupported bins are NaN, which the decoder treats as
  outside the track.
- **Candidate events.** Pass your event intervals in seconds, on the same
  clock as the spike times. `decode_intervals` decodes them in
  non-overlapping time bins with one set of rate maps. Several papers
  decoded overlapping windows (`method.decoding`, e.g. 20 ms windows stepped
  by 10 ms for `WU_2017`): for those, bin each interval with
  `neurospatial.decoding.bin_spikes_in_time` and decode with
  `decode_position`, the same way for real and null events.
- **Shuffles.** Keep each recipe's published `n_shuffles`. Give `rng` a
  fixed seed and report it.
- **Two running directions or two tracks.** Decode each event's spike
  counts with one rate-map set per direction
  (`decode_jointly(env, counts, {"outbound": ..., "inbound": ...}, dt=dt,
  times=...)`) and pass the joint result as the event, with the mapping of
  rate maps as `encoding_models`. Recipes such as `MALLORY_2025` and
  `BERNERS_LEE_2022` choose or sum the templates as their papers did; for
  other methods, pass `template=` to test one. A method with the
  pseudo-event shuffle (`DAVIDSON_2009`) cannot draw from joint decodes:
  pass it the summed posterior, as its recipe notes say.
- **Mazes with arms.** `ArmReplayMethod` recipes (`WU_FOSTER_2014`,
  `BERNERS_LEE_2021`) classify replay per arm: pass `arms=` to
  `compare_replay_methods`.
- **Several sessions.** Score each session's events in its own call, add a
  `session` column, and pass it as `groups=` to `detection_rates`,
  `method_agreement` and `matched_alpha`.

## References

- Davidson TJ, Kloosterman F, Wilson MA (2009). Hippocampal replay of
  extended experience. *Neuron* 63:497–507.
- Diba K, Buzsáki G (2007). Forward and reverse hippocampal place-cell
  sequences during ripples. *Nat Neurosci* 10:1241–1242.
- Silva D, Feng T, Foster DJ (2015). Trajectory events across hippocampal
  place cells require previous experience. *Nat Neurosci* 18:1772–1779.
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
