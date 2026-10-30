# Comparing Reactivation Methods

**The question.** You recorded a population of cells during rest, a task
and rest again. Do the cells that fired together during the task fire
together again during rest after it, more than during rest before it?
Published analyses answer this in two ways. Pairwise methods correlate every
pair of cells in each period and ask whether the task's pattern of pair
correlations predicts rest after it beyond what rest before it already
predicts: the explained variance, EV (Kudrimoti et al. 1999). Assembly
methods find groups of cells that fire together during the task and measure
how strongly each group co-fires in each rest period (Peyrache et al. 2010;
van de Ven et al. 2016). Within each family, the papers differ in the time
bin, the cell pairs, the windows of rest and the test. This guide applies
the published recipes to one session and asks:

1. What does each method report, and how do they differ?
2. What do they report on data without reactivation, and which kinds of
   "no reactivation" fool them?
3. Can equal firing rates or larger time bins separate reactivation from
   a change in rate, and how much does the time bin matter?
4. How do the methods that count spikes in sharp-wave ripples (SWRs) or
   theta cycles, or follow the assemblies around SWRs, differ from those
   that bin time?

Detecting sleep stages, ripples or theta cycles is a separate step and is
not covered here: start from your own epochs and events.

## The session

A simulated session of 30 units: 20 minutes of rest, 20 minutes of a task
and 20 minutes of rest. Every unit fires at 2 Hz. Three assemblies of five
units fire together in brief events once per second during the task, at the
same rate during rest after it, and never during rest before it. Both rest
periods also have population bursts, 100 ms in which every unit fires five
times faster, which correlate every pair of cells during rest, as the
population events of sleep do. The simulation returns the bursts as
`control_events` and `match_events`, stand-ins for SWRs, for the
event-based methods at the end of this guide.

<!-- docs-test: run -->
```python
import dataclasses

import numpy as np
import pandas as pd

from neurospatial.decoding import bin_spikes_in_time
from neurospatial.reactivation import (
    KUDRIMOTI_1999,
    PUBLISHED_REACTIVATION_METHODS,
    ReactivationWindows,
    compare_reactivation_methods,
    detect_assemblies,
    null_reactivation_data,
    pairwise_correlations,
    reactivation_detection_rates,
    simulated_reactivation_session,
)

session = simulated_reactivation_session(n_units=30, duration=1200.0, rng=0)
print("assemblies:", session.assemblies)
print("rest before:", session.control_epochs, "task:", session.template_epochs)
print("rest after:", session.match_epochs)
```

The pair rules need each unit's tetrode, shank or brain region. Here three
units share a tetrode, two tetrodes a shank, and the first 15 units are in
one region and the rest in another.

<!-- docs-test: run -->
```python
unit_groups = {
    "tetrode": np.arange(30) // 3,
    "shank": np.arange(30) // 6,
    "region": np.arange(30) // 15,
}
```

## The published methods

`PUBLISHED_REACTIVATION_METHODS` holds the recipes. This section and the
next two use those that bin time; the recipes that count spikes in events
are compared [at the end](#methods-that-use-events). A recipe records the
paper's time bin, pair rule, windows, statistic and test, and its `notes`
quote the paper for each value and say where the paper leaves a choice
open. Unlike the replay recipes, a reactivation recipe applies its time bin,
because the correlation of 100 ms counts is a different number from that of
250 ms counts.

| Recipe | Bin | Pairs | Windows | Statistic | Within-session test |
|---|---|---|---|---|---|
| `KUDRIMOTI_1999` | 100 ms | different tetrodes | last 10 min before; 10-min blocks after | EV | t test of the partial correlation |
| `PENNARTZ_2004` | 50 ms | all | last 10 min before; 10-min blocks after (clock) | EV − REV | none (across sessions) |
| `TATSUNO_2006` | 250 ms | different tetrodes | 15-min segments; EV averaged over segments before | EV | none |
| `LANSINK_2009` | 50 ms | different regions | 20-min blocks of rest | EV − REV | bootstrap of pairs: 2.5th percentile above 0 |
| `PEYRACHE_2010` | 100 ms | (assemblies, PCA) | whole periods | mean R(t) after − before | per-bin cell-identity shuffles (shares only) |
| `VAN_DE_VEN_2016` | 25 ms | (assemblies, ICA) | whole periods; smoothed spikes | mean R(t) after − before | none (across patterns) |
| `OLIVA_2020_ASSEMBLY` | 20 ms | (assemblies, ICA) | whole periods; smoothed spikes, z-scored over the whole session | mean R(t) after − before | none |
| `GIRI_2019`, `GIRI_2024` | 250 ms | different shanks; `pair_mask` | 15-min windows sliding by 5 min | EV averaged over windows before | none |

REV, the reversed EV (Pennartz et al. 2004), swaps the two rest periods: how
much the task's pattern predicts rest before it beyond rest after it. Its
size shows how large EV is without any reactivation.

The published 10,000 bootstrap samples (Lansink) and 1,000 shuffles
(Peyrache) are slow for a guide, so here they get 1,000 and 200. Keep the
published counts for your analysis. Kudrimoti's notes, for example, give
the source of each value and the choices the paper leaves open:

<!-- docs-test: run -->
```python
fewer = {"lansink_2009": 1000, "peyrache_2010": 200}
methods = [
    dataclasses.replace(method, n_shuffles=fewer[method.name])
    if method.name in fewer
    else method
    for method in PUBLISHED_REACTIVATION_METHODS
    if not method.needs_events
]
for note in KUDRIMOTI_1999.notes:
    print("-", note)
```

## Apply them to the same session

`compare_reactivation_methods` bins the spikes at each method's bin size in
each of its windows and returns one row per method, window of rest after
and assembly pattern. `value` is the method's statistic; `is_reactivated` is
its within-session verdict, missing for methods that tested across sessions
or patterns instead. A method that cannot be applied has an `error` row:
one for the method when its task or rest before fails (for example a
period shorter than one window), or one for each window of rest after that
fails (for example one without events), keeping the method's other
windows. Each method draws from its own random stream, derived from `rng`
and its name, so adding or removing methods does not change the others.

<!-- docs-test: run -->
```python
table = compare_reactivation_methods(
    session.spike_times,
    session.template_epochs,
    session.match_epochs,
    session.control_epochs,
    reactivation_methods=methods,
    unit_groups=unit_groups,
    rng=0,
)
columns = ["method", "window", "component", "value", "reversed_explained_variance"]
print(table[[*columns, "is_reactivated"]].round(3).to_string(index=False))
```

Every EV method finds EV above 0.9 and REV below 0.01: the task's pattern of
pair correlations returns in rest after it and was absent before it.
Kudrimoti's 20 minutes of rest after hold two of its three 10-minute blocks,
and its partial correlation is significant in both; Lansink's bootstrap
interval of EV − REV lies above 0. The three assembly methods find the three
assemblies and a mean activation strength R(t) more than 1 higher after the
task than before it.

The assembly rows say more about where the difference comes from:

<!-- docs-test: run -->
```python
assembly = table[table["component"].notna()]
print(
    assembly[
        [
            "method",
            "member_unit_ids",
            "control_mean_activation",
            "match_mean_activation",
            "fraction_above_control_threshold",
            "control_fraction_above_shuffle",
            "match_fraction_above_shuffle",
        ]
    ]
    .round(3)
    .to_string(index=False)
)
```

- Mean R(t) before the task is above 0 (about 0.3 for Peyrache's 100 ms
  bins), because the population bursts make every cell, including the
  assembly's, fire together. Comparing with rest before removes this.
- About 5% of the bins after the task exceed the 99th percentile of R(t)
  before it (Peyrache et al. 2010 found 4.5%), against 1% expected without
  reactivation.
- Permuting the cells' identities in each bin keeps how many cells fire and
  destroys which cells fire together. Before the task, 1.4 to 1.9% of bins
  beat the 99th percentile of their 200 shuffles (a bin that ranks in the
  top 3 of 201 by chance does so 1.5% of the time); after it, about 8%. The
  co-firing after the task is specific to the assembly, not a rise in
  population activity.

## Data without reactivation

How large is each statistic when there is no reactivation, and what fools a
method? `null_reactivation_data` keeps the spikes outside rest after the
task and replaces those inside it in one of three ways:

- `"cell_identity"`: each unit gets another unit's spikes. Rest after keeps
  its own co-firing, bursts included, but which cells fire together no
  longer matches the task.
- `"circular_shift"`: each unit's spikes are rotated in time by their own
  offset. Rates and spike timing within a unit are kept; all co-firing,
  bursts included, is lost.
- `"rate_poisson"`: Poisson spikes at each unit's rate. Only the rates are
  kept.

Two simulated sessions add other kinds: one without reactivation
(`reactivation=0`), and one in which the assembly members fire twice as
much after the task without firing together (`rate_change=1.0`). The
second is the confound Tingley & Peyrache (2020) warn about: EV "does not
account for the possible change in firing rates (independent of
correlations) that inevitably leads to changes in correlation".

<!-- docs-test: run -->
```python
def compare(spike_times, epochs_from, seed):
    return compare_reactivation_methods(
        spike_times,
        epochs_from.template_epochs,
        epochs_from.match_epochs,
        epochs_from.control_epochs,
        reactivation_methods=methods,
        unit_groups=unit_groups,
        rng=seed,
    )


tables = [table.assign(data="reactivated")]
for kind in ("cell_identity", "circular_shift", "rate_poisson"):
    null = null_reactivation_data(
        session.spike_times, session.match_epochs, kind=kind, rng=1
    )
    tables.append(compare(null, session, seed=1).assign(data=kind))
for name, options in {
    "no_reactivation": {"reactivation": 0.0},
    "rate_change": {"reactivation": 0.0, "rate_change": 1.0},
}.items():
    other = simulated_reactivation_session(n_units=30, duration=1200.0, rng=2, **options)
    tables.append(compare(other.spike_times, other, seed=2).assign(data=name))
everything = pd.concat(tables, ignore_index=True)
mean_value = everything.groupby(["method", "data"], sort=False)["value"].mean()
print(mean_value.unstack().round(3).to_string())
```

- **The null data.** Every EV method drops from above 0.9 to about 0 on all
  three kinds of null data and on the session without reactivation. Neither
  the pair-correlation pattern of rest after the task nor its rates alone
  produce EV.
- **Assemblies on null data.** The assembly difference depends on the kind
  of null. With cell identities permuted it is small but positive, 0.07 to
  0.23 or 5 to 11% of the reactivated value, because a permutation can send
  an assembly member's spikes to another member. With trains rotated or
  redrawn it is negative: the bursts still
  correlate the cells in rest before the task but no longer in rest after
  it. A null that removes co-firing the control period keeps is not neutral
  for a before-after difference; prefer the cell-identity null, or rotate
  the spikes of both rest periods.
- **The rate confound.** Doubling the members' rates after the task, with no
  co-firing beyond the bursts, raises every EV method to 0.12 to 0.18 and
  the assembly differences to 0.1 to 0.23. Higher rates make the bursts
  correlate the members more, along exactly the pattern the task set up.

`reactivation_detection_rates` counts the verdicts of the methods that have
one, with exact binomial intervals. With one session per kind the counts
are small; on your own data, simulate or shuffle many sessions.

<!-- docs-test: run -->
```python
rates = reactivation_detection_rates(everything, groups=["data"])
tested = rates[rates["n_tested"] > 0]
called = tested["n_reactivated"].astype(str) + " of " + tested["n_tested"].astype(str)
print(tested.assign(called=called).pivot(index="data", columns="method", values="called"))
```

Kudrimoti's t test and Lansink's bootstrap call none of the null data
reactivated, and both call the rate-change session reactivated. A
within-session test of EV cannot tell a change in co-firing from a change
in rate. The next section applies the two controls Tingley & Peyrache
(2020) propose.

## Controlling for a change in rate

Tingley & Peyrache (2020) name two controls for the rate confound: "It is
possible to control for this by subsampling spike trains to equalize rates
[Giri et al. 2019], or greatly increase the time bin size in order to
estimate the contribution of slow time-scale rate fluctuations." Both are
applied below to four simulated sessions: the reactivated one, one without
reactivation, one whose assembly members only fire twice as much after the
task, and one with both reactivation and the rate change.

<!-- docs-test: run -->
```python
from neurospatial.reactivation import (
    GIRI_2019,
    PEYRACHE_2010,
    equalize_firing_rates,
    explained_variance_reactivation,
    pairwise_activation,
)

sessions = {
    "reactivated": session,
    "no_reactivation": simulated_reactivation_session(
        n_units=30, duration=1200.0, rng=2, reactivation=0.0
    ),
    "rate_change": simulated_reactivation_session(
        n_units=30, duration=1200.0, rng=2, reactivation=0.0, rate_change=1.0
    ),
    "reactivated_and_rate_change": simulated_reactivation_session(
        n_units=30, duration=1200.0, rng=2, rate_change=1.0
    ),
}


def compare_sessions(methods, names):
    tables = []
    for name in names:
        data = sessions[name]
        table = compare_reactivation_methods(
            data.spike_times,
            data.template_epochs,
            data.match_epochs,
            data.control_epochs,
            reactivation_methods=methods,
            unit_groups=unit_groups,
            rng=0,
        )
        tables.append(table.assign(data=name))
    return pd.concat(tables, ignore_index=True)


def by_session(table, column="value"):
    return table.groupby(["data", "method"], sort=False)[column].mean().unstack()
```

### Equal rates (Giri et al. 2019)

Giri et al. "repeated this analysis after subsampling to equalize the
firing rate for each neuron in PRE and POST", over "100 different random
subsamples equalizing PRE and POST spike counts of each cell", and reported
the median correlation and p-value. `equalize_firing_rates` draws one such
subsample from binned counts: for each unit, the rest period in which it
fires more per bin loses spikes chosen at random until it fires at its rate
in the other period. The task is not subsampled. A recipe with
`n_rate_subsamples=` does this for each pair of windows of rest before and
after and reports the subsample whose statistic is the median (the lower
of the two middle ones for an even count), so EV, REV, the partial
correlation and its p-value all come from one subsample and stay
consistent with each other. Giri et al. drew 100; 10 keep this guide fast.

<!-- docs-test: run -->
```python
equal_rates = []
for method in (KUDRIMOTI_1999, GIRI_2019):
    equal_rates += [
        method,
        dataclasses.replace(
            method, name=f"{method.name}_equal_rates", n_rate_subsamples=10
        ),
    ]
equal_table = compare_sessions(equal_rates, sessions)
print("EV:")
print(by_session(equal_table).round(3).to_string())
print("Kudrimoti windows called reactivated (of 2):")
called = by_session(equal_table, "is_reactivated").dropna(axis=1)
print(called.mul(2).astype(int).to_string())
```

- **The rate change alone.** With every spike, Kudrimoti's EV is 0.12 and
  Giri's 0.14, and Kudrimoti's t test calls both windows reactivated. With
  equal rates both fall to 0.002 to 0.004, the EV of the session without
  reactivation, and the test calls neither window.
- **Real reactivation survives.** EV stays at 0.84 to 0.88 in the
  reactivated session and 0.69 to 0.74 when the rates also change. It is
  lower than with every spike because the subsample also removes spikes of
  the assembly events after the task, which raise the members' rates too.
  The control is conservative: it can lower a genuine effect.

On your own counts, draw the subsamples from one generator and take the
median of the statistic:

<!-- docs-test: run -->
```python
rate_session = sessions["rate_change"]


def counts_of(data, epochs, bin_size=0.1):
    return bin_spikes_in_time(data.spike_times, dt=bin_size, epochs=epochs)[0]


template_correlations = pairwise_correlations(
    counts_of(rate_session, rate_session.template_epochs)
)
before = counts_of(rate_session, rate_session.control_epochs)
after = counts_of(rate_session, rate_session.match_epochs)
members = list(rate_session.assemblies[0])
print("member spikes before:   ", before[:, members].sum(axis=0))
print("member spikes after:    ", after[:, members].sum(axis=0))

generator = np.random.default_rng(0)
explained = []
for _ in range(10):
    equal_before, equal_after = equalize_firing_rates(before, after, rng=generator)
    explained.append(
        explained_variance_reactivation(
            template_correlations,
            pairwise_correlations(equal_after),
            control_correlations=pairwise_correlations(equal_before),
        ).explained_variance
    )
print("after, equal rates:     ", equal_after[:, members].sum(axis=0))
print(f"median EV over 10 subsamples: {np.median(explained):.4f}")
```

The members of the first assembly fired about 2,900 spikes each in the 20
minutes before the task and about 5,700 after it. After subsampling each
fires exactly as many spikes after as before, and EV falls to 0.001. Bin
both periods with the same bin size: the rate being equalized is spikes per
bin. This uses every pair and all 20 minutes, so the numbers differ a
little from Kudrimoti's recipe, which keeps pairs on different tetrodes and
cuts rest after into 10-minute windows.

### Larger time bins

The second control bins the same spikes in much larger bins. Kudrimoti's
analysis at 0.1 to 5 s:

<!-- docs-test: run -->
```python
wide = [
    dataclasses.replace(KUDRIMOTI_1999, name=f"{bin_size:g} s", bin_size=bin_size)
    for bin_size in (0.1, 0.25, 1.0, 5.0)
]
names = ["reactivated", "no_reactivation", "rate_change"]
wide_table = compare_sessions(wide, names)
print("EV:")
print(by_session(wide_table).round(3).to_string())
print("windows called reactivated (of 2):")
print(by_session(wide_table, "is_reactivated").mul(2).astype(int).to_string())
```

At 5 s the rate-change session is no longer called reactivated (EV 0.006),
while the reactivated session still is (EV 0.49). The large bin did not
remove the confound. Over six simulated sessions, the members' pair
correlations after the task exceed those of the other cells by 0.09 at 1 s
and still by 0.07 at 5 s: the bursts that carry the rate change last
100 ms, and any bin wider than a burst counts all of it. What changes is
the noise. Kudrimoti's 10-minute windows hold only 120 bins of 5 s, so each
pair correlation scatters by about 0.09, as much as the effect, and EV can no
longer see it. The real reactivation loses half its EV for the same
reason. The large bin lost the power to see the confound, and most of the
power to see reactivation with it.

What a large bin does measure is co-firing on slow timescales. If EV stays
as high at seconds as at 100 ms, slow co-fluctuations of rates, such as
cells whose rates drift together over sleep, may carry it rather than the
fast co-firing of SWRs and theta sequences. It does not correct a
difference in rate between the two rest periods; equal rates do.

## How much does the time bin matter?

Published reactivation analyses use bins of 25 to 250 ms; Kudrimoti et al.
found "comparable results" at 50 and 200 ms and Giri et al. "highly similar
and consistent results" from 50 to 500 ms. Tingley & Peyrache (2020) note
that "a systemic variation of bin size and its impact on pattern detection
is lacking". Here are Kudrimoti's EV and Peyrache's assembly difference
(without the cell-identity shuffles, for speed) at 50, 100 and 250 ms on
three sessions:

<!-- docs-test: run -->
```python
bin_sizes = (0.05, 0.1, 0.25)
by_bin = [
    dataclasses.replace(KUDRIMOTI_1999, name=f"EV {b * 1000:g} ms", bin_size=b)
    for b in bin_sizes
] + [
    dataclasses.replace(
        PEYRACHE_2010, name=f"assembly {b * 1000:g} ms", bin_size=b, test=None
    )
    for b in bin_sizes
]
print(by_session(compare_sessions(by_bin, names)).round(3).T.to_string())
```

- **With reactivation**, both measures find it at every bin size. EV falls
  slightly as the bin grows (0.97, 0.95, 0.91) while the assembly
  difference stays near 1.5. The assembly events last about 10 ms, and the
  100 ms bursts, which every pair shares, make up more of each bin's
  co-firing as the bin grows: the mean pair correlation of rest rises from
  0.05 to 0.06 at 50 ms to about 0.10 at 250 ms.
- **Without reactivation**, both stay near 0 at every bin size.
- **The rate confound grows with the bin for assemblies** (0.17, 0.23,
  0.28): 50 ms bins split the 100 ms bursts that carry it, and 250 ms bins
  count them whole. For EV it stays at 0.11 to 0.12.

Which bin is right depends on the timescale of the co-firing you are
after. Report the bin, and check that the conclusion holds at its
neighbors, as Kudrimoti et al. and Giri et al. did.

## Following reactivation in time

EV summarizes a whole period. The reactivation strength of Tingley &
Peyrache (2020) follows it bin by bin: each bin's z-scored population
vector is projected on the task's correlation matrix, with the diagonal set
to 0 so that a cell firing alone adds nothing. With an assembly pattern in
place of the whole matrix this is `assembly_activation`; with the whole
matrix ("in the simplest case, the correlation matrix") it is
`pairwise_activation`. Its mean over a period is the sum, over pairs of
cells, of the products of their correlations in the task and in that
period.

<!-- docs-test: run -->
```python
for name in names:
    data = sessions[name]
    template_correlations = pairwise_correlations(
        counts_of(data, data.template_epochs)
    )
    before = pairwise_activation(
        counts_of(data, data.control_epochs), template_correlations
    )
    after = pairwise_activation(
        counts_of(data, data.match_epochs), template_correlations
    )
    above = np.mean(after > np.percentile(before, 99))
    print(
        f"{name:16s} mean before {before.mean():5.2f}, after {after.mean():5.2f}; "
        f"{above:.1%} of bins after above the 99th percentile before"
    )
```

In the reactivated session the mean rises from about 2.3 before the task to
13 after it, and 5.7% of the bins after it exceed the 99th percentile of
those before, against 1.0% without reactivation. The 2.3 before the task
comes from the population bursts, which correlate every pair in both rest
periods. The measure sums over all pairs, weighted by their correlations
in the task, so it grows with the number of cells and keeps every noisy
pair; assembly patterns keep only the components above chance. The rate
change raises it too, from 2.35 to 3.95, so the rate controls above apply
to it as well.

## How many assemblies?

Assembly methods keep the principal components whose eigenvalue exceeds a
threshold. The default is the Marchenko-Pastur edge for independent units.
`detect_assemblies(component_threshold=...)` also takes the Tracy-Widom
correction (Peyrache et al. 2010) and surrogate thresholds that shuffle or
rotate each unit's counts in time (Lopes-dos-Santos et al. 2013). Rotation
keeps each unit's autocorrelation, so slow rate drifts, which spread the
eigenvalues beyond the Marchenko-Pastur edge, do not count as assemblies.
The paper gives no single percentile for the surrogates (it used the 95th
and the 99th), so you choose it.

<!-- docs-test: run -->
```python
counts, _ = bin_spikes_in_time(
    session.spike_times, dt=0.025, epochs=session.template_epochs
)
for threshold, options in (
    ("marchenko_pastur", {}),
    ("tracy_widom", {}),
    ("circular_shift", {"percentile": 95.0, "n_shuffles": 200}),
):
    result = detect_assemblies(counts, component_threshold=threshold, rng=0, **options)
    print(f"{threshold:17s} threshold {result.threshold:.4f}: {result.n_significant}")
print("largest eigenvalues:", result.eigenvalues[:5].round(2))
```

The three eigenvalues near 2.4 are far above every threshold, and the next
ones near 1.0 are below all of them: here the choice does not matter. It
matters when an eigenvalue is close to the edge, in short recordings or with
slowly drifting rates.

## Methods that use events

Several labs measure reactivation in sharp-wave ripples (SWRs) rather than
across all of rest, and some bin the task by theta cycle rather than by a
fixed time bin. Their recipes count the spikes of each event as one bin
(`rest_bins="events"`, `template_bins="events"`), or follow each assembly's
activation strength R(t) around SWR onsets (`event_response`):

| Recipe | Task | Rest | Statistic | Within-session test |
|---|---|---|---|---|
| `OLIVA_2020_EV` | 100 ms bins | one bin per SWR | EV − REV | none (across mice) |
| `EL_GABY_2021` | 25 ms bins; ICA members above mean + 2 SD | one bin per SWR, at least 200 | mean correlation of member pairs, after − before | none |
| `HARVEY_2023_ASSEMBLY` | 25 ms bins; ICA, Otsu members | spikes smoothed by 7.2 ms, every 2 ms; first NREM hour | peak of the mean R(t) 0–50 ms after SWR onsets, after − before | none |
| `LIU_2023_EV` | one bin per theta cycle | one bin per SWR | EV | none (across sessions) |
| `LIU_2023_PAIRWISE` | one bin per theta cycle | one bin per SWR | semipartial correlation | none |
| `LIU_2023_ASSEMBLY` | 25 ms bins; members above mean + 2 SD | 25 ms bins; last and first NREM hour | mean R(t) 0–100 ms after SWR onsets, after − before | none (across assemblies) |
| `CHANG_2025_ASSEMBLY` | 50 ms bins; members above mean + 3 SD | 50 ms bins every 10 ms | (after − before) / before of the 0–100 ms response | signed-rank of each SWR's response against −500 to −100 ms |
| `ROBINSON_2026_PAIRWISE` | 100 ms bins | one bin per SWR | partial correlation | none (hierarchical bootstrap) |
| `ROBINSON_2026_ASSEMBLY` | 50 ms bins, each region and across regions; members above 1/√N | 50 ms bins every 10 ms | mean R(t) 0–100 ms after SWR onsets, after − before; share of SWRs with z > 1.96 | none |

The assembly recipes that follow R(t) around SWRs z-score each unit over
rest before and after together (`zscore_over="control_and_match"`), as the
labs' code does. `HARVEY_2023_ASSEMBLY` and `OLIVA_2020_ASSEMBLY` z-score
over the whole session (`"all_periods"`), and `PEYRACHE_2010` and
`VAN_DE_VEN_2016` each period on its own.

`compare_reactivation_methods` takes the events as `(start, stop)` rows:
`control_events` and `match_events` for the SWRs of rest before and after,
and `template_events` for the theta cycles of the task. A period uses the
events that start inside its window. Events may touch but not overlap. A
recipe's `needs_events` names the arguments it uses. Here the simulated
bursts stand in for SWRs, and the task is cut into 125 ms theta cycles. The
simulated rest periods last 20 minutes, shorter than the hour of NREM that
Harvey et al. and Liu et al. used; as in their code, a shorter sleep is used
whole (`ReactivationWindows(allow_shorter=True)`).

<!-- docs-test: run -->
```python
start, stop = session.template_epochs[0]
edges = np.arange(start, stop + 1e-9, 0.125)
theta_cycles = np.column_stack([edges[:-1], edges[1:]])
print("SWRs before:", len(session.control_events), "after:", len(session.match_events))

event_methods = [
    method for method in PUBLISHED_REACTIVATION_METHODS if method.needs_events
]


def compare_events(data, methods, seed):
    return compare_reactivation_methods(
        data.spike_times,
        data.template_epochs,
        data.match_epochs,
        data.control_epochs,
        reactivation_methods=methods,
        template_events=theta_cycles,
        control_events=data.control_events,
        match_events=data.match_events,
        unit_groups=unit_groups,
        rng=seed,
    )


events_table = compare_events(session, event_methods, seed=0)
pairwise = events_table[events_table["component"].isna()]
columns = ["method", "value", "explained_variance", "reversed_explained_variance"]
print(pairwise[[*columns, "semipartial_correlation"]].round(3).to_string(index=False))
```

The four pairwise recipes find the task's pair correlations again in the
SWRs after it. EV is 0.66 whether the task is binned by 100 ms or by theta
cycle, lower than the 0.92 to 0.98 of the time-binned recipes: only the
assembly events that fall inside a burst (about 1 in 20) enter the SWR
counts. Robinson et al.'s residual correlation is the partial correlation
whose square is EV, 0.815. Liu et al.'s semipartial correlation regresses
rest before out of rest after only; it equals the partial correlation here
because rest before holds no trace of the task's correlations.

<!-- docs-test: run -->
```python
patterns = events_table[events_table["component"].notna()]
columns = [
    "method",
    "region",
    "member_unit_ids",
    "value",
    "control_member_correlation",
    "match_member_correlation",
    "control_event_response",
    "match_event_response",
    "p_value",
]
print(patterns[columns].round(3).to_string(index=False))
```

- **Member co-firing (El-Gaby et al. 2021).** The members of each assembly
  correlate at about 0.03 over the SWRs before the task and 0.24 to 0.29
  over those after it.
- **Responses to SWRs.** Harvey's and Liu's recipes find R(t) after SWR
  onsets about 3 higher after the task than before it. Chang et al. divide
  the difference by the response before the task: a value of 3 is a
  response four times as large. Their signed-rank test of each SWR's
  response against its baseline is significant for one assembly of three,
  since only a few SWRs hold an assembly event.
- **Members.** Chang's rule (weights above the mean by 3 SD) names no
  members here: with 5 equal weights among 30 units the members' z-score is
  √(25 / 5) ≈ 2.2. The members do not enter R(t), so the responses are
  unaffected; with few units, lower `member_threshold`.
- **Regions (Robinson et al. 2026).** Assemblies detected in each region
  hold that region's part of each simulated assembly; the cross-region
  assemblies, detected on all units, keep those with members in both
  regions.

### Event lengths

A count per event depends on the event's length: every unit fires more in a
longer SWR. The counts of every pair then share a length component that
raises all pair correlations alike. Here the simulated bursts are 100 ms
long, except where two overlapped; giving the same events random lengths
from 20 to 300 ms raises the mean pair correlation over them:

<!-- docs-test: run -->
```python
def counts_in(events):
    return np.column_stack(
        [
            np.searchsorted(train, events[:, 1]) - np.searchsorted(train, events[:, 0])
            for train in session.spike_times
        ]
    ).astype(float)


swrs = session.control_events
lengths = np.random.default_rng(0).uniform(0.02, 0.3, len(swrs))
stretched = np.column_stack([swrs[:, 0], swrs[:, 0] + lengths])
for name, events in (("100 ms bursts", swrs), ("20 to 300 ms", stretched)):
    mean_correlation = np.nanmean(pairwise_correlations(counts_in(events)))
    print(f"{name}: mean pair correlation {mean_correlation:.3f}")
```

The component cancels between rest before and after only when their SWRs
have similar lengths. Compare the length distributions of the two periods'
SWRs before comparing their correlations.

### The rate confound around SWRs

The simulated session in which the assembly members fire twice as much
after the task, without firing together, shows how the event-based recipes
respond to a change in rate (Harvey's recipe behaves as Liu's and is left
out to keep this short):

<!-- docs-test: run -->
```python
quick = [method for method in event_methods if method.name != "harvey_2023_assembly"]
rows = []
for name, options in {
    "no_reactivation": {"reactivation": 0.0},
    "rate_change": {"reactivation": 0.0, "rate_change": 1.0},
}.items():
    other = simulated_reactivation_session(n_units=30, duration=1200.0, rng=2, **options)
    rows.append(compare_events(other, quick, seed=2).assign(data=name))
by_data = pd.concat(rows).groupby(["method", "data"], sort=False)["value"].mean()
print(by_data.unstack().round(3).to_string())
```

Without reactivation every statistic is near 0. Doubling the members'
rates moves the member correlations of El-Gaby et al. by 0.02, EV by less
than 0.01 and the residual correlations by about 0.08, but raises the SWR
responses of the assembly recipes by about 5, more than the reactivation in
the first session did. With rest before and after z-scored together,
members that fire more after the task have higher z-scores in every SWR
after it, and since the bursts multiply every unit's rate, they deviate
together. Z-scoring each period on its own (`zscore_over="period"`) does not
remove the confound either: the members' larger variance after the task
then shrinks their z-scores in SWRs, and the response falls instead. Before
reading an SWR response as reactivation, compare the members' firing rates
in the two rest periods.

## Using your own recording

- **Epochs.** Pass the task as `template_epochs` (for example running, or
  its theta periods) and the rest periods as `control_epochs` and
  `match_epochs`, restricted to the state the paper used: slow-wave sleep
  for Kudrimoti and Peyrache, quiet rest and sleep for Lansink. Epochs are
  `(n, 2)` arrays of start and stop times in seconds on the spike clock, or
  a pynapple `IntervalSet`. Windows of a recipe's `ReactivationWindows`
  count time inside the epochs (`clock="concatenated"`) or on the clock
  (`clock="wall"`), as the recipe's notes explain.
- **Events.** Pass SWRs as `control_events` and `match_events` and theta
  cycles as `template_events`, `(n, 2)` arrays of start and stop times on
  the spike clock. Merge or drop overlapping events first. Restrict them to
  the state each paper used (NREM for Harvey, Liu and Chang); El-Gaby et
  al. used every SWR of a sleep session and required at least 200.
- **Units.** Use the units recorded and stable in all three periods; the
  papers' inclusion criteria (pyramidal cells, stability, minimum counts)
  are not applied by the recipes. A dict of unit ID to spike train, a
  `SpikeTrains` or a pynapple `TsGroup` names assembly members by its own
  IDs; for a list of trains, pass `unit_ids=` with your sorter's labels.
- **Pairs.** Give each unit's tetrode, shank or region in `unit_groups=`.
  Pairs on the same tetrode can share spike-sorting errors. For Giri et al.
  (2024), pass the pairs with waveform similarity below 0.8 as
  `pair_mask=`, in `np.triu_indices(n_units, k=1)` order.
- **Rates.** Compare each unit's firing rate in the two rest periods. If
  they differ, also run the EV recipes with equal rates,
  `dataclasses.replace(method, name=..., n_rate_subsamples=100)`, or draw
  the subsamples yourself with `equalize_firing_rates`, and report both.
  Event-binned rest (`rest_bins="events"`) is refused: spikes per event and
  spikes per second disagree when events differ in length.
- **Shuffles.** Keep the published `n_shuffles`, give `rng` a fixed seed and
  report `table.attrs["rng_entropy"]`.
- **Sessions.** Most papers tested across sessions (Wilcoxon signed-rank of
  EV against REV) or across patterns (t test of the differences against 0).
  Run each session in its own call, add a `session` column and concatenate
  the tables; `reactivation_detection_rates(table, groups=["session"])`
  summarizes them.
- **Your own variant.** `dataclasses.replace(KUDRIMOTI_1999, name=...,
  bin_size=0.2)` changes one choice and keeps the rest; give it a new name.

## References

- Chang H, Tang W, Wulf AM, Nyasulu T, Wolf ME, Fernandez-Ruiz A, Oliva A
  (2025). Sleep microstructure organizes memory replay. *Nature*
  637:1161–1169.
- El-Gaby M, Reeve HM, Lopes-dos-Santos V, Campo-Urriza N, Perestenko PV,
  Morley A, Strickland LAM, Lukács IP, Paulsen O, Dupret D (2021). An
  emergent neural coactivity code for dynamic memory. *Nat Neurosci*
  24:694–704.
- Giri B, Miyawaki H, Mizuseki K, Cheng S, Diba K (2019). Hippocampal
  reactivation extends for several hours following novel experience.
  *J Neurosci* 39:866–875.
- Giri B, Kinsky N, Kaya U, Maboudi K, Abel T, Diba K (2024). Sleep loss
  diminishes hippocampal reactivation and replay. *Nature* 630:935–942.
- Harvey RE, Robinson HL, Liu C, Oliva A, Fernandez-Ruiz A (2023).
  Hippocampo-cortical circuits for selective memory encoding, routing, and
  replay. *Neuron* 111:2076–2090.
- Kudrimoti HS, Barnes CA, McNaughton BL (1999). Reactivation of
  hippocampal cell assemblies: effects of behavioral state, experience, and
  EEG dynamics. *J Neurosci* 19:4090–4101.
- Lansink CS, Goltstein PM, Lankelma JV, McNaughton BL, Pennartz CMA (2009).
  Hippocampus leads ventral striatum in replay of place-reward information.
  *PLoS Biol* 7:e1000173.
- Liu C, Todorova R, Tang W, Oliva A, Fernandez-Ruiz A (2023).
  Associative and predictive hippocampal codes support memory-guided
  behaviors. *Science* 382:eadi8237.
- Lopes-dos-Santos V, Ribeiro S, Tort ABL (2013). Detecting cell assemblies
  in large neuronal populations. *J Neurosci Methods* 220:149–166.
- Oliva A, Fernández-Ruiz A, Leroy F, Siegelbaum SA (2020). Hippocampal CA2
  sharp-wave ripples reactivate and promote social memory. *Nature*
  587:264–269.
- Pennartz CMA, Lee E, Verheul J, Lipa P, Barnes CA, McNaughton BL (2004).
  The ventral striatum in off-line processing: ensemble reactivation during
  sleep and modulation by hippocampal ripples. *J Neurosci* 24:6446–6456.
- Peyrache A, Benchenane K, Khamassi M, Wiener SI, Battaglia FP (2010).
  Principal component analysis of ensemble recordings reveals cell
  assemblies at high temporal resolution. *J Comput Neurosci* 29:309–325.
- Robinson HL, Todorova R, Nagy GA, Gruzdeva A, Paudel P, Oliva A,
  Fernandez-Ruiz A (2026). Large sharp-wave ripples promote
  hippocampo-cortical memory reactivation and consolidation during sleep.
  *Neuron* 114:226–236 (read from bioRxiv preprint v2).
- Tatsuno M, Lipa P, McNaughton BL (2006). Methodological considerations on
  the use of template matching to study long-lasting memory trace replay.
  *J Neurosci* 26:10727–10742.
- Tingley D, Peyrache A (2020). On the methods for reactivation and replay
  analysis. *Phil Trans R Soc B* 375:20190231.
- van de Ven GM, Trouche S, McNamara CG, Allen K, Dupret D (2016).
  Hippocampal offline reactivation consolidates recently formed cell
  assembly patterns during sharp wave-ripples. *Neuron* 92:968–974.
