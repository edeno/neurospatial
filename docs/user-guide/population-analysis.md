# Population Analysis

A hippocampal recording gives you tens to hundreds of units at once, and many
questions are about the population rather than one cell. How much of the track
do the place fields cover? Does the spatial map change when the cues change?
Which cells fire together, and do they fire together again during rest after
the task? What position does the population encode at each moment?

This guide answers each question on a small simulated recording in which we
planted the answer, so you can see what each number means before you compute
it on your own data. Every example runs in a few seconds, and the code blocks
build on each other, so run them in order.

## Where these live

Population tools are in three subpackages. Spatial-map summaries are with the
encoding tools. Co-firing and reactivation work on spike counts in time bins
and are in `neurospatial.reactivation`; decoding is in `neurospatial.decoding`,
which also bins the spike counts.

| Question | Functions | Import from |
| --- | --- | --- |
| Rate maps for every unit | `compute_spatial_rates` | `neurospatial` or `neurospatial.encoding` |
| How much of the environment is covered, and by how many place cells? | `population_coverage`, `plot_population_coverage`, `field_density_map`, `count_place_cells` | `neurospatial.encoding` (defined in `neurospatial.encoding.population`) |
| Does the map change between conditions? | `population_vector_correlation`, `rate_map_correlations`, `field_overlap` | `neurospatial.encoding` (defined in `neurospatial.encoding.population`) |
| Spike counts in time bins | `bin_spikes_in_time` | `neurospatial.decoding` |
| Which cells fire together, and does that pattern return during rest? | `pairwise_correlations`, `detect_assemblies`, `marchenko_pastur_threshold`, `assembly_activation`, `assembly_reactivation`, `explained_variance_reactivation` | `neurospatial.reactivation` (defined in `neurospatial.reactivation.assemblies`) |
| What position does the population encode? | `decode_position`, `decode_session`, `BayesianDecoder` | `neurospatial.decoding` |

## How much of the track does the population cover?

Coverage asks whether the place fields of the recorded population tile the
environment or leave gaps. Before counting fields, decide which units are
spatially tuned: a field detector finds peaks in any rate map, including the
map of a cell that fires the same everywhere.

We simulate two minutes of running on a 200 cm linear track. Twenty units are
place cells; five more fire at a steady 2 Hz wherever the animal is, like
untuned cells. We compute all 25 rate maps in one call with
5 cm smoothing, and leave bins with less than 0.05 s of occupancy as NaN
(unvisited, not 0 Hz).

```python
import matplotlib.pyplot as plt
import numpy as np

from neurospatial import compute_spatial_rates
from neurospatial.encoding import (
    count_place_cells,
    field_density_map,
    plot_population_coverage,
    population_coverage,
)
from neurospatial.simulation import generate_poisson_spikes, linear_track_session

session = linear_track_session(duration=120.0, n_place_cells=20, n_laps=10, seed=0)
env, times, positions = session.env, session.times, session.positions
untuned = [
    generate_poisson_spikes(np.full(len(times), 2.0), times, seed=100 + i)
    for i in range(5)
]
spike_times = [*session.spike_times, *untuned]  # units 0-19 tuned, 20-24 untuned
unit_ids = np.arange(len(spike_times))

rates = compute_spatial_rates(
    env, spike_times, times, positions,
    unit_ids=unit_ids, bandwidth=5.0, min_occupancy=0.05,
)
information = rates.spatial_information()  # bits/spike, one value per unit
bias = rates.spatial_information_bias()  # expected for an untuned unit
n_spatial = count_place_cells(rates, min_info=0.5)
print(f"Units at or above 0.5 bits/spike beyond chance: {n_spatial} of {len(unit_ids)}")
print("Information of the untuned units:", np.round(information[20:], 2))
print("Expected for untuned units:      ", np.round(bias[20:], 2))
# Quoted numbers: atol = 0.6 * 10**-decimals
assert n_spatial == 20
assert 0.004 <= information[20:].min() <= information[20:].max() <= 0.046
```

All 20 simulated place cells pass the 0.5 bits/spike screen and none of the
untuned units do. The untuned units show 0.01-0.04 bits/spike, about what an
untuned unit with their spike counts is expected to show from Poisson noise
alone (`spatial_information_bias()`, 0.03-0.04 here). The screen subtracts that
expectation before it applies the cutoff, so short or sparse recordings do not
pass untuned cells (see
[information bias and cell verdicts](neuroscience-metrics.md#information-bias-and-cell-verdicts)).
This is a screen, not a verdict.
For a per-cell decision you can report, use the shuffle labels of
`spatial_cell_type_significance` on the raw spike times and tracking (they
test grid and border scores as well), or `is_place_cell(...,
criterion="shuffle")` for one cell. Both call a cell place only when its
information test is significant and its bias-corrected information reaches the
screen's cutoff; `place_cell_significance` gives the information test's
p-values alone.

Now compare coverage computed from every map with coverage computed from the
screened units only:

```python
coverage_all = population_coverage(env, rates.firing_rates, bandwidth=rates.bandwidth)
is_spatial = rates.classify(min_info=0.5)  # the screen count_place_cells counts
coverage = population_coverage(
    env, rates.firing_rates[is_spatial], bandwidth=rates.bandwidth
)
print(f"All 25 units: {coverage_all.coverage_fraction:.0%} of bins covered, "
      f"{coverage_all.n_fields} fields")
print(f"20 screened units: {coverage.coverage_fraction:.0%} of bins covered, "
      f"{coverage.n_fields} fields, {coverage.fields_per_place_cell:.1f} per cell")
gap_positions = coverage.uncovered_positions[:, 0]
print("Uncovered positions, rounded to 10 cm:",
      np.unique(np.round(gap_positions, -1)))
print("Uncovered bins the animal never visited:",
      int((~rates.spatial_support[coverage.uncovered_bins]).sum()))
assert np.isclose(coverage.coverage_fraction, 1.0, atol=0.006)
assert np.isclose(coverage.fields_per_place_cell, 1.0, atol=0.06)
```

Both sets cover the whole track, but the 25 maps give 25 fields: the detector
also found fields in the flat maps of the untuned units. Screen first, so
that noise neither adds fields nor fills gaps. The 20 screened place cells
cover every bin with one field each, because the simulated fields tile the
track from end to end. In your own recording a gap is either a location
without a field or a location without data; check `rates.spatial_support` at
`coverage.uncovered_bins` as above, because coverage counts an unvisited bin
as a gap.

`population_coverage` detects fields with `detect_place_fields` (field edge at
20% of the peak rate by default, set with `threshold=`). Pass the smoothing
`bandwidth` the maps were estimated with: a field must be larger than what one
noisy bin becomes after that smoothing. If you detect fields
yourself, for example with a different threshold, `field_density_map` counts
how many fields include each bin:

```python
density = field_density_map(coverage.place_fields, env.n_bins)
covered = density[coverage.covered_bins]
print(f"Fields per covered bin: median {np.median(covered):.0f}, max {covered.max()}")
assert covered.max() == 7

fig, ax = plt.subplots(figsize=(8, 2.5))
plot_population_coverage(env, coverage, show_field_count=True, ax=ax)
plt.show()
```

`plot_population_coverage` draws the same count from the coverage result:
how many place fields include each position, here six at a typical position
and up to seven. Dashed lines mark the uncovered bins, and the
title gives the coverage fraction. A location covered by several fields is
represented redundantly; a location covered by none is decoded poorly, if at
all. For an open-field arena, the same call draws a map and marks
each gap with a cross; leave out `show_field_count=True` to show covered
versus uncovered bins only.

## Does the spatial map change between conditions?

Remapping experiments compare the same cells in two conditions: a cue
rotation, a new room, or the same room on another day. We simulate one session
on the linear track with a change halfway through. After 120 s, every other
place cell moves its field 80 cm along the track; the rest keep their field.

Use one environment and the same rate-map settings for both conditions,
select each condition with `epochs=` instead of slicing the arrays, and pass
the sorter's labels as `unit_ids=` so each result knows which cell each row
is.

```python
from dataclasses import replace

from neurospatial.encoding import (
    field_overlap,
    population_vector_correlation,
    rate_map_correlations,
)
from neurospatial.simulation import PlaceCellModel

session = linear_track_session(duration=240.0, n_place_cells=16, n_laps=20, seed=1)
env, times, positions = session.env, session.times, session.positions
change_time = 120.0
moves = np.arange(16) % 2 == 1  # odd-numbered units remap
spike_times = []
for unit, cell in enumerate(session.models):
    rate = cell.firing_rate(times, positions)
    if moves[unit]:
        # Move the field 80 cm, wrapping around within 20-180 cm; replace()
        # keeps every other argument of the cell.
        new_center = 20.0 + (cell.ground_truth["center"] - 20.0 + 80.0) % 160.0
        moved = replace(cell, center=new_center)
        rate = np.where(times < change_time, rate, moved.firing_rate(times, positions))
    spike_times.append(generate_poisson_spikes(rate, times, seed=200 + unit))
unit_ids = np.arange(len(spike_times))  # your sorter's unit labels

conditions = {"before": (0.0, change_time), "after": (change_time, float(times[-1]))}
maps = {
    name: compute_spatial_rates(
        env, spike_times, times, positions,
        unit_ids=unit_ids, epochs=window, bandwidth=5.0, min_occupancy=0.05,
    )
    for name, window in conditions.items()
}
# Compare only bins with occupancy in both conditions; the others are NaN.
shared = maps["before"].spatial_support & maps["after"].spatial_support
before = maps["before"].firing_rates[:, shared]
after = maps["after"].firing_rates[:, shared]
print(f"Bins compared: {shared.sum()} of {env.n_bins}")
```

`rate_map_correlations` returns the map-by-map correlation matrix: the
Pearson correlation of every pair of rows in a `(n_maps, n_bins)` matrix.
Stack the two conditions, and the diagonal of the before-versus-after block
is each cell's rate-map correlation across the change. The classic
population-vector correlation of remapping studies works the other way round:
`population_vector_correlation` correlates the vector of all cells' rates at
each location, giving one value per bin. Pass it the two rate results: it
pairs the cells by `unit_ids` and compares only bins with occupancy in both
conditions, so you do not line up rows or select bins yourself. A location
outside that shared support is NaN, as is a location where every cell has the
same rate (no defined correlation); summarize over the defined locations and
report how many were undefined.

```python
n_units = len(spike_times)
map_similarity = rate_map_correlations(np.vstack([before, after]))
per_cell = np.diag(map_similarity[:n_units, n_units:])
print("Map correlation, cells that kept their field:", np.round(per_cell[~moves], 2))
print("Map correlation, cells that moved their field:", np.round(per_cell[moves], 2))
assert per_cell[~moves].min() >= 0.98 - 0.006  # 0.98-1.00
assert -0.486 <= per_cell[moves].min() <= per_cell[moves].max() <= -0.294

# Identical maps with one flat location: the flat one is NaN, the rest are 1.
demo = np.array([[1.0, 2.0, 5.0], [3.0, 2.0, 1.0], [2.0, 2.0, 4.0]])
demo_pv = population_vector_correlation(demo, demo)
assert np.isnan(demo_pv[1]) and np.allclose(demo_pv[[0, 2]], 1.0)

pv = population_vector_correlation(maps["before"], maps["after"])  # (n_bins,)
defined = np.isfinite(pv)
print(f"Population-vector correlation per location: median {np.median(pv[defined]):.2f} "
      f"({(~defined).sum()} of {pv.size} locations undefined)")
assert np.isclose(np.median(pv[defined]), 0.29, atol=0.006)
assert np.array_equal(defined, shared)  # undefined only where a condition lacks data

# Units are paired by label: listing the cells in another order changes nothing.
order = np.random.default_rng(2).permutation(len(spike_times))
after_reordered = compute_spatial_rates(
    env, [spike_times[i] for i in order], times, positions,
    unit_ids=unit_ids[order], epochs=conditions["after"],
    bandwidth=5.0, min_occupancy=0.05,
)
pv_reordered = population_vector_correlation(maps["before"], after_reordered)
assert np.array_equal(pv_reordered, pv, equal_nan=True)
# Bare arrays pair rows by position, so the same reordering pairs different cells.
by_position = population_vector_correlation(
    maps["before"].firing_rates[:, shared], after_reordered.firing_rates[:, shared]
)
print(f"Rows in another order, paired by position: median {np.median(by_position):.2f}")
assert np.isclose(np.median(by_position), -0.1, atol=0.06)
```

The cells that kept their field have map correlations of 0.98-1.00, and the
cells that moved have correlations of -0.48 to -0.30: you can read off which
cells remapped. The median population-vector correlation of 0.29 says the
population code changed substantially even though half the cells kept their
fields, because a moved field lowers the correlation both where it was and
where it went. An unchanged map gives values near 1 and a completely new map
values near 0. Correlations of smoothed maps are high when the fields are
well sampled; with few laps per condition, compare against the correlation
between two halves of the same condition before calling a change remapping.

Reordering the units of one condition gives the same values, because the
results pair cells by label. The same arrays paired by row position correlate
different cells, and the median drops to about -0.1 although nothing about
the cells changed. If you pass arrays (for example matrices from another
tool), lining up the rows is your job. With results, units recorded in only
one condition are left out by default; `unit_selection="exact"` refuses them
instead. Maps estimated on different environments are refused unless you pass
an explicit `bin_mapping=` that pairs bins at the same place; for a subset of the
arena, `part.bin_mapping_from(env).pairs` builds it (see
[Carry values to a subset or a coarser grid](environments.md#bin-mapping)).

`field_overlap` asks the same question with field boundaries instead of rates:
the Jaccard index of the area inside a cell's fields in the two conditions
(1 = same bins, 0 = no shared bin). Only bins with occupancy in both
conditions (`shared`) count, because a bin one condition never visited cannot
show whether a field is there. A cell with no field bin on shared bins in
either condition has no defined overlap and gets NaN.

```python
fields_before = population_coverage(
    env, maps["before"].firing_rates, bandwidth=maps["before"].bandwidth
).place_fields
fields_after = population_coverage(
    env, maps["after"].firing_rates, bandwidth=maps["after"].bandwidth
).place_fields


def field_bins(fields):
    """All bins inside any of one cell's fields."""
    return np.concatenate(list(fields)) if len(fields) else np.empty(0, dtype=np.int64)


def shared_field_overlap(a, b, shared):
    """Jaccard overlap of two cells' fields, counting only bins in `shared`.

    `field_overlap` returns NaN when neither condition has a field bin left.
    """
    bins_a = np.intersect1d(field_bins(a), np.flatnonzero(shared))
    bins_b = np.intersect1d(field_bins(b), np.flatnonzero(shared))
    return field_overlap(env, bins_a, bins_b)


# Identical fields, but the second condition visited fewer bins: overlap is 1.
demo_shared = np.array([True, True, True, False, False, False])
assert shared_field_overlap([np.array([0, 1, 2, 3])], [np.array([0, 1, 2])], demo_shared) == 1.0

overlap = np.array([
    shared_field_overlap(a, b, shared)
    for a, b in zip(fields_before, fields_after, strict=True)
])
print("Field overlap, cells that kept their field:", np.round(overlap[~moves], 2))
print("Field overlap, cells that moved their field:", np.round(overlap[moves], 2))
assert 0.834 <= overlap[~moves].min() <= overlap[~moves].max() <= 1.0 + 1e-9  # Jaccard is at most 1
assert (overlap[moves] == 0).all()
```

Cells that kept their field share most of their field bins (overlap
0.84-0.99), and cells that moved share none. Neither per-cell measure detects
a cell that keeps its field but changes its peak rate (rate remapping): field
overlap ignores rates, and a correlation ignores a change of scale. Compare
in-field rates for that; the per-location `population_vector_correlation` does
drop when cells change their rates relative to each other.

## Which cells fire together, and does that pattern return during rest?

Cells that fire together during a task can fire together again during rest or
sleep afterwards, a signature of memory reactivation. These analyses work on
spike counts in short time bins rather than on rate maps.

We simulate a session with three periods on one clock: 60 s of rest before
the task, 120 s of running on the track, and 60 s of rest after. Units 0-11
form three assemblies of four cells; the cells of an assembly share a place
field, so they fire together during running. Units 12-27 fire at a steady
1 Hz and belong to no assembly. During both rest periods each assembly cell
fires 30 short bursts (30 Hz for 100 ms) on top of a 1 Hz background. Before
the task the bursts are independent; after the task the four cells of an
assembly burst together, a stand-in for replay. Both rest periods therefore
have the same firing rates and differ only in which cells fire together.

```python
from neurospatial.decoding import bin_spikes_in_time

rng = np.random.default_rng(1)  # rest burst times; spikes use their own seeds
track = linear_track_session(duration=120.0, n_place_cells=1, n_laps=10, seed=3)
env, positions = track.env, track.positions
times = track.times + 60.0  # running occupies 60-180 s of the session clock
periods = {"pre_rest": (0.0, 60.0), "run": (60.0, 180.0), "post_rest": (180.0, 240.0)}

assembly_of = np.r_[np.repeat([0, 1, 2], 4), np.full(16, -1)]  # -1: no assembly
field_centers = np.array([40.0, 100.0, 160.0])
n_units = assembly_of.size
unit_ids = 100 + np.arange(n_units)  # your sorter's unit labels
rest_clock = np.arange(0.0, 60.0, 0.002)
replay_starts = rng.uniform(0.0, 59.9, size=(3, 30))  # seconds into post-task rest


def rest_rates(coordinated):
    """Firing rate (Hz) of every unit on the rest clock."""
    rest = np.ones((n_units, rest_clock.size))
    for unit, assembly in enumerate(assembly_of):
        if assembly < 0:
            continue
        starts = replay_starts[assembly] if coordinated else rng.uniform(0.0, 59.9, 30)
        for start in starts:
            rest[unit, (rest_clock >= start) & (rest_clock < start + 0.1)] += 30.0
    return rest


pre_rates, post_rates = rest_rates(coordinated=False), rest_rates(coordinated=True)
spike_times = []
for unit, assembly in enumerate(assembly_of):
    if assembly >= 0:
        cell = PlaceCellModel(
            env, center=field_centers[[assembly]], width=8.0,
            max_rate=15.0, baseline_rate=0.5,
        )
        run_rate = cell.firing_rate(times, positions)
    else:
        run_rate = np.ones(len(times))
    spike_times.append(np.concatenate([
        generate_poisson_spikes(pre_rates[unit], rest_clock, seed=300 + unit),
        generate_poisson_spikes(run_rate, times, seed=400 + unit),
        180.0 + generate_poisson_spikes(post_rates[unit], rest_clock, seed=500 + unit),
    ]))

dt = 0.05  # 50 ms bins
counts, bin_times = {}, {}
for name, (start, stop) in periods.items():
    counts[name], bin_times[name] = bin_spikes_in_time(
        spike_times, dt=dt, t_start=start, t_stop=stop
    )
print({name: matrix.shape for name, matrix in counts.items()})
```

Each count matrix is `(n_time_bins, n_units)`, with the same unit order in
every period; `unit_ids` labels the columns. The bin size sets the time scale
of "together". Published assembly analyses use 25 ms bins (van de Ven et al.,
2016) up to 100 ms (Peyrache et al., 2010); shorter bins catch tighter
synchrony but hold fewer spikes and give noisier correlations. The
Marchenko-Pastur threshold below is exact only for long recordings, so with
few time bins per unit check the number of selected dimensions on counts
shuffled in time.

### Which cells fire together during running?

`pairwise_correlations` returns the correlation of every pair of units, in the
order of `np.triu_indices(n_units, k=1)`. Those are column positions; map them
through `unit_ids` to name the cells:

```python
from neurospatial.reactivation import (
    assembly_activation,
    assembly_reactivation,
    detect_assemblies,
    explained_variance_reactivation,
    marchenko_pastur_threshold,
    pairwise_correlations,
)

run_correlations = pairwise_correlations(counts["run"])
first, second = np.triu_indices(n_units, k=1)
strongest = np.argsort(run_correlations)[::-1][:5]
for pair in strongest:
    print(f"units {unit_ids[first[pair]]} and {unit_ids[second[pair]]}: "
          f"r = {run_correlations[pair]:.2f}")
print(f"Pairs with r > 0.1: {(run_correlations > 0.1).sum()} "
      f"of {run_correlations.size}")
assert ((run_correlations > 0.1).sum(), run_correlations.size) == (18, 378)
```

The most correlated pairs are cells of the same assembly. Only 18 of the 378
pairs exceed r = 0.1, exactly the six pairs within each of the three
assemblies; the other pairs are near 0. Correlations between counts in 50 ms
bins are small even for cells that share a field; what matters is that they
stand out from the rest.

`detect_assemblies` finds groups of co-active cells. It z-scores each unit's
counts, finds the eigenvalues of the correlation matrix, and keeps the
dimensions whose eigenvalues exceed the Marchenko-Pastur threshold: the
largest eigenvalue expected if the units were independent, given the number of
units and time bins. ICA then separates the kept dimensions into patterns of
neuron weights, and units whose weight stands out (absolute weight more than
2 standard deviations above the mean, set with `z_threshold=`) are listed as
core members. Pass the column labels as `unit_ids=`: each pattern then names
its members by label (`member_unit_ids`), and later calls can match the
columns of other periods to the pattern's weights by label.

```python
threshold = marchenko_pastur_threshold(n_units, counts["run"].shape[0])
assemblies = detect_assemblies(counts["run"], unit_ids=unit_ids, rng=0)
print(f"Marchenko-Pastur threshold: {threshold:.2f}")
print("Largest eigenvalues:", np.round(assemblies.eigenvalues[:5], 2))
print("Dimensions above threshold:", assemblies.n_significant)
planted = []
for pattern in assemblies.patterns:
    members = pattern.member_indices  # column positions of the members
    print(f"Core members {pattern.member_unit_ids.tolist()} -> planted assembly "
          f"{np.unique(assembly_of[members]).tolist()}")
    planted.append(np.unique(assembly_of[members]).tolist())
assert assemblies.n_significant == 3
assert all(len(assembly) == 1 for assembly in planted)  # each pattern: one assembly
```

Three eigenvalues exceed the threshold, matching the three planted
assemblies, and each pattern's core members are the four cells of one planted
assembly. The threshold selects dimensions; it is not a p-value for a pattern
or a cell. In real data, cells with overlapping place fields form patterns
too, because they fire together during running. A pattern can also have no
core members at the default cutoff even when its dimension passes: with `k`
equally weighted members among `n` units their z-score is sqrt((n - k) / k),
so 4 members of 28 units reach 2.4, but 3 members of 9 units only 1.4. Pass
`rng=` so ICA gives the same patterns on every run.

### Do the patterns return during rest?

`assembly_activation` measures, in each time bin, how strongly the cells of
a pattern fire together. It z-scores each unit's counts within the period you
pass and computes the activation strength
R(t) = sum over pairs i != j of w_i w_j z_i(t) z_j(t), where w are the
pattern's weights and z(t) the z-scored counts (Lopes-dos-Santos et al., 2013;
Peyrache et al., 2010). Only pairs of cells contribute, so a burst of one
cell alone adds nothing; R(t) is large only when several cells with large
weights fire above their means in the same bin, and it averages 0 for cells
that fire independently. With `unit_ids=`, the columns are matched to the
pattern's weights by label, so counts binned from a differently ordered unit
list still give the right answer, and a unit missing from either side raises
instead of shifting every weight onto the wrong cell. ICA determines a
pattern only up to its sign; R(t) does not depend on the sign, and
`detect_assemblies` orients each pattern so the neurons with the largest
positive weights are the members.

```python
reactivated = []
for k, pattern in enumerate(assemblies.patterns):
    assembly = assembly_of[pattern.member_indices[0]]
    activation = assembly_activation(counts["post_rest"], pattern, unit_ids=unit_ids)
    activation_pre = assembly_activation(counts["pre_rest"], pattern, unit_ids=unit_ids)
    # In this simulation we know when each assembly burst together.
    starts = 180.0 + replay_starts[assembly]
    centers = bin_times["post_rest"][:, None]
    in_event = ((centers >= starts) & (centers < starts + 0.1)).any(axis=1)
    comparison = assembly_reactivation(
        counts["pre_rest"], counts["post_rest"], pattern,
        control_unit_ids=unit_ids, match_unit_ids=unit_ids,
    )
    print(f"Pattern {k}: activation {activation[in_event].mean():.1f} during "
          f"its events vs {activation[~in_event].mean():.2f} otherwise; "
          f"bins above 5: pre {np.mean(activation_pre > 5):.1%}, "
          f"post {np.mean(activation > 5):.1%}; "
          f"mean pre {comparison.control_mean_activation:.2f}, "
          f"post {comparison.match_mean_activation:.2f}, "
          f"difference {comparison.mean_activation_difference:.2f}; "
          f"post bins above the pre 99th percentile: "
          f"{comparison.fraction_above_control_threshold:.1%}")
    reactivated.append((activation[in_event].mean(), activation[~in_event].mean(),
                        comparison.control_mean_activation,
                        comparison.mean_activation_difference,
                        comparison.fraction_above_control_threshold))
in_events, otherwise, pre_mean, difference, above = np.array(reactivated).T
assert np.abs(pre_mean).max() < 0.1  # no co-firing before the task
assert difference.min() > 0.5 and above.min() > 0.02  # co-firing after it
# Reproducibility pins for this seed:
assert 19.9 <= in_events.min() <= in_events.max() <= 23.9
assert np.allclose(otherwise, 0.09, atol=0.02)
assert 1.0 <= difference.min() <= difference.max() <= 1.3
assert 0.039 <= above.min() <= above.max() <= 0.045

# The columns of rest counts in another order give the same activation.
order = np.random.default_rng(3).permutation(n_units)
pattern = assemblies.patterns[0]
reordered = assembly_activation(
    counts["post_rest"][:, order], pattern, unit_ids=unit_ids[order]
)
in_order = assembly_activation(counts["post_rest"], pattern, unit_ids=unit_ids)
assert np.allclose(reordered, in_order)
```

In post-task rest, each pattern's activation strength averages 20 to 24 in
the bins where that assembly burst together and about 0.1 otherwise. Bins
above 5, the cutoff van de Ven et al. (2016) used to call assembly
activations, make up about 4% of post-task rest and under 1% of pre-task
rest. With real data you do not know the replay times; align the activation
to sharp-wave ripples or to rest epochs instead, and compare with pre-task
rest. For independent cells R(t) averages 0 with a standard deviation of at
most about 1.4, but spike counts give heavier tails, so the cutoff of 5 is
descriptive, not a p-value.

`assembly_reactivation` makes that comparison with the two published
statistics. Its `mean_activation_difference` is the pattern's mean activation
strength in post-task rest minus its mean in pre-task rest: van de Ven et al.
(2016) call this difference the pattern's reactivation strength, and Peyrache
et al. (2009) compare the same two averages. A period's mean activation
strength is the pattern's co-firing beyond independent firing, so it stays
near 0 before the task (the bursts were independent), and the difference is
1.0 to 1.3. Its `fraction_above_control_threshold` is the share
of post-task bins whose activation strength exceeds the 99th percentile of the
pre-task bins (Peyrache et al., 2010); without reactivation it is about 1%,
and here it is about 4%. Each period is z-scored on its own, so a change of
gain does not change either statistic, but independent spikes added to the
members dilute their correlation and lower it. Peyrache et al. call R(t)
itself the "reactivation strength"; this guide calls R(t) the activation
strength to keep the two apart. Neither statistic is a p-value: for a claim,
compare across patterns or sessions (van de Ven et al. tested the differences
of a set of patterns against 0) or build a null for your recording, such as
activation strengths after shuffling each unit's counts in time. van de Ven et
al. computed R(t) on spike trains smoothed with a Gaussian kernel of standard
deviation 25 ms / sqrt(12) instead of on binned counts. Each period's column
labels go to their own argument, `control_unit_ids=` for pre-task rest and
`match_unit_ids=` for post-task rest; here both periods were binned from one
list of spike trains, so they get the same labels. Pass both or neither:
labels for only one period raise.

The explained-variance measure asks whether the whole correlation structure of
running returns during rest. It correlates the pairwise correlations of
running with those of post-task rest, controlling for the correlations that
were already present before the task. The reversed measure (REV) swaps the
two rest periods: it measures how much the running correlations resemble
pre-task rest, controlling for post-task rest. Reactivation predicts EV larger
than REV.

Each correlation must be compared with the correlation of the same two cells
in the other periods. Pass each period's column labels (`template_unit_ids=`,
`match_unit_ids=`, `control_unit_ids=`) and the pairs are matched by cell, so
periods sorted or binned with their units in different orders still compare
the right pairs, and a unit missing from one period raises. Without labels the
pairs are matched by position.

```python
rest_correlations = {
    name: pairwise_correlations(counts[name]) for name in ("pre_rest", "post_rest")
}
reactivation = explained_variance_reactivation(
    run_correlations,
    rest_correlations["post_rest"],
    control_correlations=rest_correlations["pre_rest"],
    template_unit_ids=unit_ids,
    match_unit_ids=unit_ids,
    control_unit_ids=unit_ids,
)
print(f"EV = {reactivation.explained_variance:.2f}, "
      f"REV = {reactivation.reversed_explained_variance:.3f}, over {reactivation.n_pairs} pairs")
assert np.isclose(reactivation.explained_variance, 0.74, atol=0.006)
assert np.isclose(reactivation.reversed_explained_variance, 0.011, atol=0.0006)

# Post-task rest with its units in another order gives the same EV.
reordered = explained_variance_reactivation(
    run_correlations,
    pairwise_correlations(counts["post_rest"][:, order]),
    control_correlations=rest_correlations["pre_rest"],
    template_unit_ids=unit_ids,
    match_unit_ids=unit_ids[order],
    control_unit_ids=unit_ids,
)
assert np.isclose(reordered.explained_variance, reactivation.explained_variance)
```

EV (0.74) is far above REV (0.011): the correlation structure of running
reappears after the task and was absent before it, which is what we planted.
EV and REV are effect sizes. To claim reactivation in a real recording, compare against a
null built for your data (for example, EV computed after shuffling unit
identities or using other rest epochs) and report the control period, the bin
size and the unit selection. Without `control_correlations` there is no
period to swap, so `reversed_explained_variance` is NaN and EV alone says
nothing about direction.

## What position does the population encode?

Decoding turns the population's spike counts into an estimate of position in
each time bin, using the rate maps as the encoding model. It uses the same
ingredients as this guide: rate maps from `compute_spatial_rates` and counts
from `bin_spikes_in_time`, in the same unit order.

- [One-call decoding with `decode_session`](workflows.md#workflow-2-bayesian-decoding-one-call)
  fits the maps and decodes the same recording.
- [Held-out decoding](workflows.md#decode-held-out-laps) with
  `BayesianDecoder` fits on training time and scores separate test time;
  report this error, not the in-sample one.
- [Shared units for decoding and population statistics](workflows.md#workflow-6-shared-units-for-decoding-and-population-statistics)
  shows one unit selection feeding both decoding and the assembly analyses.
- The [position decoding example](../examples/06_position_decoding.ipynb)
  walks through posteriors, errors and plots.

## Using your own recording

You need spike times for each unit in seconds, tracking times in seconds,
positions in cm on the same clock, and the start and stop time of each
behavioral period. Build the environment from the tracking or from the
arena outline (see [Environments](environments.md)). The block below runs the
whole guide on the simulated session above; replace its first five
assignments with your data.

```python
# Replace with your recording. Spikes, tracking and periods share one clock.
my_spike_times = spike_times  # list of 1-D arrays of spike times (s), one per unit
my_unit_ids = np.arange(len(my_spike_times))  # your sorter's unit labels
my_times, my_positions = times, positions  # tracking during the task
my_periods = periods  # {"pre_rest": (start, stop), "run": ..., "post_rest": ...}
my_env = env

# Bin every period, then keep units that fire in all of them. Select once, so
# every analysis below uses the same units in the same order.
my_counts = {
    name: bin_spikes_in_time(my_spike_times, dt=0.05, t_start=start, t_stop=stop)[0]
    for name, (start, stop) in my_periods.items()
}
keep = np.logical_and.reduce([m.std(axis=0) > 0 for m in my_counts.values()])
my_unit_ids = my_unit_ids[keep]
my_spike_times = [
    train for train, kept in zip(my_spike_times, keep, strict=True) if kept
]
my_counts = {name: m[:, keep] for name, m in my_counts.items()}

# Rate maps from running only: keep the spikes recorded during the tracking,
# drop slow samples (< 4 cm/s) and keep unvisited bins NaN.
run_start, run_stop = my_periods["run"]
run_spikes = [
    train[(train >= run_start) & (train < run_stop)] for train in my_spike_times
]
my_rates = compute_spatial_rates(
    my_env, run_spikes, my_times, my_positions,
    unit_ids=my_unit_ids, epochs=my_periods["run"],
    bandwidth=5.0, min_occupancy=0.05, min_speed=4.0,
)
spatial = my_rates.classify(min_info=0.5)  # bias-corrected information screen
my_coverage = population_coverage(
    my_env, my_rates.firing_rates[spatial], bandwidth=my_rates.bandwidth
)
my_assemblies = detect_assemblies(my_counts["run"], unit_ids=my_unit_ids, rng=0)
my_reactivation = explained_variance_reactivation(
    pairwise_correlations(my_counts["run"]),
    pairwise_correlations(my_counts["post_rest"]),
    control_correlations=pairwise_correlations(my_counts["pre_rest"]),
    template_unit_ids=my_unit_ids,
    match_unit_ids=my_unit_ids,
    control_unit_ids=my_unit_ids,
)
print(f"{spatial.sum()} of {len(my_unit_ids)} units pass the information screen; "
      f"they cover {my_coverage.coverage_fraction:.0%} of the track")
print(f"{my_assemblies.n_significant} assembly dimensions; "
      f"EV {my_reactivation.explained_variance:.2f}, "
      f"REV {my_reactivation.reversed_explained_variance:.3f}")
```

Checks before you interpret the numbers:

- **Units and clocks.** Positions in cm and times in seconds, with spikes and
  tracking on the same clock. `min_occupancy` is seconds per bin and
  `min_speed` is cm/s.
- **Running versus rest.** Compute rate maps from running periods
  (`min_speed=` or `epochs=`). For rest, choose the epochs that match your
  question, such as sleep, immobility or sharp-wave ripples, and pass them
  through `t_start`/`t_stop` or `epochs=` of `bin_spikes_in_time`.
- **One unit selection.** Remove silent or constant units once across all
  periods and keep `unit_ids` with every result; a different selection per
  period compares different cells. Pass the labels as `unit_ids=` to
  `compute_spatial_rates`, `detect_assemblies` and `assembly_activation`, and
  as `control_unit_ids=` and `match_unit_ids=` to `assembly_reactivation`
  (plus `template_unit_ids=` for `explained_variance_reactivation`), so cells
  are matched by label. Without labels these calls pair cells by column
  position, so every count matrix must share one column order.
- **Screens, not verdicts.** The information cutoff, the Marchenko-Pastur
  threshold, the membership cutoff, activation strength above 5, the share
  of bins above the pre-task 99th percentile and EV versus REV are
  descriptive. Use `spatial_cell_type_significance` labels or
  `is_place_cell(..., criterion="shuffle")` for cell-type decisions and a
  null built for your recording for reactivation claims.
- **Missing is not zero.** Unvisited bins stay NaN in rate maps; compare
  conditions only over bins visited in both, and check whether coverage gaps
  were visited at all.

## See also

- [Neuroscience metrics](neuroscience-metrics.md): single-cell place-field
  metrics, information bias and shuffle tests.
- [Complete workflows](workflows.md): condition comparisons, decoding and a
  combined decoding and assembly pipeline.
- [Spatial analysis](spatial-analysis.md): occupancy and operations on spatial
  fields.
