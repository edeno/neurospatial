# Neuroscience Metrics for Spatial Analysis

This guide covers metrics for spatial firing fields, boundary association and
population representations. The implementations draw on published methods
and adapt some of them to graph-based environments. Agreement with a formula
or successful synthetic tests does not by itself establish equivalence to
another package or identify a biological cell type.

## Overview

The `neurospatial.encoding` package provides three categories of metrics:

1. **Place Field Metrics** - Single-cell spatial tuning analysis
2. **Population Metrics** - Multi-cell spatial representation analysis
3. **Boundary Cell Metrics** - Border cell detection and analysis

Spatial analysis functions take the **environment first**, followed by data.
Non-spatial statistics that operate only on arrays do not need an environment.

## Place Field Detection and Metrics

### Detecting Place Fields

Place fields are localized regions of elevated firing rate that characterize spatial cells. The `detect_place_fields()` function implements an iterative peak-based detection algorithm adapted from neurocode (AyA Lab):

```python
import numpy as np
from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rate, detect_place_fields

# Create environment and compute firing rate map
env = Environment.from_samples(positions, bin_size=2.5)
rates = compute_spatial_rate(env, spike_times, times, positions)

# Detect place fields
fields = detect_place_fields(env, rates.firing_rate,
    bandwidth=rates.bandwidth,  # smoothing of the map: sets the smallest field
    threshold=0.2,              # 20% of each field's peak rate
    max_mean_rate=np.inf,       # Default: no rate-based exclusion
    detect_subfields=True       # Discriminate coalescent fields
)

# Result: iterable PlaceFieldsResult with bin arrays and optional exclusion metadata
# fields[0] = np.array([10, 11, 12, 23, 24, 25])  # Bin indices for field 1
# fields[1] = np.array([45, 46, 56, 57])          # Bin indices for field 2
```

**Algorithm details:**

1. **Iterative peak detection**: Identifies local maxima above threshold
2. **Connected component extraction**: Groups neighboring bins into fields
3. **Optional map-mean filter**: An explicitly finite `max_mean_rate` can skip maps
4. **Subfield discrimination**: FMAToolbox `MapStats`'s rule. The core of a
   field is the region above `m + threshold * (peak - m)`, with
   `m = threshold * peak` (36% of the peak at the default 0.2). When the core
   is less than half the field, two fields share it: the core is kept and the
   neighboring field is found from its own peak

**Key parameters:**

- `threshold`: Fraction of peak rate (default 0.2 = 20%, following Muller & Kubie, 1989)
- `bandwidth`: The smoothing bandwidth the map was estimated with
  (`rates.bandwidth`; 0 for an unsmoothed map). A field must be larger than
  what one noisy bin becomes after that smoothing: with a 5 cm bandwidth and
  the 0.2 threshold, 18 cm on a track and 253 cm² in an open field, whatever
  the bin size. `has_place_field` passes it for you.
- `min_secondary_peak_fraction`: A second field must peak at least this
  fraction of the map's maximum (default 0: every field peaking above
  `min_peak_rate` counts). neurocode's 1-D pipeline uses 0.6, which on
  simulated tracks and mazes missed every secondary field at 20-50% of the main
  peak without reducing the fields found in untuned maps.
- `max_mean_rate`: Optional upper bound on the unweighted mean of non-NaN spatial-bin rates
  (default `np.inf`, disabled). This is not spike count divided by recording time,
  and does not identify interneurons or another cell type.
- `detect_subfields`: Enable subfield discrimination (default True, following
  FMAToolbox and neurocode `MapStats`)
- `min_peak_rate`: The smallest peak that starts a field (default 1 Hz, the
  `minPeak` of FMAToolbox and neurocode `MapStats` and opexebo; neurocode's
  1-D `findPlaceFieldsAvg1D` uses 2 Hz)

Segmentation is descriptive, not a test: even a flat map can yield a region.
In simulations, every untuned unit firing at 1 Hz or more had a "field" at
every recording length from 1 to 20 minutes, and so did interneuron-like units
with a high, broadly modulated rate, whose field covered the whole track. Field
count and `has_place_field` do not establish a place cell; use
`place_cell_significance` for a verdict. Keep cell-type tests and
quality selection separate from map segmentation. To recover the previous
10 Hz map-mean exclusion, pass `max_mean_rate=10.0` explicitly and inspect
`fields.excluded_reason`; it remains `"mean_rate_above_threshold"` when the
requested filter skips the map.

### Field Size

Compute the physical area of a place field:

```python
from neurospatial.encoding import field_size

# Get size of first detected field
area = field_size(env, fields[0])
# Returns: area in squared physical units (e.g., cm²)
```

Field size is the sum of the field's bin sizes, `env.bin_sizes`. On a regular grid each bin has area `bin_size²`; hexagonal and triangular-mesh bins use their own cell areas.

### Field Centroid

Compute the firing-rate-weighted center of mass of a place field:

```python
from neurospatial.encoding import rate_map_centroid

# Compute centroid of first field
center = rate_map_centroid(env, firing_rate, fields[0])
# Returns: array of shape (n_dims,) with N-D coordinates
```

The centroid is the weighted average position where weights are the firing rates:

$$
\text{centroid} = \frac{\sum_i \text{rate}_i \cdot \text{position}_i}{\sum_i \text{rate}_i}
$$

### Skaggs Spatial Information

Skaggs spatial information quantifies how much information (in bits) a cell's firing conveys about the animal's spatial location (Skaggs et al., 1996):

```python
from neurospatial.encoding import spatial_information

# Compute occupancy
occupancy = env.occupancy(times, positions, return_seconds=True)

# Compute spatial information
info = spatial_information(firing_rate, occupancy, base=2.0)
# Returns: bits per spike
```

**Formula:**

$$
I = \sum_i p_i \frac{r_i}{\bar{r}} \log_2\left(\frac{r_i}{\bar{r}}\right)
$$

where:
- $p_i$ = occupancy probability (normalized occupancy)
- $r_i$ = firing rate in bin $i$
- $\bar{r}$ = mean firing rate across all bins

**Interpretation:**
- **High values (>1 bit/spike)**: Strong spatial selectivity
- **Low values (<0.5 bit/spike)**: Weak or no spatial tuning
- **Units**: bits per spike (information per action potential)

### Sparsity

Sparsity measures how selectively a cell fires across space (Skaggs et al., 1996):

```python
from neurospatial.encoding import sparsity

sparseness = sparsity(firing_rate, occupancy)
# Returns: value in [0, 1]
```

**Formula:**

$$
\text{sparsity} = \frac{\left(\sum_i p_i r_i\right)^2}{\sum_i p_i r_i^2}
$$

**Interpretation:**
- **1**: Uniform positive firing over the occupied environment
- **Smaller values**: Firing concentrated in a smaller part of the occupied environment
- For equal occupancy and equal positive rates in a subset of bins, the value
  equals the fraction of occupied bins in that subset. A zero-rate map returns 0.

### Field Stability

Measure the stability of place fields across recording sessions:

```python
from neurospatial.encoding import field_stability
from neurospatial.encoding import compute_spatial_rate

# Compute firing rate maps from two sessions
rate_map_session1 = compute_spatial_rate(env, spikes1, times1, positions1).firing_rate
rate_map_session2 = compute_spatial_rate(env, spikes2, times2, positions2).firing_rate

# Compute stability
stability = field_stability(
    rate_map_session1,
    rate_map_session2,
    method='pearson'  # or 'spearman'
)
# Returns: correlation coefficient [-1, 1], or NaN when undefined
# (fewer than 3 bins finite in both maps, or a constant map)
```

**Methods:**
- `'pearson'`: Pearson correlation (linear relationship, parametric)
- `'spearman'`: Spearman rank correlation (monotonic relationship, non-parametric)

**Interpretation:**
- **>0.7**: Highly stable field
- **0.3-0.7**: Moderately stable
- **<0.3**: Unstable or remapped field

## Population-Level Metrics

### Population Coverage

Compute spatial coverage of a place cell population with comprehensive statistics:

```python
from neurospatial.encoding import population_coverage, plot_population_coverage
import numpy as np

# Stack firing rate maps from all neurons: shape (n_neurons, n_bins)
firing_rates = np.array([rate_map_cell1, rate_map_cell2, rate_map_cell3])

# Compute coverage (runs detect_place_fields internally); pass the bandwidth
# the maps were smoothed with, e.g. rates.bandwidth from compute_spatial_rates
result = population_coverage(env, firing_rates, bandwidth=5.0)

# Access results
print(f"Coverage: {result.coverage_fraction:.1%}")
print(f"Place cells: {result.n_place_cells}/{result.n_neurons}")
print(f"Total fields: {result.n_fields}")
print(f"Uncovered bins: {len(result.uncovered_bins)}")

# Get gap locations (bin indices and coordinates)
gap_bins = result.uncovered_bins        # Array of bin indices
gap_coords = result.uncovered_positions  # Array of (x, y) coordinates

# Visualize coverage with gap highlighting
ax = plot_population_coverage(env, result)

# Show field count per bin (redundancy visualization)
ax = plot_population_coverage(env, result, show_field_count=True)
```

The `PopulationCoverageResult` object contains:

- `coverage_fraction`: Fraction of observed bins covered (0.0-1.0)
- `uncovered_bins`: Indices of gaps (observed bins outside every field)
- `uncovered_positions`: Coordinates of gaps
- `field_count`: Number of place fields per bin
- `n_neurons`, `n_place_cells`, `n_fields`: Population statistics
- `place_fields`: Detected fields for each neuron
- `is_observed`: Bins with a finite rate for at least one neuron

A bin that is NaN in every rate map was never visited (or fell below
`min_occupancy`); it is neither covered nor a gap. Compute the rates with the
default `fill_value=None` so these bins stay NaN.

**Computed properties** (standard hippocampal metrics):

- `place_cell_fraction`: n_place_cells / n_neurons (typical CA1: 0.3-0.5)
- `fields_per_place_cell`: n_fields / n_place_cells (typical: 1.0-2.0)
- `mean_redundancy`: Average fields per covered bin (Wilson & McNaughton 1993)

Coverage is the fraction of observed bins contained in at least one place field. High coverage (>0.8) indicates the population represents most of the environment.

### Field Density Map

Count how many place fields overlap at each location:

```python
from neurospatial.encoding import field_density_map

density = field_density_map(all_place_fields, env.n_bins)
# Returns: array of shape (n_bins,) with overlap counts
```

Useful for identifying "hotspots" where many cells have overlapping fields, which may indicate salient locations (e.g., reward zones, decision points).

### Field Overlap

Measure the spatial overlap between two fields using the Jaccard index:

```python
from neurospatial.encoding import field_overlap

# Compare fields from two cells
overlap = field_overlap(field_bins_cell1, field_bins_cell2)
# Returns: Jaccard coefficient in [0, 1]
```

**Formula:**

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

**Interpretation:**
- **0**: No overlap (fields completely disjoint)
- **1**: Perfect overlap (identical fields)
- **~0.5**: Moderate overlap (typical for neighboring place cells)
- **NaN**: Neither set has a field bin, so there is nothing to compare

### Count Place-Cell Candidates

Count the units that pass the place-cell information screen. The screen
subtracts the information an untuned cell shows by chance on the same maps
before it compares with the cutoff, so it needs the rate result, not bare
information values:

```python
from neurospatial.encoding import compute_spatial_rates, count_place_cells

rates = compute_spatial_rates(env, all_spike_trains, times, positions)
n_place_cells = count_place_cells(rates, min_info=0.5)  # bits/spike, inclusive
# Equivalent: int(rates.classify(min_info=0.5).sum())
```

The 0.5 bits/spike cutoff is a screening choice, not a universal significance
threshold; see
[information bias and cell verdicts](#information-bias-and-cell-verdicts).

### Rate-map correlations

Compute the correlation between every pair of rate maps:

```python
from neurospatial.encoding import rate_map_correlations

# Stack rate maps into matrix (n_maps × n_bins)
firing_rates = np.array([
    rate_map_cell1,
    rate_map_cell2,
    rate_map_cell3,
])

# Compute correlation matrix
corr_matrix = rate_map_correlations(firing_rates)
# Returns: (n_maps, n_maps) correlation matrix
```

This describes similarity of spatial maps. Shared spatial tuning alone does
not establish coordinated temporal activity or membership in a cell assembly.

### Population-vector correlation

To ask whether the population code at each location changed between two
conditions, correlate the vector of all cells' rates at each location. Pass
the two rate results, estimated with the sorter's labels as `unit_ids=`: cells
are paired by label and bins by place, so the order the units were listed in
does not matter.

<!-- docs-test: run -->
```python
import numpy as np

from neurospatial.encoding import compute_spatial_rates, population_vector_correlation
from neurospatial.simulation import linear_track_session

session = linear_track_session(duration=120.0, n_place_cells=12, n_laps=10, seed=0)
unit_ids = np.arange(101, 113)  # the sorter's unit labels
first_half, second_half = (
    compute_spatial_rates(
        session.env, session.spike_times, session.times, session.positions,
        unit_ids=unit_ids, epochs=window, bandwidth=5.0, min_occupancy=0.05,
    )
    for window in ([0.0, 60.0], [60.0, 120.0])
)
pv = population_vector_correlation(first_half, second_half)  # (n_bins,)
defined = np.isfinite(pv)
print(f"median {np.median(pv[defined]):.2f}, "
      f"{(~defined).sum()} of {pv.size} locations undefined")
assert np.isclose(np.median(pv[defined]), 1.00, atol=0.006)
```

The simulated place fields do not move, so the two halves have a median
correlation near 1 (1.00). A location is NaN, not 0, where either condition
lacks occupancy or has a NaN rate there, or where every cell has the same rate. Units recorded in only one
condition are left out (`unit_selection="exact"` refuses them instead), and
maps estimated on different environments need an explicit `bin_mapping=`, such
as `part.bin_mapping_from(env).pairs` for a subset of the arena.

With your own `(n_units, n_bins)` matrices, pass two arrays instead. Rows must
then be the same units in the same order in both, and pass
`spatial_support=rates_1.spatial_support & rates_2.spatial_support`:
zero-filled maps (`fill_value=0`) read unvisited bins as 0 Hz, so without it
unvisited bins are compared. See
[Does the spatial map change between conditions?](population-analysis.md#does-the-spatial-map-change-between-conditions)
for a worked remapping example.

## Boundary Cell Metrics

### Border Score

Border cells of the medial entorhinal cortex fire along one or more walls of
the arena (Solstad et al., 2008). The border score compares how much of one
wall a firing field covers with how far the cell's firing lies from the walls.
On a rectangular box it is the score of Solstad et al. (2008), as computed by
BNT and opexebo; the same definition applies to polygonal and circular arenas,
arenas with internal walls and combined rooms.

Which boundary counts as a wall is a scientific choice. By default
(`boundary_source="physical"`), the score uses measured walls: the arena's
declared footprint and any internal walls. An environment built from tracked
positions (`Environment.from_samples`) has no measured walls, so the default
raises and asks for that choice. Either build the environment from the
measured arena outline (`Environment.from_polygon(outline, bin_size=...)`), or
pass `boundary_source="analysis"` to score against the edges of the visited
bins, as in the examples below. See
[Choose physical walls or analysis boundaries](environments.md#choose-physical-walls-or-analysis-boundaries).

<!-- docs-test: run -->
```python
from neurospatial.encoding import border_score
from neurospatial.encoding import compute_spatial_rate

# Compute firing rate map
firing_rate = compute_spatial_rate(env, spike_times, times, positions).firing_rate

# Compute border score against the edges of the visited bins
score = border_score(
    env,
    firing_rate,
    threshold=0.3,    # 30% of peak (Solstad et al. standard)
    min_area=200.0,   # Minimum field area (cm²)
    boundary_source="analysis",
)
# Returns: score in [-1, 1], or NaN when no field reaches min_area
```

**Algorithm:**

1. *Fields*: groups of neighboring bins at or above `threshold × peak_rate`
   (default 0.3), each with an area of at least `min_area` (default 200 cm²,
   converted to the environment's units; without metric length units, at
   least 9 bins, the BNT/opexebo default). `summary_table()` records the rule
   it applied in `df.attrs["border_field_minimum"]`.
2. *Walls*: each outline is split into walls at its corners, in physical
   units. A polygon's vertices are corners unless they sample a curve: a
   rectangle has four walls, an octagon eight, and a box with rounded corners
   or a hand-traced square four. A ring without corners (a circular arena) is
   one wall that offers each field any half of it, following Høydal et al.
   (2018). Each named wall (`OpenArena(walls={name: line})`) is one wall; one
   that lies on the outline replaces that piece of it.
3. *Coverage* `cM`: the largest fraction of one wall's observed length covered
   by one field. Unvisited wall bins are missing, not uncovered;
   `observed_wall_fraction` reports how much of the walls was observed.
4. *Distance* `dm`: the firing-rate-weighted mean distance from the bins of all
   fields to the nearest wall, divided by the radius of the largest circle that
   fits in the arena (half the shorter side of a rectangle).
5. Border score: `(cM - dm) / (cM + dm)`, in [-1, 1].

**Interpretation:**

- **Near +1**: a thin field along the whole of one wall.
- **-1**: firing only away from the walls.
- **0.5 on a square for a uniform map**: one field covers every wall and its
  mean distance is a third of the half side. The usual 0.5 cutoff therefore
  does not reject untuned cells: in a simulated 20-minute session in a 100 cm
  box, 46% of untuned units reached it, and on a radial-arm maze 6 of 12 arm
  place cells did.

The 0.5 cutoff is a convention from rectangular boxes. The score of a map that
fires everywhere depends on the arena: 0.5 in convex arenas, higher in corridor
mazes (every bin is near a wall), lower on a narrow linear track.
`untuned_border_score(env)` computes it, and `summary_table()` records it in
`df.attrs["border_score_untuned"]`. `label_cell_types` judges each label by its
own cutoff and is a screen; for a verdict, `spatial_cell_type_significance`
compares each unit's border score (and its spatial information and grid score)
with the scores of its own circularly shifted spike trains, which adapts to the
arena's shape.

**Differences from BNT and opexebo** on rectangular boxes: distances run from
each bin center to the wall (a bin against the wall is half a bin away, where
they count it as one bin away, so their scores are lower: about 0.07 for a
round field on one wall of a 100 cm box with 2.5 cm bins); fields follow Solstad et al. (2008) rather than their place-field
detection (0.2 of the peak, at least 9 bins, peak at least 1 Hz); and where
they look up to 8 bins inward past unvisited wall bins, unvisited pieces of
wall are left out of the wall's length here.

**Example: Checking a synthetic boundary field**

<!-- docs-test: run -->
```python
# Construct an idealized boundary-associated field:
# - High firing along one or more walls
# - Low firing in center

from neurospatial.encoding import border_score

# Generate the synthetic field
firing_rate = np.zeros(env.n_bins)
boundary_bins = env.boundary_bins
firing_rate[boundary_bins] = 5.0  # High firing at boundaries

score = border_score(env, firing_rate, boundary_source="analysis")
print(f"Border score: {score:.3f}")  # near 1: thin fields along the walls

# Visualize
import matplotlib.pyplot as plt
env.plot_field(firing_rate, cmap='hot')
plt.title(f"Boundary field (score = {score:.2f})")
plt.show()
```

### Boundary-vector and egocentric boundary cells

A boundary-vector cell fires when a wall is at a preferred distance in a
preferred room direction, for example 15 cm to the East (Lever et al., 2009).
An egocentric boundary cell fires when a wall is at a preferred distance and
direction relative to the animal's heading, for example 15 cm to its left
(Hinman et al., 2019; Alexander et al., 2020). A border score cannot tell
these apart; a rate map over distance and direction to the walls can.

`compute_boundary_vector_rate` casts one ray per direction bin from every
tracking sample and records how far away the first wall is. Each interval
adds its duration, and each spike one count, to the (distance, direction) bin
of every ray. A bin's rate is then the firing rate while a wall lay at that
distance in that direction. Pass `direction_frame="allocentric"` for
boundary-vector cells and `direction_frame="egocentric"` with `headings` for
egocentric boundary cells. The walls are the arena's declared walls
(`boundary_source="physical"`), so build the environment from the measured
outline; an unvisited corner is then not a wall.

The map covers walls 0-50 cm away by default, the range Hinman et al. (2019)
used, in 5 cm rings and 30° direction bins. It does not reach the far side of
the arena on purpose: in a box, a cell firing 15 cm from the East wall is
also firing 85 cm from the West wall, and a map out to the far walls lets the
peak jump there. `analysis_settings["fraction_of_time_in_distance_range"]`
records the share of ray time the map covers. A field may continue beyond the
range: for a cell whose preferred distance is in the last ring, widen
`distance_range=` and check that the peak stays on the near wall.

This example simulates an egocentric boundary cell that fires when a wall is
about 15 cm to the animal's left:

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.encoding import compute_boundary_vector_rate
from neurospatial.ops import compute_boundary_distances
from neurospatial.simulation import generate_poisson_spikes

env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")
rng = np.random.default_rng(0)
times = np.arange(0, 600, 1 / 30)  # 10 minutes at 30 Hz
positions = rng.uniform(2, 98, (len(times), 2))  # cm
headings = rng.uniform(-np.pi, np.pi, len(times))  # radians, 0 = East

# 18 direction bins of 20 degrees; one is centered straight to the left.
distances, directions = compute_boundary_distances(
    env, positions, headings, direction_frame="egocentric", n_directions=18
)
wall_left = distances[:, np.isclose(directions, np.pi / 2)][:, 0]  # cm
rate = 0.5 + 20 * np.exp(-0.5 * ((wall_left - 15) / 4) ** 2)  # Hz
spike_times = generate_poisson_spikes(rate, times, seed=1)

result = compute_boundary_vector_rate(
    env, spike_times, times, positions, headings,
    direction_frame="egocentric", direction_bin_size=np.deg2rad(20),
)
print(f"Preferred direction: {np.degrees(result.preferred_direction()):.0f} deg")
print(f"Preferred distance: {result.preferred_distance():.1f} cm")
assert np.isclose(result.preferred_direction(), np.pi / 2)
```

The peak is at +90° (left) in a distance bin near 15 cm, as simulated.
`result.plot()` draws the map with distance on one axis and direction on the
other. A bin whose rate is NaN never had a wall at that distance and
direction, which means it was not measured, not that the cell was silent.
Skaggs information is not defined over this whole map, because every sample
counts once per ray; `result.distance_information()` gives the information
about wall distance along the preferred direction, where the distance bins do
partition time. It is biased upward for few spikes, so compare it with a
shuffled control before calling a cell boundary-tuned.
`neurospatial.ops.compute_boundary_distances` gives the ray distances
themselves, and `neurospatial.ops.nearest_boundary` the distance and
direction to the closest wall at each sample.

## Specialized tuning recipes

Choose the coordinate system for the scientific question before computing a
score. The recipes below use one unit's `spike_times` in seconds, aligned
`times` and `positions`, and an independently defined `env`. Spatial examples
assume coordinates in cm. Headings are measured head direction in radians,
with zero pointing east and positive angles counterclockwise. A heading
derived from velocity measures movement direction; it does not provide a
measurement of where the head or eyes point.

| Question | Rate-map domain | Main API |
| --- | --- | --- |
| Does spatial tuning depend on travel direction? | Animal position, separately per condition | `compute_directional_place_fields` |
| Is firing associated with head direction? | Circular world-frame heading | `compute_directional_rate` |
| Is spatial firing hexagonally periodic? | 2D spatial autocorrelation | `spatial_autocorrelation`, `grid_score` |
| Is firing associated with a vector to an object? | Object distance and allocentric or egocentric bearing | `compute_object_vector_rate`, `compute_egocentric_rate` |
| Is firing associated with a vector to a wall? | Wall distance and allocentric or egocentric direction | `compute_boundary_vector_rate` |
| Is firing associated with a viewed location? | Projected gaze location in the arena | `compute_view_rate` |
| Does spike theta phase advance across a place field? | Distance travelled into one field, per spike | `compute_phase_precession` |

Keep the same analysis epochs and known spike-recording window across related
maps. Each snippet assumes spikes were recorded throughout the tracked span;
replace its `spike_window` with the actual electrophysiology recording
intervals. The default `max_gap=0.5` excludes longer sampling intervals. NaNs
and gaps remain missing observations, and each domain has its own support.
If you set `max_gap` yourself, keep it above the sampling interval, for example
`max_gap=1.5 / sampling_rate`. Timestamps such as `np.arange(n) / 30` carry
float rounding, so with `max_gap` equal to the sampling interval a quarter to a
half of the intervals count as gaps; a warning says when this happens.

### Spatial maps conditioned on movement direction

On a track aligned with the x axis, label rightward and leftward travel using
a predeclared movement criterion. For a task with known trial directions, use
those sample-aligned labels instead. Samples labeled `None` (pauses) are
left out of both occupancy and spikes.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial.encoding import compute_directional_place_fields

velocity_x = np.gradient(positions[:, 0], times)
direction_labels = np.full(len(times), None, dtype=object)
direction_labels[velocity_x > 2.0] = "rightward"   # cm/s
direction_labels[velocity_x < -2.0] = "leftward"
directional = compute_directional_place_fields(
    env, spike_times, times, positions, direction_labels,
    bandwidth=5.0, min_occupancy=0.5,
    epochs=(float(times[0]), float(times[-1])),
    spike_window=(float(times[0]), float(times[-1])),
)
ax = directional.plot()  # Horizontal axis is spatial-bin index
ax.set_ylabel("Rate (Hz)")
```

These are spatial maps conditioned on travel direction, rather than angular
head-direction tuning curves. Compare rates only where both conditions have
adequate occupancy. Different paths, speeds or trial outcomes can explain a
map difference; the helper does not perform a condition-comparison test.
Labels from `heading_direction_labels` also include `"stationary"` for an
animal that was observed but not moving; passing them straight to
`compute_directional_place_fields` gives a "stationary" map as well, and
`missing_label="stationary"` drops it (a `None` label is always dropped).
For gapped tracking, derive velocity within each observed run rather than
across a dropout.

### Head-direction tuning

Directional encoding operates on the angular domain and takes no environment.
Use angular units consistently for headings, bin width and bandwidth.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial.encoding import compute_directional_rate

hd = compute_directional_rate(
    spike_times, times, headings,
    bin_size=np.deg2rad(6.0), bandwidth=np.deg2rad(10.0),
    epochs=(float(times[0]), float(times[-1])),
    spike_window=(float(times[0]), float(times[-1])),
)
hd.plot()
print("Rate-weighted mean vector length:", hd.mean_vector_length())
```

Inspect angular occupancy as well as rates. Mean vector length describes the
estimated tuning curve. A fixed cutoff is a screen. `hd.tuning_pvalue()` tests
for one preferred direction given how long the animal faced each heading, so
uneven heading dwell time alone does not make a cell look tuned. Real cells
fire in bursts and drift in rate, so their spike counts vary more than
independent spikes would; the test estimates that extra variability from spike
counts in time windows (`hd.spike_count_dispersion`, 1 for independent spikes,
about 2b - 1 for bursts of b spikes) and divides its statistic by it. It needs
at least 50 spikes and returns NaN below that. A result you build yourself
from tuning curves has no dispersion, so its test is NaN until you pass
`spike_count_dispersion` (1.0 assumes independent spikes). Mean vector length uses only visited headings, so check
`hd.largest_unvisited_arc()`: when the animal never faced an arc wider than
90°, untuned cells get inflated MVLs and the screen warns. A circular-shift
test on the raw recording is the verdict when its stationarity assumptions fit
the experiment:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial.encoding import is_head_direction_cell

hd_shift_verdict = is_head_direction_cell(
    spike_times, times, headings,
    criterion="shuffle", n_shuffles=199, min_shift=5.0, rng=2026,
    bin_size=np.deg2rad(6.0), bandwidth=np.deg2rad(10.0),
    epochs=(float(times[0]), float(times[-1])),
    spike_window=(float(times[0]), float(times[-1])),
)
```

The selected analyzed span must be long enough for the minimum shift. Report
the null, estimator, seed and number of shuffles; a small p-value establishes
association against that null rather than independence from position or speed.

**Heading and position are confounded on tracks and mazes.** On a linear
track, T-maze or W-maze the animal faces one direction on each arm and run,
so a place cell that fires on only one running direction has a sharp heading
tuning curve. Both the screen and the circular-shift test flag such cells (in
simulation, 83-100% of one-direction place cells on tracks and T/W mazes),
because shifting spikes in time keeps the cell's position-locked firing. A
head-direction claim needs a control that holds position fixed: compare the
cell's spatial maps for each running direction with
`compute_directional_place_fields` (a place-by-direction cell has a field in
one direction only, while a head-direction cell fires everywhere the animal
faces its preferred direction), fit a model in which the rate is the product
of a position map and a direction tuning curve and ask which factor explains
the firing, or use a shuffle that keeps each spike's position.

### Grid periodicity on a regular 2D arena

Use a regular 2D grid of rectangular cells for the autocorrelogram. Preserve
missing support; filling unvisited locations with zero changes the map being
analyzed.

<!-- docs-test: run -->
```python
from neurospatial.encoding import compute_spatial_rate, grid_score, spatial_autocorrelation

spatial = compute_spatial_rate(
    env, spike_times, times, positions,
    bandwidth=5.0, min_occupancy=0.5,
    epochs=(float(times[0]), float(times[-1])),
    spike_window=(float(times[0]), float(times[-1])),
)
autocorrelation = spatial_autocorrelation(env, spatial.firing_rate)
score = grid_score(autocorrelation)
print("Hexagonal rotational-symmetry score:", score)
```

At each shift the autocorrelogram is the Pearson correlation between the map
and its shifted copy over the bins visited in both, without wrapping around
the arena. Shifts at which fewer than a quarter of the visited bins overlap
are NaN: their correlation rests on a strip of bins along two walls and is
more than twice as noisy as at zero lag. Unvisited bins are left out rather
than counted as silent, but that does not remove sampling bias. The score
sweeps an annulus from the edge of the central peak out to those shifts and
keeps the best 60°-versus-30° rotational contrast, so it does not depend on
the grid spacing as long as the first ring of fields fits, for spacings up to
about 0.6 of the arena's side. It is NaN when the central peak is not bounded
inside the defined shifts or the arena is too small for an annulus around it.
`grid_properties` reads spacing and orientation only from peaks that can be one
hexagon's first ring, and leaves them NaN otherwise. Inspect the rate map,
autocorrelogram and occupancy together. A score of 0.4 is a common screening
convention, not a significance test: on circularly time-shifted spike trains of
simulated grid, place and untuned cells, 1-8% of the shuffled maps reached it.
`spatial_cell_type_significance` tests each unit's grid score against its own
spike train circularly shifted in time. On layouts without a rectangular grid (hexagonal bins,
tracks, meshes) there is no grid score: `summary_table()` records it in
`attrs["unavailable_metrics"]`. `spatial_autocorrelation_radial` from
`neurospatial.encoding.grid` gives a distance-correlation profile there, but a
radial profile does not test for sixfold symmetry.

### Allocentric and egocentric object vectors

`object_positions` has shape `(n_objects, 2)` and shares the animal position's
coordinate system. Both APIs use the nearest object. Allocentric bearing
measures the vector **from animal to object** in world coordinates;
egocentric bearing measures that vector relative to the animal's heading
(zero is straight ahead).

The map is a grid of distance rings and direction bins around the object. By
default it covers every distance inside the arena, as in Høydal et al. (2019),
so a cell tuned to a far point is not cut off, and maps of one arena with the
object moved share a grid. The default rings are 5 cm wide and the direction
bins 30°, a resolution unsmoothed maps of 5-15 minute sessions support. Pass
`distance_range=` to restrict the map: the share of time it then covers is in
`analysis_settings["fraction_of_time_in_distance_range"]`, and a unit whose
peak lies in the outermost ring of a range that cut time off raises a warning.

<!-- docs-test: run -->
```python
from neurospatial.encoding import compute_egocentric_rate, compute_object_vector_rate

polar_settings = {
    "method": "binned",
    "min_occupancy": 0.5,
    "epochs": (float(times[0]), float(times[-1])),
    "spike_window": (float(times[0]), float(times[-1])),
}
allocentric = compute_object_vector_rate(
    env, spike_times, times, positions, object_positions,
    **polar_settings,
)
egocentric = compute_egocentric_rate(
    env, spike_times, times, positions, headings, object_positions,
    **polar_settings,
)
print(allocentric.direction_frame, egocentric.direction_frame)
allocentric.plot()
egocentric.plot()
```

`object_positions` also accepts names of `env.landmarks` (see
[landmarks](environments.md#landmarks)). A point landmark gives the same map as
its coordinates. A line or polygon landmark, such as a monitor along a wall,
is measured from its nearest edge at each sample, so a cell tuned to the
screen's surface is not smeared by distances to the screen's center. The names
are kept in `result.analysis_settings["objects"]`.

<!-- docs-test: run -->
```python
from neurospatial.encoding import compute_object_vector_rate
from neurospatial.landmarks import Landmark, Landmarks

(x_low, x_high), (y_low, y_high) = env.dimension_ranges
screen = Landmark("monitor", "line", [(x_low, y_high), (x_high, y_high)])
with_screen = env.with_landmarks(Landmarks([screen]))
by_screen = compute_object_vector_rate(
    with_screen, spike_times, times, positions, "monitor",
    epochs=(float(times[0]), float(times[-1])),
    spike_window=(float(times[0]), float(times[-1])),
)
print(by_screen.analysis_settings["objects"])
```

Compare support in the two polar domains before comparing their information
scores. Polar bin area grows with distance, and occupancy corrects dwell time;
do not smooth a finished rate map as if it contained counts. A place field near
one stationary object yields a localized object-vector map, and within one
session it scores like an object-vector cell. Move the object and correlate the
two vector maps: an object-vector field keeps its vector to the object, a place
field stays where it was (Høydal et al., 2019).

### Viewed-location maps

A view map indexes the location the animal looked at rather than its
position. The default `gaze_model="ray_cast"` takes the first wall along each
sample's heading; `gaze_model="fixed_distance"` takes the point
`view_distance` ahead, or the first wall when one is nearer. A gaze never
passes through a wall, so the map needs the arena's walls: build the
environment from the measured outline (`Environment.from_polygon`) to use its
walls, or pass `boundary_source="analysis"` to treat the edge of the binned
area as the wall, as below. Either model is a geometric assumption about where
the animal looked, not a measurement of eye gaze.

<!-- docs-test: run -->
```python
from neurospatial.encoding import compute_view_rate

view = compute_view_rate(
    env, spike_times, times, positions, headings,
    boundary_source="analysis",
    bandwidth=5.0, min_occupancy=0.5,
    epochs=(float(times[0]), float(times[-1])),
    spike_window=(float(times[0]), float(times[-1])),
)
view.plot(vmin=0)
print("View information (bits/spike):", view.spatial_information())
```

Bins that were never viewed lack support (NaN). View and position
maps generally have different occupancy and correlated predictors, so a larger
view-information score alone does not demonstrate a spatial view cell. A
head-direction cell also has more view than place information, because its
heading fixes which wall is in view. A spatial view cell's field stays in place
when the view map is computed separately from the two halves of the arena; a
head-direction cell's field slides along the wall with the animal. Match the
scientific interpretation to the head/eye measurements and gaze model.
The plural `compute_*_rates` APIs share preprocessing across a population and
preserve supplied unit labels.

### Theta phase precession on a linear track

A place cell precesses when its spikes come at earlier theta phases as the
animal moves through the field (O'Keefe & Recce, 1993). The test fits a line
to spike phase against position by circular-linear regression, one field and
one running direction at a time, in a coordinate that increases along the
run: the distance travelled into the field. Precession is then a negative
slope whichever way the animal ran.

The example simulates back-and-forth laps over one field, with an 8 Hz LFP
standing in for the recording. Replace the simulated block with your LFP
channel, its sampling rate, the spike times, the linearized track position
and the field bounds from the rate map.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial.encoding import compute_phase_precession, theta_phase
from neurospatial.simulation import generate_poisson_spikes

# --- Simulated session; replace with your recording ---
lfp_rate = 1000.0  # Hz
lfp_times = np.arange(0, 400, 1 / lfp_rate)  # s
lfp = np.cos(2 * np.pi * 8.0 * lfp_times)  # 8 Hz theta
times = lfp_times[::20]  # 50 Hz tracking
track_position = 200 * np.abs((times / 10 + 1) % 2 - 1)  # cm, 10 s per run
field_start, field_end = 86.0, 114.0  # field bounds from the rate map, cm
x = np.interp(lfp_times, times, track_position)
rightward = np.gradient(x, lfp_times) > 0
crossed = np.where(rightward, x - field_start, field_end - x) / (field_end - field_start)
preferred_phase = 5.5 - 5.0 * crossed  # rad: falls by 5 rad across the field
theta_locking = np.exp(2.0 * (np.cos(2 * np.pi * 8.0 * lfp_times - preferred_phase) - 1))
rate = 40.0 * ((x > field_start) & (x < field_end)) * theta_locking  # Hz
spike_times = generate_poisson_spikes(rate, lfp_times, seed=1)
# ---

# Theta phase at each spike: interpolate the unwrapped phase, then re-wrap.
phase = theta_phase(lfp, sampling_rate=lfp_rate)  # 0 = peak of the filtered LFP
spike_phases = np.mod(np.interp(spike_times, lfp_times, np.unwrap(phase)), 2 * np.pi)

# Position and running direction at each spike.
spike_positions = np.interp(spike_times, times, track_position)
spike_velocity = np.interp(spike_times, times, np.gradient(track_position, times))
in_field = (spike_positions >= field_start) & (spike_positions <= field_end)
field_width = field_end - field_start

runs = {"rightward": spike_velocity > 0, "leftward": spike_velocity < 0}
for direction, running in runs.items():
    keep = in_field & running
    if direction == "rightward":
        distance_into_field = spike_positions[keep] - field_start  # cm
    else:
        distance_into_field = field_end - spike_positions[keep]  # cm
    result = compute_phase_precession(
        distance_into_field, spike_phases[keep],
        position_range=(0.0, field_width), rng=0,
    )
    print(f"{direction}: slope {result.slope:.3f} rad/cm, "
          f"{result.normalized_slope:.1f} rad across the field, p = {result.p_value:.3f}")
```

Both directions give a slope near -0.18 rad/cm, a fall of about 5 rad (a
little less than one theta cycle) across the 28 cm field, with p = 0.001,
the smallest p-value 1000 shuffles can give. `slope` is in radians per cm;
`normalized_slope` is the phase change across `position_range`. The
p-value comes from shuffling which phase goes with which position and
re-fitting the slope each time, and it is calibrated: on simulated spikes
with no phase-position relationship it fell below 0.05 in 2-5% of cells
for 10 to 200 spikes. `result.correlation` and `result.mean_resultant_length`
describe how tightly the spikes follow the line; neither has a cutoff.

Choices that change the answer:

- **Direction of travel.** With raw track positions, the leftward runs give
  +0.18 rad/cm, which reads as recession. Pooling both directions in raw
  positions mixes the two slopes and the correlation drops to near zero.
- **One field, its own range.** Spikes from two fields, or from the whole
  track with the default range (the spike extent), give a slope pinned at
  the search bound of one theta cycle across the range, several times too
  shallow, and still a small p-value. A warning flags any fit whose best
  slope lies on the bound, and `result.slope_at_bound` records it for
  batch analyses. Pass one field's spikes with
  `position_range=(field_start, field_end)` in the same coordinate.
- **Fields cut by the end of the track.** The spikes cover only part of the
  field, so with the default range `normalized_slope` is the phase change
  across that part. `slope` is unaffected; compare cells by `slope`, or pass
  the field's full extent as `position_range`.
- **Theta reference.** Phase 0 is the peak of the theta-filtered LFP on the
  channel you used. Theta reverses polarity across hippocampal layers, so
  phase offsets compare between recordings only on the same reference;
  slopes do not depend on it.

`has_phase_precession` screens many cells with the same test: it returns
`True` when `p_value < alpha` and the slope is negative, so untuned cells pass
at about `alpha / 2`. `plot_phase_precession(distance_into_field,
spike_phases[keep], result)` draws the spikes twice, at phase and phase +
2π, with the fitted line.

## References and Cross-References

### Scientific References

**Place Field Analysis:**
- O'Keefe, J., & Recce, M. L. (1993). Phase relationship between hippocampal place units and the EEG theta rhythm. Hippocampus, 3(3), 317-330.
- O'Keefe, J., & Dostrovsky, J. (1971). The hippocampus as a spatial map. Brain Research, 34(1), 171-175.
- Muller, R. U., & Kubie, J. L. (1989). The effects of changes in the environment on the spatial firing of hippocampal complex-spike cells. Journal of Neuroscience, 9(1), 137-154.
- Skaggs, W. E., McNaughton, B. L., Wilson, M. A., & Barnes, C. A. (1996). Theta phase precession in hippocampal neuronal populations and the compression of temporal sequences. Hippocampus, 6(2), 149-172.
- Thompson, L. T., & Best, P. J. (1989). Place cells and silent cells in the hippocampus of freely-behaving rats. Journal of Neuroscience, 9(7), 2382-2390.

**Boundary Cell Analysis:**
- Solstad, T., Boccara, C. N., Kropff, E., Moser, M. B., & Moser, E. I. (2008). Representation of geometric borders in the entorhinal cortex. Science, 322(5909), 1865-1868.
- Høydal, Ø. A., Skytøen, E. R., Moser, M.-B., & Moser, E. I. (2018). Object-vector coding in the medial entorhinal cortex. bioRxiv 286286. https://doi.org/10.1101/286286

### Validation Notes

The repository tests analytical properties, synthetic examples and behavior
across operations. The cell-significance tests also exercise untuned spike
trains and planted tuning under specified simulation conditions. These checks
have a defined scope; they are not evidence of biological classification
accuracy on a new recording.

When comparing with opexebo, neurocode or buzcode, match bin geometry,
occupancy windows, smoothing, NaN handling and thresholds, and record package
versions and numerical differences. This guide does not provide an executable
cross-package benchmark. On rectangular boxes the border score is the
per-wall calculation of Solstad et al. (2008); see the border score section for
the remaining differences from BNT and opexebo.

## Information bias and cell verdicts

Plug-in Skaggs information is positive even for a cell that fires at the same
rate everywhere: Poisson noise makes its rate map uneven. On average, an
untuned cell shows `K / (2 ln(2) N)` bits/spike, where `N` is its number of
spikes and `K` is the number of effectively independent bins of the map.
`spatial_information_bias()` on spatial, view and object-vector rate results
returns this value for each unit. For an unsmoothed map, `K` is the number of
occupied bins minus one: the Miller (1955) and Treves & Panzeri (1995) bias.
Smoothing makes neighboring bins share spikes, so `K` is much smaller than the
number of bins. It is derived from the estimator itself (each spike's
footprint on the map, weighted by occupancy). For a 1 m box with 5 cm bins and
a 5 cm kernel, `K` is about 46 for 441 bins. With a 10 cm kernel it is about
11. The derivation is in the Notes of
`SpatialRateResult.spatial_information_bias`. In simulations of untuned cells
across arena sizes, bin sizes, bandwidths and estimators, it matched the mean
information to within 10% (16% for `method="binned"`) whenever the units had at
least as many spikes as independent bins.

Every information screen subtracts this bias before it applies its cutoff:
`is_place_cell(criterion="threshold")`, `classify()`, `label_cell_types()`,
`summary_table()` labels, `count_place_cells`, `is_object_vector_cell()` and
`is_spatial_view_cell()`. `spatial_information()` itself still returns the
plug-in value. The expansion behind the bias needs at least as many spikes as
independent bins. A unit with fewer spikes gets a NaN bias and fails the
screen with a warning. Record longer, use coarser bins or a larger bandwidth,
or run the shuffle test for such units.

Measured on 20 independent untuned 0.5 Hz Poisson units with a 30 Hz OU
trajectory in a 100 x 100 cm arena (trajectory seed 0, spikes seed 1):
place/view maps used 5 cm bins and diffusion KDE with bandwidth 5;
egocentric maps used the default binned polar grid (5 cm rings and 30°
directions out to the arena's diagonal). Entries are
counts flagged by the screen, not cells with ground-truth tuning. "Raw" is the
uncorrected plug-in information against the same cutoff.

| Minutes (approximate spikes) | Egocentric information median / raw flags / screen flags (0.3) | Place information median / raw flags / screen flags (0.5) | Detected fields | View raw / screen flags (0.5) | HD flags |
| --- | --- | --- | --- | --- | --- |
| 1 (29) | 2.05 / 20/20 / 0/20 | 0.86 / 19/20 / 0/20 | 20/20 | 20/20 / 0/20 | 0/20 |
| 2 (59) | 1.42 / 20/20 / 0/20 | 0.54 / 14/20 / 0/20 | 20/20 | 20/20 / 0/20 | 0/20 |
| 5 (148) | 0.76 / 20/20 / 0/20 | 0.22 / 0/20 / 0/20 | 20/20 | 0/20 / 0/20 | 0/20 |
| 10 (298) | 0.44 / 20/20 / 0/20 | 0.11 / 0/20 / 0/20 | 20/20 | 0/20 / 0/20 | 0/20 |
| 20 (598) | 0.22 / 0/20 / 0/20 | 0.05 / 0/20 / 0/20 | 16/20 | 0/20 / 0/20 | 0/20 |

The allocentric object-vector screen behaves like the egocentric one: raw
information flagged 20/20 up to 10 minutes, the corrected screen none. The
correction costs power at low counts. Place cells with a 2 Hz peak in this
arena passed the corrected screen at 10 minutes (about 110 spikes) but were
refused at 2 minutes (about 18 spikes, fewer than the 46 independent bins).
Object-vector cells with a 2 Hz peak fired about 70 spikes in 10 minutes,
fewer than the 150-170 occupied polar bins: the default grid covers the
whole arena (348 bins in a 1 m box), so a narrower `distance_range` or wider
bins lower the floor. The shuffle test has no such floor.

HD's screen combines the rate-weighted mean vector length
with an occupancy-corrected test of one preferred direction (a score test with
heading occupancy as exposure, divided by the estimated spike-count
dispersion), which keeps its false-positive rate near `alpha` for untuned
cells under uneven heading sampling, including bursting cells.

`has_place_field()` preserves field detection. `is_place_cell()` requires
`criterion="threshold"` (screen) or `"shuffle"` (circular-shift verdict).
The other four cell predicates default to `criterion="threshold"` and allow
opt-in `"shuffle"`. Result methods and batch `classify()` offer threshold
screens; results keep no raw input arrays. Threshold comparisons are inclusive;
shuffle and tuning-test p-values must be strictly below `alpha`.

For grid and border cells there are no predicates:
`spatial_cell_type_significance(env, spike_times, times, positions)` shifts
each unit's spikes once per shuffle and recomputes spatial information, grid
score and border score from the same shifted trains. Its `summary_table()`
gives each statistic's observed score, p-value and null percentile, and a
`cell_type` label in which each type is judged by its own test (a unit that
passes the grid and border tests is `"grid+border"`). Grid and border do not
apply on a 1-D track, and scores that cannot be computed (no measured walls
for the physical border score) have NaN p-values with the reason recorded.

For publication, report a shuffle test through the raw-array predicate or the
matching `*_cell_significance` function, together with its estimator, windows,
shift range, number of shuffles and seed. The shift null assumes stable firing
statistics on the joined analyzed clock. Gaps, invalid samples and excluded
epochs never receive shifted spikes. Stable integer seeds and matching unit
labels give the same shifts alone, in a population and after reordering.
The corrected upper-tail rank counts ties; its minimum p is `1/(n_shuffles+1)`
when all null scores are finite. NaN observed scores give NaN p-values.
The cost is about `n_shuffles` plural-map recomputes; significance rejects
`method="glm"` because pooled REML couples units. Neither a significant polar
association nor a chosen cutoff distinguishes an object-vector cell from a
place-cell control by itself.

## Common Workflows

### Complete Place Cell Analysis Pipeline

```python
import numpy as np
from neurospatial import Environment
from neurospatial.encoding import (
    detect_place_fields,
    field_size,
    rate_map_centroid,
    spatial_information,
    sparsity,
)
from neurospatial.encoding import compute_spatial_rate

# 1. Create environment and compute firing rate
env = Environment.from_samples(positions, bin_size=2.5)
rates = compute_spatial_rate(env, spike_times, times, positions, min_occupancy=0.5)
firing_rate = rates.firing_rate

# 2. Detect place fields
fields = detect_place_fields(
    env, firing_rate, bandwidth=rates.bandwidth, detect_subfields=True
)
print(f"Detected {len(fields)} place fields")

# 3. Compute field properties
for i, field in enumerate(fields):
    area = field_size(env, field)
    center = rate_map_centroid(env, firing_rate, field)
    print(f"Field {i+1}: area={area:.1f} cm², center={center}")

# 4. Compute single-cell metrics
occupancy = env.occupancy(times, positions, return_seconds=True)
info = spatial_information(firing_rate, occupancy)
sparse = sparsity(firing_rate, occupancy)
print(f"Spatial information: {info:.3f} bits/spike")
print(f"Sparsity: {sparse:.3f}")

# 5. Explicit fast screen (subtracts the untuned-cell information bias)
from neurospatial.encoding import is_place_cell
candidate = is_place_cell(env, spike_times, times, positions, criterion="threshold")
print(f"Place candidate: {candidate}")
# For publication: criterion="shuffle" on these raw arrays.
```

### Population-Level Analysis

```python
from neurospatial.encoding import (
    compute_spatial_rates,
    population_coverage,
    plot_population_coverage,
    spatial_information,
)
import numpy as np

# Batch-encode firing rate maps for all cells: shape (n_neurons, n_bins)
# all_spike_trains is a list of per-cell spike-time arrays.
rates = compute_spatial_rates(env, all_spike_trains, times, positions)

# Analyze population coverage (runs detect_place_fields internally)
result = population_coverage(env, rates.firing_rates, bandwidth=rates.bandwidth)

print(f"Population coverage: {result.coverage_fraction:.1%}")
print(f"Place cells: {result.n_place_cells}/{result.n_neurons}")
print(f"Total fields detected: {result.n_fields}")
print(f"Gap bins: {len(result.uncovered_bins)}")

# Visualize coverage with gap highlighting
import matplotlib.pyplot as plt
ax = plot_population_coverage(env, result)
plt.show()

# Show field count per bin (redundancy)
ax = plot_population_coverage(env, result, show_field_count=True)
plt.show()

# Access detected place fields for further analysis
for i, neuron_fields in enumerate(result.place_fields):
    if len(neuron_fields) > 0:
        print(f"Neuron {i}: {len(neuron_fields)} field(s)")
```

### Border Cell Detection

<!-- docs-test: run -->
```python
from neurospatial.encoding import border_score
from neurospatial.encoding import compute_spatial_rate

# Compute firing rate
firing_rate = compute_spatial_rate(env, spike_times, times, positions).firing_rate

# Compute border score. This environment was built from tracked positions,
# so score against the edges of the visited bins; with a measured arena
# outline (Environment.from_polygon), omit boundary_source to use its walls.
score = border_score(
    env, firing_rate, threshold=0.3, min_area=200.0, boundary_source="analysis"
)

# Screen for a boundary-associated candidate; not a calibrated cell verdict
is_border_candidate = score >= 0.5  # Illustrative cutoff
print(f"Border score: {score:.3f}")
print(f"Border candidate: {is_border_candidate}")

# Visualize with boundaries highlighted
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Firing rate map
env.plot_field(firing_rate, ax=ax1, cmap='hot')
ax1.set_title(f"Firing Rate (Border Score = {score:.2f})")

# Highlight boundaries
boundary_field = np.zeros(env.n_bins)
boundary_field[env.boundary_bins] = 1.0
env.plot_field(boundary_field, ax=ax2, cmap='binary', alpha=0.5)
ax2.set_title("Boundary Bins")

plt.tight_layout()
plt.show()
```

## See Also

- [Spike Train to Spatial Field Conversion](spike-field-primitives.md) - Computing firing rate maps
- [Signal Processing Primitives](signal-processing-primitives.md) - Smoothing and filtering fields
- [Spatial Analysis](spatial-analysis.md) - Core spatial operations
