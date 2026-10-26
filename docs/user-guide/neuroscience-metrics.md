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
firing_rate = compute_spatial_rate(env, spike_times, times, positions).firing_rate

# Detect place fields
fields = detect_place_fields(env, firing_rate,
    threshold=0.2,          # 20% of peak rate
    min_size=None,          # Default: at least 9 bins
    max_mean_rate=np.inf,   # Default: no rate-based exclusion
    detect_subfields=True   # Discriminate coalescent fields
)

# Result: iterable PlaceFieldsResult with bin arrays and optional exclusion metadata
# fields[0] = np.array([10, 11, 12, 23, 24, 25])  # Bin indices for field 1
# fields[1] = np.array([45, 46, 56, 57])          # Bin indices for field 2
```

**Algorithm details:**

1. **Iterative peak detection**: Identifies local maxima above threshold
2. **Connected component extraction**: Groups neighboring bins into fields
3. **Optional map-mean filter**: An explicitly finite `max_mean_rate` can skip maps
4. **Subfield discrimination**: Recursively re-thresholds at 50% and 70% to separate coalescent fields

**Key parameters:**

- `threshold`: Fraction of peak rate (default 0.2 = 20%, following Muller & Kubie, 1989)
- `min_size`: Minimum field size in bins (defaults to 9; choose for your bin size)
- `max_mean_rate`: Optional upper bound on the unweighted mean of non-NaN spatial-bin rates
  (default `np.inf`, disabled). This is not spike count divided by recording time,
  and does not identify interneurons or another cell type.
- `detect_subfields`: Enable subfield discrimination (default True, following neurocode)

Segmentation is descriptive: even a flat map can yield a region. Field count
and `has_place_field` do not establish a place cell. Keep cell-type tests and
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

# Compute coverage (runs detect_place_fields internally)
result = population_coverage(env, firing_rates)

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

Count cells at or above a spatial information threshold (inclusive):

```python
from neurospatial.encoding import count_place_cells

# Compute spatial information for all cells
spatial_info = [
    spatial_information(rate_map1, occupancy),
    spatial_information(rate_map2, occupancy),
    spatial_information(rate_map3, occupancy),
]

# Count candidates at or above the selected information screen
n_place_cells = count_place_cells(
    spatial_info,
    threshold=0.5  # bits/spike
)
# Returns: integer count
```

The 0.5 bits/spike cutoff is a screening choice, not a universal significance
threshold. Information depends on sample size, occupancy, binning and smoothing;
see [information bias and cell verdicts](#information-bias-and-cell-verdicts).

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
`spatial_support=rates_1.spatial_support & rates_2.spatial_support`: smoothing
gives finite rates in bins the animal never visited, and zero-filling reads
them as 0 Hz, so without it unvisited bins are compared. See
[Does the spatial map change between conditions?](population-analysis.md#does-the-spatial-map-change-between-conditions)
for a worked remapping example.

## Boundary Cell Metrics

### Border Score

The border score quantifies how strongly a cell's firing field is aligned with environmental boundaries (walls). Implements the algorithm from Solstad et al. (2008), adapted for arbitrary graph-based environments.

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

**Algorithm with `boundary_source="analysis"` (adapted for irregular graphs):**

1. Segment field at threshold: bins where `firing_rate >= threshold × peak_rate`
2. Compute boundary coverage (cM): fraction of boundary bins in field
3. Compute normalized mean distance (d): mean distance from field bins to nearest boundary,
   divided by the bin-center bounding-box diagonal
4. Border score: `(cM - d) / (cM + d)`

**Interpretation:**
- **+1**: Field covers boundary bins with zero mean boundary distance
- **0**: Boundary coverage equals normalized boundary distance
- **-1**: No boundary coverage with positive boundary distance
- **>0.5**: A possible candidate-screen cutoff; its calibration does not
  automatically transfer from a rectangular arena to this graph adaptation

A uniform positive map also covers every boundary and can score well above 0.5;
zero is not its untuned baseline. For a complete 5 × 5 unit grid, mean boundary
distance is `(8*1 + 1*2)/25`, the bin-center diagonal is `4*sqrt(2)`, and this
formula gives approximately 0.868 for a flat map. Compare with untuned controls
and spatial information before interpreting a boundary-selectivity claim;
this cutoff alone is insufficient.

**Key differences from original algorithm:**

The original Solstad et al. (2008) paper used rectangular arenas with 4 discrete walls. This implementation generalizes to arbitrary graph-based environments:

- **Boundary definition**: Analysis mode uses `env.boundary_bins` (graph-based boundary detection) instead of discrete walls; physical mode uses the declared wall geometry
- **Distance metric**: Uses graph geodesic distances instead of Euclidean distances
- **Coverage**: Computed over all boundary bins (not per-wall)
- **Normalization**: Distance normalized by environment extent (bounding box diagonal)

This adaptation is appropriate for irregular layouts, mazes, and complex environments where the concept of "walls" doesn't apply cleanly.

**Parameters:**

- `threshold`: Fraction of peak for field segmentation (default 0.3 = 30%, following Solstad et al.)
- `min_area`: Minimum field area to compute score (default 0.0, Solstad et al. used 200 cm² for rats)

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
print(f"Border score: {score:.3f}")  # Should be > 0.5

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
    direction_frame="egocentric", n_direction_bins=18,
)
print(f"Preferred direction: {np.degrees(result.preferred_direction()):.0f} deg")
print(f"Preferred distance: {result.preferred_distance():.1f} cm")
assert np.isclose(result.preferred_direction(), np.pi / 2)
```

The peak is at +90° (left) in a distance bin near 15 cm, as simulated.
`result.plot()` draws the map with distance on one axis and direction on the
other. A bin whose rate is NaN never had a wall at that distance and
direction, which means it was not measured, not that the cell was silent.
Spatial information on this map is biased upward for few spikes, so compare
it with a shuffled control before calling a cell boundary-tuned.
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
estimated tuning curve. A fixed cutoff is a screen, and a Rayleigh statistic
on spike angles can reflect uneven heading dwell time. A circular-shift test
on the raw recording is available when its stationarity assumptions fit the
experiment:

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

### Grid periodicity on a regular 2D arena

Use a regular 2D grid for the FFT autocorrelation. Preserve missing support;
filling unvisited locations with zero changes the map being analyzed.

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

The autocorrelation uses circular FFT boundaries and centers observed values;
its missing-bin treatment does not remove sampling bias. Inspect the rate map,
autocorrelogram and occupancy together. The annulus and binning affect the
score, which may be unavailable for small or constant maps. A conventional
score cutoff is not a calibrated significance test. For an irregular graph,
`spatial_autocorrelation_radial` and `periodicity_score` from
`neurospatial.encoding.grid` describe radial periodicity; a radial profile
does not establish sixfold symmetry.

### Allocentric and egocentric object vectors

`object_positions` has shape `(n_objects, 2)` and shares the animal position's
coordinate system. Both APIs use the nearest object. Allocentric bearing
measures the vector **from animal to object** in world coordinates;
egocentric bearing measures that vector relative to the animal's heading
(zero is straight ahead).

<!-- docs-test: run -->
```python
from neurospatial.encoding import compute_egocentric_rate, compute_object_vector_rate

polar_settings = {
    "distance_range": (0.0, 50.0),
    "n_distance_bins": 10,
    "n_direction_bins": 12,
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
    distance_range=(0.0, 50.0), n_distance_bins=10, n_direction_bins=12,
    epochs=(float(times[0]), float(times[-1])),
    spike_window=(float(times[0]), float(times[-1])),
)
print(by_screen.analysis_settings["objects"])
```

Compare support in the two polar domains before comparing their information
scores. Polar bin area grows with distance, and occupancy corrects dwell time;
do not smooth a finished rate map as if it contained counts. A place field near
one stationary object can yield a localized object-vector map. Stable vector
tuning after object relocation, diverse heading coverage and appropriate
controls are needed to distinguish those explanations.

### Viewed-location maps

A view map indexes projected gaze locations rather than the animal's position.
The fixed-distance model below projects 15 cm along measured heading; it is a
geometric assumption, not a measurement of eye gaze.

<!-- docs-test: run -->
```python
from neurospatial.encoding import compute_view_rate

view = compute_view_rate(
    env, spike_times, times, positions, headings,
    gaze_model="fixed_distance", view_distance=15.0,
    bandwidth=5.0, min_occupancy=0.5,
    epochs=(float(times[0]), float(times[-1])),
    spike_window=(float(times[0]), float(times[-1])),
)
view.plot(vmin=0)
print("View information (bits/spike):", view.view_spatial_information())
```

`gaze_model="ray_cast"` takes the first wall along the gaze instead.
Projected locations outside the environment lack support. View and position
maps generally have different occupancy and correlated predictors, so a larger
view-information score alone does not demonstrate a spatial view cell. Match
the scientific interpretation to the head/eye measurements and gaze model.
The plural `compute_*_rates` APIs share preprocessing across a population and
preserve supplied unit labels.

## References and Cross-References

### Scientific References

**Place Field Analysis:**
- O'Keefe, J., & Dostrovsky, J. (1971). The hippocampus as a spatial map. Brain Research, 34(1), 171-175.
- Muller, R. U., & Kubie, J. L. (1989). The effects of changes in the environment on the spatial firing of hippocampal complex-spike cells. Journal of Neuroscience, 9(1), 137-154.
- Skaggs, W. E., McNaughton, B. L., Wilson, M. A., & Barnes, C. A. (1996). Theta phase precession in hippocampal neuronal populations and the compression of temporal sequences. Hippocampus, 6(2), 149-172.
- Thompson, L. T., & Best, P. J. (1989). Place cells and silent cells in the hippocampus of freely-behaving rats. Journal of Neuroscience, 9(7), 2382-2390.

**Boundary Cell Analysis:**
- Solstad, T., Boccara, C. N., Kropff, E., Moser, M. B., & Moser, E. I. (2008). Representation of geometric borders in the entorhinal cortex. Science, 322(5909), 1865-1868.

### Validation Notes

The repository tests analytical properties, synthetic examples and behavior
across operations. The cell-significance tests also exercise untuned spike
trains and planted tuning under specified simulation conditions. These checks
have a defined scope; they are not evidence of biological classification
accuracy on a new recording.

When comparing with opexebo, neurocode or buzcode, match bin geometry,
occupancy windows, smoothing, NaN handling and thresholds, and record package
versions and numerical differences. This guide does not provide an executable
cross-package benchmark. The border score intentionally uses a graph
adaptation and should not be treated as numerically interchangeable with the
original per-wall calculation.

## Information bias and cell verdicts

Plug-in information has approximate upward finite-count bias
`(n_bins - 1) / (2 ln(2) N_spikes)`. This approximation describes independent
multinomial sampling; occupancy imbalance, smoothing and temporal dependence
change its magnitude. A fixed cutoff is a screening heuristic, and raw binning
is not universally lower or more conservative than smoothed estimation.

Measured on 20 independent untuned 0.5 Hz Poisson units with a 30 Hz OU
trajectory in a 100 x 100 cm arena (trajectory seed 0, spikes seed 1):
place/view maps used 5 cm bins and diffusion KDE with bandwidth 5;
egocentric maps used the default binned 10 x 12 polar grid. Entries are
counts flagged by the screen, not cells with ground-truth tuning.

| Minutes (approximate spikes) | Egocentric information median / flags (0.3) | Place information median / flags (0.5) | Detected fields | View flags (0.5) | HD flags |
| --- | --- | --- | --- | --- | --- |
| 1 (30) | 2.12 / 20/20 | 0.86 / 19/20 | 11/20 | 19/20 | 0/20 |
| 2 (60) | 1.41 / 20/20 | 0.53 / 15/20 | 9/20 | 19/20 | 0/20 |
| 5 (150) | 0.70 / 20/20 | 0.22 / 0/20 | 20/20 | 0/20 | 0/20 |
| 10 (300) | 0.41 / 20/20 | 0.11 / 0/20 | 20/20 | 0/20 | 0/20 |
| 20 (600) | 0.21 / 0/20 | 0.05 / 0/20 | 16/20 | 0/20 | 0/20 |

The allocentric object-vector information screen also flagged 20/20 in the
10-minute fixture. HD's MVL screen weights firing rates, while its Rayleigh
statistic weights spike counts (or reconstructs counts from rate times
occupancy). The spike-angle distribution can reflect heading dwell time, so
Rayleigh can be occupancy-biased; these empirical zero counts do not calibrate
the combined screen for another occupancy distribution.

`has_place_field()` preserves field detection. `is_place_cell()` requires
`criterion="spatial_info"` (screen) or `"shuffle"` (circular-shift verdict).
The other four cell predicates default to `criterion="threshold"` and allow
opt-in `"shuffle"`. Result methods and batch `classify()` offer threshold
screens; results keep no raw input arrays. Threshold comparisons are inclusive;
shuffle and Rayleigh p-values must be strictly below `alpha`.

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
firing_rate = compute_spatial_rate(
    env, spike_times, times, positions, min_occupancy=0.5
).firing_rate

# 2. Detect place fields
fields = detect_place_fields(env, firing_rate, detect_subfields=True)
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

# 5. Explicit fast screen (biased at low counts)
from neurospatial.encoding import is_place_cell
candidate = is_place_cell(env, spike_times, times, positions, criterion="spatial_info")
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
firing_rates = compute_spatial_rates(env, all_spike_trains, times, positions).firing_rates

# Analyze population coverage (runs detect_place_fields internally)
result = population_coverage(env, firing_rates)

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
