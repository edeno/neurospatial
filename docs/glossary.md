# Glossary

A short reference for the spatial-coding vocabulary used throughout
neurospatial. Each entry says what the term means, names the
neurospatial concept it maps to, and links to the closest notebook.

---

### Allocentric

A reference frame fixed in the world, not the animal. Allocentric
coordinates are the cartesian `(x, y)` or `(x, y, z)` of physical
space; an allocentric bearing is measured against compass directions
(east is the standard zero in neurospatial). Contrast with
[*egocentric*](#egocentric). See
[`ops.egocentric.allocentric_to_egocentric`](api/index.md) and
[`24_object_vector_cells`](examples/24_object_vector_cells.ipynb).

### Bayesian decoding

Reading the animal's position back out from a population of neurons.
Given a vector of spike counts across cells in a short window,
Bayesian decoding computes a probability ("posterior") over every
spatial bin: how likely the animal is to be there given the observed
firing pattern. From the posterior you can extract a single
best-guess location (the **MAP estimate**, the bin with the highest
probability) or the posterior mean. The discrepancy between the
estimate and the animal's actual position is the **decoding error**.
The required inputs are a *rate map* per cell (the "encoding model"
— what each cell fires across space, built from a training segment
of the recording) and the spike counts during the segment you want
to decode. See [`decode_position`](api/index.md) and
[`20_bayesian_decoding`](examples/20_bayesian_decoding.ipynb).

### Bin

A single discrete cell of the discretized environment. Each bin
carries a fixed area (2D) or volume (3D), a centroid in the
allocentric frame, and a node ID in the connectivity graph. neurospatial
identifies bins by integer index throughout; see
[`Environment.bin_at`](api/index.md). Bins are sometimes called
**nodes** when the graph view is more relevant — e.g. when computing
geodesic distances.

### Cell (graph)

Synonym for *bin* in graph contexts. neurospatial uses "cell" only
when interfacing with libraries that prefer the graph vocabulary; the
canonical term in the public API is *bin*.

### Egocentric

A reference frame fixed in the animal's body. The egocentric origin
is the animal's current position, and angles are measured relative to
its current heading: `0 = ahead`, `π/2 = left`, `-π/2 = right`. Used
to describe how the animal perceives nearby objects regardless of
which compass direction it's currently facing. See
[`compute_egocentric_rate`](api/index.md),
[`22_spatial_view_cells`](examples/22_spatial_view_cells.ipynb), and
[`24_object_vector_cells`](examples/24_object_vector_cells.ipynb).

### Field

A scalar quantity defined per bin: shape `(n_bins,)` for single-time
quantities, `(n_time, n_bins)` for time-varying. Place fields,
boundary fields, occupancy maps, and decoded posteriors are all
fields. See
[`compute_spatial_rate`](api/index.md) which returns a *rate field*.

### Linearization

Mapping a 2D position onto a 1D coordinate by projecting it onto a
track graph (a linear graph embedded in 2D, like a T-maze or W-track).
The 1D coordinate respects topological adjacency on the track even
when bins are far apart in 2D space. See
[`Environment.to_linear`](api/index.md) and
[`05_track_linearization`](examples/05_track_linearization.ipynb).

### Object-vector cell (OVC)

A neuron tuned to distance and allocentric direction relative to an object
(Høydal et al., 2019). The analysis uses animal-to-object direction:
0 = East and +pi/2 = North. Add pi and wrap to obtain the reverse,
object-to-animal vector. See [`compute_object_vector_rate`](api/index.md),
[`is_object_vector_cell`](api/index.md), and
[`24_object_vector_cells`](examples/24_object_vector_cells.ipynb).

### Egocentric object-vector (bearing) cell

A neuron tuned to object bearing relative to the animal's heading
(Wang et al., 2018): 0 = ahead and +pi/2 = left. This frame requires headings;
it is separate from world-relative object-vector tuning. See
[`compute_egocentric_rate`](api/index.md) and
[`is_egocentric_object_vector_cell`](api/index.md). Both map families return
`ObjectVectorRateResult(s)` and record `direction_frame`.

### Occupancy

The amount of time the animal spent in each bin: shape `(n_bins,)`
with units of seconds. Bins the animal never visited have occupancy 0
(or NaN, depending on the consumer). Firing-rate computation divides
spike counts by occupancy bin-wise.

### Place cell

A neuron whose firing rate is selective for the animal's location in
the environment, irrespective of head direction or behavior. The
spatial firing-rate map of a place cell typically has one or a few
compact *place fields*. See
[`compute_spatial_rate`](api/index.md),
[`detect_place_fields`](api/index.md), and
[`11_place_field_analysis`](examples/11_place_field_analysis.ipynb).

### Place field

A contiguous region of high firing rate in a place cell's rate map.
Detected via thresholding + connected-component analysis; quantified
by location, size, peak rate, and information content. See
[`detect_place_fields`](api/index.md).

### Rate map

The spatial firing-rate field for one neuron: shape `(n_bins,)` in
units of Hz. Built from spike counts ÷ occupancy, optionally smoothed
via graph diffusion or Gaussian KDE. The phrase "place field" refers
to a thresholded subset of the rate map.

### Shuffle significance

An observed tuning statistic compared with a null distribution made by
circularly shifting each spike train on the joined valid recording clock.
Shifts preserve the retained spike count and compressed-clock circular
spacing, avoid recording gaps, and break alignment with behavior. This null
assumes stable firing statistics over analyzed time. Upper-tail p-values use
`(1 + number of finite null scores >= observed) / (1 + number of finite scores)`.
The five `encoding.*_cell_significance` functions return label-keyed results;
free predicates use `criterion="shuffle"`. A threshold screen or detected
place field is separate from a calibrated verdict, and a significant
association alone does not establish cell identity.

### Spike-triggered

Anything indexed by spike times rather than time bins. A
*spike-triggered average* of position gives the animal's expected
location at the moment each spike fired; a *spike-triggered
event-aligned histogram* (PSTH) does the analogous slicing around
event times. See [`peri_event_histogram`](api/index.md) and
[`26_peri_event_psth`](examples/26_peri_event_psth.ipynb).

### Tuning curve

A 1-D firing-rate function over some behavioral variable other than
position — head direction, speed, time-from-event. Computed by
binning the variable, counting spikes per bin, and dividing by
occupancy. Direction tuning is the most common case. See
[`compute_directional_rate`](api/index.md) and
[`25_head_direction_tuning`](examples/25_head_direction_tuning.ipynb).

### View field

A spatial firing-rate field indexed by the location the animal is
*looking at*, not where the animal is standing. Built from
gaze-direction trajectories and a view model (fixed-distance,
ray-cast, or boundary-intersection). See
[`compute_view_rate`](api/index.md),
[`is_spatial_view_cell`](api/index.md), and
[`22_spatial_view_cells`](examples/22_spatial_view_cells.ipynb).
