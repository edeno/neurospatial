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
[`ops.allocentric_to_egocentric`](api/index.md) and
[specialized recipes](user-guide/neuroscience-metrics.md#specialized-tuning-recipes).

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
[`06_position_decoding`](examples/06_position_decoding.ipynb).

### Bin

A single discrete cell of the discretized environment. Each bin
carries a fixed area (2D) or volume (3D), a centroid in the
allocentric frame, and a node ID in the connectivity graph. neurospatial
identifies bins by integer index throughout; see
[`Environment.bin_at`](api/index.md). Bins are sometimes called
**nodes** when the graph view is more relevant — e.g. when computing
geodesic distances.

### Bin cell

Synonym for *bin* in graph and grid contexts, such as a cell of a
regular grid. It is not a neuron: for neurons this documentation says
*unit*, *neuron* or *place cell*. The canonical term in the public API
is *bin*.

### dt

The width of a time bin, in seconds. `dt=` is used wherever activity is
counted in time bins: `decode_session` and `decode_position` (the
decoding bin), `peri_event_histogram` and `BayesianDecoder`. `decode_session`,
`BayesianDecoder` and `peri_event_histogram` default to 0.025 s;
`decode_position` requires `dt=`. Spatial,
angular and distance bin widths are called `bin_size`.

### Egocentric

A reference frame fixed in the animal's body. The egocentric origin
is the animal's current position, and angles are measured relative to
its current heading: `0 = ahead`, `π/2 = left`, `-π/2 = right`. Used
to describe how the animal perceives nearby objects regardless of
which compass direction it's currently facing. See
[`compute_egocentric_rate`](api/index.md),
[specialized recipes](user-guide/neuroscience-metrics.md#specialized-tuning-recipes), and
[specialized recipes](user-guide/neuroscience-metrics.md#specialized-tuning-recipes).

### Epoch

A stretch of time to analyze, such as a running bout, a session or one
trial. Pass `epochs=` to a rate-map, tuning or decoding function to
restrict it to those windows; each window is half-open, `[start, stop)`,
in seconds on the same clock as `times`. An epoch is an *interval* used
as a selector. See [`Intervals`](api/index.md).

### Event-triggered

Anything aligned to event times rather than to the clock: a *peri-event
time histogram* (PSTH) of spikes around reward deliveries, or a signal
averaged around ripples. The events can be stimuli, behavior or detected
events. See [`peri_event_histogram`](api/index.md) and
[`07_peri_event_analysis`](examples/07_peri_event_analysis.ipynb).

### Field

A scalar quantity defined per bin: shape `(n_bins,)` for single-time
quantities, `(n_time, n_bins)` for time-varying. Place fields,
boundary fields, occupancy maps, and decoded posteriors are all
fields. See
[`compute_spatial_rate`](api/index.md) which returns a *rate field*.

### Interval

A half-open time window `[start, stop)` in seconds, such as a lap, a trial
or a row of a task log. An `Intervals` table holds many of them with stable
ids and named fields (outcome, direction, region) and works anywhere an
*epoch* is accepted. Detectors such as `detect_laps` and `detect_trials`
return intervals.

### Lap

One circuit of a loop, or one pass along a linear track, found by
`detect_laps`. A lap is a time interval with a direction (clockwise or
counter-clockwise on a loop; increasing or decreasing on a linear track)
and an overlap score against the template lap. Contrast with *trial*.

### Linearization

Mapping a 2D position onto a 1D coordinate by projecting it onto a
track graph (a linear graph embedded in 2D, like a T-maze or W-track).
The 1D coordinate respects topological adjacency on the track even
when bins are far apart in 2D space. See
[`Environment.to_linear`](api/index.md) and
[`03_trajectories_and_behavior`](examples/03_trajectories_and_behavior.ipynb).

### `max_gap`

The longest stretch of missing tracking, in seconds, that is bridged: the
interval between the valid samples on either side still counts as time the
animal spent somewhere. With the default `time_allocation="start"` the
interval is charged to the bin it started in; `"linear"` spreads it along
the straight line between the two samples. A longer gap is treated as a
tracking loss: it is left out of occupancy and its spikes are not counted. The default is 1.0 s, chosen for video and
DeepLabCut tracking. Choose a value longer than routine dropped frames and
shorter than a real loss, such as the animal being picked up. `None`
bridges every gap.

### Object-vector cell (OVC)

A neuron tuned to distance and allocentric direction relative to an object
(Høydal et al., 2019). The analysis uses animal-to-object direction:
0 = East and +pi/2 = North. Add pi and wrap to obtain the reverse,
object-to-animal vector. See [`compute_object_vector_rate`](api/index.md),
[`is_object_vector_cell`](api/index.md), and
[specialized recipes](user-guide/neuroscience-metrics.md#specialized-tuning-recipes).

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
[`02_spatial_rate_maps`](examples/02_spatial_rate_maps.ipynb).

### Place field

A contiguous region of high firing rate in a place cell's rate map.
Detected via thresholding + connected-component analysis; quantified
by location, size, peak rate, and information content. See
[`detect_place_fields`](api/index.md).

### Posterior

In decoding, the probability that the animal was in each bin given the
spikes in one time bin, as probability mass per bin. A `DecodingResult` has
one posterior row per time bin, shape `(n_time_bins, n_bins)`; each row
sums to 1. The default prior is uniform over space, and the MAP bin and
credible region rank bins by density (mass divided by bin size). See
*Bayesian decoding*.

### PSTH

Peri-stimulus (peri-event) time histogram: firing rate as a function of
time relative to a set of events such as reward delivery, a stimulus or a
ripple, averaged over events. `peri_event_histogram(spike_times,
event_times, window, dt=...)` returns the rate in each time bin relative
to the event. See *Event-triggered*.

### Rate map

The spatial firing-rate field for one neuron: shape `(n_bins,)` in
units of Hz. Built from spike counts ÷ occupancy, optionally smoothed
via graph diffusion or Gaussian KDE. A *place field*
is a thresholded region of the rate map, not the whole map: `compute_spatial_rate`
returns a rate map and `detect_place_fields` finds the fields in it.

### Shuffle significance

An observed tuning statistic compared with a null distribution made by
circularly shifting each spike train on the joined valid recording clock.
Shifts preserve the retained spike count and compressed-clock circular
spacing, avoid recording gaps, and break alignment with behavior. This null
assumes stable firing statistics over analyzed time. Upper-tail p-values use
`(1 + number of finite null scores >= observed) / (1 + number of finite scores)`.
The five `encoding.*_cell_significance` functions return label-keyed results;
free predicates run the same test by default (`criterion="shuffle"`). A threshold screen or detected
place field is separate from a calibrated verdict, and a significant
association alone does not establish cell identity: the place, view and
object-vector verdicts also need bias-corrected information at or above the
screen's cutoff.

### Spike-triggered

Anything indexed by spike times rather than time bins. A
*spike-triggered average* of position gives the animal's expected
location at the moment each spike fired.

### Spike window (`spike_window`)

The time during which the electrophysiology was recording. Intervals
outside it are left out of occupancy and their spikes are not counted.
The default, `None`, assumes spikes were recorded whenever position was;
pass it when tracking started before, or continued after, the spike
recording. The result records the window applied and whether it was
assumed.

### Trial

One run of a task, such as leaving the start arm and arriving at a goal
arm, found by `detect_trials` or built from a task log with
`Intervals.from_arrays`. A trial is a time interval with an outcome
(`reached`, `returned` or `censored`) and the regions it started from and
ended in. Contrast with *lap*.

### Tuning curve

A 1-D firing-rate function over some behavioral variable other than
position — head direction, speed, time-from-event. Computed by
binning the variable, counting spikes per bin, and dividing by
occupancy. Direction tuning is the most common case. See
[`compute_directional_rate`](api/index.md) for head direction,
[`compute_tuning_curves`](api/index.md) for any other variable
([Workflow 9](user-guide/workflows.md#workflow-9-firing-along-the-route-to-a-goal)), and
[specialized recipes](user-guide/neuroscience-metrics.md#specialized-tuning-recipes).

### View field

A spatial firing-rate field indexed by the location the animal is
*looking at*, not where the animal is standing. Built from
gaze-direction trajectories and a view model (fixed-distance,
ray-cast, or boundary-intersection). See
[`compute_view_rate`](api/index.md),
[`is_spatial_view_cell`](api/index.md), and
[specialized recipes](user-guide/neuroscience-metrics.md#specialized-tuning-recipes).
