- [ ] `compute_place_field` and metrics are single-cell; for large populations you re-do occupancy, kernels, and sometimes binning. Adding a population-level API (e.g., take an occupancy once, compute kernels once, then accept a spike-time list per unit) would give big speedups with minimal conceptual cost.
- [ ] _extract_connected_component_graph in src/neurospatial/metrics/place_fields.py (line 200) uses frontier.pop(0) on a list; switching to collections.deque (popleft()) would avoid O(n²) behavior on large components.
- [ ] Some tight loops over bins (e.g., in Gaussian KDE path of src/neurospatial/spike_field.py and certain layout helpers) could be further vectorized or batched, but these are secondary unless you’re routinely at very large n_bins.
- [ ] One-click workflows: On top of the primitives, expose a few high-level “analysis recipes”:
e.g. analyze_place_cell(env, spike_times, times, positions) that returns a small results object (field map, detected fields, Skaggs info, sparsity, stability, maybe some pre-made matplotlib axes).
Similar wrappers for boundary-cell analysis and track-based behavioral segmentation (laps, trials, region dwell times) that internally use Environment.occupancy, bin_sequence, segmentation utilities, and metrics.
- [ ] Richer result objects: Instead of returning only NumPy arrays, consider lightweight results containers for complex operations:
Place-field analysis → an object with field, fields (list of bin-index arrays), metrics (Skaggs, sparsity, stability), and params. Each could have .to_dict() for logging/serialization. This makes provenance and reproducibility much easier for multi-analyst projects.
- [ ] Visualization utilities: Add built-in plotting functions for common visualizations:
  - [ ]Place fields with detected fields overlaid.
  - [ ]Stability plots (e.g., correlation between first and second half).
  - [ ]Occupancy maps with spike overlays.
  - [ ]Fields over time for dynamic analyses.
    - rendering of regions
    - how does this work with different layouts? different dimensions?
    - integrating with video of animal behavior (could be different sampling rates)
    - should be able to display head direction
    - should be able to display multiple body parts (e.g., nose, tail base)
    - multiple synced environments
    - Multi-Animal Support
    - Mark specific events (rewards, errors, choice points) on timeline.
    - Extend to 3D environments with 3D position tracking.
- [ ] Head direction
  - [ ] Implement head direction cell analysis, including circular occupancy, tuning curves, and metrics like mean vector length and preferred direction.
  - [ ] Add utilities for circular statistics (e.g., Rayleigh test) to assess significance of head direction tuning.
- [ ] Help users define environment boundaries and obstacle.
- [ ] phase precession analysis
- [ ] Distance of animal to goal over time

viz

- scale bar option
- colorbar option
- 3D with napari
- viz spike raster on position heatmap (either static or dynamic)
- visualize events timeline
- visualize continuous variables (speed, acceleration, head direction)
- visualize region events
- track graph

Session-level pipeline + config (Flagship feature, leverages existing code).

One population / remapping module (even a minimal version).

NWB (or similar) adapter + one or two figure helpers.

Annotation quality checks + small dashboard for quick exploration.

Streaming/chunked analysis and, later, optional SAM-based assist.

Cost-distance maps: distance under movement constraints (e.g., obstacles).
Cost-distance maps (graph shortest paths with per-edge cost).

Visibility / line-of-sight (for studying cue visibility).
zonal stats APIs

Construct good spatio-temporal regressors from events and spatial features for GLMs

Cross-session / cross-animal comparability

With proper environment alignment and spatial representation:

You can bring multiple animals’ arenas into a common coordinate frame.

Events module: standardized way to represent behavioral events (rewards, choices, errors). Could visualize when events are active (opto stim, reward delivery, stimulus presentation). Easily compute event-triggered averages of neural activity (zone entries, exits, rewards, opto).

NWB import/export utilities

decoding module (bayesian, ica, etc.)

Need to consider scalability and performance for large datasets (many neurons, long sessions).

Moran’s I on firing rate maps

Ripley’s K for spike patterns accounting for occupancy

Spatio-temporal K-functions

Variogram estimation for rate maps

GLM-based “residual spatial structure” tests

Point-process co-clustering (spikes vs reward events)

Baddeley’s inhomogeneous K estimator

v
