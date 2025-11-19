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
- [ ] Head direction
  - [ ] Implement head direction cell analysis, including circular occupancy, tuning curves, and metrics like mean vector length and preferred direction.
  - [ ] Add utilities for circular statistics (e.g., Rayleigh test) to assess significance of head direction tuning.
- [ ] Help users define environment boundaries and obstacle.
- [ ] phase precession analysis
- [ ] Distance of animal to goal over time
