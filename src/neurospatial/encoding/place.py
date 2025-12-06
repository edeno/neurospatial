"""Place cell encoding analysis.

This module provides tools for analyzing place cell representations,
including place field computation, detection, and spatial metrics.

Functions are re-exported from their source modules to provide a unified
interface under ``neurospatial.encoding.place``.

Functions from spike_field.py
-----------------------------
compute_place_field : Compute place field from spike train.
compute_directional_place_fields : Compute direction-conditioned place fields.
spikes_to_field : Convert spike train to spatial firing rate field.
DirectionalPlaceFields : Container for directional place field results.

Functions from metrics/place_fields.py
--------------------------------------
detect_place_fields : Detect place fields using iterative peak-based approach.
skaggs_information : Compute Skaggs spatial information (bits/spike).
sparsity : Compute sparsity of spatial firing.
selectivity : Compute spatial selectivity (peak/mean rate).
field_centroid : Compute firing-rate-weighted centroid.
field_size : Compute field size (area) in physical units.
field_stability : Compute stability between two firing rate maps.
field_shape_metrics : Compute geometric shape metrics for a place field.
field_shift_distance : Compute distance between field centroids.
in_out_field_ratio : Compute ratio of in-field to out-of-field firing rate.
information_per_second : Compute spatial information in bits per second.
mutual_information : Compute mutual information between position and firing rate.
rate_map_coherence : Compute spatial coherence of a firing rate map.
spatial_coverage_single_cell : Compute fraction of environment covered by cell.
compute_field_emd : Compute Earth Mover's Distance between rate distributions.

Examples
--------
>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.encoding.place import compute_place_field, detect_place_fields
>>>
>>> # Create environment and trajectory
>>> positions = np.random.uniform(0, 100, (1000, 2))
>>> times = np.linspace(0, 100, 1000)
>>> env = Environment.from_samples(positions, bin_size=10.0)
>>>
>>> # Compute place field
>>> spike_times = np.random.uniform(0, 100, 50)
>>> firing_rate = compute_place_field(
...     env, spike_times, times, positions, bandwidth=10.0
... )
>>>
>>> # Detect place fields
>>> fields = detect_place_fields(firing_rate, env)

See Also
--------
neurospatial.encoding.grid : Grid cell analysis.
neurospatial.encoding.head_direction : Head direction cell analysis.
neurospatial.encoding.border : Border/boundary cell analysis.
"""

from __future__ import annotations

# Re-export from metrics/place_fields.py (field metrics and detection)
from neurospatial.metrics.place_fields import (
    compute_field_emd,
    detect_place_fields,
    field_centroid,
    field_shape_metrics,
    field_shift_distance,
    field_size,
    field_stability,
    in_out_field_ratio,
    information_per_second,
    mutual_information,
    rate_map_coherence,
    selectivity,
    skaggs_information,
    sparsity,
    spatial_coverage_single_cell,
)

# Re-export from spike_field.py (spike→field conversion)
from neurospatial.spike_field import (
    DirectionalPlaceFields,
    compute_directional_place_fields,
    compute_place_field,
    spikes_to_field,
)

# ruff: noqa: RUF022  - Intentionally organized into groups with comments
__all__ = [
    # Classes
    "DirectionalPlaceFields",
    # Spike → field conversion
    "compute_directional_place_fields",
    "compute_place_field",
    "spikes_to_field",
    # Field detection
    "detect_place_fields",
    # Information-theoretic metrics
    "skaggs_information",
    "information_per_second",
    "mutual_information",
    # Sparsity/selectivity metrics
    "sparsity",
    "selectivity",
    "spatial_coverage_single_cell",
    # Field geometry metrics
    "field_centroid",
    "field_size",
    "field_shape_metrics",
    # Field comparison metrics
    "field_stability",
    "field_shift_distance",
    "compute_field_emd",
    "in_out_field_ratio",
    "rate_map_coherence",
]
