"""Object-vector cell analysis.

This module provides tools for analyzing object-vector cells (OVCs), which fire
when an animal is at a specific distance and direction from an object in the
environment. OVCs encode spatial relationships in an egocentric reference frame.

Imports
-------
Object-vector field computation:

>>> from neurospatial.encoding.object_vector import compute_object_vector_field
>>> from neurospatial.encoding.object_vector import ObjectVectorFieldResult

Object-vector tuning and metrics:

>>> from neurospatial.encoding.object_vector import compute_object_vector_tuning
>>> from neurospatial.encoding.object_vector import ObjectVectorMetrics
>>> from neurospatial.encoding.object_vector import object_vector_score
>>> from neurospatial.encoding.object_vector import is_object_vector_cell
>>> from neurospatial.encoding.object_vector import plot_object_vector_tuning

Or import everything:

>>> from neurospatial.encoding import object_vector

Which Function Should I Use?
----------------------------
**Computing object-vector field from spikes?**
    Use ``compute_object_vector_field()`` to compute firing rate as a function
    of egocentric distance and direction to nearest object.

**Computing tuning metrics from raw data?**
    Use ``compute_object_vector_tuning()`` to get a comprehensive
    ``ObjectVectorMetrics`` dataclass with preferred distance/direction,
    selectivity, and tuning curve.

**Quantifying selectivity?**
    Use ``object_vector_score()`` to compute combined distance/direction
    selectivity metric.

**Screening many neurons?**
    Use ``is_object_vector_cell()`` for fast boolean filtering.

**Visualizing tuning?**
    Use ``plot_object_vector_tuning()`` for polar heatmap visualization.

Typical Workflow
----------------
1. Compute object-vector field from spike times and behavioral data::

    >>> result = compute_object_vector_field(  # doctest: +SKIP
    ...     spike_times, times, positions, headings, object_positions
    ... )

2. Access the field and egocentric environment::

    >>> field = result.field  # Firing rate per egocentric bin  # doctest: +SKIP
    >>> ego_env = result.ego_env  # Egocentric polar environment

3. Compute tuning metrics and classify::

    >>> metrics = compute_object_vector_tuning(  # doctest: +SKIP
    ...     spike_times, times, positions, headings, object_positions, env
    ... )
    >>> if is_object_vector_cell(metrics.tuning_curve, metrics.peak_rate):
    ...     print(f"OVC! Preferred distance: {metrics.preferred_distance:.1f}")

4. Visualize::

    >>> plot_object_vector_tuning(  # doctest: +SKIP
    ...     metrics.tuning_curve, metrics.distance_bins, metrics.direction_bins
    ... )

Coordinate Conventions
----------------------
**Egocentric direction**:
- 0 radians = object is directly ahead of animal
- pi/2 radians = object is to the left
- -pi/2 radians = object is to the right
- +/-pi radians = object is behind

This matches the coordinate convention in ``neurospatial.ops.egocentric``.

References
----------
Hoydal, O. A., et al. (2019). Object-vector coding in the medial entorhinal
    cortex. Nature, 568(7752), 400-404.
Deshmukh, S. S., & Knierim, J. J. (2011). Representation of non-spatial and
    spatial information in the lateral entorhinal cortex. Frontiers in
    Behavioral Neuroscience, 5, 69.

See Also
--------
neurospatial.ops.egocentric : Egocentric coordinate transforms
neurospatial.Environment.from_polar_egocentric : Egocentric polar environment
"""

# Re-export from object_vector_field.py
# Re-export from metrics/object_vector_cells.py
from neurospatial.metrics.object_vector_cells import (
    ObjectVectorMetrics,
    compute_object_vector_tuning,
    is_object_vector_cell,
    object_vector_score,
    plot_object_vector_tuning,
)
from neurospatial.object_vector_field import (
    ObjectVectorFieldResult,
    compute_object_vector_field,
)

__all__ = [  # noqa: RUF022 - organized by source module
    # From object_vector_field.py
    "ObjectVectorFieldResult",
    "compute_object_vector_field",
    # From metrics/object_vector_cells.py
    "ObjectVectorMetrics",
    "compute_object_vector_tuning",
    "is_object_vector_cell",
    "object_vector_score",
    "plot_object_vector_tuning",
]
