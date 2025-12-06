"""Spatial view cell analysis.

Spatial view cells fire when an animal *views* a specific location, regardless
of where the animal is positioned. This module provides tools for computing
spatial view fields and metrics for classifying spatial view cells.

This module re-exports functions from:
- ``neurospatial.spatial_view_field``: Field computation
- ``neurospatial.metrics.spatial_view_cells``: Metrics and classification
- ``neurospatial.ops.visibility``: Visibility utilities (for convenience)

Import Paths
------------
After package reorganization (v0.4.0+), use::

    from neurospatial.encoding.spatial_view import (
        # Field computation
        SpatialViewFieldResult,
        compute_spatial_view_field,
        # Metrics and classification
        SpatialViewMetrics,
        spatial_view_cell_metrics,
        is_spatial_view_cell,
        # Visibility utilities (re-exports)
        compute_viewed_location,
        compute_viewshed,
        visibility_occupancy,
        FieldOfView,
    )

    # Or import from encoding package
    from neurospatial.encoding import (
        compute_spatial_view_field,
        spatial_view_cell_metrics,
    )

Typical Workflow
----------------
1. Compute spatial view field from spike times and behavioral data::

    >>> result = compute_spatial_view_field(  # doctest: +SKIP
    ...     env, spike_times, times, positions, headings
    ... )

2. Compute metrics and classify as spatial view cell::

    >>> metrics = spatial_view_cell_metrics(  # doctest: +SKIP
    ...     env, spike_times, times, positions, headings
    ... )
    >>> if metrics.is_spatial_view_cell:  # doctest: +SKIP
    ...     print("Spatial view cell detected!")

3. Or use quick classifier for screening::

    >>> if is_spatial_view_cell(env, spike_times, times, positions, headings):
    ...     print("Spatial view cell!")  # doctest: +SKIP

Key Difference: Place Cells vs Spatial View Cells
-------------------------------------------------
- **Place cell**: Fires when animal is *at* a specific location
- **Spatial view cell**: Fires when animal is *looking at* a specific location

For place cells, both place field and view field are similar (because viewing
location correlates with position). For spatial view cells, the view field
has higher spatial information than the place field.

See Also
--------
neurospatial.encoding.place : Place cell analysis
neurospatial.ops.visibility : Visibility and gaze computation
"""

# Field computation from spatial_view_field.py
# Metrics and classification from metrics/spatial_view_cells.py
from neurospatial.metrics.spatial_view_cells import (
    SpatialViewMetrics,
    is_spatial_view_cell,
    spatial_view_cell_metrics,
)

# Re-exports from ops.visibility for convenience in SVC workflow
from neurospatial.ops.visibility import (
    FieldOfView,
    compute_viewed_location,
    compute_viewshed,
    visibility_occupancy,
)
from neurospatial.spatial_view_field import (
    SpatialViewFieldResult,
    compute_spatial_view_field,
)

__all__ = [  # noqa: RUF022 - organized by category
    # Field computation
    "SpatialViewFieldResult",
    "compute_spatial_view_field",
    # Metrics and classification
    "SpatialViewMetrics",
    "spatial_view_cell_metrics",
    "is_spatial_view_cell",
    # Re-exports from ops.visibility
    "compute_viewed_location",
    "compute_viewshed",
    "visibility_occupancy",
    "FieldOfView",
]
