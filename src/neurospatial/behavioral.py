"""Behavioral analysis and goal-directed navigation metrics.

.. deprecated::
    This module is deprecated. Import from ``neurospatial.behavior.navigation`` instead.
    This module will be removed in a future release.

For new code, use::

    from neurospatial.behavior.navigation import (
        path_progress,
        distance_to_region,
        cost_to_goal,
        time_to_goal,
        trials_to_region_arrays,
        graph_turn_sequence,
        goal_pair_direction_labels,
        heading_direction_labels,
    )

Or import from the behavior module::

    from neurospatial.behavior import path_progress, distance_to_region
"""

# Re-export from new location for backward compatibility
from neurospatial.behavior.navigation import (
    cost_to_goal,
    distance_to_region,
    goal_pair_direction_labels,
    graph_turn_sequence,
    heading_direction_labels,
    path_progress,
    time_to_goal,
    trials_to_region_arrays,
)
from neurospatial.behavior.trajectory import (
    compute_trajectory_curvature,
)

__all__ = [
    "compute_trajectory_curvature",
    "cost_to_goal",
    "distance_to_region",
    "goal_pair_direction_labels",
    "graph_turn_sequence",
    "heading_direction_labels",
    "path_progress",
    "time_to_goal",
    "trials_to_region_arrays",
]
