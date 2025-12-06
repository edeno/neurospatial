"""Goal-directed navigation metrics.

.. deprecated::
    This module is deprecated. Import from ``neurospatial.behavior.navigation`` instead.
    This module will be removed in a future release.

For new code, use::

    from neurospatial.behavior.navigation import (
        GoalDirectedMetrics,
        goal_vector,
        goal_direction,
        instantaneous_goal_alignment,
        goal_bias,
        approach_rate,
        compute_goal_directed_metrics,
    )
"""

# Re-export from new location for backward compatibility
from neurospatial.behavior.navigation import (
    GoalDirectedMetrics,
    approach_rate,
    compute_goal_directed_metrics,
    goal_bias,
    goal_direction,
    goal_vector,
    instantaneous_goal_alignment,
)

__all__ = [
    "GoalDirectedMetrics",
    "approach_rate",
    "compute_goal_directed_metrics",
    "goal_bias",
    "goal_direction",
    "goal_vector",
    "instantaneous_goal_alignment",
]
