"""Path efficiency metrics for spatial navigation analysis.

.. deprecated::
    This module is deprecated. Import from ``neurospatial.behavior.navigation`` instead.
    This module will be removed in a future release.

For new code, use::

    from neurospatial.behavior.navigation import (
        PathEfficiencyResult,
        SubgoalEfficiencyResult,
        traveled_path_length,
        shortest_path_length,
        path_efficiency,
        time_efficiency,
        angular_efficiency,
        subgoal_efficiency,
        compute_path_efficiency,
    )
"""

# Re-export from new location for backward compatibility
from neurospatial.behavior.navigation import (
    PathEfficiencyResult,
    SubgoalEfficiencyResult,
    angular_efficiency,
    compute_path_efficiency,
    path_efficiency,
    shortest_path_length,
    subgoal_efficiency,
    time_efficiency,
    traveled_path_length,
)

__all__ = [
    "PathEfficiencyResult",
    "SubgoalEfficiencyResult",
    "angular_efficiency",
    "compute_path_efficiency",
    "path_efficiency",
    "shortest_path_length",
    "subgoal_efficiency",
    "time_efficiency",
    "traveled_path_length",
]
