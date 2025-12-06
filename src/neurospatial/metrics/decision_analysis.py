"""Spatial decision analysis at choice points.

.. deprecated::
    This module is deprecated. Import from ``neurospatial.behavior.decisions`` instead.
    This module will be removed in a future release.

For new code, use::

    from neurospatial.behavior.decisions import (
        PreDecisionMetrics,
        DecisionBoundaryMetrics,
        DecisionAnalysisResult,
        compute_decision_analysis,
        compute_pre_decision_metrics,
        decision_region_entry_time,
        detect_boundary_crossings,
        distance_to_decision_boundary,
        extract_pre_decision_window,
        geodesic_voronoi_labels,
        pre_decision_heading_stats,
        pre_decision_speed_stats,
    )
"""

# Re-export from new location for backward compatibility
from neurospatial.behavior.decisions import (
    DecisionAnalysisResult,
    DecisionBoundaryMetrics,
    PreDecisionMetrics,
    compute_decision_analysis,
    compute_pre_decision_metrics,
    decision_region_entry_time,
    detect_boundary_crossings,
    distance_to_decision_boundary,
    extract_pre_decision_window,
    geodesic_voronoi_labels,
    pre_decision_heading_stats,
    pre_decision_speed_stats,
)

__all__ = [
    "DecisionAnalysisResult",
    "DecisionBoundaryMetrics",
    "PreDecisionMetrics",
    "compute_decision_analysis",
    "compute_pre_decision_metrics",
    "decision_region_entry_time",
    "detect_boundary_crossings",
    "distance_to_decision_boundary",
    "extract_pre_decision_window",
    "geodesic_voronoi_labels",
    "pre_decision_heading_stats",
    "pre_decision_speed_stats",
]
