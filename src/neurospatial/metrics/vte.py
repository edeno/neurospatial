"""Vicarious Trial and Error (VTE) detection and analysis.

.. deprecated::
    This module is deprecated. Import from ``neurospatial.behavior.decisions`` instead.
    This module will be removed in a future release.

For new code, use::

    from neurospatial.behavior.decisions import (
        VTETrialResult,
        VTESessionResult,
        compute_vte_index,
        compute_vte_trial,
        compute_vte_session,
        classify_vte,
        head_sweep_from_positions,
        head_sweep_magnitude,
        integrated_absolute_rotation,
        normalize_vte_scores,
        wrap_angle,
    )
"""

# Re-export from new location for backward compatibility
from neurospatial.behavior.decisions import (
    VTESessionResult,
    VTETrialResult,
    classify_vte,
    compute_vte_index,
    compute_vte_session,
    compute_vte_trial,
    head_sweep_from_positions,
    head_sweep_magnitude,
    integrated_absolute_rotation,
    normalize_vte_scores,
    wrap_angle,
)

__all__ = [
    "VTESessionResult",
    "VTETrialResult",
    "classify_vte",
    "compute_vte_index",
    "compute_vte_session",
    "compute_vte_trial",
    "head_sweep_from_positions",
    "head_sweep_magnitude",
    "integrated_absolute_rotation",
    "normalize_vte_scores",
    "wrap_angle",
]
