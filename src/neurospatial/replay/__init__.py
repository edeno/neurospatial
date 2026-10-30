"""Replay analysis: score decoded candidate events and test them against shuffles.

A candidate replay event is decoded first, with
:func:`neurospatial.decoding.decode_position`. The trajectory scores take that
posterior and ask whether it describes a trajectory; the shuffles build null
events from the posterior, the spike counts or the rate maps. Detecting the
candidate events themselves (sharp-wave ripples, population bursts) is not
part of this subpackage.

Trajectory Scores
-----------------
detect_trajectory_radon : Replay line fit (Davidson et al. 2009)
    The constant-speed line holding the most posterior mass within a band.

fit_linear_trajectory : Linear trajectory fitting
    Linear regression with optional Monte Carlo uncertainty.

fit_isotonic_trajectory : Monotonic trajectory fitting
    Isotonic regression on posterior sequence.

Significance
------------
replay_trajectory_significance : Replay line significance
    Test the replay score against column-cycle, unit-identity and
    pseudo-event shuffles.

Shuffles
--------
shuffle_cell_identity : Permute which spike train goes with which rate map
shuffle_place_fields_circular : Circularly shift each 1-D rate map
shuffle_place_fields_circular_2d : Circularly shift each 2-D rate map
shuffle_posterior_circular : Shift each time bin's posterior along the track
shuffle_posterior_weighted_circular : Posterior shift with edge control

Result Containers
-----------------
RadonDetectionResult, LinearFitResult, IsotonicFitResult
    The fitted trajectory and its score.
ReplaySignificanceResult
    The detected trajectory, one shuffle test per null, and the largest
    p-value.

Monte Carlo p-value helpers and temporal shuffles
(:func:`~neurospatial.stats.shuffle_time_bins`) live in
:mod:`neurospatial.stats`.
"""

from neurospatial.replay.nulls import (
    shuffle_cell_identity,
    shuffle_place_fields_circular,
    shuffle_place_fields_circular_2d,
    shuffle_posterior_circular,
    shuffle_posterior_weighted_circular,
)
from neurospatial.replay.trajectory import (
    IsotonicFitResult,
    LinearFitResult,
    RadonDetectionResult,
    ReplaySignificanceResult,
    detect_trajectory_radon,
    fit_isotonic_trajectory,
    fit_linear_trajectory,
    replay_trajectory_significance,
)

__all__ = [
    # Result containers
    "IsotonicFitResult",
    "LinearFitResult",
    "RadonDetectionResult",
    "ReplaySignificanceResult",
    # Trajectory scores
    "detect_trajectory_radon",
    "fit_isotonic_trajectory",
    "fit_linear_trajectory",
    # Significance
    "replay_trajectory_significance",
    # Shuffles
    "shuffle_cell_identity",
    "shuffle_place_fields_circular",
    "shuffle_place_fields_circular_2d",
    "shuffle_posterior_circular",
    "shuffle_posterior_weighted_circular",
]
