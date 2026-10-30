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

weighted_correlation : Weighted correlation of time and position
    Pearson correlation over the posterior (Wu & Foster 2014).

fit_linear_trajectory : Linear trajectory fitting
    Linear regression with optional Monte Carlo uncertainty.

fit_isotonic_trajectory : Monotonic trajectory fitting
    Isotonic regression on posterior sequence.

Rank Order
----------
rank_order_correlation : Field order vs firing order (Foster & Wilson 2006)
    Spearman correlation between each active cell's field position and when it
    fired; needs no decoding.

rank_order_significance : Rank-order correlation against cell-order shuffles

Significance
------------
replay_significance : Test any replay score against shuffled events
    Column-cycle, unit-identity and pseudo-event nulls; the event is
    significant when every null rejects it.

Shuffles
--------
shuffle_cell_identity : Permute which spike train goes with which rate map
shuffle_spike_counts_circular : Shift each cell's counts in time within an event
shuffle_place_fields_circular : Circularly shift each 1-D rate map
shuffle_place_fields_circular_2d : Circularly shift each 2-D rate map
shuffle_posterior_circular : Shift each time bin's posterior along the track
shuffle_posterior_weighted_circular : Posterior shift with edge control

Result Containers
-----------------
RadonDetectionResult, LinearFitResult, IsotonicFitResult, WeightedCorrelationResult
    The fitted trajectory or correlation, and its ``score``.
ReplaySignificanceResult
    The event's score, one shuffle test per null, and the largest p-value.

Monte Carlo p-value helpers and temporal shuffles
(:func:`~neurospatial.stats.shuffle_time_bins`) live in
:mod:`neurospatial.stats`.
"""

from neurospatial.replay.correlation import (
    WeightedCorrelationResult,
    weighted_correlation,
)
from neurospatial.replay.nulls import (
    shuffle_cell_identity,
    shuffle_place_fields_circular,
    shuffle_place_fields_circular_2d,
    shuffle_posterior_circular,
    shuffle_posterior_weighted_circular,
    shuffle_spike_counts_circular,
)
from neurospatial.replay.rank_order import (
    RankOrderResult,
    RankOrderSignificanceResult,
    rank_order_correlation,
    rank_order_significance,
)
from neurospatial.replay.significance import (
    ReplaySignificanceResult,
    replay_significance,
)
from neurospatial.replay.trajectory import (
    IsotonicFitResult,
    LinearFitResult,
    RadonDetectionResult,
    detect_trajectory_radon,
    fit_isotonic_trajectory,
    fit_linear_trajectory,
)

__all__ = [  # noqa: RUF022 (organized by category, not alphabetically)
    # Result containers
    "IsotonicFitResult",
    "LinearFitResult",
    "RadonDetectionResult",
    "RankOrderResult",
    "RankOrderSignificanceResult",
    "ReplaySignificanceResult",
    "WeightedCorrelationResult",
    # Trajectory scores
    "detect_trajectory_radon",
    "fit_isotonic_trajectory",
    "fit_linear_trajectory",
    "weighted_correlation",
    # Significance
    "replay_significance",
    # Rank order
    "rank_order_correlation",
    "rank_order_significance",
    # Shuffles
    "shuffle_cell_identity",
    "shuffle_place_fields_circular",
    "shuffle_place_fields_circular_2d",
    "shuffle_posterior_circular",
    "shuffle_posterior_weighted_circular",
    "shuffle_spike_counts_circular",
]
