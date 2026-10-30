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

fit_linear_trajectory : Line fit (Karlsson & Frank 2009; Wu et al. 2017)
    R² of a line through positions sampled from the posterior, or through
    the MAP positions.

regression_line_score : Band mass under the MAP regression line (Harvey et al.
    2023; Soula et al. 2023)
    Posterior mass within a band of the least-squares line through each
    time bin's most likely position.
fit_isotonic_trajectory : Monotonic fit (Michon et al. 2019)
    Weighted isotonic fit of the MAP positions, scored by the mean MAP
    probability times R².

fit_circular_sequence : Circular-track fit (Wikenheiser & Redish 2013)
    R² of a line through the cumulative decoded path around a loop, from
    each time bin's circular mean, so sequences that cross the start of the
    loop's linear position are scored whole.

Criteria Metrics
----------------
trajectory_metrics : Per-event metrics thresholded by criteria-based detection
    MAP and posterior-mean jumps, spread, range, path length, dispersion,
    coverage, peak probability and empty time bins, each named for the
    papers that threshold it.

template_fractions : Share of a joint decode in each template
    Directional fraction of Ambrose et al. (2016), with optional masking of
    the track ends (Mallory et al. 2025).

replay_segments : The part of an event that is a smooth trajectory
    Split at jumps, reject blurred or weak time bins, bridge gaps, merge
    and filter segments; parameters reproduce Pfeiffer & Foster (2013,
    2015), Igata et al. (2021), Carey et al. (2019), Widloski & Foster
    (2022) and Mallory et al. (2025).

Rank Order
----------
rank_order_correlation : Field order vs firing order (Foster & Wilson 2006)
    Spearman correlation between each active cell's field position and when it
    fired; needs no decoding.

rank_order_significance : Rank-order correlation against cell-order shuffles

Template Matching
-----------------
best_sequence_match : Best (x, y) match of an event's letters (Lee & Wilson 2002)
    Bursts of each template cell are letters; the p-value is the share of
    the orderings of the letters that match the field order as well.

sequence_matching_index : Matching index of firing order (Ji & Wilson 2007)
    Pairs of active cells in field order minus pairs in opposite order,
    with its exact p-value.

burst_onset_times, spike_words : Letters and words of Lee & Wilson (2002)
detect_template_match_replay, TemplateMatchMethod : Published template tests
template_match_dataset_test : The papers' data-set tests (binomial counts
    of matches, chance and template-shuffle significance)
recurring_spike_sequences, recurring_sequences_significance : Sequences that
    recur within a session, against surrogate spike trains (Nádasdy et al.
    1999); RecurringSequenceMethod records the test
surrogate_spike_trains : Interval-shuffle, jitter, across-train and
    theta-phase surrogates
pairwise_order_score : Firing order of spike pairs (Gupta et al. 2010)
    Over every pair of spikes less than 80 ms apart, +1 in field order and
    -1 against it, summed; multi-field cells take their best field.

pairwise_order_significance : Pairwise-order score against spike-time and
    cell-identity shuffles

Published Methods
-----------------
detect_replay : Apply one published replay method to one decoded event
compare_replay_methods : Several methods on several events, as a table
ReplayMethod, ReplayCriterion : A method's score, shuffles and thresholds
ReplaySegmentation : How a criteria-based method finds the trajectory segment
detect_rank_order_replay, RankOrderMethod : Published rank-order tests on spikes
detect_pairwise_order_replay, PairwiseOrderMethod : Pairwise firing-order test
    on spikes, every template forward and backward (Gupta et al. 2010)
detect_arm_replay, ArmReplayMethod : Single-arm and joint-arm replay on a maze
event_properties, detection_rates, method_agreement : Calibrate methods on a
    comparison table: rates (false positives on null data) by event
    property, with confidence intervals, and agreement between methods
null_events, simulated_null_events, decode_intervals : Events without replay
    (cell-identity, spike-shift, uniform-spike-time and rate-matched
    Poisson copies of real events, or
    simulated random and stationary events) to measure false positives
matched_alpha, apply_alpha : Each method's alpha that holds its
    false-positive rate on null data to a target, and verdicts at it

Data-Set-Level Tests
--------------------
dataset_values : Per-event values of the real data set or of null data sets
    Each null data set shuffles every event once, measured by the same
    method; ``shared=True`` draws one template shuffle per data set.
dataset_test, CriterionGrid : A data-set statistic against null data sets
    The share of events passing every criterion on a grid of thresholds
    (Silva et al. 2015's significance matrix), or any statistic.
distribution_test : Real events' values against pooled null events' values
    KS, rank-sum or Levene test, the side stated in terms of the values
    (Stella et al. 2019; Xu et al. 2019; Grosmark & Buzsáki 2016).
shuffle_events : Null data sets as decoded events, for nested analyses
relative_rank_template, template_correlation_test : Liu et al. (2019)
    Sleep template from relative ranks, tested against shuffled running
    templates.
DatasetTestResult, DistributionTestResult, ShuffledDataset : Their results
neurospatial.replay.dataset : The papers' data-set tests as recipes
    (SILVA_2015_MATRIX, SILVA_2015_RANK_ORDER, GROSMARK_2016_SURROGATES,
    FAROOQ_2019_TIME_BIN, FAROOQ_2019_POISSON, FAROOQ_2019_SPACE_BIN,
    FAROOQ_2019_WEIGHTED, LIU_2019_TEMPLATES, STELLA_2019_JITTER,
    XU_2019_ROTATION)
neurospatial.replay.methods : The published recipes (FOSTER_WILSON_2006,
    DIBA_BUZSAKI_2007, DAVIDSON_2009, KARLSSON_FRANK_2009, GUPTA_2010,
    LEE_WILSON_2002, JI_WILSON_2007,
    PFEIFFER_FOSTER_2013, WIKENHEISER_REDISH_2013, WU_FOSTER_2014,
    OLAFSDOTTIR_2015, PFEIFFER_FOSTER_2015, SILVA_2015, AMBROSE_2016,
    GROSMARK_2016, JADHAV_2016, OLAFSDOTTIR_2016, WU_2017, OLAFSDOTTIR_2017,
    TANG_2017, YAMAMOTO_TONEGAWA_2017, MABOUDI_2018, CAREY_2019, CHENANI_2019,
    FAROOQ_2019, MICHON_2019, MUESSIG_2019, SHIN_2019, IGATA_2021,
    BERNERS_LEE_2021, BUSH_2022, TIROLE_2022, BERNERS_LEE_2022,
    WIDLOSKI_FOSTER_2022, HARVEY_2023, HARVEY_2023_LINE, LIU_2023,
    SOULA_2023, MALLORY_2025, CHANG_2025, WIDLOSKI_FOSTER_2025) and the
    session-level NADASDY_1999

Significance
------------
replay_significance : Test any replay score against shuffled events
    Eight nulls: column-cycle, unit-identity and pseudo-event (the default,
    Davidson et al. 2009), spike-train and place-field circular shifts,
    time-bin permutation (the "time swap" or "time-bin shuffle" of Silva
    et al. 2015, Grosmark & Buzsáki 2016, Maboudi et al. 2018 and Farooq
    et al. 2019), spike times redrawn uniformly in the event (Stella et
    al. 2019) and the weighted column cycle (Farooq et al. 2019); the
    event is significant when every null run rejects it.

Shuffles
--------
shuffle_cell_identity : Permute which spike train goes with which rate map
shuffle_spike_counts_circular : Shift each cell's counts in time within an event
shuffle_spike_counts_uniform : Spread each cell's spikes uniformly over an event
shuffle_place_fields_circular : Circularly shift each 1-D rate map
shuffle_place_fields_circular_2d : Circularly shift each 2-D rate map
shuffle_posterior_circular : Shift each time bin's posterior along the track
shuffle_posterior_weighted_circular : Shift each time bin's peak to a position
    drawn from the session's mean posterior (Farooq et al. 2019)

Result Containers
-----------------
RadonDetectionResult, LinearFitResult, IsotonicFitResult, WeightedCorrelationResult,
RegressionLineScoreResult
    The fitted trajectory or correlation, and its ``score``.
PairwiseOrderResult, PairwiseOrderSignificanceResult
    A pairwise-order score with its scored cells and assigned fields, and its
    shuffle tests.
CircularSequenceResult
    The circular-track fit: replay speed around the loop and its ``score``.
TrajectoryMetrics
    Per-time-bin arrays and per-event scalars of one event.
ReplaySegmentsResult
    The segments of one event, their durations and distances, and the
    longest.
ReplaySignificanceResult
    The event's score, one shuffle test per null, and the largest p-value.

Monte Carlo p-value helpers and temporal shuffles
(:func:`~neurospatial.stats.shuffle_time_bins`) live in
:mod:`neurospatial.stats`.
"""

from neurospatial.replay.arms import ArmReplayMethod, detect_arm_replay
from neurospatial.replay.calibration import (
    NullSpikeData,
    apply_alpha,
    decode_intervals,
    detection_rates,
    event_properties,
    matched_alpha,
    method_agreement,
    null_events,
    simulated_null_events,
)
from neurospatial.replay.circular import (
    CircularSequenceResult,
    fit_circular_sequence,
)
from neurospatial.replay.correlation import (
    WeightedCorrelationResult,
    weighted_correlation,
)
from neurospatial.replay.criteria import (
    TrajectoryMetrics,
    template_fractions,
    trajectory_metrics,
)
from neurospatial.replay.dataset import (
    CriterionGrid,
    DatasetTestResult,
    DistributionTestResult,
    ShuffledDataset,
    dataset_test,
    dataset_values,
    distribution_test,
    relative_rank_template,
    shuffle_events,
    template_correlation_test,
)
from neurospatial.replay.methods import (
    RankOrderMethod,
    ReplayCriterion,
    ReplayDetectionResult,
    ReplayMethod,
    ReplaySegmentation,
    compare_replay_methods,
    detect_rank_order_replay,
    detect_replay,
)
from neurospatial.replay.nulls import (
    shuffle_cell_identity,
    shuffle_place_fields_circular,
    shuffle_place_fields_circular_2d,
    shuffle_posterior_circular,
    shuffle_posterior_weighted_circular,
    shuffle_spike_counts_circular,
    shuffle_spike_counts_uniform,
)
from neurospatial.replay.pairwise_order import (
    PairwiseOrderMethod,
    PairwiseOrderResult,
    PairwiseOrderSignificanceResult,
    detect_pairwise_order_replay,
    pairwise_order_score,
    pairwise_order_significance,
)
from neurospatial.replay.rank_order import (
    RankOrderResult,
    RankOrderSignificanceResult,
    rank_order_correlation,
    rank_order_significance,
)
from neurospatial.replay.recurring_sequences import (
    RecurringSequenceMethod,
    RecurringSequencesResult,
    RecurringSequencesSignificanceResult,
    recurring_sequences_significance,
    recurring_spike_sequences,
    surrogate_spike_trains,
)
from neurospatial.replay.segments import ReplaySegmentsResult, replay_segments
from neurospatial.replay.sequence_match import (
    MatchingIndexResult,
    SequenceMatchResult,
    SpikeWordsResult,
    best_sequence_match,
    burst_onset_times,
    sequence_matching_index,
    spike_words,
)
from neurospatial.replay.significance import (
    ReplaySignificanceResult,
    replay_significance,
)
from neurospatial.replay.template_match import (
    TemplateMatchDatasetResult,
    TemplateMatchMethod,
    detect_template_match_replay,
    template_match_dataset_test,
)
from neurospatial.replay.trajectory import (
    IsotonicFitResult,
    LinearFitResult,
    RadonDetectionResult,
    RegressionLineScoreResult,
    detect_trajectory_radon,
    fit_isotonic_trajectory,
    fit_linear_trajectory,
    regression_line_score,
)

__all__ = [  # noqa: RUF022 (organized by category, not alphabetically)
    # Published methods
    "ArmReplayMethod",
    "PairwiseOrderMethod",
    "ReplayCriterion",
    "ReplayDetectionResult",
    "RankOrderMethod",
    "ReplayMethod",
    "ReplaySegmentation",
    "compare_replay_methods",
    "detect_arm_replay",
    "detect_pairwise_order_replay",
    "detect_rank_order_replay",
    "RecurringSequenceMethod",
    "TemplateMatchMethod",
    "detect_template_match_replay",
    "template_match_dataset_test",
    # Calibration and comparison
    "NullSpikeData",
    "apply_alpha",
    "decode_intervals",
    "detection_rates",
    "event_properties",
    "matched_alpha",
    "method_agreement",
    "null_events",
    "simulated_null_events",
    "detect_replay",
    # Data-set-level tests
    "CriterionGrid",
    "DatasetTestResult",
    "DistributionTestResult",
    "ShuffledDataset",
    "dataset_test",
    "dataset_values",
    "distribution_test",
    "relative_rank_template",
    "shuffle_events",
    "template_correlation_test",
    # Result containers
    "CircularSequenceResult",
    "IsotonicFitResult",
    "LinearFitResult",
    "MatchingIndexResult",
    "PairwiseOrderResult",
    "PairwiseOrderSignificanceResult",
    "RadonDetectionResult",
    "RankOrderResult",
    "RankOrderSignificanceResult",
    "RecurringSequencesResult",
    "RegressionLineScoreResult",
    "RecurringSequencesSignificanceResult",
    "ReplaySegmentsResult",
    "ReplaySignificanceResult",
    "SequenceMatchResult",
    "SpikeWordsResult",
    "TemplateMatchDatasetResult",
    "TrajectoryMetrics",
    "WeightedCorrelationResult",
    # Trajectory scores
    "detect_trajectory_radon",
    "fit_circular_sequence",
    "fit_isotonic_trajectory",
    "fit_linear_trajectory",
    "regression_line_score",
    "weighted_correlation",
    # Criteria metrics
    "template_fractions",
    "trajectory_metrics",
    "replay_segments",
    # Significance
    "replay_significance",
    # Rank order
    "rank_order_correlation",
    "rank_order_significance",
    # Template matching
    "best_sequence_match",
    "burst_onset_times",
    "sequence_matching_index",
    "spike_words",
    "recurring_sequences_significance",
    "recurring_spike_sequences",
    "surrogate_spike_trains",
    # Pairwise order
    "pairwise_order_score",
    "pairwise_order_significance",
    # Shuffles
    "shuffle_cell_identity",
    "shuffle_place_fields_circular",
    "shuffle_place_fields_circular_2d",
    "shuffle_posterior_circular",
    "shuffle_posterior_weighted_circular",
    "shuffle_spike_counts_circular",
    "shuffle_spike_counts_uniform",
]
