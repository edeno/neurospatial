"""Published replay detection methods as explicit, comparable recipes.

Each :class:`ReplayMethod` records one paper's replay test: the score, the
shuffles and significance level, how time bins without spikes are handled,
how the event is cut into trajectory segments (:class:`ReplaySegmentation`),
and the thresholds an event must meet. :func:`detect_replay` applies a method
to one decoded event and :func:`compare_replay_methods` applies several
methods to several events, so the same data can be scored the way each lab
scored it. :class:`RankOrderMethod` records a rank-order test, which scores
the event's spikes rather than a decoded posterior
(:func:`detect_rank_order_replay`),
:class:`~neurospatial.replay.PairwiseOrderMethod` the pairwise firing-order
test of Gupta et al. (2010) on the event's spikes
(:func:`~neurospatial.replay.detect_pairwise_order_replay`),
:class:`~neurospatial.replay.BiasSequenceMethod` the pairwise-bias sequence
test of Liu et al. (2023)
(:func:`~neurospatial.replay.detect_bias_sequence_replay`), and
:class:`~neurospatial.replay.ArmReplayMethod` a test for replay of single
arms and pairs of arms on a maze.

A recipe records the paper's decoding settings (time-bin and position-bin
sizes) for reference but does not decode: methods are compared on the same
decoded posterior, so differences come from the replay test alone. Choices a
paper does not specify, and bugs in the authors' code that are not
reproduced, are listed in each recipe's ``notes``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import operator
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._constants import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE
from neurospatial._exceptions import _format_error
from neurospatial._results import ResultMixin, reduce_through_constructor
from neurospatial.replay.arms import ArmReplayMethod, detect_arm_replay
from neurospatial.replay.circular import fit_circular_sequence
from neurospatial.replay.correlation import weighted_correlation
from neurospatial.replay.pairwise_order import (
    PairwiseOrderMethod,
    _FieldSearchTooLargeError,
    detect_pairwise_order_replay,
)
from neurospatial.replay.recurring_sequences import NADASDY_1999
from neurospatial.replay.sequence_bias import (
    BiasSequenceMethod,
    detect_bias_sequence_replay,
)
from neurospatial.replay.significance import (
    _ALTERNATIVES,
    _COUNT_CHANGING_NULLS,
    _DECODING_NULLS,
    _NULLS,
    _ROW_MOVING_NULLS,
    _checked_null_options,
)
from neurospatial.replay.template_match import (
    JI_WILSON_2007,
    LEE_WILSON_2002,
    TemplateMatchMethod,
    detect_template_match_replay,
)
from neurospatial.replay.trajectory import (
    _SCORE_DECIMALS,
    detect_trajectory_radon,
    fit_isotonic_trajectory,
    fit_linear_trajectory,
    regression_line_score,
)

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial.decoding import DecodingResult, JointDecodingResult
    from neurospatial.environment import Environment
    from neurospatial.replay.segments import ReplaySegmentsResult

__all__ = [
    "AMBROSE_2016",
    "BERNERS_LEE_2021",
    "BERNERS_LEE_2022",
    "BUSH_2022",
    "CAREY_2019",
    "CHANG_2025",
    "CHENANI_2019",
    "DAVIDSON_2009",
    "DIBA_BUZSAKI_2007",
    "FAROOQ_2019",
    "FOSTER_WILSON_2006",
    "GROSMARK_2016",
    "GUPTA_2010",
    "HARVEY_2023",
    "HARVEY_2023_LINE",
    "IGATA_2021",
    "JADHAV_2016",
    "JI_WILSON_2007",
    "KARLSSON_FRANK_2009",
    "LEE_WILSON_2002",
    "LIU_2023",
    "LIU_2023_BIAS",
    "MABOUDI_2018",
    "MALLORY_2025",
    "MICHON_2019",
    "MUESSIG_2019",
    "NADASDY_1999",
    "OLAFSDOTTIR_2015",
    "OLAFSDOTTIR_2016",
    "OLAFSDOTTIR_2017",
    "PFEIFFER_FOSTER_2013",
    "PFEIFFER_FOSTER_2015",
    "PUBLISHED_METHODS",
    "SHIN_2019",
    "SILVA_2015",
    "SOULA_2023",
    "TANG_2017",
    "TIROLE_2022",
    "WIDLOSKI_FOSTER_2022",
    "WIDLOSKI_FOSTER_2025",
    "WIKENHEISER_REDISH_2013",
    "WU_2017",
    "WU_FOSTER_2014",
    "YAMAMOTO_TONEGAWA_2017",
    "RankOrderMethod",
    "ReplayCriterion",
    "ReplayDetectionResult",
    "ReplayMethod",
    "ReplaySegmentation",
    "compare_replay_methods",
    "detect_rank_order_replay",
    "detect_replay",
]

_OPERATION = "detect_replay()"
# Criterion quantities read from the tested segment of ReplaySegmentsResult.
_SEGMENT_QUANTITIES = {
    "segment_duration": "durations",
    "segment_time_bin_count": "time_bin_counts",
    "segment_dispersion": "dispersions",
    "segment_start_to_end_distance": "start_to_end_distances",
    "segment_path_length": "path_lengths",
}
# The scale each ReplaySegmentation option may be given relative to.
_SEGMENT_OPTION_SCALES = {
    "max_jump": "track_length",
    "max_spread": "track_length",
    "merge_max_distance": "track_length",
    "min_dispersion": "track_length",
    "min_peak_probability": "chance",
}
# Criterion quantities that compare the templates of a joint decode.
_TEMPLATE_QUANTITIES = ("template_fraction", "segment_template_fraction")
_TEMPLATE_CHOICES = (None, "most_posterior", "sum", "smallest_p", "max_score")
_COMPARISONS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


# TrajectoryMetrics scalars a shuffle does not change, or that need spike counts.
_UNTESTED_QUANTITIES = ("n_time_bins", "duration", "track_length", "empty_fraction")


def _testable_quantities() -> tuple[str, ...]:
    """The quantities ``ReplayMethod.tested_quantities`` can test against nulls."""
    from typing import get_type_hints

    from neurospatial.replay.criteria import TrajectoryMetrics

    hints = get_type_hints(TrajectoryMetrics)
    return tuple(
        item.name
        for item in dataclasses.fields(TrajectoryMetrics)
        if hints[item.name] in (int, float) and item.name not in _UNTESTED_QUANTITIES
    )


def _criterion_quantities() -> tuple[str, ...]:
    """The quantities a criterion of a :class:`ReplayMethod` can test."""
    from typing import get_type_hints

    from neurospatial.replay.criteria import TrajectoryMetrics

    hints = get_type_hints(TrajectoryMetrics)
    scalars = [
        item.name
        for item in dataclasses.fields(TrajectoryMetrics)
        if hints[item.name] in (int, float)
    ]
    return (
        "score",
        "n_active_cells",
        "active_fraction",
        "n_spikes",
        *_SEGMENT_QUANTITIES,
        *_TEMPLATE_QUANTITIES,
        *scalars,
    )


@dataclass(frozen=True)
class ReplayCriterion:
    """One threshold an event must meet to count as replay.

    Attributes
    ----------
    quantity : str
        ``"score"`` (the method's score), ``"n_active_cells"`` (cells with a
        spike anywhere in the event, also for a method with a
        segmentation), ``"active_fraction"`` (``n_active_cells`` over the
        columns of ``spike_counts``), ``"n_spikes"`` (the spikes of all
        cells in the event, before any time bins are dropped; Wikenheiser &
        Redish 2013 required at least 5), a scalar attribute (not a method)
        of :class:`~neurospatial.replay.TrajectoryMetrics`, such as
        ``"max_map_jump"``, or, for a method with a
        :class:`ReplaySegmentation`, a value of the tested segment:
        ``"segment_duration"``, ``"segment_time_bin_count"``,
        ``"segment_dispersion"``, ``"segment_start_to_end_distance"`` or
        ``"segment_path_length"`` (see
        :class:`~neurospatial.replay.ReplaySegmentsResult`). For an event
        given as a joint decode, ``"template_fraction"`` is the tested
        template's share of the event's posterior, leaving out the method's
        ``exclude_end_distance`` at each track end (see
        :func:`~neurospatial.replay.template_fractions`), and
        ``"segment_template_fraction"`` its share over the tested segment's
        trajectory time bins, with no ends left out.
    comparison : {"<", "<=", ">", ">="}
        How the value is compared with the threshold, as the paper states it.
    threshold : float
        The threshold, in environment units for distances unless
        ``relative_to`` is set.
    relative_to : {"track_length"} or None
        Scale the threshold by the event's track length, for thresholds the
        paper gives as a fraction of the track.
    absolute : bool
        Compare the absolute value, for signed quantities such as the
        weighted correlation.
    """

    quantity: str
    comparison: Literal["<", "<=", ">", ">="]
    threshold: float
    relative_to: Literal["track_length"] | None = None
    absolute: bool = False

    def __post_init__(self) -> None:
        """Refuse unknown comparisons and scales, and non-numeric thresholds."""
        threshold = self.threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float, np.integer, np.floating))
            or not np.isfinite(threshold)
        ):
            raise ValueError(
                _format_error(
                    f"ReplayCriterion {self.quantity!r}: threshold must be a "
                    f"finite number, got {threshold!r}.",
                    why="an event is compared with the threshold, and a NaN "
                    "threshold fails every event",
                    fix="give the paper's threshold as a number, such as 0.5",
                )
            )
        if not isinstance(self.absolute, (bool, np.bool_)):
            raise ValueError(
                f"ReplayCriterion {self.quantity!r}: absolute must be True or "
                f"False, got {self.absolute!r}."
            )
        if self.comparison not in _COMPARISONS:
            raise ValueError(
                f"ReplayCriterion: comparison must be one of "
                f"{list(_COMPARISONS)}, got {self.comparison!r}."
            )
        if self.relative_to not in (None, "track_length"):
            raise ValueError(
                f"ReplayCriterion: relative_to must be None or 'track_length', "
                f"got {self.relative_to!r}."
            )

    @property
    def label(self) -> str:
        """A readable form, such as ``"|score| > 0.6"``."""
        name = f"|{self.quantity}|" if self.absolute else self.quantity
        scale = " x track_length" if self.relative_to else ""
        return f"{name} {self.comparison} {self.threshold:g}{scale}"

    def evaluate(self, value: float, track_length: float) -> bool:
        """Whether ``value`` meets this criterion; False when either is NaN.

        A value within :math:`10^{-9}` (absolute plus relative) of the
        threshold counts as equal to it, as in
        :func:`~neurospatial.replay.replay_segments`, so that rounding in
        times such as ``t0 + 0.005 * k`` does not flip a strict test.
        """
        threshold = self.threshold * (track_length if self.relative_to else 1.0)
        value = abs(value) if self.absolute else value
        if not (np.isfinite(value) and np.isfinite(threshold)):
            return False
        if abs(value - threshold) <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(
            threshold
        ):
            value = threshold
        return bool(_COMPARISONS[self.comparison](value, threshold))


@dataclass(frozen=True)
class ReplaySegmentation:
    """How a method cuts an event into trajectory segments before testing it.

    Criteria-based methods keep only the part of a candidate event in which
    the decoded position moves in small steps. The segment a method tests
    is found with :func:`~neurospatial.replay.replay_segments`; its score,
    shuffles and criteria then apply to that segment's rows.

    Attributes
    ----------
    options : Mapping[str, Any]
        Keyword arguments of :func:`~neurospatial.replay.replay_segments`,
        such as ``{"max_jump": 20.0, "min_time_bins": 11}``. ``max_jump`` is
        required, and ``valid_time_bins`` is passed to
        :func:`detect_replay` instead. Distances are in the method's
        ``units``.
    relative_to : Mapping[str, {"track_length", "chance"}]
        Options given as a multiple of a property of the environment.
        ``"track_length"`` multiplies a distance option (``max_jump``,
        ``max_spread``, ``merge_max_distance``, ``min_dispersion``) by the
        track's length; ``"chance"`` multiplies ``min_peak_probability`` by
        ``1 / n``, the mass per bin of a uniform posterior on ``n`` equal
        bins decoded together (Igata et al. 2021: a MAP above 5 times
        chance). ``n`` is ``env.n_bins`` for a decode on one set of rate
        maps, and every template's bins together for one template's share
        of a joint decode.
    choose : {"longest", "any"}
        Which segment is tested. ``"longest"`` tests the longest segment,
        as papers that truncate the event to its longest sequence and then
        test it (Pfeiffer & Foster 2013, 2015; Igata et al. 2021).
        ``"any"`` tests the longest of the segments that meet the method's
        ``segment_*`` criteria, so the event is replay when any segment
        meets them (Carey et al. 2019; Widloski & Foster 2022). Without a
        segment that meets them, the longest is reported.
    min_active_cells_per_time_bin : int or None
        Time bins in which fewer cells spike cannot be on a trajectory
        (Carey et al. 2019: 4). They are passed to ``replay_segments`` as
        invalid time bins, so :func:`detect_replay` needs ``spike_counts``.
    rows : {"segment", "trajectory"}
        Rows of the tested segment that the score, shuffles and
        trajectory-metric criteria use. ``"segment"`` uses every row the
        segment spans, including rows skipped or merged across;
        ``"trajectory"`` uses only its trajectory time bins
        (``ReplaySegmentsResult.trajectory_time_bins``), as the Mallory et
        al. (2025) code does. Times keep their values either way.
    """

    options: Mapping[str, Any]
    relative_to: Mapping[str, Literal["track_length", "chance"]] = field(
        default_factory=dict
    )
    choose: Literal["longest", "any"] = "longest"
    min_active_cells_per_time_bin: int | None = None
    rows: Literal["segment", "trajectory"] = "segment"

    def __post_init__(self) -> None:
        """Freeze the mappings and refuse options replay_segments would refuse."""
        from neurospatial.replay.segments import (
            _checked_options,
            _threshold,
            replay_segments,
        )

        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        object.__setattr__(
            self, "relative_to", MappingProxyType(dict(self.relative_to))
        )
        parameters = inspect.signature(replay_segments).parameters
        accepted = [
            name
            for name, parameter in parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and name != "valid_time_bins"
        ]
        unknown = sorted(set(self.options) - set(accepted))
        if unknown:
            raise ValueError(
                f"ReplaySegmentation: options has {unknown}, which "
                f"replay_segments() does not take here; it takes {accepted}. "
                "Pass valid_time_bins to detect_replay()."
            )
        if "max_jump" not in self.options:
            raise ValueError(
                "ReplaySegmentation: options needs max_jump, the largest step "
                "of a trajectory (replay_segments() has no default)."
            )
        unscaled = sorted(set(self.relative_to) - set(self.options))
        if unscaled:
            raise ValueError(
                f"ReplaySegmentation: relative_to names {unscaled}, which are "
                "not in options; add their values to options or remove them "
                "from relative_to."
            )
        mismatched = {
            name: scale
            for name, scale in self.relative_to.items()
            if _SEGMENT_OPTION_SCALES.get(name) != scale
        }
        if mismatched:
            raise ValueError(
                _format_error(
                    f"ReplaySegmentation: relative_to={mismatched} scales "
                    "options by a property they are not measured in.",
                    why="'track_length' multiplies a distance (max_jump, "
                    "max_spread, merge_max_distance, min_dispersion) and "
                    "'chance' multiplies min_peak_probability, a posterior "
                    "mass; times and counts have no scale",
                    fix="pair each option with its scale, or give it in its own "
                    "units and leave it out of relative_to",
                )
            )
        # Check every option as replay_segments would; an option relative to
        # chance is checked against 1 only once n_bins is known.
        chance = [name for name, scale in self.relative_to.items() if scale == "chance"]
        defaults = {
            name: parameters[name].default
            for name in accepted
            if parameters[name].default is not inspect.Parameter.empty
        }
        given = {
            name: value for name, value in self.options.items() if name not in chance
        }
        _checked_options("ReplaySegmentation", **{**defaults, **given})
        for name in chance:
            _threshold(
                name, self.options[name], minimum="zero", operation="ReplaySegmentation"
            )
        if self.choose not in ("longest", "any"):
            raise ValueError(
                f"ReplaySegmentation: choose must be 'longest' or 'any', got "
                f"{self.choose!r}."
            )
        if self.rows not in ("segment", "trajectory"):
            raise ValueError(
                f"ReplaySegmentation: rows must be 'segment' or 'trajectory', "
                f"got {self.rows!r}."
            )
        cells = self.min_active_cells_per_time_bin
        if cells is not None and (
            isinstance(cells, bool)
            or not isinstance(cells, (int, np.integer))
            or cells < 1
        ):
            raise ValueError(
                "ReplaySegmentation: min_active_cells_per_time_bin must be a "
                f"positive integer or None, got {cells!r}."
            )

    def __hash__(self) -> int:
        """Hash by value; the read-only mappings are not hashable."""
        return hash(
            (
                tuple(sorted(self.options.items())),
                tuple(sorted(self.relative_to.items())),
                self.choose,
                self.min_active_cells_per_time_bin,
                self.rows,
            )
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mappings cannot pickle."""
        return reduce_through_constructor(self)

    def resolved_options(
        self, env: Environment, *, n_decoded_bins: int | None = None
    ) -> dict[str, Any]:
        """The options for ``replay_segments`` on ``env``, with scales applied.

        Parameters
        ----------
        env : Environment
            The environment the event was decoded on.
        n_decoded_bins : int or None, default=None
            Bins decoded together, for options relative to chance. For one
            template's share of a joint decode, the bins of every template
            (:func:`detect_replay` counts them); None uses ``env.n_bins``.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for :func:`~neurospatial.replay.replay_segments`.

        Raises
        ------
        ValueError
            If an option is relative to the track length and ``env`` is not
            a track.
        """
        options = dict(self.options)
        for name, scale in self.relative_to.items():
            if scale == "chance":
                options[name] = options[name] / (
                    env.n_bins if n_decoded_bins is None else n_decoded_bins
                )
            else:
                options[name] = options[name] * _track_length(
                    env, f"ReplaySegmentation with {name} relative to track_length"
                )
        return options


@dataclass(frozen=True)
class ReplayMethod:
    """A published replay test: score, shuffles, and thresholds.

    Attributes
    ----------
    name : str
        Short name, for example ``"davidson_2009"``.
    reference : str
        The paper the method follows.
    score : callable or None
        The replay score, a function ``score(env, posterior, times,
        **score_options)`` returning a result with ``score``, such as
        :func:`~neurospatial.replay.weighted_correlation`. None for a method
        that uses criteria only. A method with a ``segmentation`` and no
        score may still have ``nulls``: the tested statistic is then whether
        the event has a segment that meets the ``segment_*`` criteria, so
        the p-value is the share of shuffles that also have one (Igata et
        al. 2021).
    score_options : Mapping[str, Any]
        Options for ``score``, checked against its signature when the
        method is made. ``"n_samples_per_time_bin"`` is converted to
        ``n_samples`` for each event, for samplers like
        :func:`~neurospatial.replay.fit_linear_trajectory` whose papers
        give a count per time bin.
    nulls : tuple of str
        Shuffles the score is tested against (see
        :func:`~neurospatial.replay.replay_significance`), each named once;
        a single name is one null. Empty for a method without a per-event
        shuffle test. A method needs nulls, criteria or a segmentation:
        without any, it would call every event replay.
    null_options : Mapping[str, Mapping[str, Any]]
        Options of the method's nulls, by null name, passed to
        :func:`~neurospatial.replay.replay_significance`, such as
        ``{"column_cycle_weighted": {"edge_exclusion": 10.0}}`` (Farooq et
        al. 2019; a distance in the method's ``units``). Checked when the
        method is made: each name must be one of ``nulls`` and each option
        one the null takes.
    n_shuffles : int
        Shuffles per null.
    alpha : float
        Significance level: the event's largest p-value must be below it.
    alternative : {"greater", "less", "two-sided"}
        Tail of the shuffle test.
    use_absolute : bool or None
        Test the absolute score; None uses the default of
        :func:`~neurospatial.replay.replay_significance`.
    tested_quantities : Mapping[str, {"less", "greater", "two-sided"}]
        Scalar :class:`~neurospatial.replay.TrajectoryMetrics` of the tested
        rows, each tested against every null in the given tail, such as
        ``{"mean_spread": "less", "mean_mean_jump": "less"}`` (Widloski &
        Foster 2025: the event's posterior is sharper and steps less than
        its shuffles'). Each quantity's p-value is ``(k + 1) / (n + 1)``
        over the ``n`` shuffles of a null, ``k`` of them at least as extreme
        as the event, and the largest over the nulls; the event's
        ``p_value`` is the largest over the quantities and the score's
        test, so every test must reject. Every test draws the same shuffles
        from one seed. The quantities are computed on the rows the score
        would use: the tested segment's rows (``segmentation.rows``), less
        the time bins ``empty_time_bins="drop"`` drops. Shuffles are those
        rows shuffled or decoded again; the segmentation is not run on
        them. With a segmentation and no score, this test replaces the
        segment-detection test, and with ``choose="any"`` the segments that
        meet the ``segment_*`` criteria are tested from the longest down,
        and the first whose quantities are significant is reported (the
        longest when none is). Needs nulls. Quantities a shuffle does not
        change (``n_time_bins``, ``duration``, ``track_length``) and
        ``empty_fraction``, which needs the spike counts, cannot be tested;
        ``empty_time_bins="zero"`` (a zeroed time bin has no position),
        ``template_choice="max_score"`` and, with a score, a segmentation
        with ``choose="any"`` are refused. Empty for a method that tests
        no quantity.
    empty_time_bins : {"keep", "drop", "zero"}
        Time bins without spikes: keep them, drop them with their times
        (Karlsson & Frank 2009; Wu et al. 2017), or set their posterior to
        zero (the Bendor-lab code). With a ``segmentation``, "drop" and
        "zero" also make them invalid time bins, so no jump is measured
        across them, and "drop" removes them from the tested segment.
        "drop" cannot be used with the nulls that move spikes to other time
        bins (``"spike_train_circular"`` and ``"spike_time_uniform"``): they
        would spread the spikes over the kept time bins only, not over the
        event, and their shuffles would have time bins without spikes that
        cannot be dropped. Use "zero", which zeroes each shuffle's own
        empty time bins.
    criteria : tuple of ReplayCriterion
        Thresholds the event must meet as well.
    segmentation : ReplaySegmentation or None
        How the event is cut into trajectory segments. When given, the
        score, the shuffles and the criteria apply to the tested segment's
        rows, and the event is not replay without a segment. A method with
        a score and a segmentation cannot use ``"spike_train_circular"`` or
        ``"spike_time_uniform"``: they would move spikes over the segment's
        rows only, not over the event.
    units : str or None
        Units of the method's distance thresholds, such as ``"cm"``;
        :func:`detect_replay` refuses an environment in other units. None
        for a method whose thresholds have no units or scale with the track.
    template_choice : {"most_posterior", "sum", "smallest_p", "max_score"} or None
        For an event given as several templates without ``template=``,
        which template is tested. ``"most_posterior"`` tests the template
        of a joint decode that holds the most posterior over the event,
        leaving out ``exclude_end_distance`` at each track end (Mallory et
        al. 2025); the last template wins a tie, as in the Mallory et al.
        code, which tests the second map unless the first holds more.
        ``"sum"`` tests the posterior of a joint decode summed over the
        templates, which must share one environment, such as the two
        running directions of a joint position x direction decode
        (Berners-Lee et al. 2022); the result has no ``template``, and
        nulls that decode again cannot be used. ``"smallest_p"`` tests
        every template, each as ``template=`` would (a joint decode's share,
        or a posterior of templates decoded separately), with the method's
        score, nulls and criteria, and reports the template with the
        smallest p-value and its verdict (Karlsson & Frank 2009 lineage:
        Jadhav et al. 2016; Tang et al. 2017; Shin et al. 2019; Ólafsdóttir
        et al. 2016 for two running directions). As in those papers, the
        p-value is not corrected for the number of templates, so with ``k``
        templates an event without replay is called replay more often than
        ``alpha``, up to ``1 - (1 - alpha)**k``. A tie goes to the first
        template in the decode's order, as MATLAB's ``min`` in the
        Jadhav-lab code; a template without a p-value is chosen only when
        no template has one. Needs nulls. ``"max_score"`` takes templates
        decoded separately, not jointly (a mapping of posteriors): the
        event's statistic is the largest score over the templates, and
        each shuffle's statistic is the largest over the templates'
        independently drawn shuffles (Muessig et al. 2019, after Silva et
        al. 2015; see :func:`~neurospatial.replay.replay_significance`),
        so its p-value needs no correction; the result's ``template`` is
        the one with the largest score, and criteria are evaluated on it.
        Needs a score and nulls, with ``alternative="greater"``, no
        segmentation, no template criteria and no pseudo-event null. With
        either choice, each template draws from its own random stream,
        derived from ``rng`` and the template's name, so results do not
        depend on the templates' order. An explicit ``template=`` tests
        that template of a joint decode instead, and a posterior given as
        an array is scored as it is, so pass the joint decode or the
        mapping. None needs ``template=``.
    exclude_end_distance : float
        Distance, in the method's ``units``, left out at each end of each
        template's track when the templates are compared: by
        ``template_choice`` and the ``"template_fraction"`` criterion.
    decoding : Mapping[str, Any]
        The paper's decoding settings, for reference (not enforced).
    notes : tuple of str
        Choices the paper does not specify, and reference-code bugs not
        reproduced.
    """

    name: str
    reference: str
    score: Callable[..., Any] | None
    score_options: Mapping[str, Any] = field(default_factory=dict)
    nulls: tuple[str, ...] = ()
    null_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    n_shuffles: int = 1000
    alpha: float = 0.05
    alternative: Literal["greater", "less", "two-sided"] = "greater"
    use_absolute: bool | None = None
    tested_quantities: Mapping[str, Literal["less", "greater", "two-sided"]] = field(
        default_factory=dict
    )
    empty_time_bins: Literal["keep", "drop", "zero"] = "keep"
    criteria: tuple[ReplayCriterion, ...] = ()
    segmentation: ReplaySegmentation | None = None
    units: str | None = None
    template_choice: (
        Literal["most_posterior", "sum", "smallest_p", "max_score"] | None
    ) = None
    exclude_end_distance: float = 0.0
    decoding: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the mappings and refuse settings that cannot run."""
        object.__setattr__(
            self, "score_options", MappingProxyType(dict(self.score_options))
        )
        object.__setattr__(self, "decoding", MappingProxyType(dict(self.decoding)))
        object.__setattr__(
            self, "tested_quantities", MappingProxyType(dict(self.tested_quantities))
        )
        given: Any = self.nulls  # one name is one null, as in replay_significance
        object.__setattr__(
            self, "nulls", (given,) if isinstance(given, str) else tuple(given)
        )
        options = self.null_options
        if isinstance(options, Mapping) and all(
            isinstance(value, Mapping) for value in options.values()
        ):
            options = MappingProxyType(
                {name: MappingProxyType(dict(value)) for name, value in options.items()}
            )
        object.__setattr__(self, "null_options", options)
        object.__setattr__(self, "criteria", tuple(self.criteria))
        object.__setattr__(self, "notes", tuple(self.notes))
        self._check_score()
        quantities = _criterion_quantities()
        unknown = sorted({c.quantity for c in self.criteria} - set(quantities))
        if unknown:
            raise ValueError(
                _format_error(
                    f"ReplayMethod {self.name!r}: criteria test {unknown}, which "
                    "are not quantities of an event.",
                    why="a criterion compares one number per event with its threshold",
                    fix=f"use one of {list(quantities)}",
                )
            )
        labels = [c.label for c in self.criteria]
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            raise ValueError(
                _format_error(
                    f"ReplayMethod {self.name!r}: criteria {repeated} appear "
                    "more than once.",
                    why="each criterion's verdict and value are reported under "
                    "its label, so criteria with one label would overwrite "
                    "each other",
                    fix="keep one criterion per label; two thresholds on one "
                    "quantity in one direction reduce to the stricter one",
                )
            )
        if self.score is None and any(c.quantity == "score" for c in self.criteria):
            raise ValueError(
                f"ReplayMethod {self.name!r}: a method without a score cannot "
                "threshold one."
            )
        if (
            self.score is None
            and self.nulls
            and self.segmentation is None
            and not self.tested_quantities
        ):
            raise ValueError(
                _format_error(
                    f"ReplayMethod {self.name!r}: a method without a score or a "
                    "segmentation has nothing to test against shuffles.",
                    fix="give a score, tested_quantities, or a segmentation "
                    "whose detection is tested (Igata et al. 2021)",
                )
            )
        segment_criteria = [
            c.label for c in self.criteria if c.quantity in _SEGMENT_QUANTITIES
        ]
        if segment_criteria and self.segmentation is None:
            raise ValueError(
                f"ReplayMethod {self.name!r}: criteria {segment_criteria} "
                "describe a segment, but the method has no segmentation; give "
                "it a ReplaySegmentation or use event-level quantities."
            )
        if (
            self.score is None
            and self.segmentation is not None
            and self.nulls
            and not self.tested_quantities
        ):
            # The detection test scores 0 or 1 against shuffles at least as
            # extreme, with the event's time-bin rules applied to each shuffle.
            if self.alternative != "greater" or self.use_absolute is not None:
                raise ValueError(
                    f"ReplayMethod {self.name!r}: a segment-detection test is "
                    "one-sided on a 0/1 score; leave alternative='greater' and "
                    "use_absolute=None."
                )
            moving = [name for name in self.nulls if name in _ROW_MOVING_NULLS]
            row_rules = (
                self.segmentation.min_active_cells_per_time_bin is not None
                or self.empty_time_bins != "keep"
            )
            if moving and row_rules:
                raise ValueError(
                    f"ReplayMethod {self.name!r}: nulls {moving} move or change "
                    "the spikes in each time bin, so the event's active-cell "
                    "and empty-time-bin rules do not apply to its shuffles; "
                    "use nulls that keep the time bins in place."
                )
        if self.empty_time_bins not in ("keep", "drop", "zero"):
            raise ValueError(
                f"ReplayMethod {self.name!r}: empty_time_bins must be 'keep', "
                f"'drop' or 'zero', got {self.empty_time_bins!r}."
            )
        if self.template_choice not in _TEMPLATE_CHOICES:
            raise ValueError(
                f"ReplayMethod {self.name!r}: template_choice must be one of "
                f"{list(_TEMPLATE_CHOICES)}, got {self.template_choice!r}."
            )
        if self.template_choice == "smallest_p" and not self.nulls:
            raise ValueError(
                _format_error(
                    f"ReplayMethod {self.name!r}: template_choice='smallest_p' "
                    "needs nulls.",
                    why="it keeps the template whose shuffle test gives the "
                    "smallest p-value",
                    fix="give nulls, or choose the template with template=",
                )
            )
        if self.template_choice == "max_score":
            self._check_max_score()
        self._check_tested_quantities()
        if self.template_choice == "sum":
            refused = [
                c.label for c in self.criteria if c.quantity in _TEMPLATE_QUANTITIES
            ]
            decoding = [name for name in self.nulls if name in _DECODING_NULLS]
            if refused or decoding:
                raise ValueError(
                    f"ReplayMethod {self.name!r}: template_choice='sum' tests "
                    "the templates' summed posterior, which has no template to "
                    f"compare or decode again; remove {refused + decoding}."
                )
        distance = self.exclude_end_distance
        if not (
            isinstance(distance, (int, float, np.integer, np.floating))
            and not isinstance(distance, bool)
            and np.isfinite(distance)
            and distance >= 0
        ):
            raise ValueError(
                f"ReplayMethod {self.name!r}: exclude_end_distance must be a "
                f"finite distance >= 0, got {distance!r}."
            )
        if self.segmentation is None and any(
            c.quantity == "segment_template_fraction" for c in self.criteria
        ):
            raise ValueError(
                f"ReplayMethod {self.name!r}: segment_template_fraction needs a "
                "segmentation; give the method a ReplaySegmentation."
            )
        unknown = [name for name in self.nulls if name not in _NULLS]
        if unknown:
            raise ValueError(
                f"ReplayMethod {self.name!r}: unknown nulls {unknown}; choose "
                f"from {list(_NULLS)}."
            )
        repeated = sorted({name for name in self.nulls if self.nulls.count(name) > 1})
        if repeated:
            raise ValueError(
                f"ReplayMethod {self.name!r}: nulls {repeated} are named more "
                "than once; name each null once and set the number of shuffles "
                "per null with n_shuffles."
            )
        _checked_null_options(
            self.null_options, self.nulls, f"ReplayMethod {self.name!r}"
        )
        spreading = [name for name in self.nulls if name in _COUNT_CHANGING_NULLS]
        if spreading and self.empty_time_bins == "drop":
            raise ValueError(
                _format_error(
                    f"ReplayMethod {self.name!r}: empty_time_bins='drop' cannot "
                    f"be used with the {spreading} null"
                    f"{'s' if len(spreading) > 1 else ''}.",
                    why="they move each cell's spikes to other time bins of the "
                    "event: after dropping the time bins without spikes they "
                    "would spread the spikes over the kept time bins only, and "
                    "their shuffles have new time bins without spikes, which "
                    "cannot be dropped",
                    fix="use empty_time_bins='zero', which zeroes each "
                    "shuffle's own time bins without spikes, or 'keep'",
                )
            )
        if spreading and self.score is not None and self.segmentation is not None:
            raise ValueError(
                _format_error(
                    f"ReplayMethod {self.name!r}: a segmentation cannot be used "
                    f"with the {spreading} null"
                    f"{'s' if len(spreading) > 1 else ''} when the method has a "
                    "score.",
                    why="the score is tested on the segment's rows, so these "
                    "nulls would move each cell's spikes over the segment only "
                    "(with rows='trajectory', over rows that are not even "
                    "adjacent), not over the event",
                    fix="leave out the segmentation and test the whole event, "
                    "use nulls that do not spread spikes across time bins "
                    "(such as 'place_field_circular' or 'column_cycle'), or "
                    "test the segment's detection with score=None",
                )
            )
        if self.alternative not in _ALTERNATIVES:
            raise ValueError(
                f"ReplayMethod {self.name!r}: alternative must be one of "
                f"{list(_ALTERNATIVES)}, got {self.alternative!r}."
            )
        if not 0 < self.alpha < 1:
            raise ValueError(
                f"ReplayMethod {self.name!r}: alpha must be in (0, 1), got "
                f"{self.alpha!r}."
            )
        if (
            isinstance(self.n_shuffles, bool)
            or not isinstance(self.n_shuffles, (int, np.integer))
            or self.n_shuffles < 1
        ):
            raise ValueError(
                f"ReplayMethod {self.name!r}: n_shuffles must be a positive "
                f"integer, got {self.n_shuffles!r}."
            )
        if not (self.nulls or self.criteria or self.segmentation is not None):
            raise ValueError(
                _format_error(
                    f"ReplayMethod {self.name!r}: the method has no nulls, "
                    "criteria or segmentation, so it tests nothing.",
                    why="without a shuffle test, a threshold or a trajectory "
                    "segment to find, every event would be called replay",
                    fix="give nulls such as ('column_cycle',), criteria such as "
                    "(ReplayCriterion('score', '>', 0.5, absolute=True),), or a "
                    "ReplaySegmentation",
                )
            )

    def _check_max_score(self) -> None:
        """Refuse settings the largest score over templates cannot test."""
        refused = [
            reason
            for reason, applies in (
                ("a score", self.score is None),
                ("nulls", not self.nulls),
                ("alternative='greater'", self.alternative != "greater"),
            )
            if applies
        ]
        if refused:
            raise ValueError(
                _format_error(
                    f"ReplayMethod {self.name!r}: template_choice='max_score' "
                    f"needs {' and '.join(refused)}.",
                    why="the event's largest score over the templates is tested "
                    "against the largest of each shuffle's scores, and the "
                    "largest is the best fit only when larger scores are better",
                    fix="give a score and nulls with alternative='greater' "
                    "(use_absolute=True for a signed score), or use "
                    "template_choice='smallest_p'",
                )
            )
        unused = [c.label for c in self.criteria if c.quantity in _TEMPLATE_QUANTITIES]
        unused += [name for name in self.nulls if name == "pseudo_event"]
        if self.segmentation is not None:
            unused.append("segmentation")
        if unused:
            raise ValueError(
                _format_error(
                    f"ReplayMethod {self.name!r}: template_choice='max_score' "
                    f"cannot be used with {unused}.",
                    why="the templates are decoded separately, so no template "
                    "holds a share of a joint posterior to compare, one "
                    "pseudo-event pool cannot serve every template's "
                    "environment, and segments would be cut in each template "
                    "differently",
                    fix=f"remove {unused}, or decode the templates jointly and "
                    "use template_choice='smallest_p'",
                )
            )

    def _check_tested_quantities(self) -> None:
        """Refuse quantities a shuffle test cannot compare, and unsupported settings."""
        tested = self.tested_quantities
        if not tested:
            return
        testable = _testable_quantities()
        unknown = sorted(set(tested) - set(testable))
        if unknown:
            raise ValueError(
                _format_error(
                    f"ReplayMethod {self.name!r}: tested_quantities names "
                    f"{unknown}, which a shuffle test cannot compare.",
                    why="each tested quantity is a scalar of TrajectoryMetrics "
                    "computed again on every shuffled posterior; the "
                    "segmentation is not run on shuffles, n_time_bins, duration "
                    "and track_length are the same in every shuffle, and "
                    "empty_fraction needs spike counts",
                    fix=f"use one of {list(testable)}",
                )
            )
        tails = {
            name: tail for name, tail in tested.items() if tail not in _ALTERNATIVES
        }
        if tails:
            raise ValueError(
                f"ReplayMethod {self.name!r}: tested_quantities tails must be "
                f"one of {list(_ALTERNATIVES)}, got {tails}."
            )
        refusals = (
            (
                not self.nulls,
                "tested_quantities need nulls",
                "each quantity is compared with its value in shuffled events",
                "give nulls, such as ('unit_identity',)",
            ),
            (
                self.empty_time_bins == "zero",
                "tested_quantities cannot be used with empty_time_bins='zero'",
                "a zeroed time bin has no decoded position, so its spread and "
                "jumps are undefined",
                "use empty_time_bins='drop' or 'keep'",
            ),
            (
                self.template_choice == "max_score",
                "tested_quantities cannot be used with template_choice='max_score'",
                "it tests the largest score over separately decoded templates, "
                "not the quantities of one template",
                "use template_choice='smallest_p'",
            ),
            (
                self.score is not None
                and self.segmentation is not None
                and self.segmentation.choose == "any",
                "tested_quantities with a score cannot be used with "
                "segmentation choose='any'",
                "the segment would be chosen by the quantities' test alone, "
                "leaving out the score's",
                "use choose='longest', or leave out the score",
            ),
        )
        for applies, message, why, fix in refusals:
            if applies:
                raise ValueError(
                    _format_error(
                        f"ReplayMethod {self.name!r}: {message}.", why=why, fix=fix
                    )
                )

    def _check_score(self) -> None:
        """Refuse a score that is not callable or does not take score_options."""
        if self.score is None:
            if self.score_options:
                raise ValueError(
                    f"ReplayMethod {self.name!r}: score_options "
                    f"{sorted(self.score_options)} need a score; give one or "
                    "remove them."
                )
            return
        if not callable(self.score):
            raise TypeError(
                f"ReplayMethod {self.name!r}: score must be a function "
                "score(env, posterior, times, **score_options), such as "
                f"weighted_correlation, or None; got {self.score!r}."
            )
        options = dict(self.score_options)
        if "n_samples_per_time_bin" in options:
            options["n_samples"] = options.pop("n_samples_per_time_bin")
        try:
            signature = inspect.signature(self.score)
        except (TypeError, ValueError) as error:
            raise TypeError(
                _format_error(
                    f"ReplayMethod {self.name!r}: cannot read the signature of "
                    f"score {self.score!r} ({error}).",
                    why="the signature tells which options the score takes and "
                    "whether it takes the seeded rng of the shuffle test",
                    fix="wrap the score in a Python function, def score(env, "
                    "posterior, times, **options), with an rng keyword if it "
                    "draws random numbers",
                )
            ) from error
        try:
            signature.bind_partial(None, None, None, **options)
        except TypeError as error:
            name = getattr(self.score, "__name__", repr(self.score))
            raise ValueError(
                _format_error(
                    f"ReplayMethod {self.name!r}: score_options "
                    f"{sorted(self.score_options)} do not fit {name}"
                    "(env, posterior, times, ...).",
                    why=str(error),
                    fix=f"use options {name} takes: {list(signature.parameters)[3:]}",
                )
            ) from None

    def __hash__(self) -> int:
        """Hash by name and reference; the option mappings are not hashable."""
        return hash((self.name, self.reference))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mappings cannot pickle."""
        rebuild, (cls, state) = reduce_through_constructor(self)
        state["null_options"] = {
            name: dict(options) for name, options in self.null_options.items()
        }
        return rebuild, (cls, state)


@dataclass(frozen=True)
class RankOrderMethod:
    """A published rank-order replay test.

    The event's score is the Spearman correlation between when each active
    template cell fired and where its field lies on the track
    (:func:`~neurospatial.replay.rank_order_correlation`), tested with the
    t-distribution of Spearman's rho or against cell-order permutations
    (:func:`~neurospatial.replay.rank_order_significance`).

    Attributes
    ----------
    name : str
        Short name, for example ``"diba_buzsaki_2007"``.
    reference : str
        The paper the method follows.
    spike_statistic : {"first", "median", "mean", "all_spikes"}
        When each cell fired in the event (see
        :func:`~neurospatial.replay.rank_order_correlation`).
    field_order : {"peak", "center"}
        Each cell's field position: its peak or rate-weighted center.
    min_peak_rate : float or None
        Template cells: cells whose rate map peaks at this rate or more, in
        Hz (Foster & Wilson 2006; Diba & Buzsáki 2007: 5 Hz). Other cells'
        spikes are ignored and they do not count toward
        ``min_active_fraction``. None makes every cell with a field a
        template cell.
    min_active_cells : int
        Fewest active template cells; with fewer the event is not replay.
    min_active_fraction : float or None
        Fewest active template cells as a fraction of all template cells
        (Foster & Wilson 2006: one third; Diba & Buzsáki 2007: 30%).
    test : {"t_distribution", "cell_order", "pooled_ssi"}
        The two-sided p-value of rho from the t-distribution with
        ``n - 2`` degrees of freedom over the ``n`` (time, field) pairs, as
        Numerical Recipes' ``spear`` computes it (Foster & Wilson 2006);
        ``n_shuffles`` cell-order permutations (Diba & Buzsáki 2007); or the
        spatial similarity index of Chenani et al. (2019), compared with the
        cell-order shuffles of every event in the epoch. ``"pooled_ssi"``
        runs only in :func:`compare_replay_methods`, on one epoch's events:
        each event's SSI per template is ``rho * sqrt(n - 1)`` over its
        ``n`` active cells with a field in that template (rho over the
        standard deviation of rho under random orders), the event scores
        its largest SSI over the templates, and it is replay when that
        exceeds the ``1 - alpha`` quantile of the largest SSIs of
        ``n_shuffles`` cell-order shuffles of every event, taken on a 0.01
        grid as in the authors' code; its ``p_value``, the share of pooled
        shuffles at least as large, is reported but not used for the verdict.
    alternative : {"greater", "less", "two-sided"}
        Tail of the permutation test; the t-distribution test is two-sided
        and the pooled SSI test is one-sided (``"greater"``).
    n_shuffles : int
        Cell-order permutations.
    alpha : float
        Significance level: the p-value must be below it.
    min_epoch_events : int or None
        For ``"pooled_ssi"``, fewest events with enough active cells in the
        epoch; with fewer, no event is replay (Chenani et al. 2019: 20).
    template_choice : {"most_active"} or None
        With several templates (one rate-map set per running direction) and
        no ``template=``, which template is reported. ``"most_active"``
        takes, among the templates on which the event is replay, the one
        with the most active cells and then the larger ``|rho|`` (Diba &
        Buzsáki 2007); without such a template, the same rule over all
        templates. None needs ``template=``.
    notes : tuple of str
        Choices the paper does not specify, and event-selection rules not
        applied.
    """

    name: str
    reference: str
    spike_statistic: Literal["first", "median", "mean", "all_spikes"] = "median"
    field_order: Literal["peak", "center"] = "peak"
    min_peak_rate: float | None = None
    min_active_cells: int = 5
    min_active_fraction: float | None = None
    test: Literal["t_distribution", "cell_order", "pooled_ssi"] = "cell_order"
    alternative: Literal["greater", "less", "two-sided"] = "two-sided"
    n_shuffles: int = 1000
    alpha: float = 0.05
    min_epoch_events: int | None = None
    template_choice: Literal["most_active"] | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse settings that cannot run."""
        from neurospatial.replay.rank_order import _FIELD_ORDERS, _SPIKE_STATISTICS

        object.__setattr__(self, "notes", tuple(self.notes))
        prefix = f"RankOrderMethod {self.name!r}"
        choices = {
            "spike_statistic": _SPIKE_STATISTICS,
            "field_order": _FIELD_ORDERS,
            "test": ("t_distribution", "cell_order", "pooled_ssi"),
            "alternative": _ALTERNATIVES,
            "template_choice": (None, "most_active"),
        }
        for field_name, allowed in choices.items():
            if getattr(self, field_name) not in allowed:
                raise ValueError(
                    f"{prefix}: {field_name} must be one of {list(allowed)}, got "
                    f"{getattr(self, field_name)!r}."
                )
        if self.test == "t_distribution" and self.alternative != "two-sided":
            raise ValueError(
                f"{prefix}: the t-distribution test of rho is two-sided; use "
                "test='cell_order' for a one-sided test."
            )
        if self.test == "pooled_ssi":
            if self.alternative != "greater" or self.spike_statistic == "all_spikes":
                raise ValueError(
                    f"{prefix}: the pooled SSI test is one-sided on one time "
                    "per cell; use alternative='greater' and a spike_statistic "
                    "other than 'all_spikes'."
                )
            if self.template_choice is not None:
                raise ValueError(
                    f"{prefix}: the pooled SSI test scores every template and "
                    "keeps the largest SSI; leave template_choice=None."
                )
            if self.min_active_fraction is not None:
                raise ValueError(
                    f"{prefix}: the pooled SSI test counts active cells only; "
                    "leave min_active_fraction=None."
                )
        elif self.min_epoch_events is not None:
            raise ValueError(
                f"{prefix}: min_epoch_events applies only to test='pooled_ssi'."
            )
        if not 0 < self.alpha < 1:
            raise ValueError(f"{prefix}: alpha must be in (0, 1), got {self.alpha!r}.")
        for field_name, minimum in (
            ("min_active_cells", 3),
            ("n_shuffles", 1),
            ("min_epoch_events", 1),
        ):
            value = getattr(self, field_name)
            if value is None and field_name == "min_epoch_events":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(
                    f"{prefix}: {field_name} must be an integer of at least "
                    f"{minimum}, got {value!r}."
                )
        for field_name, upper in (
            ("min_active_fraction", 1.0),
            ("min_peak_rate", None),
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value <= 0
                or (upper is not None and value > upper)
            ):
                bounds = "in (0, 1]" if upper is not None else "a positive rate in Hz"
                raise ValueError(
                    f"{prefix}: {field_name} must be {bounds} or None, got {value!r}."
                )

    @property
    def criteria(self) -> tuple[ReplayCriterion, ...]:
        """The active-cell thresholds, as criteria."""
        criteria = [ReplayCriterion("n_active_cells", ">=", self.min_active_cells)]
        if self.min_active_fraction is not None:
            criteria.append(
                ReplayCriterion("active_fraction", ">=", self.min_active_fraction)
            )
        return tuple(criteria)


@dataclass(frozen=True, repr=False, eq=False)
class ReplayDetectionResult(ResultMixin):
    """One method's verdict on one event.

    Attributes
    ----------
    method : str
        The method's name.
    is_replay : bool
        For a :class:`ReplayMethod`, a rank-order method with a p-value
        test, a pairwise-order method and a template-matching method: True
        when the event meets every criterion and, for a method with a
        test, its largest p-value is below ``alpha`` (at or below it for a
        template-matching method with ``p_comparison="<="``). For the
        pooled-SSI test (``test="pooled_ssi"``): True when the score
        exceeds the ``1 - alpha`` quantile of the epoch's pooled shuffles;
        ``p_value`` is reported but not used. For an arm method: True when
        an arm, arms or a pair is reported in ``template``; ``criteria``
        holds every arm's and pair's checks, which need not all pass, and
        with ``arm_fallback`` a replay can have ``criteria["sequence"]``
        False.
    score : float
        The method's score (NaN for a criteria-only method).
    p_value : float
        The largest p-value over the method's nulls (NaN without shuffles),
        and over its tested quantities (NaN when one is undefined).
    criteria : Mapping[str, bool]
        Whether each criterion holds, keyed by its label.
    values : Mapping[str, float]
        The value each criterion was compared on, keyed by its label.
    segments : ReplaySegmentsResult or None
        The event's segments, for a method with a segmentation.
    segment : int or None
        Index in ``segments`` of the tested segment; None without a
        segmentation or without segments. ``segments.rows(segment)`` selects
        its rows.
    template : str or None
        The tested template of a joint decode or of separately decoded
        templates, chosen or given as ``template=``: for
        ``template_choice="smallest_p"`` the template with the smallest
        p-value, for ``"max_score"`` the one with the largest score. None
        for an event decoded on one set of rate maps, for a joint decode's
        summed posterior (``template_choice="sum"``) and for a
        ``"max_score"`` event without a verdict.
        For a pairwise-order method given a mapping of rate maps, the
        template of the test with the smallest p-value.
        For an arm method (:func:`~neurospatial.replay.detect_arm_replay`),
        the replayed arm, arms (``"center,left"``) or pair
        (``"center+left"``); None without a replay.
    n_shuffles_used : int or None
        Shuffles the p-value counts: the method's ``n_shuffles`` less those
        whose score is undefined (NaN), the smallest over its nulls; for a
        pooled SSI method, the epoch's pooled shuffles. None when no
        shuffle was scored. With tested quantities, the smallest over them
        and the score's test.
    tested_values : Mapping[str, float]
        For a method with ``tested_quantities``, the event's value of each
        tested quantity (NaN when it was not tested, such as without a
        segment); empty for other methods.
    tested_p_values : Mapping[str, float]
        Each tested quantity's p-value, the largest over the nulls; the
        same keys as ``tested_values``.
    z_scores : Mapping[str, float]
        The score's z-score against each null's shuffles, keyed by null
        name: ``(statistic - mean) / std`` over the shuffles with a defined
        score, on the tested statistic (the absolute score with
        ``use_absolute``), as
        :attr:`~neurospatial.stats.shuffle.ShuffleTestResult.z_score`. With
        ``use_absolute=True`` this is the sequence score rZ of Grosmark &
        Buzsáki (2016), the replay score-z of Xu et al. (2019) and the
        sequence score of Farooq et al. (2019). Empty for a method without a
        score's shuffle test. The standard deviation has ``ddof = 0``;
        MATLAB's ``std`` uses ``ddof = 1``, a factor ``sqrt(n / (n - 1))``.
    """

    method: str
    is_replay: bool
    score: float
    p_value: float
    criteria: Mapping[str, bool]
    values: Mapping[str, float]
    segments: ReplaySegmentsResult | None = None
    segment: int | None = None
    template: str | None = None
    n_shuffles_used: int | None = None
    tested_values: Mapping[str, float] = field(default_factory=dict)
    tested_p_values: Mapping[str, float] = field(default_factory=dict)
    z_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the mappings."""
        for name in (
            "criteria",
            "values",
            "tested_values",
            "tested_p_values",
            "z_scores",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mappings cannot pickle."""
        return reduce_through_constructor(self)

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result.

        ``n_shuffles_used`` is NaN when no shuffle was scored. A joint
        decode adds the tested ``template``. A method with a segmentation
        adds ``n_segments`` and the tested segment's ``segment_start_time``
        and ``segment_end_time`` (NaN without one). A method with tested
        quantities adds each quantity's value under its name, such as
        ``mean_spread``, and its p-value, ``p_value[mean_spread]``. Each
        criterion adds the value it was compared on, ``value[label]`` (the
        raw value, before ``absolute`` and the ``relative_to`` scale), and
        each null of the score's test its z-score, ``z_score[null]``.
        """
        values: dict[str, Any] = {
            "method": self.method,
            "is_replay": self.is_replay,
            "score": self.score,
            "p_value": self.p_value,
            "n_shuffles_used": np.nan
            if self.n_shuffles_used is None
            else self.n_shuffles_used,
            **{f"passes[{label}]": ok for label, ok in self.criteria.items()},
            **{f"value[{label}]": value for label, value in self.values.items()},
            **{f"z_score[{name}]": z for name, z in self.z_scores.items()},
            **self.tested_values,
            **{f"p_value[{name}]": p for name, p in self.tested_p_values.items()},
        }
        if self.template is not None:
            values["template"] = self.template
        if self.segments is not None:
            tested = self.segment is not None
            values["n_segments"] = self.segments.n_segments
            values["segment_start_time"] = (
                float(self.segments.start_times[self.segment]) if tested else np.nan
            )
            values["segment_end_time"] = (
                float(self.segments.end_times[self.segment]) if tested else np.nan
            )
        return values


def _verdict_uses_score(method: ReplayMethod) -> bool:
    """Whether a method's verdict reads its score: through nulls or a criterion."""
    return method.score is not None and (
        bool(method.nulls) or any(c.quantity == "score" for c in method.criteria)
    )


def _environment_mismatch(first: Environment, second: Environment) -> str | None:
    """How two environments' bins or connectivity differ, or None if they agree."""
    from neurospatial.environment import _environment_differences

    bins, topology = _environment_differences(first, second)
    return bins or topology


def detect_replay(
    env: Environment | Mapping[str, Environment],
    posterior: NDArray[np.float64]
    | JointDecodingResult
    | Mapping[str, NDArray[np.float64]],
    times: NDArray[np.float64],
    *,
    replay_method: ReplayMethod,
    spike_counts: NDArray[np.int64] | None = None,
    encoding_models: Any = None,
    dt: float | None = None,
    prior: Any = None,
    template: str | None = None,
    pseudo_event_pool: NDArray[np.float64] | None = None,
    valid_time_bins: NDArray[np.bool_] | None = None,
    rng: np.random.Generator | int | None = None,
) -> ReplayDetectionResult:
    """Apply one published replay method to one decoded event.

    Parameters
    ----------
    env : Environment or Mapping[str, Environment]
        The environment the posterior was decoded on; for one template's
        share of a joint decode over separate tracks, the mapping of
        environments passed to :func:`~neurospatial.decoding.decode_jointly`,
        with ``template`` naming the share. For templates decoded separately
        on different environments, a mapping keyed like ``posterior``.
    posterior : NDArray[np.float64], JointDecodingResult or Mapping
        The event's posterior, as decoded, shape (n_time_bins, n_bins); the
        event's joint decode over several templates, from
        :func:`~neurospatial.decoding.decode_jointly`, whose tested
        template is ``template=`` or the method's ``template_choice``; or,
        for a method with ``template_choice="smallest_p"`` or
        ``"max_score"``, the posteriors of templates decoded separately,
        such as the two running directions each decoded with
        :func:`~neurospatial.decoding.decode_position` on its own rate
        maps, as a mapping of template name to posterior. ``encoding_models``
        and ``prior`` are then mappings keyed like it (``prior`` may also be
        one prior for every template).
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds.
    replay_method : ReplayMethod
        The method, for example ``neurospatial.replay.methods.DAVIDSON_2009``.
    spike_counts : NDArray[np.int64] or None, shape (n_time_bins, n_neurons)
        The event's spike counts. Needed by methods that drop or zero time
        bins without spikes, count active cells, or decode shuffled counts.
        Counts summed over cells, shape (n_time_bins,), are accepted only by
        methods that do not count cells.
    encoding_models, dt, prior, template
        The decoding inputs, for the nulls that decode again (see
        :func:`~neurospatial.replay.replay_significance`), including one
        template's share of a joint decode (``template=``). With a joint
        decode as ``posterior``, ``template`` names the tested template. A
        prior with one row per time bin is cut with the time bins the
        method tests (its segment, or the time bins with spikes).
    pseudo_event_pool : NDArray[np.float64] or None
        Posterior rows of the session's candidate events, for the
        pseudo-event and weighted column-cycle nulls.
    valid_time_bins : NDArray[np.bool_] or None, shape (n_time_bins,)
        For a method with a segmentation, False for time bins that cannot
        be on a trajectory, such as bins while the animal runs (Widloski &
        Foster 2022: 5 cm/s or faster); see
        :func:`~neurospatial.replay.replay_segments`. Methods without a
        segmentation ignore it, but it must still have one entry per time
        bin.
    rng : np.random.Generator, int, or None
        Random number generator or seed. With ``template_choice``
        ``"smallest_p"`` or ``"max_score"``, each template draws from its
        own stream, derived from ``rng`` and the template's name. With
        ``tested_quantities``, one seed drawn from ``rng`` starts the
        score's test and each quantity's, so all draw the same shuffles.

    Returns
    -------
    ReplayDetectionResult
        The verdict, score, p-value and each criterion; for a method with
        a segmentation, the segments and the tested segment; for several
        templates, the tested template; and for a method with
        ``tested_quantities``, each quantity's value and p-value. An event
        whose score is undefined (NaN, such as the R² of a decoded position
        that does not change) gets no verdict: ``is_replay`` and every
        criterion are False, and the score and p-value are NaN. A tested
        quantity that is undefined (such as the mean jump of one time bin)
        makes the p-value NaN, and the event is not replay.

    Raises
    ------
    ValueError
        If the method needs an input that is not given (spike counts,
        decoding inputs or a pseudo-event pool); if the method counts
        active cells and ``spike_counts`` is summed over cells; if
        ``valid_time_bins`` is not one boolean per time bin; if the method's
        ``units`` are not the environment's; if a joint decode is given
        without ``template=`` to a method without a ``template_choice``, or
        a template criterion is used without a joint decode; if separately
        decoded templates are given to a method whose ``template_choice`` is
        not ``"smallest_p"`` or ``"max_score"``, with ``template=``, or with
        inputs not keyed like them; if a joint decode is given without
        ``template=`` to a ``"max_score"`` method; if a summed
        joint decode's templates are on different environments; if
        ``valid_time_bins`` is
        given with a segment-detection test whose nulls move time bins; or
        for inputs the score, the shuffles,
        :func:`~neurospatial.replay.replay_segments` or
        :func:`~neurospatial.replay.trajectory_metrics` refuse.

    See Also
    --------
    compare_replay_methods : Several methods on several events.
    neurospatial.replay.replay_significance : The shuffle test.
    neurospatial.replay.trajectory_metrics : The quantities criteria use.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import detect_replay
    >>> from neurospatial.replay.methods import SILVA_2015
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=4.0, units="cm"
    ... )
    >>> posterior = np.zeros((20, env.n_bins))
    >>> posterior[np.arange(20), np.arange(20)] = 1.0  # 4 cm per time bin
    >>> result = detect_replay(
    ...     env, posterior, np.arange(20) * 0.02, replay_method=SILVA_2015
    ... )
    >>> result.is_replay, round(result.score, 2)
    (True, 1.0)

    A criteria-based method tests the event's trajectory segment: here a
    sweep of 4 cm per 5 ms step that starts after two time bins at the far
    end of the track.

    >>> from neurospatial.replay.methods import PFEIFFER_FOSTER_2013
    >>> posterior = np.eye(env.n_bins)[[24, 24, *range(0, 14)]]
    >>> result = detect_replay(
    ...     env, posterior, np.arange(16) * 0.005, replay_method=PFEIFFER_FOSTER_2013
    ... )
    >>> result.is_replay, result.segments.rows(result.segment)
    (True, slice(2, 16, None))
    >>> round(result.values["segment_start_to_end_distance >= 40"], 6)
    52.0
    """
    from neurospatial.decoding import JointDecodingResult
    from neurospatial.replay.criteria import template_fractions, trajectory_metrics
    from neurospatial.replay.significance import (
        _conditional_rows,
        _separate_inputs,
        _takes_rng,
        _UndefinedNullError,
        _UndefinedScoreError,
        _uses_joint_weights,
        replay_significance,
    )

    joint = posterior if isinstance(posterior, JointDecodingResult) else None
    choice = replay_method.template_choice
    if isinstance(posterior, (JointDecodingResult, Mapping)) and (
        template is None and choice == "smallest_p"
    ):
        return _smallest_p_template(
            env,
            posterior,
            times,
            replay_method=replay_method,
            spike_counts=spike_counts,
            encoding_models=encoding_models,
            dt=dt,
            prior=prior,
            pseudo_event_pool=pseudo_event_pool,
            valid_time_bins=valid_time_bins,
            rng=rng,
        )
    # Templates decoded separately, tested by the largest score over them.
    separate: dict[str, NDArray[np.float64]] | None = None
    if isinstance(posterior, Mapping):
        if choice != "max_score" or template is not None:
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: method {replay_method.name!r} got templates "
                    "decoded separately"
                    + ("" if template is None else f" and template={template!r}")
                    + ".",
                    why="separately decoded templates are tested together, by "
                    "template_choice='smallest_p' or 'max_score'",
                    fix="pass one template's posterior with its own rate maps, "
                    "or use a method with one of those template choices, e.g. "
                    "dataclasses.replace(method, template_choice='smallest_p')",
                )
            )
        names, template_envs, *_ = _separate_inputs(
            env, posterior, encoding_models, prior, _OPERATION
        )
        separate = {
            name: np.asarray(posterior[name], dtype=np.float64) for name in names
        }
        # Every template has the event's rows; checks below read the first.
        posterior = separate[names[0]]
    elif joint is not None and template is None and choice == "max_score":
        raise ValueError(
            _format_error(
                f"{_OPERATION}: method {replay_method.name!r} takes the largest "
                "score over templates decoded separately, not a joint decode.",
                why="each template's posterior and shuffles are decoded on "
                "that template's rate maps alone (Muessig et al. 2019)",
                fix="decode each template with decode_position and pass "
                "{name: result.posterior, ...} with encoding_models keyed alike, "
                "or pass template= to test one share",
            )
        )
    event_values: dict[str, float] = {}
    summed = False
    if joint is not None:
        needs_fractions = replay_method.template_choice == "most_posterior" or any(
            c.quantity == "template_fraction" for c in replay_method.criteria
        )
        fractions = (
            template_fractions(
                joint, exclude_end_distance=replay_method.exclude_end_distance
            )
            if needs_fractions
            else {}
        )
        summed = template is None and replay_method.template_choice == "sum"
        if summed:
            joint_env = joint.conditional[joint.template_names[0]].env
            for name in joint.template_names[1:]:
                mismatch = _environment_mismatch(joint.conditional[name].env, joint_env)
                if mismatch is not None:
                    raise ValueError(
                        _format_error(
                            f"{_OPERATION}: method {replay_method.name!r} sums "
                            "the templates' posteriors, but their environments "
                            f"differ in {mismatch}.",
                            why="a summed posterior needs the same bins in every "
                            "template",
                            fix="decode the templates on one environment, or "
                            "pass template= to test one of them",
                        )
                    )
            posterior = np.sum(
                [joint.joint(name) for name in joint.template_names], axis=0
            )
        else:
            if template is None:
                if replay_method.template_choice is None:
                    raise ValueError(
                        _format_error(
                            f"{_OPERATION}: method {replay_method.name!r} needs "
                            "template= for a joint decode.",
                            why="it tests one template's share and has no rule "
                            "to choose one",
                            fix=f"pass template= one of {list(joint.template_names)}",
                        )
                    )
                template = max(reversed(list(fractions)), key=fractions.__getitem__)
            elif template not in joint.template_names:
                raise ValueError(
                    f"{_OPERATION}: template {template!r} is not a template of "
                    f"the joint decode; choose one of {list(joint.template_names)}."
                )
            tested: str = template
            joint_env = joint.conditional[tested].env
            if fractions:
                event_values["template_fraction"] = fractions[tested]
            posterior = joint.joint(tested)
    else:
        template_criteria = [
            c.label
            for c in replay_method.criteria
            if c.quantity in _TEMPLATE_QUANTITIES
        ]
        if template_criteria:
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: criteria {template_criteria} of method "
                    f"{replay_method.name!r} compare templates.",
                    why="they need every template's posterior, not one share",
                    fix="pass the event's JointDecodingResult from "
                    "decode_jointly as posterior",
                )
            )

    if separate is not None:
        track_env = template_envs[names[0]]
    elif isinstance(env, Mapping) and summed:
        # The summed templates share one environment.
        assert joint is not None
        track_env = env[joint.template_names[0]]
    elif isinstance(env, Mapping):
        if template is None:
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: a mapping of environments needs template=.",
                    why="the posterior is one template's share of a joint "
                    "decode, scored on that template's environment",
                    fix="pass template=name, the share's key in env",
                )
            )
        track_env = env[template]
    else:
        track_env = env
    if joint is not None:
        mismatch = _environment_mismatch(track_env, joint_env)
        if mismatch is not None:
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: env differs from the environment the joint "
                    f"decode used in {mismatch}.",
                    why="positions and distances would be read on other bins",
                    fix="pass the environment (or mapping of environments) "
                    "given to decode_jointly",
                )
            )
    for unit_env in (track_env,) if separate is None else template_envs.values():
        if replay_method.units is None or unit_env.units == replay_method.units:
            continue
        stated = (
            "has no units" if unit_env.units is None else f"is in {unit_env.units!r}"
        )
        raise ValueError(
            _format_error(
                f"{_OPERATION}: method {replay_method.name!r} has thresholds in "
                f"{replay_method.units!r}, but the environment {stated}.",
                why="its distance thresholds would be compared with distances "
                "in other units",
                fix=f"build the environment with units={replay_method.units!r}, "
                "converting its coordinates first if they are in other units",
            )
        )
    posterior = np.asarray(posterior, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    segmentation = replay_method.segmentation
    counts_cells = any(
        c.quantity in ("n_active_cells", "active_fraction")
        for c in replay_method.criteria
    ) or (
        segmentation is not None
        and segmentation.min_active_cells_per_time_bin is not None
    )
    counts_spikes = any(c.quantity == "n_spikes" for c in replay_method.criteria)
    counts = None if spike_counts is None else np.asarray(spike_counts)
    if counts is not None and counts.ndim == 1:
        if counts_cells:
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: method {replay_method.name!r} counts active "
                    f"cells, but spike_counts has shape {counts.shape}.",
                    why="spike counts summed over cells cannot count active "
                    "cells: every spike would count as one cell",
                    fix="pass (n_time_bins, n_cells) counts, one column per cell",
                )
            )
        # Summed counts say only which time bins have spikes.
        counts = counts[:, np.newaxis]
    if counts is not None and len(counts) != len(posterior):
        raise ValueError(
            f"{_OPERATION}: spike_counts has {len(counts)} rows but posterior "
            f"has {len(posterior)}; pass one row of counts per posterior row."
        )
    if counts is not None and counts_cells:
        event_values["n_active_cells"] = float(np.sum(counts.sum(axis=0) > 0))
        event_values["active_fraction"] = event_values["n_active_cells"] / max(
            counts.shape[1], 1
        )
    if counts is not None and counts_spikes:
        event_values["n_spikes"] = float(counts.sum())
    if valid_time_bins is not None:
        from neurospatial.replay.segments import _valid_time_bins

        valid_time_bins = _valid_time_bins(
            valid_time_bins, len(posterior), operation=_OPERATION
        )
    needs_counts = (
        replay_method.empty_time_bins != "keep" or counts_cells or counts_spikes
    )
    if needs_counts and counts is None:
        raise ValueError(
            _format_error(
                f"{_OPERATION}: method {replay_method.name!r} needs spike_counts.",
                why="it drops or zeroes time bins without spikes, counts the "
                "event's spikes, or counts the cells active in the event or in "
                "each time bin",
                fix="pass the event's spike_counts, one row per posterior row",
            )
        )
    # A method that segments keeps time bins without spikes as invalid rows,
    # so no jump is measured across them; it drops them from the segment.
    valid = (
        None
        if segmentation is None
        else _valid_rows(replay_method, counts, valid_time_bins, len(posterior))
    )

    score = np.nan
    p_value = np.nan
    n_shuffles_used = None
    z_scores: dict[str, float] = {}
    segments = None
    segment = None
    # Rows of the event still tested, to cut a per-time-bin prior with them.
    n_event_rows = len(posterior)
    kept_rows = np.arange(n_event_rows)
    quantities = replay_method.tested_quantities
    untested = dict.fromkeys(quantities, np.nan)
    quantity_test: _QuantityTest | None = None
    score_seed = None
    if quantities:
        # The score's test and every quantity's test start from one seed, so
        # they draw the same shuffles; a sampled score gets its own seed.
        generator = np.random.default_rng(rng)
        rng = shuffle_seed = int(generator.integers(2**63 - 1))
        score_seed = int(generator.integers(2**63 - 1))
        event_posterior, event_times, event_counts = posterior, times, counts

        def test_quantities(rows: NDArray[np.intp]) -> _QuantityTest:
            """The method's quantities on the event's ``rows``, against its nulls."""
            return _quantity_test(
                replay_method,
                env,
                track_env,
                event_posterior[rows],
                event_times[rows],
                spike_counts=None if event_counts is None else event_counts[rows],
                encoding_models=encoding_models,
                dt=dt,
                prior=_prior_rows(prior, rows, n_event_rows),
                template=template,
                pseudo_event_pool=pseudo_event_pool,
                seed=shuffle_seed,
            )

    track_length = (
        _track_length(track_env, f"method {replay_method.name!r}")
        if any(c.relative_to for c in replay_method.criteria)
        else np.nan
    )
    if segmentation is not None:
        from neurospatial.replay.segments import replay_segments

        segment_options = segmentation.resolved_options(
            track_env,
            n_decoded_bins=_decoded_bins(
                env,
                track_env,
                template,
                encoding_models,
                None if template is None else joint,
            )
            if "chance" in segmentation.relative_to.values()
            else None,
        )
        if replay_method.score is None and replay_method.nulls and not quantities:
            # The segment-detection test (Igata et al. 2021).
            if valid_time_bins is not None and any(
                name in _ROW_MOVING_NULLS for name in replay_method.nulls
            ):
                raise ValueError(
                    _format_error(
                        f"{_OPERATION}: valid_time_bins cannot be used with the "
                        f"nulls of method {replay_method.name!r}.",
                        why="those nulls move or change the time bins, so the "
                        "event's valid time bins do not apply to its shuffles",
                        fix="leave out valid_time_bins, or use nulls that keep "
                        "the time bins in place",
                    )
                )
            p_value, n_shuffles_used = _segment_detection_p_value(
                replay_method,
                env,
                track_env,
                posterior,
                times,
                segment_options=segment_options,
                valid=valid,
                track_length=track_length,
                spike_counts=counts,
                encoding_models=encoding_models,
                dt=dt,
                prior=prior,
                template=template,
                pseudo_event_pool=pseudo_event_pool,
                rng=rng,
            )
        segments = replay_segments(
            track_env, posterior, times, valid_time_bins=valid, **segment_options
        )
        segment = _tested_segment(replay_method, segments, track_length)
        if segment is None:
            return ReplayDetectionResult(
                method=replay_method.name,
                is_replay=False,
                score=np.nan,
                p_value=p_value,
                criteria={c.label: False for c in replay_method.criteria},
                values={c.label: np.nan for c in replay_method.criteria},
                segments=segments,
                segment=None,
                template=template,
                n_shuffles_used=n_shuffles_used,
                tested_values=untested,
                tested_p_values=untested,
            )
        if quantities and segmentation.choose == "any":
            # Test the segments that meet the segment criteria from the
            # longest down; report the first significant one, else the longest.
            passing = np.flatnonzero(
                _segment_passes(replay_method, segments, track_length)
            )
            order = passing[
                np.argsort(-segments.time_bin_counts[passing], kind="stable")
            ]
            for candidate in order:
                candidate_rows = _segment_rows(segmentation, segments, int(candidate))
                if replay_method.empty_time_bins == "drop":
                    assert counts is not None
                    candidate_rows = candidate_rows[
                        counts[candidate_rows].sum(axis=1) > 0
                    ]
                found = test_quantities(candidate_rows)
                if quantity_test is None or found.p_value < replay_method.alpha:
                    segment, quantity_test = int(candidate), found
                if found.p_value < replay_method.alpha:
                    break
        rows = segments.rows(segment)
        if joint is not None and not summed:
            on_trajectory = segments.trajectory_time_bins[rows]
            masses = {
                name: float(np.sum(joint.joint(name)[rows][on_trajectory]))
                for name in joint.template_names
            }
            event_values["segment_template_fraction"] = masses[tested] / sum(
                masses.values()
            )
        tested_rows = _segment_rows(segmentation, segments, segment)
        posterior, times = posterior[tested_rows], times[tested_rows]
        counts = None if counts is None else counts[tested_rows]
        kept_rows = kept_rows[tested_rows]

    if replay_method.empty_time_bins == "drop":
        assert counts is not None
        keep = counts.sum(axis=1) > 0
        posterior = np.asarray(posterior[keep])
        times, counts = times[keep], counts[keep]
        kept_rows = kept_rows[keep]
        if separate is not None:
            separate = {name: rows[keep] for name, rows in separate.items()}

    options = dict(replay_method.score_options)
    if "n_samples_per_time_bin" in options:
        options["n_samples"] = int(options.pop("n_samples_per_time_bin")) * len(
            posterior
        )
    if (
        score_seed is not None
        and replay_method.score is not None
        and _takes_rng(replay_method.score)
        and "rng" not in options
    ):
        # Seeded here, so the score's test does not draw the seed from the
        # shuffles' stream and draws the same shuffles as the quantities'.
        options["rng"] = score_seed
    # A template's share of a joint decode is scored as the posterior given
    # the template, unless the score is built for joint weights.
    share = joint is not None and template is not None
    condition = (
        share
        and replay_method.score is not None
        and not _uses_joint_weights(replay_method.score, options)
    )

    def conditioned(rows: NDArray[np.float64]) -> NDArray[np.float64]:
        return _conditional_rows(rows) if condition else rows

    if replay_method.score is not None and replay_method.nulls:
        # The test needs the counts only to decode shuffles again or to zero
        # empty time bins; given without the rate maps they would be refused.
        decodes = any(name in _DECODING_NULLS for name in replay_method.nulls)
        try:
            result = replay_significance(
                env if decodes or separate is not None else track_env,
                # Nulls that decode again condition the share themselves.
                separate
                if separate is not None
                else posterior
                if decodes
                else conditioned(posterior),
                times,
                score=replay_method.score,
                score_options=options,
                nulls=replay_method.nulls,
                null_options=replay_method.null_options,
                spike_counts=counts
                if decodes or replay_method.empty_time_bins == "zero"
                else None,
                encoding_models=encoding_models if decodes else None,
                dt=dt if decodes else None,
                prior=_prior_rows(prior, kept_rows, n_event_rows),
                template=template if decodes else None,
                pseudo_event_pool=pseudo_event_pool,
                alternative=replay_method.alternative,
                use_absolute=replay_method.use_absolute,
                empty_time_bins="zero"
                if replay_method.empty_time_bins == "zero"
                else "keep",
                n_shuffles=replay_method.n_shuffles,
                rng=rng,
            )
        except _UndefinedScoreError:
            pass  # The event's score is NaN: no verdict, below.
        except _UndefinedNullError as error:
            # The score is defined but a null left no shuffle to compare with.
            return ReplayDetectionResult(
                method=replay_method.name,
                is_replay=False,
                score=error.score,
                p_value=np.nan,
                criteria={c.label: False for c in replay_method.criteria},
                values={c.label: np.nan for c in replay_method.criteria},
                segments=segments,
                segment=segment,
                template=template,
                tested_values=untested,
                tested_p_values=untested,
            )
        else:
            score, p_value = float(result.observed.score), float(result.p_value)
            n_shuffles_used = min(result.n_shuffles_used.values())
            z_scores = {
                name: float(test.z_score) for name, test in result.shuffles.items()
            }
            if separate is not None:
                # Criteria are evaluated on the template with the largest score.
                assert result.template is not None
                template = result.template
                posterior, track_env = separate[template], template_envs[template]
                if any(c.relative_to for c in replay_method.criteria):
                    track_length = _track_length(
                        track_env, f"method {replay_method.name!r}"
                    )
    elif replay_method.score is not None:
        scored = conditioned(posterior)
        if replay_method.empty_time_bins == "zero":
            assert counts is not None
            scored = np.where(counts.sum(axis=1)[:, None] == 0, 0.0, scored)
        if _takes_rng(replay_method.score) and "rng" not in options:
            # A sampled score draws from the method's seed, as it does in a test.
            options["rng"] = int(np.random.default_rng(rng).integers(2**63 - 1))
        score = float(replay_method.score(track_env, scored, times, **options).score)

    if _verdict_uses_score(replay_method) and not np.isfinite(score):
        # An undefined score (such as R² of a decoded position that does not
        # change) gives no verdict rather than a failed or passed criterion.
        return ReplayDetectionResult(
            method=replay_method.name,
            is_replay=False,
            score=np.nan,
            p_value=np.nan,
            criteria={c.label: False for c in replay_method.criteria},
            values={c.label: np.nan for c in replay_method.criteria},
            segments=segments,
            segment=segment,
            template=template,
            tested_values=untested,
            tested_p_values=untested,
        )
    if quantities:
        if quantity_test is None:
            quantity_test = test_quantities(kept_rows)
        # Every test must reject: the largest p-value, NaN when one is NaN.
        tests = [quantity_test.p_value] + (
            [] if replay_method.score is None else [p_value]
        )
        p_value = float(np.max(tests))
        used = [
            n for n in (quantity_test.n_shuffles_used, n_shuffles_used) if n is not None
        ]
        n_shuffles_used = min(used) if used else None

    needs_metrics = any(
        c.quantity
        not in (
            "score",
            "n_active_cells",
            "active_fraction",
            "n_spikes",
            *_SEGMENT_QUANTITIES,
            *_TEMPLATE_QUANTITIES,
        )
        for c in replay_method.criteria
    )
    metrics = (
        trajectory_metrics(track_env, posterior, times, spike_counts=counts)
        if needs_metrics
        else None
    )
    passes: dict[str, bool] = {}
    values: dict[str, float] = {}
    for criterion in replay_method.criteria:
        if criterion.quantity == "score":
            value = score
        elif criterion.quantity in event_values:
            value = event_values[criterion.quantity]
        elif criterion.quantity in _SEGMENT_QUANTITIES:
            assert segments is not None and segment is not None
            value = float(
                getattr(segments, _SEGMENT_QUANTITIES[criterion.quantity])[segment]
            )
        else:
            value = float(getattr(metrics, criterion.quantity))
        values[criterion.label] = value
        passes[criterion.label] = criterion.evaluate(value, track_length)
    significant = bool(p_value < replay_method.alpha) if replay_method.nulls else True
    return ReplayDetectionResult(
        method=replay_method.name,
        is_replay=significant and all(passes.values()),
        score=score,
        p_value=p_value,
        criteria=passes,
        values=values,
        segments=segments,
        segment=segment,
        template=template,
        n_shuffles_used=n_shuffles_used,
        tested_values={} if quantity_test is None else quantity_test.values,
        tested_p_values={} if quantity_test is None else quantity_test.p_values,
        z_scores=z_scores,
    )


def _smallest_p_template(
    env: Environment | Mapping[str, Environment],
    posterior: JointDecodingResult | Mapping[str, NDArray[np.float64]],
    times: NDArray[np.float64],
    *,
    replay_method: ReplayMethod,
    spike_counts: NDArray[np.int64] | None,
    encoding_models: Any,
    dt: float | None,
    prior: Any,
    pseudo_event_pool: NDArray[np.float64] | None,
    valid_time_bins: NDArray[np.bool_] | None,
    rng: np.random.Generator | int | None,
) -> ReplayDetectionResult:
    """Test every template and report the one with the smallest p-value.

    Each template is tested as :func:`detect_replay` with ``template=``
    would (a joint decode's share), or on its own posterior and rate maps
    (templates decoded separately), with its own random stream. The first
    template in the decode's order wins a tie; templates without a p-value
    come last.
    """
    from neurospatial.decoding import JointDecodingResult
    from neurospatial.replay.significance import _separate_inputs, _template_streams

    # Per template: environment, posterior, rate maps, prior and template=.
    tests: dict[str, tuple[Any, Any, Any, Any, str | None]]
    if isinstance(posterior, JointDecodingResult):
        tests = {
            name: (env, posterior, encoding_models, prior, name)
            for name in posterior.template_names
        }
    else:
        names, envs, models, priors = _separate_inputs(
            env, posterior, encoding_models, prior, _OPERATION
        )
        tests = {
            name: (
                envs[name],
                np.asarray(posterior[name], dtype=np.float64),
                models[name],
                priors[name],
                None,
            )
            for name in names
        }
    streams = _template_streams(rng, list(tests))
    results = []
    for name, (template_env, tested, models_, prior_, share) in tests.items():
        result = detect_replay(
            template_env,
            tested,
            times,
            replay_method=replay_method,
            spike_counts=spike_counts,
            encoding_models=models_,
            dt=dt,
            prior=prior_,
            template=share,
            pseudo_event_pool=pseudo_event_pool,
            valid_time_bins=valid_time_bins,
            rng=streams[name],
        )
        results.append(dataclasses.replace(result, template=name))
    return min(
        results,
        key=lambda result: (
            bool(np.isnan(result.p_value)),
            0.0 if np.isnan(result.p_value) else result.p_value,
        ),
    )


def _prior_rows(prior: Any, rows: NDArray[np.intp], n_time_bins: int) -> Any:
    """A per-time-bin prior cut to the event's tested ``rows``.

    A prior with one row per time bin, ``(n_time_bins, n_bins)`` or a
    mapping of them for a joint decode, is cut like the posterior and the
    counts; a stationary prior is returned as it is.
    """
    if prior is None:
        return None
    if isinstance(prior, Mapping):
        return {
            name: _prior_rows(weights, rows, n_time_bins)
            for name, weights in prior.items()
        }
    values = np.asarray(prior)
    if values.ndim == 2 and len(values) == n_time_bins:
        return values[rows]
    return prior


def _track_length(env: Environment, operation: str) -> float:
    """Summed length of the track's bins, as TrajectoryMetrics.track_length."""
    from neurospatial.environment._queries import _require_track_extents

    starts, stops = _require_track_extents(
        env, operation, alternative="or give thresholds in environment units"
    )
    return float(np.sum(stops - starts))


def _decoded_bins(
    env: Environment | Mapping[str, Environment],
    track_env: Environment,
    template: str | None,
    encoding_models: Any,
    joint: JointDecodingResult | None,
) -> int:
    """Bins of every template decoded together, for thresholds over chance."""
    if joint is not None:
        return sum(result.env.n_bins for result in joint.conditional.values())
    if template is None:
        return track_env.n_bins
    if isinstance(env, Mapping):
        return sum(template_env.n_bins for template_env in env.values())
    if isinstance(encoding_models, Mapping):
        return len(encoding_models) * track_env.n_bins
    raise ValueError(
        _format_error(
            f"{_OPERATION}: a threshold relative to chance on one template's "
            "share needs every template's rate maps to count the decoded bins.",
            why="chance is 1 / (bins of all templates decoded together)",
            fix="pass encoding_models, the mapping of every template's rate "
            "maps given to decode_jointly",
        )
    )


def _valid_rows(
    replay_method: ReplayMethod,
    counts: NDArray[np.int64] | None,
    valid_time_bins: NDArray[np.bool_] | None,
    n_time_bins: int,
) -> NDArray[np.bool_] | None:
    """The caller's valid time bins, the active-cell rule and empty bins."""
    from neurospatial.replay.segments import _valid_time_bins

    assert replay_method.segmentation is not None
    valid = (
        None
        if valid_time_bins is None
        else _valid_time_bins(valid_time_bins, n_time_bins, operation=_OPERATION)
    )
    cells = replay_method.segmentation.min_active_cells_per_time_bin
    if cells is not None:
        assert counts is not None
        active = np.sum(counts > 0, axis=1) >= cells
        valid = active if valid is None else valid & active
    if replay_method.empty_time_bins != "keep":
        assert counts is not None
        spiking = counts.sum(axis=1) > 0
        valid = spiking if valid is None else valid & spiking
    return valid


def _segment_passes(
    replay_method: ReplayMethod,
    segments: ReplaySegmentsResult,
    track_length: float,
) -> NDArray[np.bool_]:
    """Whether each segment meets every ``segment_*`` criterion."""
    ok = np.ones(segments.n_segments, dtype=bool)
    for criterion in replay_method.criteria:
        if criterion.quantity in _SEGMENT_QUANTITIES:
            values = getattr(segments, _SEGMENT_QUANTITIES[criterion.quantity])
            ok &= [criterion.evaluate(float(v), track_length) for v in values]
    return ok


def _tested_segment(
    replay_method: ReplayMethod,
    segments: ReplaySegmentsResult,
    track_length: float,
) -> int | None:
    """Index of the segment the method tests; None without segments."""
    assert replay_method.segmentation is not None
    if segments.longest is None or replay_method.segmentation.choose == "longest":
        return segments.longest
    passing = np.flatnonzero(_segment_passes(replay_method, segments, track_length))
    if passing.size == 0:
        return segments.longest
    return int(passing[np.argmax(segments.time_bin_counts[passing])])


def _segment_detected(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    replay_method: ReplayMethod,
    segment_options: dict[str, Any],
    valid: NDArray[np.bool_] | None,
    track_length: float,
) -> SimpleNamespace:
    """Score 1.0 when the event has a segment meeting the segment criteria."""
    from neurospatial.replay.segments import replay_segments

    segments = replay_segments(
        env, posterior, times, valid_time_bins=valid, **segment_options
    )
    segment = _tested_segment(replay_method, segments, track_length)
    detected = (
        segment is not None
        and _segment_passes(replay_method, segments, track_length)[segment]
    )
    return SimpleNamespace(score=float(detected))


def _segment_detection_p_value(
    replay_method: ReplayMethod,
    env: Environment | Mapping[str, Environment],
    track_env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    segment_options: dict[str, Any],
    valid: NDArray[np.bool_] | None,
    track_length: float,
    spike_counts: NDArray[np.int64] | None,
    encoding_models: Any,
    dt: float | None,
    prior: Any,
    template: str | None,
    pseudo_event_pool: NDArray[np.float64] | None,
    rng: np.random.Generator | int | None,
) -> tuple[float, int | None]:
    """Share of shuffles that also have a segment meeting the criteria.

    Without such a segment in the event, every shuffle is at least as
    extreme and the p-value is 1, so no shuffle is drawn. Also returns the
    shuffles the p-value counts, the smallest over the nulls (None when no
    shuffle is drawn).
    """
    from neurospatial.replay.significance import _nulls_to_run, replay_significance

    decodes = any(name in _DECODING_NULLS for name in replay_method.nulls)
    # Refuse missing null inputs whether or not the event has a segment.
    _nulls_to_run(
        replay_method.nulls,
        spike_counts if decodes else None,
        encoding_models if decodes else None,
        dt if decodes else None,
        pseudo_event_pool,
    )
    options: dict[str, Any] = {
        "replay_method": replay_method,
        "segment_options": segment_options,
        "valid": valid,
        "track_length": track_length,
    }
    if not _segment_detected(track_env, posterior, times, **options).score:
        return 1.0, None
    result = replay_significance(
        env if decodes else track_env,
        posterior,
        times,
        score=_segment_detected,
        score_options=options,
        nulls=replay_method.nulls,
        null_options=replay_method.null_options,
        spike_counts=spike_counts if decodes else None,
        encoding_models=encoding_models if decodes else None,
        dt=dt if decodes else None,
        prior=prior,
        template=template if decodes else None,
        pseudo_event_pool=pseudo_event_pool,
        alternative="greater",
        n_shuffles=replay_method.n_shuffles,
        rng=rng,
    )
    return float(result.p_value), min(result.n_shuffles_used.values())


def _segment_rows(
    segmentation: ReplaySegmentation, segments: ReplaySegmentsResult, segment: int
) -> NDArray[np.intp]:
    """The event rows of a segment that its test uses (``segmentation.rows``)."""
    rows = segments.rows(segment)
    tested_rows = np.arange(rows.start, rows.stop)
    if segmentation.rows == "trajectory":
        tested_rows = tested_rows[segments.trajectory_time_bins[rows]]
    return tested_rows


def _trajectory_quantity(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    quantity: str,
) -> SimpleNamespace:
    """One scalar of :func:`trajectory_metrics`, as a score to test."""
    from neurospatial.replay.criteria import trajectory_metrics

    metrics = trajectory_metrics(env, posterior, times)
    return SimpleNamespace(score=float(getattr(metrics, quantity)))


@dataclass(frozen=True)
class _QuantityTest:
    """A method's tested quantities on one event's rows, against its nulls."""

    values: dict[str, float]
    p_values: dict[str, float]
    n_shuffles_used: int | None

    @property
    def p_value(self) -> float:
        """The largest p-value over the quantities; NaN when one is NaN."""
        return float(np.max(list(self.p_values.values())))


def _quantity_test(
    replay_method: ReplayMethod,
    env: Environment | Mapping[str, Environment],
    track_env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    spike_counts: NDArray[np.int64] | None,
    encoding_models: Any,
    dt: float | None,
    prior: Any,
    template: str | None,
    pseudo_event_pool: NDArray[np.float64] | None,
    seed: int,
) -> _QuantityTest:
    """Test each of the method's ``tested_quantities`` against its nulls.

    Each quantity is tested with :func:`replay_significance` from ``seed``,
    so every quantity, and the score's test from the same seed, sees the
    same shuffles. A quantity undefined on the event (such as the mean jump
    of one time bin), or on every shuffle of a null, has a NaN p-value.
    """
    from neurospatial.replay.significance import (
        _UndefinedNullError,
        _UndefinedScoreError,
        replay_significance,
    )

    decodes = any(name in _DECODING_NULLS for name in replay_method.nulls)
    values: dict[str, float] = {}
    p_values: dict[str, float] = {}
    used = []
    for quantity, tail in replay_method.tested_quantities.items():
        try:
            result = replay_significance(
                env if decodes else track_env,
                posterior,
                times,
                score=_trajectory_quantity,
                score_options={"quantity": quantity},
                nulls=replay_method.nulls,
                null_options=replay_method.null_options,
                spike_counts=spike_counts if decodes else None,
                encoding_models=encoding_models if decodes else None,
                dt=dt if decodes else None,
                prior=prior,
                template=template if decodes else None,
                pseudo_event_pool=pseudo_event_pool,
                alternative=tail,
                use_absolute=False,
                n_shuffles=replay_method.n_shuffles,
                rng=seed,
            )
        except _UndefinedScoreError:
            values[quantity], p_values[quantity] = np.nan, np.nan
        except _UndefinedNullError as error:
            values[quantity], p_values[quantity] = error.score, np.nan
        else:
            values[quantity] = float(result.observed.score)
            p_values[quantity] = float(result.p_value)
            used.append(min(result.n_shuffles_used.values()))
    return _QuantityTest(values, p_values, min(used) if used else None)


def detect_rank_order_replay(
    env: Environment | Mapping[str, Environment],
    spike_times: Any,
    interval: tuple[float, float] | NDArray[np.float64],
    encoding_models: Any,
    *,
    rank_order_method: RankOrderMethod,
    template: str | None = None,
    rng: np.random.Generator | int | None = None,
) -> ReplayDetectionResult:
    """Apply one published rank-order method to one candidate event.

    Parameters
    ----------
    env : Environment or Mapping[str, Environment]
        The environment of the rate maps: a 1-D environment or a linearized
        track; or one environment per template.
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell, in the order of the rows
        of ``encoding_models``.
    interval : (start, stop)
        The candidate event, in seconds.
    encoding_models : array of shape (n_cells, n_bins), SpatialRatesResult, or Mapping
        Each cell's rate map on ``env``, in Hz; or a mapping of such rate
        maps, one per template (such as each running direction), with the
        cells in the same order.
    rank_order_method : RankOrderMethod
        The method, for example
        ``neurospatial.replay.methods.DIBA_BUZSAKI_2007``.
    template : str or None
        With a mapping of rate maps, the template to test; None lets the
        method's ``template_choice`` choose.
    rng : np.random.Generator, int, or None
        Random number generator or seed for the cell-order permutations.

    Returns
    -------
    ReplayDetectionResult
        The verdict, the rank-order correlation as ``score``, its p-value,
        the active-cell criteria and, with several templates, the tested
        ``template``. An event with too few active template cells, or with
        all field positions or firing times equal, has a NaN score and is
        not replay.

    Raises
    ------
    ValueError
        If a mapping of rate maps is given without ``template=`` to a method
        without a ``template_choice``, or ``template`` is not one of its
        keys; or for inputs
        :func:`~neurospatial.replay.rank_order_correlation` refuses.

    See Also
    --------
    neurospatial.replay.rank_order_correlation : The correlation.
    neurospatial.replay.rank_order_significance : The permutation test.
    compare_replay_methods : Several methods, rank-order and decoded, on
        several events.

    Examples
    --------
    Eight cells fire in reverse field order within 80 ms:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import detect_rank_order_replay
    >>> from neurospatial.replay.methods import DIBA_BUZSAKI_2007
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [120.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> rates = 20.0 * np.exp(
    ...     -0.5 * ((x - np.linspace(10.0, 110.0, 8)[:, None]) / 8.0) ** 2
    ... )
    >>> spikes = [np.array([1.07 - 0.01 * i]) for i in range(8)]
    >>> result = detect_rank_order_replay(
    ...     env, spikes, (1.0, 1.1), rates, rank_order_method=DIBA_BUZSAKI_2007, rng=0
    ... )
    >>> result.is_replay, result.score
    (True, -1.0)
    """
    method = rank_order_method
    if method.test == "pooled_ssi":
        raise ValueError(
            _format_error(
                f"detect_rank_order_replay(): method {method.name!r} compares "
                "each event with the shuffles of every event in its epoch.",
                why="its threshold is a quantile of shuffles pooled over the "
                "epoch's events, which one event cannot give",
                fix="pass the epoch's events to compare_replay_methods",
            )
        )
    if isinstance(encoding_models, Mapping):
        names: list[str | None] = list(encoding_models)
        if template is not None:
            if template not in encoding_models:
                raise ValueError(
                    f"detect_rank_order_replay(): template {template!r} is not "
                    f"one of {list(encoding_models)}."
                )
            names = [template]
        elif method.template_choice is None:
            raise ValueError(
                _format_error(
                    f"detect_rank_order_replay(): method {method.name!r} needs "
                    "template= with several rate-map sets.",
                    why="it tests one template and has no rule to choose one",
                    fix=f"pass template= one of {list(encoding_models)}",
                )
            )
    else:
        names = [None]
    generator = np.random.default_rng(rng)
    seeds = generator.integers(2**63, size=len(names))
    results = []
    for name, seed in zip(names, seeds, strict=True):
        if isinstance(env, Mapping):
            if name is None:
                raise ValueError(
                    "detect_rank_order_replay(): a mapping of environments "
                    "needs a mapping of rate maps with the same templates."
                )
            template_env = env[name]
        else:
            template_env = env
        results.append(
            _rank_order_verdict(
                template_env,
                spike_times,
                interval,
                encoding_models if name is None else encoding_models[name],
                method,
                template=name,
                rng=np.random.default_rng(seed),
            )
        )
    if len(results) == 1:
        return results[0]
    detected = [result for result in results if result.is_replay]

    def strength(result: ReplayDetectionResult) -> tuple[float, float]:
        cells = result.values[method.criteria[0].label]
        return cells, abs(result.score) if np.isfinite(result.score) else -1.0

    return max(detected or results, key=strength)


def _rank_order_verdict(
    env: Environment,
    spike_times: Any,
    interval: Any,
    encoding_models: Any,
    method: RankOrderMethod,
    *,
    template: str | None,
    rng: np.random.Generator,
) -> ReplayDetectionResult:
    """One template's rank-order verdict."""
    from scipy import stats

    from neurospatial.replay.rank_order import (
        _event_pairs,
        _field_positions,
        _permuted_correlations,
        _result,
        _supported_rates,
    )
    from neurospatial.stats.shuffle import (
        compute_shuffle_pvalue,
        compute_shuffle_zscore,
    )

    operation = "detect_rank_order_replay()"
    # Validates the rates against env, then keeps the template cells' maps.
    _field_positions(env, encoding_models, method.field_order, operation)
    rates = _supported_rates(encoding_models)
    if method.min_peak_rate is not None:
        peak = np.where(np.isfinite(rates), rates, 0.0).max(axis=1)
        rates = np.where((peak >= method.min_peak_rate)[:, None], rates, np.nan)
    pairs = _event_pairs(
        env,
        spike_times,
        interval,
        rates,
        spike_statistic=method.spike_statistic,
        field_order=method.field_order,
        min_active_cells=method.min_active_cells,
        operation=operation,
    )
    observed = _result(pairs, method.spike_statistic, method.field_order)
    score = float(observed.score)
    p_value = np.nan
    n_shuffles_used = None
    z_scores: dict[str, float] = {}
    if np.isfinite(score):
        if method.test == "t_distribution":
            n = pairs.times.size
            if abs(score) >= 1.0:
                p_value = 0.0
            else:
                t = score * np.sqrt((n - 2) / (1.0 - score**2))
                p_value = float(2.0 * stats.t.sf(abs(t), n - 2))
        else:
            n_active = observed.n_active_cells
            permutations = np.argsort(rng.random((method.n_shuffles, n_active)), axis=1)
            null = _permuted_correlations(pairs, permutations)
            n_shuffles_used = int(np.isfinite(null).sum())
            # Ties in rho differ by rounding between the two formulas.
            p_value = float(
                compute_shuffle_pvalue(
                    round(score, _SCORE_DECIMALS),
                    np.round(null, _SCORE_DECIMALS),
                    tail=method.alternative,
                )
            )
            z_scores["cell_order"] = float(
                compute_shuffle_zscore(score, null[np.isfinite(null)])
            )
    n_templates = int(
        np.isfinite(_field_positions(env, rates, method.field_order, operation)).sum()
    )
    event_values = {
        "n_active_cells": float(observed.n_active_cells),
        "active_fraction": observed.n_active_cells / max(n_templates, 1),
    }
    passes: dict[str, bool] = {}
    values: dict[str, float] = {}
    for criterion in method.criteria:
        values[criterion.label] = event_values[criterion.quantity]
        passes[criterion.label] = criterion.evaluate(
            event_values[criterion.quantity], np.nan
        )
    return ReplayDetectionResult(
        method=method.name,
        is_replay=bool(p_value < method.alpha) and all(passes.values()),
        score=score,
        p_value=p_value,
        criteria=passes,
        values=values,
        template=template,
        n_shuffles_used=n_shuffles_used,
        z_scores=z_scores,
    )


def _pooled_ssi_verdicts(
    method: RankOrderMethod,
    env: Environment | Mapping[str, Environment],
    spike_times: Any,
    intervals: list[tuple[float, float]],
    encoding_models: Any,
    template: str | None,
    streams: list[np.random.Generator],
) -> list[tuple[ReplayDetectionResult, str]]:
    """Each event's verdict under the epoch-pooled SSI test (Chenani 2019)."""
    from scipy import stats

    from neurospatial._spikes import as_spike_trains
    from neurospatial.replay.rank_order import _field_positions, _supported_rates

    operation = "compare_replay_methods()"
    if isinstance(encoding_models, Mapping):
        if template is not None and template not in encoding_models:
            raise ValueError(
                f"{operation}: template {template!r} is not one of "
                f"{list(encoding_models)}."
            )
        names: list[str | None] = (
            [template] if template is not None else list(encoding_models)
        )
    else:
        names = [None]
    positions = []
    for name in names:
        models = encoding_models if name is None else encoding_models[name]
        if isinstance(env, Mapping):
            if name is None:
                raise ValueError(
                    f"{operation}: a mapping of environments needs a mapping of "
                    "rate maps with the same templates."
                )
            template_env = env[name]
        else:
            template_env = env
        position = _field_positions(template_env, models, method.field_order, operation)
        if method.min_peak_rate is not None:
            rates = _supported_rates(models)
            peak = np.where(np.isfinite(rates), rates, 0.0).max(axis=1)
            position = np.where(peak >= method.min_peak_rate, position, np.nan)
        positions.append(position)
    fields = np.vstack(positions)  # (n_templates, n_cells)
    trains = as_spike_trains(spike_times)
    if len(trains) != fields.shape[1]:
        raise ValueError(
            f"{operation}: spike_times has {len(trains)} cells but "
            f"encoding_models has {fields.shape[1]} rate maps."
        )
    summaries: dict[str, Callable[..., Any]] = {
        "first": np.min,
        "median": np.median,
        "mean": np.mean,
    }
    summary = summaries[method.spike_statistic]
    template_cell = np.isfinite(fields).any(axis=0)

    def ssi(field: NDArray[np.float64], times: NDArray[np.float64]) -> Any:
        """SSI of each row of ``times`` against the cells with a field."""
        known = np.isfinite(field)
        n = int(known.sum())
        if n < 2:
            return np.full(times.shape[0], np.nan)
        field_dev = stats.rankdata(field[known]) - (n + 1) / 2
        time_ranks = stats.rankdata(times[:, known], axis=1)
        time_dev = time_ranks - time_ranks.mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = (time_dev @ field_dev) / np.sqrt(
                (time_dev**2).sum(axis=1) * (field_dev**2).sum()
            )
        # testsession.m rounds rho to 1e-5 before scaling it.
        return np.round(rho, 5) * np.sqrt(n - 1)

    observed: list[tuple[int, NDArray[np.float64]] | None] = []
    surrogates: list[NDArray[np.float64]] = []
    for (start, stop), stream in zip(intervals, streams, strict=True):
        in_event = [train[(train >= start) & (train < stop)] for train in trains]
        active = np.flatnonzero(
            np.array([spikes.size > 0 for spikes in in_event]) & template_cell
        )
        if active.size < method.min_active_cells:
            observed.append(None)
            continue
        times = np.array([summary(in_event[cell]) for cell in active])
        order = np.argsort(stream.random((method.n_shuffles, active.size)), axis=1)
        event_scores = np.vstack(
            [ssi(field[active], times[None, :]) for field in fields]
        )
        shuffled = np.vstack([ssi(field[active], times[order]) for field in fields])
        observed.append((active.size, event_scores[:, 0]))
        # fmax skips a template without two shared cells, without warnings.
        surrogates.append(np.fmax.reduce(shuffled, axis=0))
    pooled = np.concatenate(surrogates) if surrogates else np.empty(0)
    pooled = pooled[np.isfinite(pooled)]
    n_events = len(surrogates)
    threshold = np.nan
    no_threshold = "the epoch has no shuffle with an SSI to set the threshold"
    if pooled.size:
        grid = np.arange(-400, 401) / 100.0  # testsession.m: b0 = -4:.01:4
        cdf = np.searchsorted(np.sort(pooled), grid, side="right") / pooled.size
        above = np.flatnonzero(cdf > 1.0 - method.alpha)
        if above.size:
            threshold = float(grid[above[0]])
        else:
            no_threshold = (
                f"the 1 - alpha quantile of the epoch's {pooled.size} shuffled "
                "SSIs is above 4, the end of the threshold grid (testsession.m "
                "b0 = -4:.01:4); use a larger alpha"
            )
    epoch_ok = method.min_epoch_events is None or n_events >= method.min_epoch_events
    label = method.criteria[0].label
    verdicts = []
    for entry in observed:
        if entry is None:
            verdicts.append(
                (
                    ReplayDetectionResult(
                        method=method.name,
                        is_replay=False,
                        score=np.nan,
                        p_value=np.nan,
                        criteria={label: False},
                        values={label: np.nan},
                    ),
                    f"fewer than {method.min_active_cells} active template cells",
                )
            )
            continue
        n_active, scores = entry
        finite = np.isfinite(scores)
        best = int(np.nanargmax(scores)) if finite.any() else 0
        score = float(scores[best]) if finite.any() else np.nan
        p_value = (
            float((np.sum(pooled >= score) + 1) / (pooled.size + 1))
            if np.isfinite(score)
            else np.nan
        )
        if not finite.any():
            error = (
                "rank-order correlation undefined: no variation in the active "
                "cells' order"
            )
        elif not epoch_ok:
            error = (
                f"the epoch has {n_events} events with at least "
                f"{method.min_active_cells} active cells, fewer than "
                f"{method.min_epoch_events}"
            )
        elif np.isnan(threshold):
            error = no_threshold
        else:
            error = ""
        verdicts.append(
            (
                ReplayDetectionResult(
                    method=method.name,
                    is_replay=bool(epoch_ok and score > threshold),
                    score=score,
                    p_value=p_value,
                    criteria={label: True},
                    values={label: float(n_active)},
                    template=names[best] if finite.any() else None,
                    n_shuffles_used=int(pooled.size) if np.isfinite(p_value) else None,
                ),
                error,
            )
        )
    return verdicts


def compare_replay_methods(
    env: Environment | Mapping[str, Environment],
    events: Sequence[
        DecodingResult | JointDecodingResult | Mapping[str, DecodingResult]
    ],
    *,
    replay_methods: Sequence[
        ReplayMethod
        | RankOrderMethod
        | PairwiseOrderMethod
        | ArmReplayMethod
        | TemplateMatchMethod
        | BiasSequenceMethod
    ],
    spike_counts: Sequence[NDArray[np.int64]] | None = None,
    encoding_models: Any = None,
    dt: float | None = None,
    prior: Any = None,
    template: str | None = None,
    valid_time_bins: Sequence[NDArray[np.bool_]] | None = None,
    spike_times: Any = None,
    arms: Mapping[str, NDArray[np.intp]] | None = None,
    template_bias: NDArray[np.float64] | None = None,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Apply several published replay methods to several decoded events.

    Parameters
    ----------
    env : Environment or Mapping[str, Environment]
        The environment the events were decoded on, or the mapping of
        environments of a joint decode (see :func:`detect_replay`).
    events : sequence of DecodingResult, JointDecodingResult or Mapping
        The decoded candidate events, each with ``times``: a decode on one
        set of rate maps, a joint decode over several templates, or, for
        methods with ``template_choice="smallest_p"`` or ``"max_score"``,
        the templates decoded separately, as a mapping of template name to
        :class:`~neurospatial.decoding.DecodingResult` with the same times
        (see :func:`detect_replay`).
    replay_methods : sequence of ReplayMethod, RankOrderMethod, PairwiseOrderMethod, ArmReplayMethod, TemplateMatchMethod or BiasSequenceMethod
        The methods to compare. An arm method classifies each event's
        posterior as a single-arm or joint-arm replay on ``arms`` (see
        :func:`~neurospatial.replay.detect_arm_replay`). A rank-order method scores each event's
        spikes over the event's decoding windows, from the first window's
        start to the last window's end (``times[0] - dt / 2`` to
        ``times[-1] + dt / 2``), against ``encoding_models``: with a mapping
        of rate maps, the one named by ``template``, or the one its
        ``template_choice`` picks. A method with ``test="pooled_ssi"``
        treats the events as one epoch (pass one epoch per call) and scores
        every template unless ``template`` names one. A pairwise-order
        method (:class:`~neurospatial.replay.PairwiseOrderMethod`) scores
        the spikes over the same window, on every template of a mapping of
        rate maps in each of its directions, and reports the test with the
        smallest p-value; it ignores ``template``.
        A :class:`~neurospatial.replay.TemplateMatchMethod` scores the same
        window's spikes as a rank-order method does (see
        :func:`~neurospatial.replay.detect_template_match_replay`). A
        :class:`~neurospatial.replay.BiasSequenceMethod` compares the same
        window's skew-bias matrix with ``template_bias`` (see
        :func:`~neurospatial.replay.detect_bias_sequence_replay`).
    spike_counts : sequence of arrays or None
        Each event's spike counts, shape (n_time_bins, n_cells), for methods
        that need them (see :func:`detect_replay`).
    encoding_models, dt, prior, template
        The decoding inputs, for the nulls that decode again (see
        :func:`detect_replay`). For joint decodes, ``template`` is the
        template tested by methods without a ``template_choice``; methods
        with one choose the template of each event themselves. For
        separately decoded templates, ``encoding_models`` is a mapping keyed
        like each event.
    valid_time_bins : sequence of arrays or None
        Each event's valid time bins, one boolean per time bin, used by
        methods with a segmentation and ignored by the others (see
        :func:`detect_replay`).
    spike_times : array, sequence of arrays, or spike group, optional
        The session's spike times, one train per row of
        ``encoding_models``, for rank-order, pairwise-order,
        template-matching and bias-sequence methods.
    arms : Mapping[str, NDArray[np.intp]] or None
        Each arm's bins from the junction outward, for arm methods.
    template_bias : NDArray[np.float64] or None
        The template's skew-bias matrix, shape (n_cells, n_cells), for
        bias-sequence methods, such as
        :func:`~neurospatial.replay.mean_pairwise_bias` of the task's theta
        cycles, with the cells in the order of ``spike_times``.
    rng : numpy.random.Generator, int, or None
        Random number generator or seed for the shuffles. Each (event, method)
        pair draws from its own stream, derived from ``rng``, the event's
        position and the method's name, so adding, removing or reordering
        methods does not change the other pairs' results.

    Returns
    -------
    pandas.DataFrame
        One row per (event, method): ``event``, ``method``, ``is_replay``,
        ``score``, ``p_value``, ``n_shuffles_used`` (the shuffles the
        p-value counts, smallest over the method's nulls; NaN for methods
        without shuffles), one ``passes[...]`` column per criterion and a
        ``value[...]`` column with the value it was compared on (such as
        ``value[max_map_jump < 0.4 x track_length]``, the raw jump in
        environment units), a ``z_score[...]`` column per null of the
        score's shuffle test (``z_score[place_field_circular]`` is rZ of
        Grosmark & Buzsáki 2016 with ``use_absolute=True``; ``z_score[
        cell_order]`` for a rank-order permutation test), each
        tested quantity's value and ``p_value[...]`` for methods with
        ``tested_quantities`` (NaN for the others),
        ``n_segments``, ``segment_start_time`` and ``segment_end_time`` for
        methods with a segmentation (NaN for the others), ``template`` for
        joint and separately decoded templates (the template tested or
        chosen), for rank-order methods given a mapping of rate maps and
        for the arm or pair an arm method found, and ``error``.
        ``error`` says why an event has no verdict, and ``is_replay`` and
        every ``passes[...]`` of the method are then False: fewer than two
        time bins left to score (``"fewer than two time bins with spikes to
        score"`` for a method that drops or zeroes time bins without
        spikes), a null whose every shuffle has an undefined score (the
        score is kept, the p-value is NaN), a tested quantity undefined on
        the event or on every shuffle of a null (the p-value is NaN), an
        undefined score of a method
        whose verdict uses it, through nulls or a score criterion (for
        example the R² of a decoded position that does not change; score
        and p-value are
        NaN), an undefined rank-order correlation (too few active template
        cells, or no variation in their order), for a pairwise-order method,
        an event or shuffle whose multi-field cells have more field
        combinations than are searched (``"too many combinations of
        multi-field cells' fields to search"``), for a bias-sequence
        method, a similarity undefined because no pair of cells fired in
        both the event and the template, or, for a pooled SSI
        method, an epoch with too few events or without a threshold (score
        and p-value are still reported). A method with a segmentation that
        finds no segment has a verdict, not an error: score NaN,
        ``n_segments`` 0, ``is_replay`` False and an empty ``error``.
        ``table.attrs["rng_entropy"]`` holds the seed the streams derive
        from; passing it as ``rng`` reproduces the table, also when ``rng``
        was None.

        ``is_replay`` follows each method's rule (see
        :class:`ReplayDetectionResult`). For the pooled-SSI test it compares
        the score with the epoch's shuffles, not ``p_value`` with
        ``alpha``. For an arm method it means an arm or pair was reported,
        and the ``passes[...]`` columns hold every arm's and pair's checks,
        which need not all pass. An arm method with ``arm_fallback``
        (``BERNERS_LEE_2021``) also calls an event replay on a sustained
        run that met none of the sequence rules: its row has
        ``passes[sequence]`` False and a ``score`` that can be below the
        threshold or NaN. Count only sequence replays with
        ``table["is_replay"] & table["passes[sequence]"].eq(True)``.

    Raises
    ------
    ValueError
        If an event has no ``times``, or separately decoded templates of an
        event have different times; if ``spike_counts`` or
        ``valid_time_bins`` does not have one entry per event, or an entry
        does not have one row per time bin of its event; if a method
        draws pseudo-events or weighted column cycles from joint decodes; or
        if a rank-order, pairwise-order or template-matching method is given without
        ``spike_times``, ``encoding_models`` or ``dt``, or a bias-sequence
        method without ``spike_times``, ``template_bias`` or ``dt``.

    Warns
    -----
    UserWarning
        For a shuffle test that leaves out shuffles with an undefined score;
        the warning names the event, the method and the null.

    See Also
    --------
    detect_replay : One method on one event.

    Notes
    -----
    The pseudo-event null draws from the posterior rows of every event, as
    Davidson et al. (2009) and Michon et al. (2019) drew from the complete
    set of candidate events.
    """
    import zlib

    import pandas as pd

    for name, per_event in (
        ("spike_counts", spike_counts),
        ("valid_time_bins", valid_time_bins),
    ):
        if per_event is not None and len(per_event) != len(events):
            raise ValueError(
                f"compare_replay_methods(): {name} has {len(per_event)} "
                f"entries but there are {len(events)} events."
            )
    from neurospatial.decoding import JointDecodingResult

    event_times = [
        event.conditional[event.template_names[0]].times
        if isinstance(event, JointDecodingResult)
        else _separate_times(event, index)
        if isinstance(event, Mapping)
        else event.times
        for index, event in enumerate(events)
    ]
    for index, times in enumerate(event_times):
        if times is None:
            raise ValueError(
                _format_error(
                    f"compare_replay_methods(): event {index} has no times.",
                    why="replay scores and segments need each time bin's time",
                    fix="decode each event with times=, e.g. "
                    "decode_position(env, counts, rates, dt=dt, times=grid)",
                )
            )
        event = events[index]
        expected = env[template] if isinstance(env, Mapping) and template else env
        # (environment given, environment decoded on) for each decode.
        decoded: list[tuple[Any, Any]]
        if isinstance(event, Mapping):
            # Templates missing from a mapping of env are refused when tested.
            decoded = [
                (env[name] if isinstance(env, Mapping) else env, result.env)
                for name, result in event.items()
                if not isinstance(env, Mapping) or name in env
            ]
        elif isinstance(event, JointDecodingResult) or isinstance(expected, Mapping):
            decoded = []
        else:
            decoded = [(expected, getattr(event, "env", None))]
        for expected_env, event_env in decoded:
            if event_env is None:
                continue
            mismatch = _environment_mismatch(expected_env, event_env)
            if mismatch is not None:
                raise ValueError(
                    _format_error(
                        f"compare_replay_methods(): event {index} was decoded "
                        f"on an environment that differs from env in {mismatch}.",
                        why="distances, jumps and positions would be read on "
                        "bins the event was not decoded on",
                        fix="pass the environment the events were decoded on",
                    )
                )
        n_time_bins = len(np.asarray(times))
        if spike_counts is not None:
            shape = np.shape(spike_counts[index])
            if len(shape) not in (1, 2) or shape[0] != n_time_bins:
                raise ValueError(
                    _format_error(
                        f"compare_replay_methods(): spike_counts[{index}] has "
                        f"shape {shape}, but event {index} has {n_time_bins} "
                        "time bins.",
                        why="each row of counts holds one time bin's spikes",
                        fix=f"pass shape ({n_time_bins}, n_cells); transpose "
                        "counts given as (n_cells, n_time_bins)",
                    )
                )
        if valid_time_bins is not None:
            from neurospatial.replay.segments import _valid_time_bins

            _valid_time_bins(
                valid_time_bins[index],
                n_time_bins,
                operation=f"compare_replay_methods(): event {index}",
            )
    method_names = [method.name for method in replay_methods]
    if len(set(method_names)) != len(method_names):
        raise ValueError(
            _format_error(
                f"compare_replay_methods(): method names must be unique, got "
                f"{method_names}.",
                why="each row and each random stream is labeled by the method's name",
                fix="give variants their own name, e.g. "
                "dataclasses.replace(method, name='diba_alpha_0.05', alpha=0.05)",
            )
        )
    if any(isinstance(m, ArmReplayMethod) for m in replay_methods):
        if arms is None:
            raise ValueError(
                _format_error(
                    "compare_replay_methods(): arm methods need arms=.",
                    why="they classify replay per arm and per pair of arms",
                    fix="pass arms={'center': bins, 'left': bins, ...} with each "
                    "arm's bins from the junction outward",
                )
            )
        if isinstance(env, Mapping) or any(
            isinstance(event, (JointDecodingResult, Mapping)) for event in events
        ):
            raise ValueError(
                "compare_replay_methods(): arm methods take events decoded on "
                "one environment, not joint decodes, separately decoded "
                "templates or a mapping of environments."
            )
    rank_methods = [m for m in replay_methods if isinstance(m, RankOrderMethod)]
    spike_methods = [
        m
        for m in replay_methods
        if isinstance(m, (RankOrderMethod, PairwiseOrderMethod, TemplateMatchMethod))
    ]
    if spike_methods:
        missing = [
            name
            for name, value in (
                ("spike_times", spike_times),
                ("encoding_models", encoding_models),
                ("dt", dt),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                _format_error(
                    f"compare_replay_methods(): spike-order methods "
                    f"{[m.name for m in spike_methods]} need {missing}.",
                    why="they score each cell's spikes in the event against its "
                    "field, over the event's decoding windows of width dt",
                    fix="pass the session's spike_times, the rate maps as "
                    "encoding_models and the decoding window dt",
                )
            )
    bias_methods = [m for m in replay_methods if isinstance(m, BiasSequenceMethod)]
    if bias_methods:
        missing = [
            name
            for name, value in (
                ("spike_times", spike_times),
                ("template_bias", template_bias),
                ("dt", dt),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                _format_error(
                    f"compare_replay_methods(): bias-sequence methods "
                    f"{[m.name for m in bias_methods]} need {missing}.",
                    why="they compare the order of each event's spikes, over "
                    "the event's decoding windows of width dt, with a "
                    "template's skew-bias matrix",
                    fix="pass the session's spike_times, the template's matrix "
                    "as template_bias (e.g. mean_pairwise_bias(spike_times, "
                    "theta_cycles)) and the decoding window dt",
                )
            )
    pool = None
    if any(
        name in method.nulls
        for method in replay_methods
        if isinstance(method, ReplayMethod)
        for name in ("pseudo_event", "column_cycle_weighted")
    ):
        if any(isinstance(event, (JointDecodingResult, Mapping)) for event in events):
            raise ValueError(
                _format_error(
                    "compare_replay_methods(): the pseudo-event and weighted "
                    "column-cycle nulls cannot draw from joint decodes or "
                    "separately decoded templates.",
                    why="a joint decode's rows hold several templates, and these "
                    "nulls draw rows, or the mean row, of one posterior",
                    fix="pass each event's tested share as a DecodingResult, "
                    "or leave out methods with these nulls",
                )
            )
        pool = np.vstack(
            [
                event.posterior
                for event in events
                if not isinstance(event, (JointDecodingResult, Mapping))
            ]
        )
    entropy = (
        int(rng.integers(2**63))
        if isinstance(rng, np.random.Generator)
        else np.random.SeedSequence(rng).entropy
    )

    def stream_for(
        index: int,
        method: ReplayMethod
        | RankOrderMethod
        | PairwiseOrderMethod
        | ArmReplayMethod
        | TemplateMatchMethod
        | BiasSequenceMethod,
    ) -> Any:
        return np.random.default_rng(
            np.random.SeedSequence(
                entropy, spawn_key=(index, zlib.crc32(method.name.encode()))
            )
        )

    pooled_verdicts = {}
    if dt is not None:
        windows = [
            (float(t[0]) - dt / 2, float(t[-1]) + dt / 2)
            for t in (np.asarray(times, dtype=np.float64) for times in event_times)
        ]
        for rank_method in rank_methods:
            if rank_method.test == "pooled_ssi":
                pooled_verdicts[rank_method.name] = _pooled_ssi_verdicts(
                    rank_method,
                    env,
                    spike_times,
                    windows,
                    encoding_models,
                    template,
                    [stream_for(index, rank_method) for index in range(len(events))],
                )
    rows = []
    for index, event in enumerate(events):
        for method in replay_methods:
            stream = stream_for(index, method)
            # A shuffle test's warning names the event and method it is about.
            with _warnings_with_context(
                f"compare_replay_methods(): event {index}, method {method.name!r}"
            ):
                row = _compare_row(
                    env,
                    event,
                    np.asarray(event_times[index]),
                    method,
                    spike_counts=None
                    if spike_counts is None
                    else np.asarray(spike_counts[index]),
                    encoding_models=encoding_models,
                    dt=dt,
                    prior=prior,
                    template=template,
                    valid_time_bins=None
                    if valid_time_bins is None
                    else valid_time_bins[index],
                    spike_times=spike_times,
                    arms=arms,
                    template_bias=template_bias,
                    pool=pool,
                    pooled=pooled_verdicts[method.name][index]
                    if method.name in pooled_verdicts
                    else None,
                    rng=stream,
                )
            rows.append({"event": index, **row})
    table = _fail_criteria_of_error_rows(pd.DataFrame(rows), replay_methods)
    table.attrs["rng_entropy"] = entropy
    return table


def _separate_times(
    event: Mapping[str, DecodingResult], index: int
) -> NDArray[np.float64] | None:
    """The times of an event's separately decoded templates, which must agree."""
    times = [decoded.times for decoded in event.values()]
    if not times:
        raise ValueError(
            f"compare_replay_methods(): event {index} is an empty mapping of "
            "templates; give each template's DecodingResult."
        )
    if any(t is None for t in times):
        return None
    arrays = [np.asarray(t) for t in times]
    if any(not np.array_equal(arrays[0], other) for other in arrays[1:]):
        raise ValueError(
            _format_error(
                f"compare_replay_methods(): event {index}'s templates were "
                "decoded at different times.",
                why="separately decoded templates are the same event's time "
                "bins decoded on each template's rate maps",
                fix="decode every template from the event's counts with the "
                "same times=",
            )
        )
    return arrays[0]


@contextlib.contextmanager
def _warnings_with_context(context: str) -> Iterator[None]:
    """Re-emit the warnings raised in the block with ``context`` in front."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for caught_warning in caught:
        # Frames: this generator, contextlib's __exit__, the caller, its caller.
        warnings.warn(
            f"{context}: {caught_warning.message}",
            caught_warning.category,
            stacklevel=4,
        )


def _compare_row(
    env: Environment | Mapping[str, Environment],
    event: DecodingResult | JointDecodingResult | Mapping[str, DecodingResult],
    times: NDArray[np.float64],
    method: ReplayMethod
    | RankOrderMethod
    | PairwiseOrderMethod
    | ArmReplayMethod
    | TemplateMatchMethod
    | BiasSequenceMethod,
    *,
    spike_counts: NDArray[np.int64] | None,
    encoding_models: Any,
    dt: float | None,
    prior: Any,
    template: str | None,
    valid_time_bins: NDArray[np.bool_] | None,
    spike_times: Any,
    arms: Mapping[str, NDArray[np.intp]] | None,
    template_bias: NDArray[np.float64] | None,
    pool: NDArray[np.float64] | None,
    pooled: tuple[ReplayDetectionResult, str] | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """One method's row of :func:`compare_replay_methods` for one event."""
    from neurospatial.decoding import JointDecodingResult

    def error_row(error: str) -> dict[str, Any]:
        return {
            "method": method.name,
            "is_replay": False,
            "score": np.nan,
            "p_value": np.nan,
            "error": error,
        }

    if isinstance(method, ArmReplayMethod):
        assert arms is not None and not isinstance(env, Mapping)
        assert not isinstance(event, (JointDecodingResult, Mapping))
        if np.unique(times).size < 2:
            return error_row("fewer than two time bins to score")
        verdict = detect_arm_replay(
            env, event.posterior, times, arms=arms, arm_replay_method=method
        )
        return {**verdict.summary(), "error": ""}
    if isinstance(method, TemplateMatchMethod):
        assert dt is not None
        centers = np.asarray(times, dtype=np.float64)
        matched = detect_template_match_replay(
            env,
            spike_times,
            (centers[0] - dt / 2, centers[-1] + dt / 2),
            encoding_models,
            template_match_method=method,
            template=None
            if isinstance(encoding_models, Mapping)
            and method.template_choice is not None
            else template,
            rng=rng,
        )
        error = (
            ""
            if np.isfinite(matched.score)
            else f"fewer than {method.min_active_cells} active template cells"
        )
        return {**matched.summary(), "error": error}
    if isinstance(method, RankOrderMethod) and pooled is not None:
        verdict, error = pooled
        return {**verdict.summary(), "error": error}
    if isinstance(method, RankOrderMethod):
        assert dt is not None
        centers = np.asarray(times, dtype=np.float64)
        ranked = detect_rank_order_replay(
            env,
            spike_times,
            (centers[0] - dt / 2, centers[-1] + dt / 2),
            encoding_models,
            rank_order_method=method,
            template=None
            if isinstance(encoding_models, Mapping)
            and method.template_choice is not None
            else template,
            rng=rng,
        )
        error = (
            ""
            if np.isfinite(ranked.score)
            else "rank-order correlation undefined: too few active "
            "template cells, or no variation in their order"
        )
        return {**ranked.summary(), "error": error}
    if isinstance(method, BiasSequenceMethod):
        assert dt is not None and template_bias is not None
        centers = np.asarray(times, dtype=np.float64)
        biased = detect_bias_sequence_replay(
            spike_times,
            (centers[0] - dt / 2, centers[-1] + dt / 2),
            template_bias,
            bias_sequence_method=method,
            rng=rng,
        )
        error = (
            ""
            if np.isfinite(biased.score)
            else "bias similarity undefined: no pair of cells fired in both "
            "the event and the template"
        )
        return {**biased.summary(), "error": error}
    if isinstance(method, PairwiseOrderMethod):
        assert dt is not None
        centers = np.asarray(times, dtype=np.float64)
        try:
            ordered = detect_pairwise_order_replay(
                env,
                spike_times,
                (centers[0] - dt / 2, centers[-1] + dt / 2),
                encoding_models,
                pairwise_order_method=method,
                rng=rng,
            )
        except _FieldSearchTooLargeError:
            return error_row(
                "too many combinations of multi-field cells' fields to search"
            )
        return {**ordered.summary(), "error": ""}
    usable = times
    if method.empty_time_bins != "keep" and spike_counts is not None:
        spiking = (
            spike_counts.sum(axis=1) if spike_counts.ndim == 2 else spike_counts
        ) > 0
        usable = usable[spiking]
    if np.unique(usable).size < 2:
        return error_row(
            "fewer than two time bins to score"
            if method.empty_time_bins == "keep"
            else "fewer than two time bins with spikes to score"
        )
    several = isinstance(event, (JointDecodingResult, Mapping))
    result = detect_replay(
        env,
        event
        if isinstance(event, JointDecodingResult)
        else {name: decoded.posterior for name, decoded in event.items()}
        if isinstance(event, Mapping)
        else event.posterior,
        times,
        replay_method=method,
        spike_counts=spike_counts,
        encoding_models=encoding_models,
        dt=dt,
        prior=prior,
        template=None if several and method.template_choice is not None else template,
        pseudo_event_pool=pool,
        valid_time_bins=valid_time_bins,
        rng=rng,
    )
    undefined = (
        _verdict_uses_score(method)
        and not np.isfinite(result.score)
        # A segmentation that finds no segment is a verdict, not an error.
        and (method.segmentation is None or result.segment is not None)
    )
    if undefined:
        error = "score undefined (e.g. the decoded position does not change)"
    elif (
        method.tested_quantities
        and np.isnan(result.p_value)
        and (method.segmentation is None or result.segment is not None)
    ):
        error = (
            "a tested quantity is undefined (e.g. the mean jump of one time "
            "bin), or every shuffle of a null has an undefined value"
        )
    elif method.nulls and np.isfinite(result.score) and np.isnan(result.p_value):
        error = "every shuffle of a null has an undefined score"
    else:
        error = ""
    return {**result.summary(), "error": error}


def _fail_criteria_of_error_rows(
    table: pd.DataFrame,
    replay_methods: Sequence[
        ReplayMethod
        | RankOrderMethod
        | PairwiseOrderMethod
        | ArmReplayMethod
        | TemplateMatchMethod
        | BiasSequenceMethod
    ],
) -> pd.DataFrame:
    """Set every criterion of a row's method to False on rows with an error.

    A row without a verdict passes no criterion; left NaN, a criterion
    column would read as True after ``astype(bool)``.
    """
    import pandas as pd

    if table.empty:
        return table
    failed = (table["error"] != "").to_numpy()
    for method in replay_methods:
        mine = (table["method"] == method.name).to_numpy()
        if not (mine & failed).any():
            continue
        columns = {
            column
            for column in table.columns
            if column.startswith("passes[") and table.loc[mine, column].notna().any()
        }
        if isinstance(
            method,
            (
                ReplayMethod,
                RankOrderMethod,
                PairwiseOrderMethod,
                TemplateMatchMethod,
                BiasSequenceMethod,
            ),
        ):
            columns |= {f"passes[{c.label}]" for c in method.criteria}
        for column in sorted(columns):
            if column not in table:
                table[column] = pd.Series(np.nan, index=table.index, dtype=object)
            table.loc[mine & failed, column] = False
    return table


# --------------------------------------------------------------------------
# Published methods
# --------------------------------------------------------------------------

DAVIDSON_2009 = ReplayMethod(
    name="davidson_2009",
    reference=(
        "Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009). Hippocampal "
        "replay of extended experience. Neuron, 63(4), 497-507."
    ),
    score=detect_trajectory_radon,
    nulls=("column_cycle", "unit_identity", "pseudo_event"),
    n_shuffles=1500,
    alpha=0.01,
    decoding={"time_bin": 0.02},
    notes=(
        "Band of 15 cm and speeds up to 30 m/s, as in detect_trajectory_radon.",
        "Significant when the p-value of every null is below 0.01.",
        "The paper decoded position x direction jointly in 20 ms windows and "
        "summed over direction. The unit-identity null decodes the shuffled "
        "counts again, which reproduces only a posterior decoded on one set "
        "of rate maps, so the summed posterior is refused with it (and "
        "template_choice='sum' is not used). Either decode on one set of "
        "rate maps and pass that posterior, or drop the null with "
        "dataclasses.replace(DAVIDSON_2009, nulls=('column_cycle', "
        "'pseudo_event')) and pass the summed posterior.",
        "Candidate events of at least 100 ms; event selection is not applied here.",
        "Pseudo-events draw from the complete set of candidate events.",
    ),
)

KARLSSON_FRANK_2009 = ReplayMethod(
    name="karlsson_frank_2009",
    reference=(
        "Karlsson, M. P., & Frank, L. M. (2009). Awake replay of remote "
        "experiences in the hippocampus. Nature Neuroscience, 12(7), 913-918."
    ),
    score=fit_linear_trajectory,
    score_options={"method": "sample", "n_samples_per_time_bin": 10_000},
    nulls=("time_bin_permutation",),
    n_shuffles=10_000,
    alpha=0.05,
    empty_time_bins="drop",
    criteria=(ReplayCriterion("n_active_cells", ">=", 5),),
    decoding={"time_bin": 0.015, "position_bin": 2.0, "prior": "uniform"},
    notes=(
        "10,000 posterior samples per decoded time bin, one pooled regression "
        "of position on time; R² against 10,000 shuffles of time-bin order.",
        "Time bins without spikes cannot be decoded and are dropped.",
        "At least 5 active place cells; place-cell selection is not applied here.",
        "The paper regresses on bin number; whether the remaining bins are "
        "renumbered after dropping is not stated. Here they keep their times.",
        "The paper's p-value is the share of shuffles with a larger R² (no "
        "+1); here it is (k + 1) / (n + 1).",
        "Carr et al. (2012; Carr, M. F., Karlsson, M. P., & Frank, L. M., "
        "Transient slow gamma synchrony underlies hippocampal memory replay, "
        "Neuron, 75(4), 700-713) used this method unchanged (15 ms bins, "
        "10,000 shuffles, p < 0.05, at least 5 place cells), decoding with "
        "the templates of the current and the previously run W-track 'to "
        "minimize false negatives'. How the two were combined is not stated; "
        "replay on either template is the natural reading: test each with "
        "template= and keep the smaller p-value.",
    ),
)

WU_2017 = ReplayMethod(
    name="wu_2017",
    reference=(
        "Wu, C.-T., Haggerty, D., Kemere, C., & Ji, D. (2017). Hippocampal "
        "awake replay in fear memory retrieval. Nature Neuroscience, 20(4), "
        "571-580."
    ),
    score=fit_linear_trajectory,
    score_options={"method": "map"},
    nulls=("time_bin_permutation",),
    n_shuffles=1000,
    alpha=0.05,
    empty_time_bins="drop",
    criteria=(ReplayCriterion("n_active_cells", ">=", 4),),
    decoding={"time_bin": 0.02, "time_step": 0.01},
    notes=(
        "Linear regression of the MAP position on time over time bins with "
        "at least one spike; R² against 1,000 shuffles of decoded positions "
        "in time.",
        "At least 4 active template cells (single-field place cells); the "
        "template-cell selection is not applied here.",
        "Dropped time bins keep their times; the paper regresses on time-bin "
        "number and does not say whether bins are renumbered.",
        "The paper's p-value has no +1; here it is (k + 1) / (n + 1).",
        "Mou et al. (2022; Mou, X., Pokhrel, A., Suresh, P., & Ji, D., "
        "Observational learning promotes hippocampal remote awake replay "
        "toward future reward locations, Neuron, 110(5), 891-902) used this "
        "method (20 ms bins stepped by 10 ms, at least 4 active template "
        "cells, 1,000 shuffles, p < 0.05), testing each template on its own "
        "candidate events. They compared the correlation R rather than R² "
        "with shuffled values 'greater than the actual R value', yet "
        "defined reverse replays by R < 0, which a one-sided test on signed "
        "R could never find; read as |R|, which gives the same p-value as "
        "R².",
    ),
)

SILVA_2015 = ReplayMethod(
    name="silva_2015",
    reference=(
        "Silva, D., Feng, T., & Foster, D. J. (2015). Trajectory events "
        "across hippocampal place cells require previous experience. Nature "
        "Neuroscience, 18(12), 1772-1779."
    ),
    score=weighted_correlation,
    criteria=(
        ReplayCriterion("score", ">", 0.6, absolute=True),
        ReplayCriterion("max_map_jump", "<", 0.4, relative_to="track_length"),
    ),
    decoding={"time_bin": 0.02},
    notes=(
        "Thresholds on |weighted correlation| and the maximum jump between "
        "peak decoded positions (fraction of the track). The paper also used "
        "(0.7, 0.4) and (0.7, 0.3).",
        "Significance was tested on the number of events passing, against "
        "5,000 column-cycle shuffled datasets, not per event.",
    ),
)

MICHON_2019 = ReplayMethod(
    name="michon_2019",
    reference=(
        "Michon, F., Sun, J.-J., Kim, C. Y., Ciliberti, D., & Kloosterman, F. "
        "(2019). Post-learning hippocampal replay selectively reinforces "
        "spatial memory for highly rewarded locations. Current Biology. "
        "https://doi.org/10.1016/j.cub.2019.03.048"
    ),
    score=fit_isotonic_trajectory,
    score_options={"method": "map"},
    nulls=("pseudo_event",),
    n_shuffles=500,
    alpha=0.01,
    decoding={"time_bin": 0.01},
    notes=(
        "Weighted isotonic regression of the MAP position, score = mean MAP "
        "probability x R², against 500 pseudo-events drawn from the complete "
        "set of candidate events.",
        "The paper fits each arm of the maze separately; restrict the "
        "posterior to one arm to match it.",
        "Clusterless decoding in the paper; any decode can be scored here.",
    ),
)

TIROLE_2022 = ReplayMethod(
    name="tirole_2022",
    reference=(
        "Tirole, M., Huelin Gorriz, M., Takigawa, M., Kukovska, L., & Bendor, "
        "D. (2022). Experience-driven rate modulation is reinstated during "
        "hippocampal replay. eLife, 11, e79031."
    ),
    score=weighted_correlation,
    score_options={"row_normalization": "none"},
    nulls=("spike_train_circular", "place_field_circular", "column_cycle"),
    n_shuffles=1000,
    alpha=0.05,
    use_absolute=True,
    empty_time_bins="zero",
    criteria=(ReplayCriterion("n_active_cells", ">=", 5),),
    decoding={"time_bin": 0.02, "position_bin": 10.0},
    notes=(
        "Weighted correlation of the posterior with time bins without spikes "
        "set to zero, against three shuffles; significant when above the 95th "
        "percentile of all three. The paper does not say |r|; the Bendor-lab "
        "code (weighted_correlation.m) returns abs(r).",
        "The paper normalizes the posterior across both tracks: pass one "
        "track's share, decode_jointly(...).joint(name), with every track's "
        "rate maps as encoding_models and template=name, so shuffles are "
        "decoded jointly too.",
        "Candidate events of 100-750 ms with at least 5 time bins; event "
        "selection is not applied here.",
        "Events split in halves at the multi-unit minimum (tested at "
        "p < 0.025) are not included; split the event first.",
        "The Bendor-lab code zeroes the original event's empty time bins in "
        "shuffles that move time bins; here each shuffle zeroes its own.",
    ),
)

GROSMARK_2016 = ReplayMethod(
    name="grosmark_2016",
    reference=(
        "Grosmark, A. D., & Buzsáki, G. (2016). Diversity in neural firing "
        "dynamics supports both rigid and learned hippocampal sequences. "
        "Science, 351(6280), 1440-1443."
    ),
    score=weighted_correlation,
    nulls=("place_field_circular",),
    n_shuffles=1000,
    alpha=0.05,
    use_absolute=True,
    criteria=(
        ReplayCriterion("n_active_cells", ">=", 5),
        ReplayCriterion("active_fraction", ">=", 0.1),
    ),
    decoding={"time_bin": 0.02, "position_bin": 2.0, "prior": "uniform"},
    notes=(
        "Weighted correlation of the posterior, its sign giving forward or "
        "reverse; significance of |r| against 1,000 circular place-field "
        "shuffles, decoded again. The paper reports rZ, |r| z-scored "
        "against the shuffles, and the share of events with p < 0.05 (Fig. "
        "S5C); it states neither the p-value formula nor a per-event "
        "threshold, so alpha = 0.05 and the test on |r| are readings.",
        "The paper shifted each cell's unsmoothed rate map and smoothed it "
        "again; here the given rate maps are shifted.",
        "Each running direction had its own decoder; an event's direction "
        "was the larger Pr(direction | spikes) from each direction's mean "
        "rates over the event (Eq. 6). Decode with that direction's maps.",
        "Events lasted at least 100 ms with at least max(10% of place "
        "cells, 5) firing; the cell rule is applied (pass the place cells' "
        "spike counts, so the fraction counts place cells), the duration is "
        "event selection.",
        "Secondary nulls: 1,000 time-bin permutations; cell identity (Fig. S7).",
    ),
)

AMBROSE_2016 = ReplayMethod(
    name="ambrose_2016",
    reference=(
        "Ambrose, R. E., Pfeiffer, B. E., & Foster, D. J. (2016). Reverse "
        "replay of hippocampal place cells is uniquely modulated by changing "
        "reward. Neuron, 91(5), 1124-1136."
    ),
    score=weighted_correlation,
    criteria=(ReplayCriterion("score", ">", 0.6, absolute=True),),
    decoding={
        "time_bin": 0.02,
        "time_step": 0.01,
        "position_bin": 1.8,
        "prior": "uniform",
    },
    notes=(
        "Candidate events whose posterior's weighted correlation exceeded "
        "0.6 were replay events; the absolute value is used, since replays "
        "moving up and down the track both count. Decode with rate maps "
        "from both running directions together; units with CSI > 5.",
        "Two checks, not the definition, gave similar results (Tables S2 "
        "and S3, not available here): a higher threshold, and a Monte Carlo "
        "test on 1,500 shuffles of the posterior, p = (k + 1) / 1501 < "
        "0.05, where k most naturally counts shuffles that pass the 0.6 "
        "threshold; the shuffle is not named.",
        "Direction is a separate classification on a joint decode with "
        "directional rate maps: downward when more than 66.5% of the "
        "posterior is in the downward template, upward below 33.5%, and "
        "omitted otherwise (template_fractions on that decode).",
    ),
)

PFEIFFER_FOSTER_2013 = ReplayMethod(
    name="pfeiffer_foster_2013",
    reference=(
        "Pfeiffer, B. E., & Foster, D. J. (2013). Hippocampal place-cell "
        "sequences depict future paths to remembered goals. Nature, 497(7447), "
        "74-79."
    ),
    score=None,
    segmentation=ReplaySegmentation(
        {"max_jump": 20.0, "method": "map", "min_time_bins": 11}
    ),
    criteria=(ReplayCriterion("segment_start_to_end_distance", ">=", 40.0),),
    units="cm",
    decoding={"time_bin": 0.02, "time_step": 0.005},
    notes=(
        "The event is truncated to the longest sequence of time bins whose MAP "
        "lies less than 20 cm from the previous one; events with fewer than "
        "10 steps (11 time bins) or a start-to-end distance below 40 cm are "
        "eliminated.",
        "Decoded in a 2-D open field; distances are Euclidean there.",
        "Each trajectory event was also tested against 5,000 cell-identity "
        "and 5,000 2-D place-field shuffles, p = (n + 1) / (r + 1) over the "
        "shuffles classified as trajectory events. Every trajectory event had "
        "p < 0.02 for both, so the test did not change which events counted "
        "and is not applied. To add it, use dataclasses.replace(..., "
        "nulls=('unit_identity',), n_shuffles=5000, alpha=0.02); the 2-D "
        "place-field shuffle is not available as a named null.",
    ),
)

PFEIFFER_FOSTER_2015 = ReplayMethod(
    name="pfeiffer_foster_2015",
    reference=(
        "Pfeiffer, B. E., & Foster, D. J. (2015). Autoassociative dynamics in "
        "the generation of sequences of hippocampal place cells. Science, "
        "349(6244), 180-183."
    ),
    score=None,
    segmentation=ReplaySegmentation(
        {"max_jump": 50.0, "method": "expected", "min_time_bins": 11}
    ),
    criteria=(ReplayCriterion("segment_start_to_end_distance", ">=", 80.0),),
    units="cm",
    decoding={"time_bin": 0.02, "time_step": 0.005},
    notes=(
        "The event is truncated to the longest sequence of time bins whose "
        "posterior mean lies less than 50 cm from the previous one; events "
        "with fewer than 10 steps (11 time bins) or a start-to-end distance "
        "below 80 cm are eliminated.",
        "Fig. S9 repeats the analysis with steps of at most 20 cm and a "
        "total distance of 50 cm or more; build that variant with "
        "dataclasses.replace.",
        "No per-event shuffle test.",
    ),
)

OLAFSDOTTIR_2017 = ReplayMethod(
    name="olafsdottir_2017",
    reference=(
        "Ólafsdóttir, H. F., Carpenter, F., & Barry, C. (2017). Task demands "
        "predict a dynamic switch in the content of awake hippocampal replay. "
        "Neuron, 96(4), 925-935."
    ),
    score=detect_trajectory_radon,
    score_options={"band": 30.0, "max_speed": 5000.0, "min_speed": 200.0},
    nulls=("place_field_circular",),
    n_shuffles=100,
    alpha=0.025,
    empty_time_bins="drop",
    criteria=(
        ReplayCriterion("n_active_cells", ">", 5),
        ReplayCriterion("active_fraction", ">=", 0.15),
    ),
    units="cm",
    decoding={"time_bin": 0.01, "position_bin": 2.0},
    notes=(
        "Line fit of Davidson et al. (2009): the mean posterior within 30 cm "
        "of a line, speeds from 2 to 50 m/s in either direction, normalized "
        "by the summed probability with time bins without spikes set to 0. "
        "Dropping those time bins gives the same score for every line, so "
        "the same best line and p-value up to the resolution of the dense "
        "search, with the score on another scale. Start positions are "
        "searched densely here rather than in 1 cm steps from -15 to 21 m.",
        "Time bins where the line leaves the track: Eq. 1 read literally, "
        "and the Barry-lab port, give 0; here the median over positions of "
        "Davidson et al.",
        "On short events a 2 m/s line stays within 30 cm of a stationary "
        "decode, so stationary events can still score highly, with an "
        "arbitrary direction.",
        "100 shuffles rotating each cell's rate map by 1 to n_bins - 1 bins, "
        "decoded again; here shifts of 0 to n_bins - 1 (a zero shift for one "
        "cell, probability 1 / n_bins, is kept).",
        "Replay when the score exceeded the 97.5th percentile of the "
        "shuffles: with 100 shuffles, at most 2 scoring as high; here p = "
        "(k + 1) / (n + 1) < 0.025 allows at most 1.",
        "Inbound and outbound runs were decoded separately and the event "
        "takes the direction with the higher best-fit score: score each "
        "direction's decode and keep the higher.",
        "Events need at least 15% and more than 5 of the place cells active "
        "(applied: pass the place cells' counts). Place cells fire above "
        "their mean rate over 20 contiguous bins with a peak above 1 Hz; "
        "events are multi-unit activity above 3 SD lasting at least 40 ms, "
        "with the animal below 3 cm/s at a maze corner (event selection). "
        "Rate maps in 2 cm bins smoothed with a Gaussian of sigma 5 bins "
        "(10 cm).",
        "Ólafsdóttir et al. (2016) used the same line search with p < 0.2 "
        "for 'robust' events and p < 0.025 for the strongest, choosing the "
        "direction with the lower p-value. The Barry-lab Python port differs "
        "(speeds of 1 to 50 m/s, zero padding off the track, a third of the "
        "points on the track) and is not confirmed as the lab's code.",
    ),
)

CAREY_2019 = ReplayMethod(
    name="carey_2019",
    reference=(
        "Carey, A. A., Tanaka, Y., & van der Meer, M. A. A. (2019). Reward "
        "revaluation biases hippocampal replay content away from the preferred "
        "outcome. Nature Neuroscience, 22(9), 1450-1459."
    ),
    score=None,
    segmentation=ReplaySegmentation(
        {"max_jump": 40.0, "method": "map", "min_time_bins": 10},
        choose="any",
        min_active_cells_per_time_bin=4,
    ),
    criteria=(ReplayCriterion("segment_duration", ">=", 0.05),),
    units="cm",
    decoding={"time_bin": 0.025, "time_step": 0.005, "position_bin": 3.0},
    notes=(
        "From DecSeqDetectZ.m: sequences of at least 10 time bins whose MAP "
        "moves round(40 / bin size) bins or fewer, which on 3 cm bins is "
        "less than 40 cm; time bins with fewer than 4 active cells break "
        "sequences.",
        "getEligibleSequences.m then keeps sequences of at least 50 ms, "
        "between the first and last time-bin centers: 11 time bins on 5 ms "
        "steps.",
        "The code scans whole sessions and keeps every sequence; here an "
        "event is replay when it holds one. The paper also kept only "
        "sequences that overlap a sharp-wave ripple and match one arm, not "
        "both; those event-level rules are not applied here.",
        "Counts in 5 ms bins summed over a 25 ms boxcar.",
        "The paper discards sequences whose time bins appear in more than "
        "5% of sequences found in per-time-bin cell-identity shuffles of the "
        "session. Generate_DecSeqShuf.m reads every shuffled time bin from "
        "the first one, so its shuffles are not that null; the session-level "
        "test is not applied.",
    ),
)

IGATA_2021 = ReplayMethod(
    name="igata_2021",
    reference=(
        "Igata, H., Ikegaya, Y., & Sasaki, T. (2021). Prioritized experience "
        "replays on a hippocampal predictive map for learning. Proceedings of "
        "the National Academy of Sciences, 118(1), e2011266118."
    ),
    score=None,
    nulls=("place_field_circular",),
    n_shuffles=1000,
    alpha=0.05,
    segmentation=ReplaySegmentation(
        {
            "max_jump": 40.0,
            "method": "map",
            "max_gap_bins": 1,
            "min_peak_probability": 5.0,
            "min_time_bins": 5,
        },
        relative_to={"min_peak_probability": "chance"},
    ),
    criteria=(ReplayCriterion("segment_path_length", ">", 20.0),),
    units="cm",
    decoding={"time_bin": 0.02, "time_step": 0.01},
    notes=(
        "Time bins whose MAP exceeds 5 times chance (5 / number of location "
        "bins); MAP steps below 40 cm, allowing one skipped time bin; the "
        "longest sequence must cover more than 4 time bins (60 ms) and a "
        "cumulative distance of more than 20 cm.",
        "Tested against 1,000 surrogate events decoded with each cell's "
        "tuning curve shifted across all location bins: p is the share of "
        "surrogates in which a trajectory event is also detected. The paper "
        "shifted the unsmoothed rate maps and smoothed them again (5 cm); "
        "here the given rate maps are shifted. The place-field shuffle needs "
        "a 1-D environment or a track.",
        "The paper computed the MAP of each maze path separately and "
        "extended sequences into neighboring paths; here one MAP over the "
        "whole environment.",
        "The paper's p-value is the share of surrogates; here it is (k + 1) / (n + 1).",
        "Shifted fields keep cells that fire in neighboring time bins "
        "together, so surrogates often decode a short trajectory too. In "
        "simulated sweeps, about half of the surrogates on a 2 m track and "
        "2-6% on a 10 m track held one: the test has little power where "
        "40 cm is a large share of the environment.",
    ),
)

BERNERS_LEE_2022 = ReplayMethod(
    name="berners_lee_2022",
    reference=(
        "Berners-Lee, A., Feng, T., Silva, D., Wu, X., Ambrose, E. R., "
        "Pfeiffer, B. E., & Foster, D. J. (2022). Hippocampal replays appear "
        "after a single experience and incorporate greater detail with more "
        "experience. Neuron, 110(11), 1829-1842."
    ),
    score=weighted_correlation,
    criteria=(
        ReplayCriterion("score", ">", 0.6, absolute=True),
        ReplayCriterion("max_map_jump", "<", 0.4, relative_to="track_length"),
    ),
    template_choice="sum",
    decoding={
        "time_bin": 0.02,
        "time_step": 0.005,
        "position_bin": 2.5,
        "prior": "uniform",
        "templates": "position x running direction, decoded jointly",
    },
    notes=(
        "Replays have |weighted correlation| > 0.6 and a maximum jump "
        "between MAP positions below 0.4 of the track, on the whole event. "
        "The code decodes position x direction jointly "
        "(decode_spikedensity_events.m) and sums the two directions (D = "
        "OutMatrix' + InMatrix', shuffle_wc_replayscore.m): pass the joint "
        "decode. The lap-by-lap analyses decode with fields from the two "
        "most recent passes (decode_spikedensity_events_lapbylap.m).",
        "With 5 ms steps the jump is measured between windows 5 ms apart, "
        "as in the code; the single-lap significance matrices of Fig. 1 "
        "used non-overlapping 20 ms bins (shuffle_wc_nonoverlap.m).",
        "Significance was tested on the number of events passing, against "
        "5,000 datasets with time bins shuffled (p = (n + 1) / (r + 1)), "
        "not per event; it is not applied here.",
        "Code issues not reproduced: shuffle_wc_nonoverlap.m pads events "
        "with zeros that enter the real event's jump; the quadrant "
        "cumulative p counts an event once per grid cell; column-cycle "
        "offsets are shared by all events and include 0.",
        "The line score (25 cm band) is reported, not a criterion. Putative "
        "pyramidal cells only.",
    ),
)

WIDLOSKI_FOSTER_2022 = ReplayMethod(
    name="widloski_foster_2022",
    reference=(
        "Widloski, J., & Foster, D. J. (2022). Flexible rerouting of "
        "hippocampal replay sequences around changing barriers in the absence "
        "of global place field remapping. Neuron, 110(9), 1547-1558."
    ),
    score=None,
    segmentation=ReplaySegmentation(
        {
            "max_jump": 20.0,
            "method": "expected",
            "jump_rule": "drop_earlier",
            "max_spread": 10.0,
            "merge_max_time_gap": 0.05,
            "merge_max_distance": 20.0,
        },
        choose="any",
    ),
    criteria=(
        ReplayCriterion("segment_duration", ">", 0.1),
        ReplayCriterion("segment_dispersion", ">", 12.0),
    ),
    units="cm",
    decoding={"time_bin": 0.08, "time_step": 0.005, "position_bin": 2.0},
    notes=(
        "Time bins are kept when the animal is slower than 5 cm/s, the "
        "posterior spread is below 10 cm and the center-of-mass jump is "
        "below 20 cm; pass the speed rule as valid_time_bins. The deposited "
        "code gives each time bin the jump to the next one, so a jump "
        "rejects the earlier bin.",
        "Subsequences are merged across gaps below 50 ms and 20 cm; a "
        "replay lasts more than 100 ms and has a dispersion above 12 cm.",
        "The paper prints the spread as a product of the x and y deviations, "
        "which has units of area against a 10 cm threshold; the isotropic "
        "spread of TrajectoryMetrics is used.",
        "The 2022 method scans the whole immobility period rather than "
        "candidate events; applied to event windows, sequences that cross a "
        "window's edge are cut. Pass windows that hold the whole sequence.",
        "No per-event shuffle test in the 2022 paper.",
    ),
)

WIDLOSKI_FOSTER_2025 = dataclasses.replace(
    WIDLOSKI_FOSTER_2022,
    name="widloski_foster_2025",
    reference=(
        "Widloski, J., & Foster, D. J. (2025). Replay without sharp wave "
        "ripples in a spatial memory task. Nature Communications, 16, 10287. "
        "https://doi.org/10.1038/s41467-025-65181-5"
    ),
    nulls=("unit_identity",),
    n_shuffles=100,
    alpha=0.05,
    tested_quantities={"mean_spread": "less", "mean_mean_jump": "less"},
    decoding={
        "time_bin": 0.08,
        "time_step": 0.005,
        "position_bin": 2.0,
        "rate_map_smoothing_sd": 8.0,
        "prior": "uniform",
    },
    notes=(
        "Candidate events are found as in Widloski & Foster (2022), which "
        "the paper cites for the procedure (Methods, p. 11): time bins are "
        "kept when the animal is slower than 5 cm/s, the posterior spread is "
        "below 10 cm and the center-of-mass jump is below 20 cm (pass the "
        "speed rule as valid_time_bins); subsequences are merged across gaps "
        "below 50 ms and 20 cm; a candidate lasts more than 100 ms and has a "
        "dispersion above 12 cm. The deposited code gives each time bin the "
        "jump to the next one, so a jump rejects the earlier bin.",
        "Each candidate is decoded again 100 times with the rate maps of all "
        "decoding place cells permuted across cells "
        "(load_replayEvents_cellIDshuffle_sameEvents.m, randperm over every "
        "place cell: the unit_identity null with active_only=False). Its mean "
        "posterior spread over every time bin of the merged segment, and its "
        "mean center-of-mass jump over consecutive time bins, are compared "
        "with the shuffles'. The segmentation is not run on the shuffles.",
        "Each p-value is (k + 1) / 101, k the shuffles at least as small as "
        "the event, and both must be below 0.05. The code counts strictly "
        "smaller shuffles and keeps p <= 0.05, and the paper states p < "
        "0.05; with 100 shuffles all three readings allow at most 4 smaller "
        "shuffles.",
        "The code leaves out zero jumps, between overlapping windows with "
        "the same spike counts. These decode to the same posterior under any "
        "permutation, so they are zero in the event and in every shuffle, "
        "and leaving them out rescales every mean alike: the p-value does "
        "not change. The code also counts the last jump twice (padding for "
        "the per-bin jump rule); that is not reproduced.",
        "The paper prints the spread as a product of the x and y deviations, "
        "which has units of area against a 10 cm threshold. The deposited "
        "compute_imageSpread.m takes the exponent as an argument, but the "
        "decoder that calls it is not deposited; the isotropic spread of "
        "TrajectoryMetrics is used, as in WIDLOSKI_FOSTER_2022.",
        "The bugs in load_replayEvents_cellIDshuffle.m (shuffled jumps "
        "computed from the p-value vector, l. 137, which makes every jump "
        "test pass; the wrong segment's time bins, l. 123) are in the "
        "session-level script that repeats the whole detection on cell-ID "
        "shuffles to count replays expected by chance (Supplementary "
        "Fig. 3C-D), not in the per-event test used for the published "
        "replays; they are not reproduced.",
        "The code tests every candidate. detect_replay reports one segment "
        "per window: the segments that meet the duration and dispersion "
        "criteria are tested from the longest down, and the first whose "
        "spread and jump are both significant is reported. Pass windows "
        "that hold the whole sequence, since the method scans the whole "
        "stopping period and a sequence that crosses a window's edge is cut.",
        "Decoding: 80 ms windows stepped by 5 ms, all of the session's place "
        "cells, 2 cm square bins and a uniform prior; rate maps from running "
        "faster than 10 cm/s, smoothed with an 8 cm SD Gaussian. Pass "
        "dt=0.08 so the shuffles are decoded with the windows' duration.",
    ),
)

MALLORY_2025 = ReplayMethod(
    name="mallory_2025",
    reference=(
        "Mallory, C. S., Widloski, J., & Foster, D. J. (2025). The time course "
        "and organization of hippocampal replay. Science, 387(6733), 541-548."
    ),
    score=weighted_correlation,
    score_options={"row_normalization": "none"},
    segmentation=ReplaySegmentation(
        {
            "max_jump": 0.4,
            "method": "map",
            "min_peak_probability": 5.0,
            "merge_max_time_gap": 0.05,
            "merge_max_distance": 0.4,
            "min_duration": 0.03,
        },
        relative_to={
            "max_jump": "track_length",
            "merge_max_distance": "track_length",
            "min_peak_probability": "chance",
        },
        rows="trajectory",
    ),
    criteria=(
        ReplayCriterion("segment_template_fraction", ">=", 0.5),
        ReplayCriterion("template_fraction", ">", 0.665),
        ReplayCriterion("score", ">", 0.6, absolute=True),
        ReplayCriterion("max_map_jump", "<", 0.4, relative_to="track_length"),
        ReplayCriterion("map_range", ">", 0.2, relative_to="track_length"),
        ReplayCriterion("n_active_cells", ">=", 10),
    ),
    units="cm",
    template_choice="most_posterior",
    exclude_end_distance=20.0,
    decoding={
        "time_bin": 0.02,
        "time_step": 0.005,
        "position_bin": 2.0,
        "templates": "the two running directions, decoded jointly",
    },
    notes=(
        "Pass each event's joint decode over the two running directions "
        "(decode_jointly). The tested template holds the most posterior over "
        "the event with 10 bins (20 cm) left out at each track end "
        "(filter_candidate_events.m).",
        "Segmentation from filter_candidate_events.m: MAP jumps below 40% of "
        "the track, peak above 5 times chance over both maps, runs merged "
        "across gaps below 50 ms and 40% of the track, runs of at least "
        "30 ms, longest kept. The code divides the merge distance by the "
        "bin size twice, halving it; the intended threshold is used.",
        "The code rejects the segment when the other map holds more "
        "posterior over its time bins: segment_template_fraction >= 0.5.",
        "The score and metrics use the segment's trajectory time bins only "
        "(load_candidateEvents_cm_new.m drops the NaN rows of the segment), "
        "at their own times; whether the code's weighted_correlation.m, "
        "which is not deposited, uses times or row numbers is unknown.",
        "Criteria from load_replay_criterion.m and the bioRxiv methods: more "
        "than 66.5% of the posterior in one map, weighted correlation above "
        "0.6 in absolute value, maximum MAP jump below 40% of the track, MAP "
        "range above 20% of the track, and at least 10 cells active in the "
        "event.",
        "The script that applies load_replay_criterion.m is not deposited. "
        "The directional rule is read as the |left - right| > 0.33 that "
        "filter_candidate_events.m stores for the whole event with 10 bins "
        "(20 cm) left out at each end, the same as a fraction above 0.665; "
        "the metrics script also computes fractions with 15 bins (30 cm) "
        "left out.",
        "The weighted correlation uses the template's columns of the joint "
        "posterior as weights (best_posterior in "
        "load_candidateEventMetricsDirectionalDecoding.m).",
        "load_replay_criterion.m sets a 50 ms duration threshold; measured "
        "from the start of the first 20 ms window to the end of the last, "
        "that is the segmentation's 30 ms between centers.",
    ),
)

WIKENHEISER_REDISH_2013 = ReplayMethod(
    name="wikenheiser_redish_2013",
    reference=(
        "Wikenheiser, A. M., & Redish, A. D. (2013). The balance of forward "
        "and backward hippocampal sequences shifts across behavioral states. "
        "Hippocampus, 23(1), 22-29. https://doi.org/10.1002/hipo.22049"
    ),
    score=fit_circular_sequence,
    nulls=("time_bin_permutation",),
    n_shuffles=10_000,
    alpha=0.05,
    empty_time_bins="drop",
    criteria=(
        ReplayCriterion("n_active_cells", ">=", 3),
        ReplayCriterion("n_spikes", ">=", 5),
    ),
    decoding={"time_bin": 0.01, "position_bin": 4.0, "prior": "uniform"},
    notes=(
        "Circular mean of each 10 ms time bin's posterior on a circular "
        "track 80 cm across (64 bins of about 4 cm, uniform prior), steps "
        "between successive time bins taken the short way around the loop, "
        "and a line fit to their cumulative sum; its R² is tested against "
        "shuffles of time-bin order, 'the bootstrap procedure of Karlsson "
        "and Frank (2009)', p < 0.05. The slope's sign gives forward or "
        "backward replay. Pass the circular track (Environment.from_graph "
        "on a cycle graph); fit_circular_sequence lists the readings of the "
        "fit (wrapped steps, the first point at 0, regression on time, R² "
        "as the statistic).",
        "The number of shuffles is not stated; 10,000, Karlsson & Frank's "
        "count, is a choice made here.",
        "Only time bins with at least one spike were decoded: the others are "
        "dropped with their times, and steps are taken between successive "
        "decoded time bins.",
        "Events need at least 3 active cells firing at least 5 spikes in "
        "total (applied). Candidate events are 150 ms windows centered where "
        "ripple power (140-220 Hz) exceeded 1 SD above baseline, overlapping "
        "windows concatenated; event selection is not applied here.",
        "The p-value of Karlsson & Frank is the share of shuffles with a "
        "larger R²; here it is (k + 1) / (n + 1), which gives the same "
        "verdict at alpha 0.05 with 10,000 shuffles.",
    ),
)

OLAFSDOTTIR_2015 = ReplayMethod(
    name="olafsdottir_2015",
    reference=(
        "Ólafsdóttir, H. F., Barry, C., Saleem, A. B., Hassabis, D., & "
        "Spiers, H. J. (2015). Hippocampal place cells construct reward "
        "related sequences through unexplored space. eLife, 4, e06063. "
        "https://doi.org/10.7554/eLife.06063"
    ),
    score=detect_trajectory_radon,
    score_options={"band": 20.0, "max_speed": 5000.0, "min_speed": 200.0},
    nulls=("unit_identity",),
    n_shuffles=1000,
    alpha=0.05,
    empty_time_bins="keep",
    criteria=(ReplayCriterion("n_active_cells", ">=", 7),),
    units="cm",
    decoding={
        "time_bin": 0.005,
        "position_bin": 1.0,
        "rate_map_smoothing_sd": 2.0,
        "prior": "uniform",
    },
    notes=(
        "The Bayesian line fit that corroborates the paper's rank-order "
        "preplay (Fig. 4): the mean over 5 ms time bins of the posterior "
        "within 20 cm of a line, speeds from 2 to 50 m/s in either "
        "direction, against 1,000 posteriors decoded again with the "
        "identities of the cells shuffled. Preplay when the best fit "
        "exceeded the 95th percentile of the shuffles; here p = (k + 1) / "
        "(n + 1) < 0.05, which allows at most 49 shuffles at least as high, "
        "the percentile rule up to interpolation at 50.",
        "Time bins without spikes are not mentioned. They are kept here, "
        "reading Eq. 3 as the mean over all n time bins of the decoded "
        "matrix; a choice made here. The lab's 2017 paper set them to zero, "
        "which for this score is the same as dropping them: "
        "dataclasses.replace(OLAFSDOTTIR_2015, empty_time_bins='drop'). With "
        "5 ms time bins most rows of an event are empty, so the choice "
        "matters more than in other recipes: the empty rows decode alike in "
        "the event and in every cell-identity shuffle, but they can change "
        "which line fits best.",
        "'Shuffling the identities of cells included in the event' is read "
        "as permuting every cell the decoder uses: pass the template's "
        "cells as spike_counts and encoding_models; a choice made here. "
        "Permuting only the cells active in the event, as Bush et al. "
        "(2022) of the same lab state, is the other reading: "
        "dataclasses.replace(OLAFSDOTTIR_2015, null_options={'unit_identity': "
        "{'active_only': True}}).",
        "Time bins where a line leaves the track score the median over "
        "positions of Davidson et al. (not stated in the paper). Start "
        "positions are searched densely rather than in 1 cm steps from -15 "
        "to 16 m, speeds rather than in 0.2 m/s steps.",
        "Each arm was decoded with its own rate maps: 1 cm bins smoothed "
        "with a Gaussian of sigma 2 cm, speeds below 3 cm/s and the last "
        "20 cm of each arm excluded, cells with a peak of at least 0.5 Hz, "
        "five contiguous bins above their mean rate and a correlation of at "
        "least 0.3 between session halves. Events need at least 7 active "
        "cells (applied); candidate events, in which at least 15% of a "
        "template's cells fired within 300 ms bounded by 50 ms of silence, "
        "are event selection and not applied here.",
    ),
)

OLAFSDOTTIR_2016 = dataclasses.replace(
    OLAFSDOTTIR_2017,
    name="olafsdottir_2016",
    reference=(
        "Ólafsdóttir, H. F., Carpenter, F., & Barry, C. (2016). Coordinated "
        "grid and place cell replay during rest. Nature Neuroscience, 19(6), "
        "792-794. https://doi.org/10.1038/nn.4291"
    ),
    alpha=0.2,
    empty_time_bins="keep",
    criteria=(ReplayCriterion("active_fraction", ">=", 0.15),),
    template_choice="smallest_p",
    decoding={
        "time_bin": 0.01,
        "position_bin": 2.0,
        "templates": "inbound and outbound rate maps, decoded separately",
    },
    notes=(
        "Line fit of Ólafsdóttir et al. (2015): the mean posterior within "
        "30 cm of a line, speeds from 2 to 50 m/s in either direction in "
        "0.5 m/s steps and start positions from -15 to 21 m in 1 cm steps "
        "(searched densely here). Time bins where a line leaves the track "
        "score the median over positions of Davidson et al.",
        "100 shuffles rotating each place cell's rate map by 1 to n_bins - 1 "
        "bins, decoded again; here shifts of 0 to n_bins - 1. 'Robust events, "
        "with an individual P-value of less than 0.2, were accepted as "
        "replay events': with 100 shuffles, p = (k + 1) / (n + 1) < 0.2 "
        "allows at most 19 shuffles at least as high, the same as k / 100 < "
        "0.2.",
        "Supplementary Fig. 3 marks the 95th percentile of the shuffles as "
        "the threshold for significance, which contradicts the Methods and "
        "main text (p < 0.2 for the 1,826 replay events); the Methods are "
        "followed. The strongest events (p < 0.025) were a control: "
        "dataclasses.replace(OLAFSDOTTIR_2016, alpha=0.025), which with 100 "
        "shuffles allows at most 1 shuffle at least as high where a "
        "percentile rule allows 2.",
        "Each event was decoded separately with inbound and outbound rate "
        "maps, each tested against its own shuffles, and took the direction "
        "with the lower p-value: template_choice='smallest_p', with no "
        "correction for the two directions. Pass a mapping of both "
        "directions' posteriors with encoding_models keyed alike.",
        "Time bins without spikes are not mentioned (the 2017 paper set "
        "them to zero). They are kept here, reading Eq. 2 as the mean over "
        "all n time bins; a choice made here. "
        "dataclasses.replace(OLAFSDOTTIR_2016, empty_time_bins='drop') "
        "gives the 2017 rule, which under the place-field shuffle has the "
        "same p-value as zeroing and normalizing by the summed probability.",
        "Events need activity from at least 15% of the recorded place cells "
        "(applied: pass the place cells' counts); the 'more than 5 cells' "
        "rule is the 2017 paper's. Candidate events (place-cell multi-unit "
        "activity in 1 ms bins smoothed with a 5 ms Gaussian, above its mean "
        "by 3 SD, at least 40 ms) are event selection, not applied here. "
        "Rate maps: 2 cm bins smoothed with a Gaussian of sigma 5 bins, "
        "speeds below 3 cm/s, the last 10 cm at either end and 5 cm around "
        "each corner excluded; place cells peak above 1 Hz with fields at "
        "least 20 cm long.",
    ),
)

JADHAV_2016 = dataclasses.replace(
    KARLSSON_FRANK_2009,
    name="jadhav_2016",
    reference=(
        "Jadhav, S. P., Rothschild, G., Roumis, D. K., & Frank, L. M. (2016). "
        "Coordinated excitation and inhibition of prefrontal ensembles "
        "during awake hippocampal sharp-wave ripple events. Neuron, 90(1), "
        "113-127. https://doi.org/10.1016/j.neuron.2016.02.010"
    ),
    n_shuffles=500,
    criteria=(ReplayCriterion("n_active_cells", ">=", 4),),
    template_choice="smallest_p",
    decoding={"time_bin": 0.015, "templates": "the four W-track trajectories"},
    notes=(
        "Karlsson & Frank (2009), as the paper cites it: 10,000 posterior "
        "samples per decoded 15 ms time bin, one regression of position on "
        "time, R² against 500 shuffles of time-bin order, p < 0.05. The "
        "score is given only by that citation.",
        "At least 4 CA1 cells spiking in the event (applied). The lineage "
        "code (JadhavLab/PrefrontalRipples) counts place cells (peak rate "
        "of at least 3 Hz): pass those cells' counts.",
        "Each of the four W-track trajectories is tested and the event takes "
        "the one with the smallest p-value, with no correction for four "
        "tests: template_choice='smallest_p'. The paper implies it ('a "
        "series of spatial locations on a trajectory', 'the corresponding "
        "trajectory'); Tang et al. (2017) from the same lab state it, and "
        "the lineage code does it. A choice made here. Pass a joint decode "
        "over the four trajectories (decode_jointly) or each trajectory's "
        "posterior.",
        "The position bin, rate-map smoothing and prior are not stated; the "
        "lineage uses 2 cm bins, a 4 cm SD Gaussian and a uniform prior.",
        "Time bins without spikes are dropped, by citation of Karlsson & "
        "Frank; the lineage code keeps the dropped bins' indices, as these "
        "keep their times.",
        "The lineage code (written for Shin & Jadhav 2024) draws 10,000 "
        "samples in total split by spike count, averages R² over 500 "
        "resamples and scores each of 1,500 shuffles once; the recipe "
        "follows the paper. Its p-value is the share of shuffles with a "
        "larger R²; here (k + 1) / (n + 1), the same verdict at alpha 0.05 "
        "with 500 shuffles.",
        "Candidate events are SWRs (150-250 Hz, 3 SD, speed below 4 cm/s, "
        "SWRs within 1 s of another discarded); event selection is not "
        "applied here.",
    ),
)

TANG_2017 = dataclasses.replace(
    KARLSSON_FRANK_2009,
    name="tang_2017",
    reference=(
        "Tang, W., Shin, J. D., Frank, L. M., & Jadhav, S. P. (2017). "
        "Hippocampal-prefrontal reactivation during learning is stronger in "
        "awake compared with sleep states. Journal of Neuroscience, 37(49), "
        "11789-11805. https://doi.org/10.1523/JNEUROSCI.2291-17.2017"
    ),
    template_choice="smallest_p",
    decoding={
        "time_bin": 0.01,
        "position_bin": 2.0,
        "prior": "uniform",
        "templates": "the four W-track trajectories",
    },
    notes=(
        "Karlsson & Frank (2009) in 10 ms time bins: 10,000 posterior "
        "samples per decoded time bin, a regression of position on time-bin "
        "number, R² against 10,000 shuffles of time-bin order, p < 0.05; at "
        "least 5 active place cells (peak rate of at least 3 Hz; pass those "
        "cells' counts).",
        "Four posteriors, one per W-track trajectory (center to left, left "
        "to center, center to right, right to center); 'the decoded "
        "trajectory was determined as the one (among the four possible "
        "trajectories) with the lowest p value': template_choice="
        "'smallest_p', with no correction for four tests. Pass a joint "
        "decode over the four trajectories (decode_jointly) or each "
        "trajectory's posterior.",
        "Time bins without spikes are dropped, by citation of Karlsson & "
        "Frank, and keep their times; the lineage code keeps their indices.",
        "The paper states only 'p < 0.05 versus its shuffled data'; the "
        "p-value as the share of shuffles with a larger R² comes from Karlsson "
        "& Frank (2009), which it cites, and the lineage code "
        "(JadhavLab/PrefrontalRipples). Here (k + 1) / (n + 1), the same "
        "verdict at alpha 0.05 with 10,000 shuffles.",
        "Candidate events are SWRs (150-250 Hz, mean + 3 SD on at least one "
        "tetrode, speed below 4 cm/s) in which at least 5 place cells fired; "
        "event selection is not applied here.",
    ),
)

YAMAMOTO_TONEGAWA_2017 = dataclasses.replace(
    DAVIDSON_2009,
    name="yamamoto_tonegawa_2017",
    reference=(
        "Yamamoto, J., & Tonegawa, S. (2017). Direct medial entorhinal "
        "cortex input to hippocampal CA1 is crucial for extended quiet awake "
        "replay. Neuron, 96(1), 217-227. "
        "https://doi.org/10.1016/j.neuron.2017.09.017"
    ),
    nulls=("column_cycle",),
    n_shuffles=2000,
    alpha=0.05,
    decoding={"prior": "uniform"},
    notes=(
        "A reading of an under-specified method. The paper gives the replay "
        "score of Davidson et al. (2009), the mean over time bins of the "
        "posterior within D of a line, a 'column shuffle' of the posterior "
        "2,000 times, Monte Carlo p-values, and a uniform prior over "
        "position; nothing more.",
        "Unstated, each a choice made here: the band D and speed range, "
        "Davidson et al.'s 15 cm and up to 30 m/s with no minimum (the "
        "defaults of detect_trajectory_radon), since the score and the event "
        "detection cite Davidson et al.; alpha = 0.05 (about 12-15% of "
        "candidates were significant); p = (k + 1) / (n + 1), the usual "
        "Monte Carlo p-value; time bins without spikes kept; and time bins "
        "where a line leaves the track scored with Davidson et al.'s median.",
        "'Spatial column of PDF were randomly shuffled (column shuffle)' is "
        "read as Davidson et al.'s column cycle, each time bin's posterior "
        "shifted circularly in position. Other readings: time bins permuted "
        "(dataclasses.replace(..., nulls=('time_bin_permutation',))), or each "
        "column's values permuted, which is not available.",
        "The decoding time bin, position bin and rate-map smoothing are not "
        "stated (the 10 ms bins in the Methods are for multi-unit activity). "
        "Candidate events: ripple power above 3 SD and multi-unit activity "
        "peaking above 3 SD, extended to 1 SD; event selection is not "
        "applied here.",
    ),
)

FAROOQ_2019 = ReplayMethod(
    name="farooq_2019",
    reference=(
        "Farooq, U., Sibille, J., Liu, K., & Dragoi, G. (2019). Strengthened "
        "temporal coordination within pre-existing sequential cell "
        "assemblies supports trajectory replay. Neuron, 103(4), 719-733. "
        "https://doi.org/10.1016/j.neuron.2019.05.040"
    ),
    score=weighted_correlation,
    nulls=("time_bin_permutation",),
    n_shuffles=500,
    alpha=0.05,
    alternative="two-sided",
    use_absolute=False,
    criteria=(ReplayCriterion("n_active_cells", ">=", 5),),
    decoding={"time_bin": 0.02, "prior": "uniform"},
    notes=(
        "Weighted correlation of each frame against 500 permutations of its "
        "decoded time bins (the 'time-bin shuffle'); a significant sequence "
        "is 'above the 97.5th percentile or below the 2.5th percentile (for "
        "reverse sequences)'. Here the signed correlation is tested two-"
        "sided, p = 2 min(p_greater, p_less) < 0.05 with (k + 1) / (n + 1) "
        "on each side, which rejects with at most 11 shuffles at least as "
        "extreme on one side where the percentile rule allows about 12.",
        "Each running direction was decoded and tested on its own: decode "
        "with one direction's rate maps (1 cm bins smoothed with a 5 cm "
        "Gaussian; the decoding position bin is not stated).",
        "Frames of 100-800 ms with at least 5 distinct neurons active "
        "(applied to the cells passed as spike_counts); frame detection is "
        "not applied here. The paper is inconsistent about the cells: the "
        "main text asks for 'a minimum of 5 distinct place cells', the STAR "
        "Methods for 'at least 5 distinct neurons' among all pyramidal "
        "cells. Pass the counts of the place cells or of every pyramidal "
        "cell, following one reading.",
        "Time bins without spikes are kept, a reading: the paper does not "
        "say how they are treated.",
        "The paper's sequence score rZ is the absolute weighted correlation "
        "z-scored against the shuffles' absolute values, (|r| - mean "
        "|r_shuffled|) / std |r_shuffled|: ShuffleTestResult.z_score of "
        "replay_significance with use_absolute=True (and "
        "alternative='greater'). The recipe's signed two-sided test reports "
        "the z-score of the signed r instead.",
        "The paper's main tests are on the data set, not on each frame: the "
        "share of frames passing |r| and maximum-jump thresholds against "
        "500 time-bin-shuffled data sets, 500 rate-matched Poisson data sets "
        "(make them with null_events(kind='rate_poisson') and decode them), "
        "and, as a check, data sets with each time bin's peak moved to a "
        "position drawn from the mean decoded posterior and kept 10 cm from "
        "the track's ends (the weighted column cycle: "
        "dataclasses.replace(FAROOQ_2019, nulls=('column_cycle_weighted',), "
        "null_options={'column_cycle_weighted': {'edge_exclusion': 10.0}}, "
        "units='cm'), so that detect_replay checks the 10 cm against the "
        "environment's units, "
        "with the session's candidate events as pseudo_event_pool). None of "
        "these is the per-frame test.",
    ),
)

MUESSIG_2019 = ReplayMethod(
    name="muessig_2019",
    reference=(
        "Muessig, L., Lasek, M., Varsavsky, I., Cacucci, F., & Wills, T. J. "
        "(2019). Coordinated emergence of hippocampal replay and theta "
        "sequences during post-natal development. Current Biology, 29(5), "
        "834-840. https://doi.org/10.1016/j.cub.2019.01.005"
    ),
    score=detect_trajectory_radon,
    score_options={"band": 12.5, "max_speed": 2500.0, "min_speed": 0.0, "wrap": True},
    nulls=("unit_identity",),
    n_shuffles=500,
    alpha=0.05,
    units="cm",
    template_choice="max_score",
    decoding={
        "time_bin": 0.02,
        "time_step": 0.01,
        "position_bin": 2.5,
        "rate_map_smoothing_sd": 5.0,
        "templates": "clockwise and counterclockwise rate maps, decoded separately",
    },
    notes=(
        "Summed posterior under 'a linear band, 25 cm wide', read as the "
        "band's full width: a half-width (band) of 12.5 cm, a choice made "
        "here (the Barry lab, of the same lineage, calls a 30 cm band 'the "
        "width of the trajectory (60cm)'); a 25 cm half-width is the other "
        "reading. Lines start every 2.5 cm, with speeds of 0 to 25 m/s in "
        "50 cm/s steps in either direction (searched densely here). The sum "
        "and the mean over time bins give the same best line and p-value.",
        "Lines with speed 0 are searched: events representing a single "
        "location ('stationary events') are part of the paper's results.",
        "The square track is a closed loop, and bands wrap around its first "
        "and last spatial bins: wrap=True. Pass the track built with "
        "Environment.from_graph with an edge_order that closes the loop.",
        "Each event is decoded separately with clockwise and "
        "counterclockwise rate maps; the best fit over both is the event's "
        "score, and each shuffle's score is the best over both directions, "
        "each shuffled independently, not yoked to the direction of the "
        "event's best fit (after Silva et al. 2015): "
        "template_choice='max_score'. Pass a mapping of both posteriors, "
        "with encoding_models keyed alike.",
        "500 cell-identity shuffles (spikes matched to rate maps at random, "
        "decoded again); significant when the best fit exceeded the 95th "
        "percentile of the shuffles: here p = (k + 1) / (n + 1) < 0.05, at "
        "most 24 shuffles at least as high.",
        "The 'map shuffle' (each rate map given its own random offset, "
        "decoded again) was an alternative test of significance (Figs. "
        "S3D-S3F), not the per-event definition: "
        "dataclasses.replace(MUESSIG_2019, nulls=('place_field_circular',)). "
        "Its offset range is not stated; here 0 to n_bins - 1.",
        "Time bins without spikes are not mentioned; they are kept.",
        "Rate maps from 5 s epochs of constant clockwise or counterclockwise "
        "running, 2.5 cm bins smoothed with a 5 cm SD Gaussian, speeds below "
        "2.5 cm/s excluded; ensembles with more than 25 complex-spike cells "
        "firing more than 75 spikes during the run and a median decoding "
        "error below 10 cm. Candidate events are multi-unit bursts of "
        "100-750 ms overlapping a sharp-wave ripple at rest; ensemble and "
        "event selection are not applied here.",
    ),
)

SHIN_2019 = dataclasses.replace(
    KARLSSON_FRANK_2009,
    name="shin_2019",
    reference=(
        "Shin, J. D., Tang, W., & Jadhav, S. P. (2019). Dynamics of awake "
        "hippocampal-prefrontal replay for spatial learning and "
        "memory-guided decision making. Neuron, 104(6), 1110-1125. "
        "https://doi.org/10.1016/j.neuron.2019.09.012"
    ),
    n_shuffles=1500,
    template_choice="smallest_p",
    decoding={
        "time_bin": 0.01,
        "position_bin": 2.0,
        "rate_map_smoothing_sd": 4.0,
        "prior": "uniform over position and trajectory",
        "templates": "the four W-track trajectories",
    },
    notes=(
        "10,000 posterior samples per decoded 10 ms time bin, a regression "
        "of position on time-bin number, R² against 1,500 shuffles of "
        "time-bin order, p < 0.05; at least 5 active place cells (peak rate "
        "of at least 3 Hz; pass those cells' counts).",
        "One decoder over the four W-track trajectories with a uniform "
        "prior over position and trajectory; 'we considered the trajectory "
        "with the lowest p value ... as the replay trajectory (instead of "
        "the one with the highest summed posterior probability)': "
        "template_choice='smallest_p', with no correction for four tests. "
        "Pass the joint decode over the four trajectories (decode_jointly).",
        "Outside the recipe: the place-field templates leave out positions "
        "within 15 cm of the reward wells (main results were similar "
        "without), so decode on trajectories without those bins; and only "
        "SWRs of at least 50 ms were candidate events, an event-selection "
        "rule not applied here.",
        "Time bins without spikes are dropped, by citation of Karlsson & "
        "Frank, and keep their times.",
        "The lineage code (JadhavLab/PrefrontalRipples, written for Shin & "
        "Jadhav 2024) draws 10,000 samples in total split by spike count, "
        "averages R² over 500 resamples and scores each shuffle once; the "
        "recipe follows the paper's text. The paper says only that 'the p "
        "value was calculated based on a Monte Carlo shuffle'; the p-value as "
        "the share of shuffles with a larger R² comes from Karlsson & Frank "
        "(2009), which it cites, and the lineage code. Here (k + 1) / (n + "
        "1), the same verdict at alpha 0.05 with 1,500 shuffles.",
    ),
)

BUSH_2022 = ReplayMethod(
    name="bush_2022",
    reference=(
        "Bush, D., Ólafsdóttir, H. F., Barry, C., & Burgess, N. (2022). "
        "Ripple band phase precession of place cell firing during replay. "
        "Current Biology, 32(1), 64-73. "
        "https://doi.org/10.1016/j.cub.2021.10.033"
    ),
    score=detect_trajectory_radon,
    score_options={"band": 15.0, "max_speed": 5000.0, "min_speed": 100.0},
    nulls=("unit_identity",),
    null_options={"unit_identity": {"active_only": True}},
    n_shuffles=100,
    alpha=0.05,
    empty_time_bins="drop",
    criteria=(
        ReplayCriterion("n_active_cells", ">=", 5),
        ReplayCriterion("active_fraction", ">=", 0.15),
    ),
    units="cm",
    decoding={
        "time_bin": 0.01,
        "position_bin": 2.0,
        "rate_map_smoothing_sd": 10.0,
        "templates": "outbound and inbound rate maps",
    },
    notes=(
        "The line search, citing Davidson et al. (2009) for the fit (and "
        "Ólafsdóttir et al. 2015 for the decoding and fitting procedure): "
        "'the linear fit ... which maximizes the summed probability under a "
        "line of fifteen spatial bins in width (i.e., 30cm), testing all "
        "potential spatial bins as a starting point and movement speeds of "
        "100 to 5000 cm/s in steps of 50 cm/s, in either forward or reverse "
        "directions along the track'. Speeds and start positions are "
        "searched densely here; time bins where a line leaves the track "
        "score the median over positions of Davidson et al. (not stated).",
        "The band is a choice made here: 30 cm read as the band's full "
        "width, so band (a half-width) is 15 cm. The same lab's 2016 and "
        "2017 papers score the posterior within d = 30 cm of the line (a 60 "
        "cm band; OLAFSDOTTIR_2016, OLAFSDOTTIR_2017), and the Barry-lab "
        "line-search code counts its y_range bins on each side of the line, "
        "so a 30 cm half-width is as plausible: "
        "dataclasses.replace(BUSH_2022, score_options={"
        "**BUSH_2022.score_options, 'band': 30.0}).",
        "'We decoded location independently in each time bin where at least "
        "one spike was fired by any cell', and the fit score is 'the average "
        "summed probability per time bin': time bins without spikes are "
        "dropped, keeping their times.",
        "'We shuffled the identity of active cells within each event 100 "
        "times ... An event is considered to be significant if the true fit "
        "score exceeds the 95th percentile of the fit score distribution': "
        "null_options={'unit_identity': {'active_only': True}}, decoded "
        "again. Here p = (k + 1) / (n + 1) < 0.05 allows at most 4 of the "
        "100 shuffles at least as high, where a percentile rule allows "
        "about 5, as for OLAFSDOTTIR_2017.",
        "'We excluded any events during which < 5 or 15% of all pyramidal "
        "cells were active (whichever is the larger)': at least 5 active "
        "cells and 15% of the cells passed (applied). The decoder 'includes "
        "all putative pyramidal cells, whether or not they have a place "
        "field on the track', so pass every putative pyramidal cell as "
        "spike_counts and encoding_models.",
        "Decoding: 'consecutive time bins of T = 10ms throughout the replay "
        "event, starting at the time of the first spike fired by any cell'. "
        "Rate maps: 'dwell time and the total number of spikes fired at "
        "running speeds of >= 10cm/s in 2cm bins ... smoothed using a "
        "Gaussian kernel with standard deviation of 5 bins', spikes in the "
        "final 10 cm at either end and 5 cm around each corner left out, "
        "with 'separate rate maps for runs in the outbound and inbound "
        "directions'. How the two directions are combined for replay is not "
        "stated: run the recipe on each direction's decode.",
        "Candidate events are periods of multi-unit activity or ripple-band "
        "amplitude (Z >= 0 with a peak Z >= 3, merged across gaps of 40 ms "
        "or less, longer than 40 ms) with a median running speed of at most "
        "10 cm/s and a duration of at most 0.5 s; event selection is not "
        "applied here.",
    ),
)

HARVEY_2023 = ReplayMethod(
    name="harvey_2023",
    reference=(
        "Harvey, R. E., Robinson, H. L., Liu, C., Oliva, A., & "
        "Fernandez-Ruiz, A. (2023). Hippocampo-cortical circuits for "
        "selective memory encoding, routing, and replay. Neuron, 111(13), "
        "2076-2090. https://doi.org/10.1016/j.neuron.2023.04.015"
    ),
    score=weighted_correlation,
    nulls=("column_cycle",),
    n_shuffles=1000,
    alpha=0.05,
    use_absolute=True,
    empty_time_bins="zero",
    criteria=(
        ReplayCriterion("n_active_cells", ">=", 5),
        ReplayCriterion("empty_fraction", "<=", 0.5),
    ),
    decoding={
        "time_bin": 0.02,
        "position_bin": 3.0,
        "rate_map_smoothing_sd": 3.0,
        "prior": "uniform",
    },
    notes=(
        "Weighted correlation of the posterior against 1,000 shuffles "
        "circularly shifting each time bin's posterior; p = (k + 1) / (n + "
        "1) < 0.05 with k the shuffles whose |r| is at least the event's. "
        "The paper says 'replay scores higher than the observed score'; the "
        "authors' code (ryanharvey1/ripple_heterogeneity, "
        "get_significant_events with tail='both') compares |r|, so forward "
        "and reverse replay both count.",
        "Time bins without spikes have no posterior in the authors' decoder "
        "(nelpy) and are set to 0 before scoring, so they carry no weight: "
        "empty_time_bins='zero'.",
        "At least 5 active pyramidal cells with a peak rate of at least 1 Hz "
        "and a peak-to-mean ratio of 1 (1.5 and at least 100 running spikes "
        "in the code): pass those cells' spike counts. Events with more than "
        "50% of their time bins without spikes are discarded, so at most "
        "half is kept (applied); the code keeps only events below 50%.",
        "Code differences not followed: the code's default is 1,500 "
        "shuffles (the paper states 1,000), and it shifts each time bin by "
        "1 to n_bins - 1 bins (here 0 to n_bins - 1). With 1,000 shuffles, "
        "the code's p <= 0.05 in some scripts and the paper's p < 0.05 "
        "select the same events.",
        "One decoder per running direction (rate maps in 3 cm bins smoothed "
        "with a 3 cm SD Gaussian, speeds above 4 cm/s): decode each event "
        "with each direction's maps, test each, and call it replay if either "
        "is.",
        "The paper's alternative check, the posterior within a 21 cm band "
        "of a line fit to the decoded positions, is HARVEY_2023_LINE.",
        "Candidate events are sharp-wave ripples; sessions with a median "
        "running decoding error of 25 cm or more, or a decoding accuracy "
        "that does not beat circular spike-time shuffles, were excluded. "
        "Event and session selection are not applied here.",
    ),
)

LIU_2023 = ReplayMethod(
    name="liu_2023",
    reference=(
        "Liu, C., Todorova, R., Tang, W., Oliva, A., & Fernandez-Ruiz, A. "
        "(2023). Associative and predictive hippocampal codes support "
        "memory-guided behaviors. Science, 382(6668), eadi8237. "
        "https://doi.org/10.1126/science.adi8237"
    ),
    score=detect_trajectory_radon,
    score_options={"band": 22.5, "max_speed": None, "line_ends": "on_track"},
    nulls=("column_cycle",),
    n_shuffles=500,
    alpha=0.05,
    units="cm",
    decoding={
        "time_bin": 0.02,
        "position_bin": 1.5,
        "templates": "OFF and ON running directions, decoded and tested separately",
    },
    notes=(
        "Mean over 20 ms time bins of the posterior within 22.5 cm of a line "
        "(Davidson et al. 2009), against 500 shuffles shifting each time "
        "bin's posterior along the track by a random distance; significant "
        "when the score exceeds 95% of the shuffles. The authors' code "
        "(FindReplayScore.m, ayalab1/neurocode) computes p = k / 500 with k "
        "shuffles at least as high; here (k + 1) / 501. Both reject at k <= "
        "24.",
        "Lines start and end on the track at the first and last time bins, "
        "as in the code's line family (every start bin to every end bin): "
        "line_ends='on_track', a choice made here from the code, since the "
        "paper gives no speed range. Speeds are then bounded by the track's "
        "length over the event's duration (max_speed=None), stationary lines "
        "included, and band beyond the track's ends holds the time bin's "
        "median posterior, as the code pads it.",
        "The code shifts the band sums rather than the posterior, by "
        "round(rand * (n_bins - 1)) bins (0 and n_bins - 1 at half weight); "
        "here each posterior row is shifted by 0 to n_bins - 1 bins, "
        "uniformly.",
        "OFF and ON running directions were decoded with their own rate maps "
        "and scored independently: run the recipe on each direction's "
        "decode. Rate maps on 1.5 cm bins (the 300 cm track in 200 bins), "
        "speeds above 4 cm/s; the code smooths them with 5 bins (7.5 cm), "
        "which the paper does not state.",
        "The code's decoder (ReconstructPosition.m) uses an occupancy prior "
        "by default and leaves time bins without spikes uniform over "
        "position. To match, decode with the occupancy as prior and set the "
        "posterior rows of time bins without spikes to 1 / n_bins before "
        "detect_replay; the column-cycle shuffles do not decode again, so "
        "this reproduces the code exactly. A Poisson decoder does not give "
        "such time bins a uniform row by itself, even with a uniform prior: "
        "with no spikes the posterior is the prior times exp(-dt * the "
        "cells' summed rate at x), flat only where the summed rate is flat.",
        "Trajectory events (weighted correlation above 0.6 and a maximum "
        "jump below 75 cm), compared in number with 500 cell-identity-"
        "shuffled surrogates over a grid of thresholds, are a test of the "
        "data set, not of each event; not applied here.",
        "Candidate events: pyramidal-cell rate above its mean with a peak "
        "more than 2 SD above it, 100-500 ms, overlapping an independently "
        "detected SWR; event selection is not applied here.",
    ),
)

SOULA_2023 = ReplayMethod(
    name="soula_2023",
    reference=(
        "Soula, M., Maslarova, A., Harvey, R. E., Valero, M., Brandner, S., "
        "Hamer, H., Fernández-Ruiz, A., & Buzsáki, G. (2023). Interictal "
        "epileptiform discharges affect memory in an Alzheimer's disease "
        "mouse model. Proceedings of the National Academy of Sciences, "
        "120(34), e2302676120. https://doi.org/10.1073/pnas.2302676120"
    ),
    score=regression_line_score,
    score_options={"band": 10.5},
    nulls=("column_cycle",),
    n_shuffles=1000,
    alpha=0.05,
    empty_time_bins="zero",
    criteria=(
        ReplayCriterion("n_active_cells", ">=", 5),
        ReplayCriterion("empty_fraction", "<=", 0.5),
    ),
    units="cm",
    decoding={"time_bin": 0.02, "position_bin": 3.0, "prior": "uniform"},
    notes=(
        "'Linear regression was used to fit a line to the posterior "
        "probability distribution. A Bayesian replay score for a given event "
        "was defined as the sum probability mass under the fit line within a "
        "bandwidth (21 cm)', against 1,000 shuffles 'circularly shifting each "
        "column of the posterior probability matrix by a random amount'; p = "
        "(k + 1) / (n + 1) < 0.05 with k the shuffles scoring at least the "
        "event (the paper: 'higher than the observed score').",
        "The line is read as the least-squares line of each time bin's most "
        "likely position on time, as the same co-author's code does for "
        "Harvey et al. 2023 (nelpy's trajectory_score_array, which regresses "
        "the MAP bin on the time-bin number); no Soula-specific code was "
        "read. A search for the line holding the most posterior (Davidson et "
        "al. 2009, detect_trajectory_radon with band 10.5) is another reading "
        "of 'fit a line', not followed.",
        "Band: 21 cm read as the full width, the 2w + 1 = 7 bins of 3 cm the "
        "code sums around the line (w = 3), so band (a half-width) is 10.5 "
        "cm. The code rounds the line to a bin and wraps the band around the "
        "track's ends (each time bin's posterior is shifted circularly); "
        "here band past an end holds nothing. "
        "score_options={'band': 10.5, 'wrap_band': True} wraps it as the "
        "code does; with the line on bin centers the two then agree.",
        "Time bins without spikes have no posterior in nelpy: set to zero "
        "here (empty_time_bins='zero') and left out of the line and of the "
        "mean band mass, as in the code.",
        "At least five active pyramidal cells with a peak rate of at least 1 "
        "Hz and a peak-to-mean ratio of 1.5: pass those cells' spike counts. "
        "Events with more than 50% of their time bins without spikes are "
        "discarded, so at most half is kept (applied).",
        "Candidate events are SWRs (30-100 ms) in home-cage sleep; sessions "
        "with a median running decoding error of 25 cm or more (speed > 3 "
        "cm/s) were excluded. Event and session selection are not applied "
        "here.",
    ),
)

HARVEY_2023_LINE = dataclasses.replace(
    SOULA_2023,
    name="harvey_2023_line",
    reference=HARVEY_2023.reference,
    decoding=HARVEY_2023.decoding,
    notes=(
        "The paper's 'alternative approach' (STAR Methods, 'Detection of "
        "replay'): 'linear regression to fit a line to the posterior "
        "probability distribution. A Bayesian replay score for a given event "
        "was defined as the sum probability mass under the fit line within a "
        "bandwidth (21 cm)', with the same 1,000 column-cycle shuffles, alpha "
        "and event rules as its weighted correlation (HARVEY_2023).",
        "The authors' code (ryanharvey1/ripple_heterogeneity, "
        "replay/score.py) scores nelpy's trajectory_score_array(w=3, "
        "normalize=True): the least-squares line of the MAP bin on the "
        "time-bin number, rounded to bins, and the posterior in the 7 bins "
        "of 3 cm around it (21 cm, so band is 10.5 cm), averaged over the "
        "time bins with a posterior. The band wraps around the track's ends "
        "in the code; here band past an end holds nothing. "
        "score_options={'band': 10.5, 'wrap_band': True} wraps it as the "
        "code does.",
        "Code differences not followed: 1,500 shuffles by default (the paper "
        "states 1,000), each time bin shifted by 1 to n_bins - 1 bins "
        "(null_options={'column_cycle': {'exclude_zero_shift': True}}), and "
        "the replay-participation scripts select replay with this score "
        "against time-bin permutations (nulls=('time_bin_permutation',)), "
        "which the paper does not state.",
        "Time bins without spikes have no posterior in nelpy: set to zero "
        "here (empty_time_bins='zero') and left out of the line and of the "
        "mean band mass, as in the code.",
        "At least 5 active pyramidal cells with a peak rate of at least 1 Hz "
        "and a peak-to-mean ratio of 1 (1.5 and at least 100 running spikes "
        "in the code): pass those cells' spike counts. Events with more than "
        "50% of their time bins without spikes are discarded (applied).",
        "One decoder per running direction: decode each event with each "
        "direction's maps and test each. Candidate events are sharp-wave "
        "ripples; session selection (median running decoding error below 25 "
        "cm) is not applied here.",
    ),
)

CHANG_2025 = ReplayMethod(
    name="chang_2025",
    reference=(
        "Chang, H., Tang, W., Wulf, A. M., Nyasulu, T., Wolf, M. E., "
        "Fernandez-Ruiz, A., & Oliva, A. (2025). Sleep microstructure "
        "organizes memory replay. Nature, 637(8048), 1161-1169. "
        "https://doi.org/10.1038/s41586-024-08340-w"
    ),
    score=weighted_correlation,
    score_options={"row_normalization": "none"},
    nulls=("column_cycle",),
    n_shuffles=1000,
    alpha=0.05,
    alternative="two-sided",
    use_absolute=False,
    empty_time_bins="zero",
    criteria=(ReplayCriterion("n_active_cells", ">=", 5),),
    template_choice="smallest_p",
    decoding={
        "time_bin": 0.01,
        "prior": "uniform over position and trajectory",
        "templates": "left and right T-maze trajectories, decoded jointly",
        "rate_map_speed": 2.0,
    },
    notes=(
        "'a weighted correlation was calculated for the posterior "
        "probability matrix of each trajectory, and the significance was "
        "assessed by circularly shifting the time-bins of the posterior "
        "probability matrix (n = 1000 times). A sequence was considered "
        "significant if its weighted correlation exceeded the 97.5th "
        "percentile or was below the 2.5th percentile (for reverse "
        "sequences) of the shuffled distributions (i.e., p < 0.025)': the "
        "signed r tested two-sided at alpha = 0.05 against column cycles "
        "(each time bin's posterior shifted along the track, as the code "
        "does). The two-sided p-value here is at least 2 / (n + 1), so an "
        "event at the edge of a percentile rule can differ.",
        "'If more than one trajectory type were significant, the trajectory "
        "with the lowest p-value was considered as the replayed trajectory': "
        "template_choice='smallest_p', not corrected for the two "
        "trajectories, as in the paper.",
        "The decoder is joint over position and trajectory with a uniform "
        "prior, so each trajectory's posterior matrix is its share of the "
        "joint posterior: pass decode_jointly(...) as the event with both "
        "trajectories' rate maps. The weighted correlation uses the share "
        "as weights (row_normalization='none'), so time bins the trajectory "
        "explains less weigh less. Each time bin weighted equally "
        "(score_options={}) is the code's separate decode of each "
        "trajectory: with a uniform prior, a share renormalized per time bin "
        "is that trajectory's own posterior.",
        "Each candidate event was 'divided into 10-ms non-overlapping bins'. "
        "Time bins without spikes are set to zero (empty_time_bins='zero'), "
        "as the code shifts only time bins with spikes; the paper does not "
        "say. At least 5 active cells (the paper's 'at least 5 place cells "
        "fired', applied to the event's cells given).",
        "Code differences not followed (pupil_project, "
        "replayHSEseq_decoding_CA1_slidingwin_linfittingR2.m): 20 ms windows "
        "stepped by 10 ms; 500 shuffles of 1 to n_bins - 1 bins; p = k / n "
        "with k shuffles strictly beyond the event, one-sided in the "
        "direction of r's sign; among significant trajectories the largest "
        "|r|; cells with a peak rate of at least 3 Hz, counted for a "
        "trajectory above 2 Hz there; and an event must also pass a "
        "Davidson-type line test (p < 0.025: band of 2 bins of 2 cm on each "
        "side, 100 log-spaced speeds of 100 to 10,000 cm/s in both "
        "directions, mean band mass over the time bins with spikes), which "
        "one method cannot express (it tests one score). Nearest variant: "
        "dataclasses.replace(CHANG_2025, n_shuffles=500, score_options={}, "
        "null_options={'column_cycle': {'exclude_zero_shift': True}}) on 20 "
        "ms windows stepped by 10 ms (sum adjacent 10 ms counts and decode "
        "with dt=0.02), still without the line test.",
        "Rate maps: speed above 2 cm/s, 100 linear bins per trajectory, "
        "Gaussian SD of 1 bin. Candidate events are high-synchrony events "
        "(1 ms bins smoothed with a 15 ms SD Gaussian, above 3 SD, at least "
        "50 ms, overlapping an SWR, immobility below 1 cm/s); event "
        "selection is not applied here.",
    ),
)

LIU_2023_BIAS = BiasSequenceMethod(
    name="liu_2023_bias",
    reference=LIU_2023.reference,
    n_shuffles=100,
    alpha=0.05,
    use_absolute=True,
    notes=(
        "Pairwise bias correlation (Materials and methods, 'Pairwise bias "
        "correlations', after Drieu, Todorova & Zugaro 2018): each event's "
        "skew-bias matrix S(i, j) = 2 n(i before j) / (n_i n_j) - 1 is "
        "compared with the template's by the cosine between them, 'The "
        "activity of cells that are not common to the two sequences is "
        "ignored': the cosine runs over the pairs defined in both. The lab's "
        "code (neuro_py pairwise_bias_correlation) fills undefined pairs "
        "with 0 and keeps them, which shrinks its cosines.",
        "The paper prints the unwrapped matrix as '2Bk - 2', which runs from "
        "-2 to 0; its stated range [-1, 1] and the lab's code give 2B - 1, "
        "used here.",
        "Template: 'the mean bias matrix of all sequences taking place in "
        "theta cycles during the task': pass mean_pairwise_bias(spike_times, "
        "theta_cycles) as template_bias, with the running theta cycles' "
        "intervals; each pair is averaged over the cycles in which both "
        "cells fired (not stated; the code computes one matrix over all task "
        "theta spikes). No theta-cycle detector is applied here.",
        "Shuffles: 'shuffling the spikes sequence n = 100 times', read as "
        "permuting the cell labels of the event's spikes, as the code does "
        "(each cell keeps its spike count, the event its spike times). "
        "'Significant those sequences with correlations whose absolute value "
        "exceeded 5% of the shuffled distribution': |cosine| against the "
        "shuffles' |cosine|, p = (k + 1) / 101 < 0.05, which allows at most "
        "4 shuffles at or above the event where a percentile rule allows 5.",
        "At least 2 active cells (a pair; not stated). Candidate events "
        "(post-task sleep SWRs) are not detected here.",
        "'We normalized the proportion of significant events by subtracting "
        "the mean proportion of significant events in pre-task sleep': a "
        "session statistic, not applied per event. Run "
        "compare_replay_methods on the pre- and post-task events and "
        "subtract the pre-task detection rate (detection_rates) from the "
        "post-task one.",
    ),
)

MABOUDI_2018 = ReplayMethod(
    name="maboudi_2018",
    reference=(
        "Maboudi, K., Ackermann, E., de Jong, L. W., Pfeiffer, B. E., Foster, "
        "D., Diba, K., & Kemere, C. (2018). Uncovering temporal structure in "
        "hippocampal output patterns. eLife, 7, e34467. "
        "https://doi.org/10.7554/eLife.34467"
    ),
    score=detect_trajectory_radon,
    score_options={
        "band": 3.0,
        "off_track": "bin_median",
        "zero_rows": "line_median",
        "n_lines": 35000,
    },
    nulls=("column_cycle",),
    n_shuffles=5000,
    alpha=0.01,
    empty_time_bins="zero",
    criteria=(
        ReplayCriterion("n_active_cells", ">=", 4),
        ReplayCriterion("n_time_bins", ">=", 4),
    ),
    units="cm",
    decoding={"time_bin": 0.02, "prior": "uniform", "rate_map_time_bin": 0.1},
    notes=(
        "Mean posterior mass within 3 cm of the best of 35,000 random lines "
        "(Davidson et al. 2009, with lines drawn as in Kloosterman 2012), "
        "against 5000 shuffles circularly shifting each time bin's posterior "
        "along the track; replay at the paper's 99% threshold, p = (k + 1) / "
        "(n + 1) < 0.01 with k the shuffles scoring at least the event. The "
        "published scores came from nelpy's score_Davidson_final_bst_fast "
        "(nelpy 444d5243, archived with the paper; called with w = 3, "
        "n_samples = 35000 and n_shuffles = 5000 in the analysis notebooks, "
        "kemerelab/UncoveringTemporalStructureHippocampus Figure7.ipynb).",
        "Band: the paper's 'bandwidth (of 3 cm)', read as the half-width d of "
        "Davidson et al. The code sums the 2w + 1 = 7 bins centered on the "
        "line's bin (w = 3), a half-width of 3.5 bins; the decoding bin size "
        "is in no public file, so the two cannot be reconciled.",
        "A time bin where the line leaves the track scores its median "
        "single-bin posterior mass (off_track='bin_median'), as both the text "
        "and the code say; Davidson et al.'s median, used by default, is "
        "over the band's mass.",
        "Time bins without spikes, which the authors' decoder leaves without "
        "a posterior, are set to zero (empty_time_bins='zero') and each "
        "scores the median of the line's band mass over its on-track time "
        "bins with spikes (zero_rows='line_median'), as the text says. The "
        "code computes that median but never adds it, so its empty time bins "
        "score 0; the text's rule is followed and the code's is not "
        "reproduced by default.",
        "Lines: 35,000 drawn at random over every slope (n_lines=35000, "
        "seeded by line_seed=0), the same for the event and its shuffles, as "
        "in the code, and about the code's center point, one bin past the "
        "center of the event's image of time bins by position bins on both "
        "axes; the code draws new lines for each event, here every "
        "event with the same numbers of time and position bins gets the "
        "same lines. Positions along a line are exact; the code rounds them "
        "to bins. For the dense grid of Davidson et al., which cannot miss "
        "the best line, pass n_lines=None: on a 200 cm track in 2 cm bins, a "
        "200 ms event with two empty time bins took 139 s for 5000 shuffles "
        "on the dense grid and 3.5 s with 35,000 random lines (one laptop).",
        "Shuffles: 5000 independent column cycles, each time bin shifted by "
        "0 to n_bins - 1 bins. The code draws shifts of 1 to n_bins - 1 but "
        "accumulates them from one shuffle to the next (a random walk, not "
        "independent shuffles; not reproduced), so from the second shuffle "
        "on every shift, 0 included, occurs; "
        "null_options={'column_cycle': {'exclude_zero_shift': True}} draws "
        "the code's 1 to n_bins - 1 independently.",
        "p-value: the code calls an event replay at a percentile of at least "
        "99 with ties counted half (scipy percentileofscore, kind='mean'), "
        "so #greater + #ties / 2 <= 50 of 5000; here ties count in full, "
        "k <= 49.",
        "To reproduce the code as run: dataclasses.replace(MABOUDI_2018, "
        "empty_time_bins='drop', score_options={'band': 3.5 * bin_size, "
        "'off_track': 'bin_median', 'n_lines': 35000}). Dropping the time "
        "bins without spikes scores every line as the code does, times "
        "n_time_bins over the time bins with spikes, so it gives the code's "
        "ranking of lines and its p-values (up to the random lines and "
        "shuffles drawn and the tie rule). The n_time_bins criterion then "
        "counts only time bins with spikes: apply the four-time-bin rule to "
        "the events before.",
        "Event rules applied: at least 4 time bins of 20 ms and 4 active "
        "pyramidal cells (pass pyramidal cells' counts). Not applied: "
        "interneurons (mean rate above 10 Hz while moving) are excluded, and "
        "population burst events are detected from multi-unit activity (3 "
        "SD) while the animal moves at most 5 cm/s.",
        "Decoding: Poisson, uniform prior, rate maps estimated in 100 ms "
        "bins and events decoded in 20 ms bins; the analysis notebooks "
        "smooth the tuning curves with a 4 cm SD Gaussian (sigma_tc = 4), "
        "which the paper does not state.",
    ),
)

FOSTER_WILSON_2006 = RankOrderMethod(
    name="foster_wilson_2006",
    reference=(
        "Foster, D. J., & Wilson, M. A. (2006). Reverse replay of behavioural "
        "sequences in hippocampal place cells during the awake state. Nature, "
        "440(7084), 680-683."
    ),
    spike_statistic="all_spikes",
    field_order="peak",
    min_peak_rate=5.0,
    min_active_cells=3,
    min_active_fraction=1 / 3,
    test="t_distribution",
    alpha=0.05,
    notes=(
        "Rank-order correlation between cell number, in order of field "
        "peaks, and time, with its probability from Numerical Recipes "
        "(the t-distribution for every sample size); P < 0.05.",
        "That p-value is not calibrated for few cells. When each cell fires "
        "once, in a random order, it calls a third of events replay with 3 "
        "cells (rho = 1 or -1 gives P = 0), 1/12 with 4 or 5 cells and "
        "about 5.4% with 10, the rates behind Tingley & Peyrache's (2020) "
        "simulation with MATLAB's corr. A permutation test holds the rate: "
        "dataclasses.replace(FOSTER_WILSON_2006, test='cell_order'), "
        "rank_order_significance, or RankOrderResult.analytic_p_value "
        "(exact up to 9 pairs). Exactly, no event of 3 or 4 cells reaches "
        "P < 0.05, since the smallest two-sided P is 2 / n!.",
        "The paper does not say which spikes it correlated. Every spike is "
        "used here; Takigawa et al. (2024) group Foster & Wilson with "
        "analyses of whole proto-events rather than of all spikes, so first "
        "or median spikes are defensible readings too. With every spike, a "
        "cell's spikes share one field position, and Takigawa et al. found "
        "about 17% false positives when spikes were permuted one by one, "
        "which, like this p-value, treats them as independent.",
        "Probe sequence: cells with a peak rate of at least 5 Hz for one "
        "running direction, interneurons excluded (exclude them from "
        "encoding_models); events are proto-events in which at least a "
        "third of them fired. The paper gives no fewer-cell limit; 3 is the "
        "fewest for a correlation.",
        "Each running direction was analyzed with its own probe sequence, "
        "during stops facing the direction just run: pass that direction's "
        "rate maps, or a mapping of both with template=.",
        "Events longer than 500 ms were rejected; event selection is not "
        "applied here. The 100 cell-order permutations per event fed a "
        "session-level Kolmogorov-Smirnov test, not a per-event test.",
    ),
)

DIBA_BUZSAKI_2007 = RankOrderMethod(
    name="diba_buzsaki_2007",
    reference=(
        "Diba, K., & Buzsáki, G. (2007). Forward and reverse hippocampal "
        "place-cell sequences during ripples. Nature Neuroscience, 10(10), "
        "1241-1242."
    ),
    spike_statistic="first",
    field_order="peak",
    min_peak_rate=5.0,
    min_active_cells=5,
    min_active_fraction=0.3,
    test="cell_order",
    alternative="two-sided",
    n_shuffles=500,
    alpha=0.1,
    template_choice="most_active",
    notes=(
        "First spike of each cell against the place-field order; 500 "
        "shuffles of neuron identity. An event is significant when its "
        "correlation is more positive or more negative than 95% of the "
        "shuffles: 5% in each tail, a two-sided p-value below 0.10.",
        "Templates are each trajectory's cells with a stable, single place "
        "field peaking at 5 Hz or more; the peak rule is applied, stability "
        "and single fields are left to the rate maps passed.",
        "Events were detected for each trajectory template; an event "
        "detected on both was assigned to the one with more participating "
        "cells, or else the stronger correlation: pass a mapping of both "
        "templates' rate maps.",
        "Events are 300 ms windows, preceded by at least 60 ms of silence, "
        "in which at least 30% or 5 of a template's place cells fired, "
        "whichever was greater; the cell thresholds are applied, the "
        "windows are not.",
    ),
)

CHENANI_2019 = RankOrderMethod(
    name="chenani_2019",
    reference=(
        "Chenani, A., Sabariego, M., Schlesiger, M. I., Leutgeb, J. K., "
        "Leutgeb, S., & Leibold, C. (2019). Hippocampal CA1 replay becomes "
        "less prominent but more rigid without inputs from medial entorhinal "
        "cortex. Nature Communications, 10, 1341."
    ),
    spike_statistic="mean",
    field_order="center",
    min_active_cells=5,
    test="pooled_ssi",
    alternative="greater",
    n_shuffles=100,
    alpha=0.05,
    min_epoch_events=20,
    notes=(
        "Each place cell's mean spike time in a burst against a template "
        "for each running direction, ordered by place-field centers (the "
        "Methods and Fig. 2a; Supplementary Fig. 2b says peak rates, "
        "and the paper does not define the center). Pass the place cells, "
        "with one rate-map set per direction.",
        "The SSI divides the rank correlation over the cells shared by the "
        "event and the template by the standard deviation of correlations "
        "of random orders of the same length; testsession.m estimates it "
        "from 100,000 permutations and subtracts their mean, here the exact "
        "moments are used (mean 0, standard deviation 1 / sqrt(n - 1)).",
        "Each event's cell order is shuffled 100 times and scored against "
        "both templates; the largest SSI of every shuffle of every event in "
        "the epoch forms the null. An event is replay when its largest SSI "
        "exceeds the 95% quantile, taken as in testsession.m on a 0.01 grid "
        "with a strict comparison. Negative SSIs are never significant. "
        "Pass one epoch's events to each compare_replay_methods call.",
        "Only bursts with at least 5 active cells count, and epochs with "
        "fewer than 20 such bursts are excluded (the code keeps exactly 20; "
        "the paper says '20 or less' and the supplement 'more than 20').",
        "testsession.m indexes the shuffled SSIs with the observed lengths "
        "(id0R = find(lenR == n)), so only the first shuffle of each event "
        "enters its null while the cdf divides by all of them: the threshold "
        "falls to -4 and every burst is significant. The intended null, every "
        "shuffle of every event, is used here; checked against a port of the "
        "corrected code on 1,018 simulated bursts, every verdict agrees.",
        "Ties in field positions or spike times get average ranks; the code's "
        "sequences are strict orders. Shuffles without an SSI (fewer than two "
        "shared cells on every template) are left out of the null rather than "
        "counted below the threshold.",
        "Each template holds every cell with a field in encoding_models; to "
        "leave a cell out of one direction's template, set its rate map to NaN "
        "there.",
        "The session-level binomial tests are not applied; testsession.m "
        "also refers to an undefined ns and its binocdf(..., 'upper') leaves "
        "out the observed count. Bursts (place-cell rate 1 SD above its mean, "
        "peaking at 3 SD) are not detected here.",
    ),
)

GUPTA_2010 = PairwiseOrderMethod(
    name="gupta_2010",
    reference=(
        "Gupta, A. S., van der Meer, M. A. A., Touretzky, D. S., & Redish, A. "
        "D. (2010). Hippocampal replay is not a simple function of "
        "experience. Neuron, 65(5), 695-705. "
        "https://doi.org/10.1016/j.neuron.2010.01.034"
    ),
    field_threshold=0.05,
    merge_gap_bins=1,
    field_center="rate_weighted",
    max_time_lag=0.08,
    directions=("forward", "backward"),
    nulls=("spike_time_permutation", "cell_identity"),
    n_shuffles=300,
    alpha=0.05,
    min_active_cells=3,
    notes=(
        "Score (Supplemental Experimental Procedures, 'Sequence detection'): "
        "every pair of spikes less than 80 ms apart adds +1 when the cells "
        "fire in the order of their place-field centers along the lap and -1 "
        "against it, summed over the event; each pair counts once (counting "
        "each twice, as 'compared with the other spikes within its local "
        "region' could mean, doubles every score and leaves the p-values "
        "unchanged). 'Within 80 ms' and 'separated by less than 80 ms' are "
        "read as strictly less.",
        "Multi-field cells: the paper assigns each spike the field center "
        "that maximizes the score without giving the optimizer. Here all "
        "spikes of a cell take one field, the best combination over the "
        "event's multi-field cells is found exactly, in the event and in "
        "every shuffle, and searches over more than 65,536 combinations "
        "are refused.",
        "Fields: contiguous linearized pixels (about 3.5 cm) above 5% of "
        "the highest rate of any cell at any pixel, merged across one-pixel "
        "gaps; the maximum is read as one population-wide value per "
        "template, taken over bins with spatial support. Field centers are "
        "not defined in the paper and are read as rate-weighted. Pass rate "
        "maps in about 3.5 cm bins for the paper's merge rule.",
        "Templates: each lap's linearization (left and right laps, positions "
        "running from maze start in the direction run) is one template: "
        "pass a mapping of environments and rate maps. Each is tested "
        "forward and backward and the smallest p-value is reported without "
        "correction, as in the paper; the score is the forward score or "
        "minus the backward score.",
        "Shuffles: 300 permutations of the scored spikes' times (each spike "
        "keeps its cell; the paper 'randomly shuffled the spike times', and "
        "a uniform redraw is the other reading) and 300 permutations of the "
        "cells' field sets across every cell given (the cell-identity "
        "shuffle of Davidson et al. 2009). Replay when the score is greater "
        "than 95% of both sets: p = (k + 1) / 301 < 0.05 for each, ties "
        "counted against the event.",
        "The 80 cm limit on the distance between the two spikes' field "
        "centers applied only to the detection scan; the paper removed it "
        "for significance testing, and it is not applied here.",
        "Events: the paper scanned the session in 10 ms steps with windows "
        "growing from 40 ms to 600 ms until five steps added no score, at "
        "reward sites with SWR power more than 2 SD above the mean, and "
        "kept events with at least 3 active cells (counted here over every "
        "cell given). The scan and the event selection are not applied: "
        "pass candidate windows.",
        "Cells firing more than 15,000 spikes in the 40 min session or "
        "fewer than 100 were excluded; leave them out of spike_times and "
        "encoding_models. Replays found on both laps over the central stem "
        "were merged into one in the paper's counts; that is not applied.",
    ),
)

WU_FOSTER_2014 = ArmReplayMethod(
    name="wu_foster_2014",
    reference=(
        "Wu, X., & Foster, D. J. (2014). Hippocampal replay captures the "
        "unique topological structure of a novel environment. Journal of "
        "Neuroscience, 34(19), 6459-6469."
    ),
    peak_threshold=5.0,
    smoothing_sigma=0.01,
    min_duration=0.05,
    min_coverage=0.5,
    min_abs_correlation=0.5,
    joint_min_abs_correlation=0.5,
    decoding={"time_bin": 0.01, "position_bin": 1.85},
    notes=(
        "Subregions around the largest peak of each arm's MAP function "
        "(largest posterior over the arm's bins, smoothed with a 10 ms SD "
        "Gaussian) above 5 / (bins of the whole maze); single-arm replay "
        "when at least 50 ms long (five 10 ms time bins of the smoothed "
        "run), more than 50% of the arm represented and |weighted "
        "correlation| > 0.5.",
        "Joint subregions span the two single-arm subregions over both "
        "arms' bins; replay when |weighted correlation| > 0.5, both arms "
        "meet the length and coverage rules and their correlations have the "
        "joint subregion's sign along the path. A single-arm replay counts "
        "only if no pair is a replay, a joint replay only if no other pair "
        "is.",
        "Positions along a pair run from one arm's end through the junction "
        "to the other's. Fig. 1A lays the center arm out end to junction "
        "and the side arms junction to end, and Fig. 2C draws right-left "
        "replays as V shapes on that axis, which could never pass a "
        "same-sign rule there; the path reading is the one consistent with "
        "the reported right-left replays.",
        "Unstated choices: the posterior is not renormalized within an arm; "
        "the smoothing pads with zeros at the event's edges and is truncated "
        "at 4 SD; the first time bin of the largest peak wins a tie; several "
        "single-arm replays in one event are all reported.",
        "No per-event shuffle: 5,000 place-field shifts tested the number "
        "of replays in a session. Candidate events (not applied here): spike "
        "density of place cells in 10 ms bins smoothed with a 15 ms SD, "
        "above its mean with a peak above 2 SD, while the rat ran slower "
        "than 5 cm/s. Fields were smoothed per arm (SD 5 bins); cells with "
        "peak rate above 1 Hz and spikes wider than 0.34 ms. "
        "Directionality (> 0.3) of a joint position x direction decode was "
        "a separate analysis.",
    ),
)

BERNERS_LEE_2021 = ArmReplayMethod(
    name="berners_lee_2021",
    reference=(
        "Berners-Lee, A., Wu, X., & Foster, D. J. (2021). Prefrontal cortical "
        "neurons are selective for non-local hippocampal representations "
        "during replay and behavior. Journal of Neuroscience, 41(27), "
        "5894-5908."
    ),
    peak_threshold=4.0,
    smoothing_sigma=0.005,
    min_duration=0.055,
    min_coverage=0.5,
    min_abs_correlation=0.3,
    joint_min_abs_correlation=0.3,
    subregions="runs",
    merge_max_gap=0.05,
    coverage="mean_range",
    joint="strongest_pair",
    joint_same_sign=False,
    arm_fallback=True,
    decoding={"time_bin": 0.02, "time_step": 0.005, "position_bin": 2.0},
    notes=(
        "From make_singleandjointreplay_fromPosterior.m (the version "
        "RunAnalysis.m calls). Pass the posterior summed over running "
        "directions (D = OutMatrix + InMatrix), e.g. the joint decode's "
        "shares added together.",
        "Each arm's largest posterior per window, smoothed with a Gaussian "
        "of one 5 ms step (the paper cites Wu & Foster for the smoothing "
        "without a width), against 4 / (bins of the maze). Runs of at least "
        "11 windows (end - start >= 10 steps: min_duration 55 ms here) are "
        "kept and merged when the next starts at most 10 steps after the "
        "previous ends (merge_max_gap 50 ms). The code's merge loses the end "
        "of a chain of three or more runs; the whole chain is merged here.",
        "Single-arm replay: |weighted correlation| > 0.3, the range of the "
        "posterior mean along the arm over more than 50% of its bins, and "
        "at least 50 ms; the code's maximum jump (< the arm's length) never "
        "binds.",
        "Joint replay: among single-arm replays on different arms, ordered "
        "by start, the subregion runs from the first's start to the second's "
        "end along the path through the junction; the strongest pair with "
        "|weighted correlation| >= 0.3 in the code (> here). The paper's "
        "condition that the two replays do not overlap is not checked by the "
        "code, nor here; Wu & Foster's same-sign rule is off "
        "(joint_same_sign=False).",
        "Otherwise the strongest single-arm replay; otherwise, as the paper "
        "defines arm-replays, the run with the largest absolute correlation "
        "on any arm that stayed above the threshold for 50 ms, unless "
        "another run ties it, even when it fails the correlation and "
        "coverage rules "
        "(criteria['sequence'] is False for these). In simulations these "
        "were 40-70% of the single-arm outcomes.",
        "Events were padded by 20 ms on each side and centered on whole "
        "20 ms windows; column-cycle shuffles (5,000, resampling time bins "
        "with replacement) are computed but not part of the arm-replay "
        "definition. Place fields were smoothed with 2 bins in the code, 3 "
        "in the paper. A port of the code agrees with this recipe on 24,000 "
        "of 24,000 simulated events once its merge is corrected.",
    ),
)

PUBLISHED_METHODS: tuple[
    ReplayMethod
    | RankOrderMethod
    | PairwiseOrderMethod
    | ArmReplayMethod
    | TemplateMatchMethod
    | BiasSequenceMethod,
    ...,
] = (
    LEE_WILSON_2002,
    FOSTER_WILSON_2006,
    DIBA_BUZSAKI_2007,
    JI_WILSON_2007,
    DAVIDSON_2009,
    KARLSSON_FRANK_2009,
    GUPTA_2010,
    PFEIFFER_FOSTER_2013,
    WIKENHEISER_REDISH_2013,
    WU_FOSTER_2014,
    OLAFSDOTTIR_2015,
    PFEIFFER_FOSTER_2015,
    SILVA_2015,
    AMBROSE_2016,
    GROSMARK_2016,
    JADHAV_2016,
    OLAFSDOTTIR_2016,
    WU_2017,
    OLAFSDOTTIR_2017,
    TANG_2017,
    YAMAMOTO_TONEGAWA_2017,
    MABOUDI_2018,
    CAREY_2019,
    CHENANI_2019,
    FAROOQ_2019,
    MICHON_2019,
    MUESSIG_2019,
    SHIN_2019,
    IGATA_2021,
    BERNERS_LEE_2021,
    BUSH_2022,
    TIROLE_2022,
    BERNERS_LEE_2022,
    WIDLOSKI_FOSTER_2022,
    HARVEY_2023,
    HARVEY_2023_LINE,
    LIU_2023,
    LIU_2023_BIAS,
    SOULA_2023,
    MALLORY_2025,
    CHANG_2025,
    WIDLOSKI_FOSTER_2025,
)
"""Every published per-event method recipe in this module.

``NADASDY_1999`` (a :class:`~neurospatial.replay.RecurringSequenceMethod`)
is a session-level test with no per-event verdict and is left out.
"""
