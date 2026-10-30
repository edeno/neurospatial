"""Published replay detection methods as explicit, comparable recipes.

Each :class:`ReplayMethod` records one paper's replay test: the score, the
shuffles and significance level, how time bins without spikes are handled,
how the event is cut into trajectory segments (:class:`ReplaySegmentation`),
and the thresholds an event must meet. :func:`detect_replay` applies a method
to one decoded event and :func:`compare_replay_methods` applies several
methods to several events, so the same data can be scored the way each lab
scored it. :class:`RankOrderMethod` records a rank-order test, which scores
the event's spikes rather than a decoded posterior
(:func:`detect_rank_order_replay`).

A recipe records the paper's decoding settings (time-bin and position-bin
sizes) for reference but does not decode: methods are compared on the same
decoded posterior, so differences come from the replay test alone. Choices a
paper does not specify, and bugs in the authors' code that are not
reproduced, are listed in each recipe's ``notes``.
"""

from __future__ import annotations

import inspect
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._constants import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE
from neurospatial._exceptions import _format_error
from neurospatial._results import ResultMixin
from neurospatial.replay.correlation import weighted_correlation
from neurospatial.replay.trajectory import (
    detect_trajectory_radon,
    fit_isotonic_trajectory,
    fit_linear_trajectory,
)

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial.decoding import DecodingResult, JointDecodingResult
    from neurospatial.environment import Environment
    from neurospatial.replay.segments import ReplaySegmentsResult

__all__ = [
    "AMBROSE_2016",
    "BERNERS_LEE_2022",
    "CAREY_2019",
    "CHENANI_2019",
    "DAVIDSON_2009",
    "DIBA_BUZSAKI_2007",
    "FOSTER_WILSON_2006",
    "GROSMARK_2016",
    "IGATA_2021",
    "KARLSSON_FRANK_2009",
    "MALLORY_2025",
    "MICHON_2019",
    "OLAFSDOTTIR_2017",
    "PFEIFFER_FOSTER_2013",
    "PFEIFFER_FOSTER_2015",
    "PUBLISHED_METHODS",
    "SILVA_2015",
    "TIROLE_2022",
    "WIDLOSKI_FOSTER_2022",
    "WU_2017",
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
_DECODING_NULLS = ("unit_identity", "spike_train_circular", "place_field_circular")
# Nulls whose shuffled rows no longer match the event's valid time bins.
_ROW_MOVING_NULLS = ("pseudo_event", "spike_train_circular", "time_bin_permutation")
# Criterion quantities read from the tested segment of ReplaySegmentsResult.
_SEGMENT_QUANTITIES = {
    "segment_duration": "durations",
    "segment_time_bin_count": "time_bin_counts",
    "segment_dispersion": "dispersions",
    "segment_start_to_end_distance": "start_to_end_distances",
    "segment_path_length": "path_lengths",
}
# Criterion quantities that compare the templates of a joint decode.
_TEMPLATE_QUANTITIES = ("template_fraction", "segment_template_fraction")
_COMPARISONS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


@dataclass(frozen=True)
class ReplayCriterion:
    """One threshold an event must meet to count as replay.

    Attributes
    ----------
    quantity : str
        ``"score"`` (the method's score), ``"n_active_cells"`` (cells with a
        spike anywhere in the event, also for a method with a
        segmentation), ``"active_fraction"`` (``n_active_cells`` over the
        columns of ``spike_counts``), a scalar of
        :class:`~neurospatial.replay.TrajectoryMetrics`, such as
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
        """Refuse unknown comparisons and scales."""
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
        ``"track_length"`` multiplies the option by the track's length;
        ``"chance"`` multiplies it by ``1 / env.n_bins``, the mass per bin
        of a uniform posterior on equal bins (Igata et al. 2021: a MAP
        above 5 times chance).
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
        """Freeze the mappings and refuse options replay_segments does not take."""
        from neurospatial.replay.segments import replay_segments

        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        object.__setattr__(
            self, "relative_to", MappingProxyType(dict(self.relative_to))
        )
        accepted = [
            name
            for name, parameter in inspect.signature(replay_segments).parameters.items()
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
        scales = sorted(set(self.relative_to.values()) - {"track_length", "chance"})
        if scales:
            raise ValueError(
                f"ReplaySegmentation: relative_to values must be 'track_length' "
                f"or 'chance', got {scales}."
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
        return (
            type(self),
            (
                dict(self.options),
                dict(self.relative_to),
                self.choose,
                self.min_active_cells_per_time_bin,
                self.rows,
            ),
        )

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
        Options for ``score``. ``"n_samples_per_time_bin"`` is converted to
        ``n_samples`` for each event, for samplers like
        :func:`~neurospatial.replay.fit_linear_trajectory` whose papers
        give a count per time bin.
    nulls : tuple of str
        Shuffles the score is tested against (see
        :func:`~neurospatial.replay.replay_significance`). Empty for a method
        without a per-event shuffle test.
    n_shuffles : int
        Shuffles per null.
    alpha : float
        Significance level: the event's largest p-value must be below it.
    alternative : {"greater", "less", "two-sided"}
        Tail of the shuffle test.
    use_absolute : bool or None
        Test the absolute score; None uses the default of
        :func:`~neurospatial.replay.replay_significance`.
    empty_time_bins : {"keep", "drop", "zero"}
        Time bins without spikes: keep them, drop them with their times
        (Karlsson & Frank 2009; Wu et al. 2017), or set their posterior to
        zero (the Bendor-lab code). With a ``segmentation``, "drop" and
        "zero" also make them invalid time bins, so no jump is measured
        across them, and "drop" removes them from the tested segment.
    criteria : tuple of ReplayCriterion
        Thresholds the event must meet as well.
    segmentation : ReplaySegmentation or None
        How the event is cut into trajectory segments. When given, the
        score, the shuffles and the criteria apply to the tested segment's
        rows, and the event is not replay without a segment.
    units : str or None
        Units of the method's distance thresholds, such as ``"cm"``;
        :func:`detect_replay` refuses an environment in other units. None
        for a method whose thresholds have no units or scale with the track.
    template_choice : {"most_posterior", "sum"} or None
        For an event given as a joint decode without ``template=``, which
        template is tested. ``"most_posterior"`` tests the template that
        holds the most posterior over the event, leaving out
        ``exclude_end_distance`` at each track end (Mallory et al. 2025);
        the last template wins a tie, as in the Mallory et al. code, which
        tests the second map unless the first holds more. ``"sum"`` tests
        the posterior summed over the templates, which must share one
        environment, such as the two running directions of a joint position
        x direction decode (Berners-Lee et al. 2022); the result has no
        ``template``, and nulls that decode again cannot be used. An
        explicit ``template=`` tests that template instead, and a posterior
        given as an array is scored as it is, so pass the joint decode.
        None needs ``template=``.
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
    n_shuffles: int = 1000
    alpha: float = 0.05
    alternative: Literal["greater", "less", "two-sided"] = "greater"
    use_absolute: bool | None = None
    empty_time_bins: Literal["keep", "drop", "zero"] = "keep"
    criteria: tuple[ReplayCriterion, ...] = ()
    segmentation: ReplaySegmentation | None = None
    units: str | None = None
    template_choice: Literal["most_posterior", "sum"] | None = None
    exclude_end_distance: float = 0.0
    decoding: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the mappings and refuse settings that cannot run."""
        object.__setattr__(
            self, "score_options", MappingProxyType(dict(self.score_options))
        )
        object.__setattr__(self, "decoding", MappingProxyType(dict(self.decoding)))
        object.__setattr__(self, "nulls", tuple(self.nulls))
        object.__setattr__(self, "criteria", tuple(self.criteria))
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.score is None and any(c.quantity == "score" for c in self.criteria):
            raise ValueError(
                f"ReplayMethod {self.name!r}: a method without a score cannot "
                "threshold one."
            )
        if self.score is None and self.nulls and self.segmentation is None:
            raise ValueError(
                _format_error(
                    f"ReplayMethod {self.name!r}: a method without a score or a "
                    "segmentation has nothing to test against shuffles.",
                    fix="give a score, or a segmentation whose detection is "
                    "tested (Igata et al. 2021)",
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
        if self.score is None and self.segmentation is not None and self.nulls:
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
        if self.template_choice not in (None, "most_posterior", "sum"):
            raise ValueError(
                f"ReplayMethod {self.name!r}: template_choice must be None, "
                f"'most_posterior' or 'sum', got {self.template_choice!r}."
            )
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
        from neurospatial.replay.significance import _NULLS

        unknown = [name for name in self.nulls if name not in _NULLS]
        if unknown:
            raise ValueError(
                f"ReplayMethod {self.name!r}: unknown nulls {unknown}; choose "
                f"from {list(_NULLS)}."
            )
        if self.alternative not in ("greater", "less", "two-sided"):
            raise ValueError(
                f"ReplayMethod {self.name!r}: alternative must be 'greater', "
                f"'less' or 'two-sided', got {self.alternative!r}."
            )
        if not 0 < self.alpha < 1:
            raise ValueError(
                f"ReplayMethod {self.name!r}: alpha must be in (0, 1), got "
                f"{self.alpha!r}."
            )
        if isinstance(self.n_shuffles, bool) or int(self.n_shuffles) < 1:
            raise ValueError(
                f"ReplayMethod {self.name!r}: n_shuffles must be a positive "
                f"integer, got {self.n_shuffles!r}."
            )

    def __hash__(self) -> int:
        """Hash by name and reference; the option mappings are not hashable."""
        return hash((self.name, self.reference))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mappings cannot pickle."""
        return (
            type(self),
            (
                self.name,
                self.reference,
                self.score,
                dict(self.score_options),
                self.nulls,
                self.n_shuffles,
                self.alpha,
                self.alternative,
                self.use_absolute,
                self.empty_time_bins,
                self.criteria,
                self.segmentation,
                self.units,
                self.template_choice,
                self.exclude_end_distance,
                dict(self.decoding),
                self.notes,
            ),
        )


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
            "alternative": ("greater", "less", "two-sided"),
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
        True when the event meets every criterion and, for a method with
        shuffles, its largest p-value is below ``alpha``.
    score : float
        The method's score (NaN for a criteria-only method).
    p_value : float
        The largest p-value over the method's nulls (NaN without shuffles).
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
        The tested template of a joint decode, chosen or given as
        ``template=``; None for an event decoded on one set of rate maps and
        for a joint decode's summed posterior (``template_choice="sum"``).
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

    def __post_init__(self) -> None:
        """Freeze the mappings."""
        object.__setattr__(self, "criteria", MappingProxyType(dict(self.criteria)))
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mappings cannot pickle."""
        return (
            type(self),
            (
                self.method,
                self.is_replay,
                self.score,
                self.p_value,
                dict(self.criteria),
                dict(self.values),
                self.segments,
                self.segment,
                self.template,
            ),
        )

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result.

        A joint decode adds the tested ``template``. A method with a
        segmentation adds ``n_segments`` and the tested segment's
        ``segment_start_time`` and ``segment_end_time`` (NaN without one).
        """
        values: dict[str, Any] = {
            "method": self.method,
            "is_replay": self.is_replay,
            "score": self.score,
            "p_value": self.p_value,
            **{f"passes[{label}]": ok for label, ok in self.criteria.items()},
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


def detect_replay(
    env: Environment | Mapping[str, Environment],
    posterior: NDArray[np.float64] | JointDecodingResult,
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
        with ``template`` naming the share.
    posterior : NDArray[np.float64] or JointDecodingResult
        The event's posterior, as decoded, shape (n_time_bins, n_bins); or
        the event's joint decode over several templates, from
        :func:`~neurospatial.decoding.decode_jointly`, whose tested
        template is ``template=`` or the method's ``template_choice``.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds.
    replay_method : ReplayMethod
        The method, for example ``neurospatial.replay.methods.DAVIDSON_2009``.
    spike_counts : NDArray[np.int64] or None, shape (n_time_bins, n_neurons)
        The event's spike counts. Needed by methods that drop or zero time
        bins without spikes, count active cells, or decode shuffled counts.
    encoding_models, dt, prior, template
        The decoding inputs, for the nulls that decode again (see
        :func:`~neurospatial.replay.replay_significance`), including one
        template's share of a joint decode (``template=``). With a joint
        decode as ``posterior``, ``template`` names the tested template.
    pseudo_event_pool : NDArray[np.float64] or None
        Posterior rows of the session's candidate events, for the
        pseudo-event null.
    valid_time_bins : NDArray[np.bool_] or None, shape (n_time_bins,)
        For a method with a segmentation, False for time bins that cannot
        be on a trajectory, such as bins while the animal runs (Widloski &
        Foster 2022: 5 cm/s or faster); see
        :func:`~neurospatial.replay.replay_segments`. Methods without a
        segmentation ignore it.
    rng : np.random.Generator, int, or None
        Random number generator or seed.

    Returns
    -------
    ReplayDetectionResult
        The verdict, score, p-value and each criterion; for a method with
        a segmentation, the segments and the tested segment; and for a
        joint decode, the tested template.

    Raises
    ------
    ValueError
        If the method needs an input that is not given (spike counts,
        decoding inputs or a pseudo-event pool); if the method's
        ``units`` are not the environment's; if a joint decode is given
        without ``template=`` to a method without a ``template_choice``, or
        a template criterion is used without a joint decode; if a summed
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
    from neurospatial.replay.significance import replay_significance

    joint = posterior if isinstance(posterior, JointDecodingResult) else None
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
                other = joint.conditional[name].env
                if other is not joint_env and (
                    other.n_bins != joint_env.n_bins
                    or other.units != joint_env.units
                    or other.bin_centers.shape != joint_env.bin_centers.shape
                    or not np.allclose(other.bin_centers, joint_env.bin_centers)
                ):
                    raise ValueError(
                        _format_error(
                            f"{_OPERATION}: method {replay_method.name!r} sums "
                            "the templates' posteriors, but they are on "
                            "different environments.",
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

    if isinstance(env, Mapping) and summed:
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
        decoded_env = joint_env
        if track_env is not decoded_env and (
            track_env.n_bins != decoded_env.n_bins
            or track_env.units != decoded_env.units
            or not np.allclose(track_env.bin_centers, decoded_env.bin_centers)
        ):
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: env is not the environment the joint decode used.",
                    why="positions and distances would be read on other bins",
                    fix="pass the environment (or mapping of environments) "
                    "given to decode_jointly",
                )
            )
    if replay_method.units is not None and track_env.units != replay_method.units:
        stated = (
            "has no units" if track_env.units is None else f"is in {track_env.units!r}"
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
    counts = None if spike_counts is None else np.asarray(spike_counts)
    if counts is not None and counts.ndim == 1:
        counts = counts[:, np.newaxis]
    if counts is not None and len(counts) != len(posterior):
        raise ValueError(
            f"{_OPERATION}: spike_counts has {len(counts)} rows but posterior "
            f"has {len(posterior)}; pass one row of counts per posterior row."
        )
    if counts is not None:
        event_values["n_active_cells"] = float(np.sum(counts.sum(axis=0) > 0))
        event_values["active_fraction"] = event_values["n_active_cells"] / max(
            counts.shape[1], 1
        )
    segmentation = replay_method.segmentation
    needs_counts = (
        replay_method.empty_time_bins != "keep"
        or any(
            c.quantity in ("n_active_cells", "active_fraction")
            for c in replay_method.criteria
        )
        or (
            segmentation is not None
            and segmentation.min_active_cells_per_time_bin is not None
        )
    )
    if needs_counts and counts is None:
        raise ValueError(
            _format_error(
                f"{_OPERATION}: method {replay_method.name!r} needs spike_counts.",
                why="it drops or zeroes time bins without spikes, or counts "
                "the cells active in the event or in each time bin",
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
    segments = None
    segment = None
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
        if replay_method.score is None and replay_method.nulls:
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
            p_value = _segment_detection_p_value(
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
            )
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
        tested_rows = np.arange(rows.start, rows.stop)
        if segmentation.rows == "trajectory":
            tested_rows = tested_rows[segments.trajectory_time_bins[rows]]
        posterior, times = posterior[tested_rows], times[tested_rows]
        counts = None if counts is None else counts[tested_rows]

    if replay_method.empty_time_bins == "drop":
        assert counts is not None
        keep = counts.sum(axis=1) > 0
        posterior = np.asarray(posterior[keep])
        times, counts = times[keep], counts[keep]

    options = dict(replay_method.score_options)
    if "n_samples_per_time_bin" in options:
        options["n_samples"] = int(options.pop("n_samples_per_time_bin")) * len(
            posterior
        )

    if replay_method.score is not None and replay_method.nulls:
        # The test needs the counts only to decode shuffles again or to zero
        # empty time bins; given without the rate maps they would be refused.
        decodes = any(name in _DECODING_NULLS for name in replay_method.nulls)
        result = replay_significance(
            env if decodes else track_env,
            posterior,
            times,
            score=replay_method.score,
            score_options=options,
            nulls=replay_method.nulls,
            spike_counts=counts
            if decodes or replay_method.empty_time_bins == "zero"
            else None,
            encoding_models=encoding_models if decodes else None,
            dt=dt if decodes else None,
            prior=prior,
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
        score, p_value = float(result.observed.score), float(result.p_value)
    elif replay_method.score is not None:
        scored = posterior
        if replay_method.empty_time_bins == "zero":
            assert counts is not None
            scored = np.where(counts.sum(axis=1)[:, None] == 0, 0.0, posterior)
        score = float(replay_method.score(track_env, scored, times, **options).score)

    needs_metrics = any(
        c.quantity
        not in (
            "score",
            "n_active_cells",
            "active_fraction",
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
    )


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
) -> float:
    """Share of shuffles that also have a segment meeting the criteria.

    Without such a segment in the event, every shuffle is at least as
    extreme and the p-value is 1, so no shuffle is drawn.
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
        return 1.0
    result = replay_significance(
        env if decodes else track_env,
        posterior,
        times,
        score=_segment_detected,
        score_options=options,
        nulls=replay_method.nulls,
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
    return float(result.p_value)


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
    )
    from neurospatial.stats.shuffle import compute_shuffle_pvalue

    operation = "detect_rank_order_replay()"
    # Validates the rates against env, then keeps the template cells' maps.
    _field_positions(env, encoding_models, method.field_order, operation)
    rates = np.asarray(
        getattr(encoding_models, "firing_rates", encoding_models), dtype=np.float64
    )
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
            # Ties in rho differ by rounding between the two formulas.
            p_value = float(
                compute_shuffle_pvalue(
                    round(score, 12), np.round(null, 12), tail=method.alternative
                )
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
    from neurospatial.replay.rank_order import _field_positions

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
            rates = np.asarray(getattr(models, "firing_rates", models), dtype=float)
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
    if pooled.size:
        grid = np.arange(-400, 401) / 100.0  # testsession.m: b0 = -4:.01:4
        cdf = np.searchsorted(np.sort(pooled), grid, side="right") / pooled.size
        above = np.flatnonzero(cdf > 1.0 - method.alpha)
        if above.size:
            threshold = float(grid[above[0]])
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
                ),
                error,
            )
        )
    return verdicts


def compare_replay_methods(
    env: Environment | Mapping[str, Environment],
    events: Sequence[DecodingResult | JointDecodingResult],
    *,
    replay_methods: Sequence[ReplayMethod | RankOrderMethod],
    spike_counts: Sequence[NDArray[np.int64]] | None = None,
    encoding_models: Any = None,
    dt: float | None = None,
    prior: Any = None,
    template: str | None = None,
    valid_time_bins: Sequence[NDArray[np.bool_]] | None = None,
    spike_times: Any = None,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Apply several published replay methods to several decoded events.

    Parameters
    ----------
    env : Environment or Mapping[str, Environment]
        The environment the events were decoded on, or the mapping of
        environments of a joint decode (see :func:`detect_replay`).
    events : sequence of DecodingResult or JointDecodingResult
        The decoded candidate events, each with ``times``: a decode on one
        set of rate maps, or a joint decode over several templates (see
        :func:`detect_replay`).
    replay_methods : sequence of ReplayMethod or RankOrderMethod
        The methods to compare. A rank-order method scores each event's
        spikes over the event's decoding windows, from the first window's
        start to the last window's end (``times[0] - dt / 2`` to
        ``times[-1] + dt / 2``), against ``encoding_models``: with a mapping
        of rate maps, the one named by ``template``, or the one its
        ``template_choice`` picks. A method with ``test="pooled_ssi"``
        treats the events as one epoch (pass one epoch per call) and scores
        every template unless ``template`` names one.
    spike_counts : sequence of arrays or None
        Each event's spike counts, for methods that need them.
    encoding_models, dt, prior, template
        The decoding inputs, for the nulls that decode again (see
        :func:`detect_replay`). For joint decodes, ``template`` is the
        template tested by methods without a ``template_choice``; methods
        with one choose the template of each event themselves.
    valid_time_bins : sequence of arrays or None
        Each event's valid time bins, for methods with a segmentation (see
        :func:`detect_replay`).
    spike_times : array, sequence of arrays, or spike group, optional
        The session's spike times, one train per row of
        ``encoding_models``, for rank-order methods.
    rng : numpy.random.Generator, int, or None
        Random number generator or seed for the shuffles. Each (event, method)
        pair draws from its own stream, derived from ``rng``, the event's
        position and the method's name, so adding, removing or reordering
        methods does not change the other pairs' results.

    Returns
    -------
    pandas.DataFrame
        One row per (event, method): ``event``, ``method``, ``is_replay``,
        ``score``, ``p_value``, one ``passes[...]`` column per criterion,
        ``n_segments``, ``segment_start_time`` and ``segment_end_time`` for
        methods with a segmentation (NaN for the others), ``template`` for
        joint decodes, and ``error``. ``error`` says why an event has no
        verdict, and ``is_replay`` is then False: fewer than two time bins
        left to score (after the method drops time bins without spikes), an
        undefined rank-order correlation (too few active template cells, or
        no variation in their order), or, for a pooled SSI method, an epoch
        with too few events (score and p-value are still reported).

    Notes
    -----
    The pseudo-event null draws from the posterior rows of every event, as
    Davidson et al. (2009) and Michon et al. (2019) drew from the complete
    set of candidate events.

    Raises
    ------
    ValueError
        If an event has no ``times``; if ``spike_counts`` or
        ``valid_time_bins`` does not have one entry per event; if a method
        draws pseudo-events from joint decodes; or if a rank-order method is
        given without ``spike_times``, ``encoding_models`` or ``dt``.

    See Also
    --------
    detect_replay : One method on one event.
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
        else event.times
        for event in events
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
    rank_methods = [m for m in replay_methods if isinstance(m, RankOrderMethod)]
    if rank_methods:
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
                    f"compare_replay_methods(): rank-order methods "
                    f"{[m.name for m in rank_methods]} need {missing}.",
                    why="they score each cell's spikes in the event against its "
                    "field, over the event's decoding windows of width dt",
                    fix="pass the session's spike_times, the rate maps as "
                    "encoding_models and the decoding window dt",
                )
            )
    pool = None
    if any(
        "pseudo_event" in method.nulls
        for method in replay_methods
        if isinstance(method, ReplayMethod)
    ):
        if any(isinstance(event, JointDecodingResult) for event in events):
            raise ValueError(
                _format_error(
                    "compare_replay_methods(): the pseudo-event null cannot "
                    "draw from joint decodes.",
                    why="a joint decode's rows hold several templates, and a "
                    "pseudo-event draws rows of one posterior",
                    fix="pass each event's tested share as a DecodingResult, "
                    "or leave out methods with the pseudo-event null",
                )
            )
        pool = np.vstack(
            [
                event.posterior
                for event in events
                if not isinstance(event, JointDecodingResult)
            ]
        )
    entropy = (
        int(rng.integers(2**63))
        if isinstance(rng, np.random.Generator)
        else np.random.SeedSequence(rng).entropy
    )

    def stream_for(index: int, method: ReplayMethod | RankOrderMethod) -> Any:
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
            if isinstance(method, RankOrderMethod) and method.name in pooled_verdicts:
                verdict, error = pooled_verdicts[method.name][index]
                rows.append({"event": index, **verdict.summary(), "error": error})
                continue
            if isinstance(method, RankOrderMethod):
                assert dt is not None
                centers = np.asarray(event_times[index], dtype=np.float64)
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
                    rng=stream,
                )
                error = (
                    ""
                    if np.isfinite(ranked.score)
                    else "rank-order correlation undefined: too few active "
                    "template cells, or no variation in their order"
                )
                rows.append({"event": index, **ranked.summary(), "error": error})
                continue
            counts = None if spike_counts is None else np.asarray(spike_counts[index])
            usable = np.asarray(event_times[index])
            if method.empty_time_bins == "drop" and counts is not None:
                usable = usable[counts.reshape(len(usable), -1).sum(axis=1) > 0]
            if np.unique(usable).size < 2:
                rows.append(
                    {
                        "event": index,
                        "method": method.name,
                        "is_replay": False,
                        "score": np.nan,
                        "p_value": np.nan,
                        "error": "fewer than two time bins to score",
                    }
                )
                continue
            result = detect_replay(
                env,
                event if isinstance(event, JointDecodingResult) else event.posterior,
                np.asarray(event_times[index]),
                replay_method=method,
                spike_counts=counts,
                encoding_models=encoding_models,
                dt=dt,
                prior=prior,
                template=None
                if isinstance(event, JointDecodingResult)
                and method.template_choice is not None
                else template,
                pseudo_event_pool=pool,
                valid_time_bins=None
                if valid_time_bins is None
                else valid_time_bins[index],
                rng=stream,
            )
            rows.append({"event": index, **result.summary(), "error": ""})
    return pd.DataFrame(rows)


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
        "summed over direction; decode the same way and pass the summed "
        "posterior. template_choice='sum' is not used because the unit-"
        "identity null decodes again.",
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
    decoding={"time_bin": 0.01, "position_bin": 2.0, "prior": "uniform"},
    notes=(
        "Line fit of Davidson et al. (2009): the mean posterior within 30 cm "
        "of a line, speeds from 2 to 50 m/s in either direction, normalized "
        "by the summed probability with time bins without spikes set to 0. "
        "Dropping those time bins gives the same best line and p-value (a "
        "rate-map shuffle keeps them), with the score on another scale. "
        "Start positions are searched densely here rather than in 1 cm "
        "steps from -15 to 21 m; how time bins where the line leaves the "
        "track count is not stated (here the median over positions, as "
        "Davidson et al.).",
        "On short events a 2 m/s line stays within 30 cm of a stationary "
        "decode, so stationary events can still score highly.",
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
        "(applied: pass the place cells' counts), with the animal below 3 "
        "cm/s at a maze corner (event selection). Rate maps in 2 cm bins "
        "smoothed with a 5-bin Gaussian.",
        "Ólafsdóttir et al. (2016) used the same line search with p < 0.2 "
        "for 'robust' events and p < 0.025 for the strongest. The Barry-lab "
        "Python port differs (speeds of 1 to 50 m/s, zero padding off the "
        "track, a third of the points on the track) and is not confirmed as "
        "the lab's code.",
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
        "than 66% of the posterior in one map, weighted correlation above "
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

PUBLISHED_METHODS: tuple[ReplayMethod | RankOrderMethod, ...] = (
    FOSTER_WILSON_2006,
    DIBA_BUZSAKI_2007,
    DAVIDSON_2009,
    KARLSSON_FRANK_2009,
    PFEIFFER_FOSTER_2013,
    PFEIFFER_FOSTER_2015,
    SILVA_2015,
    AMBROSE_2016,
    GROSMARK_2016,
    WU_2017,
    OLAFSDOTTIR_2017,
    CAREY_2019,
    CHENANI_2019,
    MICHON_2019,
    IGATA_2021,
    TIROLE_2022,
    BERNERS_LEE_2022,
    WIDLOSKI_FOSTER_2022,
    MALLORY_2025,
)
"""Every published method recipe in this module."""
