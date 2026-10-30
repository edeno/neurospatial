"""Published replay detection methods as explicit, comparable recipes.

Each :class:`ReplayMethod` records one paper's replay test: the score, the
shuffles and significance level, how time bins without spikes are handled,
how the event is cut into trajectory segments (:class:`ReplaySegmentation`),
and the thresholds an event must meet. :func:`detect_replay` applies a method
to one decoded event and :func:`compare_replay_methods` applies several
methods to several events, so the same data can be scored the way each lab
scored it.

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
    "CAREY_2019",
    "DAVIDSON_2009",
    "IGATA_2021",
    "KARLSSON_FRANK_2009",
    "MALLORY_2025",
    "MICHON_2019",
    "PFEIFFER_FOSTER_2013",
    "PFEIFFER_FOSTER_2015",
    "PUBLISHED_METHODS",
    "SILVA_2015",
    "TIROLE_2022",
    "WIDLOSKI_FOSTER_2022",
    "WU_2017",
    "ReplayCriterion",
    "ReplayDetectionResult",
    "ReplayMethod",
    "ReplaySegmentation",
    "compare_replay_methods",
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
        segmentation), a scalar of
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
    """

    options: Mapping[str, Any]
    relative_to: Mapping[str, Literal["track_length", "chance"]] = field(
        default_factory=dict
    )
    choose: Literal["longest", "any"] = "longest"
    min_active_cells_per_time_bin: int | None = None

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
    template_choice : {"most_posterior"} or None
        For an event given as a joint decode without ``template=``, which
        template is tested. ``"most_posterior"`` tests the template that
        holds the most posterior over the event, leaving out
        ``exclude_end_distance`` at each track end (Mallory et al. 2025);
        the first template wins a tie. None needs ``template=``.
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
    template_choice: Literal["most_posterior"] | None = None
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
        if self.template_choice not in (None, "most_posterior"):
            raise ValueError(
                f"ReplayMethod {self.name!r}: template_choice must be None or "
                f"'most_posterior', got {self.template_choice!r}."
            )
        distance = self.exclude_end_distance
        if not (np.isfinite(distance) and distance >= 0):
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
        The tested template of a joint decode; None for an event decoded on
        one set of rate maps.
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
        a template criterion is used without a joint decode; if
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
    if joint is not None:
        needs_fractions = replay_method.template_choice is not None or any(
            c.quantity == "template_fraction" for c in replay_method.criteria
        )
        fractions = (
            template_fractions(
                joint, exclude_end_distance=replay_method.exclude_end_distance
            )
            if needs_fractions
            else {}
        )
        if template is None:
            if replay_method.template_choice is None:
                raise ValueError(
                    _format_error(
                        f"{_OPERATION}: method {replay_method.name!r} needs "
                        "template= for a joint decode.",
                        why="it tests one template's share and has no rule to "
                        "choose one",
                        fix=f"pass template= one of {list(joint.template_names)}",
                    )
                )
            template = max(fractions, key=fractions.__getitem__)
        tested: str = template
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

    if isinstance(env, Mapping):
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
    segmentation = replay_method.segmentation
    needs_counts = (
        replay_method.empty_time_bins != "keep"
        or any(c.quantity == "n_active_cells" for c in replay_method.criteria)
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
                env, track_env, template, encoding_models, joint
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
                template=None if joint is None else template,
            )
        rows = segments.rows(segment)
        if joint is not None:
            on_trajectory = segments.trajectory_time_bins[rows]
            masses = {
                name: float(np.sum(joint.joint(name)[rows][on_trajectory]))
                for name in joint.template_names
            }
            event_values["segment_template_fraction"] = masses[tested] / sum(
                masses.values()
            )
        posterior, times = posterior[rows], times[rows]
        counts = None if counts is None else counts[rows]

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
        not in ("score", "n_active_cells", *_SEGMENT_QUANTITIES, *_TEMPLATE_QUANTITIES)
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
        template=None if joint is None else template,
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


def compare_replay_methods(
    env: Environment | Mapping[str, Environment],
    events: Sequence[DecodingResult | JointDecodingResult],
    *,
    replay_methods: Sequence[ReplayMethod],
    spike_counts: Sequence[NDArray[np.int64]] | None = None,
    encoding_models: Any = None,
    dt: float | None = None,
    prior: Any = None,
    template: str | None = None,
    valid_time_bins: Sequence[NDArray[np.bool_]] | None = None,
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
    replay_methods : sequence of ReplayMethod
        The methods to compare.
    spike_counts : sequence of arrays or None
        Each event's spike counts, for methods that need them.
    encoding_models, dt, prior, template
        The decoding inputs, for the nulls that decode again (see
        :func:`detect_replay`).
    valid_time_bins : sequence of arrays or None
        Each event's valid time bins, for methods with a segmentation (see
        :func:`detect_replay`).
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
        joint decodes, and ``error``.
        ``error`` is empty unless the event has fewer than two time bins
        left to score (after the method drops time bins without spikes);
        then ``is_replay`` is False and the score and p-value are NaN.

    Notes
    -----
    The pseudo-event null draws from the posterior rows of every event, as
    Davidson et al. (2009) and Michon et al. (2019) drew from the complete
    set of candidate events.

    Raises
    ------
    ValueError
        If an event has no ``times``; if ``spike_counts`` or
        ``valid_time_bins`` does not have one entry per event; or if a
        method draws pseudo-events from joint decodes.

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
    pool = None
    if any("pseudo_event" in method.nulls for method in replay_methods):
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
    rows = []
    for index, event in enumerate(events):
        for method in replay_methods:
            stream = np.random.default_rng(
                np.random.SeedSequence(
                    entropy, spawn_key=(index, zlib.crc32(method.name.encode()))
                )
            )
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
                template=template,
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
        "summed over direction; decode the same way to match it.",
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
        "Criteria from load_replay_criterion.m and the bioRxiv methods: more "
        "than 66% of the posterior in one map (the code tests |left - right| "
        "> 0.33 on the whole event with the ends left out, the same as a "
        "fraction above 0.665), weighted correlation above 0.6 in absolute "
        "value, maximum MAP jump below 40% of the track, MAP range above 20% "
        "of the track, and at least 10 cells active in the event.",
        "The weighted correlation uses the template's jointly normalized "
        "posterior as weights, as the bioRxiv formula states; the code's "
        "weighted_correlation.m is not deposited.",
        "load_candidateEvents_cm_new.m skips segments shorter than 50 ms "
        "from the start of the first window to the end of the last; with "
        "20 ms windows that is the segmentation's 30 ms between centers.",
        "The deposit calls the metrics step in commented-out code and does "
        "not include the script that applies load_replay_criterion.m.",
    ),
)

PUBLISHED_METHODS: tuple[ReplayMethod, ...] = (
    DAVIDSON_2009,
    KARLSSON_FRANK_2009,
    PFEIFFER_FOSTER_2013,
    PFEIFFER_FOSTER_2015,
    SILVA_2015,
    WU_2017,
    CAREY_2019,
    MICHON_2019,
    IGATA_2021,
    TIROLE_2022,
    WIDLOSKI_FOSTER_2022,
    MALLORY_2025,
)
"""Every published method recipe in this module."""
