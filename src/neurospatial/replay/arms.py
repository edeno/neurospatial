"""Replay of single arms and pairs of arms on a multi-arm maze.

On a maze with several arms, a replayed trajectory may run along one arm or
cross the junction from one arm into another. Wu & Foster (2014) cut each
candidate event into one subregion per arm, around the strongest
representation of that arm, scored each subregion and each pair of
subregions, and called an event a single-arm or a joint-arm replay by
rules that keep the two from overlapping. Berners-Lee et al. (2021) cut
every sustained run instead and took the strongest pair.
:class:`ArmReplayMethod` records such a method and :func:`detect_arm_replay`
applies it to one decoded event.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations, pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import reduce_through_constructor
from neurospatial._validation import validate_choice

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.replay.methods import ReplayDetectionResult

__all__ = ["ArmReplayMethod", "detect_arm_replay"]

_OPERATION = "detect_arm_replay()"


@dataclass(frozen=True)
class ArmReplayMethod:
    """A published test for replay of single arms and of pairs of arms.

    Attributes
    ----------
    name : str
        Short name, for example ``"wu_foster_2014"``.
    reference : str
        The paper the method follows.
    peak_threshold : float
        Multiple of chance, ``1 / env.n_bins`` over the whole maze, that an
        arm's peak posterior must exceed to be represented (Wu & Foster
        2014: 5).
    smoothing_sigma : float
        Standard deviation, in seconds, of the Gaussian that smooths each
        arm's peak posterior over time before the subregion is cut (Wu &
        Foster 2014: 10 ms).
    min_duration : float
        Shortest single-arm subregion, in seconds: its time bins times their
        spacing (Wu & Foster 2014: 50 ms, five 10 ms time bins).
    min_coverage : float
        Share of an arm's bins that must be represented in its subregion,
        exceeded strictly (Wu & Foster 2014: 50%).
    min_abs_correlation : float
        Absolute weighted correlation a single-arm subregion must exceed
        (Wu & Foster 2014: 0.5).
    joint_min_abs_correlation : float
        Absolute weighted correlation a joint subregion must exceed (Wu &
        Foster 2014: 0.5).
    subregions : {"largest_peak", "runs"}
        ``"largest_peak"``: one subregion per arm, the run around the
        largest peak (Wu & Foster 2014). ``"runs"``: every run above the
        threshold lasting ``min_duration``, merged across short gaps
        (Berners-Lee et al. 2021).
    merge_max_gap : float or None
        With ``"runs"``, runs merge when the time between the last time bin
        of one and the first of the next is at most this, in seconds.
    coverage : {"represented", "mean_range"}
        ``"represented"``: the share of the arm's bins whose largest
        posterior over the subregion exceeds the threshold (Wu & Foster
        2014). ``"mean_range"``: the range of the posterior mean along the
        arm, in bins, over the arm's number of bins (Berners-Lee et al.
        2021).
    joint : {"only_pair", "strongest_pair"}
        ``"only_pair"``: each pair of arms is tested from the arms'
        subregions, a joint replay is reported only when no other pair is
        one, and every single-arm replay only when no pair is one (Wu &
        Foster 2014). ``"strongest_pair"``: pairs of single-arm replays on
        different arms, the earlier first, are tested and the strongest is
        reported; otherwise the strongest single-arm replay (Berners-Lee et
        al. 2021).
    joint_same_sign : bool
        A joint replay also needs both arms' correlations to have the joint
        subregion's sign along the path (Wu & Foster 2014), for either
        ``joint``.
    arm_fallback : bool
        Without a replay, an event with a run on some arm is still a replay
        of that arm: the run with the largest absolute correlation, unless
        another run ties it (Berners-Lee et al. 2021's arm-replays).
    decoding : Mapping[str, Any]
        The paper's decoding settings, for reference (not enforced).
    notes : tuple of str
        Choices the paper does not specify.
    """

    name: str
    reference: str
    peak_threshold: float = 5.0
    smoothing_sigma: float = 0.01
    min_duration: float = 0.05
    min_coverage: float = 0.5
    min_abs_correlation: float = 0.5
    joint_min_abs_correlation: float = 0.5
    subregions: Literal["largest_peak", "runs"] = "largest_peak"
    merge_max_gap: float | None = None
    coverage: Literal["represented", "mean_range"] = "represented"
    joint: Literal["only_pair", "strongest_pair"] = "only_pair"
    joint_same_sign: bool = True
    arm_fallback: bool = False
    decoding: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the mapping and refuse settings that cannot run."""
        object.__setattr__(self, "decoding", MappingProxyType(dict(self.decoding)))
        object.__setattr__(self, "notes", tuple(self.notes))
        prefix = f"ArmReplayMethod {self.name!r}"
        for name in (
            "peak_threshold",
            "smoothing_sigma",
            "min_duration",
            "min_coverage",
            "min_abs_correlation",
            "joint_min_abs_correlation",
            "merge_max_gap",
        ):
            value = getattr(self, name)
            if value is None and name == "merge_max_gap":
                continue
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{prefix}: {name} must be a number, got {value!r}.")
        for name in ("peak_threshold", "smoothing_sigma", "min_duration"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value >= 0):
                raise ValueError(f"{prefix}: {name} must be >= 0, got {value!r}.")
        for name in (
            "min_coverage",
            "min_abs_correlation",
            "joint_min_abs_correlation",
        ):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{prefix}: {name} must be in [0, 1), got {value!r}.")
        choices = {
            "subregions": ("largest_peak", "runs"),
            "coverage": ("represented", "mean_range"),
            "joint": ("only_pair", "strongest_pair"),
        }
        for name, allowed in choices.items():
            validate_choice(getattr(self, name), allowed, name=name, owner=prefix)
        for name in ("joint_same_sign", "arm_fallback"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(
                    f"{prefix}: {name} must be True or False, got "
                    f"{getattr(self, name)!r}."
                )
        gap = self.merge_max_gap
        if gap is not None and (self.subregions != "runs" or not gap >= 0):
            raise ValueError(
                f"{prefix}: merge_max_gap must be a time >= 0 in seconds, and "
                f"only with subregions='runs'; got {gap!r}."
            )

    def __hash__(self) -> int:
        """Hash by name and reference; the decoding mapping is not hashable."""
        return hash((self.name, self.reference))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mappings cannot pickle."""
        return reduce_through_constructor(self)


def detect_arm_replay(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    arms: Mapping[str, NDArray[np.intp]],
    arm_replay_method: ArmReplayMethod,
) -> ReplayDetectionResult:
    """Find a single-arm or joint-arm replay in one decoded event.

    An arm's subregions are runs of time bins in which the arm's smoothed
    peak posterior stays above ``peak_threshold`` times chance: the run
    around its largest peak (Wu & Foster 2014), or every long-enough run,
    merged across short gaps (Berners-Lee et al. 2021). A subregion is a
    single-arm replay when it is long enough, covers enough of the arm and
    its posterior is correlated with time. Two arms form a joint
    subregion over the path from one arm's end through the junction to
    the other's, which is a joint replay when its correlation passes,
    both arms' subregions are long and cover enough and, with
    ``joint_same_sign``, both arms' correlations have the joint's sign.
    ``joint`` sets which replay is reported: the only joint pair, else
    every single arm (Wu & Foster 2014), or the strongest joint pair,
    else the strongest single arm (Berners-Lee et al. 2021); with
    ``arm_fallback``, else the strongest sustained run.

    Parameters
    ----------
    env : Environment
        The maze the posterior was decoded on.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        The event's posterior over every bin of the maze; each row sums to
        1.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds, evenly spaced.
    arms : Mapping[str, NDArray[np.intp]]
        The bins of each arm, in order from the junction to the arm's end,
        such as ``{"center": ..., "left": ..., "right": ...}``. Consecutive
        bins of an arm must be neighbors in ``env``, and each arm's first
        bin must be nearer the other arms than its last. Arms must not
        share bins; bins in no arm are ignored. Names may not contain
        ``"+"`` or ``","``.
    arm_replay_method : ArmReplayMethod
        The method, for example ``neurospatial.replay.methods.WU_FOSTER_2014``.

    Returns
    -------
    ReplayDetectionResult
        ``is_replay`` and ``template``: the replayed arm (``"left"``), arms
        (``"center,left"`` for single-arm replays of each), or pair
        (``"center+left"``, in time order for ``"strongest_pair"``);
        ``score`` is that replay's absolute weighted correlation (the
        largest, for several arms). ``criteria`` and ``values`` hold each
        arm's ``duration``, ``coverage`` and ``|wc|`` (of its strongest
        subregion) and each pair's ``|wc|`` and, with ``joint_same_sign``,
        ``same sign as joint``, keyed like ``"left: coverage > 0.5"``. With
        ``arm_fallback``, ``criteria["sequence"]`` is False for an
        arm-replay that met none of the sequence rules; its ``score`` may
        be below the threshold or NaN.

    Raises
    ------
    ValueError
        If ``posterior`` does not have one column per bin of ``env`` or has
        NaN, Inf or negative entries; if ``times`` are not evenly spaced and
        increasing; or if ``arms`` has fewer than two arms, an arm with fewer
        than two bins, a bin out of range or in two arms, an arm whose bins
        are not neighbors in order or that runs toward the junction, or a
        name that repeats or contains ``"+"`` or ``","``.

    Notes
    -----
    The weighted correlation is over every pixel of a subregion: the time
    bin's index against the position's index along the arm (or the
    path), weighted by the posterior (Wu & Foster 2014). A position bin is
    represented when its largest posterior over the subregion's time bins
    exceeds the threshold; ``coverage="represented"`` is the share of the
    arm's bins that are represented and ``"mean_range"`` the range of the
    posterior mean, in bins, over the arm's bins. A largest-peak
    subregion surrounds the first time bin of the largest smoothed peak,
    and the Gaussian is truncated at 4 standard deviations.

    References
    ----------
    Berners-Lee, A., Wu, X., & Foster, D. J. (2021). Prefrontal cortical
    neurons are selective for non-local hippocampal representations during
    replay and behavior. *Journal of Neuroscience*, 41(27), 5894-5908.

    Wu, X., & Foster, D. J. (2014). Hippocampal replay captures the unique
    topological structure of a novel environment. *Journal of
    Neuroscience*, 34(19), 6459-6469.

    Examples
    --------
    A Y maze of three 40 cm arms in 4 cm bins, and an event that runs out
    along the center arm and on into the left arm:

    >>> import numpy as np
    >>> import networkx as nx
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import detect_arm_replay
    >>> from neurospatial.replay.methods import WU_FOSTER_2014
    >>> graph = nx.Graph()
    >>> for node, xy in {0: (0, 0), 1: (0, -40), 2: (-28, 28), 3: (28, 28)}.items():
    ...     graph.add_node(node, pos=xy)
    >>> graph.add_edges_from([(0, 1), (0, 2), (0, 3)])
    >>> env = Environment.from_graph(
    ...     graph,
    ...     edge_order=[(1, 0), (0, 2), (0, 3)],
    ...     edge_spacing=0.0,
    ...     bin_size=4.0,
    ...     units="cm",
    ... )
    >>> center, left, right = np.arange(10)[::-1], np.arange(10, 20), np.arange(20, 30)
    >>> path = np.r_[center[::-1], left]  # center's end to left's end
    >>> posterior = np.eye(env.n_bins)[path]
    >>> result = detect_arm_replay(
    ...     env,
    ...     posterior,
    ...     np.arange(20) * 0.01,
    ...     arms={"center": center, "left": left, "right": right},
    ...     arm_replay_method=WU_FOSTER_2014,
    ... )
    >>> result.is_replay, result.template
    (True, 'center+left')
    """
    from neurospatial.replay.methods import ReplayCriterion, ReplayDetectionResult

    method = arm_replay_method
    posterior = np.asarray(posterior, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    n_bins = env.n_bins
    if posterior.ndim != 2 or posterior.shape[1] != n_bins:
        raise ValueError(
            f"{_OPERATION}: posterior must have shape (n_time_bins, {n_bins}), "
            f"one column per bin of env, got {posterior.shape}."
        )
    if not np.all(np.isfinite(posterior)) or np.any(posterior < 0):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: posterior has NaN, Inf or negative entries.",
                why="each row is a probability mass over the maze's bins",
                fix="decode the event again, or drop rows that could not be "
                "decoded and pass their times too",
            )
        )
    steps = np.diff(times)
    if (
        times.shape != (len(posterior),)
        or len(times) < 2
        or not np.all(np.isfinite(times))
        or np.any(steps <= 0)
        or not np.allclose(steps, steps[0], rtol=1e-6)
    ):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: times must be one evenly spaced, increasing time "
                "per posterior row.",
                why="the subregions are runs of consecutive time bins, and the "
                "peak posterior is smoothed over time",
                fix="pass the decoding time grid of the event",
            )
        )
    arm_bins = _validated_arms(arms, env)
    threshold = method.peak_threshold / n_bins
    dt = float(steps[0])
    runs = {
        name: _arm_runs(method, posterior, bins, threshold, dt, name)
        for name, bins in arm_bins.items()
    }

    values: dict[str, float] = {}
    passes: dict[str, bool] = {}
    for name, arm_runs in runs.items():
        # Report the arm's subregion: its only one, or its strongest run.
        shown = max(arm_runs, key=lambda run: run.strength, default=None)
        checks = (
            (
                ReplayCriterion(f"{name}: duration", ">=", method.min_duration),
                0.0 if shown is None else shown.duration,
            ),
            (
                ReplayCriterion(f"{name}: coverage", ">", method.min_coverage),
                0.0 if shown is None else shown.coverage,
            ),
            (
                ReplayCriterion(f"{name}: |wc|", ">", method.min_abs_correlation),
                np.nan if shown is None else abs(shown.correlation),
            ),
        )
        for criterion, value in checks:
            values[criterion.label] = float(value)
            passes[criterion.label] = criterion.evaluate(float(value), np.nan)

    def joint_correlation(first: _Run, second: _Run, start: int, stop: int) -> float:
        # The path runs from the first arm's end, through the junction, to
        # the second arm's end.
        path = np.r_[arm_bins[first.arm][::-1], arm_bins[second.arm]]
        return _pixel_correlation(posterior[start:stop][:, path])

    def same_sign(first: _Run, second: _Run, correlation: float) -> bool:
        # Along the path, the first arm is read from its end inward.
        return bool(
            np.sign(-first.correlation)
            == np.sign(second.correlation)
            == np.sign(correlation)
            != 0
        )

    def strongest(arm_runs: list[_Run]) -> _Run:
        return max(arm_runs, key=lambda run: run.strength)

    joint_criterion = ReplayCriterion("|wc|", ">", method.joint_min_abs_correlation)
    template: str | None = None
    score = np.nan
    sequence = True
    if method.joint == "only_pair":
        joint: list[tuple[str, float]] = []
        for first_arm, second_arm in combinations(arm_bins, 2):
            pair = f"{first_arm}+{second_arm}"
            if runs[first_arm] and runs[second_arm]:
                first, second = strongest(runs[first_arm]), strongest(runs[second_arm])
                correlation = joint_correlation(
                    first,
                    second,
                    min(first.start, second.start),
                    max(first.stop, second.stop),
                )
                same_way = same_sign(first, second, correlation)
                long_and_covered = first.long_and_covered and second.long_and_covered
            else:
                correlation, same_way, long_and_covered = np.nan, False, False
            label = f"{pair}: {joint_criterion.label}"
            values[label] = float(abs(correlation))
            passes[label] = joint_criterion.evaluate(abs(correlation), np.nan)
            if method.joint_same_sign:
                values[f"{pair}: same sign as joint"] = float(same_way)
                passes[f"{pair}: same sign as joint"] = same_way
            if (
                passes[label]
                and long_and_covered
                and (same_way or not method.joint_same_sign)
            ):
                joint.append((pair, abs(correlation)))
        single = [
            name
            for name, arm_runs in runs.items()
            if arm_runs and strongest(arm_runs).passes
        ]
        if len(joint) == 1:
            template, score = joint[0]
        elif not joint and single:
            template = ",".join(single)
            score = max(abs(strongest(runs[name]).correlation) for name in single)
    else:
        replays = [run for arm_runs in runs.values() for run in arm_runs if run.passes]
        best_pair: tuple[str, float] | None = None
        for one, other in combinations(replays, 2):
            if one.arm == other.arm:
                continue
            first, second = (one, other) if one.start <= other.start else (other, one)
            pair = f"{first.arm}+{second.arm}"
            # From the first replay's start to the second's end, as the
            # Berners-Lee et al. code does.
            signed = joint_correlation(first, second, first.start, second.stop)
            correlation = abs(signed)
            label = f"{pair}: {joint_criterion.label}"
            if correlation > values.get(label, -np.inf):
                values[label] = float(correlation)
                passes[label] = joint_criterion.evaluate(correlation, np.nan)
            agrees = same_sign(first, second, signed)
            if method.joint_same_sign:
                sign_label = f"{pair}: same sign as joint"
                passes[sign_label] = passes.get(sign_label, False) or agrees
                values[sign_label] = float(passes[sign_label])
            if (
                joint_criterion.evaluate(correlation, np.nan)
                and (agrees or not method.joint_same_sign)
                and (best_pair is None or correlation > best_pair[1])
            ):
                best_pair = (pair, correlation)
        if best_pair is not None:
            template, score = best_pair
        elif replays:
            best = strongest(replays)
            template, score = best.arm, abs(best.correlation)
    if template is None and method.arm_fallback:
        sustained = [run for arm_runs in runs.values() for run in arm_runs]
        strengths = [run.strength for run in sustained]
        if sustained and strengths.count(max(strengths)) == 1:
            best = sustained[strengths.index(max(strengths))]
            template, score = best.arm, abs(best.correlation)
            sequence = False
    if method.arm_fallback:
        passes["sequence"] = template is not None and sequence
        values["sequence"] = float(passes["sequence"])
    return ReplayDetectionResult(
        method=method.name,
        is_replay=template is not None,
        score=float(score),
        p_value=np.nan,
        criteria=passes,
        values=values,
        template=template,
    )


@dataclass(frozen=True)
class _Run:
    """One subregion of one arm and its single-arm measures."""

    arm: str
    start: int
    stop: int
    duration: float
    coverage: float
    correlation: float
    long_and_covered: bool
    passes: bool

    @property
    def strength(self) -> float:
        """Absolute correlation, -1 when undefined, to rank runs."""
        return abs(self.correlation) if np.isfinite(self.correlation) else -1.0


def _arm_runs(
    method: ArmReplayMethod,
    posterior: NDArray[np.float64],
    bins: NDArray[np.intp],
    threshold: float,
    dt: float,
    arm: str,
) -> list[_Run]:
    """The arm's subregions, in time order, with their measures."""
    from scipy.ndimage import gaussian_filter1d

    from neurospatial.replay.methods import ReplayCriterion

    peak = posterior[:, bins].max(axis=1)
    if method.smoothing_sigma > 0:
        peak = gaussian_filter1d(peak, method.smoothing_sigma / dt, mode="constant")
    above = peak > threshold
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(np.int8), 0]))
    spans = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]
    long_enough = ReplayCriterion("duration", ">=", method.min_duration)
    if method.subregions == "largest_peak":
        top = int(np.argmax(peak))
        spans = [span for span in spans if span[0] <= top < span[1]]
    else:
        spans = [
            span
            for span in spans
            if long_enough.evaluate((span[1] - span[0]) * dt, np.nan)
        ]
        if method.merge_max_gap is not None:
            gap = ReplayCriterion("gap", "<=", method.merge_max_gap)
            merged: list[tuple[int, int]] = []
            for span in spans:
                # Time between the previous run's last bin and this run's first.
                if merged and gap.evaluate((span[0] - merged[-1][1] + 1) * dt, np.nan):
                    merged[-1] = (merged[-1][0], span[1])
                else:
                    merged.append(span)
            spans = merged
    measures = [
        ReplayCriterion("coverage", ">", method.min_coverage),
        ReplayCriterion("|wc|", ">", method.min_abs_correlation),
    ]
    result = []
    for start, stop in spans:
        block = posterior[start:stop][:, bins]
        duration = (stop - start) * dt
        if method.coverage == "represented":
            coverage = float(np.mean(block.max(axis=0) > threshold))
        else:
            mass = block.sum(axis=1)
            positions = np.arange(1, len(bins) + 1, dtype=np.float64)
            means = (block[mass > 0] @ positions) / mass[mass > 0]
            coverage = float(np.ptp(means)) / len(bins) if means.size else 0.0
        correlation = _pixel_correlation(block)
        long_and_covered = long_enough.evaluate(duration, np.nan) and measures[
            0
        ].evaluate(coverage, np.nan)
        result.append(
            _Run(
                arm=arm,
                start=start,
                stop=stop,
                duration=duration,
                coverage=coverage,
                correlation=correlation,
                long_and_covered=long_and_covered,
                passes=long_and_covered
                and measures[1].evaluate(abs(correlation), np.nan),
            )
        )
    return result


def _validated_arms(arms: Any, env: Environment) -> dict[str, NDArray[np.intp]]:
    """Each arm's bins: disjoint, in range, contiguous and running outward."""
    n_bins = env.n_bins
    if not isinstance(arms, Mapping) or len(arms) < 2:
        raise ValueError(
            _format_error(
                f"{_OPERATION}: arms must map at least two arm names to bins.",
                why="replay is classified per arm and per pair of arms",
                fix="pass arms={'center': bins, 'left': bins, ...} with each "
                "arm's bins from the junction outward",
            )
        )
    checked: dict[str, NDArray[np.intp]] = {}
    seen: set[int] = set()
    graph = env.connectivity
    for raw_name, bins in arms.items():
        name = str(raw_name)
        if name in checked or "+" in name or "," in name:
            raise ValueError(
                f"{_OPERATION}: arm name {raw_name!r} repeats another or "
                "contains '+' or ','; results join arm names with them."
            )
        array = np.asarray(bins)
        if (
            array.ndim != 1
            or array.size < 2
            or not np.issubdtype(array.dtype, np.integer)
            or np.any(array < 0)
            or np.any(array >= n_bins)
        ):
            raise ValueError(
                f"{_OPERATION}: arm {name!r} must be at least two bin indices "
                f"in [0, {n_bins}), got {bins!r}."
            )
        shared = seen & set(array.tolist())
        if shared or len(set(array.tolist())) != array.size:
            raise ValueError(
                f"{_OPERATION}: arm {name!r} repeats or shares bins "
                f"{sorted(shared) or 'within itself'}; each bin belongs to one arm."
            )
        gaps = [
            (int(a), int(b))
            for a, b in pairwise(array)
            if not graph.has_edge(int(a), int(b))
        ]
        if gaps:
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: arm {name!r} has consecutive bins that are "
                    f"not neighbors in env, such as {gaps[0]}.",
                    why="an arm's bins are its positions in order along it",
                    fix="list each arm's bins in order from the junction to its end",
                )
            )
        seen |= set(array.tolist())
        checked[name] = array.astype(np.intp)
    centers = np.asarray(env.bin_centers, dtype=np.float64)
    for name, bins in checked.items():
        others = np.concatenate(
            [other[:1] for key, other in checked.items() if key != name]
        )
        near = np.linalg.norm(centers[others] - centers[bins[0]], axis=1).min()
        far = np.linalg.norm(centers[others] - centers[bins[-1]], axis=1).min()
        if near > far:
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: arm {name!r} runs toward the junction: its "
                    "last bin is nearer the other arms than its first.",
                    why="positions along each arm, and along a path through "
                    "the junction, are read from the junction outward",
                    fix=f"reverse the arm's bins, e.g. arms[{name!r}][::-1]",
                )
            )
    return checked


def _pixel_correlation(weights: NDArray[np.float64]) -> float:
    """Weighted correlation of time index and position index over pixels."""
    total = weights.sum()
    if total <= 0:
        return np.nan
    n_time, n_position = weights.shape
    t = np.arange(n_time, dtype=np.float64)[:, None]
    y = np.arange(n_position, dtype=np.float64)[None, :]
    mean_t = float((weights * t).sum() / total)
    mean_y = float((weights * y).sum() / total)
    cov = float((weights * (t - mean_t) * (y - mean_y)).sum() / total)
    var_t = float((weights * (t - mean_t) ** 2).sum() / total)
    var_y = float((weights * (y - mean_y) ** 2).sum() / total)
    if var_t <= 0 or var_y <= 0:
        return np.nan
    return float(cov / np.sqrt(var_t * var_y))
