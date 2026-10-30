"""Replay of single arms and pairs of arms on a multi-arm maze.

On a maze with several arms, a replayed trajectory may run along one arm or
cross the junction from one arm into another. Wu & Foster (2014) cut each
candidate event into one subregion per arm, around the strongest
representation of that arm, scored each subregion and each pair of
subregions, and called an event a single-arm or a joint-arm replay by
rules that keep the two from overlapping. :class:`ArmReplayMethod` records
such a method and :func:`detect_arm_replay` applies it to one decoded event.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error

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
    min_time_bins : int
        Fewest time bins in a single-arm subregion (Wu & Foster 2014: 5,
        that is 50 ms).
    min_coverage : float
        Share of an arm's bins that must be represented in its subregion,
        exceeded strictly (Wu & Foster 2014: 50%).
    min_abs_correlation : float
        Absolute weighted correlation a single-arm subregion must exceed
        (Wu & Foster 2014: 0.5).
    joint_min_abs_correlation : float
        Absolute weighted correlation a joint subregion must exceed (Wu &
        Foster 2014: 0.5).
    decoding : Mapping[str, Any]
        The paper's decoding settings, for reference (not enforced).
    notes : tuple of str
        Choices the paper does not specify.
    """

    name: str
    reference: str
    peak_threshold: float = 5.0
    smoothing_sigma: float = 0.01
    min_time_bins: int = 5
    min_coverage: float = 0.5
    min_abs_correlation: float = 0.5
    joint_min_abs_correlation: float = 0.5
    decoding: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the mapping and refuse settings that cannot run."""
        object.__setattr__(self, "decoding", MappingProxyType(dict(self.decoding)))
        object.__setattr__(self, "notes", tuple(self.notes))
        prefix = f"ArmReplayMethod {self.name!r}"
        for name in ("peak_threshold", "smoothing_sigma"):
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
        cells = self.min_time_bins
        if (
            isinstance(cells, bool)
            or not isinstance(cells, (int, np.integer))
            or cells < 2
        ):
            raise ValueError(
                f"{prefix}: min_time_bins must be an integer of at least 2, got "
                f"{cells!r}."
            )

    def __hash__(self) -> int:
        """Hash by name and reference; the decoding mapping is not hashable."""
        return hash((self.name, self.reference))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mapping cannot pickle."""
        return (
            type(self),
            (
                self.name,
                self.reference,
                self.peak_threshold,
                self.smoothing_sigma,
                self.min_time_bins,
                self.min_coverage,
                self.min_abs_correlation,
                self.joint_min_abs_correlation,
                dict(self.decoding),
                self.notes,
            ),
        )


def detect_arm_replay(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    arms: Mapping[str, NDArray[np.intp]],
    arm_replay_method: ArmReplayMethod,
) -> ReplayDetectionResult:
    """Find a single-arm or joint-arm replay in one decoded event.

    Each arm's subregion is the run of time bins around the largest peak
    of the arm's smoothed peak posterior, while that stays above
    ``peak_threshold`` times chance. A single-arm subregion is a replay
    when it is long enough, covers enough of the arm and its posterior
    is correlated with time. Each pair of arms forms a joint subregion,
    from the first to the last time bin of their subregions, over the
    path from one arm's end through the junction to the other's: it is a
    replay when its correlation passes, both arms' subregions are long
    and cover enough, and both run the same way along the path. A joint
    replay is reported only when no other pair is one; single-arm replays
    only when no pair is one (Wu & Foster 2014).

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
        such as ``{"center": ..., "left": ..., "right": ...}``. Arms must
        not share bins; bins in no arm are ignored.
    arm_replay_method : ArmReplayMethod
        The method, for example ``neurospatial.replay.methods.WU_FOSTER_2014``.

    Returns
    -------
    ReplayDetectionResult
        ``is_replay`` and ``template``: the replayed arm (``"left"``), arms
        (``"center,left"`` for single-arm replays of each), or pair
        (``"center+left"`` for a joint replay); ``score`` is that replay's
        absolute weighted correlation (the largest, for several arms).
        ``criteria`` and ``values`` hold each arm's ``time bins``,
        ``coverage`` and ``|wc|`` and each pair's ``|wc|`` and ``same
        direction``, keyed like ``"left: coverage > 0.5"``.

    Raises
    ------
    ValueError
        If ``posterior`` does not have one column per bin of ``env``, if
        ``times`` are not evenly spaced and increasing, or if ``arms`` has
        fewer than two arms, an arm with fewer than two bins, a bin out of
        range or a bin in two arms.

    Notes
    -----
    The weighted correlation is over every pixel of a subregion: the time
    bin's index against the position's index along the arm (or the
    path), weighted by the posterior (Wu & Foster 2014). A position bin is
    represented when its largest posterior over the subregion's time bins
    exceeds the threshold; coverage is the share of the arm's bins that
    are represented.

    References
    ----------
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
    from scipy.ndimage import gaussian_filter1d

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
    arm_bins = _validated_arms(arms, n_bins)
    threshold = method.peak_threshold / n_bins
    sigma = method.smoothing_sigma / float(steps[0])

    subregions: dict[str, tuple[int, int] | None] = {}
    values: dict[str, float] = {}
    passes: dict[str, bool] = {}
    local_correlation: dict[str, float] = {}
    long_and_covered: dict[str, bool] = {}
    single: list[str] = []
    for name, bins in arm_bins.items():
        peak = posterior[:, bins].max(axis=1)
        if sigma > 0:
            peak = gaussian_filter1d(peak, sigma, mode="constant")
        top = int(np.argmax(peak))
        above = peak > threshold
        if not above[top]:
            subregions[name] = None
            n_time, coverage, correlation = 0, 0.0, np.nan
        else:
            start, stop = top, top + 1
            while start > 0 and above[start - 1]:
                start -= 1
            while stop < len(peak) and above[stop]:
                stop += 1
            subregions[name] = (start, stop)
            block = posterior[start:stop][:, bins]
            n_time = stop - start
            coverage = float(np.mean(block.max(axis=0) > threshold))
            correlation = _pixel_correlation(block)
        local_correlation[name] = correlation
        checks = (
            (ReplayCriterion(f"{name}: time bins", ">=", method.min_time_bins), n_time),
            (ReplayCriterion(f"{name}: coverage", ">", method.min_coverage), coverage),
            (
                ReplayCriterion(f"{name}: |wc|", ">", method.min_abs_correlation),
                abs(correlation),
            ),
        )
        for criterion, value in checks:
            values[criterion.label] = float(value)
            passes[criterion.label] = criterion.evaluate(float(value), np.nan)
        long_and_covered[name] = all(
            passes[criterion.label] for criterion, _ in checks[:2]
        )
        if all(passes[criterion.label] for criterion, _ in checks):
            single.append(name)

    joint: list[str] = []
    joint_correlation: dict[str, float] = {}
    for first, second in combinations(arm_bins, 2):
        pair = f"{first}+{second}"
        both = subregions[first] is not None and subregions[second] is not None
        if both:
            spans = (subregions[first], subregions[second])
            start = min(span[0] for span in spans if span is not None)
            stop = max(span[1] for span in spans if span is not None)
            # The path runs from the first arm's end, through the junction,
            # to the second arm's end.
            path = np.r_[arm_bins[first][::-1], arm_bins[second]]
            correlation = _pixel_correlation(posterior[start:stop][:, path])
            # Along the path, the first arm is read from its end inward.
            same_way = bool(
                np.sign(-local_correlation[first])
                == np.sign(local_correlation[second])
                != 0
            )
        else:
            correlation, same_way = np.nan, False
        joint_correlation[pair] = correlation
        criterion = ReplayCriterion(
            f"{pair}: |wc|", ">", method.joint_min_abs_correlation
        )
        values[criterion.label] = float(abs(correlation))
        passes[criterion.label] = criterion.evaluate(abs(correlation), np.nan)
        values[f"{pair}: same direction"] = float(same_way)
        passes[f"{pair}: same direction"] = same_way
        if (
            passes[criterion.label]
            and same_way
            and long_and_covered[first]
            and long_and_covered[second]
        ):
            joint.append(pair)

    if len(joint) == 1:
        template, score = joint[0], abs(joint_correlation[joint[0]])
    elif not joint and single:
        template = ",".join(single)
        score = max(abs(local_correlation[name]) for name in single)
    else:
        template, score = None, np.nan
    return ReplayDetectionResult(
        method=method.name,
        is_replay=template is not None,
        score=float(score),
        p_value=np.nan,
        criteria=passes,
        values=values,
        template=template,
    )


def _validated_arms(arms: Any, n_bins: int) -> dict[str, NDArray[np.intp]]:
    """Each arm's bins as integer arrays, checked to be disjoint and in range."""
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
    for name, bins in arms.items():
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
        seen |= set(array.tolist())
        checked[str(name)] = array.astype(np.intp)
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
