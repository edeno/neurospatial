"""Replay score for a circular track (Wikenheiser & Redish 2013).

On a circular track the decoded position has no ends: a sequence that runs
past the point where the track's linear position starts again (the seam)
jumps from the largest linear position to the smallest, and a line fit along
the linear position breaks there. :func:`fit_circular_sequence` follows the
decoded position around the loop instead, as Wikenheiser & Redish (2013) did:
the circular mean of each time bin's posterior, the step between successive
time bins taken the short way around the loop, and a line fit to the
cumulative sum of those steps.

Functions
---------
fit_circular_sequence
    R² and replay speed of the cumulative decoded path around a loop.

Result Dataclasses
------------------
CircularSequenceResult
    Container for the circular-track sequence fit.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial.replay._inputs import _FIT_WHY, _posterior_times, _probability_rows
from neurospatial.replay.significance import _Scorer
from neurospatial.replay.trajectory import _SCORE_DECIMALS

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from neurospatial.environment import Environment

_OPERATION = "fit_circular_sequence()"

_UNDEFINED_RESULTANT = 1e-12
"""Mean resultant length at or below which a circular mean is undefined.

The default of :func:`neurospatial.events.align_continuous_to_events` for
circular features. A posterior spread evenly around the loop, such as the
prior of a time bin decoded without spikes, has no direction.
"""

_BUILD_A_LOOP = (
    "Environment.from_graph(graph, edge_order=[(0, 1), (1, 2), ..., "
    "(n - 1, 0)], bin_size=..., edge_spacing=0.0) from a cycle graph, "
    "networkx.cycle_graph(n), whose nodes carry 'pos' around the track"
)


@dataclass(frozen=True, repr=False, eq=False)
class CircularSequenceResult(ReadOnlyArraysMixin, ResultMixin):
    """A sequence fit to a decoded event on a circular track.

    :func:`fit_circular_sequence` takes the circular mean of each time bin's
    posterior, steps between successive time bins the short way around the
    loop, and fits a least-squares line of the cumulative sum of the steps on
    time (Wikenheiser & Redish 2013).

    Attributes
    ----------
    slope : float
        Slope of the fitted line as replay speed around the loop, in
        environment units per second (cm/s for a track in cm): the angular
        slope times the loop length over 2π. Positive when the decoded
        position moves in the direction of increasing linear position, the
        direction of the track's ``edge_order``; whether that is forward or
        backward replay depends on the direction the animal ran. 0 for a
        stationary decode.
    angular_slope : float
        Slope of the fitted line in radians per second, with the sign of
        ``slope``.
    r_squared : float
        Coefficient of determination (R²) of the line, in [0, 1]. NaN when
        the decoded position never changes (a stationary decode), which has
        no variance to explain.
    decoded_angles : NDArray[np.float64], shape (n_time_bins,)
        Circular mean of each time bin's posterior, in radians in
        ``[0, 2π)``: the decoded position's distance along the loop from
        the start of its linear position, over the loop length, times 2π.
    cumulative_angles : NDArray[np.float64], shape (n_time_bins,)
        Cumulative sum of the steps between successive decoded angles, each
        wrapped to ``(-π, π]``, in radians: 0 at the first time bin. This is
        the decoded path unrolled around the loop, the values the line is
        fit to.

    See Also
    --------
    fit_circular_sequence : Function that produces this result.
    LinearFitResult : Line fit along a track that does not close on itself.

    Examples
    --------
    >>> result = fit_circular_sequence(env, posterior, times)  # doctest: +SKIP
    >>> print(f"{result.slope:.0f} cm/s, R² = {result.r_squared:.2f}")  # doctest: +SKIP
    """

    slope: float
    angular_slope: float
    r_squared: float
    decoded_angles: NDArray[np.float64]
    cumulative_angles: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own read-only copies of the decoded and cumulative angles."""
        self._own_arrays()

    @property
    def score(self) -> float:
        """The replay score of this fit: its R²."""
        return self.r_squared

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "slope": self.slope,
            "angular_slope": self.angular_slope,
            "r_squared": self.r_squared,
        }


def fit_circular_sequence(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
) -> CircularSequenceResult:
    """Fit the decoded path around a circular track (Wikenheiser & Redish 2013).

    Each time bin's decoded position is the circular mean of its posterior:
    each bin's position is read as an angle around the loop, and the decoded
    angle is the direction of the posterior-weighted sum of the bins' unit
    vectors. The steps between successive decoded angles, each taken the
    short way around the loop, are summed cumulatively, and a least-squares
    line of that cumulative sum on time is fit. The R² of the line is the
    replay score; its slope, the replay speed around the loop, gives the
    direction.

    Test the R² against shuffles with
    :func:`~neurospatial.replay.replay_significance` and
    ``nulls="time_bin_permutation"``, the shuffle of Karlsson & Frank (2009)
    that Wikenheiser & Redish used.

    Parameters
    ----------
    env : Environment
        The circular track the posterior was decoded on: a track whose
        ``edge_order`` runs once around a loop, each edge starting where
        the previous one ends and the last ending where the first starts,
        with no ``edge_spacing``. Build it with
        :meth:`~neurospatial.Environment.from_graph` from a cycle graph.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability mass over spatial bins for each time bin. Each
        row must sum to 1. Drop time bins without spikes, with their times,
        first: Wikenheiser & Redish decoded only time bins with at least one
        spike, and the prior of an empty time bin, spread evenly around the
        loop, has no circular mean.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds; the independent
        variable of the fit. Need not be evenly spaced, so the time bins
        left after dropping empty ones keep their own times.

    Returns
    -------
    CircularSequenceResult
        Slope (environment units per second), angular slope (radians per
        second), R², and the decoded and cumulative angles of each time bin.

    Raises
    ------
    ValueError
        If ``env`` is not a circular track (see ``env``), if ``posterior``
        is not 2-D with one column per bin, if a posterior row is not a
        probability mass (NaN or Inf, negative, or not summing to 1) or has
        no circular mean (mass spread so evenly around the loop that its
        mean resultant length is at most 1e-12), or if ``times`` does not
        have one finite entry per row or holds a single distinct time.

    See Also
    --------
    CircularSequenceResult : Container for the fit.
    fit_linear_trajectory : Line fit along a track that does not close.
    neurospatial.replay.replay_significance : Test the score against shuffles.

    Notes
    -----
    Wikenheiser & Redish (2013, Methods) decoded 10 ms time bins with a
    uniform prior over 64 bins of an 80 cm diameter circular track, only
    time bins with at least one spike, took "the circular mean of the
    posterior probability distribution (that is, the vector sum with the
    central angle of each bin weighted by the amount of probability it
    contained)", "computed the cumulative sum of the difference between
    successive decoded positions", "performed a linear regression on the
    cumulative sum of differences", and tested events with the shuffle of
    Karlsson & Frank (2009). They state no more than that; this function
    makes these choices where they are silent:

    - **Bin angles.** A bin's angle is its center's distance along the
      loop over the loop length, times 2π. On a circular track this is the
      central angle the paper uses; on a loop of another shape it is the
      fraction of the loop, so equal steps along the track are equal steps
      in angle.
    - **Wrapped steps.** Each step between successive decoded angles is
      wrapped to ``(-π, π]``, the shorter way around the loop. The paper
      does not say so, but without it the cumulative sum jumps by 2π where
      the decoded position crosses the seam, and following the position
      around the loop is what the cumulative sum is for. A step of exactly
      half the loop is counted as positive.
    - **Fitted points.** The cumulative sum starts at 0 at the first time
      bin, so the line is fit to one point per time bin (``n_time_bins``
      points, not ``n_time_bins - 1`` steps).
    - **Regressor.** The line is fit on ``times``, not on the time-bin
      number, so dropped time bins leave a gap in time and the slope is in
      physical units. For evenly spaced time bins with none dropped this
      gives the same R² as fitting on the bin number. Steps are between
      successive decoded time bins, across any gap.
    - **Score.** The tested statistic is the R² of the line, the statistic
      of the Karlsson & Frank (2009) procedure the paper cites; the paper
      reports the slope's sign as the direction ("forward sequences have
      positive slopes") and median slopes without units.

    **Stationary decodes.** If the decoded angle never changes, the slope
    is 0 and R² is NaN, as for :func:`fit_linear_trajectory`, so a
    stationary decode passes no R² screen and
    :func:`~neurospatial.replay.replay_significance` refuses to test it. A
    decode that only jitters about one place has a low R².

    The R² of a cumulative sum is not that of the positions: a cumulative
    sum of steps that are random in sign drifts like a random walk, and a
    line fit to a random walk has a sizeable R² by chance. Judge the R²
    against shuffles of the same event, not against a fixed threshold.

    References
    ----------
    Wikenheiser, A. M., & Redish, A. D. (2013). The balance of forward and
    backward hippocampal sequences shifts across behavioral states.
    *Hippocampus*, 23(1), 22-29. https://doi.org/10.1002/hipo.22049

    Karlsson, M. P., & Frank, L. M. (2009). Awake replay of remote
    experiences in the hippocampus. *Nature Neuroscience*, 12(7), 913-918.
    https://doi.org/10.1038/nn.2344

    Examples
    --------
    A circular track 80 cm across in 64 bins of about 4 cm, as in
    Wikenheiser & Redish, built from a 16-sided loop, and an event that sweeps 4 bins per 10 ms
    across the seam where the linear position starts again:

    >>> import networkx as nx
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import fit_circular_sequence
    >>>
    >>> n_nodes, radius = 16, 40.0
    >>> loop = nx.cycle_graph(n_nodes)
    >>> for node in loop.nodes:
    ...     angle = 2 * np.pi * node / n_nodes
    ...     loop.nodes[node]["pos"] = (radius * np.cos(angle), radius * np.sin(angle))
    >>> env = Environment.from_graph(
    ...     loop,
    ...     edge_order=[(i, (i + 1) % n_nodes) for i in range(n_nodes)],
    ...     bin_size=4.0,
    ...     units="cm",
    ... )
    >>> env.n_bins
    64
    >>> n_time_bins = 12
    >>> posterior = np.zeros((n_time_bins, env.n_bins))
    >>> posterior[np.arange(n_time_bins), (40 + 4 * np.arange(n_time_bins)) % 64] = 1.0
    >>> times = np.arange(n_time_bins) * 0.01
    >>> result = fit_circular_sequence(env, posterior, times)
    >>> print(
    ...     f"{result.slope:.0f} cm/s, {result.angular_slope:.1f} rad/s, "
    ...     f"R² = {result.r_squared:.2f}"
    ... )
    1561 cm/s, 39.3 rad/s, R² = 1.00

    A line along the linear position breaks where the sweep crosses the
    seam:

    >>> from neurospatial.replay import fit_linear_trajectory
    >>> linear = fit_linear_trajectory(env, posterior, times, method="map")
    >>> print(f"R² = {linear.r_squared:.2f}")
    R² = 0.44
    """
    return _CircularSequenceScorer(env, posterior, times).observe()


class _CircularSequenceScorer(_Scorer):
    """Fits one event and scores its shuffles in batches.

    ``replay_significance`` uses it for :func:`fit_circular_sequence`, so
    thousands of shuffles cost a few matrix products rather than a call
    each; the event and its shuffles go through the same arithmetic.
    """

    _BATCH = 256

    def __init__(self, env: Environment, posterior: Any, times: Any) -> None:
        self.angles, self.loop_length = _bin_angles(env)
        posterior = np.asarray(posterior)
        if posterior.ndim != 2:
            raise ValueError(
                f"{_OPERATION}: posterior must be 2-D, (n_time_bins, n_bins), "
                f"got shape {posterior.shape}."
            )
        if posterior.shape[1] != env.n_bins:
            raise ValueError(
                f"{_OPERATION}: posterior has {posterior.shape[1]} columns but "
                f"the environment has {env.n_bins} bins. Pass the posterior "
                "decoded on this environment, with one column per bin."
            )
        self.posterior = _probability_rows(posterior, _OPERATION, sums="one")
        self.times = _posterior_times(
            times, len(self.posterior), _OPERATION, why=_FIT_WHY, distinct=True
        )

    def observe(self) -> CircularSequenceResult:
        decoded = _decoded_angles(self.posterior[np.newaxis], self.angles)
        cumulative = _cumulative_angles(decoded)
        angular_slope, r_squared = _fit_lines(self.times, cumulative)
        return CircularSequenceResult(
            slope=float(angular_slope[0] * self.loop_length / (2 * np.pi)),
            angular_slope=float(angular_slope[0]),
            r_squared=float(r_squared[0]),
            decoded_angles=decoded[0],
            cumulative_angles=cumulative[0],
        )

    def scores(
        self, posteriors: Iterable[NDArray[np.float64]], n_shuffles: int
    ) -> NDArray[np.float64]:
        out = np.empty(n_shuffles)
        filled = 0
        iterator = iter(posteriors)
        while batch := list(itertools.islice(iterator, self._BATCH)):
            decoded = _decoded_angles(np.stack(batch), self.angles)
            _, r_squared = _fit_lines(self.times, _cumulative_angles(decoded))
            out[filled : filled + len(batch)] = r_squared
            filled += len(batch)
        return out


def _bin_angles(env: Environment) -> tuple[NDArray[np.float64], float]:
    """Each bin's angle around a circular track, and the loop's length.

    Parameters
    ----------
    env : Environment
        A track whose ``edge_order`` runs once around a loop, head to tail,
        with no ``edge_spacing``.

    Returns
    -------
    angles : NDArray[np.float64], shape (n_bins,)
        Each bin center's distance along the loop over the loop length,
        times 2π, in radians in ``[0, 2π)``, in bin numbering.
    loop_length : float
        Length of the loop, in environment units.

    Raises
    ------
    ValueError
        If ``env`` is not such a track.
    """
    linearization = env.binning.linearization
    if linearization is None:
        _refuse_open_track(
            f"this is a {env.n_dims}-D {env.binning.kind} environment, not a track"
        )
    edge_order = list(linearization.edge_order)
    heads = [edge[0] for edge in edge_order]
    gaps = np.atleast_1d(np.asarray(linearization.edge_spacing, dtype=np.float64))
    if len(edge_order) < 3 or len(set(heads)) != len(edge_order):
        _refuse_open_track(
            "its edge_order does not run once around a loop of three or more "
            "track nodes"
        )
    unjoined = [
        k
        for k, edge in enumerate(edge_order)
        if edge[1] != edge_order[(k + 1) % len(edge_order)][0]
    ]
    if unjoined:
        k = unjoined[0]
        following = edge_order[(k + 1) % len(edge_order)]
        _refuse_open_track(
            f"its edge_order does not close on itself: edge {k}, "
            f"{tuple(edge_order[k])!r}, ends where edge "
            f"{(k + 1) % len(edge_order)}, {tuple(following)!r}, does not "
            "start"
        )
    if np.any(gaps > 0):
        _refuse_open_track(
            f"its edge_spacing puts gaps of up to {gaps.max():g} between "
            "segments, so the linear position is not distance around the loop"
        )
    from neurospatial.environment import _track_extents

    edges = linearization.grid_edges[0]
    loop_start, loop_length = float(edges[0]), float(edges[-1] - edges[0])
    extents = _track_extents(env)
    assert extents is not None
    centers = (extents[0] + extents[1]) / 2
    angles = np.mod(2 * np.pi * (centers - loop_start) / loop_length, 2 * np.pi)
    return np.asarray(angles, dtype=np.float64), loop_length


def _refuse_open_track(problem: str) -> NoReturn:
    """Refuse an environment that is not a circular track."""
    raise ValueError(
        _format_error(
            f"{_OPERATION} needs a circular track, but {problem}.",
            why="the score follows the decoded position around a loop, reading "
            "each bin's position as an angle (its distance along the loop over "
            "the loop length, times 2π); that needs a linear position that "
            "runs once around the loop and returns to its start",
            fix=f"build the loop with {_BUILD_A_LOOP}; on a track with ends, "
            "use fit_linear_trajectory",
        )
    )


def _decoded_angles(
    posteriors: NDArray[np.float64], angles: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Circular mean of each posterior row, in radians in ``[0, 2π)``.

    Parameters
    ----------
    posteriors : NDArray[np.float64], shape (..., n_time_bins, n_bins)
        Probability rows.
    angles : NDArray[np.float64], shape (n_bins,)
        Each bin's angle around the loop.

    Returns
    -------
    NDArray[np.float64], shape (..., n_time_bins)
        The direction of each row's probability-weighted sum of unit vectors.

    Raises
    ------
    ValueError
        If a row's mean resultant length is at most ``_UNDEFINED_RESULTANT``.
    """
    cosine = posteriors @ np.cos(angles)
    sine = posteriors @ np.sin(angles)
    resultant = np.hypot(cosine, sine)
    undefined = resultant <= _UNDEFINED_RESULTANT
    if np.any(undefined):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: {int(np.sum(undefined))} posterior rows have no "
                "circular mean: their mass is spread so evenly around the loop "
                f"that its mean resultant length is at most {_UNDEFINED_RESULTANT:g}.",
                why="the decoded position is the direction of the posterior's "
                "weighted sum of bin directions, and a sum of zero length has "
                "no direction; a time bin decoded without spikes keeps the "
                "uniform prior, which is such a row",
                fix="drop time bins without spikes, with their times, before "
                "scoring (empty_time_bins='drop' in a ReplayMethod), as "
                "Wikenheiser & Redish (2013) decoded only time bins with spikes",
            )
        )
    decoded: NDArray[np.float64] = np.mod(np.arctan2(sine, cosine), 2 * np.pi)
    # A tiny negative angle wraps to 2π in floating point; it is 0.
    decoded[decoded == 2 * np.pi] = 0.0
    return decoded


def _cumulative_angles(decoded: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative sum of successive steps wrapped to ``(-π, π]``, from 0.

    Parameters
    ----------
    decoded : NDArray[np.float64], shape (..., n_time_bins)
        Decoded angles, in radians.

    Returns
    -------
    NDArray[np.float64], shape (..., n_time_bins)
        0 at the first time bin, then the running sum of the steps.
    """
    # pi - ((pi - step) mod 2 pi) lies in (-pi, pi]: a half-loop step is +pi.
    steps = np.pi - np.mod(np.pi - np.diff(decoded, axis=-1), 2 * np.pi)
    cumulative: NDArray[np.float64] = np.concatenate(
        [np.zeros((*decoded.shape[:-1], 1)), np.cumsum(steps, axis=-1)], axis=-1
    )
    return cumulative


def _fit_lines(
    times: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[Any, Any]:
    """Least-squares slope and R² of each sequence of ``values`` on ``times``.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_time_bins,)
        At least two distinct times.
    values : NDArray[np.float64], shape (..., n_time_bins)
        One sequence per leading index.

    Returns
    -------
    slope, r_squared : NDArray[np.float64], shape (...)
        Slope, and R² in [0, 1]; R² is NaN for a sequence that never changes,
        which has no variance to explain. R² is rounded to
        ``_SCORE_DECIMALS`` so that an event and a shuffle equal in exact
        arithmetic but computed in batches of different sizes tie.
    """
    centered_times = times - times.mean()
    centered = values - values.mean(axis=-1, keepdims=True)
    covariance = centered @ centered_times
    time_variance = centered_times @ centered_times
    value_variance = np.einsum("...t,...t->...", centered, centered)
    slope = covariance / time_variance
    constant = np.ptp(values, axis=-1) == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        r_squared = np.where(
            constant,
            np.nan,
            np.clip(covariance**2 / (time_variance * value_variance), 0.0, 1.0),
        )
    return slope, np.round(r_squared, _SCORE_DECIMALS)
