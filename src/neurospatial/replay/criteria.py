"""Per-event metrics for criteria-based replay detection.

Criteria-based replay detection accepts a decoded candidate event when
simple properties of its posterior pass thresholds: the decoded position
never jumps too far between time bins, the posterior is sharp, the event
covers enough of the track, and so on. Each lab combines different
properties with different thresholds. :func:`trajectory_metrics` computes
every published property under its own name, so a detection recipe can
threshold exactly the quantities a paper used. :func:`template_fractions`
gives the share of a jointly decoded event in each template (running
direction or track).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin

if TYPE_CHECKING:
    from neurospatial.decoding import JointDecodingResult
    from neurospatial.environment import Environment

_OPERATION = "trajectory_metrics()"
_METRICS = ("geodesic", "euclidean")


@dataclass(frozen=True, repr=False, eq=False)
class TrajectoryMetrics(ReadOnlyArraysMixin, ResultMixin):
    """Per-time-bin and per-event metrics of one decoded candidate event.

    Positions and distances are in environment units (cm for an environment
    in cm). Distances between bins use ``metric``; distances between
    posterior means are Euclidean. ``track_length`` is NaN off a track, and
    ``map_range`` and ``mean_range`` are NaN off a track and on a branched
    track (see :func:`trajectory_metrics`).

    Attributes
    ----------
    map_bins : NDArray[np.int64], shape (n_time_bins,)
        Bin of highest posterior density in each time bin, as
        :func:`neurospatial.decoding.map_bin` returns.
    map_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Center of each MAP bin, in environment coordinates.
    peak_probability : NDArray[np.float64], shape (n_time_bins,)
        Posterior mass in the MAP bin, as given (not renormalized). On equal
        bins this is the largest mass in the time bin.
    mean_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Posterior mean position of each renormalized row, as
        :func:`neurospatial.decoding.mean_position` returns. On a track whose
        bins are 2-D points the mean of mass on two arms can lie off the
        track.
    map_jumps : NDArray[np.float64], shape (n_time_bins - 1,)
        Distance, by ``metric``, between the MAP bins of consecutive time
        bins.
    mean_jumps : NDArray[np.float64], shape (n_time_bins - 1,)
        Euclidean distance between the posterior means of consecutive time
        bins.
    spread : NDArray[np.float64], shape (n_time_bins,)
        Square root of the posterior's second central moment,
        :math:`\\sqrt{\\sum_j \\lVert x_j - \\bar x \\rVert^2 P_j}`, over
        the bin centers in environment coordinates.
    empty_time_bins : NDArray[np.bool_] or None, shape (n_time_bins,)
        True for a time bin without spikes; None when no ``spike_counts``
        were given.
    max_posterior : NDArray[np.float64], shape (n_bins,)
        Largest posterior mass, as given, that each bin reaches during the
        event.
    space_fractions : NDArray[np.float64], shape (n_bins,)
        Each bin's share of the environment's length, area or volume.
    metric : {"geodesic", "euclidean"}
        How distances between bins were measured.
    units : str or None
        Units of positions and distances (the environment's units).
    n_time_bins : int
        Number of time bins in the event.
    duration : float
        Last minus first time-bin center, in seconds. Add one time-bin width
        for the time from the start of the first bin to the end of the last.
    track_length : float
        Summed length of the track's bins, without the spacing between
        track segments; NaN off a track.
    max_map_jump, max_mean_jump : float
        Largest of ``map_jumps`` and ``mean_jumps``; NaN with one time bin.
    mean_mean_jump : float
        Mean of ``mean_jumps``, the average center-of-mass jump between
        consecutive time bins; NaN with one time bin.
    map_range, mean_range : float
        Largest minus smallest position along the track of the MAP bin, and
        of the posterior mean along the track; NaN off a track and on a
        branched track.
    start_to_end_map_distance : float
        Distance, by ``metric``, between the MAP bins of the first and last
        time bins.
    start_to_end_mean_distance : float
        Euclidean distance between the first and last posterior means.
    map_path_length : float
        Sum of ``map_jumps``.
    dispersion : float
        Root-mean-square distance of the posterior means from their average
        over the event.
    mean_spread : float
        Mean of ``spread`` over time bins.
    mean_peak_probability : float
        Mean of ``peak_probability`` over time bins.
    empty_fraction : float
        Fraction of time bins without spikes; NaN without ``spike_counts``.

    See Also
    --------
    trajectory_metrics : Function that produces this result.
    """

    map_bins: NDArray[np.int64]
    map_positions: NDArray[np.float64]
    peak_probability: NDArray[np.float64]
    mean_positions: NDArray[np.float64]
    map_jumps: NDArray[np.float64]
    mean_jumps: NDArray[np.float64]
    spread: NDArray[np.float64]
    empty_time_bins: NDArray[np.bool_] | None
    max_posterior: NDArray[np.float64]
    space_fractions: NDArray[np.float64]
    metric: str
    units: str | None
    n_time_bins: int
    duration: float
    track_length: float
    max_map_jump: float
    max_mean_jump: float
    mean_mean_jump: float
    map_range: float
    mean_range: float
    start_to_end_map_distance: float
    start_to_end_mean_distance: float
    map_path_length: float
    dispersion: float
    mean_spread: float
    mean_peak_probability: float
    empty_fraction: float

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    def well_decoded_fraction(
        self, threshold: float = 5.0, *, chance_bins: int | None = None
    ) -> float:
        """Fraction of space the event decodes well at some time (Wu & Foster 2014).

        A bin is well decoded when its posterior mass exceeds ``threshold``
        times chance, ``threshold / chance_bins``, in at least one time bin
        of the event. The fraction is the share of the environment's length,
        area or volume in such bins.

        Parameters
        ----------
        threshold : float, default=5.0
            Multiple of chance that a bin's largest posterior mass must
            exceed (Wu & Foster 2014).
        chance_bins : int or None, default=None
            Number of bins the posterior was normalized over, so that chance
            is ``1 / chance_bins``. None uses the environment's bins. For one
            template's share of a joint decode, pass the bins of all
            templates.

        Returns
        -------
        float
            Fraction of the environment's space, in [0, 1].

        Raises
        ------
        ValueError
            If ``threshold`` is not a positive finite number or
            ``chance_bins`` is not a positive integer.

        Notes
        -----
        The Mallory et al. (2025) code (``coverage_wu``) applies the
        threshold to one direction's share of a posterior decoded jointly
        over both running directions, with chance ``1 / (2 * n_bins)``. To
        reproduce it, pass that share, ``result.joint(name)`` of a
        :func:`~neurospatial.decoding.decode_jointly` result, to
        :func:`trajectory_metrics` and call
        ``well_decoded_fraction(chance_bins=2 * env.n_bins)``. On unequal
        bins, ``1 / chance_bins`` is not the mass a uniform posterior puts in
        each bin; the papers used equal bins.

        References
        ----------
        Wu, X., & Foster, D. J. (2014). Hippocampal replay captures the
        unique topological structure of a novel environment. *Journal of
        Neuroscience*, 34(19), 6459-6469.
        """
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(
                f"well_decoded_fraction(): threshold must be a positive finite "
                f"multiple of chance, such as 5.0, got {threshold}."
            )
        if chance_bins is None:
            chance_bins = len(self.max_posterior)
        elif (
            isinstance(chance_bins, bool)
            or not isinstance(chance_bins, (int, np.integer))
            or chance_bins < 1
        ):
            raise ValueError(
                "well_decoded_fraction(): chance_bins must be a positive "
                f"number of bins, such as 2 * env.n_bins, got {chance_bins!r}."
            )
        covered = self.max_posterior > threshold / chance_bins
        return float(np.sum(self.space_fractions[covered]))

    def summary(self) -> dict[str, Any]:
        """Flat dict of the event's scalar metrics."""
        return {
            "metric": self.metric,
            "units": self.units,
            "n_time_bins": self.n_time_bins,
            "duration": self.duration,
            "track_length": self.track_length,
            "max_map_jump": self.max_map_jump,
            "max_mean_jump": self.max_mean_jump,
            "mean_mean_jump": self.mean_mean_jump,
            "map_range": self.map_range,
            "mean_range": self.mean_range,
            "start_to_end_map_distance": self.start_to_end_map_distance,
            "start_to_end_mean_distance": self.start_to_end_mean_distance,
            "map_path_length": self.map_path_length,
            "dispersion": self.dispersion,
            "mean_spread": self.mean_spread,
            "mean_peak_probability": self.mean_peak_probability,
            "well_decoded_fraction": self.well_decoded_fraction(),
            "empty_fraction": self.empty_fraction,
        }


def trajectory_metrics(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    spike_counts: NDArray[np.int64] | None = None,
    metric: Literal["geodesic", "euclidean"] | None = None,
) -> TrajectoryMetrics:
    """Every published per-event metric used to accept or reject a replay event.

    Criteria-based replay detection keeps a decoded candidate event when its
    posterior moves in small steps, stays sharp and covers enough of the
    environment. This computes each quantity the published criteria
    threshold, under its own name, so that a recipe can apply a paper's
    criteria exactly. Works on 1-D environments, linearized tracks and mazes,
    and 2-D or 3-D arenas.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability mass per bin in each time bin of the event.
        Each row sums to 1, or to less than 1 when it is one template's share
        of a joint decode, ``result.joint(name)``, as in the Mallory et al.
        (2025) and Bendor-lab code. Positions, jumps and spread use each row
        rescaled to sum to 1; ``peak_probability`` and
        ``well_decoded_fraction()`` use the mass as given.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds, increasing.
    spike_counts : NDArray[np.int64] or None, shape (n_time_bins, n_neurons), optional
        Spike counts the event was decoded from; needed for
        ``empty_time_bins`` and ``empty_fraction``.
    metric : {"geodesic", "euclidean"} or None, default=None
        Distance between bins. ``"geodesic"`` is the shortest path through
        the environment's bin graph, as :meth:`Environment.distance_between`
        measures it: on a straight track it is the distance along the track,
        and on a maze it goes through the junctions. ``"euclidean"`` is the
        straight-line distance between bin centers. None uses geodesic on a
        track or 1-D environment and Euclidean on a 2-D or 3-D arena, where
        paths along the bin graph are longer than straight lines (Pfeiffer &
        Foster 2013 measured straight-line distances in the arena).

    Returns
    -------
    TrajectoryMetrics
        Per-time-bin arrays and per-event scalars. ``summary()`` returns the
        scalars.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the options; if ``posterior`` does not
        have one column per bin, or a row has NaN, Inf or negative entries,
        no mass, or more than 1; if ``times`` does not have one finite,
        increasing value per row; or if ``spike_counts`` does not have one
        row of finite, non-negative counts per time bin.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.

    Notes
    -----
    Which papers use which metric:

    - ``map_jumps`` (distance between consecutive MAP positions): Pfeiffer
      & Foster (2013), jumps < 20 cm; Silva et al. (2015), the largest jump
      divided by ``track_length``; Igata et al. (2021), < 40 cm; the
      Mallory et al. (2025) code.
    - ``mean_jumps`` (distance between consecutive posterior means):
      Pfeiffer & Foster (2015), < 50 cm; Widloski & Foster (2022, 2025),
      "the L2 norm of the change in center of mass", < 20 cm.
    - ``start_to_end_map_distance``: Pfeiffer & Foster (2013), >= 40 cm;
      ``start_to_end_mean_distance``: Pfeiffer & Foster (2015), >= 80 cm.
    - ``map_path_length``: the "cumulative distance" of Igata et al. (2021),
      > 20 cm. It sums unsigned jumps. The Mallory code's ``cum_coverage``
      is the absolute value of the summed signed steps, the net displacement
      along the track, which ``start_to_end_map_distance`` gives on a
      straight track.
    - ``map_range``: the MAP range of the Mallory code; ``mean_range``: the
      center-of-mass range of Berners-Lee et al. (2021), who divide it by
      the arm length and compute it on each arm separately.
    - ``spread``: Widloski & Foster (2022, 2025), spread < 10 cm, described
      as the square root of the posterior's second moment. The formula
      printed in the paper, :math:`\\sum (x - \\bar x)^2 (y - \\bar y)^2 P`,
      has a square root in units of length squared, which a threshold in cm
      cannot apply to. The deposited ``compute_imageSpread.m`` takes the
      exponent as an argument, but the decoder that calls it on the
      posterior is not deposited, so the exponent used is unknown; the
      isotropic second central moment is used.
    - ``mean_spread`` and ``mean_mean_jump``: Widloski & Foster (2025)
      tested the mean posterior spread and the mean center-of-mass jump of
      each candidate against cell-identity shuffles (see
      ``ReplayMethod.tested_quantities``).
    - ``dispersion``: Widloski & Foster (2022) ``compute_sequenceDispersion.m``,
      > 12 cm, the root-mean-square distance of the posterior means from
      their average. The paper prints the mean of these distances instead.
    - ``mean_peak_probability``: the "decoding confidence" of Stella et al.
      (2019), mean over time bins of the largest posterior value, > 0.1.
      Here the peak is the mass in the MAP bin, which is the largest mass on
      equal bins. Within admitted events they also kept only the time bins
      whose own peak passes 0.1: ``peak_probability > 0.1``, on rows that
      sum to 1.
    - ``well_decoded_fraction()``: the coverage of Wu & Foster (2014), as in
      the Mallory code's ``coverage_wu``.
    - ``empty_fraction``: Harvey et al. (2023) discarded events with more
      than half of their time bins inactive.
    - ``duration``: Widloski & Foster (2022, 2025) required > 100 ms.

    The MAP bin is the bin of highest density, mass divided by bin size, as
    :func:`neurospatial.decoding.map_bin` returns; on equal bins it is the
    bin of highest mass. Geodesic distances between bins in parts of the
    environment that are not connected are infinite.

    Positions along the track, used by the ranges, run through the track's
    bins in order and skip the spacing between track segments, so
    ``edge_spacing`` does not add to a range; on a 1-D environment they are
    its coordinate. On a branched track (a maze with a junction), positions
    along the linearization jump from the end of one arm to the start of the
    next, so a range across arms has no meaning, and both ranges are NaN:
    restrict the posterior to one arm's environment to measure a range
    along it.

    References
    ----------
    Berners-Lee, A., Wu, X., & Foster, D. J. (2021). Prefrontal cortical
    neurons are selective for non-local hippocampal representations during
    replay and behavior. *Journal of Neuroscience*, 41(27), 5894-5908.

    Harvey, R. E., Robinson, H. L., Liu, C., Oliva, A., & Fernandez-Ruiz, A.
    (2023). Hippocampo-cortical circuits for selective memory encoding,
    routing, and replay. *Neuron*, 111(13), 2076-2090.e9.

    Igata, H., Ikegaya, Y., & Sasaki, T. (2021). Prioritized experience
    replays on a hippocampal predictive map for learning. *Proceedings of
    the National Academy of Sciences*, 118(1), e2011266118.

    Mallory, C. S., Widloski, J., & Foster, D. J. (2025). The time course and
    organization of hippocampal replay. *Science*, 387(6733), 541-548.

    Pfeiffer, B. E., & Foster, D. J. (2013). Hippocampal place-cell sequences
    depict future paths to remembered goals. *Nature*, 497(7447), 74-79.

    Pfeiffer, B. E., & Foster, D. J. (2015). Autoassociative dynamics in the
    generation of sequences of hippocampal place cells. *Science*,
    349(6244), 180-183.

    Silva, D., Feng, T., & Foster, D. J. (2015). Trajectory events across
    hippocampal place cells require previous experience. *Nature
    Neuroscience*, 18(12), 1772-1779.

    Stella, F., Baracskay, P., O'Neill, J., & Csicsvari, J. (2019).
    Hippocampal reactivation of random trajectories resembling Brownian
    diffusion. *Neuron*, 102(2), 450-461.

    Widloski, J., & Foster, D. J. (2022). Flexible rerouting of hippocampal
    replay sequences around changing barriers in the absence of global
    place field remapping. *Neuron*, 110(9), 1547-1558.

    Widloski, J., & Foster, D. J. (2025). Replay without sharp wave ripples
    in a spatial memory task. *Nature Communications*, 16.
    https://doi.org/10.1038/s41467-025-65181-5

    Wu, X., & Foster, D. J. (2014). Hippocampal replay captures the unique
    topological structure of a novel environment. *Journal of
    Neuroscience*, 34(19), 6459-6469.

    Examples
    --------
    A sweep that moves one 10 cm bin per 20 ms time bin along a 100 cm
    track:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import trajectory_metrics
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> posterior = np.eye(env.n_bins)[:6]  # bins 0 to 5
    >>> times = np.arange(6) * 0.02
    >>> metrics = trajectory_metrics(env, posterior, times)
    >>> metrics.map_jumps.round(6).tolist()
    [10.0, 10.0, 10.0, 10.0, 10.0]
    >>> round(metrics.max_map_jump / metrics.track_length, 6)  # Silva et al. 2015
    0.1
    >>> round(metrics.map_range, 6), round(metrics.map_path_length, 6)
    (50.0, 50.0)
    >>> round(metrics.duration, 3), metrics.mean_peak_probability
    (0.1, 1.0)
    >>> round(metrics.well_decoded_fraction(), 2)  # 6 of 10 bins
    0.6
    """
    return _trajectory_metrics(
        env,
        posterior,
        times,
        spike_counts=spike_counts,
        metric=metric,
        operation=_OPERATION,
    )


def _check_metric(metric: Any, operation: str) -> None:
    """Refuse a distance metric other than geodesic, Euclidean or None."""
    if metric is not None and metric not in _METRICS:
        raise ValueError(
            f"{operation}: metric must be one of {_METRICS} or None, got {metric!r}."
        )


def _trajectory_metrics(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    spike_counts: NDArray[np.int64] | None,
    metric: Literal["geodesic", "euclidean"] | None,
    operation: str,
) -> TrajectoryMetrics:
    """:func:`trajectory_metrics`, naming ``operation`` in its errors."""
    from neurospatial.decoding import map_bin, mean_position
    from neurospatial.environment import _track_extents
    from neurospatial.geometry import require_cartesian

    _check_metric(metric, operation)
    require_cartesian(
        env.coordinate_system,
        operation,
        why="replay is decoded over allocentric positions, and distances "
        "between (distance, angle) bins are not path lengths",
        fix="decode the event on the allocentric environment the animal ran in",
    )
    raw = _posterior_rows(posterior, env.n_bins, operation)
    times = _increasing_times(times, len(raw), operation)
    empty_time_bins = (
        None
        if spike_counts is None
        else _empty_time_bins(spike_counts, len(raw), operation)
    )
    extents = _track_extents(env)
    if metric is None:
        metric = "euclidean" if extents is None else "geodesic"

    posterior = raw / raw.sum(axis=1, keepdims=True)
    bins = map_bin(env, posterior)
    rows = np.arange(len(raw))
    centers = np.asarray(env.bin_centers, dtype=np.float64)
    means = mean_position(env, posterior)
    between, rows_of = _bin_distance_lookup(env, bins, metric)
    map_jumps = between[rows_of[:-1], rows_of[1:]]
    mean_jumps = np.linalg.norm(np.diff(means, axis=0), axis=1)
    squared = np.sum((centers[None, :, :] - means[:, None, :]) ** 2, axis=2)
    spread = np.sqrt(np.sum(posterior * squared, axis=1))

    track_length = map_range = mean_range = np.nan
    if extents is not None:
        starts, stops = extents
        track_length = float(np.sum(stops - starts))
        along = _along_track(env, starts, stops)
        if along is not None:
            map_range = float(np.ptp(along[bins]))
            mean_range = float(np.ptp(posterior @ along))

    sizes = np.asarray(env.bin_sizes, dtype=np.float64)
    deviations = means - means.mean(axis=0)
    return TrajectoryMetrics(
        map_bins=bins,
        map_positions=centers[bins],
        peak_probability=raw[rows, bins],
        mean_positions=means,
        map_jumps=map_jumps,
        mean_jumps=mean_jumps,
        spread=spread,
        empty_time_bins=empty_time_bins,
        max_posterior=raw.max(axis=0),
        space_fractions=sizes / sizes.sum(),
        metric=metric,
        units=env.units,
        n_time_bins=len(raw),
        duration=float(times[-1] - times[0]),
        track_length=track_length,
        max_map_jump=_max_or_nan(map_jumps),
        max_mean_jump=_max_or_nan(mean_jumps),
        mean_mean_jump=float(np.mean(mean_jumps)) if len(mean_jumps) else np.nan,
        map_range=map_range,
        mean_range=mean_range,
        start_to_end_map_distance=float(between[rows_of[0], rows_of[-1]]),
        start_to_end_mean_distance=float(np.linalg.norm(means[-1] - means[0])),
        map_path_length=float(np.sum(map_jumps)),
        dispersion=float(np.sqrt(np.mean(np.sum(deviations**2, axis=1)))),
        mean_spread=float(np.mean(spread)),
        mean_peak_probability=float(np.mean(raw[rows, bins])),
        empty_fraction=(
            np.nan if empty_time_bins is None else float(np.mean(empty_time_bins))
        ),
    )


def _posterior_rows(posterior: Any, n_bins: int, operation: str) -> NDArray[np.float64]:
    """Check rows of finite, non-negative mass in (0, 1]."""
    from neurospatial.decoding import _ROW_SUM_TOLERANCE

    values = np.asarray(posterior, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != n_bins or len(values) == 0:
        raise ValueError(
            _format_error(
                f"{operation}: posterior must have shape (n_time_bins, "
                f"{n_bins}) with at least one time bin, one column per "
                f"environment bin, got {values.shape}.",
                fix="pass the posterior decoded on this environment",
            )
        )
    finite = np.isfinite(values).all(axis=1)
    negative = finite & (values < 0).any(axis=1)
    sums = values.sum(axis=1)
    empty = finite & ~negative & (sums <= 0)
    excess = finite & ~negative & (sums > 1.0 + _ROW_SUM_TOLERANCE)
    bad = ~finite | negative | empty | excess
    if not np.any(bad):
        return values
    problems = []
    if np.any(~finite):
        problems.append(f"{int(np.sum(~finite))} contain NaN or Inf")
    if np.any(negative):
        problems.append(f"{int(np.sum(negative))} have negative entries")
    if np.any(empty):
        problems.append(f"{int(np.sum(empty))} have no mass")
    if np.any(excess):
        problems.append(
            f"{int(np.sum(excess))} sum to more than 1 (largest "
            f"{float(np.max(sums[excess])):.6g})"
        )
    raise ValueError(
        _format_error(
            f"{operation} needs each posterior row to be a probability mass "
            f"over the bins, or one template's share of one, but "
            f"{int(np.sum(bad))} of {len(values)} rows are not: "
            f"{'; '.join(problems)}.",
            why="a row without mass, or with NaN, has no decoded position, and "
            "a row summing to more than 1 is not a probability",
            fix="drop undecodable time bins together with their times, "
            "keep = np.isfinite(posterior).all(axis=1) & (posterior.sum(axis=1) "
            "> 0); posterior, times = posterior[keep], times[keep]",
        )
    )


def _along_track(
    env: Environment, starts: NDArray[np.float64], stops: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Each bin's position along an unbranched track, in bin numbering.

    On a linearized track, positions run through the bins in track order and
    skip the spacing between segments; on a 1-D environment they are its
    coordinate. None on a branched track, where positions along the
    linearization jump between arms.
    """
    linearization = env.binning.linearization
    if linearization is None:
        along: NDArray[np.float64] = (starts + stops) / 2
        return along
    if any(degree > 2 for _, degree in linearization.graph.degree):
        return None
    order = np.argsort(starts, kind="stable")
    lengths = (stops - starts)[order]
    along = np.empty_like(starts)
    along[order] = np.cumsum(lengths) - lengths / 2
    return along


def _increasing_times(
    times: Any, n_time_bins: int, operation: str
) -> NDArray[np.float64]:
    """Check one finite time per posterior row, strictly increasing."""
    values: NDArray[np.float64] = np.asarray(times, dtype=np.float64)
    if values.shape != (n_time_bins,) or not np.isfinite(values).all():
        raise ValueError(
            _format_error(
                f"{operation}: times must hold one finite time per posterior "
                f"row ({n_time_bins}), got shape {values.shape}"
                + ("" if values.shape != (n_time_bins,) else " with NaN or Inf")
                + ".",
                why="jumps and the duration compare consecutive time bins",
                fix="pass the center time of each posterior row, in seconds; "
                "drop the times of any rows you dropped",
            )
        )
    if np.any(np.diff(values) <= 0):
        raise ValueError(
            _format_error(
                f"{operation}: times must increase from one posterior row to the next.",
                why="a jump is the step from one time bin to the next, so the "
                "rows must be in time order",
                fix="sort the rows: order = np.argsort(times); posterior, "
                "times = posterior[order], times[order]",
            )
        )
    return values


def _empty_time_bins(
    spike_counts: Any, n_time_bins: int, operation: str
) -> NDArray[np.bool_]:
    """True for each time bin without spikes."""
    counts = np.asarray(spike_counts, dtype=np.float64)
    if counts.ndim != 2 or len(counts) != n_time_bins:
        raise ValueError(
            _format_error(
                f"{operation}: spike_counts must have shape (n_time_bins, "
                f"n_neurons) with {n_time_bins} rows, got {counts.shape}.",
                fix="pass the spike counts the event was decoded from, one row "
                "per posterior row",
            )
        )
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError(f"{operation}: spike_counts must be finite and non-negative.")
    empty: NDArray[np.bool_] = counts.sum(axis=1) == 0
    return empty


def _bin_distance_lookup(
    env: Environment, bins: NDArray[np.int64], metric: str
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Distances between the distinct ``bins``, and each entry's row in them.

    The distance between ``bins[i]`` and ``bins[j]`` is
    ``between[rows_of[i], rows_of[j]]``. Only distinct bins are paired, so
    a long event costs no more than its distinct MAP bins.
    """
    from neurospatial.ops import pairwise_distances

    unique, rows_of = np.unique(bins, return_inverse=True)
    if metric == "geodesic":
        between = pairwise_distances(env.connectivity, unique)
    else:
        points = env.bin_centers[unique]
        between = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return between, rows_of


def _max_or_nan(values: NDArray[np.float64]) -> float:
    """Largest value, or NaN for an empty array."""
    return float(np.max(values)) if len(values) else np.nan


def template_fractions(
    result: JointDecodingResult, *, exclude_end_distance: float = 0.0
) -> dict[str, float]:
    """Fraction of a jointly decoded event's posterior in each template.

    For an event decoded jointly over several templates (the two running
    directions on a track, or two tracks) with
    :func:`neurospatial.decoding.decode_jointly`, the fraction of template
    :math:`k` is its share of all posterior mass over the event (Ambrose et
    al. 2016, eq. 1):

    .. math::

        f_k = \\frac{\\sum_t \\sum_x P(k, x \\mid n_t)}
        {\\sum_t \\sum_{k'} \\sum_x P(k', x \\mid n_t)}.

    Without masking, this is the mean over time bins of the template
    probability.

    Parameters
    ----------
    result : JointDecodingResult
        The joint decode of one event.
    exclude_end_distance : float, default=0.0
        Leave out bins whose center lies closer than this to either end of
        their template's track, in environment units, before summing. The
        Mallory et al. (2025) code leaves out 10 bins (20 cm) at each end.
        Needs every template on a track environment when positive.

    Returns
    -------
    dict[str, float]
        Fraction of the posterior in each template, keyed by template name,
        in the order of ``result.template_names``; the fractions sum to 1.

    Raises
    ------
    TypeError
        If ``result`` is not a :class:`~neurospatial.decoding.JointDecodingResult`.
    ValueError
        If ``exclude_end_distance`` is negative or not finite, if it is
        positive and a template's environment is not a track, or if no
        posterior mass is left after leaving out the ends.

    Notes
    -----
    Ambrose et al. (2016) called an event directional when one direction
    held at least 66.5% of the posterior. Mallory et al. (2025) required
    ``abs(fraction_left - fraction_right) > 0.33`` with two templates,
    which is the same as the larger fraction exceeding 0.665.

    The ends of a track are the start and end of its linearization: on a
    straight track, its two ends. On a branched maze, the ends of the other
    arms are not left out.

    References
    ----------
    Ambrose, R. E., Pfeiffer, B. E., & Foster, D. J. (2016). Reverse replay of
    hippocampal place cells is uniquely modulated by changing reward.
    *Neuron*, 91(5), 1124-1136.

    Mallory, C. S., Widloski, J., & Foster, D. J. (2025). The time course and
    organization of hippocampal replay. *Science*, 387(6733), 541-548.

    Examples
    --------
    A sweep drawn from the outbound rate maps, with the inbound maps holding
    the same fields in other cells:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import decode_jointly
    >>> from neurospatial.replay import template_fractions
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> centers = np.linspace(5.0, 95.0, 8)
    >>> outbound = 1.0 + 20.0 * np.exp(-0.5 * ((x - centers[:, None]) / 8.0) ** 2)
    >>> inbound = outbound[[3, 6, 1, 4, 7, 2, 5, 0]]
    >>> rng = np.random.default_rng(0)
    >>> counts = rng.poisson(outbound[:, 2:7].T * 0.1)  # bins 2 to 6
    >>> result = decode_jointly(
    ...     env, counts, {"outbound": outbound, "inbound": inbound}, dt=0.1
    ... )
    >>> fractions = template_fractions(result, exclude_end_distance=10.0)
    >>> {name: round(value, 2) for name, value in fractions.items()}
    {'outbound': 0.78, 'inbound': 0.22}
    >>> fractions["outbound"] > 0.665  # directional (Ambrose et al. 2016)
    True
    """
    from neurospatial.decoding import JointDecodingResult
    from neurospatial.environment import _require_track_extents

    operation = "template_fractions()"
    if not isinstance(result, JointDecodingResult):
        raise TypeError(
            _format_error(
                f"{operation} needs a JointDecodingResult, got "
                f"{type(result).__name__}.",
                why="the fraction compares templates decoded together, whose "
                "posteriors share one normalization",
                fix="decode the event with neurospatial.decoding.decode_jointly",
            )
        )
    distance = float(exclude_end_distance)
    if not np.isfinite(distance) or distance < 0:
        raise ValueError(
            f"{operation}: exclude_end_distance must be a finite distance >= 0 "
            f"in environment units, got {exclude_end_distance}."
        )
    masses = []
    for name in result.template_names:
        joint = result.joint(name)
        if distance > 0:
            starts, stops = _require_track_extents(
                result.conditional[name].env,
                f"{operation} with exclude_end_distance > 0",
                alternative="or pass exclude_end_distance=0.0 to keep every bin",
            )
            along = (starts + stops) / 2
            near_end = (along - starts.min() < distance) | (
                stops.max() - along < distance
            )
            joint = joint[:, ~near_end]
        masses.append(float(np.sum(joint)))
    total = sum(masses)
    if total <= 0:
        raise ValueError(
            _format_error(
                f"{operation}: no posterior mass is left after leaving out "
                f"{distance} at each end of the tracks.",
                fix="use a smaller exclude_end_distance",
            )
        )
    return {
        name: mass / total
        for name, mass in zip(result.template_names, masses, strict=True)
    }
