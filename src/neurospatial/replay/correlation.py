"""Weighted correlation between time and decoded position.

The weighted correlation of a decoded event (Wu & Foster 2014) is the Pearson
correlation between time and position over every (time bin, position bin)
pair of the posterior, each pair weighted by its posterior probability. A
posterior that sweeps along the track gives a correlation near +1 or -1; a
stationary or scattered posterior gives a correlation near 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ResultMixin
from neurospatial.replay.trajectory import _SCORE_DECIMALS

if TYPE_CHECKING:
    from neurospatial.environment import Environment

_OPERATION = "weighted_correlation()"
_ROW_NORMALIZATIONS = ("per_time_bin", "none")


@dataclass(frozen=True, repr=False, eq=False)
class WeightedCorrelationResult(ResultMixin):
    """Weighted correlation between time and decoded position in one event.

    Attributes
    ----------
    score : float
        The weighted correlation, between -1 and 1. Positive when the
        decoded position moves toward larger track positions over time,
        negative when it moves toward smaller ones. NaN when the weighted
        positions or times do not vary (all mass at one position or in one
        time bin).
    row_normalization : {"per_time_bin", "none"}
        How the posterior rows were weighted.

    See Also
    --------
    weighted_correlation : Function that produces this result.
    """

    score: float
    row_normalization: str

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {"score": self.score, "row_normalization": self.row_normalization}


def weighted_correlation(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    row_normalization: Literal["per_time_bin", "none"] = "per_time_bin",
) -> WeightedCorrelationResult:
    """Weighted correlation between time and decoded position in one event.

    Pearson correlation of time and position over every (time bin, position
    bin) pair, each pair weighted by its posterior probability (Wu & Foster
    2014; Silva et al. 2015; Grosmark & Buzsáki 2016).

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track. Positions are bin centers along the track.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Non-negative posterior mass per bin for each time bin of the event.
        Rows may sum to less than 1, as in one template's share of a joint
        decode (:meth:`~neurospatial.decoding.JointDecodingResult.joint`).
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds.
    row_normalization : {"per_time_bin", "none"}, default="per_time_bin"
        ``"per_time_bin"`` scales each row to sum to 1, so every time bin
        weighs the same (Grosmark & Buzsáki 2016); rows with no mass are left
        out. ``"none"`` uses the rows as given, so time bins with more mass
        weigh more. The Bendor-lab analyses (Tirole et al. 2022) use
        ``"none"`` on a posterior normalized across tracks, with time bins
        without spikes set to zero.

    Returns
    -------
    WeightedCorrelationResult
        The weighted correlation, signed, rounded to 12 decimals so that
        shuffles tied with the event compare equal.

    Raises
    ------
    ValueError
        If ``env`` is neither 1-D nor a linearized track, if ``posterior`` is
        not 2-D with one column per bin, has a NaN, Inf or negative entry, or
        no mass, if ``times`` does not have one finite value per row, or if
        ``row_normalization`` is not one of the options.

    Notes
    -----
    With weights :math:`w_{ij}` (posterior mass at time :math:`t_i` and
    position :math:`x_j`, divided by their total),

    .. math::

        r = \\frac{\\sum_{ij} w_{ij} (t_i - \\bar t)(x_j - \\bar x)}
        {\\sqrt{\\sum_{ij} w_{ij} (t_i - \\bar t)^2
        \\sum_{ij} w_{ij} (x_j - \\bar x)^2}},

    where :math:`\\bar t` and :math:`\\bar x` are the weighted means. The
    papers use the time-bin index for :math:`t`; the correlation does not
    change under a linear change of time axis, so with evenly spaced time
    bins the two agree. Unevenly spaced bins are placed at their times.

    The papers do not say how time bins without spikes are handled. Decide
    before scoring: drop those rows with their times, or set them to zero
    with ``row_normalization="none"`` as the Bendor-lab code does.

    References
    ----------
    Wu, X., & Foster, D. J. (2014). Hippocampal replay captures the unique
    topological structure of a novel environment. *Journal of Neuroscience*,
    34(19), 6459-6469.

    Silva, D., Feng, T., & Foster, D. J. (2015). Trajectory events across
    hippocampal place cells require previous experience. *Nature
    Neuroscience*, 18(12), 1772-1779.

    Grosmark, A. D., & Buzsáki, G. (2016). Diversity in neural firing
    dynamics supports both rigid and learned hippocampal sequences.
    *Science*, 351(6280), 1440-1443.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import weighted_correlation
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> times = np.arange(env.n_bins) * 0.02
    >>> forward = np.eye(env.n_bins)  # one bin further along each time bin
    >>> round(weighted_correlation(env, forward, times).score, 3)
    1.0
    >>> round(weighted_correlation(env, forward[::-1], times).score, 3)
    -1.0
    """
    positions = _track_bin_centers(env, _OPERATION)
    weights = _validated_weights(posterior, env.n_bins, row_normalization)
    times = _validated_times(times, len(weights))
    return WeightedCorrelationResult(
        score=float(_weighted_correlations(weights[None], times, positions)[0]),
        row_normalization=row_normalization,
    )


def _track_bin_centers(env: Environment, operation: str) -> NDArray[np.float64]:
    """Each bin's center along the track, in bin numbering."""
    from neurospatial.environment import _require_track_extents
    from neurospatial.replay.trajectory import _TRACK_ALTERNATIVE

    starts, stops = _require_track_extents(
        env, operation, alternative=_TRACK_ALTERNATIVE
    )
    return (np.asarray(starts) + np.asarray(stops)) / 2.0


def _validated_weights(
    posterior: Any, n_bins: int, row_normalization: str
) -> NDArray[np.float64]:
    """Check the posterior and apply the row normalization."""
    if row_normalization not in _ROW_NORMALIZATIONS:
        raise ValueError(
            f"{_OPERATION}: row_normalization must be one of "
            f"{_ROW_NORMALIZATIONS}, got {row_normalization!r}."
        )
    weights = np.asarray(posterior, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != n_bins:
        raise ValueError(
            f"{_OPERATION}: posterior must have shape (n_time_bins, {n_bins}), "
            f"one column per environment bin, got {weights.shape}."
        )
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(
            _format_error(
                f"{_OPERATION} needs finite, non-negative posterior mass, but "
                "the posterior has NaN, Inf or negative entries.",
                why="each entry weighs one (time, position) pair",
                fix="drop undecodable time bins together with their times, "
                "keep = np.isfinite(posterior).all(axis=1); posterior, times = "
                "posterior[keep], times[keep]",
            )
        )
    row_mass = weights.sum(axis=1)
    if not np.any(row_mass > 0):
        raise ValueError(f"{_OPERATION}: the posterior has no mass.")
    if row_normalization == "per_time_bin":
        with np.errstate(invalid="ignore", divide="ignore"):
            weights = np.where(row_mass[:, None] > 0, weights / row_mass[:, None], 0.0)
    return weights


def _validated_times(times: Any, n_time: int) -> NDArray[np.float64]:
    """Check one finite time per posterior row."""
    values: NDArray[np.float64] = np.asarray(times, dtype=np.float64)
    if values.shape != (n_time,) or not np.isfinite(values).all():
        raise ValueError(
            f"{_OPERATION}: times must hold one finite time per posterior row "
            f"({n_time}), got shape {values.shape}."
        )
    return values


def _weighted_correlations(
    weights: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Weighted time-position correlation of each event in a batch.

    Parameters
    ----------
    weights : NDArray[np.float64], shape (n_events, n_time_bins, n_bins)
        Non-negative weights, already row-normalized if wanted.
    times : NDArray[np.float64], shape (n_time_bins,)
    positions : NDArray[np.float64], shape (n_bins,)

    Returns
    -------
    NDArray[np.float64], shape (n_events,)
        The correlation of each event; NaN where time or position does not
        vary under the weights.
    """
    total = weights.sum(axis=(1, 2))
    time_mass = weights.sum(axis=2) / total[:, None]  # (n_events, n_time)
    position_mass = weights.sum(axis=1) / total[:, None]  # (n_events, n_bins)
    mean_time = time_mass @ times
    mean_position = position_mass @ positions
    time_dev = times[None, :] - mean_time[:, None]
    position_dev = positions[None, :] - mean_position[:, None]
    covariance = np.einsum("et,etb,eb->e", time_dev, weights, position_dev) / total
    var_time = np.einsum("et,et->e", time_mass, time_dev**2)
    var_position = np.einsum("eb,eb->e", position_mass, position_dev**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = covariance / np.sqrt(var_time * var_position)
    # Round so that scores equal in exact arithmetic compare equal: a shuffle
    # tied with the event then counts as a tie in the Monte Carlo p-value.
    rounded = np.clip(np.round(correlation, _SCORE_DECIMALS), -1.0, 1.0)
    return np.where((var_time > 0) & (var_position > 0), rounded, np.nan)
