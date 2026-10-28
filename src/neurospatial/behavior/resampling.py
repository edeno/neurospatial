"""Resample a tracked path at equal distances along its observed arclength.

Equal-distance samples let a scientist compare routes independently of running
speed and camera frame rate. Each output sample records the two original
samples that bracket it and the interpolation weight, so time, speed or spikes
can be recovered on the original clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import DEFAULT_MAX_GAP, as_intervals
from neurospatial._observation import validated_observations
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial.behavior.trajectory import _observed_cumulative

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["ResampledPathResult", "resample_by_distance"]

_SNAP_TOLERANCE = 1e-9  # fraction of ``spacing``


@dataclass(frozen=True, repr=False, eq=False)
class ResampledPathResult(ReadOnlyArraysMixin, ResultMixin):
    """Equal-distance samples of a path, returned by ``resample_by_distance``.

    Output sample ``k`` lies between original samples ``source_left[k]`` and
    ``source_right[k]``: its position is
    ``(1 - weights[k]) * positions[source_left[k]] + weights[k] * positions[source_right[k]]``
    and its time is the same combination of the original times.

    Attributes
    ----------
    times : NDArray[np.float64], shape (n_out,)
        Time of each sample on the original clock, in seconds. Strictly
        increasing.
    positions : NDArray[np.float64], shape (n_out, n_dims)
        Position of each sample, in the positions' units.
    run_distance : NDArray[np.float64], shape (n_out,)
        Distance since the start of the sample's run: a multiple of
        ``spacing``, or the run's full length for an off-grid endpoint.
    cumulative_distance : NDArray[np.float64], shape (n_out,)
        Path length of all earlier runs plus ``run_distance``.
    run_ids : NDArray[np.intp], shape (n_out,)
        Run of each sample, ``0..n_runs - 1`` in time order.
    source_left : NDArray[np.intp], shape (n_out,)
        Index of the original sample at or before each output sample.
    source_right : NDArray[np.intp], shape (n_out,)
        Equal to ``source_left`` when the sample falls exactly on an original
        sample, otherwise the next valid original sample after
        ``source_left`` (missing samples between them are skipped, as if
        deleted).
    weights : NDArray[np.float64], shape (n_out,)
        Interpolation weight toward ``source_right``, in ``[0, 1)``; 0 for
        exact samples.
    off_grid_endpoint : NDArray[np.bool_], shape (n_out,)
        True for the run-end sample added by ``include_endpoint=True`` when
        the run length is not a multiple of ``spacing``.
    run_sample_bounds : NDArray[np.intp], shape (n_runs, 2)
        Inclusive ``(first, last)`` original sample indices of each run.
    run_time_bounds : NDArray[np.float64], shape (n_runs, 2)
        Observed start and end time of each run, in seconds.
    run_path_lengths : NDArray[np.float64], shape (n_runs,)
        Observed path length of each run.
    n_source_samples : int
        Number of original samples.
    spacing : float
        Distance between consecutive samples within a run.
    include_endpoint : bool
        Whether off-grid run ends were added.
    max_gap : float or None
        The gap limit that was applied, in seconds.
    epochs : NDArray[np.float64] or None, shape (n_epochs, 2)
        Normalized half-open ``[start, stop)`` windows that were applied.
    """

    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    run_distance: NDArray[np.float64]
    cumulative_distance: NDArray[np.float64]
    run_ids: NDArray[np.intp]
    source_left: NDArray[np.intp]
    source_right: NDArray[np.intp]
    weights: NDArray[np.float64]
    off_grid_endpoint: NDArray[np.bool_]
    run_sample_bounds: NDArray[np.intp]
    run_time_bounds: NDArray[np.float64]
    run_path_lengths: NDArray[np.float64]
    n_source_samples: int
    spacing: float
    include_endpoint: bool
    max_gap: float | None
    epochs: NDArray[np.float64] | None

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    @property
    def break_before(self) -> NDArray[np.bool_]:
        """Run starts, shape ``(n_out,)``.

        True at the first sample of each run. Pass it with ``max_gap=None`` to
        time-and-position functions such as ``compute_planar_curvature`` so
        that a resampled path is not bridged across its breaks.
        """
        starts = np.ones(self.run_ids.shape, dtype=bool)
        starts[1:] = self.run_ids[1:] != self.run_ids[:-1]
        return starts

    def interpolate_signal(self, values: ArrayLike) -> NDArray[np.float64]:
        """Carry a continuous source signal onto the resampled samples.

        Computes ``(1 - weights) * values[source_left] + weights *
        values[source_right]``. Samples that fall exactly on an original sample
        return that sample's value. A NaN in a bracketing source value gives
        NaN: missing stays missing.

        Use this for linear quantities only (speed, original time, position,
        LFP amplitude). For angles, interpolate ``np.cos(angle)`` and
        ``np.sin(angle)`` separately and recombine with ``np.arctan2``. Labels
        and counts are not interpolated; index them with ``source_left``.

        Parameters
        ----------
        values : array-like, shape (n_source_samples, ...)
            Floating-point signal on the original samples (the ``times`` and
            ``positions`` passed to ``resample_by_distance``).

        Returns
        -------
        NDArray[np.float64], shape (n_out, ...)
            The signal at each resampled sample.

        Raises
        ------
        ValueError
            If the first axis of ``values`` is not ``n_source_samples`` long.
        TypeError
            If ``values`` is not floating point.
        """
        arr = np.asarray(values)
        if arr.ndim == 0 or arr.shape[0] != self.n_source_samples:
            n = "a scalar" if arr.ndim == 0 else f"{arr.shape[0]} rows"
            raise ValueError(
                _format_error(
                    "values do not match the resampled path's source samples",
                    why=(
                        f"values has {n} but the path was resampled from "
                        f"{self.n_source_samples} samples"
                    ),
                    fix="pass one value per original sample, with the first axis "
                    f"of length {self.n_source_samples}",
                )
            )
        if not np.issubdtype(arr.dtype, np.floating):
            raise TypeError(
                _format_error(
                    f"cannot interpolate values of dtype {arr.dtype}",
                    why="only floating-point signals have a meaningful value "
                    "between two samples",
                    fix="labels and counts are not interpolated; take "
                    "values[result.source_left] for the label at or before "
                    "each sample",
                )
            )
        w = self.weights.reshape((-1,) + (1,) * (arr.ndim - 1))
        result: NDArray[np.float64] = (1 - w) * arr[self.source_left] + w * arr[
            self.source_right
        ]
        return result

    def samples_table(self) -> pd.DataFrame:
        """One row per resampled sample.

        Returns
        -------
        pandas.DataFrame
            Columns ``run_id``, ``time``, ``run_distance``,
            ``cumulative_distance``, the coordinates (``x``, ``y``, ``z`` for
            up to three dimensions, otherwise ``position_0``, ``position_1``,
            ...), ``source_left``, ``source_right``, ``weight`` and
            ``off_grid_endpoint``. ``df.attrs`` holds ``spacing``,
            ``include_endpoint``, ``max_gap`` and ``epochs`` (a copy of the
            applied windows, or None).
        """
        import pandas as pd

        n_dims = self.positions.shape[1]
        names = (
            ["x", "y", "z"][:n_dims]
            if n_dims <= 3
            else [f"position_{i}" for i in range(n_dims)]
        )
        data: dict[str, Any] = {
            "run_id": self.run_ids,
            "time": self.times,
            "run_distance": self.run_distance,
            "cumulative_distance": self.cumulative_distance,
        }
        for i, name in enumerate(names):
            data[name] = self.positions[:, i]
        data["source_left"] = self.source_left
        data["source_right"] = self.source_right
        data["weight"] = self.weights
        data["off_grid_endpoint"] = self.off_grid_endpoint
        df = pd.DataFrame(data)
        df.attrs.update(
            spacing=self.spacing,
            include_endpoint=self.include_endpoint,
            max_gap=self.max_gap,
            epochs=None if self.epochs is None else self.epochs.copy(),
        )
        return df

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result.

        ``n_resampled_samples`` counts the output samples and
        ``n_source_samples`` the input samples they were resampled from.
        """
        lengths = self.run_path_lengths
        return {
            "n_resampled_samples": int(self.times.size),
            "n_source_samples": int(self.n_source_samples),
            "n_runs": int(lengths.size),
            "spacing": float(self.spacing),
            "total_distance": float(lengths.sum()) if lengths.size else float("nan"),
            "n_off_grid_endpoints": int(np.count_nonzero(self.off_grid_endpoint)),
            "n_stationary_runs": int(np.count_nonzero(lengths == 0)),
        }


def resample_by_distance(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    spacing: float,
    include_endpoint: bool = False,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> ResampledPathResult:
    """
    Resample a path every ``spacing`` along its observed distance.

    Samples are placed at run distances ``0, spacing, 2 * spacing, ...`` along
    each run of continuous observation. Comparing routes on this distance axis
    removes differences in running speed and frame rate; the original time of
    each sample and the two original samples around it (``source_left``,
    ``source_right``, ``weights``) let you recover time, speed or spikes at
    each sample.

    Missing samples (non-finite coordinates) are skipped as if their rows
    were deleted. Runs are separated wherever the gap between consecutive
    valid samples is not usable: longer than ``max_gap``, outside ``epochs``,
    or across a declared break. Each run is resampled on its own, starting
    from 0 at its first sample; nothing is placed or interpolated across a
    break, and isolated samples produce no output.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples,) or (n_samples, n_dims)
        Positions in physical units. A 1-D array is one coordinate (for example
        a linearized track). Rows with a non-finite coordinate are missing.
    spacing : float
        Distance between consecutive samples, in the positions' units. No
        default: choose the distance scale of your analysis.
    include_endpoint : bool, default=False
        Also keep the last sample of each run whose length is not a multiple
        of ``spacing``, marked in ``off_grid_endpoint``.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : None, (start, stop) or array-like of shape (n_epochs, 2), optional
        Half-open ``[start, stop)`` windows in seconds, on the clock of
        ``times``. An interval must lie inside one window.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    ResampledPathResult
        Times, positions and run distances of the new samples with their
        source brackets. All arrays are empty when no interval was usable.

    Raises
    ------
    ValueError
        If ``spacing`` is not a positive finite number, ``times`` and
        ``positions`` are empty, swapped or misaligned, ``times`` is not
        strictly increasing, ``max_gap`` is not positive and finite or
        ``None``, or ``break_before`` is not a Boolean array with one entry per
        sample.

    Notes
    -----
    Distance is Euclidean arclength along the observed polyline over all
    coordinate columns, in the positions' units. It is not geodesic distance
    through an environment.

    A target that coincides with an original sample is taken from that sample
    exactly (``weights == 0``). When the animal stays put, several samples
    share one distance; the target there is the **last** of them, so its time
    is when the animal leaves. The first sample of a run is always the run
    start itself, and a run that never moves gives that one sample.

    By default each run ends at its last whole multiple of ``spacing``.
    ``include_endpoint=True`` adds the run's last sample when the remainder is
    longer than the snapping tolerance; an end that is already on the grid is
    never repeated.

    Cumulative sums of step lengths carry floating-point error (``0.1 + 0.2``
    is ``0.30000000000000004``). Distances within ``1e-9 * spacing`` of a
    target therefore count as on the target.

    See Also
    --------
    compute_cumulative_distance : Observed distance at every original sample.

    Examples
    --------
    The animal pauses at x = 1 from 1 s to 10 s. The sample at distance 1 is
    the departure time, 10 s, and later samples are interpolated after it.

    >>> import numpy as np
    >>> from neurospatial.behavior import resample_by_distance
    >>> times = np.array([0.0, 1.0, 10.0, 11.0])
    >>> positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    >>> result = resample_by_distance(times, positions, spacing=0.5, max_gap=None)
    >>> result.run_distance.tolist()
    [0.0, 0.5, 1.0, 1.5, 2.0]
    >>> result.times.tolist()
    [0.0, 0.5, 10.0, 10.5, 11.0]
    >>> result.source_left.tolist(), result.source_right.tolist()
    ([0, 0, 2, 2, 3], [0, 1, 2, 3, 3])
    >>> result.weights.tolist()
    [0.0, 0.5, 0.0, 0.5, 0.0]
    """
    try:
        valid_spacing = bool(np.isfinite(spacing) and spacing > 0)
    except TypeError:  # None, strings and other non-numbers
        valid_spacing = False
    if not valid_spacing:
        raise ValueError(
            _format_error(
                f"resample_by_distance() needs spacing to be a positive finite "
                f"number; got {spacing!r}.",
                why="spacing is the distance between consecutive output samples.",
                fix="pass spacing in the positions' units, e.g. spacing=2.0 for 2 cm",
            )
        )
    spacing = float(spacing)
    times_arr, positions_arr, support = validated_observations(
        times,
        positions,
        call="resample_by_distance",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no path was resampled",
    )
    _, cumulative = _observed_cumulative(positions_arr, support)
    tol = _SNAP_TOLERANCE * spacing

    distances = [np.empty(0)]
    lefts = [np.empty(0, dtype=np.intp)]
    rights = [np.empty(0, dtype=np.intp)]
    weight_parts = [np.empty(0)]
    ends = [np.empty(0, dtype=bool)]
    run_ids = [np.empty(0, dtype=np.intp)]
    # Missing samples inside a run are skipped, as if their rows were deleted.
    for run, samples in enumerate(support.run_samples()):
        run_cumulative = cumulative[samples] - cumulative[samples[0]]
        run_end = run_cumulative[-1]
        n_grid = int(np.floor((run_end + tol) / spacing))
        targets = np.minimum(spacing * np.arange(n_grid + 1), run_end)
        left = np.searchsorted(run_cumulative, targets + tol, side="right") - 1
        left[0] = 0
        exact = run_cumulative[left] >= targets - tol
        right = np.where(exact, left, left + 1)
        span = np.where(exact, 1.0, run_cumulative[right] - run_cumulative[left])
        weights = np.where(exact, 0.0, (targets - run_cumulative[left]) / span)
        end = np.zeros(targets.size, dtype=bool)
        if include_endpoint and run_end - spacing * n_grid > tol:
            targets = np.r_[targets, run_end]
            left = np.r_[left, samples.size - 1]
            right = np.r_[right, samples.size - 1]
            weights = np.r_[weights, 0.0]
            end = np.r_[end, True]
        distances.append(targets)
        lefts.append(samples[left])
        rights.append(samples[right])
        weight_parts.append(weights)
        ends.append(end)
        run_ids.append(np.full(targets.size, run, dtype=np.intp))

    run_distance = np.concatenate(distances)
    source_left = np.concatenate(lefts)
    source_right = np.concatenate(rights)
    weights = np.concatenate(weight_parts)
    out_run_ids = np.concatenate(run_ids)
    bounds = support.run_sample_bounds
    return ResampledPathResult(
        times=(1 - weights) * times_arr[source_left]
        + weights * times_arr[source_right],
        positions=(1 - weights)[:, None] * positions_arr[source_left]
        + weights[:, None] * positions_arr[source_right],
        run_distance=run_distance,
        cumulative_distance=cumulative[bounds[out_run_ids, 0]] + run_distance,
        run_ids=out_run_ids,
        source_left=source_left,
        source_right=source_right,
        weights=weights,
        off_grid_endpoint=np.concatenate(ends),
        run_sample_bounds=bounds,
        run_time_bounds=support.run_time_bounds,
        run_path_lengths=cumulative[bounds[:, 1]] - cumulative[bounds[:, 0]],
        n_source_samples=times_arr.size,
        spacing=spacing,
        include_endpoint=include_endpoint,
        max_gap=max_gap,
        epochs=as_intervals(epochs, name="epochs"),
    )
