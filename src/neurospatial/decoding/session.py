"""One-call encode->bin->decode golden path.

Provides :func:`decode_session`, a convenience wrapper that glues together
:func:`~neurospatial.encoding.compute_spatial_rates`,
:func:`~neurospatial.decoding.bin_spikes_in_time`, and
:func:`~neurospatial.decoding.decode_position` for the standard
encode-then-decode workflow.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import DEFAULT_MAX_GAP, resolve_time_windows
from neurospatial._observation import validate_break_before
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial._stacklevel import _external_stacklevel
from neurospatial._time_bins import _count_spikes_in_time_bins
from neurospatial._validation import validate_float_dtype
from neurospatial.decoding._time_bins import validate_dt
from neurospatial.encoding._spatial_counts import (
    _clock_ratio_hint,
    _emit_time_window_warning,
)

# Share of observed time that whole-dt bins may leave undecoded before warning.
_DROPPED_TIME_WARN_FRACTION = 0.1

# Default time-block size for the streaming summary path. Matches
# decode_position_summary's default ``time_chunk`` so the two paths block the
# time axis identically.
_SUMMARY_DEFAULT_TIME_CHUNK = 1024
_DECODE_OPTIONS = ("prior", "validate", "time_chunk")

if TYPE_CHECKING:
    from neurospatial._typing import SpikeTrainsLike
    from neurospatial.decoding._result import DecodingResult, DecodingSummary
    from neurospatial.decoding.likelihood import _PoissonRates
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment


def _warn_if_spikes_out_of_window(
    trains: list[NDArray[np.float64]],
    t_start: float,
    t_stop: float,
) -> None:
    """Warn once if most spikes fall outside the decode window ``[t_start, t_stop]``.

    Aggregates across all spike trains; an empty session never warns. The
    warning names a likely millisecond/second mix-up.
    """
    nonempty = [t for t in trains if t.size > 0]
    if not nonempty:
        return
    n_out = sum(int(np.count_nonzero((t < t_start) | (t > t_stop))) for t in nonempty)
    # The message needs only the spike-time range.
    spike_range = np.array(
        [min(float(t.min()) for t in nonempty), max(float(t.max()) for t in nonempty)]
    )
    _emit_time_window_warning(
        n_out,
        sum(int(t.size) for t in nonempty),
        t_start,
        t_stop,
        spike_range,
        window="decode",
    )


def _refuse_spikes_on_another_clock(
    trains: list[NDArray[np.float64]],
    bin_left: NDArray[np.float64],
    bin_right: NDArray[np.float64],
    *,
    context: str,
) -> None:
    """Raise when every spike lies before, or every spike after, the decode bins.

    Spike trains that straddle the bins (a silent window inside a longer
    recording) are legitimate and pass.
    """
    nonempty = [t for t in trains if t.size > 0]
    if not nonempty:
        return
    first = min(float(t.min()) for t in nonempty)
    last = max(float(t.max()) for t in nonempty)
    start, stop = float(bin_left[0]), float(bin_right[-1])
    if last >= start and first < stop:
        return
    side = "before" if last < start else "after"
    total = sum(int(t.size) for t in nonempty)
    hint = _clock_ratio_hint(np.array([first, last]), start, stop)
    raise ValueError(
        _format_error(
            f"{context}: all {total} spike times lie {side} the decode time "
            f"bins [{start:.6g}, {stop:.6g}) s (spike_times range "
            f"{first:.6g} to {last:.6g}). {hint}",
            why="spike_times and times are on different clocks or in different "
            "units, so every decode bin would be empty and the posterior "
            "would only repeat the prior.",
            fix="express spike_times in seconds on the clock of times, for "
            "example spike_times / 1000 for milliseconds, or subtract the "
            "offset between the two clocks",
        )
    )


def _warn_if_bins_drop_observed_time(
    runs: NDArray[np.float64],
    bin_left: NDArray[np.float64],
    bin_right: NDArray[np.float64],
    dt: float,
    max_gap: float | None,
) -> None:
    """Warn when whole-``dt`` bins leave much of the observed time undecoded.

    Each observed run keeps only whole bins, so the end of every run shorter
    than ``dt`` is not decoded. Short, frequent runs (tracking dropouts with a
    small ``max_gap``) can lose a large share of the recording this way.

    Parameters
    ----------
    runs : ndarray, shape (n_runs, 2)
        Observed half-open runs in seconds.
    bin_left, bin_right : ndarray, shape (n_time_bins,)
        Decode bin edges tiled inside ``runs``.
    dt, max_gap : float, float or None
        The decode bin width and the gap threshold, named in the message.
    """
    observed = float(np.sum(runs[:, 1] - runs[:, 0]))
    if observed <= 0:
        return
    binned = float(np.sum(bin_right - bin_left))
    dropped = (observed - binned) / observed
    if dropped <= _DROPPED_TIME_WARN_FRACTION:
        return
    warnings.warn(
        f"Decode bins cover {binned:.4g} s of the {observed:.4g} s observed in "
        f"{runs.shape[0]} recording run(s); {100 * dropped:.0f}% is not "
        f"decoded. Each run keeps only whole dt={dt} s bins, so the end of "
        f"every run shorter than dt is dropped, with its spikes. Fix: use a "
        f"smaller dt, or a larger max_gap (now {max_gap}) if the short runs "
        f"come from brief tracking dropouts. Set warn_on_drop=False to "
        f"suppress this warning.",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


@reads_pynapple_samples
def decode_session(
    env: Environment,
    spike_times: SpikeTrainsLike,
    times: ArrayLike,
    positions: NDArray[np.float64],
    *,
    dt: float = 0.025,
    bandwidth: float | None = None,
    method: str = "diffusion_kde",
    min_occupancy: float | None = None,
    penalty: float | None = None,
    rank: int | None = None,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    warn_on_drop: bool = True,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    **decode_kwargs: Any,
) -> DecodingResult:
    """Encode, bin, and decode in one call.

    Glues together :func:`~neurospatial.encoding.compute_spatial_rates`,
    :func:`~neurospatial.decoding.bin_spikes_in_time`, and
    :func:`~neurospatial.decoding.decode_position` into a single entry point
    for the standard encode-then-decode workflow. It requires only the four
    positional arguments to return a full
    :class:`~neurospatial.decoding.DecodingResult`.

    The rate maps are fitted on the same spikes and positions that are then
    decoded, so the result is an in-sample decode: its error against the
    tracked position is optimistic. To measure decoding accuracy, fit on
    training epochs and predict on separate test epochs with
    ``BayesianDecoder(env).fit(..., epochs=train).predict(..., epochs=test)``.

    Parameters
    ----------
    env : Environment
        Spatial environment that defines the bin layout and
        connectivity graph.
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds for one or more neurons; every spike time must
        be finite. Accepted formats:

        - 1-D array / list of scalars / pynapple ``Ts`` → single neuron
        - 2-D array, shape ``(n_neurons, max_spikes)``, padded with trailing NaN
        - List/tuple of 1-D arrays → one array per neuron (canonical)
        - pynapple ``TsGroup``, ``SpikeTrains`` or a dict of unit id to train
    times : array-like, shape (n_frames,)
        Tracking timestamps in seconds. Decode bins tile each observed run;
        the returned clock can contain gaps. For pynapple, pass ``tsd.t``.
    positions : ndarray, shape (n_frames, n_dims)
        Required tracking coordinates aligned with ``times``. For pynapple,
        pass ``tsd.values`` explicitly.
    dt : float, optional
        Decoding time-bin width in seconds.  Default is 0.025 (25 ms).
    bandwidth : float or None, optional
        Smoothing bandwidth (same units as positions) for the ratio-method
        encoding step. ``None`` (default) resolves to the encoder's default
        (5 cm in declared metric units, otherwise 2.5 median bin spacings); a ratio-only param, so it must stay ``None`` when
        ``method="glm"``.
    method : str, optional
        Estimator passed to :func:`~neurospatial.encoding.compute_spatial_rates`.
        Options: ``"diffusion_kde"`` (default), ``"gaussian_kde"``, ``"binned"``,
        and ``"glm"`` (penalized-Poisson GAM, tuned with ``penalty`` / ``rank``).
    min_occupancy : float or None, optional
        Minimum occupancy (seconds) for a spatial bin to be included in the
        ratio-method encoding model. Unoccupied bins and bins below this
        threshold have zero posterior probability. ``None``
        (default) resolves to the encoder's default (0.0, no threshold); a
        ratio-only param, so it must stay ``None`` when ``method="glm"``.
    penalty : float or None, optional
        ``method="glm"`` smoothness penalty ``lambda``. ``None`` (default)
        chooses it by REML. Mutually exclusive with the ratio params
        (``bandwidth`` / ``min_occupancy``).
    rank : int or None, optional
        ``method="glm"`` requested basis rank cap. ``None`` (default) uses the
        encoder default.
    speed : NDArray[np.float64], shape (n_frames,) or None
        Precomputed instantaneous speed at each trajectory sample, forwarded to
        :func:`~neurospatial.encoding.compute_spatial_rates`. Only used when
        ``min_speed`` is set; auto-derived when ``None``.
    min_speed : float or None
        Minimum speed threshold (physical units / second), forwarded to the
        encoding step so the decode golden path can speed-filter encoding. When
        set, low-speed periods are excluded from BOTH the spike numerator and
        the occupancy denominator of the encoding model via one shared gate.
        When ``None`` (default) no speed filtering is applied.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_frames,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from encoding and decoding, as for a gap longer than
        ``max_gap``: no decode time bin spans it and spikes inside it are not
        counted.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from the rate maps' occupancy and from decoding, and their
        spikes are not counted. ``None`` (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    warn_on_drop : bool, optional
        If ``True`` (the default), emit a single ``UserWarning`` when a large
        fraction (>50%, which includes the all-dropped case) of spikes fall
        outside the decode time window ``[times.min(), times.max()]`` and are
        therefore excluded from the count matrix, for example
        ``spike_times`` in milliseconds while ``times`` is in seconds. The
        warning fires once per call, from the encoder, and also covers spikes
        mapped outside the environment. It also warns when more than 10% of
        the observed time falls in run remainders shorter than ``dt``, which
        are not decoded. Set ``False`` to suppress these diagnostics. When
        every spike lies before, or every spike after, the decode bins, the
        call raises whatever ``warn_on_drop`` is.
    dtype : {np.float32, np.float64}, default=np.float64
        Dtype of the stored rate maps and posterior; ``np.float32`` halves
        their memory. Rates are computed in float64 and then cast. Any other
        dtype raises ``ValueError``.
    **decode_kwargs
        ``prior``, ``validate`` and ``time_chunk``, with the meaning they have
        in :func:`~neurospatial.decoding.decode_position`. ``method`` above
        names the rate estimator, and ``dtype`` is the explicit parameter
        above. Unknown kwargs raise ``TypeError``.

    Returns
    -------
    DecodingResult
        Container with the posterior distribution over positions for each
        decoding time bin.  Key properties:

        - ``.posterior``, shape ``(n_time_bins, n_bins)`` — full posterior
        - ``.map_position``, shape ``(n_time_bins, n_dims)`` — MAP estimate
        - ``.times``, shape ``(n_time_bins,)`` — decoding bin centers
        - ``.posterior_entropy``, shape ``(n_time_bins,)`` — posterior
          entropy in bits per time bin

    Raises
    ------
    ValueError
        If ``dt`` is not finite and positive, if no decode time bin fits in
        the observed runs, if every spike lies before or after the decode
        bins (spike_times and times on different clocks or in different
        units), or if the encoding model has no supported bin.

    Notes
    -----
    Decode time bins are formed separately within each run of samples whose
    gaps are no longer than ``max_gap`` and that lie inside ``epochs`` and
    ``spike_window``; no bin spans a pause, and spikes between runs are not
    counted. ``result.times`` may therefore be non-contiguous. A gap is the
    time between consecutive tracked samples: a NaN position is missing
    tracking, bridged like a deleted row, so a tracking loss longer than
    ``max_gap`` is not decoded. Without ``spike_window``, spikes are assumed
    recorded only while position was. To decode ``[t0, t1)`` without
    tracking, fit a :class:`~neurospatial.decoding.BayesianDecoder` and
    predict with ``times=np.arange(t0, t1 + dt / 2, dt)``.

    **Orientation contract**:
    :func:`~neurospatial.decoding.bin_spikes_in_time` returns a count
    matrix of shape ``(n_time_bins, n_neurons)`` (default
    ``orient="time_x_neuron"``), which is exactly what
    :func:`~neurospatial.decoding.decode_position` expects for its
    ``spike_counts`` argument.  No transposition is performed.

    **Prior and training support**:
    The default prior is uniform over space restricted to the training
    support: bin mass proportional to ``env.bin_sizes``, zero in unoccupied
    bins, bins below ``min_occupancy`` and non-finite model bins, also during
    silent decoding intervals. An explicit ``prior`` is intersected with this
    support. Choose ``bandwidth`` smaller than the place fields and ``dt``
    long enough that most bins hold a few spikes (20-50 ms for tens of place
    cells during running). Internal finite
    placeholders in excluded bins do not represent measured zero activity.
    For precomputed maps use ``BayesianDecoder.from_rates(rates)``; an explicit
    array call must carry ``prior=rates.spatial_support * env.bin_sizes``.

    **Time grid**:
    Each valid run is tiled separately with half-open decode bins. The spike
    coverage warning compares the full timestamp span, while the actual counts
    exclude gaps, epochs and unobserved spike windows.

    See Also
    --------
    BayesianDecoder : Fit tracking once and predict without position inputs;
        ``from_rates`` accepts precomputed spatial results.
    decode_position : Decode explicit binned counts and rate arrays.
    bin_spikes_in_time : Bin spikes for the explicit-array route.

    Examples
    --------
    Minimal usage with simulated data:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import decode_session
    >>> from neurospatial.simulation import (
    ...     PlaceCellModel,
    ...     generate_poisson_spikes,
    ...     simulate_trajectory_ou,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> positions_raw = np.column_stack([np.linspace(0.0, 100.0, 500), np.zeros(500)])
    >>> env = Environment.from_samples(positions_raw, bin_size=5.0)
    >>> env = env.with_metadata(units="cm")  # required by simulate_trajectory_ou
    >>> positions, times = simulate_trajectory_ou(
    ...     env, duration=10.0, speed_units="cm", seed=0
    ... )
    >>> n_neurons = 10
    >>> spike_times = [
    ...     generate_poisson_spikes(
    ...         PlaceCellModel(env, width=15.0, seed=i).firing_rate(times, positions),
    ...         times,
    ...         seed=i,
    ...     )
    ...     for i in range(n_neurons)
    ... ]
    >>> result = decode_session(env, spike_times, times, positions, dt=0.1)
    >>> result.posterior.shape[1] == env.n_bins
    True
    >>> result.map_position.shape[1]
    2

    Reuse explicit rate arrays on this continuous recording:

    >>> from neurospatial.encoding import compute_spatial_rates
    >>> from neurospatial.decoding import bin_spikes_in_time, decode_position
    >>> rates = compute_spatial_rates(env, spike_times, times, positions, bandwidth=5.0)
    >>> counts, centers = bin_spikes_in_time(
    ...     spike_times, dt=0.1, t_start=times[0], t_stop=times[-1]
    ... )
    >>> result = decode_position(
    ...     env,
    ...     counts,
    ...     rates.firing_rates,
    ...     dt=0.1,
    ...     times=centers,
    ...     prior=rates.spatial_support * env.bin_sizes,
    ... )
    """
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    dt = validate_dt(dt)
    dtype = validate_float_dtype(dtype, stored="rate maps and posteriors")
    options = _decode_options(decode_kwargs, caller="decode_session", summary=False)
    trains, firing_rates, rates_result = _fit_encoding_model(
        env,
        spike_times,
        times,
        positions,
        bandwidth=bandwidth,
        method=method,
        min_occupancy=min_occupancy,
        penalty=penalty,
        rank=rank,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=resolved_epochs,
        break_before=break_before,
        spike_window=resolved_spike_window,
        warn_on_drop=warn_on_drop,
        dtype=dtype,
    )
    return _decode_with_models(
        env,
        trains,
        times,
        firing_rates,
        dt=dt,
        max_gap=max_gap,
        epochs=resolved_epochs,
        break_before=break_before,
        spike_window=resolved_spike_window,
        warn_spikes_outside=False,
        warn_partial_bins=warn_on_drop,
        dtype=dtype,
        spatial_support=rates_result.spatial_support,
        sample_valid=_tracked(times, positions),
        **options,
    )


def _decode_options(
    decode_kwargs: dict[str, Any], *, caller: str, summary: bool
) -> dict[str, Any]:
    """Validate the forwarded decode keywords before the encoding fit.

    Returns ``prior``, ``validate`` and ``time_chunk``; a summary decoder
    refuses ``time_chunk=None``.
    """
    from neurospatial.decoding.posterior import (
        _summary_time_chunk,
        _validate_time_chunk,
    )

    unknown = sorted(set(decode_kwargs) - set(_DECODE_OPTIONS))
    if unknown:
        raise TypeError(
            f"{caller} got unexpected keyword argument(s): {unknown}. Supported "
            f"decode kwargs are {', '.join(_DECODE_OPTIONS)} (dtype is an "
            "explicit parameter)."
        )
    time_chunk = decode_kwargs.get(
        "time_chunk", _SUMMARY_DEFAULT_TIME_CHUNK if summary else None
    )
    return {
        "prior": decode_kwargs.get("prior"),
        "validate": decode_kwargs.get("validate", True),
        "time_chunk": (
            _summary_time_chunk(
                time_chunk, caller=caller, full_decoder="decode_session"
            )
            if summary
            else _validate_time_chunk(time_chunk, allow_none=True)
        ),
    }


def _tracked(times: ArrayLike, positions: ArrayLike) -> NDArray[np.bool_]:
    """True at samples whose every coordinate is finite, shape (n_samples,)."""
    from neurospatial.encoding._spatial_counts import tracked_samples

    return tracked_samples(len(np.asarray(times)), np.asarray(positions))


def _prepare_session_decode(
    env: Environment,
    spike_times: Any,
    times: ArrayLike,
    encoding_models: NDArray[np.float64],
    *,
    dt: float,
    max_gap: float | None,
    epochs: Any,
    break_before: ArrayLike | None,
    spike_window: Any,
    sample_valid: NDArray[np.bool_] | None,
    warn_spikes_outside: bool,
    warn_partial_bins: bool,
    dtype: type[np.float32] | type[np.float64],
    spatial_support: NDArray[np.bool_] | None,
    prior: Any,
    validate: bool,
    context: str,
) -> tuple[
    Callable[[int, int], NDArray[np.int64]],
    _PoissonRates,
    NDArray[np.float64] | None,
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Tile decode bins inside the observed runs and prepare the rate maps.

    ``break_before`` is validated here (naming ``context``); no decode bin
    spans a declared break. ``sample_valid`` marks the tracked samples; a
    missing one is bridged like a deleted row, so a tracking loss longer than
    ``max_gap`` is not decoded.

    Returns
    -------
    count_block : callable
        ``count_block(start, stop)`` counts the spikes of decode bins
        ``start:stop``, so no full count matrix is built.
    rates, prior
        Likelihood terms and resolved prior, as from ``_prepare_decode_inputs``.
    centers : NDArray[np.float64], shape (n_time_bins,)
        Decode bin centers in seconds.
    spike_window : NDArray[np.float64] or None
        The resolved spike-recording windows.
    """
    from neurospatial._intervals import TimeSupport, interval_valid_mask
    from neurospatial._spikes import _as_spike_trains_with_ids
    from neurospatial._time_bins import _time_bins_in_windows
    from neurospatial.decoding.posterior import _prepare_decode_inputs
    from neurospatial.encoding import _validate_times

    epochs, spike_window = resolve_time_windows(epochs, spike_window)
    # Normalize array timestamps and spike groups at the boundary.
    trains, _ = _as_spike_trains_with_ids(spike_times)
    times_arr = np.asarray(times, dtype=np.float64)
    if times_arr.ndim != 1:
        raise ValueError(
            f"times must be a 1-D array of timestamps for decode_session, "
            f"got shape {times_arr.shape}."
        )
    # Require finite, sorted timestamps before constructing observed runs.
    _validate_times(times_arr, context=context)
    breaks = validate_break_before(break_before, times_arr.size, call=context)

    if warn_spikes_outside:
        _warn_if_spikes_out_of_window(
            trains, float(times_arr.min()), float(times_arr.max())
        )

    # Decode bins exist only where the recording was observed: the gap, epochs and
    # spike_window gates, with missing tracking bridged like deleted rows. The
    # speed and out-of-bounds gates restrict only the ENCODING step, so periods
    # of immobility (for example replay) are still decoded.
    observed = interval_valid_mask(
        times_arr,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        break_before=breaks,
        sample_valid=sample_valid,
    )
    recording_support = TimeSupport.from_mask(
        times_arr,
        observed,
        epochs=epochs,
        spike_window=spike_window,
        max_gap=max_gap,
        break_before=breaks,
    )
    runs = recording_support.runs
    bin_left, bin_right = _time_bins_in_windows(runs, dt)
    if bin_left.size:
        _refuse_spikes_on_another_clock(trains, bin_left, bin_right, context=context)
    if warn_partial_bins:
        _warn_if_bins_drop_observed_time(runs, bin_left, bin_right, dt, max_gap)
    if bin_left.size == 0:
        longest = float(np.max(np.diff(runs, axis=1), initial=0.0))
        raise ValueError(
            f"No decode time bin fits: the {runs.shape[0]} observed recording "
            f"run(s) are at most {longest:.3g} s long, shorter than dt={dt}. "
            f"\nWhy: time bins are formed only inside runs of samples with gaps "
            f"<= max_gap={max_gap} s that lie inside epochs and spike_window "
            f"and do not cross a declared break_before. "
            f"\nFix: use a smaller dt, widen epochs/spike_window, or pass a larger "
            f"max_gap (max_gap=None decodes across gaps)."
        )

    rates, prior = _prepare_decode_inputs(
        env,
        np.asarray(encoding_models, dtype=dtype),
        prior,
        bin_left.size,
        dt=dt,
        validate=validate,
        context=context,
        spatial_support=spatial_support,
    )
    return (
        _spike_count_blocks(trains, bin_left, bin_right),
        rates,
        prior,
        bin_left + dt / 2.0,
        spike_window,
    )


def _spike_count_blocks(
    trains: list[NDArray[np.float64]],
    bin_left: NDArray[np.float64],
    bin_right: NDArray[np.float64],
) -> Callable[[int, int], NDArray[np.int64]]:
    """Count spikes in decode bins ``start:stop`` on demand.

    Sorted trains let each block take its spikes by binary search instead of
    a full-length mask per block.
    """
    trains = [
        t if t.size < 2 or bool(np.all(t[1:] >= t[:-1])) else np.sort(t) for t in trains
    ]

    def count_block(start: int, stop: int) -> NDArray[np.int64]:
        lo, hi = bin_left[start], bin_right[stop - 1]
        block_trains = [
            s[np.searchsorted(s, lo, side="left") : np.searchsorted(s, hi, side="left")]
            for s in trains
        ]
        return _count_spikes_in_time_bins(
            block_trains, bin_left[start:stop], bin_right[start:stop]
        )

    return count_block


def _fit_encoding_model(
    env: Environment,
    spike_times: Any,
    times: ArrayLike,
    positions: NDArray[np.float64],
    *,
    bandwidth: float | None,
    method: str,
    min_occupancy: float | None,
    penalty: float | None = None,
    rank: int | None = None,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: NDArray[np.float64] | None = None,
    break_before: ArrayLike | None = None,
    spike_window: NDArray[np.float64] | None = None,
    warn_on_drop: bool,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    context: str = "decode_session",
) -> tuple[
    list[NDArray[np.float64]],
    NDArray[np.float64],
    SpatialRatesResult,
]:
    """Fit supported rate maps independently of the prediction time bins."""
    from neurospatial._spikes import _as_spike_trains_with_ids
    from neurospatial._validation import validate_times_positions
    from neurospatial.encoding.spatial import compute_spatial_rates

    times_arr, positions = validate_times_positions(times, positions, call=context)
    breaks = validate_break_before(break_before, times_arr.size, call=context)
    trains, _ = _as_spike_trains_with_ids(spike_times)
    # Mirror the encoder's method-specific validation (mutual exclusivity +
    # value domains) at the decoder boundary, reusing the SAME validator so
    # the errors are identical.
    from neurospatial.encoding import _validate_spatial_method_params

    penalty, rank = _validate_spatial_method_params(
        method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        penalty=penalty,
        rank=rank,
    )
    _method = cast("Literal['diffusion_kde', 'gaussian_kde', 'binned', 'glm']", method)
    # Unsupported bins stay NaN; spatial_support excludes them from the
    # posterior.
    # The encoder owns spike/position coverage diagnostics and already
    # emits the spike-drop warning (and additionally an inactive-bin /
    # wrong-coordinate-frame warning the decode-time check cannot), so we
    # let it own the warning here and just thread warn_on_drop through.
    rates_result = compute_spatial_rates(
        env,
        trains,
        times_arr,
        positions,
        bandwidth=bandwidth,
        method=_method,
        min_occupancy=min_occupancy,
        penalty=penalty,
        rank=rank,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        warn_on_drop=warn_on_drop,
        dtype=dtype,
        epochs=epochs,
        break_before=breaks,
        spike_window=spike_window,
    )
    # compute_spatial_rates already stores the result in `dtype`; the cast
    # is a cheap no-op guard so the working set is honored end-to-end. The
    # array is float32 OR float64; the declared NDArray[np.float64] return
    # type is the family annotation (cast keeps mypy happy).
    firing_rates = cast(
        "NDArray[np.float64]",
        np.asarray(rates_result.firing_rates, dtype=dtype),
    )
    # Training requires observed spatial support, independent of whether a
    # prediction bin of any particular width fits in the training intervals.
    occupancy = np.asarray(rates_result.occupancy)
    if not np.any((occupancy > 0) & (occupancy >= (min_occupancy or 0.0))):
        raise ValueError(
            _format_error(
                f"{context}: the encoding model has no occupied bin, so every "
                f"decode time bin would get a uniform posterior.",
                why=(
                    "Why: the gates (max_gap, min_speed, epochs, break_before, "
                    "spike_window and min_occupancy) left no training interval "
                    "with time in any kept bin."
                ),
                fix=(
                    "check that times, epochs and spike_window share one "
                    "clock in seconds, lower min_speed or min_occupancy, or "
                    "pass max_gap=None for coarsely sampled tracking"
                ),
            )
        )
    return trains, firing_rates, rates_result


def _decode_with_models(
    env: Environment,
    spike_times: Any,
    times: ArrayLike,
    encoding_models: NDArray[np.float64],
    *,
    dt: float = 0.025,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    warn_spikes_outside: bool = True,
    warn_partial_bins: bool = True,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    spatial_support: NDArray[np.bool_] | None = None,
    sample_valid: NDArray[np.bool_] | None = None,
    prior: Any = None,
    validate: bool = True,
    time_chunk: int | None = None,
    context: str = "decode_session",
) -> DecodingResult:
    """Decode existing models on the per-run recording clock."""
    from neurospatial.decoding.posterior import _decode_posterior

    count_block, rates, prior, centers, spike_window = _prepare_session_decode(
        env,
        spike_times,
        times,
        encoding_models,
        dt=dt,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        sample_valid=sample_valid,
        warn_spikes_outside=warn_spikes_outside,
        warn_partial_bins=warn_partial_bins,
        dtype=dtype,
        spatial_support=spatial_support,
        prior=prior,
        validate=validate,
        context=context,
    )
    return _decode_posterior(
        env,
        count_block,
        centers.size,
        rates,
        prior,
        dtype=dtype,
        time_chunk=time_chunk,
        times=centers,
        spike_window=spike_window,
    )


@reads_pynapple_samples
def decode_session_summary(
    env: Environment,
    spike_times: SpikeTrainsLike,
    times: ArrayLike,
    positions: NDArray[np.float64],
    *,
    dt: float = 0.025,
    bandwidth: float | None = None,
    method: str = "diffusion_kde",
    min_occupancy: float | None = None,
    penalty: float | None = None,
    rank: int | None = None,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    warn_on_drop: bool = True,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    **decode_kwargs: Any,
) -> DecodingSummary:
    """Memory-safe sibling of :func:`decode_session`.

    Same encode step as :func:`decode_session`, but **streams the
    time-binning** so the full ``(n_time, n_neurons)`` count matrix is never
    materialized, and reduces the posterior block-by-block so the full
    ``(n_time, n_bins)`` posterior is never materialized either. Returns a
    :class:`~neurospatial.decoding.DecodingSummary` of per-time reductions. Use
    this for long sessions where the dense count matrix and/or posterior would
    not fit in memory.

    The rate maps are fitted once over the whole session. Time is then
    processed in blocks of ``time_chunk`` bins: each block counts only its own
    spikes and is decoded and reduced like
    :func:`~neurospatial.decoding.decode_position_summary` would on the full
    count matrix. The transient memory of a block is
    ``O(time_chunk * max(n_neurons, n_bins))``; the stored per-time reductions
    and the decode bin edges grow with session length. Like
    :func:`decode_session`, this is an in-sample decode of the training data.

    Parameters
    ----------
    env, spike_times, times, positions, dt, bandwidth, method, \
min_occupancy, penalty, rank, speed, min_speed, max_gap, \
warn_on_drop, dtype
        Same as :func:`decode_session` -- including ``method="glm"`` and its
        ``penalty`` / ``rank`` knobs, and the nullable ``bandwidth`` /
        ``min_occupancy`` (``max_gap`` gates encoding and decoding).
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_frames,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from encoding and decoding, as for a gap longer than
        ``max_gap``: no decode time bin spans it and spikes inside it are not
        counted.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from the rate maps' occupancy and from decoding, and their
        spikes are not counted. ``None`` (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    **decode_kwargs
        ``prior``, ``validate`` and ``time_chunk`` (a positive block size,
        default 1024), with the meaning they have in
        :func:`~neurospatial.decoding.decode_position_summary`. Unknown kwargs
        raise ``TypeError``.

    Returns
    -------
    DecodingSummary
        Per-time reductions (MAP position/bin, mean position, entropy, peak
        probability) plus ``times`` and ``env``.

    Raises
    ------
    ValueError
        If ``time_chunk`` is ``None`` or not a positive integer; if a forwarded
        ``prior`` has a shape inconsistent with the decode (1-D must be
        ``(n_bins,)``, 2-D must be ``(n_time, n_bins)``); plus the same
        conditions as :func:`~neurospatial.decoding.decode_position`.

    Notes
    -----
    Decode time bins are formed separately within each run of samples whose
    gaps are no longer than ``max_gap`` and that lie inside ``epochs`` and
    ``spike_window``; no bin spans a pause, and spikes between runs are not
    counted. ``result.times`` may therefore be non-contiguous. A gap is the
    time between consecutive tracked samples: a NaN position is missing
    tracking, bridged like a deleted row, so a tracking loss longer than
    ``max_gap`` is not decoded. Without ``spike_window``, spikes are assumed
    recorded only while position was. To decode ``[t0, t1)`` without
    tracking, fit a :class:`~neurospatial.decoding.BayesianDecoder` and
    predict with ``times=np.arange(t0, t1 + dt / 2, dt)``.

    See Also
    --------
    decode_session : Full-posterior golden path.
    neurospatial.decoding.decode_position_summary : Array-first streamed decoder.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import decode_session_summary
    >>> times = np.arange(300) / 30.0
    >>> positions = np.c_[np.linspace(0.0, 10.0, len(times)), np.zeros(len(times))]
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> spikes = [times[::10], times[::15]]
    >>> summary = decode_session_summary(env, spikes, times, positions, dt=0.1, time_chunk=8)
    >>> len(summary.map_position) > 0
    True
    """
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    dt = validate_dt(dt)
    dtype = validate_float_dtype(dtype, stored="rate maps and posteriors")
    # Reject unsupported reduction options before performing the encoding fit.
    options = _decode_options(
        decode_kwargs, caller="decode_session_summary", summary=True
    )
    trains, firing_rates, rates_result = _fit_encoding_model(
        env,
        spike_times,
        times,
        positions,
        bandwidth=bandwidth,
        method=method,
        min_occupancy=min_occupancy,
        penalty=penalty,
        rank=rank,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=resolved_epochs,
        break_before=break_before,
        spike_window=resolved_spike_window,
        warn_on_drop=warn_on_drop,
        dtype=dtype,
        context="decode_session_summary",
    )
    return _decode_with_models_summary(
        env,
        trains,
        times,
        firing_rates,
        dt=dt,
        max_gap=max_gap,
        epochs=resolved_epochs,
        break_before=break_before,
        spike_window=resolved_spike_window,
        warn_spikes_outside=False,
        warn_partial_bins=warn_on_drop,
        dtype=dtype,
        spatial_support=rates_result.spatial_support,
        sample_valid=_tracked(times, positions),
        **options,
    )


def _decode_with_models_summary(
    env: Environment,
    spike_times: Any,
    times: ArrayLike,
    encoding_models: NDArray[np.float64],
    *,
    dt: float = 0.025,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    warn_spikes_outside: bool = True,
    warn_partial_bins: bool = True,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    spatial_support: NDArray[np.bool_] | None = None,
    sample_valid: NDArray[np.bool_] | None = None,
    prior: Any = None,
    validate: bool = True,
    time_chunk: int = _SUMMARY_DEFAULT_TIME_CHUNK,
    context: str = "decode_session_summary",
) -> DecodingSummary:
    """Stream reductions over existing models and per-run time bins.

    The spikes are binned one block at a time, so neither the full count
    matrix nor the full posterior is built.
    """
    from neurospatial.decoding.posterior import _decode_summary

    count_block, rates, prior, centers, spike_window = _prepare_session_decode(
        env,
        spike_times,
        times,
        encoding_models,
        dt=dt,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        sample_valid=sample_valid,
        warn_spikes_outside=warn_spikes_outside,
        warn_partial_bins=warn_partial_bins,
        dtype=dtype,
        spatial_support=spatial_support,
        prior=prior,
        validate=validate,
        context=context,
    )
    return _decode_summary(
        env,
        count_block,
        centers.size,
        rates,
        prior,
        dtype=dtype,
        time_chunk=time_chunk,
        validate=validate,
        times=centers,
        spike_window=spike_window,
        context=context,
    )
