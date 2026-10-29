"""One-call encode->bin->decode golden path.

Provides :func:`decode_session`, a convenience wrapper that glues together
:func:`~neurospatial.encoding.compute_spatial_rates`,
:func:`~neurospatial.decoding.bin_spikes_in_time`, and
:func:`~neurospatial.decoding.decode_position` so a beginner can decode
position from spikes in ≤10 lines.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import DEFAULT_MAX_GAP, resolve_time_windows
from neurospatial._observation import validate_break_before
from neurospatial._stacklevel import _external_stacklevel
from neurospatial._time_bins import _count_spikes_in_time_bins
from neurospatial.decoding._time_bins import validate_dt

# Warn when more than this fraction of spikes fall outside the decode window.
# Always < 1.0, so the 100%-dropped case (frac == 1.0) always warns.
_DROP_WARN_THRESHOLD = 0.5
# Share of observed time that whole-dt bins may leave undecoded before warning.
_DROPPED_TIME_WARN_FRACTION = 0.1

# Default time-block size for the streaming summary path. Matches
# decode_position_summary's default ``time_chunk`` so the two paths block the
# time axis identically.
_SUMMARY_DEFAULT_TIME_CHUNK = 1024

if TYPE_CHECKING:
    from neurospatial._typing import SpikeTrainsLike
    from neurospatial.decoding._result import DecodingResult, DecodingSummary
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment


def _clock_ratio_hint(
    spikes: NDArray[np.float64], t_start: float, t_stop: float
) -> str:
    """Name a milliseconds/seconds mix-up when the time spans differ ~1000x."""
    spike_span = float(spikes.max() - spikes.min())
    times_span = t_stop - t_start
    if spike_span <= 0 or times_span <= 0:
        return ""
    # Within a factor of ~3 of 1000x either way.
    log_ratio = np.log10(spike_span / times_span)
    if abs(log_ratio - 3) < 0.5:
        return (
            f"The spike times span {spike_span:.6g}, about 1000 times the "
            f"{times_span:.6g} s of times: spike_times are probably in "
            "milliseconds; divide them by 1000. "
        )
    if abs(log_ratio + 3) < 0.5:
        return (
            f"The spike times span {spike_span:.6g}, about 1/1000 of the "
            f"{times_span:.6g} span of times: times are probably in "
            "milliseconds; divide them by 1000. "
        )
    return ""


def _warn_if_spikes_out_of_window(
    trains: list[NDArray[np.float64]],
    t_start: float,
    t_stop: float,
) -> None:
    """Emit one UserWarning if most spikes fall outside the decode window.

    Aggregates across all spike trains. Warns (does not raise) when the
    dropped fraction exceeds ``_DROP_WARN_THRESHOLD``. A genuinely empty
    session is legitimate, so an empty input never warns. When the spike
    times span about 1000 times (or 1/1000 of) the window, the message names
    the likely millisecond/second mix-up.

    Parameters
    ----------
    trains : list of ndarray
        Per-neuron spike-time arrays (already normalized).
    t_start, t_stop : float
        Decode time window bounds ``[t_start, t_stop]`` (seconds).
    """
    total = sum(int(t.size) for t in trains)
    if total == 0:
        return

    n_out = sum(int(np.count_nonzero((t < t_start) | (t > t_stop))) for t in trains)
    frac = n_out / total
    if frac <= _DROP_WARN_THRESHOLD:
        return

    all_spikes = np.concatenate([t for t in trains if t.size > 0])
    warnings.warn(
        f"{n_out}/{total} spike_times "
        f"({100 * frac:.0f}%) fell outside the decode time window "
        f"[{t_start:.6g}, {t_stop:.6g}]; "
        f"spike_times.min()={all_spikes.min():.6g} "
        f"spike_times.max()={all_spikes.max():.6g}. "
        f"{_clock_ratio_hint(all_spikes, t_start, t_stop)}"
        f"Check that spike_times and times share units (both seconds). "
        f"Dropped spikes do not contribute to the posterior. "
        f"Set warn_on_drop=False to suppress this warning.",
        UserWarning,
        stacklevel=_external_stacklevel(),
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
    for the standard encode-then-decode workflow. Beginner-friendly: requires
    only the four positional arguments and a ``dt`` keyword to get a full
    :class:`~neurospatial.decoding.DecodingResult`.

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
        When ``None`` (default) no speed filtering is applied (unchanged).
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
    break_before : ArrayLike of bool, shape (n_frames,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from encoding and decoding, as for a gap longer than
        ``max_gap``: no decode time bin spans it and spikes inside it are not
        counted.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
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
        "Decode in this dtype." Controls BOTH the encoding-model working set
        AND the posterior dtype end-to-end. ``np.float32`` halves the
        encoding-model + posterior working set on the beginner golden path;
        values match the float64 default within float32 tolerance (the rate
        computation itself is done in float64 and only the stored result is
        cast, per :func:`~neurospatial.encoding.compute_spatial_rates`). Any
        other dtype raises ``ValueError``. Default ``np.float64`` leaves every
        existing caller byte-for-byte unchanged. Note: do not also pass
        ``dtype`` via ``decode_kwargs`` — this explicit parameter is the single
        source forwarded to :func:`~neurospatial.decoding.decode_position`, and
        a duplicate would raise ``TypeError``.
    **decode_kwargs
        Additional keyword arguments forwarded verbatim to
        :func:`~neurospatial.decoding.decode_position`.  Supported kwargs
        include ``prior`` and ``validate``.  (``method`` now names the smoothing
        estimator above, not a decode kwarg; pass ``dtype`` via the explicit
        ``dtype`` parameter, not here.)

    Returns
    -------
    DecodingResult
        Container with the posterior distribution over positions for each
        decoding time bin.  Key properties:

        - ``.posterior``, shape ``(n_time_bins, n_bins)`` — full posterior
        - ``.map_position``, shape ``(n_time_bins, n_dims)`` — MAP estimate
        - ``.times``, shape ``(n_time_bins,)`` — decoding bin centers
        - ``.posterior_entropy`` — per-bin uncertainty in bits

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
    ``max_gap`` is not decoded, as without ``spike_window`` spikes are assumed
    recorded only while position was. To decode spans without tracking, fit
    a :class:`~neurospatial.decoding.BayesianDecoder` and predict with
    ``times=np.arange(t0, t1, dt)``.

    **Orientation contract**:
    :func:`~neurospatial.decoding.bin_spikes_in_time` returns a count
    matrix of shape ``(n_time_bins, n_neurons)`` (default
    ``orient="time_x_neuron"``), which is exactly what
    :func:`~neurospatial.decoding.decode_position` expects for its
    ``spike_counts`` argument.  No transposition is performed.

    **Training spatial support**:
    Unoccupied bins, bins below ``min_occupancy`` and non-finite model bins
    are excluded from the posterior, even during silent decoding intervals.
    An explicit ``prior`` is intersected with this support. Internal finite
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
    ...         PlaceCellModel(env, width=15.0, seed=i).firing_rate(positions, times),
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
    validate_dt(dt)
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
        **decode_kwargs,
    )


def _validate_session_dtype(
    dtype: type[np.float32] | type[np.float64],
) -> type[np.float32] | type[np.float64]:
    """Resolve the supported encoding/posterior working precision."""
    # Validate dtype: only single/double precision working sets are supported.
    # Mirrors compute_spatial_rates' dtype validation wording. Wrap the parse so
    # an unparseable dtype string (e.g. "bogus") raises this clean ValueError
    # naming `dtype`, not a raw NumPy
    # ``TypeError: data type 'bogus' not understood``.
    _dtype_msg = (
        f"dtype must be np.float32 or np.float64, got {dtype!r}. "
        "Only single- and double-precision rate maps are supported "
        "(float32 halves the encoding-model working set and the "
        "downstream decode posterior)."
    )
    try:
        _resolved_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(_dtype_msg) from exc
    if _resolved_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(_dtype_msg)
    # Normalize to the canonical numpy scalar type for downstream casts.
    dtype = np.float32 if _resolved_dtype == np.dtype(np.float32) else np.float64

    return dtype


def _tracked(times: ArrayLike, positions: ArrayLike) -> NDArray[np.bool_]:
    """True at samples whose every coordinate is finite, shape (n_samples,)."""
    from neurospatial.encoding._spatial_counts import tracked_samples

    return tracked_samples(len(np.asarray(times)), np.asarray(positions))


def _prepare_session_decode(
    spike_times: Any,
    times: ArrayLike,
    encoding_models: NDArray[np.float64],
    *,
    dt: float,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: NDArray[np.float64] | None = None,
    break_before: ArrayLike | None = None,
    spike_window: NDArray[np.float64] | None = None,
    sample_valid: NDArray[np.bool_] | None = None,
    warn_spikes_outside: bool,
    warn_partial_bins: bool,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    context: str = "decode_session",
) -> tuple[
    list[NDArray[np.float64]],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Normalize spike inputs/models and tile bins inside observed runs.

    ``break_before`` is validated here (naming ``context``); no decode bin
    spans a declared break. ``sample_valid`` marks the tracked samples; a
    missing one is bridged like a deleted row, so a tracking loss longer than
    ``max_gap`` is not decoded.
    """
    from neurospatial._spikes import _as_spike_trains_with_ids
    from neurospatial.encoding import _validate_times

    # Validate dt up front, BEFORE the grid math below builds the decode time
    # grid directly (bypassing bin_spikes_in_time's own guard). Without this,
    # invalid dt leaks cryptic errors: dt=0 → ZeroDivisionError; dt=NaN →
    # "cannot convert float NaN to integer"; dt<0 → a MISLEADING "span smaller
    # than one bin dt" message; dt=inf → a similar cryptic failure. Route
    # through the shared bin_spikes_in_time guard so both paths report
    # identically. The legitimate n_time < 1 "span smaller than one bin" check
    # below still covers a valid positive dt with a too-short span.
    dt = validate_dt(dt)

    dtype = _validate_session_dtype(dtype)

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

    # Decode window — computed ONCE and reused for both the out-of-window drop
    # check and the time-grid construction so they agree exactly.
    t_start = float(times_arr.min())
    t_stop = float(times_arr.max())

    firing_rates = cast("NDArray[np.float64]", np.asarray(encoding_models, dtype=dtype))
    if warn_spikes_outside:
        _warn_if_spikes_out_of_window(trains, t_start, t_stop)

    from neurospatial._intervals import TimeSupport, interval_valid_mask
    from neurospatial._time_bins import _time_bins_in_windows

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

    return trains, firing_rates, bin_left, bin_right


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
    dtype = _validate_session_dtype(dtype)
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
    context: str = "decode_session",
    **decode_kwargs: Any,
) -> DecodingResult:
    """Decode existing models on the per-run recording clock."""
    from neurospatial.decoding.posterior import decode_position

    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    trains, firing_rates, bin_left, bin_right = _prepare_session_decode(
        spike_times,
        times,
        encoding_models,
        dt=dt,
        max_gap=max_gap,
        epochs=resolved_epochs,
        break_before=break_before,
        spike_window=resolved_spike_window,
        sample_valid=sample_valid,
        warn_spikes_outside=warn_spikes_outside,
        warn_partial_bins=warn_partial_bins,
        dtype=dtype,
        context=context,
    )
    from neurospatial.decoding._support import models_on_support, resolve_prior

    decode_kwargs["prior"] = resolve_prior(
        env, decode_kwargs.get("prior"), spatial_support, len(bin_left), context=context
    )
    firing_rates = models_on_support(firing_rates, spatial_support)
    counts = _count_spikes_in_time_bins(trains, bin_left, bin_right)
    centers = bin_left + dt / 2.0
    return decode_position(
        env, counts, firing_rates, dt=dt, times=centers, dtype=dtype, **decode_kwargs
    )._evolve(spike_window=resolved_spike_window)


_SUMMARY_TIME_CHUNK_NONE_MSG = (
    "time_chunk=None is not allowed for decode_session_summary: this "
    "streamed summary decoder bins time and reduces the posterior one "
    "time-block at a time, and None would materialize the full "
    "(n_time, n_bins) posterior, defeating its purpose. Use "
    "decode_session if you want the full posterior, or pass a positive "
    "time_chunk (default 1024) here."
)


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

    The encoding model (firing rates, shape ``(n_neurons, n_bins)``) is built
    once over the whole session (it is small). Then time is processed in blocks
    of ``time_chunk`` bins: each block bins ONLY that block's spikes (a
    slice of the observed-run time grid) and decodes + reduces it via the
    SAME shared inner-loop helper as
    :func:`~neurospatial.decoding.decode_position_summary`. Peak memory is
    therefore ``O(time_chunk * max(n_neurons, n_bins))`` plus the
    ``(n_neurons, n_bins)`` encoding model, **independent of session length**.
    The result is identical to running
    :func:`~neurospatial.decoding.decode_position_summary` on the fully
    materialized count matrix.

    Parameters
    ----------
    env, spike_times, times, positions, dt, bandwidth, method, \
min_occupancy, penalty, rank, speed, min_speed, max_gap, \
warn_on_drop, dtype
        Same as :func:`decode_session` -- including ``method="glm"`` and its
        ``penalty`` / ``rank`` knobs, and the nullable ``bandwidth`` /
        ``min_occupancy`` (``max_gap`` gates encoding and decoding). ``dtype``
        ("decode in this dtype") controls BOTH the encoding-model working set
        AND the streamed per-block posterior: ``np.float32`` halves both;
        default ``np.float64`` is byte-for-byte unchanged. Pass it via this
        explicit parameter, NOT via ``decode_kwargs``.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
    break_before : ArrayLike of bool, shape (n_frames,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from encoding and decoding, as for a gap longer than
        ``max_gap``: no decode time bin spans it and spikes inside it are not
        counted.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    **decode_kwargs
        Forwarded to the per-block decode (same semantics as
        :func:`~neurospatial.decoding.decode_position_summary`): ``prior``,
        ``validate``, and ``time_chunk`` (the streaming
        block size; a positive integer, defaults to 1024 — ``None`` is rejected
        because it would materialize the full posterior; use
        :func:`decode_session` for the full posterior). ``dtype`` is the
        explicit parameter above, not a ``decode_kwargs`` entry. Unknown kwargs
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
    ``max_gap`` is not decoded, as without ``spike_window`` spikes are assumed
    recorded only while position was. To decode spans without tracking, fit
    a :class:`~neurospatial.decoding.BayesianDecoder` and predict with
    ``times=np.arange(t0, t1, dt)``.

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
    # Reject unsupported reduction options before performing the encoding fit.
    unknown = set(decode_kwargs) - {"prior", "validate", "time_chunk"}
    if unknown:
        raise TypeError(
            f"decode_session_summary got unexpected keyword argument(s): {sorted(unknown)}"
        )
    from neurospatial.decoding.posterior import _validate_time_chunk

    if decode_kwargs.get("time_chunk", _SUMMARY_DEFAULT_TIME_CHUNK) is None:
        raise ValueError(_SUMMARY_TIME_CHUNK_NONE_MSG)
    _validate_time_chunk(
        decode_kwargs.get("time_chunk", _SUMMARY_DEFAULT_TIME_CHUNK), allow_none=False
    )
    validate_dt(dt)
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
        **decode_kwargs,
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
    context: str = "decode_session_summary",
    **decode_kwargs: Any,
) -> DecodingSummary:
    """Stream reductions over existing models and per-run time bins."""
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)

    from neurospatial.decoding._result import DecodingSummary
    from neurospatial.decoding.posterior import (
        _decode_and_reduce_block,
        _prepare_decode_inputs,
        _validate_time_chunk,
    )

    # Split out the decode-time knobs from decode_kwargs; everything else is an
    # unknown kwarg and must error rather than be silently dropped.
    prior = decode_kwargs.pop("prior", None)
    validate = decode_kwargs.pop("validate", True)
    time_chunk = decode_kwargs.pop("time_chunk", _SUMMARY_DEFAULT_TIME_CHUNK)
    if decode_kwargs:
        raise TypeError(
            f"decode_session_summary got unexpected keyword argument(s): "
            f"{sorted(decode_kwargs)}. Supported decode kwargs are prior, "
            f"validate, time_chunk (dtype is an explicit parameter)."
        )
    # The Poisson observation model is the only supported likelihood. It is fixed
    # here rather than read from decode_kwargs because ``method`` now names the
    # smoothing estimator (the explicit parameter forwarded to the encoder).
    likelihood_method: Literal["poisson"] = "poisson"

    if time_chunk is None:
        raise ValueError(_SUMMARY_TIME_CHUNK_NONE_MSG)
    time_chunk = _validate_time_chunk(time_chunk, allow_none=False)

    trains, firing_rates, bin_left, bin_right = _prepare_session_decode(
        spike_times,
        times,
        encoding_models,
        dt=dt,
        max_gap=max_gap,
        epochs=resolved_epochs,
        break_before=break_before,
        spike_window=resolved_spike_window,
        sample_valid=sample_valid,
        warn_spikes_outside=warn_spikes_outside,
        warn_partial_bins=warn_partial_bins,
        dtype=dtype,
        context=context,
    )

    from neurospatial.decoding._support import (
        models_on_support,
        relative_bin_measures,
        resolve_prior,
    )

    prior = resolve_prior(env, prior, spatial_support, len(bin_left), context=context)
    firing_rates = models_on_support(firing_rates, spatial_support)
    n_time = bin_left.size
    bin_centers_time = bin_left + dt / 2.0

    # Validate the encoding model once against a single row of counts, so the
    # neuron-count check runs. The streamed per-block counts come from
    # _count_spikes_in_time_bins and are non-negative integers by construction.
    _, firing_rates, prior = _prepare_decode_inputs(
        env,
        np.zeros((1, firing_rates.shape[0]), dtype=np.int64),
        firing_rates,
        prior=prior,
        method=likelihood_method,
        validate=validate,
        context=context,
    )
    prior_is_time_varying = prior is not None and np.ndim(prior) == 2

    from neurospatial.decoding.estimates import (
        _finish_posterior_mean,
        _posterior_mean_basis,
    )

    bin_centers = np.asarray(env.bin_centers, dtype=np.float64)
    mean_basis = _posterior_mean_basis(env)
    measures = relative_bin_measures(env, context=context)
    n_dims = bin_centers.shape[1]

    map_bin = np.empty(n_time, dtype=np.int64)
    mean_position = np.empty((n_time, n_dims), dtype=np.float64)
    posterior_entropy = np.empty(n_time, dtype=np.float64)
    peak_prob = np.empty(n_time, dtype=np.float64)

    # time_chunk is guaranteed a positive int by the up-front guard, so the
    # streamed-binning loop and the posterior reduction below both stay bounded
    # — the full (n_time, n_bins) posterior is never materialized in one shot.
    block = time_chunk
    for start in range(0, n_time, block):
        stop = min(start + block, n_time)
        lo, hi = bin_left[start], bin_right[stop - 1]
        block_trains = [s[(s >= lo) & (s < hi)] for s in trains]
        counts_block = _count_spikes_in_time_bins(
            block_trains, bin_left[start:stop], bin_right[start:stop]
        )

        block_prior = prior
        if prior_is_time_varying:
            block_prior = np.asarray(prior)[start:stop]

        (
            block_map_bin,
            block_mean,
            block_entropy,
            block_peak,
        ) = _decode_and_reduce_block(
            counts_block,
            firing_rates,
            dt,
            bin_centers,
            prior_block=block_prior,
            validate=validate,
            dtype=dtype,
            mean_basis=mean_basis,
            measures=measures,
        )
        map_bin[start:stop] = block_map_bin
        mean_position[start:stop] = block_mean
        posterior_entropy[start:stop] = block_entropy
        peak_prob[start:stop] = block_peak
        # counts_block / block posterior go out of scope before the next block.

    mean_position = _finish_posterior_mean(env, mean_position)
    return DecodingSummary._from_owned(
        times=bin_centers_time,
        mean_position=mean_position,
        posterior_entropy=posterior_entropy,
        peak_probability=peak_prob,
        map_bin=map_bin,
        env=env,
        spike_window=resolved_spike_window,
    )
