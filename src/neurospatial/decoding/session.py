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
from neurospatial._intervals import resolve_time_windows
from neurospatial.decoding._binning import count_spikes_in_time_bins

# Warn when more than this fraction of spikes fall outside the decode window.
# Always < 1.0, so the 100%-dropped case (frac == 1.0) always warns.
_DROP_WARN_THRESHOLD = 0.5

# Default time-block size for the streaming summary path. Matches
# decode_position_summary's default ``time_chunk`` so the two paths block the
# time axis identically.
_SUMMARY_DEFAULT_TIME_CHUNK = 1024

if TYPE_CHECKING:
    from neurospatial.decoding._result import DecodingResult, DecodingSummary
    from neurospatial.environment import Environment


def _warn_if_spikes_out_of_window(
    trains: list[NDArray[np.float64]],
    t_start: float,
    t_stop: float,
) -> None:
    """Emit one UserWarning if most spikes fall outside the decode window.

    Aggregates across all spike trains. Warns (does not raise) when the
    dropped fraction exceeds ``_DROP_WARN_THRESHOLD`` (which also covers the
    all-dropped case, since ``1.0 > 0.5``). A genuinely empty session is
    legitimate, so an empty input never warns.

    The message mirrors the wording of the encoding-path warning
    (``_emit_time_window_warning`` in ``neurospatial.encoding._binning``):
    it names the dropped count and total, the percentage, the decode time
    window, the spike range, the units hypothesis, and the escape hatch.

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
    if n_out == 0:
        return

    frac = n_out / total
    if frac <= _DROP_WARN_THRESHOLD:
        return

    nonempty = [t for t in trains if t.size > 0]
    if nonempty:
        all_spikes = np.concatenate(nonempty)
        range_part = (
            f"spike_times.min()={all_spikes.min():.6g} "
            f"spike_times.max()={all_spikes.max():.6g}. "
        )
    else:
        range_part = ""

    warnings.warn(
        f"{n_out}/{total} spike_times "
        f"({100 * frac:.0f}%) fell outside the decode time window "
        f"[{t_start:.6g}, {t_stop:.6g}]; "
        f"{range_part}"
        f"Check that spike_times and times share units (both seconds). "
        f"Dropped spikes do not contribute to the posterior. "
        f"Set warn_on_drop=False to suppress this warning.",
        UserWarning,
        stacklevel=2,
    )


def decode_session(
    env: Environment,
    spike_times: Any,
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
    max_gap: float | None = 0.5,
    epochs: Any = None,
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
        Fitted spatial environment that defines the bin layout and
        connectivity graph.
    spike_times : array or sequence of arrays
        Spike times for one or more neurons.  Accepted formats mirror
        :func:`~neurospatial.encoding.as_spike_trains`:

        - 1-D array / list of scalars → single neuron
        - 2-D array, shape ``(n_neurons, max_spikes)``, NaN-padded
        - List/tuple of 1-D arrays → one array per neuron (canonical)
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
        (5.0); a ratio-only param, so it must stay ``None`` when
        ``method="glm"``.
    method : str, optional
        Estimator passed to :func:`~neurospatial.encoding.compute_spatial_rates`.
        Options: ``"diffusion_kde"`` (default), ``"gaussian_kde"``, ``"binned"``,
        and ``"glm"`` (penalized-Poisson GAM, tuned with ``penalty`` / ``rank``).
    min_occupancy : float or None, optional
        Minimum occupancy (seconds) for a spatial bin to be included in the
        ratio-method encoding model. Bins below threshold are set to
        ``fill_value=0.0`` so the decoder never receives NaN rates. ``None``
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
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals are excluded from encoding and decoding.
        ``None`` disables the gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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
        therefore silently excluded from the count matrix.  This guards the
        common units footgun — ``spike_times`` in milliseconds while ``times``
        is in seconds — which would otherwise produce an all-zero count matrix
        and a plausible-but-wrong posterior.  The warning fires exactly once
        per call, from the encoder. This also covers spikes mapped outside
        the environment. Set ``False`` to suppress the coverage diagnostics.
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
        Propagated from the underlying helpers if inputs are invalid (e.g.
        ``dt`` is not finite/positive, ``t_stop <= t_start``, or the
        encoding model has no finite bins).

    Notes
    -----
    Decode time bins are formed separately within each run of samples whose
    gaps are no longer than ``max_gap`` and that lie inside ``epochs`` and
    ``spike_window``; no bin spans a pause, and spikes between runs are not
    counted. ``result.times`` may therefore be non-contiguous.

    **Orientation contract**:
    :func:`~neurospatial.decoding.bin_spikes_in_time` returns a count
    matrix of shape ``(n_time_bins, n_neurons)`` (default
    ``orient="time_x_neuron"``), which is exactly what
    :func:`~neurospatial.decoding.decode_position` expects for its
    ``spike_counts`` argument.  No transposition is performed.

    **Encoding fill value**:
    When a ratio method is used, this
    function passes ``fill_value=0.0`` to the encoder so that low-occupancy bins
    produce zero-rate predictions rather than NaN, keeping the posterior valid.
    ``method="glm"`` needs no fill (occupancy enters as a log-offset, so every
    bin gets a finite rate), so no ``fill_value`` is passed there. If you need
    NaN-masked bins in the encoding model, compute
    :func:`~neurospatial.encoding.compute_spatial_rates` separately and
    use ``BayesianDecoder.from_rates(rates).predict(spike_times, times)``.
    For explicit rate arrays, bin counts with ``bin_spikes_in_time`` and pass
    the count array and ``rates.firing_rates`` to ``decode_position``.

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
    >>> env.units = "cm"  # required by simulate_trajectory_ou
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
    >>> rates = compute_spatial_rates(
    ...     env, spike_times, times, positions, bandwidth=5.0, fill_value=0.0
    ... )
    >>> counts, centers = bin_spikes_in_time(
    ...     spike_times, dt=0.1, t_start=times[0], t_stop=times[-1]
    ... )
    >>> result = decode_position(env, counts, rates.firing_rates, 0.1, times=centers)
    """
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    trains, firing_rates, _, _ = _build_encoding_model(
        env,
        spike_times,
        times,
        positions,
        dt=dt,
        bandwidth=bandwidth,
        method=method,
        min_occupancy=min_occupancy,
        penalty=penalty,
        rank=rank,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=resolved_epochs,
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
        spike_window=resolved_spike_window,
        warn_on_drop=False,
        dtype=dtype,
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


def _prepare_session_decode(
    spike_times: Any,
    times: ArrayLike,
    encoding_models: NDArray[np.float64],
    *,
    dt: float,
    max_gap: float | None = 0.5,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    warn_on_drop: bool,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    context: str = "decode_session",
) -> tuple[
    list[NDArray[np.float64]],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Normalize spike inputs/models and tile bins inside observed runs."""
    from neurospatial.decoding._binning import validate_dt
    from neurospatial.encoding._spikes import as_spike_trains_with_ids
    from neurospatial.encoding._validation import validate_times

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
    trains, _ = as_spike_trains_with_ids(spike_times)
    times_arr = np.asarray(times, dtype=np.float64)
    if times_arr.ndim != 1:
        raise ValueError(
            f"times must be a 1-D array of timestamps for decode_session, "
            f"got shape {times_arr.shape}."
        )
    # Require finite, sorted timestamps before constructing observed runs.
    validate_times(times_arr, context=context)

    # Decode window — computed ONCE and reused for both the out-of-window drop
    # check and the time-grid construction so they agree exactly.
    t_start = float(times_arr.min())
    t_stop = float(times_arr.max())

    firing_rates = cast("NDArray[np.float64]", np.asarray(encoding_models, dtype=dtype))
    if warn_on_drop:
        _warn_if_spikes_out_of_window(trains, t_start, t_stop)

    from neurospatial._intervals import run_time_bounds
    from neurospatial.decoding._binning import time_bins_in_windows
    from neurospatial.environment.trajectory import interval_valid_mask

    # Decode bins exist only where the recording was observed: the gap, epochs and
    # spike_window gates. The speed and out-of-bounds gates restrict only the
    # ENCODING step, so periods of immobility (for example replay) are still
    # decoded.
    observed = interval_valid_mask(
        times_arr, max_gap=max_gap, epochs=epochs, spike_window=spike_window
    )
    runs = run_time_bounds(times_arr, observed)
    bin_left, bin_right = time_bins_in_windows(runs, dt)
    if bin_left.size == 0:
        longest = float(np.max(np.diff(runs, axis=1), initial=0.0))
        raise ValueError(
            f"No decode time bin fits: the {runs.shape[0]} observed recording "
            f"run(s) are at most {longest:.3g} s long, shorter than dt={dt}. "
            f"\nWhy: time bins are formed only inside runs of samples with gaps "
            f"<= max_gap={max_gap} s that lie inside epochs and spike_window. "
            f"\nFix: use a smaller dt, widen epochs/spike_window, or pass a larger "
            f"max_gap (max_gap=None decodes across gaps)."
        )

    return trains, firing_rates, bin_left, bin_right


def _build_encoding_model(
    env: Environment,
    spike_times: Any,
    times: ArrayLike,
    positions: NDArray[np.float64],
    *,
    dt: float,
    bandwidth: float | None,
    method: str,
    min_occupancy: float | None,
    penalty: float | None = None,
    rank: int | None = None,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    warn_on_drop: bool,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    context: str = "decode_session",
) -> tuple[
    list[NDArray[np.float64]],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Encode tracking data, then prepare the shared observed-run decode bins."""
    from neurospatial._validation import validate_times_positions
    from neurospatial.decoding._binning import validate_dt
    from neurospatial.encoding._spikes import as_spike_trains_with_ids
    from neurospatial.encoding.spatial import compute_spatial_rates

    times_arr, positions = validate_times_positions(times, positions, call=context)
    dt = validate_dt(dt)
    dtype = _validate_session_dtype(dtype)
    trains, _ = as_spike_trains_with_ids(spike_times)
    # Mirror the encoder's method-specific validation (mutual exclusivity +
    # value domains) at the decoder boundary, reusing the SAME validator so
    # the errors are identical. fill_value is not a decoder-exposed param, so
    # it is passed as None here (the golden-path 0.0 fill for ratio methods is
    # applied in the compute_spatial_rates call below, never for glm).
    from neurospatial.encoding._smoothing import validate_spatial_method_params

    penalty, rank = validate_spatial_method_params(
        method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        fill_value=None,
        penalty=penalty,
        rank=rank,
    )
    _method = cast("Literal['diffusion_kde', 'gaussian_kde', 'binned', 'glm']", method)
    # glm produces finite rates everywhere (occupancy is a log-offset), so it
    # needs no NaN fill; passing fill_value to a glm result would be rejected
    # as a ratio-only param. Ratio methods keep the golden-path 0.0 fill so
    # low-occupancy bins decode as zero-rate, never NaN.
    fill_value = None if method == "glm" else 0.0
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
        fill_value=fill_value,
        penalty=penalty,
        rank=rank,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        warn_on_drop=warn_on_drop,
        dtype=dtype,
        epochs=epochs,
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
    prepared = _prepare_session_decode(
        trains,
        times_arr,
        firing_rates,
        dt=dt,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        warn_on_drop=False,
        dtype=dtype,
        context=context,
    )
    # After the decode-bin check, whose error names epochs/max_gap precisely:
    # time bins can exist while speed or occupancy gates leave no trained bin.
    occupancy = np.asarray(rates_result.occupancy)
    if not np.any((occupancy > 0) & (occupancy >= (min_occupancy or 0.0))):
        raise ValueError(
            _format_error(
                f"{context}: the encoding model has no occupied bin, so every "
                f"decode time bin would get a uniform posterior.",
                why=(
                    "Why: the gates (max_gap, min_speed, epochs, spike_window "
                    "and min_occupancy) left no training interval with time "
                    "in any kept bin."
                ),
                fix=(
                    "check that times, epochs and spike_window share one "
                    "clock in seconds, lower min_speed or min_occupancy, or "
                    "pass max_gap=None for coarsely sampled tracking"
                ),
            )
        )
    return prepared


def _decode_with_models(
    env: Environment,
    spike_times: Any,
    times: ArrayLike,
    encoding_models: NDArray[np.float64],
    *,
    dt: float = 0.025,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    warn_on_drop: bool = True,
    dtype: type[np.float32] | type[np.float64] = np.float64,
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
        spike_window=resolved_spike_window,
        warn_on_drop=warn_on_drop,
        dtype=dtype,
    )
    counts = count_spikes_in_time_bins(trains, bin_left, bin_right)
    centers = bin_left + dt / 2.0
    return decode_position(
        env, counts, firing_rates, dt, times=centers, dtype=dtype, **decode_kwargs
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
    spike_times: Any,
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
    max_gap: float | None = 0.5,
    epochs: Any = None,
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
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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
    counted. ``result.times`` may therefore be non-contiguous.

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
    trains, firing_rates, _, _ = _build_encoding_model(
        env,
        spike_times,
        times,
        positions,
        dt=dt,
        bandwidth=bandwidth,
        method=method,
        min_occupancy=min_occupancy,
        penalty=penalty,
        rank=rank,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=resolved_epochs,
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
        spike_window=resolved_spike_window,
        warn_on_drop=False,
        dtype=dtype,
        **decode_kwargs,
    )


def _decode_with_models_summary(
    env: Environment,
    spike_times: Any,
    times: ArrayLike,
    encoding_models: NDArray[np.float64],
    *,
    dt: float = 0.025,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    warn_on_drop: bool = True,
    dtype: type[np.float32] | type[np.float64] = np.float64,
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
        spike_window=resolved_spike_window,
        warn_on_drop=warn_on_drop,
        dtype=dtype,
        context="decode_session_summary",
    )

    n_time = bin_left.size
    bin_centers_time = bin_left + dt / 2.0

    # Validate the encoding model + resolve the non-finite mask ONCE (the same
    # front-half decode_position_summary runs). spike_counts is faked with a
    # zero-row block here only to satisfy the helper's interface; its actual
    # per-block counts come from the streamed binning below. We pass a
    # (1, n_neurons) row so the neuron-count agreement check still fires.
    #
    # NOTE: the real per-block counts produced by the streamed binning below are
    # intentionally NOT routed through _validate_inputs. They come straight from
    # count_spikes_in_time_bins on float spike times, so they are non-negative int64 by
    # construction (cannot be fractional, negative, or NaN) — the value checks
    # _validate_inputs performs are already guaranteed, so the exemption is
    # deliberate, not an oversight.
    n_neurons = firing_rates.shape[0]
    _dummy_counts = np.zeros((1, n_neurons), dtype=np.int64)
    _checked_counts, firing_rates, nonfinite_mask = _prepare_decode_inputs(
        env,
        _dummy_counts,
        firing_rates,
        prior=prior,
        method=likelihood_method,
        validate=validate,
        context="decode_session_summary",
    )

    # Validate prior shape ONCE, up front, against the GLOBAL (n_time, n_bins)
    # grid — mirrors decode_position_summary so an over-long 2-D prior raises
    # here instead of being silently truncated by the block loop (R2).
    n_bins = firing_rates.shape[1]
    prior_is_time_varying = False
    if prior is not None:
        prior_arr = np.asarray(prior)
        if prior_arr.ndim == 1:
            if prior_arr.shape[0] != n_bins:
                raise ValueError(
                    f"1D prior must have shape ({n_bins},) to match the number "
                    f"of position bins, got shape {prior_arr.shape}"
                )
        elif prior_arr.ndim == 2:
            if prior_arr.shape != (n_time, n_bins):
                raise ValueError(
                    f"2D prior must have shape {(n_time, n_bins)} to match the "
                    f"({n_time} time bins, {n_bins} position bins) being "
                    f"decoded, got shape {prior_arr.shape}"
                )
            prior_is_time_varying = True
        else:
            raise ValueError(
                f"prior must be 1D (stationary) or 2D (time-varying), "
                f"got {prior_arr.ndim}D with shape {prior_arr.shape}"
            )

    bin_centers = np.asarray(env.bin_centers, dtype=np.float64)
    n_dims = bin_centers.shape[1]

    map_bin = np.empty(n_time, dtype=np.int64)
    map_position = np.empty((n_time, n_dims), dtype=np.float64)
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
        counts_block = count_spikes_in_time_bins(
            block_trains, bin_left[start:stop], bin_right[start:stop]
        )

        block_prior = prior
        if prior_is_time_varying:
            block_prior = np.asarray(prior)[start:stop]

        (
            block_map_bin,
            block_map_position,
            block_mean,
            block_entropy,
            block_peak,
        ) = _decode_and_reduce_block(
            counts_block,
            firing_rates,
            dt,
            bin_centers,
            prior_block=block_prior,
            nonfinite_mask=nonfinite_mask,
            validate=validate,
            dtype=dtype,
        )
        map_bin[start:stop] = block_map_bin
        map_position[start:stop] = block_map_position
        mean_position[start:stop] = block_mean
        posterior_entropy[start:stop] = block_entropy
        peak_prob[start:stop] = block_peak
        # counts_block / block posterior go out of scope before the next block.

    return DecodingSummary(
        times=bin_centers_time,
        map_position=map_position,
        mean_position=mean_position,
        posterior_entropy=posterior_entropy,
        peak_prob=peak_prob,
        map_bin=map_bin,
        env=env,
        spike_window=resolved_spike_window,
    )
