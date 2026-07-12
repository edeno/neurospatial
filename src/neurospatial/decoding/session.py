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

# Warn when more than this fraction of spikes fall outside the decode window.
# Always < 1.0, so the 100%-dropped case (frac == 1.0) always warns.
_DROP_WARN_THRESHOLD = 0.5

# Default time-block size for the streaming summary path. Matches
# decode_position_summary's default ``time_chunk`` so the two paths block the
# time axis identically.
_SUMMARY_DEFAULT_TIME_CHUNK = 1024

if TYPE_CHECKING:
    from neurospatial._typing import PositionLike
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
    times: ArrayLike | PositionLike,
    positions: NDArray[np.float64] | None = None,
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
    encoding_models: NDArray[np.float64] | None = None,
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
    times : array-like, shape (n_frames,), or PositionLike
        Timestamps (seconds) at which ``positions`` were recorded.
        Used both to build encoding models and to set the decoding time
        grid via ``t_start = times.min()`` / ``t_stop = times.max()``. May
        instead be a single ``PositionLike`` object (exposing ``.t`` and
        ``.values``, e.g. a pynapple ``Tsd`` / ``TsdFrame``) carrying both
        times and positions, in which case ``positions`` must be omitted.
    positions : NDArray[np.float64], shape (n_frames, n_dims), optional
        Animal position at each frame in ``times``. Omit only when ``times``
        is a ``PositionLike`` object carrying the positions.
    dt : float, optional
        Decoding time-bin width in seconds.  Default is 0.025 (25 ms).
    bandwidth : float or None, optional
        Smoothing bandwidth (same units as positions) for the ratio-method
        encoding step. ``None`` (default) resolves to the encoder's default
        (5.0); a ratio-only param, so it must stay ``None`` when
        ``method="glm"``. Ignored when ``encoding_models`` is provided.
    method : str, optional
        Estimator passed to :func:`~neurospatial.encoding.compute_spatial_rates`.
        Options: ``"diffusion_kde"`` (default), ``"gaussian_kde"``, ``"binned"``,
        and ``"glm"`` (penalized-Poisson GAM, tuned with ``penalty`` / ``rank``).
        Ignored when ``encoding_models`` is provided.
    min_occupancy : float or None, optional
        Minimum occupancy (seconds) for a spatial bin to be included in the
        ratio-method encoding model. Bins below threshold are set to
        ``fill_value=0.0`` so the decoder never receives NaN rates. ``None``
        (default) resolves to the encoder's default (0.0, no threshold); a
        ratio-only param, so it must stay ``None`` when ``method="glm"``. Ignored
        when ``encoding_models`` is provided.
    penalty : float or None, optional
        ``method="glm"`` smoothness penalty ``lambda``. ``None`` (default)
        chooses it by REML. Mutually exclusive with the ratio params
        (``bandwidth`` / ``min_occupancy``). Ignored when ``encoding_models`` is
        provided.
    rank : int or None, optional
        ``method="glm"`` requested basis rank cap. ``None`` (default) uses the
        encoder default. Ignored when ``encoding_models`` is provided.
    speed : NDArray[np.float64], shape (n_frames,) or None
        Precomputed instantaneous speed at each trajectory sample, forwarded to
        :func:`~neurospatial.encoding.compute_spatial_rates`. Only used when
        ``min_speed`` is set; auto-derived when ``None``. Ignored when
        ``encoding_models`` is provided.
    min_speed : float or None
        Minimum speed threshold (physical units / second), forwarded to the
        encoding step so the decode golden path can speed-filter encoding. When
        set, low-speed periods are excluded from BOTH the spike numerator and
        the occupancy denominator of the encoding model via one shared gate.
        When ``None`` (default) no speed filtering is applied (unchanged).
        Ignored when ``encoding_models`` is provided.
    max_gap : float or None, optional
        Maximum trajectory time gap (seconds), forwarded to the encoding step
        (see :func:`~neurospatial.encoding.compute_spatial_rates`). Intervals
        with ``dt > max_gap`` are dropped from BOTH the spike numerator and the
        occupancy denominator of the encoding model. Default 0.5 (matches
        ``compute_spatial_rates``); pass ``None`` to count all intervals
        regardless of gap size (e.g. for intentionally-gappy data). Ignored
        when ``encoding_models`` is provided.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins) or None
        Pre-computed place-field firing-rate maps.  When provided, the
        encoding step (``compute_spatial_rates``) is skipped entirely and
        this array is passed directly to the decoder.  Useful for re-using
        models across multiple decoding passes or for injecting custom
        encoding models.  ``bandwidth``, ``method``, and
        ``min_occupancy`` are ignored when this is set.
    warn_on_drop : bool, optional
        If ``True`` (the default), emit a single ``UserWarning`` when a large
        fraction (>50%, which includes the all-dropped case) of spikes fall
        outside the decode time window ``[times.min(), times.max()]`` and are
        therefore silently excluded from the count matrix.  This guards the
        common units footgun — ``spike_times`` in milliseconds while ``times``
        is in seconds — which would otherwise produce an all-zero count matrix
        and a plausible-but-wrong posterior.  The warning fires exactly once
        per call: in the ``encoding_models=None`` branch the encoder
        (``compute_spatial_rates``) emits it (and additionally warns when
        spikes map to inactive bins / the wrong coordinate frame); in the
        ``encoding_models``-provided branch the encoder is skipped, so
        ``decode_session`` performs the out-of-window check itself.  Set to
        ``False`` to suppress these warnings (e.g. for a genuinely sparse
        session) — note this also silences the encoder's inactive-bin warning.
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
    **Orientation contract**:
    :func:`~neurospatial.decoding.bin_spikes_in_time` returns a count
    matrix of shape ``(n_time_bins, n_neurons)`` (default
    ``orient="time_x_neuron"``), which is exactly what
    :func:`~neurospatial.decoding.decode_position` expects for its
    ``spike_counts`` argument.  No transposition is performed.

    **Encoding fill value**:
    When ``encoding_models`` is not provided and a ratio method is used, this
    function passes ``fill_value=0.0`` to the encoder so that low-occupancy bins
    produce zero-rate predictions rather than NaN, keeping the posterior valid.
    ``method="glm"`` needs no fill (occupancy enters as a log-offset, so every
    bin gets a finite rate), so no ``fill_value`` is passed there. If you need
    NaN-masked bins in the encoding model, compute
    :func:`~neurospatial.encoding.compute_spatial_rates` separately and
    pass the result as ``encoding_models``.

    **Time grid**:
    The decoding grid spans ``[times.min(), times.max()]`` in steps of
    ``dt``.  Spikes outside this window are excluded by
    :func:`~neurospatial.decoding.bin_spikes_in_time`; when a large fraction
    fall outside (the usual sign of a milliseconds-vs-seconds unit mismatch)
    a single ``UserWarning`` is emitted naming the window, the dropped
    fraction, and the spike range (unless ``warn_on_drop=False``).

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
    >>> env = Environment.from_samples(positions_raw, bin_size=5.0)  # doctest: +SKIP
    >>> env.units = "cm"  # required by simulate_trajectory_ou  # doctest: +SKIP
    >>> positions, times = simulate_trajectory_ou(  # doctest: +SKIP
    ...     env, duration=10.0, speed_units="cm", seed=0
    ... )
    >>> n_neurons = 10
    >>> spike_times = [  # doctest: +SKIP
    ...     generate_poisson_spikes(
    ...         PlaceCellModel(env, width=15.0, seed=i).firing_rate(positions, times),
    ...         times,
    ...         seed=i,
    ...     )
    ...     for i in range(n_neurons)
    ... ]
    >>> result = decode_session(  # doctest: +SKIP
    ...     env, spike_times, times, positions, dt=0.1
    ... )
    >>> result.posterior.shape  # doctest: +SKIP
    (n_time_bins, n_bins)
    >>> result.map_position.shape  # doctest: +SKIP
    (n_time_bins, 2)

    Re-use precomputed encoding models across multiple sessions:

    >>> from neurospatial.encoding import compute_spatial_rates  # doctest: +SKIP
    >>> models = compute_spatial_rates(  # doctest: +SKIP
    ...     env,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     bandwidth=5.0,
    ...     fill_value=0.0,
    ... ).firing_rates  # shape (n_neurons, n_bins)
    >>> result = decode_session(  # doctest: +SKIP
    ...     env,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     dt=0.1,
    ...     encoding_models=models,
    ... )
    """
    from neurospatial.decoding.posterior import decode_position

    firing_rates, counts, centers = _encode_and_bin(
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
        encoding_models=encoding_models,
        warn_on_drop=warn_on_drop,
        dtype=dtype,
    )

    # --- Decode ---
    # Forward the explicit `dtype` as the single source for the posterior dtype.
    # It is intentionally NOT left in decode_kwargs, so there is no duplicate
    # `dtype` keyword (which would be a TypeError at call time).
    return decode_position(
        env,
        counts,
        firing_rates,
        dt,
        times=centers,
        dtype=dtype,
        **decode_kwargs,
    )


def _build_encoding_model(
    env: Environment,
    spike_times: Any,
    times: ArrayLike | PositionLike,
    positions: NDArray[np.float64] | None,
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
    encoding_models: NDArray[np.float64] | None,
    warn_on_drop: bool,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    context: str = "decode_session",
) -> tuple[
    list[NDArray[np.float64]],
    NDArray[np.float64],
    int,
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Encode-only glue: build firing rates and the global decode time grid.

    Does everything :func:`_encode_and_bin` does **except** building the full
    ``(n_time, n_neurons)`` count matrix. It normalizes spike trains, validates
    timestamps, builds (or accepts) the encoding models, emits the
    units-footgun warning, and computes the global decode time grid
    (``n_time``, the global bin ``edges``, and ``bin_centers``) using the SAME
    grid math as :func:`~neurospatial.decoding.bin_spikes_in_time`. The
    streaming summary path uses this to bin spikes block-by-block against the
    GLOBAL edges (a contiguous edge slice per block, NOT recomputed per block),
    so the dense count matrix is never materialized AND the per-block counts are
    byte-for-byte identical to a single global histogram. (Recomputing edges as
    ``block_t_start + k*dt`` per block would drift by float rounding and break
    parity; slicing the precomputed global ``edges`` does not.)

    The ``dtype`` knob controls the dtype of the returned ``firing_rates``
    (the encoding-model working set): in the computed branch it is threaded
    into :func:`~neurospatial.encoding.compute_spatial_rates` so the model is
    built directly in that dtype (no promotion); in the passthrough branch the
    supplied ``encoding_models`` array is cast to that dtype (so ``dtype`` is
    authoritative end-to-end). Default ``np.float64`` keeps current behavior
    byte-for-byte.

    ``context`` names the caller in the up-front timestamp-validation error
    (``validate_times``) so a too-few-samples failure points at the real entry
    point. It defaults to ``"decode_session"`` (the message every existing
    caller already produced); ``BayesianDecoder.fit`` passes its own name so an
    epoch that selects too-few training samples names ``fit``, not the internal
    ``decode_session``.

    Returns
    -------
    trains : list of NDArray[np.float64]
        Normalized per-neuron spike-time arrays.
    firing_rates : NDArray[np.float64], shape (n_neurons, n_bins)
        Encoding-model firing-rate maps, in the requested ``dtype``.
    n_time : int
        Number of decode time bins on the global grid,
        ``floor((t_stop - t_start) / dt + 1e-9)``.
    edges : NDArray[np.float64], shape (n_time + 1,)
        Global decode time-bin edges, ``t_start + dt * arange(n_time + 1)``.
    bin_centers : NDArray[np.float64], shape (n_time,)
        Global decode time-bin centers (left edge + ``dt / 2``).
    """
    # Defer the `encoding` imports until call time: this keeps the decoding
    # package importable even if `encoding` were ever to import from `decoding`
    # (it does not today), so there is no circular-import risk at module load.
    # Mirrors how encoding/spatial.py defers its own heavy imports.
    from neurospatial._typing import _is_position_like, as_times_positions
    from neurospatial.decoding._binning import validate_dt
    from neurospatial.encoding import as_spike_trains_with_ids
    from neurospatial.encoding._validation import validate_times
    from neurospatial.encoding.spatial import compute_spatial_rates

    # Validate dt up front, BEFORE the grid math below builds the decode time
    # grid directly (bypassing bin_spikes_in_time's own guard). Without this,
    # invalid dt leaks cryptic errors: dt=0 → ZeroDivisionError; dt=NaN →
    # "cannot convert float NaN to integer"; dt<0 → a MISLEADING "span smaller
    # than one bin dt" message; dt=inf → a similar cryptic failure. Route
    # through the shared bin_spikes_in_time guard so both paths report
    # identically. The legitimate n_time < 1 "span smaller than one bin" check
    # below still covers a valid positive dt with a too-short span.
    dt = validate_dt(dt)

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

    # --- Normalize inputs ---
    # Boundary adapters: accept EITHER a PositionLike (e.g. a pynapple
    # Tsd/TsdFrame) OR explicit (times, positions) arrays, and a SpikeTrainsLike
    # group OR the canonical array formats. The scientific core below is
    # array-only; a plain-array caller is byte-for-byte unchanged. Decoding
    # results carry no unit axis, so extracted unit ids are intentionally
    # dropped here (identity is surfaced by the encoding path, not the decode).
    #
    # The position track is required only for the ENCODE step. When
    # ``encoding_models`` is supplied (passthrough decode) the positions are
    # never touched, so a caller may omit ``positions`` entirely — the
    # fitted-model decode path (e.g. ``BayesianDecoder.predict``) has no
    # position track to pass. In that one case we normalize only ``times``
    # (still handling a PositionLike, whose positions are simply unused);
    # otherwise the full ``(times, positions)`` normalization runs unchanged, so
    # every existing caller is byte-for-byte identical.
    if (
        positions is None
        and encoding_models is not None
        and not _is_position_like(times)
    ):
        times = np.asarray(times, dtype=np.float64)
    else:
        times, positions = as_times_positions(times, positions)
    trains, _ = as_spike_trains_with_ids(spike_times)
    times_arr = np.asarray(times, dtype=np.float64)
    if times_arr.ndim != 1:
        raise ValueError(
            f"times must be a 1-D array of timestamps for decode_session, "
            f"got shape {times_arr.shape}."
        )
    # Validate timestamps up front (>=2 samples, finite, sorted). This runs in
    # BOTH branches — in particular the encoding_models passthrough branch skips
    # the encoder's own validate_trajectory, so without this a NaN/inf in
    # `times` would leak a raw "cannot convert float NaN to integer" from
    # bin_spikes_in_time instead of a beginner-grade message.
    validate_times(times_arr, context=context)

    # Decode window — computed ONCE and reused for both the out-of-window drop
    # check and the time-grid construction so they agree exactly.
    t_start = float(times_arr.min())
    t_stop = float(times_arr.max())

    # --- Build encoding models if not provided ---
    # The units-footgun matters because the time-binning counts via
    # np.histogram(..., bins=edges), which silently drops spikes outside
    # [t_start, t_stop]; a ms-vs-s mismatch → all-zero counts → a
    # plausible-but-wrong posterior. Exactly one of the two branches below
    # surfaces it (never both, so no duplicate warning):
    if encoding_models is None:
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
        _method = cast(
            "Literal['diffusion_kde', 'gaussian_kde', 'binned', 'glm']", method
        )
        # glm produces finite rates everywhere (occupancy is a log-offset), so it
        # needs no NaN fill; passing fill_value to a glm result would be rejected
        # as a ratio-only param. Ratio methods keep the golden-path 0.0 fill so
        # low-occupancy bins decode as zero-rate, never NaN.
        fill_value = None if method == "glm" else 0.0
        # The encoder runs over the same [t_start, t_stop] window and already
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
        )
        # compute_spatial_rates already stores the result in `dtype`; the cast
        # is a cheap no-op guard so the working set is honored end-to-end. The
        # array is float32 OR float64; the declared NDArray[np.float64] return
        # type is the family annotation (cast keeps mypy happy).
        firing_rates = cast(
            "NDArray[np.float64]",
            np.asarray(rates_result.firing_rates, dtype=dtype),
        )
    else:
        # Passthrough: cast the supplied models to the requested dtype so
        # `dtype` is authoritative end-to-end (default np.float64 keeps existing
        # float64-in callers byte-for-byte unchanged).
        firing_rates = cast(
            "NDArray[np.float64]", np.asarray(encoding_models, dtype=dtype)
        )
        # Passthrough: the encoder was skipped, so nothing has checked the
        # spike/trajectory time window. Do the out-of-window check here so the
        # headline path still warns on a ms-vs-s mismatch before binning.
        if warn_on_drop:
            _warn_if_spikes_out_of_window(trains, t_start, t_stop)

    # --- Global decode time grid ---
    # Mirror bin_spikes_in_time exactly: n_bins = floor(span/dt + 1e-9), edges
    # at t_start + k*dt, centers at left-edge + dt/2. Computing it here (rather
    # than calling bin_spikes_in_time) lets the streaming path slice the grid
    # into blocks whose per-block bin() calls land on exactly these edges.
    n_time = int(np.floor((t_stop - t_start) / dt + 1e-9))
    if n_time < 1:
        raise ValueError(
            f"Span t_stop - t_start ({t_stop - t_start}) is smaller than one "
            f"bin dt ({dt}); no whole time bin fits."
        )
    edges = t_start + dt * np.arange(n_time + 1, dtype=np.float64)
    bin_centers = edges[:-1] + dt / 2.0

    return trains, firing_rates, n_time, edges, bin_centers


def _encode_and_bin(
    env: Environment,
    spike_times: Any,
    times: ArrayLike | PositionLike,
    positions: NDArray[np.float64] | None,
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
    encoding_models: NDArray[np.float64] | None,
    warn_on_drop: bool,
    dtype: type[np.float32] | type[np.float64] = np.float64,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Shared encode->bin glue for the FULL-posterior :func:`decode_session`.

    Builds (or accepts) the encoding models via :func:`_build_encoding_model`
    and bins spikes into a time-grid count matrix, returning
    ``(firing_rates, counts, centers)`` ready to hand to a decoder. This is the
    materialize-the-full-count-matrix path; :func:`decode_session` uses it
    unchanged. :func:`decode_session_summary` does NOT use this — it streams the
    binning (see :func:`_build_encoding_model`) so the full count matrix is
    never materialized.

    Returns
    -------
    firing_rates : NDArray[np.float64], shape (n_neurons, n_bins)
        Encoding-model firing-rate maps.
    counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike-count matrix (``orient="time_x_neuron"``).
    centers : NDArray[np.float64], shape (n_time_bins,)
        Decode time-bin centers (seconds).
    """
    trains, firing_rates, _n_time, edges, centers = _build_encoding_model(
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
        encoding_models=encoding_models,
        warn_on_drop=warn_on_drop,
        dtype=dtype,
    )

    # --- Bin spikes against the GLOBAL edges (orient="time_x_neuron") ---
    # Histogram each train against the precomputed global `edges`, exactly as
    # bin_spikes_in_time does internally. This yields the identical
    # (n_time, n_neurons) count matrix as before (same edges, same right-closed
    # last bin via np.histogram), with no behavior change for decode_session.
    counts = np.stack([np.histogram(s, bins=edges)[0] for s in trains], axis=1).astype(
        np.int64
    )
    # counts shape: (n_time_bins, n_neurons)  ← what decode_position expects
    return firing_rates, counts, centers


def decode_session_summary(
    env: Environment,
    spike_times: Any,
    times: ArrayLike | PositionLike,
    positions: NDArray[np.float64] | None = None,
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
    encoding_models: NDArray[np.float64] | None = None,
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
    contiguous slice of the global time grid) and decodes + reduces it via the
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
min_occupancy, penalty, rank, speed, min_speed, max_gap, encoding_models, \
warn_on_drop, dtype
        Same as :func:`decode_session` -- including ``method="glm"`` and its
        ``penalty`` / ``rank`` knobs, and the nullable ``bandwidth`` /
        ``min_occupancy`` (``max_gap`` forwards to
        :func:`~neurospatial.encoding.compute_spatial_rates`). ``dtype``
        ("decode in this dtype") controls BOTH the encoding-model working set
        AND the streamed per-block posterior: ``np.float32`` halves both;
        default ``np.float64`` is byte-for-byte unchanged. Pass it via this
        explicit parameter, NOT via ``decode_kwargs``.
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

    See Also
    --------
    decode_session : Full-posterior golden path.
    neurospatial.decoding.decode_position_summary : Array-first streamed decoder.
    """
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
        raise ValueError(
            "time_chunk=None is not allowed for decode_session_summary: this "
            "streamed summary decoder bins time and reduces the posterior one "
            "time-block at a time, and None would materialize the full "
            "(n_time, n_bins) posterior, defeating its purpose. Use "
            "decode_session if you want the full posterior, or pass a positive "
            "time_chunk (default 1024) here."
        )
    time_chunk = _validate_time_chunk(time_chunk, allow_none=False)

    # --- Encode once + build the global decode time grid (no count matrix) ---
    (
        trains,
        firing_rates,
        n_time,
        edges,
        bin_centers_time,
    ) = _build_encoding_model(
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
        encoding_models=encoding_models,
        warn_on_drop=warn_on_drop,
        dtype=dtype,
    )

    # Validate the encoding model + resolve the non-finite mask ONCE (the same
    # front-half decode_position_summary runs). spike_counts is faked with a
    # zero-row block here only to satisfy the helper's interface; its actual
    # per-block counts come from the streamed binning below. We pass a
    # (1, n_neurons) row so the neuron-count agreement check still fires.
    #
    # NOTE: the real per-block counts produced by the streamed binning below are
    # intentionally NOT routed through _validate_inputs. They come straight from
    # np.histogram on float spike times, so they are non-negative int64 by
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
        is_last_block = stop == n_time

        # Bin ONLY this block's spikes, against the GLOBAL edge slice
        # edges[start : stop + 1]. Slicing the precomputed global edges (rather
        # than recomputing block_t_start + k*dt) is what makes the per-block
        # counts byte-for-byte identical to a single global histogram: the
        # interior edges are the SAME float values, so each spike lands in the
        # same bin either way.
        lo = edges[start]
        hi = edges[stop]
        block_edges = edges[start : stop + 1]

        # Avoid double-counting boundary spikes. np.histogram right-closes its
        # LAST bin, so a spike exactly on an interior global edge `stop` would
        # otherwise be counted in BOTH this block's last bin (right-closed) AND
        # the next block's first (left-closed) bin. For every block except the
        # final one, drop spikes sitting on the right edge by scoping to
        # [lo, hi). The final block keeps the right edge closed (matching the
        # global grid's right-closed final bin: a spike at edges[-1] counts).
        if is_last_block:
            block_trains = [s[(s >= lo) & (s <= hi)] for s in trains]
        else:
            block_trains = [s[(s >= lo) & (s < hi)] for s in trains]

        counts_block = np.stack(
            [np.histogram(s, bins=block_edges)[0] for s in block_trains], axis=1
        ).astype(np.int64)  # (stop - start, n_neurons)

        # Block alignment contract: the block has exactly `stop - start` bins
        # and its centers equal the global centers slice (no off-by-one / gap /
        # double-count at block boundaries). These guard a load-bearing
        # correctness invariant, so raise unconditionally (do NOT use bare
        # `assert`, which `python -O` strips).
        if counts_block.shape != (stop - start, n_neurons):
            raise RuntimeError(
                f"block count shape {counts_block.shape} != expected "
                f"{(stop - start, n_neurons)} for block [{start}, {stop})"
            )
        block_centers = block_edges[:-1] + dt / 2.0
        if not np.array_equal(block_centers, bin_centers_time[start:stop]):
            raise RuntimeError(
                f"streamed block centers drifted from the global time grid for "
                f"block [{start}, {stop}); block-boundary alignment is broken"
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
    )
