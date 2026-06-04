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

if TYPE_CHECKING:
    from neurospatial.decoding._result import DecodingResult
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
    bandwidth: float = 5.0,
    smoothing_method: str = "diffusion_kde",
    min_occupancy: float = 0.0,
    encoding_models: NDArray[np.float64] | None = None,
    warn_on_drop: bool = True,
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
        Timestamps (seconds) at which ``positions`` were recorded.
        Used both to build encoding models and to set the decoding time
        grid via ``t_start = times.min()`` / ``t_stop = times.max()``.
    positions : NDArray[np.float64], shape (n_frames, n_dims)
        Animal position at each frame in ``times``.
    dt : float, optional
        Decoding time-bin width in seconds.  Default is 0.025 (25 ms).
    bandwidth : float, optional
        Smoothing bandwidth (same units as positions) for the KDE encoding
        step.  Ignored when ``encoding_models`` is provided.  Default 5.0.
    smoothing_method : str, optional
        KDE method passed to :func:`~neurospatial.encoding.compute_spatial_rates`.
        Options: ``"diffusion_kde"`` (default), ``"gaussian_kde"``,
        ``"binned"``.  Ignored when ``encoding_models`` is provided.
    min_occupancy : float, optional
        Minimum occupancy (seconds) for a spatial bin to be included in the
        encoding model.  Bins below threshold are set to ``fill_value=0.0``
        so the decoder never receives NaN rates.  Default is 0.0 (no
        threshold).  Ignored when ``encoding_models`` is provided.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins) or None
        Pre-computed place-field firing-rate maps.  When provided, the
        encoding step (``compute_spatial_rates``) is skipped entirely and
        this array is passed directly to the decoder.  Useful for re-using
        models across multiple decoding passes or for injecting custom
        encoding models.  ``bandwidth``, ``smoothing_method``, and
        ``min_occupancy`` are ignored when this is set.
    warn_on_drop : bool, optional
        If ``True`` (the default), emit a single ``UserWarning`` when a large
        fraction (>50%, which includes the all-dropped case) of spikes fall
        outside the decode time window ``[times.min(), times.max()]`` and are
        therefore silently excluded from the count matrix.  This guards the
        common units footgun — ``spike_times`` in milliseconds while ``times``
        is in seconds — which would otherwise produce an all-zero count matrix
        and a plausible-but-wrong posterior.  The warning fires exactly once
        per call and covers both the ``encoding_models``-provided branch (which
        skips the encoder) and the ``encoding_models=None`` branch (where the
        encoder's own redundant warning is suppressed).  Set to ``False`` to
        suppress this warning (e.g. for a genuinely sparse session).
    **decode_kwargs
        Additional keyword arguments forwarded verbatim to
        :func:`~neurospatial.decoding.decode_position`.  Supported kwargs
        include ``prior``, ``method``, and ``validate``.

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
    When ``encoding_models`` is not provided this function always passes
    ``fill_value=0.0`` to the encoder so that low-occupancy bins produce
    zero-rate predictions rather than NaN, keeping the posterior valid.
    If you need NaN-masked bins in the encoding model, compute
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
    # Defer the `encoding` imports until call time: this keeps the decoding
    # package importable even if `encoding` were ever to import from `decoding`
    # (it does not today), so there is no circular-import risk at module load.
    # Mirrors how encoding/spatial.py defers its own heavy imports.
    from neurospatial.decoding._binning import bin_spikes_in_time
    from neurospatial.decoding.posterior import decode_position
    from neurospatial.encoding import as_spike_trains
    from neurospatial.encoding.spatial import compute_spatial_rates

    # --- Normalize inputs ---
    trains = as_spike_trains(spike_times)
    times_arr = np.asarray(times, dtype=np.float64)
    if times_arr.size < 2:
        raise ValueError(
            f"times must have at least 2 samples to define a trajectory "
            f"window, got {times_arr.size}."
        )

    # Decode window — computed ONCE and reused for both the out-of-window drop
    # check and the bin_spikes_in_time call so they agree exactly.
    t_start = float(times_arr.min())
    t_stop = float(times_arr.max())

    # --- Out-of-window drop check (owns the single units-footgun warning) ---
    # bin_spikes_in_time counts via np.histogram(..., bins=edges), which
    # silently drops spikes outside [t_start, t_stop].  If spike_times are in
    # milliseconds while times are in seconds, (nearly) all spikes fall outside
    # the window → all-zero counts → a plausible-but-wrong posterior.  Surface
    # this once here so it fires whether or not encoding_models is provided.
    if warn_on_drop:
        _warn_if_spikes_out_of_window(trains, t_start, t_stop)

    # --- Build encoding models if not provided ---
    if encoding_models is None:
        _method = cast(
            "Literal['diffusion_kde', 'gaussian_kde', 'binned']", smoothing_method
        )
        # warn_on_drop=False: decode_session already warned once above for the
        # whole golden path; the encoder must not emit a duplicate.
        rates_result = compute_spatial_rates(
            env,
            trains,
            times_arr,
            positions,
            bandwidth=bandwidth,
            smoothing_method=_method,
            min_occupancy=min_occupancy,
            fill_value=0.0,
            warn_on_drop=False,
        )
        firing_rates = np.asarray(rates_result.firing_rates, dtype=np.float64)
    else:
        firing_rates = np.asarray(encoding_models, dtype=np.float64)

    # --- Bin spikes in time (default orient="time_x_neuron" → (n_time, n_neurons)) ---
    counts, centers = bin_spikes_in_time(
        trains,
        dt,
        t_start=t_start,
        t_stop=t_stop,
    )
    # counts shape: (n_time_bins, n_neurons)  ← what decode_position expects

    # --- Decode ---
    return decode_position(
        env,
        counts,
        firing_rates,
        dt,
        times=centers,
        **decode_kwargs,
    )
