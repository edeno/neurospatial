"""One-call encode->bin->decode golden path.

Provides :func:`decode_session`, a convenience wrapper that glues together
:func:`~neurospatial.encoding.compute_spatial_rates`,
:func:`~neurospatial.decoding.bin_spikes_in_time`, and
:func:`~neurospatial.decoding.decode_position` so a beginner can decode
position from spikes in ≤10 lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.decoding._result import DecodingResult
    from neurospatial.environment import Environment


def decode_session(
    env: Environment,
    spike_times: Any,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    dt: float = 0.025,
    bandwidth: float = 5.0,
    smoothing_method: str = "diffusion_kde",
    min_occupancy: float = 0.0,
    encoding_models: NDArray[np.float64] | None = None,
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
        :func:`~neurospatial.encoding.normalize_spike_times`:

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
    ``dt``.  Spikes outside this window are silently excluded by
    :func:`~neurospatial.decoding.bin_spikes_in_time`.

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
    >>> positions, times = simulate_trajectory_ou(  # doctest: +SKIP
    ...     env, duration=10.0, seed=0
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
    # Lazy imports: decoding -> encoding is an acceptable dependency direction,
    # but we import inside the function to guarantee zero risk of circular
    # imports at module-load time (encoding/__init__.py never imports from
    # decoding, so this is safe; the lazy pattern mirrors how encoding/spatial.py
    # handles its own heavy imports).
    from neurospatial.decoding._binning import bin_spikes_in_time
    from neurospatial.decoding.posterior import decode_position
    from neurospatial.encoding import normalize_spike_times
    from neurospatial.encoding.spatial import compute_spatial_rates

    # --- Normalize inputs ---
    trains = normalize_spike_times(spike_times)
    times_arr = np.asarray(times, dtype=np.float64)

    # --- Build encoding models if not provided ---
    if encoding_models is None:
        _method = cast(
            "Literal['diffusion_kde', 'gaussian_kde', 'binned']", smoothing_method
        )
        rates_result = compute_spatial_rates(
            env,
            trains,
            times_arr,
            positions,
            bandwidth=bandwidth,
            smoothing_method=_method,
            min_occupancy=min_occupancy,
            fill_value=0.0,
        )
        firing_rates = np.asarray(rates_result.firing_rates, dtype=np.float64)
    else:
        firing_rates = np.asarray(encoding_models, dtype=np.float64)

    # --- Bin spikes in time (default orient="time_x_neuron" → (n_time, n_neurons)) ---
    counts, centers = bin_spikes_in_time(
        trains,
        dt,
        t_start=times_arr.min(),
        t_stop=times_arr.max(),
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
