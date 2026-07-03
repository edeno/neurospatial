"""Posterior estimation and normalization for Bayesian decoding.

This module provides functions for converting log-likelihoods to posterior
probability distributions using Bayes' rule with numerically stable
log-sum-exp computation.

Functions
---------
normalize_to_posterior : Convert log-likelihood to posterior
    Applies Bayes' rule with optional prior and handles degenerate cases.

decode_position : Main entry point for Bayesian position decoding
    Combines likelihood computation and posterior normalization into a
    single function that returns a DecodingResult.

Notes
-----
All computations are performed in log-domain for numerical stability.
The log-sum-exp trick is used to prevent overflow/underflow when
exponentiating log-likelihoods.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, overload

import numpy as np
from numpy.typing import NDArray

from neurospatial.decoding._binning import validate_dt
from neurospatial.decoding._result import DecodingResult, DecodingSummary
from neurospatial.decoding.likelihood import log_poisson_likelihood

if TYPE_CHECKING:
    from neurospatial.environment import Environment


class SpatialRatesLike(Protocol):
    """Duck-typed protocol for population rate result objects.

    Any object exposing a ``firing_rates`` attribute of shape
    ``(n_neurons, n_bins)`` -- e.g.
    :class:`~neurospatial.encoding.spatial.SpatialRatesResult` -- can be passed
    directly to :func:`decode_position` in place of a raw NumPy array.
    """

    @property
    def firing_rates(self) -> NDArray[np.float64]: ...


def _validate_posterior_dtype(dtype: Any) -> np.dtype:
    """Validate and resolve the posterior working/storage dtype.

    Accepts ``np.float32`` / ``np.float64`` (and their ``np.dtype`` forms);
    anything else raises a clear ``ValueError`` naming the param and the value.

    Parameters
    ----------
    dtype : Any
        The requested posterior dtype.

    Returns
    -------
    numpy.dtype
        The resolved float dtype (``float32`` or ``float64``).

    Raises
    ------
    ValueError
        If ``dtype`` is not float32 or float64.
    """
    msg = (
        f"dtype must be np.float32 or np.float64, got {dtype!r}. "
        "Only single- and double-precision posteriors are supported "
        "(float32 halves stored and transient memory)."
    )
    # Wrap the parse so an unparseable dtype string (e.g. "bogus") raises this
    # clean ValueError naming `dtype`, not a raw NumPy
    # ``TypeError: data type 'bogus' not understood``.
    try:
        resolved = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(msg) from exc
    if resolved not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(msg)
    return cast("np.dtype", resolved)


@overload
def _validate_time_chunk(
    time_chunk: Any,
    *,
    allow_none: Literal[True],
    context: str = ...,
) -> int | None: ...


@overload
def _validate_time_chunk(
    time_chunk: Any,
    *,
    allow_none: Literal[False],
    context: str = ...,
) -> int: ...


def _validate_time_chunk(
    time_chunk: Any,
    *,
    allow_none: bool,
    context: str = "time_chunk",
) -> int | None:
    """Validate ``time_chunk`` as a genuine positive integer (or ``None``).

    Centralizes the ``time_chunk`` contract so a clear ``ValueError`` is raised
    instead of a raw ``TypeError`` leaking from ``range(...)`` (for floats /
    strings) and instead of silently accepting ``True`` as a chunk size of 1.

    Rules: ``time_chunk`` must be a real Python/NumPy integer (``bool`` is
    rejected even though it subclasses ``int``) and ``>= 1``. When
    ``allow_none`` is ``True``, ``None`` passes through unchanged; when it is
    ``False``, ``None`` raises (the caller is expected to pre-handle ``None``
    with its own actionable message before calling this helper).

    Parameters
    ----------
    time_chunk : Any
        The received ``time_chunk`` value.
    allow_none : bool
        Whether ``None`` is a permitted value.
    context : str, default="time_chunk"
        Parameter name used in the error message.

    Returns
    -------
    int or None
        The validated ``time_chunk`` (``None`` only when ``allow_none``).

    Raises
    ------
    ValueError
        If ``time_chunk`` is not a positive integer (or ``None`` when allowed).
    """
    if time_chunk is None:
        if allow_none:
            return None
        none_clause = ""
        raise ValueError(
            f"{context} must be a positive integer{none_clause}, "
            f"got {time_chunk!r} (type {type(time_chunk).__name__})."
        )
    # bool is a subclass of int but is never a valid chunk size here.
    if isinstance(time_chunk, bool) or not isinstance(time_chunk, (int, np.integer)):
        none_clause = " or None" if allow_none else ""
        raise ValueError(
            f"{context} must be a positive integer{none_clause}, "
            f"got {time_chunk!r} (type {type(time_chunk).__name__})."
        )
    if time_chunk < 1:
        none_clause = " or None" if allow_none else ""
        raise ValueError(
            f"{context} must be a positive integer{none_clause}, "
            f"got {time_chunk!r} (type {type(time_chunk).__name__})."
        )
    return int(time_chunk)


def _normalize_block(
    ll_block: NDArray[np.float64],
    *,
    axis: int,
    handle_degenerate: Literal["uniform", "nan", "raise"],
    out: NDArray[np.float64],
) -> None:
    """Normalize one time-block of log-likelihood in place into ``out``.

    Performs the log-sum-exp softmax (per-row ``ll_max`` shift, ``exp``,
    normalize) and the degenerate-row handling for a single block, writing the
    result into the preallocated ``out`` array. ``ll_block`` is the
    already-prior-applied log-likelihood for the block; ``out`` must be the
    same shape and the desired output dtype.

    Degenerate-row handling matches the full-array path exactly: rows whose
    per-row max is non-finite (all ``-inf`` zero-rate rows, or rows containing
    ``NaN``) are filled per ``handle_degenerate``.
    """
    ll_max = ll_block.max(axis=axis, keepdims=True)

    ll_max_squeezed = ll_max.squeeze(axis=axis)
    degenerate_mask = ~np.isfinite(ll_max_squeezed)
    nan_row_mask = np.isnan(ll_block).any(axis=axis)

    # Shift for stability, exponentiate into the output dtype, normalize.
    # Degenerate rows are all -inf, so ``-inf - -inf = nan`` here; that NaN is
    # expected and handled by the degenerate-row branch below, so suppress the
    # noisy "invalid value encountered in subtract" warning it would emit.
    with np.errstate(invalid="ignore"):
        ll_shifted = ll_block - ll_max
        np.exp(ll_shifted, out=out)
    out /= out.sum(axis=axis, keepdims=True)

    if degenerate_mask.any():
        if handle_degenerate == "raise":
            n_degenerate = int(degenerate_mask.sum())
            n_nan = int(nan_row_mask.sum())
            if n_nan > 0:
                n_neg_inf = n_degenerate - n_nan
                raise ValueError(
                    f"Found {n_degenerate} degenerate row(s): {n_nan} contain "
                    f"NaN values (upstream corruption, e.g. a NaN firing rate "
                    f"leaking into the likelihood) and {n_neg_inf} are all -inf "
                    f"(zero-rate). Fix the NaN source; the -inf rows can be "
                    f"handled with handle_degenerate='uniform' or 'nan'."
                )
            raise ValueError(
                f"Found {n_degenerate} degenerate row(s) with all -inf values "
                f"(zero-rate). Consider using handle_degenerate='uniform' or "
                f"'nan'."
            )
        elif handle_degenerate == "uniform":
            n_bins = ll_block.shape[axis]
            uniform_prob = 1.0 / n_bins
            out[degenerate_mask] = uniform_prob
        elif handle_degenerate == "nan":
            out[degenerate_mask] = np.nan


def normalize_to_posterior(
    log_likelihood: NDArray[np.float64],
    *,
    prior: NDArray[np.float64] | None = None,
    axis: int = -1,
    handle_degenerate: Literal["uniform", "nan", "raise"] = "uniform",
    dtype: Any = np.float64,
    time_chunk: int | None = None,
) -> NDArray[np.float64]:
    """Convert log-likelihood to posterior using Bayes' rule.

    Applies the formula:

        P(position | spikes) = P(spikes | position) * P(position) / P(spikes)

    using numerically stable log-sum-exp normalization.

    Parameters
    ----------
    log_likelihood : NDArray[np.float64], shape (n_time_bins, n_bins)
        Log-likelihood from `log_poisson_likelihood` or similar.
    prior : NDArray[np.float64] | None, default=None
        Prior probability over positions. If None, uses uniform prior.
        Shape (n_bins,) for stationary prior, (n_time_bins, n_bins) for
        time-varying prior.

        **Note**: Priors are treated as **probability distributions** (not
        unnormalized weights). They are normalized internally to sum to 1.0
        along the position axis before applying.
    axis : int, default=-1
        Axis along which to normalize.
    handle_degenerate : {"uniform", "nan", "raise"}, default="uniform"
        How to handle degenerate rows. A row is degenerate if it is all
        ``-inf`` (a legitimate zero-rate row, e.g. no spikes with a flat
        encoding model) or if it contains a ``NaN`` (upstream corruption,
        e.g. a NaN firing rate leaking into the likelihood):

        - "uniform": Return uniform distribution (1/n_bins per bin) for
          every degenerate row. This masks NaN corruption the same as a
          zero-rate row; use "raise" if you need corruption to surface.
        - "nan": Return NaN for degenerate rows.
        - "raise": Raise ValueError if any row is degenerate. The message
          distinguishes NaN-corrupted rows from all -inf zero-rate rows.
    dtype : np.float32 or np.float64, default=np.float64
        Stored/working dtype of the returned posterior. ``np.float32`` halves
        the stored and transient memory at the cost of single-precision
        rounding (parity with float64 to ~1e-6 relative). Must be float32 or
        float64; any other dtype raises ``ValueError``.
    time_chunk : int or None, default=None
        When set, the exp/normalize is computed in time-blocks of
        ``time_chunk`` rows (axis 0) into a single preallocated output array,
        so the per-block ``ll_shifted`` temporary is materialized one
        time-block at a time rather than full-size. This removes only that
        full-size temporary: the incoming ``log_likelihood`` and its working
        copy ``ll`` are still full-size and persist across the whole loop
        alongside the output, so the in-flight peak drops from ~4x to ~3x the
        stored posterior, not to ~1x. When ``None`` (the default), the whole
        array is normalized at once and the result is byte-for-byte identical
        to the pre-chunking behavior.

    Returns
    -------
    posterior : NDArray, shape (n_time_bins, n_bins)
        Posterior probability distribution in ``dtype``. Each row sums to 1.0.

    Raises
    ------
    ValueError
        If handle_degenerate="raise" and degenerate rows are detected.
        If handle_degenerate has an invalid value.

    Notes
    -----
    Implementation uses numerically stable log-sum-exp:

    .. code-block:: python

        # Add log-prior to log-likelihood
        if prior is not None:
            prior = prior / prior.sum(axis=axis, keepdims=True)  # Normalize
            log_prior = np.log(np.clip(prior, 1e-10, 1.0))
            ll = log_likelihood + log_prior
        else:
            ll = log_likelihood

        # Log-sum-exp normalization (stable softmax)
        ll_max = ll.max(axis=axis, keepdims=True)
        ll_shifted = ll - ll_max  # Shift to prevent overflow
        posterior = np.exp(ll_shifted)
        posterior /= posterior.sum(axis=axis, keepdims=True)

    For rows where all entries are -inf (e.g., no spikes and flat encoding):

    - ll_max will be -inf, ll_shifted will be NaN
    - These are detected and handled according to `handle_degenerate`

    **Memory.** ``dtype=np.float32`` halves the stored posterior; ``time_chunk``
    blocks the exp/normalize so the full-size ``ll_shifted`` temporary is no
    longer materialized, dropping the transient peak from ~4x to ~3x the stored
    posterior. It removes only that temporary -- the incoming ``log_likelihood``
    and its working copy ``ll`` remain full-size across the loop -- so this is
    not a ~1x path; for that, compute the likelihood per block and never hold
    the full log-likelihood (see :func:`decode_position_summary`). Both knobs
    default to the original behavior (``dtype=np.float64``,
    ``time_chunk=None``), which is reproduced byte-for-byte.

    Examples
    --------
    >>> ll = np.array([[-1.0, -2.0, -0.5], [-0.2, -0.3, -0.1]])
    >>> posterior = normalize_to_posterior(ll)
    >>> posterior.sum(axis=1)  # Each row sums to 1.0
    array([1., 1.])

    >>> # With prior favoring first bin
    >>> prior = np.array([0.5, 0.25, 0.25])
    >>> posterior_prior = normalize_to_posterior(ll, prior=prior)
    >>> bool(posterior_prior[0, 0] > posterior[0, 0])  # First bin gets boost
    True

    See Also
    --------
    log_poisson_likelihood : Compute log-likelihood for Poisson model
    decode_position : Main entry point combining likelihood and posterior
    """
    # Validate handle_degenerate parameter
    if handle_degenerate not in ("uniform", "nan", "raise"):
        raise ValueError(
            f"handle_degenerate must be 'uniform', 'nan', or 'raise', "
            f"got {handle_degenerate!r}"
        )

    out_dtype = _validate_posterior_dtype(dtype)

    time_chunk = _validate_time_chunk(time_chunk, allow_none=True)

    log_likelihood = np.asarray(log_likelihood, dtype=np.float64)

    # Validate axis parameter
    # The degeneracy handling logic assumes axis is the last dimension.
    # Normalize axis to positive form for comparison.
    effective_axis = axis if axis >= 0 else log_likelihood.ndim + axis
    if effective_axis != log_likelihood.ndim - 1:
        raise ValueError(
            f"axis must be the last dimension (axis=-1 or axis={log_likelihood.ndim - 1}), "
            f"got axis={axis}. The current implementation's degeneracy handling "
            f"only supports normalization along the last axis."
        )
    ll = log_likelihood.copy()

    # Apply prior if provided
    if prior is not None:
        prior_arr = np.asarray(prior, dtype=np.float64)

        # Validate prior shape
        n_bins = log_likelihood.shape[-1]  # Position axis (last dimension)
        if prior_arr.ndim == 1:
            # Stationary prior: shape (n_bins,)
            if prior_arr.shape[0] != n_bins:
                raise ValueError(
                    f"1D prior must have shape ({n_bins},) to match log_likelihood "
                    f"position axis, got shape {prior_arr.shape}"
                )
        elif prior_arr.ndim == 2:
            # Time-varying prior: shape (n_time_bins, n_bins)
            if prior_arr.shape != log_likelihood.shape:
                raise ValueError(
                    f"2D prior must have shape {log_likelihood.shape} to match "
                    f"log_likelihood, got shape {prior_arr.shape}"
                )
        else:
            raise ValueError(
                f"prior must be 1D (stationary) or 2D (time-varying), "
                f"got {prior_arr.ndim}D with shape {prior_arr.shape}"
            )

        # Normalize prior along the specified axis
        # Handle 1D prior (stationary) vs 2D prior (time-varying)
        if prior_arr.ndim == 1:
            prior_sum = prior_arr.sum()
            if prior_sum > 0:
                prior_arr = prior_arr / prior_sum
        else:
            prior_sum = prior_arr.sum(axis=axis, keepdims=True)
            # Avoid division by zero
            prior_arr = np.where(prior_sum > 0, prior_arr / prior_sum, prior_arr)

        # Clip prior to avoid log(0)
        prior_clipped = np.clip(prior_arr, 1e-10, 1.0)
        log_prior = np.log(prior_clipped)

        # Add log-prior to log-likelihood
        ll = ll + log_prior

    # Log-sum-exp normalization (numerically stable softmax). Compute into a
    # preallocated output buffer of the requested dtype. When time_chunk is set,
    # normalize one time-block at a time so the per-block `ll_shifted` temporary
    # is materialized one block at a time rather than full-size. This removes
    # only that temporary: `log_likelihood` (the incoming array) and its working
    # copy `ll` are still full-size and persist across the whole loop alongside
    # `out`, so the transient peak drops from ~4x to ~3x the stored posterior,
    # NOT to ~1x. Otherwise normalize the whole array in one block, which is
    # byte-for-byte identical to the pre-chunking path (modulo the requested
    # dtype).
    #
    # Degenerate rows have two distinct causes, handled identically per block:
    # - all -inf: a legitimate zero-rate row (e.g. no spikes with a flat
    #   encoding model); the per-row max is -inf (non-finite).
    # - any NaN: upstream corruption; the per-row max propagates NaN.
    # Both fail np.isfinite on the per-row max, so they share the degenerate
    # set; the NaN subset is tracked so handle_degenerate="raise" reports
    # corruption explicitly rather than blaming a flat row.
    out = np.empty(ll.shape, dtype=out_dtype)

    n_time = ll.shape[0]
    if time_chunk is None:
        _normalize_block(ll, axis=axis, handle_degenerate=handle_degenerate, out=out)
    else:
        for start in range(0, n_time, time_chunk):
            stop = min(start + time_chunk, n_time)
            _normalize_block(
                ll[start:stop],
                axis=axis,
                handle_degenerate=handle_degenerate,
                out=out[start:stop],
            )

    return cast("NDArray[np.float64]", out)


def decode_position(
    env: Environment,
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64] | SpatialRatesLike,
    dt: float,
    *,
    prior: NDArray[np.float64] | None = None,
    method: Literal["poisson"] = "poisson",
    times: NDArray[np.float64] | None = None,
    validate: bool = True,
    dtype: Any = np.float64,
    time_chunk: int | None = None,
) -> DecodingResult:
    """Decode position from population spike counts.

    Main entry point for Bayesian decoding. Computes posterior probability
    distribution over positions for each time bin.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64] or SpatialRatesResult, shape (n_neurons, n_bins)
        Firing rate maps (place fields) for each neuron.
        Expected units: Hz (spikes/second). Typical values: 0-50 Hz.
        Very high rates (>100 Hz) may cause numerical issues.

        A population rate result object (anything exposing a ``firing_rates``
        attribute, e.g.
        :class:`~neurospatial.encoding.spatial.SpatialRatesResult`) may be
        passed directly; its ``firing_rates`` array is used. This removes the
        ``np.stack([r.firing_rate ...])`` glue between the encoding and
        decoding steps.

        NaN entries (e.g. low-occupancy bins masked by an encoder's
        ``min_occupancy`` when ``fill_value=None``) are tolerated: each such
        ``(neuron, bin)`` is treated as a zero-rate observation and excluded
        from that neuron's contribution to the Poisson log-likelihood at that
        bin, and a single :class:`UserWarning` is emitted per call. This is
        defense-in-depth so a ``fill_value=None`` encoding model still decodes;
        the recommended path is still to pass ``fill_value=0.0`` to the
        encoder so no NaN reaches the decoder.

        Inf entries are handled the same way, but only with ``validate=False``:
        each Inf ``(neuron, bin)`` is excluded as a zero-rate observation, so a
        partial-Inf model such as ``rates=[inf, inf, 5]`` concentrates posterior
        mass on the one finite bin rather than collapsing to a uniform
        posterior. Under ``validate=True`` (the default) Inf entries are instead
        rejected by validation.

        A model with **no finite bins at all** -- every spatial bin is
        non-finite (NaN or Inf) across all neurons -- is a special case: it
        carries zero information and cannot decode, so a :class:`ValueError`
        is raised rather than returning a meaningless (uniform-looking)
        posterior. This guard is unconditional (it fires even with
        ``validate=False``) and catches Inf as well as NaN. Partial models
        with at least one finite bin still decode normally.
    dt : float
        Time bin width in seconds. Typical values: 0.001-0.1s.
        Note: For typical firing rates, lambda*dt should be in [0, 5].
    prior : NDArray[np.float64] | None, default=None
        Prior probability over positions. If None, uses uniform prior.
        Shape (n_bins,) for stationary prior, (n_time_bins, n_bins) for
        time-varying prior. Normalized internally to sum to 1.0.
    method : {"poisson"}, default="poisson"
        Likelihood model. Currently only Poisson supported.
        Future: "gaussian", "clusterless".
    times : NDArray[np.float64] | None, default=None
        Time bin centers (seconds). If provided, stored in DecodingResult.
    validate : bool, default=True
        If True (the default), run extra validation checks before and
        after the core computation:

        - Reject NaN/Inf entries in spike_counts and prior. NaN entries in
          ``encoding_models`` are NOT rejected: they are absorbed by the
          zero-rate exclusion described above (which runs first), so a
          ``fill_value=None`` model decodes under ``validate=True`` instead
          of raising. Inf entries in ``encoding_models`` are still rejected.
        - Reject negative spike_counts (must be non-negative integers).
        - Reject fractional spike_counts when given a float dtype
          (cast to integer first or fix the upstream binning bug).
        - Reject negative encoding_models firing rates.
        - Reject negative prior entries.
        - Verify each posterior row sums to 1.0 within atol=1e-6.

        Pass ``validate=False`` to opt out (e.g., inside a hot inner
        loop where the inputs are guaranteed-clean and the per-call
        overhead matters).
    dtype : np.float32 or np.float64, default=np.float64
        Stored/working dtype of the posterior. ``np.float32`` halves the
        stored and transient memory at the cost of single-precision rounding
        (parity with float64 to ~1e-6 relative). The returned
        :attr:`DecodingResult.posterior` is still a fully materialized
        ``ndarray`` in this dtype, and every :class:`DecodingResult` method
        works unchanged. Must be float32 or float64; any other value raises
        ``ValueError``.
    time_chunk : int or None, default=None
        Hybrid memory knob. Two distinct paths:

        - ``None`` (the default): the Poisson log-likelihood is computed once
          over the whole window (a single full-size matmul) and the whole
          posterior is normalized at once. This is the original full-matmul
          path, reproduced **byte-for-byte** -- every existing caller gets
          bit-identical output. Its in-flight transient peak is ~3x over the
          returned posterior (the full-size log-likelihood, its working copy,
          and the output coexist).
        - an explicit positive int: the Poisson log-likelihood is computed and
          normalized **one time-block of ``time_chunk`` rows at a time, directly
          into the preallocated posterior**, so the full ``(n_time_bins,
          n_bins)`` log-likelihood and its working copy are never materialized.
          This cuts the transient peak to **~1x over the returned posterior**
          (the posterior itself is unavoidably 1x -- this path is returned
          fully materialized). It is **tolerance-equal, not byte-exact**, to the
          ``None`` path: the per-block likelihood matmul is a different BLAS
          shape than the full matmul, so it differs by ~1e-15 ULPs. MAP/argmax
          is identical and every row sums to 1.0.

        Either way the return type is unchanged: ``.posterior`` is the full
        materialized ``(n_time_bins, n_bins)`` array. For a path that never
        materializes even the full posterior, use
        :func:`decode_position_summary`.

    Returns
    -------
    DecodingResult
        Container with posterior, estimates, and metadata.

    Raises
    ------
    ValueError
        If method is not "poisson".
        If validate=True and validation checks fail.
        If ``encoding_models`` has no finite bins (every spatial bin is
        non-finite -- NaN or Inf -- across all neurons); such a model carries
        no information and cannot decode. This is checked unconditionally,
        even with ``validate=False``.
        If ``encoding_models`` bin count (axis 1) does not match
        ``env.n_bins``. This is checked unconditionally, even with
        ``validate=False``, because a wrong bin count yields a
        well-formed posterior over the wrong positions.
        If ``times`` is provided and its length does not match the number
        of posterior time bins.

    Notes
    -----
    Memory usage: The posterior array is shape (n_time_bins, n_bins) and
    stored as float64 by default. For long recordings (e.g., 1 hour at 25ms
    bins = 144,000 time bins) with fine spatial resolution (e.g., 1000 bins),
    this requires ~1.1 GB stored. The default ``time_chunk=None`` full-matmul
    path has a ~3x transient peak (the full-size log-likelihood, its working
    copy, and the output coexist), and is byte-for-byte the original behavior.
    Two knobs cut the transient without changing the return contract:
    ``dtype=np.float32`` halves the stored and transient memory, and an explicit
    ``time_chunk=N`` computes the likelihood blockwise into the preallocated
    posterior so the full-size log-likelihood and its copy are never
    materialized -- dropping the transient peak to ~1x over the returned
    posterior (tolerance-equal to the full path; ~1e-15; MAP identical). When
    even the stored ``(n_time_bins, n_bins)`` array is too large to hold, use
    :func:`decode_position_summary`, which streams over time -- computing the
    likelihood per block so it never materializes the full log-likelihood -- and
    keeps only ``(n_time_bins, ...)`` per-time reductions.

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.posterior import decode_position
    >>> import numpy as np
    >>>
    >>> # Setup
    >>> positions = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>>
    >>> # Generate test data
    >>> n_time_bins, n_neurons = 50, 5
    >>> spike_counts = np.random.poisson(2, (n_time_bins, n_neurons)).astype(np.int64)
    >>> encoding_models = np.random.uniform(0.1, 10, (n_neurons, env.n_bins))
    >>>
    >>> # Decode
    >>> result = decode_position(env, spike_counts, encoding_models, dt=0.025)
    >>> result.posterior.shape
    (50, ...)
    >>> result.map_estimate.shape
    (50,)

    See Also
    --------
    DecodingResult : Container for decoding results
    log_poisson_likelihood : Likelihood function used internally
    normalize_to_posterior : Posterior normalization used internally
    """
    # Validate dt up front so a non-numeric/bool/non-finite dt raises the same
    # clean message as every other decoding entry point, rather than leaking a
    # raw TypeError (dt="0.1") or a misleading downstream error (dt=NaN/<=0)
    # from log_poisson_likelihood.
    dt = validate_dt(dt)
    time_chunk = _validate_time_chunk(time_chunk, allow_none=True)

    spike_counts, encoding_models, nonfinite_mask = _prepare_decode_inputs(
        env,
        spike_counts,
        encoding_models,
        prior=prior,
        method=method,
        validate=validate,
        context="decode_position",
    )

    if time_chunk is None:
        # Default path: full-matmul log-likelihood over the whole window, then
        # normalize the full array at once. Kept byte-for-byte identical to the
        # pre-hybrid behavior -- the likelihood matmul runs once at full size, so
        # there is no BLAS shape-dependence and every existing caller gets
        # bit-identical output.
        log_ll = _poisson_log_likelihood(
            spike_counts, encoding_models, nonfinite_mask, dt
        )
        posterior = normalize_to_posterior(
            log_ll, prior=prior, dtype=dtype, time_chunk=None
        )

        # Validate output if requested
        if validate:
            _validate_output(posterior)
    else:
        # Opt-in path: compute the Poisson log-likelihood ONE time-block at a
        # time directly into the preallocated posterior, never materializing the
        # full (n_time, n_bins) log-likelihood or its copy. This cuts the
        # transient peak from ~3x to ~1x over the returned posterior. It is
        # tolerance-equal (not byte-exact) to the full path: the per-block
        # likelihood matmul is a different BLAS shape than the full matmul, so it
        # differs by ~1e-15 ULPs (MAP/argmax identical; rows still sum to 1).
        posterior = _decode_blockwise(
            env,
            spike_counts,
            encoding_models,
            dt,
            prior=prior,
            nonfinite_mask=nonfinite_mask,
            validate=validate,
            dtype=dtype,
            time_chunk=time_chunk,
        )

    # Handle times
    if times is not None:
        times = np.asarray(times, dtype=np.float64)
        n_time_bins = posterior.shape[0]
        if times.ndim != 1 or len(times) != n_time_bins:
            raise ValueError(
                f"Length mismatch: times has {times.shape} but posterior has "
                f"{n_time_bins} time bins. `times` must be a 1-D array of bin "
                f"centers, one per row of spike_counts."
            )

    # Return DecodingResult
    return DecodingResult(posterior=posterior, env=env, times=times)


def _validate_prior_shape(
    prior: NDArray[np.float64],
    *,
    n_time: int,
    n_bins: int,
) -> bool:
    """Validate a prior's shape against the decode dimensions.

    Mirrors :func:`normalize_to_posterior`'s prior-shape checks so the blockwise
    ``decode_position`` path reports mismatches up front (before the block loop)
    rather than only at whichever block first slices a bad row.

    Parameters
    ----------
    prior : NDArray[np.float64]
        Prior over positions, already an ndarray.
    n_time : int
        Number of time bins being decoded.
    n_bins : int
        Number of position bins.

    Returns
    -------
    bool
        ``True`` if the prior is time-varying (2-D), ``False`` if stationary
        (1-D).

    Raises
    ------
    ValueError
        If the prior is not 1-D ``(n_bins,)`` or 2-D ``(n_time, n_bins)``.
    """
    if prior.ndim == 1:
        if prior.shape[0] != n_bins:
            raise ValueError(
                f"1D prior must have shape ({n_bins},) to match log_likelihood "
                f"position axis, got shape {prior.shape}"
            )
        return False
    if prior.ndim == 2:
        if prior.shape != (n_time, n_bins):
            raise ValueError(
                f"2D prior must have shape {(n_time, n_bins)} to match "
                f"log_likelihood, got shape {prior.shape}"
            )
        return True
    raise ValueError(
        f"prior must be 1D (stationary) or 2D (time-varying), "
        f"got {prior.ndim}D with shape {prior.shape}"
    )


def _decode_blockwise(
    env: Environment,
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    dt: float,
    *,
    prior: NDArray[np.float64] | None,
    nonfinite_mask: NDArray[np.bool_] | None,
    validate: bool,
    dtype: Any,
    time_chunk: int,
) -> NDArray[np.float64]:
    """Fill a preallocated posterior one time-block at a time.

    Opt-in (``time_chunk=N``) path for :func:`decode_position`: computes the
    Poisson log-likelihood for each ``spike_counts[start:stop]`` block via the
    R5 block machinery (:func:`_poisson_log_likelihood` +
    :func:`normalize_to_posterior` per block, as :func:`decode_position_summary`
    does) and writes the normalized posterior directly into the corresponding
    slice of a single preallocated ``(n_time, n_bins)`` output. The full
    ``(n_time, n_bins)`` log-likelihood and its working copy are never
    materialized, so the transient peak is ~1x over the returned posterior.

    The result is the same full materialized posterior as the ``time_chunk=None``
    path, but tolerance-equal (not byte-exact): the per-block likelihood matmul
    is a different BLAS shape than the full matmul, so it differs by ~1e-15 ULPs.
    MAP/argmax is identical and every row sums to 1.0.

    Parameters
    ----------
    env : Environment
        Spatial environment (used for ``n_bins``).
    spike_counts : NDArray[np.int64], shape (n_time, n_neurons)
        Validated spike counts.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Validated firing-rate maps.
    dt : float
        Time-bin width in seconds.
    prior : NDArray[np.float64] | None
        Prior over positions (1-D stationary or 2-D time-varying), or ``None``.
        Its shape is validated up front.
    nonfinite_mask : NDArray[np.bool_] | None, shape (n_neurons, n_bins)
        ``True`` where ``encoding_models`` is non-finite; ``None`` if all finite.
    validate : bool
        If ``True``, run :func:`_validate_output` on each posterior block.
    dtype : np.float32 or np.float64
        Stored/working dtype of the posterior.
    time_chunk : int
        Positive time-block size (rows per block).

    Returns
    -------
    posterior : NDArray, shape (n_time, n_bins)
        Full materialized posterior in ``dtype``; each row sums to 1.0.
    """
    out_dtype = _validate_posterior_dtype(dtype)
    n_time = spike_counts.shape[0]
    n_bins = encoding_models.shape[1]

    # Validate prior shape ONCE, up front, before the time-block loop -- the loop
    # only slices n_time rows, so a wrong-shaped 2-D prior would otherwise be
    # silently truncated or caught only at the final block. Mirrors
    # normalize_to_posterior's checks so both paths report mismatches the same.
    prior_is_time_varying = False
    prior_arr: NDArray[np.float64] | None = None
    if prior is not None:
        prior_arr = np.asarray(prior, dtype=np.float64)
        prior_is_time_varying = _validate_prior_shape(
            prior_arr, n_time=n_time, n_bins=n_bins
        )

    posterior = np.empty((n_time, n_bins), dtype=out_dtype)

    for start in range(0, n_time, time_chunk):
        stop = min(start + time_chunk, n_time)
        block_prior: NDArray[np.float64] | None = prior_arr
        if prior_is_time_varying and prior_arr is not None:
            block_prior = prior_arr[start:stop]
        # Compute the likelihood for THIS block only (never the full window) and
        # normalize it straight into the output slice. This is the ~1x-transient
        # core: no full (n_time, n_bins) log-likelihood is ever held.
        log_ll_block = _poisson_log_likelihood(
            spike_counts[start:stop], encoding_models, nonfinite_mask, dt
        )
        post_block = normalize_to_posterior(
            log_ll_block, prior=block_prior, dtype=dtype
        )
        if validate:
            _validate_output(post_block)
        posterior[start:stop] = post_block

    return cast("NDArray[np.float64]", posterior)


def _prepare_decode_inputs(
    env: Environment,
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64] | SpatialRatesLike,
    *,
    prior: NDArray[np.float64] | None,
    method: Literal["poisson"],
    validate: bool,
    context: str,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_] | None]:
    """Validate inputs and resolve the encoding model (no likelihood yet).

    Shared front half of :func:`decode_position` and
    :func:`decode_position_summary`: resolves a duck-typed encoding-result
    object, runs the unconditional correctness guards (no-finite-bins,
    bin-count agreement) and the optional ``validate=True`` checks, and detects
    non-finite encoding-model bins (to be treated as zero-rate observations).
    The actual log-likelihood is computed separately (per time-block in the
    streaming summary path) by :func:`_poisson_log_likelihood`.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64] or SpatialRatesLike
        Firing-rate maps, or a population rate result exposing
        ``firing_rates``.
    prior : NDArray[np.float64] | None
        Prior over positions (validated here when ``validate=True``).
    method : {"poisson"}
        Likelihood model. Only ``"poisson"`` is supported.
    validate : bool
        Whether to run the optional input validation checks.
    context : str
        Caller name used in error messages.

    Returns
    -------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts as an ndarray.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Validated firing-rate maps as an ndarray.
    nonfinite_mask : NDArray[np.bool_] | None, shape (n_neurons, n_bins)
        ``True`` where ``encoding_models`` is non-finite (NaN or Inf), or
        ``None`` when the model is entirely finite.
    """
    from neurospatial.encoding._validation import validate_env_fitted

    validate_env_fitted(env, context=context)

    # Validate method
    if method != "poisson":
        raise ValueError(
            f"Unknown method '{method}'. Currently only 'poisson' is supported."
        )

    # Accept a population rate result object (duck-typed): if it exposes a
    # ``firing_rates`` attribute, use that array directly. This removes the
    # np.stack([r.firing_rate ...]) glue between encode and decode.
    if hasattr(encoding_models, "firing_rates"):
        provenance = type(encoding_models).__name__
        firing_rates = encoding_models.firing_rates
        # The `.firing_rates` attribute is only a usable encoding model when it
        # is a 2-D (n_neurons, n_bins) numeric array. Some result objects expose
        # a `.firing_rates` that is None or a Mapping (e.g.
        # DirectionalPlaceFields.firing_rates is a dict keyed by direction);
        # passing those straight to np.asarray below yields an obscure
        # NumPy-internal TypeError. Reject them here with a clear message.
        if firing_rates is None:
            raise ValueError(
                f"{context}: the encoding result's `.firing_rates` must "
                f"be a 2-D (n_neurons, n_bins) array, got None "
                f"(from {provenance}.firing_rates)."
            )
        # Convert WITHOUT forcing a dtype so a float32 `.firing_rates` stays
        # float32 (matching the raw-array path, which never re-casts) -- forcing
        # dtype=float here would silently promote float32 to float64 and erase
        # part of the dtype=np.float32 memory win.
        try:
            firing_rates_arr = np.asarray(firing_rates)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context}: the encoding result's `.firing_rates` must "
                f"be a 2-D (n_neurons, n_bins) array, got "
                f"{type(firing_rates).__name__} (from {provenance}.firing_rates)."
            ) from exc
        # Validate it became a usable numeric float array. A Mapping/dict
        # produces a 0-d object-dtype array (not floating); an integer rate map
        # is unusual but should still work, so promote it to float64.
        if not np.issubdtype(firing_rates_arr.dtype, np.floating):
            if np.issubdtype(firing_rates_arr.dtype, np.integer):
                firing_rates_arr = firing_rates_arr.astype(np.float64)
            else:
                # object/non-numeric (e.g. a dict that became a 0-d object
                # array) -- reject with the same clear message.
                raise ValueError(
                    f"{context}: the encoding result's `.firing_rates` must "
                    f"be a 2-D (n_neurons, n_bins) array, got "
                    f"{type(firing_rates).__name__} "
                    f"(from {provenance}.firing_rates)."
                )
        if firing_rates_arr.ndim != 2:
            raise ValueError(
                f"{context}: the encoding result's `.firing_rates` must "
                f"be a 2-D (n_neurons, n_bins) array, got a "
                f"{firing_rates_arr.ndim}-D array with shape "
                f"{firing_rates_arr.shape} (from {provenance}.firing_rates)."
            )
        encoding_models = firing_rates_arr

    # Convert inputs to arrays
    spike_counts = np.asarray(spike_counts)
    encoding_models = np.asarray(encoding_models)

    # A model with NO finite bins -- every spatial bin is non-finite (NaN OR
    # Inf) across all neurons -- is unusable, not merely low-occupancy.
    # Excluding/clipping all terms would leave every bin's log-likelihood
    # degenerate, and the "uniform" degeneracy handler would then hand back a
    # confident-looking uniform posterior over positions that carry zero
    # information. Refuse loudly here instead of returning a meaningless
    # posterior. This is an unconditional guard (fires even with
    # validate=False) and recognizes Inf as well as NaN -- the all-NaN case is
    # a subset of "no finite bins". A bin is usable if it is finite for any
    # neuron; partial models (>=1 finite bin) fall through and decode normally.
    # (Partial-Inf models are out of scope here: validate=True rejects Inf via
    # _validate_inputs, and this guard will not fire when a finite bin exists.)
    if encoding_models.ndim == 2 and encoding_models.size:
        finite_bin = np.isfinite(encoding_models).any(axis=0)
        if not finite_bin.any():
            raise ValueError(
                "encoding_models has no finite bins; every spatial bin is "
                "non-finite (NaN or Inf) across all neurons -- cannot decode. "
                "Recompute place fields or pass fill_value (e.g. "
                "fill_value=0.0 to the encoder) so the model is explicitly "
                "zero-rate there."
            )

    # Defense-in-depth for non-finite encoding-model bins. Two distinct
    # sources:
    #   * NaN bins -- e.g. an encoder's min_occupancy mask with
    #     fill_value=None. These are the low-occupancy "unobserved" bins.
    #   * Inf bins -- a degenerate rate at one bin (only reachable with
    #     validate=False; validate=True rejects Inf via _validate_inputs).
    # Both are treated identically as zero-rate observations and excluded from
    # that neuron's Poisson contribution at that bin (handled inside
    # _log_poisson_likelihood_nan_safe), so a partial-Inf model such as
    # rates=[inf, inf, 5] concentrates mass on the one finite bin instead of
    # collapsing to a uniform posterior. We detect them HERE, before the
    # validate=True guard runs, emitting one warning per call.
    #
    # The validation substitution below stays NaN-ONLY on purpose: the
    # validate=True NaN guard should not reject NaN bins the zero-rate path has
    # already claimed, but Inf bins must still reach _validate_inputs and be
    # rejected. So validate=True keeps its non-finite rejection unchanged, and
    # the Inf-as-zero-rate handling only ever fires when validate=False.
    encoding_model_nan_mask: NDArray[np.bool_] | None = None
    encoding_model_nonfinite_mask: NDArray[np.bool_] | None = None
    if encoding_models.dtype.kind == "f" and not np.isfinite(encoding_models).all():
        encoding_model_nonfinite_mask = ~np.isfinite(encoding_models)
        nan_only = np.isnan(encoding_models)
        encoding_model_nan_mask = nan_only if nan_only.any() else None
        n_bad = int(encoding_model_nonfinite_mask.sum())
        warnings.warn(
            f"encoding_models contains {n_bad} non-finite bin(s) (NaN or Inf; "
            "e.g. low-occupancy bins masked by the encoder's min_occupancy "
            "with fill_value=None). Treating each as a zero-rate observation "
            "(excluded from that neuron's Poisson contribution at that bin). "
            "Pass fill_value=0.0 to the encoder to silence this and make the "
            "model explicitly zero-rate there.",
            UserWarning,
            stacklevel=3,
        )

    # Validate inputs if requested. Pass a NaN-free view of encoding_models so
    # the validate=True NaN guard does not reject the bins the zero-rate path
    # has already claimed; Inf and negative-rate checks still apply (the
    # substitution is NaN-only, so Inf bins are still seen and rejected).
    if validate:
        encoding_models_for_validation = encoding_models
        if encoding_model_nan_mask is not None:
            encoding_models_for_validation = np.where(
                encoding_model_nan_mask, 0.0, encoding_models
            )
        _validate_inputs(spike_counts, encoding_models_for_validation, prior, env)

    # Bin-count agreement is a *correctness* check, not an opt-in one:
    # an encoding model with the wrong number of bins produces a
    # well-formed posterior over the WRONG positions with no error.
    # Enforce it even when validate=False.
    if encoding_models.ndim != 2:
        raise ValueError(
            f"encoding_models must be 2-D (n_neurons, n_bins), got shape "
            f"{encoding_models.shape}."
        )
    if encoding_models.shape[1] != env.n_bins:
        raise ValueError(
            f"encoding_models has {encoding_models.shape[1]} bins (axis 1) "
            f"but env has {env.n_bins} active bins. The encoding models must "
            f"be defined on the same environment used for decoding."
        )

    return spike_counts, encoding_models, encoding_model_nonfinite_mask


def _poisson_log_likelihood(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    nonfinite_mask: NDArray[np.bool_] | None,
    dt: float,
) -> NDArray[np.float64]:
    """Compute the Poisson log-likelihood for prepared decode inputs.

    Thin dispatch over :func:`log_poisson_likelihood` and the NaN-safe variant,
    callable on a full ``spike_counts`` or on a single time-block slice (the
    streaming summary path computes it block-by-block to avoid materializing the
    full ``(n_time_bins, n_bins)`` log-likelihood).

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_block, n_neurons)
        Spike counts (possibly a time-block slice).
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Validated firing-rate maps.
    nonfinite_mask : NDArray[np.bool_] | None, shape (n_neurons, n_bins)
        ``True`` where ``encoding_models`` is non-finite; ``None`` if all finite.
    dt : float
        Time-bin width in seconds.

    Returns
    -------
    log_likelihood : NDArray[np.float64], shape (n_block, n_bins)
        Poisson log-likelihood, with non-finite encoding-model bins excluded.
    """
    # When some encoding-model bins are non-finite (NaN or Inf), route through
    # the NaN-safe path that excludes each such (neuron, bin) from the per-bin
    # Poisson sum. Inf bins only reach here with validate=False.
    if nonfinite_mask is not None:
        log_ll = _log_poisson_likelihood_nan_safe(
            spike_counts, encoding_models, nonfinite_mask, dt=dt
        )
    else:
        log_ll = log_poisson_likelihood(spike_counts, encoding_models, dt=dt)
    return log_ll


def _reduce_posterior_block(
    posterior_block: NDArray[np.float64],
    bin_centers: NDArray[np.float64],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Reduce a posterior time-block to per-time scalars/vectors.

    Computes, for each row (time bin) of ``posterior_block``: the MAP bin
    index, the posterior-mean position, the posterior entropy (bits), and the
    peak (max) posterior probability. Mirrors the reductions on
    :class:`DecodingResult` so streaming and full-posterior paths agree.

    Parameters
    ----------
    posterior_block : NDArray[np.float64], shape (n_block, n_bins)
        Posterior probabilities for a block of time bins (rows sum to 1).
    bin_centers : NDArray[np.float64], shape (n_bins, n_dims)
        Environment bin-center coordinates.

    Returns
    -------
    map_bin : NDArray[np.int64], shape (n_block,)
        MAP bin index per time bin (``argmax`` along bins).
    mean_position : NDArray[np.float64], shape (n_block, n_dims)
        Posterior-mean position per time bin (``posterior @ bin_centers``).
    entropy : NDArray[np.float64], shape (n_block,)
        Posterior entropy in bits.
    peak_prob : NDArray[np.float64], shape (n_block,)
        Max posterior probability per time bin.
    """
    # Upcast the block once so all reductions accumulate in float64. The block
    # is already the smallest array in play (one time-chunk), so this does not
    # change the memory story, but it prevents float32 working-posterior dtypes
    # from silently losing precision in the entropy/peak/mean reductions.
    #
    # NOTE: this summary path upcasts for precision, whereas the full-posterior
    # path (DecodingResult.posterior_entropy) historically accumulates entropy
    # in the stored dtype. The two are intentionally left consistent with each
    # other for now; a future change that fixes one should align both.
    pb = np.asarray(posterior_block, dtype=np.float64)

    map_bin = np.argmax(pb, axis=1).astype(np.int64)
    mean_position = pb @ bin_centers
    peak_prob = pb.max(axis=1)

    # Entropy (bits), mask-based to avoid log(0) bias -- identical to
    # DecodingResult.posterior_entropy.
    p = np.clip(pb, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log2(p), 0.0)
    entropy = -np.sum(p * log_p, axis=1)

    return (
        map_bin,
        np.asarray(mean_position, dtype=np.float64),
        np.asarray(entropy, dtype=np.float64),
        np.asarray(peak_prob, dtype=np.float64),
    )


def _decode_and_reduce_block(
    counts_block: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    dt: float,
    bin_centers: NDArray[np.float64],
    *,
    prior_block: NDArray[np.float64] | None,
    nonfinite_mask: NDArray[np.bool_] | None,
    validate: bool,
    dtype: Any,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Decode one time-block and reduce it to per-time vectors.

    Single shared implementation of the streaming summary inner loop, used by
    BOTH :func:`decode_position_summary` (array-first) and
    :func:`~neurospatial.decoding.decode_session_summary` (which streams the
    time-binning). Computes the Poisson log-likelihood for this block only,
    normalizes it to a posterior block, optionally validates the block's row
    sums, and reduces it to per-time scalars/vectors. Never holds more than one
    block's ``(n_block, n_bins)`` posterior at a time.

    Parameters
    ----------
    counts_block : NDArray[np.int64], shape (n_block, n_neurons)
        Spike counts for this time-block.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Validated firing-rate maps (whole session; small).
    dt : float
        Time-bin width in seconds.
    bin_centers : NDArray[np.float64], shape (n_bins, n_dims)
        Environment bin-center coordinates.
    prior_block : NDArray[np.float64] | None
        Prior for this block: a 1-D stationary prior (reused per block) or the
        ``(n_block, n_bins)`` slice of a 2-D time-varying prior, or ``None``.
    nonfinite_mask : NDArray[np.bool_] | None, shape (n_neurons, n_bins)
        ``True`` where ``encoding_models`` is non-finite; ``None`` if all finite.
    validate : bool
        If ``True``, run the per-block posterior-sum output check.
    dtype : np.float32 or np.float64
        Working dtype of the posterior block.

    Returns
    -------
    map_bin : NDArray[np.int64], shape (n_block,)
        MAP bin index per time bin.
    map_position : NDArray[np.float64], shape (n_block, n_dims)
        MAP position per time bin (``bin_centers[map_bin]``).
    mean_position : NDArray[np.float64], shape (n_block, n_dims)
        Posterior-mean position per time bin.
    entropy : NDArray[np.float64], shape (n_block,)
        Posterior entropy (bits) per time bin.
    peak_prob : NDArray[np.float64], shape (n_block,)
        Max posterior probability per time bin.
    """
    log_ll_block = _poisson_log_likelihood(
        counts_block, encoding_models, nonfinite_mask, dt
    )
    post_block = normalize_to_posterior(
        log_ll_block,
        prior=prior_block,
        dtype=dtype,
    )
    if validate:
        _validate_output(post_block)
    block_map_bin, block_mean, block_entropy, block_peak = _reduce_posterior_block(
        post_block, bin_centers
    )
    block_map_position = bin_centers[block_map_bin]
    # post_block goes out of scope and is reclaimed before the next block.
    return block_map_bin, block_map_position, block_mean, block_entropy, block_peak


def decode_position_summary(
    env: Environment,
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64] | SpatialRatesLike,
    dt: float,
    *,
    prior: NDArray[np.float64] | None = None,
    method: Literal["poisson"] = "poisson",
    times: NDArray[np.float64] | None = None,
    validate: bool = True,
    dtype: Any = np.float64,
    time_chunk: int = 1024,
) -> DecodingSummary:
    """Memory-safe decode that streams per-time reductions, not the posterior.

    Same inputs as :func:`decode_position`, but **never materializes the full
    ``(n_time_bins, n_bins)`` posterior**. It computes the posterior one
    time-block at a time, reduces each block to per-time scalars/vectors (MAP
    position, mean position, entropy, peak probability), discards the block,
    and returns a :class:`~neurospatial.decoding.DecodingSummary` holding only
    the ``(n_time_bins, ...)`` reductions. Use this when even the stored dense
    posterior is too large to hold (e.g. a 1-hour session at 25 ms / 5000 bins
    is ~5.8 GB stored).

    This is a SEPARATE function with its OWN return type by design: it does not
    change :func:`decode_position`'s return contract, whose ``.posterior`` is a
    real, fully-materialized ndarray everywhere.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64] or SpatialRatesResult, shape (n_neurons, n_bins)
        Firing-rate maps (place fields) for each neuron, or a population rate
        result object exposing ``firing_rates``. Same semantics (NaN/Inf
        handling, no-finite-bins guard) as :func:`decode_position`.
    dt : float
        Time-bin width in seconds.
    prior : NDArray[np.float64] | None, default=None
        Prior over positions. Same shapes/semantics as :func:`decode_position`.
    method : {"poisson"}, default="poisson"
        Likelihood model. Currently only Poisson is supported.
    times : NDArray[np.float64] | None, default=None
        Time-bin centers (seconds). Stored in the returned summary.
    validate : bool, default=True
        Run the same input validation as :func:`decode_position`. The
        per-row posterior-sum output check is performed per block.
    dtype : np.float32 or np.float64, default=np.float64
        Working dtype of each posterior block. ``np.float32`` halves the
        transient per-block memory.
    time_chunk : int, default=1024
        Time-block size (rows) processed at a time; must be a positive integer.
        Smaller blocks use less transient memory. ``None`` is **rejected**:
        processing all time bins in one block would materialize the full
        ``(n_time_bins, n_bins)`` posterior transiently, defeating the purpose
        of this streamed summary decoder. Use :func:`decode_position` if you
        want the full posterior.

    Returns
    -------
    DecodingSummary
        Per-time reductions: ``map_position`` / ``map_bin`` / ``mean_position``
        / ``posterior_entropy`` / ``peak_prob`` (all length ``n_time_bins``),
        plus ``times`` and ``env``.

    Raises
    ------
    ValueError
        If ``time_chunk`` is ``None`` or not a positive integer. Plus the same
        conditions as :func:`decode_position` (invalid method, no finite bins,
        bin-count mismatch, bad ``times`` length, validation failures).

    Notes
    -----
    The per-time reductions are bit-for-bit identical to reducing the full
    :class:`DecodingResult` posterior: ``map_position`` / ``map_bin`` match
    exactly; ``mean_position`` / ``posterior_entropy`` / ``peak_prob`` match to
    floating-point tolerance. See :func:`decode_position` for a path that keeps
    the full posterior (with optional ``dtype`` / ``time_chunk`` memory knobs).

    See Also
    --------
    decode_position : Full-posterior decode (return contract unchanged).
    DecodingSummary : Streamed per-time reductions container.
    decode_session_summary : One-call encode->bin->summary-decode wrapper.
    """
    if time_chunk is None:
        raise ValueError(
            "time_chunk=None is not allowed for decode_position_summary: this "
            "streamed summary decoder processes the posterior one time-block at "
            "a time, and None would materialize the full (n_time, n_bins) "
            "posterior, defeating its purpose. Use decode_position if you want "
            "the full posterior, or pass a positive time_chunk (default 1024) "
            "here."
        )
    time_chunk = _validate_time_chunk(time_chunk, allow_none=False)

    # Validate dt up front so a non-numeric/bool/non-finite dt raises the same
    # clean message as every other decoding entry point, rather than leaking a
    # raw TypeError (dt="0.1") or silently accepting dt=True as 1.
    dt = validate_dt(dt)

    spike_counts, encoding_models, nonfinite_mask = _prepare_decode_inputs(
        env,
        spike_counts,
        encoding_models,
        prior=prior,
        method=method,
        validate=validate,
        context="decode_position_summary",
    )

    n_time = spike_counts.shape[0]

    # Validate times length up front (cheap, avoids a wasted full decode).
    times_arr: NDArray[np.float64] | None = None
    if times is not None:
        times_arr = np.asarray(times, dtype=np.float64)
        if times_arr.ndim != 1 or len(times_arr) != n_time:
            raise ValueError(
                f"Length mismatch: times has {times_arr.shape} but there are "
                f"{n_time} time bins. `times` must be a 1-D array of bin "
                f"centers, one per row of spike_counts."
            )

    # Validate prior shape ONCE, up front, before the time-block loop. The
    # loop only iterates over n_time rows, so an over-long 2-D prior would be
    # silently truncated (the extra rows never sliced) and a short 2-D prior
    # would only be caught by normalize_to_posterior at the final block. Route
    # through the shared _validate_prior_shape helper so decode_position and
    # decode_position_summary enforce one prior-shape rule with one message.
    if prior is not None:
        _validate_prior_shape(
            np.asarray(prior),
            n_time=n_time,
            n_bins=encoding_models.shape[1],
        )

    bin_centers = np.asarray(env.bin_centers, dtype=np.float64)
    n_dims = bin_centers.shape[1]

    map_bin = np.empty(n_time, dtype=np.int64)
    map_position = np.empty((n_time, n_dims), dtype=np.float64)
    mean_position = np.empty((n_time, n_dims), dtype=np.float64)
    posterior_entropy = np.empty(n_time, dtype=np.float64)
    peak_prob = np.empty(n_time, dtype=np.float64)

    # A time-varying (2-D) prior must be sliced to each block so its shape
    # matches the block log-likelihood; a stationary (1-D) prior is reused.
    prior_is_time_varying = prior is not None and np.asarray(prior).ndim == 2

    # time_chunk is guaranteed a positive int by the up-front guard, so the
    # block is always bounded — the full (n_time, n_bins) posterior is never
    # materialized in one shot.
    block = time_chunk
    for start in range(0, n_time, block):
        stop = min(start + block, n_time)
        block_prior = prior
        if prior_is_time_varying:
            block_prior = np.asarray(prior)[start:stop]
        # Decode + reduce THIS time-block only via the shared inner-loop helper.
        # Computing the likelihood per block (rather than once for the whole
        # session) is what keeps peak memory at one block, never the full
        # (n_time, n_bins) array. The math matches decode_position exactly. The
        # same helper backs decode_session_summary's streaming-binning loop.
        (
            block_map_bin,
            block_map_position,
            block_mean,
            block_entropy,
            block_peak,
        ) = _decode_and_reduce_block(
            spike_counts[start:stop],
            encoding_models,
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

    return DecodingSummary(
        times=times_arr,
        map_position=map_position,
        mean_position=mean_position,
        posterior_entropy=posterior_entropy,
        peak_prob=peak_prob,
        map_bin=map_bin,
        env=env,
    )


def _log_poisson_likelihood_nan_safe(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    nan_mask: NDArray[np.bool_],
    dt: float,
    *,
    min_rate: float = 1e-10,
) -> NDArray[np.float64]:
    """Poisson log-likelihood that excludes non-finite encoding-model bins.

    Identical to :func:`~neurospatial.decoding.likelihood.log_poisson_likelihood`
    except that each ``(neuron, bin)`` flagged in ``nan_mask`` contributes
    zero to the per-bin log-likelihood: both that neuron's spike term
    ``n_i * log(lambda_i * dt)`` and its rate-penalty term ``-lambda_i * dt``
    are dropped at that bin, as if the neuron had not been observed there.
    This is the "treat non-finite encoding-model bins (NaN or Inf) as zero-rate
    (excluded from the Poisson sum)" defense-in-depth behavior.

    A bin that is non-finite for *every* neuron is a special case: it has no
    observing neuron, so excluding all terms would leave a neutral
    log-likelihood of 0 and let an uninformative bin spuriously win the MAP.
    Such all-masked bins are therefore set to ``-inf`` (zero posterior mass).
    Partial bins -- masked for some neurons but observed by at least one other
    -- still decode normally from the observing neurons.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps in Hz; may contain NaN at masked bins.
    nan_mask : NDArray[np.bool_], shape (n_neurons, n_bins)
        ``True`` where ``encoding_models`` is non-finite (NaN or Inf). Those
        entries are excluded.
    dt : float
        Time bin width in seconds.
    min_rate : float, default=1e-10
        Minimum firing rate floor to avoid log(0).

    Returns
    -------
    log_likelihood : NDArray[np.float64], shape (n_time_bins, n_bins)
        Log-likelihood up to an additive constant per time bin, with NaN
        encoding-model bins excluded from each neuron's contribution and
        all-NaN bins set to ``-inf``.
    """
    dt = validate_dt(dt)
    if min_rate <= 0:
        raise ValueError(
            f"min_rate must be positive, got {min_rate}. "
            f"This floor prevents log(0) in likelihood computation."
        )

    spike_counts = np.asarray(spike_counts)
    encoding_models = np.asarray(encoding_models, dtype=np.float64)

    if spike_counts.ndim != 2:
        raise ValueError(
            f"spike_counts must be 2-D with shape (n_time_bins, n_neurons), "
            f"got ndim={spike_counts.ndim} with shape {spike_counts.shape}."
        )
    if encoding_models.ndim != 2:
        raise ValueError(
            f"encoding_models must be 2-D with shape (n_neurons, n_bins), "
            f"got ndim={encoding_models.ndim} with shape {encoding_models.shape}."
        )
    if spike_counts.shape[1] != encoding_models.shape[0]:
        raise ValueError(
            f"Neuron-count mismatch: spike_counts has {spike_counts.shape[1]} "
            f"neurons (axis 1) but encoding_models has {encoding_models.shape[0]} "
            f"neurons (axis 0). These must agree for the Poisson likelihood."
        )

    # Replace NaN rates with the floor so log/exp stay finite, then zero out
    # both terms at the masked entries so they contribute nothing to the sum.
    rates = np.where(nan_mask, min_rate, encoding_models)
    clipped_rates = np.maximum(rates, min_rate)
    expected_counts = clipped_rates * dt  # (n_neurons, n_bins)
    log_expected = np.log(expected_counts)  # (n_neurons, n_bins)

    # Exclude masked (neuron, bin) entries: set their per-neuron contributions
    # to zero so they drop out of both the spike term and the rate penalty.
    log_expected = np.where(nan_mask, 0.0, log_expected)
    expected_counts = np.where(nan_mask, 0.0, expected_counts)

    # Spike term: (n_time_bins, n_neurons) @ (n_neurons, n_bins)
    spike_term = spike_counts.astype(np.float64) @ log_expected
    # Rate penalty: -sum_i lambda_i * dt over neurons -> (n_bins,)
    rate_penalty = -np.sum(expected_counts, axis=0)

    log_likelihood = spike_term + rate_penalty

    # A bin that is NaN for EVERY neuron carries no information: there is no
    # observing neuron to decode it from, so its excluded-term log-likelihood
    # collapses to a neutral 0 and would let an uninformative bin win the MAP.
    # Force such all-NaN bins to -inf so they receive zero posterior mass and
    # can never be the argmax. Partial-NaN bins (NaN for some neurons but
    # observed by >=1 other) are untouched and still decode from the observing
    # neurons.
    all_nan = np.all(nan_mask, axis=0)  # (n_bins,)
    if all_nan.any():
        log_likelihood[:, all_nan] = -np.inf

    return cast("NDArray[np.float64]", log_likelihood)


def _validate_inputs(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    prior: NDArray[np.float64] | None,
    env: Environment,
) -> None:
    """Validate inputs for ``decode_position``.

    Raises
    ------
    ValueError
        If inputs contain NaN or Inf values, if any spike count is
        negative, if any encoding-model rate is negative, or if the
        encoding-model bin count does not match ``env.n_bins``.
    """
    # Check spike counts: finite first (NaN/Inf can't pass the < 0 check
    # cleanly), then non-negative. The error message tells the user how
    # many bad entries we saw and the worst value, since
    # "spike_counts has a negative value" without a count or value is
    # not actionable on a 100k-row array.
    if not np.isfinite(spike_counts).all():
        n_bad = int(np.sum(~np.isfinite(spike_counts)))
        raise ValueError(
            f"spike_counts contains {n_bad} NaN or Inf entries. "
            "All spike counts must be finite non-negative integers."
        )
    if (spike_counts < 0).any():
        n_negative = int(np.sum(spike_counts < 0))
        worst_count = float(spike_counts.min())
        raise ValueError(
            f"spike_counts contains {n_negative} negative entr"
            f"{'y' if n_negative == 1 else 'ies'} (min: {worst_count:g}). "
            "Spike counts represent spike events per time bin and must "
            "be non-negative integers."
        )

    # Spike counts must also be integer-valued. Allow integer dtypes
    # outright and float dtypes whose values happen to be exact integers
    # (a common shape after building counts via histogramming and storing
    # them in a float64 column for ergonomic reasons). A fractional value
    # like 0.5 is silently meaningless under the Poisson likelihood and
    # would produce a "valid"-looking posterior; reject it at the boundary.
    if not np.issubdtype(spike_counts.dtype, np.integer):
        rounded = np.floor(spike_counts)
        is_integer_valued = np.equal(spike_counts, rounded)
        if not is_integer_valued.all():
            n_fractional = int(np.sum(~is_integer_valued))
            # Pick the entry with the largest fractional part so the user
            # can grep for it in their input rather than guessing.
            fractional_part = np.abs(spike_counts - rounded)
            worst_value = float(spike_counts.flat[int(fractional_part.argmax())])
            raise ValueError(
                f"spike_counts contains {n_fractional} fractional entr"
                f"{'y' if n_fractional == 1 else 'ies'} (e.g., {worst_value:g}). "
                "Spike counts must be integer-valued. Cast with "
                "`spike_counts.astype(np.int64)` after confirming the "
                "non-integer values are not a binning bug upstream."
            )

    # Check encoding models: finite then non-negative (firing rates can't
    # physically be negative).
    if not np.isfinite(encoding_models).all():
        n_bad = int(np.sum(~np.isfinite(encoding_models)))
        raise ValueError(
            f"encoding_models contains {n_bad} NaN or Inf entries. "
            "All firing rates must be finite non-negative values (Hz)."
        )
    if (encoding_models < 0).any():
        n_negative = int(np.sum(encoding_models < 0))
        worst_rate = float(encoding_models.min())
        raise ValueError(
            f"encoding_models contains {n_negative} negative entr"
            f"{'y' if n_negative == 1 else 'ies'} (min: {worst_rate:.6g} Hz). "
            "Firing rates must be non-negative."
        )

    # Encoding models must be defined on the decoding environment.
    if encoding_models.ndim == 2 and encoding_models.shape[1] != env.n_bins:
        raise ValueError(
            f"encoding_models has {encoding_models.shape[1]} bins (axis 1) "
            f"but env has {env.n_bins} active bins. Recompute the place "
            f"fields on this environment before decoding."
        )

    # Check prior if provided. Convert to ndarray first because the
    # downstream normalize_to_posterior accepts array-like (list, tuple)
    # priors via np.asarray; performing the finite/non-negative/sum
    # checks on the raw object would crash with a TypeError ("<' not
    # supported between instances of 'list' and 'int'") on a perfectly
    # legitimate input. Then check:
    #
    # - finite (NaN/Inf can't pass the < 0 check cleanly),
    # - non-negative (a probability mass cannot be negative),
    # - has positive total mass (a zero-sum prior would otherwise be
    #   silently rebuilt as a uniform prior by normalize_to_posterior's
    #   1e-10 clip, which is the silent-wrong-result path the validator
    #   exists to prevent). For time-varying priors, every row must
    #   carry positive mass.
    if prior is not None:
        prior_arr = np.asarray(prior, dtype=np.float64)
        if not np.isfinite(prior_arr).all():
            n_bad = int(np.sum(~np.isfinite(prior_arr)))
            raise ValueError(
                f"prior contains {n_bad} NaN or Inf entries. "
                "Prior must be finite non-negative values."
            )
        if (prior_arr < 0).any():
            n_negative = int(np.sum(prior_arr < 0))
            worst_prior = float(prior_arr.min())
            raise ValueError(
                f"prior contains {n_negative} negative entr"
                f"{'y' if n_negative == 1 else 'ies'} (min: {worst_prior:.6g}). "
                "A prior over positions is a probability mass and must "
                "be non-negative."
            )
        if prior_arr.ndim == 1:
            total = float(prior_arr.sum())
            if total <= 0:
                raise ValueError(
                    f"prior has zero total mass (sum={total:.6g}). A prior "
                    "over positions must integrate to a positive total; an "
                    "all-zero prior would otherwise be silently rebuilt as "
                    "a uniform prior by internal normalization. Pass a "
                    "non-zero prior, or pass `prior=None` for an explicit "
                    "uniform."
                )
        elif prior_arr.ndim == 2:
            row_sums = prior_arr.sum(axis=-1)
            zero_rows = ~(row_sums > 0)
            if zero_rows.any():
                n_zero = int(zero_rows.sum())
                raise ValueError(
                    f"time-varying prior has {n_zero} row"
                    f"{'' if n_zero == 1 else 's'} with zero total mass "
                    "(e.g., prior[t, :] all zeros). Each time bin's prior "
                    "must integrate to a positive total or it is silently "
                    "rebuilt as a uniform prior by internal normalization."
                )


def _validate_output(posterior: NDArray[np.float64]) -> None:
    """Validate output posterior.

    Rows that are entirely NaN are treated as intentional degenerate rows
    (produced by ``normalize_to_posterior(..., handle_degenerate="nan")``)
    and are excluded from both the finite check and the row-sum check.
    Otherwise the all-NaN row would trip the NaN/Inf guard, and even if it
    passed, ``posterior.sum(axis=1)`` would be NaN for that row and raise a
    misleading "rows do not sum to 1.0" error.

    Raises
    ------
    ValueError
        If any non-degenerate row contains NaN/Inf, or if any
        non-degenerate row does not sum to 1.0.
    """
    # Rows that are entirely NaN are intentional degenerate rows; skip them.
    all_nan_rows = np.isnan(posterior).all(axis=1)
    finite_rows = posterior[~all_nan_rows]

    # Check for NaN/Inf among the non-degenerate rows. A partial NaN/Inf in
    # an otherwise-finite row still signals real numerical instability.
    if not np.isfinite(finite_rows).all():
        raise ValueError(
            "Output posterior contains NaN or Inf values in non-degenerate "
            "rows. This may indicate numerical instability."
        )

    # Check row sums on the non-degenerate rows only.
    if finite_rows.size > 0:
        row_sums = finite_rows.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            max_deviation = np.abs(row_sums - 1.0).max()
            raise ValueError(
                f"Posterior rows do not sum to 1.0. "
                f"Maximum deviation: {max_deviation:.2e}"
            )
