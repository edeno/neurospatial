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
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import IncompatibleEnvironmentError, _format_error
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.decoding._result import DecodingResult, DecodingSummary
from neurospatial.decoding._time_bins import validate_dt
from neurospatial.decoding.likelihood import log_poisson_likelihood

if TYPE_CHECKING:
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment


def _validate_encoding_environment(
    env: Environment, encoding_models: Any, *, context: str
) -> None:
    """Check ``env`` and a rate result's spatial meaning before using either."""
    from neurospatial.encoding import _validate_env_argument

    _validate_env_argument(
        env, context=context, arguments="spike_counts, encoding_models, dt=dt"
    )
    source = getattr(encoding_models, "env", None)
    if source is None or source is env:
        return

    from neurospatial.environment import _environment_differences

    bins, topology = _environment_differences(source, env)
    mismatch = bins or topology
    if mismatch is not None:
        raise IncompatibleEnvironmentError(
            f"{context}: the encoding result's environment differs from the "
            f"decoding environment in {mismatch}.",
            fix="decode on the result's env, or recompute rates on the decoding environment.",
            first=source,
            second=env,
        )


def _rates_and_prior(
    env: Environment,
    encoding_models: NDArray[np.float64] | SpatialRatesResult,
    prior: Any,
    n_time: int,
    *,
    context: str,
) -> tuple[Any, Any]:
    """Return the rate array and the prior restricted to training support.

    A :class:`~neurospatial.encoding.spatial.SpatialRatesResult` contributes
    its ``spatial_support``: unsupported bins get a finite placeholder rate and
    zero prior mass. Other objects with ``firing_rates`` are refused, because
    only spatial rate results carry the support and environment the decode
    needs.
    """
    from neurospatial.decoding._support import models_on_support, resolve_prior
    from neurospatial.encoding.spatial import SpatialRatesResult

    _validate_encoding_environment(env, encoding_models, context=context)
    if isinstance(encoding_models, SpatialRatesResult):
        support = encoding_models.spatial_support
        return (
            models_on_support(np.asarray(encoding_models.firing_rates), support),
            resolve_prior(env, prior, support, n_time, context=context),
        )
    if hasattr(encoding_models, "firing_rates"):
        raise TypeError(
            _format_error(
                f"{context} expects a rate array or a SpatialRatesResult, got "
                f"{type(encoding_models).__name__}.",
                why="position decoding needs spatial rate maps with their "
                "training support and environment.",
                fix="pass compute_spatial_rates(env, spike_times, times, "
                "positions), or an (n_neurons, n_bins) array of rates on env",
            )
        )
    return encoding_models, resolve_prior(env, prior, None, n_time, context=context)


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
    handle_degenerate: Literal["uniform", "nan", "raise"],
    out: NDArray[np.float64],
    fill: NDArray[np.float64] | NDArray[np.bool_] | None = None,
) -> None:
    """Normalize one time-block of log-likelihood in place into ``out``.

    Performs the log-sum-exp softmax over the last axis and the degenerate-row
    handling for a single block, writing the result into the preallocated
    ``out`` array. ``ll_block`` is the prior-applied log-likelihood for the
    block; ``out`` must be the same shape and the desired output dtype.

    A row is degenerate when every entry is ``-inf``: no bin the prior allows
    has positive likelihood. With ``"uniform"``, ``fill`` (shape ``(n_bins,)``
    or matching ``ll_block``) holds non-negative weights normalized into each
    degenerate row: the positive-prior mask for :func:`normalize_to_posterior`,
    the prior itself for the decoders. A row whose weights are all zero becomes
    NaN; ``None`` fills ``1 / n_bins``. Rows containing NaN or ``+inf`` raise.
    """
    bad_rows = np.isnan(ll_block).any(axis=-1) | np.isposinf(ll_block).any(axis=-1)
    if bad_rows.any():
        raise ValueError(
            _format_error(
                f"log_likelihood has {int(bad_rows.sum())} row(s) containing NaN "
                "or +inf.",
                why="a posterior can be normalized only from finite or -inf "
                "log-likelihoods; NaN comes from a NaN spike count or rate.",
                fix="compute the log-likelihood from finite, non-negative rates; "
                "decode a rate result with decode_position(env, counts, rates, "
                "dt=dt) so unsupported bins are excluded.",
            )
        )
    ll_max = ll_block.max(axis=-1, keepdims=True)
    degenerate_mask = np.isneginf(ll_max[..., 0])

    # Degenerate rows give ``-inf - -inf = nan`` here; they are overwritten
    # below, so the "invalid value" warning is suppressed.
    with np.errstate(invalid="ignore"):
        np.exp(ll_block - ll_max, out=out)
    out /= out.sum(axis=-1, keepdims=True)

    if not degenerate_mask.any():
        return
    if handle_degenerate == "raise":
        raise ValueError(
            _format_error(
                f"Found {int(degenerate_mask.sum())} degenerate row(s) whose "
                "log-likelihood is -inf in every bin the prior allows.",
                why="such a row has no likelihood mass to normalize.",
                fix="pass handle_degenerate='uniform' to keep the prior for "
                "these rows, or 'nan' to mark them missing.",
            )
        )
    if handle_degenerate == "nan":
        out[degenerate_mask] = np.nan
    elif fill is None:
        out[degenerate_mask] = 1.0 / ll_block.shape[-1]
    else:
        weights = np.broadcast_to(fill, ll_block.shape)[degenerate_mask]
        total = weights.sum(axis=-1, keepdims=True)
        out[degenerate_mask] = np.divide(
            weights,
            total,
            out=np.full(weights.shape, np.nan, dtype=out.dtype),
            where=total > 0,
        )


def normalize_to_posterior(
    log_likelihood: NDArray[np.float64],
    *,
    prior: NDArray[np.float64] | None = None,
    handle_degenerate: Literal["uniform", "nan", "raise"] = "uniform",
    dtype: Any = np.float64,
    time_chunk: int | None = None,
) -> NDArray[np.float64]:
    """Convert log-likelihood to posterior using Bayes' rule.

    Applies the formula:

        P(position | spikes) = P(spikes | position) * P(position) / P(spikes)

    using numerically stable log-sum-exp normalization over the last axis.

    Parameters
    ----------
    log_likelihood : NDArray[np.float64], shape (n_time_bins, n_bins)
        Log-likelihood from `log_poisson_likelihood` or similar. Entries must
        be finite or ``-inf``.
    prior : NDArray[np.float64] | None, default=None
        Non-negative weight per bin, normalized internally to sum to 1 in each
        row. Shape ``(n_bins,)`` for a stationary prior, ``(n_time_bins,
        n_bins)`` for a time-varying prior. Exact zeros exclude bins from the
        posterior; positive weights are not floored. If None, every bin gets
        equal mass (this function has no environment and does not know bin
        sizes; the decoders default to a prior uniform over space).
    handle_degenerate : {"uniform", "nan", "raise"}, default="uniform"
        How to handle degenerate rows, whose log-likelihood is ``-inf`` in
        every bin the prior allows. They arise when the prior is zero exactly
        where the likelihood is positive, or from a likelihood that is ``-inf``
        everywhere:

        - "uniform": spread the row's mass evenly over the bins with positive
          prior (``1 / n_bins`` per bin without a prior). A row whose prior has
          no positive bin becomes NaN.
        - "nan": Return NaN for degenerate rows.
        - "raise": Raise ValueError if any row is degenerate.
    dtype : np.float32 or np.float64, default=np.float64
        Dtype of the returned posterior. ``np.float32`` halves its memory.
    time_chunk : int or None, default=None
        When set, the softmax runs in blocks of ``time_chunk`` rows into one
        preallocated output, so the shifted log-likelihood temporary is only
        one block in size. ``None`` normalizes all rows at once.

    Returns
    -------
    posterior : NDArray, shape (n_time_bins, n_bins)
        Posterior probability mass per bin in ``dtype``. Each row sums to 1,
        except degenerate rows under ``handle_degenerate="nan"`` (or rows whose
        prior has no positive bin), which are NaN.

    Raises
    ------
    ValueError
        If ``log_likelihood`` is not 2-D, or a row contains NaN or ``+inf``.
        If ``prior`` has the wrong shape or a negative or non-finite entry.
        If ``handle_degenerate="raise"`` and a row is degenerate, or
        ``handle_degenerate`` is not one of the allowed values.
        If ``dtype`` is not float32 or float64, or ``time_chunk`` is not a
        positive integer.

    Notes
    -----
    Implementation uses numerically stable log-sum-exp:

    .. code-block:: python

        prior = prior / prior.sum(axis=-1, keepdims=True)
        ll = log_likelihood + np.log(prior)  # log(0) = -inf excludes the bin
        posterior = np.exp(ll - ll.max(axis=-1, keepdims=True))
        posterior /= posterior.sum(axis=-1, keepdims=True)

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
    return _normalize_to_posterior(
        log_likelihood,
        prior=prior,
        handle_degenerate=handle_degenerate,
        dtype=dtype,
        time_chunk=time_chunk,
        fill_with_prior=False,
    )


def _normalize_to_posterior(
    log_likelihood: NDArray[np.float64],
    *,
    prior: NDArray[np.float64] | None,
    handle_degenerate: Literal["uniform", "nan", "raise"] = "uniform",
    dtype: Any = np.float64,
    time_chunk: int | None = None,
    fill_with_prior: bool,
) -> NDArray[np.float64]:
    """Run :func:`normalize_to_posterior`, choosing the degenerate-row fill.

    ``fill_with_prior=True`` fills a degenerate row with the normalized prior
    instead of a uniform mass over its positive bins. The decoders use it: a
    row whose likelihood carries no usable information keeps the prior, which
    by default is uniform over space rather than uniform per bin.
    """
    if handle_degenerate not in ("uniform", "nan", "raise"):
        raise ValueError(
            f"handle_degenerate must be 'uniform', 'nan', or 'raise', "
            f"got {handle_degenerate!r}"
        )
    out_dtype = _validate_posterior_dtype(dtype)
    time_chunk = _validate_time_chunk(time_chunk, allow_none=True)
    ll = np.asarray(log_likelihood, dtype=np.float64)
    if ll.ndim != 2:
        raise ValueError(
            f"log_likelihood must be 2-D with shape (n_time_bins, n_bins), got "
            f"shape {ll.shape}. For a single time bin pass ll[np.newaxis, :]."
        )
    fill: NDArray[np.float64] | NDArray[np.bool_] | None = None
    if prior is not None:
        n_time, n_bins = ll.shape
        prior_arr = np.asarray(prior, dtype=np.float64)
        if prior_arr.shape not in ((n_bins,), (n_time, n_bins)):
            raise ValueError(
                f"prior must have shape ({n_bins},) or ({n_time}, {n_bins}) to "
                f"match log_likelihood, got shape {prior_arr.shape}."
            )
        if not np.isfinite(prior_arr).all() or np.any(prior_arr < 0):
            raise ValueError(
                _format_error(
                    "prior has negative or non-finite entries.",
                    why="a prior is a non-negative weight per bin.",
                    fix="pass finite non-negative weights; use 0 to exclude a bin.",
                )
            )
        prior_sum = prior_arr.sum(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            prior_arr = np.where(prior_sum > 0, prior_arr / prior_sum, prior_arr)
            # Exact zeros stay -inf in log space, so no likelihood can select
            # a bin the prior excludes.
            log_prior = np.log(prior_arr)
        fill = prior_arr if fill_with_prior else prior_arr > 0
        ll = ll + log_prior

    out = np.empty(ll.shape, dtype=out_dtype)
    n_time = ll.shape[0]
    step = n_time if time_chunk is None else time_chunk
    for start in range(0, n_time, max(step, 1)):
        stop = min(start + step, n_time)
        _normalize_block(
            ll[start:stop],
            handle_degenerate=handle_degenerate,
            out=out[start:stop],
            fill=fill[start:stop] if fill is not None and fill.ndim == 2 else fill,
        )
    return cast("NDArray[np.float64]", out)


def decode_position(
    env: Environment,
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64] | SpatialRatesResult,
    *,
    dt: float,
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
        Firing rate maps (place fields) for each neuron, in Hz.

        A :class:`~neurospatial.encoding.spatial.SpatialRatesResult` from
        :func:`~neurospatial.compute_spatial_rates` may be passed directly.
        Its ``spatial_support`` excludes unobserved, below-threshold or
        non-finite training bins and is intersected with any supplied prior.
        Its environment must match the decoding environment's ordered bin
        coordinates, measures, coordinate system, units, frame and weighted
        connectivity; equivalent copies are accepted, also with
        ``validate=False``. Other result objects raise ``TypeError``. When
        passing a raw array, provide a prior that excludes unsupported bins,
        such as ``prior=rates.spatial_support * env.bin_sizes``.

        NaN entries mark missing rate estimates (for example bins below an
        encoder's ``min_occupancy``). A bin where any neuron's rate is NaN
        gets zero posterior probability, and a :class:`UserWarning` reports
        how many bins were excluded. A neuron's missing rate cannot be left
        out of one bin only: every observed Poisson term is at most 1, so the
        bins where fewer neurons are observed would be favoured. Pass the
        rate result itself to exclude unsupported bins without the warning.
        With ``validate=False``, Inf entries are excluded the same way; with
        ``validate=True`` they raise.

        If no bin has a finite rate for every neuron, a :class:`ValueError`
        is raised, even with ``validate=False``.
    dt : float
        Time bin width in seconds, the width used to bin ``spike_counts``.
    prior : NDArray[np.float64] | None, default=None
        Non-negative weight per bin, normalized internally to a probability
        mass. Shape (n_bins,) for stationary prior, (n_time_bins, n_bins) for
        time-varying prior. Exact zeros exclude bins.

        If None, the prior is uniform over space: each bin's mass is
        proportional to its length, area or volume (``env.bin_sizes``), so
        bins of unequal size do not bias the decode toward small bins. On
        equal bins this is the flat prior. A prior that is uniform over space
        on a subset of bins is ``mask * env.bin_sizes``. An environment that
        mixes bins measured by length and by area has no uniform-over-space
        prior and raises ``ValueError``.

        Rates are floored at 1e-10 Hz, so a spike from a neuron whose rate is
        zero in a bin makes that bin very unlikely, not impossible.
    method : {"poisson"}, default="poisson"
        Likelihood model. Only the Poisson model is available.
    times : NDArray[np.float64] | None, default=None
        Time bin centers (seconds). If provided, stored in DecodingResult.
    validate : bool, default=True
        If True (the default), run extra validation checks before and
        after the core computation:

        - Reject NaN/Inf entries in spike_counts.
        - Reject negative spike_counts (must be non-negative integers).
        - Reject fractional spike_counts when given a float dtype
          (cast to integer first or fix the upstream binning bug).
        - Reject Inf entries in encoding_models.
        - Verify each posterior row sums to 1.0 within atol=1e-6.

        Pass ``validate=False`` to skip these checks inside a hot loop with
        known-clean inputs. The prior is always checked (shape, finite
        non-negative weights, positive total mass), as are negative rates,
        the bin count and the no-decodable-bin condition.
    dtype : np.float32 or np.float64, default=np.float64
        Dtype of the stored posterior; ``np.float32`` halves its memory. See
        Notes.
    time_chunk : int or None, default=None
        Number of time bins computed at once; ``None`` computes all of them
        together. See Notes.

    Returns
    -------
    DecodingResult
        Container with posterior, estimates, and metadata.

    Raises
    ------
    TypeError
        If ``encoding_models`` is a result object other than a
        ``SpatialRatesResult``.
    ValueError
        If method is not "poisson".
        If validate=True and validation checks fail.
        If ``encoding_models`` has no bin where every neuron's rate is finite,
        or has a negative rate; checked even with ``validate=False``.
        If ``prior`` has the wrong shape, a negative or non-finite weight, or
        a row with zero total mass; checked even with ``validate=False``.
        If ``encoding_models`` bin count (axis 1) does not match
        ``env.n_bins``, or a rate result's environment differs from ``env``
        (see ``encoding_models``); checked even with ``validate=False``.
        If ``times`` is provided and its length does not match the number
        of posterior time bins.

    Notes
    -----
    **Choosing dt.** Each time bin is decoded independently, so ``dt`` trades
    time resolution against the number of spikes per bin: 20-50 ms suits
    running behaviour with a few tens of place cells, shorter bins need more
    cells for the same accuracy.

    **Memory.** The posterior has shape ``(n_time_bins, n_bins)``: one hour at
    25 ms (144,000 bins) over 1000 spatial bins is about 1.1 GB in float64.
    With ``time_chunk=None`` the full log-likelihood and its temporaries exist
    at once, a few times the stored posterior. ``time_chunk=N`` computes the
    likelihood and posterior ``N`` time bins at a time into the stored
    posterior, so the transient memory is a few ``(N, n_bins)`` blocks. The
    two paths agree to floating-point rounding (about 1e-15 relative), so the
    MAP bin differs only where two bins' densities tie to that precision.
    ``dtype=np.float32`` halves the stored posterior and rounds it to about
    1e-7 relative. When even the stored posterior is too large, use
    :func:`decode_position_summary`, which keeps only per-time reductions.

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.posterior import decode_position
    >>> import numpy as np
    >>>
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>>
    >>> # Generate test data
    >>> n_time_bins, n_neurons = 50, 5
    >>> spike_counts = rng.poisson(2, (n_time_bins, n_neurons))
    >>> encoding_models = rng.uniform(0.1, 10, (n_neurons, env.n_bins))
    >>>
    >>> # Decode
    >>> result = decode_position(env, spike_counts, encoding_models, dt=0.025)
    >>> result.posterior.shape
    (50, ...)
    >>> result.map_bin.shape
    (50,)

    See Also
    --------
    DecodingResult : Container for decoding results
    log_poisson_likelihood : Likelihood function used internally
    normalize_to_posterior : Posterior normalization used internally
    """
    dt = validate_dt(dt)
    time_chunk = _validate_time_chunk(time_chunk, allow_none=True)

    models, prior = _rates_and_prior(
        env, encoding_models, prior, len(spike_counts), context="decode_position"
    )
    spike_counts, encoding_models, prior = _prepare_decode_inputs(
        env,
        spike_counts,
        models,
        prior=prior,
        method=method,
        validate=validate,
        context="decode_position",
    )

    if time_chunk is None:
        log_ll = log_poisson_likelihood(spike_counts, encoding_models, dt=dt)
        posterior = _normalize_to_posterior(
            log_ll, prior=prior, dtype=dtype, time_chunk=None, fill_with_prior=True
        )
        if validate:
            _validate_output(posterior)
    else:
        posterior = _decode_blockwise(
            spike_counts,
            encoding_models,
            dt,
            prior=prior,
            validate=validate,
            dtype=dtype,
            time_chunk=time_chunk,
        )

    # The posterior was allocated above and nothing else references it, so the
    # result takes it without a copy.
    return DecodingResult._from_owned_posterior(
        posterior, env=env, times=times, spike_window=None
    )


def _decode_blockwise(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    dt: float,
    *,
    prior: NDArray[np.float64] | None,
    validate: bool,
    dtype: Any,
    time_chunk: int,
) -> NDArray[np.float64]:
    """Fill a preallocated posterior one time-block at a time.

    The ``time_chunk=N`` path of :func:`decode_position`: computes the Poisson
    log-likelihood for each ``spike_counts[start:stop]`` block and writes the
    normalized posterior directly into the matching slice of one preallocated
    ``(n_time, n_bins)`` output, so the full log-likelihood is never held.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time, n_neurons)
        Validated spike counts.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Validated, finite firing-rate maps.
    dt : float
        Time-bin width in seconds.
    prior : NDArray[np.float64] | None, shape (n_bins,) or (n_time, n_bins)
        Validated prior, or ``None`` for a flat prior.
    validate : bool
        If ``True``, run :func:`_validate_output` on each posterior block.
    dtype : np.float32 or np.float64
        Dtype of the posterior.
    time_chunk : int
        Positive time-block size (rows per block).

    Returns
    -------
    posterior : NDArray, shape (n_time, n_bins)
        Full posterior in ``dtype``; each row sums to 1.0.
    """
    n_time = spike_counts.shape[0]
    posterior = np.empty(
        (n_time, encoding_models.shape[1]), dtype=_validate_posterior_dtype(dtype)
    )
    time_varying = prior is not None and np.ndim(prior) == 2
    for start in range(0, n_time, time_chunk):
        stop = min(start + time_chunk, n_time)
        block_prior = prior[start:stop] if time_varying else prior  # type: ignore[index]
        post_block = _normalize_to_posterior(
            log_poisson_likelihood(spike_counts[start:stop], encoding_models, dt=dt),
            prior=block_prior,
            dtype=dtype,
            fill_with_prior=True,
        )
        if validate:
            _validate_output(post_block)
        posterior[start:stop] = post_block
    return cast("NDArray[np.float64]", posterior)


def _prepare_decode_inputs(
    env: Environment,
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    *,
    prior: NDArray[np.float64] | None,
    method: Literal["poisson"],
    validate: bool,
    context: str,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Validate inputs and exclude bins without a finite rate for every neuron.

    Shared front half of :func:`decode_position`,
    :func:`decode_position_summary` and the session decoders: runs the
    unconditional guards (bin-count agreement, at
    least one decodable bin) and the optional ``validate=True`` checks.

    A bin where any neuron's rate is non-finite (NaN, or Inf with
    ``validate=False``) is excluded: its rates are replaced by a finite
    placeholder and its prior mass is set to zero, so it gets zero posterior
    probability. Leaving only the masked neurons out of such a bin would
    favour it, because each observed Poisson term is at most 1.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing-rate maps.
    prior : NDArray[np.float64] | None
        Prior resolved by :func:`~neurospatial.decoding._support.resolve_prior`.
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
        Finite firing-rate maps.
    prior : NDArray[np.float64] | None
        The prior, with zero mass in excluded bins.
    """
    from neurospatial.decoding._support import exclude_bins
    from neurospatial.encoding import _validate_env_argument

    _validate_env_argument(
        env, context=context, arguments="spike_counts, encoding_models, dt=dt"
    )
    if method != "poisson":
        raise ValueError(
            f"Unknown method '{method}'. Currently only 'poisson' is supported."
        )

    spike_counts = np.asarray(spike_counts)
    encoding_models = np.asarray(encoding_models)
    if encoding_models.ndim != 2:
        raise ValueError(
            f"encoding_models must be 2-D (n_neurons, n_bins), got shape "
            f"{encoding_models.shape}."
        )
    # A wrong bin count yields a well-formed posterior over the wrong
    # positions, so this check does not depend on ``validate``.
    if encoding_models.shape[1] != env.n_bins:
        raise IncompatibleEnvironmentError(
            f"{context}: encoding_models has {encoding_models.shape[1]} bins "
            f"(axis 1) but env has {env.n_bins} active bins.",
            fix="compute the rate maps on the decoding environment.",
        )
    if validate:
        _validate_inputs(spike_counts, encoding_models)

    if encoding_models.dtype.kind == "f" and not np.isfinite(encoding_models).all():
        excluded = np.asarray(~np.isfinite(encoding_models).all(axis=0))
        n_bins = encoding_models.shape[1]
        if excluded.all():
            raise ValueError(
                _format_error(
                    f"{context}: encoding_models has no bin where every neuron "
                    "has a finite rate; every bin has a NaN or Inf rate for at "
                    "least one neuron, so nothing can be decoded.",
                    why="a bin is decoded only when every neuron has a rate "
                    "estimate there.",
                    fix="recompute the rate maps from valid training data "
                    "(check coverage, epochs and min_occupancy), or decode the "
                    "neurons whose maps cover the arena. Filling missing rates "
                    "with zero cannot supply the absent training evidence.",
                )
            )
        warnings.warn(
            _format_error(
                f"{context}: encoding_models has a NaN or Inf rate for at least "
                f"one neuron in {int(excluded.sum())} of {n_bins} bins; these "
                "bins get zero posterior probability.",
                why="a bin can be compared with the others only when every "
                "neuron has a rate there; leaving out only the missing neurons "
                "would favour bins where fewer neurons are observed.",
                fix="decode the rate result itself, decode_position(env, counts, "
                "rates, dt=dt) or BayesianDecoder.from_rates(rates), whose "
                "spatial_support excludes these bins without this warning; for "
                "hand-built maps pass spatial_support= to "
                "BayesianDecoder.from_models.",
            ),
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
        encoding_models = np.where(excluded, 0.0, encoding_models).astype(
            encoding_models.dtype, copy=False
        )
        prior = exclude_bins(env, prior, excluded, context=context)

    return spike_counts, encoding_models, prior


def _reduce_posterior_block(
    posterior_block: NDArray[np.float64],
    bin_centers: NDArray[np.float64],
    measures: NDArray[np.float64] | None = None,
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
    measures : NDArray[np.float64] or None, shape (n_bins,)
        Bin measures relative to the largest bin, or ``None`` for equal bins.

    Returns
    -------
    map_bin : NDArray[np.int64], shape (n_block,)
        MAP bin index per time bin: the bin of highest posterior density,
        ``argmax(posterior / measures)``.
    mean_position : NDArray[np.float64], shape (n_block, n_dims)
        Posterior-mean position per time bin (``posterior @ bin_centers``).
    entropy : NDArray[np.float64], shape (n_block,)
        Posterior entropy in bits.
    peak_probability : NDArray[np.float64], shape (n_block,)
        Max posterior probability per time bin.
    """
    # Reductions accumulate in float64 whatever the posterior dtype.
    pb = np.asarray(posterior_block, dtype=np.float64)

    from neurospatial.decoding.estimates import _density_mode, posterior_entropy

    map_bin = _density_mode(pb, measures, rounding_dtype=posterior_block.dtype)
    mean_position = pb @ bin_centers
    peak_probability = pb.max(axis=1)

    entropy = posterior_entropy(pb)

    return (
        map_bin,
        np.asarray(mean_position, dtype=np.float64),
        np.asarray(entropy, dtype=np.float64),
        np.asarray(peak_probability, dtype=np.float64),
    )


def _decode_and_reduce_block(
    counts_block: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    dt: float,
    bin_centers: NDArray[np.float64],
    *,
    prior_block: NDArray[np.float64] | None,
    validate: bool,
    dtype: Any,
    mean_basis: NDArray[np.float64],
    measures: NDArray[np.float64] | None,
) -> tuple[
    NDArray[np.int64],
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
    validate : bool
        If ``True``, run the per-block posterior-sum output check.
    dtype : np.float32 or np.float64
        Working dtype of the posterior block.
    mean_basis : NDArray[np.float64], shape (n_bins, n_dims)
        Coordinates averaged for the posterior mean, from
        :func:`~neurospatial.decoding.estimates._posterior_mean_basis`: bin
        centers, or egocentric-plane coordinates for polar environments.
        Pass the block means through ``_finish_posterior_mean``.
    measures : NDArray[np.float64] or None, shape (n_bins,)
        Bin measures relative to the largest bin, used for the MAP bin;
        ``None`` for equal bins.

    Returns
    -------
    map_bin : NDArray[np.int64], shape (n_block,)
        MAP bin index per time bin.
    mean_position : NDArray[np.float64], shape (n_block, n_dims)
        Posterior-weighted mean of ``mean_basis`` per time bin.
    entropy : NDArray[np.float64], shape (n_block,)
        Posterior entropy (bits) per time bin.
    peak_probability : NDArray[np.float64], shape (n_block,)
        Max posterior probability per time bin.
    """
    post_block = _normalize_to_posterior(
        log_poisson_likelihood(counts_block, encoding_models, dt=dt),
        prior=prior_block,
        dtype=dtype,
        fill_with_prior=True,
    )
    if validate:
        _validate_output(post_block)
    block_map_bin, block_mean, block_entropy, block_peak = _reduce_posterior_block(
        post_block, mean_basis, measures
    )
    # post_block goes out of scope and is reclaimed before the next block.
    return block_map_bin, block_mean, block_entropy, block_peak


def decode_position_summary(
    env: Environment,
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64] | SpatialRatesResult,
    *,
    dt: float,
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

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64] or SpatialRatesResult, shape (n_neurons, n_bins)
        Firing-rate maps (place fields) for each neuron, or a
        ``SpatialRatesResult`` whose ``spatial_support`` excludes unsupported
        training bins. Same semantics (missing rates, environment
        compatibility) as :func:`decode_position`.
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
        / ``posterior_entropy`` / ``peak_probability`` (all length
        ``n_time_bins``), plus ``times`` and ``env``.

    Raises
    ------
    ValueError
        If ``time_chunk`` is ``None`` or not a positive integer. Plus the same
        conditions as :func:`decode_position` (invalid method, no finite bins,
        bin-count mismatch, bad ``times`` length, validation failures).

    Notes
    -----
    The reductions equal those of the full :class:`DecodingResult` posterior to
    floating-point rounding; ``map_bin`` differs only where two bins'
    densities tie to that precision. Peak memory is one ``(time_chunk,
    n_bins)`` block plus the reductions; see :func:`decode_position` Notes.

    See Also
    --------
    decode_position : Decode and keep the full posterior.
    DecodingSummary : Streamed per-time reductions container.
    decode_session_summary : One-call encode->bin->summary-decode wrapper.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import decode_position_summary
    >>> positions = np.c_[np.arange(10.0), np.zeros(10)]
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> counts = np.ones((4, 2), dtype=np.int64)
    >>> models = np.full((2, env.n_bins), 5.0)
    >>> summary = decode_position_summary(env, counts, models, dt=0.1, time_chunk=2)
    >>> summary.map_position.shape
    (4, 2)
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

    models, prior = _rates_and_prior(
        env,
        encoding_models,
        prior,
        len(spike_counts),
        context="decode_position_summary",
    )
    spike_counts, encoding_models, prior = _prepare_decode_inputs(
        env,
        spike_counts,
        models,
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

    from neurospatial.decoding._support import relative_bin_measures
    from neurospatial.decoding.estimates import (
        _finish_posterior_mean,
        _posterior_mean_basis,
    )

    bin_centers = np.asarray(env.bin_centers, dtype=np.float64)
    mean_basis = _posterior_mean_basis(env)
    measures = relative_bin_measures(env, context="decode_position_summary")
    n_dims = bin_centers.shape[1]

    map_bin = np.empty(n_time, dtype=np.int64)
    mean_position = np.empty((n_time, n_dims), dtype=np.float64)
    posterior_entropy = np.empty(n_time, dtype=np.float64)
    peak_probability = np.empty(n_time, dtype=np.float64)

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
            block_mean,
            block_entropy,
            block_peak,
        ) = _decode_and_reduce_block(
            spike_counts[start:stop],
            encoding_models,
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
        peak_probability[start:stop] = block_peak

    return DecodingSummary._from_owned(
        times=times_arr,
        mean_position=_finish_posterior_mean(env, mean_position),
        posterior_entropy=posterior_entropy,
        peak_probability=peak_probability,
        map_bin=map_bin,
        env=env,
        spike_window=None,
    )


def _validate_inputs(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
) -> None:
    """Run the optional ``validate=True`` checks of ``decode_position``.

    NaN rates are not rejected here: they mark missing rate estimates and the
    affected bins are excluded from the posterior.

    Raises
    ------
    ValueError
        If spike counts are non-finite, negative or fractional, or if an
        encoding-model rate is infinite or negative.
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

    n_inf = int(np.isinf(encoding_models).sum())
    if n_inf:
        raise ValueError(
            _format_error(
                f"encoding_models contains {n_inf} infinite rate(s).",
                why="an infinite firing rate is not a rate estimate.",
                fix="fix the rate computation; mark missing rates with NaN so "
                "their bins are excluded from the posterior.",
            )
        )
    if (encoding_models < 0).any():
        n_negative = int(np.sum(encoding_models < 0))
        worst_rate = float(np.nanmin(encoding_models))
        raise ValueError(
            f"encoding_models contains {n_negative} negative entr"
            f"{'y' if n_negative == 1 else 'ies'} (min: {worst_rate:.6g} Hz). "
            "Firing rates must be non-negative."
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
