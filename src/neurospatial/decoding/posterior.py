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
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import IncompatibleEnvironmentError, _format_error
from neurospatial._stacklevel import _external_stacklevel
from neurospatial._validation import validate_float_dtype
from neurospatial.decoding._result import DecodingResult, DecodingSummary
from neurospatial.decoding._time_bins import validate_dt
from neurospatial.decoding.likelihood import _PoissonRates

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


def _validate_time_chunk(time_chunk: Any, *, allow_none: bool) -> Any:
    """Return ``time_chunk`` as a positive ``int``, or ``None`` when allowed.

    ``bool`` is rejected even though it subclasses ``int``, so ``True`` is not
    read as a chunk size of 1.
    """
    from neurospatial._validation import is_integer_at_least

    if time_chunk is None and allow_none:
        return None
    if not is_integer_at_least(time_chunk, 1):
        raise ValueError(
            f"time_chunk must be a positive integer{' or None' if allow_none else ''}, "
            f"got {time_chunk!r} (type {type(time_chunk).__name__})."
        )
    return int(time_chunk)


def _summary_time_chunk(time_chunk: Any, *, caller: str, full_decoder: str) -> int:
    """Validate the block size of a streamed summary decoder; ``None`` is refused."""
    if time_chunk is None:
        raise ValueError(
            f"time_chunk=None is not allowed for {caller}: this streamed summary "
            "decoder reduces the posterior one time-block at a time, and None "
            "would materialize the full (n_time, n_bins) posterior, defeating its "
            f"purpose. Use {full_decoder} if you want the full posterior, or pass "
            "a positive time_chunk (default 1024) here."
        )
    return cast("int", _validate_time_chunk(time_chunk, allow_none=False))


def _normalized_prior(
    prior: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``prior`` normalized per row and its log; exact zeros give ``-inf``.

    A row with no positive weight is left unnormalized.
    """
    prior_sum = prior.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(prior_sum > 0, prior / prior_sum, prior)
        # Exact zeros stay -inf in log space, so no likelihood can select a
        # bin the prior excludes.
        return normalized, np.log(normalized)


def _normalize_block(
    ll_block: NDArray[np.float64],
    *,
    handle_degenerate: Literal["uniform", "nan", "raise"],
    out: NDArray[np.floating[Any]],
    fill: NDArray[np.float64] | NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Normalize one time-block of log-likelihood into ``out``.

    Performs the log-sum-exp softmax over the last axis and the degenerate-row
    handling for a single block, writing the result into the preallocated
    ``out`` array. ``ll_block`` is the prior-applied log-likelihood for the
    block and is not modified; ``out`` must be the same shape and the desired
    output dtype.

    A row is degenerate when every entry is ``-inf``: no bin the prior allows
    has positive likelihood. With ``"uniform"``, ``fill`` (shape ``(n_bins,)``
    or matching ``ll_block``) holds non-negative weights normalized into each
    degenerate row: the positive-prior mask for :func:`normalize_to_posterior`,
    the prior itself for the decoders. A row whose weights are all zero becomes
    NaN; ``None`` fills ``1 / n_bins``. Rows containing NaN or ``+inf`` raise.

    Returns
    -------
    NDArray[np.float64], shape (n_block,)
        Log of each row's normalizer, ``log(sum(exp(ll_block)))``; ``-inf``
        for a degenerate row.
    """
    ll_max = ll_block.max(axis=-1, keepdims=True)
    # The row maximum is NaN or +inf exactly when the row contains one.
    bad_rows = np.isnan(ll_max[..., 0]) | np.isposinf(ll_max[..., 0])
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
    degenerate_mask = np.isneginf(ll_max[..., 0])

    # Shift into ``out`` when it can hold the float64 shift exactly, so no
    # block-sized temporary is needed. Degenerate rows give ``-inf - -inf =
    # nan`` here; they are overwritten below, so the "invalid value" warning
    # is suppressed.
    shifted = out if out.dtype == ll_block.dtype else np.empty(ll_block.shape)
    with np.errstate(invalid="ignore"):
        np.subtract(ll_block, ll_max, out=shifted)
    np.exp(shifted, out=out)
    row_sums = out.sum(axis=-1, keepdims=True)
    out /= row_sums
    log_norm: NDArray[np.float64] = ll_max[..., 0] + np.log(row_sums[..., 0])
    log_norm[degenerate_mask] = -np.inf

    if not degenerate_mask.any():
        return log_norm
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
    return log_norm


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
    from neurospatial._validation import validate_choice

    validate_choice(
        handle_degenerate,
        ("uniform", "nan", "raise"),
        name="handle_degenerate",
        owner="normalize_to_posterior()",
    )
    out_dtype = validate_float_dtype(dtype, stored="posteriors")
    time_chunk = _validate_time_chunk(time_chunk, allow_none=True)
    ll = np.asarray(log_likelihood, dtype=np.float64)
    if ll.ndim != 2:
        raise ValueError(
            f"log_likelihood must be 2-D with shape (n_time_bins, n_bins), got "
            f"shape {ll.shape}. For a single time bin pass ll[np.newaxis, :]."
        )
    n_time, n_bins = ll.shape
    fill: NDArray[np.bool_] | None = None
    if prior is not None:
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
        normalized, log_prior = _normalized_prior(prior_arr)
        fill = normalized > 0
        ll = ll + log_prior

    out = np.empty(ll.shape, dtype=out_dtype)
    step = max(n_time if time_chunk is None else time_chunk, 1)
    for start in range(0, n_time, step):
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

        Pass ``validate=False`` to skip these checks inside a hot loop with
        known-clean inputs. The prior is always checked (shape, finite
        non-negative weights, positive total mass), as are negative rates,
        the bin count and the no-decodable-bin condition. The returned
        :class:`DecodingResult` always checks that each posterior row sums to
        1 within 1e-6.
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
        of posterior time bins, or it is not finite and strictly increasing.

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
    dtype = validate_float_dtype(dtype, stored="posteriors")
    time_chunk = _validate_time_chunk(time_chunk, allow_none=True)
    spike_counts = np.asarray(spike_counts)
    rates, prior = _prepare_decode_inputs(
        env,
        encoding_models,
        prior,
        len(spike_counts),
        dt=dt,
        method=method,
        validate=validate,
        context="decode_position",
    )
    if validate:
        _validate_spike_counts(spike_counts)
    return _decode_posterior(
        env,
        lambda start, stop: spike_counts[start:stop],
        len(spike_counts),
        rates,
        prior,
        dtype=dtype,
        time_chunk=time_chunk,
        times=times,
        spike_window=None,
    )


def _prepare_decode_inputs(
    env: Environment,
    encoding_models: NDArray[np.float64] | SpatialRatesResult,
    prior: Any,
    n_time: int,
    *,
    dt: float,
    validate: bool,
    context: str,
    spatial_support: NDArray[np.bool_] | None = None,
    method: str = "poisson",
) -> tuple[_PoissonRates, NDArray[np.float64] | None]:
    """Validate the rate maps and prior once and build the likelihood terms.

    Shared front half of every decoder. A
    :class:`~neurospatial.encoding.spatial.SpatialRatesResult` contributes
    its ``spatial_support``; a bare array may come with ``spatial_support``.
    Unsupported bins get a finite placeholder rate and zero prior mass. Other
    objects with ``firing_rates`` are refused, because only spatial rate
    results carry the support and environment the decode needs.

    A bin where any neuron's rate is non-finite (NaN, or Inf with
    ``validate=False``) is excluded the same way, with a warning. Leaving only
    the masked neurons out of such a bin would favour it, because each
    observed Poisson term is at most 1.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    encoding_models : NDArray[np.float64] or SpatialRatesResult
        Firing-rate maps, shape ``(n_neurons, n_bins)``.
    prior : array-like or None
        User prior, shape ``(n_bins,)`` or ``(n_time, n_bins)``.
    n_time : int
        Number of decoded time bins.
    dt : float
        Validated time-bin width in seconds.
    validate : bool
        Whether to refuse infinite rates instead of excluding their bins.
    context : str
        Caller name used in error messages.
    spatial_support : NDArray[np.bool_] or None
        Training support of a bare rate array.
    method : str
        Likelihood model. Only ``"poisson"`` is supported.

    Returns
    -------
    rates : _PoissonRates
        Likelihood terms of finite, non-negative rate maps.
    prior : NDArray[np.float64] or None
        Prior mass per bin, zero in excluded bins; ``None`` is flat.
    """
    from neurospatial.decoding._support import (
        exclude_bins,
        models_on_support,
        resolve_prior,
    )
    from neurospatial.encoding.spatial import SpatialRatesResult

    _validate_encoding_environment(env, encoding_models, context=context)
    if method != "poisson":
        raise ValueError(
            f"Unknown method '{method}'. Currently only 'poisson' is supported."
        )
    rate_maps: Any = encoding_models
    if isinstance(encoding_models, SpatialRatesResult):
        spatial_support = encoding_models.spatial_support
        rate_maps = encoding_models.firing_rates
    elif hasattr(encoding_models, "firing_rates"):
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
    prior = resolve_prior(env, prior, spatial_support, n_time, context=context)
    models = np.asarray(models_on_support(np.asarray(rate_maps), spatial_support))
    if models.ndim != 2:
        raise ValueError(
            f"encoding_models must be 2-D (n_neurons, n_bins), got shape "
            f"{models.shape}."
        )
    # A wrong bin count yields a well-formed posterior over the wrong
    # positions, so this check does not depend on ``validate``.
    if models.shape[1] != env.n_bins:
        raise IncompatibleEnvironmentError(
            f"{context}: encoding_models has {models.shape[1]} bins "
            f"(axis 1) but env has {env.n_bins} active bins.",
            fix="compute the rate maps on the decoding environment.",
        )
    if validate:
        n_inf = int(np.isinf(models).sum())
        if n_inf:
            raise ValueError(
                _format_error(
                    f"encoding_models contains {n_inf} infinite rate(s).",
                    why="an infinite firing rate is not a rate estimate.",
                    fix="fix the rate computation; mark missing rates with NaN "
                    "so their bins are excluded from the posterior.",
                )
            )
    n_negative = int(np.sum(models < 0))
    if n_negative:
        raise ValueError(
            f"encoding_models contains {n_negative} negative entr"
            f"{'y' if n_negative == 1 else 'ies'} "
            f"(min: {float(np.nanmin(models)):.6g} Hz). "
            "Firing rates must be non-negative."
        )

    if models.dtype.kind == "f" and not np.isfinite(models).all():
        excluded = np.asarray(~np.isfinite(models).all(axis=0))
        n_bins = models.shape[1]
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
        models = np.where(excluded, 0.0, models).astype(models.dtype, copy=False)
        prior = exclude_bins(env, prior, excluded, context=context)

    return _PoissonRates.from_rates(models, dt), prior


def _validate_spike_counts(spike_counts: NDArray[Any]) -> None:
    """Refuse non-finite, negative or fractional spike counts.

    Raises
    ------
    ValueError
        If a spike count is NaN or Inf, negative, or not integer-valued.
    """
    # Finite first (NaN/Inf can't pass the < 0 check cleanly), then
    # non-negative. The messages give the number of bad entries and the worst
    # value, which a 100k-row array needs to be actionable.
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

    # Float counts must hold exact integers (a common shape after histogramming
    # into a float64 column). A fractional value like 0.5 is meaningless under
    # the Poisson likelihood and would produce a "valid"-looking posterior.
    if not np.issubdtype(spike_counts.dtype, np.integer):
        rounded = np.floor(spike_counts)
        is_integer_valued = np.equal(spike_counts, rounded)
        if not is_integer_valued.all():
            n_fractional = int(np.sum(~is_integer_valued))
            # The entry with the largest fractional part, to find in the input.
            fractional_part = np.abs(spike_counts - rounded)
            worst_value = float(spike_counts.flat[int(fractional_part.argmax())])
            raise ValueError(
                f"spike_counts contains {n_fractional} fractional entr"
                f"{'y' if n_fractional == 1 else 'ies'} (e.g., {worst_value:g}). "
                "Spike counts must be integer-valued. Cast with "
                "`spike_counts.astype(np.int64)` after confirming the "
                "non-integer values are not a binning bug upstream."
            )


def _posterior_blocks(
    count_block: Callable[[int, int], NDArray[np.int64]],
    n_time: int,
    rates: _PoissonRates,
    prior: NDArray[np.float64] | None,
    *,
    time_chunk: int,
    dtype: Any,
    out: NDArray[np.float64] | None = None,
) -> Iterator[tuple[int, int, NDArray[np.float64], NDArray[np.float64]]]:
    """Decode the posterior one block of ``time_chunk`` time bins at a time.

    The single decode loop behind every decoder: the full-posterior decoders
    write each block into ``out``, the summary decoders reduce and discard it.

    Parameters
    ----------
    count_block : callable
        ``count_block(start, stop)`` returns the ``(stop - start, n_neurons)``
        spike counts of time bins ``start:stop``.
    n_time : int
        Number of time bins.
    rates : _PoissonRates
        Likelihood terms of the rate maps.
    prior : NDArray[np.float64] or None, shape (n_bins,) or (n_time, n_bins)
        Resolved prior; ``None`` is flat.
    time_chunk : int
        Time bins per block.
    dtype : np.float32 or np.float64
        Dtype of each posterior block.
    out : NDArray or None, shape (n_time, n_bins)
        Preallocated posterior to write the blocks into.

    Yields
    ------
    start, stop : int
        Time bins of the block.
    posterior : NDArray, shape (stop - start, n_bins)
        Posterior block, a view of ``out`` when given. A degenerate row keeps
        the normalized prior.
    log_norm : NDArray[np.float64], shape (stop - start,)
        Log of each row's normalizer: the log evidence under the normalized
        prior, up to the per-row constant the likelihood omits.
    """
    if prior is None:
        log_prior = fill = None
    else:
        fill, log_prior = _normalized_prior(prior)
    time_varying = log_prior is not None and log_prior.ndim == 2
    # At least one (possibly empty) block, so the count shape is always checked.
    for start in range(0, max(n_time, 1), time_chunk):
        stop = min(start + time_chunk, n_time)
        ll = rates.log_likelihood(count_block(start, stop))
        if log_prior is not None:
            ll += log_prior[start:stop] if time_varying else log_prior
        block = np.empty(ll.shape, dtype=dtype) if out is None else out[start:stop]
        log_norm = _normalize_block(
            ll,
            handle_degenerate="uniform",
            out=block,
            fill=fill[start:stop] if time_varying and fill is not None else fill,
        )
        del ll  # free the log-likelihood before the caller reduces the block
        yield start, stop, block, log_norm


def _decode_posterior(
    env: Environment,
    count_block: Callable[[int, int], NDArray[np.int64]],
    n_time: int,
    rates: _PoissonRates,
    prior: NDArray[np.float64] | None,
    *,
    dtype: Any,
    time_chunk: int | None,
    times: NDArray[np.float64] | None,
    spike_window: Any,
) -> DecodingResult:
    """Decode the full posterior into one preallocated array; see ``_posterior_blocks``."""
    posterior = np.empty((n_time, env.n_bins), dtype=dtype)
    for _ in _posterior_blocks(
        count_block,
        n_time,
        rates,
        prior,
        time_chunk=max(n_time, 1) if time_chunk is None else time_chunk,
        dtype=dtype,
        out=posterior,
    ):
        pass
    # Nothing else references the posterior, so the result takes it without a
    # copy; its constructor checks every row.
    return DecodingResult._from_owned_posterior(
        posterior, env=env, times=times, spike_window=spike_window
    )


def _decode_summary(
    env: Environment,
    count_block: Callable[[int, int], NDArray[np.int64]],
    n_time: int,
    rates: _PoissonRates,
    prior: NDArray[np.float64] | None,
    *,
    dtype: Any,
    time_chunk: int,
    validate: bool,
    times: NDArray[np.float64] | None,
    spike_window: Any,
    context: str,
) -> DecodingSummary:
    """Reduce each posterior block to per-time estimates and discard it.

    Never holds more than one ``(time_chunk, n_bins)`` posterior block. The
    reductions match the :class:`DecodingResult` properties of the full
    posterior; with ``validate`` every block is checked like a result's
    posterior.
    """
    from neurospatial.decoding._result import _posterior_value_problems
    from neurospatial.decoding._support import relative_bin_measures
    from neurospatial.decoding.estimates import (
        _density_mode,
        _finish_posterior_mean,
        _posterior_mean_basis,
        posterior_entropy,
    )

    mean_basis = _posterior_mean_basis(env)
    measures = relative_bin_measures(env, context=context)
    map_bin = np.empty(n_time, dtype=np.int64)
    weighted = np.empty((n_time, mean_basis.shape[1]), dtype=np.float64)
    entropy = np.empty(n_time, dtype=np.float64)
    peak = np.empty(n_time, dtype=np.float64)
    for start, stop, block, _ in _posterior_blocks(
        count_block, n_time, rates, prior, time_chunk=time_chunk, dtype=dtype
    ):
        if validate:
            problems = _posterior_value_problems(block)
            if problems:
                raise ValueError(
                    f"{context}: the decoded posterior of time bins {start} to "
                    f"{stop} is invalid: {'; '.join(problems)}. This indicates "
                    "numerical instability."
                )
        # The reductions accumulate in float64 whatever the block dtype.
        map_bin[start:stop] = _density_mode(block, measures)
        weighted[start:stop] = block @ mean_basis
        entropy[start:stop] = posterior_entropy(block)
        peak[start:stop] = block.max(axis=1)
    return DecodingSummary._from_owned(
        times=times,
        mean_position=_finish_posterior_mean(env, weighted),
        posterior_entropy=entropy,
        peak_probability=peak,
        map_bin=map_bin,
        env=env,
        spike_window=spike_window,
    )


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
        Run the same input validation as :func:`decode_position`, and check
        each posterior block as a :class:`DecodingResult` checks its
        posterior.
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
        bin-count mismatch, bad ``times``, validation failures).

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
    time_chunk = _summary_time_chunk(
        time_chunk, caller="decode_position_summary", full_decoder="decode_position"
    )
    dt = validate_dt(dt)
    dtype = validate_float_dtype(dtype, stored="posteriors")
    spike_counts = np.asarray(spike_counts)
    n_time = len(spike_counts)
    rates, prior = _prepare_decode_inputs(
        env,
        encoding_models,
        prior,
        n_time,
        dt=dt,
        method=method,
        validate=validate,
        context="decode_position_summary",
    )
    if validate:
        _validate_spike_counts(spike_counts)

    # Check times before decoding (cheap, avoids a wasted full decode).
    times_arr: NDArray[np.float64] | None = None
    if times is not None:
        times_arr = np.asarray(times, dtype=np.float64)
        if times_arr.ndim != 1 or len(times_arr) != n_time:
            raise ValueError(
                f"Length mismatch: times has {times_arr.shape} but there are "
                f"{n_time} time bins. `times` must be a 1-D array of bin "
                f"centers, one per row of spike_counts."
            )

    return _decode_summary(
        env,
        lambda start, stop: spike_counts[start:stop],
        n_time,
        rates,
        prior,
        dtype=dtype,
        time_chunk=time_chunk,
        validate=validate,
        times=times_arr,
        spike_window=None,
        context="decode_position_summary",
    )
