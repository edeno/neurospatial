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
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial.decoding._result import DecodingResult
from neurospatial.decoding.likelihood import log_poisson_likelihood

if TYPE_CHECKING:
    from neurospatial.environment import Environment


def normalize_to_posterior(
    log_likelihood: NDArray[np.float64],
    *,
    prior: NDArray[np.float64] | None = None,
    axis: int = -1,
    handle_degenerate: Literal["uniform", "nan", "raise"] = "uniform",
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

    Returns
    -------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution. Each row sums to 1.0.

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

    # Log-sum-exp normalization (numerically stable softmax)
    ll_max = ll.max(axis=axis, keepdims=True)

    # Detect degenerate rows. There are two distinct causes, and we keep
    # them separate because they mean different things:
    #
    # - all -inf: a legitimate zero-rate row (e.g. no spikes with a flat
    #   encoding model). ll.max() is -inf, which is not finite.
    # - any NaN: upstream corruption (e.g. a NaN firing rate leaking into
    #   the likelihood). ll.max() propagates NaN, which is also not finite.
    #
    # Both fail np.isfinite on the per-row max, so the union is the set of
    # degenerate rows; the NaN subset is tracked so handle_degenerate="raise"
    # can report corruption explicitly rather than blaming a flat row.
    ll_max_squeezed = ll_max.squeeze(axis=axis)
    degenerate_mask = ~np.isfinite(ll_max_squeezed)
    nan_row_mask = np.isnan(ll).any(axis=axis)

    # Shift for stability
    ll_shifted = ll - ll_max

    # Exponentiate
    posterior = np.exp(ll_shifted)

    # Normalize
    posterior_sum = posterior.sum(axis=axis, keepdims=True)
    posterior = posterior / posterior_sum

    # Handle degenerate rows
    if degenerate_mask.any():
        if handle_degenerate == "raise":
            n_degenerate = int(degenerate_mask.sum())
            n_nan = int(nan_row_mask.sum())
            # Distinguish corruption (NaN) from a legitimate zero-rate row
            # (all -inf): the former points at an upstream data bug, the
            # latter at a likelihood/encoding-model mismatch. Conflating
            # them in the message would send the user looking in the wrong
            # place.
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
            # Replace degenerate rows with uniform distribution. This fills
            # both all -inf rows and NaN-corrupted rows uniformly; if you
            # need NaN corruption to surface rather than be masked, use
            # handle_degenerate="raise".
            n_bins = ll.shape[axis]
            uniform_prob = 1.0 / n_bins
            posterior[degenerate_mask] = uniform_prob
        elif handle_degenerate == "nan":
            # Keep NaN values (already set by exp of NaN)
            posterior[degenerate_mask] = np.nan

    return cast("NDArray[np.float64]", posterior)


def decode_position(
    env: Environment,
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    dt: float,
    *,
    prior: NDArray[np.float64] | None = None,
    method: Literal["poisson"] = "poisson",
    times: NDArray[np.float64] | None = None,
    validate: bool = True,
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
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) for each neuron.
        Expected units: Hz (spikes/second). Typical values: 0-50 Hz.
        Very high rates (>100 Hz) may cause numerical issues.

        NaN entries (e.g. low-occupancy bins masked by an encoder's
        ``min_occupancy`` when ``fill_value=None``) are tolerated: each such
        ``(neuron, bin)`` is treated as a zero-rate observation and excluded
        from that neuron's contribution to the Poisson log-likelihood at that
        bin, and a single :class:`UserWarning` is emitted per call. This is
        defense-in-depth so a ``fill_value=None`` encoding model still decodes;
        the recommended path is still to pass ``fill_value=0.0`` to the
        encoder so no NaN reaches the decoder.
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

    Returns
    -------
    DecodingResult
        Container with posterior, estimates, and metadata.

    Raises
    ------
    ValueError
        If method is not "poisson".
        If validate=True and validation checks fail.
        If ``encoding_models`` bin count (axis 1) does not match
        ``env.n_bins``. This is checked unconditionally, even with
        ``validate=False``, because a wrong bin count yields a
        well-formed posterior over the wrong positions.
        If ``times`` is provided and its length does not match the number
        of posterior time bins.

    Notes
    -----
    Memory usage: The posterior array is shape (n_time_bins, n_bins) and
    stored as float64. For long recordings (e.g., 1 hour at 25ms bins =
    144,000 time bins) with fine spatial resolution (e.g., 1000 bins),
    this requires ~1.1 GB. Consider processing in chunks for very long
    recordings, or using float32 dtype in future versions.

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
    from neurospatial.encoding._validation import validate_env_fitted

    validate_env_fitted(env, context="decode_position")

    # Validate method
    if method != "poisson":
        raise ValueError(
            f"Unknown method '{method}'. Currently only 'poisson' is supported."
        )

    # Convert inputs to arrays
    spike_counts = np.asarray(spike_counts)
    encoding_models = np.asarray(encoding_models)

    # Defense-in-depth for NaN encoding-model bins (e.g. an encoder's
    # min_occupancy mask with fill_value=None). Detect them HERE, before the
    # validate=True NaN guard runs, so the two do not both fire on the same
    # bins: we treat each NaN (neuron, bin) as a zero-rate observation and
    # exclude it from that neuron's Poisson contribution at that bin (handled
    # inside _log_poisson_likelihood_nan_safe), emitting one warning per call.
    # The validate=True guard then runs on the *other* inputs (spike_counts,
    # prior) and on Inf-in-encoding-models, but no longer sees NaN encoding
    # bins to reject.
    encoding_model_nan_mask: NDArray[np.bool_] | None = None
    if encoding_models.dtype.kind == "f" and np.isnan(encoding_models).any():
        encoding_model_nan_mask = np.isnan(encoding_models)
        n_nan = int(encoding_model_nan_mask.sum())
        warnings.warn(
            f"encoding_models contains {n_nan} NaN bin(s) (e.g. low-occupancy "
            "bins masked by the encoder's min_occupancy with fill_value=None). "
            "Treating each as a zero-rate observation (excluded from that "
            "neuron's Poisson contribution at that bin). Pass fill_value=0.0 "
            "to the encoder to silence this and make the model explicitly "
            "zero-rate there.",
            UserWarning,
            stacklevel=2,
        )

    # Validate inputs if requested. Pass a NaN-free view of encoding_models so
    # the validate=True NaN guard does not reject the bins the zero-rate path
    # has already claimed; Inf and negative-rate checks still apply.
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

    # Compute log-likelihood using Poisson model. When some encoding-model
    # bins are NaN, route through the NaN-safe path that excludes each NaN
    # (neuron, bin) from the per-bin Poisson sum.
    if encoding_model_nan_mask is not None:
        log_ll = _log_poisson_likelihood_nan_safe(
            spike_counts, encoding_models, encoding_model_nan_mask, dt=dt
        )
    else:
        log_ll = log_poisson_likelihood(spike_counts, encoding_models, dt=dt)

    # Normalize to posterior
    posterior = normalize_to_posterior(log_ll, prior=prior)

    # Validate output if requested
    if validate:
        _validate_output(posterior)

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


def _log_poisson_likelihood_nan_safe(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    nan_mask: NDArray[np.bool_],
    dt: float,
    *,
    min_rate: float = 1e-10,
) -> NDArray[np.float64]:
    """Poisson log-likelihood that excludes NaN encoding-model bins.

    Identical to :func:`~neurospatial.decoding.likelihood.log_poisson_likelihood`
    except that each ``(neuron, bin)`` flagged in ``nan_mask`` contributes
    zero to the per-bin log-likelihood: both that neuron's spike term
    ``n_i * log(lambda_i * dt)`` and its rate-penalty term ``-lambda_i * dt``
    are dropped at that bin, as if the neuron had not been observed there.
    This is the "treat NaN encoding-model bins as zero-rate (excluded from the
    Poisson sum)" defense-in-depth behavior.

    A bin that is NaN for *every* neuron is a special case: it has no observing
    neuron, so excluding all terms would leave a neutral log-likelihood of 0
    and let an uninformative bin spuriously win the MAP. Such all-NaN bins are
    therefore set to ``-inf`` (zero posterior mass). Partial-NaN bins -- NaN for
    some neurons but observed by at least one other -- still decode normally
    from the observing neurons.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps in Hz; may contain NaN at masked bins.
    nan_mask : NDArray[np.bool_], shape (n_neurons, n_bins)
        ``True`` where ``encoding_models`` is NaN. Those entries are excluded.
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
    if dt <= 0:
        raise ValueError(
            f"dt must be positive, got {dt}. "
            f"Time bin width should be in seconds (typical values: 0.001-0.1s)."
        )
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
