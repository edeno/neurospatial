"""Bayesian decoding subpackage for population neural analysis.

This subpackage provides tools for decoding spatial position from population
neural activity using Bayesian methods.

Start Here
----------
decode_session : Encode, time-bin and decode one recording in a single call
    Returns a :class:`DecodingResult` with the full posterior.

BayesianDecoder : Fit encoding models on training data, decode new spikes
    ``BayesianDecoder(env).fit(...)`` or ``BayesianDecoder.from_rates(rates)``,
    then ``predict`` (a :class:`DecodingResult`) or ``predict_summary`` (a
    :class:`DecodingSummary`); use it for held-out evaluation.

decode_session_summary, decode_position_summary : Memory-safe decoding
    Stream the decode in time chunks and return a :class:`DecodingSummary`,
    never holding the full ``(n_time_bins, n_bins)`` posterior.

Result Containers
-----------------
DecodingResult : Full posterior with derived estimates
    ``map_bin``, ``map_position``, ``mean_position``, ``posterior_entropy``,
    plus ``to_dataframe``, ``to_xarray``, ``plot`` and ``error_against``.

DecodingSummary : Per-time reductions from the summary decoders
    The same estimate names as :class:`DecodingResult`, without the posterior.

Building Blocks
---------------
bin_spikes_in_time : Bin spike times into a count matrix on a regular time grid
    Turns per-neuron spike-time arrays into a ``(time, neuron)`` (or
    ``(neuron, time)``) count matrix, owning the time-grid construction.

decode_position : Decode precomputed spike counts with encoding models
    Decode position from spike counts using encoding models (place fields).

decode_jointly : Decode several sets of rate maps in one posterior
    For example both running directions, or two tracks; returns a
    :class:`JointDecodingResult` with the probability of each set.

log_poisson_likelihood : Compute log-likelihood under Poisson model
    Numerically stable log-likelihood computation.

normalize_to_posterior : Convert log-likelihood to posterior
    Bayes' rule with configurable prior and degenerate handling.

Estimate Functions
------------------
map_bin : Maximum a posteriori bin index
    Bin index of highest posterior density per time bin.

map_position : MAP position in environment coordinates
    Coordinate position of the MAP bin.

mean_position : Posterior mean position
    Probability-weighted average position.

posterior_entropy : Posterior entropy in bits
    Uncertainty measure over bins (0 = one bin; log2(n_bins) = equal mass in
    every bin, which on equal bins is a posterior flat over space).

credible_region : Highest-density credible region
    Bins added in order of density until they hold a given probability mass.

Quality Metrics
---------------
decoding_error : Position error per time bin
    Euclidean or graph-based distance between decoded and actual positions.

median_decoding_error : Median decoding error
    Summary statistic for decoding accuracy.

decoding_correlation : Weighted correlation
    Correlation between decoded and actual positions.

confusion_matrix : Confusion matrix between bins
    Spatial confusion analysis.

Replay scores and their shuffles live in :mod:`neurospatial.replay`, cell
assemblies and reactivation in :mod:`neurospatial.reactivation`, and general
shuffle controls and Poisson surrogates in :mod:`neurospatial.stats`.

Examples
--------
One-call golden path (encode + time-bin + decode in a single call). This
recording visits only the first bin, so decoding excludes the other bins::

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import decode_session
    >>> times = np.arange(101) / 10
    >>> positions = np.full((len(times), 1), 5.0)
    >>> spike_times = [np.arange(0.05, 10.0, 0.2), np.arange(0.1, 10.0, 0.4)]
    >>> env = Environment.from_grid_mask(
    ...     np.ones(4, dtype=bool), grid_edges=(np.arange(5.0) * 10,)
    ... )
    >>> env = env.with_metadata(units="cm")
    >>> result = decode_session(env, spike_times, times, positions, dt=0.1)
    >>> result.map_position.shape
    (100, 1)
    >>> bool(np.all(result.posterior[:, 1:] == 0.0))
    True

Explicit two-step form (build batch encoding models, then decode)::

    >>> from neurospatial.encoding import compute_spatial_rates
    >>> from neurospatial.decoding import bin_spikes_in_time, decode_position
    >>> # Batch place fields for the whole population in one call
    >>> rates = compute_spatial_rates(env, spike_times, times, positions)
    >>> spike_counts, time_grid = bin_spikes_in_time(
    ...     spike_times, dt=0.1, t_start=times[0], t_stop=times[-1]
    ... )
    >>> explicit = decode_position(
    ...     env, spike_counts, rates, dt=0.1, times=time_grid
    ... )
    >>> np.array_equal(explicit.posterior, result.posterior)
    True

Pass the rate result itself to preserve training support. Extracting
``rates.firing_rates`` after zero filling loses the distinction between
unobserved space and measured silence; a bare array needs an explicit prior
such as ``prior=rates.spatial_support * env.bin_sizes``.

See Also
--------
neurospatial.decoding.decode_session : One-call encode + time-bin + decode
neurospatial.encoding.compute_spatial_rates : Batch encoding models (place fields)
neurospatial.Environment : Spatial discretization

Notes
-----
Functions that need the environment's bins take it first, for example
``decode_session(env, spike_times, times, positions)`` and ``map_bin(env,
posterior)``. Functions that do not, such as :func:`posterior_entropy` and
:func:`normalize_to_posterior`, take no environment. :class:`BayesianDecoder` is an immutable estimator: ``fit`` returns a
new fitted decoder. Result containers are frozen and their arrays read-only.

A posterior holds a probability mass per bin. The MAP bin and credible
regions rank bins by density, mass divided by bin size, so a large bin does
not win only by holding more space. Likelihoods are combined in log space for
numerical stability.
"""

from neurospatial.decoding._result import _ROW_SUM_TOLERANCE as _ROW_SUM_TOLERANCE
from neurospatial.decoding._result import DecodingResult, DecodingSummary
from neurospatial.decoding._time_bins import bin_spikes_in_time
from neurospatial.decoding.estimates import _density_mode as _density_mode
from neurospatial.decoding.estimates import (
    credible_region,
    map_bin,
    map_position,
    mean_position,
    posterior_entropy,
)
from neurospatial.decoding.estimator import BayesianDecoder
from neurospatial.decoding.joint import JointDecodingResult, decode_jointly
from neurospatial.decoding.likelihood import log_poisson_likelihood
from neurospatial.decoding.metrics import (
    confusion_matrix,
    decoding_correlation,
    decoding_error,
    median_decoding_error,
)
from neurospatial.decoding.posterior import (
    decode_position,
    decode_position_summary,
    normalize_to_posterior,
)
from neurospatial.decoding.session import decode_session, decode_session_summary

# =============================================================================
# Public API exports
# =============================================================================

# The ``_`` names imported here are internal helpers shared with other
# neurospatial packages; they are not public API.
__all__ = [  # noqa: RUF022 (organized by category, not alphabetically)
    # Immutable fit/predict estimator
    "BayesianDecoder",
    # Result containers
    "DecodingResult",
    "DecodingSummary",
    "JointDecodingResult",
    # Spike-time binning
    "bin_spikes_in_time",
    # Metrics
    "confusion_matrix",
    "credible_region",
    # One-call golden path
    "decode_session",
    "decode_session_summary",
    # Core decoding
    "decode_position",
    "decode_position_summary",
    "decode_jointly",
    "decoding_correlation",
    "decoding_error",
    # Likelihood
    "log_poisson_likelihood",
    # Estimates
    "map_bin",
    "posterior_entropy",
    "map_position",
    "mean_position",
    "median_decoding_error",
    "normalize_to_posterior",
]
