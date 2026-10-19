"""Bayesian decoding subpackage for population neural analysis.

This subpackage provides tools for decoding spatial position from population
neural activity using Bayesian methods.

Core Functions
--------------
bin_spikes_in_time : Bin spike times into a count matrix on a regular time grid
    Turns per-neuron spike-time arrays into a ``(time, neuron)`` (or
    ``(neuron, time)``) count matrix, owning the time-grid construction.

decode_position : Main entry point for Bayesian decoding
    Decode position from spike counts using encoding models (place fields).

log_poisson_likelihood : Compute log-likelihood under Poisson model
    Numerically stable log-likelihood computation.

normalize_to_posterior : Convert log-likelihood to posterior
    Bayes' rule with configurable prior and degenerate handling.

Result Containers
-----------------
DecodingResult : Container for decoding results
    Stores posterior with lazy-computed derived properties (MAP, mean, entropy).

Estimate Functions
------------------
posterior_mode : Maximum a posteriori bin index
    Bin index of highest posterior probability per time bin.

map_position : MAP position in environment coordinates
    Coordinate position of the MAP bin.

mean_position : Posterior mean position
    Probability-weighted average position.

posterior_entropy : Posterior entropy in bits
    Uncertainty measure (0 = certain, log2(n_bins) = uniform).

credible_region : Highest posterior density region
    Smallest set of bins containing specified probability mass.

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

Trajectory Analysis
-------------------
fit_isotonic_trajectory : Monotonic trajectory fitting
    Isotonic regression on posterior sequence.

fit_linear_trajectory : Linear trajectory fitting
    Linear regression with optional Monte Carlo uncertainty.

detect_trajectory_radon : Radon transform trajectory detection
    Detect linear trajectories in posterior images.

Cell Assembly Detection
-----------------------
detect_assemblies : Detect cell assemblies from population spike counts
    ICA/PCA/NMF-based detection of coordinated neural ensembles.

assembly_activation : Compute activation strength of an assembly
    Project spike counts onto assembly pattern.

pairwise_correlations : Compute pairwise correlations between neurons
    Extract upper triangle of correlation matrix.

explained_variance_reactivation : Explained variance reactivation analysis
    Measure how template correlations predict match correlations.

reactivation_strength : Compare assembly activation between periods
    Simple reactivation measure based on mean activation.

marchenko_pastur_threshold : Random matrix theory significance threshold
    Upper bound of eigenvalue distribution for random correlations.

AssemblyPattern : Single assembly pattern with weights and members
    Frozen dataclass with neuron weights and member indices.

AssemblyDetectionResult : Results from assembly detection
    Contains patterns, activations, eigenvalues, and metadata.

ExplainedVarianceResult : Results from explained variance analysis
    Contains EV, reversed EV, partial correlation, and n_pairs.

Shuffle controls and Poisson surrogates live under :mod:`neurospatial.stats`
(``neurospatial.stats.shuffle`` and ``neurospatial.stats.surrogates``). They
are no longer re-exported here — import them directly from their canonical
locations.

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
    >>> rates = compute_spatial_rates(
    ...     env, spike_times, times, positions, fill_value=0.0
    ... )
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
such as ``prior=rates.spatial_support.astype(float)``.

See Also
--------
neurospatial.decoding.decode_session : One-call encode + time-bin + decode
neurospatial.encoding.compute_spatial_rates : Batch encoding models (place fields)
neurospatial.Environment : Spatial discretization

Notes
-----
This subpackage follows neurospatial's design patterns:

- Stateless functions (not classes) for all operations
- Environment as first argument for spatial context
- NumPy docstrings with examples
- Immutable result containers with lazy-computed properties

The implementation is based on standard Bayesian methods for neural decoding,
with numerical stability considerations for log-domain computation.
"""

from neurospatial.decoding._result import DecodingResult, DecodingSummary
from neurospatial.decoding._time_bins import bin_spikes_in_time
from neurospatial.decoding.assemblies import (
    AssemblyDetectionResult,
    AssemblyPattern,
    ExplainedVarianceResult,
    assembly_activation,
    detect_assemblies,
    explained_variance_reactivation,
    marchenko_pastur_threshold,
    pairwise_correlations,
    reactivation_strength,
)
from neurospatial.decoding.estimates import (
    credible_region,
    map_position,
    mean_position,
    posterior_entropy,
    posterior_mode,
)
from neurospatial.decoding.estimator import BayesianDecoder
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
from neurospatial.decoding.trajectory import (
    IsotonicFitResult,
    LinearFitResult,
    RadonDetectionResult,
    detect_trajectory_radon,
    fit_isotonic_trajectory,
    fit_linear_trajectory,
)

# =============================================================================
# Public API exports
# =============================================================================

# The ``_`` names imported here are internal helpers shared with other
# neurospatial packages; they are not public API.
__all__ = [  # noqa: RUF022 (organized by category, not alphabetically)
    # Immutable fit/predict estimator
    "BayesianDecoder",
    # Result containers
    "AssemblyDetectionResult",
    "AssemblyPattern",
    "DecodingResult",
    "DecodingSummary",
    "ExplainedVarianceResult",
    "IsotonicFitResult",
    "LinearFitResult",
    "RadonDetectionResult",
    # Spike-time binning
    "bin_spikes_in_time",
    # Cell assembly detection
    "assembly_activation",
    # Metrics
    "confusion_matrix",
    "credible_region",
    # One-call golden path
    "decode_session",
    "decode_session_summary",
    # Core decoding
    "decode_position",
    "decode_position_summary",
    "decoding_correlation",
    "decoding_error",
    "detect_assemblies",
    # Trajectory analysis
    "detect_trajectory_radon",
    "explained_variance_reactivation",
    "fit_isotonic_trajectory",
    "fit_linear_trajectory",
    # Likelihood
    "log_poisson_likelihood",
    # Estimates
    "posterior_mode",
    "posterior_entropy",
    "map_position",
    "marchenko_pastur_threshold",
    "mean_position",
    "median_decoding_error",
    "normalize_to_posterior",
    "pairwise_correlations",
    "reactivation_strength",
]
