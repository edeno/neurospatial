"""Bayesian decoding subpackage for population neural analysis.

This subpackage provides tools for decoding spatial position from population
neural activity using Bayesian methods.

Core Functions
--------------
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
map_estimate : Maximum a posteriori bin index
    Bin index of highest posterior probability per time bin.

map_position : MAP position in environment coordinates
    Coordinate position of the MAP bin.

mean_position : Posterior mean position
    Probability-weighted average position.

entropy : Posterior entropy in bits
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

Shuffle Controls (Re-exported from neurospatial.stats)
------------------------------------------------------
shuffle_time_bins : Shuffle temporal order of time bins
    Test that sequential structure is significant.

shuffle_cell_identity : Shuffle cell identity labels
    Test that spatial code coherence is significant.

compute_shuffle_pvalue : Compute p-value from null distribution
    Statistical significance from shuffle controls.

ShuffleTestResult : Container for shuffle test results
    Stores observed score, null distribution, and p-value.

generate_poisson_surrogates : Generate rate-matched Poisson surrogates
    Test that observed structure exceeds rate-based expectations.

Examples
--------
Basic decoding workflow::

    >>> from neurospatial import Environment, compute_place_field
    >>> from neurospatial.decoding import decode_position
    >>>
    >>> # Create environment and compute place fields
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> encoding_models = np.array([
    ...     compute_place_field(env, spikes, times, positions)
    ...     for spikes in spike_times_list
    ... ])
    >>>
    >>> # Decode position
    >>> result = decode_position(env, spike_counts, encoding_models, dt=0.025)
    >>> print(f"Decoded positions: {result.map_position[:5]}")
    >>> print(f"Uncertainty: {result.uncertainty.mean():.2f} bits")

See Also
--------
neurospatial.compute_place_field : Compute encoding models (place fields)
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

from neurospatial.decoding._result import DecodingResult
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
    entropy,
    map_estimate,
    map_position,
    mean_position,
)
from neurospatial.decoding.likelihood import (
    log_poisson_likelihood,
    poisson_likelihood,
)
from neurospatial.decoding.metrics import (
    confusion_matrix,
    decoding_correlation,
    decoding_error,
    median_decoding_error,
)
from neurospatial.decoding.posterior import (
    decode_position,
    normalize_to_posterior,
)
from neurospatial.decoding.trajectory import (
    IsotonicFitResult,
    LinearFitResult,
    RadonDetectionResult,
    detect_trajectory_radon,
    fit_isotonic_trajectory,
    fit_linear_trajectory,
)

# Re-export shuffle controls from stats for discoverability in decoding workflows.
# Canonical location: neurospatial.stats.shuffle
from neurospatial.stats.shuffle import (
    ShuffleTestResult,
    compute_shuffle_pvalue,
    shuffle_cell_identity,
    shuffle_time_bins,
)

# Re-export surrogate generation from stats for discoverability in decoding workflows.
# Canonical location: neurospatial.stats.surrogates
from neurospatial.stats.surrogates import generate_poisson_surrogates

# =============================================================================
# Public API exports
# =============================================================================

__all__ = [  # noqa: RUF022 (organized by category, not alphabetically)
    # Result containers
    "AssemblyDetectionResult",
    "AssemblyPattern",
    "DecodingResult",
    "ExplainedVarianceResult",
    "IsotonicFitResult",
    "LinearFitResult",
    "RadonDetectionResult",
    "ShuffleTestResult",
    # Cell assembly detection
    "assembly_activation",
    # Metrics
    "confusion_matrix",
    "credible_region",
    # Shuffle controls (re-exported from stats)
    "compute_shuffle_pvalue",
    # Core decoding
    "decode_position",
    "decoding_correlation",
    "decoding_error",
    "detect_assemblies",
    # Trajectory analysis
    "detect_trajectory_radon",
    "entropy",
    "explained_variance_reactivation",
    "fit_isotonic_trajectory",
    "fit_linear_trajectory",
    # Surrogates (re-exported from stats)
    "generate_poisson_surrogates",
    # Likelihood
    "log_poisson_likelihood",
    # Estimates
    "map_estimate",
    "map_position",
    "marchenko_pastur_threshold",
    "mean_position",
    "median_decoding_error",
    "normalize_to_posterior",
    "pairwise_correlations",
    "poisson_likelihood",
    "reactivation_strength",
    # Shuffle controls (re-exported from stats)
    "shuffle_cell_identity",
    "shuffle_time_bins",
]
