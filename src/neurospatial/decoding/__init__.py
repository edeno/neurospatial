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

Shuffle-Based Significance Testing
----------------------------------
shuffle_time_bins : Temporal order shuffle
    Test sequential structure within events.

shuffle_cell_identity : Cell identity shuffle
    Test spatial code coherence.

shuffle_posterior_circular : Posterior circular shuffle
    Test trajectory detection bias.

generate_poisson_surrogates : Poisson surrogate generation
    Test against rate-based null hypothesis.

compute_shuffle_pvalue : P-value from shuffle null distribution
    Monte Carlo p-value with correction.

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
from neurospatial.decoding.posterior import (
    decode_position,
    normalize_to_posterior,
)

# =============================================================================
# Public API exports
# =============================================================================

__all__ = [
    "DecodingResult",
    "credible_region",
    "decode_position",
    "entropy",
    "log_poisson_likelihood",
    "map_estimate",
    "map_position",
    "mean_position",
    "normalize_to_posterior",
    "poisson_likelihood",
]
