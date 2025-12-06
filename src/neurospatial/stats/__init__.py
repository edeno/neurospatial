"""
Statistical methods.

This module provides general-purpose statistical utilities
used across encoding, decoding, and behavior analysis.

Submodules
----------
circular : Circular statistics, Rayleigh test
shuffle : Shuffle controls, permutation tests
surrogates : Surrogate data generation

Imports
-------
>>> from neurospatial.stats import rayleigh_test, circular_mean
>>> from neurospatial.stats import shuffle_time_bins, compute_shuffle_pvalue
>>> from neurospatial.stats.circular import circular_basis
>>> from neurospatial.stats.shuffle import ShuffleTestResult
"""

from neurospatial.stats.circular import (
    CircularBasisResult,
    circular_basis,
    circular_basis_metrics,
    circular_circular_correlation,
    circular_linear_correlation,
    circular_mean,
    circular_variance,
    is_modulated,
    mean_resultant_length,
    phase_position_correlation,
    plot_circular_basis_tuning,
    rayleigh_test,
    wrap_angle,
)
from neurospatial.stats.shuffle import (
    ShuffleTestResult,
    compute_shuffle_pvalue,
    compute_shuffle_zscore,
    generate_inhomogeneous_poisson_surrogates,
    generate_poisson_surrogates,
    shuffle_cell_identity,
    shuffle_place_fields_circular,
    shuffle_place_fields_circular_2d,
    shuffle_posterior_circular,
    shuffle_posterior_weighted_circular,
    shuffle_spikes_isi,
    shuffle_time_bins,
    shuffle_time_bins_coherent,
    shuffle_trials,
)

__all__ = [  # noqa: RUF022
    # Core circular statistics
    "rayleigh_test",
    "circular_linear_correlation",
    "circular_circular_correlation",
    "phase_position_correlation",
    # Public circular statistics
    "circular_mean",
    "circular_variance",
    "mean_resultant_length",
    # Angle utilities
    "wrap_angle",
    # GLM circular basis
    "CircularBasisResult",
    "circular_basis",
    "circular_basis_metrics",
    "is_modulated",
    "plot_circular_basis_tuning",
    # Shuffle functions
    "ShuffleTestResult",
    "compute_shuffle_pvalue",
    "compute_shuffle_zscore",
    "generate_inhomogeneous_poisson_surrogates",
    "generate_poisson_surrogates",
    "shuffle_cell_identity",
    "shuffle_place_fields_circular",
    "shuffle_place_fields_circular_2d",
    "shuffle_posterior_circular",
    "shuffle_posterior_weighted_circular",
    "shuffle_spikes_isi",
    "shuffle_time_bins",
    "shuffle_time_bins_coherent",
    "shuffle_trials",
]
