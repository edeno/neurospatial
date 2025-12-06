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
>>> from neurospatial.stats.circular import circular_basis
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
]
