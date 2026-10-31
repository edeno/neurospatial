"""
Statistical methods.

This module provides general-purpose statistical utilities
used across encoding, decoding, and behavior analysis. Shuffles of rate maps
and posteriors for replay tests live in :mod:`neurospatial.replay`.

Submodules
----------
circular : Circular statistics, Rayleigh test
shuffle : Temporal, trial and spike-train shuffles; Monte Carlo p-values
surrogates : Surrogate data generation

Imports
-------
>>> from neurospatial.stats import rayleigh_test, circular_mean
>>> from neurospatial.stats import shuffle_time_bins, compute_shuffle_pvalue
>>> from neurospatial.stats import generate_poisson_surrogates, generate_jittered_spikes
>>> from neurospatial.stats.circular import circular_basis
>>> from neurospatial.stats.shuffle import ShuffleTestResult
>>> from neurospatial.stats.surrogates import generate_inhomogeneous_poisson_surrogates
"""

from neurospatial.stats._utils import _binomial_rate as _binomial_rate
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
from neurospatial.stats.circular import _validate_paired_input as _validate_paired_input
from neurospatial.stats.shuffle import _ALTERNATIVES as _ALTERNATIVES
from neurospatial.stats.shuffle import (
    ShuffleTestResult,
    compute_shuffle_pvalue,
    compute_shuffle_zscore,
    shuffle_spike_times_circular,
    shuffle_spikes_isi,
    shuffle_time_bins,
    shuffle_time_bins_coherent,
    shuffle_trials,
)
from neurospatial.stats.shuffle import (
    _validate_circular_shift_settings as _validate_circular_shift_settings,
)
from neurospatial.stats.surrogates import (
    generate_inhomogeneous_poisson_surrogates,
    generate_jittered_spikes,
    generate_poisson_surrogates,
)

# The ``_`` names imported here are internal helpers shared with other
# neurospatial packages; they are not public API.
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
    "shuffle_spikes_isi",
    "shuffle_spike_times_circular",
    "shuffle_time_bins",
    "shuffle_time_bins_coherent",
    "shuffle_trials",
    # Surrogate generation
    "generate_poisson_surrogates",
    "generate_inhomogeneous_poisson_surrogates",
    "generate_jittered_spikes",
]
