"""Reactivation analysis: cell assemblies and co-firing compared between periods.

These analyses use co-firing in binned spike counts, not decoded position, so
they take no environment.

Cell Assemblies
---------------
detect_assemblies : Detect cell assemblies from population spike counts
    ICA/PCA/NMF-based detection of coordinated neural ensembles.

assembly_activation : Compute activation strength of an assembly
    Activation strength R(t): co-firing of the pattern's neurons per time bin.

reactivation_strength : Compare assembly activation between periods
    Mean activation strength in a match period over a template period.

marchenko_pastur_threshold : Random matrix theory significance threshold
    Upper bound of eigenvalue distribution for random correlations.

Pairwise Reactivation
---------------------
pairwise_correlations : Compute pairwise correlations between neurons
    Extract upper triangle of correlation matrix.

explained_variance_reactivation : Explained variance reactivation analysis
    Measure how template correlations predict match correlations.

Result Containers
-----------------
AssemblyPattern : Single assembly pattern with weights and members
AssemblyDetectionResult : Patterns, activations, eigenvalues and metadata
ExplainedVarianceResult : Explained variance, reversed explained variance and
    the number of pairs

Bin spike times with :func:`neurospatial.decoding.bin_spikes_in_time`.
"""

from neurospatial.reactivation.assemblies import (
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

__all__ = [  # noqa: RUF022 (organized by category, not alphabetically)
    # Result containers
    "AssemblyDetectionResult",
    "AssemblyPattern",
    "ExplainedVarianceResult",
    # Cell assemblies
    "assembly_activation",
    "detect_assemblies",
    "marchenko_pastur_threshold",
    "reactivation_strength",
    # Pairwise reactivation
    "explained_variance_reactivation",
    "pairwise_correlations",
]
