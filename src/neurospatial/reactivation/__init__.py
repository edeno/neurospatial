"""Reactivation analysis: cell assemblies and co-firing compared between periods.

These analyses use co-firing in binned spike counts, not decoded position, so
they take no environment.

Cell Assemblies
---------------
detect_assemblies : Detect cell assemblies from population spike counts
    ICA/PCA/NMF-based detection of coordinated neural ensembles.

assembly_activation : Compute activation strength of an assembly
    Activation strength R(t): co-firing of the pattern's neurons per time bin.

assembly_reactivation : Compare a pattern's activation after a task with before
    Mean activation strength after minus before (van de Ven et al. 2016) and
    the share of bins above the control's 99th percentile (Peyrache et al.
    2010).

marchenko_pastur_threshold : Random matrix theory significance threshold
    Upper bound of eigenvalue distribution for random correlations.

Pairwise Reactivation
---------------------
pairwise_correlations : Compute pairwise correlations between neurons
    Extract upper triangle of correlation matrix.

explained_variance_reactivation : Explained variance reactivation analysis
    Partial correlation of template and match pair correlations controlling
    for control periods (Kudrimoti et al. 1999), with REV (Pennartz et al.
    2004), pair selection and a p-value.

Published Methods
-----------------
ReactivationMethod : A published reactivation analysis as a recipe
    Bin size, pairs, windows, statistic and within-session test.
ReactivationWindows : How a period's epochs are cut into windows
compare_reactivation_methods : Apply several recipes to one session
    One row per (method, match window, component).
PUBLISHED_REACTIVATION_METHODS : Every published recipe
    KUDRIMOTI_1999, PENNARTZ_2004, TATSUNO_2006, LANSINK_2009,
    PEYRACHE_2010, VAN_DE_VEN_2016, GIRI_2019 and GIRI_2024.

Null Data and Calibration
-------------------------
null_reactivation_data : Match-period spikes without reactivation
    Cell identities permuted, trains rotated in time, or Poisson spikes.
simulated_reactivation_session : Rest, task and rest with assemblies
    Reactivation strength and a rate-change confound are settings.
reactivation_detection_rates : Share of rows called reactivated
    With Clopper-Pearson intervals, overall or by group.
SimulatedReactivationSession : Spike times, epochs and assemblies

Result Containers
-----------------
AssemblyPattern : Single assembly pattern with weights and members
AssemblyDetectionResult : Patterns, activations, eigenvalues and metadata
AssemblyReactivationResult : Mean activation strength before and after a task,
    their difference and the share of bins above the control's percentile
ExplainedVarianceResult : Explained variance, reversed explained variance,
    partial correlation, its p-value and the number of pairs

Bin spike times with :func:`neurospatial.decoding.bin_spikes_in_time`.
"""

from neurospatial.reactivation.assemblies import (
    AssemblyDetectionResult,
    AssemblyPattern,
    AssemblyReactivationResult,
    ExplainedVarianceResult,
    assembly_activation,
    assembly_reactivation,
    detect_assemblies,
    explained_variance_reactivation,
    marchenko_pastur_threshold,
    pairwise_correlations,
)
from neurospatial.reactivation.calibration import (
    SimulatedReactivationSession,
    null_reactivation_data,
    reactivation_detection_rates,
    simulated_reactivation_session,
)
from neurospatial.reactivation.methods import (
    GIRI_2019,
    GIRI_2024,
    KUDRIMOTI_1999,
    LANSINK_2009,
    PENNARTZ_2004,
    PEYRACHE_2010,
    PUBLISHED_REACTIVATION_METHODS,
    TATSUNO_2006,
    VAN_DE_VEN_2016,
    ReactivationMethod,
    ReactivationWindows,
    compare_reactivation_methods,
)

__all__ = [  # noqa: RUF022 (organized by category, not alphabetically)
    # Result containers
    "AssemblyDetectionResult",
    "AssemblyPattern",
    "AssemblyReactivationResult",
    "ExplainedVarianceResult",
    # Cell assemblies
    "assembly_activation",
    "assembly_reactivation",
    "detect_assemblies",
    "marchenko_pastur_threshold",
    # Pairwise reactivation
    "explained_variance_reactivation",
    "pairwise_correlations",
    # Published methods
    "ReactivationMethod",
    "ReactivationWindows",
    "compare_reactivation_methods",
    "PUBLISHED_REACTIVATION_METHODS",
    "GIRI_2019",
    "GIRI_2024",
    "KUDRIMOTI_1999",
    "LANSINK_2009",
    "PENNARTZ_2004",
    "PEYRACHE_2010",
    "TATSUNO_2006",
    "VAN_DE_VEN_2016",
    # Null data and calibration
    "SimulatedReactivationSession",
    "null_reactivation_data",
    "reactivation_detection_rates",
    "simulated_reactivation_session",
]
