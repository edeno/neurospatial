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

assembly_event_response : Activation strength around events
    R(t) averaged around SWR onsets and each event's response in a window
    (Harvey et al. 2023; Liu et al. 2023; Chang et al. 2025; Robinson et al.
    2026), with a signed-rank test against a baseline window.

assembly_member_cofiring : Correlation of a pattern's members in two periods
    Mean member-pair correlation over SWRs before and after (El-Gaby et al.
    2021).

assembly_similarity : Absolute cosine similarity of two sets of patterns
    Each pattern's best match in another session (El-Gaby et al. 2021).

marchenko_pastur_threshold : Random matrix theory significance threshold
    Upper bound of eigenvalue distribution for random correlations.

Pairwise Reactivation
---------------------
pairwise_correlations : Compute pairwise correlations between neurons
    Extract upper triangle of correlation matrix.

explained_variance_reactivation : Explained variance reactivation analysis
    Partial correlation of template and match pair correlations controlling
    for control periods (Kudrimoti et al. 1999), with REV (Pennartz et al.
    2004), the semipartial correlation (Giri et al. 2019; Liu et al. 2023),
    pair selection and a p-value.

explained_variance_contributions : Each pair's and unit's share of EV
    EV with all pairs minus EV without each pair (Oliva et al. 2020).

ev_rev_contrast : (EV - REV) / (EV + REV)
    The explained variance ratio of Karaba et al. (2024).

Published Methods
-----------------
ReactivationMethod : A published reactivation analysis as a recipe
    Bin size, pairs, windows, statistic and within-session test.
ReactivationWindows : How a period's epochs are cut into windows
EventResponse : How an assembly recipe measures R(t) around events
compare_reactivation_methods : Apply several recipes to one session
    One row per (method, match window, component), with spikes counted in
    fixed bins or one bin per event (SWR, theta cycle).
PUBLISHED_REACTIVATION_METHODS : Every published recipe
    KUDRIMOTI_1999, PENNARTZ_2004, TATSUNO_2006, LANSINK_2009,
    PEYRACHE_2010, VAN_DE_VEN_2016, GIRI_2019, OLIVA_2020_EV,
    OLIVA_2020_ASSEMBLY, EL_GABY_2021, HARVEY_2023_ASSEMBLY, LIU_2023_EV,
    LIU_2023_PAIRWISE, LIU_2023_ASSEMBLY, GIRI_2024, CHANG_2025_ASSEMBLY,
    ROBINSON_2026_PAIRWISE and ROBINSON_2026_ASSEMBLY.

Null Data and Calibration
-------------------------
null_reactivation_data : Match-period spikes without reactivation
    Cell identities permuted, trains rotated in time, or Poisson spikes.
simulated_reactivation_session : Rest, task and rest with assemblies
    Reactivation strength and a rate-change confound are settings.
reactivation_detection_rates : Share of rows called reactivated
    With Clopper-Pearson intervals, overall or by group.
equalize_firing_rates : Subsample spikes to one rate before and after
    The rate control of Giri et al. (2019).
SimulatedReactivationSession : Spike times, epochs and assemblies

Result Containers
-----------------
AssemblyPattern : Single assembly pattern with weights and members
AssemblyDetectionResult : Patterns, activations, eigenvalues and metadata
AssemblyReactivationResult : Mean activation strength before and after a task,
    their difference and the share of bins above the control's percentile
ExplainedVarianceResult : Explained variance, reversed explained variance,
    partial and semipartial correlations, a p-value and the number of pairs
AssemblyEventResponseResult : Peri-event R(t), each event's response, the
    response and the baseline test
AssemblyMemberCofiringResult : Member-pair correlations in two periods
AssemblySimilarityResult : Pattern similarities and best matches
ExplainedVarianceContributionResult : Per-pair and per-unit EV contributions

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
    equalize_firing_rates,
    null_reactivation_data,
    reactivation_detection_rates,
    simulated_reactivation_session,
)
from neurospatial.reactivation.coactivity import (
    AssemblyEventResponseResult,
    AssemblyMemberCofiringResult,
    AssemblySimilarityResult,
    ExplainedVarianceContributionResult,
    assembly_event_response,
    assembly_member_cofiring,
    assembly_similarity,
    ev_rev_contrast,
    explained_variance_contributions,
)
from neurospatial.reactivation.methods import (
    CHANG_2025_ASSEMBLY,
    EL_GABY_2021,
    GIRI_2019,
    GIRI_2024,
    HARVEY_2023_ASSEMBLY,
    KUDRIMOTI_1999,
    LANSINK_2009,
    LIU_2023_ASSEMBLY,
    LIU_2023_EV,
    LIU_2023_PAIRWISE,
    OLIVA_2020_ASSEMBLY,
    OLIVA_2020_EV,
    PENNARTZ_2004,
    PEYRACHE_2010,
    PUBLISHED_REACTIVATION_METHODS,
    ROBINSON_2026_ASSEMBLY,
    ROBINSON_2026_PAIRWISE,
    TATSUNO_2006,
    VAN_DE_VEN_2016,
    EventResponse,
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
    "AssemblyEventResponseResult",
    "AssemblyMemberCofiringResult",
    "AssemblySimilarityResult",
    "ExplainedVarianceContributionResult",
    # Cell assemblies
    "assembly_activation",
    "assembly_reactivation",
    "assembly_event_response",
    "assembly_member_cofiring",
    "assembly_similarity",
    "detect_assemblies",
    "marchenko_pastur_threshold",
    # Pairwise reactivation
    "explained_variance_reactivation",
    "explained_variance_contributions",
    "ev_rev_contrast",
    "pairwise_correlations",
    # Published methods
    "EventResponse",
    "ReactivationMethod",
    "ReactivationWindows",
    "compare_reactivation_methods",
    "PUBLISHED_REACTIVATION_METHODS",
    "CHANG_2025_ASSEMBLY",
    "EL_GABY_2021",
    "GIRI_2019",
    "GIRI_2024",
    "HARVEY_2023_ASSEMBLY",
    "KUDRIMOTI_1999",
    "LANSINK_2009",
    "LIU_2023_ASSEMBLY",
    "LIU_2023_EV",
    "LIU_2023_PAIRWISE",
    "OLIVA_2020_ASSEMBLY",
    "OLIVA_2020_EV",
    "PENNARTZ_2004",
    "PEYRACHE_2010",
    "ROBINSON_2026_ASSEMBLY",
    "ROBINSON_2026_PAIRWISE",
    "TATSUNO_2006",
    "VAN_DE_VEN_2016",
    # Null data and calibration
    "SimulatedReactivationSession",
    "equalize_firing_rates",
    "null_reactivation_data",
    "reactivation_detection_rates",
    "simulated_reactivation_session",
]
