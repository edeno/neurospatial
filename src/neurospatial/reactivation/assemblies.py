"""
Cell assembly detection and reactivation analysis.

This module provides functions for detecting coordinated neural ensembles
(cell assemblies), measuring their activation strength over time, and
comparing co-firing between behavioral periods.

Cell assemblies are groups of neurons that fire together more often than
expected by chance, representing stored neural patterns. Detecting these
assemblies and measuring their reactivation is fundamental for studying
memory consolidation and replay.

See also: Population analysis guide,
<https://edeno.github.io/neurospatial/user-guide/population-analysis/>
Place-field coverage, place-cell counts and remapping measures are in
``neurospatial.encoding``.

Choosing an Analysis Method
---------------------------
**Use `detect_assemblies()` when you want to:**

- Identify specific neural ensembles and their member neurons
- Track assembly activation over time
- Compare which neurons participate in each assembly
- Analyze individual assembly properties (weights, explained variance)

**Use `explained_variance_reactivation()` when you want to:**

- Measure overall correlation structure preservation
- Compare population-level reactivation between periods
- Classic Kudrimoti et al. (1999) style pairwise correlation analysis
- Don't need to identify specific assemblies

Typical Workflows
-----------------
**Assembly Detection**:

1. Prepare spike count matrix (n_time_bins, n_neurons):

   >>> import numpy as np
   >>> # Example: 50 neurons, 2000 time bins (25ms bins = 50s recording)
   >>> spike_counts = np.random.poisson(5, (2000, 50)).astype(np.float64)

2. Detect assemblies:

   >>> result = detect_assemblies(
   ...     spike_counts, method="ica", unit_ids=sorter_labels
   ... )  # doctest: +SKIP
   >>> print(
   ...     f"Found {result.n_significant} dimensions above threshold"
   ... )  # doctest: +SKIP

3. Analyze assembly patterns:

   >>> for i, pattern in enumerate(result.patterns):  # doctest: +SKIP
   ...     members = pattern.member_unit_ids.tolist()
   ...     ev = pattern.explained_variance_ratio
   ...     print(f"Assembly {i}: units {members}, {ev:.1%} variance")

4. Measure the patterns in another period, matching columns by label:

   >>> activation = assembly_activation(
   ...     counts_sleep, result.patterns[0], unit_ids=sleep_labels
   ... )  # doctest: +SKIP

5. Compare rest after the task with rest before it:

   >>> comparison = assembly_reactivation(
   ...     counts_pre_rest,
   ...     counts_post_rest,
   ...     result.patterns[0],
   ...     control_unit_ids=pre_labels,
   ...     match_unit_ids=post_labels,
   ... )  # doctest: +SKIP
   >>> comparison.mean_activation_difference  # doctest: +SKIP

**Reactivation Analysis**:

1. Compute pairwise correlations in behavior and sleep:

   >>> corr_behavior = pairwise_correlations(counts_behavior)  # doctest: +SKIP
   >>> corr_sleep = pairwise_correlations(counts_sleep)  # doctest: +SKIP

2. Measure explained variance, controlling for rest before the task and
   using only pairs of cells on different tetrodes:

   >>> ev_result = explained_variance_reactivation(
   ...     corr_behavior,
   ...     corr_sleep,
   ...     control_correlations=corr_pre_sleep,
   ...     unit_groups=tetrode_of_unit,
   ... )  # doctest: +SKIP
   >>> print(f"EV = {ev_result.explained_variance:.3f}")  # doctest: +SKIP

Notes
-----
**Bin size**:

The bin size sets the time scale of "firing together". Published analyses
use 25 ms bins (van de Ven et al., 2016, following the assembly time scale
of Harris et al., 2003) up to 100 ms (Peyrache et al., 2010). Shorter bins
hold fewer spikes and give noisier correlations; longer bins merge activity
that is not simultaneous.

**Data requirements**:

The Marchenko-Pastur bound is an asymptotic edge for independent neurons
(see :func:`marchenko_pastur_threshold`); in a short recording an
eigenvalue can pass it by chance, and neurons whose rates drift slowly
spread the eigenvalues further. ``detect_assemblies(component_threshold=...)``
also takes surrogate thresholds from counts shuffled or rotated in time
(Lopes-dos-Santos et al., 2013). Neurons that fire in few bins
give noisy correlations, and neurons with the same count in every bin are
excluded.

References
----------
Lopes-dos-Santos V, Ribeiro S, Tort ABL (2013). Detecting cell assemblies
    in large neuronal populations. J Neurosci Methods 220(2):149-166.
Peyrache A, Khamassi M, Benchenane K, Wiener SI, Battaglia FP (2009).
    Replay of rule-learning related neural patterns in the prefrontal cortex
    during sleep. Nat Neurosci 12(7):919-926.
Peyrache A, Benchenane K, Khamassi M, Wiener SI, Battaglia FP (2010).
    Principal component analysis of ensemble recordings reveals cell
    assemblies at high temporal resolution. J Comput Neurosci
    29(1-2):309-325.
Harris KD, Csicsvari J, Hirase H, Dragoi G, Buzsaki G (2003). Organization
    of cell assemblies in the hippocampus. Nature 424(6948):552-556.
van de Ven GM, Trouche S, McNamara CG, Allen K, Dupret D (2016).
    Hippocampal offline reactivation consolidates recently formed cell
    assembly patterns during sharp wave-ripples. Neuron 92(5):968-974.
Kudrimoti HS, Barnes CA, McNaughton BL (1999). Reactivation of hippocampal
    cell assemblies: effects of behavioral state, experience, and EEG dynamics.
    J Neurosci 19(10):4090-4101.
Pennartz CMA, Lee E, Verheul J, Lipa P, Barnes CA, McNaughton BL (2004).
    The ventral striatum in off-line processing: ensemble reactivation during
    sleep and modulation by hippocampal ripples. J Neurosci 24(29):6446-6456.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, stats

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin, resolve_unit_ids

__all__ = [
    "AssemblyDetectionResult",
    "AssemblyPattern",
    "AssemblyReactivationResult",
    "ExplainedVarianceResult",
    "assembly_activation",
    "assembly_reactivation",
    "detect_assemblies",
    "explained_variance_reactivation",
    "marchenko_pastur_threshold",
    "pairwise_activation",
    "pairwise_correlations",
]


# =============================================================================
# Data Classes
# =============================================================================

AssemblyMethod = Literal["ica", "pca", "nmf"]
_METHODS = ("ica", "pca", "nmf")
ComponentThreshold = Literal[
    "marchenko_pastur", "tracy_widom", "bin_shuffle", "circular_shift"
]
_COMPONENT_THRESHOLDS = (
    "marchenko_pastur",
    "tracy_widom",
    "bin_shuffle",
    "circular_shift",
)
_SURROGATE_THRESHOLDS = ("bin_shuffle", "circular_shift")
MemberRule = Literal["abs_z", "positive_z", "otsu", "inverse_sqrt_n"]
_MEMBER_RULES = ("abs_z", "positive_z", "otsu", "inverse_sqrt_n")
_THRESHOLD_NAMES = {
    "marchenko_pastur": "Marchenko-Pastur bound",
    "tracy_widom": "Marchenko-Pastur bound plus the Tracy-Widom term",
    "bin_shuffle": "bin-shuffle threshold",
    "circular_shift": "circular-shift threshold",
}


@dataclass(frozen=True, repr=False, eq=False)
class AssemblyPattern(ReadOnlyArraysMixin, ResultMixin):
    """
    A detected cell assembly pattern.

    Attributes
    ----------
    weights : NDArray[np.float64], shape (n_neurons,)
        Weight of each neuron in this assembly. Higher absolute values
        indicate stronger participation. :func:`detect_assemblies` scales
        each pattern to unit length and orients it so its largest-magnitude
        weight is positive.
    member_indices : NDArray[np.int64], shape (n_members,)
        Column positions of the "core" members of the assembly, chosen by
        ``detect_assemblies(member_rule=...)``: unique integers in
        ``[0, n_neurons)``, in increasing order.
    explained_variance_ratio : float
        Fraction of the population's variance this pattern accounts for, in
        [0, 1]. Higher values indicate more prominent assemblies. For PCA it
        is the pattern's eigenvalue over the number of varying neurons; for
        ICA, the variance of the z-scored counts reconstructed from the
        pattern's source; for NMF, the variance of the raw counts along the
        unit-length pattern, ``w' S w / trace(S)`` for the counts'
        covariance ``S``, which for a PCA eigenvector would equal its
        eigenvalue fraction. NMF components can overlap, so their fractions
        need not add up to the variance the fit explains.
    unit_ids : ndarray, shape (n_neurons,)
        Label of the unit each weight belongs to (read-only): the
        ``unit_ids=`` given to :func:`detect_assemblies`, else
        ``np.arange(n_neurons)``. Pass the column labels of new counts as
        ``unit_ids=`` to :func:`assembly_activation`, or as
        ``control_unit_ids=`` and ``match_unit_ids=`` to
        :func:`assembly_reactivation`, to match their columns by label
        instead of by position.
    member_unit_ids : ndarray
        Labels of the core members, ``unit_ids[member_indices]``.

    Other Parameters
    ----------------
    _unit_ids_generated : bool, default=False
        Internal identity flag: True when ``unit_ids`` were generated as
        ``np.arange(n_neurons)`` rather than supplied. Generated labels are
        column positions, so they cannot match columns by label.

    Notes
    -----
    **Interpreting Weights**:

    - Positive weights: neurons that co-activate with the assembly
    - Negative weights: neurons whose activity moves opposite to the
      positive-weight neurons
    - Near-zero weights: neurons not participating in this assembly

    **Member Selection**:

    By default members are selected by z-scoring absolute weights across
    neurons and thresholding (``member_rule="abs_z"``). With the default
    cutoff of 2.0, an absolute weight must be more than two standard
    deviations above their mean to be selected.
    This is a heuristic weight cutoff,
    not a calibrated per-neuron significance test. A detected pattern can have
    no core members even when its underlying dimension exceeds the
    Marchenko-Pastur threshold. With ``k`` equal weights among ``n``
    neurons and the rest 0, the members' z-score is
    :math:`\\sqrt{(n - k) / k}`: 3 members of 9 neurons reach only 1.41,
    below the default cutoff, so small recordings or large assemblies need a
    lower ``member_threshold``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reactivation import AssemblyPattern
    >>> pattern = AssemblyPattern(
    ...     weights=np.array([0.7, 0.1, 0.7]),
    ...     member_indices=np.array([0, 2]),
    ...     explained_variance_ratio=0.3,
    ...     unit_ids=np.array([12, 15, 31]),
    ... )
    >>> pattern.member_unit_ids.tolist()
    [12, 31]
    """

    weights: NDArray[np.float64]
    member_indices: NDArray[np.int64]
    explained_variance_ratio: float
    unit_ids: NDArray[Any] | None = field(default=None, kw_only=True, compare=False)
    _unit_ids_generated: bool = field(
        default=False, repr=False, compare=False, kw_only=True
    )

    def __post_init__(self) -> None:
        """Validate the fields, resolve labels and own read-only copies."""
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 1 or not np.all(np.isfinite(weights)):
            raise ValueError(
                "AssemblyPattern: weights must be a finite 1-D array with one "
                f"weight per neuron, got shape {weights.shape}."
            )
        n_neurons = weights.size
        members = np.asarray(self.member_indices)
        if members.size == 0:
            members = members.astype(np.int64)
        if members.ndim != 1 or not np.issubdtype(members.dtype, np.integer):
            raise ValueError(
                _format_error(
                    "AssemblyPattern: member_indices must be a 1-D array of "
                    f"integer column positions, got dtype {members.dtype} and "
                    f"shape {members.shape}.",
                    fix="pass np.flatnonzero(mask) for a boolean mask of members",
                )
            )
        if members.size and (
            members[0] < 0 or members[-1] >= n_neurons or np.any(np.diff(members) <= 0)
        ):
            raise ValueError(
                _format_error(
                    "AssemblyPattern: member_indices must be unique column "
                    f"positions in [0, {n_neurons}) in increasing order, got "
                    f"{members.tolist()}.",
                    why="Why: each member is one of the pattern's weights; a "
                    "negative index would silently name the last units, and a "
                    "repeated one would count a neuron twice.",
                    fix="pass np.unique(member_indices) of valid column positions",
                )
            )
        ratio = float(self.explained_variance_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                "AssemblyPattern: explained_variance_ratio is a fraction of "
                f"the population's variance and must lie in [0, 1], got {ratio}."
            )
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "member_indices", members.astype(np.int64))
        object.__setattr__(self, "explained_variance_ratio", ratio)
        if self.unit_ids is None:
            object.__setattr__(self, "_unit_ids_generated", True)
        labels = resolve_unit_ids(self.unit_ids, n_neurons, context="AssemblyPattern")
        object.__setattr__(self, "unit_ids", labels)
        self._own_arrays()

    @property
    def member_unit_ids(self) -> NDArray[Any]:
        """Labels of the core members, in ``member_indices`` order."""
        return np.asarray(self.unit_ids)[self.member_indices]

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "n_members": int(self.member_indices.size),
            "explained_variance_ratio": self.explained_variance_ratio,
        }


@dataclass(frozen=True, repr=False, eq=False)
class AssemblyDetectionResult(ReadOnlyArraysMixin, ResultMixin):
    """
    Results from cell assembly detection.

    Attributes
    ----------
    patterns : tuple of AssemblyPattern
        Detected assembly patterns, ordered by ``explained_variance_ratio``
        (highest first) for every method. Every pattern carries the result's
        ``unit_ids``.
    activations : NDArray[np.float64], shape (n_time_bins, n_assemblies)
        Activation strength R(t) of each pattern in each time bin of the
        detection counts, time first like the counts; column ``k`` equals
        ``assembly_activation(spike_counts, patterns[k])``.
    method : {'ica', 'pca', 'nmf'}
        Detection method used.
    n_significant : int
        Number of PCA eigenvalues above ``threshold``. This counts retained
        dimensions, not assemblies with thresholded core members. It equals
        len(patterns) with ``n_components="auto"`` (zero patterns when no
        eigenvalue exceeds the threshold) and may differ when n_components
        was specified manually.
    eigenvalues : NDArray[np.float64], shape (n_neurons,)
        Eigenvalues of the correlation matrix of the neurons that vary over
        time, in descending order, useful for scree plots. Each excluded
        constant neuron adds an eigenvalue of 0 at the end.
    threshold : float
        Eigenvalue threshold of ``component_threshold`` for the number of
        varying neurons and time bins.
    unit_ids : ndarray, shape (n_neurons,)
        Label of each column of the detection counts (read-only), the same
        labels every pattern carries: the ``unit_ids=`` given to
        :func:`detect_assemblies`, else ``np.arange(n_neurons)``.
    component_threshold : {'marchenko_pastur', 'tracy_widom', 'bin_shuffle', 'circular_shift'}
        Rule that set ``threshold`` (see :func:`detect_assemblies`).
    surrogate_max_eigenvalues : ndarray, shape (n_shuffles,)
        Largest eigenvalue of each surrogate count matrix for the
        ``"bin_shuffle"`` and ``"circular_shift"`` thresholds, in the order
        drawn; empty for the analytical thresholds.

    Other Parameters
    ----------------
    _unit_ids_generated : bool, default=False
        Internal identity flag: True when ``unit_ids`` were generated.

    Notes
    -----
    **Significance Determination**:

    ``n_significant`` counts eigenvalues above ``threshold``. This dimension-selection criterion does not assign a p-value
    to a pattern, a neuron weight, or an activation time bin. A selected
    dimension can produce a pattern with an empty ``member_indices`` array.

    **Activation Interpretation**:

    ``activations`` holds the projector activation strength of
    :func:`assembly_activation` for every method: in units of squared
    z-scores, about 0 when the pattern's neurons fire independently, and
    large when they fire together. It describes co-activation; it does not by
    itself establish significance or biological assembly identity.

    Examples
    --------
    >>> result = detect_assemblies(spike_counts)  # doctest: +SKIP
    >>> print(
    ...     f"Found {result.n_significant} dimensions above threshold"
    ... )  # doctest: +SKIP
    >>> print(f"Method: {result.method}")  # doctest: +SKIP
    >>>
    >>> # Plot activation of first assembly
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> plt.plot(result.activations[:, 0])  # doctest: +SKIP
    >>> plt.ylabel("Activation strength")  # doctest: +SKIP
    """

    patterns: tuple[AssemblyPattern, ...]
    activations: NDArray[np.float64]
    method: AssemblyMethod
    n_significant: int
    eigenvalues: NDArray[np.float64]
    threshold: float
    unit_ids: NDArray[Any] | None = field(default=None, kw_only=True, compare=False)
    component_threshold: ComponentThreshold = field(
        default="marchenko_pastur", kw_only=True
    )
    surrogate_max_eigenvalues: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0), kw_only=True, compare=False
    )
    _unit_ids_generated: bool = field(
        default=False, repr=False, compare=False, kw_only=True
    )

    def __post_init__(self) -> None:
        """Validate the fields, resolve labels and own read-only copies."""
        if self.component_threshold not in _COMPONENT_THRESHOLDS:
            raise ValueError(
                "AssemblyDetectionResult: component_threshold must be one of "
                f"{_COMPONENT_THRESHOLDS}, got {self.component_threshold!r}."
            )
        object.__setattr__(
            self,
            "surrogate_max_eigenvalues",
            np.asarray(self.surrogate_max_eigenvalues, dtype=np.float64),
        )
        patterns = tuple(self.patterns)
        if not all(isinstance(pattern, AssemblyPattern) for pattern in patterns):
            raise TypeError(
                "AssemblyDetectionResult: patterns must hold AssemblyPattern objects."
            )
        if self.method not in _METHODS:
            raise ValueError(
                f"AssemblyDetectionResult: method must be one of {_METHODS}, "
                f"got {self.method!r}."
            )
        activations = np.asarray(self.activations, dtype=np.float64)
        if activations.ndim != 2 or activations.shape[1] != len(patterns):
            raise ValueError(
                "AssemblyDetectionResult: activations must have shape "
                f"(n_time_bins, n_assemblies) with n_assemblies == "
                f"len(patterns) == {len(patterns)}, got shape {activations.shape}."
            )
        if self.unit_ids is None:
            object.__setattr__(self, "_unit_ids_generated", True)
        # One eigenvalue per neuron, from the neurons' correlation matrix.
        labels = resolve_unit_ids(
            self.unit_ids, len(self.eigenvalues), context="AssemblyDetectionResult"
        )
        for pattern in patterns:
            if not np.array_equal(np.asarray(pattern.unit_ids), labels):
                raise ValueError(
                    "AssemblyDetectionResult: every pattern must carry the "
                    "result's unit_ids, the labels of the detection counts' "
                    "columns."
                )
        object.__setattr__(self, "patterns", patterns)
        object.__setattr__(self, "activations", activations)
        object.__setattr__(self, "unit_ids", labels)
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "n_assemblies": len(self.patterns),
            "n_significant": self.n_significant,
            "method": self.method,
            "component_threshold": self.component_threshold,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, repr=False)
class AssemblyReactivationResult(ResultMixin):
    """
    A pattern's activation strength after a task compared with before it.

    Attributes
    ----------
    mean_activation_difference : float
        ``match_mean_activation - control_mean_activation``: the pattern's
        mean activation strength R(t) in the match period (e.g. rest after
        the task) minus its mean in the control period (e.g. rest before).
        van de Ven et al. (2016) call this difference the pattern's
        "reactivation strength"; Peyrache et al. (2009, Fig. 2c) report it
        as the difference in average reactivation strength between pre and
        post sleep. 0 when the pattern is expressed equally in both periods.
    control_mean_activation, match_mean_activation : float
        Mean of R(t) over each period. For unit-length weights ``w`` this is
        ``w @ C @ w - 1`` with ``C`` the period's correlation matrix
        (Peyrache et al., 2010: "the average reactivation strength is equal
        to γ − 1"), the pattern's co-firing beyond independent firing.
    control_threshold : float
        The ``percentile``-th percentile of R(t) over the control bins
        (NumPy's default linear interpolation).
    fraction_above_control_threshold : float
        Share of match bins with R(t) strictly above ``control_threshold``
        (Peyrache et al., 2010, "Reactivation is a rare event"). Without
        reactivation it is about ``1 - percentile / 100``.
    percentile : float
        Percentile of the control distribution used for the threshold.
    n_control_bins, n_match_bins : int
        Number of time bins in each period.

    Notes
    -----
    Peyrache et al. (2009, 2010) call R(t) itself the "reactivation
    strength"; here R(t) is the activation strength of
    :func:`assembly_activation`, and the comparison between periods is
    ``mean_activation_difference``. Both statistics are effect sizes: neither
    is a p-value.

    Examples
    --------
    >>> result = assembly_reactivation(
    ...     counts_pre_rest, counts_post_rest, pattern
    ... )  # doctest: +SKIP
    >>> result.mean_activation_difference  # doctest: +SKIP
    """

    mean_activation_difference: float
    control_mean_activation: float
    match_mean_activation: float
    control_threshold: float
    fraction_above_control_threshold: float
    percentile: float
    n_control_bins: int
    n_match_bins: int

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "mean_activation_difference": self.mean_activation_difference,
            "control_mean_activation": self.control_mean_activation,
            "match_mean_activation": self.match_mean_activation,
            "control_threshold": self.control_threshold,
            "fraction_above_control_threshold": (self.fraction_above_control_threshold),
            "percentile": self.percentile,
            "n_control_bins": self.n_control_bins,
            "n_match_bins": self.n_match_bins,
        }


@dataclass(frozen=True, repr=False)
class ExplainedVarianceResult(ResultMixin):
    """
    Results from explained variance reactivation analysis.

    Attributes
    ----------
    explained_variance : float
        Squared correlation between template and match correlation vectors,
        partialled for the control periods when they are supplied. Range
        [0, 1].
    reversed_explained_variance : float
        REV (Pennartz et al., 2004): with one control period, the squared
        partial correlation between template and control vectors, partialled
        for the match. NaN without a control (there is no period to swap)
        and with several controls (REV is defined for one).
    partial_correlation : float
        Partial correlation between template and match, controlling for the
        control periods; without a control, their plain correlation. It is
        also the correlation between the residuals of the template and of
        the match correlations, each regressed (with an intercept) on the
        controls (Robinson et al., 2026's pairwise reactivation).
    semipartial_correlation : float
        Correlation between the template correlations and the residuals of
        the match correlations regressed (with an intercept) on the
        controls; only the match side is residualized. With one control,
        ``(r_tm - r_tc * r_mc) / sqrt(1 - r_mc**2)``. Liu et al. (2023),
        after Giri et al. (2019), "used linear regression between CPOST and
        CPRE to remove the portion of the variance in CPOST that can be
        explained by preexisting patterns, and ... correlated the resultant
        residuals CRESIDUAL with C". Its magnitude is at most that of
        ``partial_correlation``. Without a control, the plain correlation.
    p_value : float
        Two-sided p-value of ``partial_correlation`` from the t test
        ``t = r * sqrt(df / (1 - r**2))`` with
        ``df = n_pairs - 2 - n_controls``. It treats pairs as independent
        observations, which they are not (pairs share cells), so it is
        anticonservative. NaN when ``df < 1`` or the correlation is
        undefined.
    n_pairs : int
        Number of neuron pairs used in analysis, after pair selection and
        dropping pairs that are NaN in any period.
    n_controls : int
        Number of control periods partialled out.

    Notes
    -----
    **Interpretation**:

    With a baseline control, EV > REV describes more template-related
    correlation structure after behavior than before behavior. Without that
    control, REV is NaN and EV alone provides no directional evidence.
    EV and REV are effect sizes, and a fixed cutoff such as EV > 0.1 or
    EV > 0.3 does not establish statistical significance. ``p_value`` tests
    the partial correlation over pairs within one session; because pairs
    share cells it overstates the evidence, so a claim of reactivation needs
    a comparison across sessions (Pennartz et al. 2004: EV vs REV, Wilcoxon
    signed-rank) or a null built for the recording. An in-sample or synthetic
    demonstration does not establish biological reactivation.

    Examples
    --------
    >>> result = explained_variance_reactivation(
    ...     corr_behavior, corr_sleep, control_correlations=corr_pre_behavior
    ... )  # doctest: +SKIP
    >>> if (
    ...     result.explained_variance > result.reversed_explained_variance
    ... ):  # doctest: +SKIP
    ...     print(
    ...         "Template-related effect exceeds the baseline-related effect"
    ...     )  # doctest: +SKIP
    """

    explained_variance: float
    reversed_explained_variance: float
    partial_correlation: float
    semipartial_correlation: float
    p_value: float
    n_pairs: int
    n_controls: int

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "explained_variance": self.explained_variance,
            "reversed_explained_variance": self.reversed_explained_variance,
            "partial_correlation": self.partial_correlation,
            "semipartial_correlation": self.semipartial_correlation,
            "p_value": self.p_value,
            "n_pairs": self.n_pairs,
            "n_controls": self.n_controls,
        }


def _validated_counts(
    spike_counts: NDArray[np.float64] | Sequence[Any],
    *,
    call: str,
    argument: str = "spike_counts",
    nonnegative: bool = True,
) -> NDArray[np.float64]:
    """Return ``spike_counts`` as a 2-D float array of observed counts.

    Parameters
    ----------
    spike_counts : array-like, shape (n_time_bins, n_neurons)
        Counts to check.
    call, argument : str
        Function and argument names for error messages.
    nonnegative : bool, default=True
        Refuse negative values. False for counts the caller normalized.

    Returns
    -------
    ndarray, shape (n_time_bins, n_neurons)
        The counts as float64.

    Raises
    ------
    ValueError
        If the counts are not 2-D, hold NaN or infinite values, or hold
        negative values when ``nonnegative``.
    """
    counts = np.asarray(spike_counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(
            f"{argument} must be 2D (n_time_bins, n_neurons), got shape {counts.shape}"
        )
    finite = np.isfinite(counts)
    if not finite.all():
        n_bad_bins = int(np.sum(~finite.all(axis=1)))
        columns = np.flatnonzero(~finite.all(axis=0)).tolist()
        raise ValueError(
            _format_error(
                f"{call}: {argument} has NaN or infinite values in {n_bad_bins} "
                f"of {counts.shape[0]} time bins, in neuron columns {columns}.",
                why="Why: a NaN bin was not observed, so it has no count. "
                "Z-scoring through it would remove the neuron for the whole "
                "period, and filling it with 0 would count the gap as silence.",
                fix=f"drop the gap bins before the call, e.g. "
                f"{argument}[np.isfinite({argument}).all(axis=1)], or bin only "
                "the recorded epochs",
            )
        )
    if nonnegative and np.any(counts < 0):
        columns = np.flatnonzero((counts < 0).any(axis=0)).tolist()
        raise ValueError(
            _format_error(
                f"{call}: {argument} has negative values in neuron columns {columns}.",
                why="Why: spike counts are numbers of spikes per bin and cannot "
                "be negative; negative values usually mean the counts were "
                "already normalized or a fill value marks missing bins.",
                fix="pass the raw binned spike counts from bin_spikes_in_time(), "
                "and drop bins that were not recorded",
            )
        )
    return counts


# =============================================================================
# Marchenko-Pastur Threshold
# =============================================================================


def marchenko_pastur_threshold(
    n_neurons: int,
    n_time_bins: int,
) -> float:
    """
    Compute the Marchenko-Pastur reference threshold for eigenvalues.

    Eigenvalues of the correlation matrix above this reference edge are used
    to select dimensions under the random-matrix model. This returns a
    threshold, not a calibrated p-value for a finite recording.

    Parameters
    ----------
    n_neurons : int
        Number of neurons (columns in the (n_time_bins, n_neurons) spike-count
        matrix).
    n_time_bins : int
        Number of time bins (rows in the (n_time_bins, n_neurons) spike-count
        matrix).

    Returns
    -------
    float
        Upper reference edge of the random-matrix eigenvalue distribution.
        The detector counts eigenvalues exceeding this edge.

    Notes
    -----
    **Marchenko-Pastur Distribution**:

    For a random matrix with i.i.d. entries, the eigenvalues of its
    correlation matrix follow the Marchenko-Pastur distribution with
    upper bound:

    .. math::

        \\lambda_{max} = \\left(1 + \\sqrt{\\frac{n}{p}}\\right)^2

    where n = n_neurons and p = n_time_bins.

    **Interpretation**:

    - Eigenvalues > threshold: dimensions selected by the reference criterion
    - Eigenvalues < threshold: dimensions not selected by that criterion

    Selection depends on the reference model's assumptions; it does not
    establish a biological assembly or assign significance to its members.

    **Requirements**:

    The bound is the asymptotic edge as both sizes grow at a fixed ratio.
    For a finite recording of independent neurons the largest eigenvalue
    fluctuates around it on a scale of order :math:`n^{-2/3}` (Tracy-Widom;
    Peyrache et al., 2010), so an eigenvalue just above the threshold can
    arise by chance. When it matters, use a surrogate threshold
    (``detect_assemblies(component_threshold="circular_shift", ...)``).

    Examples
    --------
    >>> threshold = marchenko_pastur_threshold(100, 1000)
    >>> print(f"Threshold: {threshold:.3f}")
    Threshold: 1.732

    >>> # Compare with eigenvalues. spike_counts is (n_time_bins, n_neurons),
    >>> # so rowvar=False makes neurons the variables (an n_neurons x n_neurons
    >>> # correlation matrix); the default would correlate time bins instead.
    >>> corr = np.corrcoef(spike_counts, rowvar=False)  # doctest: +SKIP
    >>> eigenvalues = np.linalg.eigvalsh(corr)  # doctest: +SKIP
    >>> n_significant = np.sum(eigenvalues > threshold)  # doctest: +SKIP

    References
    ----------
    Marchenko VA, Pastur LA (1967). Distribution of eigenvalues for some sets
        of random matrices. Mat Sb 114(4):507-536.
    Lopes-dos-Santos V, Ribeiro S, Tort ABL (2013). Detecting cell
        assemblies in large neuronal populations. J Neurosci Methods
        220(2):149-166.
    Peyrache A, Benchenane K, Khamassi M, Wiener SI, Battaglia FP (2010).
        Principal component analysis of ensemble recordings reveals cell
        assemblies at high temporal resolution. J Comput Neurosci
        29(1-2):309-325.
    """
    if n_neurons <= 0:
        raise ValueError(f"n_neurons must be positive, got {n_neurons}")
    if n_time_bins <= 0:
        raise ValueError(f"n_time_bins must be positive, got {n_time_bins}")

    # Aspect ratio
    q = n_neurons / n_time_bins

    if q > 1:
        warnings.warn(
            f"n_neurons ({n_neurons}) > n_time_bins ({n_time_bins}). "
            "Marchenko-Pastur threshold may be unreliable. "
            "Consider using more time bins or fewer neurons.",
            UserWarning,
            stacklevel=2,
        )

    # Upper bound of Marchenko-Pastur distribution
    threshold = (1 + np.sqrt(q)) ** 2

    return float(threshold)


# =============================================================================
# Assembly Detection
# =============================================================================


def detect_assemblies(
    spike_counts: NDArray[np.float64],
    *,
    method: AssemblyMethod = "ica",
    n_components: int | Literal["auto"] = "auto",
    component_threshold: ComponentThreshold = "marchenko_pastur",
    percentile: float | None = None,
    n_shuffles: int = 1000,
    member_rule: MemberRule = "abs_z",
    member_threshold: float = 2.0,
    rng: int | np.random.Generator | None = None,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> AssemblyDetectionResult:
    """
    Detect cell assemblies from population spike counts.

    Extracts population activity patterns using dimensionality reduction on
    the correlation structure and a random-matrix dimension-selection rule.

    Parameters
    ----------
    spike_counts : NDArray[np.float64], shape (n_time_bins, n_neurons)
        Binned spike counts for population. Will be z-scored internally.
        Typical bin size: 25-100ms for assembly detection.
    method : {'ica', 'pca', 'nmf'}, default='ica'
        Detection method:

        - 'ica': Independent Component Analysis (FastICA) in the selected
          principal subspace (Lopes-dos-Santos et al., 2013). Separates
          assemblies whose activations are statistically independent and
          non-Gaussian. Default choice.
        - 'pca': Principal Component Analysis (Peyrache et al., 2010). Finds
          orthogonal patterns ordered by variance. Faster, but assemblies of
          similar strength mix within one pattern.
        - 'nmf': Non-negative Matrix Factorization of the raw counts.
          Constrains weights to be non-negative, so a neuron cannot
          participate by firing less.

    n_components : int or 'auto', default='auto'
        Number of assemblies to detect:

        - 'auto': the number of eigenvalues above ``component_threshold``
          (recommended)
        - int: Fixed number of components

    component_threshold : {'marchenko_pastur', 'tracy_widom', 'bin_shuffle', 'circular_shift'}, default='marchenko_pastur'
        Eigenvalue threshold that selects the number of components:

        - 'marchenko_pastur': the analytical upper edge
          :math:`(1 + \\sqrt{n / B})^2` for ``n`` varying neurons and ``B``
          time bins (Lopes-dos-Santos et al., 2013; Peyrache et al., 2010).
        - 'tracy_widom': that edge plus :math:`n^{-2/3}`, the scale of the
          finite-sample fluctuation of the largest eigenvalue, which "lies
          around λmax with a standard deviation of order N^−2/3" (Peyrache
          et al., 2010). The Lopes-dos-Santos Python code adds the same term
          with ``tracywidom=True``.
        - 'bin_shuffle': the ``percentile``-th percentile of the largest
          eigenvalue of ``n_shuffles`` surrogate matrices in which the time
          bins of each neuron are shuffled independently, which keeps each
          neuron's count distribution and removes its co-firing.
        - 'circular_shift': the same with each neuron's counts rotated in
          time by its own random offset, which also keeps each neuron's
          autocorrelation (Lopes-dos-Santos et al., 2013).

        For the surrogate thresholds, "each surrogate matrix contributes its
        maximal eigenvalue to the chance distribution, which is a more
        conservative approach than considering all surrogate eigenvalues"
        (Lopes-dos-Santos et al., 2013, p. 5).
    percentile : float, optional
        Percentile of the surrogate maxima used as the threshold, in
        (0, 100). Required for ``'bin_shuffle'`` and ``'circular_shift'``
        and refused for the analytical thresholds. There is no single
        published value: Lopes-dos-Santos et al. (2013) used "a certain
        percentile (e.g., 95% or 99%)", and their MATLAB toolbox asks for
        it. Their 95th percentile of circular shifts "matches very closely"
        the Marchenko-Pastur edge, while that of bin shuffles was lower.
    n_shuffles : int, default=1000
        Number of surrogate matrices for the surrogate thresholds, the
        default of the authors' Python code (``nshu=1000``). Ignored by
        the analytical thresholds.

    member_rule : {'abs_z', 'positive_z', 'otsu', 'inverse_sqrt_n'}, default='abs_z'
        How the core members of each pattern are chosen from its weights,
        which are scaled to unit length and oriented so the largest absolute
        weight is positive:

        - 'abs_z': absolute weights z-scored across the varying neurons, above
          ``member_threshold``.
        - 'positive_z': signed weights above ``mean + member_threshold * SD``
          of the pattern's weights, and positive (El-Gaby et al., 2021: "the
          direction where weights were highest was assigned positive weights,
          and ... neurons whose weight was positive and exceeded 2 standard
          deviations from the mean were defined as pattern 'members'"). A
          large negative weight lowers the mean and raises the SD of the
          signed weights, so this can differ from 'abs_z'.
        - 'otsu': Otsu's split of the absolute weights into the two groups
          with the smallest within-group variance; the group with the larger
          absolute weights are members (Harvey et al., 2023). Every split of
          the sorted weights is tried, so the result is exact.
          ``member_threshold`` is not used.
        - 'inverse_sqrt_n': weights above ``1 / sqrt(n_neurons)``, the weight
          every neuron would have in a uniform unit-length pattern (Robinson
          et al., 2026: "units whose weights exceeded 1/√N, where N is the
          number of recorded units"). ``member_threshold`` is not used.

    member_threshold : float, default=2.0
        Z-score cutoff of the 'abs_z' and 'positive_z' rules.

        - 2.0: Default heuristic weight cutoff
        - 2.5: Select fewer core members
        - 1.5: Select more core members

        These cutoffs do not supply per-neuron p-values. A selected dimension
        may yield a pattern with no weights above the chosen cutoff; with
        ``k`` equally weighted members among ``n`` neurons and the other
        weights 0, the members' z-score is ``sqrt((n - k) / k)``.

    rng : int or np.random.Generator, optional
        Random seed or generator for reproducibility. Affects ICA and NMF
        initialization and the surrogate matrices. Both ``int`` seeds and
        ``np.random.Generator`` instances are accepted.
    unit_ids : array-like, shape (n_neurons,), optional
        Label of the unit in each column of ``spike_counts``, for example the
        sorter's unit IDs. Stored on the result and on every pattern, so
        ``pattern.member_unit_ids`` names the members, and passing the labels
        of new counts to :func:`assembly_activation` (``unit_ids=``) or
        :func:`assembly_reactivation` (``control_unit_ids=`` and
        ``match_unit_ids=``) matches columns by label. Default
        ``np.arange(n_neurons)``: column positions, which match new counts
        by position only.

    Returns
    -------
    AssemblyDetectionResult
        Contains detected patterns, activations, eigenvalues, and metadata.

    Raises
    ------
    ValueError
        If spike_counts has wrong shape, holds NaN, infinite or negative
        values, has fewer than 3 neurons that vary over time, or has invalid
        parameters (including a surrogate threshold without ``percentile``
        or an analytical one with it), or ``unit_ids`` does not hold one
        distinct label per column.

    Notes
    -----
    **Algorithm (Lopes-dos-Santos et al., 2013)**:

    1. Z-score spike counts for each neuron
    2. Compute correlation matrix
    3. Eigendecomposition to find significant dimensions (eigenvalues
       above ``component_threshold``)
    4. Project onto significant PCs (and whiten their scores for ICA)
    5. Apply ICA/PCA/NMF to extract assembly patterns
    6. Threshold weights to identify core members
    7. Compute the activation strength R(t) of each pattern (see
       :func:`assembly_activation`)

    **Excluded neurons and components**: A neuron with the same count in
    every time bin has no correlation with any other neuron. It is excluded
    from every method (with a warning), gets weight 0 in every pattern, and
    the thresholds use the number of varying neurons. (The Lopes-dos-Santos
    Python code counts silent neurons in the Marchenko-Pastur ratio; a
    silent neuron only adds an eigenvalue of 0, so here it does not loosen
    the bound.) Surrogate matrices shuffle the z-scored counts of the
    varying neurons, so their correlation matrices are computed exactly like
    the data's; the percentile uses NumPy's linear interpolation. With
    ``n_components="auto"`` and no eigenvalue above the threshold, the result
    has no patterns and a warning says so. Requested components beyond the
    numerical rank of the counts carry no variance and are dropped with a
    warning, so a result can hold fewer patterns than ``n_components``.

    **Sign convention**: ICA and PCA determine a pattern only up to its sign.
    Each pattern is scaled to unit length and flipped so that its
    largest-magnitude weight is positive; its members are then the neurons
    with the largest positive weights. The activation strength does not
    depend on the sign. NMF patterns are non-negative.

    **Choosing a Method**:

    - **ICA**: The default. It assumes the assemblies' activations are
      statistically independent and non-Gaussian; Lopes-dos-Santos et al.
      (2013) show it separates assemblies that PCA mixes, including
      assemblies that share neurons. Activations driven by nearly Gaussian
      fluctuations separate less cleanly.
    - **PCA**: Use for orthogonal patterns or fast exploratory analysis.
      When several assemblies are about equally strong their eigenvalues are
      close, and each eigenvector mixes them: a pattern can then have core
      members from two assemblies, or none, although the patterns together
      still span the assemblies.
    - **NMF**: Use when assemblies should have non-negative membership
      (no participation by firing less). It factorizes raw counts, so
      high-rate neurons weigh more than in the z-scored PCA and ICA.

    **Preprocessing**:

    Input spike counts are z-scored internally for PCA and ICA. Choose the
    bin size for the time scale of co-firing you study (25 to 100 ms in the
    publications cited in the module notes). When the recording is short
    relative to the number of neurons, or the neurons' rates drift slowly,
    compare the number of selected dimensions with a surrogate threshold:
    circular shifts keep each neuron's autocorrelation, so slow drifts that
    spread the eigenvalues beyond the Marchenko-Pastur edge raise their
    threshold too.

    Examples
    --------
    Three of twenty units fire together; label the columns with the
    sorter's unit IDs so the members are named by label:

    >>> import numpy as np
    >>> from neurospatial.reactivation import detect_assemblies
    >>> rng = np.random.default_rng(0)
    >>> counts = rng.poisson(1.0, (2000, 20)).astype(float)
    >>> counts[:, [2, 9, 15]] += 3 * rng.poisson(0.3, (2000, 1))
    >>> labels = np.arange(101, 121)  # the sorter's unit IDs
    >>> result = detect_assemblies(counts, unit_ids=labels, rng=0)
    >>> result.n_significant
    1
    >>> result.patterns[0].member_unit_ids.tolist()
    [103, 110, 116]

    The same counts with a circular-shift surrogate threshold, the 99th
    percentile of the largest eigenvalue of 200 surrogate matrices:

    >>> shifted = detect_assemblies(
    ...     counts,
    ...     component_threshold="circular_shift",
    ...     percentile=99.0,
    ...     n_shuffles=200,
    ...     rng=0,
    ... )
    >>> shifted.n_significant
    1

    >>> # Basic usage
    >>> result = detect_assemblies(spike_counts)  # doctest: +SKIP
    >>> print(
    ...     f"Found {result.n_significant} dimensions above threshold"
    ... )  # doctest: +SKIP

    >>> # With specific number of components
    >>> result = detect_assemblies(spike_counts, n_components=5)  # doctest: +SKIP

    >>> # Using PCA for faster computation
    >>> result = detect_assemblies(spike_counts, method="pca")  # doctest: +SKIP

    >>> # Access assembly patterns
    >>> for i, pattern in enumerate(result.patterns):  # doctest: +SKIP
    ...     print(f"Assembly {i}: units {pattern.member_unit_ids.tolist()}")

    See Also
    --------
    assembly_activation : Compute activation for new data
    marchenko_pastur_threshold : Significance threshold calculation

    References
    ----------
    Lopes-dos-Santos V, Ribeiro S, Tort ABL (2013). Detecting cell
        assemblies in large neuronal populations. J Neurosci Methods
        220(2):149-166.
    Peyrache A, Benchenane K, Khamassi M, Wiener SI, Battaglia FP (2010).
        Principal component analysis of ensemble recordings reveals cell
        assemblies at high temporal resolution. J Comput Neurosci
        29(1-2):309-325.
    """
    spike_counts = _validated_counts(spike_counts, call="detect_assemblies()")
    n_time_bins, n_neurons = spike_counts.shape
    if method not in _METHODS:
        raise ValueError(f"Unknown method: {method!r}. Use 'ica', 'pca', or 'nmf'.")

    if n_neurons < 3:
        raise ValueError(
            f"Need at least 3 neurons for assembly detection, got {n_neurons}"
        )
    unit_ids_generated = unit_ids is None
    labels = resolve_unit_ids(unit_ids, n_neurons, context="detect_assemblies()")
    _check_component_threshold(component_threshold, percentile, n_shuffles)
    if member_rule not in _MEMBER_RULES:
        raise ValueError(
            f"detect_assemblies(): member_rule must be one of {_MEMBER_RULES}, "
            f"got {member_rule!r}."
        )

    if n_time_bins < n_neurons:
        warnings.warn(
            f"spike_counts has n_time_bins ({n_time_bins}) < n_neurons "
            f"({n_neurons}). spike_counts is expected as (n_time_bins, n_neurons) "
            "-- time-first, matching bin_spikes_in_time's default and "
            "decode_position. Fewer time bins than neurons usually means the "
            "matrix is TRANSPOSED (n_neurons, n_time_bins); otherwise the "
            "recording is simply short and results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # A neuron whose count never changes has no variance to correlate: it
    # cannot take part in any covariance pattern, so every method excludes it.
    constant = np.ptp(spike_counts, axis=0) == 0
    retained = np.flatnonzero(~constant)
    n_retained = retained.size
    if n_retained < 3:
        raise ValueError(
            _format_error(
                f"detect_assemblies(): only {n_retained} of {n_neurons} neurons "
                f"vary over time; units {labels[constant].tolist()} have the same "
                "count in every time bin.",
                why="Why: assemblies are patterns of correlated activity, and a "
                "neuron whose count never changes has no correlation with any "
                "other; at least 3 varying neurons are needed.",
                fix="check that spike_counts covers a period in which the units "
                "fire, or select the units that fire in this period",
            )
        )
    if np.any(constant):
        warnings.warn(
            f"detect_assemblies(): {int(constant.sum())} of {n_neurons} neurons "
            f"have the same count in every time bin (units "
            f"{labels[constant].tolist()}), so they have no correlation with "
            "other neurons. They are excluded from detection and get weight 0 "
            "in every pattern.",
            UserWarning,
            stacklevel=2,
        )

    # Work internally in (n_neurons, n_time_bins): the assembly structure is a
    # correlation over neurons, so z-score each neuron over time and correlate
    # neuron-wise.
    retained_counts = spike_counts[:, retained].T
    spike_counts_z = stats.zscore(retained_counts, axis=1)

    # Eigenvalues of the retained neurons' correlation matrix, in descending
    # order; each excluded neuron adds an eigenvalue of 0.
    corr_matrix = np.corrcoef(spike_counts_z)
    eigenvalues = np.r_[
        np.sort(linalg.eigvalsh(corr_matrix))[::-1], np.zeros(n_neurons - n_retained)
    ]

    # Derive an integer seed for sklearn from the public ``rng`` argument.
    # sklearn estimators accept an int seed (or RandomState), not a numpy
    # Generator, so normalize a Generator to a reproducible integer seed.
    if rng is None:
        seed: int | None = None
    elif isinstance(rng, np.random.Generator):
        seed = int(rng.integers(0, 2**32 - 1))
    else:
        seed = int(rng)

    mp_threshold = marchenko_pastur_threshold(n_retained, n_time_bins)
    surrogate_max = np.zeros(0)
    if component_threshold == "marchenko_pastur":
        threshold = mp_threshold
    elif component_threshold == "tracy_widom":
        threshold = mp_threshold + n_retained ** (-2.0 / 3.0)
    else:
        surrogate_max = _surrogate_max_eigenvalues(
            spike_counts_z,
            component_threshold,
            n_shuffles,
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng),
        )
        assert percentile is not None  # checked by _check_component_threshold
        threshold = float(np.percentile(surrogate_max, percentile))
    n_significant = int(np.sum(eigenvalues > threshold))

    # Determine number of components
    if n_components == "auto":
        n_comp = n_significant
        if n_comp == 0:
            warnings.warn(
                f"detect_assemblies(): no eigenvalue of the correlation matrix "
                f"exceeds the {_THRESHOLD_NAMES[component_threshold]} "
                f"{threshold:.3f} (largest "
                f"{eigenvalues[0]:.3f}), so the counts show no correlation "
                "structure beyond what independent neurons would give. The "
                "result has no patterns; pass n_components= to fit exploratory "
                "patterns anyway.",
                UserWarning,
                stacklevel=2,
            )
    else:
        n_comp = int(n_components)
        if n_comp < 1:
            raise ValueError(f"n_components must be >= 1, got {n_comp}")
        if n_comp > n_neurons:
            raise ValueError(
                f"n_components ({n_comp}) cannot exceed n_neurons ({n_neurons})"
            )

    # The factorization can yield at most min(n_retained, n_time_bins)
    # components. Requesting more (short recordings, excluded neurons) would
    # index past the SVD output; clamp with a warning.
    max_rank = min(n_retained, n_time_bins)
    if n_comp > max_rank:
        warnings.warn(
            f"n_components ({n_comp}) exceeds the achievable rank "
            f"min(n_varying_neurons, n_time_bins) = {max_rank}; using {max_rank}.",
            UserWarning,
            stacklevel=2,
        )
        n_comp = max_rank

    # Perform dimensionality reduction
    if n_comp == 0:
        patterns, explained_var = np.zeros((0, n_retained)), np.zeros(0)
    elif method == "pca":
        patterns, explained_var = _detect_pca(spike_counts_z, n_comp)
    elif method == "ica":
        patterns, explained_var = _detect_ica(spike_counts_z, n_comp, seed)
    else:
        # NMF needs non-negative input, so it factorizes the raw counts.
        patterns, explained_var = _detect_nmf(retained_counts, n_comp, seed)
    n_found = patterns.shape[0]
    if n_found < n_comp:
        warnings.warn(
            f"detect_assemblies(): {n_comp - n_found} of the {n_comp} requested "
            f"components carry no variance (the counts span only {n_found} "
            "dimensions), so they define no pattern and are dropped.",
            UserWarning,
            stacklevel=2,
        )

    # Order patterns by explained variance, highest first. PCA eigenvectors
    # already are; ICA sources and NMF components come in arbitrary order.
    order = np.argsort(-explained_var, kind="stable")
    patterns, explained_var = patterns[order], explained_var[order]
    # Every definition is a fraction of the total variance; remove rounding
    # above 1 (1 + 7e-16 for ICA) for a component that explains all of it.
    explained_var = np.minimum(explained_var, 1.0)

    # ICA and eigenvector signs are arbitrary. Orient each pattern so its
    # largest-magnitude weight is positive: its members are then the neurons
    # with the largest positive weights. The activation strength does not
    # depend on the sign.
    largest = patterns[np.arange(n_found), np.argmax(np.abs(patterns), axis=1)]
    patterns = patterns * np.where(largest < 0, -1.0, 1.0)[:, None]

    # Activation strength of every pattern in every detection bin,
    # shape (n_time_bins, n_found); excluded neurons have weight 0.
    activations = _activation_strength(spike_counts_z.T, patterns)

    # Members among the retained neurons; shape (n_found, n_retained).
    member_masks = _member_masks(patterns, member_rule, member_threshold, n_neurons)
    full_patterns = np.zeros((n_found, n_neurons))
    full_patterns[:, retained] = patterns

    assembly_patterns = tuple(
        AssemblyPattern(
            weights=full_patterns[i],
            member_indices=retained[member_masks[i]].astype(np.int64),
            explained_variance_ratio=explained_var[i],
            unit_ids=labels,
            _unit_ids_generated=unit_ids_generated,
        )
        for i in range(n_found)
    )

    return AssemblyDetectionResult(
        patterns=assembly_patterns,
        activations=activations,
        method=method,
        n_significant=n_significant,
        eigenvalues=eigenvalues,
        threshold=threshold,
        unit_ids=labels,
        component_threshold=component_threshold,
        surrogate_max_eigenvalues=surrogate_max,
        _unit_ids_generated=unit_ids_generated,
    )


def _member_masks(
    patterns: NDArray[np.float64],
    rule: str,
    threshold: float,
    n_neurons: int,
) -> NDArray[np.bool_]:
    """Core members of each pattern by ``rule``.

    Parameters
    ----------
    patterns : ndarray, shape (n_patterns, n_retained)
        Unit-length weights of the varying neurons, each pattern oriented so
        its largest absolute weight is positive.
    rule : {"abs_z", "positive_z", "otsu", "inverse_sqrt_n"}
        Membership rule (see :func:`detect_assemblies`).
    threshold : float
        Z-score cutoff of the ``"abs_z"`` and ``"positive_z"`` rules.
    n_neurons : int
        Number of recorded neurons, the ``N`` of ``"inverse_sqrt_n"``.

    Returns
    -------
    ndarray of bool, shape (n_patterns, n_retained)
    """
    if rule == "abs_z":
        return np.asarray(stats.zscore(np.abs(patterns), axis=1) > threshold)
    if rule == "positive_z":
        return np.asarray((patterns > 0) & (stats.zscore(patterns, axis=1) > threshold))
    if rule == "inverse_sqrt_n":
        return np.asarray(patterns > 1.0 / np.sqrt(n_neurons))
    return np.array([_otsu_upper(np.abs(weights)) for weights in patterns]).reshape(
        patterns.shape
    )


def _otsu_upper(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """The upper group of Otsu's two-group split of ``values``.

    Tries every split of the sorted values into a lower and an upper group
    and keeps the one with the smallest within-group variance
    ``p_low var_low + p_high var_high`` (equivalently the largest variance
    between the groups), the first such split on ties.

    Parameters
    ----------
    values : ndarray, shape (n,)

    Returns
    -------
    ndarray of bool, shape (n,)
        True for values above the split.
    """
    ordered = np.sort(values)
    n = ordered.size
    if n < 2:
        return np.zeros(n, dtype=bool)
    size_low = np.arange(1, n)
    sums = np.cumsum(ordered)[:-1]
    total, total_squares = ordered.sum(), (ordered**2).sum()
    # n * within-group variance = sum of squares minus each group's n * mean^2.
    within = total_squares - sums**2 / size_low - (total - sums) ** 2 / (n - size_low)
    split = ordered[int(np.argmin(within))]
    return np.asarray(values > split)


def _check_component_threshold(
    component_threshold: str, percentile: float | None, n_shuffles: int
) -> None:
    """Refuse an unknown threshold rule or options that do not fit it.

    Parameters
    ----------
    component_threshold : str
        The threshold rule.
    percentile : float or None
        Percentile of the surrogate maxima; required for surrogates only.
    n_shuffles : int
        Number of surrogate matrices.
    """
    call = "detect_assemblies()"
    if component_threshold not in _COMPONENT_THRESHOLDS:
        raise ValueError(
            f"{call}: component_threshold must be one of "
            f"{_COMPONENT_THRESHOLDS}, got {component_threshold!r}."
        )
    surrogate = component_threshold in _SURROGATE_THRESHOLDS
    if surrogate and percentile is None:
        raise ValueError(
            _format_error(
                f"{call}: component_threshold={component_threshold!r} needs "
                "percentile=.",
                why="Why: the threshold is a percentile of the surrogates' "
                "largest eigenvalues, and Lopes-dos-Santos et al. (2013) give "
                "no single value (they used the 95th and the 99th; the 95th "
                "of circular shifts is close to the Marchenko-Pastur edge).",
                fix="pass percentile=95.0 or percentile=99.0",
            )
        )
    if not surrogate and percentile is not None:
        raise ValueError(
            _format_error(
                f"{call}: percentile= applies only to the surrogate "
                f"thresholds, not component_threshold={component_threshold!r}.",
                why="Why: the analytical thresholds have no surrogate "
                "distribution, so the percentile would be ignored.",
                fix="pass component_threshold='bin_shuffle' or "
                "'circular_shift', or drop percentile=",
            )
        )
    if percentile is not None and not 0.0 < float(percentile) < 100.0:
        raise ValueError(f"{call}: percentile must be in (0, 100), got {percentile}.")
    if isinstance(n_shuffles, bool) or int(n_shuffles) != n_shuffles or n_shuffles < 1:
        raise ValueError(
            f"{call}: n_shuffles must be a positive integer, got {n_shuffles!r}."
        )


def _surrogate_max_eigenvalues(
    spike_counts_z: NDArray[np.float64],
    kind: str,
    n_shuffles: int,
    generator: np.random.Generator,
) -> NDArray[np.float64]:
    """Largest correlation eigenvalue of each surrogate count matrix.

    Parameters
    ----------
    spike_counts_z : ndarray, shape (n_neurons, n_time_bins)
        Z-scored counts of the varying neurons.
    kind : {"bin_shuffle", "circular_shift"}
        Shuffle each neuron's time bins independently, or rotate each
        neuron's counts by its own offset drawn uniformly from
        ``0 .. n_time_bins - 1`` (as in the authors' MATLAB and Python code).
    n_shuffles : int
        Number of surrogate matrices.
    generator : numpy.random.Generator
        Source of the shuffles.

    Returns
    -------
    ndarray, shape (n_shuffles,)
        Largest eigenvalue of each surrogate's correlation matrix.
    """
    n_neurons, n_time_bins = spike_counts_z.shape
    # Shuffling or rotating a z-scored row keeps its mean 0 and variance 1,
    # so Z^T Z / B of the time-first surrogate is its correlation matrix.
    time_first = np.ascontiguousarray(spike_counts_z.T)
    columns = np.arange(n_neurons)
    steps = np.arange(n_time_bins)[:, None]
    maxima = np.empty(n_shuffles)
    for index in range(n_shuffles):
        if kind == "bin_shuffle":
            surrogate = generator.permuted(time_first, axis=0)
        else:
            offsets = np.asarray(generator.integers(0, n_time_bins, size=n_neurons))
            surrogate = time_first[(steps - offsets[None, :]) % n_time_bins, columns]
        correlation = surrogate.T @ surrogate / n_time_bins
        maxima[index] = linalg.eigvalsh(
            correlation, subset_by_index=[n_neurons - 1, n_neurons - 1]
        )[0]
    return maxima


def _detect_pca(
    spike_counts_z: NDArray[np.float64],
    n_components: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Detect assemblies using PCA via scipy SVD on the correlation matrix.

    Following Lopes-dos-Santos et al. (2013), we find eigenvectors of the
    correlation matrix C = (1/T) * Z @ Z.T where Z is the z-scored spike count
    matrix. These eigenvectors represent assembly patterns.

    Returns
    -------
    patterns : array, shape (n_live, n_neurons)
        Eigenvectors of the ``n_live <= n_components`` leading components
        with nonzero variance; components beyond the numerical rank are
        dropped.
    explained_variance_ratio : array, shape (n_live,)
    """
    # SVD: Z = U @ S @ Vh where Z is (n_neurons, n_time_bins). The correlation
    # matrix C = Z @ Z.T / T = U @ S^2 @ U.T / T has eigenvectors in U.
    u, s, _vh = linalg.svd(spike_counts_z, full_matrices=False)

    # Components beyond the numerical rank have arbitrary eigenvectors.
    rank_tol = np.finfo(np.float64).eps * max(spike_counts_z.shape) * s[0]
    n_live = int(np.sum(s[:n_components] > rank_tol))

    # Each column of u is a pattern; transpose to (n_live, n_neurons).
    patterns = u[:, :n_live].T

    # Eigenvalues of C are s^2 / T, and they sum to the total variance.
    explained_var = (s[:n_live] ** 2) / np.sum(s**2)

    return patterns, explained_var


def _detect_ica(
    spike_counts_z: NDArray[np.float64],
    n_components: int,
    random_state: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Detect assemblies using ICA via sklearn FastICA.

    Following Lopes-dos-Santos et al. (2013), we first reduce dimensionality
    with PCA to the significant subspace, then apply ICA to find statistically
    independent assembly patterns.

    Returns
    -------
    patterns : array, shape (n_live, n_neurons)
        One pattern per principal component with nonzero variance,
        ``n_live <= n_components``.
    explained_variance_ratio : array, shape (n_live,)
    """
    from sklearn.decomposition import FastICA

    # First reduce to n_components via PCA. PCA decorrelates scores but
    # leaves their variances equal to the eigenvalues; FastICA must whiten
    # these scores before its rotation. _detect_pca drops null PCs, so
    # whitening cannot amplify numerical noise or divide by zero.
    patterns_pca, _ = _detect_pca(spike_counts_z, n_components)
    n_live = patterns_pca.shape[0]
    scores_pca = patterns_pca @ spike_counts_z  # (n_live, n_time_bins)

    ica = FastICA(
        n_components=n_live,
        random_state=random_state,
        max_iter=1000,
        whiten="unit-variance",
    )
    ica.fit(scores_pca.T)

    # components_ includes the whitening transform as well as the ICA
    # rotation; composed with the projection it maps z-scored counts to the
    # sources. Scale each pattern to unit length (Lopes-dos-Santos et al.,
    # 2013), like the PCA eigenvectors, so weights are comparable across
    # methods; the pattern @ z-scored counts is then a scaled source.
    patterns = ica.components_ @ patterns_pca
    patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)

    # Unit-variance, uncorrelated sources contribute the squared norm of
    # their neuron-space mixing column to the reconstructed variance.
    # Individual PCA variances do not correspond to the rotated ICA sources.
    loadings = patterns_pca.T @ ica.mixing_
    total_var = np.sum(spike_counts_z**2) / spike_counts_z.shape[1]
    explained_var = np.sum(loadings**2, axis=0) / total_var

    return patterns, explained_var


def _detect_nmf(
    spike_counts: NDArray[np.float64],
    n_components: int,
    random_state: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Detect assemblies using NMF via sklearn.

    Note: Uses original spike counts (not z-scored) since NMF requires
    non-negative input.

    Parameters
    ----------
    spike_counts : array, shape (n_neurons, n_time_bins)
        Non-negative counts of the neurons that vary over time.

    Returns
    -------
    patterns : array, shape (n_live, n_neurons)
        Unit-length weights of the components with nonzero weights and
        activations, ``n_live <= n_components``.
    explained_variance_ratio : array, shape (n_live,)
    """
    from sklearn.decomposition import NMF

    nmf = NMF(
        n_components=n_components,
        random_state=random_state,
        max_iter=500,
        init="nndsvda",  # Better initialization
    )

    # NMF: X ≈ W @ H
    # W: (n_neurons, n_components) - basis vectors
    # H: (n_components, n_time_bins) - activations
    basis = nmf.fit_transform(spike_counts)
    activations = nmf.components_
    patterns = basis.T  # (n_components, n_neurons)

    # A component with all-zero weights or activations reconstructs nothing.
    live = (np.abs(patterns).sum(axis=1) > 0) & (np.abs(activations).sum(axis=1) > 0)
    patterns, activations = patterns[live], activations[live]

    # W @ H is unchanged when a pattern is scaled by c and its activation by
    # 1 / c. Give every pattern unit length, as for PCA and ICA.
    patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)

    # Explained variance of component k: the variance of the counts along
    # its unit pattern w_k, w_k' S w_k / trace(S) with S the counts'
    # covariance. This is the variance of the least-squares reconstruction of
    # the counts from that one direction, so it lies in [0, 1] and, unlike a
    # share of the total fit split by pattern norm, it does not depend on the
    # W/H scaling. For PCA eigenvectors the same definition gives the
    # eigenvalue fraction. Components can overlap, so the fractions need not
    # add up to the variance the whole factorization explains.
    centered = spike_counts - spike_counts.mean(axis=1, keepdims=True)
    explained_var = np.sum((patterns @ centered) ** 2, axis=1) / np.sum(centered**2)

    return patterns, explained_var


def _z_scores(
    spike_counts: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Z-score each neuron over time, with 0 for constant neurons.

    Parameters
    ----------
    spike_counts : ndarray, shape (n_time_bins, n_neurons)
        Finite counts of one period.

    Returns
    -------
    z : ndarray, shape (n_time_bins, n_neurons)
        Each column minus its mean, divided by its standard deviation, both
        over this period. A constant column has no deviation and is 0.
    constant : ndarray of bool, shape (n_neurons,)
        Columns with the same value in every time bin.
    """
    constant = np.ptp(spike_counts, axis=0) == 0
    std = np.where(constant, 1.0, spike_counts.std(axis=0))
    z = (spike_counts - spike_counts.mean(axis=0)) / std
    z[:, constant] = 0.0
    return z, constant


def _activation_strength(
    spike_counts_z: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Projector activation strength of each pattern in each time bin.

    Parameters
    ----------
    spike_counts_z : ndarray, shape (n_time_bins, n_neurons)
        Z-scored counts.
    weights : ndarray, shape (n_patterns, n_neurons)
        Pattern weights.

    Returns
    -------
    ndarray, shape (n_time_bins, n_patterns)
        ``R[t, k] = z(t)^T P_k z(t)`` with ``P_k = w_k w_k^T`` and its
        diagonal set to 0, computed as
        ``(w_k . z(t))^2 - sum_i w_ki^2 z_i(t)^2``.
    """
    projection = spike_counts_z @ weights.T
    activation: NDArray[np.float64] = projection**2 - spike_counts_z**2 @ (weights**2).T
    return activation


# =============================================================================
# Assembly Activation
# =============================================================================


def assembly_activation(
    spike_counts: NDArray[np.float64],
    pattern: AssemblyPattern,
    *,
    z_score_input: bool = True,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> NDArray[np.float64]:
    """
    Compute activation strength of an assembly over time.

    Measures, in each time bin, how strongly the pattern's neurons fire
    together: the projector activation strength of Lopes-dos-Santos et al.
    (2013) and Peyrache et al. (2009, 2010).

    Parameters
    ----------
    spike_counts : NDArray[np.float64], shape (n_time_bins, n_neurons)
        Population activity of the pattern's units, finite. Raw counts must
        be non-negative.
    pattern : AssemblyPattern
        Assembly pattern with neuron weights.
    z_score_input : bool, default=True
        If True, z-score each neuron's counts with this period's own mean and
        standard deviation, as in the published method. Set to False if
        ``spike_counts`` is already z-scored, for example against another
        period's mean and standard deviation.
    unit_ids : array-like, shape (n_neurons,), optional
        Label of the unit in each column of ``spike_counts``. The columns are
        then matched to the pattern's weights by label, in any order, and
        must hold exactly the pattern's units; the pattern needs real labels
        (``detect_assemblies(..., unit_ids=...)``). Without it, column ``i``
        is the pattern's unit ``i``, in detection order.

    Returns
    -------
    activation : NDArray[np.float64], shape (n_time_bins,)
        Activation strength R(t) in each time bin, in units of squared
        z-scores. Large positive values mark bins in which neurons with
        same-sign weights fire together above their means.

    Raises
    ------
    ValueError
        If spike_counts has the wrong number of neurons, holds NaN or
        infinite values (or negative raw counts), the pattern has no nonzero
        weight, or with ``unit_ids``, if the pattern's labels were generated
        or the labels of the two sides differ.

    Warns
    -----
    UserWarning
        If a neuron with nonzero weight has the same count in every bin of
        this period. Its z-score is undefined; it contributes 0, so it
        cannot take part in co-activation.

    Notes
    -----
    **Activation strength**:

    With the pattern's weights :math:`w` scaled to unit length and the
    z-scored counts :math:`z(t)`,

    .. math::

        R(t) = z(t)^T P z(t) = \\sum_{i \\neq j} w_i w_j z_i(t) z_j(t),
        \\qquad P = w w^T \\text{ with } P_{ii} = 0.

    Removing the diagonal removes each neuron's own squared deviation, so a
    burst of one member alone adds nothing; only pairs of neurons that
    deviate together raise R(t) (Peyrache et al., 2010). Flipping the sign of
    every weight leaves R(t) unchanged. Averaged over the period, R(t) is
    :math:`w^T C w - 1`, where :math:`C` is the period's correlation matrix:
    the pattern's co-activation beyond what independent neurons give.

    **Interpretation**:

    For independent neurons R(t) has mean 0 and, for Gaussian z-scores,
    standard deviation :math:`\\sqrt{2(1 - \\sum_i w_i^4)} \\le \\sqrt{2}`.
    Spike counts are skewed, so the tails are heavier than that. Published
    analyses call bins with R(t) above 5 assembly activations (van de Ven et
    al., 2016); this is a descriptive cutoff, not a p-value. Assess
    significance against a null built for your recording, for example
    activation strengths of patterns detected on time-shuffled counts.

    Each period is z-scored with its own mean and standard deviation, so a
    change in firing rate between periods does not by itself change R(t).
    Use :func:`assembly_reactivation` to compare a pattern's activation
    after a task with before it.

    Peyrache et al. (2009, 2010) call R(t) the "reactivation strength" and
    van de Ven et al. (2016) the "expression strength"; this package calls
    it the activation strength.

    Examples
    --------
    Units 0 and 1 form an assembly. A burst of unit 0 alone barely registers;
    the two firing together does:

    >>> import numpy as np
    >>> from neurospatial.reactivation import AssemblyPattern, assembly_activation
    >>> counts = np.random.default_rng(0).poisson(1.0, (200, 3)).astype(float)
    >>> counts[10, 0] += 6  # unit 0 alone
    >>> counts[20, :2] += 6  # units 0 and 1 together
    >>> pattern = AssemblyPattern(np.array([0.7, 0.7, 0.1]), np.array([0, 1]), 0.5)
    >>> activation = assembly_activation(counts, pattern)
    >>> print(f"alone {activation[10]:.1f}, together {activation[20]:.1f}")
    alone -0.3, together 33.5

    With labels, the columns of new counts can be in any order:

    >>> pattern = AssemblyPattern(
    ...     np.array([1.0, 0.5, 0.0]), np.array([0]), 1.0, unit_ids=[7, 8, 9]
    ... )
    >>> counts = np.random.default_rng(0).poisson(2.0, (50, 3)).astype(float)
    >>> by_label = assembly_activation(counts[:, ::-1], pattern, unit_ids=[9, 8, 7])
    >>> np.allclose(by_label, assembly_activation(counts, pattern))
    True

    >>> # Detect assemblies during behavior
    >>> result = detect_assemblies(counts_behavior)  # doctest: +SKIP
    >>> pattern = result.patterns[0]  # doctest: +SKIP
    >>>
    >>> # Compute activation during sleep
    >>> activation_sleep = assembly_activation(counts_sleep, pattern)  # doctest: +SKIP
    >>> print(f"Peak activation: {activation_sleep.max():.2f}")  # doctest: +SKIP

    See Also
    --------
    detect_assemblies : Detect assembly patterns
    assembly_reactivation : Compare activation after a task with before it

    References
    ----------
    Lopes-dos-Santos V, Ribeiro S, Tort ABL (2013). Detecting cell
        assemblies in large neuronal populations. J Neurosci Methods
        220(2):149-166.
    Peyrache A, Khamassi M, Benchenane K, Wiener SI, Battaglia FP (2009).
        Replay of rule-learning related neural patterns in the prefrontal
        cortex during sleep. Nat Neurosci 12(7):919-926.
    Peyrache A, Benchenane K, Khamassi M, Wiener SI, Battaglia FP (2010).
        Principal component analysis of ensemble recordings reveals cell
        assemblies at high temporal resolution. J Comput Neurosci
        29(1-2):309-325.
    van de Ven GM, Trouche S, McNamara CG, Allen K, Dupret D (2016).
        Hippocampal offline reactivation consolidates recently formed cell
        assembly patterns during sharp wave-ripples. Neuron 92(5):968-974.
    """
    call = "assembly_activation()"
    spike_counts = _validated_counts(spike_counts, call=call, nonnegative=z_score_input)
    spike_counts = _columns_by_label(spike_counts, unit_ids, pattern, call=call)
    _n_time_bins, n_neurons = spike_counts.shape

    if n_neurons != len(pattern.weights):
        raise ValueError(
            f"spike_counts has {n_neurons} neurons but pattern has "
            f"{len(pattern.weights)} weights. Must match."
        )
    weights = _unit_weights(pattern.weights, call=call)

    if z_score_input:
        spike_counts_z, constant = _z_scores(spike_counts)
        silent_members = constant & (weights != 0)
        if np.any(silent_members):
            warnings.warn(
                f"{call}: units {np.asarray(pattern.unit_ids)[silent_members].tolist()} "
                "have nonzero weight but the same count in every time bin of "
                "spike_counts, so they contribute 0 to the activation strength.",
                UserWarning,
                stacklevel=2,
            )
    else:
        spike_counts_z = spike_counts

    return _activation_strength(spike_counts_z, weights[None, :])[:, 0]


def pairwise_activation(
    spike_counts: NDArray[np.float64],
    template_correlations: NDArray[np.float64],
    *,
    z_score_input: bool = True,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    template_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> NDArray[np.float64]:
    """Co-firing in each time bin, weighted by every pair's template correlation.

    The reactivation strength of Tingley & Peyrache (2020) in its "simplest
    case": the z-scored population vector of each bin is projected on the
    template period's whole correlation matrix with its diagonal set to 0,
    instead of on one assembly pattern (:func:`assembly_activation`). The
    supplementary material calls it the "cell-pair coherogram". It tracks
    in time how much the co-firing of a period resembles the template's pair
    correlations, which explained variance summarizes over a whole period.

    Parameters
    ----------
    spike_counts : NDArray[np.float64], shape (n_time_bins, n_neurons)
        Population activity of the period to track (e.g. rest after the
        task), finite. Raw counts must be non-negative.
    template_correlations : NDArray[np.float64], shape (n_pairs,)
        Pair correlations of the template period (e.g. running), from
        :func:`pairwise_correlations`, in ``np.triu_indices(n_neurons, k=1)``
        order; ``n_pairs = n_neurons * (n_neurons - 1) / 2``. A NaN pair (a
        neuron constant in the template) gets weight 0.
    z_score_input : bool, default=True
        If True, z-score each neuron's counts with this period's own mean
        and standard deviation, as in the published method. Set to False if
        ``spike_counts`` is already z-scored.
    unit_ids, template_unit_ids : array-like, shape (n_neurons,), optional
        Labels of the columns of ``spike_counts`` and of the counts that
        ``template_correlations`` came from. Given together, the columns
        are matched to the template's units by label, in any order, and
        must hold the same units. Without them, column ``i`` is the
        template's unit ``i``.

    Returns
    -------
    activation : NDArray[np.float64], shape (n_time_bins,)
        :math:`R(t) = z(t)^T P z(t)`, in units of squared z-scores times
        correlation.

    Raises
    ------
    ValueError
        If ``spike_counts`` holds NaN, infinite (or negative raw) values,
        ``template_correlations`` does not hold one finite-or-NaN value in
        [-1, 1] per pair of the columns, or the labels are given alone or
        name different units.

    Warns
    -----
    UserWarning
        If some template pairs are NaN, or a neuron with a defined template
        correlation has the same count in every bin of ``spike_counts`` (its
        z-score is undefined; it contributes 0).

    Notes
    -----
    With :math:`C` the template's correlation matrix and :math:`P = C` with
    :math:`P_{ii} = 0`,

    .. math::

        R(t) = z(t)^T P z(t) = \\sum_{i \\neq j} C_{ij} z_i(t) z_j(t).

    Removing the diagonal removes each neuron's own squared deviation, so a
    neuron firing alone adds nothing (Tingley & Peyrache 2020: "To make the
    measure independent of single-neuron rate fluctuation, the diagonal of
    the matrix can be set at 0").

    **Time average.** A period's correlation matrix is the average of
    :math:`z(t) z(t)^T` over its bins, so the mean of R(t) over the bins
    of the z-scored period is

    .. math::

        \\overline{R} = \\sum_{i \\neq j} C^{template}_{ij} C^{period}_{ij}
        = 2 \\sum_{pairs} \\rho^{template}_{pair} \\rho^{period}_{pair},

    the inner product of the two periods' pair correlations: what the
    supplementary material calls "the variance during the sleep epoch
    explained by the correlation of the wake epoch". Unlike the cell-pair
    coherence (the Pearson correlation of the two pair vectors, whose
    square is EV without a control), it is neither centered nor
    normalized: it grows with the number of pairs and with the size of the
    correlations in both periods, so compare it between periods recorded
    with the same units, for example after the task minus before it.

    Every pair counts, weighted by its template correlation, including
    pairs whose template correlation is noise; the assembly projectors of
    :func:`assembly_activation` keep only the components above chance
    (Tingley & Peyrache 2020: using the correlation matrix "does not
    address the possibility that noisy correlations can potentially
    overshadow the actual signal").

    Examples
    --------
    Units 0, 1 and 2 fire together during the task, and twice during rest:

    >>> import numpy as np
    >>> from neurospatial.reactivation import (
    ...     pairwise_activation,
    ...     pairwise_correlations,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> task = rng.poisson(1.0, (2000, 6)).astype(float)
    >>> task[:, :3] += rng.poisson(1.0, (2000, 1))
    >>> rest = rng.poisson(1.0, (300, 6)).astype(float)
    >>> rest[[100, 200], :3] += 5
    >>> activation = pairwise_activation(rest, pairwise_correlations(task))
    >>> int(np.argmax(activation[:150])), int(150 + np.argmax(activation[150:]))
    (100, 200)

    The mean over the period is twice the sum of products of the two
    periods' pair correlations:

    >>> mean = 2 * np.sum(pairwise_correlations(task) * pairwise_correlations(rest))
    >>> bool(np.isclose(activation.mean(), mean))
    True

    See Also
    --------
    assembly_activation : The same projection on one assembly pattern
    explained_variance_reactivation : Whole-period comparison of pair correlations

    References
    ----------
    Tingley D, Peyrache A (2020). On the methods for reactivation and
        replay analysis. Phil Trans R Soc B 375:20190231 (and its
        electronic supplementary material).
    Peyrache A, Khamassi M, Benchenane K, Wiener SI, Battaglia FP (2009).
        Replay of rule-learning related neural patterns in the prefrontal
        cortex during sleep. Nat Neurosci 12(7):919-926.
    """
    call = "pairwise_activation()"
    spike_counts = _validated_counts(spike_counts, call=call, nonnegative=z_score_input)
    n_neurons = spike_counts.shape[1]
    template = np.asarray(template_correlations, dtype=np.float64)
    n_pairs = n_neurons * (n_neurons - 1) // 2
    if template.shape != (n_pairs,):
        raise ValueError(
            _format_error(
                f"{call}: template_correlations has shape {template.shape}, but "
                f"{n_neurons} neurons have {n_pairs} pairs.",
                fix="pass pairwise_correlations(template_counts) of the same "
                "units as spike_counts",
            )
        )
    if np.any(np.isinf(template)) or np.any(np.abs(template) > 1 + 1e-12):
        raise ValueError(
            f"{call}: template_correlations must be correlations in [-1, 1] "
            "or NaN; pass pairwise_correlations(template_counts)."
        )
    if (unit_ids is None) != (template_unit_ids is None):
        raise ValueError(
            f"{call}: pass unit_ids= and template_unit_ids= together, so the "
            "columns can be matched to the template's units by label."
        )
    if unit_ids is not None:
        labels = resolve_unit_ids(unit_ids, n_neurons, context=f"{call} for unit_ids")
        reference = resolve_unit_ids(
            template_unit_ids, n_neurons, context=f"{call} for template_unit_ids"
        )
        order, only_template, only_counts = _label_order(labels, reference)
        if only_template or only_counts:
            raise ValueError(
                f"{call}: spike_counts must hold the template's units; only in "
                f"the template: {only_template}, only in spike_counts: "
                f"{only_counts}."
            )
        spike_counts = spike_counts[:, order]

    undefined = np.isnan(template)
    if np.any(undefined):
        warnings.warn(
            f"{call}: {int(undefined.sum())} of {n_pairs} template pairs are NaN "
            "(a neuron constant in the template) and get weight 0.",
            UserWarning,
            stacklevel=2,
        )
    first, second = np.triu_indices(n_neurons, k=1)
    weights = np.where(undefined, 0.0, template)
    projector = np.zeros((n_neurons, n_neurons))
    projector[first, second] = weights
    projector[second, first] = weights

    if z_score_input:
        spike_counts_z, constant = _z_scores(spike_counts)
        silent = constant & np.any(projector != 0, axis=0)
        if np.any(silent):
            warnings.warn(
                f"{call}: neuron columns {np.flatnonzero(silent).tolist()} have the "
                "same count in every time bin of spike_counts, so they "
                "contribute 0 to the activation strength.",
                UserWarning,
                stacklevel=2,
            )
    else:
        spike_counts_z = spike_counts
    activation: NDArray[np.float64] = np.sum(
        (spike_counts_z @ projector) * spike_counts_z, axis=1
    )
    return activation


def _unit_weights(weights: NDArray[np.float64], *, call: str) -> NDArray[np.float64]:
    """Return the pattern's weights scaled to unit length.

    Parameters
    ----------
    weights : ndarray, shape (n_neurons,)
        Pattern weights.
    call : str
        Function name for the error message.

    Returns
    -------
    ndarray, shape (n_neurons,)
        ``weights / np.linalg.norm(weights)``.
    """
    norm = np.linalg.norm(weights)
    if norm == 0:
        raise ValueError(
            _format_error(
                f"{call}: the pattern's weights are all 0.",
                why="Why: the activation strength measures co-activation of the "
                "pattern's weighted neurons, and a pattern without weights has "
                "none.",
                fix="pass a pattern from detect_assemblies(), or an "
                "AssemblyPattern with at least one nonzero weight",
            )
        )
    unit: NDArray[np.float64] = weights / norm
    return unit


def _label_order(
    labels: NDArray[Any], reference: NDArray[Any]
) -> tuple[NDArray[np.intp], list[Any], list[Any]]:
    """Find where each reference label sits among ``labels``.

    Parameters
    ----------
    labels : ndarray, shape (n_units,)
        Distinct labels of the units to reorder.
    reference : ndarray, shape (n_reference_units,)
        Distinct labels in the target order.

    Returns
    -------
    order : ndarray of intp, shape (n_reference_units,)
        ``labels[order]`` equals ``reference``. Empty when the label sets
        differ.
    only_reference, only_labels : list
        Labels found in only one of the two arrays, in their own order.
    """
    position_of = {label: position for position, label in enumerate(labels.tolist())}
    reference_labels = reference.tolist()
    reference_set = set(reference_labels)
    only_reference = [label for label in reference_labels if label not in position_of]
    only_labels = [label for label in labels.tolist() if label not in reference_set]
    if only_reference or only_labels:
        return np.empty(0, dtype=np.intp), only_reference, only_labels
    order = np.array([position_of[label] for label in reference_labels], dtype=np.intp)
    return order, only_reference, only_labels


def _columns_by_label(
    spike_counts: NDArray[np.float64],
    unit_ids: NDArray[Any] | Sequence[Any] | None,
    pattern: AssemblyPattern,
    *,
    call: str,
    argument: str = "spike_counts",
    labels_argument: str = "unit_ids",
) -> NDArray[np.float64]:
    """Reorder count columns into the pattern's unit order, by label.

    Parameters
    ----------
    spike_counts : ndarray, shape (n_time_bins, n_neurons)
        Counts of new data, one unit per column.
    unit_ids : array-like or None
        Label of each column. ``None`` keeps positional pairing and returns
        ``spike_counts`` unchanged.
    pattern : AssemblyPattern
        Pattern whose ``unit_ids`` set the target order.
    call, argument, labels_argument : str
        Function, counts-argument and labels-argument names for error
        messages.

    Returns
    -------
    ndarray, shape (n_time_bins, n_neurons)
        Columns in ``pattern.unit_ids`` order.
    """
    if unit_ids is None:
        return spike_counts
    if spike_counts.ndim != 2:
        raise ValueError(
            f"{argument} must be 2D (n_time_bins, n_neurons), "
            f"got shape {spike_counts.shape}"
        )
    labels = resolve_unit_ids(
        unit_ids, spike_counts.shape[1], context=f"{call} for {argument}"
    )
    if pattern._unit_ids_generated:
        raise ValueError(
            _format_error(
                f"{call} got {labels_argument}= for {argument}, but the pattern "
                "has no unit_ids; its labels are column positions generated by "
                "detect_assemblies().",
                why="Why: columns are matched to the pattern's weights by label, "
                "and a column position does not identify the same cell in two "
                "count matrices.",
                fix="pass the sorted-unit labels as unit_ids= to "
                f"detect_assemblies() as well, or drop {labels_argument}= here "
                "to pair columns by position",
            )
        )
    order, only_pattern, only_counts = _label_order(
        labels, np.asarray(pattern.unit_ids)
    )
    if only_pattern or only_counts:
        raise ValueError(
            _format_error(
                f"{call}: {argument} must hold the same unit_ids as the pattern; "
                f"only in the pattern: {only_pattern}, only in {argument}: "
                f"{only_counts}.",
                why="Why: each weight of the pattern belongs to one cell, so "
                "every pattern unit needs its counts, and a column without a "
                "weight has no place in the projection.",
                fix=f"keep only the pattern's units, e.g. {argument}[:, keep] and "
                f"{labels_argument}[keep] with keep = np.isin({labels_argument}, "
                "pattern.unit_ids); if pattern units were not recorded, detect "
                "the assemblies again on the units recorded in both periods",
            )
        )
    reordered: NDArray[np.float64] = spike_counts[:, order]
    return reordered


# =============================================================================
# Pairwise Correlations and Explained Variance
# =============================================================================


def pairwise_correlations(
    spike_counts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute pairwise correlations between neurons.

    Parameters
    ----------
    spike_counts : NDArray[np.float64], shape (n_time_bins, n_neurons)
        Binned spike counts, finite and non-negative.

    Returns
    -------
    correlations : NDArray[np.float64], shape (n_pairs,)
        Pearson correlation of each pair of neurons over time bins, the upper
        triangle of the correlation matrix, flattened.
        n_pairs = n_neurons * (n_neurons - 1) / 2. A pair with a neuron
        whose count is constant over the period (for example silent) has no
        correlation and is NaN.

    Raises
    ------
    ValueError
        If ``spike_counts`` is not 2-D or holds NaN, infinite or negative
        values.

    Warns
    -----
    UserWarning
        If some neurons are constant over the period, naming their columns.

    Notes
    -----
    Returns only the upper triangle (excluding diagonal) to avoid
    redundancy. Use `numpy.triu_indices` to reconstruct the full matrix.

    The pairs follow the columns of ``spike_counts``: pair ``k`` is the pair
    of columns ``np.triu_indices(n_neurons, k=1)`` gives at ``k``. The
    correlations of two periods therefore describe the same pairs of cells
    only when both count matrices have the same units in the same column
    order. To compare periods whose columns are ordered differently, pass
    each period's column labels to :func:`explained_variance_reactivation`
    (``template_unit_ids=``, ``match_unit_ids=``, ``control_unit_ids=``),
    which pairs the correlations by cell.

    A constant neuron's pairs stay NaN rather than 0: a correlation of 0
    would claim the neuron fires independently of the others, while a
    neuron that never changes its count gives no evidence either way.
    :func:`explained_variance_reactivation` drops NaN pairs and reports the
    number of pairs it used.

    Every pair is returned, including pairs of cells on the same tetrode.
    Select pairs in :func:`explained_variance_reactivation` with
    ``unit_groups=`` (pairs from different tetrodes or regions) or
    ``pair_mask=``, so the pair vector keeps its ``np.triu_indices`` order.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reactivation import pairwise_correlations
    >>> counts = np.random.default_rng(0).poisson(2.0, (500, 4)).astype(float)
    >>> unit_ids = np.array([12, 15, 31, 40])  # the label of each column
    >>> corr = pairwise_correlations(counts)
    >>> corr.shape  # 4 * 3 / 2 pairs
    (6,)
    >>> first, second = np.triu_indices(4, k=1)
    >>> [(int(unit_ids[i]), int(unit_ids[j])) for i, j in zip(first[:3], second[:3])]
    [(12, 15), (12, 31), (12, 40)]

    Pass the same labels on when comparing periods:

    >>> ev = explained_variance_reactivation(  # doctest: +SKIP
    ...     corr,
    ...     pairwise_correlations(counts_sleep),
    ...     template_unit_ids=unit_ids,
    ...     match_unit_ids=sleep_unit_ids,
    ... )
    """
    spike_counts = _validated_counts(spike_counts, call="pairwise_correlations()")
    n_neurons = spike_counts.shape[1]

    constant = np.ptp(spike_counts, axis=0) == 0
    if np.any(constant):
        warnings.warn(
            f"pairwise_correlations(): neuron columns "
            f"{np.flatnonzero(constant).tolist()} have the same count in every "
            "time bin, so their correlations are undefined and returned as NaN. "
            "explained_variance_reactivation() drops those pairs; to keep the "
            "same pairs in every period, select units that fire in all periods.",
            UserWarning,
            stacklevel=2,
        )
    corr_matrix = np.full((n_neurons, n_neurons), np.nan)
    varying = np.flatnonzero(~constant)
    if varying.size >= 2:
        corr_matrix[np.ix_(varying, varying)] = np.corrcoef(
            spike_counts[:, varying], rowvar=False
        )

    # Extract upper triangle (excluding diagonal)
    correlations: NDArray[np.float64] = corr_matrix[np.triu_indices(n_neurons, k=1)]
    return correlations


def assembly_reactivation(
    control_counts: NDArray[np.float64],
    match_counts: NDArray[np.float64],
    pattern: AssemblyPattern,
    *,
    percentile: float = 99.0,
    control_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    match_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> AssemblyReactivationResult:
    """
    Compare a pattern's activation strength after a task with before it.

    Computes the activation strength R(t) of the pattern (as in
    :func:`assembly_activation`) in a control period, such as rest before
    the task, and in a match period, such as rest after it, and returns the
    two published comparisons:

    - ``mean_activation_difference``: mean R(t) in the match period minus
      mean R(t) in the control period. van de Ven et al. (2016) define a
      pattern's reactivation strength as "its average expression strength
      during the rest after minus its average expression strength during the
      rest before"; Peyrache et al. (2009, Fig. 2c) compare the average
      R(t) of post- and pre-task sleep the same way.
    - ``fraction_above_control_threshold``: the share of match bins whose
      R(t) exceeds the ``percentile``-th percentile of the control bins'
      R(t) (Peyrache et al., 2010: "4.5% of the bins from POST exceeded the
      99th percentile of the PRE distribution", against an expected 1%).

    Parameters
    ----------
    control_counts : NDArray[np.float64], shape (n_control_bins, n_neurons)
        Spike counts in the control period (e.g. rest before the task),
        finite and non-negative.
    match_counts : NDArray[np.float64], shape (n_match_bins, n_neurons)
        Spike counts in the match period (e.g. rest after the task), finite
        and non-negative.
    pattern : AssemblyPattern
        Assembly pattern to measure, usually detected in the task (template)
        period with :func:`detect_assemblies`.
    percentile : float, default=99.0
        Percentile of the control period's R(t) used as the threshold for
        ``fraction_above_control_threshold``, in (0, 100). Peyrache et al.
        (2010) used the 99th.
    control_unit_ids : array-like, shape (n_neurons,), optional
        Label of the unit in each column of ``control_counts``. Pass it
        together with ``match_unit_ids``: the columns of each period are then
        matched to the pattern's weights by label, as in
        :func:`assembly_activation`, so the two periods may list their units
        in different orders. Each must hold exactly the pattern's units, and
        the pattern needs real labels (``detect_assemblies(..., unit_ids=...)``).
        Without either, column ``i`` of both matrices is the pattern's unit
        ``i``.
    match_unit_ids : array-like, shape (n_neurons,), optional
        Label of the unit in each column of ``match_counts``. When both
        periods were binned from one list of spike trains, pass the same
        labels to both arguments.

    Returns
    -------
    AssemblyReactivationResult
        Mean activation strength in each period, their difference, the
        control threshold and the share of match bins above it. All values
        are NaN when every neuron with nonzero weight is constant in a
        period.

    Raises
    ------
    ValueError
        If the counts and the pattern have different numbers of neurons,
        either period holds NaN, infinite or negative values, the pattern
        has no nonzero weight, ``percentile`` is outside (0, 100), or with
        labels, if only one of ``control_unit_ids`` and ``match_unit_ids`` is
        given, the pattern's labels were generated, or a period's labels
        differ from the pattern's.

    Warns
    -----
    UserWarning
        If neurons with nonzero weight have the same count in every bin of
        either period (they are excluded from both periods, named in the
        warning).

    Notes
    -----
    **Terms.** Peyrache et al. (2009, 2010) call R(t) itself the
    "reactivation strength"; van de Ven et al. (2016) call R(t) the
    "expression strength" and the after-minus-before difference the
    "reactivation strength". This package calls R(t) the activation strength
    and this difference ``mean_activation_difference``, so neither name is
    ambiguous.

    **Z-scoring.** Each period is z-scored with its own per-neuron mean and
    standard deviation, as in both papers. The mean of R(t) over a period is
    then :math:`w^T C w - 1` for unit-length weights :math:`w` and the
    period's correlation matrix :math:`C` (Peyrache et al. 2010: the average
    reactivation strength "is equal to γ − 1"), so the difference is
    :math:`w^T (C_{match} - C_{control}) w`, and scaling a period's counts
    does not change it. A neuron with the same count in every bin of a
    period has no z-score there. It is excluded from both periods, so both
    means use the same neurons, and the remaining weights are rescaled to
    unit length.

    **Smoothing.** van de Ven et al. (2016) computed R(t) on spike trains
    convolved with a Gaussian kernel of standard deviation 25 ms / sqrt(12)
    and then z-scored (Supplemental Experimental Procedures); this function
    uses the counts you pass, so their binned version is reproduced exactly
    only for binned counts.

    **Noise floor.** Without co-firing, a period's mean R(t) is 0 up to
    sampling noise of standard deviation about :math:`\\sqrt{2 / n}` for
    ``n`` independent Gaussian bins; spike counts have heavier tails.
    Firing-rate changes matter when they change the correlations:
    independent spikes added to the members dilute their correlation. Neither
    statistic is a p-value: van de Ven et al. tested the differences of a
    set of patterns against 0 with a one-sample t-test, and the share of
    patterns with a positive difference against 50% with a z-test for
    proportions; Peyrache et al. (2010) built a null by permuting cell
    identities within each time bin.

    For the correlation structure of the whole population rather than one
    pattern, use :func:`explained_variance_reactivation` with pairwise
    correlations.

    Examples
    --------
    Units 7 and 8 burst together after the task but not before it; the
    post-task counts list the units in another order:

    >>> import numpy as np
    >>> from neurospatial.reactivation import AssemblyPattern, assembly_reactivation
    >>> pattern = AssemblyPattern(
    ...     np.array([0.7, 0.7, 0.1]), np.array([0, 1]), 0.5, unit_ids=[7, 8, 9]
    ... )
    >>> rng = np.random.default_rng(0)
    >>> pre = rng.poisson(1.0, (2000, 3)).astype(float)
    >>> post = rng.poisson(1.0, (2000, 3)).astype(float)
    >>> post[:, 1:] += 2 * rng.poisson(0.3, (2000, 1))  # columns: 9, 7, 8
    >>> result = assembly_reactivation(
    ...     pre, post, pattern, control_unit_ids=[7, 8, 9], match_unit_ids=[9, 7, 8]
    ... )
    >>> bool(result.mean_activation_difference > 0.3)
    True
    >>> bool(result.fraction_above_control_threshold > 0.01)
    True

    References
    ----------
    Peyrache A, Khamassi M, Benchenane K, Wiener SI, Battaglia FP (2009).
        Replay of rule-learning related neural patterns in the prefrontal
        cortex during sleep. Nat Neurosci 12(7):919-926.
    Peyrache A, Benchenane K, Khamassi M, Wiener SI, Battaglia FP (2010).
        Principal component analysis of ensemble recordings reveals cell
        assemblies at high temporal resolution. J Comput Neurosci
        29(1-2):309-325.
    van de Ven GM, Trouche S, McNamara CG, Allen K, Dupret D (2016).
        Hippocampal offline reactivation consolidates recently formed cell
        assembly patterns during sharp wave-ripples. Neuron 92(5):968-974.
    """
    call = "assembly_reactivation()"
    if not 0.0 < percentile < 100.0:
        raise ValueError(
            _format_error(
                f"{call}: percentile must be in (0, 100), got {percentile}.",
                why="Why: it is the percentile of the control period's "
                "activation strength used as the threshold for the match "
                "period.",
                fix="pass percentile=99.0 (Peyrache et al. 2010) or another "
                "value between 0 and 100",
            )
        )
    if (control_unit_ids is None) != (match_unit_ids is None):
        given, missing = (
            ("control_unit_ids", "match_unit_ids")
            if match_unit_ids is None
            else ("match_unit_ids", "control_unit_ids")
        )
        raise ValueError(
            _format_error(
                f"{call} got {given}= but not {missing}=.",
                why="Why: one period's columns would be matched to the "
                "pattern's weights by label and the other's by position, so "
                "a period whose units are listed in another order would "
                "silently pair weights with the wrong cells.",
                fix="pass the column labels of both count matrices "
                "(control_unit_ids= and match_unit_ids=; the same labels when "
                "both were binned from one list of spike trains), or neither "
                "to pair columns by position",
            )
        )
    control_counts = _columns_by_label(
        _validated_counts(control_counts, call=call, argument="control_counts"),
        control_unit_ids,
        pattern,
        call=call,
        argument="control_counts",
        labels_argument="control_unit_ids",
    )
    match_counts = _columns_by_label(
        _validated_counts(match_counts, call=call, argument="match_counts"),
        match_unit_ids,
        pattern,
        call=call,
        argument="match_counts",
        labels_argument="match_unit_ids",
    )

    n_control_neurons = control_counts.shape[1]
    n_match_neurons = match_counts.shape[1]
    n_weights = len(pattern.weights)
    if n_control_neurons != n_weights or n_match_neurons != n_weights:
        raise ValueError(
            f"{call}: control_counts has {n_control_neurons} neurons and "
            f"match_counts has {n_match_neurons} neurons but pattern has "
            f"{n_weights} weights. All must match."
        )

    weights = _unit_weights(pattern.weights, call=call)
    control_z, control_constant = _z_scores(control_counts)
    match_z, match_constant = _z_scores(match_counts)
    n_control_bins, n_match_bins = control_counts.shape[0], match_counts.shape[0]

    # A neuron constant in either period has no z-score there; drop it from
    # both so the two means use the same neurons.
    excluded = (control_constant | match_constant) & (weights != 0)
    if np.any(excluded):
        names = np.asarray(pattern.unit_ids)
        warnings.warn(
            f"{call}: units {names[excluded].tolist()} have nonzero weight but "
            "the same count in every time bin of "
            + " and ".join(
                period
                for period, constant in (
                    ("control_counts", control_constant & excluded),
                    ("match_counts", match_constant & excluded),
                )
                if constant.any()
            )
            + ", so they have no z-score there. They are excluded from both "
            "periods.",
            UserWarning,
            stacklevel=2,
        )
        weights = np.where(excluded, 0.0, weights)
        norm = np.linalg.norm(weights)
        if norm == 0:
            warnings.warn(
                f"{call}: every unit with nonzero weight is excluded, so the "
                "comparison is undefined (NaN).",
                UserWarning,
                stacklevel=2,
            )
            return AssemblyReactivationResult(
                mean_activation_difference=np.nan,
                control_mean_activation=np.nan,
                match_mean_activation=np.nan,
                control_threshold=np.nan,
                fraction_above_control_threshold=np.nan,
                percentile=float(percentile),
                n_control_bins=n_control_bins,
                n_match_bins=n_match_bins,
            )
        weights = weights / norm

    control_activation = _activation_strength(control_z, weights[None])[:, 0]
    match_activation = _activation_strength(match_z, weights[None])[:, 0]
    control_mean = float(np.mean(control_activation))
    match_mean = float(np.mean(match_activation))
    threshold = float(np.percentile(control_activation, percentile))
    return AssemblyReactivationResult(
        mean_activation_difference=match_mean - control_mean,
        control_mean_activation=control_mean,
        match_mean_activation=match_mean,
        control_threshold=threshold,
        fraction_above_control_threshold=float(np.mean(match_activation > threshold)),
        percentile=float(percentile),
        n_control_bins=n_control_bins,
        n_match_bins=n_match_bins,
    )


def explained_variance_reactivation(
    template_correlations: NDArray[np.float64],
    match_correlations: NDArray[np.float64],
    *,
    control_correlations: NDArray[np.float64]
    | Sequence[NDArray[np.float64]]
    | None = None,
    unit_groups: NDArray[Any] | Sequence[Any] | None = None,
    pair_mask: NDArray[np.bool_] | Sequence[bool] | None = None,
    template_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    match_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    control_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> ExplainedVarianceResult:
    """
    Compute explained variance for pairwise correlation reactivation.

    Measures how well pairwise correlations during 'match' period
    are predicted by correlations from 'template' period, beyond what the
    correlations of one or more control periods already predict (Kudrimoti
    et al., 1999).

    Parameters
    ----------
    template_correlations : NDArray[np.float64], shape (n_pairs,)
        Pairwise correlations during template (e.g., behavior).
        From `pairwise_correlations()`. Without labels, pairs are matched by
        position, so every period's counts must have the same units in the
        same column order.
    match_correlations : NDArray[np.float64], shape (n_pairs,)
        Pairwise correlations during match (e.g., sleep/rest).
    control_correlations : array-like, shape (n_pairs,), or sequence of them, optional
        Correlations of the control period (e.g., pre-behavior rest), or a
        sequence of several control periods' correlations. EV is then the
        squared partial correlation of template and match controlling for
        all of them at once (Kudrimoti et al. 1999, experiment 2, partialled
        out pre-task sleep and the other half of the track).
    unit_groups : array-like, shape (n_neurons,), optional
        Group of each unit, such as its tetrode, shank or brain region, in
        the column order of the template counts (``template_unit_ids``
        order when labels are given). Only pairs of units from different
        groups are used: Kudrimoti et al. (1999) and Tatsuno et al. (2006)
        used pairs on different tetrodes, since cells on one tetrode can
        share spike-sorting errors; Lansink et al. (2009) and Tang et al.
        (2017) used only cross-region pairs.
    pair_mask : array-like of bool, shape (n_pairs,), optional
        ``True`` for the pairs to use, in the template's pair order (the
        order of ``np.triu_indices(n_neurons, k=1)``), e.g. pairs with
        waveform similarity below 0.8 (Giri et al., 2024). Combined with
        ``unit_groups`` by logical AND.
    template_unit_ids : array-like, shape (n_neurons,), optional
        Label of the unit in each column of the counts that
        ``template_correlations`` came from, so pair ``k`` is the cells
        ``template_unit_ids[i]`` and ``template_unit_ids[j]`` with
        ``i, j = np.triu_indices(n_neurons, k=1)`` at ``k``. Labels match
        the pairs of each period by cell instead of by position: the match
        and control pairs are put in the template's order before the
        comparison, so each period may list its units in its own order. When
        labels are given, every supplied period needs them, and all periods
        must hold the same units.
    match_unit_ids : array-like, shape (n_neurons,), optional
        Column labels of the counts that ``match_correlations`` came from.
        Required with ``template_unit_ids``.
    control_unit_ids : array-like, shape (n_neurons,), or sequence of them, optional
        Column labels of the counts that ``control_correlations`` came from;
        with several controls, one label array per control. Required with
        ``template_unit_ids`` when ``control_correlations`` is given, and
        refused without it.

    Returns
    -------
    ExplainedVarianceResult
        EV, reversed EV, partial correlation, its p-value, the number of
        pairs used and the number of controls.

    Raises
    ------
    ValueError
        If correlation arrays have different lengths, ``pair_mask`` is not a
        boolean array with one entry per pair, or ``unit_groups`` does not
        hold one group per unit. With labels: if a supplied period has no
        labels, a period's labels are not one distinct label per unit of its
        pair vector, or the periods hold different units.

    Warns
    -----
    UserWarning
        If fewer than 3 pairs remain, or EV is undefined (a correlation
        vector is constant, or the controls explain the template or match
        correlations completely).

    Notes
    -----
    **Explained Variance (EV)**:

    With a control, EV is the squared *partial* correlation between template
    and match correlations, controlling for the control (pre-behavior)
    baseline (Kudrimoti et al., 1999, p. 4093):

    .. math::

        EV = r_{tm \\mid c}^2
           = \\left(\\frac{r_{tm} - r_{tc} \\cdot r_{mc}}
                    {\\sqrt{(1-r_{tc}^2)(1-r_{mc}^2)}}\\right)^2

    where t=template, m=match, c=control and each ``r`` is the Pearson
    correlation of two periods' pair-correlation vectors. With several
    controls, :math:`r_{tm \\mid c_1 \\ldots c_k}` is the correlation of the
    template and match vectors after regressing both on all controls. Without
    a control, EV is the plain squared correlation :math:`r_{tm}^2`
    (Kudrimoti's EV for the pre-task period, and the co-firing similarity of
    Gridchyn et al. 2020 without the square).

    Tatsuno et al. (2006) and Giri et al. (2019, 2024) used several control
    blocks differently: they computed one EV per control block and averaged
    the EVs, :math:`\\frac{1}{K} \\sum_k r_{tm \\mid c_k}^2`. Compute that
    as the mean of one call per control block.

    **Reversed EV (REV)**:

    REV was introduced by Pennartz et al. (2004) as a control measure; it
    swaps the pre- and post-behavior periods: the template's correlation
    with the control, partialled for the match:

    .. math::

        REV = r_{tc \\mid m}^2

    Kudrimoti et al. (1999) has no REV. With one control, EV exceeding REV
    indicates a larger template-related effect after behavior than before
    it. Without a control there is nothing to swap, and with several
    controls no swap is defined, so REV is NaN.

    **p-value**:

    ``p_value`` is the two-sided t test of the partial correlation,
    :math:`t = r \\sqrt{df / (1 - r^2)}` with :math:`df = n_{pairs} - 2 - k`
    for ``k`` controls (the plain Pearson test without a control).
    Kudrimoti et al. (1999) report per-session p-values of the partial
    correlation computed with SPSS without naming the test; this standard
    partial-correlation t test is inferred, not stated. The test treats
    every pair as an independent observation, but pairs share cells (``n``
    cells give ``n (n - 1) / 2`` pairs), so the p-value is anticonservative.
    Compare EV with REV across sessions (Pennartz et al. 2004 used a
    Wilcoxon signed-rank test) or build a null for your recording before
    claiming reactivation.

    **Split-half check**:

    Kudrimoti et al. (1999, p. 4094) used one half of the template period as
    the control for the other half and found EV of about 0. This check is
    not 0 in expectation: if both halves and the match period measure the
    same pair-correlation structure with independent noise, so that each two
    of the three vectors correlate with :math:`\\rho`, the partial correlation
    is :math:`\\rho / (1 + \\rho)` (up to 1/2), and it is near 0 only when the
    correlations are mostly noise.

    **Pair selection**:

    ``n_pairs`` counts the pairs left after ``unit_groups``, ``pair_mask``
    and dropping pairs that are NaN in any period (pairs with a neuron
    constant in that period).

    Examples
    --------
    Units 3, 5 and 8 fire together during behavior and sleep. The sleep
    counts list the units in another order; their labels pair each sleep
    correlation with the same two cells as in behavior:

    >>> import numpy as np
    >>> from neurospatial.reactivation import (
    ...     explained_variance_reactivation,
    ...     pairwise_correlations,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> behavior = rng.poisson(1.0, (2000, 10)).astype(float)
    >>> behavior[:, [3, 5, 8]] += rng.poisson(0.5, (2000, 1))
    >>> sleep = rng.poisson(1.0, (2000, 10)).astype(float)
    >>> sleep[:, [3, 5, 8]] += rng.poisson(0.5, (2000, 1))
    >>> labels = np.arange(10)
    >>> order = rng.permutation(10)  # the sleep recording's unit order
    >>> result = explained_variance_reactivation(
    ...     pairwise_correlations(behavior),
    ...     pairwise_correlations(sleep[:, order]),
    ...     template_unit_ids=labels,
    ...     match_unit_ids=labels[order],
    ... )
    >>> bool(result.explained_variance > 0.5)
    True

    Only pairs of units on different tetrodes (two units per tetrode here):

    >>> tetrode = np.repeat(np.arange(5), 2)
    >>> different = explained_variance_reactivation(
    ...     pairwise_correlations(behavior),
    ...     pairwise_correlations(sleep),
    ...     unit_groups=tetrode,
    ... )
    >>> different.n_pairs  # 45 pairs minus the 5 within a tetrode
    40

    With a pre-behavior baseline control:

    >>> result = explained_variance_reactivation(  # doctest: +SKIP
    ...     corr_behavior,
    ...     corr_sleep,
    ...     control_correlations=corr_pre,
    ...     template_unit_ids=behavior_labels,
    ...     match_unit_ids=sleep_labels,
    ...     control_unit_ids=pre_labels,
    ... )
    >>> print(f"EV = {result.explained_variance:.3f}")  # doctest: +SKIP
    >>> print(f"REV = {result.reversed_explained_variance:.3f}")  # doctest: +SKIP

    References
    ----------
    Kudrimoti HS, Barnes CA, McNaughton BL (1999). Reactivation of
        hippocampal cell assemblies: effects of behavioral state, experience,
        and EEG dynamics. J Neurosci 19(10):4090-4101.
    Pennartz CMA, Lee E, Verheul J, Lipa P, Barnes CA, McNaughton BL (2004).
        The ventral striatum in off-line processing: ensemble reactivation
        during sleep and modulation by hippocampal ripples. J Neurosci
        24(29):6446-6456.
    Tatsuno M, Lipa P, McNaughton BL (2006). Methodological considerations
        on the use of template matching to study long-lasting memory trace
        replay. J Neurosci 26(42):10727-10742.
    Lansink CS, Goltstein PM, Lankelma JV, McNaughton BL, Pennartz CMA
        (2009). Hippocampus leads ventral striatum in replay of place-reward
        information. PLoS Biol 7(8):e1000173.
    Giri B, Miyawaki H, Mizuseki K, Cheng S, Diba K (2019). Hippocampal
        reactivation extends for several hours following novel experience.
        J Neurosci 39(5):866-875.
    """
    call = "explained_variance_reactivation()"
    controls = _control_list(control_correlations)
    n_controls = len(controls)
    labels = {
        "template_unit_ids": template_unit_ids,
        "match_unit_ids": match_unit_ids,
        "control_unit_ids": control_unit_ids,
    }
    if any(value is not None for value in labels.values()):
        template_correlations, match_correlations, controls = _pairs_in_template_order(
            template_correlations, match_correlations, controls, labels
        )

    template_correlations = np.asarray(template_correlations, dtype=np.float64)
    match_correlations = np.asarray(match_correlations, dtype=np.float64)

    if len(template_correlations) != len(match_correlations):
        raise ValueError(
            f"template and match correlations must have same length. "
            f"Got {len(template_correlations)} and {len(match_correlations)}."
        )

    n_pairs = len(template_correlations)
    _check_correlation_values(template_correlations, "template_correlations")
    _check_correlation_values(match_correlations, "match_correlations")

    if n_pairs < 3:
        raise ValueError(f"Need at least 3 pairs for EV calculation, got {n_pairs}")

    valid_mask = _selected_pairs(n_pairs, unit_groups, pair_mask, call=call)
    valid_mask &= np.isfinite(template_correlations) & np.isfinite(match_correlations)
    for index, control in enumerate(controls):
        name = "control_correlations" + (f"[{index}]" if n_controls > 1 else "")
        if len(control) != n_pairs:
            raise ValueError(
                f"control correlations must have same length as template/match. "
                f"Got {len(control)} for {name}, expected {n_pairs}."
            )
        _check_correlation_values(control, name)
        valid_mask &= np.isfinite(control)

    n_valid = int(np.sum(valid_mask))
    if n_valid < 3:
        warnings.warn(
            f"Only {n_valid} valid pairs after pair selection and removing NaN. "
            "Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )
        # With fewer than 3 valid pairs the correlation is undefined: np.corrcoef
        # returns NaN, and coercing that to 0.0 would masquerade as a confident
        # "no reactivation". Return NaN statistics so an undefined result reads
        # as undefined.
        return ExplainedVarianceResult(
            explained_variance=np.nan,
            reversed_explained_variance=np.nan,
            partial_correlation=np.nan,
            semipartial_correlation=np.nan,
            p_value=np.nan,
            n_pairs=n_valid,
            n_controls=n_controls,
        )

    # Rows: template (0), match (1), controls (2 ...).
    vectors = np.vstack([template_correlations, match_correlations, *controls])[
        :, valid_mask
    ]
    period_correlations = _correlation_matrix(vectors)
    control_rows = list(range(2, 2 + n_controls))
    # EV: variance of match correlations explained by template, after
    # partialling out the control periods; the plain correlation without one.
    partial_corr = _partial_correlation(period_correlations, 0, 1, control_rows)
    ev = partial_corr**2
    # REV reverses time by swapping pre (control) and post (match):
    # r_partial(template, control | match). Defined for one control only.
    rev = (
        _partial_correlation(period_correlations, 0, 2, [1]) ** 2
        if n_controls == 1
        else np.nan
    )

    if np.isnan(ev):
        warnings.warn(
            "explained_variance_reactivation(): EV is undefined (NaN). A "
            "correlation vector is the same for every valid pair, or the "
            "control correlations are a perfect linear function of the "
            "template or match correlations (or of each other), so the "
            "partial correlation has no value. Check that each period's "
            "correlations come from pairwise_correlations() on that period's "
            "own counts.",
            UserWarning,
            stacklevel=2,
        )

    return ExplainedVarianceResult(
        explained_variance=float(ev),
        reversed_explained_variance=float(rev),
        partial_correlation=float(partial_corr),
        semipartial_correlation=_semipartial_correlation(
            period_correlations, 0, 1, control_rows
        ),
        p_value=_partial_correlation_p_value(partial_corr, n_valid - 2 - n_controls),
        n_pairs=n_valid,
        n_controls=n_controls,
    )


def _control_list(
    control_correlations: NDArray[np.float64] | Sequence[Any] | None,
) -> list[NDArray[np.float64]]:
    """Return the control periods' pair correlations as a list of 1-D arrays.

    Parameters
    ----------
    control_correlations : array-like or sequence of array-like, or None
        One control's pair correlations, shape (n_pairs,), or one per
        control, shape (n_controls, n_pairs).

    Returns
    -------
    list of ndarray, each shape (n_pairs,)
        Empty without controls.
    """
    if control_correlations is None:
        return []
    # Scalar entries: one period's pair vector. Vector entries (a list of
    # vectors or a 2-D array's rows): one vector per control period.
    first = control_correlations[0] if len(control_correlations) else 0.0
    items = (
        [control_correlations] if np.ndim(first) == 0 else list(control_correlations)
    )
    controls = [np.asarray(item, dtype=np.float64) for item in items]
    if not controls or any(control.ndim != 1 for control in controls):
        raise ValueError(
            _format_error(
                "explained_variance_reactivation(): control_correlations must be "
                "one period's pair correlations or a sequence of them.",
                fix="pass pairwise_correlations(control_counts), or a list "
                "[pairwise_correlations(a), pairwise_correlations(b)] for "
                "several control periods",
            )
        )
    return controls


def _selected_pairs(
    n_pairs: int,
    unit_groups: NDArray[Any] | Sequence[Any] | None,
    pair_mask: NDArray[np.bool_] | Sequence[bool] | None,
    *,
    call: str,
) -> NDArray[np.bool_]:
    """Pairs kept by ``unit_groups`` (different groups) and ``pair_mask``.

    Parameters
    ----------
    n_pairs : int
        Number of pairs, ``n_units * (n_units - 1) / 2``.
    unit_groups : array-like, shape (n_units,), or None
        Group of each unit; only pairs from different groups are kept.
    pair_mask : array-like of bool, shape (n_pairs,), or None
        Pairs to keep.
    call : str
        Function name for error messages.

    Returns
    -------
    ndarray of bool, shape (n_pairs,)
        True for the pairs to use.
    """
    selected = np.ones(n_pairs, dtype=bool)
    if pair_mask is not None:
        mask = np.asarray(pair_mask)
        if mask.dtype != np.bool_ or mask.shape != (n_pairs,):
            raise ValueError(
                _format_error(
                    f"{call}: pair_mask must be a boolean array with one entry "
                    f"per pair ({n_pairs}), got dtype {mask.dtype} and shape "
                    f"{mask.shape}.",
                    why="Why: entry k selects pair k of the template correlations; "
                    "an integer array would be read as pair indices.",
                    fix="pass a boolean array in np.triu_indices(n_units, k=1) "
                    "order, e.g. waveform_similarity < 0.8",
                )
            )
        selected &= mask
    if unit_groups is not None:
        groups = np.asarray(unit_groups)
        n_units = round((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
        if (
            groups.ndim != 1
            or groups.size != n_units
            or n_units * (n_units - 1) // 2 != n_pairs
        ):
            raise ValueError(
                _format_error(
                    f"{call}: unit_groups must hold one group per unit; got shape "
                    f"{groups.shape} for {n_pairs} pairs.",
                    why="Why: a pair is kept when its two units are in different "
                    "groups, so every unit of the template counts needs a group.",
                    fix="pass the tetrode, shank or region of each column of the "
                    "template counts (in template_unit_ids order when labels are "
                    "given)",
                )
            )
        first, second = np.triu_indices(n_units, k=1)
        selected &= groups[first] != groups[second]
    return selected


def _check_correlation_values(correlations: NDArray[np.float64], argument: str) -> None:
    """Refuse infinite values; NaN marks an undefined pair and is allowed.

    Parameters
    ----------
    correlations : ndarray, shape (n_pairs,)
        One period's pair correlations.
    argument : str
        Argument name for the error message.
    """
    if np.any(np.isinf(correlations)):
        raise ValueError(
            _format_error(
                f"explained_variance_reactivation(): {argument} holds infinite values.",
                why="Why: each entry must be the correlation of one pair of "
                "neurons; NaN marks a pair whose correlation is undefined.",
                fix="pass the output of pairwise_correlations() for each period",
            )
        )


def _correlation_matrix(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pearson correlations between the rows, NaN for a constant row.

    Parameters
    ----------
    vectors : ndarray, shape (n_vectors, n_pairs)
        Finite values.

    Returns
    -------
    ndarray, shape (n_vectors, n_vectors)
        Correlation matrix; rows and columns of a constant vector are NaN.
    """
    constant = np.ptp(vectors, axis=1) == 0
    matrix = np.full((len(vectors), len(vectors)), np.nan)
    varying = np.flatnonzero(~constant)
    if varying.size:
        matrix[np.ix_(varying, varying)] = np.atleast_2d(np.corrcoef(vectors[varying]))
    return matrix


def _partial_correlation(
    correlations: NDArray[np.float64], x: int, y: int, controls: Sequence[int]
) -> float:
    """Correlation of x and y controlling for ``controls``, NaN when undefined.

    Parameters
    ----------
    correlations : ndarray, shape (n_vectors, n_vectors)
        Correlation matrix of all vectors; NaN propagates.
    x, y : int
        Rows of the two vectors to correlate.
    controls : sequence of int
        Rows of the vectors to partial out; empty for the plain correlation.

    Returns
    -------
    float
        ``S_xy / sqrt(S_xx S_yy)`` with ``S`` the Schur complement of the
        controls' block, i.e. the correlation of the residuals of x and y
        regressed on the controls. With one control this is
        ``(r_xy - r_xz r_yz) / sqrt((1 - r_xz**2) (1 - r_yz**2))``. NaN when
        the controls are collinear or explain x or y perfectly (within
        rounding), which leaves no residual to correlate.
    """
    if not controls:
        return float(correlations[x, y])
    pair, rows = [x, y], list(controls)
    if np.isnan(correlations[np.ix_(pair + rows, pair + rows)]).any():
        return np.nan
    between_controls = correlations[np.ix_(rows, rows)]
    # Collinear controls (within rounding) leave the regression undefined.
    if not np.linalg.eigvalsh(between_controls).min() > 1e-12:
        return np.nan
    with_controls = correlations[np.ix_(pair, rows)]
    residual = correlations[np.ix_(pair, pair)] - with_controls @ np.linalg.solve(
        between_controls, with_controls.T
    )
    unexplained_x, unexplained_y = residual[0, 0], residual[1, 1]
    # |r| within rounding of 1 (np.corrcoef of identical vectors can return
    # 1 - 2e-16) leaves a residual of pure rounding error.
    if not (unexplained_x > 1e-12 and unexplained_y > 1e-12):
        return np.nan
    return float(residual[0, 1] / np.sqrt(unexplained_x * unexplained_y))


def _semipartial_correlation(
    correlations: NDArray[np.float64], x: int, y: int, controls: Sequence[int]
) -> float:
    """Correlation of x with the residual of y regressed on ``controls``.

    Parameters
    ----------
    correlations : ndarray, shape (n_vectors, n_vectors)
        Correlation matrix of all vectors; NaN propagates.
    x, y : int
        Rows of the vector kept and of the vector residualized.
    controls : sequence of int
        Rows of the vectors regressed out of y; empty for the plain
        correlation.

    Returns
    -------
    float
        ``S_xy / sqrt(S_yy)`` with ``S`` the Schur complement of the
        controls' block (the controls' share of x is not removed). With one
        control, ``(r_xy - r_xz r_yz) / sqrt(1 - r_yz**2)``. NaN when the
        controls are collinear or explain y completely.
    """
    if not controls:
        return float(correlations[x, y])
    pair, rows = [x, y], list(controls)
    if np.isnan(correlations[np.ix_(pair + rows, pair + rows)]).any():
        return np.nan
    between_controls = correlations[np.ix_(rows, rows)]
    if not np.linalg.eigvalsh(between_controls).min() > 1e-12:
        return np.nan
    with_controls = correlations[np.ix_(pair, rows)]
    residual = correlations[np.ix_(pair, pair)] - with_controls @ np.linalg.solve(
        between_controls, with_controls.T
    )
    if not residual[1, 1] > 1e-12:
        return np.nan
    return float(residual[0, 1] / np.sqrt(residual[1, 1]))


def _partial_correlation_p_value(correlation: float, df: int) -> float:
    """Two-sided t-test p-value of a (partial) correlation.

    Parameters
    ----------
    correlation : float
        Partial correlation; NaN gives NaN.
    df : int
        Degrees of freedom, ``n - 2 - n_controls``.

    Returns
    -------
    float
        ``2 * t.sf(|r| sqrt(df / (1 - r**2)), df)``; 0 for ``|r| = 1``, NaN
        when ``df < 1`` or the correlation is NaN.
    """
    if df < 1 or not np.isfinite(correlation):
        return np.nan
    unexplained = 1.0 - correlation**2
    if unexplained <= 0:
        return 0.0
    t_statistic = abs(correlation) * np.sqrt(df / unexplained)
    return float(2.0 * stats.t.sf(t_statistic, df))


def _pairs_in_template_order(
    template_correlations: NDArray[np.float64],
    match_correlations: NDArray[np.float64],
    controls: list[NDArray[np.float64]],
    labels: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]:
    """Put the match and control pairs in the template's unit order, by label.

    Parameters
    ----------
    template_correlations, match_correlations : array-like, shape (n_pairs,)
        Pair correlations from :func:`pairwise_correlations`.
    controls : list of ndarray, each shape (n_pairs,)
        Pair correlations of the control periods; empty without a control.
    labels : dict
        ``template_unit_ids``, ``match_unit_ids`` and ``control_unit_ids``
        as given; at least one is not None. With several controls,
        ``control_unit_ids`` holds one label array per control.

    Returns
    -------
    template, match : ndarray, shape (n_pairs,)
        The template pairs unchanged, and the match pairs reordered so pair
        ``k`` of every period is the same two cells.
    controls : list of ndarray, each shape (n_pairs,)
        The control pairs reordered the same way.
    """
    call = "explained_variance_reactivation()"
    if not controls and labels["control_unit_ids"] is not None:
        raise ValueError(
            _format_error(
                f"{call} got control_unit_ids= but no control_correlations=.",
                why="Why: control_unit_ids labels the control period's "
                "correlations, and there are none to label.",
                fix="pass control_correlations= with it, or drop control_unit_ids=",
            )
        )
    needed = ["template_unit_ids", "match_unit_ids"]
    if controls:
        needed.append("control_unit_ids")
    missing = [name for name in needed if labels[name] is None]
    if missing:
        given = [name for name in needed if labels[name] is not None]
        raise ValueError(
            _format_error(
                f"{call} got {'= and '.join(given)}= but not {'= or '.join(missing)}=.",
                why="Why: labels pair each correlation with its two cells, and a "
                "period without labels can only be paired by position; mixing "
                "the two in one call would compare different pairs of cells "
                "without an error.",
                fix="pass the column labels of the counts each period's "
                "correlations came from (template_unit_ids=, match_unit_ids= "
                "and, with control_correlations, control_unit_ids=; the same "
                "labels when every period was binned from one list of spike "
                "trains), or none of them to pair by position",
            )
        )
    template = np.asarray(template_correlations, dtype=np.float64)
    reference = _pair_labels(
        template,
        labels["template_unit_ids"],
        "template_correlations",
        "template_unit_ids",
    )

    def in_template_order(
        correlations: NDArray[np.float64],
        unit_ids: Any,
        argument: str,
        labels_argument: str,
    ) -> NDArray[np.float64]:
        values = np.asarray(correlations, dtype=np.float64)
        period_labels = _pair_labels(values, unit_ids, argument, labels_argument)
        order, only_template, only_period = _label_order(period_labels, reference)
        if only_template or only_period:
            raise ValueError(
                _format_error(
                    f"{call}: {argument} must describe the same units "
                    f"as template_correlations; only in template_unit_ids: "
                    f"{only_template}, only in {labels_argument}: {only_period}.",
                    why="Why: EV compares the correlation of each pair of cells "
                    "across periods, so every pair needs a correlation in every "
                    "period.",
                    fix="select the units recorded in every period, keep the "
                    "same units in each count matrix (counts[:, keep] and "
                    "unit_ids[keep]), and compute pairwise_correlations() again",
                )
            )
        return _reorder_pairs(values, order)

    match = in_template_order(
        match_correlations,
        labels["match_unit_ids"],
        "match_correlations",
        "match_unit_ids",
    )
    if not controls:
        return template, match, []
    if len(controls) == 1:
        return (
            template,
            match,
            [
                in_template_order(
                    controls[0],
                    labels["control_unit_ids"],
                    "control_correlations",
                    "control_unit_ids",
                )
            ],
        )
    control_labels = labels["control_unit_ids"]
    if (
        control_labels is None
        or len(control_labels) != len(controls)
        or any(np.ndim(item) != 1 for item in control_labels)
    ):
        raise ValueError(
            _format_error(
                f"{call}: with {len(controls)} control periods, control_unit_ids "
                "must hold one label array per control.",
                why="Why: each control period's pair correlations are matched to "
                "the template's pairs by that period's own column labels.",
                fix="pass control_unit_ids=[labels_a, labels_b, ...] in the order "
                "of control_correlations",
            )
        )
    return (
        template,
        match,
        [
            in_template_order(
                control,
                unit_ids,
                f"control_correlations[{index}]",
                f"control_unit_ids[{index}]",
            )
            for index, (control, unit_ids) in enumerate(
                zip(controls, control_labels, strict=True)
            )
        ],
    )


def _pair_labels(
    correlations: NDArray[np.float64],
    unit_ids: NDArray[Any] | Sequence[Any] | None,
    argument: str,
    labels_argument: str,
) -> NDArray[Any]:
    """Validate the unit labels of one period's pair vector.

    Parameters
    ----------
    correlations : ndarray, shape (n_pairs,)
        Pair correlations from :func:`pairwise_correlations`.
    unit_ids : array-like
        One label per unit, ``n_pairs == n_units * (n_units - 1) / 2``.
    argument, labels_argument : str
        Names of the correlations and labels arguments for error messages.

    Returns
    -------
    ndarray, shape (n_units,)
        The resolved labels.
    """
    call = "explained_variance_reactivation()"
    if correlations.ndim != 1:
        raise ValueError(
            _format_error(
                f"{call}: {argument} must be 1-D to be labelled by "
                f"{labels_argument}=, got shape {correlations.shape}.",
                fix="pass the pair correlations returned by pairwise_correlations()",
            )
        )
    n_pairs = correlations.size
    n_units = round((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
    if n_units * (n_units - 1) // 2 != n_pairs:
        raise ValueError(
            _format_error(
                f"{call}: {argument} has {n_pairs} entries, which is not "
                "n * (n - 1) / 2 pairs for any number of units n.",
                why=f"Why: {labels_argument} labels the units whose pairs "
                "pairwise_correlations() lists in np.triu_indices(n, k=1) order, "
                "so the correlations must hold every pair.",
                fix="pass the full output of pairwise_correlations() together "
                "with the column labels of its spike_counts",
            )
        )
    return resolve_unit_ids(unit_ids, n_units, context=f"{call} for {argument}")


def _reorder_pairs(
    correlations: NDArray[np.float64], order: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Reorder a pair vector so its units follow ``order``.

    Parameters
    ----------
    correlations : ndarray, shape (n_pairs,)
        Correlation of the units at ``np.triu_indices(n_units, k=1)``.
    order : ndarray of intp, shape (n_units,)
        Position of each target unit among the current units.

    Returns
    -------
    ndarray, shape (n_pairs,)
        Correlation of target units ``i`` and ``j`` at the position
        ``np.triu_indices(n_units, k=1)`` gives the pair ``(i, j)``.
    """
    n_units = order.size
    upper = np.triu_indices(n_units, k=1)
    matrix = np.zeros((n_units, n_units), dtype=np.float64)
    matrix[upper] = correlations
    matrix[upper[::-1]] = correlations
    reordered: NDArray[np.float64] = matrix[np.ix_(order, order)][upper]
    return reordered
