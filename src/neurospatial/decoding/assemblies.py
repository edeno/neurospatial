"""
Cell assembly detection and reactivation analysis.

This module provides functions for detecting coordinated neural ensembles
(cell assemblies) and measuring their reactivation strength across behavioral
states.

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
   ...     spike_counts, algorithm="ica", unit_ids=sorter_labels
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

**Reactivation Analysis**:

1. Compute pairwise correlations in behavior and sleep:

   >>> corr_behavior = pairwise_correlations(counts_behavior)  # doctest: +SKIP
   >>> corr_sleep = pairwise_correlations(counts_sleep)  # doctest: +SKIP

2. Measure explained variance:

   >>> ev_result = explained_variance_reactivation(
   ...     corr_behavior,
   ...     corr_sleep,
   ...     template_unit_ids=behavior_labels,
   ...     match_unit_ids=sleep_labels,
   ... )  # doctest: +SKIP
   >>> print(f"EV = {ev_result.explained_variance:.3f}")  # doctest: +SKIP

Notes
-----
**Bin Size Guidelines**:

Bin sizes of 25-100ms are typical for assembly detection. Smaller bins
(< 25ms) increase noise; larger bins (> 100ms) may miss fast assembly
dynamics.

**Data Requirements**:

- Need 5-10× more time bins than neurons for reliable results
- Low firing rate neurons (< 0.1 Hz) may not form detectable assemblies
- Works best with functionally related neural populations

References
----------
Lopes-dos-Santos V, Ribeiro S, Bhattacharyya A (2013). Detecting cell assemblies
    in large neuronal populations. J Neurosci Methods 220(2):149-166.
Peyrache A, Khamassi M, Benchenane K, Wiener SI, Battaglia FP (2009).
    Replay of rule-learning related neural patterns in the prefrontal cortex
    during sleep. Nat Neurosci 12(7):919-926.
Kudrimoti HS, Barnes CA, McNaughton BL (1999). Reactivation of hippocampal
    cell assemblies: effects of behavioral state, experience, and EEG dynamics.
    J Neurosci 19(10):4090-4101.
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
    "ExplainedVarianceResult",
    "assembly_activation",
    "detect_assemblies",
    "explained_variance_reactivation",
    "marchenko_pastur_threshold",
    "pairwise_correlations",
    "reactivation_strength",
]


# =============================================================================
# Data Classes
# =============================================================================


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
    member_indices : NDArray[np.int64]
        Column positions of neurons whose absolute-weight z-scores exceed
        z_threshold. These are the "core" members of the assembly.
    explained_variance_ratio : float
        Fraction of the population's variance this pattern accounts for, in
        [0, 1]. Higher values indicate more prominent assemblies. For PCA it
        is the pattern's eigenvalue over the number of varying neurons; for
        ICA, the variance of the z-scored counts reconstructed from the
        pattern's source; for NMF, the variance over time of the pattern's
        rank-1 reconstruction of the raw counts, as a fraction of their
        summed variance. NMF components can overlap in time, so their
        fractions need not add up to the variance the fit explains.
    unit_ids : ndarray, shape (n_neurons,)
        Label of the unit each weight belongs to (read-only): the
        ``unit_ids=`` given to :func:`detect_assemblies`, else
        ``np.arange(n_neurons)``. Pass the column labels of new counts as
        ``unit_ids=`` to :func:`assembly_activation`, or as
        ``template_unit_ids=`` and ``match_unit_ids=`` to
        :func:`reactivation_strength`, to match their columns by label
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

    Members are selected by z-scoring absolute weights across neurons and
    thresholding. With the default cutoff of 2.0, an absolute weight must be
    more than two standard deviations above their mean to be selected.
    This is a heuristic weight cutoff,
    not a calibrated per-neuron significance test. A detected pattern can have
    no core members even when its underlying dimension exceeds the
    Marchenko-Pastur threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding import AssemblyPattern
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
        """Resolve unit labels and own read-only copies of the arrays."""
        if self.unit_ids is None:
            object.__setattr__(self, "_unit_ids_generated", True)
        labels = resolve_unit_ids(
            self.unit_ids, len(self.weights), context="AssemblyPattern"
        )
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
    patterns : list[AssemblyPattern]
        Detected assembly patterns, ordered by ``explained_variance_ratio``
        (highest first) for every method.
    activations : NDArray[np.float64], shape (n_time_bins, n_assemblies)
        Activation strength R(t) of each pattern in each time bin of the
        detection counts, time first like the counts; column ``k`` equals
        ``assembly_activation(spike_counts, patterns[k])``.
    method : str
        Detection method used ('ica', 'pca', or 'nmf').
    n_significant : int
        Number of PCA eigenvalues above the Marchenko-Pastur reference
        threshold. This counts retained dimensions, not assemblies with
        thresholded core members. It equals len(patterns) with
        ``n_components="auto"`` (zero patterns when no eigenvalue exceeds the
        threshold) and may differ when n_components was specified manually.
    eigenvalues : NDArray[np.float64], shape (n_neurons,)
        Eigenvalues of the correlation matrix of the neurons that vary over
        time, in descending order, useful for scree plots. Each excluded
        constant neuron adds an eigenvalue of 0 at the end.
    threshold : float
        Marchenko-Pastur threshold for the number of varying neurons and
        time bins.
    unit_ids : ndarray, shape (n_neurons,)
        Label of each column of the detection counts (read-only), the same
        labels every pattern carries: the ``unit_ids=`` given to
        :func:`detect_assemblies`, else ``np.arange(n_neurons)``.

    Other Parameters
    ----------------
    _unit_ids_generated : bool, default=False
        Internal identity flag: True when ``unit_ids`` were generated.

    Notes
    -----
    **Significance Determination**:

    ``n_significant`` counts eigenvalues above the Marchenko-Pastur reference
    threshold. This dimension-selection criterion does not assign a p-value
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

    patterns: list[AssemblyPattern]
    activations: NDArray[np.float64]
    method: str
    n_significant: int
    eigenvalues: NDArray[np.float64]
    threshold: float
    unit_ids: NDArray[Any] | None = field(default=None, kw_only=True, compare=False)
    _unit_ids_generated: bool = field(
        default=False, repr=False, compare=False, kw_only=True
    )

    def __post_init__(self) -> None:
        """Resolve unit labels and own read-only copies of the arrays."""
        if self.unit_ids is None:
            object.__setattr__(self, "_unit_ids_generated", True)
        # One eigenvalue per neuron, from the neurons' correlation matrix.
        labels = resolve_unit_ids(
            self.unit_ids, len(self.eigenvalues), context="AssemblyDetectionResult"
        )
        object.__setattr__(self, "unit_ids", labels)
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "n_assemblies": len(self.patterns),
            "n_significant": self.n_significant,
            "method": self.method,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, repr=False)
class ExplainedVarianceResult(ResultMixin):
    """
    Results from explained variance reactivation analysis.

    Attributes
    ----------
    explained_variance : float
        Squared correlation between template and match correlation vectors,
        partialled for the baseline control when one is supplied. Range [0, 1].
    reversed_ev : float
        With a baseline control, the squared partial correlation between
        template and control vectors, partialled for the match. Without a
        control, equals explained_variance by construction.
    partial_correlation : float
        Partial correlation between template and match, controlling for
        baseline correlations (if control provided).
    n_pairs : int
        Number of neuron pairs used in analysis.

    Notes
    -----
    **Interpretation**:

    With a baseline control, EV > REV describes more template-related
    correlation structure after behavior than before behavior. Without that
    control, EV == REV, so the comparison provides no directional evidence.
    These are effect sizes, not p-values: a fixed cutoff such as EV > 0.1 or
    EV > 0.3 does not establish statistical significance. Statistical inference
    needs a suitable null/control analysis for the recording and question.
    An in-sample or synthetic demonstration does not establish biological
    reactivation or held-out decoding accuracy.

    Examples
    --------
    >>> result = explained_variance_reactivation(
    ...     corr_behavior, corr_sleep, control_correlations=corr_pre_behavior
    ... )  # doctest: +SKIP
    >>> if result.explained_variance > result.reversed_ev:  # doctest: +SKIP
    ...     print(
    ...         "Template-related effect exceeds the baseline-related effect"
    ...     )  # doctest: +SKIP
    """

    explained_variance: float
    reversed_ev: float
    partial_correlation: float
    n_pairs: int

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "explained_variance": self.explained_variance,
            "reversed_ev": self.reversed_ev,
            "partial_correlation": self.partial_correlation,
            "n_pairs": self.n_pairs,
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

    For reliable threshold estimation, n_time_bins should be significantly
    larger than n_neurons (typically > 5× larger).

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
    Lopes-dos-Santos V et al. (2013). Detecting cell assemblies in large
        neuronal populations. J Neurosci Methods 220(2):149-166.
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
    algorithm: Literal["ica", "pca", "nmf"] = "ica",
    n_components: int | Literal["auto"] = "auto",
    z_threshold: float = 2.0,
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
    algorithm : {'ica', 'pca', 'nmf'}, default='ica'
        Detection algorithm:

        - 'ica': Independent Component Analysis (FastICA). Best for
          separating statistically independent assemblies. Default choice.
        - 'pca': Principal Component Analysis. Finds orthogonal patterns
          ordered by variance. Faster but patterns may be mixed.
        - 'nmf': Non-negative Matrix Factorization. Constrains weights
          to be non-negative. Good when assemblies don't overlap.

    n_components : int or 'auto', default='auto'
        Number of assemblies to detect:

        - 'auto': Determined by Marchenko-Pastur threshold (recommended)
        - int: Fixed number of components

    z_threshold : float, default=2.0
        Z-score threshold for assembly membership. Neurons with absolute-weight
        z-scores above this are selected as core members.

        - 2.0: Default heuristic weight cutoff
        - 2.5: Select fewer core members
        - 1.5: Select more core members

        These cutoffs do not supply per-neuron p-values. A selected dimension
        may yield a pattern with no weights above the chosen cutoff.

    rng : int or np.random.Generator, optional
        Random seed or generator for reproducibility. Affects ICA and NMF
        initialization. Both ``int`` seeds and ``np.random.Generator``
        instances are accepted.
    unit_ids : array-like, shape (n_neurons,), optional
        Label of the unit in each column of ``spike_counts``, for example the
        sorter's unit IDs. Stored on the result and on every pattern, so
        ``pattern.member_unit_ids`` names the members, and passing the labels
        of new counts to :func:`assembly_activation` (``unit_ids=``) or
        :func:`reactivation_strength` (``template_unit_ids=`` and
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
        parameters, or ``unit_ids`` does not hold one distinct label per
        column.

    Notes
    -----
    **Algorithm (Lopes-dos-Santos et al., 2013)**:

    1. Z-score spike counts for each neuron
    2. Compute correlation matrix
    3. Eigendecomposition to find significant dimensions (Marchenko-Pastur)
    4. Project onto significant PCs (and whiten their scores for ICA)
    5. Apply ICA/PCA/NMF to extract assembly patterns
    6. Threshold weights to identify core members
    7. Compute the activation strength R(t) of each pattern (see
       :func:`assembly_activation`)

    **Excluded neurons and components**: A neuron with the same count in
    every time bin has no correlation with any other neuron. It is excluded
    from every method (with a warning), gets weight 0 in every pattern, and
    the Marchenko-Pastur threshold uses the number of varying neurons. With
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

    - **ICA**: Best for most cases. Assumes assemblies are statistically
      independent, which is often true for distinct memory representations.
    - **PCA**: Use when you want orthogonal patterns or fast computation.
      Good for exploratory analysis.
    - **NMF**: Use when assemblies should have non-negative membership
      (no inhibitory participation).

    **Preprocessing**:

    Input spike counts are automatically z-scored. For best results:
    - Use bin sizes of 25-100ms
    - Ensure sufficient time bins (> 5× n_neurons)
    - Remove neurons with very low firing rates

    Examples
    --------
    Three of twenty units fire together; label the columns with the
    sorter's unit IDs so the members are named by label:

    >>> import numpy as np
    >>> from neurospatial.decoding import detect_assemblies
    >>> rng = np.random.default_rng(0)
    >>> counts = rng.poisson(1.0, (2000, 20)).astype(float)
    >>> counts[:, [2, 9, 15]] += 3 * rng.poisson(0.3, (2000, 1))
    >>> labels = np.arange(101, 121)  # the sorter's unit IDs
    >>> result = detect_assemblies(counts, unit_ids=labels, rng=0)
    >>> result.n_significant
    1
    >>> result.patterns[0].member_unit_ids.tolist()
    [103, 110, 116]

    >>> # Basic usage
    >>> result = detect_assemblies(spike_counts)  # doctest: +SKIP
    >>> print(
    ...     f"Found {result.n_significant} dimensions above threshold"
    ... )  # doctest: +SKIP

    >>> # With specific number of components
    >>> result = detect_assemblies(spike_counts, n_components=5)  # doctest: +SKIP

    >>> # Using PCA for faster computation
    >>> result = detect_assemblies(spike_counts, algorithm="pca")  # doctest: +SKIP

    >>> # Access assembly patterns
    >>> for i, pattern in enumerate(result.patterns):  # doctest: +SKIP
    ...     print(f"Assembly {i}: units {pattern.member_unit_ids.tolist()}")

    See Also
    --------
    assembly_activation : Compute activation for new data
    marchenko_pastur_threshold : Significance threshold calculation

    References
    ----------
    Lopes-dos-Santos V, Ribeiro S, Bhattacharyya A (2013). Detecting cell
        assemblies in large neuronal populations. J Neurosci Methods.
    """
    spike_counts = _validated_counts(spike_counts, call="detect_assemblies()")
    n_time_bins, n_neurons = spike_counts.shape
    if algorithm not in ("ica", "pca", "nmf"):
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'ica', 'pca', or 'nmf'.")

    if n_neurons < 3:
        raise ValueError(
            f"Need at least 3 neurons for assembly detection, got {n_neurons}"
        )
    unit_ids_generated = unit_ids is None
    labels = resolve_unit_ids(unit_ids, n_neurons, context="detect_assemblies()")

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

    mp_threshold = marchenko_pastur_threshold(n_retained, n_time_bins)
    n_significant = int(np.sum(eigenvalues > mp_threshold))

    # Determine number of components
    if n_components == "auto":
        n_comp = n_significant
        if n_comp == 0:
            warnings.warn(
                f"detect_assemblies(): no eigenvalue of the correlation matrix "
                f"exceeds the Marchenko-Pastur bound {mp_threshold:.3f} (largest "
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

    # Derive an integer seed for sklearn from the public ``rng`` argument.
    # sklearn estimators accept an int seed (or RandomState), not a numpy
    # Generator, so normalize a Generator to a reproducible integer seed.
    if rng is None:
        seed: int | None = None
    elif isinstance(rng, np.random.Generator):
        seed = int(rng.integers(0, 2**32 - 1))
    else:
        seed = int(rng)

    # Perform dimensionality reduction
    if n_comp == 0:
        patterns, explained_var = np.zeros((0, n_retained)), np.zeros(0)
    elif algorithm == "pca":
        patterns, explained_var = _detect_pca(spike_counts_z, n_comp)
    elif algorithm == "ica":
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

    # ICA and eigenvector signs are arbitrary. Orient each pattern so its
    # largest-magnitude weight is positive: its members are then the neurons
    # with the largest positive weights. The activation strength does not
    # depend on the sign.
    largest = patterns[np.arange(n_found), np.argmax(np.abs(patterns), axis=1)]
    patterns = patterns * np.where(largest < 0, -1.0, 1.0)[:, None]

    # Activation strength of every pattern in every detection bin,
    # shape (n_time_bins, n_found); excluded neurons have weight 0.
    activations = _activation_strength(spike_counts_z.T, patterns)

    # Members: absolute weights z-scored across the retained neurons, then
    # thresholded; shape (n_found, n_retained).
    member_masks = stats.zscore(np.abs(patterns), axis=1) > z_threshold
    full_patterns = np.zeros((n_found, n_neurons))
    full_patterns[:, retained] = patterns

    assembly_patterns = [
        AssemblyPattern(
            weights=full_patterns[i],
            member_indices=retained[member_masks[i]].astype(np.int64),
            explained_variance_ratio=explained_var[i],
            unit_ids=labels,
            _unit_ids_generated=unit_ids_generated,
        )
        for i in range(n_found)
    ]

    return AssemblyDetectionResult(
        patterns=assembly_patterns,
        activations=activations,
        method=algorithm,
        n_significant=n_significant,
        eigenvalues=eigenvalues,
        threshold=mp_threshold,
        unit_ids=labels,
        _unit_ids_generated=unit_ids_generated,
    )


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
    norms = np.linalg.norm(patterns, axis=1, keepdims=True)
    patterns, activations = patterns / norms, activations * norms

    # Explained variance of component k: the variance over time of its rank-1
    # reconstruction outer(w_k, h_k), summed over neurons, as a fraction of
    # the summed variance of the counts. Unlike a share of the total fit
    # split by pattern norm, this does not depend on the W/H scaling.
    # Components can overlap in time, so the fractions need not add up to the
    # variance the whole factorization explains.
    explained_var = (
        np.sum(patterns**2, axis=1)
        * np.var(activations, axis=1)
        / np.sum(np.var(spike_counts, axis=1))
    )

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
    Use :func:`reactivation_strength` to compare a pattern's mean activation
    between periods.

    Examples
    --------
    Units 0 and 1 form an assembly. A burst of unit 0 alone barely registers;
    the two firing together does:

    >>> import numpy as np
    >>> from neurospatial.decoding import AssemblyPattern, assembly_activation
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
    reactivation_strength : Compare activations between periods

    References
    ----------
    Lopes-dos-Santos V, Ribeiro S, Tort ABL (2013). Detecting cell
        assemblies in large neuronal populations. J Neurosci Methods
        220(2):149-166.
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

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding import pairwise_correlations
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


def reactivation_strength(
    template_counts: NDArray[np.float64],
    match_counts: NDArray[np.float64],
    pattern: AssemblyPattern,
    *,
    template_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    match_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> float:
    """
    Compare a pattern's mean activation strength between two periods.

    Returns the mean activation strength R(t) of the pattern in the match
    period divided by its mean in the template period (Peyrache et al., 2009,
    2010). Values > 1 mean the pattern's neurons fire together more in the
    match period than in the template period.

    Parameters
    ----------
    template_counts : NDArray[np.float64], shape (n_template_bins, n_neurons)
        Spike counts during template period (e.g., behavior), finite and
        non-negative.
    match_counts : NDArray[np.float64], shape (n_match_bins, n_neurons)
        Spike counts during match period (e.g., sleep), finite and
        non-negative.
    pattern : AssemblyPattern
        Assembly pattern to measure.
    template_unit_ids : array-like, shape (n_neurons,), optional
        Label of the unit in each column of ``template_counts``. Pass it
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
    float
        Mean activation strength in the match period over its mean in the
        template period. 1 when both periods are the same counts, about 0
        when the pattern's neurons fire independently in the match period,
        and negative when they fire in opposition. NaN when the pattern shows
        no co-activation in the template period (mean activation strength
        <= 0) or every neuron with nonzero weight is constant in a period,
        so the ratio is undefined.

    Raises
    ------
    ValueError
        If the counts and the pattern have different numbers of neurons,
        either period holds NaN, infinite or negative values, the pattern
        has no nonzero weight, or with labels, if only one of
        ``template_unit_ids`` and ``match_unit_ids`` is given, the pattern's
        labels were generated, or a period's labels differ from the
        pattern's.

    Warns
    -----
    UserWarning
        If neurons with nonzero weight have the same count in every bin of
        either period (they are excluded from both periods, named in the
        warning), or the ratio is undefined.

    Notes
    -----
    Each period is z-scored with its own per-neuron mean and standard
    deviation, as in the published method. The mean of R(t) over a period is
    then :math:`w^T C w - 1` for unit-length weights :math:`w` and the
    period's correlation matrix :math:`C`, the pattern's co-activation
    beyond independent firing, so the ratio compares how strongly the
    pattern's correlation structure is expressed in the two periods.

    A neuron with the same count in every bin of a period has no z-score
    there. It is excluded from both periods, so both means use the same
    neurons, and the remaining weights are rescaled to unit length.

    **Rate and noise floor**: Because each period is standardized on its
    own, scaling a period's counts leaves the ratio unchanged. Firing-rate
    changes still matter when they change the correlations: independent
    spikes added to the members dilute their correlation and lower the
    ratio. With no co-activation in the match period the mean activation
    strength is 0 up to sampling noise of standard deviation about
    :math:`\\sqrt{2 / n_{match}}` (independent Gaussian z-scores), so the ratio
    has a noise floor of about :math:`\\sqrt{2 / n_{match}}` divided by the
    template mean. Compare against a control period (such as pre-task rest)
    or a shuffle null rather than against 1 or 0.

    For the correlation structure of the whole population rather than one
    pattern, use :func:`explained_variance_reactivation` with pairwise
    correlations.

    Examples
    --------
    Units 7 and 8 burst together more often in the match period, whose counts
    list the units in another order:

    >>> import numpy as np
    >>> from neurospatial.decoding import AssemblyPattern, reactivation_strength
    >>> pattern = AssemblyPattern(
    ...     np.array([0.7, 0.7, 0.1]), np.array([0, 1]), 0.5, unit_ids=[7, 8, 9]
    ... )
    >>> rng = np.random.default_rng(0)
    >>> run = rng.poisson(1.0, (500, 3)).astype(float)
    >>> run[:, :2] += 2 * rng.poisson(0.2, (500, 1))
    >>> rest = rng.poisson(1.0, (500, 3)).astype(float)
    >>> rest[:, 1:] += 2 * rng.poisson(0.6, (500, 1))  # columns: 9, 7, 8
    >>> strength = reactivation_strength(
    ...     run, rest, pattern, template_unit_ids=[7, 8, 9], match_unit_ids=[9, 7, 8]
    ... )
    >>> bool(strength > 1.0)
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
    """
    call = "reactivation_strength()"
    if (template_unit_ids is None) != (match_unit_ids is None):
        given, missing = (
            ("template_unit_ids", "match_unit_ids")
            if match_unit_ids is None
            else ("match_unit_ids", "template_unit_ids")
        )
        raise ValueError(
            _format_error(
                f"{call} got {given}= but not {missing}=.",
                why="Why: one period's columns would be matched to the "
                "pattern's weights by label and the other's by position, so "
                "a period whose units are listed in another order would "
                "silently pair weights with the wrong cells.",
                fix="pass the column labels of both count matrices "
                "(template_unit_ids= and match_unit_ids=; the same labels when "
                "both were binned from one list of spike trains), or neither "
                "to pair columns by position",
            )
        )
    template_counts = _columns_by_label(
        _validated_counts(template_counts, call=call, argument="template_counts"),
        template_unit_ids,
        pattern,
        call=call,
        argument="template_counts",
        labels_argument="template_unit_ids",
    ).T
    match_counts = _columns_by_label(
        _validated_counts(match_counts, call=call, argument="match_counts"),
        match_unit_ids,
        pattern,
        call=call,
        argument="match_counts",
        labels_argument="match_unit_ids",
    ).T

    n_template_neurons = template_counts.shape[0]
    n_match_neurons = match_counts.shape[0]
    n_weights = len(pattern.weights)
    if n_template_neurons != n_weights or n_match_neurons != n_weights:
        raise ValueError(
            f"template_counts has {n_template_neurons} neurons and "
            f"match_counts has {n_match_neurons} neurons but pattern has "
            f"{n_weights} weights. All must match."
        )

    weights = _unit_weights(pattern.weights, call=call)
    template_z, template_constant = _z_scores(template_counts.T)
    match_z, match_constant = _z_scores(match_counts.T)

    # A neuron constant in either period has no z-score there; drop it from
    # both so the two means use the same neurons.
    excluded = (template_constant | match_constant) & (weights != 0)
    if np.any(excluded):
        names = np.asarray(pattern.unit_ids)
        warnings.warn(
            f"{call}: units {names[excluded].tolist()} have nonzero weight but "
            "the same count in every time bin of "
            + " and ".join(
                period
                for period, constant in (
                    ("template_counts", template_constant & excluded),
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
                "strength is undefined (NaN).",
                UserWarning,
                stacklevel=2,
            )
            return np.nan
        weights = weights / norm

    mean_template = float(np.mean(_activation_strength(template_z, weights[None])))
    mean_match = float(np.mean(_activation_strength(match_z, weights[None])))

    if not mean_template > 0:
        warnings.warn(
            f"{call}: the pattern's mean activation strength in the template "
            f"period is {mean_template:.3g}, so its neurons do not fire together "
            "there and the ratio is undefined (NaN). Measure a pattern detected "
            "in template_counts, or compare mean assembly_activation() values "
            "directly.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan
    return mean_match / mean_template


def explained_variance_reactivation(
    template_correlations: NDArray[np.float64],
    match_correlations: NDArray[np.float64],
    *,
    control_correlations: NDArray[np.float64] | None = None,
    template_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    match_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    control_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> ExplainedVarianceResult:
    """
    Compute explained variance for pairwise correlation reactivation.

    Measures how well pairwise correlations during 'match' period
    are predicted by correlations from 'template' period.

    Parameters
    ----------
    template_correlations : NDArray[np.float64], shape (n_pairs,)
        Pairwise correlations during template (e.g., behavior).
        From `pairwise_correlations()`. Without labels, pairs are matched by
        position, so every period's counts must have the same units in the
        same column order.
    match_correlations : NDArray[np.float64], shape (n_pairs,)
        Pairwise correlations during match (e.g., sleep/rest).
    control_correlations : NDArray[np.float64], optional
        Control period correlations (e.g., pre-behavior baseline).
        If provided, computes partial correlation controlling for baseline.
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
    control_unit_ids : array-like, shape (n_neurons,), optional
        Column labels of the counts that ``control_correlations`` came from.
        Required with ``template_unit_ids`` when ``control_correlations`` is
        given, and refused without it.

    Returns
    -------
    ExplainedVarianceResult
        Contains EV, reversed EV, partial correlation, and n_pairs.

    Raises
    ------
    ValueError
        If correlation arrays have different lengths. With labels: if a
        supplied period has no labels, a period's labels are not one
        distinct label per unit of its pair vector, or the periods hold
        different units.

    Notes
    -----
    **Explained Variance (EV)**:

    When a control is provided, EV is the squared *partial* correlation
    between template and match correlations, controlling for the control
    (pre-behavior) baseline:

    .. math::

        EV = r_{partial}(template, match \\mid control)^2

    This measures how much of the match period correlation structure is
    explained by the template period beyond what the control already
    explains. When no control is provided, EV reduces to the plain
    squared correlation :math:`r(template, match)^2`.

    **Reversed EV (REV)**:

    REV (Pennartz et al. 2004) swaps the pre- and post-behavior periods:
    the template's correlation with the control, partialled for the match:

    .. math::

        REV = r_{partial}(template, control \\mid match)^2

    With a control, EV exceeding REV indicates a larger template-related
    effect after behavior than before behavior; it is not a
    calibrated significance test. Without a control there is no asymmetry to
    break, so ``EV == REV`` by construction. Neither a fixed EV cutoff nor
    EV > REV alone establishes significant reactivation.

    **Partial Correlation**:

    The partial correlation underlying EV is:

    .. math::

        r_{partial} = \\frac{r_{tm} - r_{tc} \\cdot r_{mc}}{\\sqrt{(1-r_{tc}^2)(1-r_{mc}^2)}}

    where t=template, m=match, c=control.

    Examples
    --------
    Units 3, 5 and 8 fire together during behavior and sleep. The sleep
    counts list the units in another order; their labels pair each sleep
    correlation with the same two cells as in behavior:

    >>> import numpy as np
    >>> from neurospatial.decoding import (
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
    >>> print(f"REV = {result.reversed_ev:.3f}")  # doctest: +SKIP

    References
    ----------
    Kudrimoti HS, Barnes CA, McNaughton BL (1999). Reactivation of
        hippocampal cell assemblies. J Neurosci 19(10):4090-4101.
    Pennartz CMA et al. (2004). The ventral striatum in off-line processing:
        ensemble reactivation during sleep and modulation by hippocampal
        ripples. J Neurosci 24(29):6446-6456.
    """
    labels = {
        "template_unit_ids": template_unit_ids,
        "match_unit_ids": match_unit_ids,
        "control_unit_ids": control_unit_ids,
    }
    if any(value is not None for value in labels.values()):
        template_correlations, match_correlations, control_correlations = (
            _pairs_in_template_order(
                template_correlations,
                match_correlations,
                control_correlations,
                labels,
            )
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

    # Remove NaN pairs
    valid_mask = np.isfinite(template_correlations) & np.isfinite(match_correlations)
    if control_correlations is not None:
        control_correlations = np.asarray(control_correlations, dtype=np.float64)
        if len(control_correlations) != n_pairs:
            raise ValueError(
                f"control correlations must have same length as template/match. "
                f"Got {len(control_correlations)}, expected {n_pairs}."
            )
        _check_correlation_values(control_correlations, "control_correlations")
        valid_mask &= np.isfinite(control_correlations)

    template_valid = template_correlations[valid_mask]
    match_valid = match_correlations[valid_mask]

    if len(template_valid) < 3:
        warnings.warn(
            f"Only {len(template_valid)} valid pairs after removing NaN. "
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
            reversed_ev=np.nan,
            partial_correlation=np.nan,
            n_pairs=len(template_valid),
        )

    # Pairwise correlation between template and match correlation vectors.
    r_tm = _pearson(template_valid, match_valid)

    if control_correlations is not None:
        control_valid = control_correlations[valid_mask]
        r_tc = _pearson(template_valid, control_valid)
        r_mc = _pearson(match_valid, control_valid)

        # EV: variance of match correlations explained by template, after
        # partialling out the control (pre-behavior) baseline.
        #   r_partial(template, match | control)
        partial_corr = _partial_correlation(r_tm, r_tc, r_mc)
        ev = partial_corr**2

        # Reverse time by swapping pre (control) and post (match):
        # r_partial(template, control | match).
        rev = _partial_correlation(r_tc, r_tm, r_mc) ** 2
    else:
        # No control: EV is the plain squared correlation. REV reduces to
        # the same value because there is no asymmetry to break; the
        # control is what makes EV != REV meaningful.
        partial_corr = r_tm
        ev = r_tm**2
        rev = r_tm**2

    if np.isnan(ev):
        warnings.warn(
            "explained_variance_reactivation(): EV is undefined (NaN). A "
            "correlation vector is the same for every valid pair, or the "
            "control correlations are a perfect linear function of the "
            "template or match correlations, so the partial correlation has "
            "no value. Check that each period's correlations come from "
            "pairwise_correlations() on that period's own counts.",
            UserWarning,
            stacklevel=2,
        )

    return ExplainedVarianceResult(
        explained_variance=float(ev),
        reversed_ev=float(rev),
        partial_correlation=float(partial_corr),
        n_pairs=int(np.sum(valid_mask)),
    )


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


def _pearson(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """Pearson correlation of two vectors, NaN when either is constant.

    Parameters
    ----------
    first, second : ndarray, shape (n_pairs,)
        Finite values.

    Returns
    -------
    float
        Their correlation, or NaN when a vector has no variance.
    """
    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def _partial_correlation(r_xy: float, r_xz: float, r_yz: float) -> float:
    """Correlation of x and y controlling for z, NaN when undefined.

    Parameters
    ----------
    r_xy, r_xz, r_yz : float
        Pairwise correlations; NaN propagates.

    Returns
    -------
    float
        The partial correlation, or NaN when z explains x or y perfectly
        (within rounding), which leaves no residual to correlate.
    """
    unexplained_x, unexplained_y = 1.0 - r_xz**2, 1.0 - r_yz**2
    # |r| within rounding of 1 (np.corrcoef of identical vectors can return
    # 1 - 2e-16) leaves a residual of pure rounding error.
    if not (unexplained_x > 1e-12 and unexplained_y > 1e-12):
        return np.nan
    return float((r_xy - r_xz * r_yz) / np.sqrt(unexplained_x * unexplained_y))


def _pairs_in_template_order(
    template_correlations: NDArray[np.float64],
    match_correlations: NDArray[np.float64],
    control_correlations: NDArray[np.float64] | None,
    labels: dict[str, NDArray[Any] | Sequence[Any] | None],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Put the match and control pairs in the template's unit order, by label.

    Parameters
    ----------
    template_correlations, match_correlations : array-like, shape (n_pairs,)
        Pair correlations from :func:`pairwise_correlations`.
    control_correlations : array-like, shape (n_pairs,), or None
        Pair correlations of the control period, if any.
    labels : dict
        ``template_unit_ids``, ``match_unit_ids`` and ``control_unit_ids``
        as given; at least one is not None.

    Returns
    -------
    template, match, control : ndarray, shape (n_pairs,)
        The template pairs unchanged, and the match and control pairs
        reordered so pair ``k`` of every period is the same two cells.
    """
    call = "explained_variance_reactivation()"
    if control_correlations is None and labels["control_unit_ids"] is not None:
        raise ValueError(
            _format_error(
                f"{call} got control_unit_ids= but no control_correlations=.",
                why="Why: control_unit_ids labels the control period's "
                "correlations, and there are none to label.",
                fix="pass control_correlations= with it, or drop control_unit_ids=",
            )
        )
    needed = ["template_unit_ids", "match_unit_ids"]
    if control_correlations is not None:
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
    reference = _pair_labels(template, labels["template_unit_ids"], "template")

    def in_template_order(
        correlations: NDArray[np.float64], period: str
    ) -> NDArray[np.float64]:
        values = np.asarray(correlations, dtype=np.float64)
        period_labels = _pair_labels(values, labels[f"{period}_unit_ids"], period)
        order, only_template, only_period = _label_order(period_labels, reference)
        if only_template or only_period:
            raise ValueError(
                _format_error(
                    f"{call}: {period}_correlations must describe the same units "
                    f"as template_correlations; only in template_unit_ids: "
                    f"{only_template}, only in {period}_unit_ids: {only_period}.",
                    why="Why: EV compares the correlation of each pair of cells "
                    "across periods, so every pair needs a correlation in every "
                    "period.",
                    fix="select the units recorded in every period, keep the "
                    "same units in each count matrix (counts[:, keep] and "
                    "unit_ids[keep]), and compute pairwise_correlations() again",
                )
            )
        return _reorder_pairs(values, order)

    match = in_template_order(match_correlations, "match")
    control = (
        None
        if control_correlations is None
        else in_template_order(control_correlations, "control")
    )
    return template, match, control


def _pair_labels(
    correlations: NDArray[np.float64],
    unit_ids: NDArray[Any] | Sequence[Any] | None,
    period: str,
) -> NDArray[Any]:
    """Validate the unit labels of one period's pair vector.

    Parameters
    ----------
    correlations : ndarray, shape (n_pairs,)
        Pair correlations from :func:`pairwise_correlations`.
    unit_ids : array-like
        One label per unit, ``n_pairs == n_units * (n_units - 1) / 2``.
    period : {'template', 'match', 'control'}
        Period name for error messages.

    Returns
    -------
    ndarray, shape (n_units,)
        The resolved labels.
    """
    call = "explained_variance_reactivation()"
    argument = f"{period}_correlations"
    if correlations.ndim != 1:
        raise ValueError(
            _format_error(
                f"{call}: {argument} must be 1-D to be labelled by "
                f"{period}_unit_ids=, got shape {correlations.shape}.",
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
                why=f"Why: {period}_unit_ids labels the units whose pairs "
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
