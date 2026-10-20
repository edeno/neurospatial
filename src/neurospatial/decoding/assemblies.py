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
   ...     corr_behavior, corr_sleep
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


@dataclass(frozen=True, repr=False)
class AssemblyPattern(ReadOnlyArraysMixin, ResultMixin):
    """
    A detected cell assembly pattern.

    Attributes
    ----------
    weights : NDArray[np.float64], shape (n_neurons,)
        Weight of each neuron in this assembly. Higher absolute values
        indicate stronger participation. :func:`detect_assemblies` orients
        each pattern so its largest-magnitude weight is positive.
    member_indices : NDArray[np.int64]
        Column positions of neurons whose absolute-weight z-scores exceed
        z_threshold. These are the "core" members of the assembly.
    explained_variance_ratio : float
        Fraction of total variance explained by this assembly pattern.
        Higher values indicate more prominent assemblies.
    unit_ids : ndarray, shape (n_neurons,)
        Label of the unit each weight belongs to (read-only): the
        ``unit_ids=`` given to :func:`detect_assemblies`, else
        ``np.arange(n_neurons)``. Pass the same labels as ``unit_ids=`` to
        :func:`assembly_activation` or :func:`reactivation_strength` to match
        the columns of new counts by label instead of by position.
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


@dataclass(frozen=True, repr=False)
class AssemblyDetectionResult(ReadOnlyArraysMixin, ResultMixin):
    """
    Results from cell assembly detection.

    Attributes
    ----------
    patterns : list[AssemblyPattern]
        Detected assembly patterns, ordered by explained variance (highest first).
    activations : NDArray[np.float64], shape (n_assemblies, n_time_bins)
        Activation strength of each assembly over time.
    method : str
        Detection method used ('ica', 'pca', or 'nmf').
    n_significant : int
        Number of PCA eigenvalues above the Marchenko-Pastur reference
        threshold. This counts retained dimensions, not assemblies with
        thresholded core members. It may differ from len(patterns), for example
        when n_components was specified manually or no eigenvalue exceeds
        the threshold and the algorithm returns one exploratory component.
    eigenvalues : NDArray[np.float64]
        Eigenvalues from PCA, useful for scree plots.
    threshold : float
        Marchenko-Pastur threshold used for significance.
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

    ``activations`` carries the detection algorithm's projection scale, which
    differs between PCA, ICA and NMF. Use :func:`assembly_activation` to obtain
    a standardized projection for a chosen pattern and period. Standardized
    values describe relative activity; they do not by themselves establish
    significance or biological assembly identity.

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
    >>> plt.plot(result.activations[0])  # doctest: +SKIP
    >>> plt.ylabel("Activation (algorithm projection scale)")  # doctest: +SKIP
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
        if self.patterns:
            n_neurons = len(self.patterns[0].weights)
        else:
            n_neurons = 0 if self.unit_ids is None else len(self.unit_ids)
        labels = resolve_unit_ids(
            self.unit_ids, n_neurons, context="AssemblyDetectionResult"
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
        of new counts to :func:`assembly_activation` or
        :func:`reactivation_strength` matches columns by label. Default
        ``np.arange(n_neurons)``: column positions, which match new counts
        by position only.

    Returns
    -------
    AssemblyDetectionResult
        Contains detected patterns, activations, eigenvalues, and metadata.

    Raises
    ------
    ValueError
        If spike_counts has wrong shape or invalid parameters, or
        ``unit_ids`` does not hold one distinct label per column.

    Notes
    -----
    **Algorithm (Lopes-dos-Santos et al., 2013)**:

    1. Z-score spike counts for each neuron
    2. Compute correlation matrix
    3. Eigendecomposition to find significant dimensions (Marchenko-Pastur)
    4. Project onto significant PCs (and whiten their scores for ICA)
    5. Apply ICA/PCA/NMF to extract assembly patterns
    6. Threshold weights to identify core members
    7. Compute activation time series

    For ICA, requested components beyond the numerical rank of the data
    have zero weights, activations, and explained variance.

    **Sign convention**: ICA and PCA determine a pattern only up to its sign.
    Each pattern is flipped so that its largest-magnitude weight is positive,
    and its activation is flipped with it. Bins where the neurons with the
    largest positive weights fire together then have positive activation in
    ``result.activations`` and in :func:`assembly_activation`. NMF patterns
    are non-negative and are unchanged.

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
    # Validate input
    spike_counts = np.asarray(spike_counts, dtype=np.float64)

    if spike_counts.ndim != 2:
        raise ValueError(
            f"spike_counts must be 2D (n_time_bins, n_neurons), "
            f"got shape {spike_counts.shape}"
        )

    n_time_bins, n_neurons = spike_counts.shape

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

    # Work internally in (n_neurons, n_time_bins): the assembly structure is a
    # correlation over neurons, so z-score each neuron over time and correlate
    # neuron-wise.
    spike_counts = spike_counts.T

    # Z-score each neuron's activity
    spike_counts_z = stats.zscore(spike_counts, axis=1)

    # Handle neurons with zero variance (constant firing)
    zero_var_mask = np.isnan(spike_counts_z).any(axis=1)
    if np.any(zero_var_mask):
        n_zero = np.sum(zero_var_mask)
        warnings.warn(
            f"{n_zero} neurons have zero variance and will be excluded.",
            UserWarning,
            stacklevel=2,
        )
        spike_counts_z[zero_var_mask] = 0  # Set to zero rather than NaN

    # Compute correlation matrix using numpy
    corr_matrix = np.corrcoef(spike_counts_z)

    # Handle NaN in correlation matrix
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Eigendecomposition using scipy
    eigenvalues = linalg.eigvalsh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    # Compute Marchenko-Pastur threshold
    mp_threshold = marchenko_pastur_threshold(n_neurons, n_time_bins)
    n_significant = int(np.sum(eigenvalues > mp_threshold))

    # Determine number of components
    if n_components == "auto":
        n_comp = max(1, n_significant)  # At least 1 component
    else:
        n_comp = int(n_components)
        if n_comp < 1:
            raise ValueError(f"n_components must be >= 1, got {n_comp}")
        if n_comp > n_neurons:
            raise ValueError(
                f"n_components ({n_comp}) cannot exceed n_neurons ({n_neurons})"
            )

    # The factorization can yield at most min(n_neurons, n_time_bins)
    # components. Requesting more (short recordings) would index past the
    # SVD output and raise IndexError downstream; clamp with a warning.
    max_rank = min(n_neurons, n_time_bins)
    if n_comp > max_rank:
        warnings.warn(
            f"n_components ({n_comp}) exceeds the achievable rank "
            f"min(n_neurons, n_time_bins) = {max_rank}; using {max_rank}.",
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
    if algorithm == "pca":
        patterns, activations, explained_var = _detect_pca(spike_counts_z, n_comp)
    elif algorithm == "ica":
        patterns, activations, explained_var = _detect_ica(spike_counts_z, n_comp, seed)
    elif algorithm == "nmf":
        patterns, activations, explained_var = _detect_nmf(
            spike_counts,
            n_comp,
            seed,  # NMF uses original counts
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'ica', 'pca', or 'nmf'.")

    # ICA and eigenvector signs are arbitrary. Orient each pattern so its
    # largest-magnitude weight is positive, and flip its activation with it so
    # weights @ z-scored counts == activations still holds. An assembly's
    # events then raise its activation instead of lowering it.
    largest = patterns[np.arange(n_comp), np.argmax(np.abs(patterns), axis=1)]
    signs = np.where(largest < 0, -1.0, 1.0)
    patterns = patterns * signs[:, None]
    activations = activations * signs[:, None]

    # Create AssemblyPattern objects - vectorized z-score computation
    # Compute z-scores for all patterns at once: shape (n_comp, n_neurons)
    abs_patterns = np.abs(patterns)
    # stats.zscore along axis=1 normalizes each row (pattern) independently
    weights_z_all = stats.zscore(abs_patterns, axis=1)

    # Find member masks for all patterns at once
    member_masks = weights_z_all > z_threshold  # Shape: (n_comp, n_neurons)

    assembly_patterns = []
    for i in range(n_comp):
        member_indices = np.where(member_masks[i])[0].astype(np.int64)
        assembly_patterns.append(
            AssemblyPattern(
                weights=patterns[i],
                member_indices=member_indices,
                explained_variance_ratio=explained_var[i],
                unit_ids=labels,
                _unit_ids_generated=unit_ids_generated,
            )
        )

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
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Detect assemblies using PCA via scipy SVD on the correlation matrix.

    Following Lopes-dos-Santos et al. (2013), we find eigenvectors of the
    correlation matrix C = (1/T) * Z @ Z.T where Z is the z-scored spike count
    matrix. These eigenvectors represent assembly patterns.

    Returns
    -------
    patterns : array, shape (n_components, n_neurons)
    activations : array, shape (n_components, n_time_bins)
    explained_variance_ratio : array, shape (n_components,)
    """
    # Use SVD for numerical stability
    # SVD: Z = U @ S @ Vh where Z is (n_neurons, n_time_bins)
    # The correlation matrix C = Z @ Z.T / T has eigenvectors in U
    # (since C = U @ S^2 @ U.T / T from SVD)
    u, s, _vh = linalg.svd(spike_counts_z, full_matrices=False)

    # u has shape (n_neurons, min(n_neurons, n_time_bins))
    # Each column is a pattern (eigenvector of correlation matrix)
    # Select top n_components columns and transpose to get (n_components, n_neurons)
    patterns = u[:, :n_components].T  # (n_components, n_neurons)

    # Compute activations by projecting data onto patterns
    # activation[i, t] = sum_j patterns[i, j] * spike_counts_z[j, t]
    activations = patterns @ spike_counts_z  # (n_components, n_time_bins)

    # Explained variance ratio from singular values
    # Eigenvalues of C are s^2 / T
    total_var = np.sum(s**2)
    explained_var = (
        (s[:n_components] ** 2) / total_var
        if total_var > 0
        else np.zeros(n_components, dtype=np.float64)
    )

    return patterns, activations, explained_var


def _detect_ica(
    spike_counts_z: NDArray[np.float64],
    n_components: int,
    random_state: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Detect assemblies using ICA via sklearn FastICA.

    Following Lopes-dos-Santos et al. (2013), we first reduce dimensionality
    with PCA to the significant subspace, then apply ICA to find statistically
    independent assembly patterns.

    Returns
    -------
    patterns : array, shape (n_components, n_neurons)
    activations : array, shape (n_components, n_time_bins)
    explained_variance_ratio : array, shape (n_components,)
    """
    from sklearn.decomposition import FastICA

    # First reduce to n_components via PCA
    patterns_pca, activations_pca, _ = _detect_pca(spike_counts_z, n_components)

    # PCA decorrelates scores but leaves their variances equal to the
    # eigenvalues. FastICA must whiten these scores before its rotation.
    # Exclude null PCs so whitening cannot amplify numerical noise or divide
    # by zero on constant / rank-deficient recordings.
    score_norms = np.linalg.norm(activations_pca, axis=1)
    rank_tol = np.finfo(np.float64).eps * max(spike_counts_z.shape) * score_norms.max()
    live = score_norms > rank_tol
    n_live = int(live.sum())
    patterns = np.zeros((n_components, spike_counts_z.shape[0]), dtype=np.float64)
    activations = np.zeros((n_components, spike_counts_z.shape[1]), dtype=np.float64)
    explained_var = np.zeros(n_components, dtype=np.float64)
    if n_live == 0:
        return patterns, activations, explained_var

    ica = FastICA(
        n_components=n_live,
        random_state=random_state,
        max_iter=1000,
        whiten="unit-variance",
    )
    sources = ica.fit_transform(activations_pca[live].T)
    activations[:n_live] = sources.T

    # components_ includes the whitening transform as well as the ICA
    # rotation. Compose it with the projection to preserve neuron-space
    # weights @ z-scored counts == activations.
    patterns[:n_live] = ica.components_ @ patterns_pca[live]

    # Unit-variance, uncorrelated sources contribute the squared norm of
    # their neuron-space mixing column to the reconstructed variance.
    # Individual PCA variances do not correspond to the rotated ICA sources.
    loadings = patterns_pca[live].T @ ica.mixing_
    total_var = np.sum(spike_counts_z**2) / spike_counts_z.shape[1]
    explained_var[:n_live] = np.sum(loadings**2, axis=0) / total_var

    return patterns, activations, explained_var


def _detect_nmf(
    spike_counts: NDArray[np.float64],
    n_components: int,
    random_state: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Detect assemblies using NMF via sklearn.

    Note: Uses original spike counts (not z-scored) since NMF requires
    non-negative input.

    Returns
    -------
    patterns : array, shape (n_components, n_neurons)
    activations : array, shape (n_components, n_time_bins)
    explained_variance_ratio : array, shape (n_components,)
    """
    from sklearn.decomposition import NMF

    # Ensure non-negative
    spike_counts_nn = np.maximum(spike_counts, 0)

    nmf = NMF(
        n_components=n_components,
        random_state=random_state,
        max_iter=500,
        init="nndsvda",  # Better initialization
    )

    # NMF: X ≈ W @ H
    # W: (n_neurons, n_components) - basis vectors
    # H: (n_components, n_time_bins) - activations
    basis = nmf.fit_transform(spike_counts_nn)
    activations = nmf.components_

    patterns = basis.T  # (n_components, n_neurons)

    # Compute explained variance
    reconstruction = basis @ activations
    total_var = np.var(spike_counts_nn)
    residual_var = np.var(spike_counts_nn - reconstruction)
    total_explained = 1 - residual_var / total_var if total_var > 0 else 0

    # Approximate per-component variance (proportional to pattern norm)
    pattern_norms = np.linalg.norm(patterns, axis=1)
    total_norm = np.sum(pattern_norms)
    if total_norm > 0:
        explained_var = pattern_norms / total_norm * total_explained
    else:
        explained_var = np.zeros(n_components)

    return patterns, activations, explained_var


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

    Projects new spike count data onto an assembly pattern to measure
    activation strength at each time bin.

    Parameters
    ----------
    spike_counts : NDArray[np.float64], shape (n_time_bins, n_neurons)
        Population activity of the pattern's units.
    pattern : AssemblyPattern
        Assembly pattern with neuron weights.
    z_score_input : bool, default=True
        If True, z-score spike counts before projection.
        Set to False if data is already normalized.
    unit_ids : array-like, shape (n_neurons,), optional
        Label of the unit in each column of ``spike_counts``. The columns are
        then matched to the pattern's weights by label, in any order, and
        must hold exactly the pattern's units; the pattern needs real labels
        (``detect_assemblies(..., unit_ids=...)``). Without it, column ``i``
        is the pattern's unit ``i``, in detection order.

    Returns
    -------
    activation : NDArray[np.float64], shape (n_time_bins,)
        Projection standardized to zero mean and unit standard deviation
        within this period (all zeros for a constant projection). Large
        magnitudes indicate unusual activity relative to that period's mean;
        they do not supply calibrated tail probabilities.

    Raises
    ------
    ValueError
        If spike_counts has wrong number of neurons, or with ``unit_ids``,
        if the pattern's labels were generated or the labels of the two sides
        differ.

    Notes
    -----
    **Activation Computation**:

    Activation is computed as the dot product between pattern weights
    and normalized spike counts:

    .. math::

        a(t) = \\sum_i w_i \\cdot z_i(t)

    where :math:`w_i` are pattern weights and :math:`z_i(t)` are
    z-scored spike counts when ``z_score_input=True``. The projected series
    is then standardized again within the supplied period, even when
    ``z_score_input=False``. Use :func:`reactivation_strength` to compare
    magnitudes between periods against a shared template baseline.

    **Interpretation**:

    - activation > 2: more than two standard deviations above this period's mean
    - activation < -2: more than two standard deviations below this period's mean
    - activation ≈ 0: near this period's mean projection

    Standardization does not make an arbitrary activity distribution normal.
    A threshold of 2 is a descriptive activity cutoff, not an automatic
    p < 0.05 test. Assess significance using a suitable null/control analysis.

    Examples
    --------
    A sparse one-neuron projection has 10% of bins above 2 despite unit variance:

    >>> import numpy as np
    >>> from neurospatial.decoding import AssemblyPattern, assembly_activation
    >>> counts = np.concatenate([np.zeros(90), np.ones(10)])[:, None]
    >>> pattern = AssemblyPattern(np.array([1.0]), np.array([0]), 1.0)
    >>> activation = assembly_activation(counts, pattern)
    >>> np.allclose([activation.mean(), activation.std()], [0.0, 1.0])
    True
    >>> float(np.mean(activation > 2.0))
    0.1

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
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)

    if spike_counts.ndim != 2:
        raise ValueError(
            f"spike_counts must be 2D (n_time_bins, n_neurons), "
            f"got shape {spike_counts.shape}"
        )

    spike_counts = _columns_by_label(
        spike_counts, unit_ids, pattern, call="assembly_activation()"
    )
    _n_time_bins, n_neurons = spike_counts.shape

    if n_neurons != len(pattern.weights):
        raise ValueError(
            f"spike_counts has {n_neurons} neurons but pattern has "
            f"{len(pattern.weights)} weights. Must match."
        )

    # Work internally in (n_neurons, n_time_bins).
    spike_counts = spike_counts.T

    # Z-score if requested
    if z_score_input:
        spike_counts_z = stats.zscore(spike_counts, axis=1)
        # Handle zero variance
        spike_counts_z = np.nan_to_num(spike_counts_z, nan=0.0)
    else:
        spike_counts_z = spike_counts

    # Project onto pattern
    activation = pattern.weights @ spike_counts_z  # (n_time_bins,)

    # Z-score activation
    activation_z = stats.zscore(activation)
    activation_z = np.nan_to_num(activation_z, nan=0.0)

    return np.asarray(activation_z, dtype=np.float64)


def _columns_by_label(
    spike_counts: NDArray[np.float64],
    unit_ids: NDArray[Any] | Sequence[Any] | None,
    pattern: AssemblyPattern,
    *,
    call: str,
    argument: str = "spike_counts",
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
    call, argument : str
        Function and argument names for error messages.

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
                f"{call} got unit_ids= for {argument}, but the pattern has no "
                "unit_ids; its labels are column positions generated by "
                "detect_assemblies().",
                why="Why: columns are matched to the pattern's weights by label, "
                "and a column position does not identify the same cell in two "
                "count matrices.",
                fix="pass the sorted-unit labels as unit_ids= to "
                "detect_assemblies() as well, or drop unit_ids= here to pair "
                "columns by position",
            )
        )
    column_of = {label: column for column, label in enumerate(labels.tolist())}
    pattern_labels = np.asarray(pattern.unit_ids).tolist()
    pattern_set = set(pattern_labels)
    only_pattern = [label for label in pattern_labels if label not in column_of]
    only_counts = [label for label in labels.tolist() if label not in pattern_set]
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
                "unit_ids[keep] with keep = np.isin(unit_ids, pattern.unit_ids); "
                "if pattern units were not recorded, detect the assemblies again "
                "on the units recorded in both periods",
            )
        )
    order = np.array([column_of[label] for label in pattern_labels], dtype=np.intp)
    reordered: NDArray[np.float64] = spike_counts[:, order]
    return reordered


# =============================================================================
# Pairwise Correlations and Explained Variance
# =============================================================================


def pairwise_correlations(
    spike_counts: NDArray[np.float64],
    *,
    z_score_input: bool = True,
) -> NDArray[np.float64]:
    """
    Compute pairwise correlations between neurons.

    Parameters
    ----------
    spike_counts : NDArray[np.float64], shape (n_time_bins, n_neurons)
        Binned spike counts.
    z_score_input : bool, default=True
        If True, z-score spike counts before computing correlations.

    Returns
    -------
    correlations : NDArray[np.float64], shape (n_pairs,)
        Upper triangle of correlation matrix, flattened.
        n_pairs = n_neurons * (n_neurons - 1) / 2

    Notes
    -----
    Returns only the upper triangle (excluding diagonal) to avoid
    redundancy. Use `numpy.triu_indices` to reconstruct the full matrix.

    Pair ``k`` is the pair of columns ``np.triu_indices(n_neurons, k=1)``
    gives at ``k``, so correlations of two periods describe the same pairs
    of cells only when both count matrices have the same units in the same
    column order.

    Examples
    --------
    >>> corr = pairwise_correlations(spike_counts)  # doctest: +SKIP
    >>> n_neurons = spike_counts.shape[1]  # doctest: +SKIP
    >>> n_pairs = n_neurons * (n_neurons - 1) // 2  # doctest: +SKIP
    >>> assert len(corr) == n_pairs  # doctest: +SKIP
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)

    if spike_counts.ndim != 2:
        raise ValueError(
            f"spike_counts must be 2D (n_time_bins, n_neurons), "
            f"got shape {spike_counts.shape}"
        )

    # Work internally in (n_neurons, n_time_bins).
    spike_counts = spike_counts.T
    n_neurons = spike_counts.shape[0]

    if z_score_input:
        spike_counts_z = stats.zscore(spike_counts, axis=1)
        spike_counts_z = np.nan_to_num(spike_counts_z, nan=0.0)
    else:
        spike_counts_z = spike_counts

    # Compute correlation matrix using numpy
    corr_matrix = np.corrcoef(spike_counts_z)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n_neurons, k=1)
    correlations = corr_matrix[triu_indices]

    return correlations


def reactivation_strength(
    template_counts: NDArray[np.float64],
    match_counts: NDArray[np.float64],
    pattern: AssemblyPattern,
    *,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> float:
    """
    Compare assembly activation magnitude between template and match periods.

    Computes the ratio of mean activation strength in the match period
    relative to the template period. Values > 1 indicate stronger activation
    during match (potential reactivation).

    Parameters
    ----------
    template_counts : NDArray[np.float64], shape (n_template_bins, n_neurons)
        Spike counts during template period (e.g., behavior).
    match_counts : NDArray[np.float64], shape (n_match_bins, n_neurons)
        Spike counts during match period (e.g., sleep).
    pattern : AssemblyPattern
        Assembly pattern to measure.
    unit_ids : array-like, shape (n_neurons,), optional
        Label of the unit in each column of both ``template_counts`` and
        ``match_counts``, which share one column order (as from
        ``bin_spikes_in_time`` on one list of spike trains). The columns are
        then matched to the pattern's weights by label, as in
        :func:`assembly_activation`. Without it, column ``i`` is the
        pattern's unit ``i``.

    Returns
    -------
    float
        Ratio of mean absolute activation in match vs template period.
        Values > 1 indicate stronger activation during match period.
        Returns 0 if template activation is zero.

    Notes
    -----
    This is a simple measure of reactivation based on comparing mean
    activation magnitudes. For more sophisticated analysis of correlation
    structure preservation, use `explained_variance_reactivation` with
    pairwise correlations.

    **Normalization**: Both periods are normalized against a *single*
    baseline — the template period's per-neuron mean and standard
    deviation — and projected directly onto the assembly pattern. Neither
    period is z-scored independently, so absolute magnitude survives: a
    match period with larger projected activity yields ``strength > 1``.

    **Interpretation**:

    - strength > 1: Assembly is more active during match period
    - strength ≈ 1: Similar activity levels in both periods
    - strength < 1: Assembly is less active during match period
    - strength = 0: No activation in template period

    Examples
    --------
    >>> strength = reactivation_strength(
    ...     counts_behavior, counts_sleep, pattern
    ... )  # doctest: +SKIP
    >>> if strength > 1.0:  # doctest: +SKIP
    ...     print("Stronger activation during sleep")  # doctest: +SKIP
    """
    # Compute activations WITHOUT any per-period standardization, so the
    # absolute magnitude of each period's activity survives. The default
    # assembly_activation path standardizes each period independently (an
    # inner input z-score AND an outer z-score of the projection), which
    # erases the very magnitude difference this metric is meant to detect
    # (strength would sit at ~1 no matter how much stronger the match
    # activity is). Instead normalize both periods against a *single*
    # baseline (the template mean/std) and project directly onto the
    # pattern, so a larger match-period projection produces strength > 1.
    # Inputs are (n_time_bins, n_neurons); work internally in (n_neurons, n_time).
    call = "reactivation_strength()"
    template_counts = _columns_by_label(
        np.asarray(template_counts, dtype=np.float64),
        unit_ids,
        pattern,
        call=call,
        argument="template_counts",
    ).T
    match_counts = _columns_by_label(
        np.asarray(match_counts, dtype=np.float64),
        unit_ids,
        pattern,
        call=call,
        argument="match_counts",
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

    mean = template_counts.mean(axis=1, keepdims=True)
    std = template_counts.std(axis=1, keepdims=True)
    std = np.where(std > 1e-10, std, 1.0)

    template_norm = (template_counts - mean) / std
    match_norm = (match_counts - mean) / std

    # Project onto the assembly pattern on the shared (template) scale.
    act_template = pattern.weights @ template_norm
    act_match = pattern.weights @ match_norm

    # Compare mean absolute activation magnitudes on the shared scale.
    mean_template = np.mean(np.abs(act_template))
    mean_match = np.mean(np.abs(act_match))

    if mean_template > 1e-10:
        return float(mean_match / mean_template)
    else:
        return 0.0


def explained_variance_reactivation(
    template_correlations: NDArray[np.float64],
    match_correlations: NDArray[np.float64],
    *,
    control_correlations: NDArray[np.float64] | None = None,
) -> ExplainedVarianceResult:
    """
    Compute explained variance for pairwise correlation reactivation.

    Measures how well pairwise correlations during 'match' period
    are predicted by correlations from 'template' period.

    Parameters
    ----------
    template_correlations : NDArray[np.float64], shape (n_pairs,)
        Pairwise correlations during template (e.g., behavior).
        From `pairwise_correlations()`. Pairs are matched by position, so
        every period's counts must have the same units in the same column
        order.
    match_correlations : NDArray[np.float64], shape (n_pairs,)
        Pairwise correlations during match (e.g., sleep/rest).
    control_correlations : NDArray[np.float64], optional
        Control period correlations (e.g., pre-behavior baseline).
        If provided, computes partial correlation controlling for baseline.

    Returns
    -------
    ExplainedVarianceResult
        Contains EV, reversed EV, partial correlation, and n_pairs.

    Raises
    ------
    ValueError
        If correlation arrays have different lengths.

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
    >>> # Compute correlations
    >>> corr_pre = pairwise_correlations(counts_pre_behavior)  # doctest: +SKIP
    >>> corr_behavior = pairwise_correlations(counts_behavior)  # doctest: +SKIP
    >>> corr_sleep = pairwise_correlations(counts_sleep)  # doctest: +SKIP
    >>>
    >>> # Measure reactivation with baseline control
    >>> result = explained_variance_reactivation(  # doctest: +SKIP
    ...     corr_behavior, corr_sleep, control_correlations=corr_pre
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
    template_correlations = np.asarray(template_correlations, dtype=np.float64)
    match_correlations = np.asarray(match_correlations, dtype=np.float64)

    if len(template_correlations) != len(match_correlations):
        raise ValueError(
            f"template and match correlations must have same length. "
            f"Got {len(template_correlations)} and {len(match_correlations)}."
        )

    n_pairs = len(template_correlations)

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
    r_tm = np.nan_to_num(np.corrcoef(template_valid, match_valid)[0, 1], nan=0.0)

    if control_correlations is not None:
        control_valid = control_correlations[valid_mask]
        r_tc = np.nan_to_num(np.corrcoef(template_valid, control_valid)[0, 1], nan=0.0)
        r_mc = np.nan_to_num(np.corrcoef(match_valid, control_valid)[0, 1], nan=0.0)

        # EV: variance of match correlations explained by template, after
        # partialling out the control (pre-behavior) baseline.
        #   r_partial(template, match | control)
        denom_ev = np.sqrt((1.0 - r_tc**2) * (1.0 - r_mc**2))
        partial_corr = (r_tm - r_tc * r_mc) / denom_ev if denom_ev > 1e-10 else 0.0
        ev = partial_corr**2

        # Reverse time by swapping pre (control) and post (match):
        # r_partial(template, control | match).
        denom_rev = np.sqrt((1.0 - r_tm**2) * (1.0 - r_mc**2))
        rev_partial = (r_tc - r_tm * r_mc) / denom_rev if denom_rev > 1e-10 else 0.0
        rev = rev_partial**2
    else:
        # No control: EV is the plain squared correlation. REV reduces to
        # the same value because there is no asymmetry to break — the
        # control is what makes EV != REV meaningful. This is intentional,
        # not the old hardcoded `rev = ev` shortcut.
        partial_corr = r_tm
        ev = r_tm**2
        rev = r_tm**2

    return ExplainedVarianceResult(
        explained_variance=float(ev),
        reversed_ev=float(rev),
        partial_correlation=float(partial_corr),
        n_pairs=int(np.sum(valid_mask)),
    )
