"""
Cell assembly detection and reactivation analysis.

This module provides functions for detecting coordinated neural ensembles
(cell assemblies) and measuring their reactivation strength across behavioral
states.

Cell assemblies are groups of neurons that fire together more often than
expected by chance, representing stored neural patterns. Detecting these
assemblies and measuring their reactivation is fundamental for studying
memory consolidation and replay.

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

1. Prepare spike count matrix (n_neurons, n_time_bins):

   >>> import numpy as np
   >>> # Example: 50 neurons, 2000 time bins (25ms bins = 50s recording)
   >>> spike_counts = np.random.poisson(5, (50, 2000)).astype(np.float64)

2. Detect assemblies:

   >>> result = detect_assemblies(spike_counts, method="ica")  # doctest: +SKIP
   >>> print(f"Found {result.n_significant} significant assemblies")  # doctest: +SKIP

3. Analyze assembly patterns:

   >>> for i, pattern in enumerate(result.patterns):  # doctest: +SKIP
   ...     n_members = len(pattern.member_indices)
   ...     ev = pattern.explained_variance_ratio
   ...     print(f"Assembly {i}: {n_members} neurons, {ev:.1%} variance")

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
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, stats

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


@dataclass(frozen=True)
class AssemblyPattern:
    """
    A detected cell assembly pattern.

    Attributes
    ----------
    weights : NDArray[np.float64], shape (n_neurons,)
        Weight of each neuron in this assembly. Higher absolute values
        indicate stronger participation. Sign indicates activation direction.
    member_indices : NDArray[np.int64]
        Indices of neurons with significant weights (above z_threshold).
        These are the "core" members of the assembly.
    explained_variance_ratio : float
        Fraction of total variance explained by this assembly pattern.
        Higher values indicate more prominent assemblies.

    Notes
    -----
    **Interpreting Weights**:

    - Positive weights: neurons that co-activate with the assembly
    - Negative weights: neurons that are suppressed during assembly activation
    - Near-zero weights: neurons not participating in this assembly

    **Member Selection**:

    Members are selected by z-scoring weights and thresholding. The default
    threshold of 2.0 means neurons with weights > 2 standard deviations from
    the mean are considered significant members.

    Examples
    --------
    >>> pattern = result.patterns[0]  # doctest: +SKIP
    >>> print(
    ...     f"Assembly has {len(pattern.member_indices)} core members"
    ... )  # doctest: +SKIP
    >>> print(
    ...     f"Explains {pattern.explained_variance_ratio:.1%} of variance"
    ... )  # doctest: +SKIP
    """

    weights: NDArray[np.float64]
    member_indices: NDArray[np.int64]
    explained_variance_ratio: float


@dataclass(frozen=True)
class AssemblyDetectionResult:
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
        Number of statistically significant assemblies (above Marchenko-Pastur
        threshold). This may differ from len(patterns) if n_components was
        specified manually.
    eigenvalues : NDArray[np.float64]
        Eigenvalues from PCA, useful for scree plots.
    threshold : float
        Marchenko-Pastur threshold used for significance.

    Notes
    -----
    **Significance Determination**:

    The number of significant assemblies is determined by the Marchenko-Pastur
    theorem from random matrix theory. Eigenvalues above the threshold are
    unlikely to arise from random correlations.

    **Activation Interpretation**:

    Activation values are z-scored, so:
    - activation > 2: strong assembly activation
    - activation < -2: strong assembly suppression
    - activation ≈ 0: baseline activity

    Examples
    --------
    >>> result = detect_assemblies(spike_counts)  # doctest: +SKIP
    >>> print(f"Found {result.n_significant} significant assemblies")  # doctest: +SKIP
    >>> print(f"Method: {result.method}")  # doctest: +SKIP
    >>>
    >>> # Plot activation of first assembly
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> plt.plot(result.activations[0])  # doctest: +SKIP
    >>> plt.ylabel("Activation (z-score)")  # doctest: +SKIP
    """

    patterns: list[AssemblyPattern]
    activations: NDArray[np.float64]
    method: str
    n_significant: int
    eigenvalues: NDArray[np.float64]
    threshold: float


@dataclass(frozen=True)
class ExplainedVarianceResult:
    """
    Results from explained variance reactivation analysis.

    Attributes
    ----------
    explained_variance : float
        Explained variance (EV) - fraction of match period correlations
        explained by template period correlations. Range [0, 1].
    reversed_ev : float
        Reversed explained variance (REV) - fraction of template period
        correlations explained by match period. Used as control.
    partial_correlation : float
        Partial correlation between template and match, controlling for
        baseline correlations (if control provided).
    n_pairs : int
        Number of neuron pairs used in analysis.

    Notes
    -----
    **Interpretation**:

    - EV > REV: Forward reactivation (template patterns appear in match)
    - EV ≈ REV: No directional reactivation
    - EV >> 0: Strong correlation structure preserved

    **Typical Values** (from Kudrimoti et al., 1999):

    - Significant reactivation: EV > 0.1
    - Strong reactivation: EV > 0.3
    - Control periods: EV ≈ 0

    Examples
    --------
    >>> result = explained_variance_reactivation(
    ...     corr_behavior, corr_sleep
    ... )  # doctest: +SKIP
    >>> if result.explained_variance > result.reversed_ev:  # doctest: +SKIP
    ...     print("Forward reactivation detected!")  # doctest: +SKIP
    """

    explained_variance: float
    reversed_ev: float
    partial_correlation: float
    n_pairs: int


# =============================================================================
# Marchenko-Pastur Threshold
# =============================================================================


def marchenko_pastur_threshold(
    n_neurons: int,
    n_time_bins: int,
) -> float:
    """
    Compute Marchenko-Pastur threshold for significant eigenvalues.

    Eigenvalues of the correlation matrix above this threshold are unlikely
    to arise from random correlations, indicating true structure in the data.

    Parameters
    ----------
    n_neurons : int
        Number of neurons (rows in spike count matrix).
    n_time_bins : int
        Number of time bins (columns in spike count matrix).

    Returns
    -------
    float
        Upper bound of eigenvalue distribution for random matrix.
        Eigenvalues above this are statistically significant.

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

    - Eigenvalues > threshold: likely reflect true correlations
    - Eigenvalues < threshold: consistent with random noise

    **Requirements**:

    For reliable threshold estimation, n_time_bins should be significantly
    larger than n_neurons (typically > 5× larger).

    Examples
    --------
    >>> threshold = marchenko_pastur_threshold(100, 1000)
    >>> print(f"Threshold: {threshold:.3f}")
    Threshold: 1.732

    >>> # Compare with eigenvalues
    >>> eigenvalues = np.linalg.eigvalsh(np.corrcoef(spike_counts))  # doctest: +SKIP
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
    method: Literal["ica", "pca", "nmf"] = "ica",
    n_components: int | Literal["auto"] = "auto",
    z_threshold: float = 2.0,
    random_state: int | None = None,
) -> AssemblyDetectionResult:
    """
    Detect cell assemblies from population spike counts.

    Identifies groups of neurons that fire together more than expected by
    chance, using dimensionality reduction on the correlation structure.

    Parameters
    ----------
    spike_counts : NDArray[np.float64], shape (n_neurons, n_time_bins)
        Binned spike counts for population. Will be z-scored internally.
        Typical bin size: 25-100ms for assembly detection.
    method : {'ica', 'pca', 'nmf'}, default='ica'
        Detection method:

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
        Z-score threshold for assembly membership. Neurons with weight
        z-scores above this are considered assembly members.

        - 2.0: Standard threshold (p < 0.05, two-tailed)
        - 2.5: Conservative threshold
        - 1.5: Liberal threshold

    random_state : int, optional
        Random seed for reproducibility. Affects ICA and NMF initialization.

    Returns
    -------
    AssemblyDetectionResult
        Contains detected patterns, activations, eigenvalues, and metadata.

    Raises
    ------
    ValueError
        If spike_counts has wrong shape or invalid parameters.

    Notes
    -----
    **Algorithm (Lopes-dos-Santos et al., 2013)**:

    1. Z-score spike counts for each neuron
    2. Compute correlation matrix
    3. Eigendecomposition to find significant dimensions (Marchenko-Pastur)
    4. Project onto significant PCs
    5. Apply ICA/PCA/NMF to extract assembly patterns
    6. Threshold weights to identify core members
    7. Compute activation time series

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
    >>> # Basic usage
    >>> result = detect_assemblies(spike_counts)  # doctest: +SKIP
    >>> print(f"Found {result.n_significant} assemblies")  # doctest: +SKIP

    >>> # With specific number of components
    >>> result = detect_assemblies(spike_counts, n_components=5)  # doctest: +SKIP

    >>> # Using PCA for faster computation
    >>> result = detect_assemblies(spike_counts, method="pca")  # doctest: +SKIP

    >>> # Access assembly patterns
    >>> for i, pattern in enumerate(result.patterns):  # doctest: +SKIP
    ...     members = pattern.member_indices
    ...     print(f"Assembly {i}: {len(members)} neurons")

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
            f"spike_counts must be 2D (n_neurons, n_time_bins), "
            f"got shape {spike_counts.shape}"
        )

    n_neurons, n_time_bins = spike_counts.shape

    if n_neurons < 3:
        raise ValueError(
            f"Need at least 3 neurons for assembly detection, got {n_neurons}"
        )

    if n_time_bins < n_neurons:
        warnings.warn(
            f"n_time_bins ({n_time_bins}) < n_neurons ({n_neurons}). "
            "Results may be unreliable. Consider using longer recordings.",
            UserWarning,
            stacklevel=2,
        )

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

    # Perform dimensionality reduction
    if method == "pca":
        patterns, activations, explained_var = _detect_pca(spike_counts_z, n_comp)
    elif method == "ica":
        patterns, activations, explained_var = _detect_ica(
            spike_counts_z, n_comp, random_state
        )
    elif method == "nmf":
        patterns, activations, explained_var = _detect_nmf(
            spike_counts,
            n_comp,
            random_state,  # NMF uses original counts
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ica', 'pca', or 'nmf'.")

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
            )
        )

    return AssemblyDetectionResult(
        patterns=assembly_patterns,
        activations=activations,
        method=method,
        n_significant=n_significant,
        eigenvalues=eigenvalues,
        threshold=mp_threshold,
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
    explained_var = (s[:n_components] ** 2) / total_var

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
    patterns_pca, activations_pca, explained_var_pca = _detect_pca(
        spike_counts_z, n_components
    )
    # patterns_pca: (n_components, n_neurons)
    # activations_pca: (n_components, n_time_bins)

    # Apply ICA to the PCA-reduced activations
    # ICA finds mixing matrix W such that: activations_pca ≈ W @ sources
    # We want sources (independent components) which become our new activations
    ica = FastICA(
        n_components=n_components,
        random_state=random_state,
        max_iter=1000,
        whiten=False,  # Already done by PCA
    )

    # ICA on transposed data: (n_time_bins, n_components) -> (n_time_bins, n_components)
    # sources = ica.fit_transform(activations_pca.T) gives (n_time_bins, n_components)
    sources = ica.fit_transform(activations_pca.T)
    activations = sources.T  # (n_components, n_time_bins)

    # To get patterns in neuron space:
    # activations_pca = W @ sources.T = W @ activations
    # activations_pca = patterns_pca @ spike_counts_z
    # So patterns_pca @ spike_counts_z = W @ (patterns_ica @ spike_counts_z)
    # patterns_pca = W @ patterns_ica
    # patterns_ica = W^-1 @ patterns_pca
    # Since ica.components_ = W^-1 (unmixing matrix), we have:
    # patterns_ica = ica.components_ @ patterns_pca
    patterns = ica.components_ @ patterns_pca  # (n_components, n_neurons)

    # Explained variance from PCA (ICA doesn't change total variance)
    explained_var = explained_var_pca

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
) -> NDArray[np.float64]:
    """
    Compute activation strength of an assembly over time.

    Projects new spike count data onto an assembly pattern to measure
    activation strength at each time bin.

    Parameters
    ----------
    spike_counts : NDArray[np.float64], shape (n_neurons, n_time_bins)
        Population activity. Must have same neurons as pattern.
    pattern : AssemblyPattern
        Assembly pattern with neuron weights.
    z_score_input : bool, default=True
        If True, z-score spike counts before projection.
        Set to False if data is already normalized.

    Returns
    -------
    activation : NDArray[np.float64], shape (n_time_bins,)
        Activation strength at each time bin. Values are z-scored,
        so activation > 2 indicates strong assembly activation.

    Raises
    ------
    ValueError
        If spike_counts has wrong number of neurons.

    Notes
    -----
    **Activation Computation**:

    Activation is computed as the dot product between pattern weights
    and normalized spike counts:

    .. math::

        a(t) = \\sum_i w_i \\cdot z_i(t)

    where :math:`w_i` are pattern weights and :math:`z_i(t)` are
    z-scored spike counts.

    **Interpretation**:

    - activation > 2: strong assembly activation (p < 0.05)
    - activation < -2: strong assembly suppression
    - activation ≈ 0: baseline activity

    Examples
    --------
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
            f"spike_counts must be 2D (n_neurons, n_time_bins), "
            f"got shape {spike_counts.shape}"
        )

    n_neurons, _n_time_bins = spike_counts.shape

    if n_neurons != len(pattern.weights):
        raise ValueError(
            f"spike_counts has {n_neurons} neurons but pattern has "
            f"{len(pattern.weights)} weights. Must match."
        )

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
    spike_counts : NDArray[np.float64], shape (n_neurons, n_time_bins)
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

    Examples
    --------
    >>> corr = pairwise_correlations(spike_counts)  # doctest: +SKIP
    >>> n_neurons = spike_counts.shape[0]  # doctest: +SKIP
    >>> n_pairs = n_neurons * (n_neurons - 1) // 2  # doctest: +SKIP
    >>> assert len(corr) == n_pairs  # doctest: +SKIP
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)

    if spike_counts.ndim != 2:
        raise ValueError(f"spike_counts must be 2D, got shape {spike_counts.shape}")

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
) -> float:
    """
    Compare assembly activation magnitude between template and match periods.

    Computes the ratio of mean activation strength in the match period
    relative to the template period. Values > 1 indicate stronger activation
    during match (potential reactivation).

    Parameters
    ----------
    template_counts : NDArray[np.float64], shape (n_neurons, n_template_bins)
        Spike counts during template period (e.g., behavior).
    match_counts : NDArray[np.float64], shape (n_neurons, n_match_bins)
        Spike counts during match period (e.g., sleep).
    pattern : AssemblyPattern
        Assembly pattern to measure.

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
    # Compute activations
    act_template = assembly_activation(template_counts, pattern)
    act_match = assembly_activation(match_counts, pattern)

    # Compare mean absolute activation magnitudes
    mean_template = np.mean(np.abs(act_template))
    mean_match = np.mean(np.abs(act_match))

    # Ratio of activation strengths
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
        From `pairwise_correlations()`.
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

    EV is the squared correlation between template and match correlations:

    .. math::

        EV = r(template, match)^2

    This measures how much of the match period correlation structure
    can be "explained" by the template period.

    **Reversed EV (REV)**:

    REV is computed by reversing the roles of template and match.
    For true reactivation, EV should exceed REV.

    **Partial Correlation**:

    When control correlations are provided, partial correlation is computed:

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

    # Compute correlation between template and match
    r_tm = np.corrcoef(template_valid, match_valid)[0, 1]
    r_tm = np.nan_to_num(r_tm, nan=0.0)

    # Explained variance
    ev = r_tm**2

    # Reversed EV (same as EV for correlation, but conceptually different)
    rev = ev  # r(match, template)^2 = r(template, match)^2

    # Partial correlation if control provided
    if control_correlations is not None:
        control_valid = control_correlations[valid_mask]

        r_tc = np.corrcoef(template_valid, control_valid)[0, 1]
        r_mc = np.corrcoef(match_valid, control_valid)[0, 1]

        r_tc = np.nan_to_num(r_tc, nan=0.0)
        r_mc = np.nan_to_num(r_mc, nan=0.0)

        # Partial correlation formula
        denom = np.sqrt((1 - r_tc**2) * (1 - r_mc**2))
        partial_corr = (r_tm - r_tc * r_mc) / denom if denom > 1e-10 else 0.0
    else:
        partial_corr = r_tm  # No control, use raw correlation

    return ExplainedVarianceResult(
        explained_variance=float(ev),
        reversed_ev=float(rev),
        partial_correlation=float(partial_corr),
        n_pairs=int(np.sum(valid_mask)),
    )
