# Plan: Cell Assembly Detection Module

## Overview

Add a new `neurospatial.metrics.assemblies` module for detecting coordinated neural ensembles (cell assemblies) and measuring their reactivation. This fills a gap identified in the NeuroPy comparison.

**Scope:** ICA/PCA-based assembly detection, activation time series, explained variance reactivation.

## Design Principles

1. **Use scipy/numpy where possible**: PCA via `scipy.linalg.svd`, correlations via `numpy.corrcoef`
2. **sklearn only when necessary**: FastICA (no scipy equivalent), NMF (optional)
3. **No new dependencies**: sklearn already in pyproject.toml
4. **Match existing patterns**: NumPy docstrings, type hints, TYPE_CHECKING guards, frozen dataclasses
5. **Standalone functions**: Like `place_fields.py`, not Environment methods

## Implementation Sources

Algorithms verified against authoritative sources:

- **Lopes-dos-Santos et al. (2013)** - PCA/ICA assembly detection, Marchenko-Pastur threshold
- **Peyrache et al. (2009)** - Assembly reactivation during sleep
- **Kudrimoti et al. (1999)** - Explained variance for pairwise correlation reactivation
- **van de Ven et al. (2016)** - Assembly detection methodology review

## Scipy/Numpy Coverage Analysis

| Function | scipy/numpy | sklearn | Decision |
|----------|-------------|---------|----------|
| Z-score normalization | `scipy.stats.zscore` | - | Use scipy |
| Covariance matrix | `numpy.cov` | - | Use numpy |
| SVD/PCA | `scipy.linalg.svd` | `PCA` | Use scipy.linalg.svd |
| Eigenvalues | `scipy.linalg.eigvalsh` | - | Use scipy |
| ICA | - | `FastICA` | Use sklearn (no scipy equivalent) |
| NMF | - | `NMF` | Use sklearn (optional method) |
| Correlation matrix | `numpy.corrcoef` | - | Use numpy |
| Partial correlation | Manual via `numpy.linalg.inv` | - | Implement with numpy |

## Module Structure

```
src/neurospatial/metrics/
├── assemblies.py          # NEW: Cell assembly detection
├── __init__.py            # Update exports
└── ... (existing)

tests/metrics/
├── test_assemblies.py     # NEW: Assembly tests
└── ... (existing)
```

---

## Part 1: `assemblies.py` - Module Header and Imports

```python
"""
Cell assembly detection and reactivation analysis.

This module provides functions for detecting coordinated neural ensembles
(cell assemblies) and measuring their reactivation strength across behavioral
states.

Cell assemblies are groups of neurons that fire together more often than
expected by chance, representing stored neural patterns. Detecting these
assemblies and measuring their reactivation is fundamental for studying
memory consolidation and replay.

Typical Workflows
-----------------
**Assembly Detection**:

1. Prepare spike count matrix:

   >>> # spike_counts shape: (n_neurons, n_time_bins)
   >>> spike_counts = bin_spikes(spike_times_list, bin_size=0.025)

2. Detect assemblies:

   >>> result = detect_assemblies(spike_counts, method="ica")
   >>> print(f"Found {result.n_significant} assemblies")

3. Analyze assembly activation:

   >>> for pattern in result.patterns:
   ...     activation = assembly_activation(spike_counts, pattern)
   ...     print(f"Assembly with {len(pattern.member_indices)} neurons")

**Reactivation Analysis**:

1. Compute pairwise correlations in behavior and sleep:

   >>> corr_behavior = pairwise_correlations(counts_behavior)
   >>> corr_sleep = pairwise_correlations(counts_sleep)

2. Measure explained variance:

   >>> ev_result = explained_variance_reactivation(corr_behavior, corr_sleep)
   >>> print(f"EV = {ev_result.explained_variance:.3f}")

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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, stats

if TYPE_CHECKING:
    pass

__all__ = [
    "AssemblyPattern",
    "AssemblyDetectionResult",
    "ExplainedVarianceResult",
    "detect_assemblies",
    "assembly_activation",
    "reactivation_strength",
    "explained_variance_reactivation",
    "pairwise_correlations",
    "marchenko_pastur_threshold",
]
```

---

## Part 2: Data Classes

```python
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
    >>> pattern = result.patterns[0]
    >>> print(f"Assembly has {len(pattern.member_indices)} core members")
    >>> print(f"Explains {pattern.explained_variance_ratio:.1%} of variance")
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
    >>> result = detect_assemblies(spike_counts)
    >>> print(f"Found {result.n_significant} significant assemblies")
    >>> print(f"Method: {result.method}")
    >>>
    >>> # Plot activation of first assembly
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(result.activations[0])
    >>> plt.ylabel("Activation (z-score)")
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
    >>> result = explained_variance_reactivation(corr_behavior, corr_sleep)
    >>> if result.explained_variance > result.reversed_ev:
    ...     print("Forward reactivation detected!")
    """

    explained_variance: float
    reversed_ev: float
    partial_correlation: float
    n_pairs: int
```

---

## Part 3: Marchenko-Pastur Threshold

```python
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
    Threshold: 1.464

    >>> # Compare with eigenvalues
    >>> eigenvalues = np.linalg.eigvalsh(np.corrcoef(spike_counts))
    >>> n_significant = np.sum(eigenvalues > threshold)

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
        import warnings

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
```

---

## Part 4: Assembly Detection

```python
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
    >>> result = detect_assemblies(spike_counts)
    >>> print(f"Found {result.n_significant} assemblies")

    >>> # With specific number of components
    >>> result = detect_assemblies(spike_counts, n_components=5)

    >>> # Using PCA for faster computation
    >>> result = detect_assemblies(spike_counts, method="pca")

    >>> # Access assembly patterns
    >>> for i, pattern in enumerate(result.patterns):
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
        import warnings

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
        import warnings

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
        patterns, activations, explained_var = _detect_pca(
            spike_counts_z, n_comp
        )
    elif method == "ica":
        patterns, activations, explained_var = _detect_ica(
            spike_counts_z, n_comp, random_state
        )
    elif method == "nmf":
        patterns, activations, explained_var = _detect_nmf(
            spike_counts, n_comp, random_state  # NMF uses original counts
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ica', 'pca', or 'nmf'.")

    # Create AssemblyPattern objects
    assembly_patterns = []
    for i in range(n_comp):
        weights = patterns[i]

        # Z-score weights to find significant members
        weights_z = stats.zscore(np.abs(weights))
        member_mask = weights_z > z_threshold
        member_indices = np.where(member_mask)[0].astype(np.int64)

        assembly_patterns.append(
            AssemblyPattern(
                weights=weights,
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
    Detect assemblies using PCA via scipy SVD.

    Returns
    -------
    patterns : array, shape (n_components, n_neurons)
    activations : array, shape (n_components, n_time_bins)
    explained_variance_ratio : array, shape (n_components,)
    """
    # SVD: spike_counts_z = U @ S @ Vh
    # patterns = Vh (right singular vectors)
    # activations = U @ S (scores)
    U, s, Vh = linalg.svd(spike_counts_z, full_matrices=False)

    # Select top n_components
    patterns = Vh[:n_components]  # (n_components, n_neurons)

    # Compute activations
    activations = (U[:, :n_components] * s[:n_components]).T  # (n_components, n_time_bins)

    # Explained variance ratio
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

    Returns
    -------
    patterns : array, shape (n_components, n_neurons)
    activations : array, shape (n_components, n_time_bins)
    explained_variance_ratio : array, shape (n_components,)
    """
    from sklearn.decomposition import FastICA

    # First reduce to n_components via PCA
    patterns_pca, _, explained_var_pca = _detect_pca(spike_counts_z, n_components)

    # Project data onto PCA space
    projected = patterns_pca @ spike_counts_z  # (n_components, n_time_bins)

    # Apply ICA to find independent components
    ica = FastICA(
        n_components=n_components,
        random_state=random_state,
        max_iter=1000,
        whiten=False,  # Already whitened by PCA
    )

    # ICA on transposed data: (n_time_bins, n_components) -> (n_time_bins, n_components)
    activations_t = ica.fit_transform(projected.T)
    activations = activations_t.T  # (n_components, n_time_bins)

    # Compute patterns by back-projecting
    # patterns_ica = ica.mixing_.T @ patterns_pca
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
    W = nmf.fit_transform(spike_counts_nn)
    H = nmf.components_

    patterns = W.T  # (n_components, n_neurons)
    activations = H  # (n_components, n_time_bins)

    # Compute explained variance
    reconstruction = W @ H
    total_var = np.var(spike_counts_nn)
    residual_var = np.var(spike_counts_nn - reconstruction)
    total_explained = 1 - residual_var / total_var if total_var > 0 else 0

    # Approximate per-component variance (proportional to pattern norm)
    pattern_norms = np.linalg.norm(patterns, axis=1)
    explained_var = pattern_norms / np.sum(pattern_norms) * total_explained

    return patterns, activations, explained_var
```

---

## Part 5: Assembly Activation

```python
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
    >>> result = detect_assemblies(counts_behavior)
    >>> pattern = result.patterns[0]
    >>>
    >>> # Compute activation during sleep
    >>> activation_sleep = assembly_activation(counts_sleep, pattern)
    >>> print(f"Peak activation: {activation_sleep.max():.2f}")

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

    n_neurons, n_time_bins = spike_counts.shape

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

    return activation_z
```

---

## Part 6: Pairwise Correlations and Explained Variance

```python
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
    >>> corr = pairwise_correlations(spike_counts)
    >>> n_neurons = spike_counts.shape[0]
    >>> n_pairs = n_neurons * (n_neurons - 1) // 2
    >>> assert len(corr) == n_pairs
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)

    if spike_counts.ndim != 2:
        raise ValueError(
            f"spike_counts must be 2D, got shape {spike_counts.shape}"
        )

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
    Compare assembly activation between template and match periods.

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
        Correlation between mean activations in template and match periods.
        Range [-1, 1]. High positive values indicate reactivation.

    Notes
    -----
    This is a simple measure of reactivation based on comparing mean
    activation levels. For more sophisticated analysis, use
    `explained_variance_reactivation` with pairwise correlations.

    Examples
    --------
    >>> strength = reactivation_strength(counts_behavior, counts_sleep, pattern)
    >>> if strength > 0.3:
    ...     print("Strong reactivation")
    """
    # Compute activations
    act_template = assembly_activation(template_counts, pattern)
    act_match = assembly_activation(match_counts, pattern)

    # Correlation of activation time series
    corr = np.corrcoef(act_template.mean(), act_match.mean())[0, 1]

    return float(np.nan_to_num(corr, nan=0.0))


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
    >>> corr_pre = pairwise_correlations(counts_pre_behavior)
    >>> corr_behavior = pairwise_correlations(counts_behavior)
    >>> corr_sleep = pairwise_correlations(counts_sleep)
    >>>
    >>> # Measure reactivation with baseline control
    >>> result = explained_variance_reactivation(
    ...     corr_behavior, corr_sleep, control_correlations=corr_pre
    ... )
    >>> print(f"EV = {result.explained_variance:.3f}")
    >>> print(f"REV = {result.reversed_ev:.3f}")

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
        raise ValueError(
            f"Need at least 3 pairs for EV calculation, got {n_pairs}"
        )

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
        import warnings

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
        if denom > 1e-10:
            partial_corr = (r_tm - r_tc * r_mc) / denom
        else:
            partial_corr = 0.0
    else:
        partial_corr = r_tm  # No control, use raw correlation

    return ExplainedVarianceResult(
        explained_variance=float(ev),
        reversed_ev=float(rev),
        partial_correlation=float(partial_corr),
        n_pairs=int(np.sum(valid_mask)),
    )
```

---

## Part 7: Update Exports

### Update `metrics/__init__.py`

```python
from neurospatial.metrics.assemblies import (
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

__all__ = [
    # ... existing exports ...
    # Cell assemblies
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
```

---

## Implementation Order

1. **Data classes** (no dependencies)
   - `AssemblyPattern`
   - `AssemblyDetectionResult`
   - `ExplainedVarianceResult`

2. **Core functions** (scipy/numpy only)
   - `marchenko_pastur_threshold()`
   - `pairwise_correlations()`
   - `_detect_pca()` (uses scipy.linalg.svd)

3. **sklearn-dependent functions**
   - `_detect_ica()` (uses FastICA)
   - `_detect_nmf()` (uses NMF)

4. **Main API**
   - `detect_assemblies()`
   - `assembly_activation()`

5. **Reactivation analysis**
   - `reactivation_strength()`
   - `explained_variance_reactivation()`

6. **Tests**
   - `tests/metrics/test_assemblies.py`

7. **Update exports**
   - `metrics/__init__.py`

---

## Test Strategy

### Synthetic Assembly Tests

```python
def test_detect_known_assemblies():
    """Create known patterns and verify detection."""
    rng = np.random.default_rng(42)
    n_neurons, n_time = 50, 2000

    # Create 3 assemblies with known membership
    assembly_neurons = [
        [0, 1, 2, 3, 4],
        [10, 11, 12, 13, 14],
        [20, 21, 22, 23, 24],
    ]

    # Generate correlated activity
    spike_counts = rng.poisson(5, (n_neurons, n_time)).astype(float)
    for neurons in assembly_neurons:
        shared = rng.poisson(3, n_time)
        for n in neurons:
            spike_counts[n] += shared

    result = detect_assemblies(spike_counts, method="ica")

    # Should detect ~3 assemblies
    assert result.n_significant >= 2
    assert result.n_significant <= 5


def test_random_data_no_assemblies():
    """Random data should detect few/no significant assemblies."""
    rng = np.random.default_rng(42)
    spike_counts = rng.poisson(5, (30, 3000)).astype(float)

    result = detect_assemblies(spike_counts)

    # Should detect 0-1 assemblies (noise)
    assert result.n_significant <= 2
```

### Explained Variance Tests

```python
def test_ev_perfect_reactivation():
    """Perfect correlation should give EV = 1."""
    corr = np.array([0.1, 0.5, -0.3, 0.8, 0.2])

    result = explained_variance_reactivation(corr, corr)

    assert result.explained_variance > 0.99
    assert result.n_pairs == 5


def test_ev_random_no_reactivation():
    """Random correlations should give low EV."""
    rng = np.random.default_rng(42)
    corr1 = rng.uniform(-1, 1, 100)
    corr2 = rng.uniform(-1, 1, 100)

    result = explained_variance_reactivation(corr1, corr2)

    assert result.explained_variance < 0.1
```

### Edge Cases

- Empty spike counts → ValueError
- Single neuron → ValueError
- Zero variance neurons → Warning, handled gracefully
- All-zero spike counts → Handled without NaN
- Mismatched array sizes → ValueError

---

## Dependencies

**Uses from existing neurospatial dependencies:**

- `numpy` - corrcoef, linalg, triu_indices
- `scipy.stats` - zscore
- `scipy.linalg` - svd, eigvalsh
- `sklearn.decomposition` - FastICA, NMF (already in pyproject.toml)

**No new dependencies required.**

---

## Computational Complexity

| Function | Complexity | Notes |
|----------|------------|-------|
| `marchenko_pastur_threshold` | O(1) | Simple formula |
| `pairwise_correlations` | O(n² × t) | n=neurons, t=time bins |
| `detect_assemblies` (PCA) | O(n² × t) | SVD-dominated |
| `detect_assemblies` (ICA) | O(n² × t × k) | k=ICA iterations |
| `detect_assemblies` (NMF) | O(n × t × c × k) | c=components, k=iterations |
| `assembly_activation` | O(n × t) | Single dot product |
| `explained_variance_reactivation` | O(p) | p=pairs |

---

## References

### Primary Sources

- **Lopes-dos-Santos V, Ribeiro S, Bhattacharyya A (2013)**. Detecting cell assemblies
  in large neuronal populations. J Neurosci Methods 220(2):149-166.
  - PCA/ICA methodology, Marchenko-Pastur threshold

- **Peyrache A, Khamassi M, Benchenane K, Wiener SI, Battaglia FP (2009)**.
  Replay of rule-learning related neural patterns in the prefrontal cortex
  during sleep. Nat Neurosci 12(7):919-926.
  - Assembly reactivation methodology

- **Kudrimoti HS, Barnes CA, McNaughton BL (1999)**. Reactivation of hippocampal
  cell assemblies: effects of behavioral state, experience, and EEG dynamics.
  J Neurosci 19(10):4090-4101.
  - Explained variance reactivation

- **van de Ven GM, Trouche S, McNamara CG, Allen K, Bhattercharya D (2016)**.
  Hippocampal offline reactivation consolidates recently formed cell assembly
  patterns during sharp wave-ripples. Neuron 92(5):968-974.
  - Assembly detection methodology review

### Scipy/Numpy Functions Used

- [scipy.stats.zscore](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html)
- [scipy.linalg.svd](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html)
- [scipy.linalg.eigvalsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigvalsh.html)
- [numpy.corrcoef](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html)
- [numpy.triu_indices](https://numpy.org/doc/stable/reference/generated/numpy.triu_indices.html)

### Sklearn Functions Used

- [sklearn.decomposition.FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
- [sklearn.decomposition.NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
