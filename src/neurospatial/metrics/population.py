"""Population-level place field metrics.

This module provides metrics for analyzing spatial representations across
populations of neurons, including coverage, overlap, and correlation measures.

References
----------
.. [1] Wilson, M. A., & McNaughton, B. L. (1993). Dynamics of the hippocampal
       ensemble code for space. Science, 261(5124), 1055-1058.
.. [2] O'Keefe, J., & Nadel, L. (1978). The Hippocampus as a Cognitive Map.
       Oxford: Clarendon Press.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def population_coverage(
    all_place_fields: list[list[NDArray[np.int64]]], n_bins: int
) -> float:
    """
    Compute fraction of environment covered by place fields.

    Measures the spatial extent of place field representation across a
    population of cells. Higher coverage indicates more complete spatial
    mapping by the population.

    Parameters
    ----------
    all_place_fields : list of list of array
        Place fields for each cell. Outer list: cells, inner list: fields
        per cell, arrays: bin indices in each field.
        Format matches output of `detect_place_fields()` run on multiple cells.
    n_bins : int
        Total number of bins in the environment.

    Returns
    -------
    coverage : float
        Fraction of bins covered by at least one place field. Range [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.population import population_coverage
    >>> # Two cells with non-overlapping fields
    >>> all_fields = [
    ...     [np.array([0, 1, 2])],  # Cell 1
    ...     [np.array([5, 6, 7])],  # Cell 2
    ... ]
    >>> coverage = population_coverage(all_fields, n_bins=10)
    >>> print(f"{coverage:.2f}")  # 6 bins out of 10
    0.60

    >>> # Overlapping fields
    >>> all_fields = [
    ...     [np.array([0, 1, 2])],
    ...     [np.array([2, 3, 4])],  # Overlaps at bin 2
    ... ]
    >>> coverage = population_coverage(all_fields, n_bins=10)
    >>> print(f"{coverage:.2f}")  # 5 unique bins out of 10
    0.50

    See Also
    --------
    field_density_map : Count overlapping fields per bin
    detect_place_fields : Detect place fields for individual cells

    Notes
    -----
    Population coverage is a measure of how completely the environment is
    represented by the place cell population. In hippocampus, typical coverage
    is 60-90% depending on environment size and cell sample size.

    Empty fields (no bins) are handled gracefully and contribute 0 to coverage.
    """
    # Collect all unique bins across all cells and fields
    covered_bins = set()
    for cell_fields in all_place_fields:
        for field_bins in cell_fields:
            covered_bins.update(field_bins.tolist())

    # Compute fraction
    return len(covered_bins) / n_bins


def field_density_map(
    all_place_fields: list[list[NDArray[np.int64]]], n_bins: int
) -> NDArray[np.int64]:
    """
    Count number of overlapping place fields per bin.

    Creates a density map showing how many cells have place fields at each
    location. High-density bins indicate regions represented by multiple cells.

    Parameters
    ----------
    all_place_fields : list of list of array
        Place fields for each cell. Format matches `detect_place_fields()` output.
    n_bins : int
        Total number of bins in the environment.

    Returns
    -------
    density : array, shape (n_bins,)
        Number of place fields overlapping each bin. Zero for bins with no fields.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.population import field_density_map
    >>> # Three cells with overlapping fields
    >>> all_fields = [
    ...     [np.array([2, 3, 4])],  # Cell 1
    ...     [np.array([3, 4, 5])],  # Cell 2 (overlaps at 3, 4)
    ...     [np.array([4, 5, 6])],  # Cell 3 (overlaps at 4, 5)
    ... ]
    >>> density = field_density_map(all_fields, n_bins=10)
    >>> print(density[4])  # Bin 4 has 3 overlapping fields
    3
    >>> print(density[3])  # Bin 3 has 2 overlapping fields
    2

    See Also
    --------
    population_coverage : Fraction of environment covered
    field_overlap : Pairwise field overlap (Jaccard)

    Notes
    -----
    The density map can reveal:

    - **High-density regions**: Locations represented by many cells (stable coding)
    - **Low-density regions**: Sparse representation (may be less reliable)
    - **Density gradients**: Spatial biases in population code

    In typical hippocampal recordings, density ranges from 0-5 fields per bin,
    with higher density near boundaries and goal locations.
    """
    density = np.zeros(n_bins, dtype=np.int64)

    for cell_fields in all_place_fields:
        for field_bins in cell_fields:
            # Increment count for each bin in this field
            density[field_bins] += 1

    return density


def count_place_cells(
    spatial_information: NDArray[np.float64], threshold: float = 0.5
) -> int:
    """
    Count cells exceeding spatial information threshold.

    Determines number of spatially selective cells in a population based on
    Skaggs spatial information criterion. Standard method for identifying
    place cells vs. non-selective cells.

    Parameters
    ----------
    spatial_information : array, shape (n_cells,)
        Spatial information (bits/spike) for each cell.
        Typically computed using `skaggs_information()`.
    threshold : float, default=0.5
        Minimum spatial information (bits/spike) to classify as place cell.
        Default of 0.5 bits/spike is standard in literature (Skaggs et al., 1996).

    Returns
    -------
    count : int
        Number of cells with spatial information > threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.population import count_place_cells
    >>> # Population with mix of place cells and non-selective cells
    >>> spatial_info = np.array([0.2, 0.8, 1.5, 0.3, 1.2, 0.1])
    >>> n_place_cells = count_place_cells(spatial_info, threshold=0.5)
    >>> print(n_place_cells)  # 3 cells exceed 0.5 bits/spike
    3

    See Also
    --------
    skaggs_information : Compute spatial information

    Notes
    -----
    **Classification criteria**:

    - **Place cells**: Spatial information > 0.5 bits/spike
    - **Interneurons**: Mean firing rate > 10 Hz (handled separately)
    - **Non-selective**: Low spatial information or uniform firing

    NaN values (e.g., from cells with zero firing) are excluded from count.

    References
    ----------
    .. [1] Skaggs, W. E., McNaughton, B. L., et al. (1996). Theta phase
           precession in hippocampal neuronal populations and the compression
           of temporal sequences. Hippocampus, 6(2), 149-172.
    """
    # Exclude NaN values (cells with undefined information)
    valid_mask = ~np.isnan(spatial_information)
    valid_info = spatial_information[valid_mask]

    # Count cells strictly exceeding threshold
    return int(np.sum(valid_info > threshold))


def field_overlap(
    field_bins_i: NDArray[np.int64], field_bins_j: NDArray[np.int64]
) -> float:
    """
    Compute Jaccard similarity between two place fields.

    Measures spatial overlap as the ratio of intersection to union (Jaccard
    coefficient). Used to quantify remapping and field stability.

    Parameters
    ----------
    field_bins_i : array
        Bin indices in first field.
    field_bins_j : array
        Bin indices in second field.

    Returns
    -------
    overlap : float
        Jaccard coefficient = |intersection| / |union|. Range [0, 1].
        Returns 0.0 if either field is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.population import field_overlap
    >>> # Identical fields (perfect overlap)
    >>> field1 = np.array([0, 1, 2, 3])
    >>> field2 = np.array([0, 1, 2, 3])
    >>> overlap = field_overlap(field1, field2)
    >>> print(f"{overlap:.2f}")
    1.00

    >>> # Partial overlap
    >>> field1 = np.array([0, 1, 2, 3])
    >>> field2 = np.array([2, 3, 4, 5])
    >>> overlap = field_overlap(field1, field2)
    >>> # Intersection: {2, 3}, Union: {0,1,2,3,4,5}
    >>> print(f"{overlap:.2f}")  # 2/6 = 0.33
    0.33

    See Also
    --------
    field_stability : Correlation-based stability measure
    field_density_map : Population-level overlap

    Notes
    -----
    **Interpretation**:

    - **Jaccard = 1.0**: Identical fields (perfect stability/remapping)
    - **Jaccard = 0.5**: Moderate overlap (partial remapping)
    - **Jaccard = 0.0**: Disjoint fields (complete remapping)

    **Use cases**:

    - Quantify remapping between environments (Muller & Kubie, 1987)
    - Measure field stability across sessions
    - Assess similarity of fields from different cells

    The Jaccard index is preferred over correlation for sparse binary fields
    because it's invariant to the number of non-field bins.

    References
    ----------
    .. [1] Muller, R. U., & Kubie, J. L. (1987). The effects of changes in
           the environment on the spatial firing of hippocampal complex-spike
           cells. Journal of Neuroscience, 7(7), 1951-1968.
    """
    # Handle empty fields
    if len(field_bins_i) == 0 or len(field_bins_j) == 0:
        return 0.0

    # Convert to sets for efficient set operations
    set_i = set(field_bins_i.tolist())
    set_j = set(field_bins_j.tolist())

    # Compute Jaccard coefficient
    intersection = set_i & set_j
    union = set_i | set_j

    if len(union) == 0:
        # Both fields empty (edge case already handled above, but be defensive)
        return 0.0

    return len(intersection) / len(union)


def population_vector_correlation(
    population_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute pairwise correlation matrix between firing rate maps.

    Calculates Pearson correlation between all pairs of cells' firing rate
    vectors. Used to assess similarity of spatial representations and detect
    functional cell assemblies.

    Parameters
    ----------
    population_matrix : array, shape (n_cells, n_bins)
        Firing rate maps for each cell (rows: cells, columns: bins).

    Returns
    -------
    correlation_matrix : array, shape (n_cells, n_cells)
        Pairwise Pearson correlation coefficients. Symmetric matrix with
        diagonal = 1.0. Off-diagonal entries in range [-1, 1].
        Returns NaN for pairs involving cells with zero variance.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.population import population_vector_correlation
    >>> # Three cells with different firing patterns
    >>> np.random.seed(42)
    >>> population_matrix = np.random.rand(3, 50)
    >>> corr_matrix = population_vector_correlation(population_matrix)
    >>> print(corr_matrix.shape)
    (3, 3)
    >>> # Diagonal is self-correlation (always 1.0)
    >>> print(np.diag(corr_matrix))
    [1. 1. 1.]

    See Also
    --------
    field_stability : Correlation between same cell across sessions
    field_overlap : Jaccard-based overlap measure

    Notes
    -----
    **Interpretation**:

    - **r = 1.0**: Identical spatial firing patterns
    - **r = 0.0**: Uncorrelated (orthogonal) representations
    - **r = -1.0**: Anticorrelated (complementary) firing
    - **r = NaN**: One or both cells have constant firing (zero variance)

    **Use cases**:

    - Identify functional cell assemblies (high positive correlation)
    - Detect complementary coding schemes (negative correlation)
    - Analyze remapping (correlation drop between environments)
    - Study ensemble dynamics during learning

    The matrix is symmetric by construction: `corr[i,j] = corr[j,i]`.

    For cells with constant firing rates, correlation is undefined (NaN).
    This typically indicates interneurons or cells with insufficient data.

    References
    ----------
    .. [1] Wilson, M. A., & McNaughton, B. L. (1993). Dynamics of the
           hippocampal ensemble code for space. Science, 261(5124), 1055-1058.
    .. [2] Harris, K. D., et al. (2003). Organization of cell assemblies in
           the hippocampus. Nature, 424(6948), 552-556.
    """
    n_cells = population_matrix.shape[0]

    # Use numpy's corrcoef for efficiency
    # corrcoef computes correlation between rows
    # Explicit dtype for mypy type checking
    correlation_matrix = np.corrcoef(population_matrix, dtype=np.float64)

    # Ensure output is 2D even for single cell
    if n_cells == 1:
        correlation_matrix = correlation_matrix.reshape(1, 1)

    # Fix diagonal: self-correlation is always 1.0 by definition
    # (even for constant cells where numpy returns NaN)
    np.fill_diagonal(correlation_matrix, 1.0)

    return correlation_matrix
