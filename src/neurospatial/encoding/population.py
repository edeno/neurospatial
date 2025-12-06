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

from dataclasses import dataclass

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray

from neurospatial.encoding.place import detect_place_fields
from neurospatial.environment import Environment


@dataclass(frozen=True, slots=True)
class PopulationCoverageResult:
    """Results from population coverage analysis.

    Attributes
    ----------
    coverage_fraction : float
        Fraction of environment bins covered by at least one place field (0.0-1.0).
    is_covered : NDArray[np.bool_]
        Boolean mask of shape (n_bins,) indicating which bins are covered.
    field_count : NDArray[np.int64]
        Integer array of shape (n_bins,) counting place fields per bin.
    covered_bins : NDArray[np.intp]
        Indices of bins with at least one place field.
    uncovered_bins : NDArray[np.intp]
        Indices of bins lacking coverage (gaps).
    uncovered_positions : NDArray[np.floating]
        Coordinates of uncovered bins, shape (n_uncovered, n_dims).
    n_neurons : int
        Total number of neurons in the population.
    n_place_cells : int
        Number of neurons with at least one detected place field.
    n_fields : int
        Total number of place fields detected across all neurons.
    place_fields : list[list[NDArray[np.int64]]]
        Detected place fields for each neuron (output of detect_place_fields).
    place_cell_fraction : float
        Computed property: n_place_cells / n_neurons (typical CA1: 0.3-0.5).
    fields_per_place_cell : float
        Computed property: n_fields / n_place_cells (typical: 1.0-2.0).
    mean_redundancy : float
        Computed property: average field_count over covered bins.
    """

    coverage_fraction: float
    is_covered: NDArray[np.bool_]
    field_count: NDArray[np.int64]
    covered_bins: NDArray[np.intp]
    uncovered_bins: NDArray[np.intp]
    uncovered_positions: NDArray[np.floating]
    n_neurons: int
    n_place_cells: int
    n_fields: int
    place_fields: list[list[NDArray[np.int64]]]

    @property
    def place_cell_fraction(self) -> float:
        """Fraction of neurons that are place cells.

        Typical values in hippocampal CA1: 0.3-0.5 (30-50%).

        References
        ----------
        .. [1] Wilson, M. A., & McNaughton, B. L. (1993). Dynamics of the
               hippocampal ensemble code for space. Science, 261(5124), 1055-1058.
        """
        if self.n_neurons == 0:
            return 0.0
        return self.n_place_cells / self.n_neurons

    @property
    def fields_per_place_cell(self) -> float:
        """Average number of place fields per place cell.

        Typical values: 1.0-2.0 in familiar environments.
        Higher values (>2) may indicate multifield cells or novel environments.

        References
        ----------
        .. [1] Fenton, A. A., et al. (2008). Unmasking the CA1 ensemble place
               code by exposures to small and large environments.
               Journal of Neuroscience, 28(50), 13566-13577.
        """
        if self.n_place_cells == 0:
            return 0.0
        return self.n_fields / self.n_place_cells

    @property
    def mean_redundancy(self) -> float:
        """Average number of place fields per covered bin.

        Measures the degree of overlapping representation. Higher values
        indicate more robust population coding with multiple cells
        representing each location.

        References
        ----------
        .. [1] Wilson, M. A., & McNaughton, B. L. (1993). Dynamics of the
               hippocampal ensemble code for space. Science, 261(5124), 1055-1058.
        """
        n_covered = len(self.covered_bins)
        if n_covered == 0:
            return 0.0
        return float(self.field_count[self.covered_bins].mean())

    def __str__(self) -> str:
        """Return human-readable summary for notebook inspection."""
        return (
            f"PopulationCoverageResult:\n"
            f"  Coverage: {self.coverage_fraction:.1%}\n"
            f"  Place cells: {self.n_place_cells}/{self.n_neurons} "
            f"({self.place_cell_fraction:.1%})\n"
            f"  Total fields: {self.n_fields} "
            f"({self.fields_per_place_cell:.1f} per place cell)\n"
            f"  Mean redundancy: {self.mean_redundancy:.2f} fields/bin\n"
            f"  Gaps: {len(self.uncovered_bins)} bins"
        )


def population_coverage(
    firing_rates: NDArray[np.floating],
    env: Environment,
    *,
    threshold: float = 0.2,
    min_size: int | None = None,
    max_mean_rate: float = 10.0,
    detect_subfields: bool = True,
) -> PopulationCoverageResult:
    """Compute spatial coverage of a place cell population.

    Detects place fields for each neuron, then computes comprehensive
    coverage statistics including gap locations.

    Parameters
    ----------
    firing_rates : NDArray[np.floating], shape (n_neurons, n_bins)
        Firing rate maps for each neuron.
    env : Environment
        Fitted environment defining the spatial bins.
    threshold : float, default=0.2
        Place field boundary threshold as fraction of peak firing rate (0-1).
        Lower values (0.1-0.2) detect larger fields, higher values (0.3-0.5)
        detect only core field regions. Passed to `detect_place_fields()`.
    min_size : int, optional
        Minimum number of bins for a valid field. Default is 9.
        Passed to `detect_place_fields()`.
    max_mean_rate : float, default=10.0
        Maximum mean firing rate (Hz). Neurons exceeding this are
        excluded as putative interneurons. Passed to `detect_place_fields()`.
    detect_subfields : bool, default=True
        If True, recursively detect subfields within large fields.
        Passed to `detect_place_fields()`.

    Returns
    -------
    PopulationCoverageResult
        Comprehensive coverage statistics including:
        - coverage_fraction: fraction of bins covered (0.0-1.0)
        - uncovered_bins: indices of gaps
        - uncovered_positions: coordinates of gaps
        - place_fields: detected fields for each neuron

    Raises
    ------
    RuntimeError
        If environment is not fitted.
    ValueError
        If firing_rates shape doesn't match environment bins.

    See Also
    --------
    field_density_map : Field count per bin from detected fields
    detect_place_fields : Place field detection algorithm
    plot_population_coverage : Visualize coverage with gaps

    Examples
    --------
    >>> import numpy as np  # doctest: +SKIP
    >>> from neurospatial import Environment  # doctest: +SKIP
    >>> from neurospatial.encoding.place import compute_place_field  # doctest: +SKIP
    >>> from neurospatial.encoding.population import (
    ...     population_coverage,
    ...     plot_population_coverage,
    ... )  # doctest: +SKIP
    >>>
    >>> # 1. Create environment from position samples
    >>> positions = np.random.rand(1000, 2) * 100  # (n_samples, 2)  # doctest: +SKIP
    >>> times = np.arange(1000) * 0.033  # 30 Hz timestamps  # doctest: +SKIP
    >>> env = Environment.from_samples(positions, bin_size=5.0)  # doctest: +SKIP
    >>>
    >>> # 2. Compute firing rates for each neuron from spike data
    >>> spike_times = [
    ...     ...
    ... ]  # List of spike time arrays, one per neuron  # doctest: +SKIP
    >>> firing_rates = np.array(  # doctest: +SKIP
    ...     [
    ...         compute_place_field(env, spikes, times, positions)
    ...         for spikes in spike_times
    ...     ]
    ... )  # Shape: (n_neurons, n_bins)
    >>>
    >>> # 3. Analyze coverage
    >>> result = population_coverage(firing_rates, env)  # doctest: +SKIP
    >>> print(f"Coverage: {result.coverage_fraction:.1%}")  # doctest: +SKIP
    >>> print(
    ...     f"Place cells: {result.n_place_cells}/{result.n_neurons}"
    ... )  # doctest: +SKIP
    >>> print(f"Gaps: {len(result.uncovered_bins)} bins")  # doctest: +SKIP
    >>>
    >>> # 4. Visualize
    >>> plot_population_coverage(env, result)  # doctest: +SKIP

    References
    ----------
    .. [1] Wilson, M. A., & McNaughton, B. L. (1993). Dynamics of the hippocampal
           ensemble code for space. Science, 261(5124), 1055-1058.
    """
    # Validate environment is fitted
    if not getattr(env, "_is_fitted", False):
        raise RuntimeError(
            "Environment must be fitted before computing coverage. "
            "Use a factory method like Environment.from_samples()."
        )

    if env.n_bins <= 0:
        raise ValueError(f"Environment has no bins (n_bins={env.n_bins})")

    # Validate firing_rates shape
    if firing_rates.ndim != 2:
        raise ValueError(
            f"firing_rates must be 2D (n_neurons, n_bins), got shape {firing_rates.shape}"
        )

    if firing_rates.shape[1] != env.n_bins:
        raise ValueError(
            f"firing_rates shape mismatch: expected (n_neurons, {env.n_bins}) bins, "
            f"got (n_neurons, {firing_rates.shape[1]}) bins.\n\n"
            f"This usually happens when:\n"
            f"  - Firing rates were computed for a different environment\n"
            f"  - The environment was modified after computing firing rates\n\n"
            f"To fix: Recompute firing rates for each neuron:\n"
            f"  from neurospatial import compute_place_field\n"
            f"  firing_rate = compute_place_field(env, spike_times, times, positions)\n"
            f"  firing_rates = np.stack([rate1, rate2, ...])"
        )

    # Validate detection parameters
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"threshold must be in range (0, 1), got {threshold}. "
            f"This represents the fraction of peak firing rate for field boundaries."
        )

    if max_mean_rate <= 0:
        raise ValueError(f"max_mean_rate must be positive, got {max_mean_rate}")

    n_bins = env.n_bins
    n_neurons = firing_rates.shape[0]

    # Detect place fields for each neuron
    all_fields: list[list[NDArray[np.int64]]] = []
    for neuron_idx in range(n_neurons):
        fields = detect_place_fields(
            firing_rates[neuron_idx],
            env,
            threshold=threshold,
            min_size=min_size,
            max_mean_rate=max_mean_rate,
            detect_subfields=detect_subfields,
        )
        all_fields.append(fields)

    # Count fields and compute coverage
    field_count = np.zeros(n_bins, dtype=np.int64)
    n_fields = 0
    n_place_cells = 0

    for neuron_fields in all_fields:
        if len(neuron_fields) > 0:
            n_place_cells += 1
        for field_bins in neuron_fields:
            field_count[field_bins] += 1
            n_fields += 1

    # Compute coverage
    is_covered = field_count > 0
    coverage_fraction = float(is_covered.sum() / n_bins)

    # Find covered and uncovered bins
    covered_bins = np.where(is_covered)[0]
    uncovered_bins = np.where(~is_covered)[0]
    uncovered_positions = env.bin_centers[uncovered_bins]

    return PopulationCoverageResult(
        coverage_fraction=coverage_fraction,
        is_covered=is_covered,
        field_count=field_count,
        covered_bins=covered_bins,
        uncovered_bins=uncovered_bins,
        uncovered_positions=uncovered_positions,
        n_neurons=n_neurons,
        n_place_cells=n_place_cells,
        n_fields=n_fields,
        place_fields=all_fields,
    )


def plot_population_coverage(
    env: Environment,
    result: PopulationCoverageResult,
    *,
    show_field_count: bool = False,
    highlight_gaps: bool = True,
    gap_color: str = "black",
    covered_color: str = "#0173B2",  # Colorblind-safe blue
    gap_marker: str = "x",
    gap_marker_size: float = 50.0,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Visualize place cell coverage with optional gap highlighting.

    Parameters
    ----------
    env : Environment
        Fitted environment for plotting.
    result : PopulationCoverageResult
        Output from population_coverage.
    show_field_count : bool, optional
        If True, show number of place fields per bin (redundancy colormap).
        Use this to identify over-represented regions.
        If False, show binary covered/uncovered map (recommended for gap
        detection). Default is False.
    highlight_gaps : bool, optional
        If True, overlay markers on uncovered bins. Default is True.
    gap_color : str, optional
        Color for uncovered bin markers. Default is "black".
    covered_color : str, optional
        Color for covered bins (only used when show_field_count=False).
        Default is "#0173B2" (colorblind-safe blue).
    gap_marker : str, optional
        Marker style for gaps. Default is "x".
    gap_marker_size : float, optional
        Size of gap markers. Default is 50.0.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if not provided.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the coverage plot.

    Raises
    ------
    RuntimeError
        If environment is not fitted.

    Notes
    -----
    For 1D environments, uses `env.plot_1d()` with vertical lines for gaps.

    Examples
    --------
    >>> ax = plot_population_coverage(env, result)  # doctest: +SKIP

    >>> ax = plot_population_coverage(
    ...     env, result, show_field_count=True
    ... )  # doctest: +SKIP
    """
    # Validate environment is fitted
    if not getattr(env, "_is_fitted", False):
        raise RuntimeError(
            "Environment must be fitted before plotting coverage. "
            "Use a factory method like Environment.from_samples()."
        )

    # Validate result type
    if not isinstance(result, PopulationCoverageResult):
        raise TypeError(
            f"result must be PopulationCoverageResult, got {type(result).__name__}. "
            f"Use population_coverage() to compute the result first."
        )

    # Create axes if needed
    if ax is None:
        _, ax = plt.subplots()

    # Handle 1D environments
    if env.is_1d:
        if show_field_count:
            field_to_plot = result.field_count.astype(float)
            env.plot_field(field_to_plot, ax=ax)
            ax.set_ylabel("Field count")
        else:
            field_to_plot = result.is_covered.astype(float)
            env.plot_field(field_to_plot, ax=ax)
            ax.set_ylabel("Coverage")

        # Highlight gaps as vertical lines with legend
        if highlight_gaps and len(result.uncovered_bins) > 0:
            gap_positions = result.uncovered_positions.ravel()
            for i, pos in enumerate(gap_positions):
                ax.axvline(
                    float(pos),
                    color=gap_color,
                    linestyle="--",
                    alpha=0.5,
                    linewidth=0.5,
                    label="Gap" if i == 0 else "",
                )
            ax.legend(loc="upper right")
    else:
        # 2D environment
        if show_field_count:
            env.plot_field(result.field_count.astype(float), ax=ax, cmap="viridis")
        else:
            binary_field = result.is_covered.astype(float)
            cmap = ListedColormap(["white", covered_color])
            env.plot_field(binary_field, ax=ax, cmap=cmap, vmin=0, vmax=1)

        # Overlay gap markers
        if highlight_gaps and len(result.uncovered_bins) > 0:
            ax.scatter(
                result.uncovered_positions[:, 0],
                result.uncovered_positions[:, 1],
                c=gap_color,
                marker=gap_marker,
                s=gap_marker_size,
                label="Gaps",
                zorder=10,
            )
            ax.legend(loc="upper right")

    # Add informative title
    ax.set_title(
        f"Bin Coverage: {result.coverage_fraction:.1%} "
        f"({result.n_place_cells}/{result.n_neurons} place cells, "
        f"{result.n_fields} fields)"
    )

    return ax


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
    >>> from neurospatial.encoding.population import field_density_map
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
    >>> from neurospatial.encoding.population import count_place_cells
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
    >>> from neurospatial.encoding.population import field_overlap
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
    >>> from neurospatial.encoding.population import population_vector_correlation
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


__all__ = [
    "PopulationCoverageResult",
    "count_place_cells",
    "field_density_map",
    "field_overlap",
    "plot_population_coverage",
    "population_coverage",
    "population_vector_correlation",
]
