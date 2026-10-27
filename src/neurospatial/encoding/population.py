"""Population-level place field metrics.

This module provides metrics for analyzing spatial representations across
populations of neurons, including coverage, overlap, and correlation measures.

See also: Population analysis guide,
<https://edeno.github.io/neurospatial/user-guide/population-analysis/>
Assembly detection, reactivation and pairwise spike-count correlations are in
``neurospatial.decoding``.

References
----------
.. [1] Wilson, M. A., & McNaughton, B. L. (1993). Dynamics of the hippocampal
       ensemble code for space. Science, 261(5124), 1055-1058.
.. [2] O'Keefe, J., & Nadel, L. (1978). The Hippocampus as a Cognitive Map.
       Oxford: Clarendon Press.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial.encoding._comparison import (
    _TOPOLOGIES,
    _UNIT_SELECTIONS,
    _estimator,
    _rates_2d,
    _warn_on_estimator_differences,
)
from neurospatial.encoding._comparison_inputs import (
    _bin_pairs,
    _Caller,
    _check_choice,
    _match_units,
)
from neurospatial.encoding._field_metrics import _relative_measures
from neurospatial.encoding._provenance import occupancy_support
from neurospatial.encoding.spatial import (
    PlaceFieldsResult,
    SpatialRateResult,
    SpatialRatesResult,
    _check_place_field_arguments,
    _keep_place_fields,
    _search_place_fields,
    detect_place_fields,
)
from neurospatial.environment import Environment

if TYPE_CHECKING:
    import matplotlib.axes

_PV = _Caller(
    operation="population_vector_correlation()",
    first="firing_rates_1",
    second="firing_rates_2",
    intersection_effect="correlate over the units recorded in both conditions",
)


@dataclass(frozen=True, repr=False, eq=False)
class PopulationCoverageResult(ReadOnlyArraysMixin, ResultMixin):
    """Results from population coverage analysis.

    Every count and bin set is derived from the detected fields and the
    observed bins, so they cannot disagree.

    Parameters
    ----------
    place_fields : sequence of PlaceFieldsResult, length n_neurons
        Detected place fields for each neuron (output of
        :func:`detect_place_fields`). A plain list of bin-index arrays is
        converted to a ``PlaceFieldsResult``.
    is_observed : NDArray[np.bool_], shape (n_bins,)
        Bins with a finite rate for at least one neuron. Unobserved bins are
        neither covered nor gaps.
    bin_centers : NDArray[np.floating], shape (n_bins, n_dims)
        Bin coordinates, used to locate gaps.
    bin_sizes : NDArray[np.float64], shape (n_bins,)
        Measure of each bin (``env.bin_sizes``: area, length or volume), so
        coverage is a fraction of space rather than of bins.

    Attributes
    ----------
    n_neurons : int
        Number of neurons, ``len(place_fields)``.
    n_place_cells : int
        Number of neurons with at least one detected field.
    n_fields : int
        Total number of fields across neurons.
    field_count : NDArray[np.int64], shape (n_bins,)
        Number of fields containing each bin.
    is_covered : NDArray[np.bool_], shape (n_bins,)
        Bins inside at least one field.
    covered_bins : NDArray[np.intp]
        Indices of covered bins.
    uncovered_bins : NDArray[np.intp]
        Indices of observed bins outside every field (gaps).
    uncovered_positions : NDArray[np.floating], shape (n_uncovered, n_dims)
        Coordinates of the gaps.
    coverage_fraction : float
        Covered fraction of the observed area, length or volume (0-1); NaN
        when no bin was observed.
    place_cell_fraction : float
        n_place_cells / n_neurons, a descriptive segmentation fraction.
    fields_per_place_cell : float
        n_fields / n_place_cells.
    mean_redundancy : float
        Average field_count over the covered area, length or volume.
    """

    place_fields: Sequence[PlaceFieldsResult]
    is_observed: NDArray[np.bool_]
    bin_centers: NDArray[np.floating]
    bin_sizes: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate, own the arrays and prevent replacing per-unit fields."""
        place_fields = tuple(
            fields
            if isinstance(fields, PlaceFieldsResult)
            else PlaceFieldsResult(fields=list(fields))
            for fields in self.place_fields
        )
        is_observed = np.asarray(self.is_observed, dtype=bool)
        bin_centers = np.asarray(self.bin_centers)
        bin_sizes = np.asarray(self.bin_sizes, dtype=np.float64)
        if is_observed.ndim != 1 or bin_centers.ndim != 2 or bin_sizes.ndim != 1:
            raise ValueError(
                _format_error(
                    "is_observed and bin_sizes must be (n_bins,) and bin_centers "
                    f"(n_bins, n_dims); got {is_observed.shape}, "
                    f"{bin_sizes.shape} and {bin_centers.shape}.",
                    why="Why: coverage is counted per bin",
                    fix="use population_coverage(env, firing_rates, bandwidth=rates.bandwidth) to build the result",
                )
            )
        if not len(bin_centers) == len(bin_sizes) == len(is_observed):
            raise ValueError(
                _format_error(
                    f"is_observed has {len(is_observed)} bins but bin_centers has "
                    f"{len(bin_centers)} and bin_sizes {len(bin_sizes)}.",
                    why="Why: all three describe the same environment bins",
                    fix=(
                        "pass env.bin_centers and env.bin_sizes for the "
                        "environment of the rate maps"
                    ),
                )
            )
        n_bins = len(is_observed)
        for neuron, fields in enumerate(place_fields):
            for bins in fields:
                if bins.size and bins.max() >= n_bins:
                    raise ValueError(
                        _format_error(
                            f"place_fields[{neuron}] has bin {int(bins.max())}, but "
                            f"there are only {n_bins} bins.",
                            why="Why: field bins index the environment bins",
                            fix="detect fields on rate maps from this environment",
                        )
                    )
        object.__setattr__(self, "place_fields", place_fields)
        object.__setattr__(self, "is_observed", is_observed)
        object.__setattr__(self, "bin_centers", bin_centers)
        object.__setattr__(self, "bin_sizes", bin_sizes)
        self._own_arrays()

    @property
    def n_neurons(self) -> int:
        """Number of neurons."""
        return len(self.place_fields)

    @property
    def n_place_cells(self) -> int:
        """Number of neurons with at least one detected field."""
        return sum(len(fields) > 0 for fields in self.place_fields)

    @property
    def n_fields(self) -> int:
        """Total number of detected fields across neurons."""
        return sum(len(fields) for fields in self.place_fields)

    @property
    def field_count(self) -> NDArray[np.int64]:
        """Number of fields containing each bin, shape (n_bins,)."""
        count = np.zeros(len(self.is_observed), dtype=np.int64)
        for fields in self.place_fields:
            for bins in fields:
                np.add.at(count, bins, 1)
        return count

    @property
    def is_covered(self) -> NDArray[np.bool_]:
        """Bins inside at least one field, shape (n_bins,)."""
        return self.field_count > 0

    @property
    def covered_bins(self) -> NDArray[np.intp]:
        """Indices of bins inside at least one field."""
        return np.flatnonzero(self.is_covered)

    @property
    def uncovered_bins(self) -> NDArray[np.intp]:
        """Indices of observed bins outside every field (gaps)."""
        return np.flatnonzero(self.is_observed & ~self.is_covered)

    @property
    def uncovered_positions(self) -> NDArray[np.floating]:
        """Coordinates of the gaps, shape (n_uncovered, n_dims)."""
        return self.bin_centers[self.uncovered_bins]

    @property
    def coverage_fraction(self) -> float:
        """Covered fraction of the observed space; NaN if none observed.

        Each bin counts by its measure (``bin_sizes``), so splitting a bin
        does not change the fraction. On a regular grid it is the fraction of
        observed bins that are covered.
        """
        if not self.is_observed.any():
            return np.nan
        measures = _relative_measures(self.bin_sizes)
        covered = self.is_covered & self.is_observed
        return float(np.sum(measures[covered]) / np.sum(measures[self.is_observed]))

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "coverage_fraction": self.coverage_fraction,
            "n_covered": int(self.covered_bins.size),
            "n_uncovered": int(self.uncovered_bins.size),
            "n_neurons": self.n_neurons,
            "n_place_cells": self.n_place_cells,
            "n_fields": self.n_fields,
        }

    @property
    def place_cell_fraction(self) -> float:
        """Fraction of neurons with at least one segmented map region.

        The name does not imply a cell-type test or place-cell prevalence.
        NaN when there are no neurons.

        References
        ----------
        .. [1] Wilson, M. A., & McNaughton, B. L. (1993). Dynamics of the
               hippocampal ensemble code for space. Science, 261(5124), 1055-1058.
        """
        if self.n_neurons == 0:
            return np.nan
        return self.n_place_cells / self.n_neurons

    @property
    def fields_per_place_cell(self) -> float:
        """Average number of segmented fields per map containing a field.

        This descriptive ratio depends on the segmentation settings and is
        not a cell-type classification. NaN when no map has a field.

        References
        ----------
        .. [1] Fenton, A. A., et al. (2008). Unmasking the CA1 ensemble place
               code by exposures to small and large environments: more place
               cells and multiple, irregularly arranged, and expanded place
               fields in the larger space. Journal of Neuroscience, 28(44),
               11250-11262.
        """
        if self.n_place_cells == 0:
            return np.nan
        return self.n_fields / self.n_place_cells

    @property
    def mean_redundancy(self) -> float:
        """Average number of place fields over the covered space.

        Measures the degree of overlapping representation. Higher values
        indicate more robust population coding with multiple cells
        representing each location. Each covered bin counts by its measure
        (``bin_sizes``); on a regular grid this is the mean field count per
        covered bin. NaN when no bin is covered.

        References
        ----------
        .. [1] Wilson, M. A., & McNaughton, B. L. (1993). Dynamics of the
               hippocampal ensemble code for space. Science, 261(5124), 1055-1058.
        """
        covered = self.covered_bins
        if covered.size == 0:
            return np.nan
        return float(
            np.average(
                self.field_count[covered],
                weights=_relative_measures(self.bin_sizes)[covered],
            )
        )

    def __str__(self) -> str:
        """Return human-readable summary for notebook inspection."""
        return (
            f"PopulationCoverageResult:\n"
            f"  Coverage: {self.coverage_fraction:.1%}\n"
            f"  Maps with fields: {self.n_place_cells}/{self.n_neurons} "
            f"({self.place_cell_fraction:.1%})\n"
            f"  Total fields: {self.n_fields} "
            f"({self.fields_per_place_cell:.1f} per map with fields)\n"
            f"  Mean redundancy: {self.mean_redundancy:.2f} fields/bin\n"
            f"  Gaps: {len(self.uncovered_bins)} bins"
        )


def population_coverage(
    env: Environment,
    firing_rates: NDArray[np.floating],
    *,
    bandwidth: float,
    threshold: float = 0.2,
    min_peak_rate: float = 1.0,
    min_secondary_peak_fraction: float = 0.0,
    max_mean_rate: float = np.inf,
    detect_subfields: bool = True,
    n_jobs: int = 1,
) -> PopulationCoverageResult:
    r"""Compute spatial coverage of a place cell population.

    Detects place fields for each neuron, then computes comprehensive
    coverage statistics including gap locations. Segmentation is descriptive:
    neurons with detected regions are not thereby classified as place cells.

    Parameters
    ----------
    env : Environment
        Environment defining the spatial bins.
    firing_rates : NDArray[np.floating], shape (n_neurons, n_bins)
        Firing rate maps for each neuron.
    bandwidth : float
        Smoothing bandwidth the maps were estimated with (``rates.bandwidth``;
        0 for unsmoothed maps). It sets the smallest field. Passed to
        `detect_place_fields()`.
    threshold : float, default=0.2
        Place field boundary threshold as fraction of peak firing rate (0-1).
        Lower values (0.1-0.2) detect larger fields, higher values (0.3-0.5)
        detect only core field regions. Passed to `detect_place_fields()`.
    min_peak_rate : float, default=1.0
        Smallest peak rate (Hz) that can start a field. Passed to
        `detect_place_fields()`.
    min_secondary_peak_fraction : float, default=0.0
        A field after a neuron's first must peak at least this fraction of
        that neuron's maximum rate; the default keeps every peak above
        ``min_peak_rate``. Passed to `detect_place_fields()`.
    max_mean_rate : float, default=np.inf
        Optional cutoff on the unweighted mean of non-NaN spatial-bin rates
        (Hz), passed to ``detect_place_fields``. The default disables exclusion.
        This is not a recording-time mean and does not identify cell type.
    detect_subfields : bool, default=True
        If True, separate two fields that merge at ``threshold`` (the
        FMAToolbox ``MapStats`` rule). Passed to `detect_place_fields()`.
    n_jobs : int, default=1
        Number of parallel jobs for per-neuron place-field detection. Use
        ``-1`` for all CPUs. ``1`` means sequential processing (no joblib
        overhead). Results are identical regardless of
        ``n_jobs`` (joblib preserves submission order). Note: per-neuron
        map-mean-exclusion ``UserWarning``\s from ``detect_place_fields``
        are not surfaced when ``n_jobs != 1`` (joblib worker warnings are
        commonly swallowed); inspect ``result.place_fields[i].excluded_reason``
        instead.

    Returns
    -------
    PopulationCoverageResult
        Coverage statistics, including ``coverage_fraction`` (covered
        fraction of the observed area, length or volume), ``uncovered_bins`` and
        ``uncovered_positions`` (gaps) and ``place_fields`` (detected fields
        per neuron). Bins that are NaN for every neuron were not observed:
        they are neither covered nor gaps.

    Raises
    ------
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
    >>> from neurospatial.encoding import compute_spatial_rates  # doctest: +SKIP
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
    >>> # 2. Compute firing rates for all neurons at once (shares occupancy/kernel work)
    >>> spike_times = [
    ...     ...
    ... ]  # List of spike time arrays, one per neuron  # doctest: +SKIP
    >>> firing_rates = compute_spatial_rates(  # doctest: +SKIP
    ...     env, spike_times, times, positions
    ... ).firing_rates  # Shape: (n_neurons, n_bins)
    >>>
    >>> # 3. Analyze coverage
    >>> result = population_coverage(env, firing_rates, bandwidth=5.0)  # doctest: +SKIP
    >>> print(f"Coverage: {result.coverage_fraction:.1%}")  # doctest: +SKIP
    >>> print(
    ...     f"Maps with fields: {result.n_place_cells}/{result.n_neurons}"
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
            f"To fix: Recompute firing rates for this environment with the batch function:\n"
            f"  from neurospatial.encoding import compute_spatial_rates\n"
            f"  firing_rates = compute_spatial_rates(env, spike_times, times, positions).firing_rates"
        )

    # Validate detection parameters
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"threshold must be in range (0, 1), got {threshold}. "
            f"This represents the fraction of peak firing rate for field boundaries."
        )

    if max_mean_rate <= 0:
        raise ValueError(f"max_mean_rate must be positive, got {max_mean_rate}")

    n_neurons = firing_rates.shape[0]

    if n_jobs == 1:
        # detect_place_fields for each map, with the smallest-field test of
        # every map's peaks diffused in one batch.
        if n_neurons:
            _check_place_field_arguments(
                threshold=threshold,
                min_peak_rate=min_peak_rate,
                min_secondary_peak_fraction=min_secondary_peak_fraction,
                bandwidth=bandwidth,
            )
        searches = [
            _search_place_fields(
                env,
                np.asarray(firing_rates[neuron_idx], dtype=np.float64),
                threshold=threshold,
                min_peak_rate=min_peak_rate,
                min_secondary_peak_fraction=min_secondary_peak_fraction,
                max_mean_rate=max_mean_rate,
                detect_subfields=detect_subfields,
            )
            for neuron_idx in range(n_neurons)
        ]
        all_fields = _keep_place_fields(env, searches, bandwidth, threshold)
    else:
        from joblib import Parallel, delayed

        # joblib preserves submission order, so all_fields is in neuron order.
        all_fields = list(
            Parallel(n_jobs=n_jobs)(
                delayed(detect_place_fields)(
                    env,
                    firing_rates[neuron_idx],
                    bandwidth=bandwidth,
                    threshold=threshold,
                    min_peak_rate=min_peak_rate,
                    min_secondary_peak_fraction=min_secondary_peak_fraction,
                    max_mean_rate=max_mean_rate,
                    detect_subfields=detect_subfields,
                )
                for neuron_idx in range(n_neurons)
            )
        )

    return PopulationCoverageResult(
        place_fields=all_fields,
        is_observed=np.isfinite(firing_rates).any(axis=0),
        bin_centers=env.bin_centers,
        bin_sizes=env.bin_sizes,
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
        Environment for plotting.
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

    Notes
    -----
    For 1-D environments (``env.n_dims == 1``) and linearized tracks, plots the
    coverage as a line along the track and marks each gap with a vertical
    line at its position (the linearized position on a track).

    Examples
    --------
    >>> ax = plot_population_coverage(env, result)  # doctest: +SKIP

    >>> ax = plot_population_coverage(
    ...     env, result, show_field_count=True
    ... )  # doctest: +SKIP
    """
    # Validate result type
    if not isinstance(result, PopulationCoverageResult):
        raise TypeError(
            f"result must be PopulationCoverageResult, got {type(result).__name__}. "
            f"Use population_coverage() to compute the result first."
        )

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Create axes if needed
    if ax is None:
        _, ax = plt.subplots()

    # 1-D environments: a native 1-D grid or a track drawn in its linear coordinate
    if env.n_dims == 1 or env.is_linearized_track:
        if show_field_count:
            field_to_plot = result.field_count.astype(float)
            env.plot_field(field_to_plot, ax=ax)
            ax.set_ylabel("Field count")
        else:
            field_to_plot = np.where(result.is_observed, result.is_covered, np.nan)
            env.plot_field(field_to_plot, ax=ax)
            ax.set_ylabel("Coverage")

        # Highlight gaps as vertical lines with legend
        if highlight_gaps and len(result.uncovered_bins) > 0:
            linearization = env.binning.linearization
            if linearization is None:
                gap_positions = result.uncovered_positions[:, 0]
            else:
                # The track is drawn in its linear coordinate, where segments
                # are separated by the layout's gaps; place each gap there.
                edges = linearization.grid_edges[0]
                linear_centers = ((edges[:-1] + edges[1:]) / 2)[
                    linearization.active_mask.ravel()
                ]
                gap_positions = linear_centers[result.uncovered_bins]
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
            # Unobserved bins stay NaN: they are neither covered nor gaps.
            binary_field = np.where(result.is_observed, result.is_covered, np.nan)
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
        f"({result.n_place_cells}/{result.n_neurons} maps with fields, "
        f"{result.n_fields} fields)"
    )

    return ax


def field_density_map(
    all_place_fields: list[list[NDArray[np.int64]] | PlaceFieldsResult],
    n_bins: int,
) -> NDArray[np.int64]:
    """
    Count number of overlapping place fields per bin.

    Creates a density map showing how many cells have place fields at each
    location. High-density bins indicate regions represented by multiple cells.

    Parameters
    ----------
    all_place_fields : list of (list of array OR PlaceFieldsResult)
        Place fields for each cell. Each element may be either a bare
        ``list[NDArray[np.int64]]`` (hand-built input) or a
        ``PlaceFieldsResult`` (the canonical return of
        :func:`detect_place_fields`); both forms iterate as
        ``Iterable[NDArray[np.int64]]``.
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
    rates: SpatialRatesResult, *, min_info: float | None = None
) -> int:
    """
    Count units that pass the place-cell information screen.

    The count of :meth:`SpatialRatesResult.classify`: units whose Skaggs
    spatial information [1]_, minus the information an untuned cell shows by
    chance on the same maps (``rates.spatial_information_bias()``), is at or
    above ``min_info``. Without that subtraction, short recordings count
    untuned cells; use ``place_cell_significance`` for a shuffle test.

    Parameters
    ----------
    rates : SpatialRatesResult
        Population rate maps from ``compute_spatial_rates``. The bias depends
        on the maps' occupancy, estimator and bandwidth, so information
        values alone cannot be screened.
    min_info : float or None, default=None
        Inclusive minimum bias-corrected information (bits/spike); None is
        0.5 bits/spike, a common screening convention, not derived or calibrated.

    Returns
    -------
    count : int
        Number of units that pass. Silent units and units with fewer spikes
        than the maps have effectively independent bins do not count.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import compute_spatial_rates
    >>> from neurospatial.encoding.population import count_place_cells
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (36000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0, units="cm")
    >>> times = np.arange(36000) / 30.0
    >>> field = np.exp(-np.sum((positions - 50.0) ** 2, axis=1) / (2 * 10.0**2))
    >>> place = times[rng.random(36000) < 8.0 * field / 30.0]
    >>> untuned = np.sort(rng.uniform(0, 1200, 600))
    >>> rates = compute_spatial_rates(env, [place, untuned], times, positions)
    >>> count_place_cells(rates)
    1

    See Also
    --------
    SpatialRatesResult.classify : The per-unit screen this counts.
    SpatialRatesResult.spatial_information_bias : The subtracted bias.

    Notes
    -----
    The screen does not separate interneurons or other cell classes.

    References
    ----------
    .. [1] Skaggs, W. E., McNaughton, B. L., Gothard, K. M., & Markus, E. J.
           (1993). An information-theoretic approach to deciphering the
           hippocampal code. Advances in Neural Information Processing
           Systems 5, 1030-1037.
    """
    if not isinstance(rates, SpatialRatesResult):
        raise TypeError(
            _format_error(
                "count_place_cells needs a SpatialRatesResult, got "
                f"{type(rates).__name__}.",
                why="Why: the screen subtracts the information an untuned cell "
                "shows on the same maps, which depends on their occupancy, "
                "estimator and bandwidth; information values alone lack them.",
                fix="pass rates = compute_spatial_rates(env, spike_times, times, "
                "positions).",
            )
        )
    return int(np.count_nonzero(rates.classify(min_info=min_info)))


def field_overlap(
    env: Environment,
    field_bins_i: NDArray[np.int64],
    field_bins_j: NDArray[np.int64],
) -> float:
    """
    Compute Jaccard similarity between two place fields.

    Measures spatial overlap as the area of the intersection over the area of
    the union (Jaccard coefficient). Used to quantify remapping and field
    stability.

    Parameters
    ----------
    env : Environment
        Spatial environment of the fields. Its bin measures
        (``env.bin_sizes``) make the overlap a ratio of areas.
    field_bins_i : array
        Bin indices in first field.
    field_bins_j : array
        Bin indices in second field.

    Returns
    -------
    overlap : float
        Jaccard coefficient = area(intersection) / area(union), with lengths
        on a track and volumes in 3-D. Range [0, 1].
        0.0 if exactly one field is empty: a field in one set and none in the
        other is a real absence of overlap. NaN if both are empty: with no
        field bin in either set there is nothing to compare, which is not the
        same as no overlap.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.population import field_overlap
    >>> # A 10 cm track in 1 cm bins
    >>> env = Environment.from_grid_mask(
    ...     np.ones(10, dtype=bool), grid_edges=(np.arange(11.0),), units="cm"
    ... )
    >>> # Identical fields (perfect overlap)
    >>> field1 = np.array([0, 1, 2, 3])
    >>> field2 = np.array([0, 1, 2, 3])
    >>> overlap = field_overlap(env, field1, field2)
    >>> print(f"{overlap:.2f}")
    1.00

    >>> # Partial overlap
    >>> field1 = np.array([0, 1, 2, 3])
    >>> field2 = np.array([2, 3, 4, 5])
    >>> overlap = field_overlap(env, field1, field2)
    >>> # Intersection: {2, 3}, Union: {0,1,2,3,4,5}
    >>> print(f"{overlap:.2f}")  # 2/6 = 0.33
    0.33

    >>> # No field in either set: undefined, not zero overlap
    >>> empty = np.array([], dtype=np.int64)
    >>> field_overlap(env, field1, empty)
    0.0
    >>> field_overlap(env, empty, empty)
    nan

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

    Each bin counts by its measure (``env.bin_sizes``: area, length or
    volume), so the overlap does not depend on how finely the fields are
    binned. On a regular grid it is the ratio of bin counts.

    When comparing conditions, restrict both field bin sets to bins observed
    in both conditions first; a bin one condition never visited cannot show
    whether a field is there.

    References
    ----------
    .. [1] Muller, R. U., & Kubie, J. L. (1987). The effects of changes in
           the environment on the spatial firing of hippocampal complex-spike
           cells. Journal of Neuroscience, 7(7), 1951-1968.
    """
    union = np.union1d(field_bins_i, field_bins_j).astype(np.intp)
    if union.size == 0:
        # No field bin in either set: nothing to compare.
        return np.nan
    intersection = np.intersect1d(field_bins_i, field_bins_j).astype(np.intp)
    measures = _relative_measures(env.bin_sizes)
    return float(np.sum(measures[intersection]) / np.sum(measures[union]))


def rate_map_correlations(
    firing_rates: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Pearson correlation between every pair of rate maps.

    Each row of ``firing_rates`` is one rate map: a cell, or one cell in one
    condition. The result is the map-by-map correlation matrix, computed over
    bins. It is not the population-vector correlation of remapping studies,
    which correlates all cells' rates at each location.

    Parameters
    ----------
    firing_rates : array, shape (n_maps, n_bins)
        One rate map per row, one bin per column. A NaN bin makes every
        correlation involving that row NaN, so pass only bins supported in all
        maps, for example ``rates.firing_rates[:, shared]`` with
        ``shared = a.spatial_support & b.spatial_support``.

    Returns
    -------
    correlation_matrix : array, shape (n_maps, n_maps)
        Pearson correlations, symmetric, off-diagonal entries in [-1, 1].
        The diagonal is 1.0. NaN propagates per row: a map with a NaN bin or
        with constant rates has no defined correlation, so its row, column
        and diagonal entry are NaN, and the other entries are unchanged.
        A map is constant when all its rates are exactly equal, even if their
        standard deviation rounds to a tiny nonzero value. With no bins, every
        entry is NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding import rate_map_correlations
    >>> rng = np.random.default_rng(42)
    >>> firing_rates = rng.random((3, 50))
    >>> corr = rate_map_correlations(firing_rates)
    >>> corr.shape
    (3, 3)
    >>> np.diag(corr)
    array([1., 1., 1.])

    Compare the same cells in two conditions: stack the maps, then read the
    diagonal of the before-versus-after block.

    >>> before = rng.random((4, 50))
    >>> after = before.copy()
    >>> after[2] = rng.random(50)  # cell 2 remaps
    >>> corr = rate_map_correlations(np.vstack([before, after]))
    >>> per_cell = np.diag(corr[:4, 4:])
    >>> np.round(per_cell[[0, 1, 3]], 2)
    array([1., 1., 1.])
    >>> bool(abs(per_cell[2]) < 0.5)
    True

    A constant map has no defined correlation.

    >>> flat = np.vstack([np.ones(50), firing_rates[0]])
    >>> np.isnan(rate_map_correlations(flat)).tolist()
    [[True, True], [True, False]]

    See Also
    --------
    population_vector_correlation : Correlation across cells at each location
    field_stability : Correlation between one cell's maps in two conditions
    field_overlap : Jaccard-based overlap of place fields

    Notes
    -----
    Every bin counts once, the per-pixel convention of rate-map correlations
    in remapping studies. On environments with unequal bins (clipped grids,
    meshes) small bins therefore weigh as much as large ones.
    """
    maps = np.atleast_2d(np.asarray(firing_rates, dtype=np.float64))
    if maps.shape[1] == 0:
        # No bins, for example no bin supported in both conditions.
        return np.full((maps.shape[0], maps.shape[0]), np.nan)

    # A constant map divides by zero; the NaN it produces is the documented result.
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation_matrix = np.atleast_2d(np.corrcoef(maps))

    # Exact constancy: the std of equal rates can round to ~1e-17, which
    # corrcoef would turn into a meaningless correlation instead of NaN.
    constant = maps.max(axis=1) == maps.min(axis=1)
    correlation_matrix[constant, :] = np.nan
    correlation_matrix[:, constant] = np.nan

    # Self-correlation is 1.0 up to rounding; keep NaN for undefined maps.
    diagonal = np.where(np.isnan(np.diag(correlation_matrix)), np.nan, 1.0)
    np.fill_diagonal(correlation_matrix, diagonal)

    return correlation_matrix


def population_vector_correlation(
    firing_rates_1: SpatialRatesResult | NDArray[np.float64],
    firing_rates_2: SpatialRatesResult | NDArray[np.float64],
    *,
    spatial_support: NDArray[np.bool_] | None = None,
    unit_selection: Literal["intersection", "exact"] = "intersection",
    bin_mapping: ArrayLike | None = None,
    topology: Literal["same", "may_differ"] = "same",
) -> NDArray[np.float64]:
    r"""
    Population-vector correlation at each location between two conditions.

    At each location (bin), the population vector is every cell's rate there.
    The result correlates that vector in condition 1 with the same location's
    vector in condition 2, across cells, giving one value per location. This
    is the remapping measure of Leutgeb et al. (2005): near 1 where the
    population fires the same way in both conditions, near 0 where the code at
    that location is unrelated.

    Pass the two ``SpatialRatesResult`` of ``compute_spatial_rates(...,
    epochs=..., unit_ids=...)``, one per condition. Units are then paired by
    ``unit_ids``, never by row position, and bins by place, as in
    ``compare_spatial_rates``: entry ``c`` of both population vectors is the
    same cell, and both are read at the same location. Only bins observed in
    both conditions are compared, so an unvisited bin is never read as 0 Hz.

    Parameters
    ----------
    firing_rates_1, firing_rates_2 : SpatialRatesResult or array, shape (n_units, n_bins)
        Rate maps of the two conditions. Pass two ``SpatialRatesResult``
        (recommended) or two arrays; a result and an array cannot be mixed.

        - Results: units are matched by ``unit_ids`` (see
          ``unit_selection``), so each result needs real labels: pass the
          sorter's unit labels as ``unit_ids=`` to ``compute_spatial_rates``.
          Labels generated by the estimator (``0..n-1``) are refused, since a
          row position does not identify the same cell in two results.
        - Arrays: one unit per row in the same order in both, one bin per
          column. Matching the rows is then the caller's job: a correlation
          across cells is meaningless if row ``i`` is a different unit in each
          condition. A NaN rate excludes that location.
    spatial_support : array of bool, shape (n_bins,), optional
        Arrays only. Bins observed in both conditions, for example
        ``rates_1.spatial_support & rates_2.spatial_support``. Pass it for
        any arrays taken from ``compute_spatial_rates``: smoothing (the
        default ``method="diffusion_kde"``) gives finite rates in bins the
        animal never visited, and zero-filling reads them as 0 Hz, so NaN
        rates alone do not mark every unvisited bin. Locations outside it
        are NaN. Results carry their own support, so passing it with results
        raises.
    unit_selection : {"intersection", "exact"}, default="intersection"
        Results only. ``"intersection"`` correlates over the units present in
        both conditions; the others are left out (find them with
        ``np.setdiff1d(rates_1.unit_ids, rates_2.unit_ids)``). ``"exact"``
        raises unless both conditions hold the same units, in any order.
        Arrays carry no labels, so ``"exact"`` with arrays raises.
    bin_mapping : array-like of int, shape (n_pairs, 2), optional
        Results only. Rows of ``(bin in firing_rates_1.env, bin in
        firing_rates_2.env)`` pairing bins that are the same place, for maps
        estimated on different environments such as a subset of the arena;
        ``firing_rates_2.env.bin_mapping_from(firing_rates_1.env).pairs``
        builds them for an environment derived from the other.
        Checked as in ``compare_spatial_rates``: one-to-one, and paired bins
        must be the same cells in the same coordinates. By default the two
        environments must have the same bins.
    topology : {"same", "may_differ"}, default="same"
        Results only. ``"may_differ"`` accepts environments whose bins are
        the same places but whose connectivity, walls or cell faces differ,
        as when a door opens between conditions; smoothing then passed a
        field between bins differently in each condition. See
        ``compare_spatial_rates`` for how topology is compared.

    Returns
    -------
    correlation : array, shape (n_bins,)
        Pearson correlation across units at each location, in [-1, 1]. With
        results, ``n_bins`` is ``firing_rates_1.env.n_bins`` and entry ``j``
        is bin ``j`` of that environment; with ``bin_mapping``, bins without
        a mapped partner are NaN. NaN where the location is not used (not
        observed in both conditions, a NaN rate in either map, or outside
        ``spatial_support``) or where either population vector is flat
        (every unit has the same rate there). NaN means undefined, never
        zero correlation.

    Raises
    ------
    TypeError
        If a result is mixed with an array, or an input is a single-unit
        ``SpatialRateResult`` or another kind of result.
    ValueError
        If a result has generated ``unit_ids``, the units cannot be matched,
        an option is invalid, ``spatial_support`` is passed with results, or
        a results-only option is passed with arrays. With arrays: if either
        map is not 2-D, the two shapes differ, there are fewer than two
        units, or ``spatial_support`` is not a bool array of shape
        ``(n_bins,)``.
    IncompatibleEnvironmentError
        If the two results' environments do not have the same bins and no
        ``bin_mapping`` is given, a mapping pairs bins that are not the same
        place, or their topology differs and ``topology="same"``.

    Warns
    -----
    UserWarning
        If two results were estimated with different ``method``,
        ``bandwidth`` or ``min_occupancy``: the population vectors then
        differ partly because of the estimation.

    See Also
    --------
    compare_spatial_rates : Correlation of each cell's map between conditions
    rate_map_correlations : Correlation between every pair of rate maps
    field_overlap : Jaccard overlap of place fields

    Notes
    -----
    With ``a`` and ``b`` the population vectors at one location, each centered
    by subtracting its mean across units,

    .. math::

        r = \frac{\sum_c a_c b_c}{\sqrt{\sum_c a_c^2 \sum_c b_c^2}},

    clipped to [-1, 1] against rounding. A location is flat when every unit
    has exactly the same rate there (maximum equals minimum); the correlation
    is then 0/0 and the result is NaN. The check is exact, so equal rates
    whose standard deviation rounds to about 1e-17 are still flat.

    With results, a location is observed in both conditions when each
    result's occupancy there reaches its ``min_occupancy`` and every compared
    unit's rate is finite. When every unit is compared this is
    ``firing_rates_1.spatial_support & firing_rates_2.spatial_support``; a
    unit left out by ``unit_selection="intersection"`` does not remove bins.

    Summarize over the defined locations, for example with the median, and
    report how many were undefined:
    ``np.median(r[np.isfinite(r)])`` with ``(~np.isfinite(r)).sum()``.
    Because the measure correlates rates across cells, it drops both when
    fields move (global remapping) and when cells change their rates relative
    to each other (rate remapping); a change of every rate by the same factor
    leaves it unchanged.

    References
    ----------
    .. [1] Leutgeb, S., Leutgeb, J. K., Barnes, C. A., Moser, E. I.,
           McNaughton, B. L., & Moser, M.-B. (2005). Independent codes for
           spatial and episodic memory in hippocampal neuronal ensembles.
           Science, 309(5734), 619-623. https://doi.org/10.1126/science.1114037

    Examples
    --------
    Does the population code at each place change between the two halves of a
    session? Estimate each half with the sorter's unit labels:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import (
    ...     compute_spatial_rates,
    ...     population_vector_correlation,
    ... )
    >>> times = np.arange(3000) / 30.0
    >>> positions = 50 + 40 * np.column_stack(
    ...     [np.sin(2 * np.pi * times / 20), np.cos(2 * np.pi * times / 13)]
    ... )
    >>> env = Environment.from_samples(positions, bin_size=10.0, units="cm")
    >>> rng = np.random.default_rng(0)
    >>> spike_times = [np.sort(rng.uniform(0, 100, n)) for n in (300, 200, 400)]
    >>> first_half, second_half = (
    ...     compute_spatial_rates(
    ...         env,
    ...         spike_times,
    ...         times,
    ...         positions,
    ...         epochs=[window],
    ...         unit_ids=[10, 11, 12],
    ...     )
    ...     for window in ([0.0, 50.0], [50.0, 100.0])
    ... )
    >>> pv = population_vector_correlation(first_half, second_half)
    >>> pv.shape == (env.n_bins,)
    True

    Units are paired by label, so listing the second half's units in another
    order changes nothing:

    >>> reordered = compute_spatial_rates(
    ...     env,
    ...     spike_times[::-1],
    ...     times,
    ...     positions,
    ...     epochs=[[50.0, 100.0]],
    ...     unit_ids=[12, 11, 10],
    ... )
    >>> np.array_equal(
    ...     population_vector_correlation(first_half, reordered), pv, equal_nan=True
    ... )
    True

    Summarize over the defined locations and say how many were undefined:

    >>> defined = np.isfinite(pv)
    >>> int((~defined).sum()) == int(
    ...     (~(first_half.spatial_support & second_half.spatial_support)).sum()
    ... )
    True

    With your own matrices, pass arrays whose rows are the same units in the
    same order. Location 1 below is flat (every unit fires at 2 Hz), so it
    has no defined correlation; opposite population vectors give -1:

    >>> rates = np.array([[1.0, 2.0, 5.0], [3.0, 2.0, 1.0], [2.0, 2.0, 4.0]])
    >>> np.round(population_vector_correlation(rates, rates), 2)
    array([ 1., nan,  1.])
    >>> np.round(population_vector_correlation(rates, -rates), 2)
    array([-1., nan, -1.])

    With arrays, pass the bins observed in both conditions so an unvisited
    bin is not compared, whether it holds 0 Hz or a smoothed rate:

    >>> filled = rates.copy()
    >>> filled[:, 0] = 0.0  # bin 0 was never visited
    >>> visited = np.array([False, True, True])
    >>> pv = population_vector_correlation(filled, rates, spatial_support=visited)
    >>> np.round(pv, 2)
    array([nan, nan,  1.])
    """
    _check_choice("unit_selection", unit_selection, _UNIT_SELECTIONS, _PV)
    _check_choice("topology", topology, _TOPOLOGIES, _PV)
    is_result_1 = _is_population_result(firing_rates_1, _PV.first)
    is_result_2 = _is_population_result(firing_rates_2, _PV.second)
    if is_result_1 != is_result_2:
        result_name, array_name = (
            (_PV.first, _PV.second) if is_result_1 else (_PV.second, _PV.first)
        )
        raise TypeError(
            _format_error(
                f"{_PV.operation} got a SpatialRatesResult for {result_name} and "
                f"an array for {array_name}.",
                why="Why: a result pairs units by unit_ids and bins by place, and "
                "an array has neither, so the two maps would be paired by label "
                "on one side and by row position on the other.",
                fix="pass the SpatialRatesResult of compute_spatial_rates() for "
                "both conditions (recommended), or pass both as arrays "
                "(rates.firing_rates) with the rows in the same unit order",
            )
        )
    if is_result_1:
        return _population_vector_correlation_of_results(
            cast("SpatialRatesResult", firing_rates_1),
            cast("SpatialRatesResult", firing_rates_2),
            spatial_support=spatial_support,
            unit_selection=unit_selection,
            bin_mapping=bin_mapping,
            topology=topology,
        )

    results_only = [
        option
        for option, given in (
            ("unit_selection='exact'", unit_selection == "exact"),
            ("bin_mapping=", bin_mapping is not None),
            ("topology='may_differ'", topology == "may_differ"),
        )
        if given
    ]
    if results_only:
        raise ValueError(
            _format_error(
                f"{_PV.operation} got {', '.join(results_only)} with two arrays.",
                why="Why: arrays carry no unit labels or environment, so their "
                "rows pair by position and their columns by index; these options "
                "apply only to two SpatialRatesResult.",
                fix="pass the SpatialRatesResult of compute_spatial_rates() for "
                "both conditions, or drop the option and line up the rows and "
                "columns of the arrays yourself",
            )
        )
    rates_1 = np.asarray(firing_rates_1, dtype=np.float64)
    rates_2 = np.asarray(firing_rates_2, dtype=np.float64)
    if rates_1.ndim != 2 or rates_2.ndim != 2:
        raise ValueError(
            "firing_rates_1 and firing_rates_2 must be 2-D (n_units, n_bins), "
            f"got shapes {rates_1.shape} and {rates_2.shape}. Pass one rate map "
            "per unit, for example rates.firing_rates from compute_spatial_rates."
        )
    if rates_1.shape != rates_2.shape:
        raise ValueError(
            "firing_rates_1 and firing_rates_2 must have the same shape, got "
            f"{rates_1.shape} and {rates_2.shape}. Compute both maps for the same "
            "units on the same environment, with the units in the same row order."
        )
    n_units, n_bins = rates_1.shape
    if n_units < 2:
        raise ValueError(
            f"A population-vector correlation needs at least two units, got {n_units}: "
            "it correlates rates across cells at each location. For one cell, "
            "compare its two maps with field_stability."
        )

    used = np.ones(n_bins, dtype=bool)
    if spatial_support is not None:
        support = np.asarray(spatial_support)
        if support.dtype != np.bool_ or support.shape != (n_bins,):
            raise ValueError(
                f"spatial_support must be a bool array of shape ({n_bins},), got "
                f"dtype {support.dtype} and shape {support.shape}. Pass the bins "
                "observed in both conditions, for example "
                "rates_1.spatial_support & rates_2.spatial_support."
            )
        used &= support
    return _correlate_population_vectors(rates_1, rates_2, used)


def _is_population_result(rates: object, name: str) -> bool:
    """Whether ``rates`` is a population rate result; refuse other results."""
    if isinstance(rates, SpatialRatesResult):
        return True
    if isinstance(rates, SpatialRateResult):
        raise TypeError(
            _format_error(
                f"{_PV.operation} got a single-unit SpatialRateResult for {name}.",
                why="Why: a population vector holds every cell's rate at one "
                "place, so it needs a population of units.",
                fix="pass the SpatialRatesResult of compute_spatial_rates() for "
                "each condition; to compare one cell's two maps, use "
                "compare_spatial_rates()",
            )
        )
    if isinstance(rates, ResultMixin):
        raise TypeError(
            _format_error(
                f"{_PV.operation} got {name} of type {type(rates).__name__}; it "
                "correlates two SpatialRatesResult or two (n_units, n_bins) "
                "arrays.",
                why="Why: directional, view, object-vector and boundary-vector "
                "maps are not maps of the animal's position.",
                fix="pass the results of compute_spatial_rates() for each condition",
            )
        )
    return False


def _population_vector_correlation_of_results(
    rates_1: SpatialRatesResult,
    rates_2: SpatialRatesResult,
    *,
    spatial_support: NDArray[np.bool_] | None,
    unit_selection: str,
    bin_mapping: ArrayLike | None,
    topology: str,
) -> NDArray[np.float64]:
    """Pair units by label and bins by place, then correlate per location."""
    if spatial_support is not None:
        raise ValueError(
            _format_error(
                f"{_PV.operation} got spatial_support= with two SpatialRatesResult.",
                why="Why: each result carries the bins it observed, and only bins "
                "observed in both conditions are compared "
                "(firing_rates_1.spatial_support & "
                "firing_rates_2.spatial_support); a second mask could disagree "
                "with them.",
                fix="drop spatial_support=; to look at some bins only, index the "
                "returned array, e.g. pv[env.bins_in_region('center')]",
            )
        )
    unit_ids, rows_1, rows_2, _, _ = _match_units(rates_1, rates_2, unit_selection, _PV)
    if unit_ids.size < 2:
        raise ValueError(
            _format_error(
                f"{_PV.operation}: only unit {unit_ids.tolist()} is recorded in "
                "both conditions.",
                why="Why: a population-vector correlation correlates rates across "
                "cells at each location, so it needs at least two shared units.",
                fix="check the unit_ids of both results; to compare one cell's two "
                "maps, use compare_spatial_rates()",
            )
        )
    pairs, _ = _bin_pairs(rates_1.env, rates_2.env, bin_mapping, topology, _PV)
    bins_1, bins_2 = pairs[:, 0], pairs[:, 1]
    used = occupancy_support(rates_1)[bins_1] & occupancy_support(rates_2)[bins_2]
    paired = _correlate_population_vectors(
        _rates_2d(rates_1)[np.ix_(rows_1, bins_1)],
        _rates_2d(rates_2)[np.ix_(rows_2, bins_2)],
        used,
    )
    # Called from population_vector_correlation, one frame above compare's.
    _warn_on_estimator_differences(
        _estimator(rates_1), _estimator(rates_2), _PV, stacklevel=4
    )
    correlation = np.full(rates_1.env.n_bins, np.nan)
    correlation[bins_1] = paired
    return correlation


def _correlate_population_vectors(
    rates_1: NDArray[np.float64],
    rates_2: NDArray[np.float64],
    used: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Pearson correlation across rows at each column.

    Parameters
    ----------
    rates_1, rates_2 : ndarray, shape (n_units, n_bins)
        Rates of the same units (rows) at the same places (columns).
    used : ndarray of bool, shape (n_bins,)
        Columns observed in both conditions. Columns with a non-finite rate
        are dropped as well.

    Returns
    -------
    ndarray, shape (n_bins,)
        NaN at unused or flat columns.
    """
    used = used & np.isfinite(rates_1).all(axis=0) & np.isfinite(rates_2).all(axis=0)
    a = rates_1[:, used]
    b = rates_2[:, used]
    # Exact flatness: the standard deviation of equal floats can round to ~1e-17.
    varies = (a.max(axis=0) > a.min(axis=0)) & (b.max(axis=0) > b.min(axis=0))
    a = a[:, varies] - a[:, varies].mean(axis=0)
    b = b[:, varies] - b[:, varies].mean(axis=0)
    r = np.sum(a * b, axis=0) / np.sqrt(np.sum(a**2, axis=0) * np.sum(b**2, axis=0))

    correlation = np.full(rates_1.shape[1], np.nan)
    correlation[np.flatnonzero(used)[varies]] = np.clip(r, -1.0, 1.0)
    return correlation


__all__ = [
    "PopulationCoverageResult",
    "count_place_cells",
    "field_density_map",
    "field_overlap",
    "plot_population_coverage",
    "population_coverage",
    "population_vector_correlation",
    "rate_map_correlations",
]
