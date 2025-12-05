"""Spatial view cell metrics for analyzing view-based neural representations.

Spatial view cells fire when an animal *views* a specific location, regardless
of where the animal is positioned. This is fundamentally different from place
cells, which fire based on the animal's position. This module provides metrics
for distinguishing spatial view cells from place cells.

Which Function Should I Use?
----------------------------
**Computing metrics from raw data?**
    Use ``spatial_view_cell_metrics()`` to compute place field vs view field
    metrics and classification.

**Screening many neurons?**
    Use ``is_spatial_view_cell()`` for fast boolean filtering.

Typical Workflow
----------------
1. Compute metrics from spike times and behavioral data::

    >>> metrics = spatial_view_cell_metrics(  # doctest: +SKIP
    ...     env, spike_times, times, positions, headings
    ... )

2. Inspect metrics and classify::

    >>> print(metrics)  # Human-readable interpretation  # doctest: +SKIP
    >>> if metrics.is_spatial_view_cell:  # doctest: +SKIP
    ...     print("Spatial view cell detected!")

3. Or use the quick classifier::

    >>> if is_spatial_view_cell(env, spike_times, times, positions, headings):
    ...     print("Spatial view cell!")  # doctest: +SKIP

Key Difference: Place Cells vs Spatial View Cells
-------------------------------------------------
- **Place cell**: Fires when animal is *at* a specific location
- **Spatial view cell**: Fires when animal is *looking at* a specific location

For place cells, both place field and view field are similar (because viewing
location correlates with position). For spatial view cells, the view field
has higher spatial information than the place field, and the two fields are
dissimilar (low correlation).

Classification Criteria
-----------------------
A neuron is classified as a spatial view cell if:

1. ``view_info > ratio * place_info``: View field has significantly higher
   spatial information than place field
2. ``view_place_correlation < max_corr``: View and place fields are dissimilar

The default thresholds (ratio=1.5, max_corr=0.7) are based on the principle
that spatial view cells should have view-selective, not position-selective,
firing patterns.

References
----------
Rolls, E. T., et al. (1997). Spatial view cells in the primate hippocampus.
    European Journal of Neuroscience, 9(8), 1789-1794.
Georges-François, P., Rolls, E. T., & Robertson, R. G. (1999). Spatial view
    cells in the primate hippocampus: allocentric view not head direction or
    eye position or place. Cerebral Cortex, 9(3), 197-212.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from neurospatial.metrics.place_fields import (
    rate_map_coherence,
    skaggs_information,
    sparsity,
)
from neurospatial.spatial_view_field import compute_spatial_view_field
from neurospatial.spike_field import compute_place_field

if TYPE_CHECKING:
    from neurospatial import Environment

__all__ = [
    "SpatialViewMetrics",
    "is_spatial_view_cell",
    "spatial_view_cell_metrics",
]


@dataclass(frozen=True)
class SpatialViewMetrics:
    """Metrics for spatial view cell analysis.

    Attributes
    ----------
    view_field_skaggs_info : float
        Skaggs spatial information (bits/spike) for the view field.
        Higher values indicate stronger spatial selectivity when binned
        by *viewed location*.
    place_field_skaggs_info : float
        Skaggs spatial information (bits/spike) for the place field.
        Higher values indicate stronger spatial selectivity when binned
        by *animal position*.
    view_place_correlation : float
        Pearson correlation between view field and place field.
        Low values indicate the two representations are dissimilar.
    view_field_sparsity : float
        Sparsity of the view field. Lower values indicate sparser, more
        selective firing in view space.
    view_field_coherence : float
        Spatial coherence of the view field. Higher values indicate
        smoother, more spatially structured firing.
    is_spatial_view_cell : bool
        True if classified as a spatial view cell based on default criteria.

    Examples
    --------
    >>> from neurospatial.metrics.spatial_view_cells import SpatialViewMetrics
    >>> metrics = SpatialViewMetrics(
    ...     view_field_skaggs_info=1.5,
    ...     place_field_skaggs_info=0.5,
    ...     view_place_correlation=0.3,
    ...     view_field_sparsity=0.2,
    ...     view_field_coherence=0.7,
    ...     is_spatial_view_cell=True,
    ... )
    >>> metrics.view_field_skaggs_info
    1.5
    """

    view_field_skaggs_info: float
    place_field_skaggs_info: float
    view_place_correlation: float
    view_field_sparsity: float
    view_field_coherence: float
    is_spatial_view_cell: bool

    def interpretation(self) -> str:
        """Human-readable interpretation of spatial view cell metrics.

        Returns
        -------
        str
            Multi-line interpretation.
        """
        lines = []

        if self.is_spatial_view_cell:
            lines.append("*** SPATIAL VIEW CELL ***")
            lines.append(
                f"View field info: {self.view_field_skaggs_info:.3f} bits/spike"
            )
            lines.append(
                f"Place field info: {self.place_field_skaggs_info:.3f} bits/spike"
            )
            info_ratio = (
                self.view_field_skaggs_info / self.place_field_skaggs_info
                if self.place_field_skaggs_info > 0
                else float("inf")
            )
            lines.append(f"View/Place info ratio: {info_ratio:.2f}")
            lines.append(f"View-place correlation: {self.view_place_correlation:.3f}")
            lines.append(f"View field sparsity: {self.view_field_sparsity:.3f}")
            lines.append(f"View field coherence: {self.view_field_coherence:.3f}")
        else:
            lines.append("Not classified as spatial view cell")
            lines.append(
                f"View field info: {self.view_field_skaggs_info:.3f} bits/spike"
            )
            lines.append(
                f"Place field info: {self.place_field_skaggs_info:.3f} bits/spike"
            )
            if self.place_field_skaggs_info > 0:
                info_ratio = self.view_field_skaggs_info / self.place_field_skaggs_info
                lines.append(f"View/Place info ratio: {info_ratio:.2f}")
            lines.append(f"View-place correlation: {self.view_place_correlation:.3f}")
            lines.append("")
            lines.append("Possible reasons:")
            if (
                self.place_field_skaggs_info > 0
                and self.view_field_skaggs_info <= 1.5 * self.place_field_skaggs_info
            ):
                lines.append(
                    "  - View info not sufficiently higher than place info "
                    "(ratio should be > 1.5)"
                )
            if self.view_place_correlation >= 0.7:
                lines.append(
                    "  - View and place fields too correlated "
                    "(should be < 0.7 for distinct representations)"
                )

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation with interpretation."""
        return self.interpretation()


def spatial_view_cell_metrics(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    view_distance: float = 10.0,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    smoothing_method: Literal[
        "diffusion_kde", "gaussian_kde", "binned"
    ] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy_seconds: float = 0.1,
    info_ratio: float = 1.5,
    max_correlation: float = 0.7,
) -> SpatialViewMetrics:
    """Compute spatial view cell metrics from spike data.

    Computes both place field and view field from the same spike data,
    then compares them to determine if the neuron is a spatial view cell.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Times of spikes in the same units as ``times``.
    times : NDArray[np.float64], shape (n_time,)
        Timestamps for each behavioral sample.
    positions : NDArray[np.float64], shape (n_time, 2)
        Animal positions in allocentric coordinates.
    headings : NDArray[np.float64], shape (n_time,)
        Animal heading at each time (radians, 0=East).
    view_distance : float, default=10.0
        Distance for fixed_distance gaze model (environment units).
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location.
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method for field computation.
    bandwidth : float, default=5.0
        Smoothing bandwidth for KDE methods (environment units).
    min_occupancy_seconds : float, default=0.1
        Minimum occupancy required in a bin (seconds).
    info_ratio : float, default=1.5
        Required ratio of view_info / place_info for classification.
    max_correlation : float, default=0.7
        Maximum view-place correlation for classification.

    Returns
    -------
    SpatialViewMetrics
        Dataclass with computed metrics and classification.

    Raises
    ------
    ValueError
        If spike_times is empty or arrays have mismatched lengths.

    See Also
    --------
    is_spatial_view_cell : Quick boolean classifier.
    neurospatial.spatial_view_field.compute_spatial_view_field : View field computation.
    neurospatial.spike_field.compute_place_field : Place field computation.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics.spatial_view_cells import spatial_view_cell_metrics
    >>> rng = np.random.default_rng(42)
    >>> positions = rng.uniform(0, 100, (500, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> n_time = 1000
    >>> times = np.linspace(0, 100, n_time)
    >>> trajectory = rng.uniform(20, 80, (n_time, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, n_time)
    >>> spike_times = rng.choice(times, size=100, replace=False)
    >>> metrics = spatial_view_cell_metrics(
    ...     env, spike_times, times, trajectory, headings
    ... )
    >>> isinstance(metrics.view_field_skaggs_info, float)
    True
    """
    # Convert inputs
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    times = np.asarray(times, dtype=np.float64).ravel()
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64).ravel()

    # Validate inputs
    if len(spike_times) == 0:
        raise ValueError(
            "spike_times cannot be empty. Need at least one spike "
            "to compute spatial view cell metrics."
        )

    if len(times) != len(positions):
        raise ValueError(
            f"times and positions must have the same length. "
            f"Got times: {len(times)}, positions: {len(positions)}."
        )

    if len(times) != len(headings):
        raise ValueError(
            f"times and headings must have the same length. "
            f"Got times: {len(times)}, headings: {len(headings)}."
        )

    # Compute place field (binned by animal position)
    place_field = compute_place_field(
        env,
        spike_times,
        times,
        positions,
        smoothing_method=smoothing_method,
        bandwidth=bandwidth,
        min_occupancy_seconds=min_occupancy_seconds,
    )

    # Compute view field (binned by viewed location)
    view_result = compute_spatial_view_field(
        env,
        spike_times,
        times,
        positions,
        headings,
        view_distance=view_distance,
        gaze_model=gaze_model,
        smoothing_method=smoothing_method,
        bandwidth=bandwidth,
        min_occupancy_seconds=min_occupancy_seconds,
    )
    view_field = view_result.field
    view_occupancy = view_result.view_occupancy

    # Compute occupancy for place field (standard position occupancy)
    dt = np.median(np.diff(times))
    position_bins = env.bin_at(positions)
    valid_bins = position_bins >= 0
    position_occupancy = np.zeros(env.n_bins, dtype=np.float64)
    np.add.at(position_occupancy, position_bins[valid_bins], dt)

    # Normalize occupancy to probability
    position_occupancy_prob = position_occupancy / np.sum(position_occupancy)
    view_occupancy_prob = (
        view_occupancy / np.sum(view_occupancy)
        if np.sum(view_occupancy) > 0
        else np.ones(env.n_bins) / env.n_bins
    )

    # Compute Skaggs information for both fields
    view_field_skaggs_info = skaggs_information(view_field, view_occupancy_prob)
    place_field_skaggs_info = skaggs_information(place_field, position_occupancy_prob)

    # Compute sparsity for view field
    view_field_sparsity = sparsity(view_field, view_occupancy_prob)

    # Compute coherence for view field
    try:
        view_field_coherence = rate_map_coherence(view_field, env)
    except Exception:
        view_field_coherence = np.nan

    # Compute correlation between view field and place field
    # Only use bins where both fields are valid
    valid_mask = np.isfinite(view_field) & np.isfinite(place_field)
    if np.sum(valid_mask) >= 2:
        view_vals = view_field[valid_mask]
        place_vals = place_field[valid_mask]
        # Check for zero variance
        if np.std(view_vals) > 0 and np.std(place_vals) > 0:
            view_place_correlation, _ = stats.pearsonr(view_vals, place_vals)
        else:
            view_place_correlation = np.nan
    else:
        view_place_correlation = np.nan

    # Classify as spatial view cell
    is_svc = _classify_spatial_view_cell(
        view_field_skaggs_info,
        place_field_skaggs_info,
        view_place_correlation,
        info_ratio=info_ratio,
        max_correlation=max_correlation,
    )

    return SpatialViewMetrics(
        view_field_skaggs_info=float(view_field_skaggs_info),
        place_field_skaggs_info=float(place_field_skaggs_info),
        view_place_correlation=float(view_place_correlation),
        view_field_sparsity=float(view_field_sparsity),
        view_field_coherence=float(view_field_coherence),
        is_spatial_view_cell=is_svc,
    )


def _classify_spatial_view_cell(
    view_info: float,
    place_info: float,
    view_place_corr: float,
    *,
    info_ratio: float = 1.5,
    max_correlation: float = 0.7,
) -> bool:
    """Classify neuron as spatial view cell.

    Parameters
    ----------
    view_info : float
        Skaggs info for view field.
    place_info : float
        Skaggs info for place field.
    view_place_corr : float
        Correlation between view and place fields.
    info_ratio : float, default=1.5
        Required ratio of view_info / place_info.
    max_correlation : float, default=0.7
        Maximum correlation for classification.

    Returns
    -------
    bool
        True if classified as spatial view cell.
    """
    # Handle NaN values
    if np.isnan(view_info) or np.isnan(place_info):
        return False

    # Check info ratio criterion
    # View info should be significantly higher than place info
    if place_info > 0:
        ratio_criterion = view_info >= info_ratio * place_info
    else:
        # If place info is 0, view info should be positive
        ratio_criterion = view_info > 0

    # Check correlation criterion
    # View and place fields should be dissimilar
    if np.isnan(view_place_corr):
        # If correlation is undefined, we can't classify
        correlation_criterion = False
    else:
        correlation_criterion = view_place_corr < max_correlation

    return bool(ratio_criterion and correlation_criterion)


def is_spatial_view_cell(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    view_distance: float = 10.0,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    info_ratio: float = 1.5,
    max_correlation: float = 0.7,
    **kwargs,
) -> bool:
    """Quick check: Is this a spatial view cell?

    Convenience function for fast screening.
    For detailed metrics, use ``spatial_view_cell_metrics()``.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Times of spikes.
    times : NDArray[np.float64], shape (n_time,)
        Timestamps for each behavioral sample.
    positions : NDArray[np.float64], shape (n_time, 2)
        Animal positions in allocentric coordinates.
    headings : NDArray[np.float64], shape (n_time,)
        Animal heading at each time (radians).
    view_distance : float, default=10.0
        Distance for fixed_distance gaze model.
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location.
    info_ratio : float, default=1.5
        Required ratio of view_info / place_info for classification.
    max_correlation : float, default=0.7
        Maximum view-place correlation for classification.
    **kwargs : dict
        Additional arguments passed to ``spatial_view_cell_metrics()``.

    Returns
    -------
    bool
        True if neuron passes spatial view cell criteria.

    Examples
    --------
    >>> from neurospatial.metrics.spatial_view_cells import is_spatial_view_cell
    >>> # Screen many neurons
    >>> for i, (spikes, times, pos, hd) in enumerate(all_neurons):  # doctest: +SKIP
    ...     if is_spatial_view_cell(env, spikes, times, pos, hd):
    ...         print(f"Neuron {i} is a spatial view cell")
    """
    try:
        metrics = spatial_view_cell_metrics(
            env,
            spike_times,
            times,
            positions,
            headings,
            view_distance=view_distance,
            gaze_model=gaze_model,
            info_ratio=info_ratio,
            max_correlation=max_correlation,
            **kwargs,
        )
        return bool(metrics.is_spatial_view_cell)
    except ValueError:
        return False
