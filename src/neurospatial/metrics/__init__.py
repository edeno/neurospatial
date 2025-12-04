"""
Neuroscience metrics for spatial analysis.

This module provides standard neuroscience metrics for place cells, boundary
cells, population-level analyses, and trajectory characterization validated
against field-standard packages (opexebo, neurocode, buzcode) and ecology
literature (Traja, adehabitatHR).

Modules
-------
place_fields
    Place field detection and single-cell spatial metrics.
population
    Population-level metrics for analyzing spatial representations.
boundary_cells
    Boundary cell metrics including border score.
trajectory
    Trajectory characterization metrics (turn angles, step lengths, home range, MSD).

"""

from __future__ import annotations

from neurospatial.metrics.boundary_cells import border_score, compute_region_coverage
from neurospatial.metrics.circular import rayleigh_test
from neurospatial.metrics.grid_cells import (
    GridProperties,
    grid_orientation,
    grid_properties,
    grid_scale,
    grid_score,
    periodicity_score,
    spatial_autocorrelation,
)
from neurospatial.metrics.place_fields import (
    compute_field_emd,
    detect_place_fields,
    field_centroid,
    field_shape_metrics,
    field_shift_distance,
    field_size,
    field_stability,
    in_out_field_ratio,
    information_per_second,
    mutual_information,
    rate_map_coherence,
    selectivity,
    skaggs_information,
    sparsity,
    spatial_coverage_single_cell,
)
from neurospatial.metrics.population import (
    PopulationCoverageResult,
    count_place_cells,
    field_density_map,
    field_overlap,
    plot_population_coverage,
    population_coverage,
    population_vector_correlation,
)
from neurospatial.metrics.trajectory import (
    compute_home_range,
    compute_step_lengths,
    compute_turn_angles,
    mean_square_displacement,
)

__all__ = [
    "GridProperties",
    "PopulationCoverageResult",
    "border_score",
    "compute_field_emd",
    "compute_home_range",
    "compute_region_coverage",
    "compute_step_lengths",
    "compute_turn_angles",
    "count_place_cells",
    "detect_place_fields",
    "field_centroid",
    "field_density_map",
    "field_overlap",
    "field_shape_metrics",
    "field_shift_distance",
    "field_size",
    "field_stability",
    "grid_orientation",
    "grid_properties",
    "grid_scale",
    "grid_score",
    "in_out_field_ratio",
    "information_per_second",
    "mean_square_displacement",
    "mutual_information",
    "periodicity_score",
    "plot_population_coverage",
    "population_coverage",
    "population_vector_correlation",
    "rate_map_coherence",
    "rayleigh_test",
    "selectivity",
    "skaggs_information",
    "sparsity",
    "spatial_autocorrelation",
    "spatial_coverage_single_cell",
]
