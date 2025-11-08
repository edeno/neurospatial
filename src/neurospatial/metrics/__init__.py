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

from neurospatial.metrics.boundary_cells import border_score
from neurospatial.metrics.place_fields import (
    detect_place_fields,
    field_centroid,
    field_size,
    field_stability,
    skaggs_information,
    sparsity,
)
from neurospatial.metrics.population import (
    count_place_cells,
    field_density_map,
    field_overlap,
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
    "border_score",
    "compute_home_range",
    "compute_step_lengths",
    "compute_turn_angles",
    "count_place_cells",
    "detect_place_fields",
    "field_centroid",
    "field_density_map",
    "field_overlap",
    "field_size",
    "field_stability",
    "mean_square_displacement",
    "population_coverage",
    "population_vector_correlation",
    "skaggs_information",
    "sparsity",
]
