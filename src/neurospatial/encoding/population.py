"""Population-level place field metrics.

This module re-exports population metrics from neurospatial.metrics.population.
It provides analysis tools for spatial representations across populations of
neurons, including coverage, overlap, and correlation measures.

Imports
-------
All symbols can be imported directly from this module:

>>> from neurospatial.encoding.population import (
...     PopulationCoverageResult,
...     population_coverage,
...     plot_population_coverage,
...     field_density_map,
...     count_place_cells,
...     field_overlap,
...     population_vector_correlation,
... )

Or from the encoding package:

>>> from neurospatial.encoding import (
...     PopulationCoverageResult,
...     population_coverage,
...     plot_population_coverage,
... )

References
----------
.. [1] Wilson, M. A., & McNaughton, B. L. (1993). Dynamics of the hippocampal
       ensemble code for space. Science, 261(5124), 1055-1058.
.. [2] O'Keefe, J., & Nadel, L. (1978). The Hippocampus as a Cognitive Map.
       Oxford: Clarendon Press.
"""

from neurospatial.metrics.population import (
    PopulationCoverageResult,
    count_place_cells,
    field_density_map,
    field_overlap,
    plot_population_coverage,
    population_coverage,
    population_vector_correlation,
)

__all__ = [
    "PopulationCoverageResult",
    "count_place_cells",
    "field_density_map",
    "field_overlap",
    "plot_population_coverage",
    "population_coverage",
    "population_vector_correlation",
]
