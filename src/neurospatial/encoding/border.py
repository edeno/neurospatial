"""
Border/boundary cell analysis.

This module provides tools for analyzing border cells - neurons that fire
along environmental boundaries. Implements border score (Solstad et al., 2008)
and related boundary cell analyses.

Imports
-------
>>> from neurospatial.encoding.border import border_score
>>> from neurospatial.encoding.border import compute_region_coverage

Or from the encoding package:

>>> from neurospatial.encoding import border_score, compute_region_coverage

Functions
---------
border_score
    Compute border score for a spatial firing rate map.
compute_region_coverage
    Compute field coverage for each spatial region.

References
----------
Solstad, T., Boccara, C. N., Kropff, E., Moser, M. B., & Moser, E. I. (2008).
    Representation of geometric borders in the entorhinal cortex. Science,
    322(5909), 1865-1868. https://doi.org/10.1126/science.1166466
"""

from neurospatial.metrics.boundary_cells import (
    border_score,
    compute_region_coverage,
)

__all__ = [
    "border_score",
    "compute_region_coverage",
]
