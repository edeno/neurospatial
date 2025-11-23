"""Type definitions for annotation module."""

from __future__ import annotations

from typing import Literal

# Role type alias for annotation shapes
# - "environment": Primary boundary polygon defining the spatial environment
# - "hole": Excluded areas within environment boundary (subtracted)
# - "region": Named regions of interest (ROIs)
Role = Literal["environment", "hole", "region"]

# Strategy for handling multiple environment boundaries
# - "last": Use the last drawn boundary (default, current behavior)
# - "first": Use the first drawn boundary
# - "error": Raise an error if multiple boundaries are drawn
MultipleBoundaryStrategy = Literal["last", "first", "error"]
