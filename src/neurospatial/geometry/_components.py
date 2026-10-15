"""Owned bin shapes and immutable outlines for spatial drawing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon


class _ComponentState:
    """Restore array write protection after pickle and deepcopy."""

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, eq=False)
class HexagonShapes(_ComponentState):
    """Regular hexagons with centers ``(n_bins, 2)``, radius and orientation."""

    centers: NDArray[np.float64]
    radius: float
    orientation: float

    def __post_init__(self) -> None:
        centers = np.array(self.centers, dtype=np.float64, copy=True)
        centers.setflags(write=False)
        object.__setattr__(self, "centers", centers)


@dataclass(frozen=True, eq=False)
class TriangleShapes(_ComponentState):
    """Mesh points ``(n_points, 2)`` and active triangles ``(n_bins, 3)``."""

    points: NDArray[np.float64]
    triangles: NDArray[np.intp]

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=np.float64, copy=True)
        triangles = np.array(self.triangles, dtype=np.intp, copy=True)
        points.setflags(write=False)
        triangles.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "triangles", triangles)


@dataclass(frozen=True, eq=False)
class Outline:
    """An immutable polygon or multipolygon outline."""

    geometry: Polygon | MultiPolygon
