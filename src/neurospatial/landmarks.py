"""Named things in an environment that an animal sees or approaches.

A landmark is a cue card, a monitor, an object or a reward port: geometry in
the environment's coordinates, with a name that identifies it across sessions.
Landmarks are not bins. They may lie outside the arena, and they neither block
movement nor block the view; declare a wall separately for that.

Encoding functions that take ``object_positions`` also accept landmark names,
resolved through ``env.landmarks``, and record the names in their results.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import shapely
from numpy.typing import ArrayLike, NDArray
from shapely.geometry import LineString, Polygon, mapping, shape

from neurospatial._exceptions import _format_error
from neurospatial.regions import _thaw_metadata
from neurospatial.regions.core import (
    _check_planar_geometry,
    _owned_metadata,
    _owned_point,
    _require_name,
)

__all__ = ["Landmark", "Landmarks"]

Kind = Literal["point", "line", "polygon"]


def _error(what: str, fix: str) -> ValueError:
    return ValueError(_format_error(what, fix=fix))


@dataclass(frozen=True, eq=False)
class Landmark:
    """One named landmark and its geometry.

    Parameters
    ----------
    name : str
        Identifier, the same in every session that has this landmark.
    kind : {"point", "line", "polygon"}
        ``"point"`` for a small cue or a reward port, ``"line"`` for a monitor
        or a card along a wall, ``"polygon"`` for an object's footprint.
    data : array-like or shapely geometry
        A point's coordinates, shape ``(n_dims,)``; a line's vertices, shape
        ``(n_vertices, 2)``, or a ``LineString``; a polygon's boundary, shape
        ``(n_vertices, 2)``, or a ``Polygon``. In the environment's
        coordinates and units. Lines and polygons are two-dimensional.
    metadata : mapping, optional
        JSON-serializable attributes, such as ``{"role": "cue"}``.

    Raises
    ------
    ValueError
        If the name is empty, the kind is unknown or the geometry is not
        finite and valid.
    TypeError
        If a point's data is another shapely shape or not numbers.

    Examples
    --------
    >>> from neurospatial.landmarks import Landmark
    >>> monitor = Landmark("monitor", "line", [(0.0, 100.0), (40.0, 100.0)])
    >>> monitor.n_dims
    2
    """

    name: str
    kind: Kind
    data: NDArray[np.float64] | LineString | Polygon
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _require_name(self.name, record="Landmark", example="cue_card")
        owner = f"Landmark {self.name!r}"
        object.__setattr__(
            self, "metadata", _owned_metadata(self.metadata, owner=owner)
        )
        data: NDArray[np.float64] | LineString | Polygon
        if self.kind == "point":
            data = _owned_point(
                self.data,
                owner=owner,
                shape_kinds={"Polygon": "polygon", "LineString": "line"},
            )
        elif self.kind in ("line", "polygon"):
            kind = LineString if self.kind == "line" else Polygon
            data = self.data if isinstance(self.data, kind) else kind(self.data)
            _check_planar_geometry(data, owner=owner)
        else:
            raise _error(
                f"Landmark {self.name!r} has unknown kind {self.kind!r}.",
                'use "point", "line" or "polygon"',
            )
        object.__setattr__(self, "data", data)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        # The read-only metadata proxy cannot be pickled; rebuild from plain data.
        return type(self), (
            self.name,
            self.kind,
            self.data,
            _thaw_metadata(self.metadata),
        )

    @property
    def n_dims(self) -> int:
        """Number of coordinate columns."""
        if isinstance(self.data, np.ndarray):
            return int(self.data.size)
        return 2

    @property
    def position(self) -> NDArray[np.float64]:
        """One representative point, shape ``(n_dims,)``.

        The point itself, the middle of a line along its length, or a
        polygon's centroid.
        """
        if isinstance(self.data, np.ndarray):
            return np.array(self.data, dtype=np.float64)
        if isinstance(self.data, LineString):
            return np.array(self.data.interpolate(0.5, normalized=True).coords[0])
        return np.array(self.data.centroid.coords[0])

    def nearest_points(self, positions: ArrayLike) -> NDArray[np.float64]:
        """The landmark's nearest point to each position.

        Parameters
        ----------
        positions : array-like, shape (n_time, n_dims)
            Positions in the environment's coordinates. Rows with a NaN
            coordinate are missing.

        Returns
        -------
        ndarray of float64, shape (n_time, n_dims)
            For a point, the point itself; for a line, the closest point on
            it; for a polygon, the closest point of its area (the position
            itself when inside). NaN rows for missing positions.
        """
        query = np.asarray(positions, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != self.n_dims:
            raise _error(
                f"positions for landmark {self.name!r} must have shape "
                f"(n_time, {self.n_dims}), got {query.shape}.",
                "pass one coordinate column per environment dimension",
            )
        result = np.full(query.shape, np.nan)
        finite = np.all(np.isfinite(query), axis=1)
        if isinstance(self.data, np.ndarray):
            result[finite] = self.data
            return result
        lines = shapely.shortest_line(self.data, shapely.points(query[finite]))
        result[finite] = shapely.get_coordinates(lines)[::2]
        return result

    def to_dict(self) -> dict[str, Any]:
        """Export the landmark as JSON-serializable data."""
        geometry = (
            self.data.tolist()
            if isinstance(self.data, np.ndarray)
            else mapping(self.data)
        )
        return {
            "name": self.name,
            "kind": self.kind,
            "geom": _thaw_metadata(geometry),
            "metadata": _thaw_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Landmark:
        """Rebuild a landmark from :meth:`to_dict` data."""
        kind = payload["kind"]
        geometry = payload["geom"]
        data = (
            np.asarray(geometry, dtype=np.float64)
            if kind == "point"
            else shape(geometry)
        )
        return cls(payload["name"], kind, data, payload.get("metadata", {}))


class Landmarks(Mapping[str, Landmark]):
    """An immutable, ordered collection of landmarks with unique names.

    Parameters
    ----------
    landmarks : iterable of Landmark, optional
        The landmarks, in order.

    Raises
    ------
    ValueError
        If two landmarks share a name or their dimensions differ.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.landmarks import Landmark, Landmarks
    >>> landmarks = Landmarks(
    ...     [
    ...         Landmark("cue_card", "point", [50.0, 100.0]),
    ...         Landmark("monitor", "line", [(0.0, 0.0), (40.0, 0.0)]),
    ...     ]
    ... )
    >>> landmarks.nearest_points(np.array([[10.0, 5.0]]), ["monitor"]).tolist()
    [[[10.0, 0.0]]]
    """

    def __init__(self, landmarks: Iterable[Landmark] = ()) -> None:
        items: dict[str, Landmark] = {}
        for landmark in landmarks:
            if not isinstance(landmark, Landmark):
                raise TypeError(
                    f"Landmarks holds Landmark records, got {type(landmark).__name__}."
                )
            if landmark.name in items:
                raise _error(
                    f"Two landmarks are named {landmark.name!r}.",
                    "give every landmark its own name",
                )
            items[landmark.name] = landmark
        if len({landmark.n_dims for landmark in items.values()}) > 1:
            raise _error(
                "Landmarks have different coordinate dimensions.",
                "give every landmark coordinates in the same environment",
            )
        self._items = items

    def __getitem__(self, name: str) -> Landmark:
        if name not in self._items:
            raise KeyError(
                _format_error(
                    f"No landmark is named {name!r}.",
                    fix=f"use one of {list(self._items)}, or add it with "
                    "env.with_landmarks(...)",
                )
            )
        return self._items[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Landmarks({list(self._items)})"

    def _selected(self, names: str | Sequence[str] | None) -> list[Landmark]:
        if names is None:
            return list(self._items.values())
        if isinstance(names, str):
            names = [names]
        return [self[name] for name in names]

    def positions(
        self, names: str | Sequence[str] | None = None
    ) -> NDArray[np.float64]:
        """Representative points, shape ``(n_landmarks, n_dims)``.

        See :attr:`Landmark.position`. ``names`` selects and orders them;
        every landmark by default.
        """
        selected = self._selected(names)
        if not selected:
            return np.empty((0, 0))
        return np.array([landmark.position for landmark in selected])

    def nearest_points(
        self, positions: ArrayLike, names: str | Sequence[str] | None = None
    ) -> NDArray[np.float64]:
        """Each landmark's nearest point to each position.

        Parameters
        ----------
        positions : array-like, shape (n_time, n_dims)
            Positions in the environment's coordinates.
        names : str or sequence of str, optional
            Landmarks to use, in this order; every landmark by default.

        Returns
        -------
        ndarray of float64, shape (n_time, n_landmarks, n_dims)
            See :meth:`Landmark.nearest_points`. This is the per-sample form
            the egocentric operations and ``object_positions`` accept.
        """
        selected = self._selected(names)
        query = np.asarray(positions, dtype=np.float64)
        if not selected:
            return np.empty((len(query), 0, query.shape[-1] if query.ndim else 0))
        return np.stack(
            [landmark.nearest_points(query) for landmark in selected], axis=1
        )

    def near(
        self, points: ArrayLike, name: str, *, tolerance: float
    ) -> NDArray[np.bool_]:
        """Whether each point lies within ``tolerance`` of a landmark.

        Parameters
        ----------
        points : array-like, shape (n_points, n_dims)
            Points such as gaze locations from ``compute_viewed_location``.
        name : str
            The landmark.
        tolerance : float
            Largest distance, in the environment's units, that counts as on
            the landmark.

        Returns
        -------
        ndarray of bool, shape (n_points,)
            False for points with a NaN coordinate.
        """
        if not np.isfinite(tolerance) or tolerance < 0:
            raise _error(
                f"tolerance must be finite and nonnegative, got {tolerance!r}.",
                "pass a distance in the environment's units, such as 2.0",
            )
        query = np.asarray(points, dtype=np.float64)
        nearest = self[name].nearest_points(query)
        with np.errstate(invalid="ignore"):
            return cast(
                "NDArray[np.bool_]",
                np.linalg.norm(query - nearest, axis=1) <= tolerance,
            )

    def to_list(self) -> list[dict[str, Any]]:
        """Export every landmark with :meth:`Landmark.to_dict`."""
        return [landmark.to_dict() for landmark in self._items.values()]

    @classmethod
    def from_list(cls, payload: Iterable[Mapping[str, Any]]) -> Landmarks:
        """Rebuild landmarks from :meth:`to_list` data."""
        return cls(Landmark.from_dict(item) for item in payload)
