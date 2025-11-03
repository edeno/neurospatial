"""regions/core.py
===============

Pure data layer for *continuous* regions of interest (ROIs).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Point, Polygon, mapping, shape

if TYPE_CHECKING:
    from pandas import DataFrame

# ---------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------
PointCoords: TypeAlias = NDArray[np.float64] | Iterable[float] | Point
Kind = Literal["point", "polygon"]

# ---------------------------------------------------------------------
# Region — immutable value object
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Region:
    """Immutable description of a spatial ROI.

    Parameters
    ----------
    name
        Unique region identifier.
    kind
        Either ``"point"`` or ``"polygon"``.
    data
        • point → ``np.ndarray`` with shape ``(n_dims,)``
        • polygon → :class:`shapely.geometry.Polygon` (always 2-D)
    metadata
        Optional, JSON-serialisable attributes (colour, label, …).

    """

    name: str
    kind: Kind
    data: NDArray[np.float64] | Polygon | Point
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    # filled in post-init
    n_dims: int = field(init=False, repr=False)

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    def __post_init__(self) -> None:
        # Freeze metadata to prevent accidental mutation through aliasing
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.kind == "point":
            if isinstance(self.data, Point):
                object.__setattr__(
                    self,
                    "data",
                    np.array(self.data.coords[0], dtype=float),
                )
            arr = np.asarray(self.data, dtype=float)
            if arr.ndim != 1:
                raise ValueError("Point data must be a 1-D array-like.")
            object.__setattr__(self, "data", arr)
            object.__setattr__(self, "n_dims", arr.shape[0])

        elif self.kind == "polygon":
            if not isinstance(self.data, Polygon):
                raise TypeError("data must be a Shapely Polygon for kind='polygon'.")
            object.__setattr__(self, "n_dims", 2)

        else:  # pragma: no cover
            raise ValueError(f"Unknown kind {self.kind!r}")

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------
    def __str__(self) -> str:
        return self.name

    # -----------------------------------------------------------------
    # Serialisation helpers (JSON-friendly)
    # -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Export Region to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the Region.

        """
        if self.kind == "point":
            geom = (
                self.data.tolist()
                if isinstance(self.data, np.ndarray)
                else list(self.data.coords[0])
            )
        else:
            geom = mapping(self.data)

        return {
            "name": self.name,
            "kind": self.kind,
            "geom": geom,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Region:
        """Create Region from dictionary representation.

        Parameters
        ----------
        payload : Mapping[str, Any]
            Dictionary containing region data with keys: 'name', 'kind', 'geom', 'metadata'.

        Returns
        -------
        Region
            Reconstructed Region instance.

        """
        kind_str = payload["kind"]
        if kind_str not in ("point", "polygon"):
            raise ValueError(f"Unknown kind {kind_str!r}")
        kind: Kind = kind_str

        if kind == "point":
            data = np.asarray(payload["geom"], dtype=float)
        else:  # kind == "polygon"
            data = shape(payload["geom"])
        return cls(
            name=payload["name"],
            kind=kind,
            data=data,
            metadata=payload.get("metadata", {}),
        )


# ---------------------------------------------------------------------
# Regions — mutable mapping
# ---------------------------------------------------------------------


class Regions(MutableMapping[str, Region]):
    """A small `dict`-like container mapping *name → Region*.

    Provides the usual mapping API plus a few helpers
    (`add`, `remove`, `list_names`, `buffer`, …).
    """

    __slots__ = ("_store",)

    # -------------- Mapping interface --------------------------------
    def __init__(self, items: Iterable[Region] | None = None) -> None:
        self._store: dict[str, Region] = {}
        if items is not None:
            for reg in items:
                self[reg.name] = reg

    def __getitem__(self, key: str) -> Region:
        return self._store[key]

    def __setitem__(self, key: str, value: Region) -> None:
        if key in self._store:
            raise KeyError(
                f"Region {key!r} already exists — use update_region() instead."
            )
        if key != value.name:
            raise ValueError("Key must match Region.name")
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        self._store.pop(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        inside = ", ".join(f"{n}({r.kind})" for n, r in self._store.items())
        return f"{self.__class__.__name__}({inside})"

    # -------------- Convenience helpers ------------------------------
    def add(
        self,
        name: str,
        *,
        point: PointCoords | None = None,
        polygon: Polygon | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Region:
        """Create and insert a new Region.

        Parameters
        ----------
        name : str
            Unique name for the region.
        point : PointCoords or None, optional
            Point coordinates or Shapely Point object. Mutually exclusive with polygon.
        polygon : Polygon or None, optional
            Shapely Polygon object. Mutually exclusive with point.
        metadata : Mapping[str, Any] or None, optional
            Optional metadata dictionary to attach to the region.

        Returns
        -------
        Region
            The newly created Region instance.

        Raises
        ------
        ValueError
            If both or neither of point/polygon are specified.
        KeyError
            If name already exists in the collection.

        """
        if (point is None) == (polygon is None):
            raise ValueError("Specify **one** of 'point' or 'polygon'.")
        if name in self:
            raise KeyError(f"Duplicate region name {name!r}.")

        if point is not None:
            # Accept either a coordinate array or a Shapely Point
            coords = np.asarray(
                point.coords[0] if isinstance(point, Point) else point, dtype=float
            )
            region = Region(name, "point", coords, metadata or {})
        else:
            region = Region(name, "polygon", polygon, metadata or {})

        self[name] = region
        return region

    def update_region(
        self,
        name: str,
        *,
        point: PointCoords | None = None,
        polygon: Polygon | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Region:
        """Update an existing Region.

        This method replaces an existing region with a new one. The region can
        change its type (point vs polygon) and/or data and/or metadata.
        Metadata is preserved from the existing region if not explicitly provided.

        Parameters
        ----------
        name : str
            Name of the region to update.
        point : PointCoords or None, optional
            Point coordinates or Shapely Point object. Mutually exclusive with polygon.
        polygon : Polygon or None, optional
            Shapely Polygon object. Mutually exclusive with point.
        metadata : Mapping[str, Any] or None, optional
            Optional metadata dictionary to attach to the region. If None, preserves
            the existing region's metadata.

        Returns
        -------
        Region
            The newly created Region instance that replaced the old one.

        Raises
        ------
        ValueError
            If both or neither of point/polygon are specified.
        KeyError
            If name does not exist in the collection.

        Examples
        --------
        >>> from neurospatial.regions import Regions
        >>> regs = Regions()
        >>> _ = regs.add("center", point=[0.0, 0.0], metadata={"color": "red"})
        >>> # Update coordinates while preserving metadata
        >>> _ = regs.update_region("center", point=[1.0, 1.0])
        >>> regs["center"].data
        array([1., 1.])
        >>> regs["center"].metadata["color"]
        'red'

        """
        if (point is None) == (polygon is None):
            raise ValueError("Specify **one** of 'point' or 'polygon'.")
        if name not in self:
            raise KeyError(
                f"Region {name!r} does not exist. Use add() to create new regions."
            )

        # Preserve existing metadata if not explicitly provided
        old_region = self._store[name]
        effective_metadata = metadata if metadata is not None else old_region.metadata

        # Remove the old region and add the new one
        del self._store[name]

        if point is not None:
            # Accept either a coordinate array or a Shapely Point
            coords = np.asarray(
                point.coords[0] if isinstance(point, Point) else point, dtype=float
            )
            region = Region(name, "point", coords, effective_metadata)
        else:
            region = Region(name, "polygon", polygon, effective_metadata)

        # Use direct store access to bypass __setitem__ duplicate check
        self._store[name] = region
        return region

    def remove(self, name: str) -> None:
        """Delete a region by name.

        Parameters
        ----------
        name : str
            Name of region to remove. No error if absent.

        """
        self._store.pop(name, None)

    def list_names(self) -> list[str]:
        """Get list of region names in insertion order.

        Returns
        -------
        list[str]
            Region names in the order they were added.

        """
        return list(self._store)

    # ----------- lightweight geometry helper -------------------------
    def area(self, name: str) -> float:
        """Compute area of a region.

        Parameters
        ----------
        name : str
            Name of region to query.

        Returns
        -------
        float
            Area of the polygon region, or 0.0 for point regions.

        """
        region = self[name]
        if region.kind == "polygon":
            assert isinstance(region.data, Polygon)
            return float(region.data.area)
        return 0.0

    def region_center(self, region_name: str) -> NDArray[np.float64] | None:
        """Calculate the center of a specified named region.

        Parameters
        ----------
        region_name : str
            Name of region to query.

        Returns
        -------
        Optional[NDArray[np.float64]]
            N-D coordinates of the region's center, or None if the region
            is empty or center cannot be determined.

        Raises
        ------
        KeyError
            If `region_name` is not present in this collection.

        """
        if region_name not in self._store:
            raise KeyError(f"Region '{region_name}' not found in this collection.")

        region = self._store[region_name]

        if region.kind == "point":
            return np.asarray(region.data, dtype=float)
        else:  # region.kind == "polygon"
            assert isinstance(region.data, Polygon)
            return np.array(region.data.centroid.coords[0], dtype=float)

    def buffer(
        self,
        source: str | NDArray[np.float64],
        distance: float,
        new_name: str,
        **meta: Any,
    ) -> Region:
        """Create a buffered region around a point or existing region.

        Parameters
        ----------
        source : str or NDArray[np.float64]
            Region name or point coordinates to buffer around.
        distance : float
            Buffer distance in spatial units.
        new_name : str
            Name for the new buffered region.
        **meta : Any
            Additional metadata for the new region.

        Returns
        -------
        Region
            The newly created buffered region.

        """
        # derive geometry in cm space
        if isinstance(source, str):
            src = self[source]
            if src.kind == "polygon":
                assert isinstance(src.data, Polygon)
                geom = src.data
            elif src.kind == "point" and src.n_dims == 2:
                assert isinstance(src.data, np.ndarray)
                geom = Point(src.data)
            else:
                raise ValueError("Can only buffer 2-D point or polygon regions.")
        else:  # raw coords
            arr = np.asarray(source, dtype=float)
            if arr.shape != (2,):
                raise ValueError("Raw source must be shape (2,) for buffering.")
            geom = Point(arr)

        poly = geom.buffer(distance)
        if not isinstance(poly, Polygon):
            raise ValueError("Buffer produced non-polygon geometry.")

        return self.add(new_name, polygon=poly, metadata=meta)

    def to_dataframe(self) -> DataFrame:
        """Convert this collection to a Pandas DataFrame.
        Requires Pandas to be installed.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['name', 'kind', 'data', 'metadata'].
            The 'data' column contains the coordinates for points or polygons.

        """
        from .io import regions_to_dataframe

        return regions_to_dataframe(self)

    # -------------- Serialization helpers ---------------------------
    _FMT = "Regions-v1"

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Write collection to disk in a simple, version-tagged schema.

        Parameters
        ----------
        path : str or Path
            Output file path for JSON data.
        indent : int, default=2
            Indentation level for pretty-printed JSON.

        """
        payload = {
            "format": self._FMT,
            "regions": [r.to_dict() for r in self._store.values()],
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=indent))

    @classmethod
    def from_json(cls, path: str | Path) -> Regions:
        """Load Regions from JSON file.

        Parameters
        ----------
        path : str or Path
            Path to JSON file containing regions data.

        Returns
        -------
        Regions
            Loaded Regions collection.

        """
        blob = json.loads(Path(path).read_text())
        if blob.get("format") != cls._FMT:
            warnings.warn(f"Unexpected format tag {blob.get('format')!r}")
        return cls(Region.from_dict(d) for d in blob["regions"])
