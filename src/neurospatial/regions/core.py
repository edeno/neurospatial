"""regions/core.py
===============

Pure data layer for *continuous* regions of interest (ROIs).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
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


def _freeze_metadata(value: Any) -> Any:
    """Recursively wrap mappings in read-only proxies to prevent mutation.

    Mappings (at any nesting depth) become :class:`types.MappingProxyType`
    instances; lists become tuples; other values are passed through.
    The input should already be a deep copy so the proxies wrap data the
    Region exclusively owns.

    Parameters
    ----------
    value : Any
        A (deep-copied) value to freeze in place.

    Returns
    -------
    Any
        The frozen, read-only equivalent of ``value``.

    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_metadata(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _thaw_metadata(value: Any) -> Any:
    """Recursively convert frozen metadata back into mutable, JSON-ready data.

    Inverse of :func:`_freeze_metadata`: :class:`~collections.abc.Mapping`
    instances become plain ``dict`` objects and tuples become lists, yielding
    a structure safe to hand to :func:`json.dumps`.

    Parameters
    ----------
    value : Any
        A frozen metadata value (possibly nested).

    Returns
    -------
    Any
        A mutable, JSON-serializable copy of ``value``.

    """
    if isinstance(value, Mapping):
        return {k: _thaw_metadata(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return value


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
        Optional, JSON-serialisable attributes (colour, label, …). Stored as a
        recursively read-only mapping (:class:`types.MappingProxyType` at every
        level, with lists frozen to tuples) so neither the top level nor any
        nested container can be mutated after construction. Serialization via
        :meth:`to_dict` thaws this back into plain ``dict``/``list`` data.

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
        # Thaw first (the input may already be a read-only proxy from another
        # Region, and mappingproxy cannot be deep-copied), then deep copy so the
        # Region owns the data, then recursively wrap it in read-only proxies.
        # This isolates the Region from external mutation of the source dict
        # *and* forbids mutating the stored metadata (at any nesting depth)
        # after construction.
        thawed = _thaw_metadata(self.metadata)
        object.__setattr__(self, "metadata", _freeze_metadata(copy.deepcopy(thawed)))

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
    # Copy / pickle support
    # -----------------------------------------------------------------
    # The stored ``metadata`` is a (recursively) read-only ``MappingProxyType``,
    # which the stdlib copy/pickle machinery cannot handle directly. These hooks
    # round-trip through the thawed, mutable form; the constructor re-freezes it,
    # so the copy is an independent Region with the same immutability guarantees.
    def __copy__(self) -> Region:
        return Region(
            name=self.name,
            kind=self.kind,
            data=self.data,
            metadata=_thaw_metadata(self.metadata),
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> Region:
        new = Region(
            name=copy.deepcopy(self.name, memo),
            kind=self.kind,
            data=copy.deepcopy(self.data, memo),
            metadata=_thaw_metadata(self.metadata),
        )
        memo[id(self)] = new
        return new

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        # Reconstruct via the public constructor with thawed metadata.
        return (
            self.__class__,
            (self.name, self.kind, self.data, _thaw_metadata(self.metadata)),
        )

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
            # Thaw the read-only proxy back into plain dict/list data so the
            # result is JSON-serialisable and nested values survive the trip.
            "metadata": _thaw_metadata(self.metadata),
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
        """Insert a region under ``key``; refuse to silently overwrite.

        ``regions[key] = region`` is the *strict* insertion path: it
        accepts new keys but raises :class:`KeyError` if ``key`` already
        exists. Callers that want to replace an existing region must opt
        in explicitly via :meth:`set` (idempotent replace) or
        :meth:`update_region` (replace geometry/metadata in place).

        Parameters
        ----------
        key : str
            Key to assign to. Must match the Region's name.
        value : Region
            Region object to store.

        Raises
        ------
        ValueError
            If ``key`` does not match ``value.name``.
        KeyError
            If ``key`` is already present in the collection.

        Examples
        --------
        >>> regions = Regions()
        >>> r1 = Region(name="goal", kind="point", data=[10.0, 20.0])
        >>> regions["goal"] = r1  # OK — first assignment
        >>> r2 = Region(name="goal", kind="point", data=[15.0, 25.0])
        >>> regions["goal"] = r2  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        KeyError: "Region 'goal' already exists. ..."

        To replace an existing region, pick the explicit path:

        >>> _ = regions.set("goal", r2)  # idempotent replace
        >>> _ = regions.update_region("goal", point=[15.0, 25.0])  # in-place edit

        See Also
        --------
        set : Idempotent replace (accepts both new and existing names).
        update_region : In-place geometry/metadata edit; raises if absent.
        add : Build a new Region and insert it; raises if already exists.

        """
        if key != value.name:
            raise ValueError(
                f"Key {key!r} must match Region.name {value.name!r}. "
                f"Cannot assign region with name {value.name!r} to key {key!r}."
            )
        if key in self._store:
            raise KeyError(
                f"Region {key!r} already exists. "
                "Use Regions.update_region(...) to modify it in place, or "
                "Regions.set(name, region) for the idempotent replace path."
            )
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

    def set(self, name: str, region: Region) -> Region:
        """Insert ``region`` under ``name`` (idempotent replace).

        Unlike ``regions[name] = region`` and :meth:`add`, this method
        accepts both new and existing names and is the documented path
        for callers that explicitly want to replace whatever is already
        stored under ``name``.

        Parameters
        ----------
        name : str
            Name of the region. Must match ``region.name``.
        region : Region
            Region to insert (or to use as the replacement if ``name``
            already exists).

        Returns
        -------
        Region
            The stored region (the same object passed in).

        Raises
        ------
        ValueError
            If ``name`` does not match ``region.name``.

        See Also
        --------
        add : Insert a new region; raises if ``name`` already exists.
        update_region : Replace an existing region's geometry/metadata
            in place; raises if ``name`` does not exist.
        """
        if name != region.name:
            raise ValueError(f"Key {name!r} must match Region.name {region.name!r}.")
        # Bypass __setitem__'s duplicate check; this is the idempotent path.
        self._store[name] = region
        return region

    def remove(self, name: str) -> None:
        """Delete a region by name.

        Parameters
        ----------
        name : str
            Name of region to remove.

        Raises
        ------
        KeyError
            If ``name`` is not in the collection. Mirrors ``del regions[name]``
            and the rest of the contract: every Region API
            (``add``, ``update_region``, ``__setitem__``, ``__delitem__``,
            ``remove``) raises rather than silently absorbing the case where
            the caller's mental model of the collection disagrees with reality.
        """
        del self._store[name]

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

    def region_center(self, name: str) -> NDArray[np.float64] | None:
        """Calculate the center of a specified named region.

        Parameters
        ----------
        name : str
            Name of region to query.

        Returns
        -------
        Optional[NDArray[np.float64]]
            N-D coordinates of the region's center, or None if the region
            is empty or center cannot be determined.

        Raises
        ------
        KeyError
            If `name` is not present in this collection.

        """
        if name not in self._store:
            raise KeyError(f"Region '{name}' not found in this collection.")

        region = self._store[name]

        if region.kind == "point":
            return np.asarray(region.data, dtype=float)
        # region.kind == "polygon"
        assert isinstance(region.data, Polygon)
        if region.data.is_empty:
            # Empty polygon has no centroid coordinates; the documented
            # contract is to return None rather than raise.
            return None
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

        Thin wrapper around :func:`neurospatial.regions.io.regions_to_json`;
        both entry points share one implementation and produce byte-identical
        output.

        Parameters
        ----------
        path : str or Path
            Output file path for JSON data.
        indent : int, default=2
            Indentation level for pretty-printed JSON.

        """
        from .io import regions_to_json

        regions_to_json(self, path, indent=indent)

    @classmethod
    def from_json(cls, path: str | Path) -> Regions:
        """Load Regions from JSON file.

        Thin wrapper around :func:`neurospatial.regions.io.regions_from_json`;
        both entry points share one implementation.

        Parameters
        ----------
        path : str or Path
            Path to JSON file containing regions data.

        Returns
        -------
        Regions
            Loaded Regions collection.

        """
        from .io import regions_from_json

        return regions_from_json(path)
