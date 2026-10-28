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
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString, Point, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

from neurospatial._exceptions import RegionNotFoundError, _format_error

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
    instances; lists and tuples become tuples; other values are passed through.
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
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _thaw_metadata(value: Any, *, _active: set[int] | None = None) -> Any:
    """Recursively convert frozen metadata back into mutable, JSON-ready data.

    Inverse of :func:`_freeze_metadata`: :class:`~collections.abc.Mapping`
    instances become plain ``dict`` objects and tuples become lists, yielding
    a structure safe to hand to :func:`json.dumps`.

    Parameters
    ----------
    value : Any
        A frozen metadata value (possibly nested).
    _active : set of int, optional
        Container identities on the current traversal path; detects cycles
        without rejecting shared containers in an acyclic value.

    Returns
    -------
    Any
        A mutable, JSON-serializable copy of ``value``.

    """
    if not isinstance(value, (Mapping, list, tuple)):
        return value
    active = set() if _active is None else _active
    identity = id(value)
    if identity in active:
        raise ValueError("Circular reference detected in Region metadata.")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            return {k: _thaw_metadata(v, _active=active) for k, v in value.items()}
        return [_thaw_metadata(item, _active=active) for item in value]
    finally:
        active.remove(identity)


def _require_json_metadata(
    value: Any, *, allow_nan: bool, owner: str | None = None, _path: str = "metadata"
) -> None:
    """Reject metadata that JSON would coerce or cannot store.

    Mapping keys must be strings at every level and leaves must be JSON
    scalars. Non-finite floats are rejected unless ``allow_nan``; Python's JSON
    reader restores ``NaN`` and ``Infinity`` exactly. ``owner``, such as
    ``"Region 'goal'"``, names the record in the error.
    """
    subject = "Metadata" if owner is None else f"{owner} metadata"
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    _format_error(
                        f"{subject} key {key!r} in {_path} is a "
                        f"{type(key).__name__}, not a str.",
                        why=f"Why: JSON keys are strings, so saving would turn "
                        f"it into {str(key)!r}, which can collide with an "
                        "existing key.",
                        fix=f"use the string key str({key!r})",
                    )
                )
            _require_json_metadata(
                item, allow_nan=allow_nan, owner=owner, _path=f"{_path}[{key!r}]"
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_json_metadata(
                item, allow_nan=allow_nan, owner=owner, _path=f"{_path}[{index}]"
            )
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError(
            _format_error(
                f"{subject} value {_path} is a {type(value).__name__}, which "
                "JSON cannot store.",
                fix="convert it with int(...), float(...) or str(...), or a "
                "NumPy array with .tolist()",
            )
        )
    elif not allow_nan and isinstance(value, float) and not np.isfinite(value):
        raise ValueError(
            _format_error(
                f"{subject} value {_path} is {value!r}, but metadata numbers "
                "must be finite.",
                why="Why: standard JSON has no NaN or infinity, so other tools "
                "cannot read the saved file.",
                fix="store a missing value as None",
            )
        )


def _owned_metadata(metadata: object, *, owner: str) -> Mapping[str, Any]:
    """Return a recursively read-only copy of standard-JSON ``metadata``."""
    if not isinstance(metadata, Mapping):
        raise TypeError(
            _format_error(
                f"{owner} metadata must be a mapping, got {type(metadata).__name__}.",
                fix='pass a dict of JSON values, such as {"role": "reward"}',
            )
        )
    try:
        # Thawing copies every container, so the record owns its metadata.
        thawed = _thaw_metadata(metadata)
    except ValueError as exc:
        raise ValueError(
            _format_error(
                f"{owner} metadata contains itself.",
                fix="pass metadata without circular references",
            )
        ) from exc
    _require_json_metadata(thawed, allow_nan=False, owner=owner)
    return cast("Mapping[str, Any]", _freeze_metadata(thawed))


# ---------------------------------------------------------------------
# Geometry validation shared with neurospatial.landmarks
# ---------------------------------------------------------------------
def _require_name(name: object, *, record: str, example: str) -> None:
    """Refuse a name that is not a nonempty string."""
    if not isinstance(name, str) or not name:
        raise ValueError(
            _format_error(
                f"{record} names must be nonempty strings, got {name!r}.",
                fix=f'name each {record.lower()} with a string such as "{example}"',
            )
        )


def _owned_point(
    raw: object, *, owner: str, shape_kinds: Mapping[str, str]
) -> NDArray[np.float64]:
    """Return one finite point as a read-only float64 copy, shape ``(n_dims,)``.

    ``raw`` is a coordinate sequence or a shapely ``Point``. ``shape_kinds``
    maps the shapely geometry types ``owner`` can hold under another kind to
    that kind's name, for the error when such a shape is passed as a point.
    """
    if isinstance(raw, BaseGeometry) and not isinstance(raw, Point):
        alternative = shape_kinds.get(raw.geom_type)
        raise TypeError(
            _format_error(
                f"{owner} has kind='point', but its data is a shapely {raw.geom_type}.",
                fix=(
                    f'use kind="{alternative}" for this shape, or '
                    if alternative is not None
                    else ""
                )
                + "pass the point's coordinates, such as [50.0, 100.0]",
            )
        )
    if isinstance(raw, Point):
        raw = raw.coords[0] if not raw.is_empty else ()
    try:
        point = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            _format_error(
                f"{owner} point coordinates must be numbers, got {raw!r}.",
                fix="pass the point's coordinates, such as [50.0, 100.0]",
            )
        ) from exc
    if point.ndim != 1 or point.size == 0:
        raise ValueError(
            _format_error(
                f"{owner} must be one point, shape (n_dims,), got shape {point.shape}.",
                fix="pass the point's coordinates, such as [50.0, 100.0]",
            )
        )
    if not np.all(np.isfinite(point)):
        raise ValueError(
            _format_error(
                f"{owner} point {point.tolist()} has a NaN or infinite coordinate.",
                why="Why: a point with a missing coordinate has no location, so "
                "it would silently cover no bins.",
                fix="pass finite coordinates in the environment's units",
            )
        )
    point.setflags(write=False)
    return point


def _check_planar_geometry(geometry: Polygon | LineString, *, owner: str) -> None:
    """Refuse an empty, 3-D, non-finite, invalid or zero-size polygon or line."""
    kind = "polygon" if isinstance(geometry, Polygon) else "line"
    if geometry.is_empty:
        raise ValueError(
            _format_error(
                f"{owner} is an empty {kind}.",
                fix="pass its vertices as (x, y) pairs in the environment's "
                "coordinates",
            )
        )
    if geometry.has_z:
        raise ValueError(
            _format_error(
                f"{owner} has z coordinates, but a {kind} is 2-D.",
                fix=f"drop the z coordinates with shapely.force_2d({kind})",
            )
        )
    if not np.all(np.isfinite(shapely.get_coordinates(geometry))):
        raise ValueError(
            _format_error(
                f"{owner} has a NaN or infinite vertex coordinate.",
                fix="remove the missing vertices, or redraw the shape",
            )
        )
    if not geometry.is_valid:
        what = f"{owner} is not a valid {kind}: {explain_validity(geometry)}."
        if kind == "line":
            raise ValueError(
                _format_error(
                    what,
                    fix="pass at least two distinct vertices, such as "
                    "[(0.0, 100.0), (40.0, 100.0)]",
                )
            )
        raise ValueError(
            _format_error(
                what,
                why="Why: an outline whose edges cross has no well-defined "
                "inside, so the bins it covers would be wrong.",
                fix="redraw it so its edges do not cross, or repair it with "
                "shapely.make_valid(polygon) and keep the part you meant",
            )
        )
    size = geometry.area if kind == "polygon" else geometry.length
    if size <= 0:
        measure = "area" if kind == "polygon" else "length"
        raise ValueError(
            _format_error(
                f"{owner} has zero {measure}.",
                fix='use kind="point" for a single location, or spread the '
                "vertices apart",
            )
        )


# ---------------------------------------------------------------------
# Region — immutable value object
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
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


    Examples
    --------
    >>> from neurospatial import Region
    >>> goal = Region(name="goal", kind="point", data=[5.0, 5.0])
    >>> goal.n_dims
    2
    """

    name: str
    kind: Kind
    data: NDArray[np.float64] | Polygon
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    def __post_init__(self) -> None:
        _require_name(self.name, record="Region", example="reward_zone")
        owner = f"Region {self.name!r}"
        object.__setattr__(
            self, "metadata", _owned_metadata(self.metadata, owner=owner)
        )
        if self.kind == "point":
            point = _owned_point(
                self.data, owner=owner, shape_kinds={"Polygon": "polygon"}
            )
            object.__setattr__(self, "data", point)
        elif self.kind == "polygon":
            if not isinstance(self.data, Polygon):
                raise TypeError(
                    _format_error(
                        f"{owner} has kind='polygon', so its data must be a "
                        f"shapely Polygon, got {type(self.data).__name__}.",
                        fix="wrap the vertices with shapely.Polygon(vertices); "
                        "make one region per part of a MultiPolygon",
                    )
                )
            _check_planar_geometry(self.data, owner=owner)
        else:
            raise ValueError(
                _format_error(
                    f"{owner} has unknown kind {self.kind!r}.",
                    fix='use kind="point" or kind="polygon"',
                )
            )

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------
    @property
    def n_dims(self) -> int:
        """Number of coordinates: the point's length, or 2 for a polygon."""
        if isinstance(self.data, np.ndarray):
            return int(self.data.size)
        return 2

    def __str__(self) -> str:
        return self.name

    # Value equality covers coordinate arrays and nested metadata, which have
    # no hash, so regions are unhashable; key collections by region name.
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        """Compare region values, including point coordinates elementwise."""
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, Region)
        if self is other:
            return True
        if (self.name, self.kind, self.metadata) != (
            other.name,
            other.kind,
            other.metadata,
        ):
            return False
        if self.kind == "point":
            return bool(np.array_equal(self.data, other.data))
        return bool(self.data == other.data)

    # -----------------------------------------------------------------
    # Copy / pickle support
    # -----------------------------------------------------------------
    # The stored ``metadata`` is a (recursively) read-only ``MappingProxyType``,
    # which the stdlib copy/pickle machinery cannot handle directly. Copy,
    # deepcopy and pickle rebuild the Region from its thawed metadata through
    # the constructor, which re-freezes it and owns a read-only point copy.
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
        geom = (
            self.data.tolist()
            if isinstance(self.data, np.ndarray)
            else mapping(self.data)
        )

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
    Published snapshots preserve the read API and reject mutation; construct
    ``Regions(snapshot.values())`` to obtain a mutable editing container.

    Parameters
    ----------
    items : iterable of Region or None, optional
        Initial regions, keyed by each region's name. Omit for an empty container.

    Examples
    --------
    >>> from neurospatial import Regions
    >>> regions = Regions()
    >>> _ = regions.add("home", point=(0.0, 0.0))
    >>> regions["home"].name
    'home'
    """

    __slots__ = ("_store",)

    # -------------- Mapping interface --------------------------------
    def __init__(self, items: Iterable[Region] | None = None) -> None:
        self._store: Mapping[str, Region] = {}
        if items is not None:
            for reg in items:
                self[reg.name] = reg

    def _mutable_store(self) -> dict[str, Region]:
        """Return the builder's store, refusing edits to a published snapshot."""
        if not isinstance(self._store, dict):
            raise TypeError(
                "Published regions are read-only. Create a mutable builder with "
                "Regions(regions.values()), then call env.with_regions(builder)."
            )
        return self._store

    def _snapshot(self) -> Regions:
        """Own an ordered, immutable store while sharing immutable Region values."""
        return type(self)._restore(tuple(self.values()), frozen=True)

    @classmethod
    def _restore(cls, items: tuple[Region, ...], frozen: bool) -> Regions:
        regions = cls(items)
        if frozen:
            regions._store = MappingProxyType(regions._store)
        return regions

    def __copy__(self) -> Regions:
        return type(self)._restore(
            tuple(self.values()), frozen=not isinstance(self._store, dict)
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> Regions:
        regions = type(self)._restore(
            copy.deepcopy(tuple(self.values()), memo),
            frozen=not isinstance(self._store, dict),
        )
        memo[id(self)] = regions
        return regions

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return type(self)._restore, (
            tuple(self.values()),
            not isinstance(self._store, dict),
        )

    def __getitem__(self, key: str) -> Region:
        try:
            return self._store[key]
        except KeyError:
            raise RegionNotFoundError(
                key, available=list(self._store), argument=None
            ) from None

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def get(self, key: str, default: Any = None) -> Any:
        """Return the region named ``key``, or ``default`` when it is absent."""
        return self._store.get(key, default)

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
        store = self._mutable_store()
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
        store[key] = value

    def __delitem__(self, key: str) -> None:
        store = self._mutable_store()
        if key not in store:
            raise RegionNotFoundError(key, available=list(store), argument=None)
        del store[key]

    def clear(self) -> None:
        """Remove every region from a mutable builder."""
        self._mutable_store()
        super().clear()

    def pop(self, key: str, *default: Any) -> Any:
        """Remove a region from a mutable builder, optionally returning a default."""
        self._mutable_store()
        return super().pop(key, *default)

    def popitem(self) -> tuple[str, Region]:
        """Remove and return an item from a mutable builder."""
        self._mutable_store()
        return super().popitem()

    def setdefault(self, key: str, default: Region | None = None) -> Region:
        """Return a region, inserting a supplied default into a mutable builder."""
        self._mutable_store()
        return super().setdefault(key, cast("Region", default))

    def update(self, *args: Any, **kwargs: Region) -> None:
        """Insert regions into a mutable builder using the strict assignment path."""
        self._mutable_store()
        super().update(*args, **kwargs)

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
        self._mutable_store()
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
        store = self._mutable_store()
        if (point is None) == (polygon is None):
            raise ValueError("Specify **one** of 'point' or 'polygon'.")
        if name not in self:
            raise RegionNotFoundError(name, available=list(self), argument=None)

        # Preserve existing metadata if not explicitly provided
        old_region = self._store[name]
        effective_metadata = metadata if metadata is not None else old_region.metadata

        if point is not None:
            # Accept either a coordinate array or a Shapely Point
            coords = np.asarray(
                point.coords[0] if isinstance(point, Point) else point, dtype=float
            )
            region = Region(name, "point", coords, effective_metadata)
        else:
            region = Region(name, "polygon", polygon, effective_metadata)

        # Validate before removing the old value, then preserve replacement order.
        del store[name]
        store[name] = region
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
        store = self._mutable_store()
        if name != region.name:
            raise ValueError(f"Key {name!r} must match Region.name {region.name!r}.")
        # Bypass __setitem__'s duplicate check; this is the idempotent path.
        store[name] = region
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
        del self[name]

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
            raise RegionNotFoundError(name, available=list(self._store), argument=None)

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
        self._mutable_store()
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


def _snapshot_regions(regions: Regions) -> Regions:
    """Return an ordered, immutable copy of ``regions`` for an environment to own."""
    return regions._snapshot()
