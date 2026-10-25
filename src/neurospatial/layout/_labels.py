"""The canonical layout labels and the spellings that name them.

``LayoutType`` is the one vocabulary of layout labels. Each built-in layout
engine's ``layout_type`` is a member's value, ``Binning.kind`` holds the label
an environment was built with, and ``Environment.layout_type`` reports it.
"""

from __future__ import annotations

from enum import Enum


class LayoutType(str, Enum):
    """Layout labels: the built-in layout engines and the special builds.

    Every value is also the ``layout_type`` of the engine, the ``kind`` of the
    binning it builds and the ``layout_type`` of the resulting environment.

    Attributes
    ----------
    REGULAR_GRID : str
        Standard rectangular/cuboid grids with uniform bin sizes.
    MASKED_GRID : str
        Grids with arbitrary active/inactive regions defined by a boolean mask.
    IMAGE_MASK : str
        Boolean image-based layouts where True pixels define active bins.
    HEXAGONAL : str
        Hexagonal tessellations with more uniform neighbor distances.
    GRAPH : str
        1D linearized track representations for maze/track experiments.
    TRIANGULAR_MESH : str
        Triangular tessellations for specialized spatial discretization.
    SHAPELY_POLYGON : str
        Polygon-bounded grids using Shapely geometry.
    POLAR : str
        Egocentric distance-by-angle sectors from
        ``Environment.from_polar_egocentric``. Not built with ``create_layout``.
    COMBINED : str
        Disjoint components joined by ``Environment.combine``. Not built with
        ``create_layout``.

    Examples
    --------
    >>> from neurospatial.layout import LayoutType, create_layout
    >>> layout = create_layout(
    ...     LayoutType.REGULAR_GRID, bin_size=2.0, dimension_ranges=[(0, 100), (0, 100)]
    ... )
    >>> layout.layout_type == LayoutType.REGULAR_GRID
    True
    >>> str(LayoutType.REGULAR_GRID)
    'RegularGrid'
    >>> f"{LayoutType.HEXAGONAL} bins"
    'Hexagonal bins'

    Notes
    -----
    ``create_layout`` and the ``layout=`` option of the sample factories also
    accept a layout name as a string. The match ignores case and any
    character that is not a letter or digit, so ``"RegularGrid"``,
    ``"regular_grid"`` and ``"regular-grid"`` all name ``REGULAR_GRID``.
    The short names ``"grid"``, ``"polygon"`` and ``"mesh"`` name
    ``REGULAR_GRID``, ``SHAPELY_POLYGON`` and ``TRIANGULAR_MESH``.
    ``list_available_layouts()`` returns the names ``create_layout`` builds.
    """

    REGULAR_GRID = "RegularGrid"
    MASKED_GRID = "MaskedGrid"
    IMAGE_MASK = "ImageMask"
    HEXAGONAL = "Hexagonal"
    GRAPH = "Graph"
    TRIANGULAR_MESH = "TriangularMesh"
    SHAPELY_POLYGON = "ShapelyPolygon"
    POLAR = "Polar"
    COMBINED = "Combined"

    # A str-mixin Enum prints as "LayoutType.REGULAR_GRID" on Python 3.11+
    # and as the value on 3.10; StrEnum, which prints the value, is 3.11+.
    def __str__(self) -> str:
        return self.value

    def __format__(self, format_spec: str) -> str:
        return format(self.value, format_spec)


def _key(name: str) -> str:
    """Lower-case ``name`` and drop every character that is not alphanumeric."""
    return "".join(filter(str.isalnum, name)).lower()


# Short names that name exactly one layout. ``"mask"`` is not one: it could
# mean a masked grid or an image mask.
_ALIASES = {
    "grid": LayoutType.REGULAR_GRID,
    "polygon": LayoutType.SHAPELY_POLYGON,
    "mesh": LayoutType.TRIANGULAR_MESH,
}
_BY_KEY: dict[str, LayoutType] = {
    **{_key(member.value): member for member in LayoutType},
    **_ALIASES,
}


def find_layout_type(name: object) -> LayoutType | None:
    """Return the layout that ``name`` spells, or None.

    The same match is used by ``create_layout``, the ``layout=`` option of
    the sample factories and the readers of saved environments.

    Parameters
    ----------
    name : object
        A ``LayoutType`` or a string; the match ignores case and characters
        that are not letters or digits, and accepts the short names
        ``"grid"``, ``"polygon"`` and ``"mesh"``.

    Returns
    -------
    LayoutType or None
        The named layout, or None when ``name`` names none.

    Examples
    --------
    >>> from neurospatial.layout import LayoutType, find_layout_type
    >>> find_layout_type("regular-grid") is LayoutType.REGULAR_GRID
    True
    >>> find_layout_type("grid").value
    'RegularGrid'
    >>> find_layout_type("mask") is None  # a masked grid or an image mask
    True
    """
    if isinstance(name, LayoutType):
        return name
    if not isinstance(name, str):
        return None
    return _BY_KEY.get(_key(name))


def canonical_label(name: str) -> str:
    """Return the canonical spelling of a known layout label.

    A label that names no ``LayoutType`` member, such as a custom engine's,
    is returned unchanged.
    """
    member = find_layout_type(name)
    return name if member is None else member.value
