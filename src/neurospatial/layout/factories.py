import inspect
from enum import Enum
from typing import Any

from neurospatial.layout.base import LayoutEngine
from neurospatial.layout.engines.graph import GraphLayout
from neurospatial.layout.engines.hexagonal import HexagonalLayout
from neurospatial.layout.engines.image_mask import ImageMaskLayout
from neurospatial.layout.engines.masked_grid import MaskedGridLayout
from neurospatial.layout.engines.regular_grid import RegularGridLayout
from neurospatial.layout.engines.shapely_polygon import (
    ShapelyPolygonLayout,
)
from neurospatial.layout.engines.triangular_mesh import (
    TriangularMeshLayout,
)


class LayoutType(str, Enum):
    """Available layout engine types.

    This enum provides IDE autocomplete and type safety for layout selection.
    Each value corresponds to a concrete layout engine implementation.

    Attributes
    ----------
    REGULAR_GRID : str
        Standard rectangular/cuboid grids with uniform bin sizes.
    MASKED_GRID : str
        Grids with arbitrary active/inactive regions defined by a boolean mask.
    IMAGE_MASK : str
        Binary image-based layouts where white pixels define active bins.
    HEXAGONAL : str
        Hexagonal tessellations with more uniform neighbor distances.
    GRAPH : str
        1D linearized track representations for maze/track experiments.
    TRIANGULAR_MESH : str
        Triangular tessellations for specialized spatial discretization.
    SHAPELY_POLYGON : str
        Polygon-bounded grids using Shapely geometry.

    Examples
    --------
    >>> from neurospatial.layout import LayoutType, create_layout
    >>> # Use enum for autocomplete and type safety
    >>> layout = create_layout(
    ...     LayoutType.REGULAR_GRID, bin_size=2.0, dimension_ranges=[(0, 100), (0, 100)]
    ... )  # doctest: +SKIP

    Notes
    -----
    String values are also accepted for backward compatibility:

    >>> layout = create_layout("regular_grid", bin_size=2.0)  # doctest: +SKIP
    """

    REGULAR_GRID = "RegularGrid"
    MASKED_GRID = "MaskedGrid"
    IMAGE_MASK = "ImageMask"
    HEXAGONAL = "Hexagonal"
    GRAPH = "Graph"
    TRIANGULAR_MESH = "TriangularMesh"
    SHAPELY_POLYGON = "ShapelyPolygon"


# Note: We use Any here because LayoutEngine is a Protocol, and type[Protocol]
# causes mypy errors when used with concrete class types
_LAYOUT_MAP: dict[str, Any] = {
    LayoutType.REGULAR_GRID.value: RegularGridLayout,
    LayoutType.MASKED_GRID.value: MaskedGridLayout,
    LayoutType.IMAGE_MASK.value: ImageMaskLayout,
    LayoutType.HEXAGONAL.value: HexagonalLayout,
    LayoutType.GRAPH.value: GraphLayout,
    LayoutType.TRIANGULAR_MESH.value: TriangularMeshLayout,
    LayoutType.SHAPELY_POLYGON.value: ShapelyPolygonLayout,
}


def _normalize_name(name: str) -> str:
    """Normalize a layout name by removing non-alphanumeric characters and
    converting to lowercase.

    Parameters
    ----------
    name : str
        The layout name to normalize.

    Returns
    -------
    str
        The normalized name.

    """
    return "".join(filter(str.isalnum, name)).lower()


def list_available_layouts() -> list[str]:
    """List user-friendly type strings for all available layout engines.

    Returns
    -------
    List[str]
        A sorted list of unique string identifiers for available
        `LayoutEngine` types (e.g., "RegularGrid", "Hexagonal").

    """
    # dict preserves insertion order (Python 3.7+) and deduplicates by key
    unique_layouts = {_normalize_name(name): name for name in _LAYOUT_MAP}
    return sorted(unique_layouts.values())


def get_layout_parameters(layout_type: str) -> dict[str, dict[str, Any]]:
    """Retrieve expected build parameters for a specified layout engine type.

    Inspects the `build` method signature of the specified `LayoutEngine`
    class to determine its required and optional parameters.

    Parameters
    ----------
    layout_type : str
        The string identifier of the layout engine type (case-insensitive,
        ignores non-alphanumeric characters).

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary where keys are parameter names for the `build` method.
        Each value is another dictionary containing:
        - 'annotation': The type annotation of the parameter.
        - 'default': The default value, or `None` if no default.
        - 'kind': The parameter kind (e.g., 'keyword-only').

    Raises
    ------
    ValueError
        If `layout_type` is unknown.

    """
    normalized_kind_query = _normalize_name(layout_type)
    found_key = next(
        (
            name
            for name in _LAYOUT_MAP
            if _normalize_name(name) == normalized_kind_query
        ),
        None,
    )
    if not found_key:
        raise ValueError(
            f"Unknown engine kind '{layout_type}'. Available: {list_available_layouts()}",
        )
    engine_class = _LAYOUT_MAP[found_key]
    sig = inspect.signature(engine_class.build)
    params_info: dict[str, dict[str, Any]] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        params_info[name] = {
            "annotation": (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else None
            ),
            "default": (
                param.default if param.default is not inspect.Parameter.empty else None
            ),
            "kind": param.kind.name,  # Use .name for Python 3.10+ compatibility
        }
    return params_info


def create_layout(kind: LayoutType | str, **kwargs) -> LayoutEngine:
    """Factory for creating and building a spatial-layout engine.

    Parameters
    ----------
    kind : LayoutType | str
        The layout engine type to create. Can be:
        - A LayoutType enum member (recommended for IDE autocomplete)
        - A case-insensitive string name (e.g., "RegularGrid", "Hexagonal")
    **kwargs : any
        Parameters passed to the chosen engine's `build(...)` method.

    Returns
    -------
    LayoutEngine
        A fully constructed layout engine.

    Raises
    ------
    ValueError
        - If `kind` is not one of the available layouts.
        - If any unexpected keyword arguments are passed to `build`.

    Examples
    --------
    Using the enum (recommended):

    >>> from neurospatial.layout import LayoutType, create_layout
    >>> layout = create_layout(
    ...     LayoutType.REGULAR_GRID, bin_size=2.0, dimension_ranges=[(0, 100), (0, 100)]
    ... )  # doctest: +SKIP

    Using strings (backward compatible):

    >>> layout = create_layout(
    ...     "regular_grid", bin_size=2.0, dimension_ranges=[(0, 100), (0, 100)]
    ... )  # doctest: +SKIP

    """
    # 1) Convert enum to string if needed
    kind_str = kind.value if isinstance(kind, LayoutType) else kind

    # 2) Normalize user input and find matching key
    norm_query = _normalize_name(kind_str)
    found_key = next(
        (name for name in _LAYOUT_MAP if _normalize_name(name) == norm_query),
        None,
    )
    if found_key is None:
        suggestions = ", ".join(list_available_layouts())
        raise ValueError(f"Unknown layout kind '{kind_str}'. Available: {suggestions}")

    # 2) Instantiate the class
    engine_cls = _LAYOUT_MAP[found_key]
    engine: LayoutEngine = engine_cls()

    # 3) Validate `kwargs` against `build(...)` signature
    sig = inspect.signature(engine.build)
    allowed = {param for param in sig.parameters if param != "self"}
    unexpected = set(kwargs) - allowed
    if unexpected:
        raise ValueError(f"Unexpected arguments for {found_key}.build(): {unexpected}")

    # 4) Call `build(...)` with validated params
    engine.build(**{k: kwargs[k] for k in allowed if k in kwargs})
    # At this point, engine is a LayoutEngine (Protocol check happens at runtime)
    return engine
