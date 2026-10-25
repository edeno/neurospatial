import inspect
from typing import Any, get_args

from neurospatial._exceptions import _format_error
from neurospatial.layout._labels import LayoutType, _key, find_layout_type
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

# The engines ``create_layout`` builds. Every one has the members the
# ``LayoutEngine`` docstring lists for the built-in engines, such as
# ``bin_centers``, ``connectivity``, ``bin_sizes()`` and ``plot()``, so the
# union is the return type and the registry below is derived from it.
_BuiltinLayout = (
    RegularGridLayout
    | MaskedGridLayout
    | ImageMaskLayout
    | HexagonalLayout
    | GraphLayout
    | TriangularMeshLayout
    | ShapelyPolygonLayout
)
_ENGINES: tuple[type[_BuiltinLayout], ...] = get_args(_BuiltinLayout)
# The engines keyed by their ``layout_type`` label.
_LAYOUT_MAP: dict[str, type[_BuiltinLayout]] = {
    engine.layout_type: engine for engine in _ENGINES
}


def _engine_class(kind: object, caller: str) -> type[_BuiltinLayout]:
    """Return the engine class that ``kind`` names, or explain the choices."""
    member = find_layout_type(kind)
    engine = None if member is None else _LAYOUT_MAP.get(member.value)
    if engine is not None:
        return engine
    if member is LayoutType.POLAR:
        why = "Why: polar sectors are egocentric coordinates, which only Environment.from_polar_egocentric declares."
        fix = "build polar bins with Environment.from_polar_egocentric(...)"
    elif member is LayoutType.COMBINED:
        why = "Why: a combined environment joins existing environments."
        fix = "join environments with Environment.combine({...})"
    elif isinstance(kind, str) and _key(kind) == "mask":
        why = "Why: 'mask' could mean a masked grid or an image mask."
        fix = "use LayoutType.MASKED_GRID ('MaskedGrid') or LayoutType.IMAGE_MASK ('ImageMask')"
    else:
        why = "Why: the name matches no layout engine."
        fix = f"use one of {', '.join(list_available_layouts())}"
    name = kind.value if isinstance(kind, LayoutType) else kind
    raise ValueError(
        _format_error(f"{caller} cannot build layout {name!r}.", why=why, fix=fix)
    )


def _size_parameter(kind: object) -> str | None:
    """Return the build option that sets the cell size of the layout ``kind`` names.

    Parameters
    ----------
    kind : LayoutType or str
        A layout label, such as ``Binning.kind``.

    Returns
    -------
    str or None
        The engine's ``size_parameter``, such as ``"bin_size"`` or
        ``"hexagon_width"``; None when no built-in engine builds ``kind`` or
        no single option sets its cell size.
    """
    member = find_layout_type(kind)
    engine = None if member is None else _LAYOUT_MAP.get(member.value)
    return None if engine is None else engine.size_parameter


def list_available_layouts() -> list[str]:
    """List the layout names that ``create_layout`` builds.

    Returns
    -------
    list of str
        Sorted ``LayoutType`` values with a layout engine (e.g.,
        "Hexagonal", "RegularGrid").

    """
    return sorted(_LAYOUT_MAP)


def get_layout_parameters(layout_type: str) -> dict[str, dict[str, Any]]:
    """Retrieve expected build parameters for a specified layout engine type.

    Inspects the `build` method signature of the specified `LayoutEngine`
    class to determine its required and optional parameters.

    Parameters
    ----------
    layout_type : LayoutType or str
        The layout name; the match ignores case and characters that are not
        letters or digits (see ``LayoutType``).

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
    engine_class = _engine_class(layout_type, "get_layout_parameters()")
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


def create_layout(kind: LayoutType | str, **kwargs: Any) -> _BuiltinLayout:
    """Factory for creating and building a spatial-layout engine.

    Parameters
    ----------
    kind : LayoutType | str
        The layout engine type to create. Can be:
        - A LayoutType enum member (recommended for IDE autocomplete)
        - A layout name, matched ignoring case and characters that are not
          letters or digits (e.g., "RegularGrid", "regular-grid", "grid")
    **kwargs : any
        Parameters passed to the chosen engine's `build(...)` method.

    Returns
    -------
    RegularGridLayout or another built-in layout engine
        The built engine that ``kind`` names. Every built-in engine has
        ``bin_centers``, ``layout_type``, ``connectivity``, ``bin_sizes()``,
        ``plot()`` and ``binning()``.

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

    Using a layout name, matched ignoring case and non-alphanumeric
    characters:

    >>> layout = create_layout(
    ...     "regular_grid", bin_size=2.0, dimension_ranges=[(0, 100), (0, 100)]
    ... )  # doctest: +SKIP

    """
    engine_cls = _engine_class(kind, "create_layout()")
    engine = engine_cls()

    # 3) Validate `kwargs` against `build(...)` signature
    sig = inspect.signature(engine.build)
    allowed = {param for param in sig.parameters if param != "self"}
    unexpected = set(kwargs) - allowed
    if unexpected:
        raise ValueError(
            f"Unexpected arguments for {engine_cls.layout_type}.build(): {unexpected}"
        )

    # 4) Call `build(...)` with validated params
    engine.build(**{k: kwargs[k] for k in allowed if k in kwargs})
    return engine
