import inspect
from collections.abc import Callable, Iterable, Mapping
from functools import wraps
from typing import (
    Any,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np

from neurospatial._exceptions import _format_error
from neurospatial.geometry import Binning

T = TypeVar("T")


_SCALAR_CHOICES = (bool, int, float, str, type(None))


def _choice(value: Any) -> Any:
    """Return a build choice as a plain scalar or a tuple of scalars, or raise.

    NumPy scalars become Python scalars; lists, tuples and 1-D integer or
    floating-point arrays become tuples. Anything else raises ``TypeError``.
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value[()]
        elif value.ndim == 1 and value.dtype.kind in "iuf":
            value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, _SCALAR_CHOICES):
        return value
    if isinstance(value, (list, tuple)):
        items = tuple(_choice(item) for item in value)
        if all(isinstance(item, _SCALAR_CHOICES) for item in items):
            return items
    raise TypeError(f"{type(value).__name__} is not a scalar or a flat sequence")


_DECLARE_DATA = (
    "declare it as data with @capture_build_params(data=({name!r},)) on the "
    "engine's build method"
)


def build_choices(
    params: Mapping[str, Any], *, owner: str, fix: str = _DECLARE_DATA
) -> dict[str, Any]:
    """Return build choices as plain values, refusing anything that is data.

    A choice is a scalar (bool, number, string or None) or a flat sequence of
    scalars, such as a per-axis ``bin_size``. Sequences become tuples and
    NumPy scalars become Python scalars.

    Parameters
    ----------
    params : mapping of str to object
        Choices recorded by a layout engine or a derivation, or read back
        from a saved environment.
    owner : str
        Where the choices come from, named in the error.
    fix : str, optional
        How to fix a refused value, formatted with the argument's ``name``;
        by default, declare it as data in :func:`capture_build_params`.

    Returns
    -------
    dict of str to object
        The choices, in ``params`` order.

    Raises
    ------
    TypeError
        If a value is not a scalar or a flat sequence of scalars.
    """
    choices: dict[str, Any] = {}
    for name, value in params.items():
        try:
            choices[name] = _choice(value)
        except TypeError as error:
            raise TypeError(
                _format_error(
                    f"{owner} argument {name!r} is not a build choice: {error}.",
                    why=(
                        "Why: an environment records only scalar build choices, "
                        "such as bin_size; input data (positions, masks, "
                        "polygons, graphs) is described by its bins instead."
                    ),
                    fix=fix.format(name=name),
                )
            ) from None
    return choices


@overload
def capture_build_params(build_method: Callable[..., T], /) -> Callable[..., T]: ...


@overload
def capture_build_params(
    *, data: Iterable[str] = ()
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...


def capture_build_params(
    build_method: Callable[..., T] | None = None,
    /,
    *,
    data: Iterable[str] = (),
) -> Callable[..., T] | Callable[[Callable[..., T]], Callable[..., T]]:
    """Record a layout engine's build choices, declaring its data arguments once.

    The decorated ``build()`` stores its arguments, defaults included, in
    ``self._build_params_used`` after it returns; an environment built from
    the engine reports them as ``env.layout_parameters``. Arguments named in
    ``data`` are input data, such as positions, masks, polygons or graphs,
    and are not recorded. Every other argument must be a build choice: a
    scalar or a flat sequence of scalars.

    Parameters
    ----------
    build_method : callable, optional
        The build method, when the decorator is used bare
        (``@capture_build_params``).
    data : iterable of str, default=()
        Names of the build arguments that are input data.

    Returns
    -------
    callable
        The wrapped build method, or a decorator when called with ``data``.

    Raises
    ------
    TypeError
        At decoration, if ``data`` names an argument ``build`` does not have
        (not checked when ``build`` takes ``**kwargs``). At build, if an
        argument outside ``data`` is not a build choice.

    Examples
    --------
    >>> import numpy as np
    >>> class MyLayout:
    ...     @capture_build_params(data=("positions",))
    ...     def build(self, *, positions, bin_size: float, connect: bool = True):
    ...         pass
    >>> layout = MyLayout()
    >>> layout.build(positions=np.zeros((5, 2)), bin_size=np.float64(2.0))
    >>> layout._build_params_used
    {'bin_size': 2.0, 'connect': True}
    """
    data_names = frozenset(data)

    def decorate(method: Callable[..., T]) -> Callable[..., T]:
        signature = inspect.signature(method)
        variadic = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        }
        # A build(**kwargs) method may receive any name, so only a method
        # without **kwargs can be checked against its parameters.
        unknown = sorted(data_names - set(signature.parameters))
        if unknown and not variadic:
            raise TypeError(
                _format_error(
                    f"capture_build_params(data=...) names {unknown}, which "
                    f"{method.__qualname__} does not take.",
                    why="Why: data names must be build arguments, or they would "
                    "silently never match.",
                    fix="list only names of the build method's parameters in data=",
                )
            )

        @wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            params: dict[str, Any] = {}
            for name, value in list(bound.arguments.items())[1:]:
                if name in variadic:
                    params.update(
                        (key, item)
                        for key, item in value.items()
                        if key not in data_names
                    )
                elif name not in data_names:
                    params[name] = value
            # Refuse data before building; record after, so the outermost
            # build of a subclass that calls super().build() wins.
            choices = build_choices(params, owner=f"{type(self).__name__}.build()")
            result = method(self, *args, **kwargs)
            self._build_params_used = choices
            return result

        return wrapper

    return decorate if build_method is None else decorate(build_method)


@runtime_checkable
class LayoutEngine(Protocol):
    """What an environment needs from a layout engine: ``build()`` and ``binning()``.

    A layout engine discretizes a continuous N-dimensional space into bins and
    lists which bins are neighbors. ``build()`` computes the bins from its
    arguments, and ``binning()`` returns them as a ``Binning``: owned bin
    centers and measures, neighbor pairs with their distances in
    authoritative order, point lookup, finite-volume faces and the optional
    grid, track, shape and outline components, all derived from the engine's
    own state. Environments are constructed from that record; diffusion,
    gradients and divergence use its faces, and the environment derives its
    graph, grid index and ranges from it. A custom engine implements these
    two methods and nothing else.

    An environment also reads two optional members when an engine has them:
    ``layout_type``, a string label that the built-in engines declare as a
    ``ClassVar`` (otherwise the binning's ``kind`` is the label; a subclass
    sets its own), and the build choices that :func:`capture_build_params` records in
    ``_build_params_used`` (otherwise none).

    The built-in engines offer more for working with an engine directly:
    ``bin_centers``, ``dimension_ranges``, ``grid_edges``, ``grid_shape``,
    ``active_mask``, ``is_linearized_track``, ``bin_sizes()`` and ``plot()``;
    ``connectivity``, a ``networkx.Graph`` built on first access from their
    binning by :func:`neurospatial.geometry.graph_from_binning`; and the class
    attribute ``size_parameter``, the build option that sets the cell size in
    the units of positions (``"bin_size"`` for grids and tracks,
    ``"hexagon_width"``, ``"point_spacing"`` for a mesh, ``"pixel_size"`` for
    an image mask, ``None`` for a masked grid, whose edges are given).
    ``from_samples`` passes its ``bin_size`` to that option, and a refused
    ``rebin`` reads the recorded value to suggest a rebuild at the coarser
    size. None of these is part of the protocol.
    """

    def build(self, **kwargs: Any) -> None:
        """Compute the layout's bins and their neighbor pairs from its arguments.

        The arguments depend on the engine.
        """
        ...

    def binning(self) -> Binning:
        """Return the built layout's owned geometry record.

        Neighbor pairs are in authoritative order: it becomes
        ``env.edge_pairs``, the order of edge fields in gradients and
        divergence. Every component must be a built-in geometry record (no
        callbacks to the layout), so the environment can own it.

        Returns
        -------
        Binning
            Bins, point lookup, finite-volume faces (or the reason they are
            missing) and the layout's grid, track, shape and outline
            components.

        Raises
        ------
        LayoutNotBuiltError
            If the layout has not been built.
        """
        ...
