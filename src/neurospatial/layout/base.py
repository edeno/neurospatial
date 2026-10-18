import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry import Binning

if TYPE_CHECKING:
    import matplotlib.axes

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
    """Protocol defining the interface for all spatial layout engines.

    A LayoutEngine discretizes a continuous N-dimensional space into bins and
    lists which bins are neighbors.

    An engine supplies two methods. ``build()`` computes the bins from its
    arguments, and ``binning()`` returns them as a ``Binning``: owned bin
    centers and measures, neighbor pairs with their distances in
    authoritative order, point lookup, finite-volume faces and the optional
    grid, track, shape and outline components, all derived from the engine's
    own state. Environments are constructed from that record; diffusion,
    gradients and divergence use its faces. Custom engines implement the same
    two methods. An environment derives its graph, grid index and ranges from
    the record and reads only two other, optional members: ``layout_type``
    and the build choices that :func:`capture_build_params` records in
    ``_build_params_used``. The other members below describe the built-in
    engines.

    The built-in engines also offer ``connectivity``, a convenience
    ``networkx.Graph`` derived from their binning by
    :func:`neurospatial.geometry.graph_from_binning`: nodes are bin IDs
    ``0..n_bins - 1`` with ``pos``, ``source_grid_flat_index`` and
    ``original_grid_nd_index``; edges are the neighbor pairs with
    ``distance``, ``vector``, ``edge_id`` and, in 2-D, ``angle_2d``. It is
    built on first access and is not part of the protocol.

    Each built-in engine names the build option that sets its cell size, in
    the units of positions, in the class attribute ``size_parameter``:
    ``"bin_size"`` for grids and tracks, ``"hexagon_width"``,
    ``"point_spacing"`` for a mesh, ``"pixel_size"`` for an image mask, and
    ``None`` for a masked grid, whose edges are given. ``from_samples`` passes
    its ``bin_size`` to that option, and a refused ``rebin`` reads the
    recorded value to suggest a rebuild at the coarser size. It is not part
    of the protocol either.

    Attributes
    ----------
    bin_centers : NDArray[np.float64]
        Coordinates of the center of each *active* bin/node.
        Shape is (n_active_bins, n_dims).
    is_linearized_track : bool
        True if the layout represents a primarily 1-dimensional structure
        (e.g., a linearized track), False otherwise.
    dimension_ranges : Optional[Sequence[Tuple[float, float]]]
        The actual min/max extent `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`
        covered by the layout's geometry.
    grid_edges : Optional[Tuple[NDArray[np.float64], ...]]
        For grid-based layouts: A tuple of 1D arrays, where each array
        contains the bin edge positions for one dimension of the *original,
        full grid*. `None` or `()` for non-grid or point-based layouts.
    grid_shape : Optional[Tuple[int, ...]]
        For grid-based layouts: The N-D shape (number of bins in each
        dimension) of the *original, full grid*.
        For point-based/cell-based layouts without a full grid concept:
        Typically `(n_active_bins,)`.
    active_mask : Optional[NDArray[np.bool_]]
        - For grid-based layouts: An N-D boolean mask indicating active bins
          on the *original, full grid* (shape matches `grid_shape`).
        - For point-based/cell-based layouts: A 1D array of `True` values,
          shape `(n_active_bins,)`, corresponding to `bin_centers`.
    _build_params_used : Dict[str, Any]
        The scalar build choices recorded by :func:`capture_build_params`;
        input data arguments are not recorded.
    layout_type : str
        The layout's label. Each built-in engine's label is a ``LayoutType``
        value, such as ``"RegularGrid"`` or ``"Hexagonal"``, and its
        ``binning()`` records the same label as ``Binning.kind``. A custom
        engine's label is its own; when it has none, the environment uses the
        ``kind`` of its binning.

    """

    # --- Required Data Attributes ---
    bin_centers: NDArray[np.float64]
    dimension_ranges: Sequence[tuple[float, float]] | None

    # Attributes primarily for GRID-BASED Layouts
    @property
    def grid_edges(self) -> tuple[NDArray[np.float64], ...] | None:
        """Cell edges when a grid or track representation is available."""
        ...

    @property
    def grid_shape(self) -> tuple[int, ...] | None:
        """Full grid shape, including inactive cells."""
        ...

    @property
    def active_mask(self) -> NDArray[np.bool_] | None:
        """Active cells in full grid order."""
        ...

    # Internal Attributes for Introspection/Serialization
    _build_params_used: dict[str, Any]

    # --- Public Properties for Layout Classification ---
    @property
    def layout_type(self) -> str:
        """Return the layout's label, a ``LayoutType`` value for built-in engines.

        Returns
        -------
        str
            The label the environment reports as ``env.layout_type``, for
            example ``"RegularGrid"``.

        """
        ...

    # --- Required Methods ---
    def build(self, **kwargs) -> None:
        """Construct the layout's bins and their neighbor pairs.

        This method is responsible for populating all the attributes defined
        in the `LayoutEngine` protocol (e.g., `bin_centers`) and the state
        that ``binning()`` reads, based on the provided keyword arguments.
        The specific arguments required will vary depending on the concrete
        implementation of the layout engine.
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

    @property  # pragma: no cover
    def is_linearized_track(self) -> bool:
        """Indicate if the layout structure is primarily 1-dimensional.

        Returns
        -------
        bool
            True if the layout represents a 1D structure (e.g., a linearized
            track), False otherwise.

        """
        ...

    def bin_sizes(self) -> NDArray[np.float64]:
        """Return the area (2D) or volume (3D+) of each active bin.

        For 1D layouts, this typically returns the length of each bin.
        The measures should correspond to the dimensionality of the space
        the bins occupy.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            An array where each element is the area/volume/length of the
            corresponding active bin.

        """
        ...

    def plot(
        self,
        ax: "matplotlib.axes.Axes | None" = None,
        **kwargs,
    ) -> "matplotlib.axes.Axes":
        """Plot the layout's geometry.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The Matplotlib axes to plot on. If None, a new figure and axes
            are created. Defaults to None.
        **kwargs : Any
            Additional keyword arguments for plot customization, specific to
            the layout engine implementation.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout is plotted.

        """
        ...
