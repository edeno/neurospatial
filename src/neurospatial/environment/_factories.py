"""Construction adapters that consume builders once and publish owned environments."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, cast

import networkx as nx
import numpy as np
import shapely
import shapely.geometry
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.environment import _publication
from neurospatial.environment._publication import _declared_metadata, _Provenance
from neurospatial.geometry import (
    CombinedLocator,
    OpenArena,
    TrackArena,
    own_arena,
    prepare_bins,
)
from neurospatial.layout import LayoutType, find_layout_type
from neurospatial.layout.factories import (
    _size_parameter,
    create_layout,
    list_available_layouts,
)
from neurospatial.layout.helpers.track_graphs import (
    add_edge_with_distance,
    example_node_positions,
    maze_track_graph,
    refuse_coincident_nodes,
)
from neurospatial.regions import Regions

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment
    from neurospatial.geometry import Arena, Binning
    from neurospatial.layout.base import LayoutEngine

# A spatial environment has very few dimensions (1-D linear tracks, 2-D open
# fields, occasionally 3-D). A column count above this, *combined with* more
# columns than rows, signals a transposed `positions` array rather than a
# genuine high-D environment (see `from_samples`).
_MAX_PLAUSIBLE_SPATIAL_DIMS = 8


def _owned_copy(value: Any) -> Any:
    """Copy arrays, also inside dicts, lists and tuples, so callers keep theirs."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _owned_copy(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_owned_copy(item) for item in value)
    return value


def _from_bins(
    cls: type[Environment[Any]],
    binning: Binning,
    *,
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
    regions: Regions | None = None,
    arena: Arena | None = None,
) -> Environment:
    """Implement Environment.from_bins using explicit construction inputs."""
    snapshot = prepare_bins(binning)
    # A label naming a LayoutType is spelled canonically; others are kept.
    member = find_layout_type(snapshot.kind)
    if member is not None:
        snapshot = replace(snapshot, kind=member.value)
    if arena is None and snapshot.outline is not None:
        arena = OpenArena(snapshot.outline.geometry, outline=None)
    result = _publication._publish(
        cls,
        snapshot,
        provenance=_Provenance({}),
        metadata=_declared_metadata(name=name, units=units, frame=frame),
        regions=regions,
        arena=arena,
    )
    # "Polar" belongs to from_polar_egocentric, and "Combined" to joined
    # components (a CombinedLocator), never to other Cartesian bins.
    if member is LayoutType.POLAR or (
        member is LayoutType.COMBINED
        and not isinstance(snapshot.locator, CombinedLocator)
    ):
        factory = (
            "Environment.from_polar_egocentric"
            if member is LayoutType.POLAR
            else "Environment.combine"
        )
        raise ValueError(
            _format_error(
                f"from_bins cannot label these bins {snapshot.kind!r}.",
                why=f"Why: the {member.value!r} layout label is reserved for "
                f"environments built by {factory}.",
                fix="give the Binning a kind of your own, such as 'MyBins', "
                f"or build the environment with {factory}",
            )
        )

    if not np.array_equal(result.edge_pairs, snapshot.connections.pairs):
        raise ValueError(
            "Supplied open neighbor pairs cross physical walls. Close those pairs "
            "(BinConnections.from_arrays(pairs, distances, is_open=...)) and drop "
            "their aligned face data explicitly before calling from_bins."
        )
    return result


# Defaults of the options shared by from_samples, open_field and
# environment_from_position. They are also the values that leave the bins as
# a layout without the option builds them: no clean-up, no threshold, and the
# neighbors the layout has anyway. An option that the chosen layout or
# infer_active_bins=False ignores is refused only when its value is neither
# the factory's default nor that no-op value, because only then would the
# call ask for bins it does not build.
_SAMPLE_OPTION_DEFAULTS: dict[str, Any] = {
    "bin_count_threshold": 0,
    "dilate": False,
    "fill_holes": False,
    "close_gaps": False,
    "add_boundary_bins": False,
    "connect_diagonal_neighbors": True,
}
# Environment.open_field is from_samples that fills holes by default.
_OPEN_FIELD_OPTION_DEFAULTS: dict[str, Any] = {
    **_SAMPLE_OPTION_DEFAULTS,
    "fill_holes": True,
}
# A hexagonal layout has six fixed neighbors and no mask clean-up.
_REGULAR_GRID_ONLY = (
    "add_boundary_bins",
    "dilate",
    "fill_holes",
    "close_gaps",
    "connect_diagonal_neighbors",
)
# With infer_active_bins=False every bin in the extent is active.
_MASK_OPTIONS = ("bin_count_threshold", "dilate", "fill_holes", "close_gaps")


def _changed(
    options: dict[str, Any], defaults: dict[str, Any], names: Sequence[str]
) -> list[str]:
    """Show each option in ``names`` that asks for something the bins lack.

    A value is accepted when it is the factory's default (``defaults``) or
    the no-op value in ``_SAMPLE_OPTION_DEFAULTS``.
    """
    return [
        f"{name}={options[name]!r}"
        for name in names
        if options[name] != defaults[name]
        and options[name] != _SAMPLE_OPTION_DEFAULTS[name]
    ]


def _check_sample_bin_size(
    bin_size: Any, n_dims: int, factory: str, hexagonal: bool
) -> None:
    """Refuse a bin_size whose length does not fit the layout's axes."""
    try:
        sizes = np.asarray(bin_size, dtype=float)
    except (TypeError, ValueError):
        return  # The numeric type check reports these.
    if sizes.ndim == 0:
        return
    if hexagonal:
        raise ValueError(
            _format_error(
                f"{factory}(layout='Hexagonal') takes one bin_size, the hexagon "
                f"width, got {list(sizes.tolist())}.",
                why="Why: a hexagonal tiling has the same width in every direction.",
                fix=f"pass a single number, such as bin_size={float(sizes.flat[0]):g}",
            )
        )
    if sizes.shape != (n_dims,):
        raise ValueError(
            _format_error(
                f"{factory}() got {sizes.size} bin sizes for positions with "
                f"{n_dims} column(s).",
                why="Why: a sequence gives one bin size per spatial axis, in "
                "the order of the columns of positions.",
                fix=f"pass one number for square bins, such as "
                f"bin_size={float(sizes.flat[0]):g}, or {n_dims} numbers, one per "
                "column",
            )
        )


def _check_dimension_ranges(
    dimension_ranges: Any, positions: NDArray[np.float64], factory: str
) -> None:
    """Refuse dimension_ranges that are not one finite (low, high) per axis."""
    n_dims = positions.shape[1]
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    if len(finite):
        low, high = finite.min(axis=0), finite.max(axis=0)
    else:
        low, high = np.zeros(n_dims), np.ones(n_dims)
    example = ", ".join(
        f"({lo:.4g}, {hi:.4g})" for lo, hi in zip(low, high, strict=True)
    )
    try:
        ranges = np.asarray(dimension_ranges, dtype=float)
    except (TypeError, ValueError):
        ranges = None
    if ranges is None or ranges.shape != (n_dims, 2):
        problem = (
            f"got dimension_ranges={dimension_ranges!r}, which is not "
            f"{n_dims} (min, max) pair(s), one per column of positions"
        )
    elif not np.all(np.isfinite(ranges)):
        problem = (
            f"got dimension_ranges={dimension_ranges!r}, which has a value "
            "that is not finite"
        )
    elif not np.all(ranges[:, 0] < ranges[:, 1]):
        axis = int(np.flatnonzero(ranges[:, 0] >= ranges[:, 1])[0])
        problem = (
            f"got dimension_ranges={dimension_ranges!r}, whose pair for axis "
            f"{axis} does not have its minimum below its maximum"
        )
    else:
        return
    raise ValueError(
        _format_error(
            f"{factory}() {problem}.",
            why="Why: dimension_ranges gives each spatial axis its finite "
            "(minimum, maximum) extent, with the minimum first, so the grid "
            "has a positive width along every axis.",
            fix=f"pass dimension_ranges=[{example}] (the extent of these "
            "positions), or leave it out to use the extent of the samples",
        )
    )


def diagonal_choice(n_dims: int, connect_diagonal_neighbors: bool) -> bool:
    """Return the diagonal-neighbor choice to build and record for a grid.

    A diagonal neighbor differs along two axes, so a 1-D grid, and every grid
    derived from it, has none; its record is then ``False``, the value that
    does nothing, whatever was asked for. A 2-D grid one cell wide keeps the
    choice, because refining it with ``rebin`` creates diagonal neighbors.
    """
    return bool(connect_diagonal_neighbors) if n_dims >= 2 else False


def _from_samples(
    cls: type[Environment[Any]],
    positions: NDArray[np.float64],
    *,
    bin_size: float | Sequence[float],
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
    regions: Regions | None = None,
    arena: Arena | None = None,
    wall_fit: Literal["snap", "split"] = "snap",
    layout: LayoutType | str = LayoutType.REGULAR_GRID,
    dimension_ranges: Sequence[tuple[float, float]] | None = None,
    infer_active_bins: bool = True,
    bin_count_threshold: int = 0,
    dilate: bool = False,
    fill_holes: bool = False,
    close_gaps: bool = False,
    add_boundary_bins: bool = False,
    connect_diagonal_neighbors: bool = True,
    factory: str = "from_samples",
    defaults: dict[str, Any] = _SAMPLE_OPTION_DEFAULTS,
) -> Environment:
    """Implement Environment.from_samples, open_field and the NWB builder.

    ``defaults`` holds the public factory's defaults of the shared options;
    an option the layout ignores is refused only when it differs from them.
    ``factory`` names the public function in error messages.
    """
    try:
        positions = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError(
            _format_error(
                f"{factory}() needs numeric positions, got a "
                f"{type(positions).__name__} whose values are not all numbers "
                f"({e}).",
                why="Why: each row is one sample and each column one spatial "
                "coordinate in physical units, so every value must be a number.",
                fix="convert the coordinate columns to floats, for example "
                "positions = df[['x', 'y']].to_numpy(dtype=float), and mark "
                "missing samples with np.nan",
            )
        ) from e

    if positions.ndim != 2:
        raise ValueError(
            _format_error(
                f"positions must be a 2D array of shape (n_points, n_dims), got shape {positions.shape}.",
                fix="use positions[:, None] for 1-D data; if positions has shape (n_dims, n_samples), pass positions.T",
                why="Why: each row must be one sample and each column one spatial coordinate.",
            )
        )

    # An infinite coordinate is not a position in the arena: treat it as a
    # missing sample, like NaN, which the layouts already leave out.
    if np.isinf(positions).any():
        positions = np.where(np.isinf(positions), np.nan, positions)

    # Warn on a likely-transposed positions array before building the grid.
    # The classic footgun is np.array([x, y]) -- shape (n_dims, n_samples) --
    # instead of column-stacking, so `positions` is transposed and silently
    # builds a nonsensical high-D env. Two signals together flag it: (1) more
    # columns (apparent dims) than rows (apparent samples) -- real position
    # data has many more samples than dims; and (2) an implausibly high
    # dimension count for a spatial environment. This is only a *heuristic*
    # (a genuine low-sample high-D env, e.g. shape (1, 9), trips it too), so
    # it warns rather than rejects -- from_samples supports N-dimensional
    # environments. A truly catastrophic transpose (huge n_dims) is still
    # caught downstream by the int64 bin-count guard in
    # layout.helpers.utils, so nothing silently OOMs.
    n_samples, n_dims = positions.shape
    if n_dims > n_samples and n_dims > _MAX_PLAUSIBLE_SPATIAL_DIMS:
        warnings.warn(
            f"positions has {n_dims} columns (spatial dimensions) but only "
            f"{n_samples} rows (samples) -- fewer samples than dimensions "
            f"(shape {positions.shape}). If this is a genuine "
            f"{n_dims}-D environment, ignore this; otherwise positions is "
            f"likely transposed (it must be (n_samples, n_dims), one row per "
            f"sample) -- pass positions.T.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )

    # Validate bin_size early to provide helpful error messages
    if not isinstance(bin_size, (int, float, np.number, list, tuple, np.ndarray)):
        try:
            size_text = f"{float(cast('Any', bin_size)):g}"
        except (TypeError, ValueError):
            size_text = "2.0"
        raise TypeError(
            _format_error(
                f"{factory}() takes bin_size as a number, got "
                f"{type(bin_size).__name__} {bin_size!r}.",
                why="Why: bin_size is a length in the units of positions.",
                fix=f"pass a number, such as bin_size={size_text}",
            )
        )

    layout_str = layout.value if isinstance(layout, LayoutType) else layout
    layout_type = find_layout_type(layout)

    # Only a regular grid and a hexagonal tiling are built from samples.
    tiled = (LayoutType.REGULAR_GRID, LayoutType.HEXAGONAL)
    if layout_type not in tiled:
        others = [
            name
            for name in list_available_layouts()
            if name not in {kind.value for kind in tiled}
        ]
        raise ValueError(
            _format_error(
                f"{factory}() cannot build layout={layout_str!r}; it supports "
                "layout='RegularGrid' and layout='Hexagonal'.",
                why=(
                    "Why: only those layouts are tiled from position samples. "
                    f"Other layouts ({', '.join(others)}) "
                    "are built from their own inputs."
                ),
                fix=(
                    "use layout='RegularGrid' or layout='Hexagonal'; for a "
                    "polygon, mask or track use from_polygon, from_grid_mask, "
                    "from_pixel_mask or from_graph; for other layouts build the "
                    "engine with neurospatial.layout.create_layout(...) and "
                    "pass it to Environment.from_layout"
                ),
            )
        )

    _check_sample_bin_size(
        bin_size, n_dims, factory, hexagonal=layout_type is LayoutType.HEXAGONAL
    )
    if dimension_ranges is not None:
        _check_dimension_ranges(dimension_ranges, positions, factory)

    options: dict[str, Any] = {
        "bin_count_threshold": bin_count_threshold,
        "dilate": dilate,
        "fill_holes": fill_holes,
        "close_gaps": close_gaps,
        "add_boundary_bins": add_boundary_bins,
        "connect_diagonal_neighbors": connect_diagonal_neighbors,
    }
    changed_grid_options = _changed(options, defaults, _REGULAR_GRID_ONLY)
    if layout_type is LayoutType.HEXAGONAL and changed_grid_options:
        listed = ", ".join(changed_grid_options)
        raise ValueError(
            _format_error(
                f"{factory}(layout={layout_str!r}) cannot apply {listed}.",
                why=(
                    "Why: these options shape the active-bin mask and the "
                    "neighbor graph of a regular grid. A hexagonal layout "
                    "always connects each hexagon to its six neighbors and "
                    "does not clean up the active-bin mask, so the "
                    f"{'values' if len(changed_grid_options) > 1 else 'value'} "
                    "would be ignored."
                ),
                fix=(
                    f"remove {listed} from the call (the defaults are "
                    "accepted), or use layout='RegularGrid' to apply "
                    f"{'them' if len(changed_grid_options) > 1 else 'it'}"
                ),
            )
        )

    # Mask options act only while active bins are inferred from the samples.
    changed_mask_options = _changed(options, defaults, _MASK_OPTIONS)
    if not infer_active_bins and changed_mask_options:
        listed = ", ".join(changed_mask_options)
        raise ValueError(
            _format_error(
                f"{factory}(infer_active_bins=False) cannot apply {listed}.",
                why=(
                    "Why: these options decide which bins the samples make "
                    "active. With infer_active_bins=False every bin in the "
                    f"extent is active, so the "
                    f"{'values' if len(changed_mask_options) > 1 else 'value'} "
                    "would be ignored."
                ),
                fix=(
                    f"remove {listed} from the call (the defaults are "
                    "accepted), or set infer_active_bins=True to select bins "
                    "from the samples"
                ),
            )
        )

    # The layout records every option it is given as env.layout_parameters,
    # so pass an option the bins do not use at its no-op value: the record
    # then describes the bins that were built (open_field's fill_holes=True
    # did not fill anything when every bin is active).
    if not infer_active_bins:
        options.update({key: _SAMPLE_OPTION_DEFAULTS[key] for key in _MASK_OPTIONS})
    options["connect_diagonal_neighbors"] = diagonal_choice(
        n_dims, options["connect_diagonal_neighbors"]
    )

    # Build the dict of layout parameters
    # Common parameters for all layouts
    common_params = {
        "positions": positions,
        "infer_active_bins": infer_active_bins,
        "bin_count_threshold": options["bin_count_threshold"],
    }

    # bin_size goes to the option that sets the engine's cell size, the
    # hexagon width for a hexagonal tiling.
    size_option = _size_parameter(layout_type)
    assert size_option is not None  # both sampled layouts declare one
    layout_params = {**common_params, size_option: bin_size}
    if layout_type is LayoutType.REGULAR_GRID:
        layout_params.update({name: options[name] for name in _REGULAR_GRID_ONLY})
    if arena is not None:
        # Normalizes a registered provider and refuses anything else, such as
        # a bare polygon, before its outline is read.
        arena = own_arena(arena)
    if (
        dimension_ranges is None
        and arena is not None
        and arena.domain_known
        and arena.n_dims == n_dims
    ):
        # Tile the declared arena, not the samples' extent, so the cell edges
        # start at the arena's edge and walls on that lattice fall on faces.
        domain = arena.polygonal_domain()
        if domain is not None:
            low_x, low_y, high_x, high_y = domain.bounds
            dimension_ranges = ((low_x, high_x), (low_y, high_y))
    if dimension_ranges is not None:
        layout_params["dimension_ranges"] = dimension_ranges

    finite_rows = np.all(np.isfinite(positions), axis=1)
    finite_positions = positions if np.all(finite_rows) else positions[finite_rows]
    try:
        sizes = np.asarray(bin_size, dtype=float)
    except (TypeError, ValueError):
        sizes = None  # Preserve the layout's existing invalid-value error.
    if (
        sizes is not None
        and np.all(np.isfinite(sizes))
        and np.all(sizes > 0)
        and len(finite_positions)
        and (sizes.ndim == 0 or sizes.shape == (n_dims,))
    ):
        extent = np.ptp(finite_positions, axis=0)
        if np.any(extent > 0) and np.all(sizes >= extent):
            suggested = float(np.max(extent)) / 50.0
            warnings.warn(
                _format_error(
                    f"bin_size={bin_size} is at least the per-axis data extent {extent.tolist()} in every dimension.",
                    why="Why: this produces very few bins and may indicate a units mismatch.",
                    fix=f"use bin_size={suggested:g} (in the same units as positions), or choose a size below the data extent",
                ),
                UserWarning,
                stacklevel=_external_stacklevel(),
            )
    return _build_layout(
        cls,
        layout_type,
        layout_params,
        name=name,
        units=units,
        frame=frame,
        regions=regions,
        arena=arena,
        wall_fit=wall_fit,
    )


def _track_points(
    values: Any,
    argument: str,
    factory: str = "linear_track",
    labels: Sequence[Any] | None = None,
    example: str | None = None,
) -> list[tuple[float, ...]]:
    """Read track waypoints as coordinate tuples, refusing bare numbers.

    ``labels`` names the points of a mapping (``maze(node_positions=...)``),
    so the suggested value is a mapping too. ``example`` replaces the
    suggestion of points along x, for tracks that cannot lie on one line.
    """
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is not None and array.ndim == 2 and array.shape[1] >= 2:
        return [tuple(float(c) for c in row) for row in array]
    flat = np.ravel(array) if array is not None and array.ndim == 1 else None
    fix: str | None = None
    if example is not None:
        fix = f"pass {argument}={example}, in the units of your positions"
    elif flat is not None and flat.size and labels is not None:
        example = (
            "{"
            + ", ".join(
                f"{label!r}: ({float(v):g}, 0.0)"
                for label, v in zip(labels, flat, strict=True)
            )
            + "}"
        )
    elif flat is not None and flat.size:
        example = "[" + ", ".join(f"({float(v):g}, 0.0)" for v in flat) + "]"
    else:
        example = "[(0.0, 0.0), (200.0, 0.0)]"
    if fix is None:
        fix = (
            f"pass {argument}={example} for a track along x, or the (x, y) "
            "positions of the track's ends and corners"
        )
    raise ValueError(
        _format_error(
            f"{factory}({argument}=...) takes (x, y) points, got {values!r}.",
            why="Why: the track is drawn through these positions in the arena, "
            "so each point needs a coordinate for every axis, in the units of "
            "your positions.",
            fix=fix,
        )
    )


def _require_node_positions(graph: nx.Graph) -> None:
    """Refuse a track graph whose nodes have no position."""
    missing = [node for node in graph.nodes if "pos" not in graph.nodes[node]]
    if missing:
        raise ValueError(
            _format_error(
                f"{len(missing)} of {graph.number_of_nodes()} track-graph node(s) "
                f"have no 'pos' attribute; the first is node {missing[0]!r}.",
                why="Why: bins are laid out along each edge between the "
                "positions of its two nodes, in the units of your positions.",
                fix="give every node its (x, y) position, for example "
                f"graph.add_node({missing[0]!r}, pos=(x, y)) or "
                "nx.set_node_attributes(graph, {node: (x, y), ...}, name='pos')",
            )
        )


def _linear_track(
    cls: type[Environment[Any]],
    *,
    endpoints: Sequence[Sequence[float]] | None = None,
    node_positions: Sequence[Sequence[float]] | None = None,
    bin_size: float,
    edge_spacing: float | Sequence[float] = 0.0,
    track_width: float | None = None,
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
    regions: Regions | None = None,
) -> Environment:
    """Implement Environment.linear_track using explicit construction inputs."""
    if endpoints is not None and node_positions is not None:
        raise ValueError(
            "Provide exactly one of `endpoints` or `node_positions`, not both."
        )

    if endpoints is not None:
        points = _track_points(endpoints, "endpoints")
        if len(points) != 2:
            raise ValueError(
                _format_error(
                    f"linear_track(endpoints=...) takes the two ends of a "
                    f"straight track, got {len(points)} point(s).",
                    why="Why: endpoints draws one straight segment; a track "
                    "with corners is given by its waypoints.",
                    fix=(
                        f"pass node_positions={endpoints!r} for a track through "
                        "every point, in order"
                        if len(points) > 2
                        else "pass both ends, such as endpoints=[(0.0, 0.0), "
                        "(200.0, 0.0)]"
                    ),
                )
            )
    elif node_positions is not None:
        points = _track_points(node_positions, "node_positions")
        if len(points) < 2:
            raise ValueError(
                "`node_positions` must contain at least two (x, y) waypoints "
                f"for a piecewise-linear track. Got {len(points)} point(s)."
            )
    else:
        raise ValueError(
            "linear_track requires an explicit topology: pass `endpoints` "
            "(two (x, y) points for a straight track) or `node_positions` "
            "(a sequence of (x, y) waypoints for a piecewise-linear track). "
            "Raw positions cannot be used to infer a 1D track -- a point "
            "cloud is indistinguishable from an open arena. For an open "
            "arena use Environment.open_field(positions, bin_size)."
        )

    graph = nx.Graph()
    for i, pos in enumerate(points):
        graph.add_node(i, pos=pos)
    edge_order: list[tuple[Any, Any]] = []
    for i in range(len(points) - 1):
        add_edge_with_distance(graph, i, i + 1)
        edge_order.append((i, i + 1))

    return _from_graph(
        cls,
        graph,
        edge_order,
        bin_size=bin_size,
        edge_spacing=edge_spacing,
        name=name,
        units=units,
        frame=frame,
        regions=regions,
        track_width=track_width,
    )


def _maze(
    cls: type[Environment[Any]],
    *,
    kind: str,
    track_graph: nx.Graph | None = None,
    node_positions: dict[Any, Sequence[float]] | None = None,
    bin_size: float,
    edge_spacing: float | Sequence[float] = 0.0,
    track_width: float | None = None,
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
    regions: Regions | None = None,
) -> Environment:
    """Implement Environment.maze using explicit construction inputs."""
    allowed = ("w", "plus", "t")
    kind_normalized = kind.lower() if isinstance(kind, str) else kind
    if kind_normalized not in allowed:
        raise ValueError(
            _format_error(
                f"Unknown maze kind {kind!r}. `kind` must be one of {allowed} (W maze, plus/cross maze, or T maze).",
                fix="pass kind='w', kind='plus', or kind='t' with the documented node_positions order",
                why="Why: maze topology depends on a supported maze kind.",
            )
        )

    if track_graph is not None and node_positions is not None:
        raise ValueError(
            "Provide exactly one of `track_graph` or `node_positions`, not both."
        )

    if track_graph is not None:
        # `kind` is validated above but cannot be reconciled with an
        # arbitrary supplied graph; the graph topology is authoritative.
        warnings.warn(
            f"maze(kind={kind!r}, track_graph=...): `kind` is not consulted "
            "when an explicit `track_graph` is supplied; the supplied "
            "`track_graph` is authoritative for the maze topology.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
        _require_node_positions(track_graph)
        # Operate on a copy so the caller's graph is never mutated.
        graph = track_graph.copy()
        edge_order = list(graph.edges())
        # Fill in `distance` from node positions where missing.
        # `GraphLayout.build` numbers `edge_id` by `graph.edges()` order;
        # any `edge_id` on the input graph is ignored.
        for u, v in graph.edges():
            if "distance" not in graph.edges[u, v]:
                add_edge_with_distance(graph, u, v)
    elif node_positions is not None:
        coordinates = _track_points(
            list(node_positions.values()),
            "node_positions",
            factory="maze",
            labels=list(node_positions),
            example=example_node_positions(kind_normalized, list(node_positions)),
        )
        node_positions = dict(zip(node_positions, coordinates, strict=True))
        graph, edge_order = maze_track_graph(kind_normalized, node_positions)
    else:
        raise ValueError(
            f"maze({kind!r}) requires an explicit topology: pass a ready "
            "`track_graph` (a networkx.Graph with `pos` node attributes) or "
            "`node_positions` (a mapping of node label -> (x, y) coordinate) "
            "from which the standard topology is assembled. Raw positions "
            "cannot be used to infer maze topology."
        )

    return _from_graph(
        cls,
        graph,
        edge_order,
        bin_size=bin_size,
        edge_spacing=edge_spacing,
        name=name,
        units=units,
        frame=frame,
        regions=regions,
        track_width=track_width,
    )


def _from_graph(
    cls: type[Environment[Any]],
    graph: nx.Graph,
    edge_order: list[tuple[Any, Any]],
    *,
    bin_size: float,
    edge_spacing: float | Sequence[float] = 0.0,
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
    regions: Regions | None = None,
    track_width: float | None = None,
) -> Environment:
    """Implement Environment.from_graph using explicit construction inputs."""
    _require_node_positions(graph)
    # A gap sits between consecutive edges, so a one-edge track has none.
    n_gaps = max(0, len(edge_order) - 1)
    scalar = isinstance(edge_spacing, (int, float, np.number))
    if (scalar and n_gaps == 0 and edge_spacing != 0) or (
        not scalar and np.size(cast("Any", edge_spacing)) != n_gaps
    ):
        raise ValueError(
            _format_error(
                f"edge_spacing={edge_spacing!r} does not fit a track with "
                f"{len(edge_order)} edge(s), which has {n_gaps} gap(s) between "
                "consecutive edges.",
                why="Why: edge_spacing is the linear distance inserted between "
                "consecutive edges of edge_order: one number for every gap, or "
                "one value per gap.",
                fix=(
                    "remove edge_spacing from the call (0.0 is accepted); a gap "
                    "applies only to a track of two or more edges"
                    if n_gaps == 0
                    else f"pass one number, such as edge_spacing=5.0, or {n_gaps} "
                    f"values, such as edge_spacing={[5.0] * n_gaps}"
                ),
            )
        )
    layout_params = {
        "graph_definition": graph,
        "edge_order": edge_order,
        "edge_spacing": edge_spacing,
        "bin_size": bin_size,
    }

    # The track geometry refuses a zero-length segment too, without naming it.
    refuse_coincident_nodes(graph, graph.edges)
    segments = np.array(
        [[graph.nodes[u]["pos"], graph.nodes[v]["pos"]] for u, v in graph.edges],
        dtype=np.float64,
    )
    return _build_layout(
        cls,
        "Graph",
        layout_params,
        name=name,
        units=units,
        frame=frame,
        regions=regions,
        arena=TrackArena(segments, width=track_width),
    )


def _from_polygon(
    cls: type[Environment[Any]],
    polygon: shapely.Polygon,
    *,
    bin_size: float | Sequence[float],
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
    regions: Regions | None = None,
    arena: Arena | None = None,
    wall_fit: Literal["snap", "split"] = "snap",
    connect_diagonal_neighbors: bool = True,
) -> Environment:
    """Implement Environment.from_polygon using explicit construction inputs."""
    if not isinstance(polygon, shapely.geometry.Polygon):
        raise TypeError(
            _format_error(
                f"from_polygon() takes a shapely Polygon, got "
                f"{type(polygon).__name__}.",
                why="Why: the polygon is the arena outline the grid is clipped "
                "to, with its holes, so it must be a geometry, not a list of "
                "points.",
                fix="wrap the corner points: from shapely.geometry import "
                f"Polygon; Environment.from_polygon(Polygon(polygon), "
                f"bin_size={bin_size!r})",
            )
        )
    layout_params = {
        "polygon": polygon,
        "bin_size": bin_size,
        "connect_diagonal_neighbors": connect_diagonal_neighbors,
    }
    return _build_layout(
        cls,
        "ShapelyPolygon",
        layout_params,
        name=name,
        units=units,
        frame=frame,
        regions=regions,
        arena=arena,
        wall_fit=wall_fit,
    )


def _from_grid_mask(
    cls: type[Environment[Any]],
    active_mask: NDArray[np.bool_],
    grid_edges: tuple[NDArray[np.float64], ...],
    *,
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
    regions: Regions | None = None,
    arena: Arena | None = None,
    wall_fit: Literal["snap", "split"] = "snap",
    connect_diagonal_neighbors: bool = True,
) -> Environment:
    """Implement Environment.from_grid_mask using explicit construction inputs."""
    cells = tuple(len(edges) - 1 for edges in grid_edges)
    if np.shape(active_mask) != cells:
        raise ValueError(
            _format_error(
                f"from_grid_mask() got an active_mask of shape "
                f"{np.shape(active_mask)}, but grid_edges describe "
                f"{cells} cells.",
                why="Why: grid_edges holds the cell boundaries along each axis, "
                "so each axis needs one more edge than the mask has cells.",
                fix="give each axis one more edge than cells, for example "
                "grid_edges=tuple(np.linspace(edges[0], edges[-1], n + 1) for "
                "edges, n in zip(grid_edges, active_mask.shape))",
            )
        )
    layout_params = {
        "active_mask": active_mask,
        "grid_edges": grid_edges,
        "connect_diagonal_neighbors": diagonal_choice(
            np.ndim(active_mask), connect_diagonal_neighbors
        ),
    }

    return _build_layout(
        cls,
        "MaskedGrid",
        layout_params,
        name=name,
        units=units,
        frame=frame,
        regions=regions,
        arena=arena,
        wall_fit=wall_fit,
    )


def _from_pixel_mask(
    cls: type[Environment[Any]],
    image_mask: NDArray[np.bool_],
    *,
    pixel_size: float | tuple[float, float],
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
    regions: Regions | None = None,
    arena: Arena | None = None,
    wall_fit: Literal["snap", "split"] = "snap",
    connect_diagonal_neighbors: bool = True,
) -> Environment:
    """Implement Environment.from_pixel_mask using explicit construction inputs."""
    layout_params = {
        "image_mask": image_mask,
        "pixel_size": pixel_size,
        "connect_diagonal_neighbors": connect_diagonal_neighbors,
    }

    return _build_layout(
        cls,
        "ImageMask",
        layout_params,
        name=name,
        units=units,
        frame=frame,
        regions=regions,
        arena=arena,
        wall_fit=wall_fit,
    )


def _build_layout(
    cls: type[Environment[Any]],
    kind: LayoutType | str,
    layout_params: dict[str, Any],
    *,
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
    regions: Regions | None = None,
    arena: Arena | None = None,
    wall_fit: Literal["snap", "split"] = "snap",
) -> Environment:
    """Build a named native layout from its parameters and publish it."""
    layout_params = _owned_copy(layout_params)
    if arena is not None:
        arena = own_arena(arena)
    metadata = _declared_metadata(name=name, units=units, frame=frame)
    layout_instance = create_layout(kind=kind, **layout_params)
    return _publication.publish_layout(
        cls,
        layout_instance,
        metadata=metadata,
        regions=regions,
        arena=arena,
        wall_fit=wall_fit,
    )


def _from_layout(
    cls: type[Environment[Any]],
    layout: LayoutEngine,
    *,
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
    regions: Regions | None = None,
    arena: Arena | None = None,
    wall_fit: Literal["snap", "split"] = "snap",
) -> Environment:
    """Implement Environment.from_layout: publish a built layout engine."""
    if isinstance(layout, (str, type)):
        given = repr(layout) if isinstance(layout, str) else layout.__name__
        raise TypeError(
            _format_error(
                f"Environment.from_layout() takes a built layout engine, got {given}.",
                why=(
                    "Why: a layout name or class carries no bins; the engine "
                    "must be built with its parameters first."
                ),
                fix=(
                    "build the engine and pass it, e.g. "
                    "Environment.from_layout(create_layout('TriangularMesh', "
                    "boundary_polygon=polygon, point_spacing=2.0)) with "
                    "create_layout from neurospatial.layout, or call build(...) "
                    "on your own engine first. For grids, polygons, masks and "
                    "tracks use from_samples, from_polygon, from_grid_mask, "
                    "from_pixel_mask or from_graph"
                ),
            )
        )
    if arena is not None:
        arena = own_arena(arena)
    return _publication.publish_layout(
        cls,
        layout,
        metadata=_declared_metadata(name=name, units=units, frame=frame),
        regions=regions,
        arena=arena,
        wall_fit=wall_fit,
    )


def _from_nwb(
    cls: type[Environment[Any]],
    nwbfile: Any,
    *,
    name: str,
) -> Environment:
    """Implement Environment.from_nwb: read a stored environment record."""
    # Lazy import keeps pynwb optional. The reader raises the "pynwb is
    # required" ImportError itself when pynwb is missing; any other
    # ImportError propagates unchanged.
    from neurospatial.io.nwb import read_environment

    return read_environment(nwbfile, name=name)
