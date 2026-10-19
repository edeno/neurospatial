"""Draw environments and bin-valued fields on the GPU with fastplotlib.

``Environment.plot``, ``Environment.plot_field`` and
``Environment.animate_fields`` use this module with ``backend="fastplotlib"``.
Each bin is drawn as its own cell, so a field shows exactly the bins the
environment holds: rectangles for grids, hexagons, triangles or polygons for
the other layouts, and a point sized like a bin for tracks, points and 3-D
environments. Colors come from a lookup table whose first entry is the
missing-value color; GPU colormaps would otherwise draw NaN in the top color,
the same as the highest firing rate.

Inside Jupyter, fastplotlib renders on the machine running the kernel and
streams the frames to the browser, so a GPU server works from a laptop.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry import HexagonShapes, PolygonCells, TriangleShapes

if TYPE_CHECKING:
    import fastplotlib
    import shapely
    from matplotlib.colors import Colormap

    from neurospatial.environment import Environment

_INSTALL = 'pip install "neurospatial[fastplotlib]"'
_N_COLORS = 255
_WALL_COLOR = "firebrick"
_LANDMARK_COLOR = "orange"
_REGION_COLOR = "gold"


def import_fastplotlib() -> Any:
    """Import fastplotlib, or say how to install it.

    Raises
    ------
    ImportError
        If fastplotlib is not installed; it needs Python 3.11 or newer.
    """
    try:
        import fastplotlib
    except ImportError as error:
        raise ImportError(
            _format_error(
                "backend='fastplotlib' needs the fastplotlib package "
                f"(Python 3.11 or newer; this is {sys.version.split()[0]}).",
                why="Why: fastplotlib is an optional dependency for GPU drawing.",
                fix=f"install it with {_INSTALL}, or use backend='matplotlib'",
            )
        ) from error
    return fastplotlib


def refuse_matplotlib_options(method: str, **passed: Any) -> None:
    """Refuse options that only the Matplotlib backend has.

    Parameters
    ----------
    method : str
        The plotting method, for the message.
    **passed
        Option values; None, False and empty values count as not passed.

    Raises
    ------
    TypeError
        If any option was passed.
    """
    given = sorted(name for name, value in passed.items() if value)
    if given:
        raise TypeError(
            _format_error(
                f"{method}(backend='fastplotlib') got Matplotlib-only "
                f"options: {', '.join(given)}.",
                why="Why: they configure Matplotlib axes and artists, which a "
                "fastplotlib figure does not have.",
                fix="remove them, or use backend='matplotlib'",
            )
        )


# --- colors -------------------------------------------------------------------


@dataclass(frozen=True)
class ColorScale:
    """Map bin values to RGBA colors, with a separate missing-value color.

    Parameters
    ----------
    cmap : str or matplotlib Colormap
        Colormap for finite values.
    vmin, vmax : float
        Values mapped to the colormap's ends; values outside are clipped.
    nan_color : str or None
        Color of NaN bins; None leaves them transparent.
    """

    cmap: str | Colormap
    vmin: float
    vmax: float
    nan_color: str | None

    def table(self) -> NDArray[np.float32]:
        """The colormap sampled at ``_N_COLORS`` points, shape (n_colors, 4)."""
        import matplotlib.pyplot as plt

        colormap = plt.get_cmap(self.cmap)
        return colormap(np.linspace(0.0, 1.0, _N_COLORS)).astype(np.float32)

    def missing(self) -> NDArray[np.float32]:
        """The RGBA color of NaN bins."""
        from matplotlib.colors import to_rgba

        if self.nan_color is None:
            return np.zeros(4, dtype=np.float32)
        return np.asarray(to_rgba(self.nan_color), dtype=np.float32)

    def scaled(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """``values`` mapped onto [0, 1] by the limits; NaN stays NaN."""
        scaled = (np.asarray(values, dtype=np.float64) - self.vmin) / (
            self.vmax - self.vmin
        )
        return np.clip(scaled, 0.0, 1.0)

    def colors(
        self,
        values: NDArray[np.float64],
        table: NDArray[np.float32],
        missing: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """RGBA colors of ``values``, shape (n_values, 4)."""
        scaled = self.scaled(values)
        finite = np.isfinite(scaled)
        index = np.rint(np.where(finite, scaled, 0.0) * (_N_COLORS - 1))
        colors: NDArray[np.float32] = table[index.astype(np.intp)]
        colors[~finite] = missing
        return colors


def mesh_colors(rgba: NDArray[np.float32]) -> NDArray[np.float32]:
    """sRGB colors as a mesh's per-vertex colors, which pygfx reads as linear.

    Strings and scatter or line colors are read as sRGB; per-vertex mesh color
    arrays are not, so without this a mesh draws every color too light.
    """
    linear = np.array(rgba, dtype=np.float32)
    srgb = linear[..., :3]
    linear[..., :3] = np.where(
        srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4
    )
    return linear


def color_limits(
    values: NDArray[np.float64], vmin: float | None, vmax: float | None
) -> tuple[float, float]:
    """Color limits from the finite values, as ``plot_field`` chooses them."""
    finite = values[np.isfinite(values)]
    low = float(finite.min()) if vmin is None and finite.size else vmin
    high = float(finite.max()) if vmax is None and finite.size else vmax
    low = 0.0 if low is None else float(low)
    high = 1.0 if high is None else float(high)
    if low >= high:
        high = low + 1.0
    return low, high


# --- cells ----------------------------------------------------------------------


@dataclass(frozen=True)
class Cells:
    """Bins as a triangle mesh: the vertices of every cell, and their bins.

    Attributes
    ----------
    positions : ndarray of float32, shape (n_vertices, 3)
        Cell vertices, with z = 0.
    triangles : ndarray of uint32, shape (n_triangles, 3)
        Vertex indices of each triangle.
    vertex_bin : ndarray of intp, shape (n_vertices,)
        The bin each vertex belongs to; a cell's vertices share its color.
    """

    positions: NDArray[np.float32]
    triangles: NDArray[np.uint32]
    vertex_bin: NDArray[np.intp]


def _fans(
    corners: NDArray[np.float64], n_corners: int
) -> tuple[NDArray[np.float32], NDArray[np.uint32], NDArray[np.intp]]:
    """Mesh arrays for cells that are convex polygons with ``n_corners`` each."""
    n_cells = len(corners) // n_corners
    base = (np.arange(n_cells) * n_corners)[:, None, None]
    fan = np.array([[0, k, k + 1] for k in range(1, n_corners - 1)])
    positions = np.zeros((len(corners), 3), dtype=np.float32)
    positions[:, :2] = corners
    triangles = (base + fan[None]).reshape(-1, 3).astype(np.uint32)
    return positions, triangles, np.repeat(np.arange(n_cells), n_corners)


def cell_mesh(env: Environment) -> Cells | None:
    """The bins of a 2-D environment as cells, or None to draw points.

    Grids, hexagons, triangles and polygon cells have exact cells; tracks,
    bins without cell shapes and 3-D environments are drawn as points.
    """
    binning = env._binning
    if env.n_dims != 2 or binning.linearization is not None:
        return None
    shapes = binning.shapes
    if isinstance(shapes, HexagonShapes):
        angles = np.pi / 2 + shapes.orientation + np.arange(6) * np.pi / 3
        offsets = shapes.radius * np.column_stack((np.cos(angles), np.sin(angles)))
        corners = (shapes.centers[:, None, :] + offsets[None]).reshape(-1, 2)
        return Cells(*_fans(corners, 6))
    if isinstance(shapes, TriangleShapes):
        corners = shapes.points[shapes.triangles].reshape(-1, 2)
        return Cells(*_fans(corners, 3))
    if isinstance(shapes, PolygonCells):
        return _polygon_cells(shapes)
    frame = binning.grid_frame
    grid = frame.source if frame is not None else binning.rectilinear
    if grid is None:
        return None
    x_edges, y_edges = grid.edges
    i, j = np.nonzero(grid.active_mask)  # C order: the order of the bins
    corners = np.stack(
        [
            np.column_stack((x_edges[i], y_edges[j])),
            np.column_stack((x_edges[i + 1], y_edges[j])),
            np.column_stack((x_edges[i + 1], y_edges[j + 1])),
            np.column_stack((x_edges[i], y_edges[j + 1])),
        ],
        axis=1,
    ).reshape(-1, 2)
    if frame is not None:
        corners = frame.to_world(corners)
    return Cells(*_fans(corners, 4))


def _polygon_cells(shapes: PolygonCells) -> Cells:
    """Triangulate exact polygon cells, holes included."""
    import shapely

    triangles_per_cell = [
        shapely.get_parts(shapely.constrained_delaunay_triangles(polygon))
        for polygon in shapes.polygons
    ]
    counts = np.array([len(parts) for parts in triangles_per_cell])
    corners = (
        shapely.get_coordinates(np.concatenate(triangles_per_cell))
        .reshape(-1, 4, 2)[:, :3]
        .reshape(-1, 2)
    )
    positions, triangles, _ = _fans(corners, 3)
    return Cells(positions, triangles, np.repeat(np.arange(len(counts)), counts * 3))


def bin_spacing(env: Environment) -> float:
    """Typical distance between neighboring bin centers, for point sizes."""
    from scipy.spatial import cKDTree

    centers = env.bin_centers
    if len(centers) < 2:
        return 1.0
    sample = centers[:: max(1, len(centers) // 1000)]
    distances, _ = cKDTree(centers).query(sample, k=2)
    spacing = float(np.median(distances[:, 1]))
    return spacing if spacing > 0 else 1.0


def _points(env: Environment) -> NDArray[np.float32]:
    """Bin centers as (n_bins, 3) points; 1-D bins lie on y = 0."""
    centers = np.zeros((env.n_bins, 3), dtype=np.float32)
    centers[:, : env.n_dims] = env.bin_centers
    return centers


# --- graphics -------------------------------------------------------------------


class FieldGraphic:
    """One value per bin, drawn as colored cells or points; ``update`` recolors.

    In three dimensions a point's opacity follows its value, so the bins with
    the highest values show through the volume; missing bins are faint.

    Parameters
    ----------
    subplot : fastplotlib subplot
        Where to draw.
    env : Environment
        The bins.
    values : ndarray of float64, shape (n_bins,)
        The first values to show.
    scale : ColorScale
        How values become colors.
    """

    def __init__(
        self,
        subplot: Any,
        env: Environment,
        values: NDArray[np.float64],
        scale: ColorScale,
    ) -> None:
        self.scale = scale
        self._table = scale.table()
        self._missing = scale.missing()
        self._cells = cell_mesh(env)
        self._volume = env.n_dims == 3
        colors = self._colors(values)
        alpha_mode = "blend" if scale.nan_color is None else "solid"
        if self._volume:
            # Order-independent transparency: overlapping points need no sort.
            alpha_mode = "weighted_blend"
        if self._cells is not None:
            self.graphic = subplot.add_mesh(
                self._cells.positions,
                self._cells.triangles,
                mode="basic",
                colors=colors,
                alpha_mode=alpha_mode,
            )
        else:
            self.graphic = subplot.add_scatter(
                _points(env),
                colors=colors,
                sizes=0.9 * bin_spacing(env),
                size_space="world",
                markers="square" if env.n_dims < 3 else "circle",
                edge_width=0,
                alpha_mode=alpha_mode,
            )

    def _colors(self, values: NDArray[np.float64]) -> NDArray[np.float32]:
        colors = self.scale.colors(values, self._table, self._missing)
        if self._volume:
            scaled = self.scale.scaled(values)
            colors[:, 3] = np.where(np.isfinite(scaled), 0.03 + 0.97 * scaled, 0.05)
        if self._cells is None:
            return colors
        return mesh_colors(colors)[self._cells.vertex_bin]

    def update(self, values: NDArray[np.float64]) -> None:
        """Recolor the bins with new values."""
        self.graphic.colors = self._colors(values)


def _segments(lines: Sequence[NDArray[np.float64]]) -> NDArray[np.float32]:
    """Several polylines as one, separated by NaN rows, shape (n, 3)."""
    pieces: list[NDArray[np.float64]] = []
    for line in lines:
        pieces.extend([np.asarray(line, dtype=np.float64), np.full((1, 2), np.nan)])
    joined = np.concatenate(pieces) if pieces else np.full((1, 2), np.nan)
    points = np.zeros((len(joined), 3), dtype=np.float32)
    points[:, :2] = joined[:, :2]
    return points


def _label(subplot: Any, text: str, at: ArrayLike, color: str) -> None:
    """A name next to a point, in screen-sized text."""
    at = np.asarray(at, dtype=np.float64)
    subplot.add_text(
        text,
        font_size=13,
        face_color=color,
        offset=(float(at[0]), float(at[1]), 0.0),
        anchor="bottom-left",
    )


def _area(subplot: Any, polygon: Any, color: str, alpha: float) -> None:
    """A translucent polygon with its outline."""
    import shapely

    triangles = shapely.get_coordinates(
        shapely.get_parts(shapely.constrained_delaunay_triangles(polygon))
    ).reshape(-1, 4, 2)[:, :3]
    positions, faces, _ = _fans(triangles.reshape(-1, 2), 3)
    subplot.add_mesh(positions, faces, mode="basic", colors=color, alpha=alpha)
    rings = [np.asarray(polygon.exterior.coords)]
    rings += [np.asarray(ring.coords) for ring in polygon.interiors]
    subplot.add_line(_segments(rings), colors=color, thickness=2)


def draw_walls(subplot: Any, env: Environment) -> None:
    """Declared walls as thick lines, each named once at its middle."""
    arena = env.arena
    if not arena.walls:
        return
    subplot.add_line(
        _segments([np.asarray(wall.coords) for wall in arena.walls]),
        colors=_WALL_COLOR,
        thickness=5,
    )
    named: set[str] = set()
    for name, wall in zip(arena.wall_names, arena.walls, strict=True):
        if name not in named:
            named.add(name)
            _label(
                subplot,
                name,
                wall.interpolate(0.5, normalized=True).coords[0],
                _WALL_COLOR,
            )


def draw_landmarks(subplot: Any, env: Environment) -> None:
    """Each landmark by its kind, with its name."""
    for landmark in env.landmarks.values():
        data = landmark.data
        if isinstance(data, np.ndarray):
            point = np.zeros((1, 3), dtype=np.float32)
            point[0, :2] = data[:2]
            subplot.add_scatter(
                point, colors=_LANDMARK_COLOR, sizes=16, markers="diamond"
            )
        elif data.geom_type == "LineString":
            subplot.add_line(
                _segments([np.asarray(data.coords)]),
                colors=_LANDMARK_COLOR,
                thickness=6,
            )
        else:
            _area(subplot, data, _LANDMARK_COLOR, alpha=0.4)
        _label(subplot, landmark.name, landmark.position[:2], _LANDMARK_COLOR)


def draw_regions(
    subplot: Any, env: Environment, names: Sequence[str] | None, alpha: float
) -> None:
    """Regions as translucent areas or points, with their names."""
    for name, region in env.regions.items():
        if names is not None and name not in names:
            continue
        if region.kind == "point":
            point = np.zeros((1, 3), dtype=np.float32)
            point[0, :2] = np.asarray(region.data)[:2]
            subplot.add_scatter(point, colors=_REGION_COLOR, sizes=12)
            _label(subplot, name, point[0], _REGION_COLOR)
        else:
            area = cast("shapely.Polygon | shapely.MultiPolygon", region.data)
            for polygon in getattr(area, "geoms", [area]):
                _area(subplot, polygon, _REGION_COLOR, alpha)
            _label(subplot, name, area.representative_point().coords[0], _REGION_COLOR)


def draw_connections(subplot: Any, env: Environment) -> None:
    """Every open connection between bins, as one line graphic."""
    pairs = np.asarray(env.edge_pairs)
    if len(pairs) == 0:
        return
    centers = _points(env)
    segments = np.full((len(pairs) * 3, 3), np.nan, dtype=np.float32)
    segments[0::3] = centers[pairs[:, 0]]
    segments[1::3] = centers[pairs[:, 1]]
    subplot.add_line(segments, colors="#555555", thickness=1, alpha=0.6)


def draw_layers(
    subplot: Any,
    env: Environment,
    *,
    show_walls: bool,
    show_landmarks: bool,
    show_regions: bool | Sequence[str],
    region_alpha: float = 0.3,
) -> None:
    """Walls, landmarks and regions over the bins; 3-D environments have none."""
    if env.n_dims != 2:
        return
    if show_regions:
        names = None if show_regions is True else list(show_regions)
        draw_regions(subplot, env, names, region_alpha)
    if show_walls:
        draw_walls(subplot, env)
    if show_landmarks:
        draw_landmarks(subplot, env)


def draw_colorbar(subplot: Any, scale: ColorScale, label: str) -> None:
    """A vertical color scale with its limits and label."""
    table = scale.table()
    rows = np.arange(_N_COLORS + 1, dtype=np.float64)
    corners = np.stack(
        [
            np.column_stack((np.zeros(_N_COLORS), rows[:-1])),
            np.column_stack((np.ones(_N_COLORS), rows[:-1])),
            np.column_stack((np.ones(_N_COLORS), rows[1:])),
            np.column_stack((np.zeros(_N_COLORS), rows[1:])),
        ],
        axis=1,
    ).reshape(-1, 2)
    positions, triangles, vertex_bin = _fans(corners, 4)
    subplot.add_mesh(
        positions, triangles, mode="basic", colors=mesh_colors(table)[vertex_bin]
    )
    # Limits above and below the bar, where a long number has the bar's width.
    for value, height, anchor in (
        (scale.vmax, 1.02 * _N_COLORS, "bottom-center"),
        (scale.vmin, -0.02 * _N_COLORS, "top-center"),
    ):
        subplot.add_text(
            f"{value:.3g}",
            font_size=12,
            face_color="w",
            offset=(0.5, height, 0.0),
            anchor=anchor,
        )
    if label:
        subplot.add_text(
            label,
            font_size=12,
            face_color="w",
            offset=(0.5, 1.12 * _N_COLORS, 0.0),
            anchor="bottom-center",
        )
    subplot.axes.visible = False
    subplot.camera.maintain_aspect = False
    subplot.camera.show_rect(-0.4, 1.4, -0.12 * _N_COLORS, 1.22 * _N_COLORS)


# --- figures --------------------------------------------------------------------


def new_figure(env: Environment, *, colorbar: bool) -> tuple[Any, Any, Any]:
    """A figure for ``env``, its plot, and a narrow color scale when asked.

    Returns
    -------
    figure, plot, scale
        The figure, the subplot for the bins, and the color-scale subplot or
        None.
    """
    fastplotlib = import_fastplotlib()
    camera = "3d" if env.n_dims == 3 else "2d"
    controller = "orbit" if env.n_dims == 3 else "panzoom"
    if not colorbar:
        figure = fastplotlib.Figure(
            cameras=camera, controller_types=[controller], size=(720, 700)
        )
        return figure, figure[0, 0], None
    figure = fastplotlib.Figure(
        rects=[(0.0, 0.0, 0.83, 1.0), (0.84, 0.1, 0.16, 0.8)],
        cameras=[camera, "2d"],
        controller_types=[controller, "panzoom"],
        size=(820, 700),
    )
    return figure, figure[0], figure[1]


def frame_view(subplot: Any, env: Environment) -> None:
    """Look at a 3-D environment from an angle, z up; 2-D views are unchanged."""
    if env.n_dims == 3:
        subplot.camera.show_object(subplot.scene, view_dir=(-1, -1, -1), up=(0, 0, 1))


def title(env: Environment, subject: str) -> str:
    """A plot title naming the environment, its units and what is shown."""
    name = env.name or env.layout_type
    units = f" ({env.units})" if env.units else ""
    return f"{name}{units}: {subject}" if subject else f"{name}{units}"


def plot_environment(
    env: Environment,
    *,
    show_regions: bool,
    show_walls: bool,
    show_landmarks: bool,
) -> fastplotlib.Figure:
    """The bins, their connections, walls, landmarks and regions.

    Returns
    -------
    fastplotlib.Figure
        Call ``figure.show()`` to display it.
    """
    figure, subplot, _ = new_figure(env, colorbar=False)
    cells = cell_mesh(env)
    if cells is not None:
        subplot.add_mesh(
            cells.positions, cells.triangles, mode="basic", colors="#b9d3d6"
        )
    else:
        subplot.add_scatter(
            _points(env),
            colors="#b9d3d6",
            sizes=0.9 * bin_spacing(env),
            size_space="world",
            markers="square" if env.n_dims < 3 else "circle",
        )
    draw_connections(subplot, env)
    draw_layers(
        subplot,
        env,
        show_walls=show_walls,
        show_landmarks=show_landmarks,
        show_regions=show_regions,
    )
    subplot.title = title(env, "")
    frame_view(subplot, env)
    return figure


def plot_field(
    env: Environment,
    field: NDArray[np.float64],
    *,
    cmap: str | Colormap,
    vmin: float | None,
    vmax: float | None,
    colorbar: bool,
    colorbar_label: str,
    nan_color: str | None,
) -> fastplotlib.Figure:
    """One value per bin, colored over the bins' own cells.

    Returns
    -------
    fastplotlib.Figure
        Call ``figure.show()`` to display it.
    """
    low, high = color_limits(field, vmin, vmax)
    scale = ColorScale(cmap, low, high, nan_color)
    figure, subplot, scale_plot = new_figure(env, colorbar=colorbar)
    FieldGraphic(subplot, env, field, scale)
    draw_layers(subplot, env, show_walls=True, show_landmarks=True, show_regions=False)
    subplot.title = title(env, colorbar_label)
    if scale_plot is not None:
        draw_colorbar(scale_plot, scale, colorbar_label)
    frame_view(subplot, env)
    return figure
