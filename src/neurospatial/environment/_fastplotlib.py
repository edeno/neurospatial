"""Draw environments and bin-valued fields on the GPU with fastplotlib.

``Environment.plot``, ``Environment.plot_field`` and
``Environment.animate_fields`` use this module with ``backend="fastplotlib"``.
Each bin is drawn as its own cell, so a field shows exactly the bins the
environment holds: rectangles for grids, hexagons, triangles or polygons for
the other layouts, and a point sized like a bin for tracks, points and 3-D
environments.

Cells are colored through a lookup texture whose first entry is the
missing-value color: each vertex carries the texture coordinate of its bin's
value, so recoloring uploads one number per vertex and the GPU does the
lookup. pygfx has no color for missing values, and would otherwise draw NaN in
the colormap's top color, the same as the highest firing rate.

Inside Jupyter, fastplotlib renders on the machine running the kernel and
streams the frames to the browser, so a GPU server works from a laptop.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import shapely
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial.environment._cache import CELL_POLYGONS
from neurospatial.environment._lookup import center_tree
from neurospatial.geometry import cell_polygons
from neurospatial.layout import _cell_shape

if TYPE_CHECKING:
    import fastplotlib
    from matplotlib.colors import Colormap

    from neurospatial.environment import Environment

_INSTALL = 'pip install "neurospatial[fastplotlib]"'
_N_COLORS = 255  # colormap entries; the lookup texture adds the missing color
_BIN_COLOR = "#b9d3d6"
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


def refuse_matplotlib_options(
    method: str, kwargs: Mapping[str, Any], **passed: bool
) -> None:
    """Refuse options that only the Matplotlib backend has.

    Parameters
    ----------
    method : str
        The plotting method, for the message.
    kwargs : mapping
        Extra keyword arguments, all of them Matplotlib's.
    **passed
        For each named Matplotlib option, whether it differs from its default.

    Raises
    ------
    TypeError
        If any option was passed.
    """
    given = sorted([name for name, was in passed.items() if was] + list(kwargs))
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
    """Map bin values to colors, with a separate missing-value color.

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

    @cached_property
    def lut(self) -> NDArray[np.float32]:
        """The missing color, then the colormap, shape (1 + n_colors, 4)."""
        from matplotlib import colormaps
        from matplotlib.colors import to_rgba

        colormap = colormaps[self.cmap] if isinstance(self.cmap, str) else self.cmap
        missing = (0.0, 0.0, 0.0, 0.0) if self.nan_color is None else self.nan_color
        colors = colormap(np.linspace(0.0, 1.0, _N_COLORS))
        return np.vstack([to_rgba(missing), colors]).astype(np.float32)

    def scaled(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """``values`` mapped onto [0, 1] by the limits; NaN stays NaN."""
        span = (self.vmax - self.vmin) or 1.0
        values = np.asarray(values, dtype=np.float64)
        return np.clip((values - self.vmin) / span, 0.0, 1.0)

    def index(self, values: NDArray[np.float64]) -> NDArray[np.intp]:
        """Row of ``lut`` for each value: 0 for a missing value."""
        scaled = self.scaled(values)
        finite = np.isfinite(scaled)
        rows = 1 + np.rint(np.where(finite, scaled, 0.0) * (_N_COLORS - 1))
        return np.where(finite, rows, 0).astype(np.intp)

    def texture_coordinates(self, values: NDArray[np.float64]) -> NDArray[np.float32]:
        """Each value's coordinate in the lookup texture: its row's center."""
        return ((self.index(values) + 0.5) / len(self.lut)).astype(np.float32)

    def rgba(self, values: NDArray[np.float64]) -> NDArray[np.float32]:
        """Each value's RGBA color, shape (n_values, 4)."""
        return self.lut[self.index(values)]


# --- geometry -------------------------------------------------------------------


def xyz(points: ArrayLike) -> NDArray[np.float32]:
    """Points with 1 to 3 coordinates as float32 (n, 3), padding with zeros."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    out = np.zeros((len(points), 3), dtype=np.float32)
    out[:, : min(points.shape[1], 3)] = points[:, :3]
    return out


def polylines(lines: Sequence[ArrayLike]) -> NDArray[np.float32]:
    """Several polylines as one line, separated by NaN rows, shape (n, 3)."""
    gap = np.full((1, 3), np.nan, dtype=np.float32)
    pieces = [piece for line in lines for piece in (xyz(line), gap)]
    return np.concatenate(pieces) if pieces else gap


def pair_segments(starts: ArrayLike, ends: ArrayLike) -> NDArray[np.float32]:
    """Line segments from ``starts`` to ``ends``, separated by NaN rows."""
    starts, ends = xyz(starts), xyz(ends)
    segments = np.full((3 * len(starts), 3), np.nan, dtype=np.float32)
    segments[0::3], segments[1::3] = starts, ends
    return segments


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


def _fans(corners: NDArray[np.float64], n_corners: int) -> Cells:
    """Cells that are convex polygons of ``n_corners`` corners each, in order."""
    n_cells = len(corners) // n_corners
    base = (np.arange(n_cells) * n_corners)[:, None, None]
    fan = np.array([[0, k, k + 1] for k in range(1, n_corners - 1)])
    triangles = (base + fan[None]).reshape(-1, 3).astype(np.uint32)
    return Cells(xyz(corners), triangles, np.repeat(np.arange(n_cells), n_corners))


def _rectangles(
    x0: NDArray[np.float64],
    x1: NDArray[np.float64],
    y0: NDArray[np.float64],
    y1: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Corners of rectangles, counterclockwise from (x0, y0), shape (4n, 2)."""
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return np.stack([np.column_stack(c) for c in corners], axis=1).reshape(-1, 2)


def _triangles(polygons: Sequence[Any]) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Triangulate polygons, holes included.

    Returns
    -------
    corners : ndarray of float64, shape (3 * n_triangles, 2)
        The corners of each triangle.
    owner : ndarray of intp, shape (n_triangles,)
        The polygon each triangle belongs to.
    """
    triangles, owner = shapely.get_parts(
        shapely.constrained_delaunay_triangles(np.asarray(polygons, dtype=object)),
        return_index=True,
    )
    # Each triangle is a closed ring of four points; drop the repeated one.
    corners = shapely.get_coordinates(triangles).reshape(-1, 4, 2)[:, :3]
    return corners.reshape(-1, 2), owner


def cell_mesh(env: Environment) -> Cells | None:
    """The bins of a 2-D environment as cells, or None to draw points.

    The cells are the environment's own (:func:`geometry.cell_polygons`, the
    description analyses read), cached with the environment. Grid, hexagon and
    triangle cells are convex, so each is a fan of triangles from one corner;
    clipped or mixed cells are triangulated. Tracks, bins without cell shapes
    and 3-D environments are drawn as points.
    """
    shape = _cell_shape(env._binning)
    if env.n_dims != 2 or shape in ("track", "points"):
        return None
    polygons = env._cache.get(CELL_POLYGONS, lambda: cell_polygons(env._binning))
    geometries = np.empty(len(polygons), dtype=object)
    geometries[:] = polygons
    rings, owner = shapely.get_coordinates(
        shapely.get_exterior_ring(geometries), return_index=True
    )
    counts = np.bincount(owner, minlength=len(polygons))
    if shape == "polygons" or counts.min() != counts.max():
        corners, owner = _triangles(polygons)
        cells = _fans(corners, 3)
        return Cells(cells.positions, cells.triangles, np.repeat(owner, 3))
    # A closed ring repeats its first corner; drop it.
    corners = rings.reshape(len(polygons), counts[0], 2)[:, :-1]
    return _fans(corners.reshape(-1, 2), int(counts[0]) - 1)


def bin_spacing(env: Environment) -> float:
    """Typical distance between neighboring bin centers, for point sizes."""
    centers = env.bin_centers
    if len(centers) < 2:
        return 1.0
    sample = centers[:: max(1, len(centers) // 1000)]
    distances, _ = center_tree(env._cache, env._binning).query(sample, k=2)
    spacing = float(np.median(distances[:, 1]))
    return spacing if spacing > 0 else 1.0


def add_bins(subplot: Any, env: Environment, cells: Cells | None, **style: Any) -> Any:
    """Draw every bin: its cell, or a point the size of a bin.

    ``style`` holds the colors: mesh options for cells, scatter options for
    points.
    """
    if cells is not None:
        return subplot.add_mesh(cells.positions, cells.triangles, mode="basic", **style)
    return subplot.add_scatter(
        xyz(env.bin_centers),
        sizes=0.9 * bin_spacing(env),
        size_space="world",
        markers="square" if env.n_dims < 3 else "circle",
        edge_width=0,
        **style,
    )


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
        self._volume = env.n_dims == 3
        cells = cell_mesh(env)
        # Only the bin of each vertex is needed again; the mesh owns the rest.
        self._vertex_bin = None if cells is None else cells.vertex_bin
        if self._volume:
            alpha_mode = "weighted_blend"  # order-independent: no depth sort
        else:
            alpha_mode = "blend" if scale.nan_color is None else "solid"
        if cells is not None:
            style = {
                "mapcoords": self._coordinates(values),
                "cmap": scale.lut,
                "clim": (0.0, 1.0),
            }
        else:
            style = {"colors": self._colors(values)}
        self.graphic = add_bins(subplot, env, cells, alpha_mode=alpha_mode, **style)

    def _coordinates(self, values: NDArray[np.float64]) -> NDArray[np.float32]:
        assert self._vertex_bin is not None
        return self.scale.texture_coordinates(values)[self._vertex_bin]

    def _colors(self, values: NDArray[np.float64]) -> NDArray[np.float32]:
        colors = self.scale.rgba(values)
        if self._volume:
            scaled = self.scale.scaled(values)
            colors[:, 3] = np.where(np.isfinite(scaled), 0.03 + 0.97 * scaled, 0.05)
        return colors

    def update(self, values: NDArray[np.float64]) -> None:
        """Recolor the bins with new values."""
        if self._vertex_bin is not None:
            self.graphic.mapcoords = self._coordinates(values)
        else:
            self.graphic.colors = self._colors(values)


def _label(subplot: Any, text: str, at: ArrayLike, color: str) -> None:
    """A name next to a point, in screen-sized text."""
    subplot.add_text(
        text,
        font_size=13,
        face_color=color,
        offset=tuple(xyz(at)[0]),
        anchor="bottom-left",
    )


def _area(subplot: Any, polygon: Any, color: str, alpha: float) -> None:
    """A translucent polygon with its outline."""
    corners, _ = _triangles([polygon])
    cells = _fans(corners, 3)
    subplot.add_mesh(
        cells.positions, cells.triangles, mode="basic", colors=color, alpha=alpha
    )
    rings = [polygon.exterior.coords, *(ring.coords for ring in polygon.interiors)]
    subplot.add_line(polylines(rings), colors=color, thickness=2)


def draw_walls(subplot: Any, env: Environment) -> None:
    """Declared walls as thick lines, each named once at its middle."""
    arena = env.arena
    if not arena.walls:
        return
    subplot.add_line(
        polylines([wall.coords for wall in arena.walls]),
        colors=_WALL_COLOR,
        thickness=5,
    )
    named: set[str] = set()
    for name, wall in zip(arena.wall_names, arena.walls, strict=True):
        if name not in named:
            named.add(name)
            middle = wall.interpolate(0.5, normalized=True).coords[0]
            _label(subplot, name, middle, _WALL_COLOR)


def draw_landmarks(subplot: Any, env: Environment) -> None:
    """Each landmark by its kind, with its name."""
    for landmark in env.landmarks.values():
        data = landmark.data
        if isinstance(data, np.ndarray):
            subplot.add_scatter(
                xyz(data), colors=_LANDMARK_COLOR, sizes=16, markers="diamond"
            )
        elif data.geom_type == "LineString":
            subplot.add_line(
                polylines([data.coords]), colors=_LANDMARK_COLOR, thickness=6
            )
        else:
            _area(subplot, data, _LANDMARK_COLOR, alpha=0.4)
        _label(subplot, landmark.name, landmark.position, _LANDMARK_COLOR)


def draw_regions(
    subplot: Any, env: Environment, names: Sequence[str] | None, alpha: float
) -> None:
    """Regions as translucent areas or points, with their names."""
    for name, region in env.regions.items():
        if names is not None and name not in names:
            continue
        if region.kind == "point":
            subplot.add_scatter(xyz(region.data), colors=_REGION_COLOR, sizes=12)
            _label(subplot, name, region.data, _REGION_COLOR)
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
    centers = env.bin_centers
    subplot.add_line(
        pair_segments(centers[pairs[:, 0]], centers[pairs[:, 1]]),
        colors="#555555",
        thickness=1,
        alpha=0.6,
    )


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
    """A vertical color scale, with its limits and label, docked to the right."""
    dock = subplot.docks["right"]
    dock.size = 90
    bar = _fans(_rectangles(*np.array([[0.0], [1.0], [0.0], [1.0]])), 4)
    # The bar runs through the colormap's rows, from the first to the last.
    bottom, top = 1.5 / len(scale.lut), (len(scale.lut) - 0.5) / len(scale.lut)
    dock.add_mesh(
        bar.positions,
        bar.triangles,
        mode="basic",
        mapcoords=np.array([bottom, bottom, top, top], dtype=np.float32),
        cmap=scale.lut,
        clim=(0.0, 1.0),
    )
    # Limits above and below the bar, where a long number has the bar's width.
    texts = [(f"{scale.vmax:.3g}", 1.02, "bottom-center")]
    texts += [(f"{scale.vmin:.3g}", -0.02, "top-center")]
    texts += [(label, 1.12, "bottom-center")] if label else []
    for text, height, anchor in texts:
        dock.add_text(
            text, font_size=12, face_color="w", offset=(0.5, height, 0.0), anchor=anchor
        )
    dock.camera.maintain_aspect = False
    dock.camera.show_rect(-0.4, 1.4, -0.12, 1.22)


# --- figures --------------------------------------------------------------------


def new_figure(env: Environment) -> tuple[Any, Any]:
    """A figure for ``env`` and its one subplot.

    A 3-D environment gets a rotatable 3-D camera; coordinates without a
    common scale, such as polar (distance, angle), are not drawn at equal
    aspect.
    """
    fastplotlib = import_fastplotlib()
    volume = env.n_dims == 3
    figure = fastplotlib.Figure(
        cameras="3d" if volume else "2d",
        controller_types=["orbit" if volume else "panzoom"],
        size=(800, 700),
    )
    subplot = figure[0, 0]
    subplot.camera.maintain_aspect = env._plot_labels().coordinates.equal_aspect
    return figure, subplot


def frame_view(subplot: Any, env: Environment) -> None:
    """Look at a 3-D environment from an angle, z up; 2-D views are unchanged."""
    if env.n_dims == 3:
        subplot.camera.show_object(subplot.scene, view_dir=(-1, -1, -1), up=(0, 0, 1))


def plot_title(env: Environment, subject: str = "") -> str:
    """A plot title: the environment, what is shown, and non-(x, y) axes."""
    labels = env._plot_labels()
    name = labels.name or labels.layout_type
    units = f" ({labels.units})" if labels.units else ""
    parts = [f"{name}{units}"] + ([subject] if subject else [])
    if not labels.coordinates.equal_aspect and env.n_dims == 2:
        parts.append(" vs ".join(labels.coordinates.axis_labels(labels.units)))
    return ": ".join(parts)


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
    figure, subplot = new_figure(env)
    add_bins(subplot, env, cell_mesh(env), colors=_BIN_COLOR)
    draw_connections(subplot, env)
    draw_layers(
        subplot,
        env,
        show_walls=show_walls,
        show_landmarks=show_landmarks,
        show_regions=show_regions,
    )
    subplot.title = plot_title(env)
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
    from neurospatial.environment._visualization import field_color_limits

    scale = ColorScale(cmap, *field_color_limits(field, vmin, vmax), nan_color)
    figure, subplot = new_figure(env)
    FieldGraphic(subplot, env, field, scale)
    draw_layers(subplot, env, show_walls=True, show_landmarks=True, show_regions=False)
    subplot.title = plot_title(env, colorbar_label)
    if colorbar:
        draw_colorbar(subplot, scale, colorbar_label)
    frame_view(subplot, env)
    return figure
