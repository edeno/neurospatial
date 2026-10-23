"""Display-independent complete-scene Matplotlib rendering and video writing.

The renderer creates its figure, axes and artists once and afterwards only
updates their data, so a frame costs drawing rather than figure construction.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
from matplotlib.colors import to_rgba

from neurospatial.animation._prepared import PreparedScene, TrackWindow
from neurospatial.animation._style import (
    _head_direction_color,
    _marker_area,
    _rate_opacity,
)

Update = Callable[[float], None]


def _planar(points: np.ndarray) -> np.ndarray:
    """Display coordinates: a native 1-D position is drawn at y = 0.

    A 1-D field has physical position on x and field amplitude on y.
    """
    if points.shape[-1] > 1:
        return points
    return np.concatenate([points, np.zeros_like(points)], axis=-1)


def _scatter(ax: Any, **style: Any) -> Any:
    """An empty marker collection, filled per frame by :func:`_set_points`."""
    return ax.scatter(*np.empty((3 if ax.name == "3d" else 2, 0)), **style)


def _set_points(collection: Any, points: np.ndarray) -> None:
    """Show the finite rows of ``points``, ``(n, n_dims)`` environment coordinates.

    A missing (NaN) point draws no marker.
    """
    points = _planar(points[np.isfinite(points).all(axis=1)])
    collection.set_visible(len(points) > 0)
    collection.set_offsets(points[:, :2])
    if points.shape[1] == 3:
        collection.set_3d_properties(points[:, 2], "z")


class _Segments:
    """Line segments that draw exactly like one ``ax.plot`` call per segment.

    Planar axes hold one line collection with ``Line2D``'s cap and join
    styles. 3-D axes keep a fixed pool of lines: a collection there would join
    the depth sort that ``Line3D`` stays outside of.
    """

    def __init__(
        self,
        ax: Any,
        count: int,
        *,
        color: Any,
        alpha: float | None = None,
        linewidth: float | None = None,
        zorder: float,
    ) -> None:
        from matplotlib import rcParams
        from matplotlib.collections import LineCollection

        self.lines: list[Any] = []
        self.collection: Any = None
        if ax.name == "3d":
            for _ in range(count):
                (line,) = ax.plot(
                    [],
                    [],
                    [],
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    zorder=zorder,
                )
                self.lines.append(line)
            return
        self.collection = LineCollection(
            [],
            colors=color,
            alpha=alpha,
            linewidths=rcParams["lines.linewidth"] if linewidth is None else linewidth,
            capstyle=rcParams["lines.solid_capstyle"],
            joinstyle=rcParams["lines.solid_joinstyle"],
            zorder=zorder,
        )
        ax.add_collection(self.collection, autolim=False)

    def set(
        self,
        segments: list[np.ndarray] | np.ndarray,
        *,
        alpha: float | np.ndarray | None = None,
    ) -> None:
        """Draw ``segments``, each ``(2, n_dims)`` in environment coordinates.

        ``alpha`` is one opacity, or one per segment.
        """
        if self.collection is not None:
            self.collection.set_segments([_planar(s) for s in segments])
            if alpha is not None and len(segments):
                self.collection.set_alpha(alpha)
            return
        alphas = np.broadcast_to(np.nan if alpha is None else alpha, len(segments))
        for index, line in enumerate(self.lines):
            line.set_visible(index < len(segments))
            if index < len(segments):
                line.set_data_3d(*segments[index].T)
                if alpha is not None:
                    line.set_alpha(float(alphas[index]))


def _fit_view(ax: Any, points: np.ndarray) -> None:
    """Autoscale to ``points`` alone, as Matplotlib fits a new Axes, then lock.

    Persistent axes remember earlier frames' data limits and view, so both
    are reset: an unmodified (0, 1) view is how Matplotlib recognizes axes
    that have no data limits yet.
    """
    from matplotlib.transforms import Bbox

    ax.dataLim.set_points(Bbox.null().get_points())
    ax.ignore_existing_data_limits = True
    ax.update_datalim(points)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_autoscale_on(True)
    ax.autoscale_view()
    ax.set_autoscale_on(False)


class SceneRenderer:
    """A reusable Agg canvas; video readers belong to the parent scene."""

    def __init__(
        self, scene: Any, *, dpi: int = 100, figsize: tuple[float, float] | None = None
    ) -> None:
        # Resolve capabilities and shared interpretation before allocating a canvas.
        self.prepared = PreparedScene(scene)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.scene = scene
        self.settings = (dpi, figsize)
        self.signal_groups = self.prepared.signal_groups
        self.track_env = self.prepared.track_env
        self.track_extents = self.prepared.track_extents
        self.projection = self.prepared.projection
        self.panel_count = self.prepared.panel_count
        width = max(6, 5 * len(scene.fields))
        self.figure = Figure(
            figsize=figsize or (width, 5 + 1.5 * self.panel_count), dpi=dpi
        )
        self._subplotpars = copy.copy(self.figure.subplotpars)
        self.canvas = FigureCanvasAgg(self.figure)
        # Built on the first render; each entry updates its artists to a time.
        self._updates: list[Update] = []

    def close(self) -> None:
        self.figure.clear()
        self._updates = []

    def _sample(
        self,
        item: Any,
        values: np.ndarray,
        time: float,
        *,
        circular: bool = False,
        name: str | None = None,
    ) -> np.ndarray:
        return self.prepared.sample(item, values, time, circular=circular, name=name)

    def render(self, time: float) -> np.ndarray:
        if not np.isfinite(time):
            raise ValueError("time must be finite seconds.")
        if not self._updates:
            self._build()
        for update in self._updates:
            update(time)
        self.canvas.draw()
        return cast("np.ndarray", np.asarray(self.canvas.buffer_rgba())[..., :3].copy())

    def _build(self) -> None:
        """Create every axis and artist once, laid out at the scene's start.

        One layout for every frame keeps the panels still during playback and
        makes a frame independent of which time was rendered first.
        """
        scene = self.scene
        start = float(scene.frame_times[0])
        self.figure.clear()
        self.figure.subplotpars = copy.copy(self._subplotpars)
        grid = self.figure.add_gridspec(
            1 + self.panel_count,
            len(scene.fields),
            height_ratios=[4] + [1] * self.panel_count,
        )
        updates: list[Update] = []
        for column, (name, field) in enumerate(scene.fields.items()):
            ax = self.figure.add_subplot(
                grid[0, column], projection="3d" if scene.env.n_dims == 3 else None
            )
            updates.append(self._field(ax, name, field, start))
            if scene.env.n_dims in (2, 3):
                _colorbar(ax, self.prepared.field_scales[name])
            ax.set_title(name)
            updates.extend(self._spatial(ax))
        row = 1
        panels: list[Panel] = []
        if self.track_extents is not None:
            for column, (name, field) in enumerate(scene.fields.items()):
                ax = self.figure.add_subplot(grid[row, column])
                panels.append(track_panel(ax, self.prepared, name, field))
            row += 1
        for signals in self.signal_groups.values():
            panels.append(
                signal_panel(
                    self.figure.add_subplot(grid[row, :]), self.prepared, signals
                )
            )
            row += 1
        for raster in scene._overlay_data.spike_rasters:
            panels.append(
                raster_panel(
                    self.figure.add_subplot(grid[row, :]), self.prepared, raster
                )
            )
            row += 1
        if scene._overlay_data.epochs:
            panels.append(
                epoch_panel(self.figure.add_subplot(grid[row, :]), self.prepared)
            )
        updates += [panel.update for panel in panels]
        prefix = "" if scene.title is None else f"{scene.title} — "
        title = self.figure.suptitle("")
        updates.append(lambda time: title.set_text(f"{prefix}t = {time:.3f} s"))
        for update in updates:
            update(start)
        self.figure.tight_layout(rect=(0, 0, 1, 0.95))
        self._updates = updates

    def _below_video(self, time: float) -> bool:
        return any(
            video.z_order == "below" and video.times[0] <= time <= video.times[-1]
            for video in self.scene._overlay_data.videos
        )

    def _field(self, ax: Any, name: str, field: Any, time: float) -> Update:
        """Draw a field once; later frames update its values in place."""
        env = self.scene.env
        scale = self.prepared.field_scales[name]
        low, high = scale.vmin, scale.vmax
        cmaps: dict[bool, Any] = {False: scale.cmap}
        if scale.opaque_from is not None:
            from matplotlib.colors import ListedColormap

            # Low values fade out so a video below shows through them.
            cmaps[True] = ListedColormap(scale.colors())
        if env.n_dims == 3:
            cells, faces = self._field3d(
                ax, self.prepared.field_sample(field, time), scale.cmap, low, high
            )

            def update3d(time: float) -> None:
                colors = _cell_colors(
                    self.prepared.field_sample(field, time), scale.cmap, low, high
                )
                cells.set_facecolor(np.repeat(colors, faces, axis=0))

            return update3d

        from matplotlib.cm import ScalarMappable
        from matplotlib.lines import Line2D

        from neurospatial.layout import _cell_shape

        shape = "line" if env.n_dims == 1 else _cell_shape(env.binning)
        if shape == "grid":
            from neurospatial.geometry import require_grid_context
            from neurospatial.layout.helpers.utils import map_active_data_to_grid

            grid, _ = require_grid_context(
                env.binning, "animation", fix="use a cell renderer"
            )
        state: dict[str, Any] = {"artists": []}

        def draw(values: np.ndarray, below: bool) -> None:
            for artist in state["artists"]:
                artist.remove()
            before = set(ax.get_children())
            # A redraw must autoscale and color its first line like a new Axes.
            _fit_view(ax, np.empty((0, 2)))
            ax.set_autoscale_on(True)
            ax.set_prop_cycle(None)
            env.plot_field(
                values,
                ax=ax,
                cmap=cmaps[below],
                nan_color=None if below else "lightgray",
                vmin=low,
                vmax=high,
                colorbar=False,
                scale_bar=self.scene.scale_bar or False,
            )
            added = [artist for artist in ax.get_children() if artist not in before]
            state.update(
                artists=added,
                below=below,
                drawn=np.isfinite(values) if below else np.ones(len(values), bool),
                field=next(
                    (a for a in added if isinstance(a, (ScalarMappable, Line2D))), None
                ),
            )
            ax.set_title(name)
            if shape == "line":
                # One view for every frame: the track's extent and the field's
                # fixed scale, with zero, where position markers are drawn. A
                # missing row neither moves the axes nor hides the animal.
                track = env.dimension_ranges[0]
                _fit_view(
                    ax,
                    np.array([[track[0], min(low, 0.0)], [track[1], max(high, 0.0)]]),
                )
            # Videos and overlays must not move the field's view.
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())
            # Videos keep drawing after the field, whose artists were re-added.
            for image in [image for image in ax.images if image not in added]:
                image.remove()
                ax.add_image(image)

        def set_values(values: np.ndarray, below: bool) -> bool:
            """Update the drawn artist; False if this frame needs a redraw."""
            artist = state["field"]
            if artist is None or below != state["below"]:
                return False
            if shape == "line":
                artist.set_ydata(values)
            elif shape == "grid":
                # pcolormesh draws the transposed (dim0, dim1) grid.
                artist.set_array(
                    map_active_data_to_grid(
                        grid.shape, grid.active_mask, values, fill_value=np.nan
                    ).T
                )
            elif shape == "polygons":
                artist.set_array(np.ma.masked_invalid(values))
            elif shape == "triangles":
                artist.set_array(values)
            elif shape in ("hexagons", "points"):
                # Without a missing-value color only finite bins are drawn.
                drawn = np.isfinite(values) if below else state["drawn"]
                if shape == "points":
                    artist.set_offsets(env.bin_centers[drawn, :2])
                elif not np.array_equal(drawn, state["drawn"]):
                    return False
                artist.set_array(values[drawn])
            else:
                return False
            return True

        draw(self.prepared.field_sample(field, time), self._below_video(time))

        def update(time: float) -> None:
            values = self.prepared.field_sample(field, time)
            below = self._below_video(time)
            if not set_values(values, below):
                draw(values, below)

        return update

    def _field3d(
        self, ax: Any, values: np.ndarray, cmap: Any, low: float, high: float
    ) -> tuple[Any, int]:
        """Draw actual grid cells, or explicitly labelled centers without cell geometry.

        Returns the colored collection and its number of faces per bin.
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        env = self.scene.env
        frame = env.binning.grid_frame
        grid = frame.source if frame is not None else env.binning.rectilinear
        colors = _cell_colors(values, cmap, low, high)
        if grid is not None:
            # All six faces of each active cell, including transformed grids.
            corners = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                ]
            )
            faces = np.array(
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [0, 1, 5, 4],
                    [2, 3, 7, 6],
                    [1, 2, 6, 5],
                    [0, 3, 7, 4],
                ]
            )
            indices = np.argwhere(grid.active_mask)
            starts = np.column_stack(
                [edge[indices[:, d]] for d, edge in enumerate(grid.edges)]
            )
            stops = np.column_stack(
                [edge[indices[:, d] + 1] for d, edge in enumerate(grid.edges)]
            )
            vertices = (
                starts[:, None, :] + corners[None, :, :] * (stops - starts)[:, None, :]
            )
            if frame is not None:
                vertices = vertices @ frame.forward[:3, :3].T + frame.forward[:3, 3]
            cells = Poly3DCollection(
                vertices[:, faces, :].reshape(-1, 4, 3),
                facecolors=np.repeat(colors, 6, axis=0),
                edgecolors=(0.2, 0.2, 0.2, 0.15),
                linewidths=0.2,
            )
            ax.add_collection3d(cells)
        else:
            cells = ax.scatter(*env.bin_centers.T, c=colors, marker="s", s=35)
            ax.text2D(
                0.02,
                0.02,
                "Values at bin centers (cell geometry unavailable)",
                transform=ax.transAxes,
                fontsize=8,
            )
        ranges = np.asarray(env.dimension_ranges)
        ax.set_xlim(*ranges[0])
        ax.set_ylim(*ranges[1])
        ax.set_zlim(*ranges[2])
        ax.set_box_aspect(np.maximum(np.ptp(ranges, axis=1), np.finfo(float).eps))
        for letter, setter in zip(
            "xyz", (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel), strict=True
        ):
            setter(f"{letter} ({env.units or 'units'})")
        return cells, 6 if grid is not None else 1

    def _spatial(self, ax: Any) -> list[Update]:
        """Create the spatial overlays in drawing order, each with its update."""
        scene = self.scene
        data = scene._overlay_data
        updates = [self._video(ax, video) for video in data.videos]
        if scene.show_regions:
            if scene.env.n_dims == 2:
                _render_regions_matplotlib(
                    ax, scene.env, scene.show_regions, scene.region_alpha
                )
            elif scene.env.n_dims == 3:
                from neurospatial.animation._region_names import shown_region_names

                for name in shown_region_names(scene.env, scene.show_regions):
                    point = scene.env.regions[name].data
                    ax.scatter(*point, color="white", alpha=scene.region_alpha, s=50)
                    ax.text(*point, name)
        updates += [self._position(ax, position) for position in data.positions]
        updates += [self._bodyparts(ax, body) for body in data.bodypart_sets]
        updates += [self._heading(ax, heading) for heading in data.head_directions]
        updates += [
            self._object_vectors(ax, vectors) for vectors in data.object_vectors
        ]
        updates += [self._events(ax, events) for events in data.events]
        return updates

    def _video(self, ax: Any, video: Any) -> Update:
        # A placeholder image keeps the overlay order; frames replace its data.
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        image_artist = ax.imshow(
            np.zeros((1, 1, 3), dtype=np.uint8),
            extent=video.env_bounds,
            origin="upper",
            alpha=video.alpha,
            zorder=-1 if video.z_order == "below" else 5,
            aspect=ax.get_aspect(),
            visible=False,
        )
        if video.transform_to_env is not None:
            from matplotlib.transforms import Affine2D

            image_artist.set_transform(
                Affine2D(video.transform_to_env.A) + ax.transData
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        def update(time: float) -> None:
            image = self.prepared.video_frame(video, time)
            image_artist.set_visible(image is not None)
            if image is None:
                return
            image_artist.set_data(image)
            if video.transform_to_env is not None:
                image_artist.set_extent(
                    (-0.5, image.shape[1] - 0.5, image.shape[0] - 0.5, -0.5)
                )

        return update

    def _position(self, ax: Any, position: Any) -> Update:
        trail = _Segments(
            ax,
            max(self.prepared.trail_capacity(position) - 1, 0),
            color=position.color,
            zorder=100,
        )
        marker = _scatter(
            ax,
            s=_marker_area(position.size, dpi=self.figure.dpi),
            color=position.color,
            edgecolor="white",
            zorder=101,
        )

        def update(time: float) -> None:
            point = self._sample(position, position.data, time)
            if position.trail_length:
                segments, alphas = self.prepared.trail(position, time)
                trail.set(segments, alpha=alphas)
            _set_points(marker, point[None])

        return update

    def _bodyparts(self, ax: Any, body: Any) -> Update:
        markers = [
            _scatter(
                ax,
                color=body.colors[name],
                s=_marker_area(8.0, dpi=self.figure.dpi),
                zorder=101,
            )
            for name in body.bodyparts
        ]
        edges = () if body.skeleton is None else body.skeleton.edges
        skeleton = _Segments(ax, len(edges), color="white", zorder=100)

        def update(time: float) -> None:
            pose = self.prepared.pose(body, time)
            for marker, point in zip(markers, pose.points, strict=True):
                _set_points(marker, point[None])
            skeleton.set(pose.skeleton)

        return update

    def _heading(self, ax: Any, heading: Any) -> Update:
        if ax.name == "3d":
            # A 3-D quiver cannot move in place; replace just this artist.
            state: dict[str, Any] = {"quiver": None}

            def update3d(time: float) -> None:
                if state["quiver"] is not None:
                    state["quiver"].remove()
                    state["quiver"] = None
                arrow = self.prepared.heading_arrow(heading, time)
                if arrow is not None:
                    point, direction = arrow
                    state["quiver"] = ax.quiver(
                        *point,
                        *direction,
                        color=_head_direction_color(heading.color, direction),
                        linewidth=heading.width,
                    )

            return update3d
        arrow = ax.arrow(
            0.0,
            0.0,
            1.0,
            0.0,
            width=heading.width,
            length_includes_head=True,
            zorder=103,
            visible=False,
        )

        def update(time: float) -> None:
            geometry = self.prepared.heading_arrow(heading, time)
            arrow.set_visible(geometry is not None)
            if geometry is not None:
                point, direction = geometry
                arrow.set_data(x=point[0], y=point[1], dx=direction[0], dy=direction[1])
                arrow.set_color(_head_direction_color(heading.color, direction))

        return update

    def _object_vectors(self, ax: Any, vectors: Any) -> Update:
        # Static object coordinates have no time axis.
        n_objects = np.atleast_2d(vectors.object_positions).shape[-2]
        markers = [
            _scatter(
                ax,
                color=vectors.color,
                marker=vectors.object_marker,
                s=_marker_area(vectors.object_size, dpi=self.figure.dpi),
                zorder=103,
            )
            for _ in range(n_objects if vectors.show_objects else 0)
        ]
        lines = _Segments(
            ax, n_objects, color=vectors.color, linewidth=vectors.linewidth, zorder=102
        )

        def update(time: float) -> None:
            animal = self._sample(vectors, vectors.animal_positions, time)
            objects = vectors.object_positions
            if objects.ndim == 3:
                objects = self._sample(vectors, objects, time)
            objects = np.atleast_2d(objects)
            opacity = 1.0
            if vectors.firing_rates is not None:
                rate = self._sample(
                    vectors, vectors.firing_rates, time, name="firing_rates"
                )
                opacity = _rate_opacity(float(rate), vectors._rate_peak or 1.0)
            finite = np.isfinite(objects).all(axis=1)
            for marker, point in zip(markers, objects, strict=True):
                _set_points(marker, point[None])
            segments = (
                [np.array([animal, point]) for point in objects[finite]]
                if np.isfinite(animal).all()
                else []
            )
            lines.set(segments, alpha=opacity)

        return update

    def _events(self, ax: Any, events: Any) -> Update:
        markers = {
            name: _scatter(
                ax,
                color="none",
                marker=events.markers[name],
                s=_marker_area(events.size, dpi=self.figure.dpi),
                linewidths=events.border_width,
                zorder=104,
            )
            for name in events.event_positions
        }

        drawn: dict[str, tuple[int, ...]] = {}

        def update(time: float) -> None:
            for name, shown in self.prepared.events(events, time).items():
                if drawn.get(name) == shown.key:
                    continue
                drawn[name] = shown.key
                # One collection per event type; each marker keeps its own
                # fade, applied to its face and border alike.
                face = np.tile(to_rgba(events.colors[name]), (len(shown.opacity), 1))
                edge = np.tile(to_rgba(events.border_color), (len(shown.opacity), 1))
                face[:, 3] = edge[:, 3] = shown.opacity
                markers[name].set_facecolor(face)
                markers[name].set_edgecolor(edge)
                _set_points(markers[name], shown.positions)

        return update


def _colorbar(ax: Any, scale: Any) -> None:
    """A colorbar of ``scale`` beside ``ax``, marking each clipped end."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    ax.figure.colorbar(
        ScalarMappable(Normalize(scale.vmin, scale.vmax), cmap=scale.cmap),
        ax=ax,
        label=scale.label,
        extend=scale.extend_name,
    )


def _cell_colors(values: np.ndarray, cmap: Any, low: float, high: float) -> np.ndarray:
    """RGBA per bin; missing bins are translucent gray, unlike observed zeros."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    colors = colormaps.get_cmap(cmap)(
        Normalize(low, high)(np.nan_to_num(values, nan=low))
    )
    colors[~np.isfinite(values)] = (0.75, 0.75, 0.75, 0.4)
    return cast("np.ndarray", colors)


def _render_regions_matplotlib(
    ax: Any, env: Any, show_regions: bool | list[str], region_alpha: float
) -> None:
    """Render environment regions as patches on matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to render on
    env : Environment
        Environment with regions
    show_regions : bool | list[str]
        If True, show all regions. If list, show only specified regions.
    region_alpha : float
        Alpha transparency for regions (0-1)

    Notes
    -----
    Uses PathPatch for polygon regions and circles for point regions.
    """
    from matplotlib.patches import Circle, PathPatch
    from matplotlib.path import Path as MplPath

    from neurospatial.animation._region_names import shown_region_names

    for region_name in shown_region_names(env, show_regions):
        region = env.regions[region_name]
        if region.kind == "point":
            ax.add_patch(
                Circle(
                    region.data,
                    radius=5.0,  # Visual marker size
                    facecolor="white",
                    edgecolor="white",
                    alpha=region_alpha,
                    zorder=99,
                )
            )
        elif region.kind == "polygon":
            ax.add_patch(
                PathPatch(
                    MplPath(np.array(region.data.exterior.coords)),
                    facecolor="white",
                    edgecolor="white",
                    alpha=region_alpha,
                    zorder=99,
                )
            )


class Panel(NamedTuple):
    """A scientific panel against time, built once and updated per time.

    ``prepare`` reads the panel's data at a time without touching artists,
    so a native view can run it on a worker thread; ``apply`` sets the
    panel's artists to that data.
    """

    prepare: Callable[[float], Any]
    apply: Callable[[float, Any], None]

    def update(self, time: float) -> None:
        """Prepare and apply one time."""
        self.apply(time, self.prepare(time))


def _time_axis(
    ax: Any,
    prepared: PreparedScene,
    *,
    cursor_visible: bool = True,
    cursor_color: Any = "red",
) -> Update:
    """Style a time panel; its update centers the window and cursor."""
    ax.set_facecolor("#20252b")
    cursor = ax.axvline(0.0, color=cursor_color, linewidth=1, visible=cursor_visible)
    ax.set_xlabel("Time (s)")

    def update(time: float) -> None:
        ax.set_xlim(*prepared.bounds(time))
        cursor.set_xdata([time, time])

    return update


def signal_panel(ax: Any, prepared: PreparedScene, signals: list[Any]) -> Panel:
    """Signals sharing a panel, each a line through its samples in the window."""
    lines = [
        ax.plot(
            [],
            [],
            color=signal.color,
            linewidth=signal.linewidth,
            alpha=signal.alpha,
            label=signal.label,
        )[0]
        for signal in signals
    ]
    if any(signal.label for signal in signals):
        ax.legend(loc="upper right", fontsize="small")
    ax.set_ylim(*prepared.signal_limits(signals))
    # A panel's signals share one cursor, styled by its first signal.
    time_axis = _time_axis(
        ax,
        prepared,
        cursor_visible=signals[0].show_cursor,
        cursor_color=signals[0].cursor_color,
    )

    def prepare(time: float) -> list[tuple[np.ndarray, np.ndarray]]:
        return [prepared.signal_window(signal, time) for signal in signals]

    def apply(time: float, windows: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for line, window in zip(lines, windows, strict=True):
            line.set_data(*window)
        time_axis(time)

    return Panel(prepare, apply)


def raster_panel(ax: Any, prepared: PreparedScene, raster: Any) -> Panel:
    """A spike raster: one row of ticks per unit, labeled by unit id."""
    from matplotlib.collections import LineCollection

    units = range(len(raster.spike_times))
    # One collection for every unit: the ticks share their style.
    ticks = ax.add_collection(
        LineCollection([], colors=raster.color, linewidths=raster.linewidth),
        autolim=False,
    )
    # Fixed rows: the limits Matplotlib fits when every unit has a spike
    # in view, so rows stay put while units fall silent.
    ax.update_datalim([(0.0, -0.4), (0.0, len(units) - 0.6)])
    ax.autoscale_view(scalex=False)
    ax.set_ylim(ax.get_ylim())
    ax.set_yticks(units, [str(unit) for unit in raster.unit_ids])
    ax.set_ylabel("Unit")
    time_axis = _time_axis(ax, prepared)

    def apply(time: float, segments: np.ndarray) -> None:
        ticks.set_segments(segments)
        time_axis(time)

    return Panel(lambda time: prepared.raster_ticks(raster, time), apply)


def epoch_panel(ax: Any, prepared: PreparedScene) -> Panel:
    """Every epoch overlay as shaded intervals, one row each."""
    epochs = prepared.scene._overlay_data.epochs
    for number, epoch in enumerate(epochs):
        for start, stop in epoch.intervals:
            ax.broken_barh(
                [(start, stop - start)],
                (number - 0.35, 0.7),
                facecolors=epoch.color,
                alpha=epoch.alpha,
            )
    ax.set_yticks(range(len(epochs)), [e.label for e in epochs])
    time_axis = _time_axis(ax, prepared)
    return Panel(lambda time: None, lambda time, _: time_axis(time))


def track_panel(
    ax: Any,
    prepared: PreparedScene,
    name: str,
    field: Any,
    *,
    colorbar: bool = True,
) -> Panel:
    """A field along the track against time, with the observed positions.

    The field is one mesh image of (time interval, track bin) cells, so a
    frame costs one image however many samples the window holds.
    """
    from matplotlib import rcParams
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.image import PcolorImage

    scale = prepared.track_scales[name]
    # Clipped to the axes, the mesh takes no part in the layout.
    cells = PcolorImage(
        ax,
        cmap=scale.cmap,
        norm=Normalize(scale.vmin, scale.vmax),
        visible=False,
        in_layout=False,
    )
    ax.add_image(cells)
    if prepared.projection is not None and colorbar:
        _colorbar(ax, scale)
    assert prepared.track_extents is not None
    starts, stops = prepared.track_extents
    ax.set_ylim(float(np.min(starts)), float(np.max(stops)))
    ax.set_ylabel(f"Position ({prepared.track_env.units or 'units'})")
    ax.set_title(f"{name} along track")
    # Drawn like one Line2D per observed segment.
    positions = LineCollection(
        [],
        linewidths=1,
        capstyle=rcParams["lines.solid_capstyle"],
        joinstyle=rcParams["lines.solid_joinstyle"],
        zorder=3,
    )
    ax.add_collection(positions, autolim=False)

    time_axis = _time_axis(ax, prepared)

    def apply(time: float, window: TrackWindow) -> None:
        # Missing cells are masked: transparent over the panel's background.
        cells.set_visible(window.values.shape[1] > 0)
        if window.values.shape[1]:
            cells.set_data(
                window.time_edges,
                window.track_edges,
                np.ma.masked_invalid(window.values),
            )
        # Matplotlib's stubs omit the (n, 2, 2) array input it accepts.
        positions.set_segments(cast("Any", window.paths))
        positions.set_color(window.path_colors)
        time_axis(time)

    return Panel(lambda time: prepared.track_window(field, time), apply)


def export_scene(
    scene: Any,
    path: Path,
    *,
    dpi: int,
    figsize: tuple[float, float] | None,
    **options: Any,
) -> Path:
    """Stream complete scene frames to a file, replacing only on success.

    ``options`` are those of :func:`~neurospatial.animation._export.prepare_export`.
    With ``n_workers > 1`` worker processes render equivalent scenes.
    """
    from neurospatial.animation._export import prepare_export
    from neurospatial.animation._prepared import iter_observation_clocks

    plan = prepare_export(
        path,
        scene.frame_times,
        title=scene.title,
        sample_times=iter_observation_clocks(scene),
        **options,
    )
    if plan.n_workers > 1:
        from neurospatial.animation._export_workers import render_in_workers

        return plan.write_frames(
            render_in_workers(
                scene, plan.times, dpi=dpi, figsize=figsize, n_workers=plan.n_workers
            )
        )
    renderer = SceneRenderer(scene, dpi=dpi, figsize=figsize)
    try:
        return plan.write(renderer.render)
    finally:
        renderer.close()
