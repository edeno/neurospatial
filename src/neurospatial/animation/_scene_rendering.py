"""Display-independent complete-scene Matplotlib rendering and video writing."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import numpy as np
from matplotlib.colors import to_rgba

from neurospatial.animation._prepared import PreparedScene
from neurospatial.animation._style import (
    _head_direction_color,
    _heading_displacement,
    _marker_area,
    _rate_opacity,
)


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
        self.limits = self.prepared.limits
        self.track_limits = self.prepared.track_limits
        width = max(6, 5 * len(scene.fields))
        self.figure = Figure(
            figsize=figsize or (width, 5 + 1.5 * self.panel_count), dpi=dpi
        )
        self._subplotpars = copy.copy(self.figure.subplotpars)
        self.canvas = FigureCanvasAgg(self.figure)

    def close(self) -> None:
        self.figure.clear()

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
        scene = self.scene
        self.figure.clear()
        self.figure.subplotpars = copy.copy(self._subplotpars)
        grid = self.figure.add_gridspec(
            1 + self.panel_count,
            len(scene.fields),
            height_ratios=[4] + [1] * self.panel_count,
        )
        for column, (name, field) in enumerate(scene.fields.items()):
            ax = self.figure.add_subplot(
                grid[0, column], projection="3d" if scene.env.n_dims == 3 else None
            )
            low, high = self.limits[name]
            cmap = field.cmap
            below_video = any(
                video.z_order == "below"
                and (video.times if video.times is not None else scene.frame_times)[0]
                <= time
                <= (video.times if video.times is not None else scene.frame_times)[-1]
                for video in scene._overlay_data.videos
            )
            if below_video:
                from matplotlib import colormaps
                from matplotlib.colors import ListedColormap

                colors = colormaps.get_cmap(field.cmap)(np.linspace(0, 1, 256))
                values = np.linspace(low, high, 256)
                colors[:, 3] *= np.clip(values / field._typical_peak(), 0, 1)
                cmap = ListedColormap(colors)
            if scene.env.n_dims == 3:
                self._field3d(ax, field.sample(time), cmap, low, high)
            else:
                scene.env.plot_field(
                    field.sample(time),
                    ax=ax,
                    cmap=cmap,
                    nan_color=None if below_video else "lightgray",
                    vmin=low,
                    vmax=high,
                    colorbar=False,
                    scale_bar=scene.scale_bar or False,
                )
            if scene.env.n_dims in (2, 3):
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize

                self.figure.colorbar(
                    ScalarMappable(Normalize(low, high), cmap=field.cmap),
                    ax=ax,
                    label=field.label or name,
                )
            ax.set_title(name)
            self._spatial(ax, time)
        row = 1
        if self.track_extents is not None:
            for column, (name, field) in enumerate(scene.fields.items()):
                ax = self.figure.add_subplot(grid[row, column])
                self._track(ax, name, field, time)
            row += 1
        for signals in self.signal_groups.values():
            ax = self.figure.add_subplot(grid[row, :])
            self._signals(ax, signals, time)
            row += 1
        for raster in scene._overlay_data.spike_rasters:
            ax = self.figure.add_subplot(grid[row, :])
            for unit in range(len(raster.spike_times)):
                ax.vlines(
                    self.prepared.raster_window(raster, unit, time),
                    unit - 0.4,
                    unit + 0.4,
                    color=raster.color,
                    linewidth=raster.linewidth,
                )
            ax.set_yticks(
                range(len(raster.unit_ids)), [str(unit) for unit in raster.unit_ids]
            )
            ax.set_ylabel("Unit")
            self._time_panel(ax, time)
            row += 1
        if scene._overlay_data.epochs:
            ax = self.figure.add_subplot(grid[row, :])
            for number, epoch in enumerate(scene._overlay_data.epochs):
                for start, stop in epoch.intervals:
                    ax.broken_barh(
                        [(start, stop - start)],
                        (number - 0.35, 0.7),
                        facecolors=epoch.color,
                        alpha=epoch.alpha,
                    )
            ax.set_yticks(
                range(len(scene._overlay_data.epochs)),
                [e.label for e in scene._overlay_data.epochs],
            )
            self._time_panel(ax, time)
        prefix = "" if scene.title is None else f"{scene.title} — "
        self.figure.suptitle(f"{prefix}t = {time:.3f} s")
        self.figure.tight_layout(rect=(0, 0, 1, 0.95))
        self.canvas.draw()
        return cast("np.ndarray", np.asarray(self.canvas.buffer_rgba())[..., :3].copy())

    def _field3d(
        self, ax: Any, values: np.ndarray, cmap: Any, low: float, high: float
    ) -> None:
        """Draw actual grid cells, or explicitly labelled centers without cell geometry."""
        from matplotlib import colormaps
        from matplotlib.colors import Normalize
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        env = self.scene.env
        frame = env.binning.grid_frame
        grid = frame.source if frame is not None else env.binning.rectilinear
        colors = colormaps.get_cmap(cmap)(
            Normalize(low, high)(np.nan_to_num(values, nan=low))
        )
        colors[~np.isfinite(values)] = (0.75, 0.75, 0.75, 0.4)
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
            collection = Poly3DCollection(
                vertices[:, faces, :].reshape(-1, 4, 3),
                facecolors=np.repeat(colors, 6, axis=0),
                edgecolors=(0.2, 0.2, 0.2, 0.15),
                linewidths=0.2,
            )
            ax.add_collection3d(collection)
        else:
            ax.scatter(*env.bin_centers.T, c=colors, marker="s", s=35)
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

    def _time_panel(self, ax: Any, time: float) -> None:
        ax.set_facecolor("#20252b")
        ax.set_xlim(
            time - self.scene.window_seconds / 2, time + self.scene.window_seconds / 2
        )
        ax.axvline(time, color="red", linewidth=1)
        ax.set_xlabel("Time (s)")

    def _signals(self, ax: Any, signals: list[Any], time: float) -> None:
        for signal in signals:
            times, values = self.prepared.signal_window(signal, time)
            ax.plot(
                times,
                values,
                color=signal.color,
                linewidth=signal.linewidth,
                alpha=signal.alpha,
                label=signal.label,
            )
        if any(signal.label for signal in signals):
            ax.legend(loc="upper right", fontsize="small")
        lows = [0.0 if signal.normalize else signal.global_vmin for signal in signals]
        highs = [1.0 if signal.normalize else signal.global_vmax for signal in signals]
        low, high = min(lows), max(highs)
        if low == high:
            low, high = low - 0.5, high + 0.5
        ax.set_ylim(low, high)
        self._time_panel(ax, time)

    def _track(
        self, ax: Any, name: str, field: Any, time: float, *, colorbar: bool = True
    ) -> None:
        """Draw shared prepared track cells on this renderer's axes."""
        draw_track_panel(ax, self.prepared, name, field, time, colorbar=colorbar)

    def _spatial(self, ax: Any, time: float) -> None:
        from neurospatial.animation._style import _event_visibility

        scene = self.scene
        data = scene._overlay_data

        # Spatial overlays use environment coordinates. A native 1-D field has
        # physical position on x and field amplitude on y.
        def xy(point: np.ndarray) -> Any:
            return point if len(point) > 1 else (point[0], 0)

        spatial_xlim, spatial_ylim = ax.get_xlim(), ax.get_ylim()
        for video in data.videos:
            image = self.prepared.video_frame(video, time)
            if image is None:
                continue
            image_artist = ax.imshow(
                image,
                extent=video.env_bounds,
                origin="upper",
                alpha=video.alpha,
                zorder=-1 if video.z_order == "below" else 5,
            )
            if video.transform_to_env is not None:
                from matplotlib.transforms import Affine2D

                image_artist.set_extent(
                    (-0.5, image.shape[1] - 0.5, image.shape[0] - 0.5, -0.5)
                )
                image_artist.set_transform(
                    Affine2D(video.transform_to_env.A) + ax.transData
                )
        ax.set_xlim(spatial_xlim)
        ax.set_ylim(spatial_ylim)
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
        for position in data.positions:
            point = self._sample(position, position.data, time)
            if np.isfinite(point).all():
                if position.trail_length:
                    times = (
                        position.times
                        if position.times is not None
                        else scene.frame_times
                    )
                    stop = int(np.searchsorted(times, time, side="right"))
                    start = max(0, stop - position.trail_length)
                    valid = position.interval_valid
                    for index in range(start, stop - 1):
                        if valid is None or valid[index]:
                            pair = np.array(
                                [
                                    xy(point)
                                    for point in position.data[index : index + 2]
                                ]
                            )
                            ax.plot(
                                *pair.T,
                                color=position.color,
                                alpha=0.6,
                                zorder=100,
                            )
                ax.scatter(
                    *xy(point),
                    s=_marker_area(position.size, dpi=self.figure.dpi),
                    color=position.color,
                    edgecolor="white",
                    zorder=101,
                )
        for body in data.bodypart_sets:
            points = {
                name: self._sample(body, values, time, name=name)
                for name, values in body.bodyparts.items()
            }
            for name, point in points.items():
                if np.isfinite(point).all():
                    ax.scatter(
                        *xy(point),
                        color=(body.colors or {}).get(name, "red"),
                        s=_marker_area(8.0, dpi=self.figure.dpi),
                        zorder=101,
                    )
            if body.skeleton is not None:
                for first, second in body.skeleton.edges:
                    pair = np.array([xy(points[first]), xy(points[second])])
                    if np.isfinite(pair).all():
                        ax.plot(*pair.T, color="white", zorder=100)
        for heading in data.head_directions:
            sampled_heading = self._sample(
                heading, heading.data, time, circular=heading.data.ndim == 1
            )
            point = (
                self._sample(data.positions[0], data.positions[0].data, time)
                if len(data.positions) == 1
                else np.mean(scene.env.bin_centers, axis=0)
            )
            direction = _heading_displacement(
                sampled_heading, heading.length, n_dims=len(point)
            )
            if direction is not None and np.isfinite(point).all():
                color = _head_direction_color(heading.color, direction)
                if len(point) == 3:
                    ax.quiver(
                        *point,
                        *direction,
                        color=color,
                        linewidth=heading.width,
                    )
                else:
                    ax.arrow(
                        *point,
                        *direction,
                        color=color,
                        width=heading.width,
                        length_includes_head=True,
                        zorder=103,
                    )
        for vectors in data.object_vectors:
            animal = self._sample(vectors, vectors.animal_positions, time)
            # Static object coordinates have no time axis.
            objects = vectors.object_positions
            if objects.ndim == 3:
                objects = self._sample(vectors, objects, time)
            opacity = 1.0
            if vectors.firing_rates is not None:
                rate = self._sample(vectors, vectors.firing_rates, time)
                opacity = _rate_opacity(float(rate), vectors._rate_peak or 1.0)
            for point in np.atleast_2d(objects):
                if not np.isfinite(point).all():
                    continue
                if vectors.show_objects:
                    ax.scatter(
                        *xy(point),
                        color=vectors.color,
                        marker=vectors.object_marker,
                        s=_marker_area(vectors.object_size, dpi=self.figure.dpi),
                        zorder=103,
                    )
                if np.isfinite(animal).all():
                    ax.plot(
                        *np.array([xy(animal), xy(point)]).T,
                        color=vectors.color,
                        linewidth=vectors.linewidth,
                        alpha=opacity,
                        zorder=102,
                    )
        for events in data.events:
            for name, positions in events.event_positions.items():
                event_times, event_indices = self.prepared.event_clocks[
                    id(events), name
                ]
                # Event persistence uses the complete scene clock, so a selected
                # export does not reset the age of an earlier event.
                frame = int(np.searchsorted(scene.frame_times, time, side="right")) - 1
                visible, alphas = _event_visibility(
                    event_indices,
                    frame,
                    events.decay_frames,
                )
                visible &= event_times <= time
                visible &= np.isfinite(positions).all(axis=1)
                if not visible.any():
                    continue
                # One collection per event type; each marker keeps its own
                # fade, applied to its face and border alike.
                opacity = alphas[visible] * events.opacity
                face = np.tile(to_rgba(events.colors[name]), (len(opacity), 1))
                edge = np.tile(to_rgba(events.border_color), (len(opacity), 1))
                face[:, 3] = edge[:, 3] = opacity
                points = positions[visible]
                ax.scatter(
                    points[:, 0],
                    points[:, 1] if points.shape[1] > 1 else np.zeros(len(points)),
                    *([points[:, 2]] if points.shape[1] > 2 else []),
                    c=face,
                    marker=events.markers[name],
                    s=_marker_area(events.size, dpi=self.figure.dpi),
                    linewidths=events.border_width,
                    edgecolors=edge,
                    zorder=104,
                )


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


def draw_track_panel(
    ax: Any,
    prepared: PreparedScene,
    name: str,
    field: Any,
    time: float,
    *,
    colorbar: bool = True,
) -> None:
    """Draw prepared track data; usable by Agg and native scientific docks."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    window = prepared.track_window(field, time)
    low, high = prepared.track_limits[name]
    collection = PolyCollection(
        window.polygons.tolist(),
        cmap=field.cmap,
        norm=Normalize(low, high),
        edgecolors="none",
    )
    collection.set_array(window.values)
    ax.add_collection(collection)
    if prepared.projection is not None and colorbar:
        ax.figure.colorbar(
            collection,
            ax=ax,
            label=f"Projected {field.label or name} (sum per track bin)",
        )
    assert prepared.track_extents is not None
    starts, stops = prepared.track_extents
    ax.set_ylim(float(np.min(starts)), float(np.max(stops)))
    ax.set_ylabel(f"Position ({prepared.track_env.units or 'units'})")
    ax.set_title(f"{name} along track")
    for segment, color in window.positions:
        ax.plot(*segment.T, color=color, linewidth=1, zorder=3)
    ax.set_facecolor("#20252b")
    ax.set_xlim(*prepared.bounds(time))
    ax.axvline(time, color="red", linewidth=1)
    ax.set_xlabel("Time (s)")


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
