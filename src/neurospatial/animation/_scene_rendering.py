"""Display-independent complete-scene Matplotlib rendering and video writing."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import numpy as np

from neurospatial.animation._sampling import sample_at_times


class SceneRenderer:
    """A reusable Agg canvas; video readers belong to the parent scene."""

    def __init__(
        self, scene: Any, *, dpi: int = 100, figsize: tuple[float, float] | None = None
    ) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.scene = scene
        self.settings = (dpi, figsize)
        data = scene._overlay_data
        self.signal_groups: dict[tuple[str, str | int], list[Any]] = {}
        for i, signal in enumerate(data.timeseries):
            self.signal_groups.setdefault(
                ("group", signal.group) if signal.group is not None else ("signal", i),
                [],
            ).append(signal)
        from neurospatial.animation._tracks import (
            _projection,
            track_extents,
        )

        self.track_env = scene.env if scene.track is None else scene.track
        self.track_extents = track_extents(self.track_env)
        self.projection = (
            None if scene.track is None else _projection(scene.env, scene.track)
        )
        self.panel_count = (
            len(self.signal_groups)
            + len(data.spike_rasters)
            + bool(data.epochs)
            + (self.track_extents is not None)
        )
        width = max(6, 5 * len(scene.fields))
        self.figure = Figure(
            figsize=figsize or (width, 5 + 1.5 * self.panel_count), dpi=dpi
        )
        self._subplotpars = copy.copy(self.figure.subplotpars)
        self.canvas = FigureCanvasAgg(self.figure)
        self.limits = {
            name: field.color_limits() for name, field in scene.fields.items()
        }
        self.track_limits = dict(self.limits)
        if self.projection is not None:
            from neurospatial.animation._tracks import _project
            from neurospatial.animation.rendering import animation_colormap_range

            for name, field in scene.fields.items():
                indices = np.unique(
                    np.linspace(0, len(field) - 1, min(len(field), 100), dtype=int)
                )
                rows = np.stack([field[int(index)] for index in indices])
                projected = _project(rows, self.projection)
                self.track_limits[name] = animation_colormap_range(
                    projected, None, None
                )

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
        times = item.times if item.times is not None else self.scene.frame_times
        valid = getattr(item, "sample_valid", None)
        intervals = getattr(item, "interval_valid", None)
        if name is not None:
            valid = None if valid is None else valid.get(name)
            intervals = None if intervals is None else intervals.get(name)
        sampled = sample_at_times(
            times,
            values,
            np.array([time]),
            interp=getattr(item, "interp", "linear"),
            sample_valid=valid,
            interval_valid=intervals,
            circular=circular,
        )[0]
        return cast("np.ndarray", sampled)

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
            for unit, spikes in enumerate(raster.spike_times):
                keep = (spikes >= time - scene.window_seconds / 2) & (
                    spikes <= time + scene.window_seconds / 2
                )
                ax.vlines(
                    spikes[keep],
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
            keep = (signal.times >= time - self.scene.window_seconds / 2) & (
                signal.times <= time + self.scene.window_seconds / 2
            )
            values = signal.data[keep]
            if signal.normalize:
                span = signal.global_vmax - signal.global_vmin
                values = (values - signal.global_vmin) / (span if span else 1)
            ax.plot(
                signal.times[keep],
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
        """Position-by-time cells respect the source's exact interval support."""
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import Normalize

        from neurospatial.animation._tracks import (
            _project,
            track_position,
        )

        assert self.track_extents is not None
        starts, stops = self.track_extents
        left, right = (
            time - self.scene.window_seconds / 2,
            time + self.scene.window_seconds / 2,
        )
        selected = np.flatnonzero(
            (field.times[:-1] < right) & (field.times[1:] > left) & field.interval_valid
        )
        polygons, colors = [], []
        for index in selected:
            values = field[int(index)]
            if self.projection is not None:
                values = _project(values[None, :], self.projection)[0]
            lo, hi = max(left, field.times[index]), min(right, field.times[index + 1])
            for first, last, value in zip(starts, stops, values, strict=True):
                if np.isfinite(value):
                    polygons.append([(lo, first), (hi, first), (hi, last), (lo, last)])
                    colors.append(value)
        low, high = self.track_limits[name]
        collection = PolyCollection(
            polygons, cmap=field.cmap, norm=Normalize(low, high), edgecolors="none"
        )
        collection.set_array(np.asarray(colors))
        ax.add_collection(collection)
        if self.projection is not None and colorbar:
            self.figure.colorbar(
                collection,
                ax=ax,
                label=f"Projected {field.label or name} (sum per track bin)",
            )
        ax.set_ylim(float(np.min(starts)), float(np.max(stops)))
        ax.set_ylabel(f"Position ({self.track_env.units or 'units'})")
        ax.set_title(f"{name} along track")
        for position in self.scene._overlay_data.positions:
            times = (
                position.times if position.times is not None else self.scene.frame_times
            )
            keep = (times >= left) & (times <= right)
            distances = track_position(self.track_env, position.data)
            intervals = position.interval_valid
            for index in np.flatnonzero(keep[:-1] & keep[1:]):
                if intervals is None or intervals[index]:
                    ax.plot(
                        times[index : index + 2],
                        distances[index : index + 2],
                        color=position.color,
                        linewidth=1,
                        zorder=3,
                    )
        self._time_panel(ax, time)

    def _spatial(self, ax: Any, time: float) -> None:
        from matplotlib import colormaps

        from neurospatial.animation._parallel import _render_regions_matplotlib
        from neurospatial.animation._style import _event_visibility

        scene = self.scene
        data = scene._overlay_data

        # Spatial overlays use environment coordinates. A native 1-D field has
        # physical position on x and field amplitude on y.
        def xy(point: np.ndarray) -> Any:
            return point if len(point) > 1 else (point[0], 0)

        spatial_xlim, spatial_ylim = ax.get_xlim(), ax.get_ylim()
        for video in data.videos:
            times = video.times if video.times is not None else scene.frame_times
            if time < times[0] or time > times[-1]:
                continue
            index = int(np.argmin(np.abs(times - time)))
            image = (
                video.reader[index]
                if video.times is not None
                else video.get_frame(index)
            )
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
                    if position.data.shape[1] in (2, 3):
                        for index in range(start, stop - 1):
                            if valid is None or valid[index]:
                                pair = position.data[index : index + 2]
                                ax.plot(
                                    *pair.T,
                                    color=position.color,
                                    alpha=0.6,
                                    zorder=100,
                                )
                ax.scatter(
                    *xy(point),
                    s=position.size**2,
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
                        zorder=101,
                    )
            if body.skeleton is not None:
                for first, second in body.skeleton.edges:
                    pair = np.array([points[first], points[second]])
                    if pair.shape[1] in (2, 3) and np.isfinite(pair).all():
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
            if (
                np.isfinite(sampled_heading).all()
                and np.isfinite(point).all()
                and len(point) in (2, 3)
            ):
                angle = (
                    float(sampled_heading)
                    if heading.data.ndim == 1
                    else float(np.arctan2(sampled_heading[1], sampled_heading[0]))
                )
                color = (
                    colormaps[heading.color]((angle % (2 * np.pi)) / (2 * np.pi))
                    if heading.color in colormaps
                    else heading.color
                )
                if len(point) == 3:
                    norm = float(np.linalg.norm(sampled_heading))
                    if norm > 0:
                        ax.quiver(
                            *point,
                            *(heading.length * sampled_heading / norm),
                            color=color,
                            linewidth=heading.width,
                        )
                else:
                    ax.arrow(
                        *point,
                        heading.length * np.cos(angle),
                        heading.length * np.sin(angle),
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
                finite_rates = vectors.firing_rates[np.isfinite(vectors.firing_rates)]
                peak = float(np.max(finite_rates, initial=0)) or 1.0
                opacity = (
                    float(np.clip(0.1 + 0.9 * float(rate) / peak, 0, 1))
                    if np.isfinite(rate)
                    else 0.0
                )
            for point in np.atleast_2d(objects):
                if not np.isfinite(point).all():
                    continue
                if vectors.show_objects:
                    ax.scatter(
                        *point,
                        color=vectors.color,
                        marker=vectors.object_marker,
                        s=vectors.object_size**2,
                        zorder=103,
                    )
                if np.isfinite(animal).all() and len(animal) in (2, 3):
                    ax.plot(
                        *np.array([animal, point]).T,
                        color=vectors.color,
                        linewidth=vectors.linewidth,
                        alpha=opacity,
                        zorder=102,
                    )
        for events in data.events:
            for name, positions in events.event_positions.items():
                event_times = events.event_times.get(name)
                if event_times is None:
                    event_times = scene.frame_times[events.event_frame_indices[name]]
                # Event persistence uses the complete scene clock, so a selected
                # export does not reset the age of an earlier event.
                frame = int(np.searchsorted(scene.frame_times, time, side="right")) - 1
                visible, alphas = _event_visibility(
                    np.maximum(
                        np.searchsorted(scene.frame_times, event_times, side="right")
                        - 1,
                        0,
                    ),
                    frame,
                    events.decay_frames,
                )
                visible &= event_times <= time
                for point, alpha in zip(
                    positions[visible], alphas[visible], strict=True
                ):
                    if np.isfinite(point).all():
                        ax.scatter(
                            *xy(point),
                            color=events.colors[name],
                            marker=events.markers[name],
                            s=events.size**2,
                            alpha=float(alpha) * events.opacity,
                            linewidths=events.border_width,
                            edgecolors=events.border_color,
                            zorder=104,
                        )


def export_scene(
    scene: Any,
    path: Path,
    *,
    time_range: Any,
    fps: float,
    speed: float,
    overwrite: bool,
    dpi: int,
    figsize: tuple[float, float] | None,
    codec: str | None,
) -> Path:
    """Stream complete scene frames to a file, replacing only on success."""
    from neurospatial.animation._export import prepare_export

    plan = prepare_export(
        path,
        scene.frame_times,
        time_range=time_range,
        fps=fps,
        speed=speed,
        overwrite=overwrite,
        codec=codec,
        sample_times=(field.times for field in scene.fields.values()),
    )
    renderer = SceneRenderer(scene, dpi=dpi, figsize=figsize)
    try:
        return plan.write(renderer.render)
    finally:
        renderer.close()
