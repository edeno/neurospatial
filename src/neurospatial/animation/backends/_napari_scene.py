"""Native napari layers driven by an Animation's physical clock.

The napari spatial canvas and synchronized Matplotlib scientific panels are
captured together. Controls and application chrome are excluded from exports.
Sources and video readers remain owned by the scene, not this view.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any, cast

import numpy as np

from neurospatial.animation._prepared import PreparedScene
from neurospatial.animation._scene_rendering import draw_track_panel
from neurospatial.animation._style import _heading_displacement, _rate_opacity


def _native_grid(env: Any) -> tuple[Any, np.ndarray] | None:
    """Return uniform source cells and the pixel-center to napari-world map."""
    frame = env.binning.grid_frame
    grid = frame.source if frame is not None else env.binning.rectilinear
    if grid is None or any(
        not np.allclose(np.diff(e), np.diff(e)[0]) for e in grid.edges
    ):
        return None
    dims = env.n_dims
    pixel = np.eye(dims + 1)
    pixel[np.arange(dims), np.arange(dims)] = [np.diff(e)[0] for e in grid.edges]
    pixel[:dims, -1] = [(e[0] + e[1]) / 2 for e in grid.edges]
    world = pixel if frame is None else frame.forward @ pixel
    order = [*range(dims - 1, -1, -1), dims]
    return grid, world[np.ix_(order, order)]


def _image(values: np.ndarray, grid: Any) -> np.ndarray:
    image = np.full(grid.shape, np.nan, dtype=np.float32)
    image[grid.active_mask] = values
    return cast("np.ndarray", image.transpose())


def _symbol(marker: str) -> str:
    """Translate Matplotlib marker names; napari validates unknown symbols."""
    return {
        "o": "disc",
        "s": "square",
        "^": "triangle_up",
        "v": "triangle_down",
        "*": "star",
        "+": "cross",
        "x": "x",
        "D": "diamond",
        "d": "diamond",
        ".": "disc",
        "<": "triangle_left",
        ">": "triangle_right",
    }.get(marker, marker)


def _rgba(color: Any) -> Any:
    from matplotlib.colors import to_rgba

    return np.asarray(to_rgba(color))


class _ScientificPanels:
    """Persistent Matplotlib artists, so native annotations survive seeking."""

    def __init__(self, scene: Any, helper: PreparedScene) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.scene, self.helper = scene, helper
        count = helper.panel_count + (
            len(scene.fields) - 1 if helper.track_extents is not None else 0
        )
        self.figure = Figure(figsize=(8, max(1.5, 1.5 * count)), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes: list[Any] = []
        self.signals: list[tuple[Any, Any]] = []
        self.rasters: list[tuple[Any, Any, int]] = []
        self.tracks: list[tuple[Any, Any, Any]] = []
        self.cursors: list[Any] = []
        for signals in helper.signal_groups.values():
            ax = self.figure.add_subplot(count, 1, len(self.axes) + 1)
            self.axes.append(ax)
            for signal in signals:
                (line,) = ax.plot(
                    [],
                    [],
                    color=signal.color,
                    linewidth=signal.linewidth,
                    alpha=signal.alpha,
                    label=signal.label,
                )
                self.signals.append((line, signal))
            if any(s.label for s in signals):
                ax.legend(loc="upper right", fontsize="small")
            lows = [0 if s.normalize else s.global_vmin for s in signals]
            highs = [1 if s.normalize else s.global_vmax for s in signals]
            low, high = min(lows), max(highs)
            ax.set_ylim(low - (low == high) * 0.5, high + (low == high) * 0.5)
        for raster in scene._overlay_data.spike_rasters:
            ax = self.figure.add_subplot(count, 1, len(self.axes) + 1)
            self.axes.append(ax)
            for row in range(len(raster.unit_ids)):
                collection = ax.vlines(
                    [], [], [], color=raster.color, linewidth=raster.linewidth
                )
                self.rasters.append((collection, raster, row))
            ax.set_yticks(
                range(len(raster.unit_ids)), [str(u) for u in raster.unit_ids]
            )
            ax.set_ylim(-0.5, max(0.5, len(raster.unit_ids) - 0.5))
            ax.set_ylabel("Unit")
        if scene._overlay_data.epochs:
            ax = self.figure.add_subplot(count, 1, len(self.axes) + 1)
            self.axes.append(ax)
            for row, epoch in enumerate(scene._overlay_data.epochs):
                ax.broken_barh(
                    [(a, b - a) for a, b in epoch.intervals],
                    (row - 0.35, 0.7),
                    facecolors=epoch.color,
                    alpha=epoch.alpha,
                )
            ax.set_yticks(
                range(len(scene._overlay_data.epochs)),
                [e.label for e in scene._overlay_data.epochs],
            )
        # Track images need interval polygons. Retain their axes and replace only
        # the managed cell collection, preserving annotations and other artists.
        if helper.track_extents is not None:
            for name, field in scene.fields.items():
                # One shared track panel per stream; resize the grid once.
                ax = self.figure.add_subplot(count, 1, len(self.axes) + 1)
                self.axes.append(ax)
                self.tracks.append((ax, name, field))
                if helper.projection is not None:
                    from matplotlib.cm import ScalarMappable
                    from matplotlib.colors import Normalize

                    low, high = helper.track_limits[name]
                    self.figure.colorbar(
                        ScalarMappable(Normalize(low, high), cmap=field.cmap),
                        ax=ax,
                        label=f"Projected {field.label or name} (sum per track bin)",
                    )
        for ax in self.axes:
            ax.set_facecolor("#20252b")
            ax.set_xlabel("Time (s)")
            self.cursors.append(
                ax.axvline(scene.frame_times[0], color="red", linewidth=1)
            )
        self._track_artists: list[Any] = []

    def update(self, time: float) -> None:
        left, right = (
            time - self.scene.window_seconds / 2,
            time + self.scene.window_seconds / 2,
        )
        for line, signal in self.signals:
            times, values = self.helper.signal_window(signal, time)
            line.set_data(times, values)
        for artist, raster, row in self.rasters:
            spikes = self.helper.raster_window(raster, row, time)
            segments = np.array(
                [[[t, row - 0.4], [t, row + 0.4]] for t in spikes]
            ).reshape(-1, 2, 2)
            artist.set_segments(segments)
        for artist in self._track_artists:
            artist.remove()
        self._track_artists.clear()
        for ax, name, field in self.tracks:
            before = set(ax.get_children())
            # _track also adds a cursor; remove only that managed duplicate.
            draw_track_panel(ax, self.helper, name, field, time, colorbar=False)
            self._track_artists.extend(set(ax.get_children()) - before)
        for ax, cursor in zip(self.axes, self.cursors, strict=True):
            ax.set_xlim(left, right)
            cursor.set_xdata([time, time])
        self.figure.tight_layout()
        self.canvas.draw()

    def capture(self) -> np.ndarray:
        self.canvas.draw()
        return cast("np.ndarray", np.asarray(self.canvas.buffer_rgba())[..., :3].copy())


class NapariScene:
    """One persistent native viewer and absolute recording-time controls."""

    def __init__(self, scene: Any, **options: Any) -> None:
        import napari
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import (
            QDoubleSpinBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QWidget,
        )

        scene._ensure_open()
        allowed = {"viewer", "show", "ndisplay", "title", "fps", "speed", "layout"}
        unknown = set(options) - allowed
        if unknown:
            raise TypeError(
                f"Unknown napari scene options: {', '.join(sorted(unknown))}."
            )
        if scene.env.n_dims == 1:
            raise ValueError(
                "Napari scene viewing requires 2D or 3D coordinates; use backend='widget' for a 1D track."
            )
        self.scene, self._closed = scene, False
        self._time = float(scene.frame_times[0])
        self._speed = float(options.get("speed", 1))
        fps = float(options.get("fps", 30))
        if not np.isfinite([self._speed, fps]).all() or self._speed <= 0 or fps <= 0:
            raise ValueError("fps and speed must be positive finite numbers.")
        self._step = self._speed / fps
        self._helper = PreparedScene(scene)
        self._panels: _ScientificPanels | None = None
        self.viewer: Any = options.get("viewer")
        self._owns_viewer = self.viewer is None
        if self.viewer is None:
            self.viewer = napari.Viewer(
                title=options.get("title", scene.title or "Animation"),
                ndisplay=options.get("ndisplay", scene.env.n_dims),
                show=options.get("show", True),
            )
        self._layers: list[Any] = []
        self._updates: list[Any] = []
        self.field_layers: dict[str, Any] = {}
        try:
            if not self._owns_viewer:
                if "title" in options:
                    self.viewer.title = options["title"]
                if "ndisplay" in options:
                    self.viewer.dims.ndisplay = options["ndisplay"]
            self._build_fields()
            self._build_overlays()
            if scene.scale_bar:
                from neurospatial.animation.config import (
                    ScaleBarConfig,
                    configure_napari_scale_bar,
                )

                configure_napari_scale_bar(
                    self.viewer,
                    units=scene.env.units,
                    config=scene.scale_bar
                    if isinstance(scene.scale_bar, ScaleBarConfig)
                    else None,
                )
            # Each stream retains a separate layer, scale and colormap. Native
            # grid mode can separate spatial fields when explicitly requested.
            layout = options.get("layout")
            if layout is not None:
                if layout not in ("horizontal", "vertical", "grid"):
                    raise ValueError(
                        "layout must be 'horizontal', 'vertical' or 'grid'."
                    )
                if self.scene.overlays:
                    raise ValueError(
                        "Napari grid layout separates individual overlay layers; omit layout to keep overlays registered to every field."
                    )
                self.viewer.grid.enabled = True
                self.viewer.grid.shape = {
                    "horizontal": (1, len(scene.fields)),
                    "vertical": (len(scene.fields), 1),
                    "grid": (-1, -1),
                }[layout]
            self.viewer.dims.axis_labels = tuple(reversed("xyz"[: scene.env.n_dims]))
            controls = self._controls = QWidget()
            row = QHBoxLayout(controls)
            self.play_button = QPushButton("Play")
            self.time_control = QDoubleSpinBox()
            self.time_control.setDecimals(9)
            self.time_control.setRange(
                float(scene.frame_times[0]), float(scene.frame_times[-1])
            )
            self.time_control.setSingleStep(self._step)
            self.time_control.setSuffix(" s")
            row.addWidget(self.play_button)
            row.addWidget(QLabel("Recording time"))
            row.addWidget(self.time_control)
            self._dock = self.viewer.window.add_dock_widget(
                controls, name="Recording time", area="bottom"
            )
            self.timer = QTimer(controls)
            self.timer.setInterval(max(1, round(1000 / fps)))
            self.timer.timeout.connect(self._tick)
            self.play_button.clicked.connect(self._toggle_play)
            self.time_control.valueChanged.connect(self.seek)
            if self._helper.panel_count:
                self._panels = _ScientificPanels(scene, self._helper)
                self._panel_dock = self.viewer.window.add_dock_widget(
                    self._panels.canvas, name="Scientific panels", area="bottom"
                )
            self.seek(self._time)
            from qtpy.QtCore import QEvent, QObject

            window = self.viewer.window._qt_window
            handle_ref, window_ref = weakref.ref(self), weakref.ref(window)

            class WindowLifecycle(QObject):
                def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
                    if event.type() == QEvent.Type.Close:

                        def after_close() -> None:
                            handle, native_window = handle_ref(), window_ref()
                            if handle is None:
                                return
                            visible = False
                            if native_window is not None:
                                # Native deletion can precede this deferred callback.
                                with suppress(RuntimeError):
                                    visible = native_window.isVisible()
                            if not visible:
                                try:
                                    handle._release_resources()
                                except Exception:
                                    # Exceptions escaping a Qt callback can abort
                                    # the process; keep full cleanup diagnostics.
                                    logging.getLogger(__name__).exception(
                                        "Failed to release closed napari scene resources."
                                    )

                        QTimer.singleShot(0, after_close)
                    return False

            self._lifecycle: Any = WindowLifecycle(window)
            window.installEventFilter(self._lifecycle)
            if not self._owns_viewer and "show" in options:
                window.setVisible(options["show"])
        except BaseException as error:
            from neurospatial.animation._resources import _add_cleanup_note

            try:
                self.close()
            except BaseException as cleanup:
                _add_cleanup_note(
                    error, f"Native napari cleanup also failed: {cleanup!r}"
                )
            raise

    @property
    def time(self) -> float:
        return self._time

    def _add(self, kind: str, data: Any, **kwargs: Any) -> Any:
        layer = getattr(self.viewer, f"add_{kind}")(data, **kwargs)
        self._layers.append(layer)
        return layer

    def _build_fields(self) -> None:
        from matplotlib import colormaps
        from napari.utils.colormaps import Colormap

        native = _native_grid(self.scene.env)
        for name, field in self.scene.fields.items():
            display_name = f"{name} ({field.label})" if field.label else name
            colors = colormaps[field.cmap](np.linspace(0, 1, 256))
            if any(v.z_order == "below" for v in self.scene._overlay_data.videos):
                low, high = field.color_limits()
                colors[:, 3] *= np.clip(
                    np.linspace(low, high, 256) / field._typical_peak(), 0, 1
                )
            cmap = Colormap(colors, name=field.cmap)
            values = field.sample(self._time)
            if native is not None:
                grid, affine = native
                layer = self._add(
                    "image",
                    _image(values, grid),
                    name=display_name,
                    colormap=cmap,
                    contrast_limits=field.color_limits(),
                    affine=affine,
                    blending="translucent",
                    rendering="attenuated_mip",
                )
                self._updates.append(
                    lambda time, samples, layer=layer, name=name, grid=grid: setattr(
                        layer, "data", _image(samples[name], grid)
                    )
                )
            else:
                layer = self._add(
                    "points",
                    self.scene.env.bin_centers[:, ::-1],
                    name=f"{display_name} (bin centers)",
                    features={"value": values},
                    face_color="value",
                    face_colormap=cmap,
                    face_contrast_limits=field.color_limits(),
                    border_width=0,
                    size=5,
                )

                def update(
                    time: float, samples: Any, *, layer: Any = layer, name: str = name
                ) -> None:
                    layer.features = {"value": samples[name]}
                    layer.shown = np.isfinite(samples[name])

                self._updates.append(update)
            self.field_layers[name] = layer

    def _build_overlays(self) -> None:
        from matplotlib import colormaps

        from neurospatial.animation._region_names import shown_region_names
        from neurospatial.animation._style import _event_visibility

        scene, sample = self.scene, self._helper.sample
        data, dims = scene._overlay_data, scene.env.n_dims
        empty = np.empty((0, dims))

        def points(layer: Any, coordinates: Any) -> None:
            values = np.asarray(coordinates, dtype=float).reshape(-1, dims)
            layer.data = values[np.isfinite(values).all(axis=1), ::-1]

        def paths(layer: Any, segments: list[Any]) -> None:
            layer.data = [
                np.asarray(segment)[:, ::-1]
                for segment in segments
                if np.isfinite(segment).all()
            ]

        for i, position in enumerate(data.positions):
            layer = self._add(
                "points",
                empty,
                ndim=dims,
                name=f"Position {i + 1}",
                size=position.size,
                face_color=_rgba(position.color),
            )
            trail = (
                self._add(
                    "shapes",
                    [],
                    ndim=dims,
                    shape_type="line",
                    name=f"Position trail {i + 1}",
                    edge_color=_rgba(position.color),
                    edge_width=2,
                    face_color="transparent",
                )
                if position.trail_length
                else None
            )

            def update_position(
                time: float,
                samples: Any,
                *,
                p: Any = position,
                layer: Any = layer,
                trail: Any = trail,
            ) -> None:
                points(layer, sample(p, p.data, time))
                if trail is not None:
                    times = p.times if p.times is not None else scene.frame_times
                    stop = int(np.searchsorted(times, time, side="right"))
                    start = max(0, stop - p.trail_length)
                    segments = [
                        p.data[j : j + 2]
                        for j in range(start, stop - 1)
                        if p.interval_valid is None or p.interval_valid[j]
                    ]
                    paths(trail, segments)

            self._updates.append(update_position)
        for i, body in enumerate(data.bodypart_sets):
            names = list(body.bodyparts)
            layers = {
                name: self._add(
                    "points",
                    empty,
                    ndim=dims,
                    name=f"Body {i + 1}: {name}",
                    face_color=_rgba(
                        (body.colors or {}).get(
                            name,
                            (
                                (body.skeleton.node_colors or {}).get(name, "red")
                                if body.skeleton
                                else "red"
                            ),
                        )
                    ),
                    size=8,
                )
                for name in names
            }
            skeleton = (
                self._add(
                    "shapes",
                    [],
                    ndim=dims,
                    shape_type="line",
                    name=f"Skeleton {i + 1}",
                    edge_color=_rgba(body.skeleton.edge_color),
                    edge_width=body.skeleton.edge_width,
                )
                if body.skeleton
                else None
            )

            def update_body(
                time: float,
                samples: Any,
                *,
                body: Any = body,
                layers: Any = layers,
                skeleton: Any = skeleton,
            ) -> None:
                current = {
                    name: sample(body, values, time, name=name)
                    for name, values in body.bodyparts.items()
                }
                for name, layer in layers.items():
                    points(layer, current[name])
                if skeleton is not None:
                    paths(
                        skeleton,
                        [
                            np.array([current[a], current[b]])
                            for a, b in body.skeleton.edges
                        ],
                    )

            self._updates.append(update_body)
        for i, heading in enumerate(data.head_directions):
            color = "white" if heading.color in colormaps else _rgba(heading.color)
            layer = self._add(
                "vectors",
                np.empty((0, 2, dims)),
                ndim=dims,
                name=f"Head direction {i + 1}",
                edge_color=color,
                edge_width=heading.width,
                length=1,
            )

            def update_heading(
                time: float, samples: Any, *, h: Any = heading, layer: Any = layer
            ) -> None:
                value = sample(h, h.data, time, circular=h.data.ndim == 1)
                origin = (
                    sample(data.positions[0], data.positions[0].data, time)
                    if len(data.positions) == 1
                    else np.mean(scene.env.bin_centers, axis=0)
                )
                direction = _heading_displacement(value, h.length, n_dims=dims)
                if direction is not None and np.isfinite(origin).all():
                    layer.data = np.array([[origin[::-1], direction[::-1]]])
                    if h.color in colormaps:
                        angle = np.arctan2(direction[1], direction[0])
                        layer.edge_color = np.asarray(
                            colormaps[h.color]((angle % (2 * np.pi)) / (2 * np.pi))
                        )
                else:
                    layer.data = np.empty((0, 2, dims))

            self._updates.append(update_heading)
        for i, vector in enumerate(data.object_vectors):
            layer = self._add(
                "vectors",
                np.empty((0, 2, dims)),
                ndim=dims,
                name=f"Object vectors {i + 1}",
                edge_color=_rgba(vector.color),
                edge_width=vector.linewidth,
                length=1,
            )
            objects = (
                self._add(
                    "points",
                    empty,
                    ndim=dims,
                    name=f"Objects {i + 1}",
                    size=vector.object_size,
                    symbol=_symbol(vector.object_marker),
                    face_color=_rgba(vector.color),
                )
                if vector.show_objects
                else None
            )

            def update_vectors(
                time: float,
                samples: Any,
                *,
                v: Any = vector,
                layer: Any = layer,
                objects: Any = objects,
            ) -> None:
                animal = sample(v, v.animal_positions, time)
                targets = (
                    sample(v, v.object_positions, time)
                    if v.object_positions.ndim == 3
                    else v.object_positions
                )
                targets = targets[np.isfinite(targets).all(axis=1)]
                if objects is not None:
                    points(objects, targets)
                layer.data = (
                    np.array(
                        [[animal[::-1], (target - animal)[::-1]] for target in targets]
                    ).reshape(-1, 2, dims)
                    if np.isfinite(animal).all()
                    else np.empty((0, 2, dims))
                )
                if v.firing_rates is not None:
                    rate = sample(v, v.firing_rates, time)
                    layer.opacity = _rate_opacity(float(rate), v._rate_peak or 1.0)

            self._updates.append(update_vectors)
        for i, events in enumerate(data.events):
            for name, positions in events.event_positions.items():
                layer = self._add(
                    "points",
                    empty,
                    ndim=dims,
                    name=f"Events {i + 1}: {name}",
                    face_color=_rgba(events.colors[name]),
                    size=events.size,
                    symbol=_symbol(events.markers[name]),
                    border_width=events.border_width,
                    border_color=_rgba(events.border_color),
                )

                def update_events(
                    time: float,
                    samples: Any,
                    *,
                    e: Any = events,
                    name: str = name,
                    positions: Any = positions,
                    layer: Any = layer,
                ) -> None:
                    times = e.event_times.get(name)
                    if times is None:
                        times = scene.frame_times[e.event_frame_indices[name]]
                    indices = np.maximum(
                        np.searchsorted(scene.frame_times, times, side="right") - 1, 0
                    )
                    frame = (
                        int(np.searchsorted(scene.frame_times, time, side="right")) - 1
                    )
                    visible, alphas = _event_visibility(indices, frame, e.decay_frames)
                    visible &= (times <= time) & np.isfinite(positions).all(axis=1)
                    layer.data = positions[visible, ::-1]
                    colors = np.tile(
                        _rgba(e.colors[name]), (int(np.count_nonzero(visible)), 1)
                    )
                    colors[:, 3] *= alphas[visible] * e.opacity
                    layer.face_color = colors

                self._updates.append(update_events)
        for i, video in enumerate(data.videos):
            # A custom map may intentionally begin with a missing frame. Use
            # the first mapped image only to establish layer geometry; seeking
            # below decides whether the current observation is visible.
            available = np.flatnonzero(video.frame_indices >= 0)
            image = (
                video.get_frame(int(available[0]))
                if len(available)
                else np.zeros((1, 1, 3), dtype=np.uint8)
            )
            if image is None:
                raise ValueError("VideoOverlay's first mapped frame could not be read.")
            # Video-to-environment transforms are x,y; napari uses y,x.
            if video.transform_to_env is not None:
                matrix = video.transform_to_env.A
            else:
                xmin, xmax, ymin, ymax = video.env_bounds
                matrix = np.array(
                    [
                        [
                            (xmax - xmin) / image.shape[1],
                            0,
                            xmin + (xmax - xmin) / image.shape[1] / 2,
                        ],
                        [
                            0,
                            -(ymax - ymin) / image.shape[0],
                            ymax - (ymax - ymin) / image.shape[0] / 2,
                        ],
                        [0, 0, 1],
                    ]
                )
            order = [1, 0, 2]
            layer = self._add(
                "image",
                image,
                rgb=True,
                name=f"Video {i + 1}",
                opacity=video.alpha,
                affine=matrix[np.ix_(order, order)],
                blending="translucent",
            )
            field_indices = [
                self.viewer.layers.index(field) for field in self.field_layers.values()
            ]
            destination = (
                min(field_indices)
                if video.z_order == "below"
                else max(field_indices) + 1
            )
            self.viewer.layers.move(self.viewer.layers.index(layer), destination)

            def update_video(
                time: float,
                samples: Any,
                *,
                v: Any = video,
                layer: Any = layer,
            ) -> None:
                image = self._helper.video_frame(v, time)
                layer.visible = image is not None
                if image is not None:
                    layer.data = image

            self._updates.append(update_video)
        for name in (
            shown_region_names(scene.env, scene.show_regions)
            if scene.show_regions
            else []
        ):
            region = scene.env.regions[name]
            if region.kind == "point":
                self._add(
                    "points",
                    np.asarray(region.data).reshape(1, dims)[:, ::-1],
                    name=f"Region: {name}",
                    size=10,
                    face_color="white",
                    opacity=scene.region_alpha,
                )
            elif region.kind == "polygon":
                if region.data.interiors:
                    raise ValueError(
                        "Napari region polygons with holes require a raster mask; omit show_regions or use backend='widget'."
                    )
                self._add(
                    "shapes",
                    [np.asarray(region.data.exterior.coords)[:, ::-1]],
                    shape_type="polygon",
                    name=f"Region: {name}",
                    face_color="white",
                    opacity=scene.region_alpha,
                )
            else:
                raise ValueError(
                    f"Napari cannot draw region kind {region.kind!r}; omit show_regions."
                )

    def seek(self, time: float) -> None:
        if self._closed:
            raise RuntimeError("This napari view is closed.")
        time = float(time)
        if (
            not np.isfinite(time)
            or not self.scene.frame_times[0] <= time <= self.scene.frame_times[-1]
        ):
            raise ValueError(
                "time must lie inside the scene's recording clock, in seconds."
            )
        samples = self.scene.sample_fields(time)
        for update in self._updates:
            update(time, samples)
        self._time = time
        self.time_control.blockSignals(True)
        try:
            self.time_control.setValue(time)
        finally:
            self.time_control.blockSignals(False)
        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.text = f"t = {time:.3f} s"
        if self._panels is not None:
            self._panels.update(time)

    def _tick(self) -> None:
        time = min(float(self.scene.frame_times[-1]), self.time + self._step)
        self.seek(time)
        if time >= self.scene.frame_times[-1]:
            self.timer.stop()
            self.play_button.setText("Play")

    def _toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            if self.time >= self.scene.frame_times[-1]:
                self.seek(float(self.scene.frame_times[0]))
            self.timer.start()
            self.play_button.setText("Pause")

    def capture_state(self) -> dict[str, Any]:
        return {"time": self.time, "playing": self.timer.isActive()}

    def restore_state(self, state: dict[str, Any]) -> None:
        if (
            self._closed
            or getattr(self.scene, "_closed", False)
            or getattr(self.scene, "_closing", False)
        ):
            return
        self.seek(state["time"])
        if state["playing"]:
            self.timer.start()
        self.play_button.setText("Pause" if state["playing"] else "Play")

    @contextmanager
    def capture_session(self) -> Iterator[Any]:
        if self._closed:
            raise RuntimeError("This napari view is closed.")
        state = self.capture_state()
        self.timer.stop()
        try:
            yield self.capture_frame
        finally:
            self.restore_state(state)

    def capture_frame(self, time: float) -> np.ndarray:
        from qtpy.QtWidgets import QApplication

        self.seek(time)
        # Process pending native layer/camera draws before a blocking screenshot.
        QApplication.processEvents()
        frame = np.asarray(self.viewer.screenshot(canvas_only=True))[..., :3].copy()
        if self._panels is not None and self._panel_dock.isVisible():
            from PIL import Image

            panel = self._panels.capture()
            height = max(1, round(panel.shape[0] * frame.shape[1] / panel.shape[1]))
            panel = np.asarray(Image.fromarray(panel).resize((frame.shape[1], height)))
            frame = np.concatenate((frame, panel), axis=0)
        return cast("np.ndarray", frame)

    def _release_resources(self) -> None:
        if self._closed:
            return
        self._closed = True
        from neurospatial.animation._resources import _raise_cleanup_errors

        failures: list[BaseException] = []

        def release(action: Any) -> None:
            try:
                action()
            except RuntimeError as exc:
                # Native window closure can already have destroyed Qt children.
                if "deleted" not in str(exc):
                    failures.append(exc)
            except Exception as exc:
                failures.append(exc)

        timer = getattr(self, "timer", None)
        if timer is not None:
            release(timer.stop)
        for control, signal_name, callback in (
            (timer, "timeout", self._tick),
            (getattr(self, "play_button", None), "clicked", self._toggle_play),
            (getattr(self, "time_control", None), "valueChanged", self.seek),
        ):
            if control is not None:

                def disconnect(
                    control: Any = control,
                    signal_name: str = signal_name,
                    callback: Any = callback,
                ) -> None:
                    # Partial construction may not have connected this signal.
                    with suppress(TypeError):
                        getattr(control, signal_name).disconnect(callback)

                release(disconnect)
        native_window = getattr(self.viewer.window, "_qt_window", None)
        lifecycle = getattr(self, "_lifecycle", None)
        if lifecycle is not None:
            if native_window is not None:
                release(lambda: native_window.removeEventFilter(lifecycle))
            release(lambda: lifecycle.setParent(None))
            release(lifecycle.deleteLater)
            self._lifecycle = None
        if self._panels is not None:
            release(self._panels.figure.clear)
        for layer in self._layers:
            if layer in self.viewer.layers:
                release(lambda layer=layer: self.viewer.layers.remove(layer))
        for attr in ("_dock", "_panel_dock"):
            dock = getattr(self, attr, None)
            if dock is not None and native_window is not None:
                release(lambda dock=dock: self.viewer.window.remove_dock_widget(dock))
        for widget in (
            getattr(self, "_controls", None),
            self._panels.canvas if self._panels is not None else None,
        ):
            if widget is not None:
                release(lambda widget=widget: widget.setParent(None))
                release(widget.deleteLater)
        self._updates.clear()
        self._layers.clear()
        self.field_layers.clear()
        _raise_cleanup_errors(
            "Failed to release native napari scene resources.", failures
        )

    def close(self) -> None:
        if self._closed:
            return
        # Releasing scientific controls must not close a caller's viewer.
        try:
            self._release_resources()
        finally:
            if self._owns_viewer:
                self.viewer.close()


def build_napari_scene(animation: Any, **options: Any) -> NapariScene:
    """Create native scalar layers with independent streams on physical time."""
    return NapariScene(animation, **options)
