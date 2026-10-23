"""Native napari layers driven by an Animation's physical clock.

The napari spatial canvas and synchronized Matplotlib scientific panels are
captured together. Controls and application chrome are excluded from exports.
Sources and video readers remain owned by the scene, not this view.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from time import monotonic
from typing import Any, NamedTuple, cast

import numpy as np

from neurospatial.animation._backend_view import _CloseNotification
from neurospatial.animation._prepared import PreparedScene
from neurospatial.animation._style import _head_direction_color, _rate_opacity
from neurospatial.animation.backends._napari_compat import (
    canvas_grid,
    canvas_overlay,
    qt_window,
    screen_point_options,
    set_spatial_units,
)


class _PreparedFrame(NamedTuple):
    """Keep arbitrary scientific field names separate from view metadata."""

    fields: dict[str, np.ndarray]
    videos: dict[int, np.ndarray | None]
    panels: Any


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


class _PanelLayers:
    """One overlay's copies in every field panel, assigned together.

    Overlays are sampled once per time; each assignment is applied to every
    panel's copy so all panels show the same observation.
    """

    __slots__ = ("layers",)
    layers: tuple[Any, ...]

    def __init__(self, layers: list[Any]) -> None:
        object.__setattr__(self, "layers", tuple(layers))

    def __setattr__(self, name: str, value: Any) -> None:
        for layer in self.layers:
            setattr(layer, name, value)


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
        # A panel's signals share one cursor, styled by its first signal.
        cursor_styles = [
            (signals[0].show_cursor, signals[0].cursor_color)
            for signals in helper.signal_groups.values()
        ]
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
            ax.set_ylim(*helper.signal_limits(signals))
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
                    from neurospatial.animation._scene_rendering import _colorbar

                    _colorbar(ax, helper.track_scales[name])
        cursor_styles += [(True, "red")] * (len(self.axes) - len(cursor_styles))
        for ax, (visible, color) in zip(self.axes, cursor_styles, strict=True):
            ax.set_facecolor("#20252b")
            ax.set_xlabel("Time (s)")
            self.cursors.append(
                ax.axvline(
                    scene.frame_times[0], color=color, linewidth=1, visible=visible
                )
            )
        from matplotlib.collections import LineCollection, PolyCollection
        from matplotlib.colors import Normalize

        self._track_artists = []
        for ax, name, _field in self.tracks:
            scale = helper.track_scales[name]
            cells = PolyCollection(
                [],
                cmap=scale.cmap,
                norm=Normalize(scale.vmin, scale.vmax),
                edgecolors="none",
            )
            positions = LineCollection([], linewidths=1, zorder=3)
            ax.add_collection(cells)
            ax.add_collection(positions)
            assert helper.track_extents is not None
            starts, stops = helper.track_extents
            ax.set_ylim(float(np.min(starts)), float(np.max(stops)))
            ax.set_ylabel(f"Position ({helper.track_env.units or 'units'})")
            ax.set_title(f"{name} along track")
            self._track_artists.append((cells, positions))
        # Layout is stable across ticks. Repeated tight_layout measures every
        # artist and can dominate playback even when no source I/O is needed.
        self.figure.subplots_adjust(
            left=0.12, right=0.93, bottom=0.1, top=0.94, hspace=0.75
        )

    def prepare(self, time: float) -> dict[str, Any]:
        """Build bounded scientific windows without touching Qt or artists."""
        return {
            "signals": [
                self.helper.signal_window(signal, time) for _, signal in self.signals
            ],
            "rasters": [
                self.helper.raster_window(raster, row, time)
                for _, raster, row in self.rasters
            ],
            "tracks": [
                self.helper.track_window(field, time) for _, _, field in self.tracks
            ],
        }

    def update(self, time: float, prepared: Any = None) -> None:
        prepared = self.prepare(time) if prepared is None else prepared
        left, right = (
            time - self.scene.window_seconds / 2,
            time + self.scene.window_seconds / 2,
        )
        for (line, _), (times, values) in zip(
            self.signals, prepared["signals"], strict=True
        ):
            line.set_data(times, values)
        for (artist, _, row), spikes in zip(
            self.rasters, prepared["rasters"], strict=True
        ):
            segments = np.empty((len(spikes), 2, 2))
            segments[:, :, 0] = spikes[:, None]
            segments[:, :, 1] = [row - 0.4, row + 0.4]
            artist.set_segments(segments)
        for (cells, positions), window in zip(
            self._track_artists, prepared["tracks"], strict=True
        ):
            cells.set_verts(window.polygons)
            cells.set_array(window.values)
            positions.set_segments([segment for segment, _ in window.positions])
            positions.set_color([color for _, color in window.positions])
        for ax, cursor in zip(self.axes, self.cursors, strict=True):
            ax.set_xlim(left, right)
            cursor.set_xdata([time, time])
        self.canvas.draw_idle()

    def capture(self) -> np.ndarray:
        if self.figure.stale:
            self.canvas.draw()
        return cast("np.ndarray", np.asarray(self.canvas.buffer_rgba())[..., :3].copy())


class NapariScene:
    """One persistent native viewer and absolute recording-time controls."""

    def __init__(self, scene: Any, **options: Any) -> None:
        try:
            import napari
        except ImportError as error:
            raise ImportError(
                "The napari view needs napari. Install it with "
                "pip install 'neurospatial[animation]', or use "
                "backend='widget' in a notebook or backend='fastplotlib'."
            ) from error
        from qtpy.QtCore import (  # type: ignore[attr-defined]
            QEvent,
            QObject,
            Qt,
            QTimer,
            Signal,
        )
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
        self._close_notification = _CloseNotification()
        self._time = float(scene.frame_times[0])
        self._speed = float(options.get("speed", 1))
        fps = float(options.get("fps", 30))
        if not np.isfinite([self._speed, fps]).all() or self._speed <= 0 or fps <= 0:
            raise ValueError("fps and speed must be positive finite numbers.")
        self._step = self._speed / fps
        self._helper = PreparedScene(scene)
        self._panels: _ScientificPanels | None = None
        self._source_lock = self._helper.source_lock
        self._inflight: Future[Any] | None = None
        self._pending: tuple[int, float] | None = None
        self._generation = 0
        self._capturing = False
        self.last_error: BaseException | None = None
        self._last_samples: dict[str, np.ndarray] = {}
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
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="napari-scene"
            )
            if not self._owns_viewer:
                if "title" in options:
                    self.viewer.title = options["title"]
                if "ndisplay" in options:
                    self.viewer.dims.ndisplay = options["ndisplay"]
            # Each stream retains a separate layer, scale and colormap. A layout
            # gives every field its own panel holding copies of the overlays.
            layout = options.get("layout")
            if layout not in (None, "horizontal", "vertical", "grid"):
                raise ValueError("layout must be 'horizontal', 'vertical' or 'grid'.")
            self._build_fields(split=layout is not None)
            self._build_overlays()
            if layout is not None:
                self._arrange_panels(layout)
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
            self.time_control.valueChanged.connect(self._request_seek)

            class PreparedBridge(QObject):
                ready = Signal(object)

            self._bridge = PreparedBridge(controls)
            self._bridge.ready.connect(
                self._prepared_ready, Qt.ConnectionType.QueuedConnection
            )
            if self._helper.panel_count:
                self._panels = _ScientificPanels(scene, self._helper)
                self._panel_dock = self.viewer.window.add_dock_widget(
                    self._panels.canvas, name="Scientific panels", area="bottom"
                )
            self.seek(self._time)
            window = qt_window(self.viewer)
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

    def _add_close_callback(self, callback: Callable[[], None]) -> None:
        self._close_notification.add(callback)

    def _add(self, kind: str, data: Any, **kwargs: Any) -> Any:
        layer = getattr(self.viewer, f"add_{kind}")(data, **kwargs)
        self._layers.append(layer)
        set_spatial_units(layer, self.scene.env.units, self.scene.env.n_dims)
        return layer

    def _add_overlay(
        self, kind: str, data: Any, *, name: str, **kwargs: Any
    ) -> _PanelLayers:
        """Add one overlay layer to every panel; updates fan out to each copy."""
        copies = []
        for panel, stack in self._panel_stacks.items():
            label = name if panel is None else f"{panel}: {name}"
            layer = self._add(kind, data, name=label, **kwargs)
            stack.append(layer)
            copies.append(layer)
        return _PanelLayers(copies)

    def _arrange_panels(self, layout: str) -> None:
        """Group each field with its overlay copies in one native grid panel.

        Napari places ``stride`` consecutive layers in each grid panel, so the
        panels' layers lead the layer list in field order and keep their
        single-panel stacking (video below or above the field, overlays and
        regions on top). Layers added later occupy panels after the fields.
        """
        layers = self.viewer.layers
        stacks = list(self._panel_stacks.values())
        order = [layer for stack in stacks for layer in sorted(stack, key=layers.index)]
        for index, layer in enumerate(order):
            layers.move(layers.index(layer), index)
        grid = canvas_grid(self.viewer)
        grid.stride = len(stacks[0])
        grid.shape = {
            "horizontal": (1, len(stacks)),
            "vertical": (len(stacks), 1),
            "grid": (-1, -1),
        }[layout]
        grid.enabled = True
        self.viewer.reset_view()

    def _build_fields(self, *, split: bool = False) -> None:
        from napari.utils.colormaps import Colormap

        native = _native_grid(self.scene.env)
        # Keep owned initial samples serialized with other views, then release
        # the lock before creating GUI layers. The prepared scene read the
        # scale statistics under the same lock.
        with self._helper.source_lock:
            self.scene._ensure_open()
            samples = self._helper.field_samples(self._time)
        for name, field in self.scene.fields.items():
            display_name = f"{name} ({field.label})" if field.label else name
            scale = self._helper.field_scales[name]
            limits = (scale.vmin, scale.vmax)
            cmap = Colormap(scale.colors(), name=scale.cmap)
            values = samples[name]
            if native is not None:
                grid, affine = native
                layer = self._add(
                    "image",
                    _image(values, grid),
                    name=display_name,
                    colormap=cmap,
                    contrast_limits=limits,
                    affine=affine,
                    blending="translucent",
                    rendering="attenuated_mip",
                )

                def update_image(
                    time: float,
                    samples: Any,
                    *,
                    layer: Any = layer,
                    name: str = name,
                    grid: Any = grid,
                ) -> None:
                    if (
                        getattr(self, "_last_samples", {}).get(name)
                        is not samples.fields[name]
                    ):
                        layer.data = _image(samples.fields[name], grid)

                self._updates.append(update_image)
            else:
                layer = self._add(
                    "points",
                    self.scene.env.bin_centers[:, ::-1],
                    name=f"{display_name} (bin centers)",
                    features={"value": values},
                    face_color="value",
                    face_colormap=cmap,
                    face_contrast_limits=limits,
                    border_width=0,
                    size=5,
                )

                def update(
                    time: float, samples: Any, *, layer: Any = layer, name: str = name
                ) -> None:
                    if (
                        getattr(self, "_last_samples", {}).get(name)
                        is not samples.fields[name]
                    ):
                        layer.features = {"value": samples.fields[name]}
                        layer.shown = np.isfinite(samples.fields[name])

                self._updates.append(update)
            self.field_layers[name] = layer
        # Without a layout, all fields share one panel and one overlay stack.
        self._panel_stacks: dict[str | None, list[Any]] = (
            {name: [layer] for name, layer in self.field_layers.items()}
            if split
            else {None: list(self.field_layers.values())}
        )

    def _build_overlays(self) -> None:
        from matplotlib import colormaps

        from neurospatial.animation._region_names import shown_region_names

        scene, sample = self.scene, self._helper.sample
        data, dims = scene._overlay_data, scene.env.n_dims
        empty = np.empty((0, dims))

        def points(layer: Any, coordinates: Any) -> None:
            values = np.asarray(coordinates, dtype=float).reshape(-1, dims)
            layer.data = values[np.isfinite(values).all(axis=1), ::-1]

        def paths(
            layer: Any,
            segments: list[Any],
            alphas: np.ndarray | None = None,
            color: Any = None,
        ) -> None:
            keep = [
                np.isfinite(segment).all() and bool(np.any(np.diff(segment, axis=0)))
                for segment in segments
            ]
            layer.data = [
                np.asarray(segment)[:, ::-1]
                for segment, kept in zip(segments, keep, strict=True)
                if kept
            ]
            if alphas is not None and any(keep):
                # Each segment fades with its age, as in the other views.
                colors = np.tile(np.asarray(color, dtype=float), (sum(keep), 1))
                colors[:, 3] = np.asarray(alphas)[keep]
                layer.edge_color = colors

        for i, position in enumerate(data.positions):
            layer = self._add_overlay(
                "points",
                empty,
                ndim=dims,
                name=f"Position {i + 1}",
                **screen_point_options(position.size),
                face_color=_rgba(position.color),
            )
            trail = (
                self._add_overlay(
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
                    paths(trail, *self._helper.trail(p, time), _rgba(p.color))

            self._updates.append(update_position)
        for i, body in enumerate(data.bodypart_sets):
            names = list(body.bodyparts)
            layers = {
                name: self._add_overlay(
                    "points",
                    empty,
                    ndim=dims,
                    name=f"Body {i + 1}: {name}",
                    face_color=_rgba(body.colors[name]),
                    **screen_point_options(8),
                )
                for name in names
            }
            skeleton = (
                self._add_overlay(
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
            layer = self._add_overlay(
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
                arrow = self._helper.heading_arrow(h, time)
                if arrow is not None:
                    origin, direction = arrow
                    layer.data = np.array([[origin[::-1], direction[::-1]]])
                    if h.color in colormaps:
                        layer.edge_color = np.asarray(
                            _head_direction_color(h.color, direction)
                        )
                else:
                    layer.data = np.empty((0, 2, dims))

            self._updates.append(update_heading)
        for i, vector in enumerate(data.object_vectors):
            layer = self._add_overlay(
                "vectors",
                np.empty((0, 2, dims)),
                ndim=dims,
                name=f"Object vectors {i + 1}",
                edge_color=_rgba(vector.color),
                edge_width=vector.linewidth,
                length=1,
            )
            objects = (
                self._add_overlay(
                    "points",
                    empty,
                    ndim=dims,
                    name=f"Objects {i + 1}",
                    **screen_point_options(vector.object_size),
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
                    rate = sample(v, v.firing_rates, time, name="firing_rates")
                    layer.opacity = _rate_opacity(float(rate), v._rate_peak or 1.0)

            self._updates.append(update_vectors)
        for i, events in enumerate(data.events):
            layers = {
                name: self._add_overlay(
                    "points",
                    empty,
                    ndim=dims,
                    name=f"Events {i + 1}: {name}",
                    face_color=_rgba(events.colors[name]),
                    **screen_point_options(events.size),
                    symbol=_symbol(events.markers[name]),
                    border_width=events.border_width,
                    border_color=_rgba(events.border_color),
                )
                for name in events.event_positions
            }
            drawn: dict[str, tuple[int, ...]] = {}

            def update_events(
                time: float,
                samples: Any,
                *,
                e: Any = events,
                layers: Any = layers,
                drawn: Any = drawn,
            ) -> None:
                for name, shown in self._helper.events(e, time).items():
                    if drawn.get(name) == shown.key:
                        continue
                    drawn[name] = shown.key
                    layers[name].data = shown.positions[:, ::-1]
                    colors = np.tile(_rgba(e.colors[name]), (len(shown.opacity), 1))
                    colors[:, 3] *= shown.opacity
                    layers[name].face_color = colors

            self._updates.append(update_events)
        for i, video in enumerate(data.videos):
            # A custom map may intentionally begin with a missing frame. Use
            # the first mapped image only to establish layer geometry; seeking
            # below decides whether the current observation is visible.
            available = np.flatnonzero(video.frame_indices >= 0)
            image = (
                self._helper.video_frame(video, float(video.times[available[0]]))
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
            layer = self._add_overlay(
                "image",
                image,
                rgb=True,
                name=f"Video {i + 1}",
                opacity=video.alpha,
                affine=matrix[np.ix_(order, order)],
                blending="translucent",
            )
            for copy in layer.layers:
                field_indices = [
                    self.viewer.layers.index(field)
                    for field in self.field_layers.values()
                ]
                destination = (
                    min(field_indices)
                    if video.z_order == "below"
                    else max(field_indices) + 1
                )
                self.viewer.layers.move(self.viewer.layers.index(copy), destination)

            def update_video(
                time: float,
                samples: Any,
                *,
                v: Any = video,
                layer: Any = layer,
            ) -> None:
                image = samples.videos[id(v)]
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
                self._add_overlay(
                    "points",
                    np.asarray(region.data).reshape(1, dims)[:, ::-1],
                    name=f"Region: {name}",
                    **screen_point_options(10),
                    face_color="white",
                    opacity=scene.region_alpha,
                )
            elif region.kind == "polygon":
                if region.data.interiors:
                    raise ValueError(
                        "Napari region polygons with holes require a raster mask; omit show_regions or use backend='widget'."
                    )
                self._add_overlay(
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

    def _validate_time(self, time: float) -> float:
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
        return time

    def _prepare(self, time: float) -> _PreparedFrame:
        # Decoders and lazy source adapters need not support concurrent reads.
        # Synchronous seek, worker jobs and close share this source lifetime.
        with getattr(self, "_source_lock", self._helper.source_lock):
            return _PreparedFrame(
                self._helper.field_samples(time),
                {
                    id(video): self._helper.video_frame(video, time)
                    for video in self.scene._overlay_data.videos
                },
                self._panels.prepare(time) if self._panels is not None else None,
            )

    def _apply(self, time: float, samples: _PreparedFrame) -> None:
        for update in self._updates:
            update(time, samples)
        self._last_samples = {name: samples.fields[name] for name in self.scene.fields}
        self._time = time
        self.time_control.blockSignals(True)
        try:
            self.time_control.setValue(time)
        finally:
            self.time_control.blockSignals(False)
        canvas_overlay(self.viewer, "text").visible = True
        canvas_overlay(self.viewer, "text").text = f"t = {time:.3f} s"
        if self._panels is not None:
            self._panels.update(time, samples.panels)
        self.last_error = None

    def _invalidate_pending(self) -> None:
        self._generation = getattr(self, "_generation", 0) + 1
        self._pending = None

    def seek(self, time: float) -> None:
        """Synchronously prepare and apply an exact recording-time observation."""
        time = self._validate_time(time)
        self._invalidate_pending()
        self._apply(time, self._prepare(time))
        if getattr(self, "timer", None) is not None and self.timer.isActive():
            self._play_origin, self._play_started = time, monotonic()

    def _request_seek(self, time: float, *, playback: bool = False) -> None:
        """GUI-only request: retain one in-flight job and the latest target."""
        if self._closed or self._capturing:
            return
        time = self._validate_time(time)
        if not playback:
            self._generation += 1
            if self.timer.isActive():
                self._play_origin, self._play_started = time, monotonic()
        # Playback requests share an epoch: a completed frame can advance the
        # display while a newer wall-clock target waits. A manual scrub or
        # synchronous seek invalidates that epoch and discards its results.
        request = (self._generation, time)
        if self._inflight is not None:
            self._pending = request
        else:
            self._submit(request)

    def _submit(self, request: tuple[int, float]) -> None:
        generation, time = request
        future = self._executor.submit(self._prepare, time)
        self._inflight = future
        bridge = self._bridge

        def completed(done: Future[Any]) -> None:
            # No layer, artist or widget mutation on a worker thread. The
            # queued signal is delivered only by the GUI event loop.
            with suppress(RuntimeError):
                bridge.ready.emit((generation, time, done))

        future.add_done_callback(completed)

    def _prepared_ready(self, result: tuple[int, float, Future[Any]]) -> None:
        generation, time, future = result
        if future is self._inflight:
            self._inflight = None
        if self._closed:
            return
        if generation == self._generation and not self._capturing:
            try:
                self._apply(time, future.result())
            except Exception as error:
                self.last_error = error
                text = canvas_overlay(self.viewer, "text")
                text.visible = True
                text.text = f"Read failed at {time:.3f} s: {error}\nLast frame t = {self.time:.3f} s"
                self.timer.stop()
                self.play_button.setText("Play")
                logging.getLogger(__name__).exception(
                    "Napari scene preparation failed at %.6f s", time
                )
        pending, self._pending = self._pending, None
        if pending is not None and not self._capturing:
            self._submit(pending)

    def _tick(self) -> None:
        if not hasattr(self, "_play_started"):
            self._play_origin, self._play_started = self.time, monotonic()
        time = min(
            float(self.scene.frame_times[-1]),
            self._play_origin + (monotonic() - self._play_started) * self._speed,
        )
        self._request_seek(time, playback=True)
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
            self._play_origin, self._play_started = self.time, monotonic()
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
            self._play_origin, self._play_started = self.time, monotonic()
            self.timer.start()
        self.play_button.setText("Pause" if state["playing"] else "Play")

    @contextmanager
    def capture_session(self) -> Iterator[Any]:
        if self._closed:
            raise RuntimeError("This napari view is closed.")
        state = self.capture_state()
        self.timer.stop()
        was_capturing = getattr(self, "_capturing", False)
        self._capturing = True
        self._invalidate_pending()
        try:
            yield self.capture_frame
        finally:
            self._capturing = was_capturing
            self.restore_state(state)

    def capture_frame(self, time: float) -> np.ndarray:
        from qtpy.QtWidgets import QApplication

        if not self._capturing:
            with self.capture_session() as capture:
                return cast("np.ndarray", capture(time))
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
        self._invalidate_pending()
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
        executor = getattr(self, "_executor", None)
        if executor is not None:
            # Drain reads before Animation.close closes its owned video readers.
            release(lambda: executor.shutdown(wait=True, cancel_futures=True))
        bridge = getattr(self, "_bridge", None)
        if bridge is not None:
            release(lambda: bridge.ready.disconnect(self._prepared_ready))
        self._inflight = None
        for control, signal_name, callback in (
            (timer, "timeout", self._tick),
            (getattr(self, "play_button", None), "clicked", self._toggle_play),
            (getattr(self, "time_control", None), "valueChanged", self._request_seek),
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
        native_window = qt_window(self.viewer)
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
        release(self._helper.clear_cache)
        self._last_samples = {}
        self.last_error = None
        self._panels = None
        self._updates.clear()
        self._layers.clear()
        self.field_layers.clear()
        release(self._close_notification.notify)
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
