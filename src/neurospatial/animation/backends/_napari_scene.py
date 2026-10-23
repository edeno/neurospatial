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
from functools import partial
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


def _unless_deleted(action: Callable[[], object]) -> Callable[[], None]:
    """``action``, ignoring Qt children a native window closure destroyed."""

    def run() -> None:
        try:
            action()
        except RuntimeError as error:
            if "deleted" not in str(error):
                raise

    return run


class _PreparedFrame(NamedTuple):
    """One time's layer and panel data, computed on the worker thread."""

    layers: list[Any]
    panels: Any


class _LayerUpdate(NamedTuple):
    """One layer group's update at a time.

    ``prepare`` computes the group's data on the worker thread; ``apply``
    only assigns it to the layers, on the GUI thread.
    """

    prepare: Callable[[float], Any]
    apply: Callable[[Any], None]


def _native_grid(env: Any) -> tuple[Any, np.ndarray] | None:
    """Uniform source cells and their pixel-center to napari-world affine.

    Napari orders axes in reverse: ``(y, x)`` and ``(z, y, x)``.
    """
    from neurospatial.environment._fastplotlib_volume import uniform_grid

    grid = uniform_grid(env)
    if grid is None:
        return None
    order = [*range(env.n_dims - 1, -1, -1), env.n_dims]
    return grid, grid.matrix[np.ix_(order, order)]


def _image(values: np.ndarray, grid: Any) -> np.ndarray:
    image = np.full(grid.active_mask.shape, np.nan, dtype=np.float32)
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
    """Persistent Matplotlib artists, so native annotations survive seeking.

    The panels are the scene renderer's; this dock stacks them in one column.
    """

    def __init__(self, scene: Any, helper: PreparedScene) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from neurospatial.animation._scene_rendering import (
            Panel,
            epoch_panel,
            raster_panel,
            signal_panel,
            track_panel,
        )

        # One track panel per stream.
        count = helper.panel_count + (
            len(scene.fields) - 1 if helper.track_extents is not None else 0
        )
        self.figure = Figure(figsize=(8, max(1.5, 1.5 * count)), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes: list[Any] = []

        def axis() -> Any:
            self.axes.append(self.figure.add_subplot(count, 1, len(self.axes) + 1))
            return self.axes[-1]

        self.panels: list[Panel] = [
            signal_panel(axis(), helper, signals)
            for signals in helper.signal_groups.values()
        ]
        self.panels += [
            raster_panel(axis(), helper, raster)
            for raster in scene._overlay_data.spike_rasters
        ]
        if scene._overlay_data.epochs:
            self.panels.append(epoch_panel(axis(), helper))
        if helper.track_extents is not None:
            self.panels += [
                track_panel(axis(), helper, name, field)
                for name, field in scene.fields.items()
            ]
        # Layout is stable across ticks. Repeated tight_layout measures every
        # artist and can dominate playback even when no source I/O is needed.
        self.figure.subplots_adjust(
            left=0.12, right=0.93, bottom=0.1, top=0.94, hspace=0.75
        )

    def prepare(self, time: float) -> list[Any]:
        """Read each panel's data at ``time`` without touching Qt or artists."""
        return [panel.prepare(time) for panel in self.panels]

    def update(self, time: float, prepared: list[Any]) -> None:
        for panel, data in zip(self.panels, prepared, strict=True):
            panel.apply(time, data)
        self.canvas.draw_idle()

    def capture(self) -> np.ndarray:
        if self.figure.stale:
            self.canvas.draw()
        return cast("np.ndarray", np.asarray(self.canvas.buffer_rgba())[..., :3].copy())


class NapariScene:
    """One persistent native viewer and absolute recording-time controls.

    ``Animation.show(backend="napari")`` checks the request (options,
    dimensions and regions) before creating the viewer.
    """

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
        self.scene, self._closed = scene, False
        self._close_notification = _CloseNotification()
        self._time = float(scene.frame_times[0])
        # require_scene_view checked speed and fps.
        self._speed = float(options.get("speed", 1))
        fps = float(options.get("fps", 30))
        self._step = self._speed / fps
        # Playback advances from this recording time at this wall-clock time.
        self._play_origin, self._play_started = self._time, monotonic()
        self._helper = PreparedScene(scene)
        self._panels: _ScientificPanels | None = None
        self._inflight: Future[Any] | None = None
        self._pending: tuple[int, float] | None = None
        self._generation = 0
        self._capturing = False
        self.last_error: BaseException | None = None
        self.viewer: Any = options.get("viewer")
        self._owns_viewer = self.viewer is None
        if self.viewer is None:
            self.viewer = napari.Viewer(
                title=options.get("title", scene.title or "Animation"),
                ndisplay=options.get("ndisplay", scene.env.n_dims),
                show=options.get("show", True),
            )
        self._layers: list[Any] = []
        self._updates: list[_LayerUpdate] = []
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
            from neurospatial.animation._resources import _close_keeping_error

            _close_keeping_error(self.close, error, what="Native napari")
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

                def assign(
                    values: np.ndarray, layer: Any = layer, grid: Any = grid
                ) -> None:
                    layer.data = _image(values, grid)

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

                def assign(
                    values: np.ndarray, layer: Any = layer, grid: Any = None
                ) -> None:
                    layer.features = {"value": values}
                    layer.shown = np.isfinite(values)

            self._updates.append(self._field_update(field, assign))
            self.field_layers[name] = layer
        # Without a layout, all fields share one panel and one overlay stack.
        self._panel_stacks: dict[str | None, list[Any]] = (
            {name: [layer] for name, layer in self.field_layers.items()}
            if split
            else {None: list(self.field_layers.values())}
        )

    def _field_update(
        self, field: Any, assign: Callable[[np.ndarray], None]
    ) -> _LayerUpdate:
        """A field layer's update; a held row is not assigned again."""
        drawn: list[np.ndarray | None] = [None]

        def apply(values: np.ndarray) -> None:
            if values is not drawn[0]:
                assign(values)
                drawn[0] = values

        return _LayerUpdate(lambda time: self._helper.field_sample(field, time), apply)

    def _build_overlays(self) -> None:
        from matplotlib import colormaps

        from neurospatial.animation._region_names import shown_region_names

        scene, helper = self.scene, self._helper
        data, dims = scene._overlay_data, scene.env.n_dims
        empty = np.empty((0, dims))
        no_vectors = np.empty((0, 2, dims))

        def points(coordinates: Any) -> np.ndarray:
            """Finite points in napari's (..., y, x) order."""
            values = np.asarray(coordinates, dtype=float).reshape(-1, dims)
            return values[np.isfinite(values).all(axis=1), ::-1]

        def paths(segments: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
            """Drawable segments in napari order, and which were kept."""
            keep = np.isfinite(segments).all(axis=(1, 2)) & np.any(
                np.diff(segments, axis=1) != 0, axis=(1, 2)
            )
            return list(segments[keep][:, :, ::-1]), keep

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

            def prepare_position(time: float, p: Any = position) -> Any:
                marker = points(helper.sample(p, p.data, time))
                if not p.trail_length:
                    return marker, None
                segments, alphas = helper.trail(p, time)
                shown, keep = paths(segments)
                # Each segment fades with its age, as in the other views.
                colors = np.tile(_rgba(p.color), (len(shown), 1))
                colors[:, 3] = alphas[keep]
                return marker, (shown, colors)

            def apply_position(
                prepared: Any, layer: Any = layer, trail: Any = trail
            ) -> None:
                marker, segments = prepared
                layer.data = marker
                if trail is not None:
                    shown, colors = segments
                    trail.data = shown
                    if len(colors):
                        trail.edge_color = colors

            self._updates.append(_LayerUpdate(prepare_position, apply_position))
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

            def prepare_body(time: float, body: Any = body) -> Any:
                pose = helper.pose(body, time)
                return [points(point) for point in pose.points], paths(pose.skeleton)[0]

            def apply_body(
                prepared: Any, layers: Any = layers, skeleton: Any = skeleton
            ) -> None:
                parts, edges = prepared
                for layer, part in zip(layers.values(), parts, strict=True):
                    layer.data = part
                if skeleton is not None:
                    skeleton.data = edges

            self._updates.append(_LayerUpdate(prepare_body, apply_body))
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

            def prepare_heading(time: float, h: Any = heading) -> Any:
                arrow = helper.heading_arrow(h, time)
                if arrow is None:
                    return no_vectors, None
                origin, direction = arrow
                hue = (
                    np.asarray(_head_direction_color(h.color, direction))
                    if h.color in colormaps
                    else None
                )
                return np.array([[origin[::-1], direction[::-1]]]), hue

            def apply_heading(prepared: Any, layer: Any = layer) -> None:
                arrow, hue = prepared
                layer.data = arrow
                if hue is not None:
                    layer.edge_color = hue

            self._updates.append(_LayerUpdate(prepare_heading, apply_heading))
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

            def prepare_vectors(time: float, v: Any = vector) -> Any:
                animal = helper.sample(v, v.animal_positions, time)
                targets = (
                    helper.sample(v, v.object_positions, time)
                    if v.object_positions.ndim == 3
                    else v.object_positions
                )
                targets = targets[np.isfinite(targets).all(axis=1)]
                vectors = no_vectors
                if np.isfinite(animal).all():
                    vectors = np.stack(
                        [np.broadcast_to(animal, targets.shape), targets - animal],
                        axis=1,
                    )[:, :, ::-1]
                opacity = None
                if v.firing_rates is not None:
                    rate = helper.sample(v, v.firing_rates, time, name="firing_rates")
                    opacity = _rate_opacity(float(rate), v._rate_peak)
                return points(targets), vectors, opacity

            def apply_vectors(
                prepared: Any, layer: Any = layer, objects: Any = objects
            ) -> None:
                targets, vectors, opacity = prepared
                layer.data = vectors
                if objects is not None:
                    objects.data = targets
                if opacity is not None:
                    layer.opacity = opacity

            self._updates.append(_LayerUpdate(prepare_vectors, apply_vectors))
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

            def prepare_events(time: float, e: Any = events) -> Any:
                prepared = {}
                for name, shown in helper.events(e, time).items():
                    colors = np.tile(_rgba(e.colors[name]), (len(shown.opacity), 1))
                    colors[:, 3] *= shown.opacity
                    prepared[name] = (shown.key, shown.positions[:, ::-1], colors)
                return prepared

            def apply_events(
                prepared: Any, layers: Any = layers, drawn: Any = drawn
            ) -> None:
                # Markers are assigned only when the shown events change.
                for name, (key, positions, colors) in prepared.items():
                    if drawn.get(name) != key:
                        layers[name].data = positions
                        layers[name].face_color = colors
                        drawn[name] = key

            self._updates.append(_LayerUpdate(prepare_events, apply_events))
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
            matrix = video.pixel_to_env(image.shape[1], image.shape[0])
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

            def apply_video(image: Any, layer: Any = layer) -> None:
                layer.visible = image is not None
                if image is not None:
                    layer.data = image

            def prepare_video(time: float, v: Any = video) -> Any:
                return helper.video_frame(v, time)

            self._updates.append(_LayerUpdate(prepare_video, apply_video))
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
            else:
                # The view request admitted only point and hole-free polygons.
                self._add_overlay(
                    "shapes",
                    [np.asarray(region.data.exterior.coords)[:, ::-1]],
                    shape_type="polygon",
                    name=f"Region: {name}",
                    face_color="white",
                    opacity=scene.region_alpha,
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
        """Sample, read and lay out every layer and panel; no Qt or artists.

        Runs on the worker thread. Decoders and lazy source adapters need not
        support concurrent reads: synchronous seek, worker jobs and close
        share the scene's source lock.
        """
        with self._helper.source_lock:
            return _PreparedFrame(
                [update.prepare(time) for update in self._updates],
                self._panels.prepare(time) if self._panels is not None else None,
            )

    def _apply(self, time: float, samples: _PreparedFrame) -> None:
        """Assign prepared data to the layers and panels, on the GUI thread."""
        for update, prepared in zip(self._updates, samples.layers, strict=True):
            update.apply(prepared)
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
        self._generation += 1
        self._pending = None

    def seek(self, time: float) -> None:
        """Synchronously prepare and apply an exact recording-time observation."""
        time = self._validate_time(time)
        self._invalidate_pending()
        self._apply(time, self._prepare(time))
        if self.timer.isActive():
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
        if self._closed or self.scene._closed or self.scene._closing:
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
        was_capturing = self._capturing
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
        from neurospatial.animation._resources import _run_all

        _run_all(
            map(_unless_deleted, self._release_actions()),
            "Failed to release native napari scene resources.",
        )

    def _release_actions(self) -> Iterator[Callable[[], object]]:
        """Each release step, after the previous one ran; for partial builds too."""
        timer = getattr(self, "timer", None)
        if timer is not None:
            yield timer.stop
        executor = getattr(self, "_executor", None)
        if executor is not None:
            # Drain reads before Animation.close closes its owned video readers.
            yield lambda: executor.shutdown(wait=True, cancel_futures=True)
        bridge = getattr(self, "_bridge", None)
        if bridge is not None:
            yield lambda: bridge.ready.disconnect(self._prepared_ready)
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

                yield disconnect
        native_window = qt_window(self.viewer)
        lifecycle = getattr(self, "_lifecycle", None)
        if lifecycle is not None:
            if native_window is not None:
                yield lambda: native_window.removeEventFilter(lifecycle)
            yield lambda: lifecycle.setParent(None)
            yield lifecycle.deleteLater
            self._lifecycle = None
        if self._panels is not None:
            yield self._panels.figure.clear
        for layer in self._layers:
            if layer in self.viewer.layers:
                yield partial(self.viewer.layers.remove, layer)
        for attr in ("_dock", "_panel_dock"):
            dock = getattr(self, attr, None)
            if dock is not None and native_window is not None:
                yield partial(self.viewer.window.remove_dock_widget, dock)
        for widget in (
            getattr(self, "_controls", None),
            self._panels.canvas if self._panels is not None else None,
        ):
            if widget is not None:
                yield partial(widget.setParent, None)
                yield widget.deleteLater
        yield self._helper.clear_cache
        self.last_error = None
        self._panels = None
        self._updates.clear()
        self._layers.clear()
        self.field_layers.clear()
        yield self._close_notification.notify

    def close(self) -> None:
        if self._closed:
            return
        # Releasing scientific controls must not close a caller's viewer.
        try:
            self._release_resources()
        finally:
            if self._owns_viewer:
                self.viewer.close()
