"""Compatibility boundary for fastplotlib 0.7 NDWidget lifecycle hooks.

NDWidget does not expose registration, playback state or in-flight revision
cancellation publicly. Keep those dependencies here, behind a version and
capability check performed before constructing a viewer. Graphics, buffers,
transforms and volume materials use public fastplotlib/pygfx APIs, with a
scoped shader extension for a checked upstream isosurface depth defect.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any

from packaging.version import Version


def require_animation_support(fastplotlib: Any) -> None:
    """Fail before allocation when the supported widget contract is unavailable."""
    version = Version(fastplotlib.__version__)
    if not Version("0.7") <= version < Version("0.8"):
        raise ImportError(
            f"fastplotlib {version} is unsupported for animation. "
            "Install neurospatial[fastplotlib] (fastplotlib>=0.7,<0.8)."
        )
    try:
        from fastplotlib.graphics import ImageVolumeGraphic
        from fastplotlib.layouts import Subplot
        from fastplotlib.widgets.nd_widget import (
            NDGraphic,
            NDSlicer,
            NDTimeseries,
            NDWSubplot,
        )
        from fastplotlib.widgets.nd_widget._async import run_in_thread_pool, run_sync
        from fastplotlib.widgets.nd_widget._index import ReferenceIndices
        from fastplotlib.widgets.nd_widget._ui import NDWidgetUI
    except ImportError as error:
        raise ImportError(
            "fastplotlib animation requires the 0.7 NDWidget and imgui-bundle. "
            "Install neurospatial[fastplotlib] with its optional dependencies."
        ) from error
    required = (
        (ImageVolumeGraphic, ("data", "mode", "threshold", "plane", "world_object")),
        (Subplot, ("add_image_volume",)),
        (NDGraphic, ("pause", "indices_displayed")),
        (NDSlicer, ("close",)),
        (NDWSubplot, ("nd_graphics",)),
        (NDTimeseries, ("_update_graphic",)),
        (
            ReferenceIndices,
            ("_fetch_indices", "_schedule_fetch", "_fetch_request_latest"),
        ),
        (NDWidgetUI, ("update",)),
    )
    if (
        not callable(run_in_thread_pool)
        or not callable(run_sync)
        or any(not hasattr(cls, name) for cls, names in required for name in names)
    ):
        raise ImportError(
            "The installed fastplotlib NDWidget lacks required animation hooks. "
            "Reinstall fastplotlib>=0.7,<0.8 and imgui-bundle."
        )


def playback_state(widget: Any) -> dict[str, bool]:
    """Mutable playback state; tolerate a partly constructed viewer during cleanup."""
    return getattr(getattr(widget, "ui_sliders", None), "_playing", {})


def set_playback_fps(widget: Any, fps: float) -> None:
    widget.ui_sliders._fps["time"] = fps
    widget.ui_sliders._frame_time["time"] = 1 / fps


def register_frames(widget: Any, subplot: Any, frames: Any) -> None:
    widget[subplot]._nd_graphics.append(frames)


def mark_displayed(frames: Any, time: float) -> None:
    frames._last_indices = {"time": time}


def update_graphic(graphic: Any) -> None:
    from fastplotlib.widgets.nd_widget._async import run_sync

    run_sync(graphic._set_indices_())


def invalidate_pending(widget: Any, graphics: Sequence[Any]) -> None:
    revisions = widget.indices._fetch_rev
    for graphic in graphics:
        revisions[graphic] = revisions.get(graphic, 0) + 1


def figure_output(figure: Any) -> Any:
    return getattr(figure, "_output", None)


def prepare_figure_close(figure: Any) -> None:
    # Offscreen show() returns None. Figure.close() still needs a closeable output.
    if hasattr(figure, "_output") and figure._output is None:
        figure._output = figure.canvas


def _frames_graphic(
    nd_subplot: Any,
    indices: Any,
    fields: Any,
    frame_of: Callable[[float], int],
    read: Callable[[float], Any],
    show: Callable[[float, Any], None],
    graphic: Any,
) -> Any:
    """The field and every frame-aligned panel, as one ``NDGraphic``.

    The widget's slider is in seconds; ``frame_of`` maps it to the frame
    shown, which is held until the next frame starts. ``read`` loads the
    data for a time (field values, video images) on a worker thread, so disk
    reads do not stall the controls; ``show`` then draws all of it together,
    so the overlays never show another frame than the field. Built in a
    function because fastplotlib is optional.
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from fastplotlib.widgets.nd_widget import NDGraphic, NDSlicer
    from fastplotlib.widgets.nd_widget._async import run_in_thread_pool

    class Frames(NDGraphic):  # type: ignore[misc,unused-ignore]
        def __init__(self) -> None:
            super().__init__(nd_subplot, "field")
            # NDSlicer needs array metadata only: our read callback owns all
            # source reads. A scalar broadcast stores O(1) bytes and avoids
            # imposing fastplotlib's stricter ArrayProtocol on lazy sources.
            metadata = np.broadcast_to(np.zeros(()), fields.shape)
            self._slicer = NDSlicer(
                metadata, ("time", "bin"), ("bin",), slider_maps={"time": frame_of}
            )
            self._graphic = graphic
            self._reader = ThreadPoolExecutor(max_workers=1)
            self._closed = False
            self._generation = 0

        @property
        def slicer(self) -> Any:
            return self._slicer

        @property
        def graphic(self) -> Any:
            return self._graphic

        @property
        def display_dims(self) -> tuple[str, ...]:
            return self._slicer.display_dims  # type: ignore[no-any-return]

        @property
        def indices(self) -> dict[str, float]:
            return {"time": indices["time"]}

        async def _set_indices_(self, at: dict[str, float] | None = None) -> None:
            at = self.indices if at is None else at
            time = at["time"]
            if self._closed:
                return
            generation = self._generation
            result = await run_in_thread_pool(self._reader, read, time)
            if not self._closed and generation == self._generation:
                show(time, result)
                self._last_indices = at

        def synchronize(self) -> None:
            """Invalidate pending writes and finish any read already on the worker."""
            self._generation += 1
            self._reader.submit(lambda: None).result()

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            try:
                self._reader.shutdown(wait=True, cancel_futures=True)
            finally:
                self._slicer.close()

    return Frames()


def _guard_series_updates(series: Any) -> None:
    """Keep an older async series read from replacing the current time window."""
    from concurrent.futures import CancelledError

    update = series._update_graphic

    def update_current(features: Any, indices: Mapping[str, Any]) -> None:
        if dict(indices) != dict(series.indices):
            # The NDWidget scheduler handles this as a superseded fetch.
            # Cancel before it also records these indices as displayed.
            raise CancelledError
        update(features, indices)

    series._update_graphic = update_current


def _iso_shader_source(code: str, version: str) -> str:
    """Correct the known 0.17 translation omission, or recognize its upstream fix."""
    import re

    broken = "let world_pos = world_matrix * data_pos;"
    corrected = (
        "let world_pos = (u_wobject.world_transform * vec4<f32>(data_pos, 1.0)).xyz;"
    )
    iso = code.partition("$$ elif mode == 'iso'")[2]
    if (
        Version("0.17") <= Version(version) < Version("0.18")
        and code.count(broken) == 1
        and "let world_matrix = mat3x3<f32>" in iso
    ):
        return code.replace(broken, corrected)
    if re.search(
        r"let world_pos\s*=\s*[^;]*u_wobject\.world_transform[^;]*data_pos", iso
    ):
        return code
    raise ImportError(
        f"pygfx {version} has an unsupported isosurface shader: correct world "
        "depth cannot be verified. Use field_rendering='mip', 'slice', or 'points', "
        "or install pygfx>=0.17,<0.18 for the checked compatibility fix."
    )


# Cache the extension registration, never replace the built-in shader/material.


@lru_cache(maxsize=1)
def _iso_material_type() -> Any:
    import pygfx
    from pygfx.renderers.wgpu import load_wgsl, register_wgpu_render_function
    from pygfx.renderers.wgpu.shaders.volumeshader import VolumeRayShader

    original = load_wgsl("volume_ray.wgsl")
    corrected = _iso_shader_source(original, pygfx.__version__)
    if corrected == original:
        return None  # upstream already computes world coordinates correctly

    class WorldVolumeIsoMaterial(pygfx.VolumeIsoMaterial):
        """Our volumes alone use the corrected depth and lighting position."""

    @register_wgpu_render_function(pygfx.Volume, WorldVolumeIsoMaterial)
    class WorldVolumeIsoShader(VolumeRayShader):
        def get_code(self) -> str:
            return corrected

    return WorldVolumeIsoMaterial


def require_volume_support(mode: str) -> None:
    """Check iso depth compatibility before viewer/GPU resource allocation."""
    if mode == "iso":
        _iso_material_type()


def configure_volume_material(graphic: Any, texture: Any) -> None:
    """Use public materials and an isolated fastplotlib material-owner bridge.

    Public threshold/step-size setters in fastplotlib 0.7 write _material, so
    retain that owner reference when replacing our iso material. Mode changes
    use the public event API and reapply the fix to this volume alone.
    """
    import weakref

    def configure(target: Any) -> None:
        children = target.world_object.children
        if not children:
            return
        material = children[0].material
        if target.mode == "iso":
            material_type = _iso_material_type()
            if material_type is not None and not isinstance(material, material_type):
                # All copied settings are public pygfx material properties.
                names = (
                    "clim",
                    "interpolation",
                    "gamma",
                    "opacity",
                    "pick_write",
                    "threshold",
                    "step_size",
                    "substep_size",
                    "emissive",
                    "shininess",
                    "clipping_planes",
                    "clipping_mode",
                    "alpha_config",
                    "alpha_test",
                    "alpha_compare",
                    "depth_test",
                    "depth_compare",
                    "depth_write",
                )
                material = material_type(
                    **{name: getattr(material, name) for name in names}
                )
                target._material = material
        material.map = texture
        for tile in children:
            tile.material = material

    configure(graphic)
    reference = weakref.ref(graphic)

    def mode_changed(event: Any) -> None:
        target = reference()
        if target is not None:
            configure(target)

    graphic.add_event_handler(mode_changed, "mode")
