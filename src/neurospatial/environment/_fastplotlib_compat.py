"""Compatibility boundary for fastplotlib 0.7 NDWidget lifecycle hooks.

NDWidget does not expose registration, playback state or in-flight revision
cancellation publicly. Keep those dependencies here, behind a version and
capability check performed before constructing a viewer. Graphics, buffers,
transforms and volume materials use public fastplotlib/pygfx APIs, with scoped
shader extensions for missing-aware scalar sampling and an upstream iso depth
defect. No built-in material or shader is replaced globally.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    shape: tuple[int, int],
    frame_of: Callable[[float], int],
    read: Callable[[float], Any],
    show: Callable[[float, Any], None],
    graphic: Any,
) -> Any:
    """The field and every frame-aligned panel, as one ``NDGraphic``.

    ``shape`` is ``(n_frames, n_bins)``. The widget's slider is in seconds;
    ``frame_of`` maps it to the frame shown, which is held until the next
    frame starts. ``read`` loads the data for a time (field values, video
    images) on a worker thread, so disk reads do not stall the controls;
    ``show`` then draws all of it together, so the overlays never show
    another frame than the field. Built in a function because fastplotlib
    is optional.
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
            metadata = np.broadcast_to(np.zeros(()), shape)
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


def _volume_shader_source(code: str, common: str) -> str:
    """Separate the scalar R channel from support G, checking each hook."""
    common_include = "{$ include 'pygfx.volume_common.wgsl' $}"
    color_include = "{$ include 'pygfx.image_sample.wgsl' $}"
    if code.count(common_include) != 1 or common.count(color_include) != 1:
        raise ImportError(
            "The installed pygfx volume shader lacks the checked scalar/color "
            "sampling hooks. Use field_rendering='points' or install "
            "pygfx>=0.17,<0.18."
        )
    color = """
fn sampled_value_to_color(value: vec4<f32>) -> vec4<f32> {
    if (value.g == 0.0) { return vec4<f32>(0.0); }
    // Contrast/gamma affect only colors, after scalar selection. Keep the
    // reserved missing row outside finite-value contrast/gamma correction.
    let width = u_material.maprange[1] - u_material.maprange[0];
    let span = select(width, 1.0, width == 0.0);
    let scaled = clamp((value.r - u_material.maprange[0]) / span, 0.0, 1.0);
    let rows = f32(textureDimensions(t_colormap));
    let coordinate = (1.5 + (rows - 2.0) * pow(scaled, u_material.gamma)) / rows;
    return sample_colormap_nomaprange(coordinate);
}
"""
    code = code.replace(common_include, common.replace(color_include, color))
    if "fn raycast(" not in code:  # The slice shader only needs color/support.
        return code
    replacements = {
        # A supported first sample wins even below pygfx's -999999 seed.
        "var the_value : vec4<f32>;": "var the_value = vec4<f32>(0.0);",
        "if (reff > the_ref) {": (
            "if (value.g > 0.0 && (the_value.g == 0.0 || reff > the_ref)) {"
        ),
        "if (reff < the_ref) {": (
            "if (value.g > 0.0 && (the_value.g == 0.0 || reff < the_ref)) {"
        ),
        "if (ref1 >= the_ref) {": "if (value1.g > 0.0 && ref1 >= the_ref) {",
        "if (ref2 > the_ref) {": "if (value2.g > 0.0 && ref2 > the_ref) {",
        "if (ref1 <= the_ref) {": "if (value1.g > 0.0 && ref1 <= the_ref) {",
        "if (ref2 < the_ref) {": "if (value2.g > 0.0 && ref2 < the_ref) {",
        "if (reff > iso_threshold) {": "if (value.g > 0.0 && reff > iso_threshold) {",
        "if (reff < iso_threshold){": "if (value.g == 0.0 || reff < iso_threshold){",
    }
    for old, new in replacements.items():
        expected = 3 if old == "var the_value : vec4<f32>;" else 1
        if code.count(old) != expected:
            raise ImportError(
                "The installed pygfx volume raycaster lacks the checked "
                "missing-support hooks. Use field_rendering='points' or install "
                "pygfx>=0.17,<0.18."
            )
        code = code.replace(old, new)
    return code


# Cache extension registration, never replace the built-in shader/material.


@lru_cache(maxsize=4)
def _scalar_material_type(mode: str) -> Any:
    import pygfx
    from pygfx.renderers.wgpu import load_wgsl, register_wgpu_render_function
    from pygfx.renderers.wgpu.shaders.volumeshader import (
        VolumeRayShader,
        VolumeSliceShader,
    )

    base_material = {
        "mip": pygfx.VolumeMipMaterial,
        "minip": pygfx.VolumeMinipMaterial,
        "iso": pygfx.VolumeIsoMaterial,
        "slice": pygfx.VolumeSliceMaterial,
    }[mode]
    base_shader = VolumeSliceShader if mode == "slice" else VolumeRayShader
    source: str = load_wgsl(
        "volume_slice.wgsl" if mode == "slice" else "volume_ray.wgsl"
    )
    if mode == "iso":
        source = _iso_shader_source(source, pygfx.__version__)
    source = _volume_shader_source(source, load_wgsl("volume_common.wgsl"))

    class ScalarVolumeMaterial(base_material):  # type: ignore[misc,valid-type]
        """Only our volumes carry independent scalar and support channels."""

    @register_wgpu_render_function(pygfx.Volume, ScalarVolumeMaterial)
    class ScalarVolumeShader(base_shader):  # type: ignore[misc,valid-type]
        def __init__(self, wobject: Any) -> None:
            super().__init__(wobject)
            # The scalar color lookup is one dimensional; G is support, not a
            # second colormap coordinate. Texture sampling still reads RG.
            self["img_nchannels"] = 1

        def get_code(self) -> str:
            return source

    return ScalarVolumeMaterial


@lru_cache(maxsize=1)
def _iso_material_type() -> Any:
    return _scalar_material_type("iso")


def require_volume_support(mode: str) -> None:
    """Check scalar sampling and iso depth before viewer/GPU allocation."""
    if mode == "iso":
        _iso_material_type()
    elif mode in {"mip", "slice"}:
        _scalar_material_type(mode)


def configure_volume_material(graphic: Any, texture: Any) -> None:
    """Use public materials and an isolated fastplotlib material-owner bridge.

    Public feature setters in fastplotlib 0.7 write _material, so retain that
    owner reference when replacing materials. Mode changes use the public
    event API and reapply scalar/color sampling to this volume alone.
    """
    import weakref

    def configure(target: Any) -> None:
        children = target.world_object.children
        if not children:
            return
        material = children[0].material
        material_type = (
            _iso_material_type()
            if target.mode == "iso"
            else _scalar_material_type(target.mode)
        )
        if not isinstance(material, material_type):
            # All copied settings are public pygfx material properties.
            names: tuple[str, ...] = (
                "clim",
                "interpolation",
                "gamma",
                "opacity",
                "pick_write",
                "clipping_planes",
                "clipping_mode",
                "alpha_config",
                "alpha_test",
                "alpha_compare",
                "depth_test",
                "depth_compare",
                "depth_write",
            )
            if target.mode == "iso":
                names += (
                    "threshold",
                    "step_size",
                    "substep_size",
                    "emissive",
                    "shininess",
                )
            elif target.mode == "slice":
                names += ("plane",)
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
