"""Volume coordinates, missing support and allocation-free validation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.environment._fastplotlib import ColorScale, FieldGraphic
from neurospatial.environment._fastplotlib_volume import (
    resolve_field_rendering,
    volume_grid,
)


@pytest.fixture
def grid_env():
    mask = np.ones((3, 4, 5), dtype=bool)
    mask[1, 1, 1] = False
    return Environment.from_grid_mask(
        mask,
        grid_edges=(
            10 + np.arange(4) * 2.0,
            30 + np.arange(5) * 3.0,
            50 + np.arange(6) * 4.0,
        ),
    )


def test_volume_native_zyx_centers_and_missing_are_distinct_from_zero(grid_env):
    grid = volume_grid(grid_env)
    assert grid is not None
    values = np.linspace(-10, 0, grid_env.n_bins)
    values[0] = np.nan
    values[-2] = 0
    encoded = grid.encode(values, ColorScale("viridis", -10, 0, "lightgray"))
    assert encoded.shape == (5, 4, 3)
    assert encoded.flags.c_contiguous
    assert encoded[1, 1, 1] == 0  # inactive
    assert encoded[0, 0, 0] == 0  # unsupported observation
    assert encoded[4, 3, 2] == 255  # true zero is the high endpoint
    assert np.min(encoded[encoded > 0]) >= 1
    native_centers = np.argwhere(grid.active_mask)
    mapped = native_centers @ grid.matrix[:3, :3].T + grid.matrix[:3, 3]
    np.testing.assert_allclose(mapped, grid_env.bin_centers)


def test_transformed_grid_preserves_world_xyz_and_bin_order(grid_env):
    from neurospatial.ops.transforms import AffineND

    matrix = np.array(
        [[0, -2, 0, 70], [3, 0, 0, -25], [0, 0, 4, 3], [0, 0, 0, 1]], dtype=float
    )
    transformed = grid_env.apply_transform(AffineND(matrix))
    grid = volume_grid(transformed)
    assert grid is not None
    native_centers = np.argwhere(grid.active_mask)
    mapped = native_centers @ grid.matrix[:3, :3].T + grid.matrix[:3, 3]
    np.testing.assert_allclose(mapped, transformed.bin_centers)
    assert grid.middle_plane[:3] == (0, 0, 1)


def test_irregular_grid_auto_points_and_explicit_volume_rejected():
    from neurospatial.geometry._capabilities import RectilinearGrid

    grid = RectilinearGrid(
        (np.array([0.0, 1.0, 3.0]), np.arange(3.0), np.arange(3.0)),
        (2, 2, 2),
        np.ones((2, 2, 2), dtype=bool),
    )
    env = SimpleNamespace(
        n_dims=3, binning=SimpleNamespace(grid_frame=None, rectilinear=grid)
    )
    assert resolve_field_rendering(env, "auto") == "points"
    for mode in ("mip", "iso", "slice"):
        with pytest.raises(ValueError, match="uniform regular 3D"):
            resolve_field_rendering(env, mode)


def test_explicit_volume_validation_precedes_import_and_allocation(monkeypatch):
    import neurospatial.animation.backends.fastplotlib_backend as backend

    monkeypatch.setattr(
        backend, "import_fastplotlib", lambda: pytest.fail("must validate first")
    )
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    with pytest.raises(ValueError, match="uniform regular 3D"):
        backend.FastplotlibAnimation(
            env,
            np.ones((2, env.n_bins)),
            frame_times=np.array([0.0, 1.0]),
            field_rendering="mip",
        )


@pytest.mark.parametrize("mode", ["mip", "iso", "slice"])
def test_native_volume_uses_public_api_and_lazy_single_frame_updates(grid_env, mode):
    import pygfx

    class Graphic:
        def __init__(self, data, mode):
            self.mode = mode
            self.data = data.copy()
            material_type = {
                "mip": pygfx.VolumeMipMaterial,
                "iso": pygfx.VolumeIsoMaterial,
                "slice": pygfx.VolumeSliceMaterial,
            }[mode]
            self.world_object = SimpleNamespace(
                local=SimpleNamespace(matrix=None),
                children=[SimpleNamespace(material=material_type())],
            )

        def add_event_handler(self, *args):
            pass

    class Subplot:
        def add_image_volume(self, data, **kwargs):
            self.kwargs = kwargs
            self.graphic = Graphic(data, kwargs["mode"])
            return self.graphic

    plot = Subplot()
    values = np.linspace(-10, 0, grid_env.n_bins)
    scale = ColorScale("viridis", -10, 0, "lightgray")
    field = FieldGraphic(
        plot,
        grid_env,
        values,
        scale,
        field_rendering=mode,
        volume_threshold=-3 if mode == "iso" else None,
        slice_plane=(0, 0, 1, -58) if mode == "slice" else None,
    )
    assert plot.kwargs["mode"] == mode
    assert plot.kwargs["interpolation"] == "nearest"
    assert plot.kwargs["vmin"] == -0.5
    assert plot.kwargs["vmax"] == 255.5
    if mode == "iso":
        assert plot.kwargs["threshold"] == pytest.approx(1 + 254 * 0.7)
    if mode == "slice":
        assert plot.kwargs["plane"] == (0, 0, 1, -58)
    table = plot.graphic.world_object.children[0].material.map.texture.data
    assert table[0, 3] == 0
    assert table[1, 3] == 1
    field.update(np.full(grid_env.n_bins, np.nan))
    assert np.all(plot.graphic.data == 0)
    field.update(np.zeros(grid_env.n_bins))
    assert np.all(plot.graphic.data[plot.graphic.data != 0] == 255)
    assert scale.vmin == -10 and scale.vmax == 0
    hover = field.graphic.tooltip_format
    assert "bin 58" in hover({"index": (2, 3, 4)})
    assert "value: 0" in hover({"index": (2, 3, 4)})
    assert "Z Position: 68" in hover({"index": (2, 3, 4)})
    assert "inactive cell" in hover({"index": (1, 1, 1)})
    assert "outside" in hover({"index": (20, 3, 4)})
    field.update(np.full(grid_env.n_bins, np.nan))
    assert "value: missing" in hover({"index": (2, 3, 4)})


@pytest.mark.parametrize("plane", [(0, 0, 0, 1), (1, 0, np.nan, 1), (1, 0, 1)])
def test_slice_plane_requires_finite_world_normal(grid_env, plane):
    with pytest.raises(ValueError, match="WORLD XYZ"):
        resolve_field_rendering(grid_env, "slice", slice_plane=plane)


def test_adapter_rejects_unsupported_version_without_constructing_widget():
    from neurospatial.environment._fastplotlib_compat import require_animation_support

    with pytest.raises(ImportError, match=r"fastplotlib>=0.7,<0.8"):
        require_animation_support(SimpleNamespace(__version__="0.8.0"))


@pytest.fixture
def native_fastplotlib():
    fastplotlib = pytest.importorskip("fastplotlib")
    wgpu = pytest.importorskip("wgpu")
    try:
        adapter = wgpu.gpu.request_adapter_sync()
    except Exception:
        adapter = None
    if adapter is None:
        pytest.skip("no GPU adapter for native volume captures")
    return fastplotlib


@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("mode", ["mip", "iso", "slice"])
def test_native_captures_missing_holes_negative_zero_and_lazy_updates(
    native_fastplotlib, mode, transformed
):
    from neurospatial.animation.backends.fastplotlib_backend import FastplotlibAnimation

    mask = np.ones((5, 3, 3), dtype=bool)
    mask[2] = False
    env = Environment.from_grid_mask(
        mask,
        grid_edges=(
            10 + 2 * np.arange(6.0),
            30 + 3 * np.arange(4.0),
            50 + 4 * np.arange(4.0),
        ),
    )
    values = np.where(env.bin_centers[:, 0] < 14, -5.0, 0.0)
    values[env.bin_centers[:, 0] < 12] = np.nan
    values[env.bin_centers[:, 0] > 18] = 5.0

    class Lazy:
        shape = (1000, env.n_bins)

        def __init__(self):
            self.reads = []

        def __len__(self):
            return self.shape[0]

        def __array__(self, *args, **kwargs):
            raise AssertionError("Never materialize the movie")

        def __getitem__(self, index):
            self.reads.append(index)
            return values if index == 0 else np.full(env.n_bins, np.nan)

    source = Lazy()
    matrix = np.eye(4)
    if transformed:
        from neurospatial.ops.transforms import AffineND

        matrix[:3, :3] = np.diag([2.0, 3.0, 4.0])
        matrix[:3, 3] = [100.0, -10.0, 20.0]
        env = env.apply_transform(AffineND(matrix))
    slice_z = matrix[2, 2] * 56 + matrix[2, 3]
    options = (
        {"volume_threshold": -7}
        if mode == "iso"
        else {"slice_plane": (0, 0, 1, -slice_z)}
        if mode == "slice"
        else {}
    )
    native = FastplotlibAnimation(
        env,
        source,
        frame_times=np.arange(1000.0),
        vmin=-10,
        vmax=10,
        field_rendering=mode,
        **options,
    )
    try:
        native.show()
        plot = native.figure["field"]
        volume = native._field_graphics[0].graphic
        assert type(volume).__name__ == "ImageVolumeGraphic"
        plot.camera.fov = 0
        plot.camera.show_object(volume.world_object, view_dir=(0, 0, -1), up=(0, 1, 0))
        observed = native.capture_frame(0.0)
        absent = native.capture_frame(1.0)

        def patch(image, x):
            world = matrix @ np.array([x, 34.5, 56.0, 1.0])
            screen = plot.map_world_to_screen(world[:3])
            ratio = image.shape[1] / native.figure.canvas.get_logical_size()[0]
            col, row = (np.asarray(screen[:2]) * ratio).astype(int)
            return image[row - 1 : row + 2, col - 1 : col + 2].astype(float)

        # All-NaN columns and inactive internal holes are exactly the rendered
        # background. Finite negative values and true zeros remain visible.
        for missing_x in (11.0, 15.0):
            np.testing.assert_array_equal(
                patch(observed, missing_x), patch(absent, missing_x)
            )
        for finite_x in (13.0, 17.0, 19.0):
            assert (
                np.max(np.abs(patch(observed, finite_x) - patch(absent, finite_x))) > 10
            )
        assert np.max(np.abs(patch(observed, 13.0) - patch(observed, 17.0))) > 5
        assert len(source.reads) < 20
        assert all(isinstance(index, (int, np.integer)) for index in source.reads)
        assert native._field_scales[0].vmin == -10
        assert native._field_scales[0].vmax == 10
    finally:
        native.close()


@pytest.mark.parametrize("transformed", [False, True])
def test_native_iso_depth_occludes_background_and_preserves_foreground(
    native_fastplotlib, transformed
):
    from neurospatial.animation.backends.fastplotlib_backend import FastplotlibAnimation
    from neurospatial.ops.transforms import AffineND

    env = Environment.from_grid_mask(
        np.ones((5, 3, 3), dtype=bool),
        grid_edges=(np.arange(6.0) * 2, np.arange(4.0) * 3, 50 + np.arange(4.0) * 4),
    )
    matrix = np.eye(4)
    if transformed:
        matrix[:3, :3] = np.diag([2.0, 3.0, 4.0])
        matrix[:3, 3] = [100.0, -20.0, 300.0]
        env = env.apply_transform(AffineND(matrix))
    native = FastplotlibAnimation(
        env,
        np.ones((2, env.n_bins)),
        frame_times=np.array([0.0, 1.0]),
        vmin=0.0,
        vmax=1.0,
        field_rendering="iso",
    )
    try:
        native.show()
        plot = native.figure["field"]
        volume = native._field_graphics[0].graphic
        plot.camera.fov = 0
        plot.camera.show_object(volume.world_object, view_dir=(0, 0, -1), up=(0, 1, 0))
        # One marker is behind the volume; the other is in front. All positions
        # use world XYZ, including the retained grid's affine translation.
        source_points = np.array([[3.0, 4.5, 45.0], [7.0, 4.5, 67.0]])
        world = source_points @ matrix[:3, :3].T + matrix[:3, 3]
        plot.add_scatter(world.astype(np.float32), colors="cyan", sizes=25)
        for switch_mode in (False, True):
            if switch_mode:
                volume.mode = "mip"
                volume.mode = "iso"
            # The public feature setters must keep writing the active material.
            volume.threshold = 100.0
            volume.step_size = 0.75
            volume.substep_size = 0.1
            image = native.capture_frame(0.0)
            rgb = []
            for point in world:
                screen = plot.map_world_to_screen(point)
                ratio = image.shape[1] / native.figure.canvas.get_logical_size()[0]
                col, row = (np.asarray(screen[:2]) * ratio).astype(int)
                rgb.append(image[row, col])
            assert not np.array_equal(rgb[0], [0, 255, 255])
            np.testing.assert_array_equal(rgb[1], [0, 255, 255])
            assert volume.world_object.children[0].material.threshold == pytest.approx(
                100.0
            )
    finally:
        native.close()


def test_iso_shader_guard_accepts_known_fix_and_refuses_unknown_code():
    from neurospatial.environment._fastplotlib_compat import _iso_shader_source

    source = "$$ elif mode == 'iso'\nlet world_matrix = mat3x3<f32>();\nlet world_pos = world_matrix * data_pos;"
    fixed = _iso_shader_source(source, "0.17.0")
    assert "u_wobject.world_transform * vec4<f32>(data_pos, 1.0)" in fixed
    assert _iso_shader_source(fixed, "0.18.0") == fixed
    with pytest.raises(ImportError, match="field_rendering='mip'"):
        _iso_shader_source("unrecognized shader", "0.18.0")


def test_supported_fastplotlib_widget_contract_imports_without_gpu():
    from neurospatial.environment._fastplotlib_compat import require_animation_support

    require_animation_support(pytest.importorskip("fastplotlib"))


def test_iso_compatibility_failure_precedes_widget_allocation(grid_env, monkeypatch):
    import neurospatial.animation.backends.fastplotlib_backend as backend
    import neurospatial.environment._fastplotlib_compat as compatibility

    def unavailable():
        raise ImportError("Cannot verify this iso shader")

    fake = SimpleNamespace(
        __version__="0.7.0",
        NDWidget=lambda **kwargs: pytest.fail("must validate before allocation"),
    )
    monkeypatch.setattr(backend, "import_fastplotlib", lambda: fake)
    monkeypatch.setattr(compatibility, "_iso_material_type", unavailable)
    with pytest.raises(ImportError, match="Cannot verify"):
        backend.FastplotlibAnimation(
            grid_env,
            np.ones((2, grid_env.n_bins)),
            frame_times=np.array([0.0, 1.0]),
            field_rendering="iso",
        )
