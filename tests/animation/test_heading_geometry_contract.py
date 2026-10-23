"""Undefined headings stay missing across scene and legacy renderers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._parallel import (
    OverlayArtistManager,
    _render_head_direction_overlay_matplotlib,
)
from neurospatial.animation._prepared import PreparedScene
from neurospatial.animation._style import _heading_displacement, _heading_displacements
from neurospatial.animation.backends import fastplotlib_backend as gpu
from neurospatial.animation.backends._napari_scene import NapariScene
from neurospatial.animation.overlays import HeadDirectionData, HeadDirectionOverlay
from neurospatial.animation.scene import Animation


@pytest.fixture(params=[2, 3])
def env(request):
    return Environment.from_grid_mask(
        np.ones((2,) * request.param, dtype=bool),
        tuple(np.arange(3.0) for _ in range(request.param)),
    )


def napari_heading(scene):
    """Capture native vector geometry without a Qt viewer or graphics adapter."""
    handle = NapariScene.__new__(NapariScene)
    handle.scene, handle._helper, handle._updates = scene, PreparedScene(scene), []
    handle._panel_stacks = {None: []}
    layers = []

    def add(kind, values, **options):
        layer = SimpleNamespace(data=values, **options)
        layers.append(layer)
        return layer

    handle._add = add
    handle._build_overlays()
    return handle, layers[0]


def heading_artists(scene):
    from matplotlib.patches import FancyArrow
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    ax = scene._renderer.figure.axes[0]
    return [
        artist
        for artist in [*ax.patches, *ax.collections]
        if isinstance(artist, (FancyArrow, Line3DCollection)) and artist.get_visible()
    ]


def test_opposite_vectors_midpoint_has_no_arrow_in_scene_or_export(
    env, tmp_path, monkeypatch
):
    vector = np.zeros(env.n_dims)
    vector[-1] = 1.0
    times = np.array([0.0, 0.2])
    overlay = HeadDirectionOverlay(np.array([vector, -vector]), times, length=0.5)
    frames, arrows = [], []

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def append_data(self, frame):
            frames.append(frame)
            arrows.append(len(heading_artists(scene)))

    import imageio.v2

    monkeypatch.setattr(imageio.v2, "get_writer", lambda *a, **k: Writer())
    with Animation(
        env, np.ones((3, env.n_bins)), frame_times=[0.0, 0.1, 0.2], overlays=[overlay]
    ) as scene:
        expected = scene.render_frame(0.1, dpi=25)
        assert heading_artists(scene) == []
        handle, layer = napari_heading(scene)
        for update in handle._updates:
            update(0.1, None)
        assert layer.data.shape == (0, 2, env.n_dims)
        scene.export(tmp_path / "heading.mp4", time_range=(0.1, 0.2), fps=10, dpi=25)
        assert arrows == [0]
        np.testing.assert_array_equal(frames[0], expected)
        scene.render_frame(0.0, dpi=25)
        assert len(heading_artists(scene)) == 1


@pytest.mark.parametrize("scale", [1.0, 1e-300, 1e300])
def test_finite_vectors_have_requested_length_in_environment_coordinates(env, scale):
    vector = np.arange(1.0, env.n_dims + 1)
    expected = vector / np.linalg.norm(vector) * 0.5
    direction = _heading_displacement(vector * scale, 0.5, n_dims=env.n_dims)
    np.testing.assert_allclose(direction, expected)
    origin = np.arange(env.n_dims, dtype=float)
    segment = gpu.heading_segment(origin, vector * scale, 0.5)
    np.testing.assert_allclose(segment, [origin, origin + expected])


@pytest.mark.parametrize("missing", [0.0, np.nan, np.inf])
def test_undefined_vectors_are_missing_in_shared_and_gpu_geometry(env, missing):
    vector = np.full(env.n_dims, missing)
    assert _heading_displacement(vector, 0.5, n_dims=env.n_dims) is None
    assert np.isnan(gpu.heading_segment(np.zeros(env.n_dims), vector, 0.5)).all()


@pytest.mark.parametrize("missing", [0.0, np.nan])
def test_native_napari_and_scene_omit_missing_vectors(env, missing):
    overlay = HeadDirectionOverlay(
        np.full((1, env.n_dims), missing), np.array([0.0]), length=0.5
    )
    with Animation(
        env, np.ones((1, env.n_bins)), frame_times=[0.0], overlays=[overlay]
    ) as scene:
        scene.render_frame(0.0, dpi=25)
        assert heading_artists(scene) == []
        handle, layer = napari_heading(scene)
        for update in handle._updates:
            update(0.0, None)
        assert layer.data.shape == (0, 2, env.n_dims)


def test_native_napari_and_scene_normalize_interpolated_vector(env):
    # Both observations are unit vectors; their interpolated midpoint is not.
    first = np.zeros(env.n_dims)
    second = np.zeros(env.n_dims)
    first[0], second[-1] = 1.0, 1.0
    expected = (first + second) / np.sqrt(2) * 0.5
    overlay = HeadDirectionOverlay(
        np.array([first, second]), np.array([0.0, 0.2]), length=0.5
    )
    with Animation(
        env, np.ones((2, env.n_bins)), frame_times=[0.0, 0.2], overlays=[overlay]
    ) as scene:
        # Capture precisely the geometry sent to Matplotlib in either dimension.
        from matplotlib.patches import FancyArrow
        from mpl_toolkits.mplot3d import Axes3D

        if env.n_dims == 2:
            with patch.object(
                FancyArrow, "set_data", autospec=True, side_effect=FancyArrow.set_data
            ) as sent:
                scene.render_frame(0.1, dpi=25)
            sent_geometry = sent.call_args.kwargs
            origin = [sent_geometry["x"], sent_geometry["y"]]
            direction = [sent_geometry["dx"], sent_geometry["dy"]]
        else:
            with patch.object(
                Axes3D, "quiver", autospec=True, side_effect=Axes3D.quiver
            ) as sent:
                scene.render_frame(0.1, dpi=25)
            origin, direction = sent.call_args.args[1:4], sent.call_args.args[4:7]
        np.testing.assert_allclose(origin, env.bin_centers.mean(0))
        np.testing.assert_allclose(direction, expected)
        handle, layer = napari_heading(scene)
        for update in handle._updates:
            update(0.1, None)
        np.testing.assert_allclose(layer.data[0, 1, ::-1], expected)
        np.testing.assert_allclose(layer.data[0, 0, ::-1], env.bin_centers.mean(0))


def test_gpu_native_update_hides_opposite_vectors_midpoint(env):
    vector = np.zeros(env.n_dims)
    vector[-1] = 1.0
    data = HeadDirectionData(
        np.array([vector, -vector]), "red", 0.5, times=np.array([0.0, 0.2])
    )
    clock = gpu._Samples(data.times, lambda time: 0, 1.0)
    plot = SimpleNamespace(add_line=lambda values, **opts: SimpleNamespace(data=values))
    drawn = gpu._HeadDirection(plot, data, clock, None, np.zeros(env.n_dims))
    drawn.update(0.0)
    np.testing.assert_allclose(drawn.line.data[1, : env.n_dims], vector * 0.5)
    drawn.update(0.1)
    assert np.isnan(drawn.line.data[:, : env.n_dims]).all()


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("missing", [0.0, np.nan, np.inf])
def test_legacy_matplotlib_omits_undefined_vectors(persistent, missing):
    env = SimpleNamespace(bin_centers=np.array([[0.0, 0.0]]))
    data = HeadDirectionData(np.full((1, 2), missing), "red", 0.5)
    ax = MagicMock()
    if persistent:
        manager = OverlayArtistManager(ax, env, SimpleNamespace(), False, 0.3)
        assert manager._create_head_direction_quiver(data, 0, None) is None
    else:
        _render_head_direction_overlay_matplotlib(ax, data, 0, env)
    ax.quiver.assert_not_called()


def test_legacy_napari_tracks_skip_undefined_vectors_and_keep_valid_frames():
    from neurospatial.animation.backends.napari_backend import (
        _build_head_direction_tracks,
    )

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    data = HeadDirectionData(
        np.array([[0, 1], [0, 0], [np.nan, np.nan], [np.inf, 0], [1, 0]]),
        "red",
        0.5,
    )
    tracks = _build_head_direction_tracks(data, env)
    np.testing.assert_array_equal(tracks[:, 0], [0, 0, 4, 4])
    assert np.isfinite(tracks).all()


@pytest.mark.parametrize("angles", [False, True])
def test_batch_geometry_keeps_row_identity_with_missing_samples(env, angles):
    if angles:
        values = np.array([0, np.nan, np.pi / 2, np.inf])
    else:
        values = np.zeros((4, env.n_dims))
        values[0, 0], values[2, 1], values[3, 0] = 1e-300, 1e300, np.inf
    actual = _heading_displacements(values, 0.5, n_dims=env.n_dims)
    expected = np.zeros((2, env.n_dims))
    expected[0, 0], expected[1, 1] = 0.5, 0.5
    np.testing.assert_allclose(actual[[0, 2]], expected, atol=1e-15)
    assert np.isnan(actual[[1, 3]]).all()


def test_angle_zero_is_east_and_circular_interpolation_still_wraps():
    np.testing.assert_allclose(_heading_displacement(0.0, 0.5, n_dims=2), [0.5, 0])
    assert _heading_displacement(np.inf, 0.5, n_dims=2) is None
    data = HeadDirectionData(
        np.deg2rad([170, -170]), "red", 0.5, times=np.array([0.0, 0.2])
    )
    clock = gpu._Samples(data.times, lambda time: 0, 1.0)
    plot = SimpleNamespace(add_line=lambda values, **opts: SimpleNamespace(data=values))
    drawn = gpu._HeadDirection(plot, data, clock, None, np.zeros(2))
    drawn.update(0.1)
    np.testing.assert_allclose(drawn.line.data[1, :2], [-0.5, 0], atol=1e-7)


def test_heading_with_several_positions_is_refused():
    # Two tracked points give no single start for the arrow. Drawing it at the
    # arena center would look plausible and be wrong.
    from neurospatial.animation.overlays import PositionOverlay

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    times = np.array([0.0, 0.1])
    track = np.array([[0.5, 0.5], [1.5, 1.5]])
    with pytest.raises(ValueError, match="2 PositionOverlays"):
        Animation(
            env,
            np.ones((2, env.n_bins)),
            frame_times=times,
            overlays=[
                PositionOverlay(track, times=times),
                PositionOverlay(track + 0.5, times=times),
                HeadDirectionOverlay(np.zeros(2), times=times),
            ],
        )
