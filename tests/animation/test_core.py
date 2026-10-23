"""animate_fields builds an Animation and opens or exports it on its clock.

The dispatcher validates the fields and frame times, chooses a backend, names
the field streams and derives the playback rate from the sampling rate of
``frame_times``; the scene, view and export tests cover drawing itself.
"""

import base64
import io
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from neurospatial import Environment, Regions
from neurospatial.animation import AnimationView
from neurospatial.animation.core import (
    MAX_PLAYBACK_FPS,
    _auto_reason,
    _select_backend,
    animate_fields,
)
from neurospatial.animation.overlays import PositionOverlay
from neurospatial.animation.scene import Animation, AnimationField


@pytest.fixture
def env():
    points = np.random.default_rng(42).uniform(0, 10, (100, 2))
    return Environment.from_samples(points, bin_size=2).with_metadata(units="cm")


def _fields(env, n_frames):
    return np.random.default_rng(7).random((n_frames, env.n_bins))


def _player(path: Path) -> tuple[float, list[str], list[np.ndarray]]:
    """The playback rate, time labels and frames of an embedded HTML player."""
    text = path.read_text(encoding="utf-8")
    fps = float(re.search(r"const baseFPS = ([^;]+);", text).group(1))
    labels = json.loads(re.search(r"const labels = (\[.*?\]);", text).group(1))
    frames = json.loads(re.search(r"const frames = (\[.*?\]);", text).group(1))
    images = [np.asarray(Image.open(io.BytesIO(base64.b64decode(f)))) for f in frames]
    return fps, labels, images


def _napari_installed(installed):
    """Patch napari's availability as the auto backend checks it."""
    import importlib.util

    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "napari":
            return object() if installed else None
        return real(name, *args, **kwargs)

    return patch("importlib.util.find_spec", find_spec)


class TestBackendSelection:
    """backend="auto" exports to files and otherwise picks a viewer."""

    @pytest.mark.parametrize(
        ("save_path", "backend"),
        [
            ("out.mp4", "video"),
            ("out.webm", "video"),
            ("out.gif", "video"),
            ("out.html", "html"),
        ],
    )
    def test_file_suffix_selects_the_export(self, save_path, backend):
        assert _select_backend(n_frames=100, save_path=save_path) == backend

    def test_jupyter_selects_the_widget(self):
        with patch("IPython.get_ipython", return_value=MagicMock()):
            assert _select_backend(n_frames=100, save_path=None) == "widget"

    def test_large_sessions_in_jupyter_use_napari(self):
        with (
            patch("IPython.get_ipython", return_value=MagicMock()),
            _napari_installed(True),
        ):
            assert _select_backend(n_frames=15_000, save_path=None) == "napari"
        assert "15,000 frames, more than 10,000" in _auto_reason("napari", 15_000, None)

    def test_outside_jupyter_napari_is_the_viewer(self):
        with patch("IPython.get_ipython", return_value=None), _napari_installed(True):
            assert _select_backend(n_frames=100, save_path=None) == "napari"
        assert "not a Jupyter session" in _auto_reason("napari", 100, None)

    def test_missing_napari_names_the_alternatives(self, env):
        with (
            patch("IPython.get_ipython", return_value=None),
            _napari_installed(False),
            pytest.raises(RuntimeError, match="napari is not installed") as error,
        ):
            animate_fields(env, _fields(env, 3), frame_times=np.arange(3) / 10)
        fix = str(error.value).splitlines()[-1]
        assert "pip install 'napari[all]'" in fix
        assert "backend='widget'" in fix
        assert "save_path='out.mp4'" in fix


class TestAnimateFieldsValidation:
    """Invalid fields fail with an actionable message and write nothing."""

    def test_empty_fields(self, env, tmp_path):
        for fields in ([], np.empty((0, env.n_bins))):
            with pytest.raises(ValueError, match="fields cannot be empty"):
                animate_fields(
                    env, fields, frame_times=np.empty(0), save_path=tmp_path / "a.mp4"
                )
        assert list(tmp_path.iterdir()) == []

    def test_one_dimensional_array_is_not_read_as_frames(self, env, tmp_path):
        """A single (n_bins,) row needs a frame axis, not n_bins frames."""
        with pytest.raises(ValueError, match=r"shape \(n_frames, n_bins\)") as error:
            animate_fields(
                env,
                _fields(env, 1)[0],
                frame_times=np.array([0.0]),
                save_path=tmp_path / "a.mp4",
            )
        assert "field[None, :]" in str(error.value)
        assert list(tmp_path.iterdir()) == []

    def test_field_bins_must_match_the_environment(self, env, tmp_path):
        fields = np.zeros((2, env.n_bins + 10))
        with pytest.raises(ValueError, match=f"has {env.n_bins + 10} bins"):
            animate_fields(
                env,
                fields,
                frame_times=np.array([0.0, 0.1]),
                save_path=tmp_path / "a.mp4",
            )
        assert list(tmp_path.iterdir()) == []

    def test_frame_times_must_match_the_frames(self, env):
        with pytest.raises(ValueError, match=r"frame_times length \(2\).*\(3\)"):
            animate_fields(
                env, _fields(env, 3), frame_times=np.array([0.0, 0.1]), backend="widget"
            )

    def test_each_field_needs_a_row_per_frame_time(self, env):
        """The field whose length differs is named, not only the first."""
        short, long = _fields(env, 3), _fields(env, 5)
        with pytest.raises(ValueError, match="Field 'late' has 5 frames") as error:
            animate_fields(
                env,
                [list(short), list(long)],
                layer_names=["early", "late"],
                frame_times=np.arange(3) / 10,
                backend="widget",
            )
        assert "frame_times has 3 times" in str(error.value)

    def test_frame_times_is_required(self, env):
        with pytest.raises(TypeError, match="frame_times"):
            animate_fields(env, _fields(env, 3), backend="widget")

    def test_unknown_backend(self, env):
        with pytest.raises(ValueError, match="Unknown backend: invalid") as error:
            animate_fields(
                env, _fields(env, 3), frame_times=np.arange(3) / 10, backend="invalid"
            )
        for backend in ("napari", "video", "html", "widget", "fastplotlib"):
            assert repr(backend) in str(error.value)

    @pytest.mark.parametrize("speed", [0, -1, np.inf, np.nan])
    def test_speed_must_be_positive_and_finite(self, env, speed):
        with pytest.raises(ValueError, match="speed must be a positive"):
            animate_fields(
                env, _fields(env, 3), frame_times=np.arange(3) / 10, speed=speed
            )

    @pytest.mark.parametrize("cap", [0, 2.5, True])
    def test_max_playback_fps_must_be_a_positive_integer(self, env, cap):
        with pytest.raises(ValueError, match="max_playback_fps must be"):
            animate_fields(
                env,
                _fields(env, 3),
                frame_times=np.arange(3) / 10,
                max_playback_fps=cap,
            )

    @pytest.mark.parametrize("backend", ["video", "html"])
    def test_exports_need_a_matching_save_path(self, env, backend):
        with pytest.raises(ValueError, match="save_path=") as error:
            animate_fields(
                env, _fields(env, 3), frame_times=np.arange(3) / 10, backend=backend
            )
        assert "Fix: pass save_path=" in str(error.value)

    def test_fps_option_is_refused_with_the_speed_fix(self, env, tmp_path):
        """The playback rate comes from frame_times and speed, so fps= raises."""
        with pytest.raises(TypeError, match="unknown option 'fps'") as error:
            animate_fields(
                env,
                _fields(env, 3),
                frame_times=np.arange(3) / 10,
                save_path=tmp_path / "a.html",
                fps=100,
            )
        assert "speed=2.0" in str(error.value)
        assert list(tmp_path.iterdir()) == []


class TestExports:
    """Exports are the scene the arguments describe, on the recording clock."""

    def test_html_frames_are_the_scene_the_arguments_describe(self, env, tmp_path):
        times = np.array([0.0, 0.1, 0.2])
        fields = _fields(env, 3)
        regions = Regions()
        regions.add("goal", point=(5.0, 5.0))
        env = env.with_regions(regions)
        path_xy = np.array([[2.0, 2.0], [5.0, 5.0], [8.0, 3.0]])
        options = {
            "cmap": "hot",
            "vmin": 0.0,
            "vmax": 2.0,
            "title": "Session 1",
            "show_regions": True,
            "region_alpha": 0.6,
            "scale_bar": True,
        }
        result = animate_fields(
            env,
            fields,
            frame_times=times,
            save_path=tmp_path / "player.html",
            overlays=[PositionOverlay(path_xy, times, color="red", size=12)],
            dpi=30,
            **options,
        )
        assert result == tmp_path / "player.html"
        fps, labels, frames = _player(result)
        assert fps == pytest.approx(10.0)
        # Output frames sample the half-open [first, last) frame time.
        assert labels == ["t = 0.000 s", "t = 0.100 s"]

        def scene(**changes):
            settings = {**options, **changes}
            field = AnimationField(
                fields,
                times,
                cmap=settings.pop("cmap"),
                vmin=settings.pop("vmin"),
                vmax=settings.pop("vmax"),
            )
            overlays = [PositionOverlay(path_xy, times, color="red", size=12)]
            return Animation(
                env, {"field": field}, overlays=overlays, frame_times=times, **settings
            )

        with scene() as expected:
            np.testing.assert_array_equal(frames[1], expected.render_frame(0.1, dpi=30))
        # Each option shows in the frame, so the comparison above checks it.
        for changes in (
            {"cmap": "viridis"},
            {"vmax": 1.0},
            {"show_regions": False},
            {"scale_bar": None},
        ):
            with scene(**changes) as other:
                assert not np.array_equal(frames[1], other.render_frame(0.1, dpi=30))

    def test_video_holds_one_frame_per_row_at_the_sampling_rate(self, env, tmp_path):
        imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
        times = np.arange(5) / 20.0
        path = animate_fields(
            env,
            _fields(env, 5),
            frame_times=times,
            save_path=tmp_path / "a.mp4",
            dpi=30,
        )
        assert path == tmp_path / "a.mp4"
        reader = imageio_ffmpeg.read_frames(str(path))
        meta = next(reader)
        assert meta["fps"] == pytest.approx(20.0)
        assert len(list(reader)) == 4
        assert [item.name for item in tmp_path.iterdir()] == ["a.mp4"]

    @pytest.mark.parametrize(("speed", "fps"), [(1.0, 30.0), (0.5, 15.0), (2.0, 60.0)])
    def test_speed_sets_the_rate_and_keeps_every_row(self, env, tmp_path, speed, fps):
        times = np.linspace(0, 0.2, 7)  # 30 Hz
        path = animate_fields(
            env,
            _fields(env, 7),
            frame_times=times,
            save_path=tmp_path / "a.html",
            speed=speed,
            dpi=20,
        )
        played, labels, _ = _player(path)
        assert played == pytest.approx(fps)
        assert labels == [f"t = {t:.3f} s" for t in times[:-1]]

    def test_capped_rate_keeps_speed_and_warns_at_the_call_site(self, env, tmp_path):
        times = np.linspace(0, 0.2, 101)  # 500 Hz
        with pytest.warns(UserWarning, match="max_playback_fps=60") as record:
            path = animate_fields(
                env,
                _fields(env, 101),
                frame_times=times,
                save_path=tmp_path / "a.html",
                dpi=20,
            )
        (warning,) = [w for w in record if "max_playback_fps" in str(w.message)]
        assert warning.filename == __file__
        assert "skips rows" in str(warning.message)
        played, labels, _ = _player(path)
        assert played == MAX_PLAYBACK_FPS
        # Real time: frames are 1/60 s of recording apart.
        assert labels == [f"t = {k / 60:.3f} s" for k in range(12)]

    def test_uncapped_rate_does_not_warn(self, env, tmp_path):
        import warnings

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            animate_fields(
                env,
                _fields(env, 5),
                frame_times=np.arange(5) / 20.0,
                save_path=tmp_path / "a.html",
                dpi=20,
            )
        assert not [w for w in record if "max_playback_fps" in str(w.message)]

    def test_auto_exports_a_gif_suffix_as_video(self, env, tmp_path):
        import imageio.v3 as iio

        path = animate_fields(
            env,
            _fields(env, 4),
            frame_times=np.arange(4) / 10,
            save_path=tmp_path / "a.gif",
            dpi=20,
        )
        assert iio.imread(path, index=None).shape[0] == 3


class TestViews:
    """Interactive backends return an AnimationView of the scene."""

    @pytest.fixture(autouse=True)
    def _ipywidgets(self):
        pytest.importorskip("ipywidgets")

    def test_widget_view_shows_the_scene_and_closing_releases_it(self, env):
        times = np.arange(3) / 10
        view = animate_fields(
            env, _fields(env, 3), frame_times=times, backend="widget", dpi=20
        )
        try:
            assert isinstance(view, AnimationView)
            image = view.native.children[0]
            shown = np.asarray(Image.open(io.BytesIO(image.value)))
            np.testing.assert_array_equal(
                shown, view.animation.render_frame(0.0, dpi=20)
            )
            view.seek(0.2)
            assert view.time == pytest.approx(0.2)
            np.testing.assert_array_equal(
                np.asarray(Image.open(io.BytesIO(image.value))),
                view.animation.render_frame(0.2, dpi=20),
            )
        finally:
            view.close()
        with pytest.raises(RuntimeError, match="Animation is closed"):
            view.animation.render_frame(0.0)

    def test_frame_times_and_memmap_reach_the_scene_unchanged(self, env, tmp_path):
        source = np.memmap(
            tmp_path / "fields.dat", mode="w+", shape=(4, env.n_bins), dtype=float
        )
        source[:] = _fields(env, 4)
        times = np.array([10.0, 10.1, 10.25, 10.3])
        view = animate_fields(env, source, frame_times=times, backend="widget", dpi=20)
        try:
            np.testing.assert_array_equal(view.animation.frame_times, times)
            # Rows stay on disk and are read lazily, not copied.
            assert view.animation.fields["field"].values is source
        finally:
            view.close()
        assert not source._mmap.closed
        source._mmap.close()

    @pytest.mark.parametrize(("max_gap", "held"), [(0.5, False), (None, True)])
    def test_rows_are_held_up_to_max_gap(self, env, max_gap, held):
        fields = _fields(env, 4)
        view = animate_fields(
            env,
            fields,
            frame_times=np.array([0.0, 0.1, 0.2, 2.0]),
            backend="widget",
            max_gap=max_gap,
            dpi=20,
        )
        try:
            sample = view.animation.sample_fields(1.0)["field"]
            if held:
                np.testing.assert_array_equal(sample, fields[2])
            else:
                assert np.isnan(sample).all()
        finally:
            view.close()

    @pytest.mark.parametrize("form", ["mapping", "list"])
    def test_several_fields_are_named_and_drawn_side_by_side(self, env, form):
        times = np.arange(3) / 10
        early, late = _fields(env, 3), _fields(env, 3)[::-1]
        if form == "mapping":
            fields, options = {"early": early, "late": late}, {}
        else:
            fields = [list(early), list(late)]
            options = {"layer_names": ["early", "late"]}
        view = animate_fields(
            env, fields, frame_times=times, backend="widget", dpi=20, **options
        )
        single = animate_fields(env, early, frame_times=times, backend="widget", dpi=20)
        try:
            assert list(view.animation.fields) == ["early", "late"]
            np.testing.assert_array_equal(
                view.animation.sample_fields(0.1)["late"], late[1]
            )
            wide = view.animation.render_frame(0.0, dpi=20)
            narrow = single.animation.render_frame(0.0, dpi=20)
            assert wide.shape[1] > narrow.shape[1]
        finally:
            view.close()
            single.close()

    def test_unnamed_fields_get_numbered_names(self, env):
        fields = [list(_fields(env, 2)), list(_fields(env, 2))]
        view = animate_fields(
            env, fields, frame_times=np.array([0.0, 0.1]), backend="widget", dpi=20
        )
        try:
            assert list(view.animation.fields) == ["field 1", "field 2"]
        finally:
            view.close()

    def test_layer_names_must_name_every_field(self, env):
        fields = [list(_fields(env, 2)), list(_fields(env, 2))]
        with pytest.raises(ValueError, match="layer_names has 1 names for 2 fields"):
            animate_fields(
                env,
                fields,
                frame_times=np.array([0.0, 0.1]),
                backend="widget",
                layer_names=["early"],
            )


@pytest.mark.napari
def test_napari_view_wraps_the_viewer_and_closing_releases_the_scene(env):
    napari = pytest.importorskip("napari")
    view = animate_fields(
        env,
        _fields(env, 3),
        frame_times=np.arange(3) / 10,
        backend="napari",
        show=False,
    )
    try:
        assert isinstance(view.native, napari.Viewer)
        view.seek(0.1)
        assert view.time == pytest.approx(0.1)
    finally:
        view.close()
    with pytest.raises(RuntimeError, match="Animation is closed"):
        view.animation.render_frame(0.0)


class TestSubsampleFrames:
    """Test subsample_frames() utility function."""

    def test_subsample_ndarray(self):
        """Test subsampling with ndarray input."""
        from neurospatial.animation.core import subsample_frames

        rng = np.random.default_rng(42)
        # Create test data: 100 frames at 100 Hz
        fields = rng.random((100, 50))  # 100 frames, 50 bins

        # Subsample to 25 Hz (every 4th frame)
        result = subsample_frames(fields, target_fps=25, source_fps=100)

        # Check output type
        assert isinstance(result, np.ndarray)

        # Check output shape (should be ~25 frames)
        assert result.shape[0] == 25
        assert result.shape[1] == 50

        # Check first frame matches
        np.testing.assert_array_equal(result[0], fields[0])

    def test_subsample_list(self):
        """Test subsampling with list input."""
        from neurospatial.animation.core import subsample_frames

        rng = np.random.default_rng(42)
        # Create test data as list
        fields = [rng.random(50) for _ in range(100)]

        # Subsample to 25 Hz
        result = subsample_frames(fields, target_fps=25, source_fps=100)

        # Check output type preserved
        assert isinstance(result, list)

        # Check output length
        assert len(result) == 25

        # Check first frame matches
        np.testing.assert_array_equal(result[0], fields[0])

    def test_subsample_invalid_fps(self):
        """Test that target_fps > source_fps raises error."""
        from neurospatial.animation.core import subsample_frames

        rng = np.random.default_rng(42)
        fields = rng.random((100, 50))

        with pytest.raises(ValueError, match=r"target_fps.*cannot exceed.*source_fps"):
            subsample_frames(fields, target_fps=200, source_fps=100)

    def test_subsample_memmap(self, tmp_path):
        """Test subsampling works with memory-mapped arrays."""
        from neurospatial.animation.core import subsample_frames

        rng = np.random.default_rng(42)
        # Use tmp_path so pytest handles cleanup. On Windows the previous
        # `tmpfile_path.unlink()` raised PermissionError because the memmap
        # was still mapped against the file at unlink time.
        mmap_path = tmp_path / "test_subsample.dat"
        fields_mmap = np.memmap(
            str(mmap_path), dtype="float32", mode="w+", shape=(1000, 50)
        )
        fields_mmap[:] = rng.random((1000, 50))
        fields_mmap.flush()

        # Subsample
        result = subsample_frames(fields_mmap, target_fps=30, source_fps=250)

        # Check it worked without loading all data
        assert len(result) > 0
        assert result.shape[1] == 50

        # Release the mapping before pytest cleans up tmp_path on Windows.
        del fields_mmap

    def test_subsample_calculation(self):
        """Test subsampling rate calculation is correct."""
        from neurospatial.animation.core import subsample_frames

        # 250 Hz → 30 fps should take every ~8.3 frames
        fields = np.arange(250 * 50).reshape(250, 50)  # Sequential data

        result = subsample_frames(fields, target_fps=30, source_fps=250)

        # Calculate expected number of frames
        # np.arange(0, 250, 8.333...) produces 30 indices: [0, 8, 16, 24, ...]
        subsample_rate = 250 / 30
        expected_n_frames = len(np.arange(0, 250, subsample_rate))
        assert len(result) == expected_n_frames


class TestEstimateColormapRangeFromSubset:
    """Test estimate_colormap_range_from_subset() helper function."""

    def test_returns_tuple_of_floats(self):
        """Test that function returns tuple of two floats."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = rng.random((100, 50))

        result = estimate_colormap_range_from_subset(fields)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_reasonable_range_for_uniform_random(self):
        """Test that range is reasonable for uniform random [0, 1] data."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = rng.random((10_000, 50))

        vmin, vmax = estimate_colormap_range_from_subset(fields)

        # For uniform random [0, 1], 1st and 99th percentiles should be ~0.01 and ~0.99
        assert 0.0 <= vmin <= 0.05
        assert 0.95 <= vmax <= 1.0

    def test_works_with_list_input(self):
        """Test that function works with list of arrays."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = [rng.random(50) for _ in range(100)]

        vmin, vmax = estimate_colormap_range_from_subset(fields)

        assert isinstance(vmin, float)
        assert isinstance(vmax, float)
        assert vmin < vmax

    def test_reproducible_with_seed(self):
        """Test that results are reproducible with same seed."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = rng.random((1000, 50))

        result1 = estimate_colormap_range_from_subset(fields, seed=123)
        result2 = estimate_colormap_range_from_subset(fields, seed=123)

        assert result1 == result2

    def test_matches_the_animation_default_for_a_sharp_posterior(self):
        """The top is a percentile of the frames' peaks, not of all values."""
        from neurospatial.animation._field_stats import animation_colormap_range
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = np.zeros((1000, 500))
        peaks = rng.uniform(0.2, 0.3, 1000)
        fields[np.arange(1000), rng.integers(0, 500, 1000)] = peaks

        vmin, vmax = estimate_colormap_range_from_subset(fields)

        # Nearly every frame's peak is below the top color, not above it.
        assert vmin == 0.0 and vmax > np.percentile(peaks, 95)
        assert (vmin, vmax) == animation_colormap_range(fields, None, None)

    def test_custom_percentiles(self):
        """A lower top percentile lets more frames' peaks pass the top."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = rng.random((10_000, 50))

        _, top_99 = estimate_colormap_range_from_subset(fields)
        _, top_50 = estimate_colormap_range_from_subset(fields, percentile=(1.0, 50.0))

        frame_peaks = fields.max(axis=1)
        assert top_50 == pytest.approx(np.median(frame_peaks))
        assert top_50 < top_99

    def test_n_samples_parameter(self):
        """Test that n_samples controls sampling."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = rng.random((100_000, 50))

        # Small sample should still work
        vmin, vmax = estimate_colormap_range_from_subset(fields, n_samples=100)

        assert vmin < vmax

    def test_works_with_memmap(self, tmp_path):
        """Test that function works with memory-mapped arrays."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        # Create memmap
        mmap_path = tmp_path / "test.dat"
        fields = np.memmap(
            str(mmap_path), dtype=np.float64, mode="w+", shape=(1000, 50)
        )
        rng = np.random.default_rng(42)
        fields[:] = rng.random((1000, 50))
        fields.flush()

        vmin, vmax = estimate_colormap_range_from_subset(fields)

        assert isinstance(vmin, float)
        assert isinstance(vmax, float)
        assert vmin < vmax

    def test_fast_for_large_dataset(self):
        """Test that function completes quickly for large datasets."""
        import time

        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = rng.random((1_000_000, 50))

        start = time.perf_counter()
        vmin, vmax = estimate_colormap_range_from_subset(fields)
        elapsed = time.perf_counter() - start

        # Should complete in under 0.5s
        assert elapsed < 0.5
        assert vmin < vmax


def test_show_regions_string_is_one_region_name():
    """``show_regions="goal"`` shows the goal, not regions named 'g', 'o', ..."""
    from neurospatial.animation._region_names import shown_region_names

    env = Environment.from_samples(
        np.random.default_rng(0).random((50, 2)) * 10, bin_size=2.0
    )
    regions = Regions()
    regions.add("goal", point=(5.0, 5.0))
    env = env.with_regions(regions)
    assert shown_region_names(env, "goal") == ["goal"]
    assert shown_region_names(env, True) == ["goal"]
    assert shown_region_names(env, False) == []
