"""Animation and environment exports protect existing targets by default."""

import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def animation_data():
    times = np.arange(1800) / 30.0
    positions = 50 + 40 * np.column_stack(
        [np.sin(2 * np.pi * times / 20), np.cos(2 * np.pi * times / 13)]
    )
    env = Environment.from_samples(positions, bin_size=4.0, units="cm")
    return env, np.random.default_rng(0).random((5, env.n_bins)), np.arange(5) / 30.0


def _export(data, path, backend, **kwargs):
    env, fields, times = data
    return env.animate_fields(
        fields,
        frame_times=times,
        backend=backend,
        save_path=str(path),
        n_workers=1,
        dpi=30,
        **kwargs,
    )


def test_video_refuses_existing_file(animation_data, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "out.mp4"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        _export(animation_data, path, "video")
    assert path.read_bytes() == b"keep"
    _export(animation_data, path, "video", overwrite=True)
    assert path.read_bytes() != b"keep"
    assert b"ftyp" in path.read_bytes()[:32]
    assert [item.name for item in tmp_path.iterdir()] == ["out.mp4"]


def test_html_refuses_existing_file_and_frames_dir(animation_data, tmp_path):
    path = tmp_path / "out.html"
    frames = tmp_path / "frames"
    path.write_bytes(b"keep")
    for options in ({}, {"embed": False, "frames_dir": frames}):
        with pytest.raises(FileExistsError, match="overwrite=True"):
            _export(animation_data, path, "html", **options)
        assert path.read_bytes() == b"keep"
        assert not frames.exists()
    path.unlink()
    frames.mkdir()
    sentinel = frames / "sentinel.txt"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        _export(animation_data, path, "html", embed=False, frames_dir=frames)
    assert not path.exists()
    assert sentinel.read_bytes() == b"keep"
    _export(
        animation_data, path, "html", embed=False, frames_dir=frames, overwrite=True
    )
    assert "<html" in path.read_text(encoding="utf-8")
    assert list(frames.glob("*.png"))
    path.write_bytes(b"keep")
    _export(animation_data, path, "html", overwrite=True)
    assert "data:image/png;base64," in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("backend", ["video", "html"])
def test_check_happens_before_rendering(animation_data, tmp_path, monkeypatch, backend):
    from neurospatial.animation import _scene_rendering

    def unexpected_render(*args, **kwargs):
        raise AssertionError("An existing target must be refused before rendering")

    monkeypatch.setattr(_scene_rendering, "SceneRenderer", unexpected_render)
    path = tmp_path / ("out.mp4" if backend == "video" else "out.html")
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        _export(animation_data, path, backend)
    assert path.read_bytes() == b"keep"


def test_env_to_file_wording_shared(animation_data, tmp_path):
    env, _, _ = animation_data
    path = tmp_path / "arena"
    env.to_file(path)
    before = {
        suffix: path.with_suffix(suffix).read_bytes() for suffix in (".json", ".npz")
    }
    with pytest.raises(
        FileExistsError, match="Fix: pass overwrite=True to replace them"
    ) as error:
        env.to_file(path)
    for suffix, content in before.items():
        assert str(path.with_suffix(suffix)) in str(error.value)
        assert path.with_suffix(suffix).read_bytes() == content


def test_html_without_save_path_writes_nothing(animation_data, tmp_path, monkeypatch):
    """No default output file exists that a forgotten save_path could replace."""
    monkeypatch.chdir(tmp_path)
    env, fields, times = animation_data
    with pytest.raises(ValueError, match=r"save_path='animation\.html'"):
        env.animate_fields(fields, frame_times=times, backend="html", dpi=30)
    assert not list(tmp_path.iterdir())


def test_overwrite_is_ignored_with_a_warning_by_interactive_backends(animation_data):
    """Interactive views write no file; overwrite=True warns and is dropped."""
    pytest.importorskip("ipywidgets")
    env, fields, times = animation_data
    with pytest.warns(UserWarning, match="overwrite is used by the video and html"):
        view = env.animate_fields(
            fields, frame_times=times, backend="widget", overwrite=True, dpi=30
        )
    with view:
        assert view.time == times[0]
