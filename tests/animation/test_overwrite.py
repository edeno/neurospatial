"""Animation and environment exports protect existing targets by default."""

import shutil
import subprocess

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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_video_refuses_existing_file(animation_data, tmp_path):
    path = tmp_path / "out.mp4"
    path.write_bytes(b"keep")
    for options in ({}, {"dry_run": True}):
        with pytest.raises(FileExistsError, match="overwrite=True"):
            _export(animation_data, path, "video", **options)
        assert path.read_bytes() == b"keep"
    _export(animation_data, path, "video", overwrite=True)
    assert path.read_bytes() != b"keep"
    assert b"ftyp" in path.read_bytes()[:32]


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


def test_empty_frames_directory_is_writable(animation_data, tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    path = tmp_path / "out.html"
    _export(animation_data, path, "html", embed=False, frames_dir=frames)
    assert path.exists()
    assert len(list(frames.glob("*.png"))) == 5


@pytest.mark.parametrize("backend", ["video", "html"])
def test_check_happens_before_rendering(animation_data, tmp_path, monkeypatch, backend):
    def unexpected_render(*args, **kwargs):
        raise AssertionError("An existing target must be refused before rendering")

    monkeypatch.setattr(
        "neurospatial.animation._parallel.parallel_render_frames", unexpected_render
    )
    monkeypatch.setattr(
        "neurospatial.animation.rendering.render_field_to_image_bytes",
        unexpected_render,
    )
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
@pytest.mark.parametrize("overwrite", [False, True])
def test_ffmpeg_never_overwrites_without_opt_in(
    animation_data, tmp_path, monkeypatch, overwrite
):
    commands = []
    run = subprocess.run

    def record(command, *args, **kwargs):
        if command[0] == "ffmpeg" and "-version" not in command:
            commands.append(command)
        return run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", record)
    _export(animation_data, tmp_path / "out.mp4", "video", overwrite=overwrite)
    assert len(commands) == 1
    assert ("-y" if overwrite else "-n") in commands[0]
    assert ("-n" if overwrite else "-y") not in commands[0]


def test_default_html_path_is_protected(animation_data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "animation.html"
    path.write_bytes(b"keep")
    env, fields, times = animation_data
    with pytest.raises(FileExistsError, match="overwrite=True"):
        env.animate_fields(fields, frame_times=times, backend="html", dpi=30)
    assert path.read_bytes() == b"keep"


@pytest.mark.parametrize("backend", ["napari", "widget"])
def test_overwrite_is_not_forwarded_to_interactive_backends(
    animation_data, monkeypatch, backend
):
    received = []

    def interactive(env, fields, **kwargs):
        received.append(kwargs)

    monkeypatch.setattr(
        f"neurospatial.animation.backends.{backend}_backend.render_{backend}",
        interactive,
    )
    env, fields, times = animation_data
    env.animate_fields(fields, frame_times=times, backend=backend, overwrite=True)
    assert len(received) == 1
    assert "overwrite" not in received[0]
