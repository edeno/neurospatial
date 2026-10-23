"""Long sessions stay on disk: animate_fields reads only the rows it shows."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def env():
    positions = np.random.default_rng(42).standard_normal((100, 2)) * 50
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.mark.xfail(
    strict=True,
    reason="The persistent SceneRenderer reads the first shown row three times while building its artists.",
)
def test_a_lazy_session_is_read_only_at_the_shown_time(env):
    """A view reads a bounded sample for the color scale and the shown row."""
    pytest.importorskip("ipywidgets")
    n_rows = 20_000

    class LazyRows:
        shape = (n_rows, env.n_bins)

        def __init__(self):
            self.reads = []

        def __len__(self):
            return n_rows

        def __array__(self, *args):
            raise AssertionError("animate_fields must not materialize the session")

        def __getitem__(self, index):
            self.reads.append(index)
            return np.full(env.n_bins, index / n_rows)

    rows = LazyRows()
    times = np.arange(n_rows) / 250
    view = env.animate_fields(
        {"rate": rows}, frame_times=times, backend="widget", dpi=20
    )
    with view:
        assert view.animation.fields["rate"].values is rows
        # At most 100 rows set the color scale, plus the first frame shown.
        assert len(rows.reads) <= 101
        before = len(rows.reads)
        view.seek(10.0)
        assert rows.reads[before:] == [2500]


def test_a_memory_mapped_session_is_not_copied(env, tmp_path):
    pytest.importorskip("ipywidgets")
    fields = np.memmap(
        tmp_path / "fields.dat", dtype="float32", mode="w+", shape=(15_000, env.n_bins)
    )
    fields[:] = 0.5
    fields.flush()
    view = env.animate_fields(
        fields,
        frame_times=np.arange(15_000) / 250,
        backend="widget",
        vmin=0,
        vmax=1,
        dpi=20,
    )
    with view:
        assert view.animation.fields["field"].values is fields
    assert not fields._mmap.closed


@pytest.mark.slow
def test_subsample_with_memmap_preserves_type(tmp_path, env):
    """Test subsample_frames() works with memory-mapped arrays."""
    from neurospatial.animation.core import subsample_frames

    rng = np.random.default_rng(42)
    n_frames = 1000
    memmap_file = tmp_path / "fields.dat"
    fields = np.memmap(
        memmap_file, dtype="float32", mode="w+", shape=(n_frames, env.n_bins)
    )
    fields[0] = rng.random(env.n_bins)
    fields.flush()

    subsampled = subsample_frames(fields, source_fps=250, target_fps=30)

    assert isinstance(subsampled, np.ndarray)
    expected_frames = len(np.arange(0, n_frames, 250 / 30))
    assert len(subsampled) == expected_frames
    assert subsampled.shape[1] == env.n_bins
    np.testing.assert_array_equal(subsampled[0], fields[0])


def test_subsampled_session_exports_on_its_own_clock(env, tmp_path):
    """memmap -> subsample_frames -> HTML player, as in the subsample docs."""
    from neurospatial.animation.core import subsample_frames

    n_frames = 500  # 2 s at 250 Hz
    fields = np.memmap(
        tmp_path / "fields.dat",
        dtype="float32",
        mode="w+",
        shape=(n_frames, env.n_bins),
    )
    fields[:] = np.linspace(0, 1, n_frames)[:, None]
    fields.flush()
    times = np.arange(n_frames) / 250
    path = env.animate_fields(
        subsample_frames(fields, source_fps=250, target_fps=10),
        frame_times=subsample_frames(times, source_fps=250, target_fps=10),
        save_path=tmp_path / "session.html",
        dpi=20,
    )
    text = path.read_text(encoding="utf-8")
    labels = json.loads(re.search(r"const labels = (\[.*?\]);", text).group(1))
    # 20 rows at 10 Hz span [0, 1.9] s; the player keeps their physical times.
    assert labels == [f"t = {k / 10:.3f} s" for k in range(20)]
