"""The notebook widget shows the scene frame at the selected recording time."""

from __future__ import annotations

import io

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationField

FRAME_TIMES = np.arange(5) / 10


def _grid_env():
    positions = np.random.default_rng(42).random((100, 2)) * 50
    return Environment.from_samples(positions, bin_size=10.0)


def _hex_env():
    positions = np.random.default_rng(42).random((100, 2)) * 50
    return Environment.from_samples(
        positions,
        bin_size=10.0,
        layout="Hexagonal",
        infer_active_bins=True,
        bin_count_threshold=0,
    )


def _shown(view) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(view.native.children[0].value)))


@pytest.mark.parametrize("make_env", [_grid_env, _hex_env], ids=["grid", "hexagonal"])
def test_shown_image_matches_a_fresh_render_after_other_seeks(make_env):
    """Reusing the display figure never leaks an earlier frame's values."""
    pytest.importorskip("ipywidgets")
    env = make_env()
    fields = np.random.default_rng(0).random((len(FRAME_TIMES), env.n_bins))
    view = env.animate_fields(
        fields, frame_times=FRAME_TIMES, backend="widget", vmin=0, vmax=1, dpi=30
    )
    with view:
        first = _shown(view)
        view.seek(0.3)
        view.seek(0.1)
        shown = _shown(view)
    assert not np.array_equal(first, shown)
    with Animation(
        env, {"field": AnimationField(fields, FRAME_TIMES, vmin=0, vmax=1)}
    ) as fresh:
        np.testing.assert_array_equal(shown, fresh.render_frame(0.1, dpi=30))


def test_playback_advances_recording_seconds_at_the_capped_rate():
    pytest.importorskip("ipywidgets")
    env = _grid_env()
    fields = np.zeros((len(FRAME_TIMES), env.n_bins))
    view = env.animate_fields(
        fields,
        frame_times=FRAME_TIMES,
        backend="widget",
        speed=2.0,
        max_playback_fps=20,
        dpi=20,
    )
    with view:
        play, slider = view.native.children[1].children
        # A 50 ms tick at twice real time advances 0.1 s of recording.
        assert play.interval == 50
        assert slider.step == pytest.approx(0.1)
        assert (slider.min, slider.max) == (0.0, 0.4)
        assert slider.description == "Time (s)"


def test_missing_ipywidgets_names_the_package_and_alternatives(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    env = _grid_env()
    with pytest.raises(ImportError, match=r"ipywidgets.*backend='fastplotlib'"):
        env.animate_fields(
            np.zeros((len(FRAME_TIMES), env.n_bins)),
            frame_times=FRAME_TIMES,
            backend="widget",
        )
