"""Animation errors say what failed, why, and how to fix it.

Field, frame-time, backend and save-path errors of ``animate_fields`` are
checked in ``test_core.py``; export option errors in ``test_scene_export.py``.
"""

import sys
from unittest.mock import patch

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import subsample_frames
from neurospatial.animation.core import animate_fields


@pytest.fixture
def test_env():
    rng = np.random.default_rng(42)
    return Environment.from_samples(rng.standard_normal((100, 2)) * 50, bin_size=10.0)


@pytest.fixture
def test_fields(test_env):
    return np.random.default_rng(42).random((3, test_env.n_bins))


def test_ipywidgets_missing_error_message(test_env, test_fields):
    with (
        patch.dict(sys.modules, {"ipywidgets": None}),
        pytest.raises(ImportError, match="requires ipywidgets") as error,
    ):
        animate_fields(
            test_env, test_fields, frame_times=np.arange(3) / 10, backend="widget"
        )
    assert "backend='fastplotlib'" in str(error.value)


def test_pickle_failure_error_message(test_env, test_fields, tmp_path, monkeypatch):
    """Worker export says what cannot be pickled and offers n_workers=1."""
    # Transient caches are omitted from pickle; persistent caller state is not.
    monkeypatch.setitem(test_env.__dict__, "user_annotation", {"bad": lambda: None})
    with pytest.raises(ValueError, match="environment cannot be pickled") as error:
        animate_fields(
            test_env,
            test_fields,
            frame_times=np.arange(3) / 10,
            save_path=tmp_path / "out.html",
            n_workers=2,
        )
    message = str(error.value)
    assert "n_workers=1" in message
    assert "lambdas" in message
    assert list(tmp_path.iterdir()) == []


def test_html_max_frames_error_message(test_env, tmp_path):
    """A page too large for a browser is refused before any frame renders."""
    fields = np.random.default_rng(42).random((2000, test_env.n_bins))
    with pytest.raises(ValueError, match="2000 frames") as error:
        animate_fields(
            test_env,
            fields,
            frame_times=np.linspace(0, 200, 2000),
            save_path=tmp_path / "out.html",
        )
    message = str(error.value)
    assert "max_html_frames=500" in message
    assert "embed=False" in message
    assert "time_range=" in message
    assert list(tmp_path.iterdir()) == []


def test_n_workers_validation_error_message(test_env, test_fields, tmp_path):
    with pytest.raises(ValueError, match="n_workers must be a positive integer"):
        animate_fields(
            test_env,
            test_fields,
            frame_times=np.arange(3) / 10,
            save_path=tmp_path / "out.html",
            n_workers=-1,
        )
    assert list(tmp_path.iterdir()) == []


def test_target_fps_exceeds_source_error_message():
    """Verify subsample_frames fps validation is clear."""
    fields = [np.array([1, 2, 3]) for _ in range(100)]

    with pytest.raises(ValueError) as exc_info:
        subsample_frames(fields, target_fps=100, source_fps=50)

    error_msg = str(exc_info.value)
    assert "100" in error_msg
    assert "50" in error_msg
    assert "exceed" in error_msg.lower()
