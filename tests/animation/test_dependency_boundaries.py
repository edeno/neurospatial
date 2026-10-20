"""Dependency boundaries for shared video I/O and overlay styling."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    ("blocked", "operation"),
    [
        (
            ("neurospatial.animation", "matplotlib", "cv2"),
            "from neurospatial._video_io import CacheInfo, VideoReaderProtocol, _VideoReader",
        ),
        (
            ("neurospatial.animation", "matplotlib.pyplot", "matplotlib.backends"),
            """
            from pathlib import Path
            import numpy as np
            from neurospatial import _video_io
            from neurospatial.annotation.core import _load_video_frame

            class Reader:
                n_frames = 1
                def __init__(self, path):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *exc):
                    pass
                def __getitem__(self, index):
                    return np.zeros((3, 4, 3), dtype=np.uint8)

            _video_io._VideoReader = Reader
            assert _load_video_frame(Path(__file__), 0).shape == (3, 4, 3)
            """,
        ),
        (
            (
                "neurospatial.animation._parallel",
                "matplotlib.pyplot",
                "matplotlib.collections",
                "matplotlib.quiver",
                "matplotlib.backends.backend_agg",
                "matplotlib.backends.backend_qtagg",
            ),
            """
            import numpy as np
            from neurospatial.animation._style import (
                _event_visibility, _head_direction_color, _build_trail_segments,
            )
            shown, alpha = _event_visibility(np.array([0, 1, 3]), 1, 1)
            assert shown.tolist() == [True, True, False]
            assert alpha[:2].tolist() == [0.5, 1.0]
            assert _head_direction_color("red", np.array([1., 0.])) == (1., 0., 0., 1.)
            segments, colors = _build_trail_segments(np.array([[0., 0.], [1., 1.]]), "red")
            assert len(segments) == 1 and colors[0][-1] == 0.7
            """,
        ),
    ],
    ids=["shared-reader", "annotation-frame", "overlay-style"],
)
def test_import_without_rendering_pipeline(blocked, operation):
    """Use a fresh interpreter so prior test imports cannot hide coupling."""
    script = f"""
import importlib.abc
import sys

class BlockRendering(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in {blocked!r}):
            raise AssertionError("Unexpected rendering import: " + fullname)

sys.meta_path.insert(0, BlockRendering())
__file__ = sys.executable
""" + textwrap.dedent(operation)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_legacy_reader_names_and_pickles_resolve_canonical_classes():
    """Old private imports and pickle globals still resolve after the move."""
    import pickle

    from neurospatial import _video_io
    from neurospatial.animation import _video_io as legacy

    for name in ("_VideoReader", "CacheInfo", "VideoReaderProtocol"):
        canonical = getattr(_video_io, name)
        assert getattr(legacy, name) is canonical
        old_pickle = f"cneurospatial.animation._video_io\n{name}\n.".encode()
        assert pickle.loads(old_pickle) is canonical
