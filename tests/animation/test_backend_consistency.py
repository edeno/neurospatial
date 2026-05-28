"""Cross-backend consistency for animation rendering.

The four backends do NOT produce bit-identical frames by design: the HTML
backend embeds the bare matplotlib frame (``render_field_to_png_bytes``), the
widget backend wraps that frame with a frame label, the video backend re-encodes
it with lossy H.264, and the napari backend renders the field on the GPU with
its own colormap (different resolution entirely). So pixel-exact 4-way parity is
not a property of this codebase.

What IS verifiable, and what these tests pin, is that every backend applies the
*same field-to-color mapping*:

- the HTML backend's embedded frame is pixel-identical to the shared
  ``render_field_to_png_bytes`` canonical frame;
- the video backend's decoded frame matches that canonical frame within H.264
  compression error (not a blank or wrong frame);
- the widget backend's frame matches the canonical frame everywhere except the
  small frame-label chrome;
- every backend visibly *honors* ``vmin``/``vmax`` and ``cmap`` -- changing the
  parameter changes the output. This is the regression guard for the audit's
  concern that a backend could silently ignore ``vmin``/``vmax``.

The napari backend is GPU-rendered (different resolution, no matplotlib axes)
and is not pixel-comparable to the matplotlib family, so it is not exercised
here. napari rendering is covered by the dedicated, environment-gated
napari-backend tests; a live napari viewer requires a working Qt/OpenGL context
that is not available in a headless run.
"""

from __future__ import annotations

import base64
import io
import re

import numpy as np
import pytest

from neurospatial import Environment

pytestmark = pytest.mark.integration

# Frame index compared across backends.
_FRAME = 2
_DPI = 100


@pytest.fixture
def canonical_parity_data():
    """Small deterministic (env, fields, params) shared by all parity tests.

    A compact environment keeps per-backend rendering fast while still
    exercising the full field-to-color pipeline.
    """
    rng = np.random.default_rng(42)
    env = Environment.from_samples(rng.uniform(0, 32, (2000, 2)), bin_size=2.0)
    fields = [rng.uniform(0, 10, env.n_bins) for _ in range(4)]
    return {
        "env": env,
        "fields": fields,
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 10.0,
    }


def _decode_png(png_bytes: bytes) -> np.ndarray:
    """Decode PNG/JPEG bytes to an (H, W, 3) uint8 RGB array."""
    pil = pytest.importorskip("PIL.Image")
    return np.array(pil.open(io.BytesIO(png_bytes)).convert("RGB"))


def _canonical_frame(data: dict, *, vmin=None, vmax=None, cmap=None) -> np.ndarray:
    """The shared matplotlib frame that every backend's color mapping derives from."""
    from neurospatial.animation.rendering import render_field_to_png_bytes

    return _decode_png(
        render_field_to_png_bytes(
            data["env"],
            data["fields"][_FRAME],
            cmap if cmap is not None else data["cmap"],
            data["vmin"] if vmin is None else vmin,
            data["vmax"] if vmax is None else vmax,
            dpi=_DPI,
        )
    )


def _html_frame(data: dict, tmp_path, *, vmin=None, vmax=None, cmap=None) -> np.ndarray:
    from neurospatial.animation.backends.html_backend import render_html

    out = tmp_path / "anim.html"
    render_html(
        data["env"],
        data["fields"],
        str(out),
        fps=4,
        cmap=cmap if cmap is not None else data["cmap"],
        vmin=data["vmin"] if vmin is None else vmin,
        vmax=data["vmax"] if vmax is None else vmax,
        dpi=_DPI,
    )
    # Frames are embedded as raw base64 PNGs in a `const frames = [...]` array.
    b64_frames = re.findall(r'"([A-Za-z0-9+/=]{100,})"', out.read_text())
    return _decode_png(base64.b64decode(b64_frames[_FRAME]))


def _video_frame(
    data: dict, tmp_path, *, vmin=None, vmax=None, cmap=None
) -> np.ndarray:
    cv2 = pytest.importorskip("cv2")
    from neurospatial.animation.backends.video_backend import render_video

    out = tmp_path / "anim.mp4"
    render_video(
        data["env"],
        data["fields"],
        str(out),
        fps=4,
        cmap=cmap if cmap is not None else data["cmap"],
        vmin=data["vmin"] if vmin is None else vmin,
        vmax=data["vmax"] if vmax is None else vmax,
        dpi=_DPI,
    )
    cap = cv2.VideoCapture(str(out))
    frames = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames[_FRAME]


def _widget_frame(data: dict, *, vmin=None, vmax=None, cmap=None) -> np.ndarray:
    from neurospatial.animation.backends.widget_backend import (
        render_field_to_png_bytes_with_overlays,
    )

    return _decode_png(
        render_field_to_png_bytes_with_overlays(
            data["env"],
            data["fields"][_FRAME],
            cmap if cmap is not None else data["cmap"],
            data["vmin"] if vmin is None else vmin,
            data["vmax"] if vmax is None else vmax,
            _DPI,
            _FRAME,
        )
    )


class TestBackendColorMappingParity:
    """Each matplotlib-family backend renders the shared field-to-color mapping."""

    def test_html_frame_pixel_identical_to_canonical(
        self, canonical_parity_data, tmp_path
    ):
        # HTML embeds the bare canonical PNG -> must be pixel-exact.
        canonical = _canonical_frame(canonical_parity_data)
        html_frame = _html_frame(canonical_parity_data, tmp_path)
        assert html_frame.shape == canonical.shape
        assert np.array_equal(html_frame, canonical)

    def test_video_frame_matches_canonical_within_compression(
        self, canonical_parity_data, tmp_path
    ):
        canonical = _canonical_frame(canonical_parity_data)
        video_frame = _video_frame(canonical_parity_data, tmp_path)
        assert video_frame.shape == canonical.shape
        # H.264 is lossy: individual pixels can differ a lot, but the *average*
        # difference is tiny if it is the same frame. A blank/wrong frame would
        # have a large mean difference.
        mean_abs_diff = np.abs(video_frame.astype(int) - canonical.astype(int)).mean()
        assert mean_abs_diff < 8.0

    def test_widget_frame_matches_canonical_outside_chrome(self, canonical_parity_data):
        canonical = _canonical_frame(canonical_parity_data)
        widget_frame = _widget_frame(canonical_parity_data)
        assert widget_frame.shape == canonical.shape
        # Widget adds a small frame-label; the field region is identical.
        per_pixel_diff = np.abs(widget_frame.astype(int) - canonical.astype(int)).sum(
            axis=2
        )
        frac_differing = (per_pixel_diff > 5).mean()
        assert frac_differing < 0.05  # <5% of pixels differ (the chrome)
        assert per_pixel_diff.mean() < 5.0


class TestBackendsHonorColorParameters:
    """Changing vmin/vmax or cmap must change each backend's output.

    A backend that silently ignored the parameter (the audit's specific concern)
    would produce identical output and fail these tests.
    """

    # mean abs-diff threshold: honoring the parameter produces tens of levels of
    # change; ignoring it produces ~0.
    _CHANGE_THRESHOLD = 5.0

    def test_html_honors_vmin_vmax(self, canonical_parity_data, tmp_path):
        # Each call writes and is read back immediately, so reusing tmp_path
        # (the second render overwrites the first) is safe.
        a = _html_frame(canonical_parity_data, tmp_path, vmax=10.0)
        b = _html_frame(canonical_parity_data, tmp_path, vmax=100.0)
        assert np.abs(a.astype(int) - b.astype(int)).mean() > self._CHANGE_THRESHOLD

    def test_html_honors_cmap(self, canonical_parity_data, tmp_path):
        a = _html_frame(canonical_parity_data, tmp_path, cmap="viridis")
        b = _html_frame(canonical_parity_data, tmp_path, cmap="hot")
        assert np.abs(a.astype(int) - b.astype(int)).mean() > self._CHANGE_THRESHOLD

    def test_video_honors_vmin_vmax(self, canonical_parity_data, tmp_path):
        a = _video_frame(canonical_parity_data, tmp_path, vmax=10.0)
        b = _video_frame(canonical_parity_data, tmp_path, vmax=100.0)
        assert np.abs(a.astype(int) - b.astype(int)).mean() > self._CHANGE_THRESHOLD

    def test_video_honors_cmap(self, canonical_parity_data, tmp_path):
        a = _video_frame(canonical_parity_data, tmp_path, cmap="viridis")
        b = _video_frame(canonical_parity_data, tmp_path, cmap="hot")
        assert np.abs(a.astype(int) - b.astype(int)).mean() > self._CHANGE_THRESHOLD

    def test_widget_honors_vmin_vmax(self, canonical_parity_data):
        a = _widget_frame(canonical_parity_data, vmax=10.0)
        b = _widget_frame(canonical_parity_data, vmax=100.0)
        assert np.abs(a.astype(int) - b.astype(int)).mean() > self._CHANGE_THRESHOLD

    def test_widget_honors_cmap(self, canonical_parity_data):
        a = _widget_frame(canonical_parity_data, cmap="viridis")
        b = _widget_frame(canonical_parity_data, cmap="hot")
        assert np.abs(a.astype(int) - b.astype(int)).mean() > self._CHANGE_THRESHOLD
