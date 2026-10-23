"""How animate_fields reads one field or several side-by-side fields.

A 2-D array or a list of ``(n_bins,)`` frames is one field. A mapping, or a
list whose items are themselves lists of frames, holds several fields that
share the recording clock. Every field is checked against the environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment

pytest.importorskip("ipywidgets")

TIMES = np.array([0.0, 0.1, 0.2])


@pytest.fixture(scope="module")
def env():
    return Environment.from_grid_mask(
        np.ones((3, 4), dtype=bool), (np.arange(4.0), np.arange(5.0))
    )


def _view(env, fields, **options):
    """Open a notebook view, which builds the scene without a display."""
    return env.animate_fields(fields, frame_times=TIMES, backend="widget", **options)


def _rows(env, scale):
    return [np.full(env.n_bins, scale * (k + 1)) for k in range(len(TIMES))]


@pytest.mark.parametrize(
    "make",
    [
        lambda rows: rows,  # list of frames
        lambda rows: np.array(rows),  # (n_frames, n_bins) array
        lambda rows: [np.atleast_1d(row) for row in rows],
    ],
    ids=["list", "array", "list-of-1d-arrays"],
)
def test_frames_of_one_field_are_one_stream(env, make):
    rows = _rows(env, 1.0)
    with _view(env, make(rows)) as view:
        assert list(view.animation.fields) == ["field"]
        for time, row in zip(TIMES, rows, strict=True):
            np.testing.assert_array_equal(
                view.animation.sample_fields(time)["field"], row
            )


def test_lists_of_frames_are_named_fields_on_one_clock(env):
    first, second = _rows(env, 1.0), _rows(env, 10.0)
    with _view(env, [first, second]) as view:
        assert list(view.animation.fields) == ["field 1", "field 2"]
    with _view(env, [first, second], layer_names=["posterior", "likelihood"]) as view:
        assert list(view.animation.fields) == ["posterior", "likelihood"]
        for k, time in enumerate(TIMES):
            sampled = view.animation.sample_fields(time)
            np.testing.assert_array_equal(sampled["posterior"], first[k])
            np.testing.assert_array_equal(sampled["likelihood"], second[k])
    with _view(env, {"left": np.array(first), "right": np.array(second)}) as view:
        assert list(view.animation.fields) == ["left", "right"]


def test_explicit_color_limits_apply_to_every_field(env):
    with _view(env, [_rows(env, 1.0), _rows(env, 10.0)], vmin=0.0, vmax=5.0) as view:
        assert {
            name: f.color_limits() for name, f in view.animation.fields.items()
        } == {
            "field 1": (0.0, 5.0),
            "field 2": (0.0, 5.0),
        }


def test_side_by_side_fields_share_one_color_scale_by_default(env):
    """Fields shown side by side are compared, so one color means one value."""
    with _view(env, [_rows(env, 0.1), _rows(env, 1.0)]) as view:
        limits = {f.color_limits() for f in view.animation.fields.values()}
    assert len(limits) == 1
    ((low, high),) = limits
    assert low <= 0.1 and high >= 2.9


def test_closing_the_view_releases_the_scene(env):
    view = _view(env, _rows(env, 1.0))
    view.close()
    with pytest.raises(RuntimeError):
        view.animation.sample_fields(0.0)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        # A field of three frames beside a single frame, read as 12 frames.
        (
            lambda env: [_rows(env, 1.0), np.ones(env.n_bins)],
            r"'field 2' has 12 frames, but frame_times has 3",
        ),
        # A frame beside a field is one field of two frames.
        (
            lambda env: [np.ones(env.n_bins), _rows(env, 1.0)],
            r"frame_times length \(3\) must match n_frames \(2\)",
        ),
        # Fields of different lengths cannot share the frame clock.
        (
            lambda env: [_rows(env, 1.0), _rows(env, 1.0)[:2]],
            r"'field 2' has 2 frames, but frame_times has 3",
        ),
        # A field with too many bins, alone or beside a valid one.
        (lambda env: [np.ones(env.n_bins + 2)] * 3, r"14 bins; the environment has 12"),
        (
            lambda env: [_rows(env, 1.0), [np.ones(env.n_bins + 2)] * 3],
            r"'field 2' has 14 bins; the environment has 12",
        ),
    ],
    ids=[
        "field-then-frame",
        "frame-then-field",
        "different-lengths",
        "wrong-bins",
        "second-field-wrong-bins",
    ],
)
def test_inconsistent_fields_are_refused(env, fields, match):
    with pytest.raises(ValueError, match=match):
        _view(env, fields(env))


def test_layer_names_must_name_every_field(env):
    with pytest.raises(ValueError, match="2 names for 3 fields"):
        _view(env, [_rows(env, 1.0)] * 3, layer_names=["a", "b"])
