"""animate_fields draws every overlay it is given, with regions, on each backend.

Exports and the notebook view draw the scene the arguments describe: the
same frame as an ``Animation`` built from the same overlays, and not the frame
without them. napari places each overlay in its own layer at the animal's
recorded position.
"""

import base64
import io
import json
import re

import numpy as np
import pytest
from PIL import Image

from neurospatial import Environment, Regions
from neurospatial.animation.overlays import (
    BodypartOverlay,
    HeadDirectionOverlay,
    PositionOverlay,
)
from neurospatial.animation.scene import Animation
from neurospatial.animation.skeleton import Skeleton

TIMES = np.linspace(0, 0.2, 3)


@pytest.fixture
def env():
    rng = np.random.default_rng(42)
    env = Environment.from_samples(rng.uniform(0, 100, (200, 2)), bin_size=10.0)
    regions = Regions()
    regions.add("goal", point=np.array([75.0, 75.0]))
    regions.add("start", point=np.array([25.0, 25.0]))
    return env.with_metadata(units="cm").with_regions(regions)


@pytest.fixture
def fields(env):
    return np.random.default_rng(42).random((len(TIMES), env.n_bins))


def _positions(color="red", seed=0):
    xy = np.random.default_rng(seed).uniform(20, 80, (len(TIMES), 2))
    return PositionOverlay(xy, TIMES, color=color, size=10.0, trail_length=2)


def _bodyparts(edge_color="white", seed=1):
    rng = np.random.default_rng(seed)
    return BodypartOverlay(
        data={part: rng.uniform(20, 80, (len(TIMES), 2)) for part in ("nose", "tail")},
        times=TIMES,
        skeleton=Skeleton(
            name="rat",
            nodes=("nose", "tail"),
            edges=(("nose", "tail"),),
            edge_color=edge_color,
            edge_width=2.0,
        ),
        colors={"nose": "yellow", "tail": "blue"},
    )


def _heading():
    return HeadDirectionOverlay(
        np.linspace(0, np.pi, len(TIMES)), TIMES, color="yellow", length=10.0
    )


OVERLAY_SETS = {
    "position": lambda: [_positions()],
    "bodyparts": lambda: [_bodyparts()],
    # Head direction is drawn at the animal's position.
    "head direction": lambda: [_positions(), _heading()],
    "three animals": lambda: [_positions(c, s) for s, c in enumerate("rgb")],
    "two skeletons": lambda: [_bodyparts("red", 1), _bodyparts("blue", 2)],
    "all": lambda: [_positions(), _bodyparts(), _heading()],
}


def _rendered(env, fields, overlays, show_regions, time):
    with Animation(
        env,
        fields,
        frame_times=TIMES,
        overlays=overlays,
        show_regions=show_regions,
    ) as scene:
        return scene.render_frame(time, dpi=20)


def _html_frame(path, index):
    text = path.read_text(encoding="utf-8")
    frames = json.loads(re.search(r"const frames = (\[.*?\]);", text).group(1))
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(frames[index]))))


@pytest.mark.parametrize("show_regions", [False, True])
@pytest.mark.parametrize("name", list(OVERLAY_SETS))
@pytest.mark.parametrize("backend", ["html", "widget"])
def test_overlays_and_regions_are_drawn(
    env, fields, tmp_path, backend, name, show_regions
):
    if backend == "widget":
        pytest.importorskip("ipywidgets")
    overlays = OVERLAY_SETS[name]()
    result = env.animate_fields(
        fields,
        frame_times=TIMES,
        backend=backend,
        save_path=tmp_path / "out.html" if backend == "html" else None,
        overlays=overlays,
        show_regions=show_regions,
        dpi=20,
    )
    if backend == "html":
        frame = _html_frame(result, 1)
    else:
        try:
            result.seek(0.1)
            image = result.native.children[0].value
            frame = np.asarray(Image.open(io.BytesIO(image)))
        finally:
            result.close()
    expected = _rendered(env, fields, OVERLAY_SETS[name](), show_regions, 0.1)
    np.testing.assert_array_equal(frame, expected)
    assert not np.array_equal(frame, _rendered(env, fields, None, show_regions, 0.1))
    if show_regions:
        assert not np.array_equal(frame, _rendered(env, fields, overlays, False, 0.1))


def test_mismatched_dimensions_raises_error(env, fields):
    """A 3-D trajectory cannot be drawn in a 2-D arena."""
    invalid = PositionOverlay(
        np.random.default_rng(42).uniform(0, 100, (len(TIMES), 3)), TIMES
    )
    with pytest.raises(ValueError, match=r"dimension|shape"):
        env.animate_fields(
            fields, frame_times=TIMES, backend="widget", overlays=[invalid]
        )


@pytest.mark.napari
def test_napari_draws_each_overlay_at_the_recorded_position(env, fields):
    pytest.importorskip("napari")
    animals = [_positions(c, s) for s, c in enumerate("rgb")]
    view = env.animate_fields(
        fields,
        frame_times=TIMES,
        backend="napari",
        overlays=[*animals, _bodyparts()],
        show_regions=True,
        show=False,
    )
    try:
        layers = {layer.name: layer for layer in view.native.layers}
        for name in (
            "Position 1",
            "Position 2",
            "Position 3",
            "Body 1: nose",
            "Body 1: tail",
            "Skeleton 1",
            "Region: goal",
            "Region: start",
        ):
            assert name in layers
        for k, time in enumerate(TIMES):
            view.seek(time)
            for i, animal in enumerate(animals):
                # napari orders coordinates (y, x).
                np.testing.assert_allclose(
                    layers[f"Position {i + 1}"].data, animal.positions[k : k + 1, ::-1]
                )
    finally:
        view.close()
