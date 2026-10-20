"""Native scale bars retain physical units through current napari zoom events."""

import warnings

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation
from neurospatial.animation.backends._napari_compat import canvas_overlay, scene_camera


@pytest.mark.napari
def test_scene_scale_bar_retains_layer_units_after_zoom():
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), (np.arange(4.0), np.arange(4.0)), units="cm"
    )
    with Animation(
        env, np.ones((2, env.n_bins)), frame_times=[0.0, 0.1], scale_bar=True
    ) as scene:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FutureWarning)
            view = scene.show(backend="napari", show=False)
            viewer = view.native
            assert canvas_overlay(viewer, "scale_bar").visible
            for layer in viewer.layers:
                if hasattr(layer, "units"):
                    assert all(str(unit) == "centimeter" for unit in layer.units)
            camera = scene_camera(viewer)
            camera.zoom *= 2
            canvas = viewer.window._qt_viewer.canvas
            grouped = getattr(canvas, "_viewer_overlay_to_visual", None)
            if grouped is not None:
                visuals = [visual for group in grouped.values() for visual in group]
                bars = [
                    v for v in visuals if type(v).__name__ == "VispyScaleBarOverlay"
                ]
                assert bars
                assert all(bar.node.text.text.endswith(" cm") for bar in bars)
        assert not [w for w in caught if "deprecated" in str(w.message).lower()]


def test_unrecognized_unit_label_does_not_become_a_physical_scale():
    from types import SimpleNamespace

    from neurospatial.animation.config import configure_napari_scale_bar

    bar = SimpleNamespace(visible=True)
    viewer = SimpleNamespace(
        canvas=SimpleNamespace(overlays=SimpleNamespace(scale_bar=bar))
    )
    with pytest.warns(UserWarning, match="scale bar is hidden"):
        configure_napari_scale_bar(viewer, units="arena units")
    assert not bar.visible
