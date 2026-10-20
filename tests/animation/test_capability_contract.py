"""Capability preflight never silently discards a requested overlay."""

import numpy as np
import pytest

from neurospatial.animation._capabilities import (
    _CAPABILITIES,
    _OVERLAY_NAMES,
    require_overlays,
    warn_unsupported_overlays,
)
from neurospatial.animation.overlays import ObjectVectorData, OverlayData


@pytest.mark.parametrize("backend", ["napari", "video", "widget", "html"])
def test_object_vectors_warn_in_legacy_and_fail_scene_preflight(backend):
    vector = ObjectVectorData(
        animal_positions=np.array([[0.0, 0.0]]),
        object_positions=np.array([[1.0, 1.0]]),
        firing_rates=None,
        color="red",
        linewidth=1.0,
        show_objects=True,
        object_marker="o",
        object_size=5.0,
    )
    data = OverlayData(object_vectors=[vector])
    with pytest.warns(UserWarning, match="ObjectVectorOverlay"):
        filtered = warn_unsupported_overlays(backend, data)
    assert not filtered.object_vectors
    assert data.object_vectors == [vector]
    with pytest.raises(ValueError, match="ObjectVectorOverlay"):
        require_overlays(backend, data)


def test_complete_renderers_advertise_every_known_overlay():
    for name in ("fastplotlib", "scene"):
        assert _CAPABILITIES[name].overlays == set(_OVERLAY_NAMES)
