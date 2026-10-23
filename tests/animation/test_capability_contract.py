"""Capability preflight never silently discards a requested overlay."""

from dataclasses import fields

from neurospatial.animation._capabilities import _OVERLAY_NAMES, _VIEWS
from neurospatial.animation.overlays import OverlayData


def test_every_overlay_collection_has_a_name_for_refusals():
    assert set(_OVERLAY_NAMES) == {item.name for item in fields(OverlayData)}


def test_views_leave_out_only_known_overlays_and_say_why():
    assert set(_VIEWS) == {"widget", "napari", "fastplotlib"}
    for dims in _VIEWS.values():
        assert set(dims) <= {1, 2, 3}
        for drawn in dims.values():
            assert drawn.left_out <= set(_OVERLAY_NAMES)
            assert bool(drawn.left_out) == bool(drawn.why) == bool(drawn.fix)
