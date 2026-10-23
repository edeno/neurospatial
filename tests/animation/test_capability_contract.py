"""Capability preflight never silently discards a requested overlay."""

from neurospatial.animation._capabilities import _CAPABILITIES, _OVERLAY_NAMES


def test_complete_renderers_advertise_every_known_overlay():
    for name in ("fastplotlib", "napari_scene", "scene"):
        assert _CAPABILITIES[name].overlays == set(_OVERLAY_NAMES)
