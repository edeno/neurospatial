"""Every spelling of a length unit matches and converts identically."""

from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial._units import normalize_position_units, units_per_centimeter
from neurospatial.io.nwb._adapters import position_units_from_series

SPELLINGS = {
    "m": ["m", "meter", "Meters", "metre", "metres"],
    "cm": ["cm", "centimeter", "centimeters", "centimetre", " centimetres "],
    "mm": ["mm", "millimeter", "millimeters", "millimetre", "millimetres"],
    "um": ["um", "µm", "micrometer", "micrometers", "micrometre", "micrometres"],
    "px": ["px", "pixel", "pixels"],
}


@pytest.mark.parametrize("canonical", SPELLINGS)
def test_spellings_share_one_canonical_unit_and_conversion(canonical):
    factors = {units_per_centimeter(spelling) for spelling in SPELLINGS[canonical]}
    assert len(factors) == 1
    for spelling in SPELLINGS[canonical]:
        assert normalize_position_units(spelling) == canonical
        series = SimpleNamespace(unit=spelling)
        assert position_units_from_series(series) == canonical


def test_custom_units_are_kept_and_not_converted():
    assert normalize_position_units(" Arena ") == "arena"
    assert position_units_from_series(SimpleNamespace(unit="Arena")) == "Arena"
    assert units_per_centimeter("arena") is None
    assert units_per_centimeter("px") is None


def test_overlay_units_alias_matches_environment():
    from neurospatial import Environment
    from neurospatial.animation.overlays import PositionOverlay

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    ).with_metadata(units="micrometers")
    times = np.array([0.0, 0.1])
    overlay = PositionOverlay(
        np.array([[0.5, 0.5], [1.5, 1.5]]), times=times, units="um"
    )
    overlay.convert_to_data(times, 2, env)
