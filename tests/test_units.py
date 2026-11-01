"""Every spelling of a length unit matches and converts identically."""

import numpy as np
import pytest

from neurospatial._units import (
    declared_units,
    normalize_position_units,
    same_coordinate_declaration,
    units_per_centimeter,
)

SPELLINGS = {
    "m": ["m", "meter", "Meters", "metre", "metres"],
    "cm": ["cm", "centimeter", "centimeters", "centimetre", " centimetres "],
    "mm": ["mm", "millimeter", "millimeters", "millimetre", "millimetres"],
    # Micro sign U+00B5 and Greek small mu U+03BC look identical.
    "um": [
        "um",
        "\u00b5m",
        "\u03bcm",
        "micrometer",
        "micrometers",
        "micrometre",
        "micrometres",
    ],
    "px": ["px", "pixel", "pixels"],
}


@pytest.mark.parametrize("canonical", SPELLINGS)
def test_spellings_share_one_canonical_unit_and_conversion(canonical):
    factors = {units_per_centimeter(spelling) for spelling in SPELLINGS[canonical]}
    assert len(factors) == 1
    for spelling in SPELLINGS[canonical]:
        assert normalize_position_units(spelling) == canonical
        assert declared_units(spelling) == canonical


@pytest.mark.parametrize(
    ("units", "per_centimeter"),
    [("m", 0.01), ("cm", 1.0), ("mm", 10.0), ("um", 1e4)],
)
def test_units_per_centimeter_converts(units, per_centimeter):
    assert units_per_centimeter(units) == pytest.approx(per_centimeter, rel=1e-12)


def test_custom_units_are_kept_and_not_converted():
    assert normalize_position_units(" Arena ") == "arena"
    assert declared_units(" Arena ") == "Arena"
    assert declared_units("  ") is None
    assert declared_units(None) is None
    assert units_per_centimeter("arena") is None
    assert units_per_centimeter("px") is None


def test_overlay_units_alias_matches_environment():
    from neurospatial import Environment
    from neurospatial.animation.overlays import PositionOverlay

    grid = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    with pytest.warns(UserWarning, match="not in the standard registry"):
        env = grid.with_metadata(units="micrometers")
    times = np.array([0.0, 0.1])
    overlay = PositionOverlay(
        np.array([[0.5, 0.5], [1.5, 1.5]]), times=times, units="um"
    )
    overlay.convert_to_data(times, 2, env)


@pytest.mark.parametrize(
    ("name", "first", "second", "strict", "lenient"),
    [
        ("units", "metres", "m", True, True),
        ("units", "cm", "m", False, False),
        ("frame", "arena", "Arena", False, False),
        ("frame", "arena", "arena", True, True),
        ("units", None, "cm", False, True),
        ("frame", "arena", None, False, True),
        ("units", None, None, True, True),
    ],
)
def test_coordinate_declarations_match_by_policy(name, first, second, strict, lenient):
    assert (
        same_coordinate_declaration(name, first, second, undeclared_matches=False)
        is strict
    )
    assert (
        same_coordinate_declaration(name, first, second, undeclared_matches=True)
        is lenient
    )
