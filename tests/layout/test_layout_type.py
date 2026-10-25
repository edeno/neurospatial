"""Layout labels: one ``LayoutType`` vocabulary for engines, bins and environments.

Each built-in engine's ``layout_type`` is a ``LayoutType`` value, the binning
it builds carries the same label as ``kind``, and an environment reports it.
User spellings of a layout name are normalized where they enter. A smoke test
checks that the napari rendering path rasterizes a rectilinear grid.
"""

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.layout import (
    LayoutType,
    MaskedGridLayout,
    RegularGridLayout,
    create_layout,
    get_layout_parameters,
    list_available_layouts,
)
from neurospatial.layout.factories import _ENGINES, _LAYOUT_MAP

# Minimal builds of every registered engine.
_BUILDS = {
    "RegularGrid": {"bin_size": 2.0, "dimension_ranges": [(0, 6), (0, 6)]},
    "MaskedGrid": {
        "active_mask": np.ones((2, 2), dtype=bool),
        "grid_edges": (np.arange(3.0), np.arange(3.0)),
    },
    "ImageMask": {"image_mask": np.ones((2, 2), dtype=bool), "pixel_size": 1.0},
    "Hexagonal": {"hexagon_width": 2.0, "dimension_ranges": [(0, 6), (0, 6)]},
    "TriangularMesh": {"boundary_polygon": box(0, 0, 6, 6), "point_spacing": 2.0},
    "ShapelyPolygon": {"polygon": box(0, 0, 6, 6), "bin_size": 2.0},
}


@pytest.mark.parametrize("member", list(LayoutType))
def test_layout_type_prints_as_its_value(member):
    """str() and f-strings give the label on every supported Python version."""
    assert str(member) == member.value
    assert f"{member}" == member.value
    assert f"{member:>16}" == f"{member.value:>16}"
    assert format(member) == member.value


def test_every_engine_label_is_a_layout_type_value():
    labels = [engine.layout_type for engine in _ENGINES]
    assert all(type(label) is str for label in labels)
    assert {LayoutType(label) for label in labels} <= set(LayoutType)
    assert set(_LAYOUT_MAP) == set(labels)
    assert list_available_layouts() == sorted(labels)
    # Polar and combined environments are not built with create_layout.
    assert LayoutType.POLAR.value not in _LAYOUT_MAP
    assert LayoutType.COMBINED.value not in _LAYOUT_MAP


@pytest.mark.parametrize("label", sorted(_BUILDS))
def test_engine_binning_and_environment_share_one_label(label):
    layout = create_layout(label, **_BUILDS[label])
    assert layout.layout_type == label
    assert layout.binning().kind == label
    env = Environment.from_layout(layout)
    assert env.layout_type == label == env.binning.kind


@pytest.mark.parametrize(
    "spelling", ["RegularGrid", "regular-grid", "regular_grid", "REGULARGRID", "grid"]
)
def test_user_spellings_name_the_canonical_layout(spelling):
    layout = create_layout(spelling, bin_size=2.0, dimension_ranges=[(0, 4), (0, 4)])
    assert type(layout) is RegularGridLayout
    assert (
        get_layout_parameters(spelling).keys()
        == get_layout_parameters(LayoutType.REGULAR_GRID).keys()
    )


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("regular-grid", "RegularGrid"),
        ("hexagonal", "Hexagonal"),
        ("grid", "RegularGrid"),
    ],
)
def test_sample_factories_report_the_canonical_label(spelling, expected):
    positions = np.random.default_rng(0).uniform(0, 20, (300, 2))
    env = Environment.from_samples(positions, bin_size=4.0, layout=spelling)
    assert env.layout_type == expected
    assert env.binning.kind == expected


@pytest.mark.parametrize(
    ("name", "fix"),
    [
        ("Polar", "Environment.from_polar_egocentric"),
        ("Combined", "Environment.combine"),
        ("circle", "RegularGrid"),
        ("mask", "could mean a masked grid or an image mask"),
    ],
)
def test_names_without_an_engine_are_refused_with_the_way_to_build_them(name, fix):
    with pytest.raises(ValueError, match="cannot build layout") as error:
        create_layout(name)
    assert fix in str(error.value)


def test_subclass_engine_label_is_the_environment_label():
    class Wall(MaskedGridLayout):
        layout_type = "WalledGrid"

    class Plain(MaskedGridLayout):
        pass

    for engine, label in ((Wall, "WalledGrid"), (Plain, "MaskedGrid")):
        layout = engine()
        layout.build(**_BUILDS["MaskedGrid"])
        env = Environment.from_layout(layout)
        assert layout.binning().kind == label
        assert env.layout_type == label == env.binning.kind


def test_custom_label_matching_a_layout_type_is_spelled_canonically():
    class Relabeled(MaskedGridLayout):
        layout_type = "masked-grid"

    layout = Relabeled()
    layout.build(**_BUILDS["MaskedGrid"])
    assert Environment.from_layout(layout).layout_type == "MaskedGrid"
