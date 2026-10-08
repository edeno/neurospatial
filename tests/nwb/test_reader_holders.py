"""Named NWB reader outputs preserve physical data and acquisition coverage."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

pynwb = pytest.importorskip("pynwb")

from neurospatial.io.nwb import (  # noqa: E402
    NWBHeadDirection,
    NWBPosition,
    NWBUnits,
    read_head_direction,
    read_position,
    read_units,
)


def test_read_position_holder(make_scaled_position_nwb):
    file = make_scaled_position_nwb(unit="centimeters", conversion=0.2, offset=1.0)
    series = file.processing["behavior"]["Position"].spatial_series["position"]
    pos = read_position(file)
    assert isinstance(pos, NWBPosition)
    assert pos.times.ndim == 1
    assert pos.positions.shape == series.data.shape
    np.testing.assert_array_equal(pos.positions, np.asarray(series.data) * 0.2 + 1.0)
    assert pos.units == "cm"
    with pytest.raises(TypeError):
        _, _ = read_position(file)
    with pytest.raises(FrozenInstanceError):
        pos.units = "m"


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ("meters", "m"),
        ("cm", "cm"),
        ("millimeters", "mm"),
        ("pixels", "px"),
        ("Body lengths", "Body lengths"),
        ("", None),
    ],
)
def test_position_units_eager_lazy_parity(
    make_scaled_position_nwb, declaration, expected
):
    file = make_scaled_position_nwb(unit=declaration, conversion=1.0, offset=0.0)
    eager = read_position(file)
    lazy = read_position(file, lazy=True)
    assert eager.units == lazy.units == expected
    np.testing.assert_array_equal(np.asarray(lazy.positions), eager.positions)
    np.testing.assert_array_equal(np.asarray(lazy.times), eager.times)


def test_missing_position_declaration_is_not_assumed(make_scaled_position_nwb):
    file = make_scaled_position_nwb(conversion=1.0, offset=0.0)
    file.processing["behavior"]["Position"].spatial_series["position"].fields[
        "unit"
    ] = None
    assert read_position(file).units is None
    assert read_position(file, lazy=True).units is None


def test_read_head_direction_holder(sample_nwb_with_head_direction):
    hd = read_head_direction(sample_nwb_with_head_direction)
    assert isinstance(hd, NWBHeadDirection)
    assert hd.times.ndim == hd.headings.ndim == 1
    assert hd.times.shape == hd.headings.shape
    with pytest.raises(TypeError):
        _, _ = read_head_direction(sample_nwb_with_head_direction)
    with pytest.raises(FrozenInstanceError):
        hd.headings = np.zeros(1)


def test_read_units_holder_without_coverage(empty_nwb):
    empty_nwb.add_unit(id=7, spike_times=[0.5, 0.1])
    units = read_units(empty_nwb)
    assert isinstance(units, NWBUnits)
    np.testing.assert_array_equal(units.spike_times[0], [0.1, 0.5])
    np.testing.assert_array_equal(units.unit_ids, [7])
    assert units.obs_intervals is None
    assert units.spike_window is None
    with pytest.raises(TypeError):
        _, _ = read_units(empty_nwb)
    with pytest.raises(FrozenInstanceError):
        units.unit_ids = np.array([11])
