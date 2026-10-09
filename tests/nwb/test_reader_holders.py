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


def test_read_units_spike_window(empty_nwb, tmp_path):
    from neurospatial import Environment, compute_spatial_rates

    empty_nwb.add_unit(
        id=7,
        spike_times=[20.0, 60.0, 1120.0],
        obs_intervals=[[0.0, 100.0], [1100.0, 1200.0]],
    )
    empty_nwb.add_unit(
        id=11,
        spike_times=[25.0, 65.0, 1125.0],
        obs_intervals=[[10.0, 1150.0]],
    )
    path = tmp_path / "unit_coverage.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(empty_nwb)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        file = io.read()
        units = read_units(file)
        reordered = read_units(file, unit_ids=[11, 7], lazy=True)
        np.testing.assert_array_equal(reordered.spike_window, units.spike_window)
        np.testing.assert_array_equal(reordered.obs_intervals[0], [[10.0, 1150.0]])
        np.testing.assert_array_equal(
            np.asarray(reordered.spike_times[0]), [25, 65, 1125]
        )
        selected = read_units(file, unit_ids=[7])
    np.testing.assert_array_equal(units.unit_ids, [7, 11])
    np.testing.assert_array_equal(units.obs_intervals[0], [[0, 100], [1100, 1200]])
    np.testing.assert_array_equal(units.spike_window, [[10, 100], [1100, 1150]])
    np.testing.assert_array_equal(selected.spike_window, [[0, 100], [1100, 1200]])
    times = np.arange(0, 1200.1, 0.1)
    positions = np.sin(times)[:, None]
    env = Environment.from_samples(positions, bin_size=0.2)
    rates = compute_spatial_rates(
        env,
        units.spike_times,
        times,
        positions,
        unit_ids=units.unit_ids,
        spike_window=units.spike_window,
        method="binned",
    )
    assert rates.spike_window_assumed is False
    np.testing.assert_array_equal(rates.unit_ids, [7, 11])
    assert rates.occupancy.sum() == pytest.approx(140.0)


def test_unit_coverage_intersection_folds_all_selected_units(empty_nwb):
    empty_nwb.add_unit(id=7, spike_times=[1.0], obs_intervals=[[0.0, 10.0]])
    empty_nwb.add_unit(id=11, spike_times=[3.0], obs_intervals=[[2.0, 8.0]])
    empty_nwb.add_unit(id=19, spike_times=[5.0], obs_intervals=[[4.0, 6.0]])
    np.testing.assert_array_equal(read_units(empty_nwb).spike_window, [[4.0, 6.0]])
    assert read_units(empty_nwb, unit_ids=[]).spike_window.shape == (0, 2)


def test_unit_without_observation_intervals_reads(empty_nwb):
    """NWB allows a unit with no obs_intervals rows; it was never observed."""
    empty_nwb.add_unit(id=7, spike_times=[1.0], obs_intervals=[[0.0, 10.0]])
    empty_nwb.add_unit(id=11, spike_times=[], obs_intervals=np.empty((0, 2)))
    units = read_units(empty_nwb)
    assert units.obs_intervals[1].shape == (0, 2)
    assert units.spike_window.shape == (0, 2)
    np.testing.assert_array_equal(
        read_units(empty_nwb, unit_ids=[7]).spike_window, [[0.0, 10.0]]
    )
    assert read_units(empty_nwb, unit_ids=[11]).spike_window.shape == (0, 2)


def test_disjoint_unit_coverage_is_explicitly_empty(empty_nwb):
    from neurospatial import Environment, compute_spatial_rates

    empty_nwb.add_unit(id=7, spike_times=[0.5], obs_intervals=[[0.0, 1.0]])
    empty_nwb.add_unit(id=11, spike_times=[2.5], obs_intervals=[[2.0, 3.0]])
    units = read_units(empty_nwb)
    assert units.spike_window.shape == (0, 2)
    times = np.arange(0, 3, 0.1)
    positions = times[:, None]
    env = Environment.from_samples(positions, bin_size=0.5)
    with pytest.raises(ValueError, match=r"spike_window.*no rows"):
        compute_spatial_rates(
            env, units.spike_times, times, positions, spike_window=units.spike_window
        )
