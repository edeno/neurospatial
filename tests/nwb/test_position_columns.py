"""Selecting the spatial columns of a Position series that stores more.

Some files keep x, y and speed in one SpatialSeries under a compound unit
(DANDI 000447 declares ``'centimeters; centimeters/second'``). Binning every
column would treat speed as a third spatial axis, so the readers refuse such
a series until the spatial columns are named with ``columns=``.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

pynwb = pytest.importorskip("pynwb")

_N = 200
_COMMENTS = "Video frames taken at 30fps - row1:xpos row2:ypos row3:velocity"


def _xy_speed():
    """(n, 3) samples: x and y in cm on a 40 x 20 cm box, then speed in cm/s."""
    rng = np.random.default_rng(0)
    xy = np.column_stack([rng.uniform(0, 40, _N), rng.uniform(0, 20, _N)])
    speed = rng.uniform(0, 50, _N)
    return np.column_stack([xy, speed])


def _file(data, unit, *, comments=_COMMENTS):
    from pynwb import NWBFile
    from pynwb.behavior import Position, SpatialSeries

    nwbfile = NWBFile(
        session_description="Position series with a speed column",
        identifier=str(uuid4()),
        session_start_time=datetime.now().astimezone(),
    )
    position = Position(name="Position")
    position.add_spatial_series(
        SpatialSeries(
            name="SpatialSeries",
            data=data,
            timestamps=np.arange(len(data)) / 30.0,
            reference_frame="center well",
            unit=unit,
            comments=comments,
        )
    )
    nwbfile.create_processing_module("behavior", "Tracking").add(position)
    return nwbfile


@pytest.fixture
def xy_speed_file():
    """A series mimicking DANDI 000447: x, y, speed under a compound unit."""
    return _file(_xy_speed(), "centimeters; centimeters/second")


def test_compound_unit_with_extra_column_is_refused(xy_speed_file):
    """Without columns=, the speed column is not silently binned as space."""
    from neurospatial.io.nwb import environment_from_position, read_position

    for read in (
        lambda: read_position(xy_speed_file),
        lambda: read_position(xy_speed_file, lazy=True),
        lambda: environment_from_position(xy_speed_file, bin_size=5.0, units="cm"),
    ):
        with pytest.raises(ValueError) as info:
            read()
        message = str(info.value)
        assert (
            "has 3 columns and declares unit 'centimeters; centimeters/second'"
            in message
        )
        assert "pass columns= with the spatial columns" in message
        assert "row1:xpos row2:ypos row3:velocity" in message


def test_selected_columns_are_the_spatial_samples(xy_speed_file):
    """columns=[0, 1] returns x and y; a compound unit is not assigned to them."""
    from neurospatial.io.nwb import read_position

    pos = read_position(xy_speed_file, columns=[0, 1])
    np.testing.assert_array_equal(pos.positions, _xy_speed()[:, :2])
    assert pos.units is None
    assert pos.frame == "center well"
    assert pos.times.shape == (_N,)


def test_environment_from_selected_columns_is_planar(xy_speed_file):
    """The environment spans the x-y box, not x-y-speed."""
    from neurospatial.io.nwb import environment_from_position

    env = environment_from_position(
        xy_speed_file, bin_size=5.0, columns=[0, 1], units="cm"
    )
    assert env.n_dims == 2
    assert env.units == "cm"
    assert env.frame == "center well"
    xy = _xy_speed()[:, :2]
    np.testing.assert_array_equal(env.contains(xy), np.ones(_N, dtype=bool))
    lo, hi = env.bin_centers.min(axis=0), env.bin_centers.max(axis=0)
    assert np.all(lo >= xy.min(axis=0) - 5.0) and np.all(hi <= xy.max(axis=0) + 5.0)


def test_environment_from_selected_columns_needs_units(xy_speed_file):
    """The selected columns have no single declared unit, so units= is asked."""
    from neurospatial.io.nwb import environment_from_position

    with pytest.raises(ValueError, match=r"states no single unit for columns=\[0, 1\]"):
        environment_from_position(xy_speed_file, bin_size=5.0, columns=[0, 1])


def test_recognized_unit_keeps_all_columns_and_units():
    """A single length unit with three columns is 3-D tracking, read as is."""
    from neurospatial.io.nwb import read_position

    data = _xy_speed()
    nwbfile = _file(data, "meters", comments="no comments")
    np.testing.assert_array_equal(read_position(nwbfile).positions, data)
    pos = read_position(nwbfile, columns=[1, 0])
    np.testing.assert_array_equal(pos.positions, data[:, [1, 0]])
    assert pos.units == "m"


def test_two_columns_with_compound_unit_are_refused():
    """A unit listing several units is ambiguous even with two columns."""
    from neurospatial.io.nwb import read_position

    nwbfile = _file(_xy_speed()[:, [0, 2]], "centimeters; centimeters/second")
    with pytest.raises(ValueError, match="has 2 columns and declares unit"):
        read_position(nwbfile)
    assert read_position(nwbfile, columns=[0]).positions.shape == (_N, 1)


def test_lazy_selected_columns_stay_on_disk(tmp_path, xy_speed_file):
    """Lazy column views read only when sliced and match the eager values."""
    from pynwb import NWBHDF5IO

    from neurospatial.io.nwb import read_position

    path = tmp_path / "xy_speed.nwb"
    with NWBHDF5IO(path, "w") as io:
        io.write(xy_speed_file)
    expected = _xy_speed()[:, [0, 1]]
    with NWBHDF5IO(path, "r") as io:
        pos = read_position(io.read(), lazy=True, columns=[0, 1])
        assert not isinstance(pos.positions, np.ndarray)
        assert pos.positions.shape == (_N, 2)
        assert len(pos.positions) == _N
        np.testing.assert_array_equal(pos.positions[:3], expected[:3])
        np.testing.assert_array_equal(pos.positions[5, 1], expected[5, 1])
        np.testing.assert_array_equal(pos.positions[10:20, 0], expected[10:20, 0])
        np.testing.assert_array_equal(np.asarray(pos.positions), expected)


@pytest.mark.parametrize(
    ("columns", "fragment"),
    [
        ([], "at least one column"),
        ([0, 0], "repeats column 0"),
        ([0, 3], "column 3 is out of range"),
        ([-1, 0], "column -1 is out of range"),
        ([0.0, 1.0], "integer column indices"),
        ([True, False], "integer column indices"),
    ],
)
def test_invalid_columns_are_refused(xy_speed_file, columns, fragment):
    """columns= names distinct existing column indices."""
    from neurospatial.io.nwb import read_position

    with pytest.raises(ValueError, match=fragment):
        read_position(xy_speed_file, columns=columns)


def test_columns_need_a_two_dimensional_series():
    """A 1-D series has no columns to select."""
    from neurospatial.io.nwb import read_position

    nwbfile = _file(np.linspace(0.0, 1.0, _N), "meters")
    with pytest.raises(ValueError, match="stores one value per sample"):
        read_position(nwbfile, columns=[0])
