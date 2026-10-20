"""NWB position coordinates retain their unit and frame declarations."""

import numpy as np
import pytest

pynwb = pytest.importorskip("pynwb")

from neurospatial import Environment  # noqa: E402
from neurospatial.io.nwb import (  # noqa: E402
    NWBPosition,
    position_overlay_from_nwb,
    read_position,
)


def _series(file):
    return file.processing["behavior"]["Position"].spatial_series["position"]


def test_position_holder_original_constructor_remains_valid():
    holder = NWBPosition(np.arange(2.0), np.zeros((2, 2)), "cm")
    assert holder.frame is None


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        (None, None),
        ("", None),
        ("  ", None),
        (
            " Image corner (0, 0); x right, y down ",
            " Image corner (0, 0); x right, y down ",
        ),
    ],
)
def test_position_frame_is_preserved_without_interpretation(
    make_scaled_position_nwb, lazy, declaration, expected
):
    file = make_scaled_position_nwb(conversion=1.0, offset=0.0)
    series = _series(file)
    series.fields["reference_frame"] = declaration
    holder = read_position(file, lazy=lazy)
    assert holder.frame == expected
    np.testing.assert_array_equal(np.asarray(holder.positions), series.data)
    if lazy:
        assert holder.positions is series.data
        assert holder.times is series.timestamps


def test_file_backed_position_frame_does_not_materialize_lazy_data(
    make_scaled_position_nwb, tmp_path
):
    import h5py

    file = make_scaled_position_nwb(conversion=1.0, offset=0.0)
    path = tmp_path / "position_metadata.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(file)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        loaded = io.read()
        eager = read_position(loaded)
        lazy = read_position(loaded, lazy=True)
        assert eager.frame == lazy.frame == "Image corner (0, 0)"
        assert eager.units == lazy.units == "m"
        assert isinstance(lazy.positions, h5py.Dataset)
        assert lazy.positions is _series(loaded).data
        assert lazy.times is _series(loaded).timestamps
        np.testing.assert_array_equal(lazy.positions[:], eager.positions)


@pytest.mark.parametrize(
    "overrides",
    [{"units": "cm"}, {"units": None}, {"frame": "arena"}, {"frame": None}],
)
def test_position_metadata_cannot_be_relabelled_or_erased(
    make_scaled_position_nwb, overrides
):
    file = make_scaled_position_nwb()
    with pytest.raises(ValueError, match="conflicts with NWB position"):
        position_overlay_from_nwb(file, **overrides)


@pytest.mark.parametrize(
    ("stored_unit", "requested_unit", "expected"),
    [
        ("meters", "metres", "m"),
        ("cm", "centimetres", "cm"),
        ("centimetres", "cm", "cm"),
        ("pixels", "px", "px"),
    ],
)
def test_position_metadata_accepts_aliases_without_rescaling(
    make_scaled_position_nwb, stored_unit, requested_unit, expected
):
    file = make_scaled_position_nwb(unit=stored_unit, conversion=0.2, offset=1.0)
    overlay = position_overlay_from_nwb(
        file, units=requested_unit, frame="Image corner (0, 0)"
    )
    assert overlay.units == read_position(file).units
    assert overlay.frame == "Image corner (0, 0)"
    np.testing.assert_array_equal(
        overlay.positions, np.asarray(_series(file).data) * 0.2 + 1.0
    )
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool),
        (np.arange(4.0) * 50, np.arange(4.0) * 50),
        units=expected,
        frame=overlay.frame,
    )
    data = overlay.convert_to_data(overlay.times, len(overlay.times), env)
    np.testing.assert_array_equal(data.data, overlay.positions)


@pytest.mark.parametrize("undeclared", [None, "", "  "])
def test_missing_position_metadata_may_be_supplied(
    make_scaled_position_nwb, undeclared
):
    file = make_scaled_position_nwb(conversion=1.0, offset=0.0)
    series = _series(file)
    series.fields["unit"] = undeclared
    series.fields["reference_frame"] = undeclared
    overlay = position_overlay_from_nwb(file, units="centimeters", frame="arena")
    assert overlay.units == "centimeters"
    assert overlay.frame == "arena"
    np.testing.assert_array_equal(overlay.positions, series.data)


@pytest.mark.parametrize("metadata", [{"units": "cm"}, {"frame": "arena"}])
def test_position_coordinates_are_checked_against_environment(
    make_scaled_position_nwb, metadata
):
    file = make_scaled_position_nwb()
    overlay = position_overlay_from_nwb(file)
    declarations = {"units": "m", "frame": "Image corner (0, 0)", **metadata}
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool),
        (np.arange(4.0), np.arange(4.0)),
        **declarations,
    )
    with pytest.raises(
        ValueError, match=r"PositionOverlay.*does not match environment"
    ):
        overlay.convert_to_data(overlay.times, len(overlay.times), env)
