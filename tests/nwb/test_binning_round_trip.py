"""NWB records retain geometric lookup, finite-volume faces and track projection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pynwb = pytest.importorskip("pynwb")
pytestmark = [pytest.mark.nwb, pytest.mark.integration]


@pytest.mark.parametrize("environment_round_trip", ["nwb"], indirect=True)
def test_nwb_read_back_boundary_bins(environment_round_trip):
    original, restore = environment_round_trip
    loaded = restore()
    np.testing.assert_array_equal(loaded.boundary_bins, original.boundary_bins)


@pytest.mark.parametrize("built_environment", ["track"], indirect=True)
@pytest.mark.parametrize("environment_round_trip", ["nwb"], indirect=True)
def test_nwb_read_back_track_to_linear(environment_round_trip):
    original, restore = environment_round_trip
    loaded = restore()
    points = np.random.default_rng(48).uniform(-2.0, 14.0, (500, 2))
    expected = original.to_linear(points)
    np.testing.assert_array_equal(loaded.to_linear(points), expected)
    np.testing.assert_array_equal(
        loaded.linear_to_nd(expected), original.linear_to_nd(expected)
    )


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_old_versions_refused(version):
    from neurospatial.io.nwb import read_environment

    path = (
        Path(__file__).parent
        / f"data/environment_schema_{version.replace('.', '_')}.nwb"
    )
    with pynwb.NWBHDF5IO(str(path), "r") as io, pytest.raises(ValueError) as caught:
        read_environment(io.read())
    text = str(caught.value)
    assert version in text
    assert "Why:" in text
    assert "Fix:" in text
    assert "save" in text.lower()


@pytest.mark.parametrize(
    "built_environment", ["grid8", "track", "mesh_boundary"], indirect=True
)
def test_nwb_geometry_transport_is_a_single_typed_ragged_row(
    built_environment, tmp_path
):
    from hdmf.common import VectorIndex

    from neurospatial.io import to_dict
    from neurospatial.io.nwb import read_environment, write_environment

    env = built_environment
    nwbfile = pynwb.NWBFile(
        session_description="Typed geometry payload",
        identifier="geometry-transport",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    write_environment(nwbfile, env)
    path = tmp_path / "geometry_transport.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        loaded_nwb = io.read()
        record = loaded_nwb.scratch["spatial_environment"]
        metadata = json.loads(record.description)
        expected = to_dict(env)
        assert len(record) == 1
        assert metadata["schema_version"] == "3.0"
        assert metadata["array_specs"].keys() == expected["array_specs"].keys()
        assert len(record.colnames) == len(metadata["array_specs"])
        for key, spec in metadata["array_specs"].items():
            name = spec["column"]
            assert name.startswith("array_")
            assert "/" not in name
            assert isinstance(record[name], VectorIndex)
            actual = np.asarray(record[name][0])
            assert actual.ndim == 1
            assert actual.dtype.str == spec["dtype"]
            actual = actual.reshape(spec["shape"])
            np.testing.assert_array_equal(
                actual,
                np.asarray(expected["arrays"][key]).reshape(spec["shape"]),
                err_msg=key,
            )
        restored = read_environment(loaded_nwb)
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
