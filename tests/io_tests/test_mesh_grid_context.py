"""Mesh storage retains the full triangulation's native simplex indexing."""

from datetime import datetime, timezone

import numpy as np
import pytest
from shapely.geometry import Polygon

from neurospatial import Environment


def _mesh():
    boundary = Polygon(
        [(0, 0), (12, 0), (12, 12), (8, 12), (8, 8), (4, 8), (4, 12), (0, 12)],
        [[(4, 3), (6, 3), (6, 5), (4, 5)]],
    )
    return Environment.from_layout(
        "TriangularMesh", {"boundary_polygon": boundary, "point_spacing": 1.5}
    )


@pytest.mark.parametrize("storage", ["dict", "file", "nwb"])
def test_mesh_round_trip_preserves_full_simplex_grid_context(tmp_path, storage):
    env = _mesh()
    assert env.grid_shape[0] > env.n_bins
    assert not env.active_mask.all()
    if storage == "dict":
        restored = Environment.from_dict(env.to_dict())
    elif storage == "file":
        env.to_file(tmp_path / "mesh")
        restored = Environment.from_file(tmp_path / "mesh")
    else:
        pynwb = pytest.importorskip("pynwb")
        from neurospatial.io.nwb import read_environment, write_environment

        nwbfile = pynwb.NWBFile(
            "mesh", "mesh", datetime(2020, 1, 1, tzinfo=timezone.utc)
        )
        write_environment(nwbfile, env)
        with pynwb.NWBHDF5IO(str(tmp_path / "mesh.nwb"), "w") as io:
            io.write(nwbfile)
        with pynwb.NWBHDF5IO(str(tmp_path / "mesh.nwb"), "r") as io:
            restored = read_environment(io.read())

    for candidate in (restored, restored.copy()):
        assert candidate.grid_shape == env.grid_shape
        np.testing.assert_array_equal(candidate.active_mask, env.active_mask)
        np.testing.assert_array_equal(
            candidate.bin_at(env.bin_centers), env.bin_at(env.bin_centers)
        )
        assert not candidate.active_mask.flags.writeable


@pytest.mark.parametrize("field", ["grid_shape", "active_mask"])
def test_mesh_loading_refuses_inconsistent_duplicate_grid_context(field):
    env = _mesh()
    record = env.to_dict()
    if field == "grid_shape":
        record["grid_shape"] = [record["grid_shape"][0] - 1]
    else:
        record["arrays"]["active_mask"][0] = not record["arrays"]["active_mask"][0]
    with pytest.raises(ValueError, match=field):
        Environment.from_dict(record)
