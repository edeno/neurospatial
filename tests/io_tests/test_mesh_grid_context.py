"""Mesh storage retains the full triangulation's native simplex indexing."""

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import shapely
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

import neurospatial.layout.engines.triangular_mesh as mesh_engine
from neurospatial import Environment


def _mesh():
    boundary = Polygon(
        [(0, 0), (12, 0), (12, 12), (8, 12), (8, 8), (4, 8), (4, 12), (0, 12)],
        [[(4, 3), (6, 3), (6, 5), (4, 5)]],
    )
    return Environment.from_layout(
        "TriangularMesh", {"boundary_polygon": boundary, "point_spacing": 1.5}
    )


def _round_trip(env, storage, tmp_path):
    if storage == "dict":
        return Environment.from_dict(env.to_dict())
    if storage == "file":
        env.to_file(tmp_path / "mesh")
        return Environment.from_file(tmp_path / "mesh")
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    nwbfile = pynwb.NWBFile("mesh", "mesh", datetime(2020, 1, 1, tzinfo=timezone.utc))
    write_environment(nwbfile, env)
    with pynwb.NWBHDF5IO(str(tmp_path / "mesh.nwb"), "w") as io:
        io.write(nwbfile)
    with pynwb.NWBHDF5IO(str(tmp_path / "mesh.nwb"), "r") as io:
        return read_environment(io.read())


@pytest.mark.parametrize("storage", ["dict", "file", "nwb"])
def test_mesh_round_trip_preserves_full_simplex_grid_context(tmp_path, storage):
    env = _mesh()
    assert env.grid_shape[0] > env.n_bins
    assert not env.active_mask.all()
    restored = _round_trip(env, storage, tmp_path)

    for candidate in (restored, restored.with_metadata()):
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


def _triangle_set(simplices):
    return {tuple(sorted(triangle)) for triangle in np.asarray(simplices).tolist()}


def _other_platform_triangulation(sample_points):
    """Triangulate cocircular lattice cells with the other diagonal choice.

    Qhull breaks Delaunay ties between a cell's two diagonals differently across
    platforms. A tiny shear forces one choice; an affine map keeps the triangles a
    valid triangulation of the unsheared points.
    """
    points = np.asarray(sample_points, dtype=np.float64)
    local = _triangle_set(Delaunay(points).simplices)
    for shear in (1e-6, -1e-6):
        sheared = points.copy()
        sheared[:, 0] += shear * sheared[:, 1]
        triangulation = Delaunay(sheared)
        if _triangle_set(triangulation.simplices) != local:
            return SimpleNamespace(
                points=points,
                simplices=triangulation.simplices,
                neighbors=triangulation.neighbors,
            )
    raise AssertionError("both diagonal choices match this platform's Delaunay")


@pytest.mark.parametrize("storage", ["dict", "file", "nwb"])
def test_mesh_saved_with_another_platforms_triangulation_loads(
    tmp_path, storage, monkeypatch
):
    """Files keep their stored triangles even where this platform's Qhull differs."""
    with monkeypatch.context() as patch:
        patch.setattr(mesh_engine, "_triangulate_points", _other_platform_triangulation)
        env = _mesh()
    stored = env._binning.locator
    assert _triangle_set(stored.simplices) != _triangle_set(
        Delaunay(stored.points).simplices
    )

    restored = _round_trip(env, storage, tmp_path)

    locator = restored._binning.locator
    np.testing.assert_array_equal(locator.simplices, stored.simplices)
    np.testing.assert_array_equal(locator.simplex_to_active, stored.simplex_to_active)
    np.testing.assert_array_equal(
        restored.bin_at(env.bin_centers), np.arange(env.n_bins)
    )
    # Every located point lies in its bin's stored triangle; unlocated points
    # inside the boundary lie in no stored bin triangle.
    active = np.flatnonzero(stored.simplex_to_active >= 0)
    active = active[np.argsort(stored.simplex_to_active[active])]
    bin_triangles = shapely.polygons(stored.points[stored.simplices[active]])
    points = np.random.default_rng(0).uniform(-1.0, 13.0, size=(4000, 2))
    bins = restored.bin_at(points)
    located = bins >= 0
    assert located.sum() > 1000
    assert np.all(
        shapely.distance(bin_triangles[bins[located]], shapely.points(points[located]))
        <= 1e-12
    )
    unlocated = points[~located]
    inside = shapely.contains_xy(stored.boundary, unlocated[:, 0], unlocated[:, 1])
    assert not np.any(
        shapely.contains_xy(shapely.union_all(bin_triangles), *unlocated[inside].T)
    )


def test_mesh_loading_refuses_overlapping_triangles():
    env = _mesh()
    record = env.to_dict()
    simplices = record["arrays"]["binning/locator/simplices"]
    simplex_to_active = record["arrays"]["binning/locator/simplex_to_active"]
    # An inactive triangle that duplicates a bin's triangle claims its points.
    simplices[simplex_to_active.index(-1)] = list(simplices[simplex_to_active.index(0)])
    with pytest.raises(ValueError, match="overlap"):
        Environment.from_dict(record)
