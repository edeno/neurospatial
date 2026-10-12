"""Provider boundary, numerical contracts, and geometry-cache ownership."""

import warnings
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pytest
from scipy import linalg
from shapely.geometry import Polygon

from neurospatial import Environment
from neurospatial.environment.polar import EgocentricPolarEnvironment
from neurospatial.io import from_dict, from_file, to_dict, to_file
from neurospatial.layout._diffusion import (
    DiffusionGeometry,
    FiniteVolumeGeometryProvider,
)
from neurospatial.layout.engines.masked_grid import MaskedGridLayout
from neurospatial.layout.factories import _LAYOUT_MAP
from neurospatial.ops import smoothing
from neurospatial.ops.basis import heat_kernel_wavelet_basis
from neurospatial.ops.calculus import divergence, gradient
from neurospatial.ops.smoothing import compute_diffusion_kernels
from neurospatial.ops.transforms import Affine2D


class FaceLayout(MaskedGridLayout):
    """Hand-chosen cell faces for three 1-D bins, outside the built-in engines."""

    builds = 0
    scale = 1.0
    volumes = np.array([1.0, 2.0, 4.0])

    def bin_sizes(self):
        return self.volumes

    def finite_volume_geometry(self):
        self.builds += 1
        graph = self.connectivity.copy()
        for (u, v), area in zip(graph.edges(), (2.0, 3.0), strict=True):
            graph.edges[u, v]["A"] = area * self.scale
        return graph, self.volumes


@pytest.fixture
def face_env(monkeypatch):
    monkeypatch.setitem(_LAYOUT_MAP, "Faces", FaceLayout)
    return Environment.from_layout(
        "Faces",
        {"active_mask": np.ones(3, dtype=bool), "grid_edges": (np.arange(4.0),)},
    )


def test_all_diffusion_consumers_share_provider_snapshot(face_env):
    """A new layout works without registering its name in any solver."""
    env = face_env
    mass = np.diag([1.0, 2.0, 4.0])
    stiffness = np.array([[2, -2, 0], [-2, 5, -3], [0, -3, 3]])
    generator = np.linalg.solve(mass, stiffness)
    heat = linalg.expm(-0.5 * generator)
    field = np.array([1.0, 3.0, 0.0])

    for mode, expected in (
        ("average", heat),
        ("transition", heat.T),
        ("density", heat @ np.linalg.inv(mass)),
    ):
        np.testing.assert_allclose(env.compute_kernel(1.0, mode=mode), expected)
        np.testing.assert_allclose(
            env.diffuse(field, 1.0, mode=mode), expected @ field, atol=1e-12
        )

    wavelets = heat_kernel_wavelet_basis(
        env, centers=np.array([0]), scales=[0.5], normalize="none"
    )
    np.testing.assert_allclose(wavelets[0], heat[:, 0], atol=1e-12)

    basis = env._mrf_basis(np.ones(3), rank=3)
    np.testing.assert_allclose(basis.B.T @ mass @ basis.B, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        basis.B.T @ stiffness @ basis.B, np.diag(basis.d), atol=1e-12
    )
    # The same faces drive the differential operators: div(grad f) = -M^-1 L f.
    np.testing.assert_allclose(
        divergence(env, gradient(env, field)), -generator @ field, atol=1e-12
    )
    assert env.layout.builds == 1  # one face build serves every consumer


def test_clear_cache_rebuilds_geometry_and_both_eigenbases(face_env):
    env = face_env
    before = env.compute_kernel(1.0, mode="average")
    env.diffuse(np.ones(3), 1.0)
    env._mrf_basis(np.ones(3), rank=3)
    old_geometry = env._diffusion_geometry
    old_diffusion_cache = env._diffusion_eigenbasis
    old_mrf_cache = env._mrf_eigenbasis

    env.layout.scale = 2.0
    env.clear_cache()
    after = env.compute_kernel(1.0, mode="average")
    np.testing.assert_allclose(after, before @ before, atol=1e-12)
    np.testing.assert_allclose(
        env.diffuse(np.arange(3.0), 1.0, mode="average"), after @ np.arange(3.0)
    )
    assert env._diffusion_geometry is not old_geometry
    assert env._diffusion_eigenbasis is not old_diffusion_cache
    assert env._mrf_eigenbasis is not old_mrf_cache
    assert env.layout.builds == 2


def test_existing_layout_subclass_inherits_geometry_without_registration(monkeypatch):
    class RenamedGrid(MaskedGridLayout):
        pass

    monkeypatch.setitem(_LAYOUT_MAP, "Renamed", RenamedGrid)
    params = {
        "active_mask": np.ones((2, 3), dtype=bool),
        "grid_edges": (np.arange(3.0), np.arange(4.0)),
    }
    original = Environment.from_layout("MaskedGrid", params)
    subclass = Environment.from_layout("Renamed", params)
    np.testing.assert_array_equal(
        subclass.compute_kernel(1.0), original.compute_kernel(1.0)
    )


def test_geometry_owns_its_inputs_and_ignores_zero_area_faces():
    graph = nx.Graph()
    graph.add_nodes_from([2, 0, 1])
    graph.add_edge(0, 1, A=2.0, distance=4.0)
    graph.add_edge(1, 2, A=0.0, distance=1.0)
    before = list(graph.edges(data=True))
    volumes = np.array([1.0, 3.0, 5.0])
    geometry = DiffusionGeometry.from_graph(graph, volumes)
    assert list(graph.edges(data=True)) == before
    graph.edges[0, 1]["A"] = 9.0
    volumes[:] = 9.0
    np.testing.assert_array_equal(
        geometry.W.toarray(), [[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]
    )
    np.testing.assert_array_equal(geometry.volumes, [1, 3, 5])
    assert geometry.n_components == 2
    assert geometry.labels[0] == geometry.labels[1] != geometry.labels[2]
    for buffer in (
        geometry.W.data,
        geometry.W.indices,
        geometry.W.indptr,
        geometry.volumes,
        geometry.labels,
    ):
        assert not buffer.flags.writeable


@pytest.mark.parametrize("volumes", [[1], [1, 0], [1, -1], [1, np.nan], [1, np.inf]])
def test_invalid_cell_measures_fail_at_provider_boundary(volumes):
    with pytest.raises(ValueError, match="volumes must"):
        DiffusionGeometry.from_graph(nx.path_graph(2), volumes)


def test_anisotropic_3d_grid_face_weights_and_cell_measures():
    env = Environment.from_grid_mask(
        np.ones((2, 2, 2), dtype=bool),
        grid_edges=(np.arange(3.0), np.arange(3.0) * 2, np.arange(3.0) * 4),
        connect_diagonal_neighbors=True,
    )
    geometry = env._diffusion_geometry
    np.testing.assert_array_equal(geometry.volumes, np.full(8, 8.0))
    for u, v in env.connectivity.edges:
        offset = np.abs(env.bin_centers[u] - env.bin_centers[v])
        expected = 8.0 / np.sum(offset) ** 2 if np.count_nonzero(offset) == 1 else 0.0
        assert geometry.W[u, v] == expected
        assert "A" not in env.connectivity.edges[u, v]


# ---------------------------------------------------------------------------
# Reconstructed environments keep the saved graph on their layout.
# ---------------------------------------------------------------------------
def _round_trip(env, how, tmp_path):
    if how == "dict":
        return from_dict(to_dict(env))
    if how == "file":
        to_file(env, tmp_path / "env")
        return from_file(tmp_path / "env")
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    nwbfile = pynwb.NWBFile(
        session_description="round trip",
        identifier="round-trip",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    write_environment(nwbfile, env)
    return read_environment(nwbfile)


def _four_connected_grid():
    positions = np.random.default_rng(0).uniform(0, 30, (4000, 2))
    return Environment.from_samples(
        positions, bin_size=3.0, connect_diagonal_neighbors=False
    )


def _polar():
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 6,
    )


@pytest.mark.parametrize("how", ["dict", "file", "nwb"])
@pytest.mark.parametrize(
    "make_env", [_four_connected_grid, _polar], ids=["grid4", "polar"]
)
def test_round_trips_and_their_copies_use_the_saved_graph(make_env, how, tmp_path):
    """Read-back environments and their copies keep 4-connectivity and polar seams.

    The rebuilt layout would otherwise supply an 8-connected grid graph (NWB) or
    a Cartesian, seamless polar graph, which copy() and the geometry read.
    """
    original = make_env()
    field = np.sin(np.arange(original.n_bins) * 0.37)
    expected_divergence = divergence(original, gradient(original, field))
    restored = _round_trip(original, how, tmp_path)
    for env in (restored, restored.copy()):
        assert list(env.connectivity.edges()) == list(original.connectivity.edges())
        np.testing.assert_allclose(gradient(env, field), gradient(original, field))
        np.testing.assert_allclose(
            divergence(env, gradient(env, field)), expected_divergence
        )
        np.testing.assert_allclose(
            env.compute_kernel(3.0), original.compute_kernel(3.0), atol=1e-12
        )


def test_finite_volume_geometry_refuses_a_layout_graph_out_of_sync():
    """A layout graph that diverged from env.connectivity fails loudly."""
    edges = (np.arange(4.0), np.arange(4.0))
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=edges, connect_diagonal_neighbors=False
    )
    env.layout.connectivity = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=edges, connect_diagonal_neighbors=True
    ).connectivity
    with pytest.raises(ValueError, match=r"out of sync with env\.connectivity"):
        gradient(env, np.ones(env.n_bins))


# ---------------------------------------------------------------------------
# Polar geometry belongs to the polar layout.
# ---------------------------------------------------------------------------
def test_polar_layout_supplies_annular_sector_geometry():
    """Asking the layout directly gives sector areas, not Cartesian dr*dtheta."""
    env = _polar()
    assert isinstance(env.layout, MaskedGridLayout)
    distance_edges, angle_edges = env.grid_edges
    sectors = (0.5 * np.diff(distance_edges**2)[:, None] * np.diff(angle_edges)).ravel()
    np.testing.assert_allclose(env.layout.bin_sizes(), sectors)
    np.testing.assert_allclose(env.bin_sizes, sectors)
    graph, volumes = env.layout.finite_volume_geometry()
    np.testing.assert_allclose(volumes, sectors)
    geometry = DiffusionGeometry.from_graph(graph, volumes)
    assert (geometry.W != env._diffusion_geometry.W).nnz == 0


def test_polar_from_layout_requires_the_masked_grid():
    with pytest.raises(ValueError, match=r"'MaskedGrid'.*from_polar_egocentric"):
        EgocentricPolarEnvironment.from_layout(
            "RegularGrid", {"positions": np.zeros((4, 2)), "bin_size": 1.0}
        )


# ---------------------------------------------------------------------------
# Capability protocols and actionable errors.
# ---------------------------------------------------------------------------
def test_capability_protocol_reports_what_layouts_can_supply(reloaded_graph_env):
    for layout in (_four_connected_grid().layout, _polar().layout):
        assert isinstance(layout, FiniteVolumeGeometryProvider)
    assert not isinstance(reloaded_graph_env.layout, FiniteVolumeGeometryProvider)


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (lambda env: env.smooth(np.ones(env.n_bins), 3.0), "smoothing"),
        (lambda env: env.compute_kernel(3.0), "smoothing"),
        (lambda env: env.diffuse(np.ones(env.n_bins), 3.0), "smoothing"),
        (lambda env: gradient(env, np.ones(env.n_bins)), "gradient/divergence"),
        (
            lambda env: divergence(env, np.ones(env.connectivity.number_of_edges())),
            "gradient/divergence",
        ),
        (
            lambda env: heat_kernel_wavelet_basis(env, centers=np.array([0])),
            "heat_kernel_wavelet_basis",
        ),
    ],
)
def test_unsupported_layout_errors_name_the_operation_and_a_factory(call, operation):
    env = _four_connected_grid().apply_transform(Affine2D(np.eye(3)))
    with pytest.raises(NotImplementedError) as exc:
        call(env)
    message = str(exc.value)
    assert message.startswith(f"{operation} needs ")
    assert "'TransformedLayout'" in message
    assert "\nWhy:" in message
    assert "\nFix:" in message
    assert "Environment.from_samples" in message
    assert "implement" not in message


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph])
def test_compute_diffusion_kernels_rejects_graphs_that_sum_edges(graph_type):
    """Reciprocal or parallel edges would double a conductance; say how to fix."""
    graph = nx.path_graph(4, create_using=graph_type)
    nx.set_edge_attributes(graph, 1.0, "A")
    nx.set_edge_attributes(graph, 1.0, "distance")
    with pytest.raises(ValueError, match=r"undirected simple nx\.Graph.*nx\.Graph\("):
        compute_diffusion_kernels(graph, volumes=np.ones(4), sigma=1.0, mode="average")
    kernel = compute_diffusion_kernels(
        nx.Graph(graph), volumes=np.ones(4), sigma=1.0, mode="average"
    )
    np.testing.assert_allclose(kernel.sum(axis=1), 1.0)


def test_dense_kernel_warning_points_at_the_callers_line(monkeypatch, face_env):
    monkeypatch.setattr(smoothing, "_LARGE_KERNEL_THRESHOLD", 1)
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1.0, "A")
    nx.set_edge_attributes(graph, 1.0, "distance")
    with pytest.warns(UserWarning, match="dense diffusion kernel") as record:
        compute_diffusion_kernels(graph, volumes=np.ones(3), sigma=1.0, mode="average")
        face_env.compute_kernel(1.0, cache=False)
    dense = [w for w in record if "dense diffusion kernel" in str(w.message)]
    assert [w.filename for w in dense] == [__file__, __file__]


def test_mesh_skew_warning_is_raised_once_per_geometry_at_the_callers_line():
    shear = 6.0
    corners = [(x + shear * y, y) for x, y in [(-30, -8), (30, -8), (30, 8), (-30, 8)]]
    env = Environment.from_layout(
        "TriangularMesh",
        {"boundary_polygon": Polygon(corners), "point_spacing": 4.0},
    )
    with pytest.warns(UserWarning, match="non-K-orthogonal") as record:
        env.compute_kernel(3.0)
    skew = [w for w in record if "non-K-orthogonal" in str(w.message)]
    assert [w.filename for w in skew] == [__file__]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env.compute_kernel(5.0, cache=False)  # the cached geometry is reused
    env.clear_cache()
    with pytest.warns(UserWarning, match="non-K-orthogonal"):
        env.smooth(np.ones(env.n_bins), 3.0)


# ---------------------------------------------------------------------------
# Invalid layout geometry fails at the environment boundary.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    [
        ("volumes", np.ones(4), r"volumes must have shape \(3,\)"),
        ("volumes", np.array([1.0, -2.0, 4.0]), "strictly positive"),
        ("scale", -1.0, "invalid face measure"),
    ],
)
def test_invalid_layout_faces_fail_at_the_environment(
    face_env, monkeypatch, attribute, value, match
):
    monkeypatch.setattr(FaceLayout, attribute, value)
    with pytest.raises(ValueError, match=match):
        face_env.compute_kernel(1.0)
