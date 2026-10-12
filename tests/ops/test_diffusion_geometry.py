"""Provider boundary, numerical contracts, and geometry-cache ownership."""

from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pytest
from scipy import linalg, sparse

from neurospatial import Environment
from neurospatial.io import from_dict, from_file, to_dict, to_file
from neurospatial.layout._diffusion import DiffusionGeometry
from neurospatial.layout.engines.masked_grid import MaskedGridLayout
from neurospatial.layout.factories import _LAYOUT_MAP
from neurospatial.ops.basis import heat_kernel_wavelet_basis
from neurospatial.ops.calculus import divergence, gradient


class ConductanceLayout(MaskedGridLayout):
    """Supply conductances directly, with no finite-volume face capability."""

    builds = 0
    scale = 1.0

    def bin_sizes(self):
        return np.array([1.0, 2.0, 4.0])

    def diffusion_geometry(self):
        self.builds += 1
        return DiffusionGeometry.from_weights(
            sparse.csr_matrix([[0, 2, 0], [2, 0, 3], [0, 3, 0]]) * self.scale,
            self.bin_sizes(),
        )

    def finite_volume_geometry(self):
        raise NotImplementedError("This provider supplies conductances only.")


@pytest.fixture
def conductance_env(monkeypatch):
    monkeypatch.setitem(_LAYOUT_MAP, "Conductance", ConductanceLayout)
    return Environment.from_layout(
        "Conductance",
        {"active_mask": np.ones(3, dtype=bool), "grid_edges": (np.arange(4.0),)},
    )


def test_all_diffusion_consumers_share_provider_snapshot(conductance_env):
    """A new provider works without registering its name in any solver."""
    env = conductance_env
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
    assert env.layout.builds == 1
    with pytest.raises(NotImplementedError, match="conductances only"):
        env.get_differential_operator()


def test_clear_cache_rebuilds_geometry_and_both_eigenbases(conductance_env):
    env = conductance_env
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


def test_geometry_owns_its_inputs_and_ignores_zero_conductance_edges():
    weights = sparse.csr_matrix(([2.0, 2.0, 0.0], ([0, 1, 1], [1, 0, 2])), shape=(3, 3))
    volumes = np.array([1.0, 2.0, 4.0])
    geometry = DiffusionGeometry.from_weights(weights, volumes)
    weights.data[:] = 9.0
    volumes[:] = 9.0
    np.testing.assert_array_equal(
        geometry.W.toarray(), [[0, 2, 0], [2, 0, 0], [0, 0, 0]]
    )
    np.testing.assert_array_equal(geometry.volumes, [1, 2, 4])
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


@pytest.mark.parametrize(
    "weights, message",
    [
        (sparse.csr_matrix((2, 3)), "square"),
        (sparse.csr_matrix([[0, 1], [2, 0]]), "symmetric"),
        (sparse.csr_matrix([[0, -1], [-1, 0]]), "nonnegative"),
        (sparse.csr_matrix([[0, np.nan], [np.nan, 0]]), "finite"),
        (sparse.csr_matrix([[0, np.inf], [np.inf, 0]]), "finite"),
        (sparse.csr_matrix([[0, 1j], [1j, 0]]), "real"),
    ],
)
def test_invalid_conductances_fail_at_provider_boundary(weights, message):
    with pytest.raises(ValueError, match=message):
        DiffusionGeometry.from_weights(weights, np.ones(2))


@pytest.mark.parametrize("volumes", [[1], [1, 0], [1, -1], [1, np.nan], [1, np.inf]])
def test_invalid_cell_measures_fail_at_provider_boundary(volumes):
    with pytest.raises(ValueError, match="volumes must"):
        DiffusionGeometry.from_weights(sparse.csr_matrix([[0, 1], [1, 0]]), volumes)


def test_face_geometry_assembly_preserves_input_graph_and_bin_order():
    graph = nx.Graph()
    graph.add_nodes_from([2, 0, 1])
    graph.add_edge(0, 1, A=2.0, distance=4.0)
    graph.add_edge(1, 2, A=0.0, distance=1.0)
    before = list(graph.edges(data=True))
    geometry = DiffusionGeometry.from_graph(graph, np.array([1.0, 3.0, 5.0]))
    np.testing.assert_array_equal(
        geometry.W.toarray(), [[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]
    )
    np.testing.assert_array_equal(geometry.volumes, [1, 3, 5])
    assert list(graph.edges(data=True)) == before


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
