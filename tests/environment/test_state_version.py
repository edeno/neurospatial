"""Immutable geometry ownership of Environment."""

from __future__ import annotations

import numpy as np
import pytest
from environment_helpers import publish_environment

from neurospatial import Environment
from neurospatial.environment._cache import MRF_EIGENBASIS
from neurospatial.environment._fields import diffusion_eigenbasis
from neurospatial.ops.diffusion import _MRFEigenbasis


def _mrf_holder(env):
    """The environment's cached MRF eigenbasis holder, empty until first use."""
    return env._cache.get(MRF_EIGENBASIS, lambda: _MRFEigenbasis.empty(env.n_bins))


@pytest.fixture(params=["dict", "file"])
def restored_grid(request, tmp_path) -> Environment:
    import json

    grid = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
    )
    if request.param == "dict":
        return Environment.from_dict(json.loads(json.dumps(grid.to_dict())))
    path = tmp_path / "grid.npz"
    grid.to_file(path)
    return Environment.from_file(path)


@pytest.fixture
def env() -> Environment:
    rng = np.random.default_rng(0)
    return Environment.from_samples(rng.uniform(0, 100, (200, 2)), bin_size=5.0)


class TestPublishedValueOwnership:
    """Published values retain their state; restored values own fresh caches."""

    def test_restoration_publishes_saved_geometry_with_fresh_caches(self) -> None:
        """Reading a different saved graph does not change an existing value."""
        grid = (np.ones((3, 3), dtype=bool), (np.arange(4.0),) * 2)
        env = Environment.from_grid_mask(grid[0], grid_edges=grid[1])
        expected = Environment.from_grid_mask(
            grid[0], grid_edges=grid[1], connect_diagonal_neighbors=False
        )
        env.compute_kernel(bandwidth=1.0, mode="density")
        env.diffuse(np.ones(env.n_bins), bandwidth=1.0, mode="density")
        env.get_mrf_basis(np.ones(env.n_bins), rank=3)
        env.get_differential_operator()
        env.nearest_bin(env.bin_centers)
        old_geometry = env._diffusion_geometry
        old_diffusion_cache = diffusion_eigenbasis(env._cache)
        old_mrf_cache = _mrf_holder(env)
        geometry = env._geometry
        cache, n_cached = env._cache, len(env._cache)

        restored = Environment.from_dict(expected.to_dict())

        assert env._geometry is geometry
        assert env._cache is cache and len(cache) == n_cached
        assert env._diffusion_geometry is old_geometry
        assert diffusion_eigenbasis(env._cache) is old_diffusion_cache
        assert _mrf_holder(env) is old_mrf_cache
        assert len(restored._cache) == 0
        assert restored._diffusion_geometry is not old_geometry
        assert diffusion_eigenbasis(restored._cache) is not old_diffusion_cache
        assert _mrf_holder(restored) is not old_mrf_cache
        assert list(restored.connectivity.edges) == list(expected.connectivity.edges)
        np.testing.assert_array_equal(restored.bin_centers, expected.bin_centers)
        np.testing.assert_allclose(
            restored.get_differential_operator().toarray(),
            expected.get_differential_operator().toarray(),
        )

    def test_engine_graph_attributes_do_not_reach_environment(self) -> None:
        """The published graph is derived from the binning, not the engine's graph."""
        from neurospatial.layout.engines.masked_grid import MaskedGridLayout

        layout = MaskedGridLayout()
        layout.build(
            active_mask=np.ones((2, 2), dtype=bool),
            grid_edges=(np.arange(3.0),) * 2,
        )
        assert layout.connectivity is not None
        for _, _, data in layout.connectivity.edges(data=True):
            del data["edge_id"]
            data["stale"] = True

        env = publish_environment("derived", layout, "MaskedGrid", {})

        rows = {frozenset(map(int, pair)): i for i, pair in enumerate(env.edge_pairs)}
        for u, v, data in env.connectivity.edges(data=True):
            assert data["edge_id"] == rows[frozenset((u, v))]
            assert "stale" not in data


class TestPublishedGeometryIsReadOnly:
    """Published geometry refuses in-place edits that would invalidate analyses."""

    def test_arrays_are_read_only_and_graph_is_frozen(self) -> None:
        import networkx as nx

        env = Environment.from_grid_mask(
            np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
        )
        for env_or_copy in (env, env.with_metadata()):
            for array in (
                env_or_copy.bin_centers,
                env_or_copy.active_mask,
                *env_or_copy.grid_edges,
            ):
                assert not array.flags.writeable
            assert nx.is_frozen(env_or_copy.connectivity)
        with pytest.raises(ValueError, match="read-only"):
            env.bin_centers[0, 0] = 99.0
        with pytest.raises(nx.NetworkXError, match="Frozen"):
            env.connectivity.add_edge(0, 8)

    @pytest.mark.parametrize("target", ["node", "edge", "graph"])
    @pytest.mark.parametrize("method", ["assignment", "update", "delete"])
    def test_graph_attributes_reject_edits(self, target, method) -> None:
        env = Environment.from_grid_mask(
            np.ones(3, dtype=bool), grid_edges=(np.arange(4.0),)
        )
        env.compute_kernel(bandwidth=1.0, mode="density")
        if target == "node":
            attrs, key, value = env.connectivity.nodes[0], "pos", (99.0,)
        elif target == "edge":
            attrs, key, value = env.connectivity.edges[0, 1], "distance", 100.0
        else:
            attrs, key, value = env.connectivity.graph, "test", "changed"

        with pytest.raises(TypeError, match="read-only"):
            if method == "assignment":
                attrs[key] = value
            elif method == "update":
                attrs.update({key: value})
            else:
                attrs.pop(key, None)
        assert env.distance_between(points=[np.array([0.5]), np.array([2.5])]) == 2.0

    @pytest.mark.parametrize("how", ["deepcopy", "pickle"])
    def test_graph_attributes_stay_read_only_after_copy(self, how) -> None:
        import copy
        import pickle

        env = Environment.from_grid_mask(
            np.ones(3, dtype=bool), grid_edges=(np.arange(4.0),)
        )
        restored = (
            copy.deepcopy(env) if how == "deepcopy" else pickle.loads(pickle.dumps(env))
        )
        with pytest.raises(TypeError, match="read-only"):
            restored.connectivity.edges[0, 1]["distance"] = 100.0
        editable = restored.connectivity.copy()
        editable.edges[0, 1]["distance"] = 100.0
        assert restored.connectivity.edges[0, 1]["distance"] == 1.0

    @pytest.mark.parametrize("attribute", ["pos", "original_grid_nd_index", "vector"])
    def test_restored_graph_sequences_reject_edits(
        self, restored_grid, attribute
    ) -> None:
        env = restored_grid
        boundary = env.boundary_bins.copy()
        geometry = env._geometry
        attrs = (
            env.connectivity.edges[0, 1]
            if attribute == "vector"
            else env.connectivity.nodes[4]
        )
        with pytest.raises(TypeError):
            attrs[attribute][0] = 0

        assert env._geometry is geometry
        np.testing.assert_array_equal(env.boundary_bins, boundary)

    def test_restored_graph_copy_cannot_corrupt_transformed_distances(
        self, restored_grid
    ) -> None:
        from neurospatial.ops.transforms import from_rotation_matrix

        env = restored_grid
        editable = env.connectivity.copy()
        original = tuple(env.connectivity.nodes[0]["pos"])
        with pytest.raises(TypeError):
            editable.nodes[0]["pos"][0] = 99.0
        editable.nodes[0]["pos"] = [99.0, original[1]]
        assert tuple(env.connectivity.nodes[0]["pos"]) == original

        angle = 0.2
        transform = from_rotation_matrix(
            np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        )
        transformed = env.apply_transform(transform)
        for u, v, attrs in transformed.connectivity.edges(data=True):
            assert attrs["distance"] == pytest.approx(
                np.linalg.norm(transformed.bin_centers[v] - transformed.bin_centers[u])
            )

    @pytest.mark.parametrize("container", [list, np.array])
    def test_published_graph_ignores_edits_to_the_engine_graph(self, container) -> None:
        from neurospatial.layout.engines.masked_grid import MaskedGridLayout

        layout = MaskedGridLayout()
        layout.build(
            active_mask=np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
        )
        graph = layout.connectivity
        pos = container(graph.nodes[0]["pos"])
        index = container(graph.nodes[0]["original_grid_nd_index"])
        vector = container(graph.edges[0, 1]["vector"])
        graph.nodes[0]["pos"] = pos
        graph.nodes[0]["original_grid_nd_index"] = index
        graph.edges[0, 1]["vector"] = vector
        pos[0] = index[0] = vector[0] = 99
        graph.remove_edge(0, 3)
        published = publish_environment(
            name="", layout=layout, layout_type_used=layout.layout_type
        )

        assert published.connectivity.has_edge(0, 3)
        assert tuple(published.connectivity.nodes[0]["pos"]) == (0.5, 0.5)
        assert tuple(published.connectivity.nodes[0]["original_grid_nd_index"]) == (
            0,
            0,
        )
        assert tuple(published.connectivity.edges[0, 1]["vector"]) == (0.0, 1.0)

    def test_caller_arrays_stay_writable_and_independent(self) -> None:
        """The environment freezes its own copies, not the caller's arrays."""
        base = np.ones((3, 6), dtype=bool)
        edges = (np.arange(4.0), np.arange(4.0))
        env = Environment.from_grid_mask(base[:, :3], grid_edges=edges)
        point = np.array([[0.5, 0.5]])
        before = env.bin_at(point)

        assert base.flags.writeable and edges[0].flags.writeable
        base[0, 0] = False  # through the base of the view the caller passed
        edges[0][:] += 100.0

        np.testing.assert_array_equal(env.bin_at(point), before)
        assert env.active_mask[0, 0]
        np.testing.assert_array_equal(env.grid_edges[0], np.arange(4.0))

    @pytest.mark.parametrize(
        "name",
        [
            "bin_centers",
            "connectivity",
            "dimension_ranges",
            "grid_edges",
            "grid_shape",
            "active_mask",
        ],
    )
    def test_geometry_attributes_cannot_be_reassigned(self, name) -> None:
        env = Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
        )
        before = getattr(env, name)
        with pytest.raises(AttributeError, match=f"Environment.{name} is read-only"):
            setattr(env, name, None)
        after = getattr(env, name)
        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(after, before)
        elif name == "grid_edges":
            for edge, original in zip(after, before, strict=True):
                np.testing.assert_array_equal(edge, original)
        else:
            assert after == before
