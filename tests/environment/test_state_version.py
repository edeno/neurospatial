"""Tests for Environment._state_version and versioned_cached_property."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.environment.decorators import versioned_cached_property


@pytest.fixture
def env() -> Environment:
    rng = np.random.default_rng(0)
    return Environment.from_samples(rng.uniform(0, 100, (200, 2)), bin_size=5.0)


class TestStateVersionField:
    """Environment._state_version must increment on documented mutation paths."""

    def test_restored_layout_graph_refreshes_views_and_all_caches(self) -> None:
        """Restoring a saved graph onto the layout, as the readers do, refreshes
        the environment's views and every cache."""
        from scipy.spatial import cKDTree

        grid = (np.ones((3, 3), dtype=bool), (np.arange(4.0),) * 2)
        env = Environment.from_grid_mask(grid[0], grid_edges=grid[1])
        expected = Environment.from_grid_mask(
            grid[0], grid_edges=grid[1], connect_diagonal_neighbors=False
        )
        env.compute_kernel(1.0)
        env.diffuse(np.ones(env.n_bins), 1.0)
        env._mrf_basis(np.ones(env.n_bins), rank=3)
        env.get_differential_operator()
        env._kdtree_cache = cKDTree(env.bin_centers)
        old_geometry = env._diffusion_geometry
        old_diffusion_cache = env._diffusion_eigenbasis
        old_mrf_cache = env._mrf_eigenbasis
        version = env._state_version

        env.layout.connectivity = (
            expected.connectivity
        )  # e.g. a saved 4-connected graph
        env._setup_from_layout()

        assert env._state_version == version + 1
        assert env.connectivity is env.layout.connectivity
        assert env.bin_centers is env.layout.bin_centers
        assert env._kernel_cache == {}
        assert env._kdtree_cache is None
        assert env._diffusion_geometry is not old_geometry
        assert env._diffusion_eigenbasis is not old_diffusion_cache
        assert env._mrf_eigenbasis is not old_mrf_cache
        np.testing.assert_allclose(
            env.get_differential_operator().toarray(),
            expected.get_differential_operator().toarray(),
        )

    def test_invalid_layout_graph_leaves_environment_unchanged(self) -> None:
        """Validation runs before any environment state changes."""
        env = Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
        )
        env.compute_kernel(1.0)
        connectivity, version = env.connectivity, env._state_version
        broken = connectivity.copy()
        for _, _, data in broken.edges(data=True):
            del data["edge_id"]
        env.layout.connectivity = broken

        with pytest.raises(ValueError, match="layout engine bug"):
            env._setup_from_layout()

        assert env.connectivity is connectivity
        assert env._state_version == version
        assert env._kernel_cache


class TestPublishedGeometryIsReadOnly:
    """Versioned caches cannot see in-place edits, so geometry refuses them."""

    def test_arrays_are_read_only_and_graph_is_frozen(self) -> None:
        import networkx as nx

        env = Environment.from_grid_mask(
            np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
        )
        for env_or_copy in (env, env.copy()):
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
        assert env.layout_parameters["active_mask"][0, 0]
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
        assert getattr(env, name) is before


class TestVersionedCachedProperty:
    """Decorator behavior: caches per (instance, _state_version)."""

    def test_caches_within_same_version(self) -> None:
        calls = {"n": 0}

        class Stub:
            _state_version = 0

            @versioned_cached_property
            def value(self) -> int:
                calls["n"] += 1
                return 42

        stub = Stub()
        assert stub.value == 42
        assert stub.value == 42
        assert calls["n"] == 1

    def test_recomputes_after_version_bump(self) -> None:
        calls = {"n": 0}

        class Stub:
            _state_version = 0

            @versioned_cached_property
            def value(self) -> int:
                calls["n"] += 1
                return calls["n"]

        stub = Stub()
        assert stub.value == 1
        assert stub.value == 1  # still cached
        stub._state_version += 1
        assert stub.value == 2  # recomputed
        assert stub.value == 2  # cached at the new version

    def test_separate_instances_have_separate_caches(self) -> None:
        class Stub:
            _state_version = 0

            def __init__(self, v: int) -> None:
                self._v = v

            @versioned_cached_property
            def value(self) -> int:
                return self._v

        a = Stub(1)
        b = Stub(2)
        assert a.value == 1
        assert b.value == 2
        a._v = 99
        # No version bump → still cached at the old value.
        assert a.value == 1
        a._state_version = 1
        assert a.value == 99

    def test_double_assignment_in_class_raises(self) -> None:
        # Ensure the same descriptor cannot be assigned to two different
        # attribute names on the same class — matches the cached_property
        # invariant that names are stable.
        descriptor = versioned_cached_property(lambda self: 0)

        # Python 3.10/3.11 wraps exceptions raised in __set_name__ in a
        # RuntimeError (with the original exception as __cause__); 3.12+
        # propagates the original exception as-is. Accept both shapes and
        # unwrap if needed before checking the underlying message.
        with pytest.raises((TypeError, RuntimeError)) as exc_info:

            class Stub:
                _state_version = 0
                first = descriptor
                second = descriptor

            _ = Stub  # silence unused warning

        err: BaseException | None = exc_info.value
        if isinstance(err, RuntimeError) and err.__cause__ is not None:
            err = err.__cause__
        assert isinstance(err, TypeError)
        assert "two different names" in str(err)
