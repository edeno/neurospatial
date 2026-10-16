"""Environment is an immutable value: identity equality, no copy, no fitted state."""

import copy

import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def env() -> Environment:
    positions = np.array([[x, y] for x in range(10) for y in range(10)], dtype=float)
    return Environment.from_samples(positions, bin_size=1.0, name="arena")


class TestIdentityEquality:
    def test_equal_to_itself(self, env):
        assert env == env

    def test_with_metadata_returns_unequal_object(self, env):
        relabelled = env.with_metadata(name="other")
        assert relabelled != env
        assert relabelled is not env

    def test_not_equal_to_its_name(self, env):
        assert env != "arena"
        assert env != env.name

    def test_hashable_by_identity(self, env):
        other = env.with_metadata(name="arena")
        assert hash(env) == hash(env)
        assert len({env, other}) == 2
        assert {env: 1}[env] == 1


class TestNoCopy:
    def test_copy_method_is_removed(self, env):
        assert not hasattr(Environment, "copy")
        assert not hasattr(env, "copy")

    def test_copy_and_deepcopy_return_the_same_object(self, env):
        assert copy.copy(env) is env
        assert copy.deepcopy(env) is env

    def test_deepcopy_of_a_container_keeps_the_shared_environment(self, env):
        holder = copy.deepcopy({"env": env, "values": [1, 2]})
        assert holder["env"] is env


class TestNoFittedState:
    def test_fitted_machinery_is_removed(self, env):
        import neurospatial
        import neurospatial.environment

        assert not hasattr(env, "_is_fitted")
        assert not hasattr(neurospatial, "EnvironmentNotFittedError")
        assert not hasattr(neurospatial.environment, "EnvironmentNotFittedError")
        assert not hasattr(neurospatial.environment, "check_fitted")

    def test_direct_construction_still_raises(self):
        with pytest.raises(ValueError, match="E1006"):
            Environment()

    def test_environment_is_not_a_dataclass(self):
        import dataclasses

        assert not dataclasses.is_dataclass(Environment)


class TestAnalysisCacheInitialisation:
    def test_new_environment_starts_with_empty_cache(self, env):
        assert len(env._cache) == 0


class TestBinCountWarningsAndRefusals:
    def test_n_bins_matches_bin_centers(self, env):
        assert env.n_bins == env.bin_centers.shape[0] == 100

    def test_distance_between_points_outside_raise(self, env):
        far = np.array([1e6, 1e6])
        with pytest.raises(ValueError, match="outside the environment"):
            env.distance_between(points=[far, far])

    def test_linearization_refusals_on_non_track_are_type_errors(self, env):
        with pytest.raises(TypeError, match="to_linear"):
            env.to_linear(np.array([[1.0, 1.0]]))
        with pytest.raises(TypeError, match="linear_to_nd"):
            env.linear_to_nd(np.array([0.0]))
