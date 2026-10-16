"""Environment-owned caches share explicit-data calculus builders."""

from unittest.mock import patch

import numpy as np

from neurospatial import Environment
from neurospatial.binning import FaceGeometry
from neurospatial.ops.calculus import (
    _compute_differential_operator,
    _compute_divergence_operator,
    compute_differential_operator,
)


def test_calculus_builders_accept_only_explicit_numerical_data() -> None:
    faces = FaceGeometry(
        np.array([[0, 1]]), np.array([2.0]), np.array([3.0]), np.array([4.0, 5.0])
    )
    differential = _compute_differential_operator(faces, 2)
    divergence = _compute_divergence_operator(faces, differential)
    np.testing.assert_array_equal(differential.toarray(), [[-1 / 3], [1 / 3]])
    np.testing.assert_array_equal(divergence.toarray(), [[0.5], [-0.4]])


def test_public_calculus_paths_share_owned_operator_caches() -> None:
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
    )
    with (
        patch(
            "neurospatial.environment.core._compute_differential_operator",
            wraps=_compute_differential_operator,
        ) as differential,
        patch(
            "neurospatial.environment.core._compute_divergence_operator",
            wraps=_compute_divergence_operator,
        ) as divergence,
    ):
        expected = env.get_differential_operator().toarray()
        exported = compute_differential_operator(env)
        exported.data[:] = 0
        env.get_divergence_operator()
        env.get_divergence_operator()
        np.testing.assert_array_equal(
            env.get_differential_operator().toarray(), expected
        )
        assert differential.call_count == divergence.call_count == 1
        env.clear_cache()
        env.get_divergence_operator()
        assert differential.call_count == divergence.call_count == 2
