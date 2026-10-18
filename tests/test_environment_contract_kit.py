"""The public extension contract checks work without importing pytest in the kit."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.binning import Binning, MissingFaces, NearestBinLocator, SpatialBins
from neurospatial.testing import assert_environment_contract


def test_contract_kit_on_built_in_snapshots(built_environment):
    env = built_environment
    assert_environment_contract(env, env.bin_centers, np.arange(env.n_bins))


def test_contract_kit_uses_independent_boundary_and_outside_oracles():
    env = Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])),
    )
    assert_environment_contract(
        env,
        [[0.0, 0.5], [1.0, 0.5], [2.0, 0.5], [-0.1, 0.5]],
        [0, 1, 1, -1],  # the outer edge x = 2 belongs to the last bin
    )
    with pytest.raises(AssertionError):
        assert_environment_contract(env, [[1.0, 0.5]], [0])


def test_contract_kit_exercises_missing_face_refusal():
    centers = np.array([[0.0, 0.0], [2.0, 0.0]])
    env = Environment.from_bins(
        Binning(
            "sample_sites",
            SpatialBins.from_arrays(centers, [1.0, 1.0], 2, [[0, 1]], [2.0]),
            NearestBinLocator(centers, max_distance=0.1),
            MissingFaces("No physical faces are available."),
        )
    )
    assert_environment_contract(env, [[0, 0], [2, 0], [10, 0]], [0, 1, -1])


def test_contract_kit_checks_that_a_track_without_width_refuses_lookups():
    track = Environment.linear_track(endpoints=[(0, 0), (10, 0)], bin_size=2.0)
    # Any expected bins: positions on a widthless track have none.
    assert_environment_contract(track, [[1.0, 0.0], [1.0, 50.0]], [0, 0])


def test_contract_kit_gives_a_track_with_width_the_full_lookup_checks():
    track = Environment.linear_track(
        endpoints=[(0, 0), (10, 0)], bin_size=2.0, track_width=2.0
    )
    assert_environment_contract(track, [[1.0, 0.0], [1.0, 50.0]], [0, -1])
    with pytest.raises(AssertionError):
        # A position 50 cm off a 2 cm wide track has no bin.
        assert_environment_contract(track, [[1.0, 50.0]], [0])
