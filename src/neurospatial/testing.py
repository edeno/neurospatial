"""Small contract checks for environment snapshots and extension adapters.

These helpers require no pytest dependency. Supply an independent lookup
oracle, including boundary points relevant to the geometry being published.
"""

from __future__ import annotations

import copy
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment
    from neurospatial.geometry import EgocentricPolarSystem

    PolarEnvironment = Environment[EgocentricPolarSystem]

__all__ = ["assert_environment_contract"]


def assert_environment_contract(
    env: Environment | PolarEnvironment,
    points: ArrayLike,
    expected_bins: ArrayLike,
) -> None:
    """Check ownership, native lookup and persistence against supplied answers.

    Parameters
    ----------
    env : Environment or PolarEnvironment
        Environment, including one produced by an installed adapter.
    points : array-like of float, shape (n_points, n_dims)
        Native coordinates. Include boundary points with independently known
        ownership; this helper cannot infer a scientific boundary convention.
    expected_bins : array-like of int, shape (n_points,)
        Expected bin IDs, with -1 for outside points. On a track built
        without ``track_width`` positions have no bins, so the helper checks
        instead that every position lookup refuses with the fix that names
        ``track_width``.

    Raises
    ------
    AssertionError
        If lookup, ownership, capability refusal or a supported round trip
        violates the contract. Missing required storage adapters also fail.
    """
    query = np.array(points, dtype=np.float64, copy=True)
    expected = np.asarray(expected_bins)
    assert query.ndim == 2 and query.shape[1] == env.n_dims
    assert expected.dtype.kind in "iu" and expected.shape == (len(query),)
    assert np.all((expected >= -1) & (expected < env.n_bins))
    original_query = query.copy()

    def check_lookup(value: Environment | PolarEnvironment) -> None:
        if not _assigns_positions(value):
            for refused in (query, np.full((1, value.n_dims), np.nan)):
                try:
                    value.bin_at(refused)
                except NotImplementedError as error:
                    assert "track_width" in str(error), error
                else:
                    raise AssertionError(
                        "a track without track_width assigned positions to bins"
                    )
            check_width(value)
            return
        actual = value.bin_at(query)
        assert actual.dtype == np.dtype(np.intp)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(query, original_query)
        invalid = np.full((2, value.n_dims), np.nan)
        invalid[1] = np.inf
        np.testing.assert_array_equal(value.bin_at(invalid), [-1, -1])
        check_width(value)

    def check_width(value: Environment | PolarEnvironment) -> None:
        try:
            value.bin_at(np.zeros((1, value.n_dims + 1)))
        except ValueError:
            pass
        else:
            raise AssertionError("lookup accepted an incorrect coordinate width")

    check_lookup(env)
    centers = env.bin_centers.copy()
    sizes = env.bin_sizes.copy()
    pairs = env.edge_pairs.copy()
    exported = env.bin_centers
    try:
        exported[:] = -12345
    except ValueError:
        pass
    else:
        raise AssertionError("bin_centers accepted an in-place write")
    exported.shape = (exported.size,)
    np.testing.assert_array_equal(env.bin_centers, centers)
    graph = env.to_networkx()
    graph.clear()
    assert env.connectivity.number_of_nodes() == env.n_bins
    try:
        env.regions.clear()
    except TypeError:
        pass
    else:
        raise AssertionError("environment region snapshot accepted mutation")

    same_geometry = [
        copy.copy(env),
        copy.deepcopy(env),
        env.with_metadata(name="contract replacement"),
    ]
    read_back = [
        pickle.loads(pickle.dumps(env)),
        type(env).from_dict(env.to_dict()),
    ]
    with tempfile.TemporaryDirectory(prefix="neurospatial-contract-") as directory:
        path = Path(directory) / "environment.json"
        env.to_file(path)
        read_back.append(type(env).from_file(path))
    for restored in same_geometry + read_back:
        assert type(restored) is type(env)
        check_lookup(restored)
        np.testing.assert_array_equal(restored.bin_centers, centers)
        np.testing.assert_array_equal(restored.bin_sizes, sizes)
        np.testing.assert_array_equal(restored.edge_pairs, pairs)

    if not env.binning.faces.available:
        try:
            env.get_differential_operator()
        except NotImplementedError:
            pass
        else:
            raise AssertionError("numerical operator accepted missing face geometry")


def _assigns_positions(env: Environment | PolarEnvironment) -> bool:
    """Whether ``env`` maps positions to bins; a track needs its width for that."""
    from neurospatial.geometry import require_track_width

    try:
        require_track_width(env.binning, env.arena)
    except NotImplementedError:
        return False
    return True
