"""Check that an environment behaves as the rest of neurospatial expects.

Use :func:`assert_environment_contract` when you build an environment your
own way, for example with :meth:`Environment.from_bins` or through an
installed extension, and want to confirm that position lookup, copying,
saving and loading all keep its bins. You supply the answer: points and the
bins you know they belong to, worked out by hand rather than with
neurospatial. The helper does not need pytest.
"""

from __future__ import annotations

import copy
import pickle
import tempfile
from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import ArrayLike

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.geometry import EgocentricPolarSystem

    PolarEnvironment = Environment[EgocentricPolarSystem]

__all__ = ["assert_environment_contract"]


def assert_environment_contract(
    env: Environment | PolarEnvironment,
    points: ArrayLike,
    expected_bins: ArrayLike,
) -> None:
    """Check bin lookup, read-only state and saving against known answers.

    The helper checks that

    - ``env.bin_at(points)`` returns ``expected_bins`` as ``intp`` without
      editing ``points``, returns -1 for NaN and infinite positions, and
      refuses points with the wrong number of coordinates;
    - ``env.bin_centers`` is read-only, editing the graph from
      ``env.to_networkx()`` leaves the environment unchanged, and
      ``env.regions`` refuses mutation;
    - ``copy.copy``, ``copy.deepcopy``, ``env.with_metadata``, pickling,
      ``to_dict``/``from_dict`` and ``to_file``/``from_file`` (through a
      temporary JSON file) return an environment of the same type with the
      same bin centers, bin sizes, connections and lookup answers;
    - ``env.get_differential_operator()`` raises ``NotImplementedError``
      when the bins have no face geometry.

    Parameters
    ----------
    env : Environment or PolarEnvironment
        Environment to check, including one produced by an installed
        extension.
    points : array-like of float, shape (n_points, n_dims)
        Positions in the environment's native coordinates: the coordinates
        of ``env.bin_centers``, such as (x, y) for an arena or
        (distance, angle in radians) for an egocentric polar environment.
        Include points on bin boundaries whose bin you know independently;
        the helper cannot infer which bin a boundary belongs to.
    expected_bins : array-like of int, shape (n_points,)
        Bin index each point belongs to, with -1 for points outside the
        environment. On a track built without ``track_width`` positions have
        no bins, so the helper checks instead that every position lookup
        refuses with the fix that names ``track_width``.

    Raises
    ------
    AssertionError
        If ``points`` or ``expected_bins`` are malformed, or if lookup,
        read-only state, a refusal or a round trip breaks the contract. The
        checks run under ``python -O`` too.

    Examples
    --------
    Two 1 cm bins side by side; the shared edge at x = 1 belongs to the
    second bin and the outer edge at x = 2 to the last bin:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.testing import assert_environment_contract
    >>> env = Environment.from_grid_mask(
    ...     np.ones((2, 1), dtype=bool),
    ...     grid_edges=(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])),
    ... )
    >>> assert_environment_contract(
    ...     env,
    ...     points=[[0.0, 0.5], [1.0, 0.5], [2.0, 0.5], [-0.1, 0.5]],
    ...     expected_bins=[0, 1, 1, -1],
    ... )
    """
    query = np.array(points, dtype=np.float64, copy=True)
    expected = np.asarray(expected_bins)
    if query.ndim != 2 or query.shape[1] != env.n_dims:
        raise AssertionError(
            _format_error(
                f"points must have shape (n_points, {env.n_dims}); got {query.shape}.",
                why=f"the environment has {env.n_dims} coordinate dimensions.",
                fix="pass one row per point, one column per coordinate.",
            )
        )
    if expected.dtype.kind not in "iu":
        raise AssertionError(
            _format_error(
                f"expected_bins must hold integer bin indices; got dtype "
                f"{expected.dtype}.",
                why="bin indices are integers, with -1 for outside points.",
                fix="pass expected_bins as integers, e.g. np.array([0, 1, -1]).",
            )
        )
    if expected.shape != (len(query),):
        raise AssertionError(
            _format_error(
                f"expected_bins must have shape ({len(query)},); got {expected.shape}.",
                why="each point needs exactly one expected bin.",
                fix="pass one expected bin per row of points.",
            )
        )
    if not np.all((expected >= -1) & (expected < env.n_bins)):
        raise AssertionError(
            _format_error(
                f"expected_bins must lie in -1..{env.n_bins - 1}; got "
                f"{expected.min()}..{expected.max()}.",
                why=f"the environment has {env.n_bins} bins and -1 marks a "
                "point outside it.",
                fix="check the hand-worked bin indices against env.bin_centers.",
            )
        )
    original_query = query.copy()

    def check_lookup(value: Environment | PolarEnvironment) -> None:
        if not _assigns_positions(value):
            for refused in (query, np.full((1, value.n_dims), np.nan)):
                try:
                    value.bin_at(refused)
                except NotImplementedError as error:
                    if "track_width" not in str(error):
                        raise AssertionError(
                            "a track without track_width refused a lookup "
                            f"without naming track_width: {error}"
                        ) from error
                else:
                    raise AssertionError(
                        "a track without track_width assigned positions to bins"
                    )
            check_width(value)
            return
        actual = value.bin_at(query)
        if actual.dtype != np.dtype(np.intp):
            raise AssertionError(
                f"bin_at returned dtype {actual.dtype}; bin indices must be intp"
            )
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
    sizes = env.binning.bins.sizes.copy()
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
    if env.connectivity.number_of_nodes() != env.n_bins:
        raise AssertionError("editing env.to_networkx() changed the environment")
    try:
        # Typed as a read-only Mapping; check the runtime object refuses too.
        cast("MutableMapping[str, Any]", env.regions).clear()
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
        if type(restored) is not type(env):
            raise AssertionError(
                f"a copy or round trip returned {type(restored).__name__}, "
                f"not {type(env).__name__}"
            )
        check_lookup(restored)
        np.testing.assert_array_equal(restored.bin_centers, centers)
        np.testing.assert_array_equal(restored.binning.bins.sizes, sizes)
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
