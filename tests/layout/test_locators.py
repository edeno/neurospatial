"""Point lookup retains geometric boundaries and owns its reusable data."""

from __future__ import annotations

import copy
import pickle
import warnings
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest
from environment_helpers import native_layout
from hypothesis import given
from hypothesis import strategies as st
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import Point, Polygon

from neurospatial.geometry._codecs import decode, encode
from neurospatial.geometry._locators import (
    HexLocator,
    MeshLocator,
    NearestBinLocator,
    RectilinearLocator,
    _TriangleIndex,
)


def _warnings(caught):
    return [f"{warning.category.__name__}: {warning.message}" for warning in caught]


def _assert_mesh_membership(locator, points, actual):
    """Check geometry, not platform-specific Qhull triangle numbering.

    The recorded inputs still exercise the original boundaries and missing rows.
    Shared-edge points may belong to either adjacent triangle. No lookup helper
    is used to construct the expected containing-cell sets.
    """
    cells = {
        int(active): Polygon(locator.points[triangle])
        for triangle, active in zip(
            locator.simplices, locator.simplex_to_active, strict=True
        )
        if active >= 0
    }
    for coordinates, index in zip(np.atleast_2d(points), actual, strict=True):
        if not np.all(np.isfinite(coordinates)):
            assert index == -1
            continue
        point = Point(coordinates)
        if locator.boundary is not None and not locator.boundary.contains(point):
            assert index == -1
            continue
        if index == -1:
            # At an active/inactive shared edge, the lower-numbered triangle wins.
            assert not any(cell.contains(point) for cell in cells.values())
        else:
            assert index in cells
            # Lookup permits rounding error at a triangle edge.
            assert cells[index].distance(point) <= 1e-12


@pytest.mark.parametrize(
    "probe",
    ["centers", "random", "boundaries", "nonfinite", "single", "empty", "wrong_width"],
)
def test_locator_matches_previous_lookup(
    built_environment, request, locator_parity, probe
):
    """Compare against saved values, including exact boundaries and error text."""
    env = built_environment
    case = request.node.callspec.params["built_environment"]
    prefix = f"{case}/lookup/{probe}"
    if not case.startswith("mesh"):
        np.testing.assert_array_equal(
            env.bin_centers, locator_parity[f"{case}/centers"]
        )
    locator = env._binning.locator
    assert isinstance(
        locator, (RectilinearLocator, HexLocator, MeshLocator, NearestBinLocator)
    )
    assert locator is env._binning.locator
    points = locator_parity[f"{prefix}/points"].copy()
    unchanged = points.copy()
    if case == "hex" and probe == "wrong_width":
        # This validation replaces the old hexagonal acceptance of extra columns.
        with pytest.raises(ValueError, match=r"Expected points of shape \(M, 2\)"):
            locator.locate(points)
    elif f"{prefix}/error_type" in locator_parity:
        assert locator_parity[f"{prefix}/error_type"].item() == "ValueError"
        with pytest.raises(ValueError) as exc:
            locator.locate(points)
        assert str(exc.value) == locator_parity[f"{prefix}/error_message"].item()
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            actual = locator.locate(points)
        if case.startswith("mesh"):
            _assert_mesh_membership(env._binning.locator, points, actual)
        else:
            np.testing.assert_array_equal(actual, locator_parity[f"{prefix}/values"])
        assert actual.dtype == np.intp
        expected_warnings = locator_parity[f"{prefix}/warnings"].tolist()
        if case == "hex" and probe == "nonfinite":
            # The recording cast infinite coordinates to integers, with
            # invalid-value warnings; non-finite rows are now screened first.
            expected_warnings = []
        assert _warnings(caught) == expected_warnings
    np.testing.assert_array_equal(points, unchanged)
    assert points.flags.writeable


def test_locator_non_finite_rows(built_environment):
    """A missing sample does not invalidate finite rows from the same call."""
    env = built_environment
    points = np.tile(env.bin_centers[0], (4, 1))
    points[1, 0] = np.nan
    points[2, -1] = np.inf
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", RuntimeWarning)
        actual = env._binning.locator.locate(points)
    np.testing.assert_array_equal(actual, [0, -1, -1, 0])


def test_locator_bin_centers_map_to_themselves(built_environment):
    """Every active-bin identity survives its own geometric lookup."""
    env = built_environment
    np.testing.assert_array_equal(
        env._binning.locator.locate(env.bin_centers), np.arange(env.n_bins)
    )


@pytest.mark.parametrize("width_offset", [-1, 1])
@pytest.mark.parametrize(
    ("n_rows", "fill"),
    [(0, 0.0), (2, 0.0), (2, np.nan), (2, np.inf)],
    ids=["empty", "finite", "nan", "inf"],
)
def test_wrong_coordinate_width_is_rejected_before_row_filtering(
    built_environment, width_offset, n_rows, fill
):
    """Empty or entirely missing batches still have to name the right axes."""
    env = built_environment
    points = np.full((n_rows, env.n_dims + width_offset), fill)
    with pytest.raises(ValueError):
        env._binning.locator.locate(points)
    with pytest.raises(ValueError):
        env._binning.locator.locate(points)


@st.composite
def _rectilinear_cases(draw):
    n_dims = draw(st.integers(min_value=1, max_value=3))
    shape = tuple(draw(st.integers(min_value=1, max_value=4)) for _ in range(n_dims))
    edges = tuple(
        np.concatenate(
            (
                [0.0],
                np.cumsum(draw(st.lists(st.integers(1, 4), min_size=n, max_size=n))),
            )
        )
        for n in shape
    )
    active = np.asarray(
        draw(
            st.lists(
                st.booleans(),
                min_size=int(np.prod(shape)),
                max_size=int(np.prod(shape)),
            )
        )
    ).reshape(shape)
    points = np.asarray(
        draw(
            st.lists(
                st.tuples(
                    *(
                        st.floats(
                            -1,
                            float(axis[-1]) + 1,
                            allow_nan=False,
                            allow_infinity=False,
                        )
                        for axis in edges
                    )
                ),
                min_size=1,
                max_size=20,
            )
        ),
        dtype=float,
    )
    boundaries = np.stack(
        [part.ravel() for part in np.meshgrid(*edges, indexing="ij")], axis=1
    )
    return edges, shape, active, np.concatenate((points, boundaries))


def _brute_force_boxes(edges, shape, active, points):
    """Search boxes independently of digitize or any locator helper.

    A point's half-open box owns it. A point no active half-open box owns but
    an active closed box contains (on the outline of the active cells) goes to
    the closed box whose upper faces it touches on the fewest axes, the lowest
    axes first.
    """
    result = np.full(len(points), -1, dtype=np.intp)
    best = [None] * len(points)
    active_index = 0
    for cell in np.ndindex(shape):
        if not active[cell]:
            continue
        low = np.asarray([edges[d][index] for d, index in enumerate(cell)])
        high = np.asarray([edges[d][index + 1] for d, index in enumerate(cell)])
        inside = np.all(points >= low, axis=1) & np.all(points < high, axis=1)
        result[inside] = active_index
        closed = np.all(points >= low, axis=1) & np.all(points <= high, axis=1)
        for row in np.flatnonzero(closed & ~inside):
            touched = tuple(int(d) for d in np.flatnonzero(points[row] == high))
            rank = (len(touched), touched)
            if best[row] is None or rank < best[row][0]:
                best[row] = (rank, active_index)
        active_index += 1
    for row, choice in enumerate(best):
        if result[row] < 0 and choice is not None:
            result[row] = choice[1]
    return result


@pytest.mark.hypothesis
@given(_rectilinear_cases())
def test_rectilinear_locator_matches_brute_force(case):
    """Arbitrary masks and unequal cell widths keep half-open ownership inside
    the active cells and close their outline."""
    edges, shape, active, points = case
    mapping = np.full(active.size, -1, dtype=np.intp)
    mapping[active.ravel()] = np.arange(np.count_nonzero(active))
    locator = RectilinearLocator(edges, shape, active, mapping)
    np.testing.assert_array_equal(
        locator.locate(points), _brute_force_boxes(edges, shape, active, points)
    )


@pytest.mark.parametrize("built_environment", ["mesh_boundary"], indirect=True)
def test_mesh_locator_respects_boundary(built_environment, locator_parity):
    """A hole and a concave notch remain outside even within the convex hull."""
    prefix = "mesh_boundary/lookup/boundary_exclusions"
    points = locator_parity[f"{prefix}/points"]
    expected = locator_parity[f"{prefix}/values"]
    actual = built_environment._binning.locator.locate(points)
    _assert_mesh_membership(built_environment._binning.locator, points, actual)
    np.testing.assert_array_equal(actual < 0, expected < 0)
    np.testing.assert_array_equal(actual[:2], [-1, -1])
    assert np.all(actual[2:4] >= 0)
    assert actual[4] == -1


# The corners of a unit square are cocircular, so either diagonal gives a valid
# Delaunay triangulation; which one Qhull returns differs across platforms.
_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_DIAGONAL_02 = np.array([[0, 1, 2], [0, 2, 3]])
_DIAGONAL_13 = np.array([[0, 1, 3], [1, 2, 3]])
# Probes below/above each diagonal, on both diagonals, at a corner and outside.
_SQUARE_PROBES = np.array(
    [
        [0.9, 0.2],
        [0.2, 0.1],
        [0.1, 0.6],
        [0.5, 0.5],
        [0.25, 0.75],
        [0.0, 0.0],
        [1.5, 0.5],
        [-1e-3, 0.5],
    ]
)


@pytest.mark.parametrize(
    ("triangles", "expected"),
    [
        (_DIAGONAL_02, [0, 0, 1, 0, 1, 0, -1, -1]),
        (_DIAGONAL_02[::-1], [1, 1, 0, 0, 0, 0, -1, -1]),
        (_DIAGONAL_13, [1, 0, 0, 0, 0, 0, -1, -1]),
        (_DIAGONAL_13[::-1], [0, 1, 1, 0, 0, 1, -1, -1]),
    ],
    ids=[
        "diagonal_02",
        "diagonal_02_reordered",
        "diagonal_13",
        "diagonal_13_reordered",
    ],
)
def test_mesh_locator_follows_stored_triangles(triangles, expected):
    """Lookup uses the stored triangles and numbering, whatever Qhull rebuilds.

    A point on a shared edge or vertex belongs to the lowest-numbered stored
    triangle that contains it. The codec round trip keeps the same lookup.
    """
    locator = MeshLocator(_SQUARE, triangles, np.arange(2), None)
    np.testing.assert_array_equal(locator.locate(_SQUARE_PROBES), expected)
    restored = decode(*encode(locator, "mesh"), "mesh")
    np.testing.assert_array_equal(restored.simplices, triangles)
    np.testing.assert_array_equal(restored.locate(_SQUARE_PROBES), expected)


@pytest.mark.parametrize(
    ("points", "triangles", "message"),
    [
        (_SQUARE, np.vstack([_DIAGONAL_02, _DIAGONAL_13[:1]]), "overlap"),
        (_SQUARE, np.vstack([_DIAGONAL_02, _DIAGONAL_02[:1]]), "overlap"),
        (_SQUARE, np.array([[0, 1, 2], [0, 2, 2]]), "repeat a vertex"),
        (_SQUARE, np.array([[0, 1, 2], [0, 2, 4]]), "vertex index"),
        (_SQUARE, np.array([[0, 1, 2], [0, -1, 3]]), "vertex index"),
        (_SQUARE, np.array([[0, 1, 2]]).reshape(1, 3, 1), "shape"),
        (np.vstack([_SQUARE, [[np.nan, 0.0]]]), _DIAGONAL_02, "finite"),
    ],
    ids=[
        "overlapping",
        "duplicated",
        "repeated_vertex",
        "index_too_large",
        "negative_index",
        "bad_shape",
        "nonfinite_point",
    ],
)
def test_mesh_locator_refuses_invalid_triangulation(points, triangles, message):
    """Triangles that cannot assign each point to one bin are refused."""
    locator = MeshLocator(points, triangles, np.arange(len(triangles)), None)
    with pytest.raises(ValueError, match=message):
        locator.locate(_SQUARE_PROBES)


def test_mesh_locator_skips_flat_triangles_like_qhull_output():
    """Qhull can emit zero-area slivers along straight edges; they hold no points."""
    points = np.vstack([_SQUARE, [[0.5, 0.0]]])
    flat = np.vstack([_DIAGONAL_02, [[0, 4, 1]]])
    locator = MeshLocator(points, flat, np.array([0, 1, -1]), None)
    reference = MeshLocator(_SQUARE, _DIAGONAL_02, np.arange(2), None)
    probes = np.vstack([_SQUARE_PROBES, [[0.5, 0.0], [0.25, 0.0]]])
    np.testing.assert_array_equal(locator.locate(probes), reference.locate(probes))


@pytest.fixture(params=["rectilinear", "hex", "mesh", "nearest"])
def locator_and_inputs(request):
    """Standalone locators retain no editable constructor buffers."""
    if request.param == "rectilinear":
        edges = (np.arange(6.0)[::2], np.arange(6.0)[::2])
        mask = np.array([[True, False], [True, True]])
        mapping = np.array([0, -1, 1, 2])
        inputs = {
            "edges": edges,
            "shape": (2, 2),
            "active_mask": mask,
            "flat_to_active": mapping,
        }
        return RectilinearLocator(**inputs), inputs, np.array([[1.0, 1.0], [3.0, 3.0]])
    if request.param == "hex":
        inputs = {
            "offset_x": 0.0,
            "offset_y": 0.0,
            "radius": 1.0,
            "centers_shape": (2, 2),
            "flat_to_active": np.arange(4),
        }
        return HexLocator(**inputs), inputs, np.array([[0.0, 0.0], [1.0, 1.0]])
    if request.param == "mesh":
        points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [2.0, 3.0], [0.8, 1.2]])
        triangles = Delaunay(points).simplices.copy()
        inputs = {
            "points": points,
            "simplices": triangles,
            "simplex_to_active": np.arange(len(triangles)),
            "boundary": None,
        }
        return MeshLocator(**inputs), inputs, points[triangles].mean(axis=1)
    centers = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    inputs = {"centers": centers, "bin_indices": np.array([4, 2, 9])}
    return NearestBinLocator(**inputs), inputs, centers.copy()


def _arrays(value):
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, tuple):
        for part in value:
            yield from _arrays(part)


def test_locators_own_constructor_buffers(locator_and_inputs):
    locator, inputs, points = locator_and_inputs
    expected = locator.locate(points)
    for name, value in inputs.items():
        for original, published in zip(
            _arrays(value), _arrays(getattr(locator, name)), strict=True
        ):
            assert original.flags.writeable
            assert not np.shares_memory(original, published)
            original[...] = 0
            assert not published.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                published.flat[0] = 0
    np.testing.assert_array_equal(locator.locate(points), expected)


@pytest.mark.parametrize(
    "restore", [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_locators_are_identity_hashed_and_read_only(locator_and_inputs, restore):
    """Owned data survives copying while KD-tree and triangle-index memos rebuild."""
    locator, _, points = locator_and_inputs
    expected = locator.locate(points)
    equal_values = replace(locator)
    assert locator != equal_values
    assert len({locator, equal_values}) == 2
    with pytest.raises(FrozenInstanceError):
        setattr(locator, fields(locator)[0].name, None)
    restored = restore(locator)
    assert restored != locator
    assert not any(
        isinstance(value, (_TriangleIndex, cKDTree))
        for value in vars(restored).values()
    )
    for field in fields(restored):
        for array in _arrays(getattr(restored, field.name)):
            assert not array.flags.writeable
    np.testing.assert_array_equal(restored.locate(points), expected)


def test_nearest_lookup_keeps_unlimited_distance_and_identity_mapping():
    locator = NearestBinLocator(
        np.array([[0.0, 0.0], [2.0, 0.0]]), bin_indices=np.array([8, 3])
    )
    np.testing.assert_array_equal(
        locator.locate(np.array([[-100.0, 0.0], [100.0, 0.0]])), [8, 3]
    )


def test_nearest_lookup_failure_is_raised_not_reported_as_outside(monkeypatch):
    """A failed KD-tree query is an error, never "every point is outside"."""

    class FailingTree:
        def query(self, points):
            raise MemoryError("query buffer")

    locator = NearestBinLocator(np.array([[0.0, 0.0], [2.0, 0.0]]))
    monkeypatch.setattr(
        NearestBinLocator, "kdtree", lambda self, indexes: FailingTree()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(MemoryError, match="query buffer"):
            locator.locate(np.array([[0.0, 0.0]]))


def test_empty_nearest_lookup_still_validates_axes():
    """No centers means no match; it does not permit a different coordinate space."""
    locator = NearestBinLocator(np.empty((0, 2)))
    np.testing.assert_array_equal(
        locator.locate(np.array([[0.0, 0.0], [np.nan, 0.0]])), [-1, -1]
    )
    with pytest.raises(ValueError, match="dimension"):
        locator.locate(np.empty((0, 3)))


def test_legacy_layout_rebuilds_missing_locator(
    built_environment, request, locator_parity
):
    """Saved layout data can recreate a missing memoized locator."""
    env = copy.deepcopy(built_environment)
    layout = native_layout(env)
    previous = layout.locator()
    del layout._locator
    restored = layout.locator()
    assert restored is not previous
    assert restored is layout.locator()
    case = request.node.callspec.params["built_environment"]
    points = locator_parity[f"{case}/lookup/random/points"]
    np.testing.assert_array_equal(
        restored.locate(points),
        previous.locate(points)
        if case.startswith("mesh")
        else locator_parity[f"{case}/lookup/random/values"],
    )
