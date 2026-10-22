"""The triangle-mesh skew warning appears where a field is smoothed.

Building a mesh's cell faces never warns: comparing environments, decoding,
subsetting and gradients build faces without smoothing anything. Smoothing
warns once per cached geometry, at the scientist's call.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
from shapely.geometry import Polygon

from neurospatial import Environment
from neurospatial.decoding import decode_position
from neurospatial.encoding import SpatialRatesResult, compare_spatial_rates
from neurospatial.environment._comparison import (
    _environment_differences,
    _mapped_bin_difference,
)
from neurospatial.geometry import TriangleShapes, triangle_skew_message
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import Affine2D

_SKEW = "non-K-orthogonal"


def _sheared_mesh(offset: float = 0.0) -> Environment:
    """A thin parallelogram sheared by 6: about 16% of interior edges skewed."""
    shear = 6.0
    corners = [
        (x + shear * y + offset, y) for x, y in [(-30, -8), (30, -8), (30, 8), (-30, 8)]
    ]
    return Environment.from_layout(
        create_layout(
            "TriangularMesh", boundary_polygon=Polygon(corners), point_spacing=4.0
        )
    )


def _square_mesh() -> Environment:
    """A square with uniform spacing: right-isosceles, well-shaped triangles."""
    square = Polygon([(-30, -30), (30, -30), (30, 30), (-30, 30)])
    return Environment.from_layout(
        create_layout("TriangularMesh", boundary_polygon=square, point_spacing=4.0)
    )


def _skew_warnings(record: list[warnings.WarningMessage]) -> list:
    return [w for w in record if _SKEW in str(w.message)]


def _rates(env: Environment) -> SpatialRatesResult:
    return SpatialRatesResult(
        firing_rates=np.ones((1, env.n_bins)),
        occupancy=np.ones(env.n_bins),
        env=env,
        method="binned",
        bandwidth=0.0,
        unit_ids=[1],
    )


@pytest.fixture
def meshes() -> tuple[Environment, Environment]:
    first, second = _sheared_mesh(), _sheared_mesh()
    assert first._geometry is not second._geometry
    return first, second


def _compare(first: Environment, second: Environment) -> None:
    identity = np.column_stack([np.arange(first.n_bins)] * 2)
    assert _environment_differences(first, second) == (None, None)
    assert _mapped_bin_difference(first, second, identity) is None


def test_comparing_skewed_meshes_is_silent(meshes) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _compare(*meshes)


def test_comparing_rates_and_decoding_are_silent(meshes) -> None:
    first, second = meshes
    counts = np.zeros((3, 1), dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compare_spatial_rates(_rates(first), _rates(second))
        decode_position(first, counts, _rates(second), dt=0.1)


def test_smoothing_after_comparison_still_warns(meshes) -> None:
    first, second = meshes
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _compare(first, second)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        first.smooth(np.ones(first.n_bins), bandwidth=3.0, mode="average")
    skew = _skew_warnings(record)
    assert len(skew) == 1
    assert skew[0].category is UserWarning
    assert skew[0].filename == __file__


def test_subset_and_gradient_are_silent_and_the_subset_warns_when_smoothed() -> None:
    env = _sheared_mesh()
    keep = env.bin_centers[:, 0] < np.median(env.bin_centers[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        subset = env.subset(mask=keep)
        env.get_differential_operator()
        subset.get_differential_operator()
    assert isinstance(subset.binning.shapes, TriangleShapes)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        subset.smooth(np.ones(subset.n_bins), bandwidth=3.0, mode="average")
        subset.smooth(np.ones(subset.n_bins), bandwidth=3.0, mode="average")
    skew = _skew_warnings(record)
    assert [w.filename for w in skew] == [__file__]
    # The message counts the subset's own interior edges, not the parent's.
    parent = triangle_skew_message(
        env.binning.shapes.points,
        env.binning.shapes.triangles,
        env.bin_centers,
        env.edge_pairs,
    )
    assert parent is not None
    assert _counts(str(skew[0].message)) != _counts(parent)


def test_well_shaped_mesh_never_warns() -> None:
    env = _square_mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env.smooth(np.ones(env.n_bins), bandwidth=3.0, mode="average")
        env.get_diffusion_geometry()


def test_a_mesh_that_cannot_smooth_raises_without_the_skew_warning() -> None:
    # Shearing the coordinates leaves cells whose faces are not certified.
    shear = np.array([[1.0, 1.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    env = _square_mesh().apply_transform(Affine2D(shear))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(NotImplementedError, match="finite-volume cell geometry"):
            env.smooth(np.ones(env.n_bins), bandwidth=3.0, mode="average")


def test_public_diffusion_geometry_warns_once() -> None:
    env = _sheared_mesh()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        env.get_diffusion_geometry()
        env.get_diffusion_geometry()
    assert [w.filename for w in _skew_warnings(record)] == [__file__]


def _counts(message: str) -> tuple[int, int]:
    match = re.search(r"(\d+)/(\d+) \(", message)
    assert match is not None, message
    return int(match.group(1)), int(match.group(2))


def _loop_counts(
    points: np.ndarray, triangles: np.ndarray, centers: np.ndarray, pairs: np.ndarray
) -> tuple[int, int]:
    """The per-edge skew count of the original face builder."""
    skew_angle_rad = np.deg2rad(30.0)
    n_interior = 0
    n_skewed = 0
    for u, v in pairs:
        shared = sorted(set(triangles[u].tolist()) & set(triangles[v].tolist()))
        if len(shared) != 2:
            continue
        edge_vec = np.asarray(points[shared[1]] - points[shared[0]], dtype=float)
        n_interior += 1
        centroid_line = np.asarray(centers[v] - centers[u], dtype=float)
        edge_len = np.linalg.norm(edge_vec)
        line_len = np.linalg.norm(centroid_line)
        if edge_len > 0.0 and line_len > 0.0:
            normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
            cos_angle = abs(float(np.dot(centroid_line / line_len, normal)))
            angle = float(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
            if angle > skew_angle_rad:
                n_skewed += 1
    return n_skewed, n_interior


@pytest.mark.parametrize("build", [_sheared_mesh, _square_mesh])
def test_skew_counts_match_the_per_edge_loop(build) -> None:
    env = build()
    shapes = env.binning.shapes
    arrays = (shapes.points, shapes.triangles, env.bin_centers, env.edge_pairs)
    n_skewed, n_interior = _loop_counts(*arrays)
    assert n_interior > 0
    message = triangle_skew_message(*arrays)
    if n_skewed / n_interior > 0.05:
        assert message is not None
        assert _counts(message) == (n_skewed, n_interior)
        assert message.startswith("Triangular mesh is non-K-orthogonal: ")
    else:
        assert message is None
    assert (message is None) == (build is _square_mesh)


def _combined() -> Environment:
    """A skewed mesh beside a well-shaped one, far apart."""
    square = _square_mesh().apply_transform(
        Affine2D(np.array([[1.0, 0.0, 500.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    )
    return Environment.combine({"skewed": _sheared_mesh(), "square": square})


def test_a_combined_environment_warns_when_smoothed_on_a_skewed_component() -> None:
    first, second = _combined(), _combined()
    counts = np.zeros((3, 1), dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _environment_differences(first, second) == (None, None)
        compare_spatial_rates(_rates(first), _rates(second))
        decode_position(first, counts, _rates(second), dt=0.1)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        first.smooth(np.ones(first.n_bins), bandwidth=3.0, mode="average")
        first.diffuse(np.ones(first.n_bins), bandwidth=3.0, mode="average")
    skew = _skew_warnings(record)
    assert [w.filename for w in skew] == [__file__]
    message = str(skew[0].message)
    assert message.endswith("(combined component 'skewed')")
    # The component's count is the mesh's own count.
    alone = _sheared_mesh()
    shapes = alone.binning.shapes
    expected = triangle_skew_message(
        shapes.points, shapes.triangles, alone.bin_centers, alone.edge_pairs
    )
    assert expected is not None
    assert _counts(message) == _counts(expected)


def test_a_combined_subset_warns_only_for_retained_skewed_triangles() -> None:
    env = _combined()
    square_only = env.subset(mask=env.bin_centers[:, 0] > 300.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        square_only.smooth(np.ones(square_only.n_bins), bandwidth=3.0, mode="average")
    skewed_only = env.subset(mask=env.bin_centers[:, 0] < 300.0)
    with pytest.warns(UserWarning, match="combined component 'skewed'"):
        skewed_only.smooth(np.ones(skewed_only.n_bins), bandwidth=3.0, mode="average")
