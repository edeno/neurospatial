"""Continuous arena domains, clipped-cell lookup and walls through derivations."""

import copy
import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from environment_helpers import publish_binning, publish_environment
from shapely.geometry import LineString, Point, Polygon, box

from neurospatial import Environment, IncompatibleEnvironmentError
from neurospatial.arena import OpenArena, UnknownArena
from neurospatial.decoding.posterior import _validate_encoding_environment
from neurospatial.geometry._locators import (
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._polygon import materialize_cells
from neurospatial.geometry._polygon_cells import ClippedLocator
from neurospatial.ops.transforms import AffineND


@pytest.mark.parametrize("width", [2.0**-40, 1e-100])
def test_clipping_preserves_resolved_tiny_cells_without_geometry_repair(width):
    edges = np.array([0.0, 1.0])
    parent = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool), grid_edges=(edges, edges)
    )
    child = parent.clip(box(0.0, 0.0, width, 1.0))
    assert child.n_bins == 1
    np.testing.assert_array_equal(child.bin_sizes, [width])
    np.testing.assert_array_equal(child.bin_at(child.bin_centers), [0])
    np.testing.assert_array_equal(child.clip_mapping.parent_bin_indices, [0])


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_diagonal_crop_preserves_area_and_ownership_across_units(scale):
    edges = np.linspace(0.0, scale, 21)
    parent = Environment.from_grid_mask(
        np.ones((20, 20), dtype=bool), grid_edges=(edges, edges)
    )
    child = parent.clip(Polygon([(0, 0), (scale, 0), (0, scale)]))
    assert child.n_bins == 210
    np.testing.assert_allclose(child.bin_sizes.sum(), scale**2 / 2, rtol=1e-14)
    np.testing.assert_array_equal(child.bin_at(child.bin_centers), np.arange(210))
    np.testing.assert_array_equal(
        parent.bin_at(child.bin_centers), child.clip_mapping.parent_bin_indices
    )
    restored = Environment.from_dict(child.to_dict())
    np.testing.assert_array_equal(restored.bin_sizes, child.bin_sizes)
    np.testing.assert_array_equal(restored.bin_at(child.bin_centers), np.arange(210))


@pytest.mark.parametrize("angle", [0.1, 0.3, 0.37, 0.7])
def test_rotated_grid_noop_crop_preserves_cells_and_ownership(angle):
    edges = np.linspace(0.0, 1.0, 21)
    parent = Environment.from_grid_mask(
        np.ones((20, 20), dtype=bool), grid_edges=(edges, edges)
    ).apply_transform(AffineND(_rotation(angle)))
    child = parent.clip(box(-2, -2, 2, 2))
    assert child.n_bins == parent.n_bins
    np.testing.assert_allclose(child.bin_sizes, parent.bin_sizes, rtol=1e-13)
    np.testing.assert_array_equal(child.bin_at(child.bin_centers), np.arange(400))
    np.testing.assert_array_equal(child.clip_mapping.parent_bin_indices, np.arange(400))
    # Arena and locator frames are stored independently. Their reconstructed
    # boundaries may differ at coordinate precision, but a no-op crop must not
    # lose parent support or turn rounding-sized strips into separate cells.
    for original, clipped in zip(
        materialize_cells(parent._binning),
        materialize_cells(child._binning),
        strict=True,
    ):
        assert original.symmetric_difference(clipped).area < 1e-14 * original.area
    # Exact geometry remains usable when tiny rounded kinks prevent the
    # conservative two-point certificate; clipping must not relax that check.
    if not child._binning.faces.available:
        with pytest.raises(NotImplementedError, match="certified"):
            child.compute_kernel(bandwidth=0.1, mode="density")
    restored = Environment.from_dict(child.to_dict())
    np.testing.assert_array_equal(restored.bin_sizes, child.bin_sizes)
    np.testing.assert_array_equal(restored.bin_at(child.bin_centers), np.arange(400))


def _rotation(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize(
    "restore", [lambda env: env, lambda env: Environment.from_dict(env.to_dict())]
)
def test_clipped_boundary_bins_match_the_equivalent_whole_bin_selection(restore):
    parent = _grid((20, 20))
    selected = parent.subset(polygon=box(2, 2, 17, 17))
    clipped = restore(parent.clip(box(2, 2, 17, 17)))
    assert len(clipped.boundary_bins) == 56
    np.testing.assert_array_equal(
        clipped.bin_centers[clipped.boundary_bins],
        selected.bin_centers[selected.boundary_bins],
    )


def test_clipped_boundaries_count_holes_without_vertex_only_contacts():
    parent = _grid((5, 5))
    child = parent.clip(box(0, 0, 5, 5).difference(box(2, 2, 3, 3)))
    assert len(child.boundary_bins) == 20
    centers = child.bin_centers[child.boundary_bins]
    assert any(np.array_equal(center, [1.5, 2.5]) for center in centers)
    assert not any(np.array_equal(center, [1.5, 1.5]) for center in centers)


def test_internal_physical_wall_does_not_create_an_analysis_perimeter():
    parent = _grid(
        (5, 5), arena=OpenArena(box(0, 0, 5, 5), walls=(LineString([(2, 0), (2, 5)]),))
    )
    child = parent.clip(box(0, 0, 5, 5))
    np.testing.assert_array_equal(child.boundary_bins, parent.boundary_bins)
    assert len(child.boundary_bins) == 16


def test_polygon_boundary_bins_survive_subset_and_rotation():
    parent = _grid((8, 8)).clip(box(0, 0, 8, 8))
    subset = parent.subset(polygon=box(1, 1, 6, 6))
    assert len(subset.boundary_bins) == 16
    rotated = subset.apply_transform(AffineND(_rotation(0.3)))
    np.testing.assert_array_equal(rotated.boundary_bins, subset.boundary_bins)


def _grid(shape=(2, 1), *, arena=None, diagonals=False):
    return Environment.from_grid_mask(
        np.ones(shape, dtype=bool),
        grid_edges=tuple(np.arange(n + 1, dtype=float) for n in shape),
        connect_diagonal_neighbors=diagonals,
        arena=arena,
    )


def test_decoder_accepts_semantically_equal_polygon_rings_and_wall_order():
    wall_a = LineString([(1, 0), (1, 2)])
    wall_b = LineString([(0, 1), (2, 1)])
    first = OpenArena(box(0, 0, 2, 2), walls=(wall_a, wall_b))
    second = OpenArena(
        Polygon(list(first.polygon.exterior.coords)[::-1]),
        walls=(LineString(list(wall_b.coords)[::-1]), wall_a),
    )
    source, target = _grid((2, 2), arena=first), _grid((2, 2), arena=second)
    _validate_encoding_environment(
        target, SimpleNamespace(env=source), context="review"
    )


def test_decoder_rejects_different_inherited_bin_boundary_ownership():
    source = _grid().clip(box(0, 0, 2, 1))
    reversed_grid = RectilinearLocator(
        (np.array([-2.0, -1.0, 0.0]), np.array([0.0, 1.0])),
        (2, 1),
        np.ones((2, 1), bool),
        np.array([0, 1]),
    )
    alternate = SubsetLocator(
        TransformedLocator(reversed_grid, np.diag([-1.0, 1.0, 1.0])), np.array([1, 0])
    )
    binning = replace(
        source._binning,
        locator=ClippedLocator(alternate, source._binning.shapes, source.clip_mapping),
    )
    layout = publish_binning(
        binning,
        dimension_ranges=source.dimension_ranges,
        layout_type="PolygonCells",
        build_params={},
    )
    target = publish_environment(
        "alternate", layout, "PolygonCells", {}, arena=source.arena
    )
    query = np.array([[1.0, 0.5]])
    np.testing.assert_array_equal(source.bin_at(query), [1])
    np.testing.assert_array_equal(target.bin_at(query), [0])
    with pytest.raises(IncompatibleEnvironmentError, match="bin-boundary ownership"):
        _validate_encoding_environment(
            target, SimpleNamespace(env=source), context="review"
        )


def test_transformed_polygon_domain_keeps_the_same_strict_center_selection():
    arena = OpenArena(box(0, 0, 2, 1))
    environments = [
        Environment.from_grid_mask(
            np.ones((3, 1), bool),
            grid_edges=(np.array([-0.5, 0.5, 1.5, 2.5]), np.array([0.0, 1.0])),
            arena=supplied,
        )
        for supplied in (arena, arena.transformed(np.eye(3)))
    ]
    for env in environments:
        np.testing.assert_array_equal(env.bin_centers, [[1.0, 0.5]])
        np.testing.assert_array_equal(env.bin_sizes, [1.0])


def test_closed_wall_endpoint_blocks_corner_navigation_before_and_after_clip():
    arena = OpenArena(box(0, 0, 2, 2), walls=(LineString([(1, 0), (1, 1)]),))
    env = _grid((2, 2), arena=arena, diagonals=True)
    assert not env.connectivity.has_edge(0, 3)
    clipped = env.clip(box(0, 0, 2, 2))
    np.testing.assert_array_equal(
        clipped._binning.connections.pairs, env._binning.connections.pairs
    )
    assert not clipped.connectivity.has_edge(0, 3)


def test_mapped_clipped_circle_stays_inside_its_mapped_arena():
    parent = Environment.from_polygon(Point(0, 0).buffer(3), bin_size=0.5).clip(
        box(-10, -10, 10, 10)
    )
    angle = 0.3
    transform = AffineND(
        np.array(
            [
                [np.cos(angle), -np.sin(angle), 1e9],
                [np.sin(angle), np.cos(angle), -1e9],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    child = parent.apply_transform(transform)
    assert child.n_bins == parent.n_bins
    np.testing.assert_allclose(child.bin_sizes, parent.bin_sizes, rtol=1e-14)


def test_known_arena_still_refuses_excluded_stored_center():
    parent = _grid()
    with pytest.raises(ValueError, match="bin center"):
        publish_environment(
            "invalid",
            parent,
            "MaskedGrid",
            {},
            arena=OpenArena(box(0, 0, 0.75, 1)),
        )


@pytest.mark.parametrize("origin", [0.0, 1e9])
def test_arena_cell_containment_still_rejects_resolvable_outside_slivers(origin):
    parent = (
        _grid()
        .clip(box(0, 0, 2, 1))
        .apply_transform(
            AffineND(
                np.array([[1.0, 0.0, origin], [0.0, 1.0, origin], [0.0, 0.0, 1.0]])
            )
        )
    )
    with pytest.raises(ValueError, match=r"outside.*arena"):
        publish_environment(
            "invalid",
            parent,
            "PolygonCells",
            {},
            arena=OpenArena(box(origin, origin, origin + 2 - 1e-3, origin + 1)),
        )


@pytest.mark.parametrize(
    "operation", [copy.deepcopy, lambda env: pickle.loads(pickle.dumps(env))]
)
@pytest.mark.parametrize("transformed", [False, True])
def test_decoder_accepts_identical_clipped_ownership_after_copy(operation, transformed):
    source = _grid().clip(box(0.25, 0, 2, 1))
    if transformed:
        angle = 0.3
        source = source.apply_transform(
            AffineND(
                np.array(
                    [
                        [np.cos(angle), -np.sin(angle), 1e9],
                        [np.sin(angle), np.cos(angle), -1e9],
                        [0.0, 0.0, 1.0],
                    ]
                )
            )
        )
    target = operation(source)
    _validate_encoding_environment(
        target, SimpleNamespace(env=source), context="review"
    )


class _CustomArena:
    """Unknown continuous geometry need not have a transport codec."""

    n_dims = 2
    coordinate_kind = "cartesian"
    domain_known = False
    physical_footprint_known = False
    walls = ()

    def contains(self, positions):
        raise NotImplementedError

    def polygonal_domain(self):
        return None

    def transformed(self, matrix):
        return self


def test_custom_arena_requires_owned_data_before_publication():
    with pytest.raises(
        TypeError,
        match=r"arena= takes an arena from neurospatial.arena",
    ):
        _grid(arena=_CustomArena())
    source, target = _grid(arena=UnknownArena(2)), _grid(arena=UnknownArena(2))
    _validate_encoding_environment(
        target, SimpleNamespace(env=source), context="review"
    )


class _CustomHexLayout:
    """A valid custom layout supplies shapes and delegated point containment."""

    def __init__(self, native):
        self.native = native
        self.bin_centers = native.bin_centers
        self.connectivity = native.connectivity
        self.dimension_ranges = native.dimension_ranges
        self.grid_edges = ()
        self.grid_shape = (native.n_bins,)
        self.active_mask = np.ones(native.n_bins, dtype=bool)
        self._build_params_used = {}
        self.is_linearized_track = False
        self.layout_type = "CustomHex"

    def build(self, **kwargs):
        pass

    def bin_sizes(self):
        return self.native.bin_sizes

    def bin_shapes(self):
        return self.native._binning.shapes

    def plot(self, ax=None, **kwargs):
        return self.native.plot(ax=ax, **kwargs)


@pytest.mark.parametrize("built_environment", ["hex"], indirect=True)
def test_custom_layout_shapes_do_not_force_unsupported_arena_inference(
    built_environment,
):
    with pytest.raises(
        TypeError,
        match=r"No installed snapshot adapter|owned immutable snapshot.*Migrate custom",
    ):
        publish_environment(
            "custom", _CustomHexLayout(built_environment), "CustomHex", {}
        )
    layout = publish_binning(
        built_environment._binning,
        dimension_ranges=built_environment.dimension_ranges,
        layout_type="CustomHex",
        build_params={},
    )
    env = publish_environment("custom", layout, "CustomHex", {})
    assert env.arena.domain_known
    assert not env.arena.physical_footprint_known
    np.testing.assert_array_equal(env.bin_at(env.bin_centers), np.arange(env.n_bins))


def _walled_grid():
    return _grid(
        (4, 4),
        arena=OpenArena(box(0, 0, 4, 4), walls=(LineString([(2, 0), (2, 4)]),)),
        diagonals=True,
    )


def _wall_transform(offset, scale=1.0):
    angle = 0.3
    return AffineND(
        np.array(
            [
                [scale * np.cos(angle), -scale * np.sin(angle), offset + 0.1],
                [scale * np.sin(angle), scale * np.cos(angle), -offset - 0.2],
                [0.0, 0.0, 1.0],
            ]
        )
    )


@pytest.mark.parametrize("offset", [0.0, 1e6, 1e9])
@pytest.mark.parametrize("scale", [1.0, 1.3])
@pytest.mark.parametrize("clipped", [False, True])
def test_transformed_physical_walls_use_the_shared_source_frame(offset, scale, clipped):
    parent = _walled_grid()
    if clipped:
        parent = parent.clip(box(0.25, 0.25, 3.75, 3.75))
    child = parent.apply_transform(_wall_transform(offset, scale))
    np.testing.assert_array_equal(
        child._binning.connections.pairs, parent._binning.connections.pairs
    )
    assert child.arena.physical_footprint_known
    import networkx as nx

    assert nx.number_connected_components(child.connectivity) == 2


@pytest.mark.parametrize("history", ["native", "subset", "clip", "nested"])
def test_transformed_physical_walls_survive_history_and_transport(
    history, round_trip_environment
):
    parent = _walled_grid()
    child = parent.apply_transform(_wall_transform(1e6, 1.3))
    if history == "subset":
        child = child.subset(mask=np.arange(child.n_bins) % 4 != 3)
    elif history == "clip":
        from shapely.affinity import affine_transform

        matrix = _wall_transform(1e6, 1.3).A
        region = affine_transform(
            box(0.25, 0.25, 3.75, 3.75),
            [*matrix[0, :2], *matrix[1, :2], *matrix[:2, 2]],
        )
        child = child.clip(region)
    elif history == "nested":
        child = child.subset(mask=np.arange(child.n_bins) % 4 != 3)
        child = child.apply_transform(_wall_transform(0.0, 0.75))
    loaded = round_trip_environment(child)
    np.testing.assert_array_equal(
        loaded._binning.connections.pairs, child._binning.connections.pairs
    )
    np.testing.assert_array_equal(loaded.bin_sizes, child.bin_sizes)
    np.testing.assert_array_equal(
        loaded.bin_at(child.bin_centers), np.arange(child.n_bins)
    )
    assert len(loaded.arena.walls) == 1


@pytest.mark.parametrize("matching_frame", [False, True])
def test_saved_transformed_walls_still_refuse_cell_interior_crossings(matching_frame):
    from neurospatial.geometry._codecs import encode_arena
    from neurospatial.io import from_dict, to_dict

    transform = _wall_transform(1e6, 1.3)
    child = _walled_grid().apply_transform(transform)
    matrix = transform.A.copy()
    wall_x = 2.01 if matching_frame else 2.0
    if not matching_frame:
        matrix[0, 2] += 0.01
    changed = OpenArena(
        box(0, 0, 4, 4), walls=(LineString([(wall_x, 0), (wall_x, 4)]),)
    ).transformed(matrix)
    payload = to_dict(child)
    metadata, arrays = encode_arena(changed)
    payload["arena"] = metadata
    for name, array in arrays.items():
        key = f"arena/{name}"
        payload["arrays"][key] = array.tolist()
        payload["array_specs"][key] = {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        }
    with pytest.raises(ValueError, match=r"wall 'wall_0' cuts through bin"):
        from_dict(payload)
