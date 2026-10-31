"""Scientific invariants for ordinary combined environments."""

import copy
import math
import pickle

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.binning import (
    BinConnections,
    Binning,
    MissingFaces,
    NearestBinLocator,
    SpatialBins,
)
from neurospatial.layout import create_layout


def grid(shift=0.0, *, dims=2, count=2, units="cm"):
    return Environment.from_grid_mask(
        np.ones((count,) + (1,) * (dims - 1), dtype=bool),
        grid_edges=(
            np.arange(count + 1.0) + shift,
            *[np.arange(2.0) for _ in range(dims - 1)],
        ),
        units=units,
        frame="lab",
    )


def nearest(center, *, radius=None):
    centers = np.array([center], dtype=float)
    return Environment.from_bins(
        Binning(
            "sites",
            SpatialBins.from_arrays(centers, [1.0], 2),
            BinConnections.from_arrays(np.empty((0, 2), dtype=np.intp), []),
            NearestBinLocator(centers, max_distance=radius),
            MissingFaces("Sites lack faces."),
        ),
        units="cm",
        frame="lab",
    )


@pytest.mark.parametrize("shift,expected", [(3.0, 14), (5.0, 16)])
@pytest.mark.parametrize("auto_bridge", [False, True])
def test_combined_boundary_uses_exposed_cells_independently_of_bridges(
    shift, expected, auto_bridge
):
    def square(offset):
        return Environment.from_grid_mask(
            np.ones((3, 3), dtype=bool),
            grid_edges=(np.arange(4.0) + offset, np.arange(4.0)),
        )

    env = Environment.combine(
        {"left": square(0.0), "right": square(shift)}, auto_bridge=auto_bridge
    )
    assert len(env.boundary_bins) == expected
    if shift == 3.0:
        assert 7 not in env.boundary_bins
        assert 10 not in env.boundary_bins
    restored = Environment.from_dict(env.to_dict())
    np.testing.assert_array_equal(restored.boundary_bins, env.boundary_bins)


def test_combined_planar_boundary_survives_subset_and_rotation():
    from neurospatial.ops.transforms import AffineND

    children = {
        key: Environment.from_grid_mask(
            np.ones((3, 3), dtype=bool),
            grid_edges=(np.arange(4.0) + offset, np.arange(4.0)),
        )
        for key, offset in (("left", 0.0), ("right", 3.0))
    }
    combined = Environment.combine(children)
    subset = combined.subset(mask=combined.bin_centers[:, 0] < 4.0)
    assert len(subset.boundary_bins) == 10
    angle = 0.3
    rotated = subset.apply_transform(
        AffineND(
            np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
    )
    np.testing.assert_array_equal(rotated.boundary_bins, subset.boundary_bins)


@pytest.mark.parametrize(
    "restore",
    [
        lambda x: x,
        copy.copy,
        copy.deepcopy,
        lambda x: pickle.loads(pickle.dumps(x)),
        lambda x: Environment.from_dict(x.to_dict()),
    ],
)
def test_ordinary_environment_retains_support_sources_and_path_bridges(restore):
    left, right = grid(), grid(4.0)
    joined = restore(Environment.combine({"left": left, "right": right}))
    assert type(joined) is Environment
    assert joined.component_keys == ("left", "right")
    assert joined.bin_sources == (("left", 0), ("left", 1), ("right", 0), ("right", 1))
    np.testing.assert_array_equal(
        joined.bin_at([[0.5, 0.5], [4.5, 0.5], [3.0, 0.5], [np.nan, 0.0]]),
        [0, 2, -1, -1],
    )
    np.testing.assert_array_equal(joined.edge_pairs, [[0, 1], [2, 3], [1, 2]])
    assert joined.path_between(bins=[0, 3]) == [0, 1, 2, 3]
    bridges = joined.get_bridge_attributes()
    assert len(bridges) == 1
    assert bridges.iloc[0]["distance"] == 3.0
    assert bridges.iloc[0]["conductance"] == 0.0
    bridges.loc[:, "distance"] = 100
    assert joined.get_bridge_attributes().iloc[0]["distance"] == 3.0


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_cell_overlap_rejected_and_touching_cells_accepted(dims):
    left = grid(count=1, dims=dims)
    with pytest.raises(ValueError, match=r"[Oo]verlap"):
        Environment.combine({"a": left, "b": grid(0.5, count=1, dims=dims)})
    combined = Environment.combine({"a": left, "b": grid(1.0, count=1, dims=dims)})
    assert combined.n_bins == 2


def test_path_bridges_do_not_diffuse_but_declared_conductance_does():
    children = {"a": grid(), "b": grid(4.0)}
    path_only = Environment.combine(children)
    field = np.array([1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        path_only.diffuse(field, bandwidth=1.0, mode="density"), field, atol=1e-14
    )
    coupled = Environment.combine(
        children, bridge_conductances={(("a", 1), ("b", 0)): 0.5}
    )
    output = coupled.diffuse(field, bandwidth=1.0, mode="density")
    assert output[2] > 0
    np.testing.assert_allclose(output @ coupled.bin_sizes, field @ coupled.bin_sizes)
    np.testing.assert_allclose(
        coupled.diffuse(np.ones(4), bandwidth=1.0, mode="density"), np.ones(4)
    )
    with pytest.raises(NotImplementedError, match=r"[Bb]ridge|face"):
        path_only.get_differential_operator()


def test_nearest_support_must_be_certified_and_missing_faces_remain_missing():
    with pytest.raises(ValueError, match="certif"):
        Environment.combine({"a": nearest([0, 0]), "b": nearest([10, 0])})
    with pytest.raises(ValueError, match=r"[Oo]verlap"):
        Environment.combine(
            {"a": nearest([0, 0], radius=1), "b": nearest([1.5, 0], radius=1)}
        )
    env = Environment.combine(
        {"a": nearest([0, 0], radius=0.25), "b": nearest([3, 0], radius=0.25)}
    )
    np.testing.assert_array_equal(env.bin_at([[0, 0], [3, 0], [1, 0]]), [0, 1, -1])
    with pytest.raises(NotImplementedError, match=r"geometry|faces"):
        env.diffuse(np.ones(2), bandwidth=1.0, mode="density")


def test_metadata_regions_and_cutoff():
    from neurospatial import Regions

    regions = Regions()
    regions.add("goal", point=[0.5, 0.5])
    left = grid().with_regions(regions)
    joined = Environment.combine(
        {"left": left, "right": grid(4).with_regions(regions)}, max_mnn_distance=1.0
    )
    assert tuple(joined.regions) == ("left/goal", "right/goal")
    assert joined.get_bridge_attributes().empty
    with pytest.raises(ValueError, match="units"):
        Environment.combine({"left": left, "right": grid(4, units="m")})


def test_subsets_remove_bridges_and_keep_original_sources():
    env = Environment.combine({"a": grid(), "b": grid(4.0)})
    subset = env.subset(mask=np.array([True, False, True, True]))
    assert subset.bin_sources == (("a", 0), ("b", 0), ("b", 1))
    assert subset.get_bridge_attributes().empty
    np.testing.assert_allclose(
        subset.diffuse(np.ones(3), bandwidth=1.0, mode="density"), np.ones(3)
    )


def test_no_bridge_combination_clips_without_losing_sources():
    env = Environment.combine({"a": grid(), "b": grid(4)}, auto_bridge=False)
    clipped = env.clip(box(0.1, 0.1, 1.9, 0.9))
    assert clipped.bin_sources == (("a", 0), ("a", 1))
    # Certified clipped faces, rather than the pre-clip composition, own the
    # numerical geometry even when all surviving bins come from one child.
    expected = grid().clip(box(0.1, 0.1, 1.9, 0.9))
    np.testing.assert_array_equal(
        clipped._diffusion_geometry.W.toarray(),
        expected._diffusion_geometry.W.toarray(),
    )
    np.testing.assert_allclose(
        clipped.diffuse(np.array([1.0, 0.0]), bandwidth=0.5, mode="average"),
        expected.diffuse(np.array([1.0, 0.0]), bandwidth=0.5, mode="average"),
    )


@pytest.mark.parametrize("history", ["clip", "subset", "transform", "load"])
def test_uncertified_combined_clip_cannot_regain_its_parents_diffusion(history):
    from shapely.geometry import Polygon

    from neurospatial.ops.transforms import translate

    children = {
        key: Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool),
            grid_edges=(np.arange(offset, offset + 3.0), np.arange(3.0)),
        )
        for key, offset in (("a", 0), ("b", 4))
    }
    parent = Environment.combine(children, auto_bridge=False)
    assert parent._binning.faces.available
    clipped = parent.clip(Polygon([(0, 0), (2, 0), (0, 2)]))
    if history == "subset":
        clipped = clipped.subset(mask=np.ones(clipped.n_bins, dtype=bool))
    elif history == "transform":
        clipped = clipped.apply_transform(translate(10, 0))
    elif history == "load":
        clipped = Environment.from_dict(clipped.to_dict())
    assert not clipped._binning.faces.available
    with pytest.raises(NotImplementedError, match="orthogonality"):
        clipped.diffuse(np.ones(clipped.n_bins), bandwidth=0.5, mode="average")


@pytest.mark.parametrize("kind", ["hex", "mesh", "polygon", "mask", "samples"])
def test_native_cell_families_keep_child_lookup_and_numeric_blocks(kind):
    from neurospatial.ops.transforms import translate

    if kind == "hex":
        child = Environment.from_layout(
            create_layout(
                "Hexagonal", hexagon_width=1.0, dimension_ranges=[(0, 3), (0, 3)]
            )
        )
    elif kind == "mesh":
        child = Environment.from_layout(
            create_layout(
                "TriangularMesh", boundary_polygon=box(0, 0, 3, 3), point_spacing=0.75
            )
        )
    elif kind == "polygon":
        child = Environment.from_polygon(
            box(0, 0, 3, 3).difference(box(1, 1, 2, 2)), bin_size=0.5
        )
    elif kind == "mask":
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        child = Environment.from_grid_mask(mask, grid_edges=(np.arange(4.0),) * 2)
    else:
        positions = np.random.default_rng(15).uniform(0, 3, size=(100, 2))
        child = Environment.from_samples(positions, bin_size=0.75)
    other = child.apply_transform(translate(10, 0)).with_metadata(frame=child.frame)
    combined = Environment.combine({"a": child, "b": other})
    np.testing.assert_array_equal(
        combined.bin_at(child.bin_centers), np.arange(child.n_bins)
    )
    np.testing.assert_array_equal(
        combined.bin_at(other.bin_centers), child.n_bins + np.arange(other.n_bins)
    )
    values = np.arange(child.n_bins, dtype=float)
    actual = combined.diffuse(
        np.concatenate([values, np.zeros(other.n_bins)]), bandwidth=0.5, mode="average"
    )
    np.testing.assert_allclose(
        actual[: child.n_bins],
        child.diffuse(values, bandwidth=0.5, mode="average"),
        atol=1e-12,
    )
    np.testing.assert_allclose(actual[child.n_bins :], 0, atol=1e-14)


def test_measured_embedded_tracks_and_crossing_footprint_refusal():
    import networkx as nx

    def track(shift, vertical=False):
        graph = nx.Graph()
        graph.add_node(0, pos=(shift, 0.0))
        graph.add_node(
            1, pos=(shift if vertical else shift + 4, 4.0 if vertical else 0.0)
        )
        graph.add_edge(0, 1, distance=4.0)
        return Environment.from_graph(
            graph, [(0, 1)], edge_spacing=0.0, bin_size=1.0, track_width=0.5
        )

    a, b = track(0), track(10)
    combined = Environment.combine({"a": a, "b": b})
    np.testing.assert_array_equal(
        combined.bin_at(b.bin_centers), a.n_bins + np.arange(b.n_bins)
    )
    np.testing.assert_allclose(
        combined.diffuse(np.ones(combined.n_bins), bandwidth=0.5, mode="average"), 1
    )
    with pytest.raises(ValueError, match=r"[Oo]verlap"):
        Environment.combine({"a": a, "crossing": track(2, vertical=True)})


def test_complementary_unbounded_nearest_subsets_are_valid():
    centers = np.array([[0.0, 0.0], [10.0, 0.0]])
    parent = Environment.from_bins(
        Binning(
            "sites",
            SpatialBins.from_arrays(centers, [1, 1], 2),
            BinConnections.from_arrays(np.empty((0, 2), dtype=np.intp), []),
            NearestBinLocator(centers, max_distance=math.inf),
            MissingFaces("sites"),
        )
    )
    a, b = (
        parent.subset(mask=np.array([True, False])),
        parent.subset(mask=np.array([False, True])),
    )
    combined = Environment.combine({"a": a, "b": b})
    np.testing.assert_array_equal(combined.bin_at([[-100, 0], [100, 0]]), [0, 1])


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_automatic_bridges_do_not_bypass_a_third_grid_component(dims):
    joined = Environment.combine(
        {
            key: grid(shift, count=1, dims=dims)
            for key, shift in (("a", 0), ("b", 2), ("c", 4))
        }
    )
    assert not joined.connectivity.has_edge(0, 2)
    assert joined.path_between(bins=[0, 2]) == [0, 1, 2]


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("finite", [False, True])
def test_automatic_bridges_respect_nearest_component_interiors(dims, finite):
    centers = np.array([[shift, *([0.0] * (dims - 1))] for shift in (0.0, 2.0, 4.0)])
    parent = Environment.from_bins(
        Binning(
            "sites",
            SpatialBins.from_arrays(centers, np.ones(3), dims),
            BinConnections.from_arrays(np.empty((0, 2), dtype=int), []),
            NearestBinLocator(centers, max_distance=0.25 if finite else None),
            MissingFaces("Sites lack faces."),
        )
    )
    components = {
        key: parent.subset(mask=np.arange(3) == index)
        for index, key in enumerate(("a", "b", "c"))
    }
    joined = Environment.combine(components)
    assert not joined.connectivity.has_edge(0, 2)
    assert joined.path_between(bins=[0, 2]) == [0, 1, 2]


@pytest.mark.parametrize("angle", [0.0, 0.1, 0.3, 0.37, 0.7])
def test_automatic_bridge_can_follow_a_third_components_exterior_boundary(angle):
    from neurospatial.ops.transforms import AffineND

    third = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool),
        grid_edges=(np.array([2.0, 3.0]), np.array([0.5, 1.5])),
        units="cm",
        frame="lab",
    )
    transform = AffineND(
        np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    joined = Environment.combine(
        {
            key: component.apply_transform(transform)
            for key, component in (
                ("a", grid(count=1)),
                ("b", third),
                ("c", grid(4, count=1)),
            )
        }
    )
    assert joined.connectivity.has_edge(0, 2)


def test_automatic_bridge_cannot_follow_an_internal_voxel_face():
    def end(shift):
        return Environment.from_grid_mask(
            np.ones((1, 1, 1), dtype=bool),
            grid_edges=(
                np.array([shift, shift + 1]),
                np.array([0.5, 1.5]),
                np.array([0.0, 1.0]),
            ),
        )

    middle = Environment.from_grid_mask(
        np.ones((1, 2, 1), dtype=bool),
        grid_edges=(np.array([2.0, 3.0]), np.arange(3.0), np.arange(2.0)),
    )
    joined = Environment.combine({"a": end(0.0), "b": middle, "c": end(4.0)})
    assert not joined.connectivity.has_edge(0, 3)


@pytest.mark.parametrize("angle", [0.0, 0.3])
def test_bridge_can_follow_an_external_voxel_face_after_rotation(angle):
    from neurospatial.ops.transforms import AffineND

    transform = np.eye(4)
    transform[:2, :2] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    components = {
        key: Environment.from_grid_mask(
            np.ones((1, 1, 1), dtype=bool),
            grid_edges=(
                np.array([shift, shift + 1]),
                np.array([low, low + 1]),
                np.arange(2.0),
            ),
        ).apply_transform(AffineND(transform))
        for key, shift, low in (("a", 0.0, 0.5), ("b", 2.0, 1.0), ("c", 4.0, 0.5))
    }
    joined = Environment.combine(components)
    assert joined.connectivity.has_edge(0, 2)


def test_physical_wall_still_blocks_a_boundary_following_bridge():
    from shapely.geometry import LineString

    from neurospatial.arena import OpenArena

    third = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool),
        grid_edges=(np.array([2.0, 3.0]), np.array([0.5, 1.5])),
        arena=OpenArena(box(2, 0.5, 3, 1.5), walls=(LineString([(2, 0.5), (3, 0.5)]),)),
        units="cm",
        frame="lab",
    )
    joined = Environment.combine(
        {"a": grid(count=1), "b": third, "c": grid(4, count=1)}
    )
    assert not joined.connectivity.has_edge(0, 2)


def test_bounded_nearest_clearance_uses_its_retained_partition():
    centers = np.array([[2.0, 1.0], [2.0, -1.0]])
    parent = Environment.from_bins(
        Binning(
            "sites",
            SpatialBins.from_arrays(centers, [1, 1], 2),
            BinConnections.from_arrays(np.empty((0, 2), dtype=int), []),
            NearestBinLocator(centers, max_distance=2.0),
            MissingFaces("Sites lack faces."),
        )
    )
    middle = parent.subset(mask=np.array([True, False]))
    # The radius ball extends across this bridge, but its retained Voronoi
    # cell does not: the dropped site owns the whole y=-0.5 line.
    joined = Environment.combine(
        {
            "a": nearest([-1, -0.5], radius=0.1).with_metadata(units=None, frame=None),
            "b": middle,
            "c": nearest([5, -0.5], radius=0.1).with_metadata(units=None, frame=None),
        }
    )
    assert joined.connectivity.has_edge(0, 2)


def test_third_component_point_contact_is_not_a_crossing():
    from shapely.geometry import Polygon

    middle = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool),
        grid_edges=(np.array([2.0, 3.0]), np.array([0.5, 1.5])),
        units="cm",
        frame="lab",
    ).clip(Polygon([(2, 0.5), (3, 1.5), (2, 1.5)]))
    joined = Environment.combine(
        {"a": grid(count=1), "b": middle, "c": grid(4, count=1)}
    )
    assert joined.connectivity.has_edge(0, 2)


def test_unresolved_tiny_grid_obstacle_refuses_automatic_bridges():
    components = {
        "left": Environment.from_grid_mask(
            np.ones((1,), bool), grid_edges=(np.array([-2.0, -1.0]),)
        ),
        "tiny": Environment.from_grid_mask(
            np.ones((1,), bool), grid_edges=(np.array([-1e-100, 1e-100]),)
        ),
        "right": Environment.from_grid_mask(
            np.ones((1,), bool), grid_edges=(np.array([1.0, 2.0]),)
        ),
    }
    with pytest.raises(ValueError, match=r"resolve.*crossings|disable auto_bridge"):
        Environment.combine(components)
    assert Environment.combine(components, auto_bridge=False).n_bins == 3


@pytest.mark.parametrize("direction", [[1.0, 0.0], [1.0, 1.0], [0.3, 0.7]])
def test_unresolved_tiny_nearest_obstacle_refuses_automatic_bridges(direction):
    direction = np.asarray(direction)
    components = {
        "left": nearest(-direction, radius=0.1),
        "tiny": nearest([0, 0], radius=1e-100),
        "right": nearest(direction, radius=0.1),
    }
    with pytest.raises(ValueError, match=r"resolve.*footprint|disable auto_bridge"):
        Environment.combine(components)
    assert Environment.combine(components, auto_bridge=False).n_bins == 3


@pytest.mark.parametrize(
    "vertical_shift,clear", [(-0.1, False), (0.0, True), (0.1, True)]
)
def test_bounded_nearest_boundary_tangency_is_not_an_interior_crossing(
    vertical_shift, clear
):
    # The exact integer line 7*x + 24*y = 625 is tangent to a radius-25
    # circle because 7**2 + 24**2 == 25**2. Moving it inward must block.
    components = {
        "left": nearest([-65, 45 + vertical_shift], radius=0.1),
        "middle": nearest([0, 0], radius=25.0),
        "right": nearest([175, -25 + vertical_shift], radius=0.1),
    }
    joined = Environment.combine(components)
    assert joined.connectivity.has_edge(0, 2) == clear


@pytest.mark.parametrize("dims", [2, 3])
def test_automatic_bridges_do_not_bypass_a_measured_track(dims):
    import networkx as nx

    def track(shift):
        graph = nx.Graph()
        graph.add_node(0, pos=(shift, *([0.0] * (dims - 1))))
        graph.add_node(1, pos=(shift + 1, *([0.0] * (dims - 1))))
        graph.add_edge(0, 1, distance=1.0)
        return Environment.from_graph(
            graph, [(0, 1)], edge_spacing=0.0, bin_size=1.0, track_width=0.25
        )

    joined = Environment.combine(
        {key: track(shift) for key, shift in (("a", 0), ("b", 2), ("c", 4))}
    )
    assert not joined.connectivity.has_edge(0, 2)


def test_automatic_bridges_do_not_cross_known_walls():
    from shapely.geometry import LineString

    from neurospatial.arena import OpenArena

    arena = OpenArena(box(0, 0, 6, 1), walls=(LineString([(3, 0), (3, 1)]),))
    children = {
        key: Environment.from_grid_mask(
            np.ones((2, 1), dtype=bool),
            grid_edges=(np.arange(3.0) + shift, np.arange(2.0)),
            arena=arena,
        )
        for key, shift in (("left", 0), ("right", 4))
    }
    combined = Environment.combine(children)
    assert combined.get_bridge_attributes().empty
    assert len(combined.arena.walls) == 2


def test_rigid_and_uniform_transforms_preserve_paths_and_scale_geometric_conductance():
    from neurospatial.ops.transforms import scale_3d, translate_3d

    env = Environment.combine(
        {"a": grid(dims=3), "b": grid(4, dims=3)},
        bridge_conductances={(("a", 1), ("b", 0)): 0.5},
    )
    translated = env.apply_transform(translate_3d(2, 3, 4))
    scaled = env.apply_transform(scale_3d(2), units=env.units)
    assert translated.bin_sources == env.bin_sources == scaled.bin_sources
    np.testing.assert_allclose(
        translated.get_bridge_attributes()["distance"],
        env.get_bridge_attributes()["distance"],
    )
    np.testing.assert_allclose(
        scaled.get_bridge_attributes()["conductance"],
        2 * env.get_bridge_attributes()["conductance"],
    )
    np.testing.assert_allclose(
        scaled._diffusion_geometry.W.toarray(), env._diffusion_geometry.W.toarray() * 2
    )


def test_rotated_thin_3d_boxes_with_overlapping_bounds_can_be_certified():
    from neurospatial.ops.transforms import AffineND

    angle = np.pi / 4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    child = Environment.from_grid_mask(
        np.ones((1, 1, 1), dtype=bool),
        grid_edges=(np.array([0, 4.0]), np.array([0, 0.2]), np.array([0, 0.2])),
    )
    first = child.apply_transform(AffineND(transform))
    second_matrix = transform.copy()
    second_matrix[:3, 3] = rotation @ [0, 0.8, 0]
    second = child.apply_transform(AffineND(second_matrix))
    assert Environment.combine({"a": first, "b": second}).n_bins == 2


def test_nearly_parallel_crossing_track_clearance_is_not_falsely_certified():
    import networkx as nx

    def make_track(start, end):
        graph = nx.Graph()
        graph.add_node(0, pos=start)
        graph.add_node(1, pos=end)
        graph.add_edge(0, 1, distance=float(np.hypot.reduce(np.array(end) - start)))
        return Environment.from_graph(
            graph, [(0, 1)], edge_spacing=0.0, bin_size=5e7, track_width=0.1
        )

    first = make_track((-1e8, 0.0), (1e8, 0.0))
    second = make_track((-1e8, -1.0), (1e8, 1.0))
    assert first.arena.contains(np.array([[0.0, 0.0]]))[0]
    assert second.arena.contains(np.array([[0.0, 0.0]]))[0]
    with pytest.raises(ValueError, match=r"certif|[Oo]verlap"):
        Environment.combine({"a": first, "b": second}, auto_bridge=False)


def test_removed_missing_geometry_no_longer_blocks_certified_subset():
    env = Environment.combine(
        {"grid": grid(), "site": nearest([10.0, 0.5], radius=0.1)}
    )
    reduced = env.subset(mask=np.array([True, True, False]))
    np.testing.assert_allclose(
        reduced.diffuse(np.array([1.0, 0.0]), bandwidth=1.0, mode="density"),
        grid().diffuse(np.array([1.0, 0.0]), bandwidth=1.0, mode="density"),
    )


def _walled_grid(shift=0.0, walls=()):
    """Two cells, [shift, shift + 1] and [shift + 1, shift + 2], with walls."""
    from neurospatial.arena import OpenArena

    return Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.arange(3.0) + shift, np.arange(2.0)),
        units="cm",
        frame="lab",
        arena=OpenArena(box(shift, 0, shift + 2, 1), walls=list(walls)),
    )


def _with_child_arena(data, source, child=0):
    """Saved ``data`` with child ``child``'s arena records taken from ``source``."""
    data = copy.deepcopy(data)
    prefixes = (
        f"arena/children/{child}/",
        f"binning/locator/children/{child}/arena/",
    )
    for key in [key for key in data["arrays"] if key.startswith(prefixes)]:
        del data["arrays"][key]
        data["array_specs"].pop(key, None)
    for key, value in source["arrays"].items():
        if key.startswith(prefixes):
            data["arrays"][key] = value
            if key in source["array_specs"]:
                data["array_specs"][key] = source["array_specs"][key]
    data["arena"]["children"][child] = source["arena"]["children"][child]
    data["binning"]["locator"]["children"][child]["arena"] = source["binning"][
        "locator"
    ]["children"][child]["arena"]
    return data


@pytest.mark.parametrize(
    "wall,message",
    [
        # On child a's outer edge: it closes no pair of a, but the bridge from
        # a's bin at (1.5, 0.5) to b's bin at (4.5, 0.5) crosses it.
        (
            (2.0, 0.0, 2.0, 1.0),
            "A stored navigation bridge crosses a known physical wall.",
        ),
        # Between child a's two cells, whose saved pair stays open.
        (
            (1.0, 0.0, 1.0, 1.0),
            "Component connectivity crosses its declared physical walls.",
        ),
    ],
)
def test_saved_composition_rejects_links_across_child_walls(wall, message):
    import re

    from shapely.geometry import LineString

    saved = Environment.combine({"a": _walled_grid(), "b": _walled_grid(4.0)})
    assert saved.get_bridge_attributes()["source"].tolist() == [1]
    walled = Environment.combine(
        {
            "a": _walled_grid(walls=[LineString([wall[:2], wall[2:]])]),
            "b": _walled_grid(4.0),
        },
        auto_bridge=False,
    )
    data = _with_child_arena(saved.to_dict(), walled.to_dict())
    with pytest.raises(ValueError, match=re.escape(message)):
        Environment.from_dict(data)


@pytest.mark.parametrize("corruption", ["faces", "bridge_distance", "arena"])
def test_saved_composition_rejects_inconsistent_geometry(corruption):
    env = Environment.combine(
        {"a": grid(), "b": grid(4.0)}, auto_bridge=corruption != "faces"
    )
    data = env.to_dict()
    if corruption == "faces":
        data["arrays"]["binning/faces/A"] = (
            np.asarray(data["arrays"]["binning/faces/A"]) * 2
        ).tolist()
    elif corruption == "bridge_distance":
        data["arrays"]["binning/locator/bridge_distances"] = [100.0]
        data["arrays"]["binning/connections/all_distances"][-1] = 100.0
    else:
        for key in list(data["arrays"]):
            if key.startswith("arena/children/0/edges_0"):
                data["arrays"][key] = [100.0, 101.0, 102.0]
    with pytest.raises(ValueError):
        Environment.from_dict(data)


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_linear_occupancy_retains_partial_exposure_and_excludes_gap(dims):
    env = Environment.combine(
        {"a": grid(count=1, dims=dims), "b": grid(3, count=1, dims=dims)}
    )
    start = np.full(dims, 0.5)
    stop = start.copy()
    start[0], stop[0] = -1.0, 5.0
    occupancy = env.occupancy(
        np.array([0.0, 6.0]),
        np.array([start, stop]),
        time_allocation="linear",
        max_gap=None,
    )
    np.testing.assert_allclose(occupancy, [1.0, 1.0])


def test_nested_bridges_are_reported_and_coupled_affine_refuses():
    from neurospatial.ops.transforms import AffineND

    inner = Environment.combine(
        {"a": grid(), "b": grid(4)}, bridge_conductances={(("a", 1), ("b", 0)): 0.5}
    )
    outer = Environment.combine({"inner": inner, "c": grid(8)})
    assert len(outer.get_bridge_attributes()) == 2
    matrix = np.array([[1.0, 0.3, 0], [0, 1.0, 0], [0, 0, 1.0]])
    with pytest.raises(ValueError, match="conductance"):
        outer.apply_transform(AffineND(matrix))


@pytest.mark.parametrize("derive", ["subset", "translation"])
@pytest.mark.parametrize("with_bridge", [False, True])
def test_derived_saved_authority_cannot_disagree(derive, with_bridge):
    from neurospatial.ops.transforms import translate

    env = Environment.combine({"a": grid(), "b": grid(4)}, auto_bridge=with_bridge)
    env = (
        env.subset(mask=np.array([True, True, True, True]))
        if derive == "subset"
        else env.apply_transform(translate(2, 3))
    )
    data = env.to_dict()
    if with_bridge:
        data["arrays"]["binning/connections/all_distances"][-1] = 100.0
    else:
        data["arrays"]["binning/faces/A"] = (
            2 * np.asarray(data["arrays"]["binning/faces/A"])
        ).tolist()
    with pytest.raises(ValueError, match="derived combined"):
        Environment.from_dict(data)


def test_saved_combined_faces_cannot_disagree_with_components():
    env = Environment.combine({"a": grid(), "b": grid(4)}, auto_bridge=False)
    data = env.to_dict()
    data["arrays"]["binning/faces/A"] = (
        2 * np.asarray(data["arrays"]["binning/faces/A"])
    ).tolist()
    with pytest.raises(ValueError, match=r": combined faces\.A"):
        Environment.from_dict(data)


def test_nwb_preserves_composition_after_reader_closes(tmp_path):
    from datetime import datetime, timezone

    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    env = Environment.combine(
        {"a": grid(), "b": grid(4)}, bridge_conductances={(("a", 1), ("b", 0)): 0.5}
    )
    nwb = pynwb.NWBFile(
        session_description="composition",
        identifier="combine-regression",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    write_environment(nwb, env)
    path = tmp_path / "combined.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        restored = read_environment(io.read())
    assert restored.bin_sources == env.bin_sources
    np.testing.assert_array_equal(restored.edge_pairs, env.edge_pairs)
    np.testing.assert_allclose(
        restored.compute_kernel(bandwidth=1, mode="average"),
        env.compute_kernel(bandwidth=1, mode="average"),
    )
    assert restored.get_bridge_attributes().iloc[0]["conductance"] == 0.5


def test_rotated_wall_geometry_is_retained_and_transformed_tracks_are_certified():
    from shapely.geometry import LineString

    from neurospatial.arena import OpenArena
    from neurospatial.ops.transforms import AffineND, translate

    arena = OpenArena(box(0, 0, 4, 4), walls=(LineString([(2, 0), (2, 4)]),))
    original = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool), grid_edges=(np.arange(5.0),) * 2, arena=arena
    )
    angle = 0.3
    transform = AffineND(
        np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.1],
                [np.sin(angle), np.cos(angle), -0.2],
                [0, 0, 1],
            ]
        )
    )
    left = original.apply_transform(transform)
    right = (
        original.apply_transform(translate(10, 0))
        .apply_transform(transform)
        .with_metadata(frame=left.frame)
    )
    assert (
        Environment.combine({"left": left, "right": right}, auto_bridge=False).n_bins
        == 32
    )
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(4.0, 0.0))
    graph.add_edge(0, 1, distance=4.0)
    track = Environment.from_graph(
        graph, [(0, 1)], edge_spacing=0.0, bin_size=1.0, track_width=0.5
    )
    moved = track.apply_transform(translate(10, 0)).with_metadata(frame=track.frame)
    assert Environment.combine({"track": track, "moved": moved}).n_bins == 8


def test_bridge_table_numpy_ids_are_valid_inputs_and_reverse_duplicates_refuse():
    children = {"a": grid(), "b": grid(4)}
    row = Environment.combine(children).get_bridge_attributes().iloc[0]
    endpoints = (
        (row["source_component"], row["source_local_bin"]),
        (row["target_component"], row["target_local_bin"]),
    )
    assert (
        Environment.combine(children, bridge_conductances={endpoints: 0.5})
        .get_bridge_attributes()
        .iloc[0]["conductance"]
        == 0.5
    )
    with pytest.raises(ValueError, match="one conductance"):
        Environment.combine(
            children, bridge_conductances={endpoints: 0.5, endpoints[::-1]: 0.7}
        )


@pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf, True, "unknown"])
def test_invalid_bridge_controls_refuse(invalid):
    children = {"a": grid(), "b": grid(4)}
    with pytest.raises(ValueError, match="max_mnn_distance"):
        Environment.combine(children, max_mnn_distance=invalid)
    with pytest.raises(ValueError, match="conductance"):
        Environment.combine(
            children, bridge_conductances={(("a", 1), ("b", 0)): invalid}
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_symmetric_mnn_tie_uses_local_id_and_cutoff_is_inclusive(reverse):
    left = grid()
    right = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool),
        grid_edges=(np.array([0.5, 1.5]), np.array([3.5, 4.5])),
        units="cm",
        frame="lab",
    )
    components = {"left": left, "right": right}
    if reverse:
        components = dict(reversed(tuple(components.items())))
    # Right's one center is equidistant from both left centers. Left bin 0
    # must win regardless of which component is queried first.
    distance = np.sqrt(12.5)
    env = Environment.combine(components, max_mnn_distance=distance)
    bridge = env.get_bridge_attributes().iloc[0]
    endpoints = {
        (bridge["source_component"], bridge["source_local_bin"]),
        (bridge["target_component"], bridge["target_local_bin"]),
    }
    assert endpoints == {("left", 0), ("right", 0)}
    assert bridge["distance"] == distance
    shorter = Environment.combine(
        components, max_mnn_distance=np.nextafter(distance, 0.0)
    )
    assert shorter.get_bridge_attributes().empty


def test_shared_grid_boundary_keeps_child_half_open_rule():
    """Alone, each grid owns x = 1: left on its closed outline, right by its
    half-open cell. Combined, the half-open owner wins in either order."""
    left, right = grid(count=1), grid(1.0, count=1)
    point = np.array([[1.0, 0.5]])
    np.testing.assert_array_equal(left.bin_at(point), [0])
    np.testing.assert_array_equal(right.bin_at(point), [0])
    for components in ({"left": left, "right": right}, {"right": right, "left": left}):
        env = Environment.combine(components)
        index = env.bin_at(point)[0]
        assert env.bin_sources[index] == ("right", 0)


def _rigid(angle, offset=(0.0, 0.0), reflect=False):
    matrix = np.eye(3)
    matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    if reflect:
        matrix[:2, 0] *= -1
    matrix[:2, 2] = offset
    return matrix


@pytest.mark.parametrize(
    "matrix",
    [
        _rigid(np.pi / 2),
        _rigid(0.3, (13.7, -2.1)),
        _rigid(2.0, (1e4, 0.0), reflect=True),
    ],
    ids=["quarter-turn", "rotation", "reflection"],
)
@pytest.mark.parametrize("select", [False, True])
def test_shared_face_owner_does_not_depend_on_transforms_or_order(matrix, select):
    """Rotating or mirroring each grid keeps the half-open owner of a shared
    face, in either component order, as rotating the combined environment
    does, also when the grids were subsets of larger ones."""
    from neurospatial.ops import AffineND

    transform = AffineND(matrix)
    left, right = grid(count=2), grid(2.0, count=2)
    if select:
        # Each grid loses its far cell, so the shared face stays shared.
        left = left.subset(mask=np.array([False, True]))
        right = right.subset(mask=np.array([True, False]))
    face = np.array([[2.0, 0.25], [2.0, 0.5], [2.0, 0.75]])
    plain = Environment.combine({"left": left, "right": right})
    expected = [plain.bin_sources[index] for index in plain.bin_at(face)]
    assert {source[0] for source in expected} == {"right"}
    turned = plain.apply_transform(transform)
    assert [turned.bin_sources[i] for i in turned.bin_at(transform(face))] == expected
    children = {
        "left": left.apply_transform(transform),
        "right": right.apply_transform(transform),
    }
    for order in (("left", "right"), ("right", "left")):
        env = Environment.combine({key: children[key] for key in order})
        found = env.bin_at(transform(face))
        assert [env.bin_sources[index] for index in found] == expected


def test_touching_nearest_supports_resolve_boundary_tie_by_component_order():
    left, right = nearest([0.0, 0.0], radius=1), nearest([2.0, 0.0], radius=1)
    point = np.array([[1.0, 0.0]])
    np.testing.assert_array_equal(left.bin_at(point), [0])
    np.testing.assert_array_equal(right.bin_at(point), [0])
    for components in ({"left": left, "right": right}, {"right": right, "left": left}):
        env = Environment.combine(components)
        index = env.bin_at(point)[0]
        assert env.bin_sources[index] == (next(iter(components)), 0)


def test_overlap_candidates_match_an_exhaustive_bounds_scan():
    from neurospatial.geometry._combine_geometry import _bound_candidates, _Piece

    rng = np.random.default_rng(3)
    for dims in (1, 2, 3):
        sides = []
        for _ in range(2):
            low = np.round(rng.uniform(0, 6, size=(40, dims)) * 2) / 2
            high = low + np.round(rng.uniform(0, 2, size=(40, dims)) * 2) / 2
            low[3, 0], high[5, -1], low[7] = np.nan, np.inf, -np.inf
            sides.append(
                [_Piece("box", None, lo, hi) for lo, hi in zip(low, high, strict=True)]
            )
        left, right = sides
        # Every pair the exact test would not reject by its bounds, in scan order.
        expected = [
            (a, b)
            for a in range(len(left))
            for b in range(len(right))
            if not np.any(
                np.minimum(left[a].high, right[b].high)
                <= np.maximum(left[a].low, right[b].low)
            )
        ]
        assert expected
        assert list(zip(*_bound_candidates(left, right), strict=True)) == expected


def test_touching_grid_certification_tests_no_cell_pairs(monkeypatch):
    import neurospatial.geometry._combine_geometry as geometry

    calls = []
    exact = geometry._overlap

    def counted(a, b):
        calls.append((a, b))
        return exact(a, b)

    monkeypatch.setattr(geometry, "_overlap", counted)

    def square(offset, count=30):
        edges = np.arange(count + 1.0)
        return Environment.from_grid_mask(
            np.ones((count, count), dtype=bool), grid_edges=(edges + offset, edges)
        )

    assert Environment.combine({"a": square(0.0), "b": square(30.0)}).n_bins == 1800
    assert calls == []
    with pytest.raises(ValueError, match=r"[Oo]verlap"):
        Environment.combine({"a": square(0.0), "b": square(29.5)})
    assert len(calls) == 1


def test_components_are_certified_and_validated_once_per_record(monkeypatch):
    import neurospatial.geometry._combine_geometry as geometry
    import neurospatial.geometry._validation as validation

    certified, validated = [], []
    certify, validate = geometry.certify_components, validation._validate_components

    def counted_certify(children, arenas, keys):
        certified.append(keys)
        return certify(children, arenas, keys)

    def counted_validate(binning, coordinates):
        validated.append(binning.kind)
        return validate(binning, coordinates)

    monkeypatch.setattr(geometry, "certify_components", counted_certify)
    monkeypatch.setattr(validation, "_validate_components", counted_validate)

    pair = Environment.combine({"a": grid(), "b": grid(5.0)}, auto_bridge=False)
    assert certified == [("a", "b")]
    certified.clear()
    validated.clear()

    nested = Environment.combine({"ab": pair, "c": grid(10.0)}, auto_bridge=False)
    # The published pair is neither certified nor validated again.
    assert certified == [("ab", "c")]
    assert validated.count("Combined") == 1
    certified.clear()

    nested.subset(mask=np.arange(nested.n_bins) > 0)
    assert certified == []

    # A record decoded from storage is certified again.
    Environment.from_dict(nested.to_dict())
    assert sorted(certified) == [("a", "b"), ("ab", "c")]


def test_subclass_combine_accepts_plain_environment_components():
    class Room(Environment):
        pass

    a = Environment.from_polygon(box(0, 0, 4, 4), bin_size=1.0)
    b = Environment.from_polygon(box(6, 0, 10, 4), bin_size=1.0)
    combined = Room.combine({"a": a, "b": b})
    assert combined.n_bins == a.n_bins + b.n_bins
    assert combined.layout_type == "Combined"


def test_selected_combined_bins_keep_each_components_cell_extent():
    """Dropping one grid cell leaves the grid's other cells spanning the
    same box, so the combined extent does not shrink to bin centers."""
    import networkx as nx

    field = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool), grid_edges=(np.arange(0.0, 11.0, 2.0),) * 2
    )
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 12.0))
    graph.add_node(1, pos=(10.0, 12.0))
    graph.add_edge(0, 1, distance=10.0)
    track = Environment.from_graph(graph, [(0, 1)], bin_size=2.0, track_width=2.0)
    joined = Environment.combine({"field": field, "track": track})
    assert joined.dimension_ranges == ((0.0, 10.0), (0.0, 12.0))
    keep = np.ones(joined.n_bins, dtype=bool)
    keep[0] = False  # the field's corner cell; its neighbors still reach 0
    assert joined.subset(mask=keep).dimension_ranges == ((0.0, 10.0), (0.0, 12.0))
    only_field = np.arange(joined.n_bins) < field.n_bins
    selected = joined.subset(mask=only_field).subset(mask=keep[only_field])
    assert selected.dimension_ranges == field.dimension_ranges


def test_combining_bins_without_cells_spans_their_declared_ranges():
    def points(shift, ranges):
        centers = np.array([[shift, 0.0], [shift + 1.0, 0.0]])
        return Environment.from_bins(
            Binning(
                "explicit",
                SpatialBins.from_arrays(centers, [1.0, 1.0], 2),
                BinConnections.from_arrays([[0, 1]], [1.0]),
                NearestBinLocator(centers, max_distance=0.5),
                MissingFaces("Point bins have no faces."),
                dimension_ranges=ranges,
            )
        )

    left = points(0.0, ((-5.0, 5.0), (-5.0, 5.0)))
    right = points(20.0, ((15.0, 25.0), (-4.0, 6.0)))
    joined = Environment.combine({"left": left, "right": right})
    assert joined.dimension_ranges == ((-5.0, 25.0), (-5.0, 6.0))
    assert Environment.from_dict(joined.to_dict()).dimension_ranges == (
        joined.dimension_ranges
    )


@pytest.mark.parametrize("angle", [0.3, 0.7, 2.0, 1e-9])
def test_transformed_combination_arena_shares_the_bins_inverse(angle):
    """The arena rebuilt for a moved combined environment uses the bins' own
    map, so its inverse is the locator's inverse exactly, as validation
    requires; re-inverting the inverse would round differently."""
    from neurospatial.geometry._arena_fit import _composition_arena, validate_arena
    from neurospatial.geometry._coordinates import CartesianSystem
    from neurospatial.ops import AffineND

    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 13.7],
            [np.sin(angle), np.cos(angle), -2.1],
            [0.0, 0.0, 1.0],
        ]
    )
    joined = Environment.combine({"left": grid(count=2), "right": grid(2.0, count=2)})
    moved = joined.apply_transform(AffineND(matrix))
    locator = moved.binning.locator
    np.testing.assert_array_equal(locator.forward, matrix)
    arena = _composition_arena(locator)
    np.testing.assert_array_equal(arena.forward, locator.forward)
    np.testing.assert_array_equal(arena.inverse, locator.inverse)
    validate_arena(arena, moved.binning, CartesianSystem())


def _maps_into_own_blocks(combined, children):
    """Each child's bin centers land in that child's block of combined bins."""
    offset = 0
    for child in children:
        np.testing.assert_array_equal(
            combined.bin_at(child.bin_centers), offset + np.arange(child.n_bins)
        )
        offset += child.n_bins


def _round_component(kind, center, radius):
    """One nearest-bin site (a ball) or one straight track (a capsule)."""
    import networkx as nx

    if kind == "ball":
        return nearest(center, radius=radius)
    # A track whose capsule ends in a round cap of ``radius`` at ``center``.
    graph = nx.Graph()
    graph.add_node(0, pos=tuple(center))
    graph.add_node(1, pos=(center[0] + 4.0, center[1]))
    graph.add_edge(0, 1, distance=4.0)
    return Environment.from_graph(
        graph,
        [(0, 1)],
        edge_spacing=0.0,
        bin_size=1.0,
        track_width=2 * radius,
        units="cm",
        frame="lab",
    )


# The grid's cells are [0, 1] x [0, 1] and [1, 2] x [0, 1]. Every placement
# has bounds that intersect the grid's, so the exact shape test decides.
_ROUND_PLACEMENTS = {
    # The round piece covers part of a cell.
    "interior": ((1.6, 0.5), 0.3),
    # Its bounds reach past the cell corner (2, 1), but it stays 0.71 away.
    "outside": ((2.5, 1.5), 0.6),
    # It touches the corner (2, 1) at exactly its radius: 0.75² + 1² = 1.25².
    "tangent": ((2.75, 2.0), 1.25),
}


@pytest.mark.parametrize("kind", ["ball", "capsule"])
@pytest.mark.parametrize("placement", sorted(_ROUND_PLACEMENTS))
@pytest.mark.parametrize("grid_first", [True, False])
def test_round_support_against_polygon_cells_is_refused_only_on_overlap(
    kind, placement, grid_first
):
    center, radius = _ROUND_PLACEMENTS[placement]
    round_part = _round_component(kind, center, radius)
    components = (
        {"grid": grid(), "round": round_part}
        if grid_first
        else {"round": round_part, "grid": grid()}
    )
    if placement == "interior":
        with pytest.raises(ValueError, match=r"[Oo]verlap"):
            Environment.combine(components, auto_bridge=False)
        return
    combined = Environment.combine(components, auto_bridge=False)
    _maps_into_own_blocks(combined, list(components.values()))


def _unbounded_site_in(arena, center):
    """One nearest-bin site with no lookup radius, bounded by ``arena``."""
    centers = np.array([center], dtype=float)
    n_dims = centers.shape[1]
    return Environment.from_bins(
        Binning(
            "sites",
            SpatialBins.from_arrays(centers, [1.0], n_dims),
            BinConnections.from_arrays(np.empty((0, 2), dtype=np.intp), []),
            NearestBinLocator(centers),
            MissingFaces("Sites lack faces."),
        ),
        units="cm",
        frame="lab",
        arena=arena,
    )


@pytest.mark.parametrize(
    "center,overlaps",
    # A unit square turned 45 degrees is a diamond reaching 0.71 from its
    # center. At (2.6, 1.3) its bounds meet the cell [1, 2] x [0, 1], but its
    # lower-left edge, x + y = 3.19, passes outside the cell corner (2, 1).
    [((2.3, 1.0), True), ((2.6, 1.3), False)],
)
def test_rotated_polygon_arena_of_a_site_is_checked_against_grid_cells(
    center, overlaps
):
    from neurospatial.arena import OpenArena
    from neurospatial.ops.transforms import AffineND

    site = _unbounded_site_in(OpenArena(box(-0.5, -0.5, 0.5, 0.5)), [0.0, 0.0])
    angle = np.pi / 4
    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), center[0]],
            [np.sin(angle), np.cos(angle), center[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    turned = site.apply_transform(AffineND(matrix)).with_metadata(frame="lab")
    components = {"grid": grid(), "site": turned}
    if overlaps:
        with pytest.raises(ValueError, match=r"[Oo]verlap"):
            Environment.combine(components, auto_bridge=False)
        return
    combined = Environment.combine(components, auto_bridge=False)
    _maps_into_own_blocks(combined, list(components.values()))


@pytest.mark.parametrize(
    "center,overlaps",
    # A unit cube turned 45 degrees about z reaches 0.71 from its center in
    # x and y. At (2.6, 1.3, 0.5) its bounds meet the cell [1, 2] x [0, 1] x
    # [0, 1], but its face x + y = 3.19 passes outside the cell edge at
    # (2, 1).
    [((2.3, 1.0, 0.5), True), ((2.6, 1.3, 0.5), False)],
)
def test_rotated_3d_mask_arena_of_a_site_is_checked_against_grid_cells(
    center, overlaps
):
    from neurospatial.arena import MaskArena
    from neurospatial.ops.transforms import AffineND

    unit = np.array([-0.5, 0.5])
    site = _unbounded_site_in(
        MaskArena((unit, unit, unit), np.ones((1, 1, 1), dtype=bool)),
        [0.0, 0.0, 0.0],
    )
    angle = np.pi / 4
    matrix = np.eye(4)
    matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    matrix[:3, 3] = center
    turned = site.apply_transform(AffineND(matrix)).with_metadata(frame="lab")
    components = {"grid": grid(dims=3), "site": turned}
    if overlaps:
        with pytest.raises(ValueError, match=r"[Oo]verlap"):
            Environment.combine(components, auto_bridge=False)
        return
    combined = Environment.combine(components, auto_bridge=False)
    _maps_into_own_blocks(combined, list(components.values()))


def _bridged_cubes(conductance):
    """Two unit cubes joined by one bridge with the given conductance."""
    cubes = {
        key: Environment.from_grid_mask(
            np.ones((1, 1, 1), dtype=bool),
            grid_edges=(np.array([0.0, 1.0]) + shift, *[np.array([0.0, 1.0])] * 2),
        )
        for key, shift in (("a", 0.0), ("b", 2.0))
    }
    return Environment.combine(
        cubes, bridge_conductances={(("a", 0), ("b", 0)): conductance}
    )


@pytest.mark.parametrize("conductance,scale", [(1e-300, 1e-30), (1e300, 1e30)])
def test_scaling_refuses_a_bridge_conductance_it_cannot_represent(conductance, scale):
    """A volume bridge's conductance scales with length; 0 or inf is refused.

    Rounding the conductance to zero would silently decouple the components,
    and an infinite one would join them without resistance.
    """
    from neurospatial.ops.transforms import scale_3d

    env = _bridged_cubes(conductance)
    with pytest.raises(
        ValueError, match="Transformed bridge conductance exceeds floating-point range"
    ):
        env.apply_transform(scale_3d(scale), units=env.units)


def test_scaling_keeps_a_small_representable_bridge_conductance():
    from neurospatial.ops.transforms import scale_3d

    env = _bridged_cubes(1e-300)
    scaled = env.apply_transform(scale_3d(1e-10), units=env.units)
    np.testing.assert_allclose(
        scaled.get_bridge_attributes()["conductance"], [1e-310], rtol=1e-6
    )


def test_scaling_refuses_bin_volumes_past_floating_point_range():
    from neurospatial.ops.transforms import scale_3d

    env = _bridged_cubes(1.0)
    with pytest.raises(
        ValueError, match="Transformed bin measures exceed floating-point range"
    ):
        env.apply_transform(scale_3d(1e110), units=env.units)
