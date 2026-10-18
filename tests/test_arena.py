"""Continuous arena data retain physical meaning, ownership and transport."""

from __future__ import annotations

import copy
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from neurospatial.arena import (
    EgocentricSurround,
    MaskArena,
    OpenArena,
    TrackArena,
    UnknownArena,
)
from neurospatial.geometry._codecs import decode_arena, encode_arena


def arenas():
    """Small deterministic examples exercise every stored arena kind."""
    return [
        UnknownArena(2),
        UnknownArena(2, coordinate_kind="polar"),
        OpenArena(box(0, 0, 2, 2), walls=(LineString([(1, 0), (1, 2)]),)),
        OpenArena(
            MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)]),
            physical_footprint_known=False,
        ),
        MaskArena((np.arange(3.0),) * 2, np.array([[True, False], [False, True]])),
        TrackArena(np.array([[[0.0, 0.0], [4.0, 0.0]]]), width=2.0),
        TrackArena(np.array([[[0.0, 0.0], [4.0, 0.0]]]), width=None),
        EgocentricSurround((0.0, 5.0), (-np.pi, np.pi), circular=True),
    ]


@pytest.mark.parametrize("arena", arenas())
def test_arena_roundtrip_identity_and_owned_payload(arena):
    meta, arrays = encode_arena(arena)
    restored = decode_arena(meta, arrays)
    assert restored is not arena and restored != arena
    assert restored.n_dims == arena.n_dims
    assert restored.coordinate_kind == arena.coordinate_kind
    assert restored.domain_known == arena.domain_known
    assert restored.physical_footprint_known == arena.physical_footprint_known
    if arena.domain_known:
        points = np.array([[0.0, 0.0], [0.5, 0.5], [1.5, 1.5], [20.0, 20.0]])
        np.testing.assert_array_equal(restored.contains(points), arena.contains(points))
    for array in arrays.values():
        assert array.flags.writeable
        array[...] = 0
    if arena.domain_known:
        np.testing.assert_array_equal(restored.contains(points), arena.contains(points))
    with pytest.raises(FrozenInstanceError):
        restored.n_dims = 7


def test_open_arena_hole_boundary_multipolygon_and_walls():
    polygon = Polygon(
        box(0, 0, 4, 4).exterior.coords, holes=[box(1, 1, 3, 3).exterior.coords]
    )
    wall = LineString([(0.5, 0), (0.5, 4)])
    arena = OpenArena(polygon, walls=[wall])
    np.testing.assert_array_equal(
        arena.contains(
            [[0, 0], [0.5, 0.5], [2, 2], [1, 2], [4, 2], [5, 2], [np.nan, 1]]
        ),
        [True, True, False, True, True, False, False],
    )
    assert arena.walls == (wall,)
    assert arena.polygonal_domain().equals(polygon)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: UnknownArena(2),
        lambda: TrackArena(np.array([[[0.0, 0.0], [1.0, 0.0]]]), width=None),
    ],
)
def test_unknown_footprint_refuses_physical_containment(constructor):
    arena = constructor()
    assert not arena.domain_known and not arena.physical_footprint_known
    assert arena.polygonal_domain() is None
    with pytest.raises(NotImplementedError, match="unknown"):
        arena.contains([[0.0, 0.0]])


def test_mask_domain_is_closed_cell_union_not_half_open_lookup():
    arena = MaskArena((np.arange(3.0),) * 2, np.array([[True, False], [False, True]]))
    np.testing.assert_array_equal(
        arena.contains(
            [[0, 0], [1, 0.5], [0.5, 1.5], [1, 1], [2, 2], [2, 0.5], [np.inf, 0]]
        ),
        [True, True, False, True, True, False, False],
    )
    assert arena.domain_known and not arena.physical_footprint_known
    assert arena.walls == ()
    assert arena.polygonal_domain().area == 2.0


def test_nd_mask_owns_input_and_uses_real_nonuniform_edges():
    edge = np.array([0.0, 1.0, 3.0])
    mask = np.array([True, False])
    arena = MaskArena((edge,), mask, physical_footprint_known=True)
    edge[:] = 20
    mask[:] = False
    np.testing.assert_array_equal(
        arena.contains([[0.0], [0.5], [1.0], [1.1], [3.0]]),
        [True, True, True, False, False],
    )
    assert arena.polygonal_domain() is None
    assert arena.physical_footprint_known


def test_track_uses_exact_segments_and_round_endcaps():
    segments = np.array([[[0.0, 0.0], [4.0, 0.0]], [[4.0, 0.0], [4.0, 4.0]]])
    arena = TrackArena(segments, width=2.0)
    segments[:] = 100
    np.testing.assert_array_equal(
        arena.contains(
            [
                [-1, 0],
                [-1, 0.01],
                [2, 1],
                [2, 1.01],
                [5, 2],
                [4, 5],
                [100, 100],
                [np.nan, 0],
            ]
        ),
        [True, False, True, False, True, True, False, False],
    )
    assert arena.polygonal_domain() is None  # A polygonal buffer is an approximation.


def test_egocentric_windows_periodicity_and_analysis_boundary():
    arena = EgocentricSurround((0.0, 5.0), (-np.pi / 2, np.pi / 2), circular=True)
    np.testing.assert_array_equal(
        arena.contains(
            [[0, 0], [5, np.pi / 2], [2, 2 * np.pi], [2, np.pi], [-1, 0], [6, 0]]
        ),
        [True, True, True, False, False, False],
    )
    assert arena.coordinate_kind == "polar" and not arena.physical_footprint_known
    assert arena.walls == () and arena.polygonal_domain() is None


@pytest.mark.parametrize("arena", [a for a in arenas() if a.domain_known])
def test_known_arena_query_shape_and_nonfinite_rows(arena):
    assert arena.contains(np.empty((0, 2))).shape == (0,)
    np.testing.assert_array_equal(
        arena.contains([[np.nan, 0], [0, np.inf]]), [False, False]
    )
    for points in [np.empty((0, 3)), np.full((1, 3), np.nan), np.zeros((1, 2, 1))]:
        with pytest.raises(ValueError, match=r"shape|dimension"):
            arena.contains(points)


@pytest.mark.parametrize(
    "operation", [copy.deepcopy, lambda arena: pickle.loads(pickle.dumps(arena))]
)
@pytest.mark.parametrize(
    "arena", [a for a in arenas() if isinstance(a, (MaskArena, TrackArena))]
)
def test_roundtrip_arrays_stay_owned_and_readonly(arena, operation):
    restored = operation(arena)
    arrays = [
        *getattr(restored, "edges", ()),
        getattr(restored, "active_mask", None),
        getattr(restored, "segments", None),
    ]
    for array in arrays:
        if array is not None:
            assert array.flags.owndata and not array.flags.writeable


def test_affine_maps_domain_walls_and_track_width():
    matrix = np.array([[0.0, -2.0, 10.0], [2.0, 0.0, -3.0], [0.0, 0.0, 1.0]])
    points = np.array([[0.5, 0.5], [1.5, 1.5], [5.0, 5.0]])
    mapped = points @ matrix[:2, :2].T + matrix[:2, -1]
    for arena in [arenas()[2], arenas()[4], arenas()[5]]:
        derived = arena.transformed(matrix)
        np.testing.assert_array_equal(derived.contains(mapped), arena.contains(points))
        restored = decode_arena(*encode_arena(derived))
        np.testing.assert_array_equal(restored.contains(mapped), arena.contains(points))
        assert derived.physical_footprint_known == arena.physical_footprint_known
    np.testing.assert_array_equal(
        arenas()[5].transformed(matrix).contains([[10.0, 7.0], [10.0, 7.01]]),
        [True, False],
    )


@pytest.mark.parametrize("angle", [0.3, 0.7])
@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_transformed_closed_boundaries_use_bounded_arithmetic_uncertainty(
    angle, offset
):
    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), offset],
            [np.sin(angle), np.cos(angle), -offset],
            [0.0, 0.0, 1.0],
        ]
    )
    examples = [
        (
            OpenArena(box(0, 0, 2, 2)),
            np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 1.0], [0.0, 0.0]]),
            np.array([[-1e-3, 1.0], [1.0, 2.001]]),
        ),
        (
            MaskArena((np.arange(3.0),) * 2, np.ones((2, 2), bool)),
            np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 1.0], [0.0, 0.0]]),
            np.array([[-1e-3, 1.0], [1.0, 2.001]]),
        ),
        (
            TrackArena(np.array([[[0.0, 0.0], [10.0, 0.0]]]), width=2.0),
            np.array(
                [
                    [0.0, 1.0],
                    [5.0, 1.0],
                    [10.0, 1.0],
                    [-1.0, 0.0],
                    [11.0, 0.0],
                    [0.0, -1.0],
                ]
            ),
            np.array([[5.0, 1.001], [11.001, 0.0]]),
        ),
    ]
    for arena, boundary, outside in examples:
        transformed = arena.transformed(matrix)
        for restored in [transformed, decode_arena(*encode_arena(transformed))]:
            np.testing.assert_array_equal(
                restored.contains(boundary @ matrix[:2, :2].T + matrix[:2, -1]), True
            )
            np.testing.assert_array_equal(
                restored.contains(outside @ matrix[:2, :2].T + matrix[:2, -1]), False
            )


@pytest.mark.parametrize("holes", [False, True])
def test_transformed_polygon_rings_remain_closed_after_inverse(holes):
    import shapely

    polygon = box(0, 0, 1, 1)
    if holes:
        polygon = polygon.difference(box(0.1, 0.1, 0.2, 0.2)).difference(
            box(0.1, 0.7, 0.2, 0.8)
        )
    arena = OpenArena(polygon)
    angle = 0.3
    matrix = np.array(
        [
            [-np.cos(angle), -np.sin(angle), 10.0],
            [-np.sin(angle), np.cos(angle), -7.0],
            [0.0, 0.0, 1.0],
        ]
    )
    derived = arena.transformed(matrix).transformed(np.linalg.inv(matrix))
    points = np.array([[0.5, 0.5], [0.15, 0.15], [0.15, 0.75], [2.0, 2.0]])
    for restored in (derived, decode_arena(*encode_arena(derived))):
        domain = restored.polygonal_domain()
        assert domain.is_valid
        assert len(domain.interiors) == len(polygon.interiors)
        for ring in shapely.get_rings(domain):
            vertices = shapely.get_coordinates(ring)
            np.testing.assert_array_equal(vertices[0], vertices[-1])
        np.testing.assert_array_equal(restored.contains(points), arena.contains(points))


def test_reflected_mask_and_nested_nd_frame_preserve_domain():
    arena = MaskArena(
        (np.array([0.0, 1.0, 3.0]), np.arange(3.0)),
        np.array([[True, False], [False, True]]),
    )
    reflected = arena.transformed(np.diag([-1.0, 2.0, 1.0]))
    np.testing.assert_array_equal(
        reflected.contains([[-0.5, 1.0], [-2.0, 3.0], [-2.0, 1.0]]), [True, True, False]
    )
    base = MaskArena((np.arange(3.0),) * 3, np.ones((2, 2, 2), bool))
    matrix = np.eye(4)
    matrix[0, 1] = 0.3
    derived = base.transformed(matrix).transformed(matrix)
    point = np.array([[0.5, 0.5, 0.5]])
    mapped = point @ (matrix @ matrix)[:3, :3].T
    for restored in [
        derived,
        copy.deepcopy(derived),
        pickle.loads(pickle.dumps(derived)),
        decode_arena(*encode_arena(derived)),
    ]:
        np.testing.assert_array_equal(restored.contains(mapped), [True])
        assert not restored.forward.flags.writeable
        assert not restored.inverse.flags.writeable
        assert restored.polygonal_domain() is None
    np.testing.assert_array_equal(
        derived.contains([[np.finfo(float).max, -np.finfo(float).max, 0.0]]), [False]
    )


def test_transform_refuses_unrepresentable_geometry_without_runtime_warnings():
    with pytest.raises(ValueError):
        OpenArena(box(0, 0, 2, 2)).transformed(np.diag([np.finfo(float).max, 1.0, 1.0]))
    with pytest.raises(ValueError):
        MaskArena((np.arange(3.0),), np.ones(2, bool)).transformed(
            np.diag([np.finfo(float).max, 1.0])
        )


@pytest.mark.parametrize(
    "matrix",
    [
        np.diag([2.0, 1.0, 1.0]),
        np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_track_rejects_non_similarity(matrix):
    with pytest.raises(ValueError, match="similarity"):
        arenas()[5].transformed(matrix)


def test_identity_like_rotation_does_not_refuse_harmless_underflow():
    angle = 4.144551617385651e-260
    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    with np.errstate(all="raise"):
        result = arenas()[5].transformed(matrix)
        np.testing.assert_array_equal(result.contains([[1.0, 0.0]]), [True])


def test_as_track_maps_centerline_and_scales_width():
    junction = TrackArena(
        np.array([[[0.0, 0.0], [4.0, 0.0]], [[4.0, 0.0], [4.0, 3.0]]]), width=2.0
    )
    angle = 0.3
    matrix = np.array(
        [
            [2 * np.cos(angle), -2 * np.sin(angle), 7.0],
            [2 * np.sin(angle), 2 * np.cos(angle), -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    twice = junction.transformed(matrix).transformed(matrix)
    track = twice.as_track()
    assert isinstance(track, TrackArena)
    np.testing.assert_allclose(track.width, 8.0, rtol=1e-15)
    full = matrix @ matrix
    np.testing.assert_allclose(
        track.segments,
        junction.segments @ full[:2, :2].T + full[:2, -1],
        rtol=0,
        atol=1e-13,
    )
    # The shared junction vertex stays one vertex in the new frame.
    np.testing.assert_array_equal(track.segments[0, 1], track.segments[1, 0])
    # Off the boundary, the mapped track and the transformed arena agree.
    rng = np.random.default_rng(0)
    points = rng.uniform(-20, 40, size=(500, 2))
    np.testing.assert_array_equal(track.contains(points), twice.contains(points))
    assert junction.as_track() is junction
    assert (
        TrackArena(junction.segments, None).transformed(matrix).as_track().width is None
    )
    for other in (arenas()[2], arenas()[4].transformed(matrix), arenas()[0]):
        assert other.as_track() is None


@pytest.mark.parametrize("gap", [1e-5, 1e-6])
def test_transformed_track_keeps_resolvable_collective_hole(gap):
    height = np.sqrt(3.0)
    arena = TrackArena(
        np.array(
            [
                [[0.0, 0.0], [2.0, 0.0]],
                [[2.0, 0.0], [1.0, height]],
                [[1.0, height], [0.0, 0.0]],
            ]
        ),
        width=2 * (height / 3 - gap),
    )
    point = np.array([[1.0, height / 3]])
    matrix = np.array(
        [
            [np.cos(0.7), -np.sin(0.7), 1e9],
            [np.sin(0.7), np.cos(0.7), -1e9],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(arena.contains(point), [False])
    np.testing.assert_array_equal(
        arena.transformed(matrix).contains(point @ matrix[:2, :2].T + matrix[:2, -1]),
        [False],
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((3, 3)),
        np.diag([1.0, np.nan, 1.0]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.0, 1.0]]),
    ],
)
def test_arena_refuses_invalid_affine_matrix(matrix):
    with pytest.raises(ValueError):
        arenas()[2].transformed(matrix)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: UnknownArena(True),
        lambda: UnknownArena(0),
        lambda: OpenArena(Polygon()),
        lambda: OpenArena(Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])),
        lambda: MaskArena((np.array([0.0, 2.0, 1.0]),), np.ones(2, bool)),
        lambda: MaskArena((np.arange(3.0),), np.ones(2, int)),
        lambda: MaskArena((np.arange(3.0),), np.ones(3, bool)),
        lambda: TrackArena(np.zeros((1, 2, 2)), width=1.0),
        lambda: TrackArena(np.array([[[0.0, 0.0], [1.0, 0.0]]]), width=0.0),
        lambda: EgocentricSurround((-1.0, 2.0), (-np.pi, np.pi), circular=True),
    ],
)
def test_arena_rejects_invalid_geometry(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


def test_codec_rejects_unknown_versions_missing_arrays_and_wrong_dtypes():
    meta, arrays = encode_arena(arenas()[4])
    for patch in [
        {"codec": "ExternalArena"},
        {"codec_version": 1},
        {"codec_version": 3},
        {"physical_footprint_known": "false"},
    ]:
        with pytest.raises(ValueError):
            decode_arena({**meta, **patch}, arrays)
    for broken in [{}, {**arrays, "active_mask": np.ones((2, 2), dtype=np.int64)}]:
        with pytest.raises(ValueError):
            decode_arena(meta, broken)
    polygon_meta, polygon_arrays = encode_arena(arenas()[2])
    with pytest.raises(ValueError):
        decode_arena(
            polygon_meta,
            {**polygon_arrays, "polygon_wkb": np.array([1, 2, 3], np.uint8)},
        )


@pytest.mark.parametrize(
    ("arena", "expected"),
    [
        (OpenArena(box(1, 2, 4, 3)), [[1, 2], [4, 3]]),
        (
            MaskArena(
                (np.arange(4.0), np.arange(3.0), np.array([0.0, 0.5, 2.0])),
                np.pad(np.ones((1, 1, 1), bool), ((1, 1), (0, 1), (1, 0))),
            ),
            [[1, 0, 0.5], [2, 1, 2]],
        ),
        (
            TrackArena(np.array([[[0.0, 0.0], [4.0, 1.0]]]), width=2.0),
            [[-1, -1], [5, 2]],
        ),
        (TrackArena(np.array([[[0.0, 0.0], [4.0, 0.0]]]), width=None), None),
        (UnknownArena(2), None),
        (EgocentricSurround((0.0, 5.0), (-np.pi, np.pi), circular=True), None),
    ],
)
def test_domain_bounds_enclose_known_cartesian_domain(arena, expected):
    bounds = arena.domain_bounds()
    if expected is None:
        assert bounds is None
    else:
        np.testing.assert_array_equal(bounds, expected)


def test_transformed_domain_bounds_enclose_mapped_box():
    angle = np.pi / 4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 10.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mask = MaskArena((np.array([0.0, 2.0]),) * 2, np.ones((1, 1), bool))
    bounds = mask.transformed(rotation).domain_bounds()
    half_diagonal = np.sqrt(2.0)
    np.testing.assert_allclose(
        bounds, [[10 - half_diagonal, 0.0], [10 + half_diagonal, 2 * half_diagonal]]
    )
    # Every point of the mapped domain lies inside the box.
    corners = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], float)
    mapped = corners @ rotation[:2, :2].T + rotation[:2, 2]
    assert np.all((mapped >= bounds[0] - 1e-12) & (mapped <= bounds[1] + 1e-12))
