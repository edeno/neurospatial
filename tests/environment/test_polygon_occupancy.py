"""Analytical time allocation through polygon domains and physical walls."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from neurospatial import Environment
from neurospatial.arena import MaskArena, OpenArena


def _one_cell_environment():
    base = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool),
        grid_edges=(np.array([0.0, 6.0]), np.array([0.0, 2.0])),
    )
    domain = Polygon(
        box(0, 0, 6, 2).exterior.coords,
        holes=[box(2, 0.5, 4, 1.5).exterior.coords],
    )
    return base.clip(domain)


def _wall_environment():
    return Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.arange(3.0), np.array([0.0, 2.0])),
        arena=OpenArena(box(0, 0, 2, 2), walls=(LineString([(1, 0), (1, 2)]),)),
    )


@pytest.mark.parametrize(
    ("positions", "duration", "expected"),
    [
        ([[1, 1], [5, 1]], 4.0, 2.0),
        ([[-1, 1], [1, 1]], 2.0, 1.0),
        ([[-1, 1], [7, 1]], 8.0, 4.0),
        ([[1, 1], [1, 1]], 2.0, 2.0),
        ([[3, 1], [3, 1]], 2.0, 0.0),
    ],
    ids=["same-bin-hole", "outside-entry", "entry-hole-exit", "stationary", "outside"],
)
def test_polygon_linear_occupancy_exact_duration(positions, duration, expected):
    env = _one_cell_environment()
    result = env.occupancy(
        np.array([0.0, duration]),
        np.asarray(positions, dtype=float),
        max_gap=None,
        time_allocation="linear",
    )
    np.testing.assert_allclose(result, [expected], rtol=0, atol=1e-14)


def test_polygon_fractional_counts_do_not_renormalize_retained_support():
    env = _one_cell_environment()
    result = env.occupancy(
        np.array([0.0, 8.0]),
        np.array([[-1.0, 1.0], [7.0, 1.0]]),
        max_gap=None,
        time_allocation="linear",
        return_seconds=False,
    )
    np.testing.assert_allclose(result, [0.5], rtol=0, atol=1e-14)


@pytest.mark.parametrize("gate", ["gap", "speed", "epochs"])
def test_polygon_observation_filters_precede_geometric_subdivision(gate):
    env = _one_cell_environment()
    kwargs = {"max_gap": None}
    if gate == "gap":
        kwargs["max_gap"] = 1.0
    elif gate == "speed":
        kwargs.update(speed=np.array([0.0, 1.0]), min_speed=0.5)
    else:
        kwargs["epochs"] = (0.0, 1.0)
    result = env.occupancy(
        np.array([0.0, 2.0]),
        np.array([[-1.0, 1.0], [1.0, 1.0]]),
        time_allocation="linear",
        **kwargs,
    )
    np.testing.assert_array_equal(result, [0.0])


def test_linear_polygon_wall_crossing_refused():
    env = _wall_environment()
    with pytest.raises(ValueError, match="wall"):
        env.occupancy(
            np.array([0.0, 2.0]),
            np.array([[0.5, 0.25], [1.5, 0.25]]),
            max_gap=None,
            time_allocation="linear",
        )


@pytest.mark.parametrize("operation", ["occupancy", "encoding"])
def test_crossing_a_joint_in_a_continuous_wall_is_refused(operation):
    from neurospatial import compute_spatial_rates

    env = Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.arange(3.0), np.array([0.0, 2.0])),
        arena=OpenArena(
            box(0, 0, 2, 2),
            walls=(LineString([(1, 0), (1, 1)]), LineString([(1, 1), (1, 2)])),
        ),
    )
    times = np.array([0.0, 1.0])
    positions = np.array([[0.5, 0.5], [1.5, 1.5]])
    with pytest.raises(ValueError, match="wall"):
        if operation == "occupancy":
            env.occupancy(times, positions, max_gap=None, time_allocation="linear")
        else:
            compute_spatial_rates(
                env,
                [np.array([])],
                times,
                positions,
                method="binned",
                bandwidth=0,
                max_gap=None,
                time_allocation="linear",
            )


@pytest.mark.parametrize("operation", ["occupancy", "encoding"])
@pytest.mark.parametrize(
    "junction", ["T", "split_T", "X", "split_X", "bent", "overlap"]
)
def test_wall_junction_crossings_do_not_depend_on_line_partition(operation, junction):
    from neurospatial import compute_spatial_rates

    vertical = [LineString([(2, 0), (2, 4)])]
    if junction.startswith("split"):
        vertical = [LineString([(2, 0), (2, 2)]), LineString([(2, 2), (2, 4)])]
    horizontal = [LineString([(2, 2), (4, 2)])]
    if junction == "X":
        horizontal = [LineString([(0, 2), (4, 2)])]
    elif junction == "split_X":
        horizontal = [LineString([(0, 2), (2, 2)]), *horizontal]
    positions = np.array([[0.5, 0.5], [3.5, 3.5]])
    if junction == "bent":
        vertical = [LineString([(2, 2), (2, 4)])]
    elif junction == "overlap":
        vertical = [LineString([(3, 0), (3, 4)])]
        horizontal = [LineString([(0, 2), (1, 2)])]
        positions = np.array([[0.5, 2.0], [3.5, 2.0]])
    env = Environment.from_grid_mask(
        np.ones((4, 4), bool),
        grid_edges=(np.arange(5.0),) * 2,
        arena=OpenArena(box(0, 0, 4, 4), walls=(*vertical, *horizontal)),
    )
    with pytest.raises(ValueError, match="wall"):
        if operation == "occupancy":
            env.occupancy(
                np.array([0.0, 1.0]), positions, time_allocation="linear", max_gap=None
            )
        else:
            compute_spatial_rates(
                env,
                [np.array([])],
                np.array([0.0, 1.0]),
                positions,
                method="binned",
                bandwidth=0,
                time_allocation="linear",
                max_gap=None,
            )


@pytest.mark.parametrize("contact", ["tip", "end_at_joint", "along", "graze_bend"])
def test_wall_junction_contact_without_a_crossing_remains_allowed(contact):
    walls = (LineString([(2, 0), (2, 4)]), LineString([(2, 2), (4, 2)]))
    if contact == "tip":
        walls = (LineString([(2, 0), (2, 2)]),)
        positions = [[0.5, 0.5], [3.5, 3.5]]
    elif contact == "end_at_joint":
        positions = [[0.5, 0.5], [2.0, 2.0]]
    elif contact == "along":
        positions = [[2.0, 0.5], [2.0, 3.5]]
    else:
        walls = (LineString([(2, 2), (2, 4)]), LineString([(2, 2), (4, 2)]))
        positions = [[0.5, 3.5], [3.5, 0.5]]
    env = Environment.from_grid_mask(
        np.ones((4, 4), bool),
        grid_edges=(np.arange(5.0),) * 2,
        arena=OpenArena(box(0, 0, 4, 4), walls=walls),
    )
    result = env.occupancy(
        np.array([0.0, 1.0]),
        np.array(positions),
        time_allocation="linear",
        max_gap=None,
    )
    np.testing.assert_allclose(result.sum(), 1.0)


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        ([[0.5, 0.25], [1.0, 0.25]], [2.0, 0.0]),
        ([[1.0, 0.25], [1.0, 1.75]], [0.0, 2.0]),
    ],
)
def test_polygon_wall_endpoint_and_collinear_contact_allowed(positions, expected):
    env = _wall_environment()
    result = env.occupancy(
        np.array([0.0, 2.0]),
        np.asarray(positions),
        max_gap=None,
        time_allocation="linear",
    )
    np.testing.assert_allclose(result, expected)


def test_default_start_allocation_keeps_existing_interval_contract():
    env = _one_cell_environment()
    result = env.occupancy(
        np.array([0.0, 4.0]),
        np.array([[1.0, 1.0], [5.0, 1.0]]),
        max_gap=None,
    )
    np.testing.assert_array_equal(result, [4.0])


def test_polygon_shared_edge_uses_single_bin_owner():
    base = Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.arange(3.0), np.array([0.0, 1.0])),
    )
    env = Environment(
        base.name,
        base.layout,
        base.layout_type,
        base.layout_parameters,
        arena=OpenArena(box(0, 0, 2, 1)),
    )
    owner = env.bin_at(np.array([[1.0, 0.5]]))[0]
    result = env.occupancy(
        np.array([0.0, 2.0]),
        np.array([[1.0, 0.25], [1.0, 0.75]]),
        max_gap=None,
        time_allocation="linear",
    )
    expected = np.zeros(2)
    expected[owner] = 2.0
    np.testing.assert_array_equal(result, expected)


def test_polygon_nonfinite_observation_drops_both_adjacent_intervals():
    env = _one_cell_environment()
    result = env.occupancy(
        np.arange(0.0, 10.0, 2.0),
        np.array([[-1, 1], [1, 1], [np.nan, np.nan], [5, 1], [7, 1]]),
        max_gap=None,
        time_allocation="linear",
    )
    np.testing.assert_allclose(result, [2.0], rtol=0, atol=1e-14)


def test_retained_intervals_keep_original_fractional_clock():
    from neurospatial.environment._polygon_trajectory import retained_linear_intervals

    env = _one_cell_environment()
    intervals = retained_linear_intervals(
        env,
        np.array([0.0, 4.0]),
        np.array([[1.0, 1.0], [5.0, 1.0]]),
        np.array([True]),
    )
    assert [
        (piece.sample_index, piece.start_fraction, piece.stop_fraction, piece.bin_index)
        for piece in intervals
    ] == [(0, 0.0, 0.25, 0), (0, 0.75, 1.0, 0)]


@pytest.mark.parametrize("derivation", ["clip", "subset", "transform", "chain"])
def test_clipped_cell_occupancy_survives_nested_derivations_and_transport(
    round_trip_environment, derivation
):
    from neurospatial.ops.transforms import Affine2D

    env = _one_cell_environment().clip(MultiPolygon([box(0, 0, 2, 2), box(4, 0, 6, 2)]))
    positions = np.array([[-1.0, 1.0], [7.0, 1.0]])
    expected = [2.0, 2.0]
    if derivation in {"subset", "chain"}:
        env = env.subset(bins=np.array([True, False]))
        expected = [2.0]
    if derivation in {"transform", "chain"}:
        matrix = np.array([[1.0, 0.5, 10.0], [0.0, 2.0, -3.0], [0.0, 0.0, 1.0]])
        env = env.apply_transform(Affine2D(matrix))
        positions = positions @ matrix[:2, :2].T + matrix[:2, 2]
    loaded = round_trip_environment(env)
    result = loaded.occupancy(
        np.array([0.0, 8.0]), positions, max_gap=None, time_allocation="linear"
    )
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("known_polygon", [False, True])
def test_rotated_native_cells_allocate_without_rectilinear_capability(known_polygon):
    from neurospatial.ops.transforms import Affine2D

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    if known_polygon:
        env = Environment(
            env.name,
            env.layout,
            env.layout_type,
            env.layout_parameters,
            arena=OpenArena(box(0, 0, 2, 2)),
        )
    matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    env = env.apply_transform(Affine2D(matrix))
    assert env._binning.rectilinear is None
    result = env.occupancy(
        np.array([0.0, 1.0]),
        np.array([[-0.5, 0.5], [-1.5, 1.5]]),
        max_gap=None,
        time_allocation="linear",
    )
    np.testing.assert_allclose(result, [0.5, 0.0, 0.0, 0.5], rtol=0, atol=1e-14)


def test_explicit_mask_domain_subdivides_native_grid_segment():
    arena = MaskArena(
        (np.array([0.5, 1.5]), np.array([0.0, 1.0])), np.ones((1, 1), dtype=bool)
    )
    env = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool),
        grid_edges=(np.array([0.0, 2.0]), np.array([0.0, 1.0])),
        arena=arena,
    )
    result = env.occupancy(
        np.array([0.0, 1.0]),
        np.array([[0.75, 0.5], [1.75, 0.5]]),
        max_gap=None,
        time_allocation="linear",
    )
    np.testing.assert_array_equal(result, [0.75])


@pytest.mark.parametrize("subset", [False, True])
def test_matching_native_nd_mask_preserves_linear_allocation(subset):
    env = Environment.from_grid_mask(
        np.ones((2, 2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 3
    )
    expected = np.zeros(8)
    expected[[0, 7]] = 0.5
    if subset:
        env = env.subset(bins=np.array([True] + [False] * 6 + [True]))
        expected = expected[[0, 7]]
    result = env.occupancy(
        np.array([0.0, 1.0]),
        np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]),
        max_gap=None,
        time_allocation="linear",
    )
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-16)


def test_restricted_nd_mask_refuses_unimplemented_linear_subdivision():
    arena = MaskArena(
        (np.array([0.5, 1.5]), np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        np.ones((1, 1, 1), dtype=bool),
    )
    env = Environment.from_grid_mask(
        np.ones((1, 1, 1), dtype=bool),
        grid_edges=(np.array([0.0, 2.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        arena=arena,
    )
    with pytest.raises(NotImplementedError, match="restricted MaskArena"):
        env.occupancy(
            np.array([0.0, 1.0]),
            np.array([[0.75, 0.5, 0.5], [1.75, 0.5, 0.5]]),
            max_gap=None,
            time_allocation="linear",
        )


@pytest.mark.parametrize("polygon_cells", [False, True])
def test_analytic_domain_without_polygon_refuses_linear_approximation(polygon_cells):
    class UnitDiskArena:
        n_dims = 2
        coordinate_kind = "cartesian"
        domain_known = True
        physical_footprint_known = True
        walls = ()

        def contains(self, positions):
            return np.linalg.norm(positions, axis=1) <= 1

        def polygonal_domain(self):
            return None

        def transformed(self, matrix):
            raise NotImplementedError(
                "This analytical provider has no affine transform"
            )

    base = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool), grid_edges=(np.array([-1.0, 1.0]),) * 2
    )
    if polygon_cells:
        base = base.clip(box(-1, -1, 1, 1))
    env = Environment(
        base.name,
        base.layout,
        base.layout_type,
        base.layout_parameters,
        arena=UnitDiskArena(),
    )
    positions = np.array([[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_array_equal(env.bin_at(positions), [0, -1])
    with pytest.raises(NotImplementedError, match="polygon support"):
        env.occupancy(
            np.array([0.0, 1.0]),
            positions,
            max_gap=None,
            time_allocation="linear",
        )


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("operation", ["occupancy", "encoding"])
def test_sample_exactly_on_wall_does_not_split_a_physical_crossing(
    duplicate, operation
):
    from neurospatial import compute_spatial_rates

    env = _wall_environment()
    positions = [[0.5, 0.75], [1.0, 0.75]]
    if duplicate:
        positions.append([1.0, 0.75])
    positions.append([1.5, 0.75])
    positions = np.asarray(positions)
    times = np.arange(len(positions), dtype=float)
    with pytest.raises(ValueError, match="wall"):
        if operation == "occupancy":
            env.occupancy(times, positions, time_allocation="linear", max_gap=None)
        else:
            compute_spatial_rates(
                env,
                [np.array([0.25])],
                times,
                positions,
                method="binned",
                bandwidth=0.0,
                time_allocation="linear",
                max_gap=None,
            )


@pytest.mark.parametrize("junction", ["straight", "split", "bend", "T", "X"])
def test_sample_at_a_wall_junction_does_not_open_a_route(junction):
    walls = [LineString([(2, 0), (2, 4)])]
    if junction == "split":
        walls = [LineString([(2, 0), (2, 2)]), LineString([(2, 2), (2, 4)])]
    elif junction == "bend":
        walls = [LineString([(2, 2), (2, 4)]), LineString([(2, 2), (4, 2)])]
    elif junction == "T":
        walls.append(LineString([(2, 2), (4, 2)]))
    elif junction == "X":
        walls.append(LineString([(0, 2), (4, 2)]))
    env = Environment.from_grid_mask(
        np.ones((4, 4), bool),
        grid_edges=(np.arange(5.0),) * 2,
        arena=OpenArena(box(0, 0, 4, 4), walls=tuple(walls)),
    )
    # The bend separates its small northeast sector from the large southwest one.
    positions = np.array([[0.5, 0.5], [2.0, 2.0], [3.5, 3.5]])
    with pytest.raises(ValueError, match="wall"):
        env.occupancy(np.arange(3.0), positions, time_allocation="linear", max_gap=None)


@pytest.mark.parametrize(
    "positions",
    [
        [[0.5, 0.5], [1.0, 0.75], [0.5, 1.5]],  # touch and turn back
        [[0.5, 0.5], [1.0, 0.75], [1.0, 0.75], [0.5, 1.5]],
        [[1.0, 0.25], [1.0, 0.75], [1.0, 1.75]],  # collinear wall motion
    ],
)
def test_sampled_wall_contact_without_a_crossing_remains_allowed(positions):
    env = _wall_environment()
    times = np.arange(len(positions), dtype=float)
    occupancy = env.occupancy(
        times, np.asarray(positions), time_allocation="linear", max_gap=None
    )
    assert occupancy.sum() == pytest.approx(times[-1])


@pytest.mark.parametrize("contact", ["tip", "graze_bend"])
def test_a_sample_at_a_wall_tip_or_grazing_corner_remains_allowed(contact):
    walls = (LineString([(2, 0), (2, 2)]),)
    positions = np.array([[0.5, 0.5], [2.0, 2.0], [3.5, 3.5]])
    if contact == "graze_bend":
        walls = (LineString([(2, 2), (2, 4)]), LineString([(2, 2), (4, 2)]))
        positions = np.array([[0.5, 3.5], [2.0, 2.0], [3.5, 0.5]])
    env = Environment.from_grid_mask(
        np.ones((4, 4), bool),
        grid_edges=(np.arange(5.0),) * 2,
        arena=OpenArena(box(0, 0, 4, 4), walls=walls),
    )
    occupancy = env.occupancy(
        np.arange(3.0), positions, time_allocation="linear", max_gap=None
    )
    assert occupancy.sum() == pytest.approx(2.0)


@pytest.mark.parametrize("excluded", ["gap", "epoch", "nonfinite"])
def test_sampled_wall_crossing_check_does_not_join_filtered_intervals(excluded):
    env = _wall_environment()
    positions = np.array([[0.5, 0.75], [1.0, 0.75], [1.5, 0.75]])
    times = np.arange(3.0)
    kwargs = {"max_gap": None}
    if excluded == "gap":
        times[-1] = 10.0
        kwargs = {"max_gap": 2.0}
    elif excluded == "epoch":
        kwargs["epochs"] = np.array([[1.0, 2.0]])
    else:
        positions[0] = np.nan
    occupancy = env.occupancy(times, positions, time_allocation="linear", **kwargs)
    assert occupancy.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("excluded", ["gap", "epoch"])
@pytest.mark.parametrize("moving", [False, True])
def test_filtered_wall_dwell_does_not_join_separate_valid_motion(excluded, moving):
    env = _wall_environment()
    positions = np.array([[0.5, 0.75], [1.0, 0.75], [1.0, 0.75], [1.5, 0.75]])
    if moving:
        positions[2:, 1] = 1.25
    times = np.arange(4.0)
    kwargs = {"max_gap": None}
    if excluded == "gap":
        times = np.array([0.0, 1.0, 10.0, 11.0])
        kwargs = {"max_gap": 2.0}
    else:
        kwargs["epochs"] = np.array([[0.0, 1.0], [2.0, 3.0]])
    occupancy = env.occupancy(times, positions, time_allocation="linear", **kwargs)
    assert occupancy.sum() == pytest.approx(2.0)


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("same_side", [False, True])
def test_wall_motion_preserves_the_observed_entry_side(duplicate, same_side):
    env = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool),
        grid_edges=(np.arange(5.0),) * 2,
        arena=OpenArena(box(0, 0, 4, 4), walls=(LineString([(2, 0), (2, 4)]),)),
    )
    points = [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]
    if duplicate:
        points.insert(2, [2.0, 1.0])
    points.append([1.0 if same_side else 3.0, 2.0])
    positions = np.array(points)
    times = np.arange(float(len(points)))
    if same_side:
        assert (
            env.occupancy(
                times, positions, time_allocation="linear", max_gap=None
            ).sum()
            == len(points) - 1
        )
    else:
        with pytest.raises(ValueError, match="physical wall"):
            env.occupancy(times, positions, time_allocation="linear", max_gap=None)


@pytest.mark.parametrize("sample_joint", [False, True])
@pytest.mark.parametrize("entry", ["left", "right", "unobserved"])
def test_wall_motion_propagates_possible_sides_through_a_branch(sample_joint, entry):
    env = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool),
        grid_edges=(np.arange(5.0),) * 2,
        arena=OpenArena(
            box(0, 0, 4, 4),
            walls=(LineString([(2, 0), (2, 4)]), LineString([(2, 2), (4, 2)])),
        ),
    )
    points = [[2.0, 1.0]]
    if entry != "unobserved":
        points.insert(0, [1.0 if entry == "left" else 3.0, 1.0])
    if sample_joint:
        points.append([2.0, 2.0])
    points.extend([[2.0, 3.0], [1.0, 3.0]])
    positions = np.array(points)
    times = np.arange(float(len(points)))
    if entry == "right":
        with pytest.raises(ValueError, match="physical wall"):
            env.occupancy(times, positions, time_allocation="linear", max_gap=None)
    else:
        assert (
            env.occupancy(
                times, positions, time_allocation="linear", max_gap=None
            ).sum()
            == len(points) - 1
        )


def test_wall_motion_can_round_a_genuine_isolated_tip():
    env = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool),
        grid_edges=(np.arange(5.0),) * 2,
        arena=OpenArena(box(0, 0, 4, 4), walls=(LineString([(2, 0), (2, 2)]),)),
    )
    positions = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [3.0, 2.0]])
    assert (
        env.occupancy(
            np.arange(4.0), positions, time_allocation="linear", max_gap=None
        ).sum()
        == 3.0
    )
