"""Physical walls, rather than analysis crops, define physical border scores."""

from functools import partial

import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.encoding import border_score as _border_score

# These rooms are a few unitless cells across, so keep every field.
border_score = partial(_border_score, min_area=0.0)


def physical(*, wall=False):
    return Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(np.arange(11.0),) * 2,
        arena=OpenArena(
            box(0, 0, 10, 10), walls=(LineString([(5, 0), (5, 10)]),) if wall else ()
        ),
    )


def test_unknown_walls_require_explicit_analysis_mode():
    env = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool), grid_edges=(np.arange(6.0),) * 2
    )
    with pytest.raises(NotImplementedError, match=r"physical|footprint"):
        border_score(env, np.ones(env.n_bins))
    # The full grid's outline is the box itself, so the analysis score equals
    # the physical score of the same box with its walls declared.
    measured = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool),
        grid_edges=(np.arange(6.0),) * 2,
        arena=OpenArena(box(0, 0, 5, 5)),
    )
    np.testing.assert_allclose(
        border_score(env, np.ones(env.n_bins), boundary_source="analysis"),
        border_score(measured, np.ones(env.n_bins)),
    )


def test_physical_score_matches_independent_wall_length_and_distance_oracle():
    env = physical()
    # A one-cell-wide strip covers the whole west wall. Every center is 0.5
    # from a wall, and the largest inscribed circle has radius 5.
    field = (env.bin_centers[:, 0] < 1).astype(float)
    coverage = 1.0
    distance = 0.5 / 5
    expected = (coverage - distance) / (coverage + distance)
    for metric in ("euclidean", "geodesic"):
        np.testing.assert_allclose(border_score(env, field, metric=metric), expected)


def test_analysis_crop_does_not_create_walls_or_supply_missing_wall_rates():
    env = physical(wall=True)
    field = (env.bin_centers[:, 0] == 3.5).astype(float)
    keep = env.bin_centers[:, 0] < 4
    cropped = env.subset(mask=keep)
    with pytest.raises(NotImplementedError, match="Physical-wall coverage"):
        border_score(cropped, field[keep], metric="euclidean")
    # The crop's east edge is a wall of the analysis outline: the field covers
    # it fully, half a cell from it, in a 4-wide crop (inscribed radius 2).
    assert border_score(cropped, field[keep], boundary_source="analysis") == (
        pytest.approx((1 - 0.25) / (1 + 0.25))
    )
    assert border_score(env, field, metric="euclidean") < 0


def test_internal_wall_changes_contact_and_distance():
    walled = physical(wall=True)
    plain = physical()
    field = (walled.bin_centers[:, 0] == 4.5).astype(float)
    assert border_score(walled, field, metric="euclidean") > border_score(
        plain, field, metric="euclidean"
    )


def test_no_represented_wall_coverage_is_not_silently_scored():
    env = physical().subset(
        mask=(physical().bin_centers[:, 0] > 4)
        & (physical().bin_centers[:, 0] < 6)
        & (physical().bin_centers[:, 1] > 4)
        & (physical().bin_centers[:, 1] < 6)
    )
    for metric in ("geodesic", "euclidean"):
        with pytest.raises(NotImplementedError, match="Physical-wall coverage"):
            border_score(env, np.ones(env.n_bins), metric=metric)


def test_asserted_mask_holes_remain_walls_after_storage_and_crop():
    from neurospatial.arena import MaskArena

    mask = np.ones((6, 6), dtype=bool)
    mask[2:4, 2:4] = False
    edges = (np.arange(7.0),) * 2
    env = Environment.from_grid_mask(
        mask,
        grid_edges=edges,
        arena=MaskArena(edges, mask, physical_footprint_known=True),
    )
    restored = Environment.from_dict(env.to_dict())
    field = (env.bin_centers[:, 0] == 1.5).astype(float)
    np.testing.assert_allclose(
        border_score(restored, field, metric="euclidean"),
        border_score(env, field, metric="euclidean"),
    )


def test_result_summary_explains_unavailable_walls_and_explicit_mode_propagates():
    from neurospatial.encoding.spatial import SpatialRateResult, SpatialRatesResult

    # Meters, so the default 200 cm² field minimum is defined and met.
    env = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool), grid_edges=(np.arange(6.0),) * 2
    ).with_metadata(units="m")
    single = SpatialRateResult(
        np.ones(env.n_bins), np.ones(env.n_bins), env, method="binned", bandwidth=0.0
    )
    population = SpatialRatesResult(
        np.ones((1, env.n_bins)),
        np.ones(env.n_bins),
        env,
        method="binned",
        bandwidth=0.0,
    )
    table = single.summary_table(include_classification=False)
    assert np.isnan(table["border_score"].iloc[0])
    assert "border_score" in table.attrs["unavailable_metrics"]
    assert table.attrs["border_boundary_source"] == "physical"
    expected = border_score(env, np.ones(env.n_bins), boundary_source="analysis")
    np.testing.assert_allclose(
        single.border_score(boundary_source="analysis"), expected
    )
    np.testing.assert_allclose(
        population.border_scores(boundary_source="analysis").scores, [expected]
    )
    assert np.isfinite(
        population.summary_table(boundary_source="analysis")["border_score"]
    ).all()


def test_physical_score_similarity_invariance_and_bad_source():
    from neurospatial.ops.transforms import AffineND

    env = physical(wall=True)
    field = (env.bin_centers[:, 0] < 1).astype(float)
    a = 0.3
    matrix = np.array(
        [
            [2 * np.cos(a), -2 * np.sin(a), 100],
            [2 * np.sin(a), 2 * np.cos(a), -40],
            [0, 0, 1],
        ]
    )
    transformed = env.apply_transform(AffineND(matrix))
    np.testing.assert_allclose(
        border_score(transformed, field, metric="euclidean"),
        border_score(env, field, metric="euclidean"),
    )
    with pytest.raises(ValueError, match="boundary_source"):
        border_score(env, field, boundary_source="unknown")


def test_independently_transformed_components_refuse_uncertified_contact():
    from neurospatial.ops.transforms import AffineND

    env = physical()
    a = 0.3
    matrix = np.array(
        [[np.cos(a), -np.sin(a), 20], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    )
    rotated = env.apply_transform(AffineND(matrix)).with_metadata(frame=env.frame)
    combined = Environment.combine(
        {"plain": env, "rotated": rotated}, auto_bridge=False
    )
    # Direct world-frame intersections lost wall coverage even for a uniform
    # field (0.9654 in the smaller 4x4 witness). Refuse uncertified contact.
    with pytest.raises(NotImplementedError, match="shared source frame"):
        border_score(combined, np.ones(combined.n_bins), metric="euclidean")


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("transformed", [False, True])
def test_combined_rooms_retain_shared_physical_wall(restore, transformed):
    from neurospatial.geometry._boundaries import physical_walls

    rooms = [
        Environment.from_grid_mask(
            np.ones((5, 5), dtype=bool),
            grid_edges=(np.arange(start, start + 6.0), np.arange(6.0)),
            arena=OpenArena(box(start, 0, start + 5, 5)),
        )
        for start in [0, 5]
    ]
    env = Environment.combine(
        dict(zip(["left", "right"], rooms, strict=True)), auto_bridge=False
    )
    if transformed:
        from neurospatial.ops.transforms import AffineND

        env = env.apply_transform(
            AffineND(np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 200.0], [0.0, 0.0, 1.0]]))
        )
    if restore:
        env = Environment.from_dict(env.to_dict())
    walls = physical_walls(env.arena)
    assert walls.length == pytest.approx(35.0)  # union includes the shared 5-unit wall
    assert walls.distance(Point(105.0, 202.5) if transformed else Point(5.0, 2.5)) == 0


def test_combined_partial_wall_evidence_does_not_assert_unknown_child_outline():
    from neurospatial.geometry._boundaries import physical_walls

    known = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool),
        grid_edges=(np.arange(6.0),) * 2,
        arena=OpenArena(box(0, 0, 5, 5)),
    )
    unknown = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool),
        grid_edges=(np.arange(10.0, 16.0), np.arange(6.0)),
    )
    env = Environment.combine({"known": known, "unknown": unknown}, auto_bridge=False)
    assert physical_walls(env.arena).equals(physical_walls(known.arena))


def test_unknown_wall_error_distinguishes_geometry_from_map_support():
    env = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool), grid_edges=(np.arange(6.0),) * 2
    )
    with pytest.raises(NotImplementedError) as error:
        border_score(env, np.ones(env.n_bins))
    assert "measured" in str(error.value)
    assert "bin support" in str(error.value)
    assert "boundary_source='analysis'" in str(error.value)


@pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
def test_position_footprint_inset_from_measured_walls_is_unavailable(metric):
    x, y = np.meshgrid(np.arange(3.0, 98.0), np.arange(3.0, 98.0))
    env = Environment.from_samples(
        np.column_stack((x.ravel(), y.ravel())),
        bin_size=2.0,
        arena=OpenArena(box(0, 0, 100, 100)),
    )
    field = (env.bin_centers[:, 0] < 10).astype(float)
    with pytest.raises(NotImplementedError, match="Physical-wall coverage"):
        border_score(env, field, metric=metric)
    assert border_score(env, field, boundary_source="analysis") > 0.8


def test_rebuilt_rotated_arena_owns_its_whole_wall():
    """An axis-aligned grid on a rotated square leaves slivers along every wall;
    their wall pieces belong to the neighboring bins, so a field over the whole
    arena covers the whole wall instead of reporting partial coverage."""
    from shapely.affinity import rotate
    from shapely.ops import unary_union

    from neurospatial.encoding._physical_border import border_geometry

    env = Environment.from_polygon(
        rotate(box(0, 0, 10, 10), 45, origin=(5, 5)), bin_size=0.2
    )
    geometry = border_geometry(env, "physical")
    assert len(geometry.wall_lengths) == 4
    for (_owners, pieces), length in zip(
        geometry.contact, geometry.wall_lengths, strict=True
    ):
        assert unary_union(pieces).length == pytest.approx(length, rel=1e-9)
    assert np.isfinite(border_score(env, np.ones(env.n_bins)))


@pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
def test_unobserved_wall_rates_are_unavailable_but_observed_zero_rates_are_valid(
    metric,
):
    env = physical()
    field = (
        (env.bin_centers[:, 0] > 4)
        & (env.bin_centers[:, 0] < 6)
        & (env.bin_centers[:, 1] > 4)
        & (env.bin_centers[:, 1] < 6)
    ).astype(float)
    assert border_score(env, field, metric=metric) == -1.0
    field[env.bin_centers[:, 0] < 1] = np.nan
    with pytest.raises(NotImplementedError, match="finite firing rates"):
        border_score(env, field, metric=metric)


def test_all_missing_wall_rates_refuse_before_field_detection():
    env = physical()
    with pytest.raises(NotImplementedError, match="finite firing rates"):
        border_score(env, np.full(env.n_bins, np.nan))


def test_missing_wall_rate_support_is_reported_in_result_summary():
    from neurospatial.encoding.spatial import SpatialRateResult

    env = physical()
    rates = np.ones(env.n_bins)
    rates[env.bin_centers[:, 0] < 1] = np.nan
    result = SpatialRateResult(
        rates, np.ones(env.n_bins), env, method="binned", bandwidth=0.0
    )
    table = result.summary_table(include_classification=False)
    assert np.isnan(table["border_score"].iloc[0])
    assert "finite firing rates" in table.attrs["unavailable_metrics"]["border_score"]


def test_batch_missing_wall_rate_support_raises_instead_of_hiding_as_numeric_failure():
    from neurospatial.encoding._metrics import batch_border_scores

    env = physical()
    rates = np.ones((2, env.n_bins))
    rates[1, env.bin_centers[:, 0] < 1] = np.nan
    with pytest.raises(NotImplementedError, match="finite firing rates"):
        batch_border_scores(env, rates)


def test_observed_internal_wall_side_does_not_supply_the_other_sides_missing_rates():
    env = physical(wall=True)
    rates = np.ones(env.n_bins)
    rates[env.bin_centers[:, 0] == 4.5] = np.nan
    with pytest.raises(NotImplementedError, match="finite firing rates"):
        border_score(env, rates)


def _wall_cell_with_unobserved_wall():
    """A west-wall field in a room whose east-wall cells were never observed.

    Meters, so the default 200 cm² field minimum is defined and met.
    """
    env = physical().with_metadata(units="m")
    rates = np.where(env.bin_centers[:, 0] < 1, 10.0, 0.1)
    rates[env.bin_centers[:, 0] > 9] = np.nan
    return env, rates


def test_summary_does_not_call_a_wall_cell_place_when_border_is_unavailable():
    from neurospatial.encoding.spatial import SpatialRateResult, SpatialRatesResult

    env, rates = _wall_cell_with_unobserved_wall()
    untuned = np.where(np.isnan(rates), np.nan, 1.0)
    occupancy = np.ones(env.n_bins)
    single = SpatialRateResult(rates, occupancy, env, method="binned", bandwidth=0.0)
    population = SpatialRatesResult(
        np.stack([rates, untuned]), occupancy, env, method="binned", bandwidth=0.0
    )
    with pytest.raises(NotImplementedError, match="Physical-wall coverage"):
        single.border_score()
    with pytest.warns(UserWarning, match="Physical-wall coverage"):
        labels = population.label_cell_types()

    single_table = single.summary_table()
    population_table = population.summary_table()
    # Without the border score either cell could be a border cell, whatever
    # its spatial information, so neither label is known.
    assert single_table["cell_type"].tolist() == ["undetermined"]
    assert population_table["cell_type"].tolist() == ["undetermined"] * 2
    assert labels.tolist() == population_table["cell_type"].tolist()
    assert population_table["passes_border_score"].isna().all()
    for table in (single_table, population_table):
        assert np.isnan(table["border_score"]).all()
        assert "border_score" in table.attrs["unavailable_metrics"]
    # The cell's field is on the observed west wall; the analysis boundary
    # scores it and the label follows the score.
    assert population.summary_table(boundary_source="analysis")[
        "cell_type"
    ].tolist() == ["border", "unclassified"]


@pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
def test_combined_component_without_walls_is_not_scored_against_other_walls(metric):
    known = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool),
        grid_edges=(np.arange(6.0),) * 2,
        arena=OpenArena(box(0, 0, 5, 5)),
    )
    unknown = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool),
        grid_edges=(np.arange(10.0, 16.0), np.arange(6.0)),
    )
    env = Environment.combine({"known": known, "unknown": unknown}, auto_bridge=False)
    # This field hugs the unknown room's west edge, which is not a declared wall.
    field = ((env.bin_centers[:, 0] > 10) & (env.bin_centers[:, 0] < 11)).astype(float)
    with pytest.raises(NotImplementedError) as error:
        border_score(env, field, metric=metric)
    message = str(error.value)
    assert "'unknown'" in message
    assert "'known'" not in message
    assert "boundary_source='analysis'" in message
    # The analysis outline's west edge of that room is fully covered, half a
    # cell away, in a 5-wide room (inscribed radius 2.5).
    assert border_score(
        env, field, boundary_source="analysis", metric=metric
    ) == pytest.approx((1 - 0.2) / (1 + 0.2))

    walled = Environment.combine(
        {
            "known": known,
            "walled": Environment.from_grid_mask(
                np.ones((5, 5), dtype=bool),
                grid_edges=(np.arange(10.0, 16.0), np.arange(6.0)),
                arena=OpenArena(box(10, 0, 15, 5)),
            ),
        },
        auto_bridge=False,
    )
    assert border_score(walled, field, metric=metric) > 0.5


def _room_with_internal_wall(*, footprint_known):
    """A 10 m x 10 m room with one declared internal wall at x=5."""
    return Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(np.arange(11.0),) * 2,
        arena=OpenArena(
            box(0, 0, 10, 10),
            walls=(LineString([(5, 2), (5, 8)]),),
            physical_footprint_known=footprint_known,
        ),
    ).with_metadata(units="m")


@pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
def test_internal_walls_without_footprint_do_not_score_outer_edge_fields(metric):
    from neurospatial.encoding.spatial import SpatialRateResult

    unknown = _room_with_internal_wall(footprint_known=False)
    # A strong field hugging the west edge. Only the internal wall is declared,
    # so whether the west edge is a wall is unknown.
    rates = np.where(unknown.bin_centers[:, 0] < 1, 10.0, 0.1)
    with pytest.raises(NotImplementedError) as error:
        border_score(unknown, rates, metric=metric)
    message = str(error.value)
    assert "footprint" in message
    assert "physical_footprint_known=True" in message
    assert "boundary_source='analysis'" in message

    result = SpatialRateResult(
        rates, np.ones(unknown.n_bins), unknown, method="binned", bandwidth=0.0
    )
    table = result.summary_table()
    assert np.isnan(table["border_score"].iloc[0])
    assert "footprint" in table.attrs["unavailable_metrics"]["border_score"]
    assert table["cell_type"].tolist() == ["undetermined"]

    # Declaring the same outline as the measured footprint scores the
    # outer-wall field as a border field.
    known = _room_with_internal_wall(footprint_known=True)
    assert border_score(known, rates, metric=metric) > 0.5
    known_result = SpatialRateResult(
        rates, np.ones(known.n_bins), known, method="binned", bandwidth=0.0
    )
    assert known_result.summary_table()["cell_type"].tolist() == ["border"]


def test_combined_component_with_internal_walls_but_no_footprint_is_refused():
    known = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool),
        grid_edges=(np.arange(6.0),) * 2,
        arena=OpenArena(box(0, 0, 5, 5)),
    )
    partial = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool),
        grid_edges=(np.arange(10.0, 16.0), np.arange(6.0)),
        arena=OpenArena(
            box(10, 0, 15, 5),
            walls=(LineString([(13, 1), (13, 4)]),),
            physical_footprint_known=False,
        ),
    )
    env = Environment.combine({"known": known, "partial": partial}, auto_bridge=False)
    # This field hugs the partial room's west edge, which is not a declared wall.
    field = ((env.bin_centers[:, 0] > 10) & (env.bin_centers[:, 0] < 11)).astype(float)
    with pytest.raises(NotImplementedError) as error:
        border_score(env, field, metric="euclidean")
    message = str(error.value)
    assert "'partial'" in message
    assert "'known'" not in message
