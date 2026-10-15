"""Physical walls, rather than analysis crops, define physical border scores."""

import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.encoding import border_score


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
    assert border_score(env, np.ones(env.n_bins), boundary_source="analysis") > 0.5


def test_physical_score_matches_independent_wall_length_and_distance_oracle():
    env = physical()
    # A one-cell-wide strip on the west wall. It covers length 10 on the west
    # and length 1 on each adjacent wall: 12/40. Every center is 0.5 from a wall.
    field = (env.bin_centers[:, 0] < 1).astype(float)
    coverage = 12 / 40
    distance = 0.5 / np.sqrt(200)
    expected = (coverage - distance) / (coverage + distance)
    for metric in ("euclidean", "geodesic"):
        np.testing.assert_allclose(border_score(env, field, metric=metric), expected)


def test_analysis_crop_does_not_create_wall_or_change_euclidean_score():
    env = physical(wall=True)
    field = (env.bin_centers[:, 0] == 3.5).astype(float)
    keep = env.bin_centers[:, 0] < 4
    cropped = env.subset(bins=keep)
    np.testing.assert_allclose(
        border_score(cropped, field[keep], metric="euclidean"),
        border_score(env, field, metric="euclidean"),
    )
    assert border_score(cropped, field[keep], boundary_source="analysis") == 1.0
    assert border_score(cropped, field[keep], metric="euclidean") < 0


def test_internal_wall_changes_contact_and_distance():
    walled = physical(wall=True)
    plain = physical()
    field = (walled.bin_centers[:, 0] == 4.5).astype(float)
    assert border_score(walled, field, metric="euclidean") > border_score(
        plain, field, metric="euclidean"
    )


def test_no_represented_path_to_wall_is_not_silently_dropped():
    env = physical().subset(
        bins=(physical().bin_centers[:, 0] > 4)
        & (physical().bin_centers[:, 0] < 6)
        & (physical().bin_centers[:, 1] > 4)
        & (physical().bin_centers[:, 1] < 6)
    )
    assert np.isnan(border_score(env, np.ones(env.n_bins), metric="geodesic"))
    assert border_score(env, np.ones(env.n_bins), metric="euclidean") == -1


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

    env = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool), grid_edges=(np.arange(6.0),) * 2
    )
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
    from neurospatial.ops.transforms import Affine2D

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
    transformed = env.apply_transform(Affine2D(matrix))
    np.testing.assert_allclose(
        border_score(transformed, field, metric="euclidean"),
        border_score(env, field, metric="euclidean"),
    )
    with pytest.raises(ValueError, match="boundary_source"):
        border_score(env, field, boundary_source="unknown")


def test_independently_transformed_components_refuse_uncertified_contact():
    from neurospatial.ops.transforms import Affine2D

    env = physical()
    a = 0.3
    matrix = np.array(
        [[np.cos(a), -np.sin(a), 20], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    )
    rotated = env.apply_transform(Affine2D(matrix)).with_metadata(frame=env.frame)
    combined = Environment.combine(
        {"plain": env, "rotated": rotated}, auto_bridge=False
    )
    # Direct world-frame intersections lost wall coverage even for a uniform
    # field (0.9654 in the smaller 4x4 witness). Refuse uncertified contact.
    with pytest.raises(NotImplementedError, match="shared source frame"):
        border_score(combined, np.ones(combined.n_bins), metric="euclidean")
