"""Bin-valued features are read at positions by verified containment only.

A position outside the environment, a NaN position and a NaN field value are
missing (NaN with ``support`` False), never the nearest bin's value or zero.
Values are the bins' own values: nothing is smoothed or interpolated.
"""

from __future__ import annotations

import pickle

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.ops import (
    SpatialFeatureResult,
    sample_egocentric_features,
    sample_spatial_features,
)


@pytest.fixture(scope="module")
def grid_env() -> Environment:
    x, y = np.meshgrid(np.arange(0.5, 10), np.arange(0.5, 10))
    return Environment.from_samples(
        np.c_[x.ravel(), y.ravel()], bin_size=2.0, units="cm", frame="room"
    )


def _track(track_width: float | None) -> Environment:
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(100.0, 0.0))
    graph.add_edge(0, 1)
    return Environment.from_graph(
        graph, [(0, 1)], bin_size=5.0, track_width=track_width
    )


def test_outside_position_never_reads_the_last_bin(grid_env):
    field = np.arange(grid_env.n_bins, dtype=float)
    field[-1] = 999.0
    result = sample_spatial_features(grid_env, field, np.array([[5.0, 5.0], [50, 50]]))

    assert result.bins[1] == -1
    assert np.isnan(result.values[1, 0])
    assert not result.support[1, 0]
    inside_bin = grid_env.bin_at(np.array([[5.0, 5.0]]))[0]
    assert result.bins[0] == inside_bin
    assert result.values[0, 0] == field[inside_bin]
    assert result.support[0, 0]


def test_values_are_the_exact_bin_values(grid_env):
    rng = np.random.default_rng(0)
    field = rng.normal(size=grid_env.n_bins)
    positions = rng.uniform(0.0, 10.0, size=(200, 2))
    result = sample_spatial_features(grid_env, field, positions)

    bins = grid_env.bin_at(positions)
    np.testing.assert_array_equal(result.bins, bins)
    inside = bins >= 0
    assert inside.any()
    np.testing.assert_array_equal(result.values[inside, 0], field[bins[inside]])
    assert np.isnan(result.values[~inside, 0]).all()
    np.testing.assert_array_equal(result.support[:, 0], inside)


def test_zero_field_value_is_a_supported_observation(grid_env):
    result = sample_spatial_features(
        grid_env, np.zeros(grid_env.n_bins), np.array([[5.0, 5.0], [1.0, 1.0]])
    )
    np.testing.assert_array_equal(result.values, 0.0)
    assert result.support.all()


def test_nan_field_value_is_unobserved(grid_env):
    positions = np.array([[5.0, 5.0], [1.0, 1.0]])
    field = np.ones(grid_env.n_bins)
    field[grid_env.bin_at(positions[:1])[0]] = np.nan
    result = sample_spatial_features(grid_env, field, positions)

    assert result.bins[0] >= 0
    assert np.isnan(result.values[0, 0])
    assert not result.support[0, 0]
    assert result.values[1, 0] == 1.0
    assert result.support[1, 0]


@pytest.mark.parametrize(
    "missing",
    [[np.nan, 5.0], [np.inf, 5.0], [-np.inf, 5.0], [5.0, np.inf]],
    ids=["nan", "inf", "-inf", "inf-y"],
)
def test_non_finite_position_is_missing(grid_env, missing):
    result = sample_spatial_features(
        grid_env, np.ones(grid_env.n_bins), np.array([missing, [5.0, 5.0]])
    )
    assert result.bins[0] == -1
    assert np.isnan(result.values[0, 0])
    assert not result.support[0, 0]
    assert result.support[1, 0]


def test_track_without_width_refuses_even_far_positions():
    env = _track(None)
    assert not env.arena.domain_known
    with pytest.raises(NotImplementedError) as info:
        sample_spatial_features(env, np.ones(env.n_bins), np.array([[50.0, 500.0]]))
    message = str(info.value)
    assert "Fix:" in message
    assert "cannot tell which positions are inside" in message
    assert "track_width" in message


def test_track_with_width_rejects_off_track_positions():
    env = _track(4.0)
    field = np.arange(env.n_bins, dtype=float)
    result = sample_spatial_features(env, field, np.array([[50.0, 500.0], [50.0, 1.0]]))
    assert result.bins[0] == -1
    assert not result.support[0, 0]
    assert np.isnan(result.values[0, 0])
    assert result.support[1, 0]
    assert result.values[1, 0] == field[result.bins[1]]


def test_polygon_environment():
    env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=10)
    field = np.arange(env.n_bins, dtype=float)
    result = sample_spatial_features(
        env, field, np.array([[55.0, 55.0], [150.0, 50.0]])
    )
    np.testing.assert_array_equal(result.support[:, 0], [True, False])
    assert result.values[0, 0] == field[env.bin_at(np.array([[55.0, 55.0]]))[0]]


def test_one_dimensional_grid():
    env = Environment.from_samples(np.arange(0.5, 10)[:, None], bin_size=2.0)
    field = np.arange(env.n_bins, dtype=float) + 1.0
    result = sample_spatial_features(env, field, np.array([[5.0], [50.0]]))
    np.testing.assert_array_equal(result.support[:, 0], [True, False])
    assert result.values[0, 0] == field[env.bin_at(np.array([[5.0]]))[0]]


def test_multiple_features_keep_their_order(grid_env):
    height = np.arange(grid_env.n_bins, dtype=float)
    light = -height
    positions = np.array([[5.0, 5.0], [1.0, 9.0], [50.0, 50.0]])
    result = sample_spatial_features(
        grid_env,
        np.stack([height, light]),
        positions,
        feature_ids=["height", "light"],
    )
    assert result.values.shape == (3, 2)
    assert result.support.shape == (3, 2)
    assert result.bins.shape == (3,)
    np.testing.assert_array_equal(result.feature_ids, ["height", "light"])
    bins = result.bins
    np.testing.assert_array_equal(result.values[:2, 0], height[bins[:2]])
    np.testing.assert_array_equal(result.values[:2, 1], light[bins[:2]])
    assert result.ego_points is None
    assert result.query_positions is None


def test_one_dimensional_field_is_one_feature(grid_env):
    result = sample_spatial_features(
        grid_env, np.ones(grid_env.n_bins), np.array([[5.0, 5.0]] * 4)
    )
    assert result.values.shape == (4, 1)
    np.testing.assert_array_equal(result.feature_ids, [0])


def test_units_and_frame_come_from_the_environment(grid_env):
    result = sample_spatial_features(
        grid_env, np.ones(grid_env.n_bins), np.array([[5.0, 5.0]])
    )
    assert result.units == "cm"
    assert result.frame == "room"


def test_result_owns_read_only_arrays(grid_env):
    fields = np.ones((1, grid_env.n_bins))
    positions = np.array([[5.0, 5.0], [1.0, 1.0]])
    result = sample_spatial_features(grid_env, fields, positions)
    fields[:] = 7.0
    positions[:] = 50.0
    np.testing.assert_array_equal(result.values, 1.0)
    assert result.support.all()

    restored = pickle.loads(pickle.dumps(result))
    for array in (restored.values, restored.support, restored.bins):
        assert not array.flags.writeable
    np.testing.assert_array_equal(restored.values, result.values)


def _one_field(n_bins):
    return np.ones(n_bins)


def _inf_field(n_bins):
    field = np.ones(n_bins)
    field[2] = np.inf
    return field


def _two_fields(n_bins):
    return np.ones((2, n_bins))


_INSIDE = np.array([[5.0, 5.0]])


@pytest.mark.parametrize(
    ("make_fields", "positions", "feature_ids", "match"),
    [
        (lambda n_bins: np.ones(3), _INSIDE, None, "got fields of shape \\(3,\\)"),
        (_inf_field, _INSIDE, None, "inf in fields"),
        (_one_field, np.array([[5.0, 5.0, 5.0]]), None, "positions"),
        (_one_field, np.array([5.0, 5.0]), None, "positions"),
        (_two_fields, _INSIDE, ["a", "a"], "not unique"),
        (_two_fields, _INSIDE, ["a"], "feature_ids of shape"),
        (_one_field, _INSIDE, "a", "sequence of labels"),
        (_one_field, _INSIDE, 3, "sequence of labels"),
        (lambda n_bins: np.ones((1, 1, 1)), _INSIDE, None, "got fields of shape"),
    ],
    ids=[
        "wrong-bin-count",
        "inf-field",
        "positions-wrong-width",
        "positions-flat",
        "duplicate-feature-ids",
        "feature-ids-wrong-length",
        "feature-ids-string",
        "feature-ids-scalar",
        "fields-3d",
    ],
)
def test_invalid_inputs_raise_actionable_errors(
    grid_env, make_fields, positions, feature_ids, match
):
    with pytest.raises(ValueError, match=match) as info:
        sample_spatial_features(
            grid_env,
            make_fields(grid_env.n_bins),
            positions,
            feature_ids=feature_ids,
        )
    assert "Fix:" in str(info.value)


def test_samples_table_rows_and_attrs(grid_env):
    field = np.arange(grid_env.n_bins, dtype=float)
    result = sample_spatial_features(
        grid_env,
        np.stack([field, field + 0.5]),
        np.array([[5.0, 5.0], [50.0, 50.0]]),
        feature_ids=["height", "light"],
    )
    table = result.samples_table()
    assert list(table.columns) == [
        "sample_index",
        "feature_id",
        "bin",
        "value",
        "supported",
    ]
    assert len(table) == 4
    np.testing.assert_array_equal(table["sample_index"], [0, 0, 1, 1])
    np.testing.assert_array_equal(
        table["feature_id"], ["height", "light", "height", "light"]
    )
    np.testing.assert_array_equal(table["bin"], np.repeat(result.bins, 2))
    np.testing.assert_array_equal(table["value"], result.values.reshape(-1))
    np.testing.assert_array_equal(table["supported"], [True, True, False, False])
    assert table.attrs == {"units": "cm", "frame": "room"}
    table.loc[0, "value"] = -1.0
    assert result.values[0, 0] == field[result.bins[0]]


def test_summary_counts(grid_env):
    field = np.ones(grid_env.n_bins)
    field[grid_env.bin_at(np.array([[1.0, 1.0]]))[0]] = np.nan
    result = sample_spatial_features(
        grid_env,
        field,
        np.array([[5.0, 5.0], [1.0, 1.0], [50.0, 50.0], [np.nan, np.nan]]),
    )
    assert result.summary() == {
        "n_samples": 4,
        "n_features": 1,
        "n_points": 0,
        "fraction_supported": 0.25,
        "n_outside": 2,
    }


def test_egocentric_layout_table_and_summary():
    result = SpatialFeatureResult(
        values=np.array([[[1.0], [np.nan]], [[2.0], [3.0]]]),
        support=np.array([[[True], [False]], [[True], [True]]]),
        bins=np.array([[0, -1], [1, 2]]),
        feature_ids=np.array(["height"]),
        units="cm",
        frame=None,
        ego_points=np.array([[1.0, 0.0], [0.0, 1.0]]),
        query_positions=np.zeros((2, 2, 2)),
    )
    table = result.samples_table()
    assert list(table.columns) == [
        "sample_index",
        "point_index",
        "feature_id",
        "bin",
        "value",
        "supported",
    ]
    np.testing.assert_array_equal(table["sample_index"], [0, 0, 1, 1])
    np.testing.assert_array_equal(table["point_index"], [0, 1, 0, 1])
    np.testing.assert_array_equal(table["bin"], [0, -1, 1, 2])
    summary = result.summary()
    assert summary["n_points"] == 2
    assert summary["n_outside"] == 1
    assert summary["fraction_supported"] == 0.75


@pytest.mark.parametrize(
    "change",
    [
        {"support": np.ones((2, 2), dtype=bool)},
        {"bins": np.zeros(3, dtype=np.intp)},
        {"feature_ids": np.array(["a", "b"])},
        {"ego_points": np.zeros((1, 2))},
    ],
    ids=["support-shape", "bins-shape", "feature-ids", "ego-points-on-direct"],
)
def test_result_rejects_inconsistent_fields(change):
    inputs = {
        "values": np.ones((2, 1)),
        "support": np.ones((2, 1), dtype=bool),
        "bins": np.zeros(2, dtype=np.intp),
        "feature_ids": np.array(["a"]),
        "units": None,
        "frame": None,
    }
    with pytest.raises(ValueError, match="inconsistent"):
        SpatialFeatureResult(**(inputs | change))


@pytest.mark.parametrize(
    "change",
    [
        {"ego_points": [[1.0, 0.0, 0.0]]},
        {"query_positions": [[[0.0, 0.0]]]},
        {"ego_points": None},
    ],
    ids=["ego-points-list", "query-positions-list", "ego-points-missing"],
)
def test_egocentric_result_rejects_inconsistent_points(change):
    inputs = {
        "values": np.ones((2, 1, 1)),
        "support": np.ones((2, 1, 1), dtype=bool),
        "bins": np.zeros((2, 1), dtype=np.intp),
        "feature_ids": np.array(["a"]),
        "units": None,
        "frame": None,
        "ego_points": np.zeros((1, 2)),
        "query_positions": np.zeros((2, 1, 2)),
    }
    SpatialFeatureResult(**inputs)  # consistent as given
    with pytest.raises(ValueError, match="inconsistent"):
        SpatialFeatureResult(**(inputs | change))


# --- egocentric sampling ---------------------------------------------------


@pytest.fixture(scope="module")
def box_env() -> Environment:
    return Environment.from_polygon(box(0, 0, 100, 100), bin_size=10.0, units="cm")


def _x_field(env: Environment) -> np.ndarray:
    return env.bin_centers[:, 0].copy()


def test_ahead_is_east_and_left_is_north(box_env):
    field = _x_field(box_env)
    ego = np.array([[20.0, 0.0], [0.0, 20.0]])
    result = sample_egocentric_features(
        box_env,
        field,
        np.array([[50.0, 50.0], [50.0, 50.0]]),
        np.array([0.0, np.pi / 2]),
        ego,
    )
    np.testing.assert_allclose(
        result.query_positions,
        [[[70, 50], [50, 70]], [[50, 70], [30, 50]]],
        atol=1e-9,
    )
    expected_bins = box_env.bin_at(result.query_positions.reshape(-1, 2)).reshape(2, 2)
    np.testing.assert_array_equal(result.bins, expected_bins)
    np.testing.assert_array_equal(result.values[..., 0], field[expected_bins])
    np.testing.assert_array_equal(result.ego_points, ego)
    assert result.units == "cm"


def test_shapes_with_several_points_and_features(box_env):
    rng = np.random.default_rng(0)
    fields = rng.normal(size=(2, box_env.n_bins))
    result = sample_egocentric_features(
        box_env,
        fields,
        rng.uniform(30, 70, size=(5, 2)),
        rng.uniform(-np.pi, np.pi, 5),
        rng.uniform(-10, 10, size=(3, 2)),
        feature_ids=["a", "b"],
    )
    assert result.values.shape == (5, 3, 2)
    assert result.support.shape == (5, 3, 2)
    assert result.bins.shape == (5, 3)
    assert result.query_positions.shape == (5, 3, 2)
    assert list(result.feature_ids) == ["a", "b"]


def test_matches_sampling_the_transformed_points(box_env):
    rng = np.random.default_rng(1)
    fields = rng.normal(size=(2, box_env.n_bins))
    positions = rng.uniform(-10, 110, size=(40, 2))
    headings = rng.uniform(-np.pi, np.pi, 40)
    ego = rng.uniform(-30, 30, size=(4, 2))
    result = sample_egocentric_features(box_env, fields, positions, headings, ego)
    flat = sample_spatial_features(
        box_env, fields, result.query_positions.reshape(-1, 2)
    )
    np.testing.assert_array_equal(result.values.reshape(-1, 2), flat.values)
    np.testing.assert_array_equal(result.support.reshape(-1, 2), flat.support)
    np.testing.assert_array_equal(result.bins.reshape(-1), flat.bins)


@pytest.mark.parametrize("bad", ["heading", "position"])
def test_nan_heading_or_position_is_missing_for_every_point(box_env, bad):
    positions = np.full((4, 2), 50.0)
    headings = np.zeros(4)
    if bad == "heading":
        headings[2] = np.nan
    else:
        positions[2] = np.nan
    result = sample_egocentric_features(
        box_env,
        np.ones(box_env.n_bins),
        positions,
        headings,
        np.array([[0.0, 0.0], [10.0, 0.0]]),
    )
    assert not result.support[2].any()
    assert (result.bins[2] == -1).all()
    assert np.isnan(result.values[2]).all()
    assert result.support[[0, 1, 3]].all()


def test_outside_point_is_missing_and_zero_field_is_supported(box_env):
    field = np.zeros(box_env.n_bins)
    result = sample_egocentric_features(
        box_env,
        field,
        np.array([[95.0, 50.0]]),
        np.array([0.0]),
        np.array([[2.0, 0.0], [20.0, 0.0]]),  # inside, beyond the east wall
    )
    assert result.support[0, 0, 0] and result.values[0, 0, 0] == 0.0
    assert not result.support[0, 1, 0]
    assert result.bins[0, 1] == -1
    assert np.isnan(result.values[0, 1, 0])


def test_samples_table_has_one_row_per_sample_point_and_feature(box_env):
    result = sample_egocentric_features(
        box_env,
        np.ones((2, box_env.n_bins)),
        np.full((5, 2), 50.0),
        np.zeros(5),
        np.array([[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]]),
        feature_ids=["a", "b"],
    )
    table = result.samples_table()
    assert len(table) == 5 * 3 * 2
    assert {"sample_index", "point_index", "feature_id"} <= set(table.columns)


def test_three_dimensional_environment_is_refused():
    rng = np.random.default_rng(0)
    env = Environment.from_samples(rng.uniform(0, 10, size=(500, 3)), bin_size=2.0)
    with pytest.raises(ValueError, match="2-D") as info:
        sample_egocentric_features(
            env,
            np.ones(env.n_bins),
            np.full((2, 2), 5.0),
            np.zeros(2),
            np.array([[1.0, 0.0]]),
        )
    assert "sample_spatial_features" in str(info.value)


@pytest.mark.parametrize(
    ("ego_points", "headings"),
    [
        (np.array([1.0, 0.0, 0.0]), np.zeros(2)),
        (np.zeros((2, 3)), np.zeros(2)),
        (np.zeros((2, 2)), np.zeros(3)),
    ],
)
def test_bad_shapes_raise_value_error(box_env, ego_points, headings):
    with pytest.raises(ValueError, match="sample_egocentric_features"):
        sample_egocentric_features(
            box_env,
            np.ones(box_env.n_bins),
            np.full((2, 2), 50.0),
            headings,
            ego_points,
        )


def test_positions_must_be_two_columns(box_env):
    with pytest.raises(ValueError, match="positions"):
        sample_egocentric_features(
            box_env,
            np.ones(box_env.n_bins),
            np.full((2, 3), 50.0),
            np.zeros(2),
            np.array([[1.0, 0.0]]),
        )


def test_track_without_width_refuses():
    env = _track(None)
    with pytest.raises(NotImplementedError):
        sample_egocentric_features(
            env,
            np.ones(env.n_bins),
            np.zeros((1, 2)),
            np.zeros(1),
            np.array([[1.0, 0.0]]),
        )
