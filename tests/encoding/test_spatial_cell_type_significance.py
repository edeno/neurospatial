"""One circular-shift test of spatial information, grid score and border score."""

import pickle
import warnings

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding import (
    CellTypeSignificanceResult,
    compute_spatial_rates,
    place_cell_significance,
    spatial_cell_type_significance,
)

STATISTICS = ("spatial_information", "grid_score", "border_score")


def test_single_and_population_agree_by_unit_id(significance_recording):
    r = significance_recording
    ids = [10, 20, 30]
    population = spatial_cell_type_significance(
        r.env, r.trains, r.times, r.positions, unit_ids=ids, n_shuffles=20, rng=0
    )
    reordered = spatial_cell_type_significance(
        r.env,
        r.trains[::-1],
        r.times,
        r.positions,
        unit_ids=ids[::-1],
        n_shuffles=20,
        rng=0,
    )
    for i, uid in enumerate(ids):
        alone = spatial_cell_type_significance(
            r.env,
            [r.trains[i]],
            r.times,
            r.positions,
            unit_ids=[uid],
            n_shuffles=20,
            rng=0,
        )
        for result in (alone, reordered):
            np.testing.assert_array_equal(
                result.sel(unit_id=uid)["grid_score"].null_scores,
                population.sel(unit_id=uid)["grid_score"].null_scores,
            )
        np.testing.assert_array_equal(alone.p_values[0], population.p_values[i])
    # Every statistic is scored from the same shifted trains, so the
    # information test is the one place_cell_significance runs.
    place = place_cell_significance(
        r.env, r.trains, r.times, r.positions, unit_ids=ids, n_shuffles=20, rng=0
    )
    for i, uid in enumerate(ids):
        assert place[uid].p_value == population.p_values[i, 0]
        np.testing.assert_array_equal(
            place[uid].null_scores, population.null_scores[i, :, 0]
        )


def test_result_layout_and_table(significance_recording):
    r = significance_recording
    result = spatial_cell_type_significance(
        r.env,
        r.trains,
        r.times,
        r.positions,
        unit_ids=["a", "b", "c"],
        n_shuffles=20,
        boundary_source="analysis",
        rng=0,
    )
    assert result.statistics == STATISTICS
    assert result.observed_scores.shape == (3, 3)
    assert result.null_scores.shape == (3, 20, 3)
    assert result.n_shuffles == 20
    finite = np.isfinite(result.null_scores)
    exceed = (finite & (result.null_scores >= result.observed_scores[:, None])).sum(1)
    expected = (exceed + 1) / (finite.sum(1) + 1)
    np.testing.assert_array_equal(
        result.p_values, np.where(np.isfinite(result.observed_scores), expected, np.nan)
    )
    table = result.summary_table(alpha=0.1)
    assert table.index.name == "unit_id" and table.index.tolist() == ["a", "b", "c"]
    assert list(table.columns) == [
        *STATISTICS,
        "spatial_information_bias",
        *(f"{name}_p_value" for name in STATISTICS),
        *(f"{name}_null_percentile" for name in STATISTICS),
        *(f"{name}_n_finite_null" for name in STATISTICS),
        *(f"passes_{name}" for name in STATISTICS),
        "cell_type",
    ]
    assert table.attrs["null_percentile"] == 90.0
    np.testing.assert_allclose(
        table["grid_score_null_percentile"],
        np.nanpercentile(result.null_scores[:, :, 1], 90, axis=1),
    )
    np.testing.assert_array_equal(
        table["passes_border_score"].to_numpy(dtype=bool),
        result.p_values[:, 2] < 0.1,
    )
    np.testing.assert_array_equal(
        table["cell_type"], result.label_cell_types(alpha=0.1)
    )
    assert result[1].keys() == set(STATISTICS)
    # The path never reaches a wall, so the border p-value is NaN; position
    # and label selection must still agree on every statistic.
    for name in STATISTICS:
        np.testing.assert_equal(
            result[1][name].p_value, result.sel(unit_id="b")[name].p_value
        )
    assert np.isfinite(result[1]["spatial_information"].p_value)
    summary = result.summary()
    labels = result.label_cell_types()
    assert summary["n_units"] == 3 and summary["n_shuffles"] == 20
    assert summary["n_place"] == np.sum(labels == "place")
    assert summary["n_grid_and_border"] == np.sum(labels == "grid+border")


def test_result_owns_read_only_arrays(significance_recording):
    r = significance_recording
    result = spatial_cell_type_significance(
        r.env, r.trains, r.times, r.positions, n_shuffles=5, rng=0
    )
    for array in (result.observed_scores, result.null_scores, result.p_values):
        with pytest.raises(ValueError):
            array[...] = 0
    restored = pickle.loads(pickle.dumps(result))
    np.testing.assert_array_equal(restored.p_values, result.p_values)
    assert not restored.null_scores.flags.writeable
    with pytest.raises(TypeError):
        result.unavailable_metrics["grid_score"] = "edited"


def test_track_cell_is_place_and_unavailable_scores_are_recorded():
    """On a 1-D track grid and border do not apply; the test still runs."""
    from neurospatial.simulation import generate_poisson_spikes

    env = Environment.from_samples(np.arange(0.0, 201.0)[:, None], bin_size=2.0)
    rng = np.random.default_rng(4)
    times = np.arange(0.0, 600.0, 1 / 30)
    # Irregular laps, so no circular shift realigns the trajectory with itself.
    speed = rng.uniform(5, 25, len(times) // 300 + 1).repeat(300)[: len(times)]
    x = 200 * np.abs((np.cumsum(speed) / 30 / 200) % 2 - 1)
    trains = [
        generate_poisson_spikes(
            0.1 + 12 * np.exp(-0.5 * ((x - 70) / 8) ** 2), times, seed=11
        ),
        generate_poisson_spikes(np.full(len(times), 1.5), times, seed=12),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = spatial_cell_type_significance(
            env, trains, times, x[:, None], n_shuffles=39, rng=0
        )
        labels = result.label_cell_types()
    np.testing.assert_array_equal(labels, ["place", "unclassified"])
    assert set(result.unavailable_metrics) == {"grid_score", "border_score"}
    assert set(result.inapplicable_metrics) == {"grid_score", "border_score"}
    assert np.isnan(result.p_values[:, 1:]).all()
    assert result.p_values[0, 0] == 1 / 40


def test_missing_walls_leave_border_undetermined(ou_2min_env, ou_2min, noise_trains):
    """An environment without measured walls has no physical border score."""
    times, positions, _ = ou_2min
    result = spatial_cell_type_significance(
        ou_2min_env, noise_trains(120)[:2], times, positions, n_shuffles=5, rng=0
    )
    assert "border_score" in result.unavailable_metrics
    assert result.inapplicable_metrics == ()
    assert np.isnan(result.p_values[:, 2]).all()
    with pytest.warns(UserWarning, match="border_score"):
        labels = result.label_cell_types()
    np.testing.assert_array_equal(labels, ["undetermined", "undetermined"])
    assert result.summary_table()["passes_border_score"].isna().all()


def test_rejects_glm(significance_recording):
    r = significance_recording
    with pytest.raises(ValueError, match="Fix:"):
        spatial_cell_type_significance(
            r.env, r.trains, r.times, r.positions, method="glm", n_shuffles=5
        )


@pytest.mark.parametrize(
    ("observed", "grid_spacing"),
    [((2, 2), (2,)), ((2, 3), (3,))],
)
def test_result_validates_shapes(observed, grid_spacing):
    with pytest.raises(ValueError, match="n_units, 3"):
        CellTypeSignificanceResult(
            observed_scores=np.zeros(observed),
            null_scores=np.zeros((2, 5, 3)),
            unit_ids=np.arange(2),
            min_shift=20.0,
            method="binned",
            bandwidth=5.0,
            boundary_source="physical",
            grid_spacing=np.zeros(grid_spacing),
            spatial_information_bias=np.zeros(grid_spacing),
        )


@pytest.fixture(scope="module")
def box_session():
    """A 100 cm box, 2.5 cm bins, 10 minutes of exploration."""
    import shapely

    from neurospatial.simulation import simulate_trajectory_ou

    def make(size, minutes, seed):
        env = Environment.from_polygon(
            shapely.box(0, 0, size, size), bin_size=2.5, units="cm"
        )
        positions, times = simulate_trajectory_ou(
            env,
            duration=60.0 * minutes,
            dt=1 / 30,
            speed_units="cm",
            speed_mean=15.0,
            speed_std=7.5,
            seed=seed,
        )
        return env, times, positions

    return make


def test_grid_cell_with_background_firing_is_grid(box_session):
    """A grid cell on 1 Hz background has low information but a clear grid."""
    from neurospatial.simulation import GridCellModel, generate_poisson_spikes

    env, times, positions = box_session(100, 10, seed=1)
    grid = GridCellModel(
        env, grid_spacing=45.0, max_rate=10.0, baseline_rate=1.0
    ).firing_rate(positions)
    trains = [generate_poisson_spikes(grid, times, seed=21)]
    result = spatial_cell_type_significance(
        env,
        trains,
        times,
        positions,
        n_shuffles=39,
        rng=0,
        boundary_source="analysis",
    )
    table = result.summary_table()
    # The fixed 0.5 bits/spike information cutoff used to hide this cell.
    assert table["spatial_information"].iloc[0] < 0.5
    assert table["grid_score_p_value"].iloc[0] == 1 / 40
    assert table["cell_type"].tolist() == ["grid"]


def test_border_cell_in_a_small_box_is_border(box_session):
    """A cell along every wall of a 50 cm box is a border cell."""
    import shapely

    from neurospatial.simulation import generate_poisson_spikes

    env, times, positions = box_session(50, 15, seed=2)
    walls = shapely.box(0, 0, 50, 50).exterior
    distance = shapely.distance(shapely.points(positions), walls)
    rate = 0.2 + 11.8 * np.exp(-0.5 * (distance / 4.0) ** 2)
    trains = [generate_poisson_spikes(rate, times, seed=22)]
    result = spatial_cell_type_significance(
        env, trains, times, positions, n_shuffles=39, rng=0
    )
    assert result.unavailable_metrics == {}
    assert result.p_values[0, 2] == 1 / 40
    assert result.label_cell_types().tolist() == ["border"]


@pytest.mark.slow
def test_untuned_p_values_are_uniform_for_every_statistic(box_session):
    """Each test's false-positive rate for untuned units is near alpha."""
    from neurospatial.simulation import generate_poisson_spikes

    env, times, positions = box_session(100, 10, seed=3)
    rng = np.random.default_rng(9)
    trains = [
        generate_poisson_spikes(np.full(len(times), rate), times, seed=3000 + i)
        for i, rate in enumerate(rng.uniform(0.5, 3.0, 60))
    ]
    result = spatial_cell_type_significance(
        env,
        trains,
        times,
        positions,
        n_shuffles=99,
        rng=1,
        boundary_source="analysis",
    )
    assert np.isfinite(result.p_values).all()
    for k, name in enumerate(STATISTICS):
        p = result.p_values[:, k]
        assert np.mean(p < 0.05) <= 0.1, name
        assert 0.3 <= np.median(p) <= 0.7, name


@pytest.mark.slow
def test_radial_arm_place_cells_are_not_border_cells():
    """Narrow arms put every field near a wall; the border test is not fooled."""
    from neurospatial.simulation import (
        PlaceCellModel,
        generate_poisson_spikes,
        simulate_trajectory_ou,
    )
    from neurospatial.simulation.mazes import make_radial_arm_maze

    env = make_radial_arm_maze(bin_size=2.5).env_2d
    positions, times = simulate_trajectory_ou(
        env,
        duration=1200.0,
        dt=1 / 30,
        speed_units="cm",
        speed_mean=20.0,
        speed_std=10.0,
        seed=8,
    )
    rng = np.random.default_rng(21)
    centers = env.bin_centers[rng.choice(env.n_bins, 12, replace=False)]
    trains = [
        generate_poisson_spikes(
            PlaceCellModel(
                env,
                center=center,
                width=10.0,
                max_rate=15,
                baseline_rate=0.2,
                metric="geodesic",
            ).firing_rate(positions),
            times,
            seed=500 + i,
        )
        for i, center in enumerate(centers)
    ]
    screen = compute_spatial_rates(env, trains, times, positions).label_cell_types(
        boundary_source="analysis"
    )
    assert np.sum(screen == "border") >= 3
    result = spatial_cell_type_significance(
        env,
        trains,
        times,
        positions,
        n_shuffles=99,
        rng=0,
        boundary_source="analysis",
    )
    labels = result.label_cell_types()
    assert not np.any(np.char.find(labels.astype(str), "border") >= 0), labels
    assert np.sum(labels == "place") >= 10


def _rebuilt(result, **changes):
    """The result rebuilt with some fields replaced."""
    fields = {
        name: getattr(result, name)
        for name in (
            "observed_scores",
            "null_scores",
            "unit_ids",
            "min_shift",
            "method",
            "bandwidth",
            "boundary_source",
            "grid_spacing",
            "spatial_information_bias",
            "unavailable_metrics",
            "inapplicable_metrics",
        )
    }
    return CellTypeSignificanceResult(**{**fields, **changes})


def test_significant_grid_score_without_a_ring_is_not_grid(grid_and_hexless):
    """A grid label needs a hexagonal ring as well as a significant score."""
    r = grid_and_hexless
    result = spatial_cell_type_significance(
        r.env,
        r.trains,
        r.times,
        r.positions,
        n_shuffles=79,
        rng=0,
        boundary_source="analysis",
    )
    assert np.isfinite(result.grid_spacing[0]) and np.isnan(result.grid_spacing[1])
    assert abs(result.grid_spacing[0] - 45.0) < 5.0
    assert (result.p_values[:, 1] < 0.05).all()
    labels = result.label_cell_types()
    assert labels[0] == "grid"
    assert "grid" not in labels[1]
    # Only the ring separates a significant grid score from a grid label.
    significant = result.p_values[:, 1] < 0.05
    with_ring = _rebuilt(result, grid_spacing=np.array([45.0, 45.0]))
    without_ring = _rebuilt(result, grid_spacing=np.array([np.nan, np.nan]))
    for has_ring, rebuilt in ((True, with_ring), (False, without_ring)):
        is_grid = np.char.find(rebuilt.label_cell_types().astype(str), "grid") >= 0
        np.testing.assert_array_equal(is_grid, significant & has_ring)


def test_place_needs_bias_corrected_information(ou_2min_env, ou_2min, noise_trains):
    """Untuned units significant by chance have almost no information beyond bias."""
    times, positions, _ = ou_2min
    trains = noise_trains(120)
    ids = [1, 17]  # the two untuned units whose information test has p < 0.05
    result = spatial_cell_type_significance(
        ou_2min_env,
        [trains[i] for i in ids],
        times,
        positions,
        unit_ids=ids,
        n_shuffles=99,
        rng=0,
        boundary_source="analysis",
    )
    assert (result.p_values[:, 0] < 0.05).all()
    debiased = result.observed_scores[:, 0] - result.spatial_information_bias
    assert (debiased < 0.1).all()
    table = result.summary_table()
    assert not table["passes_spatial_information"].any()
    assert "place" not in result.label_cell_types()
    np.testing.assert_array_equal(
        table["spatial_information_bias"], result.spatial_information_bias
    )
    # A lower cutoff restores the label of a significant unit.
    labels = result.label_cell_types(min_spatial_info=-np.inf)
    assert (labels[result.p_values[:, 1] >= 0.05] == "place").all()


def test_undefined_information_bias_is_not_place(significance_recording):
    """A unit with too few spikes for the bias correction is never place."""
    r = significance_recording
    result = spatial_cell_type_significance(
        r.env,
        r.trains,
        r.times,
        r.positions,
        n_shuffles=39,
        rng=0,
        boundary_source="analysis",
    )
    observed = result.observed_scores.copy()
    null = result.null_scores.copy()
    observed[:, 0] = 2.0  # every information test significant
    null[:, :, 0] = 0.0
    rebuilt = _rebuilt(
        result,
        observed_scores=observed,
        null_scores=null,
        spatial_information_bias=np.array([np.nan, 0.1, 1.8]),
    )
    with pytest.warns(UserWarning, match="1 of 3 units"):
        passes = rebuilt.summary_table()["passes_spatial_information"]
    assert passes.tolist() == [False, True, False]


def _synthetic(observed, null, bias=None):
    """A result from given scores, with a hexagonal ring for every unit."""
    observed = np.asarray(observed, dtype=float)
    n_units = len(observed)
    return CellTypeSignificanceResult(
        observed_scores=observed,
        null_scores=null,
        unit_ids=np.arange(n_units),
        min_shift=20.0,
        method="diffusion_kde",
        bandwidth=5.0,
        boundary_source="analysis",
        grid_spacing=np.full(n_units, 40.0),
        spatial_information_bias=np.zeros(n_units) if bias is None else bias,
    )


def test_too_few_finite_null_scores_leave_the_test_undetermined():
    """A test whose smallest possible p-value is not below alpha cannot reject."""
    null = np.zeros((3, 39, 3))
    null[0, 9:, 2] = np.nan  # 9 finite border nulls: p >= 1 / 10
    null[2, 9:, 0] = np.nan  # 9 finite information nulls
    result = _synthetic(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        null,
        bias=np.array([0.0, 0.0, 0.9]),
    )
    np.testing.assert_allclose(result.p_values[:, 2], [0.1, 1 / 40, 1.0])
    with pytest.warns(UserWarning, match="1 of 3 units have a test .border_score"):
        table = result.summary_table()
    np.testing.assert_array_equal(table["border_score_n_finite_null"], [9, 39, 39])
    np.testing.assert_array_equal(
        table["spatial_information_n_finite_null"], [39, 39, 9]
    )
    assert table["passes_border_score"].isna().tolist() == [True, False, False]
    # Unit 2's information test cannot reject either, but its bias-corrected
    # information (0.1 bits/spike) rules out a place label regardless.
    with pytest.warns(UserWarning, match="too few finite null scores"):
        labels = result.label_cell_types()
    np.testing.assert_array_equal(labels, ["undetermined", "border", "unclassified"])
    # At alpha = 0.2 the nine finite nulls are enough.
    assert result.label_cell_types(alpha=0.2)[0] == "border"
