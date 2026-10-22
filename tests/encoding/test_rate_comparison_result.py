"""Tables, exports, plots, selection and ownership of RateComparisonResult."""

from __future__ import annotations

import copy
import pickle
from dataclasses import fields

import matplotlib
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding import (
    RateComparisonResult,
    SpatialRatesResult,
    compare_spatial_rates,
)

matplotlib.use("Agg")

F = [1.0, 2.0, 3.0, 4.0, 5.0]
G = [5.0, 1.0, 4.0, 2.0, 3.0]
H = [2.0, 2.5, 1.0, 9.0, 3.0]
IDS = [4, 9, 2]


@pytest.fixture(scope="module")
def env() -> Environment:
    return Environment.from_samples(
        np.linspace(0, 40, 41)[:, None], bin_size=10.0, units="cm"
    )


def _rates(env, maps, occupancy, ids=IDS) -> SpatialRatesResult:
    return SpatialRatesResult(
        firing_rates=np.array(maps, dtype=float),
        occupancy=np.array(occupancy, dtype=float),
        env=env,
        method="binned",
        bandwidth=5.0,
        unit_ids=ids,
    )


@pytest.fixture(scope="module")
def result(env) -> RateComparisonResult:
    """Three units (ids 4, 9, 2); the last bin was not occupied in B."""
    rates_a = _rates(env, [F, G, H], [1, 1, 1, 1, 1])
    rates_b = _rates(env, [G, F, [1.0, 2.0, 4.0, 3.0, 7.0]], [1, 1, 1, 1, 0])
    return compare_spatial_rates(rates_a, rates_b)


@pytest.fixture(scope="module")
def result_2d() -> RateComparisonResult:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 40, (400, 2))
    env2 = Environment.from_samples(positions, bin_size=10.0, units="cm")
    maps = rng.uniform(1, 5, (3, env2.n_bins))
    occupancy = np.ones(env2.n_bins)
    return compare_spatial_rates(
        _rates(env2, maps, occupancy),
        _rates(env2, maps[::-1][[2, 1, 0]] + 1.0, occupancy),
    )


# --- summary_table -----------------------------------------------------------


def test_summary_table_layout(result) -> None:
    table = result.summary_table()

    assert table.index.name == "unit_id"
    assert table.index.tolist() == IDS
    assert table.columns.tolist() == [
        "correlation",
        "n_common_bins",
        "mean_rate_a",
        "mean_rate_b",
        "mean_rate_difference",
        "common_occupancy_a",
        "common_occupancy_b",
        "status",
    ]
    assert len(table) == 3
    assert table.attrs["method"] == "pearson"
    assert table.attrs["units"] == {
        "mean_rate_a": "Hz",
        "mean_rate_b": "Hz",
        "mean_rate_difference": "Hz",
        "common_occupancy_a": "s",
        "common_occupancy_b": "s",
        "coordinates": "cm",
    }
    assert table.attrs["analysis_settings"] == result.analysis_settings


def test_summary_table_means_are_over_each_units_common_bins(result) -> None:
    table = result.summary_table()

    # Common bins are 0..3: bin 4 was unoccupied in B.
    assert table.loc[4, "mean_rate_a"] == pytest.approx(np.mean(F[:4]))
    assert table.loc[4, "mean_rate_b"] == pytest.approx(np.mean(G[:4]))
    assert table.loc[4, "mean_rate_difference"] == pytest.approx(
        np.mean(G[:4]) - np.mean(F[:4])
    )
    assert table["common_occupancy_a"].tolist() == [4.0, 4.0, 4.0]
    assert table["common_occupancy_b"].tolist() == [4.0, 4.0, 4.0]
    np.testing.assert_array_equal(table["n_common_bins"], result.n_common_bins)


def test_summary_table_means_are_nan_not_zero_without_common_bins(env) -> None:
    rates_a = _rates(env, [F, G, H], [1, 1, 1, 1, 1])
    rates_b = _rates(env, [F, G, H], [0, 0, 0, 0, 0])
    table = compare_spatial_rates(rates_a, rates_b).summary_table()

    assert table["mean_rate_a"].isna().all()
    assert table["mean_rate_difference"].isna().all()
    assert (table["common_occupancy_a"] == 0).all()
    assert set(table["status"]) == {"insufficient_common_bins"}


def test_summary_table_shows_missing_single_unit_label_as_na(env) -> None:
    from neurospatial.encoding import SpatialRateResult

    def single(unit_id):
        return SpatialRateResult(
            firing_rate=np.array(F),
            occupancy=np.ones(5),
            env=env,
            method="binned",
            bandwidth=5.0,
            unit_id=unit_id,
        )

    one = compare_spatial_rates(single(None), single(None)).summary_table()
    assert len(one) == 1
    assert one.index.name == "unit_id"
    assert one.index.isna().all()


# --- summary -----------------------------------------------------------------


def test_summary_is_flat_scalars(result) -> None:
    summary = result.summary()

    assert list(summary) == [
        "n_units",
        "n_units_scored",
        "median_correlation",
        "n_units_only_a",
        "n_units_only_b",
        "n_compared_bins",
        "method",
        "topology_differs",
    ]
    assert summary["n_units"] == 3
    assert summary["n_units_scored"] == 3
    assert summary["median_correlation"] == pytest.approx(np.median(result.correlation))
    assert summary["n_compared_bins"] == 5
    assert summary["method"] == "pearson"
    assert summary["topology_differs"] is False
    assert all(isinstance(v, (int, float, str, bool)) for v in summary.values()), (
        summary
    )


def test_summary_median_is_nan_when_nothing_is_scored(env) -> None:
    rates_a = _rates(env, [F, G, H], [1, 1, 1, 1, 1])
    rates_b = _rates(env, [F, G, H], [0, 0, 0, 0, 0])
    summary = compare_spatial_rates(rates_a, rates_b).summary()

    assert summary["n_units_scored"] == 0
    assert np.isnan(summary["median_correlation"])


# --- to_dataframe ------------------------------------------------------------


def test_to_dataframe_is_dense_per_unit_and_bin(result, env) -> None:
    frame = result.to_dataframe()

    assert len(frame) == 3 * env.n_bins
    assert frame.columns.tolist() == [
        "unit_id",
        "bin",
        "bin_b",
        "bin_center_x",
        "firing_rate_a",
        "firing_rate_b",
        "rate_difference",
        "occupancy_a",
        "occupancy_b",
        "occupied_a",
        "occupied_b",
        "common_support",
    ]
    assert frame["unit_id"].tolist() == [4] * 5 + [9] * 5 + [2] * 5
    assert frame["bin"].tolist() == list(range(5)) * 3
    np.testing.assert_array_equal(
        frame["rate_difference"].isna().to_numpy(), ~frame["common_support"].to_numpy()
    )
    np.testing.assert_allclose(
        frame["bin_center_x"].to_numpy()[:5], env.bin_centers[:, 0]
    )
    assert frame.attrs["units"]["firing_rate_a"] == "Hz"
    assert frame.attrs["units"]["occupancy_b"] == "s"
    assert frame.attrs["units"]["bin_center_x"] == "cm"
    assert frame.attrs["analysis_settings"] == result.analysis_settings


def test_to_dataframe_rows_match_result_arrays(result) -> None:
    frame = result.to_dataframe()

    np.testing.assert_array_equal(
        frame["firing_rate_a"].to_numpy(),
        result.firing_rates_a_on_bins.ravel(),
    )
    np.testing.assert_array_equal(
        frame["common_support"].to_numpy(), result.common_support.ravel()
    )
    np.testing.assert_array_equal(
        frame["occupied_b"].to_numpy(), np.tile(result.occupied_b, 3)
    )


def test_to_dataframe_is_a_writable_copy(result) -> None:
    before = result.firing_rates_a_on_bins.copy()
    frame = result.to_dataframe()

    frame["firing_rate_a"] = -1.0
    frame.attrs["analysis_settings"]["method"] = "edited"
    frame.loc[0, "rate_difference"] = 99.0

    np.testing.assert_array_equal(result.firing_rates_a_on_bins, before)
    assert result.analysis_settings["method"] == "pearson"
    assert result.to_dataframe()["firing_rate_a"].iloc[0] == F[0]


def test_to_dataframe_has_one_center_column_per_dimension(result_2d) -> None:
    frame = result_2d.to_dataframe()

    assert {"bin_center_x", "bin_center_y"} <= set(frame.columns)
    assert len(frame) == 3 * len(result_2d.bins_a)


# --- to_xarray ---------------------------------------------------------------


def test_to_xarray_layout_and_selection(result) -> None:
    pytest.importorskip("xarray")
    dataset = result.to_xarray()

    assert dict(dataset.sizes) == {"unit_id": 3, "bin": 5}
    np.testing.assert_array_equal(dataset["bin"], result.bins_a)
    assert {"bin_b", "bin_center_x"} <= set(dataset.coords)
    for name in ("firing_rate_a", "firing_rate_b", "rate_difference", "common_support"):
        assert dataset[name].dims == ("unit_id", "bin")
    for name in ("occupancy_a", "occupancy_b", "occupied_a", "occupied_b"):
        assert dataset[name].dims == ("bin",)
    for name in ("correlation", "n_common_bins", "status"):
        assert dataset[name].dims == ("unit_id",)
    assert dataset.attrs["method"] == "pearson"
    assert dataset.attrs["units"] == "cm"
    assert "software_version" in dataset.attrs
    import json

    assert json.loads(dataset.attrs["analysis_settings"])["method"] == "pearson"

    one = dataset.sel(unit_id=9)
    selected = result.sel(unit_id=9)
    np.testing.assert_array_equal(
        one["firing_rate_b"].to_numpy(), selected.firing_rates_b_on_bins[0]
    )
    np.testing.assert_array_equal(
        one["rate_difference"].to_numpy(), selected.rate_difference[0]
    )
    assert float(one["correlation"]) == selected.correlation[0]


def test_to_xarray_is_a_writable_copy(result) -> None:
    pytest.importorskip("xarray")
    before = result.firing_rates_a_on_bins.copy()
    dataset = result.to_xarray()

    dataset["firing_rate_a"].values[...] = -1.0
    dataset["occupancy_a"].values[...] = -1.0

    np.testing.assert_array_equal(result.firing_rates_a_on_bins, before)
    assert (result.occupancy_a == 1.0).all()


def test_to_xarray_without_xarray_names_the_extra(result, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "xarray":
            raise ImportError("no xarray")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"neurospatial\[xarray\]"):
        result.to_xarray()


# --- selection ---------------------------------------------------------------


def test_positional_and_label_selection(result) -> None:
    assert result[1].unit_ids.tolist() == [9]
    assert result[-1].unit_ids.tolist() == [2]
    np.testing.assert_array_equal(
        result.sel(unit_id=2).correlation, result.correlation[2:3]
    )
    assert [r.unit_ids[0] for r in result] == IDS
    assert len(result.sel(unit_id=9).to_dataframe()) == 5


def test_label_lookup_with_brackets_points_to_sel(result) -> None:
    with pytest.raises(TypeError, match=r"sel\(unit_id="):
        result["9"]
    with pytest.raises(KeyError, match="unit_id"):
        result.sel(unit_id=123)


# --- ownership ---------------------------------------------------------------


def test_arrays_are_read_only(result) -> None:
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable, field.name
    with pytest.raises(ValueError, match="read-only"):
        result.correlation[0] = 0.0


@pytest.mark.parametrize("how", ["pickle", "deepcopy"])
def test_copies_stay_read_only(result, how) -> None:
    restored = (
        pickle.loads(pickle.dumps(result)) if how == "pickle" else copy.deepcopy(result)
    )
    assert not restored.correlation.flags.writeable
    assert not restored.common_support.flags.writeable
    assert not restored.firing_rates_a_on_bins.flags.writeable
    np.testing.assert_array_equal(restored.correlation, result.correlation)


def test_selected_unit_arrays_stay_read_only(result) -> None:
    assert not result[0].firing_rates_a_on_bins.flags.writeable


def test_analysis_settings_are_independent_copies(result) -> None:
    settings = result.analysis_settings
    settings["method"] = "edited"
    settings["estimator_a"]["bandwidth"] = -1.0

    again = result.analysis_settings
    assert again["method"] == "pearson"
    assert again["estimator_a"]["bandwidth"] == 5.0


# --- plot --------------------------------------------------------------------


def test_scatter_plot_shows_scored_units_only(env) -> None:
    from matplotlib.axes import Axes

    rates_a = _rates(env, [F, G, [3.0] * 5], [1, 1, 1, 1, 1])
    rates_b = _rates(env, [G, F, H], [1, 1, 1, 1, 1])
    mixed = compare_spatial_rates(rates_a, rates_b)
    assert mixed.status.tolist() == ["scored", "scored", "constant_rates_a"]

    ax = mixed.plot()

    assert isinstance(ax, Axes)
    assert ax.get_title() == "2 of 3 units scored"
    (collection,) = ax.collections
    points = np.asarray(collection.get_offsets())
    assert points.shape == (2, 2)
    np.testing.assert_array_equal(points[:, 0], mixed.n_common_bins[:2])
    np.testing.assert_allclose(points[:, 1], mixed.correlation[:2])
    assert ax.get_xlabel() == "Common bins"
    assert ax.get_ylabel() == "Correlation (pearson)"


def test_scatter_plot_draws_on_the_given_axes(result) -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    assert result.plot(ax=ax) is ax
    plt.close("all")


def _single_unit_pair(env, unit_id=None):
    from neurospatial.encoding import SpatialRateResult

    def single(rates):
        return SpatialRateResult(
            firing_rate=np.array(rates),
            occupancy=np.ones(5),
            env=env,
            method="binned",
            bandwidth=5.0,
            unit_id=unit_id,
        )

    return compare_spatial_rates(single(F), single(G))


def _map_label(ax) -> str:
    return ax.figure.axes[-1].get_ylabel()


@pytest.mark.parametrize("explicit", [False, True])
def test_singular_comparison_plots_its_unit(env, explicit) -> None:
    import matplotlib.pyplot as plt

    one = _single_unit_pair(env)
    assert one.unit_ids.tolist() == [None]

    ax = one.plot(unit_id=None) if explicit else one.plot()

    assert _map_label(ax) == "Rate difference, B \u2212 A (Hz)"
    assert ax.get_title() != "1 of 1 units scored"
    plt.close("all")


def test_population_plot_draws_the_scatter(env) -> None:
    import matplotlib.pyplot as plt

    two = compare_spatial_rates(
        _rates(env, [F, G], [1] * 5, ids=[7, 9]),
        _rates(env, [G, F], [1] * 5, ids=[7, 9]),
    )
    assert two.plot().get_title() == "2 of 2 units scored"
    plt.close("all")


def test_explicit_label_plots_that_units_map(result) -> None:
    import matplotlib.pyplot as plt

    ax = result.plot(unit_id=9)
    assert _map_label(ax) == "Rate difference, B \u2212 A (Hz)"
    plt.close("all")


def test_missing_label_is_a_key_error(result) -> None:
    with pytest.raises(KeyError, match="unit_id None"):
        result.plot(unit_id=None)


def test_a_one_unit_population_plots_its_unit(env) -> None:
    import matplotlib.pyplot as plt

    one = compare_spatial_rates(
        _rates(env, [F, G], [1] * 5, ids=[7, 9]),
        _rates(env, [G, H], [1] * 5, ids=[9, 3]),
        unit_selection="intersection",
    )
    assert one.unit_ids.tolist() == [9]
    assert len(one) == 1

    ax = one.plot()

    assert _map_label(ax) == "Rate difference, B \u2212 A (Hz)"
    plt.close("all")


def test_difference_map_1d_and_2d(result, result_2d) -> None:
    from matplotlib.axes import Axes

    assert isinstance(result.plot(unit_id=9), Axes)
    ax = result_2d.plot(unit_id=result_2d.unit_ids[1])
    assert isinstance(ax, Axes)
    import matplotlib.pyplot as plt

    plt.close("all")


def test_difference_map_is_symmetric_and_nan_outside_common_support(
    result, monkeypatch
) -> None:
    from neurospatial.environment import Environment as Env

    seen = {}

    def spy(self, field, ax=None, **kwargs):
        seen["field"] = np.array(field)
        seen["kwargs"] = kwargs
        return ax if ax is not None else matplotlib.pyplot.subplots()[1]

    monkeypatch.setattr(Env, "plot_field", spy)
    result.plot(unit_id=4)

    expected = result.sel(unit_id=4).rate_difference[0]
    field = seen["field"]
    assert field.shape == (result.env.n_bins,)
    np.testing.assert_array_equal(np.isnan(field), np.isnan(expected))
    bound = np.nanmax(np.abs(expected))
    assert seen["kwargs"]["vmin"] == -bound
    assert seen["kwargs"]["vmax"] == bound
    assert seen["kwargs"]["cmap"] == "RdBu_r"
    assert "B" in seen["kwargs"]["colorbar_label"]
    matplotlib.pyplot.close("all")


def test_difference_map_with_no_common_bins_has_a_finite_color_range(
    env, monkeypatch
) -> None:
    from neurospatial.environment import Environment as Env

    seen = {}

    def spy(self, field, ax=None, **kwargs):
        seen.update(kwargs)
        return matplotlib.pyplot.subplots()[1]

    monkeypatch.setattr(Env, "plot_field", spy)
    empty = compare_spatial_rates(
        _rates(env, [F, G, H], [1] * 5), _rates(env, [F, G, H], [0] * 5)
    )
    empty.plot(unit_id=4)

    assert (seen["vmin"], seen["vmax"]) == (-1.0, 1.0)
    matplotlib.pyplot.close("all")


def test_difference_map_unknown_unit_is_a_key_error(result) -> None:
    with pytest.raises(KeyError, match="unit_id"):
        result.plot(unit_id=123)
