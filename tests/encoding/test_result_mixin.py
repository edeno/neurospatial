"""Tests for the unified result-object surface (ResultMixin).

Covers the directional-result plot/correlation surface, tidy ``to_dataframe``
composability across heterogeneous result types, and the additive-only
guarantee that pre-existing spatial-result accessors keep working.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless test runs

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment
from neurospatial._results import ResultMixin
from neurospatial.encoding._base import SpatialResultMixin
from neurospatial.encoding.spatial import (
    DirectionalPlaceFields,
    PlaceFieldsResult,
    compute_spatial_rate,
)


@pytest.fixture
def linear_env() -> Environment:
    """A 1-D linearized environment for directional-field fixtures."""
    return Environment.from_samples(np.linspace(0, 9, 100)[:, None], bin_size=1.0)


@pytest.fixture
def spatial_result() -> object:
    """A single-neuron SpatialRateResult on a 2-D environment."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (500, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    times = np.linspace(0, 50, 500)
    spike_times = np.sort(rng.uniform(0, 50, 30))
    return compute_spatial_rate(env, spike_times, times, positions, bandwidth=10.0)


def _directional(env: Environment, rate_a, rate_b) -> DirectionalPlaceFields:
    n = env.n_bins
    return DirectionalPlaceFields(
        firing_rates={"fwd": np.asarray(rate_a), "rev": np.asarray(rate_b)},
        occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        env=env,
        labels=("fwd", "rev"),
    )


def test_spatial_result_mixin_reparents_under_result_mixin() -> None:
    assert issubclass(SpatialResultMixin, ResultMixin)


def test_directional_place_fields_plot(linear_env: Environment) -> None:
    n = linear_env.n_bins
    result = _directional(
        linear_env, np.linspace(1.0, 5.0, n), np.linspace(5.0, 1.0, n)
    )
    ax = result.plot()
    # One line artist per direction.
    assert len(ax.get_lines()) == len(result.labels) == 2
    # plot() returns an Axes that can be composed into multi-panel figures.
    import matplotlib.pyplot as plt

    assert isinstance(ax, plt.Axes)


def test_directional_place_fields_compare(linear_env: Environment) -> None:
    n = linear_env.n_bins
    base = np.linspace(1.0, 5.0, n)

    # Identical maps -> correlation 1.0, directionality index 0.0.
    same = _directional(linear_env, base, base.copy())
    assert np.isclose(same.correlation("fwd", "rev"), 1.0)
    assert np.isclose(same.directionality_index("fwd", "rev"), 0.0)

    # Reversed map -> perfectly anti-correlated (-1.0).
    flipped = _directional(linear_env, base, base[::-1].copy())
    assert np.isclose(flipped.correlation("fwd", "rev"), -1.0)
    # Directionality index is positive when the maps differ per bin.
    assert flipped.directionality_index("fwd", "rev") > 0.0

    # Unknown label is a clear error, not a silent NaN.
    with pytest.raises(KeyError):
        same.correlation("fwd", "nope")


def test_directional_summary_includes_correlation(linear_env: Environment) -> None:
    n = linear_env.n_bins
    result = _directional(
        linear_env, np.linspace(1.0, 5.0, n), np.linspace(5.0, 1.0, n)
    )
    s = result.summary()
    assert s["n_directions"] == 2
    assert s["n_bins"] == n
    assert np.isclose(s["correlation"], -1.0)
    assert "peak_fwd" in s and "peak_rev" in s


def test_result_to_dataframe_tidy(
    spatial_result: object, linear_env: Environment
) -> None:
    n = linear_env.n_bins
    directional = _directional(
        linear_env, np.linspace(1.0, 5.0, n), np.linspace(5.0, 1.0, n)
    )

    df_spatial = spatial_result.to_dataframe()  # type: ignore[attr-defined]
    df_directional = directional.to_dataframe()

    # Both are tidy/long: one value column "firing_rate", explicit id columns.
    assert "firing_rate" in df_spatial.columns
    assert "firing_rate" in df_directional.columns

    # Heterogeneous results concat cleanly into one frame.
    combined = pd.concat(
        [
            df_spatial.assign(source="spatial"),
            df_directional.assign(source="directional"),
        ],
        ignore_index=True,
    )
    assert len(combined) == len(df_spatial) + len(df_directional)
    assert set(combined["source"]) == {"spatial", "directional"}
    # Directional rows are 2 directions x n_bins.
    assert len(df_directional) == 2 * n


def test_place_fields_result_surface() -> None:
    result = PlaceFieldsResult(fields=[np.array([0, 1]), np.array([5])])
    s = result.summary()
    assert s["n_fields"] == 2
    assert s["total_bins"] == 3

    df = result.to_dataframe()
    assert df["field"].tolist() == [0, 0, 1]
    assert df["bin"].tolist() == [0, 1, 5]

    # Empty / excluded result yields an empty tidy frame with the columns.
    excluded = PlaceFieldsResult(
        fields=[], excluded_reason="mean_rate_above_threshold", n_excluded=1
    )
    assert excluded.summary()["n_excluded"] == 1
    empty_df = excluded.to_dataframe()
    assert list(empty_df.columns) == ["field", "bin"]
    assert len(empty_df) == 0


def test_existing_accessors_preserved(spatial_result: object) -> None:
    # Pre-existing SpatialRateResult accessors must remain unchanged (additive).
    env = spatial_result.env  # type: ignore[attr-defined]
    assert spatial_result.firing_rate.shape == (env.n_bins,)  # type: ignore[attr-defined]
    assert spatial_result.occupancy.shape == (env.n_bins,)  # type: ignore[attr-defined]
    assert float(spatial_result.spatial_information()) >= 0.0  # type: ignore[attr-defined]
    assert 0.0 <= float(spatial_result.sparsity()) <= 1.0  # type: ignore[attr-defined]
    # Peak helpers from SpatialResultMixin still work.
    assert spatial_result.peak_location().shape == (2,)  # type: ignore[attr-defined]
    assert float(spatial_result.peak_firing_rate()) >= 0.0  # type: ignore[attr-defined]
    # plot() still delegates to env.plot_field and returns an Axes.
    import matplotlib.pyplot as plt

    ax = spatial_result.plot()  # type: ignore[attr-defined]
    assert isinstance(ax, plt.Axes)


def test_result_mixin_base_raises_for_unimplemented() -> None:
    class Empty(ResultMixin):
        pass

    empty = Empty()
    with pytest.raises(NotImplementedError):
        empty.to_dataframe()
    with pytest.raises(NotImplementedError):
        empty.summary()
    with pytest.raises(NotImplementedError):
        empty.plot()
