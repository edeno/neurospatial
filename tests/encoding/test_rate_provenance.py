"""Rate families preserve coverage, physical units and analysis provenance."""

import json

import numpy as np
import pytest

from neurospatial import Environment, encoding


@pytest.fixture(
    params=["spatial", "directional", "view", "object_vector", "egocentric"]
)
def rate_family(request):
    """One recording and the explicit public arguments for each rate family."""
    family = request.param
    times = np.arange(25) / 4
    positions = np.column_stack([5 + np.arange(25) % 4 * 5, np.full(25, 10.0)])
    headings = np.arange(25) * np.pi / 12
    positions[4] = np.nan
    headings[4] = np.nan
    env = Environment.from_grid_mask(
        np.ones((6, 6), dtype=bool), grid_edges=(np.arange(7) * 5.0,) * 2
    )
    env = env.with_metadata(units="cm")
    trains = [np.array([0.3, 1.1, 1.6, 3.1, 4.1]), np.array([0.4, 1.8, 3.4])]
    kwargs = {"epochs": [[0, 2], [3, 6]], "spike_window": (0.25, 5.5), "max_gap": 0.3}
    if family == "directional":
        data = (times, headings)
        kwargs.update(bin_size=np.pi / 6, bandwidth=np.pi / 8)
    else:
        data = (times, positions)
        kwargs.update(method="diffusion_kde", bandwidth=2.0, min_occupancy=0.4)
        if family == "view":
            data += (headings,)
            kwargs.update(view_distance=2.0)
        elif family in ("object_vector", "egocentric"):
            if family == "egocentric":
                data += (headings,)
            data += (np.array([[15.0, 15.0]]),)
            kwargs.update(
                distance_range=(0.0, 25.0), n_distance_bins=5, n_direction_bins=8
            )

    def compute(spikes, *, single=False, **overrides):
        function = getattr(encoding, f"compute_{family}_rate" + ("" if single else "s"))
        args = (spikes, *data) if family == "directional" else (env, spikes, *data)
        return function(*args, **(kwargs | overrides))

    return family, compute, trains


def test_rate_support_and_provenance_match_the_actual_analysis(rate_family):
    family, compute, trains = rate_family
    result = compute(trains, unit_ids=["first", "second"])
    settings = result.analysis_settings
    assert settings["epochs"] == [[0, 2], [3, 6]]
    assert settings["spike_window"] == [[0.25, 5.5]]
    assert settings["max_gap"] == 0.3
    runs = np.asarray(settings["observed_intervals"])
    # Raw occupancy must account for exactly the intervals recorded in provenance.
    assert np.sum(runs[:, 1] - runs[:, 0]) == pytest.approx(result.occupancy.sum())
    assert np.all((runs[:, 1] <= 2) | (runs[:, 0] >= 3))
    assert np.all((runs[:, 1] <= 1) | (runs[:, 0] >= 1.25))
    assert runs.min() >= 0.25
    assert runs.max() <= 5.5
    expected = (result.occupancy > 0) & (
        result.occupancy >= (settings["min_occupancy"] or 0)
    )
    expected &= np.isfinite(result.firing_rates).all(axis=0)
    np.testing.assert_array_equal(result.spatial_support, expected)
    assert expected.any() and not expected.all()
    if family == "directional":
        assert settings["angle_unit"] == "rad"
    elif family == "view":
        assert settings["gaze_model"] == "fixed_distance"
        assert settings["view_distance"] == 2.0
    elif family in ("object_vector", "egocentric"):
        assert settings["direction_frame"] == (
            "allocentric" if family == "object_vector" else "egocentric"
        )
        assert result.env.units == "cm"

    for i, spikes in enumerate(trains):
        single = compute(spikes, single=True)
        child = result[i]
        np.testing.assert_allclose(single.firing_rate, child.firing_rate, atol=1e-10)
        assert single.analysis_settings == settings == child.analysis_settings
        assert child.unit_id == ["first", "second"][i]
        child.analysis_settings["epochs"][0][0] = -10
        assert result.analysis_settings["epochs"][0][0] == 0


def test_rate_metadata_survives_tables_and_netcdf(rate_family, tmp_path):
    _, compute, trains = rate_family
    result = compute(trains, unit_ids=["first", "second"])
    for value in (result, result[0]):
        for frame in (value.to_dataframe(), value.summary_table()):
            assert frame.attrs["analysis_settings"] == value.analysis_settings
            frame.attrs["analysis_settings"]["epochs"][0][0] = -10
            assert value.analysis_settings["epochs"][0][0] == 0
        frame = value.to_dataframe()
        np.testing.assert_array_equal(
            frame.spatial_support[: len(value.occupancy)], value.spatial_support
        )
        pytest.importorskip("xarray")
        dataset = value.to_xarray()
        assert json.loads(dataset.attrs["analysis_settings"]) == value.analysis_settings
        path = tmp_path / "rates.nc"
        dataset.to_netcdf(path)
        import xarray as xr

        with xr.open_dataset(path) as restored:
            assert (
                json.loads(restored.attrs["analysis_settings"])
                == value.analysis_settings
            )
            np.testing.assert_array_equal(
                restored.spatial_support, value.spatial_support
            )


def test_empty_populations_keep_observation_provenance(rate_family):
    _, compute, trains = rate_family
    populated = compute(trains)
    empty = compute([])
    assert empty.analysis_settings == populated.analysis_settings
    np.testing.assert_allclose(empty.occupancy, populated.occupancy)
    assert not empty.spatial_support.any()
    assert empty.to_dataframe().empty


@pytest.mark.parametrize("family", ["view", "object_vector", "egocentric"])
@pytest.mark.parametrize("method", ["diffusion_kde", "gaussian_kde"])
@pytest.mark.parametrize("declared_units", [True, False])
def test_physical_bandwidth_default_is_invariant_under_unit_conversion(
    family, method, declared_units
):
    times = np.arange(161) / 4
    phase = 0.073 + np.arange(len(times)) * np.pi / 16
    radius = 8.13 + 3.77 * np.sin(phase / 3)
    positions = np.column_stack(
        [20 + radius * np.cos(phase), 20 + radius * np.sin(phase)]
    )
    headings = phase + np.pi / 2 + 0.113
    objects = np.array([[20.0, 20.0]])
    trains = [times[::3] + 0.05, times[::5] + 0.01]
    reference = None
    for scale, units in [(1.0, "cm"), (0.01, "m"), (10.0, "mm")]:
        env = Environment.from_grid_mask(
            np.ones((8, 8), dtype=bool), grid_edges=(np.arange(9) * 5.0 * scale,) * 2
        )
        env = env.with_metadata(units=units if declared_units else None)
        # Object-vector analyses also work without a Cartesian environment.
        input_env = env if declared_units or family == "view" else None
        args = (input_env, trains, times, positions * scale)
        kwargs = {"method": method}
        if family == "view":
            args += (headings,)
            kwargs["view_distance"] = 2.0 * scale
        else:
            if family == "egocentric":
                args += (headings,)
            args += (objects * scale,)
            kwargs.update(
                distance_range=(0.0, 20.0 * scale),
                n_distance_bins=4,
                n_direction_bins=8,
            )
        result = getattr(encoding, f"compute_{family}_rates")(*args, **kwargs)
        if declared_units:
            assert result.analysis_settings["bandwidth"] == pytest.approx(5.0 * scale)
        assert result.env.units == (units if declared_units else None)
        if reference is None:
            reference = result
        else:
            assert result.analysis_settings["bandwidth"] == pytest.approx(
                reference.analysis_settings["bandwidth"] * scale
            )
            np.testing.assert_allclose(
                result.occupancy, reference.occupancy, atol=1e-10
            )
            np.testing.assert_allclose(
                result.firing_rates, reference.firing_rates, atol=1e-9, rtol=1e-9
            )
            np.testing.assert_array_equal(
                result.spatial_support, reference.spatial_support
            )
        explicit = getattr(encoding, f"compute_{family}_rates")(
            *args, **kwargs, bandwidth=2.0 * scale
        )
        assert explicit.analysis_settings["bandwidth"] == pytest.approx(2.0 * scale)
