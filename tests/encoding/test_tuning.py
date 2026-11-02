"""Tuning curves: bins, time and spike accounting, gates and exports."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Intervals
from neurospatial._intervals import value_frame_bins
from neurospatial.encoding import (
    TuningCurveResult,
    TuningCurvesResult,
    compute_tuning_curve,
    compute_tuning_curves,
)
from neurospatial.encoding.tuning import tuning_bin_edges
from neurospatial.simulation import generate_poisson_spikes

EDGES = np.arange(11) / 10.0  # the same rounding as progress k / 20


@pytest.fixture
def runs():
    """Ten 2-s runs from progress 0 to 1, sampled at 10 Hz."""
    times = np.arange(200) / 10.0
    progress = (np.arange(200) % 20) / 20.0
    return times, progress


@pytest.fixture
def trials():
    """The ten runs as trials, eight correct and two errors."""
    return Intervals.from_arrays(
        np.arange(10) * 2.0,
        np.arange(1, 11) * 2.0,
        outcome=["correct"] * 8 + ["error"] * 2,
    )


def test_counts_and_occupancy_by_hand():
    """Each interval goes to the bin of its start value; each spike likewise."""
    times = np.array([0.0, 0.1, 0.3, 0.4, 0.7, 0.8, 1.0, 1.1, 1.2, 1.4])
    values = np.array([0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 0.2, 1.2, 2.2, 2.9])
    # 0.35 lies in [0.3, 0.4), which starts at 1.5; 1.25 in [1.2, 1.4) at 2.2.
    # 0.65 lies in [0.4, 0.7), which starts at 1.5, though its nearest sample
    # (0.7) is at 2.5: a spike takes the sample before it.
    spikes = np.array([0.35, 0.65, 1.25])

    curve = compute_tuning_curve(spikes, times, values, bin_edges=[0.0, 1.0, 2.0, 3.0])

    # bin 0: 0.1 + 0.2 + 0.1; bin 1: 0.1 + 0.3 + 0.1; bin 2: 0.1 + 0.2 + 0.2
    np.testing.assert_allclose(curve.occupancy, [0.4, 0.5, 0.5])
    np.testing.assert_array_equal(curve.spike_counts, [0.0, 2.0, 1.0])
    np.testing.assert_allclose(curve.firing_rate, [0.0, 4.0, 2.0])


def test_value_range_is_exclusion_boundary():
    """value_range is tiled exactly; its upper end is in; beyond it is out."""
    times = np.arange(6) / 10.0
    values = np.array([0.1, 1.0, 1.1, 0.6, 1.0, 0.3])

    curve = compute_tuning_curve(
        np.array([0.15]), times, values, bin_size=0.25, value_range=(0, 1)
    )

    assert curve.bin_edges.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    # 1.0 is in the last (closed) bin twice; 1.1 is excluded.
    np.testing.assert_allclose(curve.occupancy, [0.1, 0.0, 0.1, 0.2])
    assert curve.n_samples_outside_range == 1
    assert curve.analysis_settings["n_samples_outside_range"] == 1
    assert curve.spike_counts[-1] == 1.0  # the spike after the sample at 1.0
    assert curve.analysis_settings["edges_from"] == "value_range"

    with pytest.raises(ValueError, match=r"bin_size=0\.25 or 0\.333.*bin_edges="):
        compute_tuning_curve(
            np.array([0.15]), times, values, bin_size=0.3, value_range=(0, 1)
        )


def test_value_range_ends_exactly_at_its_limits():
    """A value at either end of value_range is analyzed, whatever lo is."""
    times = np.arange(3) / 10.0
    curve = compute_tuning_curve(
        np.array([]),
        times,
        np.array([1.93, 1.93, 0.08]),
        bin_size=0.05,
        value_range=(0.08, 1.93),
    )
    assert curve.bin_edges[0] == 0.08
    assert curve.bin_edges[-1] == 1.93
    assert curve.n_samples_outside_range == 0
    assert curve.occupancy.sum() == pytest.approx(0.2)


def test_derived_edges_on_shared_grid():
    """Without value_range, edges are multiples of bin_size covering the data."""
    times = np.arange(5) / 10.0
    first = compute_tuning_curve(
        np.array([]), times, np.array([0.3, 1.2, 2.2, 0.9, 1.7]), bin_size=0.5
    )
    second = compute_tuning_curve(
        np.array([]), times, np.array([1.1, 3.4, 2.0, 2.6, 1.9]), bin_size=0.5
    )

    assert first.bin_edges.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert second.bin_edges.tolist() == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert first.analysis_settings["edges_from"] == "data"


def test_derived_edges_cover_the_extremes():
    """Near-multiple extremes are covered: no finite value falls outside."""
    rng = np.random.default_rng(7)
    sizes = rng.choice([0.1, 0.3, 0.7, 1 / 3, 0.05, 2.5], size=10_000)
    k = rng.integers(-1000, 1000, size=(10_000, 2))
    nudge = rng.choice([-1, 0, 1], size=(10_000, 2))
    for size, (k_lo, k_hi), (n_lo, n_hi) in zip(sizes, k, nudge, strict=True):
        lo, hi = sorted((k_lo * size, k_hi * size))
        lo = np.nextafter(lo, lo + n_lo) if n_lo else lo
        hi = np.nextafter(hi, hi + n_hi) if n_hi else hi
        values = np.array([lo, hi])
        edges, _ = tuning_bin_edges(
            values, bin_size=size, bin_edges=None, value_range=None, owner="test"
        )
        assert edges[0] <= lo
        assert edges[-1] >= hi
        assert (value_frame_bins(values, edges, closed_last=True) >= 0).all()
    assert (
        tuning_bin_edges(
            np.array([-6.500000000000001]),
            bin_size=0.1,
            bin_edges=None,
            value_range=None,
            owner="test",
        )[0][0]
        <= -6.500000000000001
    )


def test_nan_values_not_bridged(runs):
    """Undefined values exclude their intervals and spikes, not fill them."""
    times, progress = runs
    undefined = progress.copy()
    undefined[40:45] = np.nan  # 0.5 s, shorter than max_gap
    spikes = np.array([4.15, 4.25, 12.05])

    full = compute_tuning_curve(spikes, times, progress, bin_edges=EDGES)
    gapped = compute_tuning_curve(spikes, times, undefined, bin_edges=EDGES)

    assert full.occupancy.sum() - gapped.occupancy.sum() == pytest.approx(0.5)
    # The undefined samples (progress 0 to 0.2 at 4.0-4.4 s) lose their own
    # intervals; the sample before them (progress 0.95 at 3.9 s) gains nothing.
    np.testing.assert_allclose(
        full.occupancy - gapped.occupancy, [0.2, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0]
    )
    assert gapped.spike_counts.sum() == 1.0
    assert full.spike_counts.sum() == 3.0


def test_epochs_intervals_equal_windows(runs, trials):
    """An Intervals as epochs is read through its windows; its ids are kept."""
    times, progress = runs
    spikes = times[progress >= 0.8] + 0.05
    correct = trials[trials.outcome == "correct"]

    by_intervals = compute_tuning_curve(
        spikes, times, progress, bin_edges=EDGES, epochs=correct
    )
    by_windows = compute_tuning_curve(
        spikes, times, progress, bin_edges=EDGES, epochs=correct.windows
    )

    np.testing.assert_array_equal(by_intervals.firing_rate, by_windows.firing_rate)
    np.testing.assert_array_equal(by_intervals.occupancy, by_windows.occupancy)
    assert by_intervals.analysis_settings["epoch_interval_ids"] == list(range(8))
    assert by_windows.analysis_settings["epoch_interval_ids"] is None
    assert by_intervals.occupancy.sum() == pytest.approx(16.0)


def test_empty_condition(runs, trials):
    """A condition with no trials gives NaN rates, zero occupancy and a warning."""
    times, progress = runs
    never = trials[trials.outcome == "never"]

    with pytest.warns(UserWarning, match="epochs selects no time"):
        curves = compute_tuning_curves(
            [times[::7], times[::5]], times, progress, bin_edges=EDGES, epochs=never
        )

    assert np.isnan(curves.firing_rates).all()
    np.testing.assert_array_equal(curves.occupancy, np.zeros(10))
    assert curves.analysis_settings["epoch_interval_ids"] == []


def _traversals(speed_cm_s: float, n_trials: int, seed: int):
    """Constant-speed runs along a 100 cm route, a unit tuned to progress 0.6."""
    duration = 100.0 / speed_cm_s
    per_trial = round(duration * 30)
    progress = np.tile(np.arange(per_trial) / per_trial, n_trials)
    times = np.arange(progress.size) / 30.0
    rate = 2.0 + 20.0 * np.exp(-((progress - 0.6) ** 2) / (2 * 0.08**2))
    return times, progress, generate_poisson_spikes(rate, times, seed=seed)


def test_speed_invariance():
    """Slow and fast traversals give the same curve within Poisson error."""
    edges = np.linspace(0, 1, 21)
    t_slow, p_slow, s_slow = _traversals(10.0, 20, seed=11)
    t_fast, p_fast, s_fast = _traversals(40.0, 80, seed=12)
    slow = compute_tuning_curve(s_slow, t_slow, p_slow, bin_edges=edges)
    fast = compute_tuning_curve(s_fast, t_fast, p_fast, bin_edges=edges)

    assert abs(int(np.argmax(slow.firing_rate)) - int(np.argmax(fast.firing_rate))) <= 1
    standard_error = np.sqrt(
        slow.firing_rate / slow.occupancy + fast.firing_rate / fast.occupancy
    )
    assert np.all(np.abs(slow.firing_rate - fast.firing_rate) <= 4 * standard_error)


def test_speed_gate():
    """A pause is excluded from spikes and occupancy only with min_speed."""
    times = np.arange(101) / 10.0
    progress = np.interp(times, [0.0, 4.0, 6.0, 10.0], [0.0, 0.5, 0.5, 1.0])
    speed = np.where((times >= 4.0) & (times < 6.0), 0.0, 10.0)
    spikes = np.array([2.05, 4.55, 5.05, 5.55])

    all_time = compute_tuning_curve(spikes, times, progress, bin_edges=EDGES)
    running = compute_tuning_curve(
        spikes, times, progress, bin_edges=EDGES, speed=speed, min_speed=1.0
    )

    pause_bin = 5  # progress 0.5 is the lower edge of [0.5, 0.6)
    difference = all_time.occupancy - running.occupancy
    assert difference[pause_bin] == pytest.approx(2.0)
    np.testing.assert_allclose(np.delete(difference, pause_bin), 0.0, atol=1e-12)
    assert all_time.spike_counts[pause_bin] - running.spike_counts[pause_bin] == 3.0
    assert running.analysis_settings["min_speed"] == 1.0
    assert running.analysis_settings["speed_source"] == "provided"

    with pytest.raises(ValueError, match="compute_speed"):
        compute_tuning_curve(spikes, times, progress, bin_edges=EDGES, min_speed=1.0)


def test_unsupported_bins_nan(runs):
    """Bins without enough occupancy are NaN, in the result and its table."""
    times, progress = runs
    low = np.where(progress < 0.5, progress, np.nan)  # never past 0.5
    curve = compute_tuning_curve(times[::3] + 0.01, times, low, bin_edges=EDGES)

    assert np.isnan(curve.firing_rate[5:]).all()
    assert np.isfinite(curve.firing_rate[:5]).all()
    frame = curve.to_dataframe()
    assert frame["firing_rate"].isna().tolist() == [False] * 5 + [True] * 5
    assert frame["spatial_support"].tolist() == [True] * 5 + [False] * 5

    thresholded = compute_tuning_curve(
        times[::3] + 0.01, times, low, bin_edges=EDGES, min_occupancy=2.5
    )
    # Each of the first five bins holds 2.0 s (ten runs x two samples).
    np.testing.assert_allclose(thresholded.occupancy[:5], 2.0)
    assert np.isnan(thresholded.firing_rate).all()


def test_bandwidth_requires_equal_widths(runs):
    times, progress = runs
    with pytest.raises(ValueError, match="equal width"):
        compute_tuning_curve(
            times[::4], times, progress, bin_edges=[0.0, 0.1, 0.3, 1.0], bandwidth=0.1
        )


def test_bandwidth_accepts_equal_bins_far_from_zero():
    """Equal bins of a variable near 1e5 (rounded edges) can be smoothed."""
    times = np.arange(200) / 10.0
    values = 100_000.0 + (np.arange(200) % 20) / 2.0
    curve = compute_tuning_curve(
        times[::3] + 0.01,
        times,
        values,
        bin_size=0.5,
        value_range=(100_000, 100_010),
        bandwidth=1.0,
    )
    assert np.isfinite(curve.firing_rate).all()


def test_smoothing_divides_smoothed_counts_by_smoothed_occupancy(runs):
    """Counts and occupancy are smoothed apart, then divided."""
    from scipy.ndimage import gaussian_filter1d

    times, progress = runs
    spikes = times[progress >= 0.8] + 0.05
    raw = compute_tuning_curve(spikes, times, progress, bin_edges=EDGES)
    smooth = compute_tuning_curve(
        spikes, times, progress, bin_edges=EDGES, bandwidth=0.15
    )

    sigma = 0.15 / 0.1
    expected = gaussian_filter1d(raw.spike_counts, sigma, mode="constant") / (
        gaussian_filter1d(raw.occupancy, sigma, mode="constant")
    )
    np.testing.assert_allclose(smooth.firing_rate, expected)
    np.testing.assert_array_equal(smooth.spike_counts, raw.spike_counts)
    assert smooth.method == "gaussian"
    assert raw.method == "binned"


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({}, TypeError, "exactly one of bin_size= or bin_edges="),
        ({"bin_size": 0.1, "bin_edges": EDGES}, TypeError, "exactly one"),
        (
            {"bin_edges": EDGES, "value_range": (0, 1)},
            TypeError,
            "value_range= is fixed by bin_edges=",
        ),
        ({"bin_edges": [0.0, 0.5, 0.5, 1.0]}, ValueError, "strictly increasing"),
        ({"bin_size": 0.1, "value_range": (1, 0)}, ValueError, "finite lo < hi"),
        ({"bin_size": -0.1}, ValueError, "bin_size must be finite and positive"),
    ],
)
def test_bin_arguments_are_checked(runs, kwargs, error, match):
    times, progress = runs
    with pytest.raises(error, match=match):
        compute_tuning_curve(times[::4], times, progress, **kwargs)


def test_infinite_values_refused(runs):
    times, progress = runs
    values = progress.copy()
    values[3] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        compute_tuning_curve(times[::4], times, values, bin_edges=EDGES)


def test_single_vs_population_agree(runs):
    times, progress = runs
    spikes = [times[progress >= 0.8] + 0.05, times[::3] + 0.02]
    population = compute_tuning_curves(
        spikes, times, progress, bin_edges=EDGES, bandwidth=0.1, unit_ids=["a", "b"]
    )
    for k, unit_id in enumerate(["a", "b"]):
        single = compute_tuning_curve(
            spikes[k], times, progress, bin_edges=EDGES, bandwidth=0.1
        )
        child = population[k]
        assert isinstance(child, TuningCurveResult)
        assert child.unit_id == unit_id
        np.testing.assert_array_equal(child.firing_rate, single.firing_rate)
        np.testing.assert_array_equal(child.spike_counts, single.spike_counts)
        np.testing.assert_array_equal(child.occupancy, single.occupancy)
    assert population.sel(unit_id="b").unit_id == "b"


@pytest.fixture
def distance_curves(runs):
    """Two units' curves against a variable with a name and units."""
    times, progress = runs
    distance = 100.0 * (1.0 - progress)
    return compute_tuning_curves(
        [times[::3] + 0.01, times[::5] + 0.01],
        times,
        distance,
        bin_size=10.0,
        value_range=(0, 100),
        variable="distance to goal",
        units="cm",
    )


def test_variable_and_units_travel_with_the_result(distance_curves):
    """Exports and the plot name the variable and its units."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    assert isinstance(distance_curves, TuningCurvesResult)

    frame = distance_curves.to_dataframe()
    assert len(frame) == 2 * 10
    np.testing.assert_allclose(frame["bin_center"][:10], np.arange(5, 100, 10))
    assert frame.attrs["units"]["bin_center"] == "cm"
    assert frame.attrs["variable"] == "distance to goal"

    table = distance_curves.summary_table()
    assert list(table.columns[:2]) == ["peak_firing_rate", "peak_location"]
    assert table.attrs["units"]["peak_location"] == "cm"

    ax = distance_curves.plot(0)
    assert ax.get_xlabel() == "distance to goal (cm)"
    assert ax.get_ylabel() == "Firing rate (Hz)"
    plt.close(ax.figure)

    summary = distance_curves[0].summary()
    assert summary["peak_location"] == pytest.approx(
        distance_curves[0].peak_location()[0]
    )


@pytest.mark.pynapple
def test_agrees_with_pynapple_where_conventions_coincide():
    """Spike counts equal pynapple's; occupancy within one sample period.

    With every spike exactly on a sample time, the closest sample (pynapple)
    and the sample before the spike (here) are the same. Pynapple counts the
    last sample's period; this package counts only sampling intervals.
    """
    nap = pytest.importorskip("pynapple")
    rng = np.random.default_rng(3)
    fs = 30.0
    times = np.arange(3000) / fs
    values = np.abs(np.sin(times / 7.0)) + 0.1 * rng.random(times.size)
    on_samples = np.sort(rng.choice(times[:-1], size=400, replace=False))
    edges = np.linspace(0.0, 1.1, 12)

    ours = compute_tuning_curve(on_samples, times, values, bin_edges=edges)
    theirs = nap.compute_tuning_curves(
        nap.TsGroup({0: nap.Ts(on_samples)}),
        nap.Tsd(t=times, d=values),
        bins=[edges],
        fs=fs,
        return_counts=True,
    )

    np.testing.assert_array_equal(ours.spike_counts, np.asarray(theirs)[0])
    occupancy = np.asarray(theirs.attrs["occupancy"]) / fs
    assert np.all(np.abs(ours.occupancy - occupancy) <= 1 / fs + 1e-9)


@pytest.mark.pynapple
def test_pynapple_values_use_their_time_support(runs):
    """A restricted Tsd's support is the epochs: short gaps between trials are out."""
    nap = pytest.importorskip("pynapple")
    times, progress = runs
    restricted = nap.Tsd(t=times, d=progress).restrict(
        nap.IntervalSet([0.0, 2.0], [1.5, 3.5])
    )

    from_series = compute_tuning_curve(
        np.array([1.0]), restricted.t, restricted, bin_edges=EDGES
    )
    from_array = compute_tuning_curve(
        np.array([1.0]), restricted.t, restricted.d, bin_edges=EDGES
    )

    # The 0.5 s between the two pieces is shorter than max_gap: only the
    # series' support keeps it out.
    assert from_series.occupancy.sum() == pytest.approx(3.0)
    assert from_array.occupancy.sum() == pytest.approx(3.5)


@pytest.mark.pynapple
def test_explicit_epochs_are_intersected_with_the_values_support(runs):
    """Explicit epochs keep the restricted series' gaps out and record their ids."""
    nap = pytest.importorskip("pynapple")
    times, progress = runs
    restricted = nap.Tsd(t=times, d=progress).restrict(
        nap.IntervalSet([0.0, 2.0], [1.5, 3.5])
    )
    trials = Intervals.from_arrays([0.0, 1.0], [1.0, 4.0], interval_ids=[7, 9])

    curve = compute_tuning_curve(
        np.array([1.0]), restricted.t, restricted, bin_edges=EDGES, epochs=trials
    )

    # [0, 1) and [1, 4) cover the recording, but only [0, 1.5) and [2, 3.5)
    # of it was kept in the series: 3.0 s, not 3.5 s.
    assert curve.occupancy.sum() == pytest.approx(3.0)
    assert curve.analysis_settings["epoch_interval_ids"] == [7, 9]


def test_to_xarray_names_the_variable(distance_curves):
    pytest.importorskip("xarray")
    dataset = distance_curves.to_xarray()
    assert dataset["firing_rate"].dims == ("unit_id", "bin")
    np.testing.assert_allclose(dataset["bin_center"], np.arange(5, 100, 10))
    assert dataset.attrs["variable"] == "distance to goal"


def test_per_trial_curves_weighted_by_occupancy_equal_the_pooled_curve(runs, trials):
    # One curve per trial on shared edges (select each trial by id): the
    # occupancy-weighted mean over trials is total spikes over total time,
    # which is the pooled curve.
    times, progress = runs
    spikes = np.sort(np.random.default_rng(3).uniform(0, 20, 300))
    pooled = compute_tuning_curve(
        spikes, times, progress, bin_edges=EDGES, epochs=trials, max_gap=None
    )
    per_trial = [
        compute_tuning_curve(
            spikes,
            times,
            progress,
            bin_edges=EDGES,
            epochs=trials.sel(interval_id=i),
            max_gap=None,
        )
        for i in trials.interval_ids
    ]
    occupancy = np.array([curve.occupancy for curve in per_trial])
    rates = np.array([curve.firing_rate for curve in per_trial])
    weighted = np.nansum(occupancy * rates, axis=0) / occupancy.sum(axis=0)
    np.testing.assert_allclose(weighted, pooled.firing_rate, rtol=1e-12)
    np.testing.assert_allclose(occupancy.sum(axis=0), pooled.occupancy, rtol=1e-12)


@pytest.mark.parametrize(
    ("field", "value", "recorded"),
    [
        ("bin_edges", EDGES, (EDGES * 100).tolist()),
        ("variable", "progress", "speed"),
        ("units", None, "cm"),
        ("bandwidth", 0.1, 0.2),
        ("bandwidth", None, 0.1),
        ("n_samples_outside_range", 3, 4),
    ],
)
def test_analysis_settings_cannot_contradict_a_field(field, value, recorded):
    """A recorded setting that describes another analysis is refused."""
    fields = {
        "firing_rate": np.ones(10),
        "occupancy": np.ones(10),
        "bin_edges": EDGES,
        "variable": "progress",
        field: value,
    }
    with pytest.raises(ValueError, match=rf"analysis_settings\['{field}'\]"):
        TuningCurveResult(**fields, analysis_settings={field: recorded})
    population = {**fields, "firing_rates": np.ones((2, 10))}
    del population["firing_rate"]
    with pytest.raises(ValueError, match=rf"analysis_settings\['{field}'\]"):
        TuningCurvesResult(**population, analysis_settings={field: recorded})


def test_analysis_settings_method_must_match_bandwidth():
    with pytest.raises(ValueError, match=r"analysis_settings\['method'\]"):
        TuningCurveResult(
            np.ones(10), np.ones(10), EDGES, analysis_settings={"method": "gaussian"}
        )


def test_computed_settings_agree_with_fields(runs):
    """A computed result, its units and a replace() of it all construct."""
    import dataclasses

    times, progress = runs
    curves = compute_tuning_curves(
        [times[::3] + 0.01],
        times,
        progress,
        bin_edges=EDGES,
        bandwidth=0.1,
        variable="progress",
    )
    settings = curves.analysis_settings
    for name in ("bin_edges", "bandwidth", "method", "variable", "units"):
        assert name in settings
    assert dataclasses.replace(curves).bandwidth == 0.1
    assert curves[0].analysis_settings["bin_edges"] == EDGES.tolist()
    with pytest.raises(ValueError, match="disagrees with the recorded"):
        dataclasses.replace(curves, bandwidth=0.2)


@pytest.mark.parametrize("name", ["firing_rate", "occupancy", "spike_counts"])
def test_negative_rates_occupancy_and_counts_are_refused(name):
    fields = {
        "firing_rate": np.ones(10),
        "occupancy": np.ones(10),
        "spike_counts": np.full(10, 0.5),  # smoothed counts may be fractional
    }
    TuningCurveResult(**fields, bin_edges=EDGES)
    fields[name] = fields[name].copy()
    fields[name][3] = -0.25
    with pytest.raises(ValueError, match=rf"{name} has 1 negative value"):
        TuningCurveResult(**fields, bin_edges=EDGES)


def test_warns_when_most_samples_fall_outside_the_bins(runs):
    """Percent values against 0-1 edges leave almost every sample out."""
    times, progress = runs
    percent = 200.0 * progress  # 0-190: a unit mismatch
    n_inside = int(np.count_nonzero(percent <= 1.0))  # progress 0 only
    n_outside = percent.size - n_inside
    edges = np.linspace(0, 1, 11)
    with pytest.warns(
        UserWarning,
        match=rf"{n_outside} of {percent.size} analyzed samples "
        rf"\({n_outside / percent.size:.1%}\).*outside bin_edges \[0, 1\]",
    ):
        curve = compute_tuning_curve(times[::3], times, percent, bin_edges=edges)
    assert curve.n_samples_outside_range == n_outside
    with pytest.warns(UserWarning, match="outside bin_edges"):
        compute_tuning_curves([times[::3]], times, percent, bin_edges=edges)


def test_no_out_of_range_warning_for_half_or_for_samples_outside_epochs(runs):
    """Half outside is not most; values outside epochs are not analyzed."""
    import warnings

    times, progress = runs
    half = np.where(np.arange(progress.size) % 2 == 0, progress, progress + 2.0)
    late = np.where(times < 10.0, progress, progress + 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        curve = compute_tuning_curve(times[::3], times, half, bin_edges=EDGES)
        compute_tuning_curve(
            times[::3], times, late, bin_edges=EDGES, epochs=[(0.0, 10.0)]
        )
    assert curve.n_samples_outside_range == progress.size // 2


def test_a_deliberately_narrow_range_does_not_warn(runs):
    """Firing within the first fifth of the route: most samples out, on purpose."""
    times, progress = runs
    narrow = compute_tuning_curve(
        times[::3], times, progress, bin_size=0.05, value_range=(0.0, 0.2)
    )
    assert narrow.n_samples_outside_range > progress.size // 2


def test_values_in_another_frame_warn(runs):
    """Same span, shifted origin: nearly every sample is outside."""
    times, progress = runs
    # The first 2% of samples stay in range, so the curve is not empty.
    shifted = np.where(
        np.arange(progress.size) < progress.size // 50, progress, progress + 50.0
    )
    with pytest.warns(UserWarning, match=r"outside bin_edges \[0, 1\]"):
        compute_tuning_curve(times[::3], times, shifted, bin_edges=EDGES)


def test_out_of_range_fraction_is_of_analyzed_samples(runs):
    """NaN samples and samples outside epochs are not in the denominator.

    In the first 10 s, 60 of 100 values are out of range; the other 100
    samples are NaN (or outside epochs). Counted, they would make it 30%.
    """
    times, progress = runs
    first = times < 10.0
    out = first & (np.arange(times.size) % 5 < 3)
    # In percent: outside the 0-1 edges, and spanning far more than them.
    values = np.where(out, 100.0 * progress + 2.0, progress)
    with pytest.warns(UserWarning, match=r"60 of 100 analyzed samples \(60\.0%\)"):
        compute_tuning_curve(
            times[::3], times, np.where(first, values, np.nan), bin_edges=EDGES
        )
    with pytest.warns(UserWarning, match=r"60 of 100 analyzed samples \(60\.0%\)"):
        compute_tuning_curve(
            times[::3], times, values, bin_edges=EDGES, epochs=[(0.0, 10.0)]
        )


def test_epochs_one_selected_trial_records_its_id(runs, trials):
    """sel with a list keeps an Intervals, so its id is recorded."""
    times, progress = runs
    spikes = times[::3] + 0.01
    one = trials.sel(interval_id=[3])
    by_list = compute_tuning_curve(spikes, times, progress, bin_edges=EDGES, epochs=one)
    record = trials.sel(interval_id=3)
    by_record = compute_tuning_curve(
        spikes, times, progress, bin_edges=EDGES, epochs=record
    )
    assert by_list.analysis_settings["epoch_interval_ids"] == [3]
    # A record carries no id, so none is recorded.
    assert by_record.analysis_settings["epoch_interval_ids"] is None
    np.testing.assert_array_equal(by_list.occupancy, by_record.occupancy)
    assert by_list.occupancy.sum() == pytest.approx(2.0)


def test_overlapping_epochs_count_shared_time_once(runs):
    """Overlapping trials [0, 1.5) and [1, 2) analyze [0, 2) once."""
    times, progress = runs
    spikes = np.array([0.25, 1.25, 1.75, 2.5])
    overlapping = Intervals.from_arrays([0.0, 1.0], [1.5, 2.0], interval_ids=[4, 9])
    by_trials = compute_tuning_curve(
        spikes, times, progress, bin_edges=EDGES, epochs=overlapping
    )
    by_window = compute_tuning_curve(
        spikes, times, progress, bin_edges=EDGES, epochs=[(0.0, 2.0)]
    )
    # Twenty 0.1-s samples of one run, one per 0.05 of progress: two per bin.
    np.testing.assert_allclose(by_trials.occupancy, np.full(10, 0.2), rtol=1e-12)
    np.testing.assert_array_equal(by_trials.occupancy, by_window.occupancy)
    np.testing.assert_array_equal(by_trials.spike_counts, by_window.spike_counts)
    assert by_trials.spike_counts.sum() == 3.0
    assert by_trials.analysis_settings["epoch_interval_ids"] == [4, 9]
