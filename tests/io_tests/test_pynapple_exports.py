"""Export of decode results and behavior records with ``to_pynapple``.

Membership is checked against neurospatial's own half-open windows: pynapple's
``restrict`` must select exactly the samples that ``in_epochs`` selects with
the records' ``[start_time, end_time)`` windows, including samples that sit
exactly on a window boundary.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from neurospatial.behavior import (
    Crossing,
    Lap,
    RegionVisit,
    Trial,
    in_epochs,
    labels_to_epochs,
)
from neurospatial.decoding import decode_session, decode_session_summary
from neurospatial.io import from_pynapple, to_pynapple

HAS_PYNAPPLE = importlib.util.find_spec("pynapple") is not None
needs_pynapple = pytest.mark.skipif(
    not HAS_PYNAPPLE, reason="requires the pynapple extra"
)


def _half_open(closed_rows) -> np.ndarray:
    """The half-open rows holding the same floats as pynapple's ``[start, end]``."""
    rows = np.array(closed_rows, dtype=np.float64).reshape(-1, 2)
    rows[:, 1] = np.nextafter(rows[:, 1], np.inf)
    return rows


@pytest.fixture(scope="module")
def gapped_decode(continuous_recording):
    """A decode over two epochs separated by a 20 s gap, and its spikes."""
    r = continuous_recording
    epochs = np.array([[0.0, 50.0], [70.0, 120.0]])
    result = decode_session(
        r.env,
        [r.spike_times],
        r.times,
        r.positions,
        method="binned",
        dt=0.25,
        epochs=epochs,
    )
    return result, epochs, r.spike_times


# ---------------------------------------------------------------------------
# Decode results
# ---------------------------------------------------------------------------


@needs_pynapple
@pytest.mark.pynapple
def test_decoded_track_time_support_is_the_decoded_epochs(gapped_decode):
    import pynapple as nap

    result, epochs, spikes = gapped_decode
    track = to_pynapple(result)

    # The 50-70 s gap was not decoded, so it is not part of the time support.
    np.testing.assert_array_equal(_half_open(track.time_support.values), epochs)
    assert list(track.columns) == ["x", "y"]
    np.testing.assert_array_equal(track.t, result.times)
    np.testing.assert_array_equal(track.values, result.map_position)
    kept = nap.Ts(t=spikes).restrict(track.time_support).t
    np.testing.assert_array_equal(kept, spikes[in_epochs(spikes, epochs)])


@needs_pynapple
@pytest.mark.pynapple
def test_posterior_export_keeps_mass_per_bin_and_bin_centers(gapped_decode):
    result, epochs, _ = gapped_decode
    posterior = to_pynapple(result, quantity="posterior")

    assert posterior.shape == result.posterior.shape
    np.testing.assert_array_equal(posterior.values, result.posterior)
    np.testing.assert_array_equal(posterior.columns, np.arange(result.env.n_bins))
    np.testing.assert_allclose(np.nansum(posterior.values, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(posterior.bin_center_x, result.env.bin_centers[:, 0])
    np.testing.assert_array_equal(posterior.bin_center_y, result.env.bin_centers[:, 1])
    np.testing.assert_array_equal(_half_open(posterior.time_support.values), epochs)
    # Selecting bins by place keeps the matching posterior columns.
    west = posterior[:, posterior.bin_center_x < 50.0]
    np.testing.assert_array_equal(
        west.values, result.posterior[:, result.env.bin_centers[:, 0] < 50.0]
    )
    times, values, support = from_pynapple(posterior)
    np.testing.assert_array_equal(times, result.times)
    np.testing.assert_array_equal(values, result.posterior)
    np.testing.assert_array_equal(support, epochs)
    # The export is a copy: editing it never changes the result.
    posterior.values[0, 0] = -1.0
    assert result.posterior[0, 0] >= 0.0


@needs_pynapple
@pytest.mark.pynapple
def test_mean_track_and_summary_track(continuous_recording, gapped_decode):
    result, epochs, _ = gapped_decode
    mean = to_pynapple(result, quantity="mean_position")
    np.testing.assert_array_equal(mean.values, result.mean_position)

    r = continuous_recording
    summary = decode_session_summary(
        r.env,
        [r.spike_times],
        r.times,
        r.positions,
        method="binned",
        dt=0.25,
        epochs=epochs,
    )
    track = to_pynapple(summary)
    np.testing.assert_array_equal(track.values, summary.map_position)
    np.testing.assert_array_equal(_half_open(track.time_support.values), epochs)


def test_posterior_export_refusals(continuous_recording, gapped_decode):
    result, _, _ = gapped_decode
    r = continuous_recording
    summary = decode_session_summary(
        r.env, [r.spike_times], r.times, r.positions, method="binned", dt=0.25
    )
    with pytest.raises(ValueError, match="DecodingSummary, which keeps no posterior"):
        to_pynapple(summary, quantity="posterior")
    with pytest.raises(TypeError, match="columns= with quantity='posterior'"):
        to_pynapple(result, quantity="posterior", columns=["a"])
    with pytest.raises(ValueError, match="quantity must be one of"):
        to_pynapple(result, quantity="map")
    with pytest.raises(TypeError, match="quantity='posterior' with a values array"):
        to_pynapple(result.times, result.map_position, quantity="posterior")


# ---------------------------------------------------------------------------
# Interval records
# ---------------------------------------------------------------------------

# Samples on and between record boundaries, including exactly at each stop.
SAMPLES = np.array([0.0, 0.5, 1.0, 2.0, 2.5, np.nextafter(3.0, 0.0), 3.0, 3.5, 4.0])

LAPS = [  # deliberately out of time order; the first two touch at 1.0
    Lap(3.0, 4.0, "unknown", np.nan),
    Lap(0.0, 1.0, "clockwise", 0.8),
    Lap(1.0, 2.5, "counter-clockwise", 0.6),
]


def _assert_same_membership(iset, windows) -> None:
    """pynapple restrict selects what in_epochs selects, per interval and overall."""
    import pynapple as nap

    samples = nap.Ts(t=SAMPLES)
    for i, window in enumerate(windows):
        expected = SAMPLES[in_epochs(SAMPLES, window)]
        np.testing.assert_array_equal(samples.restrict(iset[i]).t, expected)
    expected = SAMPLES[in_epochs(SAMPLES, windows)]
    np.testing.assert_array_equal(samples.restrict(iset).t, expected)


@needs_pynapple
@pytest.mark.pynapple
def test_laps_become_an_interval_set_with_metadata():
    iset = to_pynapple(LAPS)

    windows = [[0.0, 1.0], [1.0, 2.5], [3.0, 4.0]]
    np.testing.assert_array_equal(_half_open(iset.values), windows)
    assert list(iset.direction) == ["clockwise", "counter-clockwise", "unknown"]
    np.testing.assert_array_equal(iset.overlap_score, [0.8, 0.6, np.nan])
    _assert_same_membership(iset, windows)
    clockwise = iset[iset.direction == "clockwise"]
    np.testing.assert_array_equal(_half_open(clockwise.values), [[0.0, 1.0]])


@needs_pynapple
@pytest.mark.pynapple
@pytest.mark.parametrize(
    ("records", "expected"),
    [
        (
            [
                RegionVisit(0, 0, "reward", 0, 5, 0.0, 1.0, True, True),
                RegionVisit(1, 1, "reward", 9, 12, 2.0, 3.0, False, True),
            ],
            {
                "visit_id": [0, 1],
                "run_id": [0, 1],
                "region_name": ["reward", "reward"],
                "start_index": [0, 9],
                "stop_index": [5, 12],
                "entry_observed": [True, False],
                "exit_observed": [True, True],
            },
        ),
    ],
    ids=["visits"],
)
def test_records_keep_every_field_as_metadata(records, expected):
    import dataclasses

    iset = to_pynapple(records)

    windows = [[r.start_time, r.end_time] for r in records]
    np.testing.assert_array_equal(_half_open(iset.values), windows)
    for name, column in expected.items():
        assert list(iset.metadata[name]) == column, name
    # Every scalar field is exported, so nothing about a record is lost.
    fields = {f.name for f in dataclasses.fields(records[0]) if f.init} - {
        "start_time",
        "end_time",
        "bins",
    }
    assert set(iset.metadata_columns) == fields
    _assert_same_membership(iset, windows)


@needs_pynapple
@pytest.mark.pynapple
def test_label_epochs_become_an_interval_set_with_labels():
    times = np.arange(9) * 0.5
    result = labels_to_epochs(times, ["a", "a", "b", "b", "a", "a", "a", "b", "b"])

    iset = to_pynapple(result)

    windows = np.concatenate([result["a"], result["b"]])
    windows = windows[np.argsort(windows[:, 0])]
    np.testing.assert_array_equal(_half_open(iset.values), windows)
    assert list(iset.label) == ["a", "b", "a", "b"]
    _assert_same_membership(iset, windows)
    np.testing.assert_array_equal(
        _half_open(iset[iset.label == "a"].values), result["a"]
    )


@needs_pynapple
@pytest.mark.pynapple
def test_empty_inputs_that_carry_their_type():
    # Labels that are all missing leave no epoch, but the result still says
    # what it holds: an IntervalSet with a label column.
    nothing = to_pynapple(labels_to_epochs(np.arange(4) * 0.5, [None] * 4))
    assert len(nothing) == 0
    assert list(nothing.metadata_columns) == ["label"]
    # An empty list with values is an empty time series, not empty records.
    assert len(to_pynapple([], [])) == 0


@needs_pynapple
@pytest.mark.pynapple
def test_records_without_duration_are_dropped_with_a_warning():
    laps = [
        Lap(0.0, 1.0, "clockwise", 0.5),
        Lap(2.0, 2.0, "unknown", 0.1),
        Lap(3.0, 4.0, "counter-clockwise", 0.7),
    ]
    with pytest.warns(UserWarning, match=r"dropped 1 of 3 Lap record\(s\) without"):
        iset = to_pynapple(laps)
    np.testing.assert_array_equal(_half_open(iset.values), [[0.0, 1.0], [3.0, 4.0]])
    # The remaining records keep their own metadata.
    assert list(iset.direction) == ["clockwise", "counter-clockwise"]
    with pytest.warns(UserWarning, match=r"dropped 1 of 1 Lap"):
        assert len(to_pynapple([laps[1]])) == 0


@needs_pynapple
@pytest.mark.pynapple
def test_intervals_export_ids_and_fields():
    from neurospatial import Intervals
    from neurospatial._intervals import stop_to_pynapple_end

    trials = Intervals(
        [
            Trial(2.0, 3.0, "home", None, "returned"),
            Trial(0.0, 1.0, "home", "left", "reached"),
            Trial(4.0, 4.0, "home", None, "censored"),
        ],
        interval_ids=[7, 3, 9],
    )
    with pytest.warns(UserWarning, match=r"dropped 1 of 3 interval record\(s\)"):
        iset = to_pynapple(trials)

    timed = trials[trials.durations > 0]
    np.testing.assert_array_equal(iset.start, timed.start)
    np.testing.assert_array_equal(iset.end, stop_to_pynapple_end(timed.stop))
    assert list(iset.metadata_columns) == ["interval_id", *trials.fields]
    assert list(iset.interval_id) == [3, 7]
    assert list(iset.end_region) == ["left", None]
    assert list(iset.outcome) == ["reached", "returned"]
    _assert_same_membership(iset, timed.windows)

    # An empty selection keeps its columns.
    none = to_pynapple(trials[[]])
    assert len(none) == 0
    assert list(none.metadata_columns) == ["interval_id", *trials.fields]


def test_trial_records_are_refused_with_the_fix():
    trials = [Trial(0.0, 1.0, "home", "left", "reached")]
    with pytest.raises(TypeError, match=r"neurospatial\.Intervals\(trials\)"):
        to_pynapple(trials)


def test_interval_record_refusals():
    visits = [
        RegionVisit(0, 0, "a", 0, 5, 0.0, 2.0, True, True),
        RegionVisit(1, 0, "b", 3, 6, 1.0, 3.0, True, True),
    ]
    with pytest.raises(ValueError, match=r"1 RegionVisit record\(s\) overlap"):
        to_pynapple(visits)
    with pytest.raises(TypeError, match="Lap records mixed with Trial"):
        to_pynapple([LAPS[0], Trial(0.0, 1.0, "home", None, "returned")])
    with pytest.raises(TypeError, match="epochs with behavior records"):
        to_pynapple(LAPS, epochs=(0.0, 5.0))
    with pytest.raises(TypeError, match="values with behavior records"):
        to_pynapple(LAPS, np.zeros(3))
    with pytest.raises(ValueError, match="empty list carries no record type"):
        to_pynapple([])


@needs_pynapple
@pytest.mark.pynapple
def test_detected_visits_and_crossings_round_trip():
    """Detector output exports with membership matching the half-open windows."""
    import pynapple as nap
    from shapely.geometry import box

    from neurospatial import Environment, Regions
    from neurospatial.behavior import (
        detect_region_crossings,
        detect_region_visits,
    )

    env = Environment.from_grid_mask(
        active_mask=np.ones((6, 1), dtype=bool),
        grid_edges=(np.arange(7, dtype=float), np.array([0.0, 1.0])),
    )
    for name, left in [("A", 0), ("B", 3), ("C", 5)]:
        regions = Regions(env.regions.values())
        regions.add(name, polygon=box(left, 0, left + 1, 1))
        env = env.with_regions(regions)
    times = np.array([0.0, 0.2, 0.7, 1.2, 2.0, 2.5, 3.0, 4.0])
    bins = np.array([1, 0, 0, 1, 3, 3, 4, 5])
    visits = detect_region_visits(
        bins, times, env, region_names=["A", "B", "C"], max_gap=None
    )

    # The visit to C is its last sample alone: no time to export.
    with pytest.warns(UserWarning, match=r"dropped 1 of 3 RegionVisit"):
        iset = to_pynapple(visits)
    timed = [v for v in visits if v.end_time > v.start_time]
    assert list(iset.region_name) == ["A", "B"]
    samples = nap.Ts(t=times)
    for i, visit in enumerate(timed):
        window = (visit.start_time, visit.end_time)
        np.testing.assert_array_equal(
            samples.restrict(iset[i]).t, times[in_epochs(times, window)]
        )
        # The selected samples are exactly the visit's inside-region samples.
        np.testing.assert_array_equal(
            samples.restrict(iset[i]).t,
            times[visit.start_index : visit.stop_index],
        )

    crossings = detect_region_crossings(bins, times, env, region_name="B", max_gap=None)
    group = to_pynapple(crossings, epochs=(times[0], np.nextafter(times[-1], 5)))
    assert list(group.direction) == ["entry", "exit"]
    np.testing.assert_array_equal(group[0].t, [2.0])
    np.testing.assert_array_equal(group[1].t, [3.0])


# ---------------------------------------------------------------------------
# Region crossings
# ---------------------------------------------------------------------------

CROSSINGS = [
    Crossing(1.0, "entry", 4, "reward"),
    Crossing(2.0, "exit", 7, "reward"),
    Crossing(3.0, "entry", 4, "reward"),
    Crossing(2.5, "entry", 11, "home"),
]


@needs_pynapple
@pytest.mark.pynapple
def test_crossings_become_a_group_per_region_and_direction():
    group = to_pynapple(CROSSINGS, epochs=(0.0, 10.0))

    assert list(group.region) == ["home", "reward", "reward"]
    assert list(group.direction) == ["entry", "entry", "exit"]
    np.testing.assert_array_equal(group[0].t, [2.5])
    np.testing.assert_array_equal(group[0].values, [11])
    np.testing.assert_array_equal(group[1].t, [1.0, 3.0])
    np.testing.assert_array_equal(group[1].values, [4, 4])
    np.testing.assert_array_equal(group[2].t, [2.0])
    np.testing.assert_array_equal(_half_open(group.time_support.values), [[0, 10]])
    entries = group[group.direction == "entry"]
    assert len(entries) == 2


@needs_pynapple
@pytest.mark.pynapple
def test_crossings_without_epochs_span_first_to_last_crossing():
    import warnings

    with warnings.catch_warnings():
        # A group with a single crossing must not trigger pynapple's
        # zero-duration and divide-by-zero warnings.
        warnings.simplefilter("error")
        group = to_pynapple(CROSSINGS)
    np.testing.assert_array_equal(group.time_support.values, [[1.0, 3.0]])
    np.testing.assert_array_equal(group[0].t, [2.5])


def test_crossings_outside_epochs_are_refused():
    with pytest.raises(ValueError, match=r"1 of 4 timestamp\(s\) lie outside"):
        to_pynapple(CROSSINGS, epochs=(0.0, 3.0))
