"""Per-sample condition labels from laps and trials with ``sample_values``.

``laps[laps.direction != "unknown"].sample_values(times, "direction")`` labels
each sample with its lap's direction, and
``trials[trials.outcome == "reached"].sample_values(times, ("start_region",
"end_region"))`` with its trial's route. The expected arrays are the outputs
of the per-sample label helpers these calls replaced, on the same inputs
(their route labels were ``"start→end"`` strings; here they are tuples).
The labels feed ``labels_to_epochs`` and ``compute_directional_place_fields``
unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Intervals
from neurospatial.behavior import detect_trials
from neurospatial.behavior.segmentation import Lap, Trial
from neurospatial.encoding.spatial import compute_directional_place_fields

_CW, _CCW = "clockwise", "counter-clockwise"


def _lap_labels(laps: Intervals[Lap], times: NDArray[np.float64]) -> NDArray:
    """Each sample's lap direction; None outside laps and in unknown laps."""
    return laps[laps.direction != "unknown"].sample_values(times, "direction")


def _route_labels(trials: Intervals[Trial], times: NDArray[np.float64]) -> NDArray:
    """Each sample's (start_region, end_region); None outside reached trials."""
    reached = trials[trials.outcome == "reached"]
    return reached.sample_values(times, ("start_region", "end_region"))


def _runs(labels: NDArray) -> list[tuple[object, int, int]]:
    """``labels`` as ``(label, first, stop)`` runs of equal consecutive values."""
    runs: list[tuple[object, int, int]] = []
    for i, label in enumerate(labels.tolist()):
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1], i + 1)
        else:
            runs.append((label, i, i + 1))
    return runs


# -----------------------------------------------------------------------------
# Laps
# -----------------------------------------------------------------------------


@pytest.fixture
def lap_times() -> NDArray[np.float64]:
    return np.linspace(0.0, 10.0, 11)


@pytest.mark.parametrize(
    ("laps", "expected"),
    [
        pytest.param(
            [Lap(0.0, 3.0, _CW, 1.0), Lap(7.0, 10.0, _CCW, 1.0)],
            [_CW, _CW, _CW, None, None, None, None, _CCW, _CCW, _CCW, None],
            id="lap-ends-are-outside",
        ),
        pytest.param(
            [Lap(0.0, 4.0, _CW, 1.0), Lap(6.0, 10.0, _CCW, 1.0)],
            [_CW, _CW, _CW, _CW, None, None, _CCW, _CCW, _CCW, _CCW, None],
            id="between-and-after-are-none",
        ),
        pytest.param(
            [Lap(0.0, 4.0, "unknown", 1.0)],
            [None] * 11,
            id="unknown-direction-is-none",
        ),
    ],
)
def test_lap_direction_labels(lap_times, laps, expected):
    """A lap labels [start_time, stop_time); its last sample leaves the lap."""
    labels = _lap_labels(Intervals(laps), lap_times)
    assert labels.dtype == object
    assert labels.shape == lap_times.shape
    assert labels.tolist() == expected


def test_overlapping_laps_are_refused(lap_times):
    """A sample in two laps would get two labels; it no longer takes the later."""
    laps = Intervals([Lap(0.0, 5.0, _CW, 1.0), Lap(3.0, 8.0, _CCW, 1.0)])
    with pytest.raises(ValueError, match=r"intervals 0 and 1 overlap"):
        _lap_labels(laps, lap_times)
    assert _lap_labels(laps[[1]], lap_times).tolist() == (
        [None, None, None, _CCW, _CCW, _CCW, _CCW, _CCW, None, None, None]
    )


# -----------------------------------------------------------------------------
# Trials
# -----------------------------------------------------------------------------


@pytest.fixture
def session_times() -> NDArray[np.float64]:
    """0-100 s at 10 Hz."""
    return np.linspace(0.0, 100.0, 1001)


def _trial(start, stop, start_region, end_region, outcome="reached") -> Trial:
    return Trial(
        start_time=start,
        stop_time=stop,
        start_region=start_region,
        end_region=end_region,
        outcome=outcome,
    )


@pytest.mark.parametrize(
    ("trials", "expected"),
    [
        pytest.param(
            [
                _trial(10.0, 20.0, "home", "goal_left"),
                _trial(30.0, 45.0, "home", "goal_right"),
            ],
            [
                (None, 0, 100),
                (("home", "goal_left"), 100, 200),
                (None, 200, 300),
                (("home", "goal_right"), 300, 450),
                (None, 450, 1001),
            ],
            id="two-trials",
        ),
        pytest.param(
            [
                _trial(10.0, 20.0, "center", "arm1"),
                _trial(30.0, 40.0, "center", "arm2"),
            ],
            [
                (None, 0, 100),
                (("center", "arm1"), 100, 200),
                (None, 200, 300),
                (("center", "arm2"), 300, 400),
                (None, 400, 1001),
            ],
            id="different-goals",
        ),
        pytest.param(
            [_trial(50.0, 65.0, "home", None, "returned")],
            [(None, 0, 1001)],
            id="returned-trial-labels-nothing",
        ),
        pytest.param([], [(None, 0, 1001)], id="no-trials"),
    ],
)
def test_route_labels(session_times, trials, expected):
    labels = _route_labels(Intervals(trials, record_type=Trial), session_times)
    assert labels.dtype == object
    assert labels.shape == session_times.shape
    assert _runs(labels) == expected


def test_route_labels_are_half_open():
    """The trial's first sample is labeled; its last, which leaves it, is not."""
    times = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    trials = Intervals([_trial(10.0, 20.0, "home", "goal")])
    assert _route_labels(trials, times).tolist() == [
        None,
        ("home", "goal"),
        ("home", "goal"),
        None,
        None,
    ]


def test_overlapping_trials_are_refused(session_times):
    """A sample in two reached trials would get two labels."""
    trials = Intervals(
        [
            _trial(10.0, 30.0, "home", "goal_left"),
            _trial(20.0, 40.0, "center", "goal_right"),
        ]
    )
    with pytest.raises(ValueError, match=r"intervals 0 and 1 overlap"):
        _route_labels(trials, session_times)
    # One of them is fine.
    assert _runs(_route_labels(trials[[1]], session_times)) == [
        (None, 0, 200),
        (("center", "goal_right"), 200, 400),
        (None, 400, 1001),
    ]


def test_route_labels_need_times():
    """Samples are required, as for every per-sample trial function."""
    trials = Intervals([_trial(10.0, 20.0, "home", "goal")])
    with pytest.raises(ValueError, match="times"):
        _route_labels(trials, np.array([], dtype=np.float64))


# -----------------------------------------------------------------------------
# Detected trials feed directional place fields
# -----------------------------------------------------------------------------


def _trial_direction_labels(fx):
    """Outbound (home to reward) and inbound labels of the reached trials."""
    trials = detect_trials(
        fx["env"],
        fx["times"],
        fx["positions"],
        regions=[fx["start_region"], fx["end_region"]],
    )
    reached = trials[trials.outcome == "reached"]
    direction = np.where(
        reached.start_region == fx["start_region"], "outbound", "inbound"
    )
    return reached.sample_values(fx["times"], direction.astype(object))


def test_trial_direction_labels_inbound_outbound(out_and_back_track):
    """Out-and-back trajectory yields contiguous outbound then inbound trials."""
    fx = out_and_back_track

    labels = _trial_direction_labels(fx)

    assert labels.shape == (len(fx["times"]),)
    assert labels.dtype == object
    assert set(labels) <= {"inbound", "outbound", None}
    assert "outbound" in labels
    assert "inbound" in labels

    # Outbound must come before inbound, with None at the turnaround.
    out_idx = np.where(labels == "outbound")[0]
    in_idx = np.where(labels == "inbound")[0]
    assert out_idx.max() < in_idx.min()

    # Each label group is a single contiguous block (no interleaving).
    assert np.all(np.diff(out_idx) == 1)
    assert np.all(np.diff(in_idx) == 1)

    # The turnaround samples between the two blocks are None.
    assert any(label is None for label in labels[out_idx.max() + 1 : in_idx.min()])


def test_labels_feed_directional_place_fields(out_and_back_track):
    """Named and route (tuple) labels give the same directional maps."""
    fx = out_and_back_track
    env = fx["env"]
    times = fx["times"]
    trials = detect_trials(
        env, times, fx["positions"], regions=[fx["start_region"], fx["end_region"]]
    )

    named = _trial_direction_labels(fx)
    routes = _route_labels(trials, times)
    route_names = {("home", "reward"): "outbound", ("reward", "home"): "inbound"}
    assert [route_names.get(r) for r in routes.tolist()] == named.tolist()

    rng = np.random.default_rng(0)
    spike_times = np.sort(rng.uniform(times[0], times[-1], 30))

    # No reshaping: pass labels straight through.
    by_name = compute_directional_place_fields(
        env, spike_times, times, fx["positions"], named, bandwidth=10.0
    )
    by_route = compute_directional_place_fields(
        env, spike_times, times, fx["positions"], routes, bandwidth=10.0
    )

    # None is excluded; each direction becomes its own map.
    assert by_name.labels == ("inbound", "outbound")
    assert by_route.labels == (("home", "reward"), ("reward", "home"))
    for route, name in route_names.items():
        assert by_route.firing_rates[route].shape == (env.n_bins,)
        np.testing.assert_array_equal(
            by_route.firing_rates[route], by_name.firing_rates[name]
        )
