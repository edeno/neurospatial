"""Observed region visits preserve timing, identity, and recording boundaries."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from neurospatial import Environment, Intervals, Regions
from neurospatial.behavior import (
    RegionVisit,
    detect_region_visits,
    detect_trials,
    trials_between_visits,
)


@pytest.fixture
def region_env() -> Environment:
    """Six bins: A, travel, travel, B, travel, C."""
    env = Environment.from_grid_mask(
        active_mask=np.ones((6, 1), dtype=bool),
        grid_edges=(np.arange(7, dtype=float), np.array([0.0, 1.0])),
    )
    for name, left in [("A", 0), ("B", 3), ("C", 5)]:
        region_builder = Regions(env.regions.values())
        region_builder.add(name, polygon=box(left, 0, left + 1, 1))
        env = env.with_regions(region_builder)
    return env


def _at(env: Environment, bins) -> np.ndarray:
    """Positions at the centers of ``bins``; NaN (lost tracking) where -1."""
    bins = np.asarray(bins, dtype=int)
    positions = env.bin_centers[np.maximum(bins, 0)].copy()
    positions[bins < 0] = np.nan
    return positions.reshape(-1, env.n_dims)


def _visits(env, bins, times, **options):
    return detect_region_visits(
        env, np.asarray(times, dtype=float), _at(env, bins), **options
    )


def test_sampled_crossings_dwell_and_trials_between(region_env):
    times = np.array([0.0, 0.2, 0.7, 1.2, 2.0, 2.5, 3.0, 4.0])
    bins = np.array([1, 0, 0, 1, 3, 3, 4, 5])
    visits = _visits(region_env, bins, times, regions=["A", "B", "C"], max_gap=None)
    assert isinstance(visits, Intervals)
    assert visits.record_type is RegionVisit
    assert visits.interval_ids.tolist() == [0, 1, 2]
    assert visits.run_id.tolist() == [0, 0, 0]
    assert visits.region.tolist() == ["A", "B", "C"]
    assert list(zip(visits.start_index, visits.stop_index, strict=True)) == [
        (1, 3),
        (4, 6),
        (7, 8),
    ]
    np.testing.assert_allclose(visits.durations, [1.0, 1.0, 0.0])
    assert visits.entry_observed.all()
    assert visits.exit_observed.tolist() == [True, True, False]
    # For this gap-free record, visit dwell equals start-allocated region occupancy.
    occupancy = region_env.occupancy(times, _at(region_env, bins), max_gap=None)
    np.testing.assert_allclose(visits.durations, occupancy[[0, 3, 5]])

    # Trials are the time between visits: departure to the next arrival.
    trials = trials_between_visits(visits)
    assert list(zip(trials.start_region, trials.end_region, strict=True)) == [
        ("A", "B"),
        ("B", "C"),
    ]
    np.testing.assert_allclose(trials.start, [1.2, 3.0])
    np.testing.assert_allclose(trials.stop, [2.0, 4.0])
    detected = detect_trials(
        region_env, times, _at(region_env, bins), regions=["A", "B", "C"], max_gap=None
    )
    assert list(detected) == list(trials)
    table = visits.intervals_table()
    assert table["exit_observed"].tolist() == [True, True, False]


def test_singleton_visit_kept_without_windows(region_env):
    """A visit seen in one sample is an observation with no time."""
    with pytest.warns(UserWarning, match="excluded ALL trajectory intervals"):
        visits = _visits(
            region_env,
            [0, 3, 5],
            [0.0, 2.0, 4.0],
            regions=["A", "B", "C"],
            max_gap=0.5,
        )
    assert len(visits) == 3
    assert not visits.entry_observed.any()
    assert not visits.exit_observed.any()
    assert visits.durations.tolist() == [0.0, 0.0, 0.0]
    assert visits.windows.shape == (0, 2)


def test_initial_visit_is_censored_without_invented_entry(region_env):
    first, second = _visits(
        region_env, [0, 0, 1, 3, 4], np.arange(5) / 10, regions=["A", "B"]
    )
    assert first.start_index == 0
    assert not first.entry_observed
    assert first.exit_observed
    assert first.stop_time == 0.2
    assert second.entry_observed


def test_reentries_are_separate_visits(region_env):
    visits = _visits(
        region_env, [1, 0, 1, 0, 1, 3], np.arange(6) / 10, regions=["A", "B"]
    )
    assert visits.region.tolist() == ["A", "A", "B"]
    trials = trials_between_visits(visits)
    assert trials.outcome.tolist() == ["returned", "reached"]
    np.testing.assert_allclose(trials.start, [0.2, 0.4])
    np.testing.assert_allclose(trials.stop, [0.3, 0.5])


def test_lost_tracking_censors_like_a_long_gap(region_env):
    """An interval touching a NaN position is not observed, as in env.occupancy.

    The visit around the lost sample is split into two censored visits,
    exactly as a gap longer than ``max_gap`` splits it, and no exit is
    fabricated at the lost sample.
    """
    bins = [1, 0, -1, 0, 1, 3]
    times = np.arange(6) / 10
    visits = _visits(region_env, bins, times, regions=["A", "B"])
    long_gap = _visits(region_env, bins, times, regions=["A", "B"], max_gap=0.15)
    assert list(visits) == list(long_gap)
    assert list(zip(visits.run_id, visits.region, strict=True)) == [
        (0, "A"),
        (1, "A"),
        (1, "B"),
    ]
    assert not visits[0].exit_observed
    assert visits[0].stop_time == 0.1
    assert not visits[1].entry_observed
    assert visits[1].start_index == 3
    assert visits[2].entry_observed
    # The visit before the loss has no departure; the one after it does.
    trials = detect_trials(region_env, times, _at(region_env, bins), regions=["A", "B"])
    assert list(zip(trials.start_region, trials.end_region, strict=True)) == [
        ("A", "B")
    ]
    assert trials.start.tolist() == [0.4]


def test_positions_outside_the_environment_are_not_located(region_env):
    """A position beyond the environment ends observation like a NaN row."""
    times = np.arange(6) / 10
    outside = _at(region_env, [1, 0, 0, 0, 1, 3])
    outside[2] = [50.0, 50.0]
    lost = _at(region_env, [1, 0, -1, 0, 1, 3])
    assert list(
        detect_region_visits(region_env, times, outside, regions=["A"])
    ) == list(detect_region_visits(region_env, times, lost, regions=["A"]))


def test_positions_need_the_environment_dimensions(region_env):
    with pytest.raises(ValueError, match="3 coordinate"):
        detect_region_visits(
            region_env, np.arange(2.0), np.zeros((2, 3)), regions=["A"]
        )


def test_time_gap_splits_runs_and_max_gap_equality_passes(region_env):
    bins = [1, 0, 1, 1, 3, 1]
    times = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 6.0])
    visits = _visits(region_env, bins, times, regions=["A", "B"], max_gap=1.0)
    assert visits.run_id.tolist() == [0, 1]
    assert visits.entry_observed.all()
    gap = detect_trials(
        region_env, times, _at(region_env, bins), regions=["A", "B"], max_gap=1.0
    )
    # The animal leaves A on the last sample before the 2 s gap: no observed
    # time outside A, so no trial; the arrival at B after it starts none.
    assert len(gap) == 0
    continuous = detect_trials(
        region_env, times, _at(region_env, bins), regions=["A", "B"], max_gap=None
    )
    assert continuous.outcome.tolist() == ["reached"]


def test_epoch_stop_closes_exposure_without_creating_entry_or_exit(region_env):
    (visit,) = _visits(
        region_env,
        [1, 0, 3],
        [0.0, 1.0, 2.0],
        regions=["A", "B"],
        max_gap=None,
        epochs=(0, 2),
    )
    assert visit.region == "A"
    assert visit.entry_observed
    assert not visit.exit_observed
    assert (visit.start_index, visit.stop_index) == (1, 2)
    assert (visit.start_time, visit.stop_time, visit.duration) == (1.0, 2.0, 1.0)


def test_disjoint_epochs_keep_original_indices_and_censor_each_restart(region_env):
    visits = _visits(
        region_env,
        [1, 0, 0, 3, 1, 3, 4, 0, 4, 5],
        np.arange(10, dtype=float),
        regions=["A", "B", "C"],
        max_gap=None,
        epochs=[(0, 3), (5, 9)],
    )
    assert list(zip(visits.region, visits.start_index, visits.run_id, strict=True)) == [
        ("A", 1, 0),
        ("B", 5, 1),
        ("A", 7, 1),
    ]
    assert visits[0].stop_time == 3.0
    assert not visits[0].exit_observed
    assert not visits[1].entry_observed
    assert visits.analysis_settings["epochs"] == [[0.0, 3.0], [5.0, 9.0]]


def test_fractional_epoch_limits_do_not_interpolate_dwell(region_env):
    with pytest.warns(UserWarning, match="excluded ALL trajectory intervals"):
        (visit,) = _visits(
            region_env,
            [0, 0, 0],
            [0.0, 1.0, 2.0],
            regions=["A"],
            max_gap=None,
            epochs=(0.5, 1.5),
        )
    assert (visit.start_index, visit.stop_index) == (1, 2)
    assert not visit.entry_observed
    assert not visit.exit_observed
    assert visit.duration == 0.0


def test_selected_region_overlap_rejected_even_when_unvisited(region_env):
    region_builder = Regions(region_env.regions.values())
    region_builder.add("overlap", polygon=box(0.4, 0, 1.4, 1))
    region_env = region_env.with_regions(region_builder)
    with pytest.raises(ValueError, match=r"overlap.*A.*active bin"):
        _visits(region_env, [4], [0.0], regions=["A", "overlap"])


def test_point_regions_refused(region_env):
    region_builder = Regions(region_env.regions.values())
    region_builder.add("well", point=(1.5, 0.5))
    region_env = region_env.with_regions(region_builder)
    with pytest.raises(ValueError, match=r"(?s)'well' is a point.*radius="):
        _visits(region_env, [1, 1], [0.0, 1.0], regions=["A", "well"])


def test_region_name_order_does_not_change_events(region_env):
    bins, times = [1, 0, 1, 3], np.arange(4) / 10
    assert list(_visits(region_env, bins, times, regions=["A", "B"])) == list(
        _visits(region_env, bins, times, regions=["B", "A"])
    )


@pytest.mark.parametrize(
    ("bins", "times"), [([], []), ([1, 1], [0.0, 0.1]), ([-1], [0.0])]
)
def test_no_observed_region_returns_no_visits(region_env, bins, times):
    visits = _visits(region_env, bins, times, regions=["A"])
    assert len(visits) == 0
    assert visits.record_type is RegionVisit


@pytest.mark.parametrize(
    ("times", "problem"),
    [
        ([0.0, 0.0], "repeat the one before"),
        ([1.0, 0.0], "decreases"),
        ([0.0, np.nan], "non-finite"),
        ([0.0, np.inf], "non-finite"),
    ],
)
def test_invalid_clock_rejected(region_env, times, problem):
    with pytest.raises(
        ValueError, match=rf"detect_region_visits\(\):\n- times.*{problem}"
    ):
        _visits(region_env, [0, 0], times, regions=["A"])


def test_times_and_positions_must_be_aligned(region_env):
    with pytest.raises(ValueError, match="must have the same length"):
        detect_region_visits(
            region_env, np.arange(3.0), _at(region_env, [0, 0]), regions=["A"]
        )


@pytest.mark.parametrize("gap", [0, -1, np.nan, np.inf])
def test_invalid_gap_rejected(region_env, gap):
    with pytest.raises(ValueError, match="max_gap"):
        _visits(region_env, [], [], regions=["A"], max_gap=gap)


@pytest.mark.parametrize(
    ("names", "match"), [([], "regions is empty"), (["A", "A"], "duplicate")]
)
def test_invalid_region_selection_rejected(region_env, names, match):
    with pytest.raises(ValueError, match=match):
        _visits(region_env, [], [], regions=names)


def test_one_region_name_must_be_in_a_list(region_env):
    with pytest.raises(TypeError, match="sequence of names"):
        _visits(region_env, [0], [0.0], regions="A")


def test_invalid_epochs_are_validated_even_for_empty_data(region_env):
    with pytest.raises(ValueError, match="epochs"):
        _visits(region_env, [], [], regions=["A"], epochs=(2, 1))


def test_records_do_not_alias_inputs_and_are_immutable(region_env):
    times = np.arange(4) / 10
    positions = _at(region_env, [1, 0, 1, 3])
    visits = detect_region_visits(region_env, times, positions, regions=["A", "B"])
    original = tuple(visits)
    positions[:] = np.nan
    times[:] = 99
    assert tuple(visits) == original
    assert isinstance(visits[0], RegionVisit)
    with pytest.raises(FrozenInstanceError):
        visits[0].entry_observed = False


def test_millisecond_times_warn_instead_of_splitting_every_visit(region_env):
    times = np.array([0.0, 0.2, 0.7, 1.2, 2.0, 2.5, 3.0, 4.0]) * 1000.0
    with pytest.warns(UserWarning, match="probably not in seconds"):
        _visits(region_env, [1, 0, 0, 1, 3, 3, 4, 5], times, regions=["A", "B", "C"])


_VISIT = {
    "visit_id": 0,
    "run_id": 0,
    "region": "A",
    "start_index": 1,
    "stop_index": 3,
    "start_time": 0.1,
    "stop_time": 0.3,
    "entry_observed": True,
    "exit_observed": True,
}


def test_records_store_plain_python_scalars():
    visit = RegionVisit(
        visit_id=np.int64(0),
        run_id=np.intp(2),
        region=np.str_("A"),
        start_index=np.int32(1),
        stop_index=np.int64(3),
        start_time=np.float64(0.25),
        stop_time=np.float64(0.5),
        entry_observed=np.bool_(True),
        exit_observed=np.bool_(False),
    )
    expected = {
        "visit_id": int,
        "run_id": int,
        "region": str,
        "start_index": int,
        "stop_index": int,
        "start_time": float,
        "stop_time": float,
        "entry_observed": bool,
        "exit_observed": bool,
        "duration": float,
    }
    assert {name: type(getattr(visit, name)) for name in expected} == expected
    assert visit.duration == 0.25
    assert pd.DataFrame([visit])["region"].tolist() == ["A"]


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"stop_index": 1}, ValueError, "start_index < stop_index"),
        ({"stop_index": 0}, ValueError, "start_index < stop_index"),
        ({"stop_time": 0.05}, ValueError, "start_time <= stop_time"),
        ({"stop_time": np.nan}, ValueError, "start_time <= stop_time"),
        ({"start_index": 1.5}, TypeError, "start_index"),
        ({"entry_observed": 1}, TypeError, "entry_observed"),
        ({"region": 3}, TypeError, "region"),
    ],
)
def test_visit_rejects_inconsistent_fields(changes, error, match):
    with pytest.raises(error, match=match):
        RegionVisit(**{**_VISIT, **changes})


def test_singleton_visit_is_valid():
    visit = RegionVisit(**{**_VISIT, "stop_index": 2, "stop_time": 0.1})
    assert visit.duration == 0.0
