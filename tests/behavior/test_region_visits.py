"""Observed region events preserve timing, identity, and recording boundaries."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from neurospatial import Environment, Regions
from neurospatial.behavior import (
    RegionTransition,
    RegionVisit,
    detect_region_visits,
    region_transitions,
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


def test_sampled_crossings_dwell_and_transition_identity(region_env):
    times = np.array([0.0, 0.2, 0.7, 1.2, 2.0, 2.5, 3.0, 4.0])
    bins = np.array([1, 0, 0, 1, 3, 3, 4, 5])
    visits = detect_region_visits(
        bins, times, region_env, region_names=["A", "B", "C"], max_gap=None
    )
    assert [(v.visit_id, v.run_id, v.region_name) for v in visits] == [
        (0, 0, "A"),
        (1, 0, "B"),
        (2, 0, "C"),
    ]
    assert [(v.start_index, v.stop_index) for v in visits] == [(1, 3), (4, 6), (7, 8)]
    np.testing.assert_allclose([v.duration for v in visits], [1.0, 1.0, 0.0])
    assert all(v.entry_observed for v in visits)
    assert [v.exit_observed for v in visits] == [True, True, False]
    # For this gap-free record, visit dwell equals start-allocated region occupancy.
    occupancy = region_env.occupancy(times, region_env.bin_centers[bins], max_gap=None)
    np.testing.assert_allclose([v.duration for v in visits], occupancy[[0, 3, 5]])

    transitions = region_transitions(visits)
    assert [(t.source_region, t.target_region) for t in transitions] == [
        ("A", "B"),
        ("B", "C"),
    ]
    assert [(t.source_visit_id, t.target_visit_id) for t in transitions] == [
        (0, 1),
        (1, 2),
    ]
    assert [(t.start_index, t.end_index) for t in transitions] == [(1, 4), (4, 7)]
    np.testing.assert_allclose([t.duration for t in transitions], [1.8, 2.0])
    assert transitions[0].end_time == transitions[1].start_time
    assert isinstance(transitions[0], RegionTransition)
    table = pd.DataFrame(transitions)
    assert table["source_visit_id"].tolist() == [0, 1]
    assert table["duration"].tolist() == [1.8, 2.0]
    assert pd.DataFrame(visits)["exit_observed"].tolist() == [True, True, False]


def test_initial_visit_is_censored_without_invented_entry(region_env):
    visits = detect_region_visits(
        np.array([0, 0, 1, 3, 4]),
        np.arange(5) / 10,
        region_env,
        region_names=["A", "B"],
    )
    first, second = visits
    assert first.start_index == 0 and not first.entry_observed
    assert first.exit_observed and first.end_time == 0.2
    assert second.entry_observed
    assert region_transitions(visits) == []


def test_reentries_preserved_and_last_source_entry_is_paired(region_env):
    visits = detect_region_visits(
        np.array([1, 0, 1, 0, 1, 3]),
        np.arange(6) / 10,
        region_env,
        region_names=["A", "B"],
    )
    assert [v.region_name for v in visits] == ["A", "A", "B"]
    (transition,) = region_transitions(visits)
    assert (transition.source_visit_id, transition.target_visit_id) == (1, 2)
    assert (transition.start_index, transition.end_index) == (3, 5)


@pytest.mark.parametrize("invalid_bin", [-1, -10, 6, 999])
def test_invalid_bins_are_bridged_like_deleted_rows(region_env, invalid_bin):
    bins = np.array([1, 0, invalid_bin, 0, 1, 3])
    times = np.arange(6) / 10
    visits = detect_region_visits(bins, times, region_env, region_names=["A", "B"])
    kept = [0, 1, 3, 4, 5]
    deleted = detect_region_visits(
        bins[kept], times[kept], region_env, region_names=["A", "B"]
    )
    as_times = [(v.region_name, v.start_time, v.end_time) for v in visits]
    assert as_times == [(v.region_name, v.start_time, v.end_time) for v in deleted]
    # The invalid sample is not an exit: one visit spans it.
    assert visits[0].region_name == "A"
    assert (visits[0].start_index, visits[0].stop_index) == (1, 4)
    assert visits[0].entry_observed and visits[0].exit_observed


@pytest.mark.parametrize("invalid_bin", [-1, -10, 6, 999])
def test_invalid_bins_beyond_max_gap_censor_instead_of_fabricating_crossings(
    region_env, invalid_bin
):
    visits = detect_region_visits(
        np.array([1, 0, invalid_bin, 0, 1, 3]),
        np.arange(6) / 10,
        region_env,
        region_names=["A", "B"],
        max_gap=0.15,
    )
    assert [(v.run_id, v.region_name) for v in visits] == [(0, "A"), (1, "A"), (1, "B")]
    assert not visits[0].exit_observed and visits[0].end_time == 0.1
    assert not visits[1].entry_observed and visits[1].start_index == 3
    assert visits[2].entry_observed
    assert region_transitions(visits) == []


def test_time_gap_breaks_pairing_and_max_gap_equality_passes(region_env):
    bins = np.array([1, 0, 1, 1, 3, 1])
    times = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 6.0])
    visits = detect_region_visits(
        bins, times, region_env, region_names=["A", "B"], max_gap=1.0
    )
    assert [v.run_id for v in visits] == [0, 1]
    assert all(v.entry_observed for v in visits)
    assert region_transitions(visits) == []
    continuous = detect_region_visits(
        bins, times, region_env, region_names=["A", "B"], max_gap=None
    )
    assert len(region_transitions(continuous)) == 1


def test_singleton_observations_are_zero_duration_censored_visits(region_env):
    visits = detect_region_visits(
        np.array([0, 3, 5]),
        np.array([0.0, 2.0, 4.0]),
        region_env,
        region_names=["A", "B", "C"],
        max_gap=0.5,
    )
    assert len(visits) == 3
    assert all(not v.entry_observed and not v.exit_observed for v in visits)
    assert [v.duration for v in visits] == [0.0, 0.0, 0.0]
    assert region_transitions(visits) == []


def test_epoch_stop_closes_exposure_without_creating_entry_or_exit(region_env):
    visits = detect_region_visits(
        np.array([1, 0, 3]),
        np.array([0.0, 1.0, 2.0]),
        region_env,
        region_names=["A", "B"],
        max_gap=None,
        epochs=(0, 2),
    )
    (visit,) = visits
    assert visit.region_name == "A"
    assert visit.entry_observed and not visit.exit_observed
    assert (visit.start_index, visit.stop_index) == (1, 2)
    assert (visit.start_time, visit.end_time, visit.duration) == (1.0, 2.0, 1.0)
    assert region_transitions(visits) == []


def test_disjoint_epochs_keep_original_indices_and_censor_each_restart(region_env):
    visits = detect_region_visits(
        np.array([1, 0, 0, 3, 1, 3, 4, 0, 4, 5]),
        np.arange(10, dtype=float),
        region_env,
        region_names=["A", "B", "C"],
        max_gap=None,
        epochs=[(0, 3), (5, 9)],
    )
    assert [(v.region_name, v.start_index, v.run_id) for v in visits] == [
        ("A", 1, 0),
        ("B", 5, 1),
        ("A", 7, 1),
    ]
    assert visits[0].end_time == 3.0 and not visits[0].exit_observed
    assert not visits[1].entry_observed
    assert region_transitions(visits) == []


def test_fractional_epoch_limits_do_not_interpolate_dwell(region_env):
    (visit,) = detect_region_visits(
        np.array([0, 0, 0]),
        np.array([0.0, 1.0, 2.0]),
        region_env,
        region_names=["A"],
        max_gap=None,
        epochs=(0.5, 1.5),
    )
    assert (visit.start_index, visit.stop_index) == (1, 2)
    assert not visit.entry_observed and not visit.exit_observed
    assert visit.duration == 0.0


def test_selected_region_overlap_rejected_even_when_unvisited(region_env):
    region_builder = Regions(region_env.regions.values())
    region_builder.add("overlap", polygon=box(0.4, 0, 1.4, 1))
    region_env = region_env.with_regions(region_builder)
    with pytest.raises(ValueError, match=r"overlap.*A.*active bin"):
        detect_region_visits(
            np.array([4]),
            np.array([0.0]),
            region_env,
            region_names=["A", "overlap"],
        )


def test_region_name_order_does_not_change_events(region_env):
    bins, times = np.array([1, 0, 1, 3]), np.arange(4) / 10
    assert detect_region_visits(
        bins, times, region_env, region_names=["A", "B"]
    ) == detect_region_visits(bins, times, region_env, region_names=["B", "A"])


@pytest.mark.parametrize("bins,times", [([], []), ([1, 1], [0.0, 0.1]), ([-1], [0.0])])
def test_no_observed_region_returns_empty_list(region_env, bins, times):
    assert (
        detect_region_visits(
            np.asarray(bins), np.asarray(times), region_env, region_names=["A"]
        )
        == []
    )
    assert region_transitions([]) == []


@pytest.mark.parametrize(
    "times", [[0.0, 0.0], [1.0, 0.0], [0.0, np.nan], [0.0, np.inf]]
)
def test_invalid_clock_rejected(region_env, times):
    with pytest.raises(ValueError, match=r"times.*finite.*strictly increasing"):
        detect_region_visits(
            np.array([0, 0]), np.asarray(times), region_env, region_names=["A"]
        )


@pytest.mark.parametrize("gap", [0, -1, np.nan, np.inf])
def test_invalid_gap_rejected(region_env, gap):
    with pytest.raises(ValueError, match="max_gap"):
        detect_region_visits(
            np.array([], dtype=int),
            np.array([]),
            region_env,
            region_names=["A"],
            max_gap=gap,
        )


@pytest.mark.parametrize("names", [[], ["A", "A"]])
def test_invalid_region_selection_rejected(region_env, names):
    with pytest.raises(ValueError, match="region_names"):
        detect_region_visits(
            np.array([], dtype=int), np.array([]), region_env, region_names=names
        )


def test_invalid_epochs_are_validated_even_for_empty_data(region_env):
    with pytest.raises(ValueError, match="epochs"):
        detect_region_visits(
            np.array([], dtype=int),
            np.array([]),
            region_env,
            region_names=["A"],
            epochs=(2, 1),
        )


def test_noninteger_bins_not_silently_truncated(region_env):
    with pytest.raises(TypeError, match="integer bin indices"):
        detect_region_visits(
            np.array([0.1, 1.0]), np.array([0.0, 0.1]), region_env, region_names=["A"]
        )


def test_records_do_not_alias_inputs_and_are_immutable(region_env):
    bins, times = np.array([1, 0, 1, 3]), np.arange(4) / 10
    visits = detect_region_visits(bins, times, region_env, region_names=["A", "B"])
    original = tuple(visits)
    bins[:] = -1
    times[:] = 99
    assert tuple(visits) == original
    assert isinstance(visits[0], RegionVisit)
    with pytest.raises(FrozenInstanceError):
        visits[0].entry_observed = False


def test_filtered_visits_do_not_invent_new_consecutive_pairs(region_env):
    visits = detect_region_visits(
        np.array([1, 0, 1, 3, 1, 5]),
        np.arange(6) / 10,
        region_env,
        region_names=["A", "B", "C"],
    )
    assert len(region_transitions(visits)) == 2
    assert region_transitions([visits[0], visits[2]]) == []
    with pytest.raises(ValueError, match="chronological"):
        region_transitions(visits[::-1])
    with pytest.raises(ValueError, match="duplicates"):
        region_transitions([visits[0], visits[0]])
    with pytest.raises(ValueError, match="chronological"):
        region_transitions([visits[0], replace(visits[1], start_time=-1)])


def test_millisecond_times_warn_instead_of_splitting_every_visit(region_env):
    times = np.array([0.0, 0.2, 0.7, 1.2, 2.0, 2.5, 3.0, 4.0]) * 1000.0
    bins = np.array([1, 0, 0, 1, 3, 3, 4, 5])
    with pytest.warns(UserWarning, match="probably not in seconds"):
        detect_region_visits(bins, times, region_env, region_names=["A", "B", "C"])
