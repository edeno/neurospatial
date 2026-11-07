"""One small, deterministic instance of every public result class.

``RESULT_FACTORIES`` maps each public result class name to a zero-argument
factory. Each factory builds its result through the public function that
returns it, on tiny seeded data, so the instances follow constructor changes.
``RESULT_VARIANTS`` adds the other forms of a class whose summary depends on
its inputs (interval results). ``public_result_classes`` lists the classes
the API snapshot publishes, so a new public result class without a factory
is caught by ``tests/test_result_instances.py``.
"""

from __future__ import annotations

import dataclasses
import importlib
import warnings
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

_SNAPSHOT = Path(__file__).parent / "data" / "public_api.txt"


def public_result_classes() -> dict[str, type]:
    """Every public result class, keyed by ``__name__``.

    A result class is a ``ResultMixin`` subclass, whatever its name
    (``DecodingSummary``, ``GridProperties``), or a class whose name ends in
    ``Result``. Re-exports of one class under several paths count once.
    """
    from neurospatial._results import ResultMixin

    classes: dict[str, type] = {}
    for line in _SNAPSHOT.read_text().splitlines():
        parts = line.strip().rsplit(" ", 1)
        if len(parts) != 2 or parts[1] != "(class)":
            continue
        module, _, attr = parts[0].rpartition(".")
        cls = getattr(importlib.import_module(module), attr)
        if issubclass(cls, ResultMixin) or cls.__name__.endswith("Result"):
            classes.setdefault(cls.__name__, cls)
    return classes


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


@cache
def _open_field() -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """A 50 cm box, 50 s of uniform positions and headings, three units."""
    from neurospatial import Environment

    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (500, 2))
    env = Environment.from_samples(positions, bin_size=5.0, units="cm")
    times = np.linspace(0, 50, 500)
    headings = rng.uniform(-np.pi, np.pi, 500)
    trains = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
    return env, times, positions, headings, trains


@cache
def _box() -> Any:
    from shapely.geometry import box

    from neurospatial import Environment

    return Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")


@cache
def _track() -> Any:
    from neurospatial import Environment

    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [120.0, 0.0]], bin_size=10.0, units="cm"
    )


@cache
def _track_rates() -> np.ndarray:
    """Eight Gaussian place fields tiling ``_track()``, in Hz."""
    x = _track().bin_centers[:, 0]
    fields = np.linspace(10.0, 110.0, 8)
    return 20.0 * np.exp(-0.5 * ((x - fields[:, None]) / 8.0) ** 2)


def _forward_spikes() -> list[np.ndarray]:
    """One spike per cell of ``_track_rates()``, in field order."""
    return [np.array([1.0 + 0.01 * i]) for i in range(8)]


@cache
def _sweep() -> tuple[Any, np.ndarray, np.ndarray]:
    """A 100 cm track and a posterior that sweeps one 4 cm bin per 20 ms."""
    from neurospatial import Environment

    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=4.0, units="cm"
    )
    posterior = np.zeros((20, env.n_bins))
    posterior[np.arange(20), np.arange(20)] = 1.0
    return env, posterior, np.arange(20) * 0.02


@cache
def _spatial_rates() -> Any:
    from neurospatial.encoding import compute_spatial_rates

    env, times, positions, _, trains = _open_field()
    return compute_spatial_rates(env, trains, times, positions, bandwidth=10.0)


@cache
def _directional_rates() -> Any:
    from neurospatial.encoding import compute_directional_rates

    _, times, _, headings, trains = _open_field()
    return compute_directional_rates(trains, times, headings)


@cache
def _object_vector_rates() -> Any:
    from neurospatial.encoding import compute_egocentric_rates

    env, times, positions, headings, trains = _open_field()
    return compute_egocentric_rates(
        env,
        trains,
        times,
        positions,
        headings,
        np.array([[25.0, 25.0]]),
        distance_range=(0.0, 50.0),
        distance_bin_size=5.0,
    )


@cache
def _boundary_vector_rates() -> Any:
    from neurospatial.encoding import compute_boundary_vector_rates

    rng = np.random.default_rng(0)
    times = np.linspace(0, 100, 1000)
    positions = rng.uniform(10, 90, (1000, 2))
    headings = rng.uniform(-np.pi, np.pi, 1000)
    trains = [np.sort(rng.uniform(0, 100, k)) for k in (100, 150, 50)]
    return compute_boundary_vector_rates(
        _box(), trains, times, positions, headings, direction_frame="egocentric"
    )


@cache
def _view_rates() -> Any:
    from neurospatial.encoding import compute_view_rates

    env, times, positions, headings, trains = _open_field()
    return compute_view_rates(
        env,
        trains,
        times,
        positions,
        headings,
        gaze_model="fixed_distance",
        view_distance=10.0,
        boundary_source="analysis",
    )


@cache
def _tuning_curves() -> Any:
    from neurospatial.encoding import compute_tuning_curves

    rng = np.random.default_rng(0)
    times = np.linspace(0, 100, 1000)
    distance = 10.0 * (times % 10)  # cm along a 100 cm run, every 10 s
    trains = [np.sort(rng.uniform(0, 100, k)) for k in (100, 150, 50)]
    return compute_tuning_curves(
        trains,
        times,
        distance,
        bin_edges=np.linspace(0, 100, 11),
        variable="distance",
        units="cm",
    )


@cache
def _population_psth() -> Any:
    from neurospatial.events import population_peri_event_histogram

    _, _, _, _, trains = _open_field()
    return population_peri_event_histogram(
        trains, np.array([10.0, 20.0, 30.0]), (-1.0, 1.0), dt=0.1
    )


@cache
def _vte_inputs() -> tuple[Any, np.ndarray, np.ndarray, Any]:
    """The four-trial T-maze of the ``compute_vte_session`` example."""
    from neurospatial import Environment, Intervals, Regions
    from neurospatial.behavior import Trial

    y = np.linspace(0, 50, 30)
    stem = np.column_stack([np.repeat([45.0, 50.0, 55.0], 30), np.tile(y, 3)])
    bar = np.column_stack([np.linspace(0, 100, 60), np.full(60, 55.0)])
    env = Environment.from_samples(np.vstack([stem, bar]), bin_size=5.0, units="cm")
    regions = Regions()
    regions.add("center", point=(50.0, 55.0))
    regions.add("choice", point=(50.0, 52.0), radius=15.0)
    regions.add("left", point=(5.0, 55.0))
    regions.add("right", point=(95.0, 55.0))
    env = env.with_regions(regions)
    times, positions, trials = [], [], []
    for i, scan in enumerate([1.0, 2.0, 3.0, None]):
        t = 3.0 * i + np.arange(60) / 30
        if scan is None:
            xy = np.column_stack([np.full(60, 50.0), 50.0 + 1.5 * (t - t[0])])
        else:
            x = 50.0 + scan * np.sin(np.linspace(0, 10 * np.pi, 60))
            xy = np.column_stack([x, np.linspace(30.0, 56.0, 60)])
        times.append(t)
        positions.append(xy)
        trials.append(Trial(t[0], t[-1], "start", "center", "reached"))
    return env, np.concatenate(times), np.vstack(positions), Intervals(trials)


@cache
def _assemblies() -> Any:
    from neurospatial.reactivation import detect_assemblies

    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, (500, 8)).astype(float)
    counts[:, [1, 4, 6]] += 3 * rng.poisson(0.3, (500, 1))
    return detect_assemblies(counts, rng=0)


def _pattern() -> Any:
    from neurospatial.reactivation import AssemblyPattern

    return AssemblyPattern(np.array([0.7, 0.7, 0.1]), np.array([0, 1]), 0.5)


def _three_unit_counts() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    pre = rng.poisson(1.0, (300, 3)).astype(float)
    post = rng.poisson(1.0, (300, 3)).astype(float)
    post[:, :2] += 2 * rng.poisson(0.3, (300, 1))
    return pre, post


def _correlations() -> tuple[np.ndarray, np.ndarray]:
    from neurospatial.reactivation import pairwise_correlations

    rng = np.random.default_rng(0)
    task = rng.poisson(1.0, (500, 6)).astype(float)
    task[:, :3] += rng.poisson(0.5, (500, 1))
    sleep = rng.poisson(1.0, (500, 6)).astype(float)
    sleep[:, :3] += rng.poisson(0.5, (500, 1))
    return pairwise_correlations(task), pairwise_correlations(sleep)


# ---------------------------------------------------------------------------
# Factories, one per class
# ---------------------------------------------------------------------------


def _decode_counts() -> np.ndarray:
    return np.random.default_rng(0).poisson(0.5, (10, len(_spatial_rates())))


def _decoding_result() -> Any:
    from neurospatial.decoding import decode_position

    rates = _spatial_rates()
    return decode_position(rates.env, _decode_counts(), rates, dt=0.1)


def _decoding_summary() -> Any:
    from neurospatial.decoding import decode_position_summary

    rates = _spatial_rates()
    return decode_position_summary(rates.env, _decode_counts(), rates, dt=0.1)


def _grid_properties() -> Any:
    from neurospatial import Environment
    from neurospatial.encoding import (
        autocorrelation_pixel_size,
        grid_properties,
        spatial_autocorrelation,
    )

    edges = np.arange(0.0, 102.5, 2.5)  # 100 cm box, 2.5 cm bins
    env = Environment.from_grid_mask(
        np.ones((40, 40), dtype=bool), grid_edges=(edges, edges), units="cm"
    )
    wavenumber = 4 * np.pi / (np.sqrt(3) * 40.0)  # 40 cm grid spacing
    rate = sum(
        np.cos(wavenumber * env.bin_centers @ [np.cos(a), np.sin(a)])
        for a in (0.3, 0.3 + np.pi / 3, 0.3 + 2 * np.pi / 3)
    )
    return grid_properties(
        spatial_autocorrelation(env, np.exp(rate)),
        pixel_size=autocorrelation_pixel_size(env),
    )


def _directional_place_fields() -> Any:
    from neurospatial.encoding import compute_directional_place_fields

    env, times, positions, _, trains = _open_field()
    labels = np.where(times < 25.0, "first", "second").astype(object)
    return compute_directional_place_fields(
        env, trains[0], times, positions, labels, bandwidth=10.0
    )


def _peri_event_result() -> Any:
    from neurospatial.events import peri_event_histogram

    _, _, _, _, trains = _open_field()
    return peri_event_histogram(
        trains[0], np.array([10.0, 20.0, 30.0]), (-1.0, 1.0), dt=0.1
    )


def _annotation_result() -> Any:
    # Built through the napari editor only; no headless public route.
    from neurospatial import Landmarks, Regions
    from neurospatial.annotation import AnnotationResult

    return AnnotationResult(
        environment=_box(),
        regions=Regions(),
        walls={},
        landmarks=Landmarks(),
        calibration=None,
    )


def _track_graph_result() -> Any:
    from neurospatial.annotation._track_helpers import (
        TrackBuilderState,
        build_track_graph_result,
    )

    state = TrackBuilderState()
    state.add_node(0.0, 0.0, label="start")
    state.add_node(100.0, 0.0, label="end")
    state.add_edge(0, 1)
    return build_track_graph_result(state, calibration=None)


def _cumulative_distance() -> Any:
    from neurospatial.behavior import compute_cumulative_distance

    _, times, positions, _, _ = _open_field()
    return compute_cumulative_distance(times, positions)


def _decision_analysis() -> Any:
    from neurospatial.behavior import compute_decision_analysis

    env, _, _, _ = _vte_inputs()
    times = np.arange(100) / 10
    positions = np.column_stack(
        [
            np.r_[np.full(30, 50.0), np.linspace(50, 95, 70)],
            np.r_[np.linspace(5, 55, 30), np.full(70, 55.0)],
        ]
    )
    return compute_decision_analysis(
        env,
        times,
        positions,
        decision_region="center",
        goal_regions=["left", "right"],
        min_speed=5.0,
    )


def _msd() -> Any:
    from neurospatial.behavior import mean_square_displacement

    rng = np.random.default_rng(0)
    positions = np.cumsum(rng.standard_normal((100, 2)) * 5, axis=0)
    return mean_square_displacement(np.arange(100) * 0.1, positions, max_tau=5.0)


@cache
def _line() -> tuple[Any, np.ndarray, np.ndarray]:
    """A 10 cm run East at 10 cm/s, sampled every 0.1 s, on a 1 cm grid."""
    from neurospatial import Environment

    positions = np.column_stack([np.arange(11.0), np.zeros(11)])
    env = Environment.from_samples(positions, bin_size=1.0, units="cm")
    return env, np.arange(11) / 10, positions


def _two_halves() -> Any:
    """The two halves of the ``_line()`` run, ``[0, 0.5)`` and ``[0.5, 1.0)`` s."""
    from neurospatial import Intervals

    return Intervals.from_arrays([0.0, 0.5], [0.5, 1.0])


def _path_efficiency() -> Any:
    from neurospatial.behavior import compute_path_efficiency

    env, times, positions = _line()
    return compute_path_efficiency(env, times, positions, goal=positions[-1])


def _path_efficiency_intervals() -> Any:
    from neurospatial.behavior import compute_path_efficiency

    env, times, positions = _line()
    return compute_path_efficiency(
        env, times, positions, _two_halves(), goal=positions[-1]
    )


def _goal_directed() -> Any:
    from neurospatial.behavior import compute_goal_directed_metrics

    env, times, positions = _line()
    return compute_goal_directed_metrics(
        env, times, positions, goal=positions[-1], goal_radius=1.5
    )


def _goal_directed_intervals() -> Any:
    from neurospatial.behavior import compute_goal_directed_metrics

    env, times, positions = _line()
    return compute_goal_directed_metrics(
        env, times, positions, _two_halves(), goal=positions[-1], goal_radius=1.5
    )


def _pre_decision() -> Any:
    from neurospatial.behavior import compute_pre_decision_metrics

    _, times, positions = _line()
    return compute_pre_decision_metrics(
        times, positions, entry_time=0.8, window_duration=0.5, min_speed=5.0
    )


def _resampled_path() -> Any:
    from neurospatial.behavior import resample_by_distance

    times = np.array([0.0, 1.0, 2.0, 3.0])
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    return resample_by_distance(times, positions, spacing=0.5)


def _subgoal_efficiency() -> Any:
    from neurospatial import Environment
    from neurospatial.behavior import subgoal_efficiency

    positions = np.c_[np.arange(11.0), np.zeros(11)]
    env = Environment.from_samples(positions, bin_size=1.0, units="cm")
    subgoals = np.array([[5.0, 0.0], [10.0, 0.0]])
    return subgoal_efficiency(
        env, np.arange(11) / 10, positions, subgoals, metric="euclidean"
    )


def _vte_session() -> Any:
    from neurospatial.behavior import compute_vte_session

    env, times, positions, trials = _vte_inputs()
    return compute_vte_session(
        env,
        times,
        positions,
        region="choice",
        trials=trials,
        min_speed=5.0,
    )


def _vte_trial() -> Any:
    from neurospatial.behavior import compute_vte_trial

    times = np.linspace(0, 3, 90)
    positions = np.column_stack([np.linspace(0, 50, 90), np.ones(90) * 50])
    return compute_vte_trial(times, positions, 1.0, 2.0, min_speed=5.0)


def _joint_decoding() -> Any:
    from neurospatial.decoding import decode_jointly

    rates = _track_rates()
    counts = np.random.default_rng(0).poisson(rates[:, 3] * 0.1, size=(5, 8))
    return decode_jointly(
        _track(), counts, {"outbound": rates, "inbound": rates[::-1]}, dt=0.1
    )


def _cell_type_significance() -> Any:
    from neurospatial.encoding import spatial_cell_type_significance

    env, times, positions, _, trains = _open_field()
    # The short random session has few spikes per bin; the data-quality
    # warnings about it are not what these instances exercise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spatial_cell_type_significance(
            env, trains, times, positions, n_shuffles=9, min_shift=1.0, rng=0
        )


def _place_cell_significance() -> Any:
    from neurospatial.encoding import place_cell_significance

    env, times, positions, _, trains = _open_field()
    # The short random session has few spikes per bin; the data-quality
    # warnings about it are not what these instances exercise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return place_cell_significance(
            env, trains, times, positions, n_shuffles=9, min_shift=1.0, rng=0
        )


def _phase_precession() -> Any:
    from neurospatial.encoding import compute_phase_precession

    positions = np.linspace(0, 50, 100)
    phases = (2 * np.pi - positions * 0.1) % (2 * np.pi)
    return compute_phase_precession(positions, phases, n_shuffles=19, rng=0)


def _place_fields() -> Any:
    from neurospatial.encoding import detect_place_fields

    env = _track()
    return detect_place_fields(env, _track_rates()[2], bandwidth=0.0)


def _population_coverage() -> Any:
    from neurospatial.encoding import population_coverage

    return population_coverage(_track(), _track_rates(), bandwidth=0.0)


def _rate_comparison() -> Any:
    from neurospatial.encoding import compare_spatial_rates, compute_spatial_rates

    env, times, positions, _, trains = _open_field()
    halves = [
        compute_spatial_rates(
            env,
            trains,
            times,
            positions,
            bandwidth=10.0,
            epochs=[window],
            unit_ids=[10, 11, 12],
        )
        for window in ([0.0, 25.0], [25.0, 50.0])
    ]
    return compare_spatial_rates(*halves)


def _continuous_event() -> Any:
    from neurospatial.events import align_continuous_to_events

    times = np.arange(100) / 10
    return align_continuous_to_events(
        times, 2 * times, [2.0, 5.0], (-1.0, 1.0), dt=0.5, feature_ids=["speed"]
    )


def _peri_event_distance() -> Any:
    from neurospatial.events import peri_event_distance_histogram

    times = np.arange(0.0, 10.0, 0.1)
    return peri_event_distance_histogram(
        np.array([5.2, 5.4]),
        np.array([5.0]),
        times,
        (times * 10.0)[:, None],
        (-20.0, 20.0),
        bin_size=10.0,
    )


def _spatial_feature() -> Any:
    from neurospatial.ops import sample_spatial_features

    env = _box()
    height = np.where(env.bin_centers[:, 0] > 50.0, 2.0, 0.0)
    positions = np.array([[20.0, 50.0], [80.0, 50.0]])
    return sample_spatial_features(env, height[None], positions, feature_ids=["h"])


def _viewshed() -> Any:
    from neurospatial.ops import viewshed

    return viewshed(_box(), np.array([50.0, 50.0]), heading=0.0)


def _assembly_event_response() -> Any:
    from neurospatial.reactivation import assembly_event_response

    times = np.arange(0.005, 10.0, 0.01)
    events = np.array([2.0, 5.0, 8.0])
    activation = np.zeros(times.size)
    for event in events:
        activation[(times > event + 0.02) & (times < event + 0.06)] = 4.0
    return assembly_event_response(times, activation, events, (-0.5, 0.5))


def _assembly_member_cofiring() -> Any:
    from neurospatial.reactivation import assembly_member_cofiring

    pre, post = _three_unit_counts()
    return assembly_member_cofiring(pre, post, _pattern())


def _assembly_reactivation() -> Any:
    from neurospatial.reactivation import assembly_reactivation

    pre, post = _three_unit_counts()
    return assembly_reactivation(pre, post, _pattern())


def _assembly_similarity() -> Any:
    from neurospatial.reactivation import AssemblyPattern, assembly_similarity

    day = [
        AssemblyPattern(np.array([0.8, 0.6, 0.0]), np.array([0]), 0.3),
        AssemblyPattern(np.array([0.0, 0.0, 1.0]), np.array([2]), 0.2),
    ]
    other = [AssemblyPattern(np.array([0.0, -0.6, -0.8]), np.array([2]), 0.3)]
    return assembly_similarity(day, other)


def _explained_variance_contributions() -> Any:
    from neurospatial.reactivation import explained_variance_contributions

    return explained_variance_contributions(*_correlations())


def _explained_variance() -> Any:
    from neurospatial.reactivation import explained_variance_reactivation

    return explained_variance_reactivation(*_correlations())


def _bias_similarity() -> Any:
    from neurospatial.replay import bias_similarity_significance

    template = np.sign(np.subtract.outer(np.arange(12), np.arange(12))).T * 1.0
    spikes = [np.array([0.005 * i, 0.06 + 0.005 * i]) for i in range(12)]
    return bias_similarity_significance(
        spikes, (0.0, 0.2), template, n_shuffles=19, rng=0
    )


def _circular_sequence() -> Any:
    import networkx as nx

    from neurospatial import Environment
    from neurospatial.replay import fit_circular_sequence

    loop = nx.cycle_graph(16)
    for node in loop.nodes:
        angle = 2 * np.pi * node / 16
        loop.nodes[node]["pos"] = (40.0 * np.cos(angle), 40.0 * np.sin(angle))
    env = Environment.from_graph(
        loop,
        edge_order=[(i, (i + 1) % 16) for i in range(16)],
        bin_size=4.0,
        units="cm",
    )
    posterior = np.zeros((12, env.n_bins))
    posterior[np.arange(12), (40 + 4 * np.arange(12)) % env.n_bins] = 1.0
    return fit_circular_sequence(env, posterior, np.arange(12) * 0.01)


def _null_tables() -> tuple[Any, Any]:
    import pandas as pd

    rng = np.random.default_rng(0)
    observed = pd.DataFrame({"score": [0.9] * 8 + [0.1] * 2})
    null = pd.DataFrame(
        {
            "dataset": np.repeat(np.arange(19), 10),
            "score": np.where(rng.random(190) < 0.3, 0.9, 0.1),
        }
    )
    return observed, null


def _dataset_test() -> Any:
    from neurospatial.replay import CriterionGrid, dataset_test

    observed, null = _null_tables()
    return dataset_test(observed, null, criteria=[CriterionGrid("score", ">", 0.5)])


def _distribution_test() -> Any:
    from neurospatial.replay import distribution_test

    observed, null = _null_tables()
    return distribution_test(observed, null, column="score")


def _isotonic_fit() -> Any:
    from neurospatial.replay import fit_isotonic_trajectory

    return fit_isotonic_trajectory(*_sweep())


def _linear_fit() -> Any:
    from neurospatial.replay import fit_linear_trajectory

    return fit_linear_trajectory(*_sweep(), method="map")


def _matching_index() -> Any:
    from neurospatial.replay import sequence_matching_index

    return sequence_matching_index(
        _track(), _forward_spikes(), (1.0, 1.1), _track_rates()
    )


def _pairwise_bias() -> Any:
    from neurospatial.replay import pairwise_bias_matrix

    spikes = [np.array([0.01, 0.02]), np.array([0.03, 0.04]), np.array([])]
    return pairwise_bias_matrix(spikes, (0.0, 0.1))


def _pairwise_order() -> Any:
    from neurospatial.replay import pairwise_order_score

    return pairwise_order_score(_track(), _forward_spikes(), (1.0, 1.1), _track_rates())


def _pairwise_order_significance() -> Any:
    from neurospatial.replay import pairwise_order_significance

    return pairwise_order_significance(
        _track(), _forward_spikes(), (1.0, 1.1), _track_rates(), n_shuffles=19, rng=0
    )


def _radon() -> Any:
    from neurospatial.replay import detect_trajectory_radon

    return detect_trajectory_radon(*_sweep())


def _rank_order() -> Any:
    from neurospatial.replay import rank_order_correlation

    return rank_order_correlation(
        _track(), _forward_spikes(), (1.0, 1.1), _track_rates()
    )


def _rank_order_significance() -> Any:
    from neurospatial.replay import rank_order_significance

    return rank_order_significance(
        _track(), _forward_spikes(), (1.0, 1.1), _track_rates(), n_shuffles=19, rng=0
    )


def _recurring_trains() -> list[np.ndarray]:
    starts = np.arange(5.0)
    return [starts, starts + 0.03, starts + 0.08]


def _recurring_sequences() -> Any:
    from neurospatial.replay import recurring_spike_sequences

    return recurring_spike_sequences(_recurring_trains(), reference_unit=0)


def _recurring_sequences_significance() -> Any:
    from neurospatial.replay import recurring_sequences_significance
    from neurospatial.replay.methods import NADASDY_1999

    method = dataclasses.replace(
        NADASDY_1999, surrogates=("across_trains",), n_surrogates=9
    )
    return recurring_sequences_significance(
        _recurring_trains(),
        recurring_sequence_method=method,
        reference_unit=0,
        rng=0,
    )


def _regression_line_score() -> Any:
    from neurospatial.replay import regression_line_score

    return regression_line_score(*_sweep())


def _replay_detection() -> Any:
    from neurospatial.replay import detect_replay
    from neurospatial.replay.methods import SILVA_2015

    return detect_replay(*_sweep(), replay_method=SILVA_2015)


def _replay_segments() -> Any:
    from neurospatial.replay import replay_segments

    return replay_segments(*_sweep(), max_jump=20.0)


def _replay_significance() -> Any:
    from neurospatial.replay import replay_significance

    return replay_significance(*_sweep(), n_shuffles=19, rng=0)


def _sequence_match() -> Any:
    from neurospatial.replay import best_sequence_match

    return best_sequence_match(_track(), _forward_spikes(), (1.0, 1.1), _track_rates())


def _spike_words() -> Any:
    from neurospatial.replay import spike_words

    spikes = [np.array([1.0, 1.02, 1.4]), np.array([1.06, 1.45])]
    return spike_words(spikes, max_isi=0.05, word_gap=0.1)


def _template_match_dataset() -> Any:
    from neurospatial.replay import template_match_dataset_test
    from neurospatial.replay.methods import JI_WILSON_2007

    env = _track()
    x = env.bin_centers[:, 0]
    rates = 20.0 * np.exp(
        -0.5 * ((x - np.linspace(15.0, 105.0, 6)[:, None]) / 8.0) ** 2
    )
    onsets = np.arange(20) * 2.0
    spikes = [onsets + 0.05 * i for i in range(6)]
    frames = np.column_stack([onsets, onsets + 0.3])
    return template_match_dataset_test(
        env, spikes, frames, rates, template_match_method=JI_WILSON_2007, rng=0
    )


def _weighted_correlation() -> Any:
    from neurospatial.replay import weighted_correlation

    return weighted_correlation(*_sweep())


def _trajectory_metrics() -> Any:
    from neurospatial.replay import trajectory_metrics

    return trajectory_metrics(*_sweep())


def _null_spike_data() -> Any:
    from neurospatial.replay import null_events

    trains = [np.array([0.01, 0.05, 0.12]), np.array([0.03, 0.15]), np.array([])]
    return null_events(trains, np.array([[0.0, 0.2]]), kind="spike_shift", rng=0)


def _shuffled_dataset() -> Any:
    from neurospatial.decoding import DecodingResult
    from neurospatial.replay import shuffle_events

    env, posterior, times = _sweep()
    event = DecodingResult(posterior=posterior, env=env, times=times)
    (dataset,) = shuffle_events(env, [event], null="time_bin_permutation", rng=0)
    return dataset


def _replay_event_simulation() -> Any:
    from neurospatial.simulation import simulate_replay_event

    return simulate_replay_event(_track(), _track_rates(), speed=500.0, rng=0)


def _simulation_validation() -> Any:
    from neurospatial.simulation import linear_track_session, validate_simulation

    session = linear_track_session(duration=60.0, n_place_cells=3, n_laps=6, rng=0)
    return validate_simulation(session)


def _circular_basis() -> Any:
    from neurospatial.stats import circular_basis

    return circular_basis(np.random.default_rng(0).uniform(0, 2 * np.pi, 50))


def _shuffle_test() -> Any:
    from neurospatial.stats import ShuffleTestResult

    null = np.random.default_rng(0).normal(2.0, 1.0, 50)
    return ShuffleTestResult.from_null(5.0, null, shuffle_type="time_bins")


RESULT_FACTORIES: dict[str, Callable[[], Any]] = {
    "DecodingResult": _decoding_result,
    "DecodingSummary": _decoding_summary,
    "PeriEventResult": _peri_event_result,
    "SpatialRateResult": lambda: _spatial_rates()[0],
    "SpatialRatesResult": _spatial_rates,
    "AnnotationResult": _annotation_result,
    "TrackGraphResult": _track_graph_result,
    "CumulativeDistanceResult": _cumulative_distance,
    "DecisionAnalysisResult": _decision_analysis,
    "DecisionBoundaryMetrics": lambda: _decision_analysis().boundary,
    "GoalDirectedMetrics": _goal_directed,
    "MSDResult": _msd,
    "PathEfficiencyResult": _path_efficiency,
    "PreDecisionMetrics": _pre_decision,
    "ResampledPathResult": _resampled_path,
    "SubgoalEfficiencyResult": _subgoal_efficiency,
    "VTESessionResult": _vte_session,
    "VTETrialResult": _vte_trial,
    "JointDecodingResult": _joint_decoding,
    "BoundaryVectorRateResult": lambda: _boundary_vector_rates()[0],
    "BoundaryVectorRatesResult": _boundary_vector_rates,
    "CellTypeSignificanceResult": _cell_type_significance,
    "PlaceCellSignificanceResult": _place_cell_significance,
    "DirectionalRateResult": lambda: _directional_rates()[0],
    "DirectionalRatesResult": _directional_rates,
    "DirectionalPlaceFields": _directional_place_fields,
    "GridProperties": _grid_properties,
    "ObjectVectorRateResult": lambda: _object_vector_rates()[0],
    "ObjectVectorRatesResult": _object_vector_rates,
    "PhasePrecessionResult": _phase_precession,
    "PlaceFieldsResult": _place_fields,
    "PopulationCoverageResult": _population_coverage,
    "RateComparisonResult": _rate_comparison,
    "TuningCurveResult": lambda: _tuning_curves()[0],
    "TuningCurvesResult": _tuning_curves,
    "ViewRateResult": lambda: _view_rates()[0],
    "ViewRatesResult": _view_rates,
    "ContinuousEventResult": _continuous_event,
    "PeriEventDistanceResult": _peri_event_distance,
    "PopulationPeriEventResult": _population_psth,
    "SpatialFeatureResult": _spatial_feature,
    "ViewshedResult": _viewshed,
    "AssemblyDetectionResult": _assemblies,
    "AssemblyPattern": lambda: _assemblies().patterns[0],
    "AssemblyEventResponseResult": _assembly_event_response,
    "AssemblyMemberCofiringResult": _assembly_member_cofiring,
    "AssemblyReactivationResult": _assembly_reactivation,
    "AssemblySimilarityResult": _assembly_similarity,
    "ExplainedVarianceContributionResult": _explained_variance_contributions,
    "ExplainedVarianceResult": _explained_variance,
    "BiasSimilarityResult": _bias_similarity,
    "CircularSequenceResult": _circular_sequence,
    "DatasetTestResult": _dataset_test,
    "DistributionTestResult": _distribution_test,
    "IsotonicFitResult": _isotonic_fit,
    "LinearFitResult": _linear_fit,
    "MatchingIndexResult": _matching_index,
    "PairwiseBiasResult": _pairwise_bias,
    "PairwiseOrderResult": _pairwise_order,
    "PairwiseOrderSignificanceResult": _pairwise_order_significance,
    "RadonDetectionResult": _radon,
    "RankOrderResult": _rank_order,
    "RankOrderSignificanceResult": _rank_order_significance,
    "RecurringSequencesResult": _recurring_sequences,
    "RecurringSequencesSignificanceResult": _recurring_sequences_significance,
    "RegressionLineScoreResult": _regression_line_score,
    "ReplayDetectionResult": _replay_detection,
    "ReplaySegmentsResult": _replay_segments,
    "ReplaySignificanceResult": _replay_significance,
    "SequenceMatchResult": _sequence_match,
    "SpikeWordsResult": _spike_words,
    "TemplateMatchDatasetResult": _template_match_dataset,
    "WeightedCorrelationResult": _weighted_correlation,
    "TrajectoryMetrics": _trajectory_metrics,
    "NullSpikeData": _null_spike_data,
    "ShuffledDataset": _shuffled_dataset,
    "ReplayEventSimulation": _replay_event_simulation,
    "SimulationValidationResult": _simulation_validation,
    "CircularBasisResult": _circular_basis,
    "ShuffleTestResult": _shuffle_test,
}

#: Further forms of a class whose summary changes with its inputs, keyed
#: ``"ClassName[form]"``: interval results summarize over their intervals.
RESULT_VARIANTS: dict[str, Callable[[], Any]] = {
    "GoalDirectedMetrics[intervals]": _goal_directed_intervals,
    "PathEfficiencyResult[intervals]": _path_efficiency_intervals,
}
