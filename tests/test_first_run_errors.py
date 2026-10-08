"""Public first-run errors explain the inputs and corrected calls."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rate
from neurospatial.encoding._validation import validate_spike_times, validate_trajectory


@dataclass(frozen=True)
class Recording:
    env: Environment
    times: np.ndarray
    positions: np.ndarray
    spike_times: np.ndarray


@pytest.fixture(scope="module")
def recording():
    times = np.arange(1800) / 30.0
    positions = (
        50 + 40 * np.c_[np.sin(2 * np.pi * times / 20), np.cos(2 * np.pi * times / 13)]
    )
    env = Environment.from_samples(positions, bin_size=4.0, units="cm")
    spikes = np.sort(np.random.default_rng(0).uniform(0, times[-1], 300))
    return Recording(env, times, positions, spikes)


def test_missing_env_names_the_call(recording):
    r = recording
    with pytest.raises(TypeError, match="expects an Environment") as caught:
        compute_spatial_rate(r.spike_times, r.times, r.positions)
    message = str(caught.value)
    assert "compute_spatial_rate" in message
    assert "ndarray" in message
    assert message.splitlines()[-1].startswith("Fix: ")
    assert "compute_spatial_rate(env, spike_times, times, positions)" in message


def test_1d_positions_on_2d_env(recording):
    r = recording
    with pytest.raises(ValueError, match=r"shape \(1800,\).*2-D") as caught:
        compute_spatial_rate(r.env, r.spike_times, r.times, r.positions[:, 0])
    assert "Fix:" in str(caught.value)
    assert "grid_edges" not in str(caught.value)


def test_swapped_times_positions(recording):
    r = recording
    with pytest.raises(ValueError, match="did you pass positions before times"):
        compute_spatial_rate(r.env, r.spike_times, r.positions, r.times)


def test_trajectory_reports_length_and_dimensions_together(recording):
    r = recording
    positions = np.zeros((len(r.times) - 1, 3))
    with pytest.raises(ValueError) as caught:
        compute_spatial_rate(r.env, r.spike_times, r.times, positions)
    message = str(caught.value)
    assert "times has 1800" in message
    assert "positions has 1799" in message
    assert "2-D" in message
    assert len([line for line in message.splitlines() if line.startswith("- ")]) == 2
    assert message.splitlines()[-1].startswith("Fix: ")


def test_egocentric_population_reports_length_and_dimensions_together(recording):
    from neurospatial.encoding import compute_egocentric_rates

    r = recording
    with pytest.raises(ValueError) as caught:
        compute_egocentric_rates(
            r.env,
            [r.spike_times],
            r.times,
            np.zeros((len(r.times) - 1, 3)),
            np.zeros(len(r.times)),
            np.array([[50.0, 50.0]]),
        )
    message = str(caught.value)
    assert "compute_egocentric_rates" in message
    assert "positions has 1799" in message
    assert "2-D" in message
    assert len([line for line in message.splitlines() if line.startswith("- ")]) == 2


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("decode_position", "spike_counts, encoding_models, dt"),
        ("decode_position_summary", "spike_counts, encoding_models, dt"),
        ("compute_view_rate", "spike_times, times, positions, headings"),
        ("compute_view_rates", "spike_times, times, positions, headings"),
        (
            "compute_egocentric_rate",
            "spike_times, times, positions, headings, object_positions",
        ),
        (
            "compute_egocentric_rates",
            "spike_times, times, positions, headings, object_positions",
        ),
    ],
)
def test_missing_environment_shows_the_complete_call(recording, name, arguments):
    import neurospatial.decoding as decoding
    import neurospatial.encoding as encoding

    r = recording
    if name.startswith("decode_"):
        function = getattr(decoding, name)
        args = (r.spike_times, np.ones((2, 1)), np.ones((r.env.n_bins, 1)), 0.1)
    else:
        function = getattr(encoding, name)
        spikes = [r.spike_times] if name.endswith("rates") else r.spike_times
        args = (r.spike_times, spikes, r.times, r.positions, np.zeros(len(r.times)))
        if "egocentric" in name:
            args += (np.array([[50.0, 50.0]]),)
    with pytest.raises(TypeError, match="expects an Environment") as caught:
        function(*args)
    assert f"{name}(env, {arguments})" in str(caught.value).splitlines()[-1]


@pytest.mark.parametrize("name", ["align_spikes_to_events", "peri_event_histogram"])
def test_window_strings_name_the_call(name):
    import neurospatial.events as events

    function = getattr(events, name)
    with pytest.raises(ValueError) as caught:
        function(np.array([0.1]), np.array([0.0, 1.0]), window=("-0.5", "1.0"))
    message = str(caught.value)
    assert name in message
    assert "window" in message
    assert "Why:" in message
    assert message.splitlines()[-1].startswith("Fix: ")


def test_unknown_animation_backend_lists_widget(recording):
    with pytest.raises(ValueError) as caught:
        recording.env.animate_fields(
            np.zeros((2, recording.env.n_bins)),
            frame_times=np.array([0.0, 0.1]),
            backend="unknown",
        )
    fix = str(caught.value).splitlines()[-1]
    for backend in ["auto", "napari", "video", "html", "widget"]:
        assert repr(backend) in fix


def test_trajectory_reports_time_and_heading_problems_together():
    with pytest.raises(ValueError) as caught:
        validate_trajectory(
            np.array([1.0, 0.0, np.nan]),
            headings=np.ones((2, 2)),
            context="compute_directional_rate",
        )
    message = str(caught.value)
    assert "compute_directional_rate" in message
    for problem in [
        "finite",
        "monotonically non-decreasing",
        "headings must be 1D",
        "headings length (2)",
    ]:
        assert problem in message
    assert "Fix:" in message


def test_spike_validation_reports_all_problems():
    with pytest.raises(ValueError) as caught:
        validate_spike_times(np.array([1.0, -1.0, np.nan]), context="decode_position")
    message = str(caught.value)
    assert "decode_position" in message
    assert "finite" in message
    assert "non-negative" in message
    assert "monotonically non-decreasing" in message
    assert "Fix:" in message


def test_valid_linear_trajectory_is_accepted():
    validate_trajectory(
        np.arange(10.0),
        positions=np.arange(10.0),
        n_dims=1,
        context="compute_spatial_rate",
    )


@pytest.mark.parametrize("bad_shape", [(10, 3), (10, 2, 1)])
def test_invalid_coordinate_dimensions_name_the_call(bad_shape):
    with pytest.raises(ValueError) as caught:
        validate_trajectory(
            np.arange(10.0),
            positions=np.ones(bad_shape),
            n_dims=2,
            context="compute_view_rate",
        )
    assert "compute_view_rate" in str(caught.value)
    assert "positions" in str(caught.value)
    assert "Fix:" in str(caught.value)


@pytest.mark.parametrize(
    "site",
    [
        "direct_environment",
        "factory_shape",
        "maze_kind",
        "all_nan",
        "bin_size_zero",
        "bin_size_nan",
        "bin_size_inf",
        "many_points",
        "outside_point",
        "unit_labels",
        "decoder_units",
        "likelihood_units",
        "decoder_bins",
        "degenerate_posterior",
        "decode_dt",
        "window_spikes",
        "window_histogram",
        "window_population",
        "window_events",
        "animation_rank",
        "animation_empty",
        "animation_path",
        "animation_backend",
        "lap_reference",
        "lap_start",
        "trials_empty",
        "files_overwrite",
        "files_metadata",
        "files_arrays",
    ],
)
def test_rewritten_sites_teach(site, recording, tmp_path):
    from neurospatial._results import resolve_unit_ids
    from neurospatial.animation.core import animate_fields
    from neurospatial.behavior import detect_laps, segment_trials
    from neurospatial.decoding import bin_spikes_in_time, decode_position
    from neurospatial.decoding.likelihood import log_poisson_likelihood
    from neurospatial.decoding.posterior import normalize_to_posterior
    from neurospatial.environment.factories import _assemble_maze_graph
    from neurospatial.events import (
        align_events,
        align_spikes_to_events,
        peri_event_histogram,
        population_peri_event_histogram,
    )

    r = recording
    bins = r.env.bin_at(r.positions)
    # A local environment avoids mutating the shared fixture's regions.
    env = r.env.copy()
    env.regions.add("home", point=tuple(env.bin_centers[0]))
    models = np.ones((2, env.n_bins))
    calls = {
        "direct_environment": lambda: Environment(),
        "factory_shape": lambda: Environment.from_samples(
            np.arange(10.0), bin_size=2.0
        ),
        "maze_kind": lambda: _assemble_maze_graph(
            "unknown", {0: (0.0, 0.0), 1: (1.0, 1.0)}
        ),
        "all_nan": lambda: Environment.from_samples(
            np.full((10, 2), np.nan), bin_size=2.0
        ),
        "bin_size_zero": lambda: Environment.from_samples(r.positions, bin_size=0),
        "bin_size_nan": lambda: Environment.from_samples(r.positions, bin_size=np.nan),
        "bin_size_inf": lambda: Environment.from_samples(r.positions, bin_size=np.inf),
        "many_points": lambda: env.neighbors(env.bin_centers[:2]),
        "outside_point": lambda: env.neighbors([-1000.0, -1000.0]),
        "unit_labels": lambda: resolve_unit_ids(
            [1], n_units=2, context="compute_spatial_rates"
        ),
        "decoder_units": lambda: decode_position(
            env, np.ones((3, 1), dtype=int), models, 0.1
        ),
        "likelihood_units": lambda: log_poisson_likelihood(
            np.ones((3, 1), dtype=int), models, 0.1
        ),
        "decoder_bins": lambda: decode_position(
            env, np.ones((3, 2), dtype=int), np.ones((2, env.n_bins + 1)), 0.1
        ),
        "degenerate_posterior": lambda: normalize_to_posterior(
            np.full((2, 3), -np.inf), handle_degenerate="raise"
        ),
        "decode_dt": lambda: bin_spikes_in_time(
            [np.array([0.1])], dt=2.0, t_start=0.0, t_stop=1.0
        ),
        "window_spikes": lambda: align_spikes_to_events(
            r.spike_times, np.array([10.0]), (1.0, -1.0)
        ),
        "window_histogram": lambda: peri_event_histogram(
            r.spike_times, np.array([10.0]), window=(1.0, -1.0)
        ),
        "window_population": lambda: population_peri_event_histogram(
            [r.spike_times], np.array([10.0]), window=(1.0, -1.0)
        ),
        "window_events": lambda: align_events(
            pd.DataFrame({"timestamp": [10.0]}),
            pd.DataFrame({"timestamp": [10.0]}),
            window=(1.0, -1.0),
        ),
        "animation_rank": lambda: animate_fields(
            env, np.ones(env.n_bins), frame_times=np.array([0.0]), backend="video"
        ),
        "animation_empty": lambda: animate_fields(
            env, np.empty((0, env.n_bins)), frame_times=np.empty(0), backend="video"
        ),
        "animation_path": lambda: animate_fields(
            env,
            np.ones((2, env.n_bins)),
            frame_times=np.array([0.0, 0.1]),
            backend="video",
        ),
        "animation_backend": lambda: animate_fields(
            env,
            np.ones((2, env.n_bins)),
            frame_times=np.array([0.0, 0.1]),
            backend="unknown",
        ),
        "lap_reference": lambda: detect_laps(bins, r.times, env, method="reference"),
        "lap_start": lambda: detect_laps(bins, r.times, env, method="region"),
        "trials_empty": lambda: segment_trials(
            bins, r.times, env, start_region="home", end_regions=[]
        ),
    }
    if site.startswith("files_"):
        path = tmp_path / "environment"
        if site != "files_metadata":
            env.to_file(path)
        if site == "files_arrays":
            path.with_suffix(".npz").unlink()
        calls.update(
            {
                "files_overwrite": lambda: env.to_file(path),
                "files_metadata": lambda: Environment.from_file(path),
                "files_arrays": lambda: Environment.from_file(path),
            }
        )
    with pytest.raises(
        (ValueError, TypeError, FileExistsError, FileNotFoundError)
    ) as caught:
        calls[site]()
    assert any(line.startswith("Fix: ") for line in str(caught.value).splitlines()), (
        str(caught.value)
    )


@pytest.mark.parametrize(
    "site",
    [
        "crossing",
        "run_source",
        "run_target",
        "lap",
        "trial_start",
        "trial_end",
        "goal",
        "boundary",
        "distance",
    ],
)
def test_segmentation_unknown_region(site, recording):
    from neurospatial import RegionNotFoundError
    from neurospatial.behavior import (
        detect_goal_directed_runs,
        detect_laps,
        detect_region_crossings,
        detect_runs_between_regions,
        segment_trials,
    )
    from neurospatial.events import distance_to_boundary

    r = recording
    env = r.env.copy()
    env.regions.add("known", point=tuple(env.bin_centers[0]))
    bins = env.bin_at(r.positions)
    calls = {
        "crossing": lambda: detect_region_crossings(
            bins, r.times, env, region_name="home"
        ),
        "run_source": lambda: detect_runs_between_regions(
            bins, r.times, env, source="home", target="known"
        ),
        "run_target": lambda: detect_runs_between_regions(
            bins, r.times, env, source="known", target="home"
        ),
        "lap": lambda: detect_laps(
            bins, r.times, r.env, method="region", start_region="home"
        ),
        "trial_start": lambda: segment_trials(
            bins, r.times, env, start_region="home", end_regions=["known"]
        ),
        "trial_end": lambda: segment_trials(
            bins, r.times, env, start_region="known", end_regions=["home"]
        ),
        "goal": lambda: detect_goal_directed_runs(
            bins, r.times, env, goal_region="home"
        ),
        "boundary": lambda: distance_to_boundary(
            env, r.positions, boundary_type="region", region_name="home"
        ),
        "distance": lambda: env.distance_to("home"),
    }
    with pytest.raises(RegionNotFoundError) as caught:
        calls[site]()
    assert "env.regions.add('home'" in str(caught.value)
    assert str(caught.value).splitlines()[-1].startswith("Fix: ")


@pytest.mark.parametrize("method", ["correlation", "directionality_index"])
def test_direction_label_errors_teach(method, recording):
    from neurospatial.encoding.spatial import DirectionalPlaceFields

    env = recording.env
    rates = {name: np.ones(env.n_bins) for name in ["forward", "reverse"]}
    result = DirectionalPlaceFields(
        firing_rates=rates, occupancy=rates, env=env, labels=tuple(rates)
    )
    with pytest.raises(KeyError) as caught:
        getattr(result, method)("missing", "reverse")
    assert "Fix:" in str(caught.value)
    assert "forward" in str(caught.value)
    assert "\\n" not in str(caught.value)


def test_neighbors_of_bin_at_output(recording):
    r = recording
    index = r.env.bin_at(r.positions[:1])
    with pytest.raises(ValueError, match=r"int\(bin_idx\[0\]\)") as caught:
        r.env.neighbors(index)
    assert "bin_at" in str(caught.value)
    # Unwrapping the returned index is a real, successful correction.
    assert r.env.neighbors(int(index[0])) == r.env.neighbors(r.positions[0])


def test_coarse_bin_size_warns():
    import warnings

    x = np.linspace(0.0, 80.0, 200)
    positions = np.c_[x, x[::-1]]
    with pytest.warns(UserWarning, match="bin_size=500") as caught:
        Environment.from_samples(positions, bin_size=500)
    assert len(caught) == 1
    assert "Fix:" in str(caught[0].message)
    with warnings.catch_warnings(record=True) as normal:
        warnings.simplefilter("always")
        Environment.from_samples(positions, bin_size=2.0)
    assert not normal
    track = np.c_[np.linspace(0.0, 200.0, 200), np.linspace(0.0, 5.0, 200)]
    with warnings.catch_warnings(record=True) as narrow:
        warnings.simplefilter("always")
        Environment.from_samples(track, bin_size=5.0)
    assert not narrow


def test_large_grid_warning_is_visible():
    from neurospatial.layout.helpers.utils import check_grid_size_safety

    with pytest.warns(UserWarning) as caught:
        check_grid_size_safety((500, 500), n_dims=2)
    assert "250,000" in str(caught[0].message)
    assert "Fix:" in str(caught[0].message)


def test_psth_window_in_ms_warns(recording):
    import warnings

    from neurospatial.events import peri_event_histogram

    r = recording
    with pytest.warns(UserWarning, match=r"window=\(-0.5, 1.0\)"):
        peri_event_histogram(
            r.spike_times, np.array([20.0, 30.0]), window=(-500, 1000), bin_size=1.0
        )
    with warnings.catch_warnings(record=True) as normal:
        warnings.simplefilter("always")
        peri_event_histogram(r.spike_times, np.array([20.0, 30.0]), window=(-1.0, 2.0))
    assert not normal


@pytest.mark.parametrize("window", [(), (0.0, 1.0, 2.0), (np.nan, 1.0), (0.0, np.inf)])
def test_invalid_peri_event_window_names_the_call(window):
    from neurospatial.events import align_spikes_to_events

    with pytest.raises(ValueError) as caught:
        align_spikes_to_events(np.array([1.0]), np.array([1.0]), window)
    assert "align_spikes_to_events" in str(caught.value)
    assert "window" in str(caught.value)
    assert "Fix:" in str(caught.value)
