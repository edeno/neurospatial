"""Every simulator reads ``rng`` like ``np.random.default_rng``."""

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.simulation import (
    BoundaryVectorCellModel,
    HeadDirectionCellModel,
    PlaceCellModel,
    border_cell_session,
    generate_poisson_spikes,
    grid_cell_session,
    linear_track_session,
    open_field_session,
    simulate_replay_event,
    simulate_session,
    simulate_trajectory_laps,
    simulate_trajectory_ou,
    tmaze_alternation_session,
)

SESSIONS = {
    "open_field": (open_field_session, {"duration": 10.0, "n_place_cells": 3}),
    "linear_track": (
        linear_track_session,
        {"duration": 10.0, "n_place_cells": 3, "n_laps": 2},
    ),
    "tmaze": (
        tmaze_alternation_session,
        {"duration": 10.0, "n_trials": 2, "n_place_cells": 3},
    ),
    "border": (
        border_cell_session,
        {"duration": 5.0, "n_border_cells": 2, "n_place_cells": 2},
    ),
    "grid": (grid_cell_session, {"duration": 5.0, "n_grid_cells": 2}),
}


def _content(session):
    return [
        session.times,
        session.positions,
        *[np.asarray(train) for train in session.spike_times],
    ]


def _assert_same(first, second):
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", SESSIONS)
def test_simulation_rng_reproducible(name):
    """An int ``rng`` repeats, and a generator seeded alike matches it."""
    function, options = SESSIONS[name]
    first = function(rng=0, **options)
    _assert_same(_content(first), _content(function(rng=0, **options)))
    seeded = function(rng=np.random.default_rng(0), **options)
    _assert_same(_content(first), _content(seeded))
    assert first.metadata["rng"] == 0
    assert seeded.metadata["rng"] is None  # a generator cannot be recorded
    other = function(rng=1, **options)
    assert not np.array_equal(first.positions, other.positions)


def test_int_rng_reproduces_the_output_of_the_same_int_seed_before():
    """Values recorded before ``seed`` became ``rng``."""
    session = open_field_session(duration=20.0, n_place_cells=5, rng=0)
    assert [len(train) for train in session.spike_times] == [0, 11, 87, 0, 31]
    np.testing.assert_allclose(
        session.positions[-1], [45.96608746350471, 43.627687202420965], rtol=1e-12
    )
    tmaze = tmaze_alternation_session(
        duration=20.0, n_trials=3, n_place_cells=3, rng=42
    )
    # The first choice is the fourth stream spawned from the seed:
    # default_rng(SeedSequence(42).spawn(4)[3].generate_state(1)[0]).
    assert tmaze.metadata["trial_choices"] == ("right", "left", "right")


def _arena():
    return Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")


def _int_and_generator(rng_seed, run):
    """``run(rng)`` for the int seed and for a generator seeded alike."""
    return run(rng_seed), run(np.random.default_rng(rng_seed))


def test_trajectory_and_spike_simulators_match_between_int_and_generator():
    env = _arena()
    rate, times = np.full(2000, 20.0), np.linspace(0, 20, 2000)
    runs = {
        "ou": lambda rng: simulate_trajectory_ou(
            env, duration=5.0, speed_units="cm", rng=rng
        ),
        "laps": lambda rng: simulate_trajectory_laps(env, n_laps=1, rng=rng)[:2],
        "poisson": lambda rng: (generate_poisson_spikes(rate, times, rng=rng),),
        "session": lambda rng: _content(
            simulate_session(
                env,
                duration=5.0,
                n_cells=4,
                cell_type="mixed",
                speed_units="cm",
                show_progress=False,
                rng=rng,
            )
        ),
        "place_cell_center": lambda rng: (
            PlaceCellModel(env, rng=rng).ground_truth["center"],
        ),
    }
    for run in runs.values():
        first, second = _int_and_generator(3, run)
        _assert_same(list(first), list(second))


def test_replay_event_rng_matches_between_int_and_generator():
    track = linear_track_session(duration=10.0, n_place_cells=4, n_laps=2, rng=0).env
    rates = [
        np.exp(-((track.bin_centers[:, 0] - center) ** 2) / 200.0)
        for center in (20, 60, 100, 140)
    ]
    events = [
        simulate_replay_event(
            track, rates, kind="sweep", speed=400.0, duration=0.3, rng=rng
        )
        for rng in (2, np.random.default_rng(2))
    ]
    _assert_same(
        [np.asarray(train) for train in events[0].spike_times],
        [np.asarray(train) for train in events[1].spike_times],
    )


def test_the_old_keyword_is_refused():
    rate, times = np.full(10, 5.0), np.linspace(0, 1, 10)
    with pytest.raises(TypeError, match="seed"):
        generate_poisson_spikes(rate, times, seed=0)
    with pytest.raises(TypeError, match="seed"):
        simulate_trajectory_ou(_arena(), duration=1.0, speed_units="cm", seed=0)


@pytest.mark.parametrize("name", SESSIONS)
def test_metadata_rng_reproduces_the_session(name):
    """The recorded ``rng`` is an int, or the entropy drawn for ``None``."""
    function, options = SESSIONS[name]
    drawn = function(rng=None, **options)
    recorded = drawn.metadata["rng"]
    assert isinstance(recorded, int)
    again = function(rng=recorded, **options)
    _assert_same(_content(drawn), _content(again))
    assert again.metadata["rng"] == recorded


def test_a_generator_that_was_drawn_from_still_starts_the_rng_0_session():
    function, options = SESSIONS["open_field"]
    used = np.random.default_rng(0)
    used.random(100)
    _assert_same(
        _content(function(rng=used, **options)), _content(function(rng=0, **options))
    )


def test_the_same_generator_twice_gives_different_sessions():
    function, options = SESSIONS["open_field"]
    generator = np.random.default_rng(0)
    first = function(rng=generator, **options)
    second = function(rng=generator, **options)
    assert not np.array_equal(first.positions, second.positions)


_DRAWN = {"center", "preferred_direction"}

MODEL_BUILDERS = {
    "place": lambda env, rng: PlaceCellModel(env, rng=rng),
    "boundary_vector": lambda env, rng: BoundaryVectorCellModel(env, rng=rng),
    "head_direction": lambda env, rng: HeadDirectionCellModel(rng=rng),
}


def _drawn(model):
    """The randomly drawn parameter of ``model``, from its ground truth."""
    truth = model.ground_truth
    (key,) = _DRAWN.intersection(truth)
    return np.asarray(truth[key])


def _expected_draws(env, name, generator, n_models):
    """The draws ``n_models`` models should make from ``generator``, in order.

    Derived from the documented distributions, not from the models: a place
    centre is a uniformly drawn bin centre, and a direction is uniform on
    [-pi, pi), wrapped to (-pi, pi].
    """
    if name == "place":
        return [
            env.bin_centers[generator.integers(0, env.n_bins)] for _ in range(n_models)
        ]
    directions = generator.uniform(-np.pi, np.pi, n_models)
    return list(np.where(directions == -np.pi, np.pi, directions))


@pytest.mark.parametrize("name", MODEL_BUILDERS)
def test_successive_models_from_one_generator_differ(name):
    """A generator is advanced by each model, as ``default_rng`` reads it."""
    env, build = _arena(), MODEL_BUILDERS[name]
    generator = np.random.default_rng(0)
    draws = [_drawn(build(env, generator)) for _ in range(3)]
    expected = _expected_draws(env, name, np.random.default_rng(0), 3)
    for drawn, value in zip(draws, expected, strict=True):
        # Wrapping a direction to (-pi, pi] can move it by one ULP.
        np.testing.assert_allclose(drawn, value, rtol=0, atol=1e-12)
    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[1], draws[2])


@pytest.mark.parametrize("name", MODEL_BUILDERS)
def test_an_int_and_a_generator_seeded_alike_give_the_same_model(name):
    env, build = _arena(), MODEL_BUILDERS[name]
    from_int = build(env, 0)
    from_generator = build(env, np.random.default_rng(0))
    np.testing.assert_array_equal(_drawn(from_int), _drawn(from_generator))
    (expected,) = _expected_draws(env, name, np.random.default_rng(0), 1)
    np.testing.assert_allclose(_drawn(from_int), expected, rtol=0, atol=1e-12)
    times = np.linspace(0.0, 1.0, 50)
    positions = np.random.default_rng(9).uniform(5, 95, (50, 2))
    headings = np.random.default_rng(10).uniform(-np.pi, np.pi, 50)
    np.testing.assert_array_equal(
        from_int.firing_rate(times, positions, headings=headings),
        from_generator.firing_rate(times, positions, headings=headings),
    )


@pytest.mark.parametrize("rng", [None, 5, "generator"])
@pytest.mark.parametrize("name", MODEL_BUILDERS)
def test_replace_pickle_and_deepcopy_keep_a_models_draw(name, rng):
    import copy
    import dataclasses
    import pickle

    env, build = _arena(), MODEL_BUILDERS[name]
    model = build(env, np.random.default_rng(5) if rng == "generator" else rng)
    drawn = _drawn(model)
    rebuilt = [
        dataclasses.replace(model, max_rate=20.0),
        dataclasses.replace(dataclasses.replace(model), max_rate=20.0),
        pickle.loads(pickle.dumps(model)),
        copy.deepcopy(model),
    ]
    for other in rebuilt:
        np.testing.assert_array_equal(_drawn(other), drawn)
    # Rebuilding again from the same model gives the same draw again.
    np.testing.assert_array_equal(_drawn(dataclasses.replace(model)), drawn)


def test_tmaze_first_choice_varies_across_calls_with_one_generator():
    """Each call advances the generator, so the first arm is not fixed."""
    generator = np.random.default_rng(1)
    sessions = [
        tmaze_alternation_session(
            duration=4.0, n_trials=2, n_place_cells=1, rng=generator
        )
        for _ in range(8)
    ]
    first_choices = {session.metadata["trial_choices"][0] for session in sessions}
    assert first_choices == {"left", "right"}
