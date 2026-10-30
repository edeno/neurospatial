"""Integration tests for simulation subpackage.

End-to-end workflows: simulate a session, estimate rate maps, and check that
the known place fields are recovered. Structure and counts of the sessions are
tested in ``test_session.py`` and ``test_examples.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.encoding import compute_spatial_rate
from neurospatial.simulation import (
    PlaceCellModel,
    generate_poisson_spikes,
    open_field_session,
    simulate_session,
    simulate_trajectory_ou,
    validate_simulation,
)


def test_validate_simulation_recovers_every_place_field(connected_arena_env):
    """Three minutes over 12 cm fields: every cell is evaluated and recovered.

    The default limits (1.5 field widths, correlation 0.9) pass about 95% of
    correctly simulated cells (see ``validate_simulation``), so this seeded
    session is one in which all five pass, not a guarantee for every seed.
    """
    session = simulate_session(
        connected_arena_env,
        duration=180.0,
        n_cells=5,
        cell_type="place",
        width=12.0,
        max_rate=20.0,
        seed=1,
        show_progress=False,
    )
    result = validate_simulation(session)
    assert result.n_evaluated == 5
    assert np.all(result.center_errors <= 1.5 * 12.0)
    assert np.all(result.correlations >= 0.9)
    assert result.passed


@pytest.mark.slow
def test_place_field_detection_accuracy():
    """Detected place-field peaks recover every true center of an open field.

    Five minutes at 50 Hz peak rate: each cell fires hundreds of spikes, and
    each rate-map peak lies within one field width (11.1 cm) of its center.
    """
    session = open_field_session(
        duration=300.0, n_place_cells=5, seed=42, max_rate=50.0
    )
    env = session.env
    errors = []
    for model, spike_times in zip(session.models, session.spike_times, strict=True):
        assert len(spike_times) > 100
        rate_map = compute_spatial_rate(
            env, spike_times, session.times, session.positions
        ).firing_rate
        peak = env.bin_centers[np.nanargmax(rate_map)]
        errors.append(np.linalg.norm(peak - model.ground_truth["center"]))
    assert np.max(errors) <= 11.1, errors
    assert np.mean(errors) <= 11.1 / 2, errors


def test_place_field_detection_with_longer_duration(connected_arena_env):
    """Three minutes locate a field better than 30 s, in the median over repetitions.

    Each repetition draws its trajectory and its spikes from separate streams.
    The peak is taken over the bins the rate map resolves (``nanargmax``):
    unvisited bins are NaN.
    """
    env = connected_arena_env
    model = PlaceCellModel(
        env, center=np.array([51.0, 51.0]), width=15.0, max_rate=30.0
    )

    def peak_errors(duration):
        errors = []
        for repetition in range(6):
            trajectory_stream, spike_stream = np.random.SeedSequence(
                [repetition, int(duration)]
            ).spawn(2)
            positions, times = simulate_trajectory_ou(
                env,
                duration=duration,
                speed_units="cm",
                seed=int(trajectory_stream.generate_state(1)[0]),
            )
            spikes = generate_poisson_spikes(
                model.firing_rate(times, positions),
                times,
                seed=int(spike_stream.generate_state(1)[0]),
            )
            rate_map = compute_spatial_rate(env, spikes, times, positions).firing_rate
            peak = env.bin_centers[np.nanargmax(rate_map)]
            errors.append(np.linalg.norm(peak - model.ground_truth["center"]))
        return np.array(errors)

    short, long = peak_errors(30.0), peak_errors(180.0)
    assert np.median(long) < np.median(short) / 2, (short, long)
    assert np.max(long) <= model.ground_truth["width"] / 2, long


def test_full_pipeline_reproducibility(connected_arena_env):
    """The same seed reproduces the trajectory, the spikes and the ground truth."""
    kwargs = {"duration": 30.0, "n_cells": 5, "cell_type": "mixed", "seed": 42}
    first = simulate_session(connected_arena_env, show_progress=False, **kwargs)
    second = simulate_session(connected_arena_env, show_progress=False, **kwargs)

    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.times, second.times)
    for a, b in zip(first.spike_times, second.spike_times, strict=True):
        np.testing.assert_array_equal(a, b)
    assert sum(len(train) for train in first.spike_times) > 0
    for label in first.unit_ids:
        truth, again = first.ground_truth[int(label)], second.ground_truth[int(label)]
        assert truth.keys() == again.keys()
        for key, value in truth.items():
            np.testing.assert_array_equal(value, again[key])
