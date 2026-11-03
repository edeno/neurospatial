"""The decoding operation's chance level, in the metric of its error."""

from __future__ import annotations

import numpy as np
import pytest
from operations import chance_errors


def _env(samples: np.ndarray):
    from neurospatial import Environment

    return Environment.from_samples(samples, bin_size=4.0, units="cm")


@pytest.fixture(scope="module")
def u_track() -> np.ndarray:
    """Samples on a U: two 100 cm arms 24 cm apart, joined at the bottom."""
    rng = np.random.default_rng(3)
    up = rng.uniform(0, 100, 3000)
    left = np.column_stack([rng.uniform(0, 6, 3000), up])
    right = np.column_stack([rng.uniform(30, 36, 3000), up])
    base = np.column_stack([rng.uniform(0, 36, 1000), rng.uniform(0, 6, 1000)])
    return np.vstack([left, right, base])


def test_geodesic_chance_exceeds_straight_line_on_a_u_track(u_track):
    env = _env(u_track)
    errors, disconnected, outside = chance_errors(env, u_track, seed=0)
    assert disconnected == 0.0
    assert outside == 0.0
    assert np.all(np.isfinite(errors))
    order = np.random.default_rng(1).permutation(len(u_track))
    straight = np.linalg.norm(u_track - u_track[order], axis=1)
    # Arm to arm is 24 cm straight across but about 200 cm along the track.
    assert np.median(errors) > 1.3 * np.median(straight)  # 63 vs 39 cm here


def test_chance_shuffles_within_each_disconnected_part():
    rng = np.random.default_rng(4)
    first = rng.uniform(0, 40, (2000, 2))
    second = rng.uniform(0, 40, (2000, 2)) + np.array([200.0, 0.0])
    samples = np.vstack([first, second])
    env = _env(samples)
    errors, disconnected, _ = chance_errors(env, samples, seed=0)
    # A global shuffle puts about half the pairs in the other part (inf apart);
    # within-part pairs stay finite and no longer than a box's diagonal path.
    assert 0.4 < disconnected < 0.6
    assert np.all(np.isfinite(errors))
    assert np.max(errors) < 2 * 40 * 1.1


def test_positions_outside_a_declared_track_are_not_a_part():
    # A straight 100 cm track 6 cm wide; a third of the true positions lie
    # 20 cm off it. They are reported as outside, not as a disconnected part.
    from _geometry import graph_environment
    from _tracks import polyline_graph

    env = graph_environment(
        polyline_graph(np.array([[0.0, 0.0], [100.0, 0.0]]), closed=False),
        bin_size=4.0,
        track_width=6.0,
        units="cm",
        name="track",
    )
    rng = np.random.default_rng(5)
    on = np.column_stack([rng.uniform(0, 100, 2000), np.zeros(2000)])
    off = on[:1000] + np.array([0.0, 20.0])
    errors, disconnected, outside = chance_errors(env, np.vstack([on, off]), seed=0)
    assert outside == pytest.approx(1 / 3)
    assert disconnected == 0.0
    assert len(errors) == 2000
    assert np.all(np.isfinite(errors))


def test_decoding_error_over_the_declared_samples_in_both_runs(simulated_recording):
    # Declare the left half of the 100 cm box: both runs report the error over
    # the same time bins, those whose true position the declared half places.
    from dataclasses import replace

    import shapely
    from operations import decode_position

    from neurospatial import Environment

    half = Environment.from_polygon(
        shapely.box(0, 0, 50, 100), bin_size=2.5, units="cm"
    )
    rec = replace(simulated_recording, geometry={"environment": lambda: half})
    with pytest.warns(UserWarning, match="outside the environment"):
        declared = decode_position(rec, {}).summary
    inferred = decode_position(rec, {"inferred_env": True}).summary
    assert declared["n_on_declared"] == inferred["n_on_declared"]
    assert 0.3 < declared["n_on_declared"] / declared["n_time_bins"] < 0.7
    assert declared["frac_actual_outside"] == pytest.approx(
        1 - declared["n_on_declared"] / declared["n_time_bins"], abs=0.02
    )
    assert inferred["frac_actual_outside"] == 0.0
    assert np.isfinite(inferred["median_error_on_declared"])
