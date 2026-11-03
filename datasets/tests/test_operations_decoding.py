"""The decoding operation's chance level, in the metric of its error."""

from __future__ import annotations

import warnings

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


def _decode_errors(rec, env):
    """An independent decode with ``env``: per-time-bin errors and true x."""
    from neurospatial.decoding import BayesianDecoder

    p, spikes = rec.params, list(rec.spike_times)
    decoder = BayesianDecoder(env, dt=p["dt"]).fit(
        spikes, rec.times, rec.positions, epochs=p["train_epochs"]
    )
    result = decoder.predict(spikes, rec.times, epochs=p["test_epochs"])
    with warnings.catch_warnings():
        # The samples-side environment leaves out x > 70 cm, and says so.
        warnings.filterwarnings("ignore", "decoding_error", UserWarning)
        error = np.asarray(result.error_against(rec.times, rec.positions), float)
    true_x = np.interp(result.times, rec.times, np.asarray(rec.positions)[:, 0])
    return error, true_x


def test_decoding_error_over_the_same_time_bins_in_both_runs(simulated_recording):
    # The declared box is the whole 100 cm arena; the samples-built side
    # (seeded in the cache, as a grid built from samples that never reach
    # x > 70 cm would be) covers x <= 70 cm only. Both runs must report the
    # error over the same time bins, those with an error in both runs, and
    # each run's median over them. The count and medians are recomputed here
    # from independent decodes, not from the operation's mask.
    from dataclasses import replace

    import shapely
    from operations import decode_position

    from neurospatial import Environment

    declared_env = Environment.from_polygon(
        shapely.box(0, 0, 100, 100), bin_size=2.5, units="cm"
    )
    samples_env = Environment.from_polygon(
        shapely.box(0, 0, 70, 100), bin_size=2.5, units="cm"
    )
    rec = replace(simulated_recording, geometry={"environment": lambda: declared_env})
    declared = decode_position(rec, {"samples_env": samples_env}).summary
    with pytest.warns(UserWarning, match="outside the environment"):
        inferred = decode_position(
            rec, {"samples_env": samples_env, "inferred_env": True}
        ).summary

    declared_error, true_x = _decode_errors(rec, declared_env)
    inferred_error, _ = _decode_errors(rec, samples_env)
    both = np.isfinite(declared_error) & np.isfinite(inferred_error)
    # The geometry: an error wherever the true x is inside the 70 cm side,
    # none a bin spacing or more past it (decoding_error measures a position
    # within about one bin spacing of an edge from its nearest bin).
    assert np.isfinite(declared_error).all()
    assert both[true_x <= 70.0].all()
    assert not both[true_x >= 72.5].any()
    expected = int(both.sum())
    assert 0.5 < expected / len(true_x) < 0.9
    assert declared["n_on_declared"] == expected
    assert inferred["n_on_declared"] == expected
    assert declared["median_error_on_declared"] == pytest.approx(
        np.median(declared_error[both]), rel=1e-12
    )
    assert inferred["median_error_on_declared"] == pytest.approx(
        np.median(inferred_error[both]), rel=1e-12
    )
    assert declared["n_time_bins"] == inferred["n_time_bins"] == len(true_x)
