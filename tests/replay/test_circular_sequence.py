"""Circular-track replay score of Wikenheiser & Redish (2013)."""

from __future__ import annotations

import copy
import pickle

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import (
    CircularSequenceResult,
    fit_circular_sequence,
    fit_linear_trajectory,
    replay_significance,
)

N_NODES = 16
RADIUS = 40.0  # an 80 cm diameter track, as in Wikenheiser & Redish (2013)


def _loop(n_nodes=N_NODES, radii=None, edge_order=None, edge_spacing=0.0):
    """A closed polygonal track; ``radii`` makes its sides unequal."""
    radii = np.full(n_nodes, RADIUS) if radii is None else np.asarray(radii)
    graph = nx.cycle_graph(n_nodes)
    for node in graph.nodes:
        angle = 2 * np.pi * node / n_nodes
        graph.nodes[node]["pos"] = (
            radii[node] * np.cos(angle),
            radii[node] * np.sin(angle),
        )
    if edge_order is None:
        edge_order = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    return Environment.from_graph(
        graph,
        edge_order=edge_order,
        bin_size=4.0,
        edge_spacing=edge_spacing,
        units="cm",
    )


@pytest.fixture(scope="module")
def loop():
    env = _loop()
    assert env.n_bins == 64
    return env


def _bin_centers_along(env):
    """Each bin's center along the loop's linear position, in bin order."""
    edges = env.binning.linearization.grid_edges[0]
    return (edges[:-1] + edges[1:]) / 2


def _bump(env, centers, concentration=8.0):
    """Posterior rows of a von Mises bump at ``centers`` (cm along the loop)."""
    length = float(np.ptp(env.binning.linearization.grid_edges[0]))
    angles = 2 * np.pi * _bin_centers_along(env) / length
    targets = 2 * np.pi * np.asarray(centers)[:, None] / length
    rows = np.exp(concentration * np.cos(angles[None, :] - targets))
    return rows / rows.sum(axis=1, keepdims=True)


def _seam_sweep(env, n_time_bins=12, speed=1500.0, start=180.0, dt=0.01):
    """A bump moving at ``speed`` cm/s that crosses the seam at 0 / loop length."""
    length = float(np.ptp(env.binning.linearization.grid_edges[0]))
    times = np.arange(n_time_bins) * dt
    centers = np.mod(start + speed * times, length)
    assert np.any(np.diff(centers) < 0), "the sweep must cross the seam"
    return _bump(env, centers), times


class TestFit:
    def test_sweep_across_the_seam_is_a_perfect_sequence(self, loop):
        """1500 cm/s through the seam: R² = 1 and the speed is recovered."""
        posterior, times = _seam_sweep(loop)
        result = fit_circular_sequence(loop, posterior, times)
        assert result.r_squared == pytest.approx(1.0, abs=1e-9)
        assert result.slope == pytest.approx(1500.0, rel=1e-6)
        length = float(np.ptp(loop.binning.linearization.grid_edges[0]))
        assert result.angular_slope == pytest.approx(
            1500.0 * 2 * np.pi / length, rel=1e-6
        )
        assert np.all(np.diff(result.cumulative_angles) > 0)
        assert result.cumulative_angles[0] == 0.0
        assert np.all(
            (result.decoded_angles >= 0) & (result.decoded_angles < 2 * np.pi)
        )

    def test_unwrapped_fits_break_at_the_seam(self, loop):
        """Without the wrap, the same sweep fits poorly: the wrap is the point.

        A line along the linear position, and a line through the decoded
        angles themselves, both see the jump of one loop at the seam.
        """
        posterior, times = _seam_sweep(loop)
        result = fit_circular_sequence(loop, posterior, times)
        linear = fit_linear_trajectory(loop, posterior, times, method="map")
        raw = np.corrcoef(times, result.decoded_angles)[0, 1] ** 2
        assert linear.r_squared < 0.6
        assert raw < 0.6

    def test_backward_sweep_has_a_negative_slope(self, loop):
        """The same sweep reversed in time runs the other way round the loop."""
        posterior, times = _seam_sweep(loop)
        forward = fit_circular_sequence(loop, posterior, times)
        backward = fit_circular_sequence(loop, posterior[::-1], times)
        assert forward.slope > 0
        assert backward.slope == pytest.approx(-forward.slope)
        assert backward.r_squared == pytest.approx(forward.r_squared)

    def test_slope_is_speed_along_an_uneven_loop(self):
        """On a loop with sides of unequal length, slope is cm/s along the loop.

        A constant 900 cm/s sweep along the linear position (bins of unequal
        length) is a straight line of cumulative angle on time.
        """
        radii = RADIUS * (1 + 0.3 * np.cos(np.arange(N_NODES)))
        env = _loop(radii=radii)
        posterior, times = _seam_sweep(env, speed=900.0, n_time_bins=20)
        result = fit_circular_sequence(env, posterior, times)
        assert result.slope == pytest.approx(900.0, rel=0.02)
        assert result.r_squared > 0.99

    def test_stationary_event_has_no_score(self, loop):
        """A decode that never moves has slope 0 and R² NaN, not a perfect fit."""
        posterior = _bump(loop, np.full(10, 37.0))
        result = fit_circular_sequence(loop, posterior, np.arange(10) * 0.01)
        assert result.slope == 0.0
        assert np.isnan(result.r_squared)
        assert np.isnan(result.score)
        with pytest.raises(ValueError, match="score is nan"):
            replay_significance(
                loop,
                posterior,
                np.arange(10) * 0.01,
                score=fit_circular_sequence,
                nulls="time_bin_permutation",
                n_shuffles=9,
                rng=0,
            )

    def test_jittering_event_scores_low(self, loop):
        """A decode that only wobbles about one place has a low R²."""
        rng = np.random.default_rng(3)
        centers = 37.0 + rng.normal(0.0, 3.0, size=30)
        result = fit_circular_sequence(loop, _bump(loop, centers), np.arange(30) * 0.01)
        assert result.r_squared < 0.5

    def test_gaps_keep_their_times(self, loop):
        """Dropped time bins leave a gap in time; the slope stays in cm/s."""
        posterior, times = _seam_sweep(loop, n_time_bins=14)
        keep = np.ones(14, dtype=bool)
        keep[[3, 4, 9]] = False
        result = fit_circular_sequence(loop, posterior[keep], times[keep])
        assert result.slope == pytest.approx(1500.0, rel=1e-6)
        assert result.r_squared == pytest.approx(1.0, abs=1e-9)

    def test_result_is_read_only_and_pickles(self, loop):
        posterior, times = _seam_sweep(loop)
        result = fit_circular_sequence(loop, posterior, times)
        assert isinstance(result, CircularSequenceResult)
        assert result.summary() == {
            "slope": result.slope,
            "angular_slope": result.angular_slope,
            "r_squared": result.r_squared,
        }
        assert result.score == result.r_squared
        assert result != fit_circular_sequence(loop, posterior, times)
        for restored in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
            assert not restored.decoded_angles.flags.writeable
            np.testing.assert_array_equal(
                restored.cumulative_angles, result.cumulative_angles
            )
        with pytest.raises(ValueError, match="read-only"):
            result.cumulative_angles[0] = 1.0


class TestRefusals:
    def test_open_track(self):
        track = Environment.linear_track(endpoints=[(0, 0), (250, 0)], bin_size=4.0)
        with pytest.raises(ValueError, match="needs a circular track") as error:
            fit_circular_sequence(
                track, np.full((5, track.n_bins), 1 / track.n_bins), np.arange(5.0)
            )
        assert "from_graph" in str(error.value)
        assert "fit_linear_trajectory" in str(error.value)

    def test_open_arc_of_a_circle(self):
        """Nodes around a circle without the closing edge have two ends."""
        env = _loop(edge_order=[(i, i + 1) for i in range(N_NODES - 1)])
        with pytest.raises(ValueError, match="does not close on itself"):
            fit_circular_sequence(
                env, np.full((5, env.n_bins), 1 / env.n_bins), np.arange(5.0)
            )

    def test_edges_out_of_order(self):
        """Edges listed out of loop order make the linear position jump."""
        order = [(i, (i + 1) % N_NODES) for i in range(N_NODES)]
        order[0], order[1] = order[1], order[0]
        env = _loop(edge_order=order)
        with pytest.raises(ValueError, match="does not close on itself"):
            fit_circular_sequence(
                env, np.full((5, env.n_bins), 1 / env.n_bins), np.arange(5.0)
            )

    def test_edge_spacing(self):
        env = _loop(edge_spacing=5.0)
        with pytest.raises(ValueError, match="edge_spacing"):
            fit_circular_sequence(
                env, np.full((5, env.n_bins), 1 / env.n_bins), np.arange(5.0)
            )

    def test_open_field(self):
        x, y = np.meshgrid(np.arange(0.0, 50.0), np.arange(0.0, 50.0))
        arena = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=5.0)
        with pytest.raises(ValueError, match="not a track"):
            fit_circular_sequence(
                arena, np.full((5, arena.n_bins), 1 / arena.n_bins), np.arange(5.0)
            )

    def test_time_bin_without_a_circular_mean(self, loop):
        """A flat row (an empty time bin's prior) has no decoded position."""
        posterior, times = _seam_sweep(loop)
        posterior[4] = 1 / loop.n_bins
        with pytest.raises(ValueError, match="no circular mean"):
            fit_circular_sequence(loop, posterior, times)


def _call_fit(env, posterior, times):
    """The score as a plain callable, scored one shuffle at a time."""
    return fit_circular_sequence(env, posterior, times)


class TestSignificance:
    def test_batched_scores_match_one_call_per_shuffle(self, loop):
        posterior, times = _seam_sweep(loop)
        posterior = 0.7 * posterior + 0.3 * _bump(loop, np.full(len(times), 90.0))
        tests = [
            replay_significance(
                loop,
                posterior,
                times,
                score=score,
                nulls=("time_bin_permutation", "column_cycle"),
                n_shuffles=300,
                rng=7,
            )
            for score in (fit_circular_sequence, _call_fit)
        ]
        for null in ("time_bin_permutation", "column_cycle"):
            np.testing.assert_allclose(
                tests[0].shuffles[null].null_scores,
                tests[1].shuffles[null].null_scores,
                rtol=0,
                atol=1e-12,
            )
        assert tests[0].p_value == tests[1].p_value

    def test_seeded_and_rejects_a_sweep(self, loop):
        posterior, times = _seam_sweep(loop, n_time_bins=15)
        first, second = (
            replay_significance(
                loop,
                posterior,
                times,
                score=fit_circular_sequence,
                nulls="time_bin_permutation",
                n_shuffles=999,
                rng=11,
            )
            for _ in range(2)
        )
        np.testing.assert_array_equal(
            first.shuffles["time_bin_permutation"].null_scores,
            second.shuffles["time_bin_permutation"].null_scores,
        )
        assert first.p_value == pytest.approx(1 / 1000)
        assert first.observed.slope > 0

    def test_false_positive_rate_on_random_events(self, loop):
        """Events whose time bins decode at random places reject at about alpha.

        400 events of 10 random positions, 199 permutations each: the
        number with p < 0.05 is Binomial(400, 0.05) under a valid test, mean
        20; the bounds hold it within about 3.5 standard deviations.
        """
        rng = np.random.default_rng(2013)
        length = float(np.ptp(loop.binning.linearization.grid_edges[0]))
        times = np.arange(10) * 0.01
        p_values = np.array(
            [
                replay_significance(
                    loop,
                    _bump(loop, rng.uniform(0, length, size=10)),
                    times,
                    score=fit_circular_sequence,
                    nulls="time_bin_permutation",
                    n_shuffles=199,
                    rng=rng,
                ).p_value
                for _ in range(400)
            ]
        )
        assert 5 <= np.sum(p_values < 0.05) <= 35
        # Not only the tail: the p-values are spread over (0, 1].
        assert 0.4 < np.median(p_values) < 0.6
