"""Maboudi et al. (2018) line scores: off-track and empty time bins, random lines.

``detect_trajectory_radon(off_track="bin_median")`` scores a time bin where
the line leaves the track by its median single-bin mass, and
``zero_rows="line_median"`` scores a time bin without spikes by the median of
the line's band mass over its on-track time bins with spikes, as the paper's
text says. ``n_lines`` draws lines at random over every slope, as nelpy does.
``null_options={"column_cycle": {"exclude_zero_shift": True}}`` draws shifts
of 1 to n_bins - 1 bins. ``MABOUDI_2018`` puts them together.
"""

from __future__ import annotations

import dataclasses

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial import Environment
from neurospatial.decoding import bin_spikes_in_time, decode_position
from neurospatial.replay import (
    detect_replay,
    detect_trajectory_radon,
    replay_significance,
    shuffle_posterior_circular,
)
from neurospatial.replay.methods import MABOUDI_2018
from neurospatial.simulation import simulate_replay_event

DT = 0.02


@pytest.fixture(scope="module")
def short_track():
    """A 20 cm track in 2 cm bins."""
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [20.0, 0.0]], bin_size=2.0, units="cm"
    )


@pytest.fixture(scope="module")
def loop():
    """A 100 cm square track as a closed loop, 5 cm bins."""
    graph = nx.Graph()
    for node, position in enumerate([(0, 0), (25, 0), (25, 25), (0, 25)]):
        graph.add_node(node, pos=position)
    edge_order = [(0, 1), (1, 2), (2, 3), (3, 0)]
    graph.add_edges_from(edge_order)
    return Environment.from_graph(graph, edge_order, bin_size=5.0, units="cm")


def _sinogram_entry(result, speed, start_position):
    """Score of the searched line of ``speed`` through ``start_position``."""
    i = np.flatnonzero(np.isclose(result.sinogram_speeds, speed))
    j = np.flatnonzero(np.isclose(result.sinogram_start_positions, start_position))
    assert i.size == 1 and j.size == 1
    return result.sinogram[i[0], j[0]]


def _rows(n_bins, *bins):
    """One-hot posterior rows; None gives a row without mass."""
    rows = np.zeros((len(bins), n_bins))
    for row, b in enumerate(bins):
        if b is not None:
            rows[row, b] = 1.0
    return rows


class TestEmptyTimeBinsScoreTheLineMedian:
    def test_stationary_line_by_hand(self, short_track):
        # A line standing at 5 cm with a 2 cm band covers [3, 7] cm: all of
        # bin 2 ([4, 6] cm) and half of bins 1 and 3. Rows with spikes put
        # 1, 0.5, 0 and 1 of their mass in the band; the row without spikes
        # scores their median, 0.75.
        posterior = _rows(short_track.n_bins, 2, 1, None, 7, 2)
        times = np.arange(5) * DT
        result = detect_trajectory_radon(
            short_track, posterior, times, band=2.0, zero_rows="line_median"
        )
        expected = (1.0 + 0.5 + 0.0 + 1.0 + 0.75) / 5
        assert _sinogram_entry(result, 0.0, 5.0) == pytest.approx(expected, abs=1e-12)

    def test_only_on_track_time_bins_with_spikes_set_the_median(self, short_track):
        # 25 cm/s from 18.2 cm leaves the track (20 cm) at the last time bin,
        # whose bin median is 0. The empty row scores the median of the three
        # on-track rows with spikes (1, 1, 0), 1; were the off-track row or
        # the empty row itself counted, the median would be lower.
        posterior = _rows(short_track.n_bins, 9, 9, None, 0, 4)
        times = np.arange(5) * DT
        result = detect_trajectory_radon(
            short_track,
            posterior,
            times,
            band=2.0,
            max_speed=100.0,
            off_track="bin_median",
            zero_rows="line_median",
        )
        expected = (1.0 + 1.0 + 1.0 + 0.0 + 0.0) / 5
        assert _sinogram_entry(result, 25.0, 18.2) == pytest.approx(expected, abs=1e-12)

    def test_a_line_without_on_track_time_bins_with_spikes_adds_zero(self, short_track):
        # 25 cm/s from 18.2 cm: on the track only in the first four rows,
        # all without spikes; off the track, the last row's bin median is 0.
        posterior = _rows(short_track.n_bins, None, None, None, None, 9)
        times = np.arange(5) * DT
        result = detect_trajectory_radon(
            short_track,
            posterior,
            times,
            band=2.0,
            max_speed=100.0,
            off_track="bin_median",
            zero_rows="line_median",
        )
        assert _sinogram_entry(result, 25.0, 18.2) == 0.0

    def test_rows_without_mass_are_refused_by_default(self, short_track):
        posterior = _rows(short_track.n_bins, 2, 1, None, 7, 2)
        with pytest.raises(ValueError, match="do not sum to 1"):
            detect_trajectory_radon(short_track, posterior, np.arange(5) * DT)

    def test_without_empty_rows_line_median_changes_nothing(self, short_track):
        posterior = np.random.default_rng(0).random((6, short_track.n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)
        times = np.arange(6) * DT
        default = detect_trajectory_radon(short_track, posterior, times, band=4.0)
        median = detect_trajectory_radon(
            short_track, posterior, times, band=4.0, zero_rows="line_median"
        )
        assert_array_equal(median.sinogram, default.sinogram)
        assert (median.speed, median.start_position, median.score) == (
            default.speed,
            default.start_position,
            default.score,
        )


class TestOffTrackBinMedian:
    def test_line_leaving_the_track_by_hand(self, short_track):
        # 25 cm/s from 18.2 cm has all of rows 0-3 (bin 9, [18, 20] cm) in its
        # 2 cm band and leaves the track at the last row, which scores the
        # median of its ten bin masses, (0.05 + 0.1) / 2.
        last = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03])
        posterior = np.vstack([_rows(short_track.n_bins, 9, 9, 9, 9), last])
        times = np.arange(5) * DT
        options = {"band": 2.0, "max_speed": 100.0}
        bin_median = detect_trajectory_radon(
            short_track, posterior, times, off_track="bin_median", **options
        )
        band_median = detect_trajectory_radon(short_track, posterior, times, **options)
        expected = (4.0 + 0.075) / 5
        assert _sinogram_entry(bin_median, 25.0, 18.2) == pytest.approx(
            expected, abs=1e-12
        )
        # Davidson's band median is over a band's mass, larger than one bin's.
        assert _sinogram_entry(band_median, 25.0, 18.2) > expected + 0.01

    @pytest.mark.parametrize("setting", [{"line_ends": "on_track"}, {"wrap": True}])
    def test_refused_where_no_line_leaves_the_track(self, loop, setting):
        posterior = np.full((4, loop.n_bins), 1.0 / loop.n_bins)
        with pytest.raises(ValueError, match="off_track would change nothing"):
            detect_trajectory_radon(
                loop,
                posterior,
                np.arange(4) * DT,
                band=10.0,
                off_track="bin_median",
                **setting,
            )


class TestRandomLines:
    def test_lines_of_every_slope_drawn_reproducibly(self, short_track):
        posterior = np.random.default_rng(1).random((8, short_track.n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)
        times = np.arange(8) * DT
        first = detect_trajectory_radon(
            short_track, posterior, times, band=2.0, n_lines=500
        )
        again = detect_trajectory_radon(
            short_track, posterior, times, band=2.0, n_lines=500
        )
        other = detect_trajectory_radon(
            short_track, posterior, times, band=2.0, n_lines=500, line_seed=1
        )
        assert first.sinogram.shape == first.sinogram_speeds.shape == (500,)
        assert_array_equal(first.sinogram, again.sinogram)
        assert not np.array_equal(first.sinogram_speeds, other.sinogram_speeds)
        # Every slope: both directions, beyond the grid's default 30 m/s.
        assert (
            first.sinogram_speeds.min() < -3000.0 < 3000.0 < first.sinogram_speeds.max()
        )
        assert first.score == np.nanmax(first.sinogram)
        k = np.flatnonzero(first.sinogram == first.score)
        assert first.speed in first.sinogram_speeds[k]

    def test_lines_are_drawn_about_nelpys_center(self, short_track):
        """nelpy 444d5243 draws rho about ((NT + 1) / 2, (NP + 1) / 2).

        Its ``calc_ri`` puts time bin ``ci`` (0-based) at position bin
        ``(rho - (ci - ci_mid) cos(phi)) / sin(phi) + ri_mid``, with rho
        uniform within half the image's diagonal: one bin past the image's
        center on both axes.
        """
        n_time, n_bins = 8, short_track.n_bins
        posterior = np.full((n_time, n_bins), 1.0 / n_bins)
        result = detect_trajectory_radon(
            short_track, posterior, np.arange(n_time) * DT, band=2.0, n_lines=5000
        )
        # Back to image units: time bins and position bins (2 cm, from 1 cm).
        slope = result.sinogram_speeds * DT / 2.0
        first_row = (result.sinogram_start_positions - 1.0) / 2.0
        phi = np.arctan(-1.0 / slope)  # the normal's angle, on (-pi/2, pi/2)
        rho = -(n_time + 1) / 2 * np.cos(phi) + (first_row - (n_bins + 1) / 2) * (
            np.sin(phi)
        )
        half_diagonal = np.hypot(n_time - 1, n_bins - 1) / 2
        assert np.abs(rho).max() <= half_diagonal + 1e-9
        # Uniform over the whole range: the extremes reach both ends.
        assert rho.min() < -half_diagonal + 0.05
        assert rho.max() > half_diagonal - 0.05

    def test_finds_a_sweep_near_the_dense_grid(self, short_track):
        posterior = np.eye(short_track.n_bins)
        times = np.arange(short_track.n_bins) * DT
        grid = detect_trajectory_radon(short_track, posterior, times, band=2.0)
        lines = detect_trajectory_radon(
            short_track, posterior, times, band=2.0, n_lines=35000
        )
        assert grid.score == 1.0
        assert lines.score > 0.9
        assert lines.speed == pytest.approx(100.0, rel=0.1)

    def test_no_line_within_the_speed_bounds_is_refused(self, short_track):
        # Lines of every slope, but none as slow as 1 cm/s over 10 time bins.
        posterior = np.full((5, short_track.n_bins), 1.0 / short_track.n_bins)
        with pytest.raises(ValueError, match=r"none of the 5 random lines") as error:
            detect_trajectory_radon(
                short_track,
                posterior,
                np.arange(5) * DT,
                band=2.0,
                n_lines=5,
                max_speed=1.0,
            )
        assert "0 have speeds between min_speed 0 and max_speed 1" in str(error.value)
        assert "band / 5" not in str(error.value)  # the dense grid's message

    def test_a_line_off_the_track_is_scored_or_refused(self, short_track):
        # One line, drawn within half the image's diagonal of its center,
        # can miss the track at every time bin: about 1 seed in 7 here.
        posterior = np.full((5, short_track.n_bins), 1.0 / short_track.n_bins)
        times = np.arange(5) * DT
        refused = 0
        for seed in range(50):
            try:
                result = detect_trajectory_radon(
                    short_track, posterior, times, band=2.0, n_lines=1, line_seed=seed
                )
            except ValueError as error:
                assert "none of the 1 random lines" in str(error)
                assert "0 lie on the track" in str(error)
                refused += 1
            else:
                assert np.isfinite(result.score)
        assert refused > 0

    @pytest.mark.parametrize(
        ("name", "value"), [("n_lines", 0), ("n_lines", 2.5), ("line_seed", -1)]
    )
    def test_invalid_counts_are_refused(self, short_track, name, value):
        posterior = np.eye(short_track.n_bins)
        with pytest.raises(ValueError, match=name):
            detect_trajectory_radon(
                short_track,
                posterior,
                np.arange(short_track.n_bins) * DT,
                **{name: value},
            )


def _with_empty_rows(env, n_time, empty, seed):
    rng = np.random.default_rng(seed)
    posterior = rng.random((n_time, env.n_bins)) ** 4
    posterior /= posterior.sum(axis=1, keepdims=True)
    counts = np.ones((n_time, 3), dtype=int)
    counts[list(empty)] = 0
    return posterior, counts


@pytest.mark.parametrize(
    ("environment", "options"),
    [
        ("short_track", {"band": 3.0, "max_speed": 600.0}),
        ("short_track", {"band": 3.0, "max_speed": 600.0, "off_track": "bin_median"}),
        ("short_track", {"band": 3.0, "line_ends": "on_track"}),
        ("short_track", {"band": 3.0, "n_lines": 3000, "off_track": "bin_median"}),
        ("loop", {"band": 10.0, "wrap": True, "max_speed": 1500.0}),
    ],
    ids=["any", "bin_median", "on_track", "random_lines", "wrap"],
)
def test_shuffle_scores_match_scoring_each_shuffle_alone(request, environment, options):
    """The batched shuffle scores, with bounds that skip lines, are exact."""
    env = request.getfixturevalue(environment)
    posterior, counts = _with_empty_rows(env, 8, empty=(2, 5), seed=4)
    times = np.arange(8) * DT
    options = {**options, "zero_rows": "line_median"}
    result = replay_significance(
        env,
        posterior,
        times,
        score_options=options,
        nulls="column_cycle",
        spike_counts=counts,
        empty_time_bins="zero",
        n_shuffles=40,
        rng=7,
    )
    zeroed = np.where(counts.sum(axis=1)[:, None] > 0, posterior, 0.0)
    shuffles = shuffle_posterior_circular(env, zeroed, n_shuffles=40, rng=7)
    alone = [detect_trajectory_radon(env, s, times, **options).score for s in shuffles]
    assert_array_equal(result.shuffles["column_cycle"].null_scores, alone)
    assert (
        result.observed.score
        == detect_trajectory_radon(env, zeroed, times, **options).score
    )


class TestColumnCycleWithoutZeroShift:
    def test_no_row_stays_in_place(self, short_track):
        posterior = np.eye(short_track.n_bins)[:6]
        shuffles = np.array(
            list(
                shuffle_posterior_circular(
                    short_track,
                    posterior,
                    n_shuffles=300,
                    rng=0,
                    exclude_zero_shift=True,
                )
            )
        )
        in_place = (shuffles == posterior).all(axis=2)
        assert not in_place.any()
        default = np.array(
            list(
                shuffle_posterior_circular(
                    short_track, posterior, n_shuffles=300, rng=0
                )
            )
        )
        assert (default == posterior).all(axis=2).any()

    def test_default_draws_every_shift_from_the_generator(self, short_track):
        posterior = np.eye(short_track.n_bins)[:6]
        shuffled = next(iter(shuffle_posterior_circular(short_track, posterior, rng=3)))
        shifts = np.random.default_rng(3).integers(0, short_track.n_bins, size=6)
        assert_array_equal(np.argmax(shuffled, axis=1), (np.arange(6) + shifts) % 10)

    def test_replay_significance_passes_the_option(self, short_track):
        posterior = np.eye(short_track.n_bins)[:6]
        times = np.arange(6) * DT
        common = {
            "score_options": {"band": 2.0, "max_speed": 600.0},
            "nulls": "column_cycle",
            "n_shuffles": 30,
            "rng": 2,
        }
        excluded = replay_significance(
            short_track,
            posterior,
            times,
            null_options={"column_cycle": {"exclude_zero_shift": True}},
            **common,
        )
        shuffles = shuffle_posterior_circular(
            short_track, posterior, n_shuffles=30, rng=2, exclude_zero_shift=True
        )
        expected = [
            detect_trajectory_radon(
                short_track, s, times, band=2.0, max_speed=600.0
            ).score
            for s in shuffles
        ]
        assert_array_equal(excluded.shuffles["column_cycle"].null_scores, expected)

    def test_non_boolean_option_is_refused(self, short_track):
        with pytest.raises(ValueError, match="exclude_zero_shift of column_cycle"):
            replay_significance(
                short_track,
                np.eye(short_track.n_bins)[:4],
                np.arange(4) * DT,
                nulls="column_cycle",
                null_options={"column_cycle": {"exclude_zero_shift": 1}},
                n_shuffles=5,
            )


class TestRecipe:
    @pytest.fixture(scope="class")
    def place_code(self):
        """A 200 cm track in 4 cm bins with 40 place cells tiling it."""
        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=4.0, units="cm"
        )
        x = env.bin_centers[:, 0]
        rates = 0.5 + 25.0 * np.exp(
            -0.5 * ((x - np.linspace(0.0, 200.0, 40)[:, None]) / 10.0) ** 2
        )
        return env, rates

    def _detect(self, env, rates, kind, seed, empty=(), **kwargs):
        event = simulate_replay_event(
            env, rates, kind=kind, rate_gain=4.0, seed=seed, **kwargs
        )
        counts, times = bin_spikes_in_time(
            list(event.spike_times), dt=DT, t_start=0.0, t_stop=event.duration
        )
        counts[list(empty)] = 0
        decoded = decode_position(env, counts, rates, dt=DT, times=times)
        # 199 shuffles: the smallest p-value, 1 / 200, is below alpha = 0.01.
        method = dataclasses.replace(MABOUDI_2018, n_shuffles=199)
        result = detect_replay(
            env,
            decoded.posterior,
            decoded.times,
            replay_method=method,
            spike_counts=counts,
            encoding_models=rates,
            dt=DT,
            rng=0,
        )
        return result, decoded, counts

    def test_detects_a_sweep_with_empty_time_bins(self, place_code):
        env, rates = place_code
        result, decoded, counts = self._detect(
            env, rates, "sweep", seed=1, empty=(3, 6), speed=900.0
        )
        assert result.is_replay, result.summary()
        assert result.p_value == pytest.approx(1 / 200)
        # The score is the line score of the posterior with empty rows zeroed.
        zeroed = np.where(counts.sum(axis=1)[:, None] > 0, decoded.posterior, 0.0)
        line = detect_trajectory_radon(
            env, zeroed, decoded.times, **MABOUDI_2018.score_options
        )
        assert result.score == line.score

    def test_rejects_random_events(self, place_code):
        env, rates = place_code
        verdicts = [
            self._detect(env, rates, "random", seed=300 + seed)[0].is_replay
            for seed in range(4)
        ]
        assert not any(verdicts)
