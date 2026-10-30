"""Line families of the replay line search: closed loops and on-track ends.

``wrap=True`` reads positions modulo a closed-loop track's length (Muessig et
al. 2019); ``line_ends="on_track"`` searches only lines that start and end on
the track, padding band beyond the track with the time bin's median posterior
density (Liu et al. 2023; neurocode ``FindReplayScore``).
"""

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial import Environment
from neurospatial.replay import detect_trajectory_radon, replay_significance

LOOP_LENGTH = 400.0
BIN_SIZE = 2.5
# Start positions are band / 5 apart, here one bin, so shifting a posterior
# by whole bins shifts the searched lines with it.
LOOP_BAND = 12.5


@pytest.fixture(scope="module")
def loop():
    """A 100 cm square track as a closed loop, 2.5 cm bins (Muessig et al.)."""
    graph = nx.Graph()
    for node, position in enumerate([(0, 0), (100, 0), (100, 100), (0, 100)]):
        graph.add_node(node, pos=position)
    edge_order = [(0, 1), (1, 2), (2, 3), (3, 0)]
    graph.add_edges_from(edge_order)
    return Environment.from_graph(graph, edge_order, bin_size=BIN_SIZE, units="cm")


@pytest.fixture(scope="module")
def track():
    """A 100 cm linear track in 2 cm bins."""
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
    )


def _loop_sweep(first_bin, *, n_time=10, bins_per_step=4, sigma_bins=1.5):
    """A Gaussian bump moving round the loop by whole bins per time bin."""
    n_bins = int(LOOP_LENGTH / BIN_SIZE)
    centers = first_bin + bins_per_step * np.arange(n_time)
    offset = (np.arange(n_bins)[None, :] - centers[:, None] + n_bins / 2) % n_bins
    rows = np.exp(-0.5 * ((offset - n_bins / 2) / sigma_bins) ** 2)
    return rows / rows.sum(axis=1, keepdims=True), np.arange(n_time) * 0.02


class TestClosedLoop:
    def test_sweep_across_the_seam_scores_as_one_away_from_it(self, loop):
        """Lines wrap across the seam where the loop's end meets its start."""
        # 500 cm/s for 180 ms: the first sweep crosses the seam at 400 cm.
        across, times = _loop_sweep(first_bin=148)
        away, _ = _loop_sweep(first_bin=40)
        options = {"band": LOOP_BAND, "max_speed": 2500.0}

        wrapped_across = detect_trajectory_radon(
            loop, across, times, wrap=True, **options
        )
        wrapped_away = detect_trajectory_radon(loop, away, times, wrap=True, **options)
        open_across = detect_trajectory_radon(loop, across, times, **options)

        assert wrapped_across.score == pytest.approx(wrapped_away.score, abs=1e-9)
        assert wrapped_across.score > 0.95
        assert wrapped_across.speed == pytest.approx(wrapped_away.speed)
        assert wrapped_across.speed == pytest.approx(500.0, abs=10.0)
        # It starts 270 cm before the seam's far side, on the loop.
        assert wrapped_across.start_position == pytest.approx(
            (148 + 0.5) * BIN_SIZE, abs=BIN_SIZE
        )
        # Read as an open track, the sweep breaks at the seam.
        assert open_across.score < wrapped_across.score - 0.2

    def test_every_line_is_on_the_loop(self, loop):
        posterior, times = _loop_sweep(first_bin=0)
        result = detect_trajectory_radon(
            loop, posterior, times, band=LOOP_BAND, max_speed=2500.0, wrap=True
        )
        assert np.all(np.isfinite(result.sinogram))
        starts = result.sinogram_start_positions
        assert starts.min() > 0.0 and starts.max() < LOOP_LENGTH
        assert_allclose(np.diff(starts), BIN_SIZE)

    def test_uniform_posterior_scores_the_band_share(self, loop):
        """Every line holds 2 * band of the loop's mass, also across the seam."""
        posterior = np.full((4, loop.n_bins), 1.0 / loop.n_bins)
        result = detect_trajectory_radon(
            loop, posterior, np.arange(4) * 0.02, band=LOOP_BAND, wrap=True
        )
        assert_allclose(result.sinogram, 2 * LOOP_BAND / LOOP_LENGTH, atol=1e-12)

    def test_rows_without_mass_add_no_band_mass_at_the_seam(self, loop):
        """An empty time bin scores its line's median, also across the seam.

        With every other row uniform, each line's median band mass is the
        band's share of the loop, so every line, including those whose band
        crosses the seam, scores that share.
        """
        posterior = np.full((6, loop.n_bins), 1.0 / loop.n_bins)
        posterior[[1, 4]] = 0.0
        result = detect_trajectory_radon(
            loop,
            posterior,
            np.arange(6) * 0.02,
            band=LOOP_BAND,
            wrap=True,
            zero_rows="line_median",
        )
        assert_allclose(result.sinogram, 2 * LOOP_BAND / LOOP_LENGTH, atol=1e-12)

    def test_empty_row_in_a_sweep_across_the_seam_scores_its_line_median(self, loop):
        posterior, times = _loop_sweep(first_bin=148)
        # Time bin 3 is centered on the seam (bin 160, 400 cm), so its band
        # crosses it.
        posterior[3] = 0.0
        result = detect_trajectory_radon(
            loop,
            posterior,
            times,
            band=LOOP_BAND,
            max_speed=2500.0,
            wrap=True,
            zero_rows="line_median",
        )
        # The best line's band mass, integrated bin by bin around the loop.
        edges = np.arange(loop.n_bins + 1) * BIN_SIZE
        position = result.start_position + result.speed * (times - times[0])
        masses = []
        for row, center in zip(posterior, position, strict=True):
            mass = 0.0
            for lap in (-LOOP_LENGTH, 0.0, LOOP_LENGTH):
                low, high = center - LOOP_BAND + lap, center + LOOP_BAND + lap
                overlap = np.minimum(edges[1:], high) - np.maximum(edges[:-1], low)
                mass += float(np.sum(row * np.clip(overlap, 0.0, None)) / BIN_SIZE)
            masses.append(mass)
        masses = np.array(masses)
        with_mass = np.arange(len(times)) != 3
        expected = (masses[with_mass].sum() + np.median(masses[with_mass])) / len(times)
        assert result.score == pytest.approx(expected, abs=1e-9)
        assert result.score <= 1.0

    def test_batched_scores_with_empty_rows_equal_single_scores(self, loop):
        from neurospatial.replay.significance import _scorer_for

        posteriors, times = _noisy_copies(loop, _event_across_the_seam, 6, seed=4)
        for index, posterior in enumerate(posteriors):
            posterior[[2, 5 + index % 3]] = 0.0
        options = {"band": LOOP_BAND, "wrap": True, "zero_rows": "line_median"}
        scorer = _scorer_for(
            detect_trajectory_radon, loop, posteriors[0], times, dict(options)
        )
        batched = scorer.scores(iter(posteriors), len(posteriors))
        single = [
            detect_trajectory_radon(loop, posterior, times, **options).score
            for posterior in posteriors
        ]
        assert_array_equal(batched, single)

    def test_event_without_mass_is_refused(self, loop):
        with pytest.raises(ValueError, match="no mass in any time bin"):
            detect_trajectory_radon(
                loop,
                np.zeros((4, loop.n_bins)),
                np.arange(4) * 0.02,
                band=LOOP_BAND,
                wrap=True,
                zero_rows="line_median",
            )

    def test_open_track_is_refused(self, track):
        posterior = np.full((3, track.n_bins), 1.0 / track.n_bins)
        with pytest.raises(ValueError, match="not a closed-loop track"):
            detect_trajectory_radon(track, posterior, np.arange(3) * 0.02, wrap=True)

    def test_loop_with_edge_spacing_is_refused(self):
        """Gaps count as loop length at each corner but not at the seam."""
        graph = nx.Graph()
        for node, position in enumerate([(0, 0), (100, 0), (100, 100), (0, 100)]):
            graph.add_node(node, pos=position)
        edge_order = [(0, 1), (1, 2), (2, 3), (3, 0)]
        graph.add_edges_from(edge_order)
        spaced = Environment.from_graph(
            graph, edge_order, bin_size=BIN_SIZE, edge_spacing=5.0, units="cm"
        )
        posterior = np.full((3, spaced.n_bins), 1.0 / spaced.n_bins)
        with pytest.raises(ValueError, match="edge_spacing puts gaps of up to 5"):
            detect_trajectory_radon(
                spaced, posterior, np.arange(3) * 0.02, band=LOOP_BAND, wrap=True
            )

    def test_one_dimensional_environment_is_refused(self):
        env = Environment.from_samples(
            np.linspace(0, 100, 200)[:, None], bin_size=5.0, units="cm"
        )
        posterior = np.full((3, env.n_bins), 1.0 / env.n_bins)
        with pytest.raises(ValueError, match="not a track built from a graph"):
            detect_trajectory_radon(env, posterior, np.arange(3) * 0.02, wrap=True)

    def test_band_of_half_the_loop_is_refused(self, loop):
        posterior = np.full((3, loop.n_bins), 1.0 / loop.n_bins)
        with pytest.raises(ValueError, match="every trajectory would score 1"):
            detect_trajectory_radon(
                loop, posterior, np.arange(3) * 0.02, band=200.0, wrap=True
            )

    def test_wrap_must_be_a_bool(self, loop):
        posterior = np.full((3, loop.n_bins), 1.0 / loop.n_bins)
        with pytest.raises(TypeError, match="wrap=True or wrap=False"):
            detect_trajectory_radon(
                loop,
                posterior,
                np.arange(3) * 0.02,
                wrap="yes",  # type: ignore[arg-type]
            )


def _sweep_off_the_end(track):
    """A 2000 cm/s sweep in the middle three of five time bins.

    The first and last time bins are flat. The best line through the sweep
    is on the track only at those three, which no line that starts and ends
    on the track within the 80 ms event can follow.
    """
    centers = np.array([10.0, 50.0, 90.0])
    position = track.bin_centers[:, 0]
    posterior = np.full((5, track.n_bins), 1.0 / track.n_bins)
    bump = np.exp(-0.5 * ((position[None, :] - centers[:, None]) / 2.0) ** 2)
    posterior[1:4] = bump / bump.sum(axis=1, keepdims=True)
    return posterior, np.arange(5) * 0.02


class TestLineEndsOnTrack:
    def test_lines_leaving_the_track_are_not_searched(self, track):
        posterior, times = _sweep_off_the_end(track)
        options = {"band": 10.0, "max_speed": 5000.0}
        any_line = detect_trajectory_radon(track, posterior, times, **options)
        on_track = detect_trajectory_radon(
            track, posterior, times, line_ends="on_track", **options
        )

        # The best of all lines is faster than any line from one end of the
        # track to the other in 80 ms, and leaves the track at both ends.
        assert any_line.speed > 100.0 / times[-1]
        assert any_line.start_position < 0.0
        assert any_line.start_position + any_line.speed * times[-1] > 100.0
        # Every searched line starts and ends on the track.
        speeds = on_track.sinogram_speeds[:, None]
        starts = on_track.sinogram_start_positions[None, :]
        searched = np.isfinite(on_track.sinogram)
        ends = starts + speeds * times[-1]
        assert np.all((starts >= 0.0) & (starts <= 100.0))
        assert np.all((ends[searched] >= 0.0) & (ends[searched] <= 100.0))
        assert np.all(~searched[(ends < 0.0) | (ends > 100.0)])
        best_end = on_track.start_position + on_track.speed * times[-1]
        assert 0.0 <= best_end <= 100.0
        assert on_track.score < any_line.score - 0.2

    def test_speed_is_bounded_by_the_track(self, track):
        posterior, times = _sweep_off_the_end(track)
        result = detect_trajectory_radon(
            track, posterior, times, band=10.0, line_ends="on_track"
        )
        searched = np.isfinite(result.sinogram).any(axis=1)
        fastest = np.abs(result.sinogram_speeds[searched]).max()
        # From the first start position (1 cm) to the last (99 cm) in 80 ms.
        assert fastest == pytest.approx(98.0 / times[-1])

    def test_band_beyond_the_end_holds_the_median_density(self, track):
        """Half the mass spread evenly, half in one bin: median 0.005 per cm."""
        row = np.full(track.n_bins, 0.5 / track.n_bins)
        row[5] += 0.5
        posterior = np.tile(row, (2, 1))
        result = detect_trajectory_radon(
            track,
            posterior,
            np.array([0.0, 0.02]),
            band=10.0,
            line_ends="on_track",
        )
        stationary = int(np.flatnonzero(result.sinogram_speeds == 0.0)[0])
        last = result.sinogram_start_positions[-1]
        assert last == pytest.approx(99.0)
        # Band [89, 109] cm: 11 cm on the track and 9 cm beyond it, both at
        # the median density 0.005 per cm.
        assert result.sinogram[stationary, -1] == pytest.approx(20 * 0.005)
        # A band inside the track holds only the mass on the track.
        middle = int(np.argmin(np.abs(result.sinogram_start_positions - 51.0)))
        assert result.sinogram[stationary, middle] == pytest.approx(20 * 0.005)

    def test_band_mass_is_capped_at_one_per_time_bin(self, track):
        """Padding cannot lift one time bin's band mass above 1."""
        position = track.bin_centers[:, 0]
        spread = np.where(position > 40.0, 1.0, 0.0)
        posterior = np.zeros((2, track.n_bins))
        posterior[0] = spread / spread.sum()  # density 1/60 per cm on 40-100 cm
        posterior[1, 5] = 1.0  # all mass at 10-12 cm: median density 0
        result = detect_trajectory_radon(
            track,
            posterior,
            np.array([0.0, 0.02]),
            band=35.0,
            line_ends="on_track",
        )
        stationary = int(np.flatnonzero(result.sinogram_speeds == 0.0)[0])
        # Near the end the first time bin holds 70 cm at 1/60 per cm,
        # capped at 1, and the second none: (1 + 0) / 2.
        assert result.sinogram_start_positions[-1] > 65.0
        assert result.sinogram[stationary, -1] == pytest.approx(0.5)

    def test_max_speed_still_bounds_the_search(self, track):
        posterior, times = _sweep_off_the_end(track)
        result = detect_trajectory_radon(
            track, posterior, times, band=10.0, max_speed=500.0, line_ends="on_track"
        )
        assert np.abs(result.sinogram_speeds).max() <= 500.0

    def test_wrap_with_on_track_ends_is_refused(self, loop):
        posterior = np.full((3, loop.n_bins), 1.0 / loop.n_bins)
        with pytest.raises(ValueError, match="a closed loop has no ends"):
            detect_trajectory_radon(
                loop,
                posterior,
                np.arange(3) * 0.02,
                wrap=True,
                line_ends="on_track",
            )

    def test_unknown_line_ends_is_refused(self, track):
        posterior = np.full((3, track.n_bins), 1.0 / track.n_bins)
        with pytest.raises(ValueError, match="line_ends must be one of"):
            detect_trajectory_radon(
                track,
                posterior,
                np.arange(3) * 0.02,
                line_ends="inside",  # type: ignore[arg-type]
            )


def _event_across_the_seam(loop):
    return _loop_sweep(first_bin=148)


FAMILIES = [
    pytest.param(
        "loop", _event_across_the_seam, {"band": LOOP_BAND, "wrap": True}, id="wrap"
    ),
    pytest.param(
        "track",
        _sweep_off_the_end,
        {"band": 10.0, "line_ends": "on_track"},
        id="line_ends_on_track",
    ),
]


def _noisy_copies(env, event, n_copies, seed):
    """The event's posterior mixed with independent noise, ``n_copies`` times.

    The events keep the sweep that only the option scores correctly: one
    across the loop's seam, or one leaving the track.
    """
    posterior, times = event(env)
    rng = np.random.default_rng(seed)
    noise = rng.dirichlet(np.ones(env.n_bins) * 0.3, size=(n_copies, len(times)))
    return list(0.6 * posterior + 0.4 * noise), times


@pytest.mark.parametrize(("environment", "event", "options"), FAMILIES)
class TestShuffleScoring:
    def test_batched_scores_equal_single_scores(
        self, environment, event, options, request
    ):
        from neurospatial.replay.significance import _RadonScorer, _scorer_for

        env = request.getfixturevalue(environment)
        posteriors, times = _noisy_copies(env, event, 6, seed=4)
        scorer = _scorer_for(
            detect_trajectory_radon, env, posteriors[0], times, dict(options)
        )
        assert isinstance(scorer, _RadonScorer)
        batched = scorer.scores(iter(posteriors), len(posteriors))
        single = [
            detect_trajectory_radon(env, posterior, times, **options).score
            for posterior in posteriors
        ]
        assert_array_equal(batched, single)
        assert scorer.observe().score == single[0]

    def test_replay_significance_is_seeded(self, environment, event, options, request):
        env = request.getfixturevalue(environment)
        (posterior,), times = _noisy_copies(env, event, 1, seed=5)
        common = {"score_options": options, "n_shuffles": 25}
        first = replay_significance(env, posterior, times, rng=1, **common)
        again = replay_significance(env, posterior, times, rng=1, **common)
        other = replay_significance(env, posterior, times, rng=2, **common)

        null = first.shuffles["column_cycle"].null_scores
        assert_array_equal(null, again.shuffles["column_cycle"].null_scores)
        assert not np.array_equal(null, other.shuffles["column_cycle"].null_scores)
        assert first.observed.score == (
            detect_trajectory_radon(env, posterior, times, **options).score
        )
