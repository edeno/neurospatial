"""Per-event metrics for criteria-based replay detection."""

from __future__ import annotations

import pickle

import networkx as nx
import numpy as np
import pytest
from environment_helpers import renumbered, strips_1d

from neurospatial import Environment
from neurospatial.decoding import (
    DecodingResult,
    JointDecodingResult,
    decode_jointly,
    map_bin,
    mean_position,
)
from neurospatial.replay import (
    TrajectoryMetrics,
    template_fractions,
    trajectory_metrics,
)

DT = 0.02


@pytest.fixture(scope="module")
def track():
    """A 100 cm straight track in ten 10 cm bins, centers 5, 15, ..., 95."""
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    )


@pytest.fixture(scope="module")
def l_track():
    """An L-shaped track: 50 cm east, then 50 cm north, in 10 cm bins.

    Bin 4 (center (45, 0)) and bin 5 (center (50, 5)) meet at the corner:
    10 cm apart along the track, sqrt(50) cm apart in a straight line.
    """
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(50.0, 0.0))
    graph.add_node(2, pos=(50.0, 50.0))
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    return Environment.from_graph(
        graph, edge_order=[(0, 1), (1, 2)], edge_spacing=0.0, bin_size=10.0, units="cm"
    )


@pytest.fixture(scope="module")
def arena():
    """A 2-D arena of 10 cm square bins."""
    x, y = np.meshgrid(np.arange(0.5, 50.0), np.arange(0.5, 50.0))
    return Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=10.0)


def _times(n):
    return np.arange(n) * DT


def _two_point(env, first, second, weight):
    """One posterior row: ``weight`` in bin ``first``, the rest in ``second``."""
    row = np.zeros((1, env.n_bins))
    row[0, first] = weight
    row[0, second] = 1.0 - weight
    return row


class TestSweep:
    """One bin per time bin from bin 2 to bin 7 of the straight track."""

    @pytest.fixture(scope="class")
    def metrics(self, track):
        posterior = np.eye(track.n_bins)[2:8]
        return trajectory_metrics(track, posterior, _times(6))

    def test_map_follows_the_sweep(self, metrics):
        np.testing.assert_array_equal(metrics.map_bins, np.arange(2, 8))
        np.testing.assert_allclose(metrics.map_positions[:, 0], np.arange(25, 85, 10))
        np.testing.assert_allclose(metrics.peak_probability, 1.0)
        assert metrics.mean_peak_probability == pytest.approx(1.0)

    def test_jumps_equal_the_bin_spacing(self, metrics):
        np.testing.assert_allclose(metrics.map_jumps, 10.0)
        np.testing.assert_allclose(metrics.mean_jumps, 10.0)
        assert metrics.max_map_jump == pytest.approx(10.0)
        assert metrics.max_mean_jump == pytest.approx(10.0)

    def test_range_path_length_and_start_to_end(self, metrics):
        assert metrics.track_length == pytest.approx(100.0)
        assert metrics.map_range == pytest.approx(50.0)
        assert metrics.mean_range == pytest.approx(50.0)
        assert metrics.map_path_length == pytest.approx(5 * 10.0)
        assert metrics.start_to_end_map_distance == pytest.approx(50.0)
        assert metrics.start_to_end_mean_distance == pytest.approx(50.0)

    def test_dispersion_is_the_rms_distance_from_the_mean_position(self, metrics):
        # Means at 25, 35, ..., 75 around 50: deviations +-5, +-15, +-25.
        expected = np.sqrt((5**2 + 15**2 + 25**2) / 3)
        assert metrics.dispersion == pytest.approx(expected)

    def test_a_point_posterior_has_no_spread(self, metrics):
        np.testing.assert_allclose(metrics.spread, 0.0, atol=1e-12)

    def test_duration_and_counts(self, metrics):
        assert metrics.n_time_bins == 6
        assert metrics.duration == pytest.approx(5 * DT)
        assert metrics.units == "cm"
        assert metrics.metric == "geodesic"

    def test_euclidean_equals_geodesic_on_a_straight_track(self, track, metrics):
        euclidean = trajectory_metrics(
            track, np.eye(track.n_bins)[2:8], _times(6), metric="euclidean"
        )
        np.testing.assert_allclose(euclidean.map_jumps, metrics.map_jumps)
        assert euclidean.map_path_length == pytest.approx(metrics.map_path_length)

    def test_means_over_the_event(self, track):
        """Half-and-half rows over neighbors, then a jump of 35 cm.

        Rows 0 and 1 split their mass over two bins 10 cm apart (spread 5 cm),
        row 2 is a point (spread 0); the means at 20, 30 and 65 cm move 10 cm,
        then 35 cm.
        """
        posterior = np.zeros((3, track.n_bins))
        posterior[0, [1, 2]] = 0.5
        posterior[1, [2, 3]] = 0.5
        posterior[2, 6] = 1.0
        metrics = trajectory_metrics(track, posterior, _times(3))
        np.testing.assert_allclose(metrics.spread, [5.0, 5.0, 0.0], atol=1e-12)
        assert metrics.mean_spread == pytest.approx(10.0 / 3)
        np.testing.assert_allclose(metrics.mean_jumps, [10.0, 35.0])
        assert metrics.mean_mean_jump == pytest.approx(22.5)
        summary = metrics.summary()
        assert summary["mean_spread"] == pytest.approx(10.0 / 3)
        assert summary["mean_mean_jump"] == pytest.approx(22.5)

    def test_reverse_sweep_has_the_same_magnitudes(self, track, metrics):
        reverse = trajectory_metrics(track, np.eye(track.n_bins)[7:1:-1], _times(6))
        for name in ("max_map_jump", "map_range", "map_path_length", "dispersion"):
            assert getattr(reverse, name) == pytest.approx(getattr(metrics, name))


class TestDistances:
    def test_geodesic_goes_around_the_corner(self, l_track):
        posterior = np.eye(l_track.n_bins)[[3, 4, 5, 6]]
        geodesic = trajectory_metrics(l_track, posterior, _times(4))
        euclidean = trajectory_metrics(
            l_track, posterior, _times(4), metric="euclidean"
        )
        np.testing.assert_allclose(geodesic.map_jumps, [10.0, 10.0, 10.0])
        np.testing.assert_allclose(euclidean.map_jumps, [10.0, np.sqrt(50.0), 10.0])
        # Bin 3 (35, 0) to bin 6 (50, 15): 30 cm along the track.
        assert geodesic.start_to_end_map_distance == pytest.approx(30.0)
        assert euclidean.start_to_end_map_distance == pytest.approx(np.hypot(15, 15))

    def test_end_to_end_of_the_l(self, l_track):
        posterior = np.eye(l_track.n_bins)[[0, 9]]
        geodesic = trajectory_metrics(l_track, posterior, _times(2))
        euclidean = trajectory_metrics(
            l_track, posterior, _times(2), metric="euclidean"
        )
        assert geodesic.max_map_jump == pytest.approx(90.0)
        assert euclidean.max_map_jump == pytest.approx(np.hypot(45.0, 45.0))
        # The range is along the track, whatever the metric.
        assert geodesic.map_range == pytest.approx(90.0)
        assert euclidean.map_range == pytest.approx(90.0)

    def test_mean_jumps_are_euclidean_between_means(self, l_track):
        posterior = np.eye(l_track.n_bins)[[4, 5]]
        metrics = trajectory_metrics(l_track, posterior, _times(2))
        assert metrics.mean_jumps[0] == pytest.approx(np.sqrt(50.0))
        np.testing.assert_allclose(
            metrics.mean_positions, mean_position(l_track, posterior)
        )


class TestSpread:
    @pytest.mark.parametrize("weight", [0.5, 0.25, 0.9])
    def test_two_point_posterior_on_a_track(self, track, weight):
        # Mass w at 15 cm and 1 - w at 55 cm: spread = 40 sqrt(w (1 - w)).
        metrics = trajectory_metrics(track, _two_point(track, 1, 5, weight), [0.0])
        assert metrics.spread[0] == pytest.approx(40.0 * np.sqrt(weight * (1 - weight)))

    def test_isotropic_spread_in_the_arena(self, arena):
        # Half the mass 40 cm right and 40 cm up of the other half: each
        # deviation is (+-20, +-20), so the spread is sqrt(20^2 + 20^2).
        first, second = arena.bin_at(np.array([[5.0, 5.0], [45.0, 45.0]]))
        metrics = trajectory_metrics(
            arena, _two_point(arena, first, second, 0.5), [0.0]
        )
        assert metrics.spread[0] == pytest.approx(np.sqrt(800.0))
        # Mass along one axis only still has spread.
        first, second = arena.bin_at(np.array([[5.0, 25.0], [45.0, 25.0]]))
        metrics = trajectory_metrics(
            arena, _two_point(arena, first, second, 0.5), [0.0]
        )
        assert metrics.spread[0] == pytest.approx(20.0)

    def test_spread_across_a_corner_uses_bin_coordinates(self, l_track):
        # Bins 4 (45, 0) and 5 (50, 5) are sqrt(50) cm apart: half that.
        metrics = trajectory_metrics(l_track, _two_point(l_track, 4, 5, 0.5), [0.0])
        assert metrics.spread[0] == pytest.approx(np.sqrt(50.0) / 2)


class TestArena:
    def test_track_only_quantities_are_nan_off_a_track(self, arena):
        rng = np.random.default_rng(0)
        posterior = rng.dirichlet(np.ones(arena.n_bins), size=4)
        metrics = trajectory_metrics(arena, posterior, _times(4))
        assert np.isnan(metrics.track_length)
        assert np.isnan(metrics.map_range)
        assert np.isnan(metrics.mean_range)
        assert np.isfinite(metrics.max_map_jump)
        np.testing.assert_array_equal(metrics.map_bins, map_bin(arena, posterior))

    def test_map_jumps_default_to_straight_lines(self, arena):
        # A (20, 10) cm step: 22.36 cm straight, 24.14 cm along the bin graph
        # (one diagonal and one straight step between 10 cm bins).
        bins = arena.bin_at(np.array([[5.0, 5.0], [25.0, 15.0], [25.0, 35.0]]))
        posterior = np.eye(arena.n_bins)[bins]
        default = trajectory_metrics(arena, posterior, _times(3))
        geodesic = trajectory_metrics(arena, posterior, _times(3), metric="geodesic")
        assert default.metric == "euclidean"
        np.testing.assert_allclose(default.map_jumps, [np.hypot(20, 10), 20.0])
        assert geodesic.map_jumps[0] == pytest.approx(10 * np.sqrt(2) + 10)
        assert default.start_to_end_map_distance == pytest.approx(np.hypot(20, 30))


class TestTrackPositions:
    @pytest.fixture(scope="class")
    def t_maze(self):
        """Stem from (0, 0) up to a junction at (0, 50); arms to (-50, 50), (50, 50)."""
        return Environment.maze(
            kind="t",
            node_positions={
                "stem": (0.0, 0.0),
                "junction": (0.0, 50.0),
                "left": (-50.0, 50.0),
                "right": (50.0, 50.0),
            },
            bin_size=10.0,
            track_width=10.0,
            units="cm",
        )

    def test_ranges_are_nan_on_a_branched_track(self, t_maze):
        bins = t_maze.bin_at(np.array([[0.0, 5.0], [0.0, 45.0], [-25.0, 50.0]]))
        metrics = trajectory_metrics(t_maze, np.eye(t_maze.n_bins)[bins], _times(3))
        assert np.isnan(metrics.map_range)
        assert np.isnan(metrics.mean_range)
        assert metrics.track_length == pytest.approx(150.0)
        assert metrics.metric == "geodesic"
        # Stem (0, 5) to (0, 45), then through the junction to (-25, 50).
        np.testing.assert_allclose(metrics.map_jumps, [40.0, 5.0 + 25.0])

    def test_spread_across_the_junction(self, t_maze):
        first, second = t_maze.bin_at(np.array([[0.0, 45.0], [-5.0, 50.0]]))
        metrics = trajectory_metrics(
            t_maze, _two_point(t_maze, first, second, 0.5), [0.0]
        )
        distance = np.linalg.norm(
            t_maze.bin_centers[first] - t_maze.bin_centers[second]
        )
        assert metrics.spread[0] == pytest.approx(distance / 2)

    def test_edge_spacing_does_not_count(self):
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0, 0.0))
        graph.add_node(1, pos=(50.0, 0.0))
        graph.add_node(2, pos=(50.0, 50.0))
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        spaced = Environment.from_graph(
            graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=20.0,
            bin_size=10.0,
            units="cm",
        )
        sweep = trajectory_metrics(spaced, np.eye(spaced.n_bins)[[0, 9]], _times(2))
        assert sweep.track_length == pytest.approx(100.0)
        assert sweep.map_range == pytest.approx(90.0)
        assert sweep.mean_range == pytest.approx(90.0)
        corner = trajectory_metrics(spaced, _two_point(spaced, 4, 5, 0.5), [0.0])
        assert corner.spread[0] == pytest.approx(np.sqrt(50.0) / 2)


class TestTemplateShare:
    def test_scaled_rows_give_the_same_positions(self, track):
        posterior = np.random.default_rng(1).dirichlet(np.ones(track.n_bins), size=5)
        share = posterior * np.array([0.9, 0.2, 0.5, 1.0, 0.01])[:, None]
        full = trajectory_metrics(track, posterior, _times(5))
        part = trajectory_metrics(track, share, _times(5))
        np.testing.assert_array_equal(part.map_bins, full.map_bins)
        np.testing.assert_allclose(part.mean_positions, full.mean_positions)
        np.testing.assert_allclose(part.spread, full.spread)
        np.testing.assert_allclose(part.map_jumps, full.map_jumps)
        np.testing.assert_allclose(part.mean_jumps, full.mean_jumps)
        assert part.dispersion == pytest.approx(full.dispersion)
        for name in (
            "map_range",
            "mean_range",
            "start_to_end_map_distance",
            "start_to_end_mean_distance",
            "map_path_length",
            "max_map_jump",
            "max_mean_jump",
        ):
            assert getattr(part, name) == pytest.approx(getattr(full, name)), name
        # Peak mass is reported as given.
        np.testing.assert_allclose(
            part.peak_probability, share[np.arange(5), full.map_bins]
        )
        np.testing.assert_allclose(part.max_posterior, share.max(axis=0))

    def test_mallory_coverage_on_a_joint_decode(self, track):
        x = track.bin_centers[:, 0]
        centers = np.linspace(5.0, 95.0, 8)
        outbound = 1.0 + 20.0 * np.exp(-0.5 * ((x - centers[:, None]) / 8.0) ** 2)
        counts = np.random.default_rng(2).poisson(outbound[:, 2:7].T * 0.1)
        result = decode_jointly(
            track,
            counts,
            {"outbound": outbound, "inbound": outbound[[3, 6, 1, 4, 7, 2, 5, 0]]},
            dt=0.1,
        )
        joint = result.joint("outbound")
        # coverage_wu: bins whose event-max joint mass exceeds 5 / (2 n_bins),
        # as a fraction of the map's bins.
        expected = np.mean(joint.max(axis=0) > 5 / (2 * track.n_bins))
        assert 0 < expected < 1
        metrics = trajectory_metrics(track, joint, _times(5))
        assert metrics.well_decoded_fraction(
            chance_bins=2 * track.n_bins
        ) == pytest.approx(expected)

    @pytest.mark.parametrize("chance_bins", [0, -3, 2.5, True])
    def test_refuses_a_bad_chance_bins(self, track, chance_bins):
        metrics = trajectory_metrics(track, np.eye(track.n_bins)[:2], _times(2))
        with pytest.raises(ValueError, match="chance_bins"):
            metrics.well_decoded_fraction(chance_bins=chance_bins)

    def test_refuses_rows_without_mass(self, track):
        posterior = np.eye(track.n_bins)[:3].copy()
        posterior[1] = 0.0
        with pytest.raises(ValueError, match="1 have no mass"):
            trajectory_metrics(track, posterior, _times(3))


class TestEmptyTimeBins:
    def test_empty_fraction(self, track):
        counts = np.array([[0, 0], [1, 0], [0, 0], [0, 3], [2, 1]])
        metrics = trajectory_metrics(
            track, np.eye(track.n_bins)[:5], _times(5), spike_counts=counts
        )
        np.testing.assert_array_equal(
            metrics.empty_time_bins, [True, False, True, False, False]
        )
        assert metrics.empty_fraction == pytest.approx(0.4)

    def test_nan_without_spike_counts(self, track):
        metrics = trajectory_metrics(track, np.eye(track.n_bins)[:5], _times(5))
        assert metrics.empty_time_bins is None
        assert np.isnan(metrics.empty_fraction)


class TestWellDecodedFraction:
    def test_counts_bins_above_threshold_times_chance(self, track):
        # Chance is 1/10. Bins 0-3 reach 0.7 at some time; the others get
        # 0.3/9 < 0.5 (5 x chance) everywhere.
        posterior = np.full((4, track.n_bins), 0.3 / 9)
        posterior[np.arange(4), np.arange(4)] = 0.7
        metrics = trajectory_metrics(track, posterior, _times(4))
        assert metrics.well_decoded_fraction() == pytest.approx(0.4)
        assert metrics.well_decoded_fraction(threshold=8.0) == pytest.approx(0.0)
        # 0.3 / 9 > 0.1 * 0.3: every bin passes a threshold of 0.3 x chance.
        assert metrics.well_decoded_fraction(threshold=0.3) == pytest.approx(1.0)

    def test_weighs_bins_by_their_length(self):
        env = strips_1d([0.0, 10.0, 40.0, 50.0])  # bins 10, 30 and 10 cm long
        posterior = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        metrics = trajectory_metrics(env, posterior, _times(2))
        # Chance 1/3, threshold 5/3 > 1: nothing passes.
        assert metrics.well_decoded_fraction() == pytest.approx(0.0)
        assert metrics.well_decoded_fraction(threshold=2.0) == pytest.approx(0.8)

    @pytest.mark.parametrize("threshold", [0.0, -1.0, np.nan])
    def test_refuses_a_non_positive_threshold(self, track, threshold):
        metrics = trajectory_metrics(track, np.eye(track.n_bins)[:2], _times(2))
        with pytest.raises(ValueError, match="positive finite"):
            metrics.well_decoded_fraction(threshold=threshold)


class TestRenumbering:
    def test_renumbered_bins_give_the_same_metrics(self):
        env = strips_1d(np.r_[0.0, np.cumsum(np.full(20, 5.0))])
        order = np.random.default_rng(3).permutation(env.n_bins)
        shuffled = renumbered(env, order)
        rng = np.random.default_rng(4)
        posterior = rng.dirichlet(np.ones(env.n_bins) * 0.3, size=8)
        counts = rng.poisson(0.5, size=(8, 6))
        original = trajectory_metrics(env, posterior, _times(8), spike_counts=counts)
        moved = trajectory_metrics(
            shuffled, posterior[:, order], _times(8), spike_counts=counts
        )
        assert moved.summary() == pytest.approx(original.summary())
        np.testing.assert_array_equal(order[moved.map_bins], original.map_bins)
        np.testing.assert_allclose(moved.spread, original.spread)
        np.testing.assert_allclose(moved.map_jumps, original.map_jumps)


class TestEdgeCases:
    def test_one_time_bin(self, track):
        metrics = trajectory_metrics(track, np.eye(track.n_bins)[:1], [0.5])
        assert metrics.map_jumps.shape == (0,)
        assert np.isnan(metrics.max_map_jump)
        assert np.isnan(metrics.max_mean_jump)
        assert np.isnan(metrics.mean_mean_jump)
        assert metrics.mean_spread == pytest.approx(0.0, abs=1e-12)
        assert metrics.map_path_length == 0.0
        assert metrics.duration == 0.0
        assert metrics.dispersion == 0.0

    def test_disconnected_bins_are_infinitely_far(self):
        import math

        from neurospatial.binning import (
            BinConnections,
            Binning,
            MissingFaces,
            NearestBinLocator,
            SpatialBins,
        )

        centers = np.array([[5.0], [25.0]])
        env = Environment.from_bins(
            Binning(
                "explicit",
                SpatialBins.from_arrays(centers, np.array([10.0, 10.0]), 1),
                BinConnections.from_arrays(np.empty((0, 2), dtype=int), np.empty(0)),
                NearestBinLocator(centers, max_distance=math.inf),
                MissingFaces("two unconnected bins"),
            ),
            units="cm",
        )
        geodesic = trajectory_metrics(env, np.eye(2), _times(2))
        euclidean = trajectory_metrics(env, np.eye(2), _times(2), metric="euclidean")
        assert geodesic.max_map_jump == np.inf
        assert euclidean.max_map_jump == pytest.approx(20.0)


class TestResult:
    @pytest.fixture(scope="class")
    def metrics(self, track):
        counts = np.ones((3, 2), dtype=int)
        return trajectory_metrics(
            track, np.eye(track.n_bins)[:3], _times(3), spike_counts=counts
        )

    def test_arrays_are_read_only(self, metrics):
        for name in ("map_bins", "map_jumps", "spread", "empty_time_bins"):
            array = getattr(metrics, name)
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0

    def test_pickle_keeps_arrays_read_only(self, metrics):
        copy = pickle.loads(pickle.dumps(metrics))
        assert not copy.map_jumps.flags.writeable
        assert copy.summary() == pytest.approx(metrics.summary(), nan_ok=True)

    def test_summary_holds_only_scalars(self, metrics):
        summary = metrics.summary()
        assert all(not isinstance(value, np.ndarray) for value in summary.values())
        assert summary["well_decoded_fraction"] == pytest.approx(0.3)
        assert isinstance(metrics, TrajectoryMetrics)
        assert "TrajectoryMetrics" in repr(metrics)


class TestRefusals:
    def test_unknown_metric(self, track):
        with pytest.raises(ValueError, match="metric must be one of"):
            trajectory_metrics(
                track, np.eye(track.n_bins)[:2], _times(2), metric="manhattan"
            )

    def test_wrong_number_of_columns(self, track):
        with pytest.raises(ValueError, match="one column per environment bin"):
            trajectory_metrics(track, np.eye(4), _times(4))

    def test_rows_that_are_not_probabilities(self, track):
        posterior = np.eye(track.n_bins)[:3] * 2
        with pytest.raises(ValueError, match="sum to more than 1"):
            trajectory_metrics(track, posterior, _times(3))

    def test_nan_rows(self, track):
        posterior = np.eye(track.n_bins)[:3].copy()
        posterior[1] = np.nan
        with pytest.raises(ValueError, match="drop undecodable time bins"):
            trajectory_metrics(track, posterior, _times(3))

    def test_times_of_the_wrong_length(self, track):
        with pytest.raises(ValueError, match="one finite time per posterior row"):
            trajectory_metrics(track, np.eye(track.n_bins)[:3], _times(4))

    def test_times_out_of_order(self, track):
        with pytest.raises(ValueError, match=r"np\.argsort\(times\)"):
            trajectory_metrics(track, np.eye(track.n_bins)[:3], [0.0, 0.04, 0.02])

    def test_spike_counts_of_the_wrong_shape(self, track):
        with pytest.raises(ValueError, match="one row per posterior row"):
            trajectory_metrics(
                track,
                np.eye(track.n_bins)[:3],
                _times(3),
                spike_counts=np.ones((4, 2)),
            )

    def test_negative_spike_counts(self, track):
        with pytest.raises(ValueError, match="non-negative"):
            trajectory_metrics(
                track,
                np.eye(track.n_bins)[:2],
                _times(2),
                spike_counts=-np.ones((2, 2)),
            )

    def test_polar_environment(self):
        from neurospatial import CartesianOnlyError

        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 30.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=15.0,
            angle_bin_size=np.pi / 4,
        )
        with pytest.raises(CartesianOnlyError, match="allocentric"):
            trajectory_metrics(env, np.eye(env.n_bins)[:2], _times(2))


def _joint(env, conditional, template_probability, names=("outbound", "inbound")):
    """A JointDecodingResult from hand-made conditional posteriors."""
    return JointDecodingResult(
        template_names=names,
        template_probability=np.asarray(template_probability, dtype=float),
        conditional={
            name: DecodingResult(posterior=posterior, env=env)
            for name, posterior in zip(names, conditional, strict=True)
        },
    )


class TestTemplateFractions:
    def test_ambrose_equation_one_by_hand(self, track):
        uniform = np.full((3, track.n_bins), 1.0 / track.n_bins)
        result = _joint(track, [uniform, uniform], [[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
        fractions = template_fractions(result)
        assert list(fractions) == ["outbound", "inbound"]
        assert fractions["outbound"] == pytest.approx((0.8 + 0.6 + 0.1) / 3)
        assert fractions["inbound"] == pytest.approx((0.2 + 0.4 + 0.9) / 3)

    def test_end_masking_by_hand(self, track):
        # Outbound puts its mass in bin 0 (center 5 cm) at time 0 and in bin 5
        # at time 1; inbound always in bin 5. Masking 10 cm at each end drops
        # bins 0 and 9 only.
        outbound = np.eye(track.n_bins)[[0, 5]]
        inbound = np.eye(track.n_bins)[[5, 5]]
        result = _joint(track, [outbound, inbound], [[0.5, 0.5], [0.5, 0.5]])
        assert template_fractions(result)["outbound"] == pytest.approx(0.5)
        masked = template_fractions(result, exclude_end_distance=10.0)
        # Outbound keeps 0.5 (time 1) of 1.5 kept mass.
        assert masked["outbound"] == pytest.approx(0.5 / 1.5)
        assert masked["inbound"] == pytest.approx(1.0 / 1.5)

    def test_masking_counts_bins_by_center(self, track):
        # 10 cm masks one 10 cm bin at each end (centers 5 and 95 cm); 15.1 cm
        # masks a second (centers 15 and 85 cm).
        outbound = np.eye(track.n_bins)[[1]]
        inbound = np.eye(track.n_bins)[[4]]
        result = _joint(track, [outbound, inbound], [[0.5, 0.5]])
        assert template_fractions(result, exclude_end_distance=10.0)[
            "outbound"
        ] == pytest.approx(0.5)
        assert template_fractions(result, exclude_end_distance=15.1)[
            "outbound"
        ] == pytest.approx(0.0)

    def test_matches_the_mean_template_probability_of_a_decode(self, track):
        x = track.bin_centers[:, 0]
        centers = np.linspace(5.0, 95.0, 8)
        outbound = 1.0 + 20.0 * np.exp(-0.5 * ((x - centers[:, None]) / 8.0) ** 2)
        counts = np.random.default_rng(0).poisson(outbound[:, :5].T * DT)
        result = decode_jointly(
            track, counts, {"outbound": outbound, "inbound": outbound[::-1]}, dt=DT
        )
        fractions = template_fractions(result)
        np.testing.assert_allclose(
            list(fractions.values()), result.template_probability.mean(axis=0)
        )

    def test_refuses_another_result(self, track):
        with pytest.raises(TypeError, match="decode_jointly"):
            template_fractions(DecodingResult(posterior=np.eye(10), env=track))

    @pytest.mark.parametrize("distance", [-1.0, np.inf])
    def test_refuses_a_bad_distance(self, track, distance):
        uniform = np.full((1, track.n_bins), 1.0 / track.n_bins)
        result = _joint(track, [uniform, uniform], [[0.5, 0.5]])
        with pytest.raises(ValueError, match="exclude_end_distance"):
            template_fractions(result, exclude_end_distance=distance)

    def test_refuses_masking_off_a_track(self, arena):
        uniform = np.full((1, arena.n_bins), 1.0 / arena.n_bins)
        result = _joint(arena, [uniform, uniform], [[0.5, 0.5]])
        assert template_fractions(result)["outbound"] == pytest.approx(0.5)
        with pytest.raises(ValueError, match=r"exclude_end_distance=0\.0"):
            template_fractions(result, exclude_end_distance=5.0)

    def test_refuses_masking_everything(self, track):
        uniform = np.full((1, track.n_bins), 1.0 / track.n_bins)
        result = _joint(track, [uniform, uniform], [[0.5, 0.5]])
        with pytest.raises(ValueError, match="smaller exclude_end_distance"):
            template_fractions(result, exclude_end_distance=60.0)
