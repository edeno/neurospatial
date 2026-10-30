"""Tests for trajectory simulation functions."""

import numpy as np
import pytest

from neurospatial.simulation.trajectory import (
    simulate_trajectory_laps,
    simulate_trajectory_ou,
    simulate_trajectory_sinusoidal,
)


@pytest.fixture
def linear_track_env():
    """A real 1D linearized track (GraphLayout) for sinusoidal motion tests."""
    import networkx as nx

    from neurospatial import Environment

    g = nx.Graph()
    g.add_node(0, pos=(0.0,))
    g.add_node(1, pos=(50.0,))
    g.add_node(2, pos=(100.0,))
    g.add_edge(0, 1, distance=50.0)
    g.add_edge(1, 2, distance=50.0)
    env = Environment.from_graph(
        graph=g, edge_order=[(0, 1), (1, 2)], edge_spacing=0.0, bin_size=2.0
    )
    env = env.with_metadata(units="cm")
    return env


@pytest.fixture(scope="module")
def large_arena():
    """A 4 x 4 m arena, where walls rarely interrupt a walk."""
    from shapely.geometry import box

    from neurospatial import Environment

    return Environment.from_polygon(box(0, 0, 400, 400), bin_size=10.0, units="cm")


class TestSimulateTrajectoryOU:
    """Tests for Ornstein-Uhlenbeck trajectory simulation."""

    def test_basic_trajectory_generation(self, simple_2d_env):
        """Test basic trajectory generation works."""
        positions, times = simulate_trajectory_ou(
            simple_2d_env,
            duration=1.0,
            dt=0.01,
            seed=42,
            speed_units="cm",
        )

        # Check shapes
        assert positions.shape[1] == 2  # 2D environment
        assert len(times) == len(positions)
        assert len(times) == 100  # 1.0s / 0.01s dt

    def test_trajectory_stays_in_bounds(self, simple_2d_env):
        """Test that trajectory stays within environment bounds."""
        positions, _times = simulate_trajectory_ou(
            simple_2d_env,
            duration=10.0,
            boundary_mode="reflect",
            seed=42,
            speed_units="cm",
        )

        # Check all positions are within environment
        for pos in positions:
            assert simple_2d_env.contains(pos), f"Position {pos} outside environment"

    def test_speed_decorrelates_over_the_coherence_time(self, large_arena):
        """The speed's normal scores have autocorrelation exp(-lag / coherence_time)."""
        from scipy.stats import gamma, norm

        coherence_time, dt = 0.5, 0.01
        positions, _times = simulate_trajectory_ou(
            large_arena,
            duration=2000.0,
            dt=dt,
            speed_mean=8.0,
            speed_std=4.0,
            coherence_time=coherence_time,
            seed=42,
            speed_units="cm",
        )
        speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
        # Speed is a gamma quantile of a standard-normal OU process; undoing
        # the quantile recovers that process.
        z = norm.ppf(gamma.cdf(speed, 4.0, scale=2.0))
        z = z[np.isfinite(z)]
        for lag_s in (0.25, 0.5, 1.0):
            lag = round(lag_s / dt)
            acf = np.corrcoef(z[:-lag], z[lag:])[0, 1]
            # About 2000 independent stretches: the standard error is ~0.02.
            assert acf == pytest.approx(np.exp(-lag_s / coherence_time), abs=0.05)

    def test_statistics_do_not_depend_on_the_time_step(self, large_arena):
        """Speed and velocity autocorrelation are the same at any dt."""
        results = []
        for dt in (0.005, 0.02, 0.1):
            positions, _times = simulate_trajectory_ou(
                large_arena, duration=1000.0, dt=dt, speed_units="cm", seed=3
            )
            velocity = np.diff(positions, axis=0) / dt
            speed = np.linalg.norm(velocity, axis=1)
            acf = [
                np.mean(np.sum(velocity[:-lag] * velocity[lag:], axis=1))
                / np.mean(speed**2)
                for lag in (round(0.5 / dt), round(1.0 / dt))
            ]
            results.append([speed.mean(), speed.std(), *acf])
        results = np.array(results)
        # The default speed is 8 +/- 4 cm/s.
        np.testing.assert_allclose(results[:, :2], [[8.0, 4.0]] * 3, atol=0.5)
        np.testing.assert_allclose(results[:, 2:], results[[0] * 3, 2:], atol=0.05)

    @pytest.mark.parametrize("dt_over_tau", [700.0, 1000.0])
    def test_3d_heading_is_uncorrelated_when_dt_dwarfs_tau(self, dt_over_tau):
        """exp(-dt/τ) underflows past dt/τ ≈ 745: every step turns uniformly."""
        from neurospatial import Environment

        corners = np.stack(
            np.meshgrid(*[np.linspace(0, 1000, 11)] * 3, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        env = Environment.from_samples(corners, bin_size=100.0, units="cm")
        dt = 0.1
        positions, _ = simulate_trajectory_ou(
            env,
            duration=4000 * dt,
            dt=dt,
            speed_units="cm",
            speed_mean=1e-3,
            speed_std=0.0,
            rotational_velocity_coherence_time=dt / dt_over_tau,
            start_position=np.array([500.0, 500.0, 500.0]),
            seed=3,
        )
        headings = np.diff(positions, axis=0)
        headings /= np.linalg.norm(headings, axis=1, keepdims=True)
        cosines = np.sum(headings[:-1] * headings[1:], axis=1)
        # Uniform turns on the sphere: mean cosine 0, variance 1/3.
        assert np.mean(cosines) == pytest.approx(0.0, abs=0.04)
        assert np.var(cosines) == pytest.approx(1 / 3, abs=0.03)

    @pytest.mark.parametrize("dt_over_tau", [1 / 8, 1 / 2, 1.0, 2.0])
    def test_3d_heading_decorrelates_as_exp_minus_t_over_tau_at_any_dt(
        self, dt_over_tau
    ):
        """Heading correlation at lag ``t`` is ``exp(-t / τ)`` at any time step.

        The walk is too slow to reach a wall, so every step's direction is
        the heading.
        """
        from neurospatial import Environment

        corners = np.stack(
            np.meshgrid(*[np.linspace(0, 1000, 11)] * 3, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        env = Environment.from_samples(corners, bin_size=100.0, units="cm")
        tau = 1.0
        dt = dt_over_tau * tau
        positions, _ = simulate_trajectory_ou(
            env,
            duration=4000 * tau,
            dt=dt,
            speed_units="cm",
            speed_mean=1e-3,
            speed_std=0.0,
            rotational_velocity_coherence_time=tau,
            start_position=np.array([500.0, 500.0, 500.0]),
            seed=3,
        )
        headings = np.diff(positions, axis=0)
        headings /= np.linalg.norm(headings, axis=1, keepdims=True)
        lag = max(1, round(tau / dt))
        correlation = np.mean(np.sum(headings[:-lag] * headings[lag:], axis=1))
        assert correlation == pytest.approx(np.exp(-lag * dt / tau), abs=0.04)

    def test_default_speed_is_rat_scale_in_the_environment_units(self):
        """8 cm/s by default, expressed in env.units."""
        from shapely.geometry import box

        from neurospatial import Environment

        env_m = Environment.from_polygon(box(0, 0, 4, 4), bin_size=0.1, units="m")
        positions, _times = simulate_trajectory_ou(
            env_m, duration=600.0, dt=0.05, speed_units="m", seed=0
        )
        speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / 0.05
        assert speed.mean() == pytest.approx(0.08, rel=0.1)
        assert speed.std() == pytest.approx(0.04, rel=0.15)

    def test_default_speed_needs_metric_units(self):
        from shapely.geometry import box

        from neurospatial import Environment

        env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="px")
        with pytest.raises(ValueError, match="needs speed_mean"):
            simulate_trajectory_ou(env, duration=1.0, speed_units="px")
        positions, _times = simulate_trajectory_ou(
            env, duration=1.0, speed_mean=8.0, speed_units="px", seed=0
        )
        assert env.contains(positions).all()

    @pytest.mark.parametrize(
        ("duration", "dt", "n_samples"),
        [(0.3, 0.1, 3), (0.29, 0.01, 29), (1.0, 0.01, 100), (0.1, 0.1, 1)],
    )
    def test_sample_count_is_duration_over_dt(
        self, connected_arena_env, duration, dt, n_samples
    ):
        """Rounding error in duration / dt does not drop a sample."""
        positions, times = simulate_trajectory_ou(
            connected_arena_env, duration=duration, dt=dt, speed_units="cm", seed=0
        )
        assert len(times) == len(positions) == n_samples
        np.testing.assert_allclose(times, np.arange(n_samples) * dt)

    def test_duration_shorter_than_dt_is_refused(self, connected_arena_env):
        with pytest.raises(ValueError, match=r"duration=0\.005 s is shorter than"):
            simulate_trajectory_ou(
                connected_arena_env, duration=0.005, dt=0.01, speed_units="cm"
            )

    def test_unknown_boundary_mode_is_refused(self, connected_arena_env):
        with pytest.raises(ValueError, match="boundary_mode must be one of"):
            simulate_trajectory_ou(
                connected_arena_env,
                duration=1.0,
                speed_units="cm",
                boundary_mode="bounce",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("shape", ["circle", "L"])
    def test_periodic_is_refused_where_the_environment_is_not_a_box(self, shape):
        """Wrapping across a circle's or an L's bounding box would leave it."""
        from shapely.geometry import Point, Polygon

        from neurospatial import Environment

        outline = {
            "circle": Point(50, 50).buffer(50),
            "L": Polygon([(0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)]),
        }[shape]
        env = Environment.from_polygon(outline, bin_size=2.0, units="cm")
        with pytest.raises(ValueError, match="not a rectangular box"):
            simulate_trajectory_ou(
                env, duration=1.0, speed_units="cm", boundary_mode="periodic"
            )

    def test_boundary_mode_reflect(self, simple_2d_env):
        """Test reflect boundary mode."""
        positions, _times = simulate_trajectory_ou(
            simple_2d_env,
            duration=5.0,
            boundary_mode="reflect",
            seed=42,
            speed_units="cm",
        )

        # All positions should be in bounds
        for pos in positions:
            assert simple_2d_env.contains(pos)

    @pytest.mark.parametrize("bin_size", [2.0, 3.0])
    def test_boundary_mode_periodic_wraps_across_opposite_sides(self, bin_size):
        """Leaving one side of the box re-enters from the opposite side.

        With 3 cm bins the grid overhangs the 100 cm box by 1 cm on each
        side; the walk wraps at the box's sides, not the grid's.
        """
        from shapely.geometry import box

        from neurospatial import Environment

        env = Environment.from_polygon(
            box(0, 0, 100, 100), bin_size=bin_size, units="cm"
        )
        dt = 0.01
        positions, _times = simulate_trajectory_ou(
            env,
            duration=120.0,
            dt=dt,
            speed_mean=30.0,
            speed_std=0.0,
            boundary_mode="periodic",
            seed=42,
            speed_units="cm",
        )
        assert env.contains(positions).all()
        assert (env.bin_at(positions) >= 0).all()
        span = np.array([100.0, 100.0])
        steps = np.diff(positions, axis=0)
        wrapped = np.abs(steps) > span / 2
        assert wrapped.any()
        # The shortest displacement on the torus is the step taken.
        torus_steps = steps - span * np.round(steps / span)
        np.testing.assert_allclose(
            np.linalg.norm(torus_steps, axis=1), 30.0 * dt, rtol=1e-9
        )

    def test_boundary_mode_stop(self, simple_2d_env):
        """Test stop boundary mode."""
        positions, _times = simulate_trajectory_ou(
            simple_2d_env,
            duration=5.0,
            boundary_mode="stop",
            seed=42,
            speed_units="cm",
        )

        # All positions should be in bounds
        for pos in positions:
            assert simple_2d_env.contains(pos)

    def test_speed_has_the_requested_mean_and_spread(self):
        """Speed is gamma(mean, std) at every time, not a constant."""
        from scipy.stats import gamma
        from shapely.geometry import box

        from neurospatial import Environment

        # A large arena keeps wall contacts rare.
        env = Environment.from_polygon(box(0, 0, 400, 400), bin_size=10.0, units="cm")
        dt = 0.05
        positions, _times = simulate_trajectory_ou(
            env,
            duration=3000.0,
            dt=dt,
            speed_mean=15.0,
            speed_std=8.0,
            speed_units="cm",
            seed=1,
        )
        speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
        # About 2100 independent samples (3000 s over twice the 0.7 s
        # coherence time): the standard error of the mean is about 0.2 cm/s.
        assert speed.mean() == pytest.approx(15.0, abs=0.8)
        assert speed.std() == pytest.approx(8.0, abs=0.8)
        expected = gamma.ppf([0.1, 0.5, 0.9], (15.0 / 8.0) ** 2, scale=8.0**2 / 15.0)
        np.testing.assert_allclose(
            np.percentile(speed, [10, 50, 90]), expected, rtol=0.1
        )

    def test_zero_speed_std_gives_a_constant_speed(self, simple_2d_env):
        positions, _times = simulate_trajectory_ou(
            simple_2d_env,
            duration=20.0,
            dt=0.01,
            speed_mean=10.0,
            speed_std=0.0,
            speed_units="cm",
            seed=2,
        )
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        moving = steps > 0
        assert moving.mean() > 0.99
        np.testing.assert_allclose(steps[moving], 10.0 * 0.01, rtol=1e-9)

    @pytest.mark.parametrize("angle", [0.0, np.pi / 6])
    def test_a_bounce_mirrors_the_velocity_about_the_wall(self, angle):
        """At a wall the component along it is kept and the one into it reversed."""
        from shapely import affinity
        from shapely.geometry import box

        from neurospatial import Environment

        square = affinity.rotate(box(0, 0, 100, 100), angle, use_radians=True)
        env = Environment.from_polygon(square, bin_size=4.0, units="cm")
        # Straight runs at constant speed: every change of step is a bounce.
        positions, _times = simulate_trajectory_ou(
            env,
            duration=200.0,
            dt=0.01,
            speed_mean=20.0,
            speed_std=0.0,
            rotational_velocity_std=0.0,
            speed_units="cm",
            seed=3,
        )
        steps = np.diff(positions, axis=0)
        changed = np.flatnonzero(np.any(~np.isclose(steps[1:], steps[:-1]), axis=1))
        assert len(changed) > 20
        before, after = steps[changed], steps[changed + 1]
        normals = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        # The step after the bounce is the mirror image about one wall.
        mismatch = np.min(
            [
                np.linalg.norm(after - (before - 2 * (before @ n)[:, None] * n), axis=1)
                for n in normals
            ],
            axis=0,
        )
        # Bounces off one wall, not into a corner, are the rule.
        assert np.mean(mismatch < 1e-9) > 0.9
        np.testing.assert_allclose(
            np.linalg.norm(after, axis=1), np.linalg.norm(before, axis=1)
        )

    @pytest.mark.parametrize("kind", ["internal wall", "hole"])
    def test_never_crosses_a_divider(self, kind):
        """A divider splits the arena: the animal goes around its end."""
        import shapely
        from shapely.geometry import LineString, box

        from neurospatial import Environment
        from neurospatial.geometry import OpenArena

        arena = box(0, 0, 100, 100)
        if kind == "internal wall":
            wall = LineString([(52, 0), (52, 60)])  # on a bin edge
            env = Environment.from_polygon(
                arena,
                bin_size=4.0,
                units="cm",
                arena=OpenArena(arena, walls={"divider": wall}),
            )
            divider = env.arena.walls[0]
        else:
            divider = box(48, 0, 52, 60)
            env = Environment.from_polygon(
                arena.difference(divider), bin_size=4.0, units="cm"
            )
        positions, _times = simulate_trajectory_ou(
            env,
            duration=900.0,
            dt=1 / 30,
            speed_mean=15.0,
            speed_std=8.0,
            speed_units="cm",
            seed=2,
        )
        steps = shapely.linestrings(np.stack([positions[:-1], positions[1:]], axis=1))
        assert not shapely.intersects(steps, divider).any()
        # Both sides below the divider's end are visited.
        low = positions[:, 1] < 50
        assert (positions[low, 0] < 48).any()
        assert (positions[low, 0] > 52).any()
        assert (env.bin_at(positions) >= 0).all()

    @pytest.mark.parametrize("boundary_mode", ["reflect", "stop", "periodic"])
    @pytest.mark.parametrize(
        "start",
        [[0, 50], [100, 50], [50, 0], [50, 100], [0, 0], [100, 0], [0, 100]],
        ids=["west", "east", "south", "north", "SW", "SE", "NW"],
    )
    def test_a_walk_starting_on_the_wall_stays_inside(self, start, boundary_mode):
        """A step from a wall point outward is turned back, on a side or a corner.

        The starting heading is random, so about half of the walks first
        head out of the arena.
        """
        from shapely.geometry import box

        from neurospatial import Environment

        env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")
        for seed in range(20):
            positions, _times = simulate_trajectory_ou(
                env,
                duration=0.5,
                start_position=np.array(start, dtype=float),
                speed_units="cm",
                boundary_mode=boundary_mode,
                seed=seed,
            )
            assert env.contains(positions).all(), seed
            if boundary_mode == "reflect":
                # Mirrored off the wall, the first step still moves the animal.
                assert not np.array_equal(positions[1], positions[0]), seed

    @pytest.mark.parametrize("boundary_mode", ["reflect", "stop", "periodic"])
    @pytest.mark.parametrize(
        "start",
        [[-1.0, 50.0], [150.0, 150.0], [np.nan, 50.0], [50.0], [50.0, 50.0, 0.0]],
        ids=["just-outside", "far-outside", "nan", "too-few", "too-many"],
    )
    def test_a_start_outside_the_environment_is_refused(self, start, boundary_mode):
        """A walk from outside would stay outside: the walls keep it there."""
        from shapely.geometry import box

        from neurospatial import Environment

        env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")
        with pytest.raises(
            ValueError, match=r"start_position .* is not a point inside"
        ):
            simulate_trajectory_ou(
                env,
                duration=0.5,
                start_position=np.array(start),
                speed_units="cm",
                boundary_mode=boundary_mode,
                seed=0,
            )

    def test_a_start_outside_a_one_dimensional_environment_is_refused(self):
        from neurospatial import Environment

        env = Environment.from_samples(
            np.linspace(0, 100, 50)[:, None], bin_size=5.0, units="cm"
        )
        with pytest.raises(
            ValueError, match=r"start_position .* is not a point inside"
        ):
            simulate_trajectory_ou(
                env, duration=0.5, start_position=np.array([200.0]), speed_units="cm"
            )

    @pytest.mark.parametrize("boundary_mode", ["reflect", "stop"])
    def test_a_walk_starting_on_a_slanted_wall_stays_inside(self, boundary_mode):
        """A start computed onto a slanted wall may round to either side of it."""
        from shapely import affinity
        from shapely.geometry import box

        from neurospatial import Environment

        square = affinity.rotate(box(0, 0, 100, 100), 30, origin=(0, 0))
        env = Environment.from_polygon(square, bin_size=4.0, units="cm")
        corners = np.asarray(square.exterior.coords)[:4]
        starts = [*corners, *(0.37 * corners + 0.63 * np.roll(corners, -1, axis=0))]
        for start in starts:
            for seed in range(5):
                positions, _times = simulate_trajectory_ou(
                    env,
                    duration=0.5,
                    start_position=start,
                    speed_units="cm",
                    boundary_mode=boundary_mode,
                    seed=seed,
                )
                # The start itself may round outside; no position the
                # animal moves to may.
                moved = np.any(positions != start, axis=1)
                assert env.contains(positions[moved]).all(), (start, seed)

    @pytest.mark.parametrize("boundary_mode", ["reflect", "stop"])
    def test_a_walk_starting_on_a_divider_leaves_it_on_one_side(self, boundary_mode):
        """Either side of an internal wall is free; the animal keeps the side it takes."""
        from shapely.geometry import LineString, box

        from neurospatial import Environment
        from neurospatial.geometry import OpenArena

        arena = box(0, 0, 100, 100)
        env = Environment.from_polygon(
            arena,
            bin_size=4.0,
            units="cm",
            arena=OpenArena(arena, walls={"divider": LineString([(52, 0), (52, 60)])}),
        )
        sides = set()
        for seed in range(20):
            # At most about 10 cm in 1 s: too short to round the divider's end.
            positions, _times = simulate_trajectory_ou(
                env,
                duration=1.0,
                start_position=np.array([52.0, 30.0]),
                speed_units="cm",
                boundary_mode=boundary_mode,
                seed=seed,
            )
            side = np.sign(positions[1:, 0] - 52)
            moved = side != 0
            assert moved.any(), seed
            assert len(set(side[moved])) == 1, seed
            sides.add(side[moved][0])
        assert sides == {-1.0, 1.0}

    def test_no_walk_leaves_the_arena_or_crosses_an_internal_wall(self):
        """Over many walks, including ones starting on walls, nothing gets through."""
        import shapely
        from shapely.geometry import LineString, Polygon

        from neurospatial import Environment
        from neurospatial.geometry import OpenArena

        # A non-convex outline with slanted edges, and a bent divider on bin
        # edges.
        arena = Polygon([(0, 0), (90, 0), (90, 40), (50, 40), (60, 90), (0, 70)])
        divider = LineString([(12, 6), (12, 30), (36, 30)])
        env = Environment.from_polygon(
            arena,
            bin_size=3.0,
            units="cm",
            arena=OpenArena(arena, walls={"divider": divider}),
        )
        rng = np.random.default_rng(7)
        outline = arena.exterior
        on_wall = [
            *shapely.line_interpolate_point(
                outline, rng.uniform(0, 1, 20), normalized=True
            ),
            *shapely.line_interpolate_point(
                divider, rng.uniform(0, 1, 10), normalized=True
            ),
            *[shapely.Point(xy) for xy in outline.coords[:-1]],
        ]
        starts = [None] * 10 + [np.array([p.x, p.y]) for p in on_wall]
        for seed, start in enumerate(starts):
            for boundary_mode in ("reflect", "stop"):
                positions, _times = simulate_trajectory_ou(
                    env,
                    duration=10.0,
                    dt=1 / 30,
                    speed_mean=30.0,
                    speed_std=15.0,
                    start_position=start,
                    speed_units="cm",
                    boundary_mode=boundary_mode,
                    seed=seed,
                )
                moved = np.any(positions != positions[0], axis=1)
                assert env.contains(positions[moved]).all(), (seed, boundary_mode)
                # Only a step leaving the divider from a start on it may touch it.
                steps = shapely.linestrings(
                    np.stack([positions[1:-1], positions[2:]], axis=1)
                )
                assert not shapely.intersects(steps, divider).any(), (
                    seed,
                    boundary_mode,
                )

    def test_reproducibility_with_seed(self, simple_2d_env):
        """Test that same seed produces same trajectory."""
        pos1, times1 = simulate_trajectory_ou(
            simple_2d_env,
            duration=1.0,
            seed=42,
            speed_units="cm",
        )
        pos2, times2 = simulate_trajectory_ou(
            simple_2d_env,
            duration=1.0,
            seed=42,
            speed_units="cm",
        )

        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(times1, times2)

    def test_different_seeds_produce_different_trajectories(self, simple_2d_env):
        """Test that different seeds produce different trajectories."""
        pos1, _ = simulate_trajectory_ou(
            simple_2d_env, duration=1.0, seed=42, speed_units="cm"
        )
        pos2, _ = simulate_trajectory_ou(
            simple_2d_env, duration=1.0, seed=43, speed_units="cm"
        )

        assert not np.array_equal(pos1, pos2)

    def test_requires_env_units(self, simple_2d_env):
        """Test that ValueError is raised if env.units is not set."""
        # Remove units
        simple_2d_env = simple_2d_env.with_metadata(units=None)

        with pytest.raises(ValueError, match="units must be set"):
            simulate_trajectory_ou(simple_2d_env, duration=1.0, speed_units="cm")

    def test_speed_units_must_match_env_units(self, simple_2d_env):
        """speed_units must match env.units exactly.

        Auto-conversion between unit families was removed in v0.4
        because silent rescaling was easy to miss. Callers must
        pre-convert if their speeds are in a different unit.
        """
        assert simple_2d_env.units == "cm"

        # Mismatched units must raise.
        with pytest.raises(ValueError, match=r"speed_units .* must match env\.units"):
            simulate_trajectory_ou(
                simple_2d_env,
                duration=1.0,
                speed_mean=0.10,
                speed_units="m",
                seed=42,
            )

        # Matching units works.
        positions, _times = simulate_trajectory_ou(
            simple_2d_env,
            duration=1.0,
            speed_mean=10.0,  # cm/s
            speed_units="cm",
            seed=42,
        )
        assert len(positions) > 0


class TestSimulateTrajectorysinusoidal:
    """Tests for sinusoidal trajectory simulation."""

    def test_requires_1d_environment(self, simple_2d_env):
        """Test that ValueError is raised for non-1D environment."""
        with pytest.raises(ValueError, match="1D environments"):
            simulate_trajectory_sinusoidal(simple_2d_env, duration=10.0)

    def test_basic_sinusoidal_generation(self, linear_track_env):
        """Produces (n_time, n_dims) positions sampled at the requested rate."""
        positions, times = simulate_trajectory_sinusoidal(
            linear_track_env,
            duration=10.0,
            sampling_frequency=100.0,
            speed=20.0,
            period=4.0,
        )
        assert positions.shape == (1000, 1)  # 10s * 100Hz
        assert times.shape == (1000,)
        assert times[1] - times[0] == pytest.approx(0.01)

    def test_stays_within_track_bounds(self, linear_track_env):
        """Sinusoidal positions never leave the linearized track range."""
        range_min, range_max = linear_track_env.dimension_ranges[0]
        positions, _ = simulate_trajectory_sinusoidal(
            linear_track_env,
            duration=20.0,
            sampling_frequency=100.0,
            speed=20.0,
            period=4.0,
        )
        assert positions.min() >= range_min - 1e-9
        assert positions.max() <= range_max + 1e-9

    def test_is_periodic_with_given_period(self, linear_track_env):
        """With pauses off, x(t) repeats every `period` seconds."""
        period = 4.0
        positions, _times = simulate_trajectory_sinusoidal(
            linear_track_env,
            duration=12.0,
            sampling_frequency=100.0,
            speed=20.0,
            period=period,
            pause_at_peaks=False,
        )
        n_per_period = int(period * 100.0)
        # Compare the first period to the second period sample-for-sample.
        first = positions[:n_per_period, 0]
        second = positions[n_per_period : 2 * n_per_period, 0]
        np.testing.assert_allclose(first, second, atol=1e-6)

    def test_runs_the_whole_length_of_a_bent_track_in_its_coordinates(self):
        """On a 120 + 80 cm L-shaped track the animal runs all 200 cm in (x, y).

        Positions are environment coordinates on the centerline, the same
        coordinates place cells and rate maps use.
        """
        from neurospatial import Environment

        track = Environment.linear_track(
            node_positions=[(0.0, 0.0), (120.0, 0.0), (120.0, 80.0)],
            bin_size=2.0,
            track_width=10.0,
            units="cm",
        )
        positions, _ = simulate_trajectory_sinusoidal(
            track,
            duration=8.0,
            sampling_frequency=100.0,
            speed=20.0,
            period=8.0,
            pause_at_peaks=False,
        )
        assert positions.shape == (800, 2)
        assert np.all(track.bin_at(positions) >= 0)
        np.testing.assert_allclose(
            track.linear_to_nd(track.to_linear(positions)), positions, atol=1e-9
        )
        linear = track.to_linear(positions)
        assert linear.min() == pytest.approx(0.0, abs=1e-3)
        assert linear.max() == pytest.approx(200.0, abs=1e-3)
        # The default period runs the track there and back at the speed.
        positions, times = simulate_trajectory_sinusoidal(
            track, duration=1.0, speed=20.0
        )
        np.testing.assert_allclose(
            track.to_linear(positions),
            100.0 + 100.0 * np.sin(2 * np.pi * times / 20.0),
            atol=1e-9,
        )

    def test_runs_between_the_far_ends_of_a_branched_track(self):
        """On a T-maze it runs stem to arm continuously, never through a gap.

        The linear coordinates of a T-maze jump between arms and leave gaps
        between segments; the animal moves on the track instead.
        """
        import networkx as nx

        from neurospatial import Environment

        graph = nx.Graph()
        graph.add_node(0, pos=(0.0, 0.0))
        graph.add_node(1, pos=(0.0, 40.0))
        graph.add_node(2, pos=(-30.0, 40.0))
        graph.add_node(3, pos=(30.0, 40.0))
        graph.add_edges_from([(0, 1), (1, 2), (1, 3)])
        track = Environment.from_graph(
            graph,
            edge_order=[(0, 1), (1, 2), (1, 3)],
            edge_spacing=10.0,
            bin_size=2.0,
            track_width=4.0,
            units="cm",
        )
        sampling_frequency, speed = 100.0, 20.0
        positions, _ = simulate_trajectory_sinusoidal(
            track, duration=10.0, sampling_frequency=sampling_frequency, speed=speed
        )
        assert np.all(np.isfinite(positions))
        assert np.all(track.bin_at(positions) >= 0)
        # The fastest a sinusoid with mean speed v moves is pi / 2 * v.
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        assert steps.max() <= np.pi / 2 * speed / sampling_frequency * (1 + 1e-6)
        # It reaches the stem's foot and the end of one arm, 70 cm apart
        # along the track: the farthest apart two points of the maze are.
        assert positions[:, 1].min() == pytest.approx(0.0, abs=1e-6)
        assert np.abs(positions[:, 0]).max() == pytest.approx(30.0, abs=1e-6)

    def test_runs_the_whole_loop_of_a_circular_track(self):
        """On a 12-sided loop of radius 50 cm it runs all 311 cm and back.

        The loop is cut where its linear position starts; the animal runs
        once around it, turns, and runs back, visiting every bin.
        """
        import networkx as nx

        from neurospatial import Environment

        n_nodes, radius = 12, 50.0
        angles = 2 * np.pi * np.arange(n_nodes) / n_nodes
        graph = nx.cycle_graph(n_nodes)
        for node, angle in enumerate(angles):
            graph.nodes[node]["pos"] = (radius * np.cos(angle), radius * np.sin(angle))
        track = Environment.from_graph(
            graph,
            edge_order=[(i, (i + 1) % n_nodes) for i in range(n_nodes)],
            edge_spacing=0.0,
            bin_size=2.0,
            track_width=5.0,
            units="cm",
        )
        loop_length = n_nodes * 2 * radius * np.sin(np.pi / n_nodes)
        speed = 20.0
        positions, times = simulate_trajectory_sinusoidal(
            track, duration=2 * loop_length / speed, speed=speed, pause_at_peaks=False
        )
        assert np.all(track.bin_at(positions) >= 0)
        assert np.unique(track.bin_at(positions)).size == track.n_bins
        # Distance along the loop from the cut at (50, 0), counted in edge
        # order, follows the sinusoid over the whole loop length.
        period = 2 * loop_length / speed
        distance = loop_length / 2 * (1 + np.sin(2 * np.pi * times / period))
        away_from_cut = (distance > 1.0) & (distance < loop_length - 1.0)
        np.testing.assert_allclose(
            track.to_linear(positions[away_from_cut]),
            distance[away_from_cut],
            atol=1e-6,
        )
        # It turns at the cut: a quarter and three quarters of a period in.
        for turn in (period / 4, 3 * period / 4):
            index = np.argmin(np.abs(times - turn))
            np.testing.assert_allclose(positions[index], [radius, 0.0], atol=1e-3)

    @pytest.mark.parametrize("maze", ["hamlet", "crossword"])
    def test_a_track_with_several_loops_is_refused(self, maze):
        """No back-and-forth route covers a maze of several loops."""
        from neurospatial.simulation.mazes import (
            make_crossword_maze,
            make_hamlet_maze,
        )

        make = {"hamlet": make_hamlet_maze, "crossword": make_crossword_maze}[maze]
        track = make().env_track
        with pytest.raises(ValueError, match=r"outbound_path="):
            simulate_trajectory_sinusoidal(track, duration=1.0)

    @pytest.mark.parametrize("first_piece", ["loop", "segment"])
    def test_a_track_in_separate_pieces_is_refused(self, first_piece):
        """No back-and-forth route joins pieces of track that do not touch.

        A straight piece beside a loop or beside another straight piece:
        the run used to cover one piece, or call the loop one with arms.
        """
        import networkx as nx

        from neurospatial import Environment

        graph = nx.Graph()
        if first_piece == "loop":
            angles = 2 * np.pi * np.arange(6) / 6
            for node, angle in enumerate(angles):
                graph.add_node(node, pos=(50 * np.cos(angle), 50 * np.sin(angle)))
            edges = [(node, (node + 1) % 6) for node in range(6)]
        else:
            graph.add_node(0, pos=(0.0, 0.0))
            graph.add_node(1, pos=(50.0, 0.0))
            edges = [(0, 1)]
        graph.add_node(10, pos=(200.0, 0.0))
        graph.add_node(11, pos=(300.0, 0.0))
        edges.append((10, 11))
        graph.add_edges_from(edges)
        track = Environment.from_graph(
            graph,
            edge_order=edges,
            edge_spacing=0.0,
            bin_size=2.0,
            track_width=5.0,
            units="cm",
        )
        with pytest.raises(ValueError, match=r"2 separate pieces\.\nWhy:.*\nFix:"):
            simulate_trajectory_sinusoidal(track, duration=1.0)

    def test_pauses_hold_at_each_end_for_pause_duration(self, linear_track_env):
        """The animal reaches both ends of a 100 cm track and waits 0.5 s at each."""
        sampling_frequency, pause_duration = 100.0, 0.5
        positions, times = simulate_trajectory_sinusoidal(
            linear_track_env,
            duration=60.0,
            sampling_frequency=sampling_frequency,
            speed=10.0,
            pause_duration=pause_duration,
        )
        np.testing.assert_allclose(np.diff(times), 1 / sampling_frequency)
        x = positions[:, 0]
        for end in (0.0, 100.0):
            at_end = np.isclose(x, end, rtol=0, atol=1e-9)
            # Each visit is one run of samples at the end, lasting the pause.
            edges = np.flatnonzero(np.diff(np.r_[0, at_end.astype(int), 0]))
            starts, stops = edges[::2], edges[1::2]
            assert len(starts) >= 2
            held = (stops - 1 - starts) / sampling_frequency
            np.testing.assert_allclose(
                held, pause_duration, atol=1 / sampling_frequency
            )
        # Nowhere else does the animal stand still.
        still = np.diff(x) == 0
        at_an_end = np.isclose(x[:-1], 0.0, atol=1e-9) | np.isclose(
            x[:-1], 100.0, atol=1e-9
        )
        assert not np.any(still & ~at_an_end)
        # One cycle: 20 s of running there and back, plus two pauses.
        crossings = np.flatnonzero(np.diff((x >= 50.0).astype(int)) > 0)
        np.testing.assert_allclose(
            np.diff(crossings) / sampling_frequency, 21.0, atol=0.02
        )

    def test_without_pauses_the_ends_are_not_held(self, linear_track_env):
        positions, _ = simulate_trajectory_sinusoidal(
            linear_track_env,
            duration=30.0,
            sampling_frequency=100.0,
            pause_duration=0.5,
            pause_at_peaks=False,
        )
        assert np.all(np.diff(positions[:, 0]) != 0)

    def test_default_speed_follows_environment_units(self, linear_track_env):
        """10 cm/s by default: a track in meters takes as long as one in cm."""
        import networkx as nx

        from neurospatial import Environment

        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))
        graph.add_node(1, pos=(0.5,))
        graph.add_node(2, pos=(1.0,))
        graph.add_edge(0, 1, distance=0.5)
        graph.add_edge(1, 2, distance=0.5)
        track_m = Environment.from_graph(
            graph=graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=0.0,
            bin_size=0.02,
            units="m",
        )
        positions_cm, times_cm = simulate_trajectory_sinusoidal(
            linear_track_env, duration=30.0, sampling_frequency=50.0
        )
        positions_m, times_m = simulate_trajectory_sinusoidal(
            track_m, duration=30.0, sampling_frequency=50.0
        )
        np.testing.assert_allclose(times_m, times_cm)
        np.testing.assert_allclose(positions_m, positions_cm / 100, atol=1e-12)

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("duration", -5.0),
            ("duration", 0.0),
            ("duration", np.nan),
            ("sampling_frequency", 0.0),
            ("sampling_frequency", np.inf),
            ("speed", 0.0),
            ("speed", -10.0),
            ("speed", np.nan),
            ("period", 0.0),
            ("pause_duration", -1.0),
            ("pause_duration", np.nan),
        ],
    )
    def test_invalid_timing_is_refused(self, linear_track_env, option, value):
        options = {"duration": 10.0, option: value}
        with pytest.raises(ValueError, match=f"{option} must be finite"):
            simulate_trajectory_sinusoidal(linear_track_env, **options)

    def test_duration_shorter_than_one_sample_is_refused(self, linear_track_env):
        with pytest.raises(ValueError, match="shorter than one sample"):
            simulate_trajectory_sinusoidal(
                linear_track_env, duration=0.001, sampling_frequency=100.0
            )


class TestSimulateTrajectoryLaps:
    """Tests for lap-based trajectory simulation."""

    def test_n_laps_produces_correct_number_of_laps(self, connected_arena_env):
        """Test that n_laps parameter produces correct number of laps."""
        n_laps = 5
        positions, _times, metadata = simulate_trajectory_laps(
            connected_arena_env,
            n_laps=n_laps,
            speed_mean=10.0,
            pause_duration=0.0,
            return_metadata=True,
            seed=42,
        )

        # Check that metadata contains lap_ids
        assert "lap_ids" in metadata, "metadata should contain 'lap_ids'"

        # Check that lap_ids has the same length as positions
        assert len(metadata["lap_ids"]) == len(positions), (
            "lap_ids should have same length as positions"
        )

        # Count unique laps (0-indexed, so n_laps should give lap_ids 0 to n_laps-1)
        unique_laps = np.unique(metadata["lap_ids"])
        assert len(unique_laps) == n_laps, (
            f"Expected {n_laps} laps, got {len(unique_laps)}"
        )
        assert np.all(unique_laps == np.arange(n_laps)), (
            "lap_ids should be 0 to n_laps-1"
        )

    def test_metadata_contains_lap_ids_and_directions(self, connected_arena_env):
        """Test that metadata contains lap_ids and direction arrays."""
        positions, _times, metadata = simulate_trajectory_laps(
            connected_arena_env,
            n_laps=3,
            return_metadata=True,
            seed=42,
        )

        # Check required metadata keys
        assert "lap_ids" in metadata, "metadata should contain 'lap_ids'"
        assert "direction" in metadata, "metadata should contain 'direction'"
        assert "lap_boundaries" in metadata, "metadata should contain 'lap_boundaries'"

        # Check shapes
        assert len(metadata["lap_ids"]) == len(positions), (
            "lap_ids should have same length as positions"
        )
        assert len(metadata["direction"]) == len(positions), (
            "direction should have same length as positions"
        )

        # Running samples are outbound or inbound; pause samples are neither
        # and belong to no lap.
        assert set(np.unique(metadata["direction"])) == {"outbound", "inbound", "pause"}
        paused = metadata["direction"] == "pause"
        np.testing.assert_array_equal(metadata["lap_ids"][paused], -1)
        assert np.all(metadata["lap_ids"][~paused] >= 0)

    def test_pauses_at_lap_ends(self, connected_arena_env):
        """Test that trajectory includes pauses at lap ends."""
        pause_duration = 0.5  # seconds
        sampling_frequency = 100.0  # Hz
        expected_pause_samples = int(pause_duration * sampling_frequency)

        positions, _times, metadata = simulate_trajectory_laps(
            connected_arena_env,
            n_laps=3,
            pause_duration=pause_duration,
            sampling_frequency=sampling_frequency,
            return_metadata=True,
            seed=42,
        )

        # Check lap boundaries
        lap_boundaries = metadata["lap_boundaries"]

        # At each lap boundary (except first), check for pauses
        # Pause is indicated by zero velocity (position not changing)
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        for i in range(1, len(lap_boundaries)):
            boundary_idx = lap_boundaries[i]
            if boundary_idx > 0 and boundary_idx < len(velocities):
                # Check samples around boundary for low velocity (paused)
                # Allow some tolerance for numerical issues
                start_idx = max(0, boundary_idx - expected_pause_samples // 2)
                end_idx = min(
                    len(velocities), boundary_idx + expected_pause_samples // 2
                )

                # At least some samples should have very low velocity (paused)
                paused_samples = velocities[start_idx:end_idx] < 0.1
                assert np.sum(paused_samples) > 0, (
                    f"Expected pause at lap boundary {i}, but found no low-velocity samples"
                )

    def test_return_metadata_false_returns_tuple_of_two(self, connected_arena_env):
        """Test that return_metadata=False returns only (positions, times)."""
        result = simulate_trajectory_laps(
            connected_arena_env,
            n_laps=3,
            return_metadata=False,
            seed=42,
        )

        # Should return tuple of 2 elements
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, (
            "Should return (positions, times) when return_metadata=False"
        )

        positions, times = result
        assert len(positions) == len(times), (
            "positions and times should have same length"
        )

    def test_custom_paths(self, simple_2d_env):
        """Test that custom outbound and inbound paths are used."""
        import networkx as nx

        # Custom paths are chains of neighboring bins.
        outbound_path = nx.shortest_path(simple_2d_env.connectivity, 0, 9)
        inbound_path = list(reversed(outbound_path))

        positions, times, metadata = simulate_trajectory_laps(
            simple_2d_env,
            n_laps=2,
            outbound_path=outbound_path,
            inbound_path=inbound_path,
            return_metadata=True,
            seed=42,
        )

        # Check that trajectory was generated
        assert len(positions) > 0, "Should generate non-empty trajectory"
        assert len(times) > 0, "Should generate non-empty times"

        # Check metadata structure
        assert "lap_ids" in metadata
        assert "direction" in metadata

    def test_reproducibility_with_seed(self, connected_arena_env):
        """Test that same seed produces same trajectory."""
        result1 = simulate_trajectory_laps(
            connected_arena_env, n_laps=3, seed=42, return_metadata=True
        )
        result2 = simulate_trajectory_laps(
            connected_arena_env, n_laps=3, seed=42, return_metadata=True
        )

        positions1, times1, metadata1 = result1
        positions2, times2, metadata2 = result2

        np.testing.assert_array_equal(positions1, positions2)
        np.testing.assert_array_equal(times1, times2)
        np.testing.assert_array_equal(metadata1["lap_ids"], metadata2["lap_ids"])

    def test_trajectory_stays_in_environment(self, connected_arena_env):
        """Test that all positions are within environment bounds."""
        positions, _times = simulate_trajectory_laps(
            connected_arena_env,
            n_laps=5,
            return_metadata=False,
            seed=42,
        )

        # Use nearest_bin to check - this is more efficient than calling
        # contains() in a loop, and it validates that positions can be mapped

        bin_indices = connected_arena_env.nearest_bin(positions)

        # All bin indices should be valid (>= 0)
        assert np.all(bin_indices >= 0), "Some positions could not be mapped to bins"

        # Spot check a few positions with contains()
        sample_indices = np.linspace(
            0, len(positions) - 1, min(10, len(positions)), dtype=int
        )
        for idx in sample_indices:
            pos = positions[idx]
            assert connected_arena_env.contains(pos), (
                f"Position {pos} at index {idx} is outside environment bounds"
            )
