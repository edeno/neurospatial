"""Tests for Morris Water Maze implementation."""

from __future__ import annotations

import numpy as np
import pytest


class TestMakeWatermaze:
    """Tests for make_watermaze factory function."""

    def test_returns_maze_environments(self):
        """make_watermaze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        maze = make_watermaze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        maze = make_watermaze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0

    def test_env_2d_connectivity_is_connected(self):
        """The open-field 2D environment must be a single connected component."""
        import networkx as nx

        from neurospatial.simulation.mazes.watermaze import make_watermaze

        maze = make_watermaze()
        assert nx.is_connected(maze.env_2d.connectivity)

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        maze = make_watermaze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_circular_extent(self):
        """env_2d should have a circular spatial extent."""
        from neurospatial.simulation.mazes.watermaze import (
            WatermazeDims,
            make_watermaze,
        )

        dims = WatermazeDims(pool_diameter=150.0)
        maze = make_watermaze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # Both extents should be approximately the pool diameter
        x_extent = x_max - x_min
        y_extent = y_max - y_min
        expected_extent = dims.pool_diameter

        # Allow some tolerance due to discretization
        assert x_extent > expected_extent * 0.8  # At least 80% of expected
        assert y_extent > expected_extent * 0.8  # At least 80% of expected

        # Both extents should be similar (circular)
        assert abs(x_extent - y_extent) < expected_extent * 0.2  # Within 20%

    def test_env_2d_has_platform_region(self):
        """env_2d should have a 'platform' region."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        maze = make_watermaze()
        assert "platform" in maze.env_2d.regions

    def test_platform_is_a_landmark_with_its_real_extent(self):
        """The hidden platform: a landmark and a 15 cm circular region."""
        from shapely.geometry import Point

        from neurospatial.simulation.mazes.watermaze import (
            WatermazeDims,
            make_watermaze,
        )

        dims = WatermazeDims()
        env = make_watermaze(dims=dims).env_2d
        assert list(env.landmarks) == ["platform"]
        platform = env.landmarks["platform"]
        assert platform.kind == "point"
        assert dict(platform.metadata) == {"role": "platform"}

        region = env.regions["platform"]
        assert region.kind == "polygon"
        disc = Point(platform.data).buffer(dims.platform_radius, quad_segs=256)
        assert region.data.symmetric_difference(disc).area < 0.01 * disc.area
        np.testing.assert_allclose(region.data.centroid.coords[0], platform.data)
        # Every bin centre within the platform, and only those.
        inside = np.hypot(*(env.bin_centers - platform.data).T) < 7.0
        assert set(np.flatnonzero(inside)) <= set(env.bins_in_region("platform"))
        assert len(env.bins_in_region("platform")) == pytest.approx(
            np.pi * 7.5**2 / 2.0**2, rel=0.15
        )

    def test_platform_default_position(self):
        """Platform should default to center of NE quadrant."""
        from neurospatial.simulation.mazes.watermaze import (
            WatermazeDims,
            make_watermaze,
        )

        dims = WatermazeDims(pool_diameter=150.0)
        maze = make_watermaze(dims=dims)

        platform = maze.env_2d.landmarks["platform"]
        # Halfway between the pool centre and the wall, on the NE diagonal.
        r = dims.pool_diameter / 2
        x, y = platform.data
        assert np.hypot(x, y) == pytest.approx(r / 2)
        assert np.degrees(np.arctan2(y, x)) == pytest.approx(45.0)

    def test_platform_lies_wholly_inside_the_pool(self):
        """The 15 cm platform at the NE quadrant centre is inside the pool."""
        from shapely.geometry import Point

        from neurospatial.simulation.mazes.watermaze import (
            WatermazeDims,
            make_watermaze,
        )

        dims = WatermazeDims()
        env = make_watermaze(dims=dims).env_2d
        platform = Point(env.landmarks["platform"].data).buffer(dims.platform_radius)
        assert env.arena.polygon.contains(platform)
        assert env.arena.polygon.contains(env.regions["platform"].data)

    def test_default_sizes_are_in_the_published_ranges(self):
        """Morris (2008): pools 1.5-2 m; platforms 10-15 cm, 1/50-1/100 of area."""
        from neurospatial.simulation.mazes.watermaze import WatermazeDims

        dims = WatermazeDims()
        assert dims.pool_diameter == 150.0
        assert 2 * dims.platform_radius == 15.0
        area_ratio = (2 * dims.platform_radius / dims.pool_diameter) ** 2
        assert 1 / 100 <= area_ratio <= 1 / 50

    def test_custom_platform_position(self):
        """Custom platform_position should be respected."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        custom_position = (20.0, 30.0)
        maze = make_watermaze(platform_position=custom_position)

        platform = maze.env_2d.landmarks["platform"]
        np.testing.assert_array_equal(platform.data, custom_position)
        centroid = maze.env_2d.regions["platform"].data.centroid
        np.testing.assert_allclose(centroid.coords[0], custom_position, atol=1e-9)

    @pytest.mark.parametrize(
        "position", [(100.0, 0.0), (70.0, 0.0), (0.0, -68.0), (np.nan, 0.0)]
    )
    def test_platform_outside_the_pool_is_refused(self, position):
        """A platform that does not fit in the 75 cm pool would put the
        landmark off the floor and leave the region with no bins."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        with pytest.raises(ValueError, match="platform_position"):
            make_watermaze(platform_position=position)

    def test_platform_touching_the_wall_from_inside_is_accepted(self):
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        maze = make_watermaze(platform_position=(67.5, 0.0))
        assert len(maze.env_2d.bins_in_region("platform")) > 0

    def test_platform_too_large_for_the_default_position_is_refused(self):
        from neurospatial.simulation.mazes.watermaze import (
            WatermazeDims,
            make_watermaze,
        )

        with pytest.raises(ValueError, match="does not fit in the pool"):
            make_watermaze(WatermazeDims(platform_radius=40.0))

    def test_env_2d_has_quadrant_regions(self):
        """env_2d should have NE, NW, SE, SW quadrant regions."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        maze = make_watermaze()
        assert "NE" in maze.env_2d.regions
        assert "NW" in maze.env_2d.regions
        assert "SE" in maze.env_2d.regions
        assert "SW" in maze.env_2d.regions

    def test_quadrants_are_quarter_discs_partitioning_the_pool(self):
        """NE, NW, SE, SW are the pool's quarters, for quadrant occupancy."""
        import shapely

        from neurospatial.simulation.mazes.watermaze import make_watermaze

        env = make_watermaze().env_2d
        pool = env.arena.polygon
        quadrants = {q: env.regions[q] for q in ("NE", "NW", "SE", "SW")}
        assert all(r.kind == "polygon" for r in quadrants.values())
        for region in quadrants.values():
            assert region.data.area == pytest.approx(pool.area / 4)
        union = shapely.union_all([r.data for r in quadrants.values()])
        assert union.symmetric_difference(pool).area == pytest.approx(0.0, abs=1e-6)
        for a in quadrants.values():
            for b in quadrants.values():
                if a is not b:
                    assert a.data.intersection(b.data).area == pytest.approx(0.0)
        # The quadrants are not landmarks: they are analysis zones.
        assert not set(quadrants) & set(env.landmarks)

    def test_quadrant_compass_directions(self):
        """North is +y and east is +x; each quarter lies on its own side."""
        from neurospatial.simulation.mazes.watermaze import (
            WatermazeDims,
            make_watermaze,
        )

        dims = WatermazeDims(pool_diameter=150.0)
        env = make_watermaze(dims=dims).env_2d
        r = dims.pool_diameter / 2
        signs = {"NE": (1, 1), "NW": (-1, 1), "SE": (1, -1), "SW": (-1, -1)}
        for name, (sx, sy) in signs.items():
            x0, y0, x1, y1 = env.regions[name].data.bounds
            assert (min(0, sx * r), min(0, sy * r)) == pytest.approx((x0, y0)), name
            assert (max(0, sx * r), max(0, sy * r)) == pytest.approx((x1, y1)), name
            # A quarter disc's centroid lies 4R / (3 pi) along each axis.
            centroid = env.regions[name].data.centroid
            expected = np.array([sx, sy]) * 4 * r / (3 * np.pi)
            np.testing.assert_allclose(centroid.coords[0], expected, rtol=1e-2)

        # The default platform lies in the NE quadrant.
        platform = env.regions["platform"].data
        assert env.regions["NE"].data.contains(platform)

    def test_quadrants_cover_every_bin(self):
        """Every bin lies in some quadrant; off-axis bins lie in exactly one."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        env = make_watermaze().env_2d
        counts = np.zeros(env.n_bins, dtype=int)
        for q in ("NE", "NW", "SE", "SW"):
            counts[env.bins_in_region(q)] += 1
        assert (counts >= 1).all()
        on_axis = np.isclose(env.bin_centers, 0.0).any(axis=1)
        assert (counts[~on_axis] == 1).all()

    def test_no_start_or_cue_landmarks(self):
        """Morris (2008) gives no fixed start positions or cue placement."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        roles = {
            lm.metadata["role"] for lm in make_watermaze().env_2d.landmarks.values()
        }
        assert roles == {"platform"}

    def test_env_track_is_none(self):
        """env_track should be None (open field, no track)."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        maze = make_watermaze()
        assert maze.env_track is None

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.watermaze import (
            WatermazeDims,
            make_watermaze,
        )

        dims = WatermazeDims()

        maze_fine = make_watermaze(dims=dims, bin_size=1.0)
        maze_coarse = make_watermaze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.watermaze import (
            WatermazeDims,
            make_watermaze,
        )

        dims = WatermazeDims(pool_diameter=200.0)
        maze = make_watermaze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Both should be roughly 200 cm (pool diameter)
        assert x_extent > 160  # Larger than default 150
        assert y_extent > 160  # Larger than default 150

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.watermaze import (
            WatermazeDims,
            make_watermaze,
        )

        maze_default = make_watermaze(dims=None)
        maze_explicit = make_watermaze(dims=WatermazeDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins

    def test_pool_centered_at_origin(self):
        """Pool should be centered at the origin."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        maze = make_watermaze()

        # Get bin centers
        bin_centers = maze.env_2d.bin_centers
        x_mean = bin_centers[:, 0].mean()
        y_mean = bin_centers[:, 1].mean()

        # Should be centered near origin (0, 0)
        assert abs(x_mean) < 10  # Within 10 cm
        assert abs(y_mean) < 10  # Within 10 cm
