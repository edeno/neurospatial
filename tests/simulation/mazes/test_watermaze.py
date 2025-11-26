"""Tests for Morris Water Maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestWatermazeDims:
    """Tests for WatermazeDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """WatermazeDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.watermaze import WatermazeDims

        dims = WatermazeDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """WatermazeDims should have correct default values."""
        from neurospatial.simulation.mazes.watermaze import WatermazeDims

        dims = WatermazeDims()
        assert dims.pool_diameter == 150.0
        assert dims.platform_radius == 5.0

    def test_is_frozen(self):
        """WatermazeDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.watermaze import WatermazeDims

        dims = WatermazeDims()
        with pytest.raises(FrozenInstanceError):
            dims.pool_diameter = 200.0  # type: ignore[misc]

    def test_custom_values(self):
        """WatermazeDims should accept custom values."""
        from neurospatial.simulation.mazes.watermaze import WatermazeDims

        dims = WatermazeDims(pool_diameter=200.0, platform_radius=10.0)
        assert dims.pool_diameter == 200.0
        assert dims.platform_radius == 10.0


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
        assert maze.env_2d._is_fitted

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

    def test_platform_region_is_point(self):
        """Platform region should be a point region."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        maze = make_watermaze()
        platform = maze.env_2d.regions["platform"]
        assert platform.kind == "point"

    def test_platform_default_position(self):
        """Platform should default to center of NE quadrant."""
        from neurospatial.simulation.mazes.watermaze import (
            WatermazeDims,
            make_watermaze,
        )

        dims = WatermazeDims(pool_diameter=150.0)
        maze = make_watermaze(dims=dims)

        platform = maze.env_2d.regions["platform"]
        # NE quadrant center: (pool_diameter/4, pool_diameter/4)
        expected_x = dims.pool_diameter / 4
        expected_y = dims.pool_diameter / 4

        assert np.isclose(platform.data[0], expected_x, atol=1.0)
        assert np.isclose(platform.data[1], expected_y, atol=1.0)

    def test_custom_platform_position(self):
        """Custom platform_position should be respected."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        custom_position = (20.0, 30.0)
        maze = make_watermaze(platform_position=custom_position)

        platform = maze.env_2d.regions["platform"]
        assert np.isclose(platform.data[0], custom_position[0], atol=0.1)
        assert np.isclose(platform.data[1], custom_position[1], atol=0.1)

    def test_env_2d_has_quadrant_regions(self):
        """env_2d should have NE, NW, SE, SW quadrant regions."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        maze = make_watermaze()
        assert "NE" in maze.env_2d.regions
        assert "NW" in maze.env_2d.regions
        assert "SE" in maze.env_2d.regions
        assert "SW" in maze.env_2d.regions

    def test_quadrant_regions_are_points(self):
        """Quadrant regions should be point regions."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        maze = make_watermaze()
        assert maze.env_2d.regions["NE"].kind == "point"
        assert maze.env_2d.regions["NW"].kind == "point"
        assert maze.env_2d.regions["SE"].kind == "point"
        assert maze.env_2d.regions["SW"].kind == "point"

    def test_quadrant_positions_correct(self):
        """Quadrant regions should be at correct positions."""
        from neurospatial.simulation.mazes.watermaze import (
            WatermazeDims,
            make_watermaze,
        )

        dims = WatermazeDims(pool_diameter=150.0)
        maze = make_watermaze(dims=dims)

        # Quadrant centers at radius/2 from origin
        r = dims.pool_diameter / 2
        half_r = r / 2

        ne = maze.env_2d.regions["NE"]
        nw = maze.env_2d.regions["NW"]
        se = maze.env_2d.regions["SE"]
        sw = maze.env_2d.regions["SW"]

        # NE: (+r/2, +r/2)
        assert np.isclose(ne.data[0], half_r, atol=1.0)
        assert np.isclose(ne.data[1], half_r, atol=1.0)

        # NW: (-r/2, +r/2)
        assert np.isclose(nw.data[0], -half_r, atol=1.0)
        assert np.isclose(nw.data[1], half_r, atol=1.0)

        # SE: (+r/2, -r/2)
        assert np.isclose(se.data[0], half_r, atol=1.0)
        assert np.isclose(se.data[1], -half_r, atol=1.0)

        # SW: (-r/2, -r/2)
        assert np.isclose(sw.data[0], -half_r, atol=1.0)
        assert np.isclose(sw.data[1], -half_r, atol=1.0)

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


class TestWatermazeDocstrings:
    """Tests for docstrings and examples."""

    def test_make_watermaze_has_docstring(self):
        """make_watermaze should have a docstring."""
        from neurospatial.simulation.mazes.watermaze import make_watermaze

        assert make_watermaze.__doc__ is not None
        assert len(make_watermaze.__doc__) > 100  # Reasonable length

    def test_watermaze_dims_has_docstring(self):
        """WatermazeDims should have a docstring."""
        from neurospatial.simulation.mazes.watermaze import WatermazeDims

        assert WatermazeDims.__doc__ is not None
