"""Tests for Cheeseboard Maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestCheeseboardDims:
    """Tests for CheeseboardDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """CheeseboardDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.cheeseboard import CheeseboardDims

        dims = CheeseboardDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """CheeseboardDims should have correct default values."""
        from neurospatial.simulation.mazes.cheeseboard import CheeseboardDims

        dims = CheeseboardDims()
        assert dims.diameter == 110.0
        assert dims.grid_spacing == 9.0
        assert dims.reward_radius == 1.5

    def test_is_frozen(self):
        """CheeseboardDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.cheeseboard import CheeseboardDims

        dims = CheeseboardDims()
        with pytest.raises(FrozenInstanceError):
            dims.diameter = 120.0  # type: ignore[misc]

    def test_custom_values(self):
        """CheeseboardDims should accept custom values."""
        from neurospatial.simulation.mazes.cheeseboard import CheeseboardDims

        dims = CheeseboardDims(diameter=120.0, grid_spacing=10.0, reward_radius=2.0)
        assert dims.diameter == 120.0
        assert dims.grid_spacing == 10.0
        assert dims.reward_radius == 2.0


class TestMakeCheeseboardMaze:
    """Tests for make_cheeseboard_maze factory function."""

    def test_returns_maze_environments(self):
        """make_cheeseboard_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        maze = make_cheeseboard_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        maze = make_cheeseboard_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        maze = make_cheeseboard_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_is_circular(self):
        """env_2d should have a circular platform shape."""
        from neurospatial.simulation.mazes.cheeseboard import (
            CheeseboardDims,
            make_cheeseboard_maze,
        )

        dims = CheeseboardDims(diameter=110.0)
        maze = make_cheeseboard_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # Extent should be roughly diameter in both dimensions
        x_extent = x_max - x_min
        y_extent = y_max - y_min
        expected_extent = dims.diameter

        # Should be circular (similar x and y extent)
        assert abs(x_extent - y_extent) < dims.diameter * 0.2  # Within 20%

        # Should be approximately the right diameter
        assert x_extent > expected_extent * 0.8  # At least 80% of expected
        assert y_extent > expected_extent * 0.8  # At least 80% of expected

    def test_env_2d_centered_at_origin(self):
        """Platform should be centered at origin (0, 0)."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        maze = make_cheeseboard_maze()

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_center = (bin_centers[:, 0].max() + bin_centers[:, 0].min()) / 2
        y_center = (bin_centers[:, 1].max() + bin_centers[:, 1].min()) / 2

        # Should be centered near origin
        assert abs(x_center) < 5.0  # Within 5 cm of origin
        assert abs(y_center) < 5.0  # Within 5 cm of origin

    def test_env_track_is_none(self):
        """env_track should be None (open field, no track)."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        maze = make_cheeseboard_maze()
        assert maze.env_track is None

    def test_has_reward_regions(self):
        """env_2d should have reward regions distributed across surface."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        maze = make_cheeseboard_maze()

        # Should have multiple reward regions (format: reward_i_j)
        reward_regions = [
            name for name in maze.env_2d.regions if name.startswith("reward_")
        ]
        assert len(reward_regions) > 0, "Should have at least one reward region"

    def test_reward_regions_are_points(self):
        """All reward regions should be point regions."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        maze = make_cheeseboard_maze()

        reward_regions = [
            name for name in maze.env_2d.regions if name.startswith("reward_")
        ]
        for region_name in reward_regions:
            region = maze.env_2d.regions[region_name]
            assert region.kind == "point", f"{region_name} should be a point region"

    def test_reward_regions_within_circular_boundary(self):
        """All reward regions should be within the circular platform boundary."""
        from neurospatial.simulation.mazes.cheeseboard import (
            CheeseboardDims,
            make_cheeseboard_maze,
        )

        dims = CheeseboardDims(diameter=110.0, reward_radius=1.5)
        maze = make_cheeseboard_maze(dims=dims)

        radius = dims.diameter / 2

        reward_regions = [
            name for name in maze.env_2d.regions if name.startswith("reward_")
        ]
        for region_name in reward_regions:
            region = maze.env_2d.regions[region_name]
            x, y = region.data
            distance_from_center = np.sqrt(x**2 + y**2)

            # Rewards should be within the platform radius minus reward_radius
            max_distance = radius - dims.reward_radius
            assert distance_from_center <= max_distance + 0.1, (
                f"{region_name} at ({x:.2f}, {y:.2f}) is outside boundary (dist={distance_from_center:.2f}, max={max_distance:.2f})"
            )

    def test_rewards_distributed_across_surface(self):
        """Rewards should be distributed across entire surface, not just perimeter."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        maze = make_cheeseboard_maze()

        reward_regions = [
            name for name in maze.env_2d.regions if name.startswith("reward_")
        ]
        positions = np.array(
            [maze.env_2d.regions[name].data for name in reward_regions]
        )

        # Check that we have rewards at various distances from center
        distances = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)

        # Should have rewards near center
        assert np.min(distances) < 20.0, (
            "Should have rewards near center (within 20 cm)"
        )

        # Should have rewards at intermediate distances
        intermediate = (distances > 20.0) & (distances < 40.0)
        assert np.sum(intermediate) > 0, (
            "Should have rewards at intermediate distances (20-40 cm)"
        )

    def test_reward_naming_convention(self):
        """Rewards should follow reward_i_j naming convention."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        maze = make_cheeseboard_maze()

        reward_regions = [
            name for name in maze.env_2d.regions if name.startswith("reward_")
        ]

        # At least one reward should have proper format
        valid_names = 0
        for name in reward_regions:
            # Should be reward_<int>_<int>
            parts = name.split("_")
            if len(parts) == 3 and parts[0] == "reward":
                try:
                    int(parts[1])  # i index
                    int(parts[2])  # j index
                    valid_names += 1
                except ValueError:
                    pass

        assert valid_names == len(reward_regions), (
            f"All {len(reward_regions)} rewards should follow reward_i_j format, found {valid_names}"
        )

    def test_regular_grid_spacing(self):
        """Rewards should be arranged in a regular grid pattern."""
        from neurospatial.simulation.mazes.cheeseboard import (
            CheeseboardDims,
            make_cheeseboard_maze,
        )

        dims = CheeseboardDims(diameter=110.0, grid_spacing=9.0)
        maze = make_cheeseboard_maze(dims=dims)

        reward_regions = [
            name for name in maze.env_2d.regions if name.startswith("reward_")
        ]
        positions = np.array(
            [maze.env_2d.regions[name].data for name in reward_regions]
        )

        # Get unique x and y coordinates (with rounding to handle float precision)
        unique_x = np.unique(np.round(positions[:, 0], 1))
        unique_y = np.unique(np.round(positions[:, 1], 1))

        # If more than one unique coordinate, check spacing
        if len(unique_x) > 1:
            x_diffs = np.diff(unique_x)
            # Most differences should be close to grid_spacing
            assert np.median(np.abs(x_diffs - dims.grid_spacing)) < 1.0

        if len(unique_y) > 1:
            y_diffs = np.diff(unique_y)
            # Most differences should be close to grid_spacing
            assert np.median(np.abs(y_diffs - dims.grid_spacing)) < 1.0

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.cheeseboard import (
            CheeseboardDims,
            make_cheeseboard_maze,
        )

        dims = CheeseboardDims()

        maze_fine = make_cheeseboard_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_cheeseboard_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.cheeseboard import (
            CheeseboardDims,
            make_cheeseboard_maze,
        )

        dims = CheeseboardDims(diameter=120.0, grid_spacing=10.0)
        maze = make_cheeseboard_maze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Both should be roughly 120 cm (diameter)
        assert x_extent > 100  # Larger than default 110
        assert y_extent > 100  # Larger than default 110

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.cheeseboard import (
            CheeseboardDims,
            make_cheeseboard_maze,
        )

        maze_default = make_cheeseboard_maze(dims=None)
        maze_explicit = make_cheeseboard_maze(dims=CheeseboardDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins

    def test_larger_grid_spacing_fewer_rewards(self):
        """Larger grid spacing should result in fewer reward locations."""
        from neurospatial.simulation.mazes.cheeseboard import (
            CheeseboardDims,
            make_cheeseboard_maze,
        )

        maze_fine = make_cheeseboard_maze(
            dims=CheeseboardDims(diameter=110.0, grid_spacing=5.0)
        )
        maze_coarse = make_cheeseboard_maze(
            dims=CheeseboardDims(diameter=110.0, grid_spacing=15.0)
        )

        fine_rewards = [
            name for name in maze_fine.env_2d.regions if name.startswith("reward_")
        ]
        coarse_rewards = [
            name for name in maze_coarse.env_2d.regions if name.startswith("reward_")
        ]

        assert len(fine_rewards) > len(coarse_rewards)


class TestCheeseboardDocstrings:
    """Tests for docstrings and examples."""

    def test_make_cheeseboard_maze_has_docstring(self):
        """make_cheeseboard_maze should have a docstring."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        assert make_cheeseboard_maze.__doc__ is not None
        assert len(make_cheeseboard_maze.__doc__) > 100  # Reasonable length

    def test_cheeseboard_dims_has_docstring(self):
        """CheeseboardDims should have a docstring."""
        from neurospatial.simulation.mazes.cheeseboard import CheeseboardDims

        assert CheeseboardDims.__doc__ is not None
