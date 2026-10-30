"""Tests for Cheeseboard Maze implementation."""

from __future__ import annotations

import numpy as np
import pytest


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

    def test_env_2d_connectivity_is_connected(self):
        """The open-field 2D environment must be a single connected component."""
        import networkx as nx

        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        maze = make_cheeseboard_maze()
        assert nx.is_connected(maze.env_2d.connectivity)

    def test_every_well_is_a_reward_port_landmark(self):
        """Any well can be baited, so each is a reward port of 2.5 cm."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        env = make_cheeseboard_maze().env_2d
        assert list(env.landmarks) == list(env.regions)
        assert all(name.startswith("well_") for name in env.landmarks)
        for name, lm in env.landmarks.items():
            assert lm.kind == "point"
            assert dict(lm.metadata) == {
                "role": "reward_port",
                "port_type": "food_well",
                "diameter": 2.5,
            }
            np.testing.assert_array_equal(lm.data, env.regions[name].data)
        np.testing.assert_array_equal(env.landmarks["well_7_7"].data, [0.0, 0.0])
        np.testing.assert_array_equal(env.landmarks["well_8_7"].data, [7.5, 0.0])

    @pytest.mark.parametrize("bin_size", [1.0, 2.0, 3.0, 5.0])
    def test_each_well_region_covers_the_bin_under_it(self, bin_size):
        """A 2.5 cm well is smaller than a bin, so its region is its centre.

        A circle of the well's real size contains no bin centre for some
        wells at these bin sizes; the point region always covers one bin.
        """
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        env = make_cheeseboard_maze(bin_size=bin_size).env_2d
        for name, region in env.regions.items():
            assert region.kind == "point", name
            bins = env.bins_in_region(name)
            assert len(bins) == 1, name
            assert bins[0] == env.bin_at(np.asarray(region.data)[np.newaxis])[0]

    def test_no_start_box_or_cue(self):
        """No source gives the start box's size or placement, or a cue's."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        roles = {
            lm.metadata["role"]
            for lm in make_cheeseboard_maze().env_2d.landmarks.values()
        }
        assert roles == {"reward_port"}

    def test_reward_regions_within_circular_boundary(self):
        """All reward regions should be within the circular platform boundary."""
        from neurospatial.simulation.mazes.cheeseboard import (
            CheeseboardDims,
            make_cheeseboard_maze,
        )

        for dims in (CheeseboardDims(), CheeseboardDims(diameter=110.0)):
            maze = make_cheeseboard_maze(dims=dims)
            positions = np.array([r.data for r in maze.env_2d.regions.values()])
            distances = np.hypot(positions[:, 0], positions[:, 1])

            # Every well centre is at least rim_margin in from the rim, so the
            # whole 2.5 cm well lies on the board.
            radius = dims.diameter / 2
            assert distances.max() <= radius - dims.rim_margin
            assert distances.max() + dims.reward_radius < radius

    def test_default_is_the_177_well_rat_board(self):
        """A 120 cm board with 177 wells of 2.5 cm diameter."""
        from neurospatial.simulation.mazes.cheeseboard import (
            CheeseboardDims,
            make_cheeseboard_maze,
        )

        dims = CheeseboardDims()
        assert dims.diameter == 120.0
        assert 2 * dims.reward_radius == 2.5
        # Derived: the mouse board's 5 cm spacing scaled by 120 / 80.
        assert dims.grid_spacing == pytest.approx(5.0 * 120.0 / 80.0)

        env = make_cheeseboard_maze().env_2d
        assert len(env.regions) == 177
        x_min, y_min, x_max, y_max = env.arena.polygon.bounds
        assert x_max - x_min == pytest.approx(120.0, abs=0.1)
        assert y_max - y_min == pytest.approx(120.0, abs=0.1)

    def test_wells_form_an_origin_centred_square_lattice(self):
        """A well at the centre, every well on the 7.5 cm lattice, none missing."""
        from neurospatial.simulation.mazes.cheeseboard import (
            CheeseboardDims,
            make_cheeseboard_maze,
        )

        dims = CheeseboardDims()
        regions = make_cheeseboard_maze(dims=dims).env_2d.regions
        positions = np.array([r.data for r in regions.values()])

        lattice = positions / dims.grid_spacing
        np.testing.assert_allclose(lattice, np.round(lattice), atol=1e-9)
        assert any(np.allclose(p, 0.0) for p in positions)

        # Every lattice point within the well radius has a well.
        limit = dims.diameter / 2 - dims.rim_margin
        k = np.arange(-10, 11)
        ii, jj = np.meshgrid(k, k)
        expected = np.hypot(ii, jj) * dims.grid_spacing <= limit
        assert expected.sum() == len(positions) == 177

    @pytest.mark.parametrize("rim_margin", [3.0, 4.0, 5.0])
    def test_177_wells_across_the_derived_rim_margin_range(self, rim_margin):
        """Any rim margin from about 2.9 to 5.4 cm keeps exactly 177 wells."""
        from neurospatial.simulation.mazes.cheeseboard import (
            CheeseboardDims,
            make_cheeseboard_maze,
        )

        dims = CheeseboardDims(rim_margin=rim_margin)
        assert len(make_cheeseboard_maze(dims=dims, bin_size=5.0).env_2d.regions) == 177

    def test_same_construction_gives_177_wells_on_the_80_cm_mouse_board(self):
        """Roux et al. (2017): 80 cm board, 177 wells, 5 cm spacing."""
        from neurospatial.simulation.mazes.cheeseboard import (
            CheeseboardDims,
            make_cheeseboard_maze,
        )

        dims = CheeseboardDims(diameter=80.0, grid_spacing=5.0, rim_margin=2.5)
        assert len(make_cheeseboard_maze(dims=dims, bin_size=4.0).env_2d.regions) == 177

    def test_rewards_distributed_across_surface(self):
        """Rewards should be distributed across entire surface, not just perimeter."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        maze = make_cheeseboard_maze()

        reward_regions = [
            name for name in maze.env_2d.regions if name.startswith("well_")
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
        """Rewards should follow well_i_j naming convention."""
        from neurospatial.simulation.mazes.cheeseboard import make_cheeseboard_maze

        maze = make_cheeseboard_maze()

        reward_regions = [
            name for name in maze.env_2d.regions if name.startswith("well_")
        ]

        # At least one reward should have proper format
        valid_names = 0
        for name in reward_regions:
            # Should be well_<int>_<int>
            parts = name.split("_")
            if len(parts) == 3 and parts[0] == "well":
                try:
                    int(parts[1])  # i index
                    int(parts[2])  # j index
                    valid_names += 1
                except ValueError:
                    pass

        assert valid_names == len(reward_regions), (
            f"All {len(reward_regions)} rewards should follow well_i_j format, found {valid_names}"
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
            name for name in maze.env_2d.regions if name.startswith("well_")
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
        assert x_extent > 100
        assert y_extent > 100

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
            name for name in maze_fine.env_2d.regions if name.startswith("well_")
        ]
        coarse_rewards = [
            name for name in maze_coarse.env_2d.regions if name.startswith("well_")
        ]

        assert len(fine_rewards) > len(coarse_rewards)
