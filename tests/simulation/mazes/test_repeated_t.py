"""Tests for Repeated T-Maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import networkx as nx
import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestRepeatedTDims:
    """Tests for RepeatedTDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """RepeatedTDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.repeated_t import RepeatedTDims

        dims = RepeatedTDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """RepeatedTDims should have correct default values."""
        from neurospatial.simulation.mazes.repeated_t import RepeatedTDims

        dims = RepeatedTDims()
        assert dims.stem_spacing == 40.0
        assert dims.stem_length == 60.0
        assert dims.top_spur_length == 20.0
        assert dims.arm_length == 15.0
        assert dims.upper_lower_gap == 30.0
        assert dims.width == 10.0

    def test_is_frozen(self):
        """RepeatedTDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.repeated_t import RepeatedTDims

        dims = RepeatedTDims()
        with pytest.raises(FrozenInstanceError):
            dims.stem_spacing = 50.0  # type: ignore[misc]

    def test_custom_values(self):
        """RepeatedTDims should accept custom values."""
        from neurospatial.simulation.mazes.repeated_t import RepeatedTDims

        dims = RepeatedTDims(
            stem_spacing=50.0,
            stem_length=80.0,
            top_spur_length=25.0,
            arm_length=20.0,
            upper_lower_gap=40.0,
            width=12.0,
        )
        assert dims.stem_spacing == 50.0
        assert dims.stem_length == 80.0
        assert dims.top_spur_length == 25.0
        assert dims.arm_length == 20.0
        assert dims.upper_lower_gap == 40.0
        assert dims.width == 12.0


class TestMakeRepeatedTMaze:
    """Tests for make_repeated_t_maze factory function."""

    def test_returns_maze_environments(self):
        """make_repeated_t_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_start_region(self):
        """env_2d should have a 'start' region."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        assert "start" in maze.env_2d.regions

    def test_env_2d_has_goal_region(self):
        """env_2d should have a 'goal' region."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        assert "goal" in maze.env_2d.regions

    def test_env_2d_has_top_spur_region(self):
        """env_2d should have a 'top_spur' region (center stem above upper horizontal)."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        assert "top_spur" in maze.env_2d.regions

    def test_env_2d_has_stem_bottom_regions(self):
        """env_2d should have regions for bottom of each stem."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        assert "left_stem_bottom" in maze.env_2d.regions
        assert "center_stem_bottom" in maze.env_2d.regions
        assert "right_stem_bottom" in maze.env_2d.regions

    def test_env_2d_has_junction_regions(self):
        """env_2d should have regions for T-junctions."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        assert "upper_junction" in maze.env_2d.regions
        assert "lower_left_junction" in maze.env_2d.regions
        assert "lower_center_junction" in maze.env_2d.regions
        assert "upper_right_junction" in maze.env_2d.regions

    def test_env_2d_has_lower_arm_regions(self):
        """env_2d should have regions for lower arm ends."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        assert "lower_left_arm" in maze.env_2d.regions
        assert "lower_center_left_arm" in maze.env_2d.regions

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_1d

    def test_env_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze(include_track=True)
        assert maze.env_track is not None
        assert nx.is_connected(maze.env_track.connectivity)

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze_fine = make_repeated_t_maze(bin_size=1.0)
        maze_coarse = make_repeated_t_maze(bin_size=5.0)

        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        maze_default = make_repeated_t_maze(dims=None)
        maze_explicit = make_repeated_t_maze(dims=RepeatedTDims())

        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestRepeatedTMazeStructure:
    """Tests for the three-stem T-maze structure."""

    def test_three_stems_exist(self):
        """Maze should have three vertical stems."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()

        # Check that stem bottom regions exist at different x positions
        left_bottom = maze.env_2d.regions["left_stem_bottom"]
        center_bottom = maze.env_2d.regions["center_stem_bottom"]
        right_bottom = maze.env_2d.regions["right_stem_bottom"]

        # X positions should be different (spaced apart)
        assert left_bottom.data[0] < center_bottom.data[0] < right_bottom.data[0]

        # Y positions should be the same (all at bottom)
        assert np.isclose(left_bottom.data[1], center_bottom.data[1])
        assert np.isclose(center_bottom.data[1], right_bottom.data[1])

    def test_center_stem_extends_above_upper_horizontal(self):
        """Center stem should extend above the upper horizontal (top_spur)."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()

        top_spur = maze.env_2d.regions["top_spur"]
        upper_junction = maze.env_2d.regions["upper_junction"]

        # Top spur should be above upper junction (same x, higher y)
        assert np.isclose(top_spur.data[0], upper_junction.data[0])
        assert top_spur.data[1] > upper_junction.data[1]

    def test_upper_horizontal_has_gap(self):
        """Upper horizontal should have two separate sections with a gap."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        dims = RepeatedTDims()
        maze = make_repeated_t_maze(dims=dims)

        # Start is at left arm end
        start = maze.env_2d.regions["start"]
        # Upper junction is where left section meets center stem
        upper_junction = maze.env_2d.regions["upper_junction"]
        # Goal is at right arm end
        goal = maze.env_2d.regions["goal"]

        # Start and upper_junction should be on same horizontal level
        assert np.isclose(start.data[1], upper_junction.data[1])

        # Goal and upper_right_junction should be on same horizontal level
        upper_right_junction = maze.env_2d.regions["upper_right_junction"]
        assert np.isclose(goal.data[1], upper_right_junction.data[1])

        # Gap exists: upper_junction to upper_right_junction has distance
        gap_distance = upper_right_junction.data[0] - upper_junction.data[0]
        assert gap_distance > dims.arm_length  # There's a gap, not just arm length

    def test_lower_horizontal_has_gap(self):
        """Lower horizontal should have two separate sections with a gap."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()

        # Lower-left arm end
        lower_left_arm = maze.env_2d.regions["lower_left_arm"]
        # Lower-left junction (at left stem)
        lower_left_junction = maze.env_2d.regions["lower_left_junction"]
        # Lower-center junction (at center stem)
        lower_center_junction = maze.env_2d.regions["lower_center_junction"]
        # Lower-center-left arm (extends left from center stem)
        lower_center_left_arm = maze.env_2d.regions["lower_center_left_arm"]

        # Lower-left arm and lower-left junction should be on same horizontal level
        assert np.isclose(lower_left_arm.data[1], lower_left_junction.data[1])

        # Lower-center junction should be at same y level
        assert np.isclose(lower_left_junction.data[1], lower_center_junction.data[1])

        # Lower-center-left arm should be at same y level
        assert np.isclose(lower_center_junction.data[1], lower_center_left_arm.data[1])

        # Gap exists between left_stem and center_stem's left arm
        # The lower_center_left_arm should be to the right of lower_left_arm
        assert lower_center_left_arm.data[0] > lower_left_arm.data[0]

    def test_start_at_upper_left_arm(self):
        """Start should be at the end of upper-left horizontal arm."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        dims = RepeatedTDims()
        maze = make_repeated_t_maze(dims=dims)

        start = maze.env_2d.regions["start"]
        upper_junction = maze.env_2d.regions["upper_junction"]

        # Start should be to the left of upper_junction
        assert start.data[0] < upper_junction.data[0]

        # Y should match upper horizontal level
        assert np.isclose(start.data[1], upper_junction.data[1])

    def test_goal_at_upper_right_arm(self):
        """Goal should be at the end of upper-right horizontal arm."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        dims = RepeatedTDims()
        maze = make_repeated_t_maze(dims=dims)

        goal = maze.env_2d.regions["goal"]
        upper_right_junction = maze.env_2d.regions["upper_right_junction"]

        # Goal should be to the right of upper_right_junction
        assert goal.data[0] > upper_right_junction.data[0]

        # Y should match upper horizontal level
        assert np.isclose(goal.data[1], upper_right_junction.data[1])

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze(include_track=True)
        assert maze.env_track is not None

        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]
