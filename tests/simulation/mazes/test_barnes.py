"""Tests for Barnes Maze implementation."""

from __future__ import annotations

import numpy as np
import pytest


def _holes(env):
    """Each hole's landmark, keyed by its ring position."""
    return {
        lm.metadata["index"]: lm
        for lm in env.landmarks.values()
        if lm.metadata["role"] in ("hole", "escape_hole")
    }


class TestMakeBarnesMaze:
    """Tests for make_barnes_maze factory function."""

    def test_returns_maze_environments(self):
        """make_barnes_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_is_circular(self):
        """env_2d should have a circular platform extent."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims(diameter=120.0)
        maze = make_barnes_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X and Y extent should both be approximately equal to diameter
        x_extent = x_max - x_min
        y_extent = y_max - y_min
        expected_extent = dims.diameter

        # Allow 20% tolerance for binning effects
        assert x_extent > expected_extent * 0.7
        assert x_extent < expected_extent * 1.3
        assert y_extent > expected_extent * 0.7
        assert y_extent < expected_extent * 1.3

        # Should be roughly circular (aspect ratio near 1.0)
        aspect_ratio = x_extent / y_extent
        assert 0.8 < aspect_ratio < 1.2

    def test_platform_centered_at_origin(self):
        """Platform should be centered at origin (0, 0)."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        bin_centers = maze.env_2d.bin_centers

        # Center of mass should be near origin
        center_x = bin_centers[:, 0].mean()
        center_y = bin_centers[:, 1].mean()

        assert abs(center_x) < 5.0
        assert abs(center_y) < 5.0

    def test_default_18_holes(self):
        """18 holes by default (original design): 17 hole_i and the escape hole."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        env = make_barnes_maze().env_2d
        assert sorted(_holes(env)) == list(range(18))
        # hole_0 is the escape hole by default; the rest keep their numbers.
        assert [f"hole_{i}" for i in range(1, 18)] == [
            name for name in env.regions if name.startswith("hole_")
        ]
        assert "hole_0" not in env.regions
        assert "hole_18" not in env.regions

    def test_custom_number_of_holes(self):
        """Custom n_holes should be respected."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims(n_holes=12)
        env = make_barnes_maze(dims=dims).env_2d
        assert sorted(_holes(env)) == list(range(12))
        assert "hole_11" in env.regions
        assert "hole_12" not in env.regions

    def test_default_is_the_original_barnes_apparatus(self):
        """Barnes (1979) via Gawel et al. (2018): 122 cm, 18 holes of 9.5 cm."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims()
        assert dims.diameter == 122.0
        assert dims.n_holes == 18
        assert 2 * dims.hole_radius == 9.5
        # Estimated from Wijnen et al. (2024) Fig. 1c: centres at ~0.8 R.
        assert dims.hole_inset == 12.0

        x_min, y_min, x_max, y_max = make_barnes_maze().env_2d.arena.polygon.bounds
        assert x_max - x_min == pytest.approx(122.0)
        assert y_max - y_min == pytest.approx(122.0)

    def test_holes_on_perimeter(self):
        """Hole centres lie on a ring hole_inset in from the rim."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims()
        maze = make_barnes_maze(dims=dims)

        radius = dims.diameter / 2.0
        ring_radius = radius - dims.hole_inset
        assert ring_radius / radius == pytest.approx(0.8, abs=0.01)

        holes = _holes(maze.env_2d)
        assert len(holes) == dims.n_holes
        for hole in holes.values():
            assert hole.kind == "point"
            assert np.hypot(*hole.data) == pytest.approx(ring_radius)
            # The whole 9.5 cm hole fits on the platform.
            assert np.hypot(*hole.data) + dims.hole_radius < radius

    def test_hole_regions_are_the_9_5_cm_holes(self):
        """Each hole's region is a circle of the published 9.5 cm diameter."""
        from shapely.geometry import Point

        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims()
        env = make_barnes_maze(dims=dims).env_2d
        for hole in _holes(env).values():
            region = env.regions[hole.name]
            assert region.kind == "polygon"
            disc = Point(hole.data).buffer(4.75, quad_segs=256)
            assert region.data.symmetric_difference(disc).area < 0.01 * disc.area
            np.testing.assert_allclose(
                region.data.centroid.coords[0], hole.data, atol=1e-9
            )
            # About pi * 4.75**2 / 2**2 = 17.7 bins of 2 cm under each hole.
            assert 12 <= len(env.bins_in_region(hole.name)) <= 24

    def test_hole_inset_sets_the_ring_radius(self):
        """A custom hole_inset moves the holes along the radius."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims(diameter=120.0, hole_inset=5.0)
        hole = _holes(make_barnes_maze(dims=dims).env_2d)[0]
        np.testing.assert_allclose(hole.data, [55.0, 0.0], atol=1e-9)

    def test_holes_evenly_distributed(self):
        """Hole i lies at angle 2 pi i / n_holes."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims(diameter=120.0, n_holes=18)
        holes = _holes(make_barnes_maze(dims=dims, escape_hole_index=5).env_2d)
        for i, hole in holes.items():
            angle = np.arctan2(hole.data[1], hole.data[0]) % (2 * np.pi)
            assert angle == pytest.approx(2 * np.pi * i / dims.n_holes, abs=1e-9)

    def test_escape_hole_is_the_hole_at_escape_hole_index(self):
        """One item per physical hole: escape_hole replaces hole_i."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        default = make_barnes_maze().env_2d
        moved = make_barnes_maze(escape_hole_index=9).env_2d
        for env, index in ((default, 0), (moved, 9)):
            escape = env.landmarks["escape_hole"]
            assert dict(escape.metadata) == {"role": "escape_hole", "index": index}
            assert f"hole_{index}" not in env.regions
            assert f"hole_{index}" not in env.landmarks
            assert "escape_hole" in env.regions
        # The escape hole sits where hole_9 would be, and hole_0 is back.
        np.testing.assert_allclose(
            moved.landmarks["escape_hole"].data, default.landmarks["hole_9"].data
        )
        assert dict(moved.landmarks["hole_0"].metadata) == {"role": "hole", "index": 0}

    def test_one_region_per_hole(self):
        """No two regions share a hole: 18 hole regions plus the start box."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        env = make_barnes_maze(escape_hole_index=4).env_2d
        assert len(env.regions) == 19
        assert list(env.landmarks) == list(env.regions)
        centres = np.array([lm.data for lm in _holes(env).values()])
        assert len(np.unique(np.round(centres, 6), axis=0)) == 18

    def test_start_box_in_the_middle(self):
        """Barnes (1979): the start chamber is placed in the middle of the maze."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        env = make_barnes_maze().env_2d
        start = env.landmarks["start_box"]
        assert dict(start.metadata) == {"role": "start_box"}
        np.testing.assert_array_equal(start.data, [0.0, 0.0])
        # Its size is not published, so its region is the point itself.
        assert env.regions["start_box"].kind == "point"
        np.testing.assert_array_equal(env.regions["start_box"].data, [0.0, 0.0])

    def test_landmark_roles(self):
        """17 holes, one escape hole, one start box and no cue."""
        from collections import Counter

        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        roles = Counter(
            lm.metadata["role"] for lm in make_barnes_maze().env_2d.landmarks.values()
        )
        assert roles == {"hole": 17, "escape_hole": 1, "start_box": 1}

    def test_env_track_is_none(self):
        """Barnes maze should not have a track graph (open field)."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        assert maze.env_track is None

    def test_env_2d_connectivity_is_connected(self):
        """The open-field 2D environment must be a single connected component."""
        import networkx as nx

        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        assert nx.is_connected(maze.env_2d.connectivity)

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims()

        maze_fine = make_barnes_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_barnes_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims(diameter=200.0, n_holes=24)
        maze = make_barnes_maze(dims=dims)

        # Check that platform extent is larger
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Should be larger than the 122 cm default
        assert x_extent > 140
        assert y_extent > 140

        # Should have 24 holes
        assert sorted(_holes(maze.env_2d)) == list(range(24))
        assert "hole_24" not in maze.env_2d.regions

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        maze_default = make_barnes_maze(dims=None)
        maze_explicit = make_barnes_maze(dims=BarnesDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins

        # Both should have 18 holes
        assert "hole_17" in maze_default.env_2d.regions
        assert "hole_17" in maze_explicit.env_2d.regions
