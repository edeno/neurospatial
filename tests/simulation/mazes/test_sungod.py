"""Tests for Sungod Maze implementation."""

from __future__ import annotations

import numpy as np


class TestMakeSungodMaze:
    """Tests for make_sungod_maze factory function."""

    def test_construct_default(self):
        """make_sungod_maze should return a fitted MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.sungod import make_sungod_maze

        maze = make_sungod_maze()
        assert isinstance(maze, MazeEnvironments)
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted
        assert maze.env_2d.units == "cm"

    def test_env_2d_is_connected(self):
        """The 2D maze surface must be a single connected component.

        The Sungod maze is a central box with 8 radiating arms unioned into one
        polygon; if a gap survived the buffer/union step the environment would
        fragment, which this catches.
        """
        import networkx as nx

        from neurospatial.simulation.mazes.sungod import make_sungod_maze

        maze = make_sungod_maze()
        assert nx.is_connected(maze.env_2d.connectivity)

    def test_n_bins_reasonable(self):
        """Bin count should match the ~1 m x 1 m geometry at the default bin size.

        Default bin_size=2 cm over a ~88 x 100 cm arm-and-box footprint yields
        order-1000 active bins (observed ~905). Bound it generously so the test
        tracks geometry/bin-size regressions without pinning an exact count.
        """
        from neurospatial.simulation.mazes.sungod import make_sungod_maze

        maze = make_sungod_maze()
        assert 400 < maze.env_2d.n_bins < 2000

    def test_finer_bins_give_more_bins(self):
        """Smaller bin_size must increase the active-bin count."""
        from neurospatial.simulation.mazes.sungod import make_sungod_maze

        fine = make_sungod_maze(bin_size=1.0)
        coarse = make_sungod_maze(bin_size=4.0)
        assert fine.env_2d.n_bins > coarse.env_2d.n_bins

    def test_centered_near_origin(self):
        """The maze polygon is translated to be centered on the origin."""
        from neurospatial.simulation.mazes.sungod import make_sungod_maze

        maze = make_sungod_maze()
        bin_centers = maze.env_2d.bin_centers
        center_x = (bin_centers[:, 0].max() + bin_centers[:, 0].min()) / 2
        center_y = (bin_centers[:, 1].max() + bin_centers[:, 1].min()) / 2

        # Extent is ~1 m; centering should land the mid-extent within ~10 cm.
        assert abs(center_x) < 10.0
        assert abs(center_y) < 10.0

    def test_meter_scale_extent(self):
        """The maze is ~1 m across (calibration target)."""
        from neurospatial.simulation.mazes.sungod import make_sungod_maze

        maze = make_sungod_maze()
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Both spans should be in the 70-120 cm range for a ~1 m maze.
        assert 70.0 < x_extent < 120.0
        assert 70.0 < y_extent < 120.0

    def test_region_positions(self):
        """Named regions exist and sit within the maze footprint.

        The maze defines a 'center' region plus 11 reward wells; all should land
        inside the (centered) ~1 m extent rather than at degenerate coordinates.
        """
        from neurospatial.simulation.mazes.sungod import make_sungod_maze

        maze = make_sungod_maze()
        regions = maze.env_2d.regions

        assert "center" in regions
        well_names = [name for name in regions if name.startswith("reward_well_")]
        assert len(well_names) == 11

        bin_centers = maze.env_2d.bin_centers
        x_lo, x_hi = bin_centers[:, 0].min() - 5.0, bin_centers[:, 0].max() + 5.0
        y_lo, y_hi = bin_centers[:, 1].min() - 5.0, bin_centers[:, 1].max() + 5.0

        for name in ["center", *well_names]:
            x, y = np.asarray(regions[name].data, dtype=float)
            assert x_lo <= x <= x_hi, (
                f"{name} x={x:.1f} outside [{x_lo:.1f}, {x_hi:.1f}]"
            )
            assert y_lo <= y <= y_hi, (
                f"{name} y={y:.1f} outside [{y_lo:.1f}, {y_hi:.1f}]"
            )

    def test_include_track_toggle(self):
        """include_track controls whether a 1D track graph is produced."""
        from neurospatial.simulation.mazes.sungod import make_sungod_maze

        with_track = make_sungod_maze(include_track=True)
        without_track = make_sungod_maze(include_track=False)

        assert with_track.env_track is not None
        assert without_track.env_track is None
