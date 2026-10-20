"""Tests for Sungod Maze implementation."""

from __future__ import annotations

import numpy as np
import pytest
import shapely

from neurospatial.simulation.mazes.sungod import make_sungod_maze

ARM_PORTS = [f"arm_port_{k}" for k in range(1, 9)]
PORTS = ["home_port", "center_port_left", "center_port_right", *ARM_PORTS]

# Port positions in cm, traced from the maze and taken as ground truth.
PORT_POSITIONS = {
    "home_port": (40.93, 1.35),
    "center_port_left": (25.93, -2.99),
    "center_port_right": (25.90, 3.91),
    "arm_port_1": (-26.95, -47.31),
    "arm_port_2": (-34.25, -34.31),
    "arm_port_3": (-39.39, -20.62),
    "arm_port_4": (-43.77, -7.36),
    "arm_port_5": (-43.25, 6.74),
    "arm_port_6": (-39.84, 20.76),
    "arm_port_7": (-34.23, 33.54),
    "arm_port_8": (-24.58, 44.43),
}


@pytest.fixture(scope="module")
def maze():
    return make_sungod_maze()


def _rightward(maze, points):
    """Distance to the right of the line from the home port to the box center.

    Positive to the right as seen from the home port facing the arms.
    """
    graph = maze.env_track.binning.linearization.graph
    home = np.asarray(graph.nodes["home_port"]["pos"])
    facing = np.asarray(graph.nodes["box_center"]["pos"]) - home
    facing /= np.linalg.norm(facing)
    left = np.array([-facing[1], facing[0]])
    return -(np.atleast_2d(points) - home) @ left


class TestPorts:
    """The 11 ports of Gillespie et al. (2021), named as in the paper."""

    def test_ports_are_reward_port_landmarks(self, maze):
        landmarks = maze.env_2d.landmarks
        assert list(landmarks) == PORTS
        assert {lm.kind for lm in landmarks.values()} == {"point"}
        assert {lm.metadata["role"] for lm in landmarks.values()} == {"reward_port"}
        port_types = [landmarks[name].metadata["port_type"] for name in PORTS]
        assert port_types == ["home", "center", "center"] + ["arm"] * 8
        assert landmarks["center_port_left"].metadata["side"] == "left"
        assert landmarks["center_port_right"].metadata["side"] == "right"
        assert [landmarks[name].metadata["arm"] for name in ARM_PORTS] == list(
            range(1, 9)
        )

    def test_port_positions(self, maze):
        for name, position in PORT_POSITIONS.items():
            np.testing.assert_allclose(
                maze.env_2d.landmarks[name].data, position, atol=0.01, err_msg=name
            )

    def test_each_port_has_a_point_region_at_its_landmark(self, maze):
        regions = maze.env_2d.regions
        assert list(regions) == [*PORTS, "box"]
        for name in PORTS:
            assert regions[name].kind == "point"
            np.testing.assert_allclose(
                regions[name].data, maze.env_2d.landmarks[name].data
            )

    def test_center_ports_are_about_7_5_cm_apart(self, maze):
        landmarks = maze.env_2d.landmarks
        gap = np.linalg.norm(
            landmarks["center_port_left"].data - landmarks["center_port_right"].data
        )
        assert gap == pytest.approx(6.9, abs=0.05)

    def test_home_and_center_ports_are_in_the_box(self, maze):
        box = shapely.Polygon(maze.env_2d.regions["box"].data)
        for name in ("home_port", "center_port_left", "center_port_right"):
            assert box.contains(shapely.Point(maze.env_2d.landmarks[name].data)), name
        for name in ARM_PORTS:
            assert not box.contains(shapely.Point(maze.env_2d.landmarks[name].data))

    def test_home_port_is_the_box_port_farthest_from_the_arms(self, maze):
        landmarks = maze.env_2d.landmarks
        arm_ports = landmarks.positions(ARM_PORTS)

        def distance_to_arms(name):
            return np.linalg.norm(arm_ports - landmarks[name].data, axis=1).min()

        assert distance_to_arms("home_port") > distance_to_arms("center_port_left")
        assert distance_to_arms("home_port") > distance_to_arms("center_port_right")

    def test_home_axis_passes_between_the_center_ports(self, maze):
        landmarks = maze.env_2d.landmarks
        left, right = _rightward(
            maze, landmarks.positions(["center_port_left", "center_port_right"])
        )
        assert left < 0 < right

    def test_arms_are_numbered_left_to_right_from_the_home_port(self, maze):
        rightward = _rightward(maze, maze.env_2d.landmarks.positions(ARM_PORTS))
        assert np.all(np.diff(rightward) > 0)

    def test_each_arm_port_is_at_the_far_end_of_its_arm(self, maze):
        graph = maze.env_track.binning.linearization.graph
        ends = np.array([graph.nodes[f"arm_{k}_end"]["pos"] for k in range(1, 9)])
        for k, name in enumerate(ARM_PORTS):
            distances = np.linalg.norm(ends - maze.env_2d.landmarks[name].data, axis=1)
            assert distances.argmin() == k, name
            # Within the last few cm of an arm about 50 cm long.
            assert distances[k] < 5.0, name

    def test_ports_scale_with_the_maze(self):
        from neurospatial.simulation.mazes.sungod import SungodDims

        large = make_sungod_maze(SungodDims(scale=2.0), include_track=False)
        for name, position in PORT_POSITIONS.items():
            np.testing.assert_allclose(
                large.env_2d.landmarks[name].data,
                2.0 * np.asarray(position),
                atol=0.05,
                err_msg=name,
            )


class TestTrack:
    def test_track_starts_at_the_home_port(self, maze):
        graph = maze.env_track.binning.linearization.graph
        np.testing.assert_allclose(
            graph.nodes["home_port"]["pos"], PORT_POSITIONS["home_port"], atol=0.01
        )
        assert set(graph.neighbors("home_port")) == {"box_center"}
        assert set(graph.neighbors("box_center")) == {
            "home_port",
            *(f"arm_{k}_start" for k in range(1, 9)),
        }

    def test_track_carries_the_ports(self, maze):
        assert list(maze.env_track.landmarks) == PORTS
        assert list(maze.env_track.regions) == list(maze.env_2d.regions)


class TestMakeSungodMaze:
    """Tests for make_sungod_maze factory function."""

    def test_construct_default(self):
        """make_sungod_maze should return a fitted MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.sungod import make_sungod_maze

        maze = make_sungod_maze()
        assert isinstance(maze, MazeEnvironments)
        assert maze.env_2d.n_bins > 0
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

    def test_include_track_toggle(self):
        """include_track controls whether a 1D track graph is produced."""
        from neurospatial.simulation.mazes.sungod import make_sungod_maze

        with_track = make_sungod_maze(include_track=True)
        without_track = make_sungod_maze(include_track=False)

        assert with_track.env_track is not None
        assert without_track.env_track is None

    def test_track_lies_on_the_maze_floor(self):
        """Every track bin is on the 2-D floor, so 1-D and 2-D positions agree.

        Each arm's track runs along the middle of the arm, not on a straight
        line from the box center that would cut across the walls between arms.
        """
        from neurospatial.simulation.mazes.sungod import make_sungod_maze

        for bin_size in (1.0, 2.0, 3.0):
            maze = make_sungod_maze(bin_size=bin_size)
            track = maze.env_track
            assert track is not None
            off = track.bin_centers[~maze.env_2d.contains(track.bin_centers)]
            assert len(off) == 0, f"bin_size={bin_size}: off-floor bins {off[:5]}"
