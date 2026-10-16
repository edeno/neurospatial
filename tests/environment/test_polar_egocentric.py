"""Tests for from_polar_egocentric() factory method.

This module tests the egocentric polar environment factory method which creates
environments in polar coordinates (distance, angle) centered on the animal.
"""

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment, Regions
from neurospatial._exceptions import CartesianOnlyError
from neurospatial.environment.core import (
    CARTESIAN_GRID_OPERATIONS,
    CARTESIAN_OPERATIONS,
)


class TestFromPolarEgocentricBasic:
    """Test basic creation and properties of egocentric polar environments."""

    def test_creates_environment_with_expected_n_bins(self):
        """Test that n_bins matches expected n_distance * n_angle."""
        # Create polar environment
        distance_range = (0.0, 100.0)  # 0-100 cm
        angle_range = (-np.pi, np.pi)  # Full circle
        distance_bin_size = 10.0  # 10 bins
        angle_bin_size = np.pi / 4  # 8 bins (45 degrees each)

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        # Expected: 10 distance bins * 8 angle bins = 80 bins
        expected_n_distance = 10
        expected_n_angle = 8
        expected_n_bins = expected_n_distance * expected_n_angle

        assert env.n_bins == expected_n_bins

    def test_bin_centers_have_correct_dimensions(self):
        """Test that bin_centers has shape (n_bins, 2) with distance and angle."""
        distance_range = (0.0, 50.0)
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0
        angle_bin_size = np.pi / 2

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        assert env.bin_centers.shape[1] == 2  # 2D: (distance, angle)
        assert env.n_dims == 2

    def test_bin_centers_distances_in_range(self):
        """Test that bin_centers[:, 0] (distances) are within distance_range."""
        distance_range = (5.0, 50.0)
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0
        angle_bin_size = np.pi / 2

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        distances = env.bin_centers[:, 0]
        # Centers should be within range (with some tolerance for bin center calculation)
        assert np.all(distances >= distance_range[0])
        assert np.all(distances <= distance_range[1])

    def test_bin_centers_angles_in_range(self):
        """Test that bin_centers[:, 1] (angles) are within angle_range."""
        distance_range = (0.0, 50.0)
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0
        angle_bin_size = np.pi / 2

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        angles = env.bin_centers[:, 1]
        # Centers should be within range
        assert np.all(angles >= angle_range[0])
        assert np.all(angles <= angle_range[1])


class TestCircularConnectivity:
    """Test circular connectivity wrapping for angle dimension."""

    def test_circular_angle_true_wraps_first_last_angle_bins(self):
        """Test that circular_angle=True connects first and last angle bins."""
        distance_range = (0.0, 10.0)  # 1 distance bin
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0  # 1 bin
        angle_bin_size = np.pi / 2  # 4 bins

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        # With 1 distance bin and 4 angle bins, bin_centers should be:
        # bin 0: (5, -3π/4), bin 1: (5, -π/4), bin 2: (5, π/4), bin 3: (5, 3π/4)
        # With circular connectivity, bin 0 and bin 3 should be connected

        connectivity = env.connectivity
        # The first and last angle bins at the same distance should be connected
        # Find bins at the first angle (-3π/4) and last angle (3π/4)
        angles = env.bin_centers[:, 1]
        first_angle_bin = np.argmin(angles)
        last_angle_bin = np.argmax(angles)

        # They should be neighbors
        assert connectivity.has_edge(first_angle_bin, last_angle_bin)

    def test_circular_angle_false_does_not_wrap(self):
        """Test that circular_angle=False does not connect first and last angle bins."""
        distance_range = (0.0, 10.0)  # 1 distance bin
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0  # 1 bin
        angle_bin_size = np.pi / 2  # 4 bins

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
            circular_angle=False,
        )

        connectivity = env.connectivity
        angles = env.bin_centers[:, 1]
        first_angle_bin = np.argmin(angles)
        last_angle_bin = np.argmax(angles)

        # They should NOT be neighbors
        assert not connectivity.has_edge(first_angle_bin, last_angle_bin)

    def test_circular_wrapping_at_multiple_distances(self):
        """Test circular wrapping works correctly at each distance ring."""
        distance_range = (0.0, 20.0)  # 2 distance bins
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0  # 2 bins
        angle_bin_size = np.pi / 2  # 4 bins

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        # Group bins by distance
        distances = env.bin_centers[:, 0]
        unique_distances = np.unique(distances)

        for d in unique_distances:
            # Get bins at this distance
            mask = np.isclose(distances, d)
            bin_indices = np.where(mask)[0]
            angles_at_d = env.bin_centers[bin_indices, 1]

            # Find first and last angle bins at this distance
            first_idx = bin_indices[np.argmin(angles_at_d)]
            last_idx = bin_indices[np.argmax(angles_at_d)]

            # They should be connected
            assert env.connectivity.has_edge(first_idx, last_idx)

    def test_circular_edges_have_required_attributes(self):
        """Test that circular connectivity edges have all required attributes."""
        distance_range = (0.0, 20.0)
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0
        angle_bin_size = np.pi / 2

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        # Find a circular edge (connects first and last angle at some distance)
        distances = env.bin_centers[:, 0]

        # Get bins at first distance ring
        d_val = distances[0]
        mask = np.isclose(distances, d_val)
        bin_indices = np.where(mask)[0]
        angles_at_d = env.bin_centers[bin_indices, 1]

        first_idx = bin_indices[np.argmin(angles_at_d)]
        last_idx = bin_indices[np.argmax(angles_at_d)]

        # Verify edge exists
        assert env.connectivity.has_edge(first_idx, last_idx)

        # Verify all required attributes
        edge_data = env.connectivity[first_idx][last_idx]
        assert "distance" in edge_data
        assert "vector" in edge_data
        assert "angle_2d" in edge_data
        assert "edge_id" in edge_data

        # Verify types
        assert isinstance(edge_data["distance"], float)
        assert isinstance(edge_data["vector"], tuple)
        with pytest.raises(TypeError):
            edge_data["vector"][0] = 99.0
        assert isinstance(edge_data["angle_2d"], float)
        assert isinstance(edge_data["edge_id"], int)

    def test_circular_angle_with_single_angle_bin(self):
        """Test that circular_angle=True with n_angle=1 doesn't create self-loops."""
        distance_range = (0.0, 20.0)
        angle_range = (-np.pi, np.pi)
        distance_bin_size = 10.0
        angle_bin_size = 2 * np.pi  # Single angle bin spanning full circle

        env = Environment.from_polar_egocentric(
            distance_range=distance_range,
            angle_range=angle_range,
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        # Should have no self-loops
        for node in env.connectivity.nodes():
            assert not env.connectivity.has_edge(node, node)


class TestCircularWrapEdgeArcLength:
    """The angular wrap edge must carry arc length, not chord length.

    Regression: the wrap edge connects the first and last angle bins on a
    single distance ring. Both bins sit at the same radius, so the Euclidean
    norm between their centers is the chord across the *full* angular span --
    not the one-``angle_bin_size`` step the seam actually represents. The
    geodesic ``distance`` weight must instead be ``ring_radius *
    angle_bin_size`` (true arc length).
    """

    @staticmethod
    def _wrap_edge(env):
        """Return (ring_radius, first_pos, last_pos, edge_data) for innermost ring."""
        distances = env.bin_centers[:, 0]
        angles = env.bin_centers[:, 1]
        ring_radius = distances.min()
        ring = np.isclose(distances, ring_radius)
        ring_idx = np.flatnonzero(ring)
        first_idx = int(ring_idx[np.argmin(angles[ring_idx])])
        last_idx = int(ring_idx[np.argmax(angles[ring_idx])])
        first_pos = np.array(env.connectivity.nodes[first_idx]["pos"])
        last_pos = np.array(env.connectivity.nodes[last_idx]["pos"])
        return ring_radius, first_pos, last_pos, env.connectivity[first_idx][last_idx]

    def test_wrap_edge_distance_is_arc_length_not_chord(self):
        angle_bin_size = np.pi / 2
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 20.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        ring_radius, first_pos, last_pos, edge_data = self._wrap_edge(env)
        expected_arc = ring_radius * angle_bin_size

        assert edge_data["distance"] == pytest.approx(expected_arc)

        # Guard against a regression to the old behavior: the previous code set
        # ``distance = norm(pos_last - pos_first)`` over the (distance, angle)
        # node positions. Because both bins lie on the same distance ring, that
        # norm collapses to the angular span across the *full* circle minus one
        # step, an entirely different quantity from the one-step arc.
        old_buggy_value = float(np.linalg.norm(last_pos - first_pos))
        assert edge_data["distance"] != pytest.approx(old_buggy_value)

    def test_wrap_edge_arc_length_scales_with_ring_radius(self):
        """Outer rings have longer wrap arcs (arc = radius * angle_bin_size)."""
        angle_bin_size = np.pi / 3
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 30.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,  # 3 distance rings
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        distances = env.bin_centers[:, 0]
        angles = env.bin_centers[:, 1]
        observed = {}
        for ring_radius in np.unique(distances):
            ring_idx = np.flatnonzero(np.isclose(distances, ring_radius))
            first_idx = int(ring_idx[np.argmin(angles[ring_idx])])
            last_idx = int(ring_idx[np.argmax(angles[ring_idx])])
            observed[ring_radius] = env.connectivity[first_idx][last_idx]["distance"]

        for ring_radius, dist in observed.items():
            assert dist == pytest.approx(ring_radius * angle_bin_size)

    def test_seam_edge_uses_actual_step_not_requested(self):
        """Seam edge arc length must use the realized angular step.

        When ``angle_bin_size`` does not evenly divide the angular range,
        ``ceil`` binning yields a realized step slightly different from the
        requested one. The regular angular edges use the realized step (via
        ``_fix_polar_edge_geometry``, which reads actual node angles), so the
        seam edge must too; otherwise the seam edge is inconsistent with its
        regular neighbors and biases seam geodesics.

        Fail-before: with ``angle_bin_size=1.0`` over [-pi, pi] the realized
        step is ~0.8976, so a regular angular edge at radius r has distance
        ~r*0.8976, but the seam edge used the requested 1.0 -> ~r*1.0 (e.g.
        4.488 vs 5.0 at r=5).
        """
        angle_bin_size = 1.0  # does NOT evenly divide 2*pi over [-pi, pi]
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=angle_bin_size,
            circular_angle=True,
        )

        # Realized step differs from the requested angle_bin_size.
        angle_edges = env.grid_edges[1]
        actual_step = float(np.diff(angle_edges).mean())
        assert actual_step != pytest.approx(angle_bin_size)

        distances = env.bin_centers[:, 0]
        angles = env.bin_centers[:, 1]

        # Pick a single (non-degenerate) distance ring.
        ring_radius = np.unique(distances)[1]
        ring_idx = np.flatnonzero(np.isclose(distances, ring_radius))
        ring_idx = ring_idx[np.argsort(angles[ring_idx])]

        # Seam edge: first and last angle bin on the ring.
        first_idx, last_idx = int(ring_idx[0]), int(ring_idx[-1])
        seam_dist = env.connectivity[first_idx][last_idx]["distance"]

        # Regular angular edge: two adjacent angle bins in the middle of the
        # same ring (same radius, so radial component is ~0).
        mid = len(ring_idx) // 2
        reg_a, reg_b = int(ring_idx[mid - 1]), int(ring_idx[mid])
        reg_dist = env.connectivity[reg_a][reg_b]["distance"]

        # Seam edge and regular angular edge at the same radius must match
        # (both use the realized step).
        assert seam_dist == pytest.approx(reg_dist)

        # And both equal radius * actual_step, NOT radius * requested step.
        assert seam_dist == pytest.approx(ring_radius * actual_step)
        assert seam_dist != pytest.approx(ring_radius * angle_bin_size)


class TestAngularCircularDerivedFromGraph:
    """``_angular_is_circular`` is derived from the connectivity graph.

    It must reflect the ``circular_angle`` build flag and survive every
    serialization path that persists the connectivity graph
    (``to_file``/``from_file``, ``copy``), so a reloaded env does not silently
    revert to the wrong wrapping behavior in polar gaussian_kde smoothing.
    """

    @staticmethod
    def _env(circular):
        return Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 6,  # 12 angular bins
            circular_angle=circular,
        )

    def test_circular_true_detected(self):
        assert self._env(True)._angular_is_circular is True

    def test_circular_false_detected(self):
        assert self._env(False)._angular_is_circular is False

    def test_round_trips_through_file(self, tmp_path):
        for circular in (True, False):
            env = self._env(circular)
            path = str(tmp_path / f"polar_{circular}")
            env.to_file(path)
            loaded = Environment.from_file(path)
            assert loaded._angular_is_circular is circular

    def test_survives_derivation(self):
        for circular in (True, False):
            assert self._env(circular).with_metadata()._angular_is_circular is circular


class TestPolarCacheFreshness:
    """Polar geometry is complete before construction-time caches are populated."""

    def test_geometry_is_complete_at_layout_setup(self, monkeypatch):
        """Initial setup already sees physical distances and circular faces."""
        from environment_helpers import record_publications

        from neurospatial import Environment

        published = record_publications(monkeypatch)
        polar_env = Environment.from_polar_egocentric(
            distance_range=(0.0, 10.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
            circular_angle=True,
        )

        assert published == [polar_env]
        assert polar_env.connectivity.has_edge(0, 3)
        assert polar_env.connectivity.edges[0, 1]["distance"] == pytest.approx(
            5.0 * np.pi / 2
        )

    def test_differential_operator_reflects_circular_edges(self):
        """The cached operator includes circular wrap edges from the owned graph.

        ``get_differential_operator`` returns a cached operator from
        shape ``(n_bins, n_edges)``. Its edge (column) dimension must match the
        connectivity graph's edge count including the circular wrap edges.
        """
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 10.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
            circular_angle=True,
        )

        n_edges = env.connectivity.number_of_edges()
        # The wrap edge is present in the graph...
        angles = env.bin_centers[:, 1]
        first_idx = int(np.argmin(angles))
        last_idx = int(np.argmax(angles))
        assert env.connectivity.has_edge(first_idx, last_idx)

        # ...and the versioned cached differential operator reflects it. The
        # operator has shape (n_bins, n_edges).
        D = env.get_differential_operator()
        assert D.shape == (env.n_bins, n_edges)
        # The edge-attribute cache (also versioned) agrees, too.
        assert len(env.get_edge_attributes()) == n_edges


class TestConnectDiagonalNeighborsParameter:
    """from_polar_egocentric exposes connect_diagonal_neighbors (default True)."""

    def test_default_connects_diagonals(self):
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 20.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )
        env_explicit = Environment.from_polar_egocentric(
            distance_range=(0.0, 20.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
            connect_diagonal_neighbors=True,
        )
        assert (
            env.connectivity.number_of_edges()
            == env_explicit.connectivity.number_of_edges()
        )

    def test_disabling_diagonals_reduces_edge_count(self):
        """An 8-connected grid has more edges than a 4-connected one."""
        diag = Environment.from_polar_egocentric(
            distance_range=(0.0, 20.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
            circular_angle=False,
            connect_diagonal_neighbors=True,
        )
        no_diag = Environment.from_polar_egocentric(
            distance_range=(0.0, 20.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
            circular_angle=False,
            connect_diagonal_neighbors=False,
        )
        assert (
            no_diag.connectivity.number_of_edges() < diag.connectivity.number_of_edges()
        )


class TestSeamDiagonalConnectivity:
    """Diagonal connectivity must cross the ±π seam (isotropic topology).

    Regression: when ``connect_diagonal_neighbors=True`` and the angular axis
    is circular, interior diagonal edges and same-ring seam edges were added,
    but the *diagonal* edges across the ±π seam were missing. That made the
    seam anisotropic -- it had fewer diagonal connections than every interior
    angular boundary. These tests pin the seam diagonals and the isotropy.

    Node indexing is row-major: ``node_id = distance_idx * n_angle +
    angle_idx``. With 2 distance rings x 4 angle bins, ring 0 is nodes 0-3 and
    ring 1 is nodes 4-7, both ordered by increasing angle, so the seam (between
    last angle index 3 and first angle index 0) diagonals are 3-4 and 0-7.
    """

    @staticmethod
    def _make(circular_angle, connect_diagonal_neighbors=True):
        # 2 distance rings (radii 5, 15) x 4 angle bins.
        return Environment.from_polar_egocentric(
            distance_range=(0.0, 20.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
            circular_angle=circular_angle,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
        )

    @staticmethod
    def _ring_nodes_sorted_by_angle(env):
        """Return list of per-ring node indices, each sorted by angle."""
        distances = env.bin_centers[:, 0]
        angles = env.bin_centers[:, 1]
        rings = []
        for d in np.unique(distances):
            idx = np.where(np.isclose(distances, d))[0]
            rings.append(idx[np.argsort(angles[idx])])
        return rings

    def test_seam_diagonal_edges_exist(self):
        """Seam diagonals (reviewer pairs 3-4 and 0-7) must be present."""
        env = self._make(circular_angle=True)
        c = env.connectivity

        # Verify the concrete node numbering first (2x4, row-major by angle).
        rings = self._ring_nodes_sorted_by_angle(env)
        assert [list(r) for r in rings] == [[0, 1, 2, 3], [4, 5, 6, 7]]

        inner_first, inner_last = int(rings[0][0]), int(rings[0][-1])
        outer_first, outer_last = int(rings[1][0]), int(rings[1][-1])

        # (r_inner, last_angle) <-> (r_outer, first_angle): nodes 3-4
        assert c.has_edge(inner_last, outer_first)
        assert (inner_last, outer_first) == (3, 4)
        # (r_inner, first_angle) <-> (r_outer, last_angle): nodes 0-7
        assert c.has_edge(inner_first, outer_last)
        assert (inner_first, outer_last) == (0, 7)

    def test_seam_diagonal_degree_matches_interior(self):
        """The seam has the same diagonal connectivity as an interior step."""
        env = self._make(circular_angle=True)
        c = env.connectivity
        rings = self._ring_nodes_sorted_by_angle(env)
        inner, outer = rings[0], rings[1]
        n_angle = len(inner)

        def n_cross_diagonals(a0, a1):
            # Diagonal edges crossing the angular boundary between angle
            # indices a0 and a1: (inner[a0]-outer[a1]) and (inner[a1]-outer[a0]).
            count = 0
            if c.has_edge(int(inner[a0]), int(outer[a1])):
                count += 1
            if c.has_edge(int(inner[a1]), int(outer[a0])):
                count += 1
            return count

        # Interior angular boundary (between angle index 0 and 1).
        interior = n_cross_diagonals(0, 1)
        # Seam angular boundary (between last angle and first angle).
        seam = n_cross_diagonals(n_angle - 1, 0)

        assert interior == 2  # both diagonal directions present interior
        assert seam == interior  # seam is now isotropic

    def test_circular_false_adds_no_seam_edges(self):
        """circular_angle=False adds NO seam edges (diagonal or same-ring)."""
        env = self._make(circular_angle=False)
        c = env.connectivity
        rings = self._ring_nodes_sorted_by_angle(env)
        inner, outer = rings[0], rings[1]

        # No same-ring seam edge (first<->last angle on a ring).
        assert not c.has_edge(int(inner[0]), int(inner[-1]))
        assert not c.has_edge(int(outer[0]), int(outer[-1]))
        # No diagonal seam edges across the ±π boundary.
        assert not c.has_edge(int(inner[-1]), int(outer[0]))
        assert not c.has_edge(int(inner[0]), int(outer[-1]))

    def test_seam_diagonal_distance_is_polar_diagonal_length(self):
        """Seam diagonal weight is sqrt(Δr² + (r̄·Δθ)²), matching interior."""
        env = self._make(circular_angle=True)
        c = env.connectivity
        rings = self._ring_nodes_sorted_by_angle(env)
        inner, outer = rings[0], rings[1]

        r_inner = float(env.bin_centers[int(inner[0]), 0])
        r_outer = float(env.bin_centers[int(outer[0]), 0])
        d_r = r_outer - r_inner
        r_mean = 0.5 * (r_inner + r_outer)
        actual_step = float(np.pi / 2)  # evenly divides 2π into 4 bins
        expected = float(np.hypot(d_r, r_mean * actual_step))

        seam_diag = c[int(inner[-1])][int(outer[0])]["distance"]
        assert seam_diag == pytest.approx(expected)

        # Should match an interior diagonal weight crossing one angular step.
        interior_diag = c[int(inner[0])][int(outer[1])]["distance"]
        assert seam_diag == pytest.approx(interior_diag)


class TestParameterValidation:
    """Test parameter validation for from_polar_egocentric()."""

    def test_raises_on_negative_distance_bin_size(self):
        """Test that negative distance_bin_size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Environment.from_polar_egocentric(
                distance_range=(0.0, 100.0),
                angle_range=(-np.pi, np.pi),
                distance_bin_size=-10.0,
                angle_bin_size=np.pi / 4,
            )

    def test_raises_on_negative_angle_bin_size(self):
        """Test that negative angle_bin_size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Environment.from_polar_egocentric(
                distance_range=(0.0, 100.0),
                angle_range=(-np.pi, np.pi),
                distance_bin_size=10.0,
                angle_bin_size=-np.pi / 4,
            )

    def test_raises_on_zero_distance_bin_size(self):
        """Test that zero distance_bin_size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Environment.from_polar_egocentric(
                distance_range=(0.0, 100.0),
                angle_range=(-np.pi, np.pi),
                distance_bin_size=0.0,
                angle_bin_size=np.pi / 4,
            )

    def test_raises_on_invalid_distance_range(self):
        """Test that distance_range[0] >= distance_range[1] raises ValueError."""
        with pytest.raises(ValueError, match="min < max"):
            Environment.from_polar_egocentric(
                distance_range=(100.0, 0.0),  # Inverted range
                angle_range=(-np.pi, np.pi),
                distance_bin_size=10.0,
                angle_bin_size=np.pi / 4,
            )

    def test_raises_on_invalid_angle_range(self):
        """Test that angle_range[0] >= angle_range[1] raises ValueError."""
        with pytest.raises(ValueError, match="min < max"):
            Environment.from_polar_egocentric(
                distance_range=(0.0, 100.0),
                angle_range=(np.pi, -np.pi),  # Inverted range
                distance_bin_size=10.0,
                angle_bin_size=np.pi / 4,
            )

    def test_raises_on_equal_distance_range(self):
        """Test that equal distance_range raises ValueError."""
        with pytest.raises(ValueError, match="min < max"):
            Environment.from_polar_egocentric(
                distance_range=(100.0, 100.0),  # Equal bounds
                angle_range=(-np.pi, np.pi),
                distance_bin_size=10.0,
                angle_bin_size=np.pi / 4,
            )


class TestEnvironmentMetadata:
    """Test that environment metadata is set correctly."""

    def test_name_parameter(self):
        """Test that name parameter is set correctly."""
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 100.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 4,
            name="test_polar_env",
        )

        assert env.name == "test_polar_env"


class TestDocstringNote:
    """Test that environment has correct coordinate convention documentation."""


class TestPolarCoordinateKindFlag:
    """Regression: polar envs are a distinct type from Cartesian.

    Downstream code that computes Euclidean distance on `bin_centers`
    would silently treat `(distance, angle)` pairs as `(x, y)` and produce
    meaningless numbers. Polar environments are now a separate type
    (``Environment``, a sibling of ``Environment``, not a
    subclass) so the geometry is carried by the type itself and the
    Cartesian-only methods are simply unavailable.
    """

    @pytest.fixture
    def polar_env(self):
        """A small polar env: 5 distance bins, 4 angle bins."""
        return Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )

    @pytest.fixture
    def cartesian_env(self):
        """A small Cartesian env for the negative-control side of every test."""
        positions = np.column_stack([np.linspace(0, 30, 16), np.linspace(0, 30, 16)])
        return Environment.from_samples(positions, bin_size=2.0)

    def test_from_polar_egocentric_returns_polar_type(self, polar_env):
        """from_polar_egocentric returns the distinct Environment."""
        from neurospatial import Environment

        assert isinstance(polar_env, Environment)
        assert polar_env.coordinate_kind == "polar"
        assert polar_env._is_polar is True

    def test_cartesian_factories_default_to_cartesian(self, cartesian_env):
        """from_samples / from_polygon / from_grid_mask / etc. produce Cartesian envs."""
        from neurospatial import Environment

        assert isinstance(cartesian_env, Environment)
        assert cartesian_env.coordinate_kind == "cartesian"
        assert cartesian_env._is_polar is False


class TestPolarRaisesOnCartesianAssumingMethods:
    """Methods that interpret inputs as (x, y) must reject polar envs.

    bin_at and distance_to(metric='euclidean') compute Euclidean operations
    on bin_centers; running them against a polar env's (distance, angle) bin
    centers produces silently wrong numbers. They should fail at the boundary
    instead.

    Methods that operate purely on the connectivity graph (path_between,
    reachable_from, neighbors, distance_between, distance_to(metric='geodesic'))
    are safe on polar envs and must keep working.
    """

    @pytest.fixture
    def polar_env(self):
        return Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )

    def test_bin_at_raises_on_polar_env(self, polar_env):
        from_xy = np.array([[5.0, 5.0]])
        with pytest.raises(NotImplementedError, match=r"polar.*bin_at|bin_at.*polar"):
            polar_env.bin_at(from_xy)

    def test_distance_between_follows_the_polar_graph(self, polar_env):
        """Bins or (distance, angle) points give the graph path length."""
        start, end = 0, polar_env.n_bins - 1
        expected = nx.shortest_path_length(
            polar_env.connectivity, start, end, weight="distance"
        )
        assert polar_env.distance_between(bins=[start, end]) == pytest.approx(expected)
        points = polar_env.bin_centers[[start, end]] + 0.01
        assert polar_env.distance_between(points=points) == pytest.approx(expected)

    def test_distance_to_euclidean_raises_on_polar_env(self, polar_env):
        with pytest.raises(
            NotImplementedError,
            match=r"polar.*distance_to.*euclidean|distance_to.*euclidean.*polar",
        ):
            polar_env.distance_to(bins=[0], metric="euclidean")

    def test_distance_to_geodesic_works_on_polar_env(self, polar_env):
        """Geodesic distance respects connectivity, so it's well-defined on polar."""
        distances = polar_env.distance_to(bins=[0], metric="geodesic")
        assert distances.shape == (polar_env.n_bins,)
        # Distance from a bin to itself is zero; everything else is positive.
        assert distances[0] == 0.0
        assert (distances[1:] >= 0).all()

    def test_neighbors_works_on_polar_env(self, polar_env):
        """neighbors only reads the connectivity graph; should not raise."""
        nbrs = polar_env.neighbors(bins=0)
        assert isinstance(nbrs, list)

    # --- Inherited Cartesian-coordinate / Cartesian-grid ops must also raise.

    def test_interpolate_raises_on_polar_env(self, polar_env):
        """interpolate bypasses bin_at and previously returned a silent array.

        Fail-before: prior to the fix this returned geometric nonsense
        (an ndarray) instead of raising on (distance, angle) coordinates.
        """
        field = np.arange(polar_env.n_bins, dtype=float)
        points = np.array([[5.0, 0.0], [25.0, 1.0]])
        with pytest.raises(
            NotImplementedError, match=r"polar.*interpolate|interpolate.*polar"
        ):
            polar_env.interpolate(field, points, mode="nearest")

    def test_occupancy_raises_on_polar_env(self, polar_env):
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[5.0, 0.0], [10.0, 0.5], [15.0, 1.0]])
        with pytest.raises(
            NotImplementedError, match=r"polar.*occupancy|occupancy.*polar"
        ):
            polar_env.occupancy(times, positions)

    def test_bin_sequence_raises_on_polar_env(self, polar_env):
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[5.0, 0.0], [10.0, 0.5], [15.0, 1.0]])
        with pytest.raises(
            NotImplementedError, match=r"polar.*bin_sequence|bin_sequence.*polar"
        ):
            polar_env.bin_sequence(times, positions)

    def test_bin_sequence_with_runs_raises_on_polar_env(self, polar_env):
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[5.0, 0.0], [10.0, 0.5], [15.0, 1.0]])
        with pytest.raises(
            NotImplementedError,
            match=r"polar.*bin_sequence_with_runs|bin_sequence_with_runs.*polar",
        ):
            polar_env.bin_sequence_with_runs(times, positions)

    def test_to_linear_raises_on_polar_env(self, polar_env):
        with pytest.raises(
            NotImplementedError, match=r"polar.*to_linear|to_linear.*polar"
        ):
            polar_env.to_linear(np.array([[5.0, 0.0]]))

    def test_linear_to_nd_raises_on_polar_env(self, polar_env):
        with pytest.raises(
            NotImplementedError, match=r"polar.*linear_to_nd|linear_to_nd.*polar"
        ):
            polar_env.linear_to_nd(np.array([5.0]))

    def test_rebin_raises_on_polar_env(self, polar_env):
        with pytest.raises(NotImplementedError, match=r"polar.*rebin|rebin.*polar"):
            polar_env.rebin(2)

    def test_subset_raises_on_polar_env(self, polar_env):
        mask = np.ones(polar_env.n_bins, dtype=bool)
        with pytest.raises(NotImplementedError, match=r"polar.*subset|subset.*polar"):
            polar_env.subset(bins=mask)

    # --- Graph operations remain valid on polar and must not regress.

    def test_path_between_works_on_polar_env(self, polar_env):
        path = polar_env.path_between(bins=[0, polar_env.n_bins - 1])
        assert isinstance(path, list)
        assert path[0] == 0 and path[-1] == polar_env.n_bins - 1

    def test_reachable_from_works_on_polar_env(self, polar_env):
        reachable = polar_env.reachable_from(bins=0)
        assert reachable.shape == (polar_env.n_bins,)
        assert bool(reachable[0]) is True

    def test_smooth_works_on_polar_env(self, polar_env):
        field = np.zeros(polar_env.n_bins, dtype=float)
        field[0] = 1.0
        smoothed = polar_env.smooth(field, bandwidth=5.0)
        assert smoothed.shape == (polar_env.n_bins,)

    def test_repr_shows_polar_class_name(self, polar_env):
        """repr must identify the concrete polar type, not 'Environment'."""
        assert "Environment" in repr(polar_env)

    @pytest.mark.parametrize("mode", ["deepcopy", "pickle"])
    def test_copy_and_pickle_preserve_polar_refusals(self, polar_env, mode):
        import copy
        import pickle

        restored = (
            copy.deepcopy(polar_env)
            if mode == "deepcopy"
            else pickle.loads(pickle.dumps(polar_env))
        )
        assert type(restored) is type(polar_env)
        np.testing.assert_array_equal(restored.bin_centers, polar_env.bin_centers)
        assert not hasattr(restored, "bin_at")
        with pytest.raises(CartesianOnlyError):
            _ = restored.bin_at

    def test_unknown_attribute_preserves_standard_attribute_error(self, polar_env):
        with pytest.raises(AttributeError) as caught:
            _ = polar_env.bin_att
        assert type(caught.value) is AttributeError
        assert caught.value.name == "bin_att"
        assert caught.value.obj is polar_env

    @pytest.mark.parametrize(
        "call, operation",
        [
            (
                lambda env: env.transitions(
                    times=np.array([0.0, 0.1]),
                    positions=np.array([[5.0, 0.0], [5.0, 0.0]]),
                ),
                "transitions() from times and positions",
            ),
            (
                lambda env: env.distance_to(bins=[0], metric="euclidean"),
                "distance_to(metric='euclidean')",
            ),
        ],
    )
    def test_refuses_cartesian_modes_of_shared_methods(
        self, polar_env, call, operation
    ):
        region_builder = Regions(polar_env.regions.values())
        region_builder.add("p", point=(25.0, 0.0))
        polar_env = polar_env.with_regions(region_builder)
        with pytest.raises(CartesianOnlyError) as caught:
            call(polar_env)
        assert str(caught.value).startswith(f"Environment.{operation} is not available")
        assert "()()" not in str(caught.value)

    def test_points_and_point_regions_use_distance_and_angle(self, polar_env):
        """Polar points and point regions are read in bin-center coordinates."""
        region_builder = Regions(polar_env.regions.values())
        region_builder.add("p", point=(25.0, 0.1))
        polar_env = polar_env.with_regions(region_builder)
        (covered,) = polar_env.bins_in_region("p")
        distance, angle = polar_env.bin_centers[covered]
        assert distance == pytest.approx(25.0)
        assert 0.0 <= angle < np.pi / 2
        np.testing.assert_array_equal(
            np.flatnonzero(polar_env.region_mask("p")), [covered]
        )
        np.testing.assert_array_equal(
            np.flatnonzero(polar_env.region_membership()[:, 0]), [covered]
        )
        assert polar_env.neighbors(points=[[25.0, 0.1]]) == polar_env.neighbors(
            bins=covered
        )
        np.testing.assert_array_equal(
            polar_env.distance_to(region="p"), polar_env.distance_to(bins=covered)
        )


class TestPolarSerializationRoundTrip:
    """The polar type must survive copy and to_file/from_file round-trips."""

    def test_derived_environment_preserves_coordinate_kind(self):
        from neurospatial import Environment

        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )
        env_copy = env.with_metadata(name="derived")
        assert isinstance(env_copy, Environment)
        assert env_copy._is_polar is True

    def test_to_file_from_file_preserves_coordinate_kind(self, tmp_path):
        from neurospatial import Environment
        from neurospatial.io.files import from_file, to_file

        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )
        path = tmp_path / "polar_env"
        to_file(env, path)
        loaded = from_file(path)
        assert isinstance(loaded, Environment)
        assert loaded._is_polar is True

    def test_polar_layout_type_is_its_own_and_survives_storage(self, tmp_path):
        """Polar sectors are not described as a Cartesian masked grid."""
        from neurospatial.io import from_dict, from_file, to_dict, to_file

        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )
        to_file(env, tmp_path / "polar_env")
        for restored in (
            env,
            from_dict(to_dict(env)),
            from_file(tmp_path / "polar_env"),
        ):
            assert restored.layout_type == "Polar"
            assert restored._provenance.category == "polar"
            assert restored._binning.kind == "Polar"
            assert repr(restored).endswith("20 bins, Polar)")


class TestPolarBoundaryGuardsExtended:
    """Regression coverage for review-found polar holes.

    Adds boundary guards for code paths the original polar-coordinate-kind
    commit missed: ``Environment.contains`` (was bypassing the polar check),
    ``Environment.plot_field`` (was silently mislabeling axes as
    X/Y), ``apply_transform`` (was silently flipping the env back to
    Cartesian), and the dict-round-trip (``from_dict`` was discarding
    ``coordinate_kind`` even though ``to_dict`` wrote it).
    """

    @pytest.fixture
    def polar_env(self):
        return Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 2,
        )

    def test_contains_raises_on_polar_env(self, polar_env):
        """contains is the boolean partner of bin_at and is unavailable on polar."""
        with pytest.raises(
            NotImplementedError, match=r"polar.*contains|contains.*polar"
        ):
            polar_env.contains(np.array([[5.0, 5.0]]))

    def test_plot_field_relabels_polar_axes(self, polar_env):
        """plot_field on a polar env labels axes Distance / Angle (rad), not X/Y."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        field = np.zeros(polar_env.n_bins)
        fig, ax = plt.subplots()
        try:
            polar_env.plot_field(field, ax=ax)
            assert "Distance" in ax.get_xlabel()
            assert "Angle" in ax.get_ylabel()
            assert "X Position" not in ax.get_xlabel()
            assert "Y Position" not in ax.get_ylabel()
        finally:
            plt.close(fig)

    def test_apply_transform_refuses_polar_env(self, polar_env):
        """apply_transform_to_environment must reject polar envs.

        An affine transform on (distance, angle) bin centers is not
        geometrically meaningful. The free-function path guards against a
        polar env being passed directly (accessing its absent Cartesian
        method raises before reaching the free function).
        """
        from neurospatial.ops.transforms import (
            apply_transform_to_environment,
            translate,
        )

        T = translate(1.0, 1.0)
        with pytest.raises(ValueError, match=r"polar") as caught:
            apply_transform_to_environment(polar_env, T)
        assert str(caught.value) == (
            "apply_transform requires a Cartesian environment but got an "
            "egocentric polar environment (an affine transform on "
            "(distance, angle) pairs is not geometrically meaningful). For "
            "polar envs, work in the egocentric coordinate space directly."
        )

    def test_apply_transform_method_refuses_polar_env(self, polar_env):
        """The apply_transform method on a polar env raises NotImplementedError."""
        from neurospatial.ops.transforms import translate

        T = translate(1.0, 1.0)
        with pytest.raises(
            NotImplementedError, match=r"polar.*apply_transform|apply_transform.*polar"
        ):
            polar_env.apply_transform(T)

    def test_from_dict_round_trip_preserves_coordinate_kind(self, polar_env):
        """to_dict / from_dict must round-trip coordinate_kind for polar envs.

        to_dict already writes the field; from_dict was discarding it,
        so an in-memory dict round-trip silently flipped polar envs to
        Cartesian even though to_file/from_file round-tripped correctly.
        """
        from neurospatial import Environment
        from neurospatial.io.files import from_dict, to_dict

        data = to_dict(polar_env)
        loaded = from_dict(data)
        assert isinstance(loaded, Environment)
        assert loaded._is_polar is True


class TestSubsetPreservesGraphLayout:
    """Regression for the review-found graph corruption.

    A 1-D linearized track (Graph layout) embedded in 2-D space
    populates ``active_mask``, ``grid_shape``, and ``grid_edges`` for
    the linearized 1-D representation -- so the original
    grid-fast-path check (looking only at those three attributes)
    routed graph parents through ``Environment.from_grid_mask``, which
    rebuilt them as MaskedGrid envs with 1-D bin_centers, dropping the
    2-D embedding and flipping ``is_linearized_track`` to False.
    """

    def _build_graph_env(self):
        import networkx as nx

        from neurospatial import Environment

        graph = nx.Graph()
        graph.add_node("A", pos=(0.0, 0.0))
        graph.add_node("B", pos=(10.0, 0.0))
        graph.add_node("C", pos=(10.0, 10.0))
        graph.add_edge("A", "B", distance=10.0)
        graph.add_edge("B", "C", distance=10.0)
        return Environment.from_graph(
            graph,
            edge_order=[("A", "B"), ("B", "C")],
            edge_spacing=0.0,
            bin_size=2.0,
        )

    def test_graph_subset_keeps_is_1d_and_n_dims(self):
        env = self._build_graph_env()
        assert env.is_linearized_track is True and env.n_dims == 2

        mask = np.zeros(env.n_bins, dtype=bool)
        mask[:5] = True
        sub = env.subset(bins=mask)

        assert sub.is_linearized_track is True, "graph subset must remain 1-D"
        assert sub.n_dims == 2, "graph subset must keep its 2-D embedding"

    def test_graph_subset_does_not_use_maskedgrid(self):
        """Graph subset stays on the inline SubsetLayout fallback, not MaskedGrid.

        MaskedGrid would flatten the 2-D embedding to 1-D bin_centers.
        """
        env = self._build_graph_env()
        mask = np.zeros(env.n_bins, dtype=bool)
        mask[:5] = True
        sub = env.subset(bins=mask)
        assert sub.layout_type != "MaskedGrid"


class TestPolarEgocentricAngularSeam:
    """Distance-field topology across the angular ±pi seam.

    The factory takes ``distance_bin_size`` / ``angle_bin_size`` (not
    ``n_distance_bins`` / ``n_direction_bins``). To probe the angular topology
    in isolation, these tests measure graph-hop distance (a unit-weighted
    distance field): the weighted geodesic ``distance`` does not distinguish
    the two cases because the seam edge carries the full chord length as its
    weight, while the seam edge's mere existence (hop = 1) is the load-bearing
    property of ``circular_angle=True``.
    """

    @staticmethod
    def _build(circular_angle):
        # distance_bin_size=5 over (0, 50) -> 10 distance bins;
        # angle_bin_size=pi/6 over (-pi, pi) -> 12 angle bins.
        return Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=5.0,
            angle_bin_size=np.pi / 6,
            circular_angle=circular_angle,
        )

    @staticmethod
    def _hop_distance_field(env, source_bin):
        from neurospatial.ops.distance import distance_field

        graph = env.connectivity.copy()
        for u, v in graph.edges():
            graph[u][v]["hop"] = 1.0
        return distance_field(graph, sources=[int(source_bin)], weight="hop")

    @staticmethod
    def _ring_bins(env):
        """Indices, angles, and key bins for the innermost distance ring."""
        distances = env.bin_centers[:, 0]
        angles = env.bin_centers[:, 1]
        ring = np.isclose(distances, distances.min())
        ring_idx = np.flatnonzero(ring)
        ring_angles = angles[ring_idx]
        source = int(ring_idx[np.argmax(ring_angles)])  # angle ~ +pi
        neg_pi = int(ring_idx[np.argmin(ring_angles)])  # angle ~ -pi
        zero = int(ring_idx[np.argmin(np.abs(ring_angles))])  # angle ~ 0
        return source, neg_pi, zero

    def test_distance_field_wraps_across_seam(self):
        """With circular_angle=True the -pi bin is closer than the 0 bin."""
        env = self._build(circular_angle=True)
        source, neg_pi, zero = self._ring_bins(env)

        field = self._hop_distance_field(env, source)

        # Source sits near +pi; the seam connects directly to the -pi bin.
        assert field[neg_pi] < field[zero]

    def test_distance_field_non_circular_does_not_wrap(self):
        """With circular_angle=False the -pi bin is farther than the 0 bin."""
        env = self._build(circular_angle=False)
        source, neg_pi, zero = self._ring_bins(env)

        field = self._hop_distance_field(env, source)

        # No seam edge: reaching -pi requires traversing the whole angular span.
        assert field[neg_pi] > field[zero]


class TestPolarTypeAndGeometry:
    """Validation slice for the distinct polar type and physical geometry."""

    @pytest.fixture
    def polar_env(self):
        # 5 distance rings (centers 5,15,25,35,45), 8 angle bins, no wrap so the
        # angular-step edges are unambiguous grid edges.
        return Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 4,
            circular_angle=False,
        )

    def test_polar_env_is_distinct_type(self, polar_env):
        """from_polar_egocentric returns Environment, not Environment."""
        from neurospatial import Environment

        assert type(polar_env) is Environment
        assert isinstance(polar_env, Environment)

        # Cartesian-only methods are absent/clearly errored, not silently disabled.
        with pytest.raises(NotImplementedError):
            polar_env.bin_at(np.array([[1.0, 1.0]]))
        with pytest.raises(NotImplementedError):
            polar_env.contains(np.array([[1.0, 1.0]]))
        with pytest.raises(NotImplementedError):
            polar_env.distance_to(bins=[0], metric="euclidean")

    def test_polar_edge_distances_physical(self, polar_env):
        """Two equal angular moves at different radii have arc length scaling with r.

        Before the geometry fix the edge "distance" was the Euclidean norm of
        (Delta_r, Delta_theta) = (0, pi/4), i.e. a constant ~0.785 at every
        radius -- so arcs at r=5 and r=45 were equal (a ~9x error). With the
        fix the arc length is r * Delta_theta, which scales with radius.
        """
        G = polar_env.connectivity
        # node id = distance_idx * n_angle + angle_idx ; n_angle = 8
        arc_inner = G.edges[0, 1]["distance"]  # ring r_center=5, angular step
        arc_outer = G.edges[32, 33]["distance"]  # ring r_center=45, angular step

        assert arc_inner == pytest.approx(5.0 * np.pi / 4, rel=1e-9)
        assert arc_outer == pytest.approx(45.0 * np.pi / 4, rel=1e-9)
        # Equal physical angular step -> arc scales with radius (45/5 = 9).
        assert arc_outer / arc_inner == pytest.approx(9.0, rel=1e-9)

        # Radial step (same angle, adjacent ring) is the radial distance Delta_r.
        radial = G.edges[0, 8]["distance"]
        assert radial == pytest.approx(10.0, rel=1e-9)

    def test_polar_geodesic_correct(self, polar_env):
        """A known polar geodesic matches the analytic value within tolerance.

        From the innermost ring center (r=5) to the outermost ring center
        (r=45) along the SAME angle, the shortest path traverses four radial
        edges of length Delta_r = 10 each, so the geodesic distance is 40.
        """
        # Same angle (idx 0), distance idx 0 -> distance idx 4: nodes 0 and 32.
        dist = polar_env.distance_to(bins=[0], metric="geodesic")
        # Node 32 is r=45 at the same angle as node 0 (r=5).
        assert dist[32] == pytest.approx(40.0, rel=1e-9)
        # A pure angular hop from node 0 to node 1 equals the inner arc length.
        assert dist[1] == pytest.approx(5.0 * np.pi / 4, rel=1e-9)


# Public members polar inherits unchanged from the shared base, each checked on
# (distance, angle) bins. A new shared-base member fails the test below until it
# is checked on a polar environment and listed here, or moved to
# CartesianCoordinates if it reads bin centers as Cartesian coordinates.
_POLAR_INHERITED_MEMBERS = {
    "nearest-bin lookup in stored radius/angle columns": {"nearest_bin"},
    "graph and diffusion operations on physical polar geometry": {
        "bin_sizes",
        "components",
        "compute_kernel",
        "compute_gaussian_kernel",
        "diffusion_accuracy",
        "get_diffusion_geometry",
        "get_mrf_basis",
        "diffuse",
        "distance_between",
        "distance_to",
        "get_differential_operator",
        "get_divergence_operator",
        "edge_pairs",
        "to_networkx",
        "neighbors",
        "path_between",
        "reachable_from",
        "rings",
        "smooth",
        "transitions",
    },
    "read-only geometry views in (distance, angle) coordinates": {
        "binning",
        "coordinate_system",
        "active_mask",
        "arena",
        "clip_mapping",
        "bin_centers",
        "connectivity",
        "dimension_ranges",
        "grid_edges",
        "grid_shape",
    },
    "bin bookkeeping, metadata and diagnostics": {
        "bin_center_of",
        "clear_cache",
        "frame",
        "get_bin_attributes",
        "get_edge_attributes",
        "info",
        "is_linearized_track",
        "layout_parameters",
        "layout_type",
        "linearization_properties",
        "n_bins",
        "n_dims",
        "name",
        "regions",
        "with_metadata",
        "with_regions",
        "units",
    },
    "serialization, which restores the polar type": {
        "boundary_bins",
        "from_dict",
        "from_file",
        "from_nwb",
        "to_dict",
        "to_file",
        "to_nwb",
    },
    "constructors: only the masked grid builds; others refuse (tested below)": {
        "from_graph",
        "from_bins",
        "combine",
        "from_grid_mask",
        "from_layout",
        "from_pixel_mask",
        "from_polygon",
        "from_samples",
        "linear_track",
        "maze",
        "open_field",
    },
    "regions, evaluated in (distance, angle) coordinates without angle wrap": {
        "bins_in_region",
        "region_mask",
        "region_membership",
    },
    "plots on distance-by-angle axes": {
        "animate_fields",
        "plot",
        "plot_1d",
        "plot_field",
    },
}


def test_owned_metadata_and_graph_exports_preserve_polar_coordinates():
    """Value replacements and graph copies do not interpret angles as lengths."""
    from neurospatial import Environment

    env = Environment.from_polar_egocentric(
        distance_range=(0, 4),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1,
        angle_bin_size=np.pi / 2,
    )
    regions = Regions()
    regions.add("goal", point=tuple(env.bin_centers[0]))
    changed = env.with_metadata(
        name="polar", units="cm", frame="egocentric"
    ).with_regions(regions)
    assert isinstance(changed, Environment)
    assert (changed.name, changed.units, changed.frame) == ("polar", "cm", "egocentric")
    assert len(env.regions) == 0 and "goal" in changed.regions
    np.testing.assert_array_equal(changed.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(changed.edge_pairs, env.edge_pairs)
    graph = changed.to_networkx()
    assert list(graph.edges) == list(changed.connectivity.edges)
    graph.remove_node(0)
    assert changed.connectivity.number_of_nodes() == env.n_bins
    from dataclasses import replace

    from neurospatial.binning import KnownFaces

    with pytest.raises(ValueError, match="cartesian cell measures"):
        Environment.from_bins(
            replace(env._binning, faces=KnownFaces(env._face_geometry))
        )


def test_every_public_member_is_classified_for_polar():
    public = {name for name in dir(Environment) if not name.startswith("_")}
    shared = public - CARTESIAN_OPERATIONS
    classified = set().union(*_POLAR_INHERITED_MEMBERS.values()) | {
        "coordinate_kind",
        "from_polar_egocentric",
    }
    assert shared == classified, f"Review coordinate meaning for {shared ^ classified}"


@pytest.mark.parametrize(
    "build",
    [
        lambda cls: cls.from_samples(np.zeros((4, 2)), bin_size=1.0),
        lambda cls: cls.open_field(np.zeros((4, 2)), bin_size=1.0),
        lambda cls: cls.from_pixel_mask(np.ones((2, 2), dtype=bool), pixel_size=1.0),
        lambda cls: cls.linear_track(endpoints=[[0.0, 0.0], [4.0, 0.0]], bin_size=1.0),
    ],
    ids=["from_samples", "open_field", "from_pixel_mask", "linear_track"],
)
def test_factories_choose_coordinates_independently_of_runtime_class(build):
    from neurospatial import Environment

    result = build(Environment)
    assert type(result) is Environment
    assert not result._is_polar


@pytest.mark.parametrize("circular", [True, False])
@pytest.mark.parametrize("n_angle", [1, 2, 3, 6])
def test_boundary_bins_respect_the_angular_seam(circular, n_angle, tmp_path):
    """A circular angle has no edge: only the inner and outer rings are boundary."""
    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 20.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=2 * np.pi / n_angle,
        circular_angle=circular,
    )
    r, theta = env.bin_centers.T
    edge = np.isclose(r, r.min()) | np.isclose(r, r.max())
    if not circular:
        edge |= np.isclose(theta, theta.min()) | np.isclose(theta, theta.max())
    expected = np.flatnonzero(edge)
    path = tmp_path / "polar"
    env.to_file(path)
    for restored in (
        env,
        env.with_metadata(),
        Environment.from_file(path),
        Environment.from_dict(env.to_dict()),
    ):
        np.testing.assert_array_equal(restored.boundary_bins, expected)
        assert restored._angular_is_circular is circular


def test_cartesian_operations_are_the_capability_methods():
    """The @cartesian_only decorators mark exactly these members and grid needs."""
    assert {
        "apply_transform",
        "component_keys",
        "bin_sources",
        "get_bridge_attributes",
        "bin_at",
        "bin_sequence",
        "bin_sequence_with_runs",
        "clip",
        "contains",
        "interpolate",
        "linear_to_nd",
        "occupancy",
        "rebin",
        "subset",
        "to_linear",
    } == CARTESIAN_OPERATIONS
    assert {"to_linear", "linear_to_nd", "rebin", "subset"} == (
        CARTESIAN_GRID_OPERATIONS
    )
    assert set(dir(Environment)) >= CARTESIAN_OPERATIONS
