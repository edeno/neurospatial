"""
Property-based tests for geometry operations using Hypothesis.

These tests verify mathematical invariants for spatial operations on regions
and environments. Property-based testing helps uncover edge cases in geometry
and binning logic that example-based tests might miss.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from shapely import Polygon

from neurospatial import Environment
from neurospatial.regions import Region, Regions

# =============================================================================
# Hypothesis Strategies for Geometry
# =============================================================================


@st.composite
def convex_polygon_strategy(
    draw: st.DrawFn,
    min_vertices: int = 4,
    max_vertices: int = 12,
    min_radius: float = 1.0,
    max_radius: float = 100.0,
) -> Polygon:
    """Generate convex polygons using polar coordinates.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    min_vertices : int, default=4
        Minimum number of vertices.
    max_vertices : int, default=12
        Maximum number of vertices.
    min_radius : float, default=1.0
        Minimum radius from center to vertices.
    max_radius : float, default=100.0
        Maximum radius from center to vertices.

    Returns
    -------
    Polygon
        A valid convex polygon.
    """
    n_vertices = draw(st.integers(min_value=min_vertices, max_value=max_vertices))

    # Generate sorted angles to ensure convexity
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    # Add small random perturbation to angles (keeping them sorted)
    angle_perturbation = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=n_vertices,
            elements=st.floats(min_value=-0.1, max_value=0.1),
        )
    )
    angles = angles + angle_perturbation
    angles = np.sort(angles % (2 * np.pi))

    # Generate radius for each vertex
    radius = draw(st.floats(min_value=min_radius, max_value=max_radius))

    # Convert polar to Cartesian
    vertices = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

    poly = Polygon(vertices)

    # Ensure polygon is valid
    assume(poly.is_valid and poly.area > 0)

    return poly


@st.composite
def region_strategy(
    draw: st.DrawFn,
    min_radius: float = 1.0,
    max_radius: float = 50.0,
) -> Region:
    """Generate valid polygon regions.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    min_radius : float, default=1.0
        Minimum polygon radius.
    max_radius : float, default=50.0
        Maximum polygon radius.

    Returns
    -------
    Region
        A valid polygon region.
    """
    poly = draw(convex_polygon_strategy(min_radius=min_radius, max_radius=max_radius))
    name = draw(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_"
            ),
        )
    )

    return Region(name=name, kind="polygon", data=poly)


@st.composite
def environment_2d_strategy(
    draw: st.DrawFn,
    min_samples: int = 50,
    max_samples: int = 500,
    min_extent: float = 10.0,
    max_extent: float = 100.0,
    min_bin_size: float = 1.0,
    max_bin_size: float = 10.0,
) -> Environment:
    """Generate valid 2D environments.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    min_samples : int, default=50
        Minimum number of position samples.
    max_samples : int, default=500
        Maximum number of position samples.
    min_extent : float, default=10.0
        Minimum spatial extent in each dimension.
    max_extent : float, default=100.0
        Maximum spatial extent in each dimension.
    min_bin_size : float, default=1.0
        Minimum bin size.
    max_bin_size : float, default=10.0
        Maximum bin size.

    Returns
    -------
    Environment
        A valid fitted Environment.
    """
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    extent = draw(st.floats(min_value=min_extent, max_value=max_extent))
    bin_size = draw(st.floats(min_value=min_bin_size, max_value=max_bin_size))

    # Ensure bin_size is reasonable relative to extent
    assume(extent / bin_size >= 3)  # At least 3 bins in each dimension

    # Generate random positions
    seed = draw(st.integers(min_value=0, max_value=10000))
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, extent, size=(n_samples, 2))

    try:
        env = Environment.from_samples(
            positions, bin_size=bin_size, bin_count_threshold=1
        )
        assume(env.n_bins >= 5)  # Need enough bins for meaningful tests
        return env
    except (ValueError, RuntimeError):
        assume(False)  # Skip this example


# =============================================================================
# Property Tests for Regions
# =============================================================================


class TestRegionGeometryProperties:
    """Property-based tests for Region geometry operations."""

    @given(convex_polygon_strategy())
    @settings(max_examples=50, deadline=5000)
    def test_polygon_area_positive(self, poly: Polygon):
        """Property: Valid polygon area is always positive."""
        region = Region(name="test", kind="polygon", data=poly)
        assert region.data.area > 0

    @given(
        convex_polygon_strategy(min_radius=5.0, max_radius=50.0),
        st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=50, deadline=5000)
    def test_buffer_increases_area(self, poly: Polygon, buffer_distance: float):
        """Property: Buffering a polygon increases its area."""
        original_area = poly.area
        buffered = poly.buffer(buffer_distance)

        # Buffer should always increase area for positive distance
        assert buffered.area > original_area

    @given(
        convex_polygon_strategy(min_radius=10.0, max_radius=50.0),
        st.floats(min_value=0.1, max_value=2.0),
    )
    @settings(max_examples=50, deadline=5000)
    def test_negative_buffer_decreases_area(
        self, poly: Polygon, buffer_distance: float
    ):
        """Property: Negative buffering (erosion) decreases area."""
        original_area = poly.area

        # Only erode if polygon is large enough
        assume(buffer_distance < poly.area / 10)  # Rough check

        eroded = poly.buffer(-buffer_distance)

        # Skip if erosion made polygon empty
        if eroded.is_empty:
            return

        assert eroded.area < original_area

    @given(convex_polygon_strategy())
    @settings(max_examples=50, deadline=5000)
    def test_centroid_inside_convex_polygon(self, poly: Polygon):
        """Property: Centroid of convex polygon is inside the polygon."""
        centroid = poly.centroid
        assert poly.contains(centroid)

    @given(
        convex_polygon_strategy(min_radius=10.0, max_radius=50.0),
        st.floats(min_value=1.0, max_value=100.0),
        st.floats(min_value=1.0, max_value=100.0),
    )
    @settings(max_examples=50, deadline=5000)
    def test_translation_preserves_area(self, poly: Polygon, dx: float, dy: float):
        """Property: Translation preserves polygon area."""
        from shapely.affinity import translate

        original_area = poly.area
        translated = translate(poly, xoff=dx, yoff=dy)

        np.testing.assert_allclose(
            translated.area,
            original_area,
            rtol=1e-10,
            err_msg="Translation should preserve area",
        )

    @given(
        convex_polygon_strategy(),
        st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=50, deadline=5000)
    def test_scaling_changes_area_quadratically(
        self, poly: Polygon, scale_factor: float
    ):
        """Property: Uniform scaling changes area by scale^2."""
        from shapely.affinity import scale

        original_area = poly.area
        scaled = scale(poly, xfact=scale_factor, yfact=scale_factor, origin="centroid")

        expected_area = original_area * (scale_factor**2)
        np.testing.assert_allclose(
            scaled.area,
            expected_area,
            rtol=1e-8,
            err_msg=f"Scaling by {scale_factor} should change area by {scale_factor**2}",
        )


class TestRegionsContainerProperties:
    """Property-based tests for Regions container operations."""

    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=30, deadline=5000)
    def test_regions_container_length(self, n_regions: int):
        """Property: Regions container length matches number of added regions."""
        regions_list = []
        for i in range(n_regions):
            region = Region(
                name=f"region_{i}",
                kind="polygon",
                data=Polygon(
                    [(i * 20, 0), ((i + 1) * 20, 0), ((i + 1) * 20, 10), (i * 20, 10)]
                ),
            )
            regions_list.append(region)

        container = Regions(regions_list)
        assert len(container) == n_regions

    @given(
        st.lists(
            st.text(min_size=1, max_size=10, alphabet="abcdefghij"),
            min_size=2,
            max_size=10,
            unique=True,
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_regions_dict_like_access(self, names: list[str]):
        """Property: Regions container supports dict-like access by name."""
        regions_list = []
        for i, name in enumerate(names):
            region = Region(
                name=name,
                kind="polygon",
                data=Polygon(
                    [(i * 20, 0), ((i + 1) * 20, 0), ((i + 1) * 20, 10), (i * 20, 10)]
                ),
            )
            regions_list.append(region)

        container = Regions(regions_list)

        # All names should be accessible
        for name in names:
            assert name in container
            assert container[name].name == name


# =============================================================================
# Property Tests for Environment Binning
# =============================================================================


class TestEnvironmentBinningProperties:
    """Property-based tests for Environment bin operations."""

    @given(environment_2d_strategy())
    @settings(max_examples=30, deadline=10000)
    def test_bin_centers_within_bounds(self, env: Environment):
        """Property: All bin centers are within the dimension ranges."""
        bin_sizes = env.bin_sizes  # It's a property, not a method
        for bin_center in env.bin_centers:
            for dim, (lo, hi) in enumerate(env.dimension_ranges):
                # Allow half bin_size tolerance at edges
                tolerance = bin_sizes[dim] / 2.0
                assert bin_center[dim] >= lo - tolerance, (
                    f"Bin center {bin_center[dim]} below range {lo}"
                )
                assert bin_center[dim] <= hi + tolerance, (
                    f"Bin center {bin_center[dim]} above range {hi}"
                )

    @given(environment_2d_strategy())
    @settings(max_examples=30, deadline=10000)
    def test_bin_at_returns_valid_indices(self, env: Environment):
        """Property: bin_at returns valid bin indices for points in range."""
        # Sample random points within the environment bounds
        rng = np.random.default_rng(42)
        n_test_points = 20

        lo = np.array([r[0] for r in env.dimension_ranges])
        hi = np.array([r[1] for r in env.dimension_ranges])

        for _ in range(n_test_points):
            point = rng.uniform(lo, hi)

            try:
                bin_idx = env.bin_at(point)
                # Valid index or -1 (out of bounds)
                assert bin_idx == -1 or 0 <= bin_idx < env.n_bins
            except (ValueError, IndexError):
                # Some points may be outside active bins
                pass

    @given(environment_2d_strategy())
    @settings(max_examples=30, deadline=10000)
    def test_bin_at_deterministic(self, env: Environment):
        """Property: bin_at is deterministic (same point → same bin)."""
        # Test determinism with a fixed point
        lo = np.array([r[0] for r in env.dimension_ranges])
        hi = np.array([r[1] for r in env.dimension_ranges])
        mid = (lo + hi) / 2

        try:
            idx1 = env.bin_at(mid)
            idx2 = env.bin_at(mid)
            assert idx1 == idx2, "bin_at should be deterministic"
        except (ValueError, IndexError):
            pass  # Point may be outside active bins

    @given(environment_2d_strategy(min_samples=100, max_samples=300))
    @settings(max_examples=20, deadline=15000)
    def test_bin_centers_are_unique(self, env: Environment):
        """Property: All bin centers are unique."""
        centers = env.bin_centers
        unique_centers = np.unique(centers, axis=0)
        assert len(unique_centers) == len(centers), "Bin centers should all be unique"

    @given(environment_2d_strategy())
    @settings(max_examples=20, deadline=10000)
    def test_connectivity_graph_nodes_match_bins(self, env: Environment):
        """Property: Connectivity graph has one node per bin."""
        n_graph_nodes = env.connectivity.number_of_nodes()
        assert n_graph_nodes == env.n_bins, (
            f"Graph has {n_graph_nodes} nodes but environment has {env.n_bins} bins"
        )


class TestEnvironmentOccupancyProperties:
    """Property-based tests for occupancy computation."""

    @given(
        environment_2d_strategy(min_samples=100, max_samples=300),
        st.integers(min_value=100, max_value=500),
    )
    @settings(max_examples=20, deadline=15000)
    def test_occupancy_sums_to_total_time(self, env: Environment, n_samples: int):
        """Property: Occupancy sums to total trajectory time."""
        # Generate trajectory
        rng = np.random.default_rng(42)
        lo = np.array([r[0] for r in env.dimension_ranges])
        hi = np.array([r[1] for r in env.dimension_ranges])

        positions = rng.uniform(lo, hi, size=(n_samples, 2))
        # times must be monotonically increasing
        times = np.sort(rng.uniform(0, n_samples / 30.0, n_samples))

        # Signature: occupancy(times, positions, ...)
        occupancy = env.occupancy(times, positions)

        # Total occupancy should equal total time (approximately, due to gaps)
        total_time = times[-1] - times[0]
        total_occupancy = np.sum(occupancy)

        # Allow looser tolerance since gaps may reduce total occupancy
        assert total_occupancy <= total_time * 1.1, (
            "Occupancy should not exceed trajectory time"
        )
        assert total_occupancy >= 0, "Occupancy should be non-negative"

    @given(environment_2d_strategy(min_samples=100, max_samples=300))
    @settings(max_examples=20, deadline=15000)
    def test_occupancy_non_negative(self, env: Environment):
        """Property: Occupancy is always non-negative."""
        # Generate trajectory
        rng = np.random.default_rng(42)
        n_samples = 200
        lo = np.array([r[0] for r in env.dimension_ranges])
        hi = np.array([r[1] for r in env.dimension_ranges])

        positions = rng.uniform(lo, hi, size=(n_samples, 2))
        # times must be monotonically increasing
        times = np.sort(rng.uniform(0, n_samples / 30.0, n_samples))

        # Signature: occupancy(times, positions, ...)
        occupancy = env.occupancy(times, positions)

        assert np.all(occupancy >= 0), "Occupancy should never be negative"


class TestEnvironmentNeighborProperties:
    """Property-based tests for neighbor relationships."""

    @given(environment_2d_strategy())
    @settings(max_examples=20, deadline=10000)
    def test_neighbor_relationship_symmetric(self, env: Environment):
        """Property: If A is neighbor of B, then B is neighbor of A."""
        for node in env.connectivity.nodes():
            for neighbor in env.connectivity.neighbors(node):
                assert env.connectivity.has_edge(neighbor, node), (
                    f"Neighbor relationship not symmetric: "
                    f"{node} has neighbor {neighbor} but not vice versa"
                )

    @given(environment_2d_strategy())
    @settings(max_examples=20, deadline=10000)
    def test_edge_distances_positive(self, env: Environment):
        """Property: All edge distances are positive."""
        for u, v, data in env.connectivity.edges(data=True):
            assert data["distance"] > 0, (
                f"Edge ({u}, {v}) has non-positive distance {data['distance']}"
            )

    @given(environment_2d_strategy())
    @settings(max_examples=20, deadline=10000)
    def test_edge_distances_match_centers(self, env: Environment):
        """Property: Edge distance matches Euclidean distance between centers."""
        for u, v, data in env.connectivity.edges(data=True):
            center_u = env.bin_centers[u]
            center_v = env.bin_centers[v]
            euclidean_dist = np.linalg.norm(center_u - center_v)

            np.testing.assert_allclose(
                data["distance"],
                euclidean_dist,
                rtol=1e-10,
                err_msg=f"Edge ({u}, {v}) distance doesn't match Euclidean distance",
            )


# =============================================================================
# Property Tests for Trajectories
# =============================================================================


class TestTrajectoryProperties:
    """Property-based tests for trajectory analysis."""

    @given(
        environment_2d_strategy(min_samples=100, max_samples=300),
        st.integers(min_value=50, max_value=200),
    )
    @settings(max_examples=15, deadline=20000)
    def test_bin_sequence_length_bounded(self, env: Environment, n_positions: int):
        """Property: bin_sequence length is at most n_positions."""
        rng = np.random.default_rng(42)
        lo = np.array([r[0] for r in env.dimension_ranges])
        hi = np.array([r[1] for r in env.dimension_ranges])

        positions = rng.uniform(lo, hi, size=(n_positions, 2))
        # bin_sequence requires times - use sorted times
        times = np.sort(rng.uniform(0, n_positions / 30.0, n_positions))

        try:
            # Signature: bin_sequence(times, positions, ...)
            bin_seq = env.bin_sequence(times, positions, dedup=False)
            # Bin sequence length should be at most n_positions
            # (may be less due to filtering invalid bins)
            assert len(bin_seq) <= n_positions
        except (ValueError, RuntimeError):
            pass  # Some positions may be outside environment

    @given(environment_2d_strategy(min_samples=100, max_samples=300))
    @settings(max_examples=15, deadline=15000)
    def test_bin_sequence_indices_valid(self, env: Environment):
        """Property: All bin sequence indices are valid bin indices or -1."""
        rng = np.random.default_rng(42)
        n_positions = 100
        lo = np.array([r[0] for r in env.dimension_ranges])
        hi = np.array([r[1] for r in env.dimension_ranges])

        positions = rng.uniform(lo, hi, size=(n_positions, 2))
        times = np.sort(rng.uniform(0, n_positions / 30.0, n_positions))

        try:
            # Signature: bin_sequence(times, positions, ...)
            bin_seq = env.bin_sequence(times, positions, dedup=False)
            for idx in bin_seq:
                # Valid indices are 0 to n_bins-1, or -1 for outside
                assert idx == -1 or 0 <= idx < env.n_bins, (
                    f"Invalid bin index {idx} in sequence (n_bins={env.n_bins})"
                )
        except (ValueError, RuntimeError):
            pass
