"""Performance regression tests for neurospatial.

This module contains performance benchmarks to prevent regressions in
critical operations. Tests are marked with @pytest.mark.slow and can be
run separately with: uv run pytest -m slow

Benchmarks cover:
- region_membership() scaling with number of regions
- Large environment creation
- Spatial query performance
"""

import time

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment


@pytest.mark.slow
class TestRegionMembershipPerformance:
    """Benchmark region_membership() performance.

    These tests verify that region_membership() scales efficiently
    with the number of regions, particularly testing the optimization
    of hoisting shapely_points() conversion outside the region loop.
    """

    def test_region_membership_scales_with_regions(self):
        """Test that region_membership() time scales linearly (or better) with regions.

        This benchmark measures the performance of region_membership() with
        varying numbers of regions. The key optimization is that converting
        bin centers to Shapely points should happen ONCE, not once per region.

        Expected behavior after optimization:
        - Time should scale approximately linearly with number of regions
        - Overhead per region should be minimal (<10ms per region)
        - Adding 10 regions shouldn't take >10x the time of 1 region
        """
        # Create a reasonably large environment (1000 bins)
        data = np.array([[i, j] for i in range(32) for j in range(32)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Benchmark with different numbers of regions
        region_counts = [1, 5, 10]
        times = []

        for n_regions in region_counts:
            # Clear existing regions
            for name in list(env.regions.keys()):
                env.regions.remove(name)

            # Add N non-overlapping regions in a grid pattern
            grid_size = int(np.ceil(np.sqrt(n_regions)))
            for i in range(n_regions):
                row = i // grid_size
                col = i % grid_size
                x0 = col * 10
                y0 = row * 10
                x1 = x0 + 8
                y1 = y0 + 8
                env.regions.add(f"region_{i}", polygon=box(x0, y0, x1, y1))

            # Benchmark
            start = time.perf_counter()
            # Run multiple times to get stable measurement
            n_iterations = 10
            for _ in range(n_iterations):
                membership = env.region_membership()
            elapsed = time.perf_counter() - start
            avg_time = elapsed / n_iterations

            times.append(avg_time)

            # Verify correctness
            assert membership.shape == (env.n_bins, n_regions)
            assert membership.dtype == bool

        # Performance checks
        time_1_region = times[0]
        time_5_regions = times[1]
        time_10_regions = times[2]

        # After optimization: 10 regions should take <5x the time of 1 region
        # (Before optimization it would take ~10x due to redundant conversions)
        speedup_ratio_5 = time_5_regions / time_1_region
        speedup_ratio_10 = time_10_regions / time_1_region

        # Key assertion: scaling should be sub-linear or linear, not quadratic
        # With optimization: expect ~1-2x for 5 regions, ~2-5x for 10 regions
        # Without optimization: would be ~5x for 5 regions, ~10x for 10 regions
        assert speedup_ratio_10 < 7.0, (
            f"Performance regression: 10 regions took {speedup_ratio_10:.1f}x "
            f"the time of 1 region (expected <7x with optimization). "
            f"Times: 1={time_1_region * 1000:.2f}ms, 10={time_10_regions * 1000:.2f}ms"
        )

        # Print timing info for manual inspection
        print("\nRegion membership performance:")
        print(f"  1 region:  {time_1_region * 1000:6.2f} ms")
        print(
            f"  5 regions: {time_5_regions * 1000:6.2f} ms (ratio: {speedup_ratio_5:.2f}x)"
        )
        print(
            f"  10 regions: {time_10_regions * 1000:6.2f} ms (ratio: {speedup_ratio_10:.2f}x)"
        )

    def test_region_membership_absolute_performance(self):
        """Test absolute performance of region_membership().

        Verifies that region_membership() completes within reasonable
        time bounds for typical use cases.
        """
        # Create environment with 2500 bins
        data = np.array([[i, j] for i in range(50) for j in range(50)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Add 10 regions
        for i in range(10):
            x0 = i * 5
            y0 = 0
            x1 = x0 + 20
            y1 = 20
            env.regions.add(f"region_{i}", polygon=box(x0, y0, x1, y1))

        # Benchmark single call
        start = time.perf_counter()
        membership = env.region_membership()
        elapsed = time.perf_counter() - start

        # Should complete in <100ms for 2500 bins x 10 regions
        assert elapsed < 0.1, (
            f"region_membership() took {elapsed * 1000:.1f}ms for 2500 bins x 10 regions "
            f"(expected <100ms)"
        )

        # Verify correctness
        assert membership.shape == (env.n_bins, 10)
        print(
            f"\nAbsolute performance: {elapsed * 1000:.2f} ms for 2500 bins x 10 regions"
        )


@pytest.mark.slow
class TestEnvironmentCreationPerformance:
    """Benchmark environment creation performance."""

    def test_large_environment_creation_time(self):
        """Test that large environments can be created in reasonable time.

        This is a regression test to ensure environment creation
        doesn't become unexpectedly slow.
        """
        # Create large dataset (10,000 points)
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, size=(10000, 2))

        # Benchmark creation
        start = time.perf_counter()
        env = Environment.from_samples(data, bin_size=2.0)
        elapsed = time.perf_counter() - start

        # Should complete in <1 second
        assert elapsed < 1.0, (
            f"Environment creation took {elapsed:.2f}s for 10k points (expected <1s)"
        )

        assert env.n_bins > 0
        print(
            f"\nEnvironment creation: {elapsed * 1000:.0f} ms for 10k points → {env.n_bins} bins"
        )
