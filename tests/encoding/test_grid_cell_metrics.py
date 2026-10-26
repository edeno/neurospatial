"""
Tests for grid cell metrics.

Tests spatial_autocorrelation, grid_score, grid_scale and grid_properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment


def _direct_autocorrelation(rate_map: np.ndarray) -> np.ndarray:
    """Pearson correlation at each lag over the overlapping finite bins, by loop.

    A lag is defined when at least 20 pairs and at least a quarter of the
    finite bins overlap.
    """
    n_x, n_y = rate_map.shape
    min_pairs = max(20, np.isfinite(rate_map).sum() / 4)
    expected = np.full((2 * n_x - 1, 2 * n_y - 1), np.nan)
    for dx in range(1 - n_x, n_x):
        for dy in range(1 - n_y, n_y):
            first = rate_map[
                max(0, -dx) : n_x - max(0, dx), max(0, -dy) : n_y - max(0, dy)
            ]
            second = rate_map[
                max(0, dx) : n_x + min(0, dx), max(0, dy) : n_y + min(0, dy)
            ]
            both = np.isfinite(first) & np.isfinite(second)
            x, y = first[both], second[both]
            if both.sum() >= min_pairs and np.ptp(x) > 0 and np.ptp(y) > 0:
                expected[dx + n_x - 1, dy + n_y - 1] = np.corrcoef(x, y)[0, 1]
    return expected


@pytest.mark.parametrize("inactive", [False, True])
def test_autocorrelation_is_pearson_over_overlapping_visited_bins(inactive):
    """Each lag correlates only the visited bins that overlap, without wrapping."""
    from neurospatial.encoding.grid import spatial_autocorrelation

    mask = np.ones((9, 12), dtype=bool)
    if inactive:
        mask[0, :4] = False
    env = Environment.from_grid_mask(mask, (np.arange(10.0), np.arange(13.0)))
    rates = np.random.default_rng(3).gamma(2.0, 2.0, env.n_bins)
    rates[[3, 17, 40]] = np.nan
    # Three silent rows make the overlap at a shift of 6 rows constant on one
    # side, where the correlation is undefined although 36 pairs overlap.
    rates[-36:] = 0.0
    rate_map = np.full(mask.shape, np.nan)
    rate_map[mask] = rates

    actual = spatial_autocorrelation(env, rates)

    assert actual.shape == (17, 23)
    assert np.isnan(actual[8 + 6, 11])
    np.testing.assert_allclose(actual[8, 11], 1.0, atol=1e-12)
    np.testing.assert_array_equal(
        np.isnan(actual), np.isnan(_direct_autocorrelation(rate_map))
    )
    np.testing.assert_allclose(actual, _direct_autocorrelation(rate_map), atol=1e-12)
    np.testing.assert_allclose(
        spatial_autocorrelation(env, rates * 3 + 100), actual, atol=1e-12
    )


def test_autocorrelation_of_a_shifted_copy_is_not_wrapped():
    """A ramp correlates perfectly at every shift; wrapping it around would not."""
    from neurospatial.encoding.grid import spatial_autocorrelation

    mask = np.ones((30, 30), dtype=bool)
    env = Environment.from_grid_mask(mask, (np.arange(31.0),) * 2)
    x = env.bin_centers[:, 0]
    # A ramp along x: shifting by +k correlates perfectly with the overlap,
    # while wrap-around would join the top of the ramp to its bottom. Shifts
    # up to 22 of 30 columns overlap in at least a quarter of the bins.
    autocorr = spatial_autocorrelation(env, x)

    finite = np.isfinite(autocorr[29 + 1 : 29 + 23, 29])
    assert finite.all()
    np.testing.assert_allclose(autocorr[29 + 1 : 29 + 23, 29], 1.0, atol=1e-12)
    assert np.isnan(autocorr[29 + 23, 29])


class TestSpatialAutocorrelation:
    """Tests for spatial_autocorrelation function."""

    def test_fft_method_on_regular_grid(self):
        """Test FFT method returns 2D autocorrelation on regular grid."""
        from neurospatial.encoding.grid import (
            spatial_autocorrelation,
        )

        # Create regular 2D grid environment using deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        # Create simple firing pattern (single peak)
        firing_rate = np.zeros(env.n_bins)
        center_bin = env.n_bins // 2
        firing_rate[center_bin] = 10.0

        # Compute autocorrelation with FFT method
        autocorr = spatial_autocorrelation(env, firing_rate)

        # Should return 2D array
        assert autocorr.ndim == 2
        assert autocorr.shape == tuple(2 * n - 1 for n in env.grid_shape)

        # Center should have highest autocorrelation (≈1.0)
        center_y, center_x = autocorr.shape[0] // 2, autocorr.shape[1] // 2
        assert autocorr[center_y, center_x] > 0.5

    def test_graph_method_returns_tuple(self):
        """Test graph method returns (distances, correlations) tuple."""
        from neurospatial.encoding.grid import (
            spatial_autocorrelation_radial,
        )

        # Create environment using deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        # Random firing pattern - local RNG for isolation
        rng = np.random.default_rng(42)
        firing_rate = rng.random(env.n_bins) * 5.0

        # Compute autocorrelation with graph method
        result = spatial_autocorrelation_radial(env, firing_rate, n_distance_bins=30)

        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        distances, correlations = result

        # Both should be 1D arrays with same length
        assert distances.ndim == 1
        assert correlations.ndim == 1
        assert distances.shape == correlations.shape
        assert len(distances) == 30

    def test_auto_method_works(self):
        """Test auto method can detect appropriate method."""
        from neurospatial.encoding.grid import (
            spatial_autocorrelation,
        )

        # Create environment using deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        rng = np.random.default_rng(42)
        firing_rate = rng.random(env.n_bins) * 5.0

        # Auto should work without error (returns either 2D array or tuple)
        result = spatial_autocorrelation(env, firing_rate)

        # Should return something valid (either 2D array or tuple)
        assert result is not None
        # Either 2D array (FFT) or tuple (graph)
        assert isinstance(result, (np.ndarray, tuple))

    def test_raises_on_shape_mismatch(self):
        """Test raises ValueError if firing_rate shape doesn't match env.n_bins."""
        from neurospatial.encoding.grid import (
            spatial_autocorrelation,
        )

        # Deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        firing_rate = np.zeros(env.n_bins + 10)  # Wrong size

        with pytest.raises(ValueError, match=r"firing_rate\.shape must be"):
            spatial_autocorrelation(env, firing_rate)

    def test_raises_on_all_nan(self):
        """Test raises ValueError if all firing rates are NaN."""
        from neurospatial.encoding.grid import (
            spatial_autocorrelation,
        )

        # Deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        firing_rate = np.full(env.n_bins, np.nan)

        with pytest.raises(ValueError, match="All firing rates are NaN"):
            spatial_autocorrelation(env, firing_rate)

    def test_raises_on_constant_rates(self):
        """Test raises ValueError if all valid firing rates are constant."""
        from neurospatial.encoding.grid import (
            spatial_autocorrelation,
        )

        # Deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        firing_rate = np.full(env.n_bins, 5.0)  # All constant

        with pytest.raises(ValueError, match="All valid firing rates are constant"):
            spatial_autocorrelation(env, firing_rate)

    def test_raises_on_irregular_env(self):
        """spatial_autocorrelation refuses irregular envs and steers callers to the radial variant."""
        import networkx as nx

        from neurospatial import Environment
        from neurospatial.encoding.grid import spatial_autocorrelation

        # Build a 1-D track graph (no regular 2D grid_shape) to trigger the
        # irregular-env path. The error message must mention the radial
        # alternative so the caller knows which function to use instead.
        graph = nx.path_graph(8)
        for i, node in enumerate(graph.nodes):
            graph.nodes[node]["pos"] = (float(i) * 5.0, 0.0)
        for u, v in graph.edges:
            graph.edges[u, v]["distance"] = 5.0
        env = Environment.from_graph(
            graph,
            edge_order=list(graph.edges),
            edge_spacing=0.0,
            bin_size=2.0,
        )
        firing_rate = np.linspace(0.0, 1.0, env.n_bins)

        with pytest.raises(NotImplementedError, match="spatial_autocorrelation_radial"):
            spatial_autocorrelation(env, firing_rate)

    def test_autocorr_is_finite_where_enough_bins_overlap(self):
        """Lags up to half the map overlap in far more than 20 bins: finite."""
        from neurospatial.encoding.grid import (
            spatial_autocorrelation,
        )

        # Deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        # Create firing pattern with local RNG
        rng = np.random.default_rng(42)
        firing_rate = rng.random(env.n_bins) * 5.0

        autocorr = spatial_autocorrelation(env, firing_rate)
        n_x, n_y = env.grid_shape
        inner = autocorr[n_x // 2 : -(n_x // 2), n_y // 2 : -(n_y // 2)]

        assert np.all(np.isfinite(inner))
        assert np.isnan(autocorr[0, 0]) and np.isnan(autocorr[-1, -1])


class TestGridScore:
    """Tests for grid_score function."""

    def test_grid_score_anchor_perfect_hexagonal(self):
        """Anchor test: Synthetic perfect hexagonal pattern should score > 0.3.

        Mathematical reasoning:
        - Perfect hexagonal symmetry has correlation at 60°, 120° rotations
        - Grid score = min(r60, r120) - max(r30, r90, r150)
        - For perfect hex: r60 ≈ r120 ≈ 1.0, r30 ≈ r90 ≈ r150 ≈ low
        - Expected score > 0.3 (literature threshold for grid cells)

        Reference: Langston et al. (2010) Science
        """
        from neurospatial.encoding.grid import grid_score

        # Create synthetic autocorrelogram with perfect hexagonal peaks
        size = 100
        autocorr = np.zeros((size, size))
        center = size // 2

        # Central peak
        y_grid, x_grid = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((y_grid - center) ** 2 + (x_grid - center) ** 2)
        autocorr = np.exp(-(dist_from_center**2) / (2 * 5**2))  # Central Gaussian

        # Add 6 peaks at 60° intervals (hexagonal pattern)
        radius = 20  # Distance from center to peaks
        for angle_deg in [0, 60, 120, 180, 240, 300]:
            angle_rad = np.radians(angle_deg)
            peak_y = center + int(radius * np.sin(angle_rad))
            peak_x = center + int(radius * np.cos(angle_rad))
            peak_dist = np.sqrt((y_grid - peak_y) ** 2 + (x_grid - peak_x) ** 2)
            autocorr += 0.8 * np.exp(-(peak_dist**2) / (2 * 5**2))

        # Normalize
        autocorr = autocorr / autocorr.max()

        score = grid_score(autocorr)

        # Perfect hexagonal pattern. Rotated about its zero-lag pixel, this
        # synthetic Gaussian-peak autocorrelogram scores ~1.52 (1.40 when the
        # rotation center sat half a pixel off zero lag); bracket it as
        # 0.4 < score < 1.8, well above the 0.3 literature threshold and below
        # the +2 theoretical maximum, so a regression that under- or
        # over-scores hex symmetry fails.
        assert 0.4 < score < 1.8, (
            f"Expected grid score in (0.4, 1.8) for perfect hexagonal pattern, "
            f"got {score}"
        )

    def test_grid_score_anchor_isotropic_noise(self):
        """Anchor test: Isotropic random noise should have score near 0.

        Mathematical reasoning:
        - Random noise has no systematic angular structure
        - Correlations at all rotation angles are approximately equal
        - Grid score = min(r60, r120) - max(r30, r90, r150) ≈ 0
        """
        from neurospatial.encoding.grid import grid_score

        # Create random autocorrelogram (seeded for reproducibility)
        rng = np.random.default_rng(12345)
        size = 50
        autocorr = rng.random((size, size)) * 0.5 + 0.5

        score = grid_score(autocorr)

        # Random noise should have score near 0 (observed ~-0.008). Use an
        # explicit two-sided bound rather than abs() so a regression returning a
        # spurious positive grid score for noise (e.g. +0.4) still fails.
        assert -0.3 < score < 0.3, f"Expected grid score near 0 for noise, got {score}"

    def test_hexagonal_pattern_high_score(self):
        """Test hexagonal grid pattern produces high grid score."""
        from neurospatial.encoding.grid import grid_score

        # Create synthetic hexagonal autocorrelogram
        # (Simplified: create peaks at 60° and 120° positions)
        size = 50
        autocorr = np.zeros((size, size))

        # Central peak
        center = size // 2
        autocorr[center, center] = 1.0

        # Add hexagonal peaks (6 peaks around center at 60° intervals)
        # This is a simplified approximation
        radius = size // 5
        for angle_deg in [0, 60, 120, 180, 240, 300]:
            angle_rad = np.radians(angle_deg)
            dy = int(radius * np.sin(angle_rad))
            dx = int(radius * np.cos(angle_rad))
            y, x = center + dy, center + dx
            if 0 <= y < size and 0 <= x < size:
                autocorr[y, x] = 0.6

        score = grid_score(autocorr)

        # Hexagonal pattern should produce positive score
        # (Not necessarily > 0.4 for this simplified pattern, but should be > 0)
        assert score > 0.0

    def test_smooth_random_maps_rarely_reach_the_screening_value(self):
        """Smooth maps without a lattice seldom score 0.4 or more.

        Smoothed random maps resemble the rate maps of time-shifted spike
        trains. When the annulus swept out to shifts where only strips along
        two walls overlap, 13.5% of these maps reached 0.4 and the 95th
        percentile was 0.60; keeping shifts where a quarter of the bins
        overlap brings them to 5% and 0.40.
        """
        from scipy.ndimage import gaussian_filter

        from neurospatial.encoding.grid import grid_score, spatial_autocorrelation

        edges = np.arange(0.0, 102.5, 2.5)
        env = Environment.from_grid_mask(
            np.ones((40, 40), dtype=bool), grid_edges=(edges, edges)
        )
        rng = np.random.default_rng(0)
        scores = np.array(
            [
                grid_score(
                    spatial_autocorrelation(
                        env, gaussian_filter(rng.gamma(1.0, 1.0, (40, 40)), 3.0).ravel()
                    )
                )
                for _ in range(200)
            ]
        )

        assert np.mean(scores >= 0.4) < 0.08
        assert np.nanpercentile(scores, 95) < 0.45

    @pytest.mark.parametrize(("side", "spacing"), [(100.0, 60.0), (150.0, 90.0)])
    @pytest.mark.parametrize("orientation", [0.0, 0.3, 0.6, 0.9])
    def test_grid_spaced_at_six_tenths_of_the_arena_scores_high(
        self, side, spacing, orientation
    ):
        """The first ring of a grid spaced at 0.6 of the arena's side is scored."""
        from neurospatial.encoding.grid import grid_score, spatial_autocorrelation

        edges = np.arange(0.0, side + 1.25, 2.5)
        n_cells = len(edges) - 1
        env = Environment.from_grid_mask(
            np.ones((n_cells, n_cells), dtype=bool), grid_edges=(edges, edges)
        )
        axes = spacing * np.array(
            [
                [np.cos(orientation), np.sin(orientation)],
                [np.cos(orientation + np.pi / 3), np.sin(orientation + np.pi / 3)],
            ]
        )
        steps = np.arange(-8, 9)
        centers = np.stack(np.meshgrid(steps, steps), axis=-1).reshape(-1, 2) @ axes
        squared = np.sum((env.bin_centers[:, None] - centers[None]) ** 2, axis=-1)
        rate = np.exp(-squared / (2 * (0.15 * spacing) ** 2)).sum(axis=1)

        assert grid_score(spatial_autocorrelation(env, rate)) > 1.2

    def test_place_cell_pattern_low_score(self):
        """A single place field's autocorrelogram has no hexagonal ring."""
        from neurospatial.encoding.grid import grid_score, spatial_autocorrelation

        edges = np.arange(0.0, 102.5, 2.5)
        env = Environment.from_grid_mask(
            np.ones((40, 40), dtype=bool), grid_edges=(edges, edges)
        )
        offset = env.bin_centers - [35.0, 60.0]
        rate = np.exp(-np.sum(offset**2, axis=1) / (2 * 8.0**2))

        score = grid_score(spatial_autocorrelation(env, rate))

        assert np.isfinite(score)
        assert score < 0.3

    def test_random_noise_near_zero_score(self):
        """Test random noise produces near-zero grid score."""
        from neurospatial.encoding.grid import grid_score

        # Random autocorrelogram with local RNG
        rng = np.random.default_rng(42)
        autocorr = rng.standard_normal((50, 50)) * 0.1

        score = grid_score(autocorr)

        # Random noise should produce score near 0 (no structure)
        assert -0.5 < score < 0.5

    def test_raises_on_1d_input(self):
        """Test raises ValueError on 1D input."""
        from neurospatial.encoding.grid import grid_score

        rng = np.random.default_rng(42)
        autocorr = rng.standard_normal(50)

        with pytest.raises(ValueError, match="autocorr_2d must be 2D"):
            grid_score(autocorr)

    def test_returns_nan_on_all_nan_input(self):
        """Test returns NaN if autocorrelation contains NaN."""
        from neurospatial.encoding.grid import grid_score

        autocorr = np.full((50, 50), np.nan)

        score = grid_score(autocorr)

        assert np.isnan(score)

    def test_score_in_valid_range(self):
        """Test grid score is in expected range [-2, 2]."""
        from neurospatial.encoding.grid import grid_score

        # Random autocorrelogram with local RNG
        rng = np.random.default_rng(42)
        autocorr = rng.standard_normal((50, 50))

        score = grid_score(autocorr)

        # Score should be in valid range
        assert -2.0 <= score <= 2.0


def _create_hexagonal_autocorr(size: int = 100, radius: float = 20.0) -> np.ndarray:
    """Create synthetic hexagonal autocorrelogram for testing."""
    autocorr = np.zeros((size, size))
    center = size // 2

    # Coordinate grids
    y_grid, x_grid = np.ogrid[:size, :size]

    # Central peak
    dist_from_center = np.sqrt((y_grid - center) ** 2 + (x_grid - center) ** 2)
    autocorr = np.exp(-(dist_from_center**2) / (2 * 5**2))

    # Add 6 peaks at 60° intervals (hexagonal pattern)
    for angle_deg in [0, 60, 120, 180, 240, 300]:
        angle_rad = np.radians(angle_deg)
        peak_y = center + int(radius * np.sin(angle_rad))
        peak_x = center + int(radius * np.cos(angle_rad))
        peak_dist = np.sqrt((y_grid - peak_y) ** 2 + (x_grid - peak_x) ** 2)
        autocorr += 0.8 * np.exp(-(peak_dist**2) / (2 * 5**2))

    return autocorr / autocorr.max()


class TestGridScale:
    """Tests for grid_scale function."""

    def test_grid_scale_anchor_hexagonal_pattern(self):
        """Anchor test: Hexagonal pattern with known radius should return correct scale.

        Mathematical reasoning:
        - Peaks are placed at radius=20 pixels from center
        - With pixel_size=2.0, expected scale = 20 * 2.0 = 40.0 units
        """
        from neurospatial.encoding.grid import grid_scale

        # Create hexagonal pattern with known radius
        radius_pixels = 20.0
        pixel_size = 2.0
        autocorr = _create_hexagonal_autocorr(size=100, radius=radius_pixels)

        scale = grid_scale(autocorr, pixel_size=pixel_size)

        # Should recover approximately the correct scale
        expected_scale = radius_pixels * pixel_size
        assert abs(scale - expected_scale) < 10.0, (
            f"Expected scale ~{expected_scale}, got {scale}"
        )

    def test_grid_scale_returns_nan_on_flat_input(self):
        """Test returns NaN when no clear peaks detected (flat data)."""
        from neurospatial.encoding.grid import grid_scale

        # Truly flat data (no local maxima possible)
        autocorr = np.ones((50, 50)) * 0.5

        scale = grid_scale(autocorr, pixel_size=2.0)

        # Should return NaN (no peaks in flat data)
        assert np.isnan(scale)

    def test_grid_scale_raises_on_1d_input(self):
        """Test raises ValueError on 1D input."""
        from neurospatial.encoding.grid import grid_scale

        autocorr = np.zeros(100)

        with pytest.raises(ValueError, match="autocorr_2d must be 2D"):
            grid_scale(autocorr, pixel_size=2.0)

    def test_grid_scale_raises_on_invalid_pixel_size(self):
        """Test raises ValueError on non-positive pixel_size."""
        from neurospatial.encoding.grid import grid_scale

        autocorr = _create_hexagonal_autocorr()

        with pytest.raises(ValueError, match="pixel_size must be positive"):
            grid_scale(autocorr, pixel_size=0.0)

        with pytest.raises(ValueError, match="pixel_size must be positive"):
            grid_scale(autocorr, pixel_size=-1.0)


class TestGridOrientationViaGridProperties:
    """Tests for grid orientation, accessed through ``GridProperties``.

    Orientation is the lattice angle in radians, in the environment's
    frame (axis 0 of the map along x, axis 1 along y), modulo pi / 3.
    """

    def test_orientation_anchor_horizontal_grid(self):
        """A grid aligned with an axis has orientation in [0, pi / 3)."""
        from neurospatial.encoding.grid import grid_properties

        autocorr = _create_hexagonal_autocorr()
        props = grid_properties(autocorr, pixel_size=1.0)

        assert props.orientation >= 0.0
        assert props.orientation < np.pi / 3
        assert props.orientation_std < np.radians(15.0)

    def test_orientation_anchor_rotated_grid(self):
        """A lattice at 30 degrees from +x has orientation ~pi / 6."""
        from neurospatial.encoding.grid import grid_properties

        size = 100
        autocorr = np.zeros((size, size))
        center = size // 2
        radius = 20.0
        # Axis 0 runs along x and axis 1 along y, as spatial_autocorrelation
        # lays the map out.
        x_grid, y_grid = np.ogrid[:size, :size]

        dist_from_center = np.sqrt((y_grid - center) ** 2 + (x_grid - center) ** 2)
        autocorr = np.exp(-(dist_from_center**2) / (2 * 5**2))

        for angle_deg in [30, 90, 150, 210, 270, 330]:
            angle_rad = np.radians(angle_deg)
            peak_y = center + int(radius * np.sin(angle_rad))
            peak_x = center + int(radius * np.cos(angle_rad))
            peak_dist = np.sqrt((y_grid - peak_y) ** 2 + (x_grid - peak_x) ** 2)
            autocorr += 0.8 * np.exp(-(peak_dist**2) / (2 * 5**2))

        autocorr = autocorr / autocorr.max()
        props = grid_properties(autocorr, pixel_size=1.0)

        assert np.radians(20.0) < props.orientation < np.radians(40.0), (
            f"Expected orientation ~pi / 6, got {props.orientation}"
        )

    def test_orientation_returns_nan_on_flat_input(self):
        """No detectable peaks -> orientation NaN."""
        from neurospatial.encoding.grid import grid_properties

        autocorr = np.ones((50, 50)) * 0.5
        props = grid_properties(autocorr, pixel_size=1.0)

        assert np.isnan(props.orientation)
        assert np.isnan(props.orientation_std)

    def test_orientation_in_valid_range(self):
        """Orientation is in [0, 60)."""
        from neurospatial.encoding.grid import grid_properties

        autocorr = _create_hexagonal_autocorr()
        props = grid_properties(autocorr, pixel_size=1.0)

        assert 0.0 <= props.orientation < np.pi / 3


class TestGridProperties:
    """Tests for grid_properties function and GridProperties dataclass."""

    def test_grid_properties_returns_dataclass(self):
        """Test grid_properties returns GridProperties dataclass."""
        from neurospatial.encoding.grid import GridProperties, grid_properties

        autocorr = _create_hexagonal_autocorr()

        props = grid_properties(autocorr, pixel_size=2.0)

        assert isinstance(props, GridProperties)
        assert hasattr(props, "score")
        assert hasattr(props, "scale")
        assert hasattr(props, "orientation")
        assert hasattr(props, "orientation_std")
        assert hasattr(props, "peak_coords")
        assert hasattr(props, "n_peaks")

    def test_grid_properties_anchor_hexagonal(self):
        """Anchor test: Hexagonal pattern should have consistent properties."""
        from neurospatial.encoding.grid import grid_properties

        autocorr = _create_hexagonal_autocorr(size=100, radius=20.0)
        pixel_size = 2.0

        props = grid_properties(autocorr, pixel_size=pixel_size)

        # Score should be positive (hexagonal pattern)
        assert props.score > 0.3

        # Scale should be approximately 20 * 2 = 40
        assert 30.0 < props.scale < 50.0

        # Orientation should be in valid range
        assert 0.0 <= props.orientation < np.pi / 3

        # Should detect 6 peaks
        assert props.n_peaks >= 5  # Allow some tolerance

        # Peak coords should be 2D array
        assert props.peak_coords.ndim == 2
        assert props.peak_coords.shape[1] == 2

    def test_grid_properties_is_frozen_and_owns_read_only_peaks(self):
        """Peaks are in the scale's units, read-only, and counted by n_peaks."""
        import dataclasses

        from neurospatial.encoding.grid import grid_properties

        autocorr = _create_hexagonal_autocorr(size=101, radius=20.0)
        props = grid_properties(autocorr, pixel_size=2.0)
        distances = np.hypot(props.peak_coords[:, 0], props.peak_coords[:, 1])

        assert props.n_peaks == len(props.peak_coords) == 6
        np.testing.assert_allclose(np.median(distances), props.scale)
        with pytest.raises(ValueError, match="read-only"):
            props.peak_coords[0, 0] = 1.0
        with pytest.raises(dataclasses.FrozenInstanceError):
            props.score = 0.0  # type: ignore[misc]

    def test_grid_properties_matches_individual_functions(self):
        """Test grid_properties matches individual function outputs."""
        from neurospatial.encoding.grid import (
            grid_properties,
            grid_scale,
            grid_score,
        )

        autocorr = _create_hexagonal_autocorr()
        pixel_size = 2.0

        # Get combined properties
        props = grid_properties(autocorr, pixel_size=pixel_size)

        # Get individual function outputs
        score_individual = grid_score(autocorr)
        scale_individual = grid_scale(autocorr, pixel_size=pixel_size)

        assert props.score == score_individual
        assert props.scale == scale_individual

    def test_grid_properties_raises_on_invalid_input(self):
        """Test raises ValueError on invalid input."""
        from neurospatial.encoding.grid import grid_properties

        # 1D input
        with pytest.raises(ValueError, match="autocorr_2d must be 2D"):
            grid_properties(np.zeros(100), pixel_size=2.0)

        # Invalid pixel_size
        with pytest.raises(ValueError, match="pixel_size must be positive"):
            grid_properties(np.zeros((50, 50)), pixel_size=0.0)

    def test_grid_properties_handles_no_peaks(self):
        """Test handles case with no detectable peaks (flat data)."""
        from neurospatial.encoding.grid import grid_properties

        # Truly flat data (no local maxima possible)
        autocorr = np.ones((50, 50)) * 0.5

        props = grid_properties(autocorr, pixel_size=2.0)

        # Score is computed independently (may be non-NaN for flat data)
        # Scale and orientation should be NaN
        assert np.isnan(props.scale)
        assert np.isnan(props.orientation)
        assert np.isnan(props.orientation_std)
        # n_peaks should be 0 for flat data
        assert props.n_peaks == 0


def _autocorr_with_peaks(peaks: list[tuple[float, float]]) -> np.ndarray:
    """Autocorrelogram with a central peak and point-symmetric peak pairs.

    ``peaks`` holds ``(distance_px, angle_deg)`` of one peak of each pair.
    """
    size = 121
    offsets = np.arange(size) - size // 2
    x, y = np.meshgrid(offsets, offsets, indexing="ij")
    autocorr = np.exp(-(x**2 + y**2) / (2 * 4.0**2))
    for distance, angle in peaks:
        for sign in (1, -1):
            cx = sign * distance * np.cos(np.radians(angle))
            cy = sign * distance * np.sin(np.radians(angle))
            autocorr += 0.6 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 4.0**2))
    return autocorr / autocorr.max()


class TestSpacingNeedsOneHexagon:
    """Spacing and orientation are read only from a hexagon's first ring.

    The nearest peaks of the autocorrelogram give a spacing only when they
    can be one hexagon: their directions agree modulo 60 degrees to within
    15 degrees (the second ring of a lattice, and half of a square lattice's
    nearest peaks, point 30 degrees away), and peaks 1.5 or more times as far
    as the nearest are not in the ring.
    """

    def test_hexagonal_ring_gives_its_spacing_and_orientation(self):
        from neurospatial.encoding.grid import grid_properties

        autocorr = _autocorr_with_peaks([(20.0, 10.0), (20.0, 70.0), (20.0, 130.0)])

        props = grid_properties(autocorr, pixel_size=2.0)

        assert props.n_peaks == 6
        assert props.scale == pytest.approx(40.0, abs=1.0)
        assert np.degrees(props.orientation) == pytest.approx(10.0, abs=2.0)

    def test_fields_merged_into_two_directions_have_no_spacing(self):
        """Wide fields leave peaks 23 degrees apart: not one hexagon.

        With fields as wide as a third of the spacing, a 50 cm grid's map
        left peaks at 61 cm (19 degrees) and 71 cm (42 degrees), and the
        spacing read 66 cm.
        """
        from neurospatial.encoding.grid import grid_properties, grid_scale

        autocorr = _autocorr_with_peaks([(24.4, 19.0), (28.4, 42.0)])

        props = grid_properties(autocorr, pixel_size=2.5)

        assert np.isfinite(props.score)
        assert np.isnan(props.scale)
        assert np.isnan(props.orientation)
        assert np.isnan(props.orientation_std)
        assert np.isnan(grid_scale(autocorr, pixel_size=2.5))

    def test_second_ring_peaks_do_not_set_the_spacing(self):
        """Oversmoothing left one direction of the ring and second-ring peaks.

        A 45 cm grid smoothed with a 12 cm bandwidth kept a ring pair at
        42 cm (30 degrees) beside second-ring peaks at 70 cm, and the spacing
        read 70 cm.
        """
        from neurospatial.encoding.grid import grid_properties

        autocorr = _autocorr_with_peaks([(17.0, 30.0), (27.8, 60.0), (28.4, 2.0)])

        props = grid_properties(autocorr, pixel_size=2.5)

        assert np.isnan(props.scale)
        assert np.isnan(props.orientation)

    def test_ring_with_one_direction_missing_keeps_its_spacing(self):
        """Second-ring peaks among the six nearest are left out, not fatal."""
        from neurospatial.encoding.grid import grid_properties

        autocorr = _autocorr_with_peaks([(18.0, 38.0), (18.0, 98.0), (31.2, 8.0)])

        props = grid_properties(autocorr, pixel_size=2.5)

        assert props.n_peaks == 4
        assert props.scale == pytest.approx(45.0, abs=1.5)
        assert np.degrees(props.orientation) == pytest.approx(38.0, abs=3.0)

    def test_square_lattice_has_no_hexagonal_spacing(self):
        from neurospatial.encoding.grid import grid_properties

        autocorr = _autocorr_with_peaks(
            [(20.0, 0.0), (20.0, 90.0), (28.3, 45.0), (28.3, 135.0)]
        )

        props = grid_properties(autocorr, pixel_size=2.0)

        assert np.isnan(props.scale)
        assert np.isnan(props.orientation)

    def test_no_spacing_when_the_score_is_undefined(self):
        """An 80 cm grid in a 50 cm box has no bounded central peak."""
        from neurospatial.encoding.grid import (
            autocorrelation_pixel_size,
            grid_properties,
            spatial_autocorrelation,
        )

        edges = np.arange(0.0, 51.25, 2.5)
        env = Environment.from_grid_mask(
            np.ones((20, 20), dtype=bool), grid_edges=(edges, edges)
        )
        wavenumber = 4 * np.pi / (np.sqrt(3) * 80.0)
        rate = np.exp(
            sum(
                np.cos(wavenumber * (env.bin_centers - 10.0) @ [np.cos(a), np.sin(a)])
                for a in (0.3, 0.3 + np.pi / 3, 0.3 + 2 * np.pi / 3)
            )
        )

        props = grid_properties(
            spatial_autocorrelation(env, rate),
            pixel_size=autocorrelation_pixel_size(env),
        )

        assert np.isnan(props.score)
        assert np.isnan(props.scale)
        assert np.isnan(props.orientation)

    @pytest.mark.parametrize(("spacing", "expected"), [(60.0, 60.0), (85.0, np.nan)])
    def test_a_ring_beyond_the_defined_lags_gives_no_spacing(self, spacing, expected):
        """Maxima on the edge of the defined lags are not peaks.

        The first ring of an 85 cm grid in a 1 m box lies where less than a
        quarter of the map overlaps; the correlation still rising toward it
        made maxima on the edge of the defined lags, and the spacing read
        71 cm. A 60 cm ring lies inside and keeps its spacing.
        """
        from neurospatial.encoding.grid import grid_scale, spatial_autocorrelation

        edges = np.arange(0.0, 101.25, 2.5)
        env = Environment.from_grid_mask(
            np.ones((40, 40), dtype=bool), grid_edges=(edges, edges)
        )
        wavenumber = 4 * np.pi / (np.sqrt(3) * spacing)
        rate = np.exp(
            sum(
                np.cos(wavenumber * (env.bin_centers - 20.0) @ [np.cos(a), np.sin(a)])
                for a in (0.3, 0.3 + np.pi / 3, 0.3 + 2 * np.pi / 3)
            )
        )

        scale = grid_scale(spatial_autocorrelation(env, rate), pixel_size=2.5)

        np.testing.assert_allclose(scale, expected, atol=1.5)

    def test_smooth_random_maps_rarely_get_a_spacing(self):
        """Without a hexagonal ring, most maps get no spacing (96% did before)."""
        from scipy.ndimage import gaussian_filter

        from neurospatial.encoding.grid import grid_scale, spatial_autocorrelation

        edges = np.arange(0.0, 102.5, 2.5)
        env = Environment.from_grid_mask(
            np.ones((40, 40), dtype=bool), grid_edges=(edges, edges)
        )
        rng = np.random.default_rng(0)
        scales = [
            grid_scale(
                spatial_autocorrelation(
                    env, gaussian_filter(rng.gamma(1.0, 1.0, (40, 40)), 3.0).ravel()
                ),
                pixel_size=2.5,
            )
            for _ in range(100)
        ]

        assert np.mean(np.isfinite(scales)) < 0.35


class TestFindAutocorrPeaks:
    """Tests for _find_autocorr_peaks helper function."""

    def test_finds_hexagonal_peaks(self):
        """Test finds 6 peaks in hexagonal pattern."""
        from neurospatial.encoding.grid import _find_autocorr_peaks

        autocorr = _create_hexagonal_autocorr()

        peaks = _find_autocorr_peaks(autocorr, threshold_rel=0.1, max_peaks=7)

        # Should find 6 peaks (excluding central peak)
        assert 5 <= len(peaks) <= 7

    def test_excludes_central_peak(self):
        """Test excludes central peak from results."""
        from neurospatial.encoding.grid import _find_autocorr_peaks

        autocorr = _create_hexagonal_autocorr()

        peaks = _find_autocorr_peaks(autocorr, threshold_rel=0.1, max_peaks=7)

        # All peaks should be non-zero distance from center
        distances = np.sqrt(peaks[:, 0] ** 2 + peaks[:, 1] ** 2)
        assert np.all(distances > 5)

    def test_returns_center_relative_coords(self):
        """Test returns coordinates relative to center."""
        from neurospatial.encoding.grid import _find_autocorr_peaks

        autocorr = _create_hexagonal_autocorr(size=100)

        peaks = _find_autocorr_peaks(autocorr, threshold_rel=0.1, max_peaks=7)

        # Peaks should be centered around (0, 0)
        # Mean should be near zero for symmetric pattern
        assert abs(np.mean(peaks[:, 0])) < 5
        assert abs(np.mean(peaks[:, 1])) < 5

    def test_sorted_by_distance(self):
        """Test peaks are sorted by distance from center."""
        from neurospatial.encoding.grid import _find_autocorr_peaks

        autocorr = _create_hexagonal_autocorr()

        peaks = _find_autocorr_peaks(autocorr, threshold_rel=0.1, max_peaks=7)

        if len(peaks) > 1:
            distances = np.sqrt(peaks[:, 0] ** 2 + peaks[:, 1] ** 2)
            # Should be sorted in ascending order
            assert np.all(np.diff(distances) >= 0)

    def test_returns_empty_on_flat_input(self):
        """Test returns empty array on flat input (no peaks)."""
        from neurospatial.encoding.grid import _find_autocorr_peaks

        autocorr = np.ones((50, 50)) * 0.5

        peaks = _find_autocorr_peaks(autocorr, threshold_rel=0.1, max_peaks=7)

        assert len(peaks) == 0
        assert peaks.shape == (0, 2)
