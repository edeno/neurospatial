"""Tests for benchmark dataset generators.

These tests verify that benchmark dataset generators produce:
- Correct shapes and types
- Reproducible data (seeded RNG)
- Valid overlay data for animation backends
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path for benchmark imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import benchmark utilities (use package-level imports)
from benchmark_datasets import (  # noqa: E402
    LARGE_CONFIG,
    MEDIUM_CONFIG,
    SMALL_CONFIG,
    BenchmarkConfig,
    create_benchmark_env,
    create_benchmark_fields,
    create_benchmark_overlays,
)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_small_config_values(self):
        """Small config has 100 frames, 40x40 grid."""
        assert SMALL_CONFIG.n_frames == 100
        assert SMALL_CONFIG.grid_size == 40
        assert SMALL_CONFIG.name == "small"

    def test_medium_config_values(self):
        """Medium config has 5k frames, typical grid."""
        assert MEDIUM_CONFIG.n_frames == 5000
        assert MEDIUM_CONFIG.grid_size == 100
        assert MEDIUM_CONFIG.name == "medium"

    def test_large_config_values(self):
        """Large config has 100k frames with overlays."""
        assert LARGE_CONFIG.n_frames == 100_000
        assert LARGE_CONFIG.grid_size == 100
        assert LARGE_CONFIG.include_skeleton is True
        assert LARGE_CONFIG.include_head_direction is True
        assert LARGE_CONFIG.name == "large"


class TestCreateBenchmarkEnv:
    """Tests for create_benchmark_env function."""

    def test_returns_environment(self):
        """Should return a fitted Environment."""
        from neurospatial import Environment

        env = create_benchmark_env(SMALL_CONFIG)
        assert isinstance(env, Environment)
        assert env._is_fitted is True

    def test_grid_size_respected(self):
        """Grid size should result in a reasonable number of bins."""
        env = create_benchmark_env(SMALL_CONFIG)
        # bin_size=1.0 with uniform samples in [0, grid_size] space
        # Bins depend on sample coverage and grid extent
        # For 40x40 extent with bin_size=1, expect roughly 1000-2000 bins
        assert env.n_bins > 0
        # Grid should have at least grid_size/2 * grid_size/2 bins
        min_bins = (SMALL_CONFIG.grid_size // 2) ** 2
        max_bins = (SMALL_CONFIG.grid_size + 5) ** 2  # Allow some padding
        assert min_bins <= env.n_bins <= max_bins, (
            f"Expected {min_bins} <= n_bins <= {max_bins}, got {env.n_bins}"
        )

    def test_reproducible_with_seed(self):
        """Same seed produces identical environment."""
        env1 = create_benchmark_env(SMALL_CONFIG, seed=42)
        env2 = create_benchmark_env(SMALL_CONFIG, seed=42)

        np.testing.assert_array_equal(env1.bin_centers, env2.bin_centers)

    def test_different_seeds_different_results(self):
        """Different seeds produce different environments."""
        env1 = create_benchmark_env(SMALL_CONFIG, seed=42)
        env2 = create_benchmark_env(SMALL_CONFIG, seed=123)

        # Bin centers should differ
        assert not np.array_equal(env1.bin_centers, env2.bin_centers)


class TestCreateBenchmarkFields:
    """Tests for create_benchmark_fields function."""

    def test_returns_correct_shape(self):
        """Fields should have shape (n_frames, n_bins)."""
        env = create_benchmark_env(SMALL_CONFIG, seed=42)
        fields = create_benchmark_fields(env, SMALL_CONFIG, seed=42)

        assert fields.shape == (SMALL_CONFIG.n_frames, env.n_bins)

    def test_returns_float32(self):
        """Fields should be float32 for memory efficiency."""
        env = create_benchmark_env(SMALL_CONFIG, seed=42)
        fields = create_benchmark_fields(env, SMALL_CONFIG, seed=42)

        assert fields.dtype == np.float32

    def test_fields_normalized(self):
        """Fields should be in [0, 1] range."""
        env = create_benchmark_env(SMALL_CONFIG, seed=42)
        fields = create_benchmark_fields(env, SMALL_CONFIG, seed=42)

        assert fields.min() >= 0.0
        assert fields.max() <= 1.0

    def test_reproducible_with_seed(self):
        """Same seed produces identical fields."""
        env = create_benchmark_env(SMALL_CONFIG, seed=42)
        fields1 = create_benchmark_fields(env, SMALL_CONFIG, seed=42)
        fields2 = create_benchmark_fields(env, SMALL_CONFIG, seed=42)

        np.testing.assert_array_equal(fields1, fields2)

    def test_memmap_output(self, tmp_path):
        """Should support memory-mapped output for large datasets."""
        env = create_benchmark_env(SMALL_CONFIG, seed=42)
        memmap_path = tmp_path / "test_fields.dat"

        fields = create_benchmark_fields(
            env, SMALL_CONFIG, seed=42, memmap_path=memmap_path
        )

        assert memmap_path.exists()
        assert isinstance(fields, np.memmap)
        assert fields.shape == (SMALL_CONFIG.n_frames, env.n_bins)


class TestCreateBenchmarkOverlays:
    """Tests for create_benchmark_overlays function."""

    def test_returns_list(self):
        """Should return a list of overlays."""
        env = create_benchmark_env(SMALL_CONFIG, seed=42)
        overlays = create_benchmark_overlays(env, SMALL_CONFIG, seed=42)

        assert isinstance(overlays, list)

    def test_position_overlay_shape(self):
        """Position overlay should have correct shape."""
        from neurospatial.animation.overlays import PositionOverlay

        env = create_benchmark_env(SMALL_CONFIG, seed=42)
        overlays = create_benchmark_overlays(env, SMALL_CONFIG, seed=42)

        position_overlays = [o for o in overlays if isinstance(o, PositionOverlay)]
        assert len(position_overlays) >= 1

        pos_overlay = position_overlays[0]
        assert pos_overlay.data.shape == (SMALL_CONFIG.n_frames, env.n_dims)

    def test_position_overlay_within_bounds(self):
        """Position overlay coordinates should be within environment bounds."""
        env = create_benchmark_env(SMALL_CONFIG, seed=42)
        overlays = create_benchmark_overlays(env, SMALL_CONFIG, seed=42)

        from neurospatial.animation.overlays import PositionOverlay

        for overlay in overlays:
            if isinstance(overlay, PositionOverlay):
                for dim in range(env.n_dims):
                    dim_min, dim_max = env.dimension_ranges[dim]
                    assert overlay.data[:, dim].min() >= dim_min
                    assert overlay.data[:, dim].max() <= dim_max

    def test_large_config_includes_skeleton(self):
        """Large config should include bodypart overlay with skeleton."""
        from neurospatial.animation.overlays import BodypartOverlay

        env = create_benchmark_env(LARGE_CONFIG, seed=42)
        overlays = create_benchmark_overlays(env, LARGE_CONFIG, seed=42)

        bodypart_overlays = [o for o in overlays if isinstance(o, BodypartOverlay)]
        assert len(bodypart_overlays) >= 1
        assert bodypart_overlays[0].skeleton is not None

    def test_large_config_includes_head_direction(self):
        """Large config should include head direction overlay."""
        from neurospatial.animation.overlays import HeadDirectionOverlay

        env = create_benchmark_env(LARGE_CONFIG, seed=42)
        overlays = create_benchmark_overlays(env, LARGE_CONFIG, seed=42)

        hd_overlays = [o for o in overlays if isinstance(o, HeadDirectionOverlay)]
        assert len(hd_overlays) >= 1

    def test_head_direction_valid_angles(self):
        """Head direction angles should be in valid range [-pi, pi]."""
        from neurospatial.animation.overlays import HeadDirectionOverlay

        env = create_benchmark_env(LARGE_CONFIG, seed=42)
        overlays = create_benchmark_overlays(env, LARGE_CONFIG, seed=42)

        for overlay in overlays:
            if isinstance(overlay, HeadDirectionOverlay):
                # Implementation wraps angles to [-pi, pi]
                assert overlay.data.min() >= -np.pi, (
                    f"Min angle {overlay.data.min()} is below -pi"
                )
                assert overlay.data.max() <= np.pi, (
                    f"Max angle {overlay.data.max()} exceeds pi"
                )

    def test_head_direction_only_config(self):
        """Config with only head direction should work."""
        from neurospatial.animation.overlays import HeadDirectionOverlay

        config = BenchmarkConfig(
            name="head_direction_only",
            n_frames=50,
            grid_size=20,
            include_position=False,
            include_skeleton=False,
            include_head_direction=True,
        )

        env = create_benchmark_env(config, seed=42)
        overlays = create_benchmark_overlays(env, config, seed=42)

        # Should have exactly one head direction overlay
        assert len(overlays) == 1
        assert isinstance(overlays[0], HeadDirectionOverlay)
        # Angles should be in valid range
        assert overlays[0].data.min() >= -np.pi
        assert overlays[0].data.max() <= np.pi

    def test_reproducible_with_seed(self):
        """Same seed produces identical overlays."""
        env = create_benchmark_env(SMALL_CONFIG, seed=42)
        overlays1 = create_benchmark_overlays(env, SMALL_CONFIG, seed=42)
        overlays2 = create_benchmark_overlays(env, SMALL_CONFIG, seed=42)

        from neurospatial.animation.overlays import PositionOverlay

        for o1, o2 in zip(overlays1, overlays2, strict=True):
            if isinstance(o1, PositionOverlay):
                np.testing.assert_array_equal(o1.data, o2.data)


class TestBenchmarkConfigCustomization:
    """Tests for custom benchmark configurations."""

    def test_custom_config(self):
        """Should support custom configurations."""
        custom = BenchmarkConfig(
            name="custom",
            n_frames=50,
            grid_size=20,
            include_position=True,
            include_skeleton=False,
            include_head_direction=False,
        )

        env = create_benchmark_env(custom, seed=42)
        fields = create_benchmark_fields(env, custom, seed=42)
        overlays = create_benchmark_overlays(env, custom, seed=42)

        assert fields.shape[0] == 50
        assert len([o for o in overlays if hasattr(o, "skeleton")]) == 0

    def test_config_no_overlays(self):
        """Config with no overlays should return empty list."""
        config = BenchmarkConfig(
            name="no_overlays",
            n_frames=10,
            grid_size=10,
            include_position=False,
            include_skeleton=False,
            include_head_direction=False,
        )

        env = create_benchmark_env(config, seed=42)
        overlays = create_benchmark_overlays(env, config, seed=42)

        assert overlays == []
