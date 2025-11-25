"""Tests for chunked LRU caching in napari backend."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment

# Mark napari GUI tests to run in same worker (prevent Qt crashes)
pytestmark = pytest.mark.xdist_group(name="napari_gui")


@pytest.fixture
def simple_env():
    """Create simple 2D environment for testing."""
    rng = np.random.default_rng(42)
    positions = rng.random((50, 2)) * 100
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def sample_fields(simple_env):
    """Create sample field sequence (30 frames)."""
    rng = np.random.default_rng(42)
    return [rng.random(simple_env.n_bins) for _ in range(30)]


@pytest.fixture
def cmap_lookup():
    """Create sample colormap lookup table."""
    return (np.linspace(0, 1, 256)[:, None] * np.ones((256, 3)) * 255).astype(np.uint8)


class TestChunkedLRUCache:
    """Test chunked LRU cache for efficient large-dataset rendering."""

    def test_chunked_renderer_basic_access(
        self, simple_env, sample_fields, cmap_lookup
    ):
        """Test basic frame access with chunked caching."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        renderer = ChunkedLazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0, chunk_size=10
        )

        # Access first frame
        frame = renderer[0]
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.ndim == 3  # (height, width, 3)

    def test_chunk_size_configuration(self, simple_env, sample_fields, cmap_lookup):
        """Test configurable chunk size."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        # Test with chunk_size=5
        renderer = ChunkedLazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0, chunk_size=5
        )

        # Access frame in first chunk (0-4)
        frame = renderer[2]
        assert frame.shape[-1] == 3  # RGB

        # Access frame in second chunk (5-9)
        frame = renderer[7]
        assert frame.shape[-1] == 3

    def test_chunk_caching_efficiency(self, simple_env, sample_fields, cmap_lookup):
        """Test that accessing frames in same chunk reuses cached data."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        renderer = ChunkedLazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0, chunk_size=10
        )

        # Access frame 5 (loads chunk 0: frames 0-9)
        _ = renderer[5]

        # Count how many chunks are cached
        initial_cache_size = len(renderer._get_chunk_cache_info())

        # Access frame 7 (same chunk, should not load new chunk)
        _ = renderer[7]

        # Cache size should be the same (no new chunk loaded)
        final_cache_size = len(renderer._get_chunk_cache_info())
        assert final_cache_size == initial_cache_size

    def test_sequential_playback_optimization(
        self, simple_env, sample_fields, cmap_lookup
    ):
        """Test that sequential access benefits from chunk pre-loading."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        renderer = ChunkedLazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0, chunk_size=10
        )

        # Sequential access through first chunk (0-9)
        frames = [renderer[i] for i in range(10)]

        # All frames should be from the same chunk
        assert len(frames) == 10
        # Only 1 chunk should be loaded
        assert len(renderer._get_chunk_cache_info()) == 1

    def test_chunk_eviction_lru(self, simple_env, sample_fields, cmap_lookup):
        """Test that least recently used chunks are evicted."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        # Create renderer with max 2 chunks in cache
        renderer = ChunkedLazyFieldRenderer(
            simple_env,
            sample_fields,
            cmap_lookup,
            0.0,
            1.0,
            chunk_size=10,
            max_chunks=2,
        )

        # Load chunk 0 (frames 0-9)
        _ = renderer[0]

        # Load chunk 1 (frames 10-19)
        _ = renderer[10]

        # Load chunk 2 (frames 20-29) - should evict chunk 0
        _ = renderer[20]

        # Cache should have only 2 chunks (1 and 2)
        assert len(renderer._get_chunk_cache_info()) <= 2

    def test_chunk_boundary_handling(self, simple_env, sample_fields, cmap_lookup):
        """Test correct handling of frames at chunk boundaries."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        renderer = ChunkedLazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0, chunk_size=10
        )

        # Frame 9 is last in chunk 0
        frame_9 = renderer[9]
        assert frame_9.shape[-1] == 3

        # Frame 10 is first in chunk 1
        frame_10 = renderer[10]
        assert frame_10.shape[-1] == 3

        # Both should render correctly
        assert frame_9.shape == frame_10.shape

    def test_negative_indexing(self, simple_env, sample_fields, cmap_lookup):
        """Test negative indexing with chunked cache."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        renderer = ChunkedLazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0, chunk_size=10
        )

        # Access last frame
        last_frame = renderer[-1]
        assert last_frame.shape[-1] == 3

        # Should be same as explicit index
        explicit_last = renderer[len(sample_fields) - 1]
        np.testing.assert_array_equal(last_frame, explicit_last)

    def test_shape_property(self, simple_env, sample_fields, cmap_lookup):
        """Test shape property reports correct dimensions."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        renderer = ChunkedLazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0, chunk_size=10
        )

        shape = renderer.shape
        assert len(shape) == 4  # (time, height, width, channels)
        assert shape[0] == len(sample_fields)
        assert shape[-1] == 3  # RGB

    def test_dtype_property(self, simple_env, sample_fields, cmap_lookup):
        """Test dtype property returns uint8."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        renderer = ChunkedLazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0, chunk_size=10
        )

        assert renderer.dtype == np.uint8

    def test_len_property(self, simple_env, sample_fields, cmap_lookup):
        """Test len returns number of frames."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        renderer = ChunkedLazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0, chunk_size=10
        )

        assert len(renderer) == len(sample_fields)

    def test_large_dataset_efficiency(self, simple_env, cmap_lookup):
        """Test memory efficiency with large dataset (100K frames)."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        # Create mock large dataset (don't actually store 100K arrays)
        # Use memory-mapped array pattern
        n_frames = 100_000

        # Create fields list with lazy generation
        class LazyFields:
            def __init__(self, n_bins, n_frames):
                self.n_bins = n_bins
                self.n_frames = n_frames
                self.rng = np.random.default_rng(42)

            def __len__(self):
                return self.n_frames

            def __getitem__(self, idx):
                # Generate on-demand (simulating memory-mapped array)
                return self.rng.random(self.n_bins)

        lazy_fields = LazyFields(simple_env.n_bins, n_frames)

        renderer = ChunkedLazyFieldRenderer(
            simple_env,
            lazy_fields,  # Simulates memory-mapped array
            cmap_lookup,
            0.0,
            1.0,
            chunk_size=100,
            max_chunks=50,
        )

        # Access a few frames
        _ = renderer[0]
        _ = renderer[5000]
        _ = renderer[99999]

        # Should only cache 3 chunks (not all 100K frames)
        assert len(renderer._get_chunk_cache_info()) <= 50

    def test_chunk_cache_info(self, simple_env, sample_fields, cmap_lookup):
        """Test chunk cache info method."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        renderer = ChunkedLazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0, chunk_size=10
        )

        # Access frames from different chunks
        _ = renderer[5]  # Chunk 0
        _ = renderer[15]  # Chunk 1

        cache_info = renderer._get_chunk_cache_info()

        # Should have 2 chunks cached
        assert len(cache_info) == 2


class TestChunkedVsSimpleCacheComparison:
    """Compare chunked cache to simple LRU cache."""

    def test_both_produce_identical_output(
        self, simple_env, sample_fields, cmap_lookup
    ):
        """Test that both cache strategies produce identical frames."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
            LazyFieldRenderer,
        )

        simple_renderer = LazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0
        )

        chunked_renderer = ChunkedLazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0, chunk_size=10
        )

        # Compare several frames
        for i in [0, 5, 10, 15, 29]:
            simple_frame = simple_renderer[i]
            chunked_frame = chunked_renderer[i]
            np.testing.assert_array_equal(
                simple_frame,
                chunked_frame,
                err_msg=f"Frame {i} differs between renderers",
            )

    def test_chunked_better_for_sequential(
        self, simple_env, sample_fields, cmap_lookup
    ):
        """Test that chunked cache is more efficient for sequential access."""
        from neurospatial.animation.backends.napari_backend import (
            ChunkedLazyFieldRenderer,
        )

        renderer = ChunkedLazyFieldRenderer(
            simple_env, sample_fields, cmap_lookup, 0.0, 1.0, chunk_size=10
        )

        # Sequential access loads only 1 chunk for 10 frames
        for i in range(10):
            _ = renderer[i]

        # Only 1 chunk should be cached
        assert len(renderer._get_chunk_cache_info()) == 1

        # Simple cache would cache 10 individual frames
        # Chunked cache pre-loads entire chunk (better for sequential)


class TestRenderNapariWithChunkedCache:
    """Test render_napari with chunked caching enabled."""

    @pytest.mark.napari
    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="napari not available",
    )
    def test_render_napari_with_chunk_size(self, simple_env, sample_fields):
        """Test render_napari with cache_chunk_size parameter."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(
            simple_env,
            sample_fields,
            fps=30,
            cache_chunk_size=10,  # Use chunked cache
        )

        # Viewer should be created successfully
        assert viewer is not None
        viewer.close()

    @pytest.mark.napari
    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="napari not available",
    )
    def test_render_napari_default_chunk_size(self, simple_env, sample_fields):
        """Test render_napari with default chunking (None = auto-select)."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(
            simple_env,
            sample_fields,
            fps=30,
            cache_chunk_size=None,  # Auto-select based on dataset size
        )

        # Should auto-select chunking for large datasets
        assert viewer is not None
        viewer.close()

    @pytest.mark.napari
    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="napari not available",
    )
    def test_render_napari_disable_chunking(self, simple_env, sample_fields):
        """Test render_napari with chunking disabled (cache_chunk_size=1)."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(
            simple_env,
            sample_fields,
            fps=30,
            cache_chunk_size=1,  # Disable chunking (1 frame per chunk = no benefit)
        )

        # Should still work (just less efficient for sequential access)
        assert viewer is not None
        viewer.close()
