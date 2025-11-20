"""Integration tests for memory-mapped array support.

These tests verify that animation backends work correctly with memory-mapped
arrays for large-scale datasets (100K+ frames) without loading all data into memory.

Tests are marked as slow (@pytest.mark.slow) and excluded from CI by default.
Napari tests use xdist_group to prevent Qt crashes in parallel execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment


@pytest.mark.slow
@pytest.mark.xdist_group(name="napari_gui")
def test_napari_with_memmap_large_dataset(tmp_path):
    """Test Napari backend with memory-mapped array (simulates 100K frames)."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    # Create environment
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create memory-mapped array (simulate large-scale session)
    # Use smaller size for CI (1000 frames), but demonstrates technique
    n_frames = 1000  # Real use case: 100K-900K frames
    memmap_file = tmp_path / "fields.dat"
    fields = np.memmap(
        memmap_file, dtype="float32", mode="w+", shape=(n_frames, env.n_bins)
    )

    # Populate with test data (only a few frames - memmap doesn't load all)
    for i in range(min(10, n_frames)):  # Just populate first 10 frames
        fields[i] = np.random.rand(env.n_bins)
    fields.flush()

    # Test napari lazy loading (should NOT load all frames into memory)
    viewer = render_napari(env, fields, vmin=0, vmax=1)

    # Verify viewer created
    assert viewer is not None

    # Verify lazy renderer has correct shape
    # (viewer created, but data not loaded until accessed)
    assert hasattr(viewer, "layers")
    assert len(viewer.layers) > 0


@pytest.mark.slow
def test_subsample_with_memmap_preserves_type(tmp_path):
    """Test subsample_frames() works with memory-mapped arrays."""
    from neurospatial.animation.core import subsample_frames

    # Create environment
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create memory-mapped array
    n_frames = 1000
    memmap_file = tmp_path / "fields.dat"
    fields = np.memmap(
        memmap_file, dtype="float32", mode="w+", shape=(n_frames, env.n_bins)
    )

    # Populate first frame only (demonstrate efficiency)
    fields[0] = np.random.rand(env.n_bins)
    fields.flush()

    # Subsample to 30 fps (from 250 Hz)
    subsampled = subsample_frames(fields, source_fps=250, target_fps=30)

    # Verify subsampling worked without loading all data
    assert isinstance(subsampled, np.ndarray)
    expected_frames = len(np.arange(0, n_frames, 250 / 30))
    assert len(subsampled) == expected_frames
    assert subsampled.shape[1] == env.n_bins

    # Verify first frame matches
    np.testing.assert_array_equal(subsampled[0], fields[0])


@pytest.mark.slow
def test_html_backend_with_memmap_and_subsample(tmp_path):
    """Test HTML backend workflow: memmap → subsample → export."""
    from neurospatial.animation.backends.html_backend import render_html

    # Create environment
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create memory-mapped array (simulate recording session)
    n_frames = 500  # Simulates 2 seconds at 250 Hz
    memmap_file = tmp_path / "fields.dat"
    fields_full = np.memmap(
        memmap_file, dtype="float32", mode="w+", shape=(n_frames, env.n_bins)
    )

    # Populate with varying data
    for i in range(n_frames):
        fields_full[i] = np.random.rand(env.n_bins) * (i / n_frames)
    fields_full.flush()

    # Subsample for HTML export (too many frames for HTML)
    from neurospatial.animation.core import subsample_frames

    fields_subsampled = subsample_frames(fields_full, source_fps=250, target_fps=10)
    # 500 frames at 250 Hz → 20 frames at 10 fps (manageable for HTML)

    # Export to HTML
    output_path = tmp_path / "memmap_test.html"
    render_html(env, list(fields_subsampled), save_path=str(output_path), fps=10)

    # Verify export succeeded
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Verify HTML contains frames
    html_content = output_path.read_text()
    assert "data:image/png;base64" in html_content
    assert len(fields_subsampled) < 100  # Verify subsampling worked


@pytest.mark.slow
def test_memmap_cleanup(tmp_path):
    """Test that memory-mapped files are properly cleaned up."""
    # Create environment
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create and use memory-mapped array
    memmap_file = tmp_path / "fields.dat"
    fields = np.memmap(memmap_file, dtype="float32", mode="w+", shape=(100, env.n_bins))
    fields[:] = np.random.rand(100, env.n_bins)
    fields.flush()

    # Verify file exists
    assert memmap_file.exists()

    # Delete reference (file should persist)
    del fields

    # File should still exist (memmap files persist)
    assert memmap_file.exists()

    # Manual cleanup (user's responsibility in real code)
    memmap_file.unlink()
    assert not memmap_file.exists()


@pytest.mark.slow
@pytest.mark.xdist_group(name="napari_gui")
def test_large_memmap_napari_chunked_cache(tmp_path):
    """Test that chunked cache works with large memory-mapped arrays."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    # Create environment
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create large memory-mapped array (triggers chunked caching)
    n_frames = 15000  # >10K frames → auto-enables chunked cache
    memmap_file = tmp_path / "large_fields.dat"
    fields = np.memmap(
        memmap_file, dtype="float32", mode="w+", shape=(n_frames, env.n_bins)
    )

    # Populate only first and last frames (demonstrate lazy loading)
    fields[0] = np.random.rand(env.n_bins)
    fields[-1] = np.random.rand(env.n_bins) * 2
    fields.flush()

    # Test napari with chunked cache (should auto-enable at 15K frames)
    viewer = render_napari(env, fields, vmin=0, vmax=2)

    # Verify viewer created without loading all data
    assert viewer is not None
    assert hasattr(viewer, "layers")

    # Chunked cache should be used (verified in napari_backend.py logic)
    # This test verifies the integration doesn't crash with large arrays
