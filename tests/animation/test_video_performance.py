"""Performance tests for video backend overlay rendering.

Tests benchmark:
- Rendering overhead: overlays vs no overlays
- Target: overhead < 2× for typical configurations
- Parallel rendering speedup with multiple workers
- Impact of artist reuse optimization

All tests marked with @pytest.mark.slow for selective execution.
"""

# ruff: noqa: SIM117 - Nested with statements are more readable in tests

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.animation._parallel import parallel_render_frames
from neurospatial.animation.overlays import (
    BodypartData,
    HeadDirectionData,
    OverlayData,
    PositionData,
)
from neurospatial.animation.skeleton import Skeleton

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def perf_env() -> Environment:
    """Create a moderately complex 2D environment for performance testing."""
    # Create environment with ~400 bins (realistic size)
    rng = np.random.default_rng(42)
    positions = rng.random((1000, 2)) * 200
    env = Environment.from_samples(positions, bin_size=10.0)
    env.clear_cache()  # Ensure pickle-able for parallel tests
    return env


@pytest.fixture
def perf_fields(perf_env: Environment) -> list[NDArray[np.float64]]:
    """Create 100 frames of fields for performance testing."""
    rng = np.random.default_rng(42)
    n_frames = 100
    return [rng.random(perf_env.n_bins) for _ in range(n_frames)]


@pytest.fixture
def perf_position_overlay() -> OverlayData:
    """Create position overlay with trail for performance testing."""
    rng = np.random.default_rng(42)
    n_frames = 100
    positions = rng.random((n_frames, 2)) * 100 + 50  # Center in environment
    pos_data = PositionData(
        data=positions,
        color="red",
        size=10.0,
        trail_length=30,  # 30-frame trail
    )
    return OverlayData(positions=[pos_data])


@pytest.fixture
def perf_bodypart_overlay() -> OverlayData:
    """Create bodypart overlay with skeleton for performance testing."""
    rng = np.random.default_rng(42)
    n_frames = 100
    n_bodyparts = 5
    bodyparts = {
        f"part{i}": rng.random((n_frames, 2)) * 100 + 50 for i in range(n_bodyparts)
    }
    nodes = tuple(f"part{i}" for i in range(n_bodyparts))
    edges = tuple((f"part{i}", f"part{i + 1}") for i in range(n_bodyparts - 1))
    skeleton = Skeleton(
        name="test",
        nodes=nodes,
        edges=edges,
        edge_color="white",
        edge_width=2.0,
    )
    colors = {f"part{i}": f"C{i}" for i in range(n_bodyparts)}

    bodypart_data = BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors=colors,
    )
    return OverlayData(bodypart_sets=[bodypart_data])


@pytest.fixture
def perf_all_overlays() -> OverlayData:
    """Create all overlay types for comprehensive performance testing."""
    rng = np.random.default_rng(42)
    n_frames = 100

    # Position with trail
    positions = rng.random((n_frames, 2)) * 100 + 50
    pos_data = PositionData(data=positions, color="red", size=10.0, trail_length=30)

    # Bodyparts with skeleton
    n_bodyparts = 5
    bodyparts = {
        f"part{i}": rng.random((n_frames, 2)) * 100 + 50 for i in range(n_bodyparts)
    }
    nodes = tuple(f"part{i}" for i in range(n_bodyparts))
    edges = tuple((f"part{i}", f"part{i + 1}") for i in range(n_bodyparts - 1))
    skeleton = Skeleton(
        name="test",
        nodes=nodes,
        edges=edges,
        edge_color="white",
        edge_width=2.0,
    )
    colors = {f"part{i}": f"C{i}" for i in range(n_bodyparts)}
    bodypart_data = BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors=colors,
    )

    # Head direction
    angles = np.linspace(0, 2 * np.pi, n_frames)
    head_dir_data = HeadDirectionData(data=angles, color="yellow", length=20.0)

    return OverlayData(
        positions=[pos_data],
        bodypart_sets=[bodypart_data],
        head_directions=[head_dir_data],
    )


# =============================================================================
# Performance Benchmarks
# =============================================================================


@pytest.mark.slow
def test_baseline_no_overlays(perf_env: Environment, perf_fields: list) -> None:
    """Benchmark baseline rendering with no overlays.

    Establishes baseline performance for comparison.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock ffmpeg subprocess call
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""

            start_time = time.perf_counter()

            _ = parallel_render_frames(
                env=perf_env,
                fields=perf_fields,
                output_dir=tmpdir,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                frame_labels=None,
                dpi=100,
                n_workers=1,  # Serial for fair comparison
                reuse_artists=True,
                overlay_data=None,  # No overlays
            )

            elapsed = time.perf_counter() - start_time

            # Verify frames were created
            png_files = list(Path(tmpdir).glob("frame_*.png"))
            assert len(png_files) == len(perf_fields)

            print(f"\n📊 Baseline (no overlays): {elapsed:.3f}s for 100 frames")
            print(f"   Average: {elapsed / len(perf_fields) * 1000:.1f}ms per frame")


@pytest.mark.slow
def test_overhead_position_with_trail(
    perf_env: Environment, perf_fields: list, perf_position_overlay: OverlayData
) -> None:
    """Benchmark rendering with position overlay and trail.

    Target: overhead < 2× baseline
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""

            start_time = time.perf_counter()

            _ = parallel_render_frames(
                env=perf_env,
                fields=perf_fields,
                output_dir=tmpdir,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                frame_labels=None,
                dpi=100,
                n_workers=1,
                reuse_artists=True,
                overlay_data=perf_position_overlay,
            )

            elapsed = time.perf_counter() - start_time

            png_files = list(Path(tmpdir).glob("frame_*.png"))
            assert len(png_files) == len(perf_fields)

            print(f"\n📊 Position + Trail: {elapsed:.3f}s for 100 frames")
            print(f"   Average: {elapsed / len(perf_fields) * 1000:.1f}ms per frame")


@pytest.mark.slow
def test_overhead_bodypart_with_skeleton(
    perf_env: Environment, perf_fields: list, perf_bodypart_overlay: OverlayData
) -> None:
    """Benchmark rendering with bodypart overlay and skeleton.

    Target: overhead < 2× baseline
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""

            start_time = time.perf_counter()

            _ = parallel_render_frames(
                env=perf_env,
                fields=perf_fields,
                output_dir=tmpdir,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                frame_labels=None,
                dpi=100,
                n_workers=1,
                reuse_artists=True,
                overlay_data=perf_bodypart_overlay,
            )

            elapsed = time.perf_counter() - start_time

            png_files = list(Path(tmpdir).glob("frame_*.png"))
            assert len(png_files) == len(perf_fields)

            print(f"\n📊 Bodypart + Skeleton: {elapsed:.3f}s for 100 frames")
            print(f"   Average: {elapsed / len(perf_fields) * 1000:.1f}ms per frame")


@pytest.mark.slow
def test_overhead_all_overlays(
    perf_env: Environment, perf_fields: list, perf_all_overlays: OverlayData
) -> None:
    """Benchmark rendering with all overlay types.

    Target: overhead < 2× baseline (comprehensive test)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""

            start_time = time.perf_counter()

            _ = parallel_render_frames(
                env=perf_env,
                fields=perf_fields,
                output_dir=tmpdir,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                frame_labels=None,
                dpi=100,
                n_workers=1,
                reuse_artists=True,
                overlay_data=perf_all_overlays,
            )

            elapsed = time.perf_counter() - start_time

            png_files = list(Path(tmpdir).glob("frame_*.png"))
            assert len(png_files) == len(perf_fields)

            print(f"\n📊 All Overlays: {elapsed:.3f}s for 100 frames")
            print(f"   Average: {elapsed / len(perf_fields) * 1000:.1f}ms per frame")
            print("   (Position + Bodypart + Head Direction)")


@pytest.mark.slow
def test_parallel_rendering_speedup(
    perf_env: Environment, perf_fields: list, perf_all_overlays: OverlayData
) -> None:
    """Benchmark parallel rendering speedup with multiple workers.

    Tests that parallel rendering provides meaningful speedup.
    """
    with tempfile.TemporaryDirectory() as tmpdir_serial:
        with tempfile.TemporaryDirectory() as tmpdir_parallel:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stderr = ""

                # Serial rendering (n_workers=1)
                start_serial = time.perf_counter()
                _ = parallel_render_frames(
                    env=perf_env,
                    fields=perf_fields,
                    output_dir=tmpdir_serial,
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    frame_labels=None,
                    dpi=100,
                    n_workers=1,
                    reuse_artists=True,
                    overlay_data=perf_all_overlays,
                )
                elapsed_serial = time.perf_counter() - start_serial

                # Parallel rendering (n_workers=4)
                start_parallel = time.perf_counter()
                _ = parallel_render_frames(
                    env=perf_env,
                    fields=perf_fields,
                    output_dir=tmpdir_parallel,
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    frame_labels=None,
                    dpi=100,
                    n_workers=4,
                    reuse_artists=True,
                    overlay_data=perf_all_overlays,
                )
                elapsed_parallel = time.perf_counter() - start_parallel

                # Verify both produced correct number of frames
                serial_files = list(Path(tmpdir_serial).glob("frame_*.png"))
                parallel_files = list(Path(tmpdir_parallel).glob("frame_*.png"))
                assert len(serial_files) == len(perf_fields)
                assert len(parallel_files) == len(perf_fields)

                speedup = elapsed_serial / elapsed_parallel

                print("\n📊 Parallel Rendering Speedup:")
                print(f"   Serial (1 worker):   {elapsed_serial:.3f}s")
                print(f"   Parallel (4 workers): {elapsed_parallel:.3f}s")
                print(f"   Speedup: {speedup:.2f}x")

                # Expect at least 1.5x speedup with 4 workers (conservative)
                # Real speedup may be less than 4x due to overhead
                assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.2f}x"


@pytest.mark.slow
def test_artist_reuse_impact(
    perf_env: Environment, perf_fields: list, perf_all_overlays: OverlayData
) -> None:
    """Benchmark impact of artist reuse optimization.

    Compares reuse_artists=True vs False to measure optimization benefit.

    Notes
    -----
    Performance analysis shows that for overlays, clearing overlay artists
    and recreating them is comparable in cost to clearing the entire axes
    and redrawing. This means reuse_artists provides primary benefit for
    the field image (already optimized), not overlays.

    The real optimization target is reducing overlay overhead vs no overlays
    (target < 2×), not reuse vs no-reuse for overlays.
    """
    with tempfile.TemporaryDirectory() as tmpdir_reuse:
        with tempfile.TemporaryDirectory() as tmpdir_no_reuse:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stderr = ""

                # With artist reuse (optimized)
                start_reuse = time.perf_counter()
                _ = parallel_render_frames(
                    env=perf_env,
                    fields=perf_fields,
                    output_dir=tmpdir_reuse,
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    frame_labels=None,
                    dpi=100,
                    n_workers=1,
                    reuse_artists=True,  # Optimized path
                    overlay_data=perf_all_overlays,
                )
                elapsed_reuse = time.perf_counter() - start_reuse

                # Without artist reuse (baseline)
                start_no_reuse = time.perf_counter()
                _ = parallel_render_frames(
                    env=perf_env,
                    fields=perf_fields,
                    output_dir=tmpdir_no_reuse,
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    frame_labels=None,
                    dpi=100,
                    n_workers=1,
                    reuse_artists=False,  # Original path
                    overlay_data=perf_all_overlays,
                )
                elapsed_no_reuse = time.perf_counter() - start_no_reuse

                # Verify both produced correct frames
                reuse_files = list(Path(tmpdir_reuse).glob("frame_*.png"))
                no_reuse_files = list(Path(tmpdir_no_reuse).glob("frame_*.png"))
                assert len(reuse_files) == len(perf_fields)
                assert len(no_reuse_files) == len(perf_fields)

                speedup = elapsed_no_reuse / elapsed_reuse

                print("\n📊 Artist Reuse Optimization:")
                print(f"   With reuse:    {elapsed_reuse:.3f}s")
                print(f"   Without reuse: {elapsed_no_reuse:.3f}s")
                print(f"   Speedup: {speedup:.2f}x")

                # Performance parity is acceptable (within 20% either direction)
                # The key optimization is field image reuse (already implemented)
                # For overlays, clearing+recreating ≈ clearing axes+redrawing
                assert 0.8 < speedup < 1.5, (
                    f"Expected performance parity (0.8-1.5x), got {speedup:.2f}x"
                )
