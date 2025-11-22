"""Parallel video export backend.

Based on parallel rendering approach from:
https://gist.github.com/edeno/652ee10a76481f00b3eb08906b41c6bf
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is installed and accessible.

    Uses shutil.which() for fast path detection before falling back to
    subprocess invocation. This avoids subprocess overhead when ffmpeg
    is not on PATH.

    Returns
    -------
    available : bool
        True if ffmpeg is available

    Examples
    --------
    .. code-block:: python

        if check_ffmpeg_available():
            print("ffmpeg is installed")
        else:
            print("Please install ffmpeg")
    """
    # Fast path: check if ffmpeg is on PATH
    if shutil.which("ffmpeg") is None:
        return False

    # Verify it actually runs (handles edge cases like broken symlinks)
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def render_video(
    env: Environment,
    fields: list[NDArray[np.float64]],
    save_path: str,
    *,
    fps: int = 30,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: list[str] | None = None,
    dpi: int = 100,
    codec: str = "h264",
    crf: int = 18,
    preset: str = "medium",
    bitrate: int | None = None,
    n_workers: int | None = None,
    dry_run: bool = False,
    title: str = "Spatial Field Animation",
    overlay_data: Any | None = None,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
    **kwargs,
) -> Path | None:
    """Export animation as video using parallel frame rendering.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure.
    fields : list of ndarray of shape (n_bins,), dtype float64
        Fields to animate. Each array contains field values for one frame.
    save_path : str
        Output path for video file (e.g., "output.mp4").
    fps : int, default=30
        Frames per second for playback.
    cmap : str, default="viridis"
        Matplotlib colormap name (e.g., "viridis", "hot", "plasma").
    vmin : float, optional
        Minimum value for color scale normalization. If None, computed from
        all fields using NaN-robust min.
    vmax : float, optional
        Maximum value for color scale normalization. If None, computed from
        all fields using NaN-robust max.
    frame_labels : list of str, optional
        Frame labels (e.g., ["Trial 1", "Trial 2", ...]).
    dpi : int, default=100
        Resolution for rendering in dots per inch.
    codec : str, default="h264"
        Video codec (h264, h265, vp9, mpeg4).
    crf : int, default=18
        Constant Rate Factor for quality control, range [0, 51].
        Lower values = higher quality and larger files.
        - 0: Lossless (very large files)
        - 18: "Visually lossless" - recommended default
        - 23: Default for libx264 (good quality)
        - 28: Noticeable quality loss
        - 51: Worst quality
        Only used when bitrate=None.
    preset : str, default="medium"
        Encoding speed preset (faster = larger file).
        Options: ultrafast, superfast, veryfast, faster, fast,
        medium, slow, slower, veryslow.
        - ultrafast: Fastest encoding, largest file
        - medium: Balanced speed/compression (default)
        - veryslow: Best compression, slowest encoding
    bitrate : int or None, optional
        Video bitrate in kbps. If specified, overrides CRF-based
        encoding and uses constant bitrate instead.
        Default: None (uses CRF mode for better quality/size ratio).
    n_workers : int, optional
        Parallel workers for rendering (default: CPU count / 2).
    dry_run : bool, default=False
        If True, estimate time and file size without rendering.
        Returns None after printing estimate.
    title : str
        Animation title (unused in video, for compatibility).
    overlay_data : OverlayData or None, optional
        Overlay data to render on top of fields. Contains positions,
        bodyparts, and head direction data aligned to frames.
        Default is None (no overlays).
    show_regions : bool or list of str, default=False
        If True, render all regions from environment. If list of strings,
        render only specified regions. Default is False (no regions).
    region_alpha : float, default=0.3
        Alpha transparency for region overlays, range [0.0, 1.0] where 0.0 is
        fully transparent and 1.0 is fully opaque. Only used if show_regions
        is True or a non-empty list. Default is 0.3.
    **kwargs : dict
        Additional parameters (accepted for compatibility).

    Returns
    -------
    save_path : Path or None
        Path to exported video file, or None if dry_run=True

    Raises
    ------
    RuntimeError
        If ffmpeg is not installed or encoding fails

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from neurospatial import Environment

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(20)]

        # Dry run to estimate time/size
        render_video(env, fields, "output.mp4", dry_run=True, n_workers=4)

        # Actual rendering with CRF mode (default)
        path = render_video(env, fields, "output.mp4", fps=10, n_workers=4)
        print(f"Video saved to {path}")

        # High quality with slower encoding
        path = render_video(
            env, fields, "high_quality.mp4", crf=15, preset="slow", n_workers=4
        )

        # Use constant bitrate mode (legacy behavior)
        path = render_video(
            env, fields, "constant_bitrate.mp4", bitrate=5000, n_workers=4
        )

    Notes
    -----
    **Parallel Rendering:**
    - Each worker process has its own matplotlib figure
    - Frames are partitioned evenly across workers
    - Workers render to temporary PNG files
    - ffmpeg combines PNGs into final video

    **Codec Selection:**
    - h264: Most compatible, good quality (default)
    - h265: Better compression, less compatible
    - vp9: Open source, good for web
    - mpeg4: Older, wider compatibility

    **Quality Control:**
    - CRF mode (default): Better quality/size ratio than constant bitrate
    - CRF values: 18 (visually lossless), 23 (good), 28 (noticeable loss)
    - Constant bitrate mode: Specify bitrate parameter to override CRF
    - Web compatibility: Uses yuv420p pixel format and faststart flag

    **Parallel Rendering Requirements:**
    When using ``n_workers > 1``, both environment and overlay_data must be
    pickle-able to send to worker processes. Pickle-ability is automatically
    validated before parallel rendering begins.

    If you encounter pickle errors:

    1. **For environment**: Call ``env.clear_cache()`` before rendering
    2. **For overlay_data**: Remove unpickleable objects (lambdas, closures)
    3. **Alternative**: Use ``n_workers=1`` for serial rendering (no pickle needed)

    Serial rendering (``n_workers=1``) does not require pickle-ability.
    """
    from neurospatial.animation._parallel import parallel_render_frames
    from neurospatial.animation.rendering import (
        compute_global_colormap_range,
        render_field_to_rgb,
    )

    # Validate ffmpeg available
    if not check_ffmpeg_available():
        raise RuntimeError(
            "Video export requires ffmpeg. Install:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html"
        )

    # Compute global color scale
    vmin, vmax = compute_global_colormap_range(fields, vmin, vmax)

    # Determine number of workers
    if n_workers is None:
        cpu_count = os.cpu_count() or 2  # Default to 2 if cpu_count() returns None
        n_workers = max(1, cpu_count // 2)
    elif n_workers < 1:
        raise ValueError(f"n_workers must be positive (got {n_workers})")

    # Warn about high DPI (can cause large file sizes and slow rendering)
    if dpi > 150:
        # Estimate output resolution (matplotlib default figsize is 8x6)
        default_figsize = (8, 6)  # inches
        width_px = int(default_figsize[0] * dpi)
        height_px = int(default_figsize[1] * dpi)
        warnings.warn(
            f"High DPI detected: dpi={dpi} will produce {width_px}x{height_px} pixel frames.\n"
            f"This may result in:\n"
            f"  - Large video file sizes\n"
            f"  - Slow rendering times\n"
            f"  - High memory usage\n"
            f"Consider using dpi=100 or dpi=150 for most use cases.",
            UserWarning,
            stacklevel=2,
        )

    # Dry run: estimate time and file size
    if dry_run:
        print("Running dry run estimation...")

        # Render one test frame to measure timing
        start = time.time()
        _ = render_field_to_rgb(env, fields[0], cmap, vmin, vmax, dpi)
        frame_time = time.time() - start

        # Estimate total time (with parallelization)
        total_time = frame_time * len(fields) / n_workers

        # Estimate file size (rough approximation)
        if bitrate is not None:
            # Constant bitrate mode
            frame_size_base_kb = 50
            frame_size_kb = (dpi / 100) ** 2 * frame_size_base_kb
            estimated_mb = frame_size_kb * len(fields) / 1024 * (bitrate / 5000)
        else:
            # CRF mode (typically smaller files than constant bitrate)
            # Empirical: CRF 18 produces ~30% smaller files than 5000 kbps
            frame_size_base_kb = 35  # Lower base for CRF mode
            frame_size_kb = (dpi / 100) ** 2 * frame_size_base_kb
            # CRF scaling: lower CRF = larger files
            crf_factor = 1.0 + (23 - crf) * 0.1  # 23 is libx264 default
            estimated_mb = frame_size_kb * len(fields) / 1024 * crf_factor

        print(f"\n{'=' * 60}")
        print("Video Export Dry Run Estimate:")
        print(f"{'=' * 60}")
        print(f"  Frames:          {len(fields):,}")
        print(f"  Workers:         {n_workers}")
        print(f"  Frame time:      {frame_time * 1000:.1f} ms")
        print(f"  Est. total time: {total_time / 60:.1f} minutes")
        print(f"  Est. file size:  {estimated_mb:.0f} MB")
        print(f"  Output path:     {save_path}")
        print("\nTo proceed, call again with dry_run=False")
        print(f"{'=' * 60}\n")
        return None

    print(f"Rendering {len(fields)} frames using {n_workers} workers...")
    print(f"Estimated time: ~{len(fields) * 0.5 / n_workers:.0f} seconds")

    # Create temporary directory for frames
    tmpdir = tempfile.mkdtemp(prefix="neurospatial_animation_")

    try:
        # Render frames in parallel
        frame_pattern = parallel_render_frames(
            env=env,
            fields=fields,
            output_dir=tmpdir,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            frame_labels=frame_labels,
            dpi=dpi,
            n_workers=n_workers,
            overlay_data=overlay_data,
            show_regions=show_regions,
            region_alpha=region_alpha,
        )

        # Encode video with ffmpeg
        print("Encoding video...")
        output_path = Path(save_path)

        # Select codec
        codec_map = {
            "h264": "libx264",
            "h265": "libx265",
            "vp9": "libvpx-vp9",
            "mpeg4": "mpeg4",
        }
        ffmpeg_codec = codec_map.get(codec, codec)

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-vf",
            "scale=ceil(iw/2)*2:ceil(ih/2)*2",  # Ensure even dimensions for h264
            "-c:v",
            ffmpeg_codec,
            "-pix_fmt",
            "yuv420p",  # Explicit pixel format for compatibility
            "-preset",
            preset,
        ]

        # Add quality control (CRF mode by default, bitrate if specified)
        if bitrate is not None:
            cmd.extend(["-b:v", f"{bitrate}k"])
        else:
            cmd.extend(["-crf", str(crf)])

        # Web-friendly flags
        cmd.extend(
            [
                "-movflags",
                "+faststart",  # Move moov atom to start for web streaming
                "-r",
                str(fps),  # Output framerate
                "-threads",
                str(n_workers),  # Encoding threads
                str(output_path),
            ]
        )

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,  # Discard ffmpeg progress output (avoids buffer issues)
            stderr=subprocess.PIPE,  # Capture errors for reporting
            text=True,
            check=False,
        )

        if result.returncode != 0:
            # Build command string for error message (filter None for safety)
            cmd_str = " ".join(str(c) for c in cmd if c is not None)
            # Provide actionable error message with full context
            raise RuntimeError(
                f"ffmpeg encoding failed (exit code {result.returncode}).\n"
                f"\n"
                f"Command:\n  {cmd_str}\n"
                f"\n"
                f"Error output:\n{result.stderr}\n"
                f"\n"
                f"Common fixes:\n"
                f"  - Ensure ffmpeg supports codec '{ffmpeg_codec}' (try: ffmpeg -encoders | grep {ffmpeg_codec})\n"
                f"  - Check output path is writable: {output_path}\n"
                f"  - For hardware acceleration, try codec='h264_videotoolbox' (macOS) "
                f"or 'h264_nvenc' (NVIDIA)"
            )

        print(f"✓ Video saved to {output_path}")
        return output_path

    finally:
        # Clean up temporary frames
        shutil.rmtree(tmpdir)
