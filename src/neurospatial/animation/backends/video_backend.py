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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is installed and accessible.

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
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
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
    **kwargs,
) -> Path | None:
    """Export animation as video using parallel frame rendering.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure
    fields : list of arrays
        Fields to animate
    save_path : str
        Output path for video file
    fps : int, default=30
        Frames per second for playback
    cmap : str, default="viridis"
        Matplotlib colormap name
    vmin, vmax : float, optional
        Color scale limits. If None, computed from all fields.
    frame_labels : list of str, optional
        Frame labels (e.g., ["Trial 1", "Trial 2", ...])
    dpi : int, default=100
        Resolution for rendering
    codec : str, default="h264"
        Video codec (h264, h265, vp9, mpeg4)
    crf : int, default=18
        Constant Rate Factor for quality control (0-51).
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
    bitrate : int | None, optional
        Video bitrate in kbps. If specified, overrides CRF-based
        encoding and uses constant bitrate instead.
        Default: None (uses CRF mode for better quality/size ratio)
    n_workers : int, optional
        Parallel workers for rendering (default: CPU count / 2)
    dry_run : bool, default=False
        If True, estimate time and file size without rendering.
        Returns None after printing estimate.
    title : str
        Animation title (unused in video, for compatibility)
    **kwargs : dict
        Additional parameters (accepted for compatibility)

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

    **Memory Management:**
    - Environment must be pickle-able for parallel rendering
    - Call env.clear_cache() if pickle errors occur
    - Use n_workers=1 for serial rendering (no pickle needed)
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
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed:\n{result.stderr}")

        print(f"✓ Video saved to {output_path}")
        return output_path

    finally:
        # Clean up temporary frames
        shutil.rmtree(tmpdir)
