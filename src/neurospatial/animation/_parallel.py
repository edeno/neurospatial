"""Parallel frame rendering utilities.

Based on approach from:
https://gist.github.com/edeno/652ee10a76481f00b3eb08906b41c6bf

Key principles:
- Each worker process has its own matplotlib figure (avoid threading issues)
- Frames saved as numbered PNGs for ffmpeg pattern matching
- Workers operate independently on partitioned frame ranges
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def parallel_render_frames(
    env: Environment,
    fields: list[NDArray[np.float64]],
    output_dir: str,
    cmap: str,
    vmin: float,
    vmax: float,
    frame_labels: list[str] | None,
    dpi: int,
    n_workers: int,
) -> str:
    """Render frames in parallel across worker processes.

    Parameters
    ----------
    env : Environment
        Must be pickle-able (will be serialized to workers)
    fields : list of arrays
        All fields to render
    output_dir : str
        Directory to save frame PNGs
    cmap : str
        Colormap name
    vmin, vmax : float
        Color scale limits
    frame_labels : list of str or None
        Frame labels for each frame
    dpi : int
        Resolution
    n_workers : int
        Number of parallel workers

    Returns
    -------
    frame_pattern : str
        ffmpeg input pattern (e.g., "/tmp/frame_%05d.png")

    Raises
    ------
    ValueError
        If environment is not pickle-able

    Examples
    --------
    .. code-block:: python

        import tempfile
        import numpy as np
        from neurospatial import Environment

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(10)]

        with tempfile.TemporaryDirectory() as tmpdir:
            pattern = parallel_render_frames(
                env, fields, tmpdir, "viridis", 0.0, 1.0, None, 100, 2
            )
            print("frame_" in pattern and ".png" in pattern)
            # True
    """
    import pickle

    n_frames = len(fields)

    # Validate environment is pickle-able
    try:
        pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise ValueError(
            f"Environment must be pickle-able for parallel rendering: {e}"
        ) from e

    # Partition frames across workers
    frames_per_worker = n_frames // n_workers
    worker_tasks = []

    for worker_id in range(n_workers):
        start_idx = worker_id * frames_per_worker
        if worker_id == n_workers - 1:
            # Last worker takes remainder
            end_idx = n_frames
        else:
            end_idx = start_idx + frames_per_worker

        worker_fields = fields[start_idx:end_idx]
        worker_frame_labels = frame_labels[start_idx:end_idx] if frame_labels else None

        worker_tasks.append(
            {
                "env": env,
                "fields": worker_fields,
                "start_frame_idx": start_idx,
                "output_dir": output_dir,
                "cmap": cmap,
                "vmin": vmin,
                "vmax": vmax,
                "frame_labels": worker_frame_labels,
                "dpi": dpi,
                "n_total_frames": n_frames,  # For consistent filename padding
            }
        )

    # Render in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(
            tqdm(
                executor.map(_render_worker_frames, worker_tasks),
                total=n_workers,
                desc="Workers",
            )
        )

    # Return ffmpeg pattern (0-indexed for compatibility)
    # ffmpeg expects: frame_00000.png, frame_00001.png, etc.
    digits = len(str(n_frames))
    pattern = str(Path(output_dir) / f"frame_%0{digits}d.png")

    return pattern


def _render_worker_frames(task: dict) -> None:
    """Render frames in a worker process.

    Each worker creates its own matplotlib figure to avoid
    threading issues and memory accumulation.

    Parameters
    ----------
    task : dict
        Worker task specification with keys:
        - env: Environment
        - fields: list of fields to render
        - start_frame_idx: global frame index offset
        - output_dir: where to save PNGs
        - cmap, vmin, vmax: colormap settings
        - frame_labels: optional frame labels
        - dpi: resolution
        - n_total_frames: total frame count (for filename padding)

    Notes
    -----
    This function is called by ProcessPoolExecutor and must be
    pickle-able (i.e., defined at module level, not nested).

    Examples
    --------
    .. code-block:: python

        import tempfile
        from pathlib import Path
        import numpy as np
        from neurospatial import Environment

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(3)]

        with tempfile.TemporaryDirectory() as tmpdir:
            task = {
                "env": env,
                "fields": fields,
                "start_frame_idx": 0,
                "output_dir": tmpdir,
                "cmap": "viridis",
                "vmin": 0.0,
                "vmax": 1.0,
                "frame_labels": None,
                "dpi": 50,
                "n_total_frames": 3,
            }
            _render_worker_frames(task)
            png_files = list(Path(tmpdir).glob("frame_*.png"))
            len(png_files)
            # 3
    """
    env = task["env"]
    fields = task["fields"]
    start_idx = task["start_frame_idx"]
    output_dir = task["output_dir"]
    cmap = task["cmap"]
    vmin = task["vmin"]
    vmax = task["vmax"]
    frame_labels = task["frame_labels"]
    dpi = task["dpi"]

    # Create figure once for this worker
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    try:
        for local_idx, field in enumerate(fields):
            global_idx = start_idx + local_idx

            # Clear previous frame
            ax.clear()

            # Render field using environment's plot_field
            env.plot_field(
                field,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                colorbar=False,
            )

            # Add label if provided
            if frame_labels and frame_labels[local_idx]:
                ax.set_title(frame_labels[local_idx], fontsize=14)

            # Save frame (0-indexed for ffmpeg compatibility)
            frame_number = global_idx
            # Get digits from task (passed from parallel_render_frames)
            # Use safe default if not provided
            n_total_frames = task.get("n_total_frames", len(fields) * 100)
            digits = len(str(n_total_frames))
            filename = f"frame_{frame_number:0{digits}d}.png"
            filepath = Path(output_dir) / filename

            fig.savefig(filepath, bbox_inches="tight", dpi=dpi)
    finally:
        # Clean up figure (prevent memory leaks)
        plt.close(fig)
