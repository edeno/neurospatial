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
    reuse_artists: bool = True,
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
    reuse_artists : bool, default=True
        If True, reuse the same AxesImage artist across frames (fast path).
        Updates image data only, avoiding layout/allocation overhead.
        If False, clear and redraw each frame (original behavior).

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

    # Cap workers to available frames
    n_workers = min(n_workers, max(1, n_frames))

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
                "digits": max(5, len(str(max(0, n_frames - 1)))),  # Pass to workers
                "reuse_artists": reuse_artists,
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
    digits = max(5, len(str(max(0, n_frames - 1))))
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
        - digits: number of digits for frame padding

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
                "digits": 5,
            }
            _render_worker_frames(task)
            png_files = list(Path(tmpdir).glob("frame_*.png"))
            len(png_files)
            # 3
    """
    # Set Agg backend BEFORE any pyplot imports
    try:
        import matplotlib

        if matplotlib.get_backend().lower() not in (
            "agg",
            "module://matplotlib_inline.backend_inline",
        ):
            matplotlib.use("Agg", force=True)
    except Exception:
        pass

    # Import pyplot only AFTER backend is set (avoids GUI backend binding in workers)
    import matplotlib.pyplot as plt
    from matplotlib.image import AxesImage

    env = task["env"]
    fields = task["fields"]
    start_idx = task["start_frame_idx"]
    output_dir = task["output_dir"]
    cmap = task["cmap"]
    vmin = task["vmin"]
    vmax = task["vmax"]
    frame_labels = task["frame_labels"]
    dpi = task["dpi"]
    # Get digits from task, fallback for backward compatibility with tests
    digits = task.get("digits", 5)
    reuse_flag = task.get("reuse_artists", True)

    # Lean rcParams for bulk rasterization
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "axes.xmargin": 0,
            "axes.ymargin": 0,
            "path.simplify": True,
            "path.simplify_threshold": 0.5,
            "agg.path.chunksize": 10000,
        }
    )

    # Create figure once for this worker
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    ax.set_axis_off()

    # Render first frame normally to establish artists and limits
    env.plot_field(
        fields[0],
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
    )

    # Add label for first frame if provided
    if frame_labels and frame_labels[0]:
        ax.set_title(frame_labels[0], fontsize=14)

    # Try to identify the primary image artist to update
    # This assumes env.plot_field uses imshow/Images; otherwise we fall back
    primary_im: AxesImage | None = ax.images[0] if ax.images else None
    reuse_artists = bool(reuse_flag and primary_im is not None)

    # Freeze autoscale to avoid changing extents while updating data
    if reuse_artists:
        ax.set_autoscale_on(False)

    try:
        # Save frame 0
        frame_number = start_idx
        filename = f"frame_{frame_number:0{digits}d}.png"
        filepath = Path(output_dir) / filename
        fig.savefig(filepath)

        if reuse_artists:
            # Fast path: update the image data only
            # Type checker: reuse_artists=True guarantees primary_im is not None
            assert primary_im is not None
            for local_idx in range(1, len(fields)):
                field = fields[local_idx]

                # Matplotlib wants array-like; ensure C-order to avoid copies later
                # If `field` is masked or not C-contiguous, ascontiguousarray avoids hidden copies
                if not isinstance(field, np.ndarray) or not field.flags["C_CONTIGUOUS"]:
                    data = np.ascontiguousarray(np.array(field))
                else:
                    data = field

                primary_im.set_data(data)  # reuse the same artist

                # Update title if labels provided
                if frame_labels and frame_labels[local_idx]:
                    ax.set_title(frame_labels[local_idx], fontsize=14)

                # No need to clear or re-layout. Draw and save.
                frame_number = start_idx + local_idx
                filename = f"frame_{frame_number:0{digits}d}.png"
                filepath = Path(output_dir) / filename
                fig.savefig(filepath)
        else:
            # Fallback: redraw per frame (original behavior)
            for local_idx in range(1, len(fields)):
                ax.clear()
                ax.set_axis_off()
                env.plot_field(
                    fields[local_idx],
                    ax=ax,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar=False,
                )

                # Add label if provided
                if frame_labels and frame_labels[local_idx]:
                    ax.set_title(frame_labels[local_idx], fontsize=14)

                frame_number = start_idx + local_idx
                filename = f"frame_{frame_number:0{digits}d}.png"
                filepath = Path(output_dir) / filename
                fig.savefig(filepath)
    finally:
        # Clean up figure (prevent memory leaks)
        plt.close(fig)
