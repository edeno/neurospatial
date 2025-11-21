"""Benchmark dataset generators for animation performance testing.

This module provides functions to create reproducible benchmark datasets
for testing animation backend performance. Each generator supports seeded
random number generation for reproducibility.

Examples
--------
>>> from benchmark_datasets import SMALL_CONFIG, create_benchmark_env
>>> env = create_benchmark_env(SMALL_CONFIG, seed=42)
>>> env.n_bins
1521
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.animation.overlays import (
        BodypartOverlay,
        HeadDirectionOverlay,
        PositionOverlay,
    )


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark dataset.

    Parameters
    ----------
    name : str
        Human-readable name for this benchmark configuration.
    n_frames : int
        Number of animation frames to generate.
    grid_size : int
        Size of the square grid (grid_size x grid_size bins).
    include_position : bool, optional
        Whether to include a position overlay. Default is True.
    include_skeleton : bool, optional
        Whether to include a bodypart overlay with skeleton. Default is False.
    include_head_direction : bool, optional
        Whether to include a head direction overlay. Default is False.
    n_bodyparts : int, optional
        Number of body parts for skeleton overlay. Default is 5.
    trail_length : int, optional
        Trail length for position overlay. Default is 10.
    """

    name: str
    n_frames: int
    grid_size: int
    include_position: bool = True
    include_skeleton: bool = False
    include_head_direction: bool = False
    n_bodyparts: int = 5
    trail_length: int = 10


# Pre-defined benchmark configurations
SMALL_CONFIG = BenchmarkConfig(
    name="small",
    n_frames=100,
    grid_size=40,
    include_position=True,
    include_skeleton=False,
    include_head_direction=False,
)

MEDIUM_CONFIG = BenchmarkConfig(
    name="medium",
    n_frames=5000,
    grid_size=100,
    include_position=True,
    include_skeleton=True,
    include_head_direction=True,
)

LARGE_CONFIG = BenchmarkConfig(
    name="large",
    n_frames=100_000,
    grid_size=100,
    include_position=True,
    include_skeleton=True,
    include_head_direction=True,
    n_bodyparts=7,
    trail_length=20,
)


def create_benchmark_env(
    config: BenchmarkConfig,
    seed: int | None = None,
) -> Environment:
    """Create a benchmark environment.

    Parameters
    ----------
    config : BenchmarkConfig
        Configuration specifying grid size and other parameters.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    Environment
        A fitted Environment with the specified grid size.

    Examples
    --------
    >>> from benchmark_datasets import SMALL_CONFIG, create_benchmark_env
    >>> env = create_benchmark_env(SMALL_CONFIG, seed=42)
    >>> env._is_fitted
    True
    """
    from neurospatial import Environment

    rng = np.random.default_rng(seed)

    # Create positions that span the full grid
    grid_extent = config.grid_size
    n_samples = max(1000, config.grid_size * 10)

    # Uniform random positions across the grid
    positions = rng.uniform(0, grid_extent, size=(n_samples, 2))

    # Create environment with bin_size=1.0 to get approximately grid_size bins
    env = Environment.from_samples(positions, bin_size=1.0)

    return env


def create_benchmark_fields(
    env: Environment,
    config: BenchmarkConfig,
    seed: int | None = None,
    memmap_path: Path | str | None = None,
) -> NDArray[np.float32]:
    """Create benchmark field data.

    Generates synthetic spatial fields for animation benchmarking.
    Fields are smooth Gaussian blobs that drift across the environment.

    Parameters
    ----------
    env : Environment
        Environment to generate fields for.
    config : BenchmarkConfig
        Configuration specifying number of frames.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    memmap_path : Path or str, optional
        If provided, create a memory-mapped array at this path.
        Useful for large datasets that don't fit in RAM.

    Returns
    -------
    NDArray[np.float32]
        Array of shape (n_frames, n_bins) with values in [0, 1].

    Examples
    --------
    >>> from benchmark_datasets import (
    ...     SMALL_CONFIG,
    ...     create_benchmark_env,
    ...     create_benchmark_fields,
    ... )
    >>> env = create_benchmark_env(SMALL_CONFIG, seed=42)
    >>> fields = create_benchmark_fields(env, SMALL_CONFIG, seed=42)
    >>> fields.shape
    (100, 1521)
    """
    rng = np.random.default_rng(seed)

    n_frames = config.n_frames
    n_bins = env.n_bins

    # Create output array (regular or memory-mapped)
    fields: NDArray[np.float32]
    if memmap_path is not None:
        memmap_path = Path(memmap_path)
        fields = np.memmap(
            memmap_path,
            dtype=np.float32,
            mode="w+",
            shape=(n_frames, n_bins),
        )
    else:
        fields = np.zeros((n_frames, n_bins), dtype=np.float32)

    # Generate smooth drifting Gaussian blobs
    bin_centers = env.bin_centers  # (n_bins, n_dims)

    # Parameters for the drifting blob
    n_blobs = 3  # Number of overlapping blobs
    blob_width = config.grid_size / 4  # Width of each blob

    # Generate random blob trajectories
    blob_centers = np.zeros((n_blobs, n_frames, 2))
    for i in range(n_blobs):
        # Random walk starting point
        start = rng.uniform(0, config.grid_size, size=2)
        # Random walk with drift
        steps = rng.normal(0, config.grid_size / n_frames, size=(n_frames, 2))
        trajectory = np.cumsum(steps, axis=0) + start
        # Wrap around boundaries
        trajectory = np.mod(trajectory, config.grid_size)
        blob_centers[i] = trajectory

    # Generate fields frame by frame
    for frame_idx in range(n_frames):
        field = np.zeros(n_bins, dtype=np.float32)

        for blob_idx in range(n_blobs):
            center = blob_centers[blob_idx, frame_idx]
            # Compute distances from blob center
            dists = np.linalg.norm(bin_centers - center, axis=1)
            # Gaussian blob
            blob = np.exp(-(dists**2) / (2 * blob_width**2))
            field += blob

        # Normalize to [0, 1]
        if field.max() > 0:
            field /= field.max()

        fields[frame_idx] = field

    # Flush if memmap
    if hasattr(fields, "flush"):
        fields.flush()

    return fields


def create_benchmark_overlays(
    env: Environment,
    config: BenchmarkConfig,
    seed: int | None = None,
) -> list[PositionOverlay | BodypartOverlay | HeadDirectionOverlay]:
    """Create benchmark overlay data.

    Generates synthetic overlay data (position, skeleton, head direction)
    based on the configuration.

    Parameters
    ----------
    env : Environment
        Environment to generate overlays for.
    config : BenchmarkConfig
        Configuration specifying which overlays to include.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    list
        List of overlay objects (PositionOverlay, BodypartOverlay,
        HeadDirectionOverlay) based on config settings.

    Examples
    --------
    >>> from benchmark_datasets import (
    ...     SMALL_CONFIG,
    ...     create_benchmark_env,
    ...     create_benchmark_overlays,
    ... )
    >>> env = create_benchmark_env(SMALL_CONFIG, seed=42)
    >>> overlays = create_benchmark_overlays(env, SMALL_CONFIG, seed=42)
    >>> len(overlays) >= 1
    True
    """
    from neurospatial.animation.overlays import (
        BodypartOverlay,
        HeadDirectionOverlay,
        PositionOverlay,
    )
    from neurospatial.animation.skeleton import Skeleton

    rng = np.random.default_rng(seed)

    overlays: list[PositionOverlay | BodypartOverlay | HeadDirectionOverlay] = []

    n_frames = config.n_frames
    dim_ranges = env.dimension_ranges
    if dim_ranges is None:
        raise ValueError("Environment must have dimension_ranges set")
    dim_ranges_list: list[tuple[float, float]] = list(dim_ranges)

    # Generate a smooth trajectory for position overlay
    if config.include_position or config.include_skeleton:
        # Create smooth random walk trajectory
        trajectory = _generate_smooth_trajectory(
            n_frames=n_frames,
            dim_ranges=dim_ranges_list,
            rng=rng,
        )

    # Position overlay
    if config.include_position:
        position_overlay = PositionOverlay(
            data=trajectory.copy(),
            color="red",
            size=10.0,
            trail_length=config.trail_length,
        )
        overlays.append(position_overlay)

    # Bodypart overlay with skeleton
    if config.include_skeleton:
        # Generate body parts around the trajectory
        bodypart_names = [f"bp{i}" for i in range(config.n_bodyparts)]

        # Create skeleton edges (chain topology)
        edges = [
            (bodypart_names[i], bodypart_names[i + 1])
            for i in range(config.n_bodyparts - 1)
        ]

        skeleton = Skeleton(
            name="benchmark_skeleton",
            nodes=tuple(bodypart_names),
            edges=tuple(edges),
            node_colors=dict.fromkeys(bodypart_names, "white"),
            edge_color="gray",
            edge_width=2.0,
        )

        # Generate bodypart positions relative to trajectory
        bodypart_data: dict[str, NDArray[np.float64]] = {}
        for _i, bp_name in enumerate(bodypart_names):
            # Each bodypart has a fixed offset from the main trajectory
            offset = rng.uniform(-2, 2, size=2)
            # Add some per-frame jitter
            jitter = rng.normal(0, 0.5, size=(n_frames, 2))
            bp_positions = trajectory + offset + jitter
            # Clip to environment bounds
            for dim in range(2):
                dim_min, dim_max = dim_ranges_list[dim]
                bp_positions[:, dim] = np.clip(bp_positions[:, dim], dim_min, dim_max)
            bodypart_data[bp_name] = bp_positions

        bodypart_overlay = BodypartOverlay(
            data=bodypart_data,
            skeleton=skeleton,
        )
        overlays.append(bodypart_overlay)

    # Head direction overlay
    if config.include_head_direction:
        # Generate smooth head direction from trajectory velocity
        if config.include_position or config.include_skeleton:
            # Use velocity direction as head direction
            velocity = np.diff(trajectory, axis=0, prepend=trajectory[:1])
            head_angles = np.arctan2(velocity[:, 1], velocity[:, 0])
            # Add some noise
            head_angles += rng.normal(0, 0.1, size=n_frames)
        else:
            # Random head direction
            head_angles = rng.uniform(-np.pi, np.pi, size=n_frames)

        # Smooth the angles
        from scipy.ndimage import gaussian_filter1d

        head_angles = gaussian_filter1d(head_angles, sigma=5)

        # Wrap to [-pi, pi]
        head_angles = np.arctan2(np.sin(head_angles), np.cos(head_angles))

        head_direction_overlay = HeadDirectionOverlay(
            data=head_angles,
            color="yellow",
            length=3.0,
        )
        overlays.append(head_direction_overlay)

    return overlays


def _generate_smooth_trajectory(
    n_frames: int,
    dim_ranges: list[tuple[float, float]],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate a smooth random walk trajectory.

    Parameters
    ----------
    n_frames : int
        Number of frames.
    dim_ranges : list of (min, max) tuples
        Bounds for each dimension.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    NDArray[np.float64]
        Trajectory of shape (n_frames, n_dims).
    """
    n_dims = len(dim_ranges)

    # Start at a random position
    trajectory = np.zeros((n_frames, n_dims))
    for dim in range(n_dims):
        dim_min, dim_max = dim_ranges[dim]
        trajectory[0, dim] = rng.uniform(dim_min, dim_max)

    # Generate smooth random walk
    step_size = 0.5  # Average step size per frame
    for frame in range(1, n_frames):
        # Random direction
        step = rng.normal(0, step_size, size=n_dims)
        trajectory[frame] = trajectory[frame - 1] + step

        # Reflect at boundaries
        for dim in range(n_dims):
            dim_min, dim_max = dim_ranges[dim]
            pos = trajectory[frame, dim]
            while pos < dim_min or pos > dim_max:
                if pos < dim_min:
                    pos = 2 * dim_min - pos
                if pos > dim_max:
                    pos = 2 * dim_max - pos
            trajectory[frame, dim] = pos

    return trajectory
