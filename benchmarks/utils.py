"""Shared utilities for benchmark scripts.

This module provides timing and memory measurement utilities
for animation backend performance testing.
"""

from __future__ import annotations

import gc
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import psutil


@dataclass
class TimingResult:
    """Result from a timing measurement.

    Parameters
    ----------
    name : str
        Name of the operation timed.
    elapsed_ms : float
        Elapsed time in milliseconds.
    memory_mb : float
        Memory usage in megabytes at end of operation.
    peak_memory_mb : float
        Peak memory usage during operation.
    """

    name: str
    elapsed_ms: float
    memory_mb: float
    peak_memory_mb: float = 0.0


@dataclass
class BenchmarkResult:
    """Collection of timing results from a benchmark run.

    Parameters
    ----------
    config_name : str
        Name of the benchmark configuration.
    backend : str
        Backend being benchmarked.
    timings : list of TimingResult
        Individual timing measurements.
    """

    config_name: str
    backend: str
    timings: list[TimingResult] = field(default_factory=list)

    def add(self, result: TimingResult) -> None:
        """Add a timing result."""
        self.timings.append(result)

    def print_table(self) -> None:
        """Print results as a formatted table."""
        print(f"\n## {self.backend.title()} Backend - {self.config_name}")
        print()
        print("| Metric | Time (ms) | Memory (MB) | Peak Memory (MB) |")
        print("|--------|-----------|-------------|------------------|")
        for t in self.timings:
            print(
                f"| {t.name} | {t.elapsed_ms:.2f} | {t.memory_mb:.2f} | {t.peak_memory_mb:.2f} |"
            )
        print()


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


def force_gc() -> None:
    """Force garbage collection to get clean memory baseline."""
    gc.collect()
    gc.collect()
    gc.collect()


@contextmanager
def timer(name: str) -> Iterator[TimingResult]:
    """Context manager for timing operations with memory tracking.

    Parameters
    ----------
    name : str
        Name for this timing measurement.

    Yields
    ------
    TimingResult
        Result object that will be populated when context exits.

    Examples
    --------
    >>> with timer("my_operation") as result:
    ...     # do something
    ...     pass
    >>> print(result.elapsed_ms)  # doctest: +SKIP
    """
    force_gc()
    start_memory = get_memory_mb()
    peak_memory = start_memory

    result = TimingResult(name=name, elapsed_ms=0.0, memory_mb=0.0)
    start_time = time.perf_counter()

    try:
        yield result
    finally:
        end_time = time.perf_counter()
        end_memory = get_memory_mb()

        result.elapsed_ms = (end_time - start_time) * 1000
        result.memory_mb = end_memory
        result.peak_memory_mb = max(peak_memory, end_memory)


def setup_sys_path() -> None:
    """Add scripts/ directory to sys.path for benchmark_datasets import."""
    scripts_dir = Path(__file__).parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def get_benchmark_configs() -> list[str]:
    """Return list of available benchmark config names."""
    return ["small", "medium", "large"]


def load_benchmark_config(name: str):
    """Load a benchmark configuration by name.

    Parameters
    ----------
    name : str
        One of "small", "medium", "large".

    Returns
    -------
    BenchmarkConfig
        Configuration for the named benchmark.
    """
    setup_sys_path()
    from benchmark_datasets import LARGE_CONFIG, MEDIUM_CONFIG, SMALL_CONFIG

    configs = {
        "small": SMALL_CONFIG,
        "medium": MEDIUM_CONFIG,
        "large": LARGE_CONFIG,
    }
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")
    return configs[name]


def create_test_data(config_name: str, seed: int = 42):
    """Create environment, fields, and overlays for a benchmark config.

    Parameters
    ----------
    config_name : str
        One of "small", "medium", "large".
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    env : Environment
        Fitted environment.
    fields : ndarray
        Field data of shape (n_frames, n_bins).
    overlays : list
        List of overlay objects.
    config : BenchmarkConfig
        The configuration used.
    """
    setup_sys_path()
    from benchmark_datasets import (
        create_benchmark_env,
        create_benchmark_fields,
        create_benchmark_overlays,
    )

    config = load_benchmark_config(config_name)
    env = create_benchmark_env(config, seed=seed)
    fields = create_benchmark_fields(env, config, seed=seed)
    overlays = create_benchmark_overlays(env, config, seed=seed)

    return env, fields, overlays, config
