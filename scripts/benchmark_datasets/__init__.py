"""Benchmark utilities for neurospatial animation backends.

This package provides:
- Dataset generators for reproducible benchmarks
- Scripts for measuring baseline performance
- Configuration for small, medium, and large datasets
"""

from .datasets import (
    LARGE_CONFIG,
    MEDIUM_CONFIG,
    SMALL_CONFIG,
    BenchmarkConfig,
    create_benchmark_env,
    create_benchmark_fields,
    create_benchmark_overlays,
)

__all__ = [
    "LARGE_CONFIG",
    "MEDIUM_CONFIG",
    "SMALL_CONFIG",
    "BenchmarkConfig",
    "create_benchmark_env",
    "create_benchmark_fields",
    "create_benchmark_overlays",
]
