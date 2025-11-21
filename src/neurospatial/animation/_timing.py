"""Timing instrumentation utilities for animation backends.

This module provides lightweight timing instrumentation for profiling
animation rendering performance across all backends.

Enable timing by setting the environment variable:
    NEUROSPATIAL_TIMING=1 uv run python your_script.py

The timing output is printed to stderr with function names and elapsed time.
For napari-specific profiling, use NAPARI_PERFMON instead.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Callable, Generator
from typing import ParamSpec, TypeVar

# Check if timing is enabled via environment variable
_TIMING_ENABLED = bool(os.environ.get("NEUROSPATIAL_TIMING"))

P = ParamSpec("P")
T = TypeVar("T")


@contextlib.contextmanager
def timing(name: str) -> Generator[None, None, None]:
    """Context manager for timing code blocks.

    When NEUROSPATIAL_TIMING is set, prints timing information to stderr.
    Otherwise, this is a no-op with minimal overhead.

    Parameters
    ----------
    name : str
        Name to identify this timing block in output.

    Examples
    --------
    >>> with timing("render_frame"):
    ...     result = expensive_operation()
    # Output (when NEUROSPATIAL_TIMING=1):
    # [TIMING] render_frame: 123.45 ms
    """
    if not _TIMING_ENABLED:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[TIMING] {name}: {elapsed_ms:.2f} ms", file=sys.stderr)


def timed(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator for timing function calls.

    When NEUROSPATIAL_TIMING is set, prints timing information to stderr.
    Otherwise, this is a no-op with minimal overhead.

    Parameters
    ----------
    func : Callable
        Function to wrap with timing.

    Returns
    -------
    Callable
        Wrapped function that logs timing when enabled.

    Examples
    --------
    >>> @timed
    ... def expensive_function():
    ...     # do work
    ...     pass
    # Output (when NEUROSPATIAL_TIMING=1):
    # [TIMING] expensive_function: 123.45 ms
    """
    if not _TIMING_ENABLED:
        return func

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"[TIMING] {func.__name__}: {elapsed_ms:.2f} ms", file=sys.stderr)

    return wrapper


def is_timing_enabled() -> bool:
    """Check if timing instrumentation is enabled.

    Returns
    -------
    bool
        True if NEUROSPATIAL_TIMING environment variable is set.
    """
    return _TIMING_ENABLED
