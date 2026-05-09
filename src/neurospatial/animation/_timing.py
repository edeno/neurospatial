"""Timing instrumentation utilities for animation backends.

This module provides lightweight timing instrumentation for profiling
animation rendering performance across all backends.

Enable timing by setting the environment variable:
    NEUROSPATIAL_TIMING=1 uv run python your_script.py

When enabled, timing records are emitted at ``logger.info`` level via the
``neurospatial.animation._timing`` logger so users can route them to
their preferred sink (stderr, a file, structured logging). For
napari-specific profiling, use NAPARI_PERFMON instead.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Callable, Generator
from typing import ParamSpec, TypeVar

# Check if timing is enabled via environment variable
_TIMING_ENABLED = bool(os.environ.get("NEUROSPATIAL_TIMING"))
_logger = logging.getLogger(__name__)

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
    >>> with timing("render_frame"):  # doctest: +SKIP
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
        _logger.info("[TIMING] %s: %.2f ms", name, elapsed_ms)


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
    >>> @timed  # doctest: +SKIP
    ... def expensive_function():
    ...     # do work
    ...     pass
    # Output (when NEUROSPATIAL_TIMING=1):
    # [TIMING] expensive_function: 123.45 ms
    """
    if not _TIMING_ENABLED:
        return func

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        """
        Execute the wrapped function and log its execution time.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the wrapped function.
        **kwargs : dict
            Keyword arguments to pass to the wrapped function.

        Returns
        -------
        T
            Return value from the wrapped function.
        """
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            _logger.info("[TIMING] %s: %.2f ms", func.__name__, elapsed_ms)

    return wrapper


def is_timing_enabled() -> bool:
    """Check if timing instrumentation is enabled.

    Returns
    -------
    bool
        True if NEUROSPATIAL_TIMING environment variable is set.
    """
    return _TIMING_ENABLED
