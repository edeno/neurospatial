"""Backend selection for encoding computations.

This module provides backend selection infrastructure for encoding functions:

- `get_backend`: Select computation backend (numpy, jax, auto)
- `get_backend_name`: Get the actual backend name used (resolves "auto")
- `is_jax_available`: Check if JAX is available on current platform
- `SUPPORTED_BACKENDS`: Tuple of valid backend names

Backend selection follows these rules:

- ``"numpy"`` (default): Works everywhere, including Windows
- ``"jax"``: Requires JAX installation (Linux/macOS only); raises ImportError
  if JAX is unavailable
- ``"auto"``: Uses JAX if available, falls back to NumPy silently on Windows
  or if JAX is not installed

Examples
--------
>>> from neurospatial.encoding._backend import get_backend, get_backend_name
>>> xp = get_backend("numpy")
>>> xp.array([1, 2, 3])
array([1, 2, 3])

>>> get_backend_name("auto")  # Returns "numpy" or "jax" depending on availability
'numpy'
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any, Literal

import numpy as np

__all__ = [
    "SUPPORTED_BACKENDS",
    "get_backend",
    "get_backend_name",
    "is_jax_available",
    "validate_and_resolve_backend",
]

# Valid backend names
SUPPORTED_BACKENDS: tuple[str, ...] = ("numpy", "jax", "auto")

# Type for backend parameter
BackendType = Literal["numpy", "jax", "auto"]


def is_jax_available() -> bool:
    """Check if JAX is available on the current platform.

    JAX is considered available only if:
    1. The platform is NOT Windows (JAX doesn't officially support Windows)
    2. JAX is installed and can be imported

    Returns
    -------
    bool
        True if JAX is available and can be used, False otherwise.

    Notes
    -----
    This function performs the actual import check each time it's called.
    JAX import can be slow on first call due to initialization.

    Examples
    --------
    >>> from neurospatial.encoding._backend import is_jax_available
    >>> is_jax_available()  # Returns True on Linux/macOS with JAX installed
    False
    """
    # JAX does not officially support Windows
    if sys.platform == "win32":
        return False

    # Check if JAX is installed
    if importlib.util.find_spec("jax") is None:
        return False

    # Try to actually import JAX to verify it works
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


def get_backend(name: BackendType) -> Any:
    """Get the array module for the specified backend.

    Parameters
    ----------
    name : {"numpy", "jax", "auto"}
        Backend name. Case-sensitive (must be lowercase).

        - ``"numpy"``: Returns the numpy module. Always works.
        - ``"jax"``: Returns jax.numpy module. Raises ImportError if JAX
          is unavailable.
        - ``"auto"``: Returns jax.numpy if JAX is available, otherwise numpy.
          Never raises an error.

    Returns
    -------
    ModuleType
        Either ``numpy`` or ``jax.numpy`` module, which can be used for
        array operations.

    Raises
    ------
    ValueError
        If ``name`` is not one of the supported backends.
    ImportError
        If ``name="jax"`` but JAX is not available.

    Examples
    --------
    >>> from neurospatial.encoding._backend import get_backend
    >>> xp = get_backend("numpy")
    >>> arr = xp.array([1.0, 2.0, 3.0])
    >>> arr.sum()
    6.0

    >>> xp = get_backend("auto")  # Uses JAX if available, else NumPy
    >>> xp.zeros(3)
    array([0., 0., 0.])
    """
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {name!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    if name == "numpy":
        return np

    if name == "jax":
        if not is_jax_available():
            platform_note = (
                " JAX is not supported on Windows."
                if sys.platform == "win32"
                else " JAX is not installed. Install with: uv add jax jaxlib"
            )
            raise ImportError(
                f"JAX backend requested but JAX is not available.{platform_note}"
            )
        import jax.numpy as jnp

        return jnp

    # name == "auto"
    if is_jax_available():
        import jax.numpy as jnp

        return jnp
    return np


def get_backend_name(name: BackendType) -> Literal["numpy", "jax"]:
    """Get the actual backend name that will be used.

    Resolves ``"auto"`` to the actual backend that would be selected.

    Parameters
    ----------
    name : {"numpy", "jax", "auto"}
        Backend name.

    Returns
    -------
    {"numpy", "jax"}
        The actual backend name. For ``"auto"``, returns the backend that
        would be selected based on availability.

    Raises
    ------
    ValueError
        If ``name`` is not one of the supported backends.
    ImportError
        If ``name="jax"`` but JAX is not available.

    Examples
    --------
    >>> from neurospatial.encoding._backend import get_backend_name
    >>> get_backend_name("numpy")
    'numpy'

    >>> get_backend_name("auto")  # Resolves to actual backend
    'numpy'
    """
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {name!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    if name == "numpy":
        return "numpy"

    if name == "jax":
        if not is_jax_available():
            platform_note = (
                " JAX is not supported on Windows."
                if sys.platform == "win32"
                else " JAX is not installed. Install with: uv add jax jaxlib"
            )
            raise ImportError(
                f"JAX backend requested but JAX is not available.{platform_note}"
            )
        return "jax"

    # name == "auto"
    if is_jax_available():
        return "jax"
    return "numpy"


def validate_and_resolve_backend(backend: str) -> Literal["numpy", "jax"]:
    """Validate backend parameter and resolve 'auto' to actual backend.

    Convenience function that combines backend validation with resolution.
    Use this at the start of compute functions to validate user input and
    get the resolved backend name.

    Parameters
    ----------
    backend : str
        Backend name. Must be one of ``"numpy"``, ``"jax"``, or ``"auto"``.

    Returns
    -------
    {"numpy", "jax"}
        The resolved backend name. For ``"auto"``, returns the backend that
        would be selected based on availability.

    Raises
    ------
    ValueError
        If ``backend`` is not one of the supported backends.
    ImportError
        If ``backend="jax"`` but JAX is not available.

    Examples
    --------
    >>> from neurospatial.encoding._backend import validate_and_resolve_backend
    >>> validate_and_resolve_backend("numpy")
    'numpy'

    >>> validate_and_resolve_backend("auto")  # Resolves to actual backend
    'numpy'

    >>> validate_and_resolve_backend("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: Unknown backend: 'invalid'. ...
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    return get_backend_name(backend)  # type: ignore[arg-type]
