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

>>> get_backend_name("numpy")
'numpy'

The result of ``get_backend_name("auto")`` depends on whether JAX is
installed and supported on the current platform — see ``is_jax_available``.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from typing import Any, get_args

import numpy as np
from numpy.typing import ArrayLike

from neurospatial.ops._types import BackendType, ResolvedBackend

__all__ = [
    "SUPPORTED_BACKENDS",
    "get_backend",
    "get_backend_name",
    "is_jax_available",
    "validate_and_resolve_backend",
]

# Valid backend names
SUPPORTED_BACKENDS: tuple[str, ...] = get_args(BackendType)


@functools.lru_cache(maxsize=1)
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
    The result is cached after the first call so subsequent backend
    resolutions in hot paths (every ``compute_*_rate(s)`` call) avoid
    repeated package-metadata lookups and import attempts.

    Examples
    --------
    >>> from neurospatial.encoding._backend import is_jax_available
    >>> isinstance(is_jax_available(), bool)
    True
    """
    if sys.platform == "win32":
        return False

    if importlib.util.find_spec("jax") is None:
        return False

    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


def _enable_jax_float64(*inputs: Any) -> None:
    """Turn on JAX float64 before the JAX backend makes any array.

    The encoding JAX paths compute rate maps in float64, and their comparisons
    are precision-sensitive (a ``min_occupancy`` threshold compared with
    occupancy in float32 keeps or drops bins differently). Enabling float64
    when the backend is chosen, not when a JAX module is first imported, keeps
    the precision independent of which call ran first.

    Parameters
    ----------
    *inputs : Any
        Arrays the caller is about to use. If float64 is off and one of them
        is being traced, enabling it now would make the compiled function
        expect float32 inputs and fail, so this raises instead.

    Raises
    ------
    RuntimeError
        If JAX float64 is off while ``inputs`` are traced (``jax.jit``,
        ``jax.grad`` or ``jax.vmap``).
    """
    import jax

    if jax.dtypes.canonicalize_dtype(np.float64) == np.float64:
        return
    if any(isinstance(value, jax.core.Tracer) for value in inputs):
        from neurospatial._exceptions import _format_error

        raise RuntimeError(
            _format_error(
                "JAX float64 is off while these inputs are traced by jax.jit, "
                "jax.grad or jax.vmap.",
                why="the JAX backend computes rate maps in float64; turning it on "
                "during tracing leaves the compiled function expecting float32 "
                "inputs.",
                fix='call jax.config.update("jax_enable_x64", True) before tracing.',
            )
        )
    jax.config.update("jax_enable_x64", True)


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
    >>> float(arr.sum())
    6.0

    The ``"auto"`` backend resolves to JAX when available (Linux/macOS with
    JAX installed) and to NumPy otherwise, so its return value depends on
    the runtime environment.
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
        _enable_jax_float64()
        import jax.numpy as jnp

        return jnp

    # name == "auto"
    if is_jax_available():
        _enable_jax_float64()
        import jax.numpy as jnp

        return jnp
    return np


def get_backend_name(name: BackendType) -> ResolvedBackend:
    """Get the actual backend name that will be used.

    Resolves ``"auto"`` to the actual backend that would be selected.
    Choosing JAX turns on JAX float64 (see :func:`_enable_jax_float64`).

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

    >>> get_backend_name("auto") in {"numpy", "jax"}
    True
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
        _enable_jax_float64()
        return "jax"

    # name == "auto"
    if is_jax_available():
        _enable_jax_float64()
        return "jax"
    return "numpy"


def validate_and_resolve_backend(backend: str) -> ResolvedBackend:
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

    >>> validate_and_resolve_backend("auto") in {"numpy", "jax"}
    True

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


def _to_backend(
    x: ArrayLike,
    resolved_backend: ResolvedBackend,
    *,
    dtype: type[np.floating[Any]] | None = None,
) -> Any:
    """Return ``x`` as a JAX array when the resolved backend is ``"jax"``.

    Parameters
    ----------
    x : array_like
        Array to hand back in the backend's array type.
    resolved_backend : {"numpy", "jax"}
        Backend from :func:`validate_and_resolve_backend`, which returns
        ``"jax"`` only when JAX is available.
    dtype : {np.float32, np.float64}, optional
        Dtype of the JAX array; ``None`` keeps the dtype of ``x``. Ignored for
        ``"numpy"``.

    Returns
    -------
    array
        ``jax.numpy.asarray(x, dtype=dtype)`` for ``"jax"``; ``x`` itself,
        unconverted, for ``"numpy"``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._backend import _to_backend
    >>> x = np.ones(3)
    >>> _to_backend(x, "numpy") is x
    True
    """
    if resolved_backend != "jax":
        return x
    import jax.numpy as jnp

    return jnp.asarray(x, dtype=dtype)
