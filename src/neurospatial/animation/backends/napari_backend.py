"""Napari GPU-accelerated viewer backend (stub for testing)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

# Check napari availability
try:
    import napari  # noqa: F401

    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False


def render_napari(
    env: Environment,
    fields: list,
    **kwargs: Any,
) -> Any:
    """Launch Napari viewer with lazy-loaded field animation.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure
    fields : list
        List of field arrays to animate
    **kwargs : dict
        Additional rendering parameters

    Returns
    -------
    viewer : napari.Viewer
        Napari viewer instance (blocking - will show window)

    Raises
    ------
    ImportError
        If napari is not installed
    """
    if not NAPARI_AVAILABLE:
        raise ImportError(
            "Napari backend requires napari. Install with:\n"
            "  pip install napari[all]\n"
            "or\n"
            "  uv add napari[all]"
        )

    # TODO: Full implementation in Milestone 4
    raise NotImplementedError("Napari backend not yet implemented")
