"""Parallel video export backend (stub for testing)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is installed and accessible.

    Returns
    -------
    available : bool
        True if ffmpeg is available
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def render_video(
    env: Environment,
    fields: list,
    save_path: str,
    **kwargs: Any,
) -> Path:
    """Export animation as video using parallel frame rendering.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure
    fields : list
        List of field arrays to animate
    save_path : str
        Output path for video file
    **kwargs : dict
        Additional rendering parameters

    Returns
    -------
    save_path : Path
        Path to exported video file

    Raises
    ------
    RuntimeError
        If ffmpeg is not installed
    """
    # TODO: Full implementation in Milestone 3
    raise NotImplementedError("Video backend not yet implemented")
