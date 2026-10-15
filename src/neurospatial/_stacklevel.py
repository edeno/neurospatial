"""Attribute warnings to the scientist's call site rather than library code.

This module has no internal dependencies, so every package tier can use it.
"""

from __future__ import annotations

import sys
from functools import cached_property
from pathlib import Path


def _external_stacklevel() -> int:
    """Stacklevel, counted from the caller, of the first frame outside the package."""
    package_dir = Path(__file__).resolve().parent
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and (
        Path(frame.f_code.co_filename).is_relative_to(package_dir)
        or frame.f_code is cached_property.__get__.__code__
    ):
        frame = frame.f_back
        level += 1
    return level
