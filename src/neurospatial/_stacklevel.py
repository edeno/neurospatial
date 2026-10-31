"""Attribute warnings to the scientist's call site rather than library code.

This module has no internal dependencies, so every package tier can use it.
"""

from __future__ import annotations

import contextlib
import os
import sys
import warnings
from collections.abc import Iterator
from functools import cached_property

# Not resolved: frame filenames come from the same sys.path entry as
# __file__, so resolving only this side would never match when the package
# is imported through a symlink. A string, not a Path, keeps the separators
# as they appear in frame filenames.
_PACKAGE_DIR = os.path.dirname(__file__)  # noqa: PTH120
# A file of the package starts with its directory and a separator; compared
# as strings because frames are walked for every warning.
_PACKAGE_PREFIXES = tuple({_PACKAGE_DIR + "/", _PACKAGE_DIR + os.sep})


def _external_stacklevel(skip: int = 0) -> int:
    """Stacklevel, counted from the caller, of the first frame outside the package.

    ``skip`` frames above the caller are passed over before the search, for a
    caller reached through a frame outside the package, such as
    ``contextlib``'s ``__exit__`` resuming a context-manager generator.
    """
    frame = sys._getframe(1 + skip)
    level = 1 + skip
    while frame.f_back is not None and (
        frame.f_code.co_filename.startswith(_PACKAGE_PREFIXES)
        or frame.f_code is cached_property.__get__.__code__
    ):
        frame = frame.f_back
        level += 1
    return level


@contextlib.contextmanager
def _warnings_with_context(context: str) -> Iterator[None]:
    """Re-emit the warnings raised in the block with ``context`` in front.

    The warnings are attributed to the caller of the function that opened
    the block. They are re-emitted even when the block raises, since a
    warning often explains the error that follows it.
    """
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
    finally:
        for caught_warning in caught:
            # Frames: this generator, contextlib's __exit__, then the caller.
            warnings.warn(
                f"{context}: {caught_warning.message}",
                caught_warning.category,
                stacklevel=_external_stacklevel(skip=2),
            )
