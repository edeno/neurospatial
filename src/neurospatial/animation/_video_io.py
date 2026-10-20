"""Compatibility imports for the shared video reader.

The implementation lives in :mod:`neurospatial._video_io` so annotation can
read video without importing animation backends. These aliases preserve
existing reader imports and unpickling through the former module path.
Internal callers and tests that replace the reader should use its shared
module as their patch target.
"""

from neurospatial._video_io import (
    CacheInfo as CacheInfo,
)
from neurospatial._video_io import (
    VideoReaderProtocol as VideoReaderProtocol,
)
from neurospatial._video_io import (
    _VideoReader as _VideoReader,
)
