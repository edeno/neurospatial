"""Analysis choices the dataset loaders share."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["alternating_blocks"]


def alternating_blocks(
    start: float, stop: float, length: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split [start, stop) into alternating train and test blocks of ``length`` s.

    Decoding trained on one set and tested on the other is always tested on
    held-out time, with both sets spread over the whole session. The last
    block ends at ``stop`` and is shorter than ``length`` unless the span is
    a whole number of blocks; an edge within a millionth of a block of
    ``stop`` (float rounding) is not a block of its own.

    Parameters
    ----------
    start, stop : float
        The span in seconds, ``start < stop``.
    length : float
        Block length in seconds.

    Returns
    -------
    train, test : ndarray, shape (n, 2)
        Even-numbered and odd-numbered blocks as [start, stop) rows.

    Examples
    --------
    >>> train, test = alternating_blocks(0.0, 150.0, 60.0)
    >>> train.tolist(), test.tolist()
    ([[0.0, 60.0], [120.0, 150.0]], [[60.0, 120.0]])
    """
    n_blocks = max(int(np.ceil((stop - start) / length - 1e-6)), 1)
    edges = np.append(start + length * np.arange(n_blocks), stop)
    blocks = np.column_stack([edges[:-1], edges[1:]])
    return blocks[::2], blocks[1::2]
