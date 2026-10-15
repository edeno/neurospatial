"""Pure array geometry shared by stored data and native builders."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def get_centers(bin_edges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the center of each bin given its edges.

    Parameters
    ----------
    bin_edges : NDArray[np.float64], shape (n_edges,)
        A 1D array of sorted coordinates representing the edges that define
        a sequence of bins. For `N` bins, there will be `N+1` edges.

    Returns
    -------
    NDArray[np.float64], shape (n_edges - 1,)
        A 1D array containing the center coordinate of each bin.

    """
    return bin_edges[:-1] + np.diff(bin_edges) / 2
