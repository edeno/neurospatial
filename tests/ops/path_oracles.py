"""Independent answers for path lengths near walls.

Nothing here calls the path engine (``neurospatial.ops._paths``) or the
distance functions built on it. A segment is clear when Shapely finds no
point of it on a wall of the environment's arena, and shortest paths are
SciPy's Dijkstra search over a visibility graph built here.

The any-angle engine turns only at bin centers, so its paths between bin
centers are paths of :func:`through_centers`'s graph, and no shorter than
that graph's shortest path. The engine grows its paths greedily (Theta*),
so they can be longer. Measured over every pair of bins of walled square
and rectangular grids (the layouts of ``tests/ops/test_distance.py``):

- pair distances (the shorter of the paths grown from either end) were at
  most 3.3% longer than the shortest path through bin centers, and equal to
  it for 78-84% of pairs (:data:`PAIR_EXCESS`);
- field lengths (the path grown from the source only) were at most 5.6%
  longer (:data:`FIELD_EXCESS`);
- bin-graph distances were up to 8.2% longer on an 8-connected grid and
  equal for 8-12% of pairs, so a bound below 8.2% tells the two apart.

The wall ends of an oracle fixture must lie on bin edges: a wall that ends
part-way along the face between two bins leaves that face open.
"""

import numpy as np
import shapely
from scipy.sparse.csgraph import shortest_path

# Most a pair distance exceeded the shortest path through bin centers.
PAIR_EXCESS = 0.033
# Most a one-direction field length exceeded it.
FIELD_EXCESS = 0.056


def walls_of(env):
    """The walls of the environment's arena as one Shapely geometry."""
    return shapely.union_all(list(env.arena.walls))


def clear_of_walls(env, starts, ends):
    """Whether each straight segment has no point on a wall of the arena.

    Parameters
    ----------
    env : Environment
        A two-dimensional environment with declared walls.
    starts, ends : ndarray, shape (n, 2)

    Returns
    -------
    ndarray of bool, shape (n,)
    """
    segments = shapely.linestrings(np.stack([starts, ends], axis=1))
    return ~shapely.intersects(segments, walls_of(env))


def through_centers(env, floor=None):
    """Shortest path length between every two bin centers, turning at centers.

    The visibility graph joins two bin centers when the straight segment
    between them is clear of the arena's walls (the room itself must then
    be convex, so a clear segment stays inside it) or, with ``floor``, when
    the segment lies in that polygon, its boundary included.

    Parameters
    ----------
    env : Environment
        A two-dimensional environment: a convex room with declared walls,
        or the bins that tile ``floor``.
    floor : shapely.Polygon, optional
        The area the bins tile, for a room whose obstacles are missing bins.

    Returns
    -------
    ndarray, shape (n_bins, n_bins)
    """
    centers = np.asarray(env.bin_centers, dtype=float)
    first, second = np.triu_indices(len(centers), k=1)
    if floor is None:
        clear = clear_of_walls(env, centers[first], centers[second])
    else:
        segments = shapely.linestrings(
            np.stack([centers[first], centers[second]], axis=1)
        )
        clear = shapely.covers(floor, segments)
    lengths = np.zeros((len(centers), len(centers)))  # 0 marks no edge
    lengths[first[clear], second[clear]] = np.linalg.norm(
        centers[first[clear]] - centers[second[clear]], axis=1
    )
    return shortest_path(lengths, method="D", directed=False)
