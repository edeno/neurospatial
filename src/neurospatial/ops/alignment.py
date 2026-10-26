"""Alignment and probability mapping between spatial environments.

Moves probability distributions between the bins of different `Environment`
instances (2D, 3D or higher), for example a posterior decoded in one session
onto another session's bins.

`map_probabilities` gives each source bin's probability to the nearest target
bin, summing the mass that lands in the same target bin, and refuses mass that
lands farther than a target bin spacing from every target bin. An `AffineND`
(from `estimate_transform` or the factories in `neurospatial.ops.transforms`)
first moves the source bin centers into the target's coordinates.

Use it for posteriors and occupancy probabilities from experiments where the
arena moved, turned or was rescaled, or that were binned differently. Mass is
summed, so do not use it for rate maps or other per-bin values; move those
with :func:`~neurospatial.ops.resample_field`.
"""

from __future__ import annotations

import numbers
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._constants import IDW_MIN_DISTANCE
from neurospatial._exceptions import _format_error
from neurospatial._validation import is_positive_integer
from neurospatial.geometry import require_cartesian
from neurospatial.ops.transforms import AffineND

if TYPE_CHECKING:
    from scipy.spatial import cKDTree

    from neurospatial.ops._protocols import (
        BinsConsumer,
        CartesianBinsConsumer,
        ProbabilityTargetConsumer,
    )

__all__ = ["map_probabilities"]


def _validated_probabilities(
    source_env: BinsConsumer,
    target_env: BinsConsumer,
    source_probs: NDArray[np.float64],
    mode: str,
    n_neighbors: object,
    eps: float,
) -> NDArray[np.float64]:
    """Check the arguments of ``map_probabilities`` and return the probabilities.

    Returns
    -------
    NDArray[np.float64], shape (source_env.n_bins,)
        ``source_probs`` as a float array.
    """
    if source_env.n_dims != target_env.n_dims:
        raise ValueError(
            f"Source and target environments must have the same number of dimensions: "
            f"{source_env.n_dims} != {target_env.n_dims}."
        )

    probs = np.asarray(source_probs, dtype=np.float64)
    n_src = source_env.n_bins
    if probs.ndim != 1 or probs.shape[0] != n_src:
        raise ValueError(
            f"source_probs must be a 1D array of length {n_src}, got shape "
            f"{probs.shape}."
        )
    bad = np.flatnonzero(~np.isfinite(probs))
    if bad.size:
        raise ValueError(
            _format_error(
                f"source_probs has NaN or infinite values in {bad.size} bin(s), "
                f"first at bin {bad[0]}.",
                why="mapped probability is summed into target bins, so one NaN "
                "would make a target bin, and the total, NaN",
                fix="set bins without an estimate to 0 if they carry no "
                "probability, or restrict the map to bins that have one",
            )
        )
    if np.any(probs < 0):
        raise ValueError("source_probs must be nonnegative.")

    if mode not in ("nearest", "inverse-distance-weighted"):
        raise ValueError(
            f"Unrecognized mode '{mode}'. "
            "Supported: 'nearest' or 'inverse-distance-weighted'."
        )
    if mode == "inverse-distance-weighted" and not is_positive_integer(n_neighbors):
        raise ValueError(
            f"In 'inverse-distance-weighted' mode, n_neighbors must be "
            f"an integer >= 1 (got n_neighbors={n_neighbors!r})."
        )
    if not (np.isfinite(eps) and eps > 0):
        raise ValueError(
            _format_error(
                f"eps must be a positive, finite distance, got {eps!r}.",
                why="inverse-distance weights are 1 / (distance + eps); eps = 0 "
                "divides by zero for a source bin on a target center and a "
                "negative eps gives wrong or negative weights",
                fix="leave eps at its default, a distance far below any bin size",
            )
        )
    return probs


def _map_inverse_distance_weighted(
    tree: cKDTree,
    src_centers: NDArray[np.float64],
    source_probs: NDArray[np.float64],
    n_tgt: int,
    n_neighbors: int,
    eps: float,
) -> NDArray[np.float64]:
    """Map probabilities using inverse distance weighting.

    Parameters
    ----------
    tree : cKDTree
        KDTree built on target bin centers.
    src_centers : NDArray[np.float64]
        Transformed source bin centers.
    source_probs : NDArray[np.float64]
        Source probabilities.
    n_tgt : int
        Number of target bins.
    n_neighbors : int
        Number of neighbors to use for weighting.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    NDArray[np.float64]
        Target probabilities.

    """
    target_probs = np.zeros(n_tgt, dtype=float)

    # Clamp n_neighbors if larger than number of target bins
    k_eff = min(n_neighbors, n_tgt)

    dists, idxs = tree.query(src_centers, k=k_eff)

    # If k_eff == 1, we may get 1D arrays; force them to 2D
    if k_eff == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    # Compute inverse-distance weights and normalize
    weights = 1.0 / (dists + eps)
    sums = np.sum(weights, axis=1, keepdims=True)
    normed = weights / sums

    # Compute contributions
    src_col = source_probs.reshape(-1, 1)
    contribs = src_col * normed

    # Accumulate contributions
    flat_idxs = idxs.ravel()
    flat_contribs = contribs.ravel()
    np.add.at(target_probs, flat_idxs, flat_contribs)

    return target_probs


def _check_source_transform(source_transform: object, n_dims: int) -> None:
    """Refuse a ``source_transform`` that is not an ``n_dims`` AffineND or None."""
    if source_transform is None:
        return
    if not isinstance(source_transform, AffineND):
        raise TypeError(
            _format_error(
                "source_transform must be an AffineND or None, got "
                f"{type(source_transform).__name__}.",
                why="the source bin centers are mapped into the target's "
                "coordinates by one affine transform",
                fix="fit one with estimate_transform(source_landmarks, "
                "target_landmarks), or build one with rotate_2d, scale_2d, "
                "translate or from_rotation_matrix and compose them with @",
            )
        )
    if source_transform.n_dims != n_dims:
        raise ValueError(
            f"source_transform is {source_transform.n_dims}-D but the "
            f"environments are {n_dims}-D."
        )


def _check_max_distance(max_distance: object) -> None:
    """Refuse a ``max_distance`` that is not None or a real number >= 0."""
    if max_distance is None:
        return
    if isinstance(max_distance, (bool, np.bool_)) or not isinstance(
        max_distance, (numbers.Real, np.floating, np.integer)
    ):
        raise TypeError(
            f"max_distance must be a distance or None, got {max_distance!r}."
        )
    if np.isnan(float(max_distance)) or float(max_distance) < 0:
        raise ValueError(f"max_distance must be zero or more, got {max_distance!r}.")


# Share of the probability that may lie beyond the limit and still be mapped to
# its nearest target bin. Decoded posteriors are positive on every bin, so their
# far tail always reaches past a smaller target; at 1e-6 of the total it adds
# nothing visible to the edge bins that receive it.
_NEGLIGIBLE_STRANDED_SHARE = 1e-6


def _require_mass_within(
    source_probs: NDArray[np.float64],
    nearest_distance: NDArray[np.float64],
    beyond: NDArray[np.bool_],
    limit: str,
) -> None:
    """Refuse to move more than a negligible share of mass ``beyond`` the limit.

    A share up to ``_NEGLIGIBLE_STRANDED_SHARE`` of the total goes to its
    nearest target bin, like the rest.

    Parameters
    ----------
    source_probs : NDArray[np.float64], shape (n_source_bins,)
    nearest_distance : NDArray[np.float64], shape (n_source_bins,)
        Distance from each transformed source bin center to the nearest
        target bin center.
    beyond : NDArray[np.bool_], shape (n_source_bins,)
        Source bins whose transformed center no target bin may receive.
    limit : str
        The rule ``beyond`` applied, completing "lies ...".
    """
    beyond = beyond & (source_probs > 0)
    if not np.any(beyond):
        return
    total = float(source_probs.sum())
    stranded = float(source_probs[beyond].sum())
    if stranded <= _NEGLIGIBLE_STRANDED_SHARE * total:
        return
    raise ValueError(
        _format_error(
            f"{100 * stranded / total:.3g}% of the source probability "
            f"({int(beyond.sum())} source bins) lies {limit}; the farthest is "
            f"{float(nearest_distance[beyond].max()):.4g} from a target bin "
            "center.",
            why="nearest-bin mapping would pile that mass onto the target's "
            "edge bins, which it never belonged to",
            fix="check source_transform: it must map source coordinates into "
            "the target's, as estimate_transform(source_landmarks, "
            "target_landmarks) does, and rotate_2d(angle) turns about the "
            "origin unless given center=. If the mismatch is expected, pass "
            "a larger max_distance, or "
            "max_distance=math.inf to give every source bin its nearest "
            "target bin however far away",
        )
    )


def map_probabilities(
    source_env: CartesianBinsConsumer,
    target_env: ProbabilityTargetConsumer,
    source_probs: NDArray[np.float64],
    *,
    mode: Literal["nearest", "inverse-distance-weighted"] = "nearest",
    n_neighbors: int = 1,
    eps: float = IDW_MIN_DISTANCE,
    source_transform: AffineND | None = None,
    max_distance: float | None = None,
) -> NDArray[np.float64]:
    """Map probabilities on source_env onto target_env.

    The source bin centers can first be moved into the target's coordinates
    by ``source_transform``; each source bin's mass then moves to the nearest
    target bin (or is split among its nearest target bins).

    Parameters
    ----------
    source_env : Environment
        An Environment whose bins currently have centers `source_env.bin_centers`.
    target_env : Environment
        An Environment onto whose bins we want to map probabilities.
    source_probs : NDArray[np.float64]
        1D array of length source_env.n_bins; finite and nonnegative.
    mode : {'nearest', 'inverse-distance-weighted'}
        - "nearest": each source bin's mass → its single nearest target bin.
        - "inverse-distance-weighted" : spread each source bin's mass over
            its `n_neighbors` nearest target bins,
            weighted by inverse distance, and summing contributions.
    n_neighbors : int
        Number of neighbors to use when mode="inverse-distance-weighted"
        (ignored if mode="nearest"). Any integer type, at least 1.
    eps : float, default=IDW_MIN_DISTANCE
        Small positive distance added before inverting distances in IDW
        weights, so a source bin on a target center does not divide by zero.
    source_transform : AffineND or None, default=None
        Map from source coordinates to target coordinates, applied to the
        source bin centers, with as many dimensions as the environments. Fit
        it from landmark pairs with ``estimate_transform(source_landmarks,
        target_landmarks)``, or compose factories such as
        ``rotate_2d(angle, center=c)``. None uses the source centers as they
        are.
    max_distance : float or None, default=None
        Farthest a transformed source bin center may be from every target bin
        center while it holds probability, in the target's length units.
        None accepts a center that lies inside a target bin (``bin_at``), so
        coarser or elongated target bins of the same arena receive all their
        mass, or that lies within the target's typical bin spacing (the
        median distance between neighboring bin centers) of a target center,
        so a small unvisited hole in the target does not refuse the map.
        ``math.inf`` sends every source bin to its nearest target bin however
        far away. A share of at most 1e-6 of the probability beyond the limit,
        such as a posterior's far tail, is mapped to its nearest bin rather
        than refused.

    Returns
    -------
    target_probabilities : array, shape (n_target_bins,)
        Each entry is the sum of `source_probs` whose (transformed)
        source bin center is nearest to that target bin center. If `n_target_bins == 0`,
        returns an empty array of shape (0,).

    Raises
    ------
    ValueError
        If `source_probs` has incorrect shape or NaN, infinite or negative
        values; if dims mismatch, `source_transform` included; if `mode` is
        unknown; if `n_neighbors`,
        `eps` or `max_distance` is invalid; or if more than 1e-6 of the
        probability lies beyond `max_distance` (by default, outside every
        target bin and farther than one bin spacing from its centers; the
        message reports how much).
    TypeError
        If `max_distance` is not a number or None, or `source_transform` is
        not an AffineND or None.
    CartesianOnlyError
        If either environment is polar: bins are matched by nearest Cartesian
        bin center after a similarity transform.

    See Also
    --------
    resample_field : Move a rate map or other intensive field onto other bins.
    estimate_transform : Fit the alignment from landmark pairs.

    Notes
    -----
    Probability mass is summed, so the total is kept but a value per bin is
    not: mapping a 1 Hz rate map onto coarser bins gives several Hz per bin.
    Use this for posteriors and occupancy probabilities, and
    :func:`~neurospatial.ops.resample_field` for rate maps.

    A rotation or scaling factory acts about the coordinate origin unless
    told otherwise: ``rotate_2d(angle, center=c)`` turns an arena about its
    center ``c``.

    Examples
    --------
    A second session recorded the same square arena turned 30 degrees about
    its center. Map a decoded posterior from session 2 back onto session 1's
    bins:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import map_probabilities, rotate_2d
    >>> rng = np.random.default_rng(0)
    >>> session1 = rng.uniform(0, 100, (5000, 2))
    >>> turn = rotate_2d(np.pi / 6, center=(50.0, 50.0))
    >>> session2 = turn(session1)
    >>> target_env = Environment.from_samples(session1, bin_size=5.0)
    >>> source_env = Environment.from_samples(session2, bin_size=5.0)
    >>> # A posterior peaked where session 2 puts session 1's point (20, 70)
    >>> peak = turn(np.array([[20.0, 70.0]]))
    >>> distance = np.linalg.norm(source_env.bin_centers - peak, axis=1)
    >>> posterior = np.exp(-0.5 * (distance / 5.0) ** 2)
    >>> posterior /= posterior.sum()
    >>> # Undo the turn about the arena center, not about the origin
    >>> mapped = map_probabilities(
    ...     source_env, target_env, posterior, source_transform=turn.inverse()
    ... )
    >>> bool(np.isclose(mapped.sum(), 1.0))
    True
    >>> bool(np.linalg.norm(target_env.bin_centers[mapped.argmax()] - [20, 70]) < 5)
    True

    Rotating about the origin instead moves part of the arena off the target
    bins. Probability spread over the whole arena, such as occupancy, would
    then pile onto the target's edge bins, so it is refused:

    >>> occupancy = np.full(source_env.n_bins, 1.0 / source_env.n_bins)
    >>> map_probabilities(
    ...     source_env, target_env, occupancy, source_transform=rotate_2d(-np.pi / 6)
    ... )  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: ...% of the source probability ... lies outside every target bin and farther than one target bin spacing (5) from every target bin center; ...

    """
    source_probs = _validated_probabilities(
        source_env, target_env, source_probs, mode, n_neighbors, eps
    )
    _check_source_transform(source_transform, source_env.n_dims)
    _check_max_distance(max_distance)
    for argument, env in (("source_env", source_env), ("target_env", target_env)):
        require_cartesian(
            env.coordinate_system,
            f"map_probabilities({argument}=...)",
            why=(
                "It matches each target bin to the nearest Cartesian source bin center."
            ),
            fix=(
                "map probabilities between allocentric environments; decode polar "
                "maps on one shared Environment.from_polar_egocentric(...) "
                "environment"
            ),
        )
    n_src = source_env.n_bins
    n_tgt = target_env.n_bins

    # Handle empty environments
    if n_src == 0 or n_tgt == 0:
        warnings.warn(
            "One of the environments has zero bins; returning zeros.",
            category=UserWarning,
            stacklevel=2,
        )
        return np.zeros(n_tgt, dtype=float)

    src_centers = (
        source_env.bin_centers
        if source_transform is None
        else source_transform(source_env.bin_centers)
    )

    # The target environment's cached KD-tree over its bin centers.
    tree = target_env.bin_center_tree()

    # cKDTree.query does not raise on "no neighbor" (it returns inf distance /
    # out-of-range index). A raised exception here means genuinely bad input
    # (e.g. non-finite coordinates) and must propagate rather than degrade to a
    # silent all-zero map.
    nearest_distance, nearest = tree.query(src_centers, k=1)
    if max_distance is None:
        spacing = target_env.bin_spacing
        beyond = nearest_distance > spacing
        # A center inside a target bin belongs to it however elongated the
        # bin; only the rest needs the target's containment test.
        check = beyond & (source_probs > 0)
        if np.any(check):
            beyond[check] = target_env.bin_at(src_centers[check]) < 0
        limit = (
            "outside every target bin and farther than one target bin "
            f"spacing ({spacing:.4g}) from every target bin center"
        )
    else:
        beyond = nearest_distance > float(max_distance)
        limit = (
            f"farther than max_distance={float(max_distance):.4g} from every "
            "target bin center"
        )
    _require_mass_within(source_probs, nearest_distance, beyond, limit)

    # Perform the requested mapping
    if mode == "nearest":
        target_probs = np.zeros(n_tgt, dtype=float)
        np.add.at(target_probs, nearest, source_probs)
        return target_probs
    return _map_inverse_distance_weighted(
        tree,
        src_centers,
        source_probs,
        n_tgt,
        n_neighbors,
        eps,
    )
