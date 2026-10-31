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
from neurospatial._validation import is_integer_at_least
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
    if mode == "inverse-distance-weighted" and not is_integer_at_least(n_neighbors, 1):
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


def _check_max_outside_fraction(max_outside_fraction: object) -> None:
    """Refuse a ``max_outside_fraction`` that is not a number from 0 to 1."""
    if isinstance(max_outside_fraction, (bool, np.bool_)) or not isinstance(
        max_outside_fraction, (numbers.Real, np.floating, np.integer)
    ):
        raise TypeError(
            "max_outside_fraction must be a number from 0 to 1, got "
            f"{max_outside_fraction!r}."
        )
    if not 0.0 <= float(max_outside_fraction) <= 1.0:
        raise ValueError(
            "max_outside_fraction must be from 0 to 1 (a share of the total "
            f"probability), got {max_outside_fraction!r}."
        )


def _outside_target(
    target_env: ProbabilityTargetConsumer,
    centers: NDArray[np.float64],
    nearest_distance: NDArray[np.float64],
    holds_mass: NDArray[np.bool_],
    limit: float,
) -> NDArray[np.bool_]:
    """Source bins holding mass whose center lies farther than ``limit`` from the target.

    The distance from a center to the target is 0 inside a target bin, by the
    target's own containment rule (``bin_at``), so a coarser or elongated bin
    keeps every point it holds; elsewhere it is the distance to the nearest
    target bin center. Only centers beyond ``limit`` of every center can be
    outside, so only those are looked up.
    """
    outside = holds_mass & (nearest_distance > limit)
    if np.any(outside):
        candidates = np.flatnonzero(outside)
        outside[candidates[target_env.bin_at(centers[candidates]) >= 0]] = False
    return outside


def _require_mass_inside(
    source_probs: NDArray[np.float64],
    nearest_distance: NDArray[np.float64],
    outside: NDArray[np.bool_],
    limit: str,
    max_outside_fraction: float,
) -> None:
    """Refuse when more than ``max_outside_fraction`` of the mass lies ``outside``.

    Parameters
    ----------
    source_probs : NDArray[np.float64], shape (n_source_bins,)
    nearest_distance : NDArray[np.float64], shape (n_source_bins,)
        Distance from each transformed source bin center to the nearest
        target bin center.
    outside : NDArray[np.bool_], shape (n_source_bins,)
        Source bins holding mass whose center lies outside the target.
    limit : str
        The distance limit, completing "farther than ...".
    max_outside_fraction : float
        Largest share of the total probability that may lie outside.
    """
    if not np.any(outside):
        return
    total = float(source_probs.sum())
    stranded = float(source_probs[outside].sum())
    if stranded <= max_outside_fraction * total:
        return
    raise ValueError(
        _format_error(
            f"{100 * stranded / total:.3g}% of the source probability "
            f"({int(outside.sum())} source bins) lies outside the target, "
            f"farther than {limit} from every target bin (the farthest "
            f"{float(nearest_distance[outside].max()):.4g} from a bin center); "
            f"max_outside_fraction={max_outside_fraction:g} allows "
            f"{100 * max_outside_fraction:.3g}%.",
            why="nearest-bin mapping would pile that mass onto the target's "
            "edge bins, which it never belonged to",
            fix="check source_transform: it must map source coordinates into "
            "the target's, as estimate_transform(source_landmarks, "
            "target_landmarks) does, and rotate_2d(angle) turns about the "
            "origin unless given center=. If the mismatch is expected, raise "
            "max_outside_fraction to accept that share on the edge bins, or "
            "pass a larger max_distance (math.inf gives every source bin its "
            "nearest target bin however far away)",
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
    max_outside_fraction: float = 1e-6,
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
        Farthest a transformed source bin center may lie from the target, in
        the target's length units: the distance is 0 inside a target bin
        (``bin_at``), so coarser or elongated target bins of the same arena
        receive all their mass, and otherwise the distance to the nearest
        target bin center. None uses ``target_env.bin_spacing``, so a small
        unvisited hole in the target does not refuse the map. ``math.inf``
        sends every source bin to its nearest target bin however far away.
    max_outside_fraction : float, default=1e-6
        Largest share of the total probability that may lie farther than
        ``max_distance`` from the target; it still goes to its nearest target
        bin. The default accepts a decoded posterior's far tail, which is
        positive on every bin and so always reaches past a smaller target,
        but adds no visible amount to the edge bins that receive it, while a
        misaligned map or a spread distribution such as occupancy strands far
        more. Raise it to accept a known mismatch; 0 refuses any mass
        outside.

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
        unknown; if `n_neighbors`, `eps`, `max_distance` or
        `max_outside_fraction` is invalid; or if more than
        `max_outside_fraction` of the probability lies outside the target
        (the message reports how much).
    TypeError
        If `max_distance` is not a number or None, `max_outside_fraction` is
        not a number, or `source_transform` is not an AffineND or None.
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
    ValueError: ...% of the source probability (... source bins) lies outside the target, farther than one target bin spacing (5) from every target bin ...

    """
    source_probs = _validated_probabilities(
        source_env, target_env, source_probs, mode, n_neighbors, eps
    )
    _check_source_transform(source_transform, source_env.n_dims)
    _check_max_distance(max_distance)
    _check_max_outside_fraction(max_outside_fraction)
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
        limit = target_env.bin_spacing
        described = f"one target bin spacing ({limit:.4g})"
    else:
        limit = float(max_distance)
        described = f"max_distance={limit:.4g}"
    outside = _outside_target(
        target_env, src_centers, nearest_distance, source_probs > 0, limit
    )
    _require_mass_inside(
        source_probs,
        nearest_distance,
        outside,
        described,
        float(max_outside_fraction),
    )

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
