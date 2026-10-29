"""Estimate functions for Bayesian decoding results.

This module provides standalone functions for computing derived quantities
from posterior distributions. These functions mirror the corresponding
properties on DecodingResult but operate on raw arrays, enabling use
without creating a DecodingResult container.

Functions
---------
map_bin : MAP bin index (highest posterior density)
map_position : MAP position (highest posterior density) in environment coordinates
mean_position : Posterior mean position (expected value)
posterior_entropy : Posterior entropy in bits (entropy measure)
credible_region : Highest-density credible region

Notes
-----
Function names mirror DecodingResult property names for consistency.
Users can use either the result properties or these standalone functions
interchangeably.

Examples
--------
>>> from neurospatial.decoding.estimates import posterior_entropy
>>> import numpy as np
>>>
>>> # Uniform posterior (maximum entropy)
>>> posterior = np.ones((10, 100)) / 100
>>> ent = posterior_entropy(posterior)
>>> print(f"Entropy: {ent[0]:.2f} bits")  # log2(100) = 6.64 bits
Entropy: 6.64 bits
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment


def _is_polar(env: Environment) -> bool:
    return env.coordinate_kind == "polar"


def _posterior_mean_basis(env: Environment) -> NDArray[np.float64]:
    """Bin coordinates that may be averaged, shape ``(n_bins, n_dims)``.

    Cartesian bin centers are returned unchanged. Polar ``(distance, angle)``
    centers are mapped to the egocentric plane, ``(distance * cos(angle),
    distance * sin(angle))`` with x ahead and y to the left.
    """
    centers = np.asarray(env.bin_centers, dtype=np.float64)
    if not _is_polar(env):
        return centers
    distance, angle = centers[:, 0], centers[:, 1]
    return np.column_stack([distance * np.cos(angle), distance * np.sin(angle)])


def _finish_posterior_mean(
    env: Environment, weighted: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Report ``posterior @ _posterior_mean_basis(env)`` in environment coordinates.

    Polar centroids return as ``(distance, angle)`` with the angle in the
    environment's angular range. A centroid within the rounding error of the
    animal's position, shape ``(n_time, 2)`` input, has no direction: NaN angle.
    """
    if not _is_polar(env):
        return weighted
    x, y = weighted[:, 0], weighted[:, 1]
    distance = np.hypot(x, y)
    ranges = env.dimension_ranges
    assert ranges is not None  # Polar environments always record their ranges.
    low = float(ranges[1][0])
    angle = low + np.mod(np.arctan2(y, x) - low, 2 * np.pi)
    # Each centroid sums n_bins terms bounded by the largest bin distance.
    max_distance = float(np.max(np.abs(env.bin_centers[:, 0])))
    rounding = env.n_bins * np.finfo(np.float64).eps * max_distance
    angle = np.where(distance <= rounding, np.nan, angle)
    return np.column_stack([distance, angle])


_UNDECODABLE_BIN = -1
"""MAP bin of a posterior row with no finite entry, as ``env.bin_at`` outside."""


def _tie_tolerance(posterior: NDArray[np.floating]) -> float:
    """Relative density difference counted as a tie: ``sqrt(eps)`` of the dtype.

    Rounding in a normalized posterior separates densities that are equal in
    exact arithmetic by far less than this, so such bins count as tied.
    """
    return float(np.sqrt(np.finfo(np.result_type(posterior, np.float32)).eps))


# Rows per block in _density_mode, sized so temporaries stay near 2**22 elements.
_BLOCK_ELEMENTS = 2**22


def _density_mode(
    posterior: NDArray[np.float64], measures: NDArray[np.float64] | None
) -> NDArray[np.int64]:
    """Bin of highest posterior density per row, shape ``(n_time_bins,)``.

    ``measures`` are bin measures relative to the largest bin (``None`` for
    equal bins). The density ``posterior / measures`` is proportional to
    probability per unit length or area, so a large bin does not win only by
    holding more space. Densities within :func:`_tie_tolerance` of the row's
    largest count as tied and the lowest bin index wins. NaN and Inf bins are
    skipped; a row with no finite entry gets ``_UNDECODABLE_BIN``.
    """
    posterior = np.asarray(posterior)
    rtol = _tie_tolerance(posterior)
    result = np.empty(posterior.shape[0], dtype=np.int64)
    step = max(1, _BLOCK_ELEMENTS // max(posterior.shape[1], 1))
    for start in range(0, posterior.shape[0], step):
        block = posterior[start : start + step]
        if measures is not None:
            block = block / measures.astype(block.dtype, copy=False)
        finite = np.isfinite(block)
        density = np.where(finite, block, -np.inf)
        peak = density.max(axis=1, keepdims=True, initial=-np.inf)
        modes = np.argmax(density >= peak - rtol * np.abs(peak), axis=1)
        modes[~finite.any(axis=1)] = _UNDECODABLE_BIN
        result[start : start + step] = modes
    return result


def _bin_positions(env: Environment, bins: NDArray[np.int64]) -> NDArray[np.float64]:
    """Centers of ``bins``, shape ``(n, n_dims)``, NaN for ``_UNDECODABLE_BIN``."""
    positions = np.full((bins.shape[0], env.n_dims), np.nan)
    decoded = bins != _UNDECODABLE_BIN
    positions[decoded] = env.bin_centers[bins[decoded]]
    return positions


def map_bin(
    env: Environment,
    posterior: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Maximum a posteriori bin index for each time bin.

    Returns the index of the bin with highest posterior density at each
    time step: the bin whose probability per unit length, area or volume,
    ``posterior / env.bin_sizes``, is largest.

    Parameters
    ----------
    env : Environment
        Spatial environment providing bin sizes.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over positions.
        Each row should sum to 1.0.

    Returns
    -------
    NDArray[np.int64], shape (n_time_bins,)
        MAP bin indices; -1 (as :meth:`Environment.bin_at` reports a point
        outside the environment) for an undecodable row with no finite entry.

    Notes
    -----
    The posterior holds a probability mass per bin. A bin twice as long
    holds twice the mass at the same density, so comparing mass would favor
    large bins; on equal bins the two orderings agree. Densities that agree
    to within ``sqrt(eps)`` of the posterior dtype, relative, count as tied,
    and ties go to the lowest bin index. NaN and Inf bins are skipped.

    This function mirrors ``DecodingResult.map_bin``.

    See Also
    --------
    map_position : MAP position in environment coordinates
    DecodingResult.map_bin : Property version on result container

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.estimates import map_bin
    >>>
    >>> positions = np.column_stack([np.linspace(0, 10, 50), np.linspace(0, 10, 50)])
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> posterior = np.zeros((1, env.n_bins))
    >>> posterior[0, 3] = 1.0
    >>> map_bin(env, posterior).tolist()
    [3]
    """
    from neurospatial.decoding._support import relative_bin_measures

    measures = relative_bin_measures(env, context="map_bin")
    return _density_mode(posterior, measures)


def map_position(
    env: Environment,
    posterior: NDArray[np.float64],
) -> NDArray[np.float64]:
    """MAP position in environment coordinates.

    Returns the center of the bin with highest posterior density at each
    time step: the bin whose probability per unit length, area or volume,
    ``posterior / env.bin_sizes``, is largest.

    Parameters
    ----------
    env : Environment
        Spatial environment providing bin centers and bin sizes.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over positions.
        Each row should sum to 1.0.

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins, n_dims)
        MAP positions in environment coordinates; NaN for an undecodable row
        with no finite entry.

    Notes
    -----
    The posterior holds a probability mass per bin. Its mode over space is
    the bin of highest density, mass divided by bin size; on equal bins this
    is the highest-mass bin. Ties, within rounding, go to the lowest bin
    index (see :func:`map_bin`).

    This function mirrors ``DecodingResult.map_position``.

    See Also
    --------
    map_bin : MAP bin index
    mean_position : Posterior mean position
    DecodingResult.map_position : Property version on result container

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.estimates import map_position
    >>>
    >>> positions = np.column_stack([np.linspace(0, 10, 50), np.linspace(0, 10, 50)])
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Delta posterior at bin 0
    >>> posterior = np.zeros((1, env.n_bins))
    >>> posterior[0, 0] = 1.0
    >>> pos = map_position(env, posterior)
    >>> pos.shape
    (1, 2)
    """
    return _bin_positions(env, map_bin(env, posterior))


def mean_position(
    env: Environment,
    posterior: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Posterior mean position (expected value).

    Computes the probability-weighted average of bin center coordinates.

    Parameters
    ----------
    env : Environment
        Spatial environment providing bin_centers.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over positions.
        Each row should sum to 1.0.

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins, n_dims)
        Mean positions in environment coordinates.

    Notes
    -----
    On Cartesian environments this is ``posterior @ env.bin_centers``.

    On egocentric polar environments the mean is the probability-weighted
    centroid in the animal's egocentric plane (x ahead, y left), returned as
    ``(distance, angle)`` with the angle in the environment's angular range.
    Averaging the raw ``(distance, angle)`` columns would be wrong: two bins
    just either side of the angle seam behind the animal would average to
    straight ahead. Mass spread around the animal pulls the centroid distance
    toward zero, exactly as a Cartesian centroid falls between modes; when it
    reaches the animal (within rounding error) the angle is NaN.

    For unimodal posteriors, this is similar to the MAP position.
    For multimodal posteriors, this may fall between modes.

    This function mirrors ``DecodingResult.mean_position``.

    See Also
    --------
    map_position : MAP position (mode of posterior)
    DecodingResult.mean_position : Property version on result container

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.estimates import mean_position
    >>>
    >>> positions = np.column_stack([np.linspace(0, 10, 50), np.linspace(0, 10, 50)])
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Uniform posterior - mean is centroid of all bins
    >>> posterior = np.ones((1, env.n_bins)) / env.n_bins
    >>> mean_pos = mean_position(env, posterior)
    >>> mean_pos.shape
    (1, 2)
    """
    return _finish_posterior_mean(env, posterior @ _posterior_mean_basis(env))


def posterior_entropy(
    posterior: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Posterior entropy in bits (entropy measure).

    Measures the entropy in the position estimate. Higher values
    indicate more spread-out (uncertain) posteriors.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over positions.
        Each row should sum to 1.0.

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins,)
        Entropy values in bits. Range: [0, log2(n_bins)].

    Notes
    -----
    Uses mask-based computation to avoid bias from exact zeros:

    .. math::

        H = -\\sum_{i: p_i > 0} p_i \\log_2(p_i)

    This is more accurate than global clipping to ``[1e-10, 1]`` which
    can slightly bias entropy upward when many exact zeros occur.

    Maximum entropy (uniform distribution) is ``log2(n_bins)``.
    Minimum entropy (delta distribution) is 0.

    This is the entropy of the discrete distribution over bins: it counts
    bins, not length or area. On an environment with unequal bins, a
    posterior that is flat over space puts more mass in larger bins, so its
    entropy is below ``log2(n_bins)``.

    This function mirrors ``DecodingResult.posterior_entropy`` (named ``entropy``
    here to be precise about the statistic being computed).

    See Also
    --------
    DecodingResult.posterior_entropy : Property version on result container

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.estimates import posterior_entropy
    >>>
    >>> # Uniform posterior (maximum entropy)
    >>> posterior = np.ones((1, 8)) / 8
    >>> ent = posterior_entropy(posterior)
    >>> print(f"{ent[0]:.2f}")  # log2(8) = 3.0
    3.00
    >>>
    >>> # Delta posterior (minimum entropy)
    >>> posterior = np.zeros((1, 8))
    >>> posterior[0, 0] = 1.0
    >>> ent = posterior_entropy(posterior)
    >>> bool(abs(ent[0]) < 0.01)  # Should be 0 bits (within numerical precision)
    True
    """
    p = np.clip(posterior, 0.0, 1.0)
    # Vectorized mask-based entropy: avoid log(0) by using np.where
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log2(p), 0.0)
    result: NDArray[np.float64] = cast(
        "NDArray[np.float64]", -np.sum(p * log_p, axis=1)
    )
    return result


def credible_region(
    env: Environment,
    posterior: NDArray[np.float64],
    *,
    level: float = 0.95,
) -> list[NDArray[np.int64]]:
    """Highest-density credible region containing a given probability mass.

    Bins are added in order of decreasing probability density (mass per
    unit length, area or volume) until their mass reaches ``level``.

    Parameters
    ----------
    env : Environment
        Spatial environment providing bin sizes.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over positions.
        Each row should sum to 1.0.
    level : float, default=0.95
        Probability mass to include. Must be in (0, 1).
        Common values: 0.5 (50% region), 0.9, 0.95, 0.99.

    Returns
    -------
    list[NDArray[np.int64]]
        List of arrays, one per time bin. Each array contains the bin
        indices in the credible region, sorted by density (highest first) and
        by bin index within tied densities.

    Raises
    ------
    ValueError
        If level is not in (0, 1), or if the environment mixes bins measured
        by length and by area.

    Notes
    -----
    This is a highest posterior density (HPD) region built greedily by
    density. The posterior holds a probability mass per bin, so each bin's
    density is its mass divided by its size (``env.bin_sizes``). Bins enter
    in order of decreasing density until the cumulative mass reaches
    ``level``; ranking by mass instead would favor large bins. On equal
    bins the region has the fewest bins that reach ``level``, apart from
    the tied bins described below. On unequal bins it can be
    slightly larger in length or area than the smallest region holding
    ``level``, because the last bin needed may overshoot: a smaller bin
    further down the ranking could reach ``level`` with less space.

    Bins tied in density with the last bin needed all enter the region, so
    the region does not depend on bin numbering. Densities that agree to
    within ``sqrt(eps)`` of the posterior dtype, relative, count as tied,
    which absorbs rounding in the posterior. A tie can make the region hold
    more than ``level``: for a posterior that is uniform over space, the
    region is every bin.

    For a unimodal posterior, the HPD region is a contiguous set of bins
    around the mode.

    NaN/Inf posterior bins (e.g. masked or off-track positions) and bins
    with zero mass are never members. A row whose finite mass is below
    ``level`` keeps every bin with mass, and an undecodable (entirely NaN)
    row has an empty region.

    See Also
    --------
    posterior_entropy : Scalar entropy measure
    map_position : Point estimate (highest-density bin)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.estimates import credible_region
    >>>
    >>> positions = np.column_stack([np.linspace(0, 10, 50), np.linspace(0, 10, 50)])
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Delta posterior - HPD is single bin
    >>> posterior = np.zeros((1, env.n_bins))
    >>> posterior[0, 5] = 1.0
    >>> regions = credible_region(env, posterior, level=0.95)
    >>> len(regions[0])
    1
    """
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1 (exclusive), got {level}")

    from neurospatial.decoding._support import relative_bin_measures

    posterior = np.asarray(posterior)
    measures = relative_bin_measures(env, context="credible_region")
    finite = np.isfinite(posterior)
    # Mass and density of each bin; NaN/Inf bins hold no mass and sort last.
    mass = np.where(finite, posterior, 0.0)
    density = mass if measures is None else mass / measures
    density = np.where(finite, density, -np.inf)

    # Stable descending sort: tied densities stay in bin order.
    order = np.argsort(-density, axis=1, kind="stable")
    sorted_density = np.take_along_axis(density, order, axis=1)
    cumsum = np.cumsum(np.take_along_axis(mass, order, axis=1), axis=1)
    reached = cumsum >= level
    # Position of the bin that brings the mass to `level`.
    last = np.argmax(reached, axis=1)
    threshold = sorted_density[np.arange(len(posterior)), last]
    rtol = _tie_tolerance(posterior)
    in_region = sorted_density >= (threshold * (1.0 - rtol))[:, np.newaxis]
    n_bins_needed = in_region.sum(axis=1)

    # A row whose finite mass never reaches `level` (e.g. off-track/NaN rows)
    # keeps every bin with mass and never a NaN/Inf or zero-mass one; an
    # undecodable row gets an empty region.
    never_reached = ~reached.any(axis=1)
    n_bins_needed[never_reached] = (mass > 0).sum(axis=1)[never_reached]

    # Build result list (variable-length arrays require Python loop)
    return [
        order[t, : n_bins_needed[t]].astype(np.int64) for t in range(len(posterior))
    ]
