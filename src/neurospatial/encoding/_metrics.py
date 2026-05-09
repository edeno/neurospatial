"""Shared metric implementations for encoding result classes.

This module provides spatial information and sparsity computations that are
used by result classes (SpatialRateResult, DirectionalRateResult, etc.).

The functions in this module operate on dense arrays:
- Single neuron: firing_rate (n_bins,), occupancy (n_bins,)
- Batch: firing_rates (n_neurons, n_bins), occupancy (n_bins,)

**Backend-aware**: These functions detect the input array type and dispatch
to the appropriate backend (NumPy or JAX). JAX arrays are routed to the
JAX implementations in ``_core_jax.py``, preserving JAX-traced compute graphs.

- NumPy in → NumPy out
- JAX in → JAX out

**Automatic detection**: Users don't need to specify a ``backend`` parameter.
The functions automatically detect whether inputs are NumPy or JAX arrays
and dispatch accordingly.

For host-only operations, use ``_to_numpy()`` from ``_base.py`` first.

References
----------
.. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
       An information-theoretic approach to deciphering the hippocampal code.
       Advances in Neural Information Processing Systems, 5, 1030-1037.

.. [2] Skaggs, W. E., McNaughton, B. L., et al. (1996). Theta phase
       precession in hippocampal neuronal populations and the compression
       of temporal sequences. Hippocampus, 6(2), 149-172.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial.encoding._base import _is_jax_array


@dataclass(frozen=True)
class BatchScoresResult:
    """Container for per-neuron batch metric scores plus a per-neuron failure mask.

    Returned by :func:`batch_grid_scores` and :func:`batch_border_scores` so a
    caller can distinguish "this neuron's score is genuinely NaN (e.g.,
    constant firing)" from "this neuron's score computation raised an
    exception that we caught". Without the mask, both look like NaN to the
    downstream code, so a population-level statistic ("fraction of grid
    cells") silently treats failures as zeros and tests pass.

    Attributes
    ----------
    scores : NDArray[np.float64], shape (n_neurons,)
        Score for each neuron. NaN where computation could not produce a
        meaningful value (constant firing, all-NaN input, or a caught
        exception). Use the ``failures`` mask to tell the latter apart.
    failures : NDArray[np.bool_], shape (n_neurons,)
        ``True`` for any neuron whose score computation raised an exception
        that was caught and converted to NaN. ``False`` otherwise (including
        for neurons whose score is NaN for legitimate reasons such as
        constant firing).

    Notes
    -----
    The result behaves like an NDArray for indexing and length: ``len(result)``
    returns ``n_neurons``, ``result[i]`` returns the i-th score (a float).
    Generic helpers expecting a 1-D float array can use ``result.scores``.
    """

    scores: NDArray[np.float64]
    failures: NDArray[np.bool_]

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    def __getitem__(self, idx: Any) -> Any:
        # Delegate to the ndarray's own __getitem__ so all forms of
        # ndarray indexing work transparently: integer (-> scalar float),
        # slice (-> 1-D ndarray), boolean mask (-> 1-D ndarray), array
        # of indices (-> 1-D ndarray). The downstream code that used to
        # write `scores[~np.isnan(scores)]` against a plain ndarray
        # keeps working when `scores` is a BatchScoresResult.
        return self.scores[idx]

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> NDArray[np.float64]:
        # NumPy ufuncs (np.isnan, np.where, etc.) call __array__ when
        # given a non-ndarray input. Returning the underlying scores
        # array lets callers use BatchScoresResult anywhere they used
        # to use a plain (n_neurons,) array of scores. The ``copy``
        # keyword is required by NumPy 2's array protocol -- without it
        # ``np.asarray(result, copy=False)`` raises in NumPy 2.
        #
        # We mirror plain ndarray semantics: copy=False with a dtype that
        # requires a cast must raise, because the request is impossible
        # to satisfy without allocating. Silently copying would mask a
        # caller bug at the place where they explicitly asked for a view.
        target_dtype = None if dtype is None else np.dtype(dtype)
        if target_dtype is None or target_dtype == self.scores.dtype:
            if copy:
                return cast("NDArray[np.float64]", self.scores.copy())
            return self.scores
        if copy is False:
            raise ValueError(
                "Unable to avoid copy while creating an array as requested. "
                "If using `np.array(obj, copy=False)` replace it with "
                "`np.asarray(obj)` to allow a copy when needed (no copy is "
                "made if not required by the dtype). Note: np.asarray with "
                "copy=False has the same behavior."
            )
        return cast(
            "NDArray[np.float64]",
            self.scores.astype(target_dtype, copy=True),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying scores array, ``(n_neurons,)``.

        Pass-through to ``self.scores.shape`` so callers reading
        ``result.shape`` can keep doing so when the function previously
        returned a plain ndarray.
        """
        return self.scores.shape

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """dtype of the underlying scores array (always ``np.float64``)."""
        return self.scores.dtype

    @property
    def n_failures(self) -> int:
        """Total number of neurons whose computation raised an exception."""
        return int(self.failures.sum())


if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol

__all__ = [
    "batch_border_scores",
    "batch_grid_scores",
    "batch_sparsity",
    "batch_spatial_information",
    "information_per_second",
    "mutual_information",
    "selectivity",
    "sparsity",
    "spatial_coverage_single_cell",
    "spatial_information",
]


def spatial_information(
    firing_rate: ArrayLike,
    occupancy: ArrayLike,
    *,
    base: float = 2.0,
) -> float | Any:
    """Compute Skaggs spatial information (bits per spike) for single neuron.

    Spatial information quantifies how much information each spike conveys
    about the animal's spatial location. This is a fundamental metric for
    classifying place cells and other spatially-tuned neurons.

    Parameters
    ----------
    firing_rate : ArrayLike, shape (n_bins,)
        Firing rate map in Hz. Can contain NaN values which are ignored.
        Accepts NumPy arrays or JAX arrays.
    occupancy : ArrayLike, shape (n_bins,)
        Time spent in each bin (seconds or any time unit). Will be normalized
        to probability internally. Can contain NaN values which are ignored.
    base : float, default=2.0
        Logarithm base. Use 2.0 for bits (standard), np.e for nats.

    Returns
    -------
    float | jax.Array
        Spatial information in bits per spike (if base=2.0).
        Returns 0.0 if mean rate is zero or undefined.
        Returns float for NumPy input, JAX scalar for JAX input.

    Raises
    ------
    ValueError
        If firing_rate and occupancy have different shapes.

    Notes
    -----
    **Formula (Skaggs et al. 1993)**:

    .. math::

        I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log \\left( \\frac{r_i}{\\bar{r}} \\right)

    where :math:`p_i` is occupancy probability, :math:`r_i` is firing rate
    in bin :math:`i`, and :math:`\\bar{r}` is mean firing rate.

    **Interpretation**:

    - Place cells typically have 1-3 bits/spike
    - Higher values indicate more spatially selective firing
    - Zero information means uniform firing (no spatial selectivity)

    **Backend-aware**: Detects input array type and dispatches to NumPy or
    JAX implementation. JAX arrays preserve JAX-traced compute graphs.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._metrics import spatial_information

    >>> # Uniform firing → zero information
    >>> firing_rate = np.ones(100) * 3.0
    >>> occupancy = np.ones(100)
    >>> info = spatial_information(firing_rate, occupancy)
    >>> abs(info) < 1e-6  # Should be ~0
    True

    >>> # Selective firing → high information
    >>> firing_rate = np.zeros(100)
    >>> firing_rate[50] = 30.0  # Fires only in bin 50
    >>> occupancy = np.ones(100)
    >>> info = spatial_information(firing_rate, occupancy)
    >>> info > 4.0  # High spatial info
    True

    See Also
    --------
    batch_spatial_information : Vectorized version for populations
    sparsity : Complementary measure of spatial selectivity
    """
    # Check if JAX array and dispatch to JAX implementation
    if _is_jax_array(firing_rate) or _is_jax_array(occupancy):
        from neurospatial.encoding._core_jax import spatial_information_single

        return spatial_information_single(firing_rate, occupancy, base=base)  # type: ignore[arg-type]

    # NumPy implementation — input validation here (boundary), kernel in _core_numpy.
    firing_rate_arr = np.asarray(firing_rate)
    occupancy_arr = np.asarray(occupancy)

    if firing_rate_arr.shape != occupancy_arr.shape:
        raise ValueError(
            f"firing_rate shape {firing_rate_arr.shape} does not match "
            f"occupancy shape {occupancy_arr.shape}"
        )
    if firing_rate_arr.size == 0:
        raise ValueError("firing_rate and occupancy cannot be empty arrays")

    from neurospatial.encoding._core_numpy import spatial_information_single as _np_si

    return float(_np_si(firing_rate_arr, occupancy_arr, base=base))


def batch_spatial_information(
    firing_rates: ArrayLike,
    occupancy: ArrayLike,
    *,
    base: float = 2.0,
) -> NDArray[np.float64] | Any:
    """Compute Skaggs spatial information for multiple neurons.

    Vectorized version of `spatial_information()` for efficient population
    analysis. Computes spatial information for each neuron in a batch.

    Parameters
    ----------
    firing_rates : ArrayLike, shape (n_neurons, n_bins)
        Firing rate maps for each neuron in Hz.
        Accepts NumPy arrays or JAX arrays.
    occupancy : ArrayLike, shape (n_bins,)
        Shared occupancy for all neurons (time spent in each bin).
    base : float, default=2.0
        Logarithm base. Use 2.0 for bits (standard), np.e for nats.

    Returns
    -------
    ndarray | jax.Array, shape (n_neurons,)
        Spatial information in bits per spike for each neuron.
        Returns NumPy array for NumPy input, JAX array for JAX input.

    Raises
    ------
    ValueError
        If firing_rates.shape[1] != occupancy.shape[0].

    Notes
    -----
    This function computes spatial information independently for each neuron.
    The occupancy is shared across all neurons (same behavioral sampling).

    **Backend-aware**: Detects input array type and dispatches to NumPy or
    JAX implementation. JAX uses ``vmap`` for efficient vectorization.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._metrics import batch_spatial_information

    >>> # 5 neurons, 100 bins
    >>> firing_rates = np.random.rand(5, 100) * 10
    >>> occupancy = np.ones(100)
    >>> info = batch_spatial_information(firing_rates, occupancy)
    >>> info.shape
    (5,)

    See Also
    --------
    spatial_information : Single-neuron version
    """
    # Check if JAX array and dispatch to JAX implementation
    if _is_jax_array(firing_rates) or _is_jax_array(occupancy):
        from neurospatial.encoding._core_jax import spatial_information_batch

        return spatial_information_batch(firing_rates, occupancy, base=base)  # type: ignore[arg-type]

    # NumPy implementation — input validation here, kernel in _core_numpy.
    firing_rates_arr = np.asarray(firing_rates)
    occupancy_arr = np.asarray(occupancy)

    if firing_rates_arr.ndim != 2:
        raise ValueError(
            f"firing_rates must be 2D (n_neurons, n_bins), got shape {firing_rates_arr.shape}"
        )
    if occupancy_arr.ndim != 1:
        raise ValueError(
            f"occupancy must be 1D (n_bins,), got shape {occupancy_arr.shape}"
        )
    if firing_rates_arr.shape[1] != occupancy_arr.shape[0]:
        raise ValueError(
            f"firing_rates has {firing_rates_arr.shape[1]} bins but "
            f"occupancy has {occupancy_arr.shape[0]} bins"
        )

    from neurospatial.encoding._core_numpy import spatial_information_batch as _np_si

    return _np_si(firing_rates_arr, occupancy_arr, base=base)


def sparsity(
    firing_rate: ArrayLike,
    occupancy: ArrayLike,
) -> float | Any:
    """Compute sparsity of spatial firing for single neuron.

    Sparsity measures what fraction of the environment elicits significant
    firing. Lower values indicate sparser, more selective place fields.

    Parameters
    ----------
    firing_rate : ArrayLike, shape (n_bins,)
        Firing rate map in Hz. Can contain NaN values which are ignored.
        Accepts NumPy arrays or JAX arrays.
    occupancy : ArrayLike, shape (n_bins,)
        Time spent in each bin (seconds or any time unit). Will be normalized
        to probability internally. Can contain NaN values which are ignored.

    Returns
    -------
    float | jax.Array
        Sparsity value in range [0, 1]. Lower values indicate sparser firing.
        Returns 0.0 if denominator is zero or undefined.
        Returns float for NumPy input, JAX scalar for JAX input.

    Raises
    ------
    ValueError
        If firing_rate and occupancy have different shapes.

    Notes
    -----
    **Formula (Skaggs et al. 1996)**:

    .. math::

        S = \\frac{\\left( \\sum_i p_i r_i \\right)^2}{\\sum_i p_i r_i^2}

    where :math:`p_i` is occupancy probability and :math:`r_i` is firing rate.

    **Interpretation**:

    - Range: [0, 1]
    - Low sparsity (0.1-0.3): Sparse, selective place field
    - High sparsity (~1.0): Uniform firing throughout environment
    - Typical place cells: 0.1-0.3

    **Backend-aware**: Detects input array type and dispatches to NumPy or
    JAX implementation. JAX arrays preserve JAX-traced compute graphs.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._metrics import sparsity

    >>> # Uniform firing → high sparsity (close to 1)
    >>> firing_rate = np.ones(100) * 5.0
    >>> occupancy = np.ones(100)
    >>> spars = sparsity(firing_rate, occupancy)
    >>> spars > 0.99
    True

    >>> # Selective firing → low sparsity
    >>> firing_rate = np.zeros(100)
    >>> firing_rate[50] = 30.0  # Fires only in bin 50
    >>> occupancy = np.ones(100)
    >>> spars = sparsity(firing_rate, occupancy)
    >>> spars < 0.1
    True

    See Also
    --------
    batch_sparsity : Vectorized version for populations
    spatial_information : Complementary measure of spatial selectivity
    """
    # Check if JAX array and dispatch to JAX implementation
    if _is_jax_array(firing_rate) or _is_jax_array(occupancy):
        from neurospatial.encoding._core_jax import sparsity_single

        return sparsity_single(firing_rate, occupancy)  # type: ignore[arg-type]

    # NumPy implementation — input validation here, kernel in _core_numpy.
    firing_rate_arr = np.asarray(firing_rate)
    occupancy_arr = np.asarray(occupancy)

    if firing_rate_arr.shape != occupancy_arr.shape:
        raise ValueError(
            f"firing_rate shape {firing_rate_arr.shape} does not match "
            f"occupancy shape {occupancy_arr.shape}"
        )
    if firing_rate_arr.size == 0:
        raise ValueError("firing_rate and occupancy cannot be empty arrays")

    from neurospatial.encoding._core_numpy import sparsity_single as _np_sp

    return float(_np_sp(firing_rate_arr, occupancy_arr))


def batch_sparsity(
    firing_rates: ArrayLike,
    occupancy: ArrayLike,
) -> NDArray[np.float64] | Any:
    """Compute sparsity for multiple neurons.

    Vectorized version of `sparsity()` for efficient population analysis.
    Computes sparsity for each neuron in a batch.

    Parameters
    ----------
    firing_rates : ArrayLike, shape (n_neurons, n_bins)
        Firing rate maps for each neuron in Hz.
        Accepts NumPy arrays or JAX arrays.
    occupancy : ArrayLike, shape (n_bins,)
        Shared occupancy for all neurons (time spent in each bin).

    Returns
    -------
    ndarray | jax.Array, shape (n_neurons,)
        Sparsity values in range [0, 1] for each neuron.
        Returns NumPy array for NumPy input, JAX array for JAX input.

    Raises
    ------
    ValueError
        If firing_rates.shape[1] != occupancy.shape[0].

    Notes
    -----
    This function computes sparsity independently for each neuron.
    The occupancy is shared across all neurons (same behavioral sampling).

    **Backend-aware**: Detects input array type and dispatches to NumPy or
    JAX implementation. JAX uses ``vmap`` for efficient vectorization.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._metrics import batch_sparsity

    >>> # 5 neurons, 100 bins
    >>> firing_rates = np.random.rand(5, 100) * 10
    >>> occupancy = np.ones(100)
    >>> spars = batch_sparsity(firing_rates, occupancy)
    >>> spars.shape
    (5,)

    See Also
    --------
    sparsity : Single-neuron version
    """
    # Check if JAX array and dispatch to JAX implementation
    if _is_jax_array(firing_rates) or _is_jax_array(occupancy):
        from neurospatial.encoding._core_jax import sparsity_batch

        return sparsity_batch(firing_rates, occupancy)  # type: ignore[arg-type]

    # NumPy implementation
    firing_rates = np.asarray(firing_rates)
    occupancy = np.asarray(occupancy)

    # Validate shapes
    if firing_rates.ndim != 2:
        raise ValueError(
            f"firing_rates must be 2D (n_neurons, n_bins), got shape {firing_rates.shape}"
        )

    if occupancy.ndim != 1:
        raise ValueError(f"occupancy must be 1D (n_bins,), got shape {occupancy.shape}")

    if firing_rates.shape[1] != occupancy.shape[0]:
        raise ValueError(
            f"firing_rates has {firing_rates.shape[1]} bins but "
            f"occupancy has {occupancy.shape[0]} bins"
        )

    from neurospatial.encoding._core_numpy import sparsity_batch as _np_sp

    return _np_sp(firing_rates, occupancy)


def batch_grid_scores(
    env: Environment,
    firing_rates: NDArray[np.float64],
    *,
    inner_radius_fraction: float = 0.2,
    outer_radius_fraction: float = 0.5,
) -> BatchScoresResult:
    """Compute grid scores for multiple neurons.

    Computes grid score (hexagonal periodicity) for each neuron's firing
    rate map. Grid score quantifies how well the firing pattern exhibits
    the characteristic 6-fold rotational symmetry of grid cells.

    Parameters
    ----------
    env : Environment
        Spatial environment containing bin centers and connectivity.
        Must be a regular 2D grid for FFT-based autocorrelation.
    firing_rates : ndarray, shape (n_neurons, n_bins)
        Firing rate maps for each neuron in Hz.
    inner_radius_fraction : float, default=0.2
        Inner radius of annular region as fraction of autocorrelogram semi-axis.
        See :func:`~neurospatial.encoding.grid.grid_score` for details.
    outer_radius_fraction : float, default=0.5
        Outer radius of annular region as fraction of autocorrelogram semi-axis.
        See :func:`~neurospatial.encoding.grid.grid_score` for details.

    Returns
    -------
    BatchScoresResult
        Container with two parallel arrays of length ``n_neurons``:

        - ``scores``: grid scores in range [-2, 2]. NaN where grid score
          cannot be computed (constant firing, invalid autocorrelation,
          non-regular grid environment, or a caught exception).
        - ``failures``: boolean mask, ``True`` for neurons whose
          computation raised an exception that was caught and converted
          to NaN. Use this to distinguish "score is NaN for legitimate
          reasons (e.g., constant firing → unitless autocorrelation)"
          from "score is NaN because grid_score crashed". A
          ``UserWarning`` is also emitted when at least one neuron
          fails so the failure surfaces in logs even if the caller
          ignores ``failures``.

    Raises
    ------
    ValueError
        If firing_rates is not 2D.
        If firing_rates.shape[1] != env.n_bins.

    Notes
    -----
    For each neuron, this function:

    1. Computes spatial autocorrelation using :func:`spatial_autocorrelation`
    2. Computes grid score using :func:`grid_score`

    **Interpretation**:

    - **score > 0.4**: Strong hexagonal grid (typical threshold for grid cells)
    - **score ≈ 0**: No hexagonal structure (place cells, non-spatial cells)
    - **score < 0**: Anti-hexagonal structure (rare)

    **Environment requirements**:

    Grid score computation requires a regular 2D grid environment for FFT-based
    autocorrelation. For irregular environments, the function returns NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._metrics import batch_grid_scores

    >>> # Create environment
    >>> x = np.linspace(-50, 50, 51)
    >>> xx, yy = np.meshgrid(x, x)
    >>> positions = np.column_stack([xx.ravel(), yy.ravel()])
    >>> env = Environment.from_samples(positions, bin_size=2.0)

    >>> # Random firing patterns (5 neurons)
    >>> rng = np.random.default_rng(42)
    >>> firing_rates = rng.random((5, env.n_bins)) * 10
    >>> result = batch_grid_scores(env, firing_rates)
    >>> result.scores.shape
    (5,)
    >>> result.failures.shape
    (5,)

    See Also
    --------
    neurospatial.encoding.grid.grid_score : Single autocorrelogram grid score
    neurospatial.encoding.grid.spatial_autocorrelation : Compute autocorrelation
    """
    from neurospatial.encoding.grid import grid_score, spatial_autocorrelation

    firing_rates = np.asarray(firing_rates)

    # Validate shapes
    if firing_rates.ndim != 2:
        raise ValueError(
            f"firing_rates must be 2D (n_neurons, n_bins), got shape {firing_rates.shape}"
        )

    if firing_rates.shape[1] != env.n_bins:
        raise ValueError(
            f"firing_rates has {firing_rates.shape[1]} bins but "
            f"env has {env.n_bins} bins"
        )

    n_neurons = firing_rates.shape[0]
    scores = np.empty(n_neurons, dtype=np.float64)
    failures = np.zeros(n_neurons, dtype=np.bool_)

    for i in range(n_neurons):
        firing_rate = firing_rates[i]

        # Pre-check legitimate-NaN inputs before calling
        # spatial_autocorrelation. Both conditions raise ValueError from
        # inside spatial_autocorrelation, but they reflect properties of
        # the input rate map (no spatial structure to autocorrelate),
        # not a computation failure. Pre-checking lets the catch-all
        # below flag only true failures.
        finite_mask = np.isfinite(firing_rate)
        if not np.any(finite_mask):
            scores[i] = np.nan
            continue
        finite_rates = firing_rate[finite_mask]
        if np.all(finite_rates == finite_rates[0]):
            scores[i] = np.nan
            continue

        try:
            # Compute spatial autocorrelation (FFT, regular 2D grid only).
            # Irregular environments (where the FFT path is undefined) raise
            # ValueError, which the catch-all below converts into a recorded
            # NaN+failure flag for the caller's failures mask. Direct callers
            # who need autocorrelation on irregular topologies should use
            # `spatial_autocorrelation_radial` (1D distance profile) instead.
            autocorr = spatial_autocorrelation(env, firing_rate)
            scores[i] = grid_score(
                autocorr,
                inner_radius_fraction=inner_radius_fraction,
                outer_radius_fraction=outer_radius_fraction,
            )
        except (ValueError, RuntimeError):
            # Caught exception: record both NaN and the failure flag so
            # the caller can distinguish this from a legitimate-NaN.
            scores[i] = np.nan
            failures[i] = True

    n_failed = int(failures.sum())
    if n_failed > 0:
        warnings.warn(
            f"batch_grid_scores: grid_score raised an exception for "
            f"{n_failed} of {n_neurons} neuron"
            f"{'' if n_neurons == 1 else 's'}; their scores were set to "
            "NaN. Inspect the `failures` mask on the returned "
            "BatchScoresResult to identify which neurons failed.",
            UserWarning,
            stacklevel=2,
        )
    return BatchScoresResult(scores=scores, failures=failures)


def batch_border_scores(
    env: Environment,
    firing_rates: NDArray[np.float64],
    *,
    threshold: float = 0.3,
    min_area: float = 0.0,
    metric: str = "geodesic",
) -> BatchScoresResult:
    """Compute border scores for multiple neurons.

    Computes border score (boundary proximity tuning) for each neuron's firing
    rate map. Border score quantifies how much a cell's firing field is aligned
    with environmental boundaries.

    Parameters
    ----------
    env : Environment
        Spatial environment containing bin centers and connectivity.
    firing_rates : ndarray, shape (n_neurons, n_bins)
        Firing rate maps for each neuron in Hz.
    threshold : float, default=0.3
        Fraction of peak firing rate used to segment the field.
        Follows Solstad et al. (2008). Must be in (0, 1).
    min_area : float, default=0.0
        Minimum field area (in physical units) to compute border score.
        Fields smaller than this return NaN. For rat hippocampal data,
        Solstad et al. (2008) used 200 cm². Adjust based on bin size
        and environment scale.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        Distance metric for computing distance from field bins to boundary bins.
        - "geodesic": Graph shortest path distance. Respects environment
          connectivity, appropriate for irregular environments or those with obstacles.
        - "euclidean": Straight-line distance in physical space. Generally faster
          for large environments.

    Returns
    -------
    BatchScoresResult
        Container with two parallel arrays of length ``n_neurons``:

        - ``scores``: border scores in range [-1, 1]. NaN where border
          score cannot be computed (zero firing, invalid field, no
          field bins, or a caught exception).
        - ``failures``: boolean mask, ``True`` for neurons whose
          ``border_score`` call raised ``ValueError`` or
          ``RuntimeError`` and was caught. A ``UserWarning`` is also
          emitted when at least one neuron fails.

    Raises
    ------
    ValueError
        If firing_rates is not 2D.
        If firing_rates.shape[1] != env.n_bins.

    Notes
    -----
    For each neuron, this function delegates to
    :func:`~neurospatial.encoding.border.border_score`.

    **Interpretation**:

    - **score > 0.5**: Strong border cell (field aligned with boundary)
    - **score ≈ 0**: No boundary preference (uniform or mixed)
    - **score < 0**: Anti-border (field in center, far from boundaries)

    References
    ----------
    .. [1] Solstad, T., Boccara, C. N., Kropff, E., Moser, M. B., & Moser, E. I.
           (2008). Representation of geometric borders in the entorhinal cortex.
           Science, 322(5909), 1865-1868.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._metrics import batch_border_scores

    >>> # Create environment
    >>> x = np.linspace(-50, 50, 51)
    >>> xx, yy = np.meshgrid(x, x)
    >>> positions = np.column_stack([xx.ravel(), yy.ravel()])
    >>> env = Environment.from_samples(positions, bin_size=2.0)

    >>> # Random firing patterns (5 neurons)
    >>> rng = np.random.default_rng(42)
    >>> firing_rates = rng.random((5, env.n_bins)) * 10
    >>> scores = batch_border_scores(env, firing_rates)
    >>> scores.shape
    (5,)

    See Also
    --------
    neurospatial.encoding.border.border_score : Single-neuron border score
    """
    from neurospatial.encoding.border import border_score

    firing_rates = np.asarray(firing_rates)

    # Validate shapes
    if firing_rates.ndim != 2:
        raise ValueError(
            f"firing_rates must be 2D (n_neurons, n_bins), got shape {firing_rates.shape}"
        )

    if firing_rates.shape[1] != env.n_bins:
        raise ValueError(
            f"firing_rates has {firing_rates.shape[1]} bins but "
            f"env has {env.n_bins} bins"
        )

    n_neurons = firing_rates.shape[0]
    scores = np.empty(n_neurons, dtype=np.float64)
    failures = np.zeros(n_neurons, dtype=np.bool_)

    for i in range(n_neurons):
        firing_rate = firing_rates[i]

        try:
            scores[i] = border_score(
                cast("EnvironmentProtocol", env),
                firing_rate,
                threshold=threshold,
                min_area=min_area,
                metric=metric,  # type: ignore[arg-type]
            )
        except (ValueError, RuntimeError):
            # Caught exception: record both NaN and the failure flag so
            # the caller can distinguish this from a legitimate-NaN.
            scores[i] = np.nan
            failures[i] = True

    n_failed = int(failures.sum())
    if n_failed > 0:
        warnings.warn(
            f"batch_border_scores: border_score raised an exception for "
            f"{n_failed} of {n_neurons} neuron"
            f"{'' if n_neurons == 1 else 's'}; their scores were set to "
            "NaN. Inspect the `failures` mask on the returned "
            "BatchScoresResult to identify which neurons failed.",
            UserWarning,
            stacklevel=2,
        )
    return BatchScoresResult(scores=scores, failures=failures)


def selectivity(
    firing_rate: ArrayLike,
    occupancy: ArrayLike,
) -> float:
    """Compute spatial selectivity (peak rate / mean rate).

    Selectivity measures how spatially selective a cell's firing is. Higher
    values indicate the cell fires strongly in a small region and weakly
    elsewhere. A value of 1.0 indicates uniform firing throughout the
    environment.

    Parameters
    ----------
    firing_rate : ArrayLike, shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    occupancy : ArrayLike, shape (n_bins,)
        Occupancy probability (normalized to sum to 1).

    Returns
    -------
    float
        Selectivity value, always >= 1.0. Returns NaN if:
        - Mean rate is zero (division by zero)
        - All firing rates are NaN
        Returns infinity if peak rate is positive but mean rate is zero.

    Notes
    -----
    **Formula**:

    .. math::

        S = \\frac{r_{\\text{peak}}}{\\bar{r}}

    where :math:`r_{\\text{peak}}` is the maximum firing rate and
    :math:`\\bar{r}` is the occupancy-weighted mean firing rate.

    **Interpretation**:

    - **Selectivity = 1.0**: Uniform firing (peak equals mean)
    - **Selectivity = 2-5**: Moderately selective place field
    - **Selectivity > 10**: Highly selective place field (fires in small region)

    **NaN handling**: NaN values in firing_rate are excluded from peak and mean
    calculations. Occupancy is renormalized to valid bins.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._metrics import selectivity

    >>> # Uniform firing -> selectivity = 1.0
    >>> firing_rate = np.ones(100) * 5.0
    >>> occupancy = np.ones(100) / 100
    >>> select = selectivity(firing_rate, occupancy)
    >>> abs(select - 1.0) < 1e-6
    True

    See Also
    --------
    spatial_information : Spatial information (bits/spike)
    sparsity : Spatial sparsity

    References
    ----------
    .. [1] opexebo package (Moser Lab):
           https://github.com/kavli-ntnu/opexebo
    """
    firing_rate = np.asarray(firing_rate)
    occupancy = np.asarray(occupancy)

    # Handle NaN values
    valid_mask = np.isfinite(firing_rate) & np.isfinite(occupancy)

    if not np.any(valid_mask):
        # All NaN
        return np.nan

    # Get valid values
    firing_rate_valid = firing_rate[valid_mask]
    occupancy_valid = occupancy[valid_mask]

    # Normalize occupancy to probability
    occupancy_prob = occupancy_valid / np.sum(occupancy_valid)

    # Peak firing rate
    peak_rate = np.max(firing_rate_valid)

    # Mean firing rate (occupancy-weighted)
    mean_rate = np.sum(occupancy_prob * firing_rate_valid)

    # Compute selectivity
    if mean_rate == 0:
        # Division by zero
        if peak_rate > 0:
            return np.inf
        else:
            return np.nan

    selectivity_value = peak_rate / mean_rate

    return float(selectivity_value)


def information_per_second(
    firing_rate: ArrayLike,
    occupancy: ArrayLike,
    *,
    base: float = 2.0,
) -> float:
    """Compute spatial information in bits per second.

    This metric combines spatial information content (bits/spike) with the
    cell's firing rate to give information transmission rate. It measures
    how many bits of spatial information the cell conveys per second.

    Parameters
    ----------
    firing_rate : ArrayLike, shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    occupancy : ArrayLike, shape (n_bins,)
        Occupancy probability (normalized to sum to 1).
    base : float, default=2.0
        Logarithm base for information calculation. Use 2.0 for bits,
        np.e for nats.

    Returns
    -------
    float
        Information rate in bits/second (or nats/second if base=e).
        Returns NaN if firing rate or occupancy are all NaN.

    Notes
    -----
    **Formula**:

    .. math::

        I_{\\text{rate}} = I_{\\text{content}} \\times \\bar{r}

    where :math:`I_{\\text{content}}` is the Skaggs spatial information
    (bits/spike) and :math:`\\bar{r}` is the mean firing rate (spikes/second).

    **Interpretation**:

    - Combines "how much info per spike" with "how many spikes per second"
    - A cell can have high bits/spike but low bits/second if it fires rarely
    - Conversely, a cell with low selectivity but high rate can have high bits/second

    **Use case**: This metric favors cells that both fire frequently AND are
    spatially selective, making it useful for identifying the most informative
    place cells for population decoding.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._metrics import information_per_second

    >>> # Highly selective but rare firing
    >>> firing_rate = np.zeros(100)
    >>> firing_rate[50] = 10.0  # 10 Hz in one bin, 0.1 Hz mean
    >>> occupancy = np.ones(100) / 100
    >>> info_rate = information_per_second(firing_rate, occupancy)
    >>> info_rate > 0
    True

    See Also
    --------
    spatial_information : Spatial information (bits/spike)
    mutual_information : Mutual information between position and firing

    References
    ----------
    .. [1] Markus et al. (1994). Interactions between location and task affect
           the spatial and directional firing of hippocampal neurons. J Neurosci 14(11).
    """
    # Compute Skaggs information (bits/spike)
    info_content = spatial_information(firing_rate, occupancy, base=base)

    firing_rate = np.asarray(firing_rate)
    occupancy = np.asarray(occupancy)

    # Handle NaN values for mean rate calculation
    valid_mask = np.isfinite(firing_rate) & np.isfinite(occupancy)

    if not np.any(valid_mask):
        return np.nan

    firing_rate_valid = firing_rate[valid_mask]
    occupancy_valid = occupancy[valid_mask]

    # Normalize occupancy
    occupancy_prob = occupancy_valid / np.sum(occupancy_valid)

    # Mean firing rate (occupancy-weighted)
    mean_rate = np.sum(occupancy_prob * firing_rate_valid)

    # Information rate = bits/spike x spikes/second = bits/second
    info_rate = info_content * mean_rate

    return float(info_rate)


def mutual_information(
    firing_rate: ArrayLike,
    occupancy: ArrayLike,
    *,
    base: float = 2.0,
) -> float:
    """Compute mutual information between position and firing rate.

    Mutual information quantifies how much knowing the animal's position
    reduces uncertainty about the neuron's firing rate. This is a fundamental
    information-theoretic measure of spatial coding.

    Parameters
    ----------
    firing_rate : ArrayLike, shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    occupancy : ArrayLike, shape (n_bins,)
        Occupancy probability (normalized to sum to 1).
    base : float, default=2.0
        Logarithm base for information calculation. Use 2.0 for bits,
        np.e for nats.

    Returns
    -------
    float
        Mutual information in bits (or nats if base=e). Returns NaN if
        firing rate or occupancy are all NaN or if mean rate is zero.

    Notes
    -----
    **Relationship to other metrics**:

    - ``mutual_information`` = ``spatial_information`` x ``mean_rate``
    - ``mutual_information`` = ``information_per_second``
    - MI is symmetric: MI(position; firing) = MI(firing; position)

    **Interpretation**:

    - **MI = 0**: Position and firing are independent (no place field)
    - **MI > 0**: Position provides information about firing
    - Higher MI indicates stronger spatial coding

    **Difference from Skaggs information**: Skaggs info is bits per spike,
    MI is total bits. A cell with high Skaggs but low firing rate will have
    lower MI than a moderately selective cell that fires frequently.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._metrics import mutual_information

    >>> # Strong place field
    >>> firing_rate = np.ones(100) * 0.5
    >>> firing_rate[40:50] = 10.0
    >>> occupancy = np.ones(100) / 100
    >>> mi = mutual_information(firing_rate, occupancy)
    >>> mi > 0
    True

    See Also
    --------
    spatial_information : Spatial information (bits/spike)
    information_per_second : Information rate (equivalent to MI)
    sparsity : Sparsity measure

    References
    ----------
    .. [1] Skaggs et al. (1993). An information-theoretic approach to deciphering
           the hippocampal code. NIPS.
    """
    # MI is mathematically equivalent to information_per_second
    return information_per_second(firing_rate, occupancy, base=base)


def spatial_coverage_single_cell(
    firing_rate: ArrayLike,
    *,
    threshold: float = 0.1,
) -> float:
    """Compute fraction of environment where cell fires above threshold.

    This metric quantifies how much of the spatial environment a single cell
    covers with its firing. Lower values indicate more spatially selective
    place fields.

    Parameters
    ----------
    firing_rate : ArrayLike, shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    threshold : float, default=0.1
        Minimum firing rate (Hz) to consider a bin as "covered".
        Standard values: 0.1 Hz (minimal activity) or 1.0 Hz (clear activity).

    Returns
    -------
    float
        Fraction of bins with firing rate > threshold, in range [0, 1].
        Returns NaN if all firing rates are NaN.

    Notes
    -----
    **Formula**:

    .. math::

        C = \\frac{\\sum_i \\mathbb{1}[r_i > \\theta]}{N}

    where :math:`r_i` is firing rate in bin :math:`i`, :math:`\\theta` is
    the threshold, and :math:`N` is the total number of bins.

    **Interpretation**:

    - **Coverage = 0.0**: Cell fires nowhere (no place field)
    - **Coverage = 0.1**: Cell fires in 10% of environment (highly selective)
    - **Coverage = 0.5**: Cell fires in half the environment (broad field)
    - **Coverage = 1.0**: Cell fires everywhere (no spatial selectivity)

    **Relationship to other metrics**:

    - Inverse of selectivity: high coverage -> low selectivity
    - Complementary to sparsity: both measure spatial specificity

    **NaN handling**: NaN values in firing_rate are treated as bins with
    zero firing (below threshold).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._metrics import spatial_coverage_single_cell

    >>> # Highly selective cell (fires in 10% of bins)
    >>> firing_rate = np.zeros(100)
    >>> firing_rate[40:50] = 5.0
    >>> coverage = spatial_coverage_single_cell(firing_rate, threshold=0.1)
    >>> abs(coverage - 0.10) < 1e-6
    True

    See Also
    --------
    sparsity : Sparsity measure (inverse of coverage)
    selectivity : Peak / mean rate ratio

    References
    ----------
    .. [1] Muller et al. (1987). The effects of changes in the environment on
           the spatial firing of hippocampal complex-spike cells. J Neurosci 7(7).
    """
    firing_rate = np.asarray(firing_rate)

    # Handle NaN values (treat as below threshold)
    valid_mask = np.isfinite(firing_rate)

    if not np.any(valid_mask):
        return np.nan

    # Count bins above threshold
    n_above = np.sum(firing_rate[valid_mask] > threshold)

    # Total number of bins (including NaN bins as zeros)
    n_total = len(firing_rate)

    coverage = n_above / n_total

    return float(coverage)
