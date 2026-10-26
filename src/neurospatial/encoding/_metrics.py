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

from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial.encoding._base import _is_jax_array


@dataclass(frozen=True, repr=False, eq=False)
class BatchScoresResult(ReadOnlyArraysMixin, ResultMixin):
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

    def __post_init__(self) -> None:
        """Own read-only copies of the scores and failure mask."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "n_scores": int(self.scores.size),
            "n_failures": int(np.count_nonzero(self.failures)),
            "mean_score": (
                float(np.nanmean(self.scores)) if self.scores.size else float("nan")
            ),
        }

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    def __getitem__(self, idx: Any) -> Any:
        # ndarray indexing: integer -> float, slice/mask/index array -> 1-D.
        return self.scores[idx]

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> NDArray[np.float64]:
        # NumPy functions (np.isnan, np.where, ...) call __array__ on a
        # non-ndarray input, so the result works wherever an (n_neurons,)
        # score array does. NumPy 2's array protocol passes ``copy``.
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
        """Shape of the underlying scores array, ``(n_neurons,)``."""
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
    from neurospatial.environment import Environment
    from neurospatial.geometry._boundaries import BoundarySource

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
        Firing rate map in Hz. Bins with non-finite rates are excluded.
        Accepts NumPy arrays or JAX arrays.
    occupancy : ArrayLike, shape (n_bins,)
        Time spent in each bin (seconds or any time unit). Will be normalized
        over bins with finite rates and occupancy. Observed zero rates remain
        included.
    base : float, default=2.0
        Logarithm base. Use 2.0 for bits (standard), np.e for nats.

    Returns
    -------
    float | jax.Array
        Spatial information in bits per spike (if base=2.0).
        NaN for a silent map (mean rate 0), an all-NaN map or zero total
        occupancy: bits per spike are undefined without spikes.
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
        import jax.numpy as jnp

        from neurospatial.encoding._core_jax import spatial_information_single

        return spatial_information_single(
            jnp.asarray(firing_rate), jnp.asarray(occupancy), base=base
        )

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
        Shared occupancy for all neurons (time spent in each bin), normalized
        separately over each neuron's finite rate/occupancy pairs.
    base : float, default=2.0
        Logarithm base. Use 2.0 for bits (standard), np.e for nats.

    Returns
    -------
    ndarray | jax.Array, shape (n_neurons,)
        Spatial information in bits per spike for each neuron, NaN for a
        silent or all-NaN map or zero total occupancy.
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
        import jax.numpy as jnp

        from neurospatial.encoding._core_jax import spatial_information_batch

        return spatial_information_batch(
            jnp.asarray(firing_rates), jnp.asarray(occupancy), base=base
        )

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
        Firing rate map in Hz. Bins with non-finite rates are excluded.
        Accepts NumPy arrays or JAX arrays.
    occupancy : ArrayLike, shape (n_bins,)
        Time spent in each bin (seconds or any time unit). Will be normalized
        over bins with finite rates and occupancy. Observed zero rates remain
        included.

    Returns
    -------
    float | jax.Array
        Sparsity value in range [0, 1]. Lower values indicate sparser firing.
        NaN for a silent map, an all-NaN map or zero total occupancy.
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
        import jax.numpy as jnp

        from neurospatial.encoding._core_jax import sparsity_single

        return sparsity_single(jnp.asarray(firing_rate), jnp.asarray(occupancy))

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
        Shared occupancy for all neurons (time spent in each bin), normalized
        separately over each neuron's finite rate/occupancy pairs.

    Returns
    -------
    ndarray | jax.Array, shape (n_neurons,)
        Sparsity values in range [0, 1] for each neuron, NaN for a silent or
        all-NaN map or zero total occupancy.
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
        import jax.numpy as jnp

        from neurospatial.encoding._core_jax import sparsity_batch

        return sparsity_batch(jnp.asarray(firing_rates), jnp.asarray(occupancy))

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
) -> BatchScoresResult:
    """Compute grid scores for multiple neurons.

    Computes grid score (hexagonal periodicity) for each neuron's firing
    rate map. Grid score quantifies how well the firing pattern exhibits
    the characteristic 6-fold rotational symmetry of grid cells.

    Parameters
    ----------
    env : Environment
        Environment with a regular 2D grid of rectangular cells, as
        :func:`~neurospatial.encoding.grid.spatial_autocorrelation` requires.
    firing_rates : ndarray, shape (n_neurons, n_bins)
        Firing rate maps for each neuron in Hz.

    Returns
    -------
    BatchScoresResult
        Container with two parallel arrays of length ``n_neurons``:

        - ``scores``: grid scores in range [-2, 2]. NaN where the grid
          score is undefined for that map (all-NaN or constant firing, no
          central autocorrelogram peak bounded inside the map) or where a
          caught exception occurred.
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
    NotImplementedError
        If ``env`` has no regular 2D grid of rectangular cells (hexagonal,
        mesh, track or 3D layouts, or uneven cells), exactly as
        :func:`~neurospatial.encoding.grid.spatial_autocorrelation` refuses
        it.
    CartesianOnlyError
        If ``env`` is polar.

    Notes
    -----
    For each neuron, this function:

    1. Computes spatial autocorrelation using :func:`spatial_autocorrelation`
    2. Computes grid score using :func:`grid_score`

    **Interpretation**:

    - **score > 0.4**: Strong hexagonal grid (typical threshold for grid cells)
    - **score ≈ 0**: No hexagonal structure (place cells, non-spatial cells)
    - **score < 0**: Anti-hexagonal structure (rare)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._metrics import batch_grid_scores

    >>> # Create environment
    >>> x = np.linspace(-50, 50, 51)
    >>> xx, yy = np.meshgrid(x, x)
    >>> positions = np.column_stack([xx.ravel(), yy.ravel()])
    >>> env = Environment.from_samples(positions, bin_size=2.0, units="cm")

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

    # The environment either has a rectangular lag grid or has no grid score
    # at all; refuse once, as spatial_autocorrelation does, rather than
    # returning NaN scores that read as "not a grid cell".
    from neurospatial.encoding.grid import _require_rectangular_grid

    _require_rectangular_grid(env, "batch_grid_scores()")

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
            autocorr = spatial_autocorrelation(env, firing_rate)
            scores[i] = grid_score(autocorr)
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
    min_area: float | None = None,
    metric: str = "geodesic",
    boundary_source: BoundarySource = "physical",
) -> BatchScoresResult:
    """Compute border scores for multiple neurons.

    Computes the border score of Solstad et al. (2008) for each neuron's
    firing rate map; see :func:`~neurospatial.encoding.border.border_score`
    for the definition.

    Parameters
    ----------
    env : Environment
        Spatial environment containing bin centers and connectivity.
    firing_rates : ndarray, shape (n_neurons, n_bins)
        Firing rate maps for each neuron in Hz.
    threshold : float, default=0.3
        Fraction of peak firing rate used to segment the field.
        Follows Solstad et al. (2008). Must be in (0, 1).
    min_area : float, optional
        Smallest area of one field, in squared position units (for example
        cm²). Smaller fields are ignored. The default, None, is the 200 cm² of
        Solstad et al. (2008) in the environment's units, or at least 9 bins
        (the BNT/opexebo default) when the units are not a metric length.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        Distance from field bins to the nearest wall.
        - "geodesic": center-to-wall length at the wall plus the shortest path
          through the connectivity graph. Respects obstacles.
        - "euclidean": straight-line distance to the nearest wall.

    boundary_source : {'physical', 'analysis'}, default='physical'
        The measured arena walls, or the explicitly requested outline of the
        retained bins. Missing physical geometry, retained wall cells or finite
        wall firing rates raise NotImplementedError rather than being converted
        to per-unit computation failures.

    Returns
    -------
    BatchScoresResult
        Container with two parallel arrays of length ``n_neurons``:

        - ``scores``: border scores in range [-1, 1]. NaN where border
          score cannot be computed (no positive peak, no field of the
          minimum size, or a caught exception).
        - ``failures``: boolean mask, ``True`` for neurons whose
          ``border_score`` call raised ``ValueError`` or
          ``RuntimeError`` and was caught. A ``UserWarning`` is also
          emitted when at least one neuron fails.

    Raises
    ------
    ValueError
        If firing_rates is not 2D, if firing_rates.shape[1] != env.n_bins,
        or if an argument is invalid.
    NotImplementedError
        If physical wall coverage or observations are unavailable for any unit.

    Notes
    -----
    For each neuron, this function delegates to
    :func:`~neurospatial.encoding.border.border_score`.

    A field along one whole wall scores near 1 and a central field -1. A
    uniform map also scores about 0.5 on a square (its one field covers every
    wall), so combine the score with spatial information or a shuffle test.

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
    >>> env = Environment.from_samples(positions, bin_size=2.0, units="cm")

    >>> # Random firing patterns (5 neurons)
    >>> rng = np.random.default_rng(42)
    >>> firing_rates = rng.random((5, env.n_bins)) * 10
    >>> scores = batch_border_scores(env, firing_rates, boundary_source="analysis")
    >>> scores.shape
    (5,)

    See Also
    --------
    neurospatial.encoding.border.border_score : Single-neuron border score
    """
    from neurospatial.encoding._physical_border import (
        _require_wall_rates,
        border_geometry,
    )
    from neurospatial.encoding.border import _validate_border_arguments, border_score

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

    # Invalid arguments and unavailable walls concern every neuron, so raise
    # them once here instead of recording a per-neuron failure.
    _validate_border_arguments(threshold, min_area, metric, boundary_source)
    geometry = border_geometry(env, boundary_source)
    if boundary_source == "physical":
        for firing_rate in firing_rates:
            _require_wall_rates(geometry, firing_rate)

    n_neurons = firing_rates.shape[0]
    scores = np.empty(n_neurons, dtype=np.float64)
    failures = np.zeros(n_neurons, dtype=np.bool_)

    for i in range(n_neurons):
        firing_rate = firing_rates[i]

        try:
            scores[i] = border_score(
                env,
                firing_rate,
                threshold=threshold,
                min_area=min_area,
                metric=metric,  # type: ignore[arg-type]
                boundary_source=boundary_source,
            )
        except NotImplementedError:
            # Missing scientific support is an unavailable measurement, not a
            # numerical failure that can be hidden in the batch failure mask.
            raise
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
    mutual_information : Mutual information from a joint position-response histogram

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
    joint_counts: ArrayLike,
    occupancy: ArrayLike | None = None,
    *,
    base: float = 2.0,
) -> float:
    r"""Compute mutual information from a joint position-response histogram.

    Parameters
    ----------
    joint_counts : ArrayLike, shape (n_position_bins, n_response_bins)
        Non-negative joint counts, weights, or probabilities. Rows index
        positions and columns index discrete neural responses (for example,
        spike counts in fixed-duration observation windows). The array is
        normalized to sum to one. All entries must be finite, with at least
        one positive entry. Zero-mass rows and columns are allowed.
    occupancy : None, optional
        Retired argument from the former rate-map alias. Supplying it raises
        an error with migration guidance; mean rates and occupancy alone do
        not determine the joint response distribution.
    base : float, default=2.0
        Finite logarithm base greater than one. Use 2.0 for bits or ``np.e``
        for nats.

    Returns
    -------
    float
        Mutual information per observation in bits (or nats if base=e).
        Zero for independent position and response distributions.

    Raises
    ------
    ValueError
        If the histogram is not a nonempty, finite, non-negative 2-D array
        with positive total mass, if ``occupancy`` is supplied, or if the
        logarithm base is invalid.

    Notes
    -----
    The discrete plug-in estimate is

    .. math::

        I(X; R) = \sum_{x,r} p(x,r)
                  \log\frac{p(x,r)}{p(x)p(r)}.

    Zero joint probabilities contribute zero. The estimate is symmetric
    and bounded above by each marginal entropy. Finite-sample bias is not
    corrected. Choose and report the response bins and observation-window
    duration; different choices define different response variables.

    Earlier versions returned Skaggs information rate under this name.
    To retain that rate-map calculation in bits/second, use
    ``information_per_second(firing_rate, occupancy)``. A mean rate map does
    not specify response variability and cannot determine mutual information
    without an additional response model or observed joint histogram.

    Examples
    --------
    >>> mutual_information([[50, 0], [0, 50]])  # Response identifies position
    1.0
    >>> mutual_information([[25, 25], [25, 25]])  # Independent response
    0.0

    See Also
    --------
    spatial_information : Skaggs spatial information (bits/spike).
    information_per_second : Skaggs information rate from a rate map (bits/second).
    """
    migration = (
        "mutual_information requires a 2-D joint position-response histogram. "
        "Mean firing rates and occupancy do not determine mutual information. "
        "For a rate map, use information_per_second(firing_rate, occupancy) "
        "for the Skaggs information rate in bits/second."
    )
    if occupancy is not None:
        raise ValueError(migration)
    joint = np.asarray(joint_counts, dtype=np.float64)
    if joint.ndim != 2 or joint.size == 0:
        raise ValueError(migration)
    if not np.all(np.isfinite(joint)) or np.any(joint < 0):
        raise ValueError("joint_counts must contain only finite, non-negative values.")
    if not np.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than one (2 for bits).")
    scale = float(joint.max())
    if scale == 0:
        raise ValueError("joint_counts must contain positive total mass.")

    # Scaling first avoids overflow when summing large but finite counts.
    joint = joint / scale
    joint /= joint.sum()
    position_prob = joint.sum(axis=1)
    response_prob = joint.sum(axis=0)
    rows, columns = np.nonzero(joint)
    probability = joint[rows, columns]
    # Subtract logs rather than dividing by a possibly underflowing product.
    log_ratio = (
        np.log(probability)
        - np.log(position_prob[rows])
        - np.log(response_prob[columns])
    )
    information = np.sum(probability * log_ratio) / np.log(base)
    return float(max(information, 0.0))


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
        Fraction of observed (finite-rate) bins with firing rate > threshold,
        in range [0, 1]. NaN if every rate is NaN.

    Notes
    -----
    **Formula**:

    .. math::

        C = \\frac{\\sum_i \\mathbb{1}[r_i > \\theta]}{N}

    where :math:`r_i` is firing rate in bin :math:`i`, :math:`\\theta` is
    the threshold, and :math:`N` is the number of bins with a finite rate.

    **Interpretation**:

    - **Coverage = 0.0**: Cell fires nowhere (no place field)
    - **Coverage = 0.1**: Cell fires in 10% of environment (highly selective)
    - **Coverage = 0.5**: Cell fires in half the environment (broad field)
    - **Coverage = 1.0**: Cell fires everywhere (no spatial selectivity)

    **Relationship to other metrics**:

    - Inverse of selectivity: high coverage -> low selectivity
    - Complementary to sparsity: both measure spatial specificity

    **NaN handling**: NaN bins (unvisited, or below the occupancy threshold)
    were not observed. They are excluded from both the count and the total,
    so a cell is not called selective because part of the arena went
    unsampled.

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
    observed = firing_rate[np.isfinite(firing_rate)]
    if observed.size == 0:
        return np.nan
    return float(np.mean(observed > threshold))
