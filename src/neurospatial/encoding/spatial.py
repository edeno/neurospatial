"""Spatial rate computation for place, grid, and border cells.

This module provides result classes and compute functions for spatial firing
rate analysis. The result classes wrap firing rate maps with their metadata
and provide convenient methods for analysis and visualization.

Result Classes
--------------
SpatialRateResult
    Single-neuron spatial rate map with convenience methods
SpatialRatesResult
    Multi-neuron spatial rate maps with batch methods and iteration

Compute Functions
-----------------
compute_spatial_rate
    Compute spatial firing rate for one neuron
compute_spatial_rates
    Compute spatial firing rates for multiple neurons

Examples
--------
>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.encoding.spatial import compute_spatial_rate

>>> # Create environment from a seeded trajectory
>>> rng = np.random.default_rng(0)
>>> positions = rng.uniform(0, 50, (500, 2))
>>> env = Environment.from_samples(positions, bin_size=5.0)

>>> # Compute a single-neuron spatial rate map (returns SpatialRateResult)
>>> times = np.linspace(0, 50, 500)
>>> spike_times = np.sort(rng.uniform(0, 50, 30))
>>> result = compute_spatial_rate(env, spike_times, times, positions, bandwidth=10.0)

>>> # Use inherited mixin methods
>>> peak = result.peak_location()  # (n_dims,) coordinates of peak
>>> peak_rate = result.peak_firing_rate()  # scalar max firing rate
>>> peak.shape
(2,)
"""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._results import ResultMixin
from neurospatial.encoding._base import SpatialResultMixin, _to_numpy
from neurospatial.encoding._metrics import BatchScoresResult

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from neurospatial import Environment
    from neurospatial._typing import PositionLike, SpikeTrainsLike
    from neurospatial.encoding.grid import GridProperties
    from neurospatial.environment._protocols import EnvironmentProtocol

# ruff: noqa: RUF022 - intentionally grouped by category
__all__ = [
    # Result classes
    "SpatialRateResult",
    "SpatialRatesResult",
    "PlaceFieldsResult",
    # Compute functions
    "compute_spatial_rate",
    "compute_spatial_rates",
    # Directional place fields
    "DirectionalPlaceFields",
    "compute_directional_place_fields",
    # Field detection
    "detect_place_fields",
    # Classification predicates
    "is_place_cell",
]


@dataclass(frozen=True, repr=False)
class PlaceFieldsResult(ResultMixin):
    """Result of ``detect_place_fields()``: detected fields plus exclusion metadata.

    Returned by :func:`detect_place_fields`. Distinguishes "this neuron
    has no detectable place fields" (``fields=[]``, ``excluded_reason=None``)
    from "this neuron was excluded by the interneuron-rate filter"
    (``fields=[]``, ``excluded_reason="mean_rate_above_threshold"``) so a
    population pipeline can branch without listening for warnings.

    Attributes
    ----------
    fields : list of NDArray[np.int64], length n_fields
        Each element is a 1-D array of bin indices belonging to one
        place field. Empty list if no fields were detected, or if the
        neuron was excluded by the ``max_mean_rate`` filter.
    excluded_reason : str | None
        ``None`` when the neuron passed all filters and ``fields``
        reflects the actual detection result. A non-None string when a
        filter caused detection to short-circuit. The only value used
        currently is ``"mean_rate_above_threshold"`` (putative interneuron);
        future filters (e.g. ``"all_nan_rate_map"``) may add more.
    n_excluded : int
        ``1`` if ``excluded_reason`` is set, else ``0``. Provided so
        downstream population aggregation can sum exclusions across
        neurons without parsing the reason string.

    Notes
    -----
    The result is iterable and indexable like a ``list[NDArray[np.int64]]``
    so callers that previously wrote ``for f in detect_place_fields(...)``
    or ``len(detect_place_fields(...))`` keep working.
    """

    fields: list[NDArray[np.int64]]
    excluded_reason: str | None = None
    n_excluded: int = 0

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, idx: int) -> NDArray[np.int64]:
        return self.fields[idx]

    def __iter__(self) -> Iterator[NDArray[np.int64]]:
        return iter(self.fields)

    def __bool__(self) -> bool:
        # Truthy iff at least one field was detected. Matches the
        # ergonomic of `if detect_place_fields(...): ...` against the
        # old list[NDArray] return.
        return len(self.fields) > 0

    def summary(self) -> dict[str, Any]:
        """Scalar headline metrics for the detected place fields.

        Returns
        -------
        dict
            Mapping with keys ``n_fields`` (int, number of detected fields),
            ``total_bins`` (int, bins across all fields), ``n_excluded``
            (int), and ``excluded_reason`` (str or ``None``).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.spatial import PlaceFieldsResult
        >>> result = PlaceFieldsResult(fields=[np.array([0, 1, 2])])
        >>> result.summary()["n_fields"]
        1
        >>> result.summary()["total_bins"]
        3
        """
        total_bins = int(sum(len(f) for f in self.fields))
        return {
            "n_fields": len(self.fields),
            "total_bins": total_bins,
            "n_excluded": int(self.n_excluded),
            "excluded_reason": self.excluded_reason,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy/long-form table of field membership: one row per (field, bin).

        Each detected place field contributes one row per member bin, with a
        ``field`` index column. Neurons excluded by a filter (``fields=[]``)
        yield an empty table; ``excluded_reason`` is exposed via
        :meth:`summary`.

        Returns
        -------
        pandas.DataFrame
            Long-form table with columns ``field`` (int, field index) and
            ``bin`` (int, member bin index). Empty (with those columns) when
            no fields were detected.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.spatial import PlaceFieldsResult
        >>> result = PlaceFieldsResult(fields=[np.array([0, 1]), np.array([5])])
        >>> df = result.to_dataframe()
        >>> df["field"].tolist()
        [0, 0, 1]
        >>> df["bin"].tolist()
        [0, 1, 5]
        """
        import pandas as pd

        field_col: list[int] = []
        bin_col: list[int] = []
        for field_idx, field_bins in enumerate(self.fields):
            for b in np.asarray(field_bins).ravel():
                field_col.append(field_idx)
                bin_col.append(int(b))
        return pd.DataFrame(
            {
                "field": np.asarray(field_col, dtype=np.int64),
                "bin": np.asarray(bin_col, dtype=np.int64),
            }
        )


#: Required GAM diagnostics -- non-``None`` for ``method="glm"``, ``None`` for
#: the ratio methods. ``penalty`` / ``reml_objective`` are excluded because they
#: are legitimately ``None`` even for glm (fixed penalty, REML-skip, or no data).
_GAM_REQUIRED_FIELDS = (
    "coefficients",
    "penalty_weights",
    "rank",
    "deviance",
    "converged",
    "n_iter",
)


def _check_gam_result_invariant(
    kind: str, fields: dict[str, Any], *, n_units: int | None
) -> None:
    """Enforce the ``None``-iff-glm invariant on a spatial-rate result.

    The GAM diagnostics travel together: they are all populated for
    ``method="glm"`` and all ``None`` for the ratio methods, and ``bandwidth`` is
    ``None`` exactly for glm. This makes the illegal states the type cannot
    express -- a ratio result carrying a stray GAM array, or a glm result missing
    or mis-shaping a diagnostic -- unrepresentable at construction (the classes
    are frozen and public, so this closes the gap the type alone leaves open).

    Parameters
    ----------
    kind : str
        Class name, for error messages.
    fields : dict
        The result's ``method``, ``bandwidth``, and the eight GAM fields.
    n_units : int or None, keyword-only
        Number of units (``firing_rates.shape[0]``) for the plural class, or
        ``None`` for the singular class (whose GAM fields are a per-unit slice).
        Selects the expected glm array shapes.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or the GAM fields / ``bandwidth`` are
        inconsistent with ``method`` (presence, ``bandwidth``, or shape).
    """
    from neurospatial.encoding._smoothing import _SPATIAL_METHODS

    method = fields["method"]
    if method not in _SPATIAL_METHODS:
        raise ValueError(
            f"{kind} has method={method!r}, which is not a known estimator; "
            f"expected one of {set(_SPATIAL_METHODS)}."
        )

    is_glm = method == "glm"
    required = {name: fields[name] for name in _GAM_REQUIRED_FIELDS}
    # ``penalty`` / ``reml_objective`` / ``reml_at_boundary`` /
    # ``penalty_selected_by_reml`` are legitimately ``None`` even for glm (fixed
    # penalty, REML-skip, no data, or pooled=True), so they are "optional".
    optional = {
        "penalty": fields["penalty"],
        "reml_objective": fields["reml_objective"],
        "reml_at_boundary": fields["reml_at_boundary"],
        "penalty_selected_by_reml": fields["penalty_selected_by_reml"],
    }
    if is_glm:
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"{kind} with method='glm' is missing required GAM field(s): "
                f"{missing}. glm results carry all of {list(_GAM_REQUIRED_FIELDS)}."
            )
        if fields["bandwidth"] is not None:
            raise ValueError(
                f"{kind} with method='glm' must have bandwidth=None (glm has no "
                f"bandwidth); got {fields['bandwidth']!r}."
            )
        # Full shape coupling: a mis-shaped diagnostic (e.g. coefficients
        # (rank, 1) on a 2-unit result) would construct and only IndexError when a
        # unit is later indexed. Pin every GAM array shape here. The singular
        # class holds a per-unit slice; the plural holds the population arrays.
        rank = int(fields["rank"])
        if n_units is None:  # singular: per-unit slice
            expected = {
                "coefficients": (rank,),
                "penalty_weights": (rank,),
                "deviance": (),  # scalar
            }
        else:  # plural: population arrays
            expected = {
                "coefficients": (rank, n_units),
                "penalty_weights": (rank,),
                "deviance": (n_units,),
            }
        for name, want in expected.items():
            got = tuple(np.shape(fields[name]))
            if got != want:
                raise ValueError(
                    f"{kind} glm field {name!r} has shape {got} but expected "
                    f"{want} (rank={rank}"
                    + (f", n_units={n_units}" if n_units is not None else "")
                    + ")."
                )
        # Per-unit fields (pooled=False) are scalar on the singular slice and
        # either scalar or a ``(n_units,)`` vector on the plural class; a stray
        # extra axis is a construction bug (indexing / summary_table rely on it).
        for name in (
            "penalty",
            "reml_objective",
            "reml_at_boundary",
            "penalty_selected_by_reml",
        ):
            value = fields[name]
            if value is None:
                continue
            ndim = np.ndim(value)
            if n_units is None:  # singular: must be a scalar
                if ndim != 0:
                    raise ValueError(
                        f"{kind} (single-unit) glm field {name!r} must be a scalar; "
                        f"got shape {tuple(np.shape(value))}."
                    )
            elif ndim not in (0, 1) or (
                ndim == 1 and tuple(np.shape(value)) != (n_units,)
            ):  # plural: scalar or (n_units,)
                raise ValueError(
                    f"{kind} glm field {name!r} must be a scalar or a ({n_units},) "
                    f"vector; got shape {tuple(np.shape(value))}."
                )
        # State-machine coupling: the per-unit-lambda fields travel together, and
        # NWB persistence keys "is this per-unit?" off the provenance mask alone.
        # So the mask's presence must agree with the vector-ness of the other
        # three (and with pooled=False) -- otherwise an inconsistent result (e.g.
        # pooled=True with a vector penalty and no mask) constructs cleanly and
        # only fails deep in the NWB writer. per-unit here = a per-unit-shaped
        # slot: a ``(n_units,)`` vector on the plural class, a scalar on the
        # singular slice.
        mask = fields["penalty_selected_by_reml"]
        per_unit_slots = ("penalty", "reml_objective", "reml_at_boundary")

        def _is_per_unit_shaped(value: Any) -> bool:
            if value is None:
                return False
            return np.ndim(value) == (0 if n_units is None else 1)

        if mask is not None:
            # Per-unit result: only under pooled=False, with all three per-unit
            # slots populated and per-unit-shaped.
            if fields["pooled"] is not False:
                raise ValueError(
                    f"{kind} carries penalty_selected_by_reml (a per-unit result) "
                    f"but pooled={fields['pooled']!r}; per-unit lambda is only "
                    "produced under pooled=False."
                )
            missing_vec = [
                n for n in per_unit_slots if not _is_per_unit_shaped(fields[n])
            ]
            if missing_vec:
                raise ValueError(
                    f"{kind} carries penalty_selected_by_reml (a per-unit result) "
                    f"but {missing_vec} are not per-unit values; penalty, "
                    "reml_objective, and reml_at_boundary must all be per-unit "
                    "when the provenance mask is present."
                )
        else:
            # Non-per-unit result: none of the three may be a per-unit vector
            # (this is exactly the state that crashes NWB writing).
            stray_vec = [
                n
                for n in per_unit_slots
                if fields[n] is not None and np.ndim(fields[n]) != 0
            ]
            if stray_vec:
                raise ValueError(
                    f"{kind} has vector {stray_vec} but no penalty_selected_by_reml "
                    "mask; per-unit penalty / reml_objective / reml_at_boundary "
                    "require the provenance mask (a per-unit pooled=False result). "
                    "For a shared/fixed lambda these must be scalar or None."
                )
        # The boolean diagnostics must be boolean (a float boundary flag or an
        # integer provenance mask would round-trip wrong through NWB).
        for name in ("reml_at_boundary", "penalty_selected_by_reml"):
            value = fields[name]
            if value is not None and np.asarray(value).dtype != np.bool_:
                raise ValueError(
                    f"{kind} glm field {name!r} must be boolean; got dtype "
                    f"{np.asarray(value).dtype}."
                )
        # ``pooled`` is a concrete bool for every glm result (the only reliable
        # NWB source, since scalar cases are value-identical under both settings).
        # Checked last so a more fundamental shape/required error surfaces first.
        if not isinstance(fields["pooled"], bool):
            raise ValueError(
                f"{kind} with method='glm' must carry pooled=True/False; got "
                f"{fields['pooled']!r}."
            )
    else:
        stray = [
            name
            for name, value in {**required, **optional}.items()
            if value is not None
        ]
        if fields["pooled"] is not None:
            stray.append("pooled")
        if stray:
            raise ValueError(
                f"{kind} with ratio method={fields['method']!r} must not carry GAM "
                f"field(s): {stray}. GAM diagnostics belong only to method='glm'."
            )
        if fields["bandwidth"] is None:
            raise ValueError(
                f"{kind} with ratio method={fields['method']!r} must have a float "
                "bandwidth; got None (bandwidth=None is reserved for method='glm')."
            )


def _index_per_unit(value: Any, idx: int) -> Any:
    """Slice a possibly-per-unit GAM field to a single unit (for indexing).

    ``None`` and shared **scalars** carry through unchanged (a shared ``lambda``
    or ``None`` is the same for every unit); a per-unit ``(n_units,)`` vector
    (``pooled=False``) is sliced to its ``idx`` element as a Python scalar. Keeps
    ``rates[i]`` field-for-field equal to the singular ``compute_spatial_rate``.
    """
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:  # shared scalar -> preserve the original Python value/type
        return value
    return arr[idx].item()


@dataclass(frozen=True, repr=False)
class SpatialRateResult(SpatialResultMixin):
    """Result of spatial rate computation for a single neuron.

    This class wraps a spatial firing rate map with its associated metadata
    (occupancy, environment, smoothing parameters). It inherits from
    `SpatialResultMixin` for common methods like `peak_location()` and
    `peak_firing_rate()`.

    Parameters
    ----------
    firing_rate : ArrayLike
        Firing rate map in Hz. Shape is (n_bins,) where n_bins is the
        number of active bins in the environment. Can contain NaN for
        bins with insufficient occupancy.
    occupancy : ArrayLike
        Time spent in each bin in seconds. Shape is (n_bins,).
    env : Environment
        The spatial environment used for the computation. Provides bin
        centers, connectivity, and plotting methods.
    method : str
        Estimator used: "diffusion_kde", "gaussian_kde", "binned", or "glm".
    bandwidth : float or None
        Smoothing bandwidth in the same units as the environment's bin_size, or
        ``None`` for ``method="glm"`` (which has no bandwidth).

    Attributes
    ----------
    firing_rate : ArrayLike
        Firing rate map in Hz. Shape is (n_bins,).
    occupancy : ArrayLike
        Time spent in each bin in seconds. Shape is (n_bins,).
    env : Environment
        The spatial environment.
    method : str
        Estimator used.
    bandwidth : float or None
        Smoothing bandwidth (``None`` for ``method="glm"``).
    unit_id : int or str or None
        Identifier for this unit. Set automatically when indexing/iterating a
        population result (``rates[i].unit_id == rates.unit_ids[i]``); ``None``
        for a standalone single-unit computation.
    coefficients, penalty, penalty_weights, rank, deviance, converged, n_iter, \
reml_objective, reml_at_boundary, penalty_selected_by_reml, pooled
        GAM diagnostics for ``method="glm"`` (all ``None`` for the ratio
        methods): the per-unit slice of a population fit -- ``coefficients``
        ``(rank,)``, scalar ``deviance``, and the shared/batch-scalar
        ``penalty`` / ``penalty_weights`` ``(rank,)`` / ``rank`` / ``converged``
        / ``n_iter`` / ``reml_objective``. Under a per-unit fit (``pooled=False``)
        ``penalty`` / ``reml_objective`` / ``reml_at_boundary`` are this unit's
        own scalar values and ``penalty_selected_by_reml`` its provenance
        (``False`` = pooled-``λ`` fallback zero-spike unit); ``pooled`` records
        the flag (``None`` for ratio results). See :class:`SpatialRatesResult`.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    Inherits from `SpatialResultMixin`, which provides:

    - `peak_location()`: Returns (n_dims,) coordinates of peak firing
    - `peak_firing_rate()`: Returns scalar max firing rate

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_spatial_rate

    >>> # Create a simple environment from a seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 50, (500, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Compute result (returns SpatialRateResult)
    >>> times = np.linspace(0, 50, 500)
    >>> spike_times = np.sort(rng.uniform(0, 50, 30))
    >>> result = compute_spatial_rate(
    ...     env, spike_times, times, positions, bandwidth=10.0
    ... )

    >>> # Access fields
    >>> result.firing_rate.shape == (env.n_bins,)
    True
    >>> result.method
    'diffusion_kde'

    >>> # Use mixin methods
    >>> peak_coords = result.peak_location()  # (n_dims,)
    >>> max_rate = result.peak_firing_rate()  # float

    See Also
    --------
    SpatialRatesResult : Batch version for multiple neurons
    compute_spatial_rate : Function to compute this result
    SpatialResultMixin : Provides peak_location() and peak_firing_rate()
    """

    firing_rate: ArrayLike
    occupancy: ArrayLike
    env: Environment
    method: str
    bandwidth: float | None
    unit_id: int | str | None = None
    # GAM (``method="glm"``) fields -- all ``None`` for the ratio methods. See
    # ``SpatialRatesResult`` for the population-level shapes; the singular fields
    # are the per-unit slices stamped when indexing a population result.
    coefficients: NDArray[np.float64] | None = None
    penalty: float | NDArray[np.float64] | None = None
    penalty_weights: NDArray[np.float64] | None = None
    rank: int | None = None
    deviance: float | NDArray[np.float64] | None = None
    converged: bool | None = None
    n_iter: int | None = None
    reml_objective: float | None = None
    # Per-unit-lambda (``pooled=False``) diagnostics: the per-unit slice of the
    # population fit (scalars here). ``None`` for ratio methods. For glm:
    # ``reml_at_boundary`` is a scalar bool when REML ran (both pooled settings)
    # and ``None`` when it did not (fixed penalty / r==0 / no data);
    # ``penalty_selected_by_reml`` is ``None`` except for a per-unit
    # (``pooled=False``) slice; ``pooled`` is ``True`` / ``False`` for glm.
    reml_at_boundary: bool | None = None
    penalty_selected_by_reml: bool | None = None
    pooled: bool | None = None

    def __post_init__(self) -> None:
        # Enforce the None-iff-glm invariant: the GAM diagnostics are all present
        # (and correctly per-unit-shaped) for method="glm" with bandwidth=None, or
        # all absent for a ratio method. n_units=None -> the singular per-unit slice.
        _check_gam_result_invariant(
            "SpatialRateResult",
            {
                "method": self.method,
                "bandwidth": self.bandwidth,
                "coefficients": self.coefficients,
                "penalty": self.penalty,
                "penalty_weights": self.penalty_weights,
                "rank": self.rank,
                "deviance": self.deviance,
                "converged": self.converged,
                "n_iter": self.n_iter,
                "reml_objective": self.reml_objective,
                "reml_at_boundary": self.reml_at_boundary,
                "penalty_selected_by_reml": self.penalty_selected_by_reml,
                "pooled": self.pooled,
            },
            n_units=None,
        )

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the spatial rate map.

        Delegates to the environment's plot_field method for consistent
        visualization across the codebase.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        **kwargs
            Additional keyword arguments passed to env.plot_field().
            Common options include:
            - cmap : str or Colormap, default="viridis"
            - vmin, vmax : float, colorbar limits
            - colorbar : bool, default=True
            - colorbar_label : str, default="Firing Rate (Hz)"

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")  # non-interactive backend for doctest
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> ax = result.plot()
        >>> type(ax).__name__
        'Axes'
        """
        kwargs.setdefault("colorbar_label", "Firing Rate (Hz)")
        return self.env.plot_field(_to_numpy(self.firing_rate), ax=ax, **kwargs)

    def spatial_information(self) -> float | Any:
        """Skaggs spatial information (bits per spike).

        Quantifies how much information each spike conveys about the
        animal's spatial location. Higher values indicate more spatially
        selective firing.

        Returns
        -------
        float | jax.Array
            Spatial information in bits/spike. Always non-negative.
            Returns 0.0 for uniform firing.

            **Backend-aware**: Returns float for NumPy input,
            JAX scalar for JAX input.

        Notes
        -----
        Uses the Skaggs et al. (1993) formula:

        .. math::

            I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log_2 \\left( \\frac{r_i}{\\bar{r}} \\right)

        **Interpretation**:

        - Place cells typically have 1-3 bits/spike
        - Higher values indicate more spatially selective firing
        - Zero means uniform firing (no spatial selectivity)

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
               An information-theoretic approach to deciphering the hippocampal code.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> info = result.spatial_information()
        >>> bool(info >= 0.0)
        True

        See Also
        --------
        neurospatial.encoding._metrics.spatial_information : Underlying computation
        """
        from neurospatial.encoding._metrics import spatial_information

        # Pass arrays directly - _metrics.py handles JAX dispatch
        return spatial_information(self.firing_rate, self.occupancy)

    def sparsity(self) -> float | Any:
        """Sparsity of spatial firing.

        Measures what fraction of the environment elicits significant
        firing. Lower values indicate sparser, more selective place fields.

        Returns
        -------
        float | jax.Array
            Sparsity value in range [0, 1].
            - Low (0.1-0.3): Sparse, selective place field
            - High (~1.0): Uniform firing throughout environment

            **Backend-aware**: Returns float for NumPy input,
            JAX scalar for JAX input.

        Notes
        -----
        Uses the Skaggs et al. (1996) formula:

        .. math::

            S = \\frac{\\left( \\sum_i p_i r_i \\right)^2}{\\sum_i p_i r_i^2}

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., et al. (1996). Theta phase
               precession in hippocampal neuronal populations.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> spars = result.sparsity()
        >>> bool(0.0 <= spars <= 1.0)
        True

        See Also
        --------
        neurospatial.encoding._metrics.sparsity : Underlying computation
        """
        from neurospatial.encoding._metrics import sparsity

        # Pass arrays directly - _metrics.py handles JAX dispatch
        return sparsity(self.firing_rate, self.occupancy)

    def grid_score(self) -> float:
        """Grid score (hexagonal periodicity).

        Quantifies the hexagonal periodicity of the firing rate map, which
        is characteristic of grid cells. Higher values indicate stronger
        hexagonal grid patterns.

        Returns
        -------
        float
            Grid score in range [-2, 2].
            - score > 0.4: Strong hexagonal grid (typical threshold)
            - score ≈ 0: No hexagonal structure
            - score < 0: Anti-hexagonal structure (rare)
            Returns NaN if grid score cannot be computed (e.g., non-2D grid).

        Notes
        -----
        Computes the spatial autocorrelation of the firing rate map and
        extracts the grid score based on rotational symmetry.

        Uses the Sargolini et al. (2006) algorithm:

        1. Compute 2D spatial autocorrelation via FFT
        2. Rotate by 30°, 60°, 90°, 120°, 150°
        3. Grid score = min(r60, r120) - max(r30, r90, r150)

        This method delegates to ``neurospatial.encoding.grid.grid_score()``.

        References
        ----------
        .. [1] Sargolini, F., Fyhn, M., et al. (2006). Conjunctive
               representation of position, direction, and velocity in
               entorhinal cortex. Science, 312(5774), 758-762.

        See Also
        --------
        grid_properties : Full grid cell metrics (score, scale, orientation)

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> score = result.grid_score()
        >>> bool(-2.0 <= score <= 2.0)
        True
        """
        from neurospatial.encoding.grid import grid_score as gs_func
        from neurospatial.encoding.grid import spatial_autocorrelation

        firing_rate = _to_numpy(self.firing_rate)

        try:
            autocorr = spatial_autocorrelation(self.env, firing_rate)
            return gs_func(autocorr)
        except (ValueError, RuntimeError):
            # Irregular env, constant firing, or all-NaN: grid_score is
            # undefined. Return NaN; callers using batch_grid_scores can
            # see the same NaN with the failures mask separating
            # legitimate-NaN from caught failures.
            return np.nan

    def grid_properties(self) -> GridProperties:
        """Full grid cell metrics (score, scale, orientation).

        Returns a comprehensive set of grid cell metrics computed from the
        spatial autocorrelation of the firing rate map.

        Returns
        -------
        GridProperties
            Dataclass containing:

            - score : float
                Grid score in range [-2, 2]
            - scale : float
                Grid spacing in physical units (same as bin_size)
            - orientation : float
                Grid orientation in degrees [0, 60)
            - orientation_std : float
                Standard deviation of orientation estimates
            - peak_coords : NDArray
                Detected peak coordinates (n_peaks, 2)
            - n_peaks : int
                Number of peaks detected

        Notes
        -----
        This method is more efficient than calling ``grid_score()`` separately
        when you need multiple grid metrics, as it performs peak detection
        only once.

        Delegates to ``neurospatial.encoding.grid.grid_properties()``.

        References
        ----------
        .. [1] Sargolini, F., Fyhn, M., et al. (2006). Conjunctive
               representation of position, direction, and velocity in
               entorhinal cortex. Science, 312(5774), 758-762.
        .. [2] Hafting, T., Fyhn, M., et al. (2005). Microstructure of a
               spatial map in the entorhinal cortex. Nature, 436(7052), 801-806.

        See Also
        --------
        grid_score : Just the grid score
        neurospatial.encoding.grid.GridProperties : Return type details

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> props = result.grid_properties()
        >>> type(props).__name__
        'GridProperties'
        """
        from neurospatial.encoding.grid import grid_properties as gp_func
        from neurospatial.encoding.grid import spatial_autocorrelation

        firing_rate = _to_numpy(self.firing_rate)
        autocorr = spatial_autocorrelation(self.env, firing_rate)
        # Use minimum bin size for grid properties (typically same for isotropic grids)
        bin_size = float(np.min(self.env.bin_sizes))
        return gp_func(autocorr, bin_size=bin_size)

    def border_score(
        self,
        threshold: float = 0.3,
        min_area: float = 0.0,
        metric: Literal["geodesic", "euclidean"] = "geodesic",
    ) -> float:
        """Border score (boundary proximity tuning).

        Quantifies how much the cell's firing field is aligned with
        environmental boundaries (walls). Higher values indicate stronger
        border cell properties.

        Parameters
        ----------
        threshold : float, default 0.3
            Fraction of peak firing rate used to segment the field.
            Bins with firing rate >= threshold * peak are included in field.
        min_area : float, default 0.0
            Minimum field area in physical units (e.g., cm²). Fields smaller
            than this return NaN. Default 0.0 (no filtering). For rat
            hippocampal data, Solstad et al. (2008) used 200 cm².
        metric : {'geodesic', 'euclidean'}, default 'geodesic'
            Distance metric for computing distance from field to boundaries.
            - 'geodesic': Graph shortest path distance (respects obstacles)
            - 'euclidean': Straight-line distance in physical space

        Returns
        -------
        float
            Border score in range [-1, 1].
            - +1: Perfect border cell (field on boundary)
            - 0: No boundary preference
            - -1: Anti-border (field in center)
            Returns NaN if border score cannot be computed.

        Notes
        -----
        Uses the Solstad et al. (2008) algorithm:

        1. Segment field at threshold * peak
        2. Compute boundary coverage (fraction of boundary in field)
        3. Compute normalized distance from field to boundary
        4. Border score = (coverage - distance) / (coverage + distance)

        Delegates to ``neurospatial.encoding.border.border_score()``.

        References
        ----------
        .. [1] Solstad, T., Boccara, C. N., et al. (2008). Representation of
               geometric borders in the entorhinal cortex. Science, 322(5909),
               1865-1868.

        See Also
        --------
        region_coverage : Coverage of specific spatial regions

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> score = result.border_score()
        >>> bool(-1.0 <= score <= 1.0)
        True
        """
        from neurospatial.encoding.border import border_score as bs_func

        firing_rate = _to_numpy(self.firing_rate)
        # Cast to EnvironmentProtocol for type checker (Environment implements it)
        env = cast("EnvironmentProtocol", self.env)
        return bs_func(
            env,
            firing_rate,
            threshold=threshold,
            min_area=min_area,
            metric=metric,
        )

    def region_coverage(
        self,
        threshold: float = 0.3,
        regions: list[str] | None = None,
    ) -> dict[str, float]:
        """Coverage of each spatial region by the firing field.

        Computes what fraction of each region's bins are covered by the
        firing field (bins where firing_rate >= threshold * peak).

        Parameters
        ----------
        threshold : float, default 0.3
            Fraction of peak firing rate used to define the field.
            Bins with firing rate >= threshold * peak are included.
        regions : list of str, optional
            Region names to analyze. If None, analyzes all regions
            defined in env.regions.

        Returns
        -------
        dict[str, float]
            Mapping from region name to coverage fraction [0, 1].
            Coverage = (region bins in field) / (total region bins).

        Notes
        -----
        This method is useful for:

        - **Border cell analysis**: Determine which wall a border cell prefers
        - **Task-relevant regions**: Check if fields overlap with reward zones
        - **Multi-zone analysis**: Quantify field distribution across zones

        Delegates to ``neurospatial.encoding.border.compute_region_coverage()``.

        See Also
        --------
        border_score : Overall border preference score
        neurospatial.encoding.border.compute_region_coverage : Direct call

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> coverage = result.region_coverage()  # no regions defined -> {}
        >>> isinstance(coverage, dict)
        True
        """
        from neurospatial.encoding.border import compute_region_coverage

        firing_rate = _to_numpy(self.firing_rate)

        # Threshold field at fraction of peak
        peak_rate = np.nanmax(firing_rate)
        if peak_rate == 0 or np.isnan(peak_rate):
            # No firing, return zero coverage for all regions
            if regions is None:
                regions = list(self.env.regions.keys())
            return dict.fromkeys(regions, 0.0)

        field_mask = firing_rate >= threshold * peak_rate
        field_bins = np.where(field_mask)[0]

        # Cast to EnvironmentProtocol for type checker (Environment implements it)
        env = cast("EnvironmentProtocol", self.env)
        return compute_region_coverage(field_bins, env, regions=regions)

    def is_place_cell(
        self,
        *,
        threshold: float = 0.2,
        min_size: int | None = None,
        max_mean_rate: float = 10.0,
        detect_subfields: bool = True,
    ) -> bool:
        """Classify as a place cell based on detected place fields.

        A neuron is classified as a place cell if :func:`detect_place_fields`
        finds at least one place field in its firing rate map. This is the
        single-neuron place predicate, the place-cell sibling of
        :meth:`is_object_vector_cell` and :meth:`is_spatial_view_cell`.

        .. note::
           This single-neuron predicate uses **place-field detection**,
           whereas the batch :meth:`SpatialRatesResult.classify` uses a
           **spatial-information threshold**. The two criteria can disagree
           for the same neuron, so this is not guaranteed to equal
           ``rates.classify()[i]``. Pick the criterion that suits your
           analysis and apply it consistently.

        Parameters
        ----------
        threshold : float, default=0.2
            Fraction of peak rate for field boundary detection (0-1).
        min_size : int, optional
            Minimum number of bins for a valid field. If None, defaults to 9.
        max_mean_rate : float, default=10.0
            Maximum mean firing rate (Hz). Neurons exceeding this are excluded
            as putative interneurons.
        detect_subfields : bool, default=True
            If True, recursively detect subfields within large fields.

        Returns
        -------
        bool
            True if the neuron has at least one detected place field.

        See Also
        --------
        detect_place_fields : Place field detection algorithm this agrees with
        is_place_cell : Free-function convenience wrapper

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> isinstance(result.is_place_cell(), bool)
        True
        """
        firing_rate = _to_numpy(self.firing_rate)
        fields = detect_place_fields(
            self.env,
            firing_rate,
            threshold=threshold,
            min_size=min_size,
            max_mean_rate=max_mean_rate,
            detect_subfields=detect_subfields,
        )
        return len(fields) > 0


@dataclass(frozen=True, repr=False)
class SpatialRatesResult(SpatialResultMixin):
    """Result of spatial rate computation for multiple neurons.

    This class wraps spatial firing rate maps for a population of neurons
    with shared occupancy and environment. It inherits from `SpatialResultMixin`
    for common methods and provides iteration over individual neuron results.

    Parameters
    ----------
    firing_rates : ArrayLike
        Firing rate maps in Hz. Shape is (n_neurons, n_bins) where n_bins
        is the number of active bins in the environment. Each row is a
        single neuron's rate map.
    occupancy : ArrayLike
        Time spent in each bin in seconds. Shape is (n_bins,). Shared
        across all neurons.
    env : Environment
        The spatial environment used for the computation.
    method : str
        Estimator used: "diffusion_kde", "gaussian_kde", "binned", or "glm".
    bandwidth : float or None
        Smoothing bandwidth in the same units as the environment's bin_size, or
        ``None`` for ``method="glm"`` (which has no bandwidth).

    Attributes
    ----------
    firing_rates : ArrayLike
        Firing rate maps in Hz. Shape is (n_neurons, n_bins).
    occupancy : ArrayLike
        Time spent in each bin in seconds. Shape is (n_bins,).
    env : Environment
        The spatial environment.
    method : str
        Estimator used.
    bandwidth : float or None
        Smoothing bandwidth (``None`` for ``method="glm"``).
    coefficients : NDArray or None
        (``method="glm"`` only; ``None`` for ratio methods.) Fitted GAM
        coefficients ``gamma`` on the live basis, shape ``(rank, n_units)``. Like
        every GLM diagnostic here, this is the ``float64`` fit result -- the
        ``dtype`` argument governs only the ``firing_rates`` storage, not the
        diagnostics.
    penalty : float or None
        Smoothness penalty ``lambda`` actually applied (scalar for the shared-λ
        fit; ``None`` for the REML-skip and no-data cases, or for ratio methods).
    penalty_weights : NDArray or None
        Basis penalty weights ``d``, shape ``(rank,)`` (``None`` for ratio).
    rank : int or None
        Effective basis rank ``r_eff`` (``None`` for ratio methods).
    deviance : NDArray or None
        Per-unit unpenalized Poisson deviance, shape ``(n_units,)`` (``None`` for
        ratio methods).
    converged : bool or None
        Batch-level convergence flag (``None`` for ratio methods). ``False`` on
        nonconvergence -- a line-search failure, the Newton iteration cap, or
        out-of-domain data (empirical rate above ``exp(30)``).
    n_iter : int or None
        Batch-level Newton iteration count (``None`` for ratio methods).
    reml_objective : float or NDArray or None
        Minimized REML objective, or ``None`` when REML did not run (a fixed
        ``penalty`` was supplied, the basis has no penalized modes, or the data
        was all-zero) -- and ``None`` for ratio methods. Under a per-unit fit
        (``pooled=False``) it is a ``(n_units,)`` vector, ``nan`` for a zero-spike
        fallback unit.
    reml_at_boundary : bool or NDArray or None
        Whether the selected ``λ`` sits on a REML search bound (weakly identified
        ``λ``, though the fitted field is stable). Scalar for the shared
        (``pooled=True``) fit, a ``(n_units,)`` bool vector for the per-unit
        (``pooled=False``) fit, and ``None`` when REML did not run or for ratio
        methods.
    penalty_selected_by_reml : NDArray or None
        Per-unit provenance mask for ``pooled=False`` only: ``True`` where
        ``λ_k`` is the unit's own REML minimum, ``False`` for a zero-spike unit
        carrying the pooled-``λ`` fallback. ``None`` under ``pooled=True`` and for
        ratio methods.
    pooled : bool or None
        The smoothing-pool flag actually applied: ``True`` (shared ``λ``) /
        ``False`` (per-unit ``λ``) for glm, ``None`` for ratio methods. Persisted
        because scalar-output cases are value-identical under both settings, so it
        cannot be reconstructed from the values (the only reliable NWB source).
    unit_ids : NDArray, shape (n_units,)
        Identifier for each unit (row), e.g. from ``read_units`` or passed via
        ``unit_ids=``. Defaults to ``np.arange(n_units)``. Carried into
        indexed/iterated single-unit results and into xarray exports.
    unit_table : pandas.DataFrame or None
        Optional per-unit metadata aligned to ``unit_ids`` (e.g. region,
        quality, depth, inclusion flags), one row per unit; ``None`` when not
        provided. Rides alongside the rates for downstream filtering/grouping.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **Iteration Support**:

    This class supports len(), indexing, and iteration:

    - `len(result)`: Number of neurons
    - `result[i]`: Returns `SpatialRateResult` for neuron i
    - `for r in result`: Iterates over single-neuron results

    **Inherited Methods from SpatialResultMixin**:

    - `peak_location()`: Returns (n_neurons, n_dims) coordinates of peaks
    - `peak_firing_rate()`: Returns (n_neurons,) max firing rates

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_spatial_rates

    >>> # Create environment from a seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 50, (500, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Compute batch result for 3 neurons (returns SpatialRatesResult)
    >>> times = np.linspace(0, 50, 500)
    >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
    >>> result = compute_spatial_rates(
    ...     env, spike_times, times, positions, bandwidth=10.0
    ... )

    >>> # Access fields
    >>> result.firing_rates.shape == (3, env.n_bins)
    True
    >>> len(result)
    3

    >>> # Index to get single-neuron result
    >>> single = result[0]
    >>> type(single).__name__
    'SpatialRateResult'

    >>> # Iterate over neurons
    >>> peaks = [round(float(r.peak_firing_rate()), 2) for r in result]
    >>> len(peaks)
    3

    >>> # Use mixin methods (batch)
    >>> peak_coords = result.peak_location()  # (n_neurons, n_dims)
    >>> max_rates = result.peak_firing_rate()  # (n_neurons,)
    >>> peak_coords.shape
    (3, 2)

    See Also
    --------
    SpatialRateResult : Single-neuron version
    compute_spatial_rates : Function to compute this result
    SpatialResultMixin : Provides peak_location() and peak_firing_rate()
    """

    firing_rates: ArrayLike
    occupancy: ArrayLike
    env: Environment
    method: str
    bandwidth: float | None
    unit_ids: NDArray[Any] | Sequence[Any] | None = field(default=None, compare=False)
    unit_table: pd.DataFrame | None = field(default=None, compare=False)
    # GAM (``method="glm"``) fields -- all ``None`` for the ratio methods,
    # populated for ``method="glm"``. Shapes (``rank == r_eff`` effective basis
    # rank, ``n_units`` neurons):
    #   coefficients      (rank, n_units)   gamma on the live basis
    #   penalty           scalar or None    lambda actually applied (None = REML skip / no data)
    #   penalty_weights   (rank,)           basis penalty weights d
    #   rank              int               effective rank r_eff
    #   deviance          (n_units,)        per-unit Poisson deviance
    #   converged         bool              batch-level convergence flag
    #   n_iter            int               batch-level Newton iterations
    #   reml_objective    scalar, (n_units,), or None   minimized REML objective
    #   reml_at_boundary  scalar bool, (n_units,) bool, or None  weak-lambda flag
    #   penalty_selected_by_reml  (n_units,) bool or None  pooled=False provenance
    #   pooled            bool or None      shared vs per-unit lambda (None ratio)
    coefficients: NDArray[np.float64] | None = None
    penalty: float | NDArray[np.float64] | None = None
    penalty_weights: NDArray[np.float64] | None = None
    rank: int | None = None
    deviance: NDArray[np.float64] | None = None
    converged: bool | None = None
    n_iter: int | None = None
    reml_objective: float | NDArray[np.float64] | None = None
    # Per-unit-lambda (``pooled=False``) diagnostics. ``penalty`` /
    # ``reml_objective`` / ``reml_at_boundary`` widen to ``(n_units,)`` vectors on
    # the per-unit automatic-REML path (informative units their ``lambda_k``,
    # zero-spike units the pooled fallback with ``reml_objective=nan``);
    # ``penalty_selected_by_reml`` is the ``(n_units,)`` bool provenance mask
    # (``True`` informative, ``False`` fallback), ``None`` otherwise.
    reml_at_boundary: bool | NDArray[np.bool_] | None = None
    penalty_selected_by_reml: NDArray[np.bool_] | None = None
    pooled: bool | None = None

    def __post_init__(self) -> None:
        from neurospatial._results import resolve_unit_ids, validate_unit_table

        n_units = int(np.asarray(self.firing_rates).shape[0])
        # Enforce the None-iff-glm invariant (GAM diagnostics travel together):
        # populated and correctly (rank, n_units)/(rank,)/(n_units,)-shaped for
        # method="glm" (bandwidth=None), all absent for a ratio method.
        _check_gam_result_invariant(
            "SpatialRatesResult",
            {
                "method": self.method,
                "bandwidth": self.bandwidth,
                "coefficients": self.coefficients,
                "penalty": self.penalty,
                "penalty_weights": self.penalty_weights,
                "rank": self.rank,
                "deviance": self.deviance,
                "converged": self.converged,
                "n_iter": self.n_iter,
                "reml_objective": self.reml_objective,
                "reml_at_boundary": self.reml_at_boundary,
                "penalty_selected_by_reml": self.penalty_selected_by_reml,
                "pooled": self.pooled,
            },
            n_units=n_units,
        )

        object.__setattr__(
            self,
            "unit_ids",
            resolve_unit_ids(self.unit_ids, n_units),
        )
        validate_unit_table(self.unit_table, n_units, context="SpatialRatesResult")

    def __len__(self) -> int:
        """Return number of units.

        Returns
        -------
        int
            Number of neurons (first dimension of firing_rates).
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        return int(rates.shape[0])

    def __getitem__(self, idx: int) -> SpatialRateResult:
        """Get single-neuron result by index.

        Parameters
        ----------
        idx : int
            Neuron index.

        Returns
        -------
        SpatialRateResult
            Result for the specified neuron with shared occupancy,
            environment, and smoothing parameters.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> single = result[0]
        >>> single.firing_rate.shape == (env.n_bins,)
        True
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        # Slice the per-unit GAM fields; shared / batch-scalar fields carry
        # through unchanged. All ``None`` for ratio results (the ``is not None``
        # guards keep them ``None``).
        coefficients = (
            None if self.coefficients is None else np.asarray(self.coefficients)[:, idx]
        )
        deviance = None if self.deviance is None else np.asarray(self.deviance)[idx]
        # ``penalty`` / ``reml_objective`` / ``reml_at_boundary`` /
        # ``penalty_selected_by_reml`` slice per-unit when they are per-unit
        # vectors (pooled=False); a shared scalar or ``None`` carries through.
        return SpatialRateResult(
            firing_rate=rates[idx],
            occupancy=self.occupancy,
            env=self.env,
            method=self.method,
            bandwidth=self.bandwidth,
            unit_id=np.asarray(self.unit_ids)[idx].item(),
            coefficients=coefficients,
            penalty=_index_per_unit(self.penalty, idx),
            penalty_weights=self.penalty_weights,
            rank=self.rank,
            deviance=deviance,
            converged=self.converged,
            n_iter=self.n_iter,
            reml_objective=_index_per_unit(self.reml_objective, idx),
            reml_at_boundary=_index_per_unit(self.reml_at_boundary, idx),
            penalty_selected_by_reml=_index_per_unit(
                self.penalty_selected_by_reml, idx
            ),
            pooled=self.pooled,
        )

    def __iter__(self) -> Iterator[SpatialRateResult]:
        """Iterate over single-neuron results.

        Yields
        ------
        SpatialRateResult
            Result for each neuron in order.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> results = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> peaks = [float(r.peak_firing_rate()) for r in results]
        >>> len(peaks)
        3
        """
        for i in range(len(self)):
            yield self[i]

    def peak_locations(self) -> NDArray[np.float64]:
        """Locations of peak firing for all neurons (batch accessor).

        Plural batch counterpart to :meth:`peak_location`. Returns the
        bin-center coordinates of the maximum firing rate for each neuron in
        the population. This is the canonical plural accessor mandated by the
        v0.6 result-class contract; it returns the same array as the batch
        form of :meth:`peak_location`.

        Returns
        -------
        ndarray, shape (n_neurons, n_dims)
            Spatial coordinates of the bins with maximum firing rate for
            each neuron. Uses ``nanargmax`` to handle NaN values.

        See Also
        --------
        SpatialRateResult.peak_location : Single-neuron version.
        peak_firing_rate : Peak firing rate values.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> result.peak_locations().shape
        (3, 2)
        """
        return self.peak_location()

    def plot(
        self, idx: int | None = None, ax: Axes | None = None, **kwargs: Any
    ) -> Axes:
        """Plot the spatial rate map for a specific neuron.

        Delegates to the environment's plot_field method for consistent
        visualization across the codebase.

        Parameters
        ----------
        idx : int
            Index of the neuron to plot (0-indexed). Required: this batch result
            holds N units, so there is no single map to plot without a choice of
            unit. Passing ``None`` raises a ``ValueError`` explaining how to pick
            one.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        **kwargs
            Additional keyword arguments passed to env.plot_field().
            Common options include:
            - cmap : str or Colormap, default="viridis"
            - vmin, vmax : float, colorbar limits
            - colorbar : bool, default=True
            - colorbar_label : str, default="Firing Rate (Hz)"

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Raises
        ------
        ValueError
            If `idx` is None. A batch result holds N units, so a unit index is
            required (e.g. ``result.plot(0)`` or iterate ``result[i].plot()``).

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")  # non-interactive backend for doctest
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> ax = result.plot(idx=0)
        >>> type(ax).__name__
        'Axes'
        """
        if idx is None:
            n_units = np.asarray(self.firing_rates).shape[0]
            raise ValueError(
                f"plot() requires a unit index: this batch result holds "
                f"{n_units} units, so there is no single rate map to plot. "
                f"Pass a unit index, e.g. result.plot(0), or iterate the units "
                f"with `for r in result: r.plot()` / `result[i].plot()`."
            )
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        kwargs.setdefault("colorbar_label", "Firing Rate (Hz)")
        return self.env.plot_field(_to_numpy(rates[idx]), ax=ax, **kwargs)

    def to_xarray(self) -> Any:
        """Convert the firing-rate maps to a labeled :class:`xarray.Dataset`.

        Wraps the ``(n_units, n_bins)`` firing-rate matrix in a labeled
        :class:`xarray.Dataset` with dims ``("unit_id", "bin")``. The
        ``unit_id`` index coordinate holds the real per-unit identity labels
        (:attr:`unit_ids`), enabling label-based selection. The ``bin``
        dimension carries non-index ``bin_center_x`` / ``bin_center_y``
        (and ``bin_center_z`` for 3-D envs) coordinates.

        Returns
        -------
        xarray.Dataset
            Dataset with:

            - data var ``firing_rate`` (Hz), dims ``("unit_id", "bin")``.
            - data var ``occupancy`` (seconds), dims ``("bin",)``.
            - index coord ``unit_id`` = :attr:`unit_ids`.
            - non-index coords ``bin_center_x`` / ``bin_center_y`` /
              ``bin_center_z`` on ``bin`` (per env dimensionality).
            - ``attrs``: ``method``, ``env`` fingerprint, ``software_version``,
              ``units`` (when set), and ``bandwidth`` (only for the ratio methods;
              omitted for ``method="glm"``, whose ``bandwidth`` is ``None`` and is
              not NetCDF-serializable).

        Raises
        ------
        ValueError
            If :attr:`unit_ids` contains duplicate labels (label-based
            ``.sel(unit_id=...)`` requires uniqueness).
        ImportError
            If ``xarray`` is not installed. xarray is an optional dependency;
            install it with ``pip install neurospatial[xarray]`` or
            ``pip install xarray``.

        Notes
        -----
        ``xarray`` is imported lazily inside this method, so it never becomes
        an import-time dependency of ``neurospatial``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> ds = result.to_xarray()  # doctest: +SKIP
        >>> ds["firing_rate"].dims  # doctest: +SKIP
        ('unit_id', 'bin')
        >>> ds.sel(unit_id=result.unit_ids[0])  # doctest: +SKIP

        See Also
        --------
        spatial_information : Per-neuron Skaggs spatial information.
        """
        from neurospatial._results import (
            build_population_dataset,
            env_fingerprint,
            software_version,
            units_attr,
        )

        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        attrs: dict[str, Any] = {
            **units_attr(self.env),
            "method": self.method,
            "env": env_fingerprint(self.env),
            "software_version": software_version(),
        }
        # Guard on the value, not on ``method``: NetCDF attributes cannot hold
        # ``None`` (``Dataset.to_netcdf()`` would raise ``TypeError``), and
        # ``bandwidth`` is ``None`` for ``method="glm"``. Keying on
        # ``bandwidth is not None`` guards exactly that serialization precondition
        # -- the same "omit-when-unset" rule ``units_attr`` uses -- so it stays
        # correct even if a ratio result ever carried a ``None`` bandwidth.
        if self.bandwidth is not None:
            attrs["bandwidth"] = self.bandwidth
        return build_population_dataset(
            rates,
            np.asarray(self.unit_ids),
            env=self.env,
            occupancy=np.asarray(self.occupancy, dtype=np.float64),
            attrs=attrs,
        )

    def spatial_information(self) -> NDArray[np.float64] | Any:
        """Skaggs spatial information (bits per spike) for all neurons.

        Quantifies how much information each spike conveys about the
        animal's spatial location. Higher values indicate more spatially
        selective firing.

        Returns
        -------
        ndarray | jax.Array, shape (n_neurons,)
            Spatial information in bits/spike for each neuron.
            Always non-negative. Returns 0.0 for uniform firing.

            **Backend-aware**: Returns NumPy array for NumPy input,
            JAX array for JAX input.

        Notes
        -----
        Uses the Skaggs et al. (1993) formula:

        .. math::

            I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log_2 \\left( \\frac{r_i}{\\bar{r}} \\right)

        **Interpretation**:

        - Place cells typically have 1-3 bits/spike
        - Higher values indicate more spatially selective firing
        - Zero means uniform firing (no spatial selectivity)

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
               An information-theoretic approach to deciphering the hippocampal code.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> info = result.spatial_information()
        >>> info.shape
        (3,)

        See Also
        --------
        neurospatial.encoding._metrics.batch_spatial_information : Underlying computation
        """
        from neurospatial.encoding._metrics import batch_spatial_information

        # Pass arrays directly - _metrics.py handles JAX dispatch
        return batch_spatial_information(self.firing_rates, self.occupancy)

    def sparsity(self) -> NDArray[np.float64] | Any:
        """Sparsity of spatial firing for all neurons.

        Measures what fraction of the environment elicits significant
        firing. Lower values indicate sparser, more selective place fields.

        Returns
        -------
        ndarray | jax.Array, shape (n_neurons,)
            Sparsity values in range [0, 1] for each neuron.
            - Low (0.1-0.3): Sparse, selective place field
            - High (~1.0): Uniform firing throughout environment

            **Backend-aware**: Returns NumPy array for NumPy input,
            JAX array for JAX input.

        Notes
        -----
        Uses the Skaggs et al. (1996) formula:

        .. math::

            S = \\frac{\\left( \\sum_i p_i r_i \\right)^2}{\\sum_i p_i r_i^2}

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., et al. (1996). Theta phase
               precession in hippocampal neuronal populations.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> spars = result.sparsity()
        >>> spars.shape
        (3,)

        See Also
        --------
        neurospatial.encoding._metrics.batch_sparsity : Underlying computation
        """
        from neurospatial.encoding._metrics import batch_sparsity

        # Pass arrays directly - _metrics.py handles JAX dispatch
        return batch_sparsity(self.firing_rates, self.occupancy)

    def grid_scores(self) -> BatchScoresResult:
        """Grid scores (hexagonal periodicity) for all neurons.

        Quantifies the hexagonal periodicity of each neuron's firing rate map,
        which is characteristic of grid cells. Higher values indicate stronger
        hexagonal grid patterns.

        Returns
        -------
        BatchScoresResult
            Container with ``scores`` (shape ``(n_neurons,)``, range [-2, 2])
            and ``failures`` (boolean mask, ``True`` for neurons whose grid
            score computation raised an exception that was caught and
            converted to NaN). Use ``result.scores`` for the raw array if
            your downstream code expects a plain ndarray.

        Notes
        -----
        For each neuron, computes the spatial autocorrelation and extracts
        the grid score based on rotational symmetry.

        Uses the Sargolini et al. (2006) algorithm:

        1. Compute 2D spatial autocorrelation via FFT
        2. Rotate by 30°, 60°, 90°, 120°, 150°
        3. Grid score = min(r60, r120) - max(r30, r90, r150)

        Delegates to ``neurospatial.encoding._metrics.batch_grid_scores()``.

        References
        ----------
        .. [1] Sargolini, F., Fyhn, M., et al. (2006). Conjunctive
               representation of position, direction, and velocity in
               entorhinal cortex. Science, 312(5774), 758-762.

        See Also
        --------
        SpatialRateResult.grid_score : Single-neuron grid score
        SpatialRateResult.grid_properties : Full grid cell metrics

        Examples
        --------
        >>> result = SpatialRatesResult(...)  # doctest: +SKIP
        >>> scores = result.grid_scores()  # doctest: +SKIP
        >>> # The result wraps an ndarray; reach in via .scores for math
        >>> # operations that need a real array.
        >>> print(f"Mean grid score: {np.nanmean(scores.scores):.3f}")  # doctest: +SKIP
        >>> n_grid_cells = int(np.sum(scores.scores > 0.4))  # doctest: +SKIP
        """
        from neurospatial.encoding._metrics import batch_grid_scores

        return batch_grid_scores(self.env, _to_numpy(self.firing_rates))

    def border_scores(
        self,
        threshold: float = 0.3,
        min_area: float = 0.0,
        metric: Literal["geodesic", "euclidean"] = "geodesic",
    ) -> BatchScoresResult:
        """Border scores (boundary proximity tuning) for all neurons.

        Quantifies how much each neuron's firing field is aligned with
        environmental boundaries (walls). Higher values indicate stronger
        border cell properties.

        Parameters
        ----------
        threshold : float, default 0.3
            Fraction of peak firing rate used to segment the field.
            Bins with firing rate >= threshold * peak are included in field.
        min_area : float, default 0.0
            Minimum field area in physical units (e.g., cm²). Fields smaller
            than this return NaN. Default 0.0 (no filtering).
        metric : {'geodesic', 'euclidean'}, default 'geodesic'
            Distance metric for computing distance from field to boundaries.
            - 'geodesic': Graph shortest path distance (respects obstacles)
            - 'euclidean': Straight-line distance in physical space

        Returns
        -------
        ndarray, shape (n_neurons,)
            Border scores in range [-1, 1] for each neuron.
            - +1: Perfect border cell (field on boundary)
            - 0: No boundary preference
            - -1: Anti-border (field in center)
            Returns NaN for neurons where border score cannot be computed.

        Notes
        -----
        Uses the Solstad et al. (2008) algorithm for each neuron.

        Delegates to ``neurospatial.encoding._metrics.batch_border_scores()``.

        References
        ----------
        .. [1] Solstad, T., Boccara, C. N., et al. (2008). Representation of
               geometric borders in the entorhinal cortex. Science, 322(5909),
               1865-1868.

        See Also
        --------
        SpatialRateResult.border_score : Single-neuron border score

        Examples
        --------
        >>> result = SpatialRatesResult(...)  # doctest: +SKIP
        >>> scores = result.border_scores()  # doctest: +SKIP
        >>> # The result wraps an ndarray; reach in via .scores for math
        >>> # operations that need a real array.
        >>> print(
        ...     f"Mean border score: {np.nanmean(scores.scores):.3f}"
        ... )  # doctest: +SKIP
        >>> n_border_cells = int(np.sum(scores.scores > 0.5))  # doctest: +SKIP
        """
        from neurospatial.encoding._metrics import batch_border_scores

        return batch_border_scores(
            self.env,
            _to_numpy(self.firing_rates),
            threshold=threshold,
            min_area=min_area,
            metric=metric,
        )

    def label_cell_types(
        self,
        min_spatial_info: float = 0.5,
        min_grid_score: float = 0.4,
        min_border_score: float = 0.5,
        *,
        grid_scores: NDArray[np.float64] | None = None,
        border_scores: NDArray[np.float64] | None = None,
    ) -> NDArray[np.str_]:
        """Label neurons with multi-class spatial cell types.

        Applies threshold-based classification to label neurons as place cells,
        grid cells, border cells, or unclassified based on their spatial
        information, grid score, and border score.

        This is the **multi-class labeler** (returns string labels). It is
        distinct from the single-type :meth:`classify` boolean predicate; use
        ``label_cell_types`` when you need ``"place"``/``"grid"``/``"border"``/
        ``"unclassified"`` labels (e.g. ``df[df["cell_type"] == "place"]``).

        Parameters
        ----------
        min_spatial_info : float, default 0.5
            Minimum spatial information (bits/spike) to be classified as a
            spatially tuned cell. Neurons below this are labeled "unclassified".
        min_grid_score : float, default 0.4
            Minimum grid score to be classified as a grid cell. Standard
            threshold from Sargolini et al. (2006).
        min_border_score : float, default 0.5
            Minimum border score to be classified as a border cell. Standard
            threshold from Solstad et al. (2008).
        grid_scores : ndarray of shape (n_neurons,), optional
            Precomputed grid scores, one per neuron. When provided, these are
            used directly instead of recomputing via :meth:`grid_scores`. Must
            be 1-D with length equal to the number of neurons. When ``None``
            (the default), grid scores are recomputed.
        border_scores : ndarray of shape (n_neurons,), optional
            Precomputed border scores, one per neuron. When provided, these are
            used directly instead of recomputing via :meth:`border_scores`. Must
            be 1-D with length equal to the number of neurons. When ``None``
            (the default), border scores are recomputed.

        Returns
        -------
        ndarray, shape (n_neurons,)
            String labels for each neuron. One of:
            - "grid": Grid cell (high grid score, passes spatial info threshold)
            - "border": Border cell (high border score, passes spatial info threshold)
            - "place": Place cell (high spatial info, not grid or border)
            - "unclassified": Does not meet criteria for any spatial cell type

        Notes
        -----
        Pass precomputed scores via ``grid_scores=``/``border_scores=`` to avoid
        recomputation when you've already computed them, e.g. from
        :meth:`summary_table` (which computes them once and forwards them here).

        **Classification priority** (higher takes precedence):

        1. **Grid cell**: grid_score >= min_grid_score
        2. **Border cell**: border_score >= min_border_score
        3. **Place cell**: spatial_info >= min_spatial_info (and not grid/border)
        4. **Unclassified**: Does not meet any criteria

        **Typical thresholds** (from literature):

        - Spatial information: 0.5-1.0 bits/spike (varies by study)
        - Grid score: 0.4-0.5 (Sargolini et al., 2006)
        - Border score: 0.5-0.6 (Solstad et al., 2008)

        References
        ----------
        .. [1] Sargolini, F., et al. (2006). Science, 312(5774), 758-762.
        .. [2] Solstad, T., et al. (2008). Science, 322(5909), 1865-1868.
        .. [3] Skaggs, W. E., et al. (1993). NIPS, 5, 1030-1037.

        See Also
        --------
        spatial_information : Compute spatial information
        grid_scores : Compute grid scores
        border_scores : Compute border scores
        classify : Single-type place-cell boolean predicate
        EgocentricRatesResult.classify : Sibling batch classifier
        ViewRatesResult.classify : Sibling batch classifier

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> labels = result.label_cell_types()
        >>> labels.shape
        (3,)
        >>> valid = {"grid", "border", "place", "unclassified"}
        >>> set(labels.tolist()).issubset(valid)
        True
        """
        n_neurons = len(self)

        spatial_info = self.spatial_information()
        # grid_scores() / border_scores() return BatchScoresResult; pull
        # the float array out via .scores for the boolean masks below.
        # Callers (e.g. summary_table) may pass precomputed score arrays to
        # avoid the expensive double recompute.
        if grid_scores is None:
            grid_scores_arr = self.grid_scores().scores
        else:
            grid_scores_arr = np.asarray(grid_scores, dtype=np.float64)
            if grid_scores_arr.ndim != 1 or grid_scores_arr.shape[0] != n_neurons:
                raise ValueError(
                    f"grid_scores must be a 1-D array of length {n_neurons} "
                    f"(one per neuron), got shape {grid_scores_arr.shape}"
                )
        if border_scores is None:
            border_scores_arr = self.border_scores().scores
        else:
            border_scores_arr = np.asarray(border_scores, dtype=np.float64)
            if border_scores_arr.ndim != 1 or border_scores_arr.shape[0] != n_neurons:
                raise ValueError(
                    f"border_scores must be a 1-D array of length {n_neurons} "
                    f"(one per neuron), got shape {border_scores_arr.shape}"
                )

        labels = np.full(n_neurons, "unclassified", dtype="<U14")
        is_place = spatial_info >= min_spatial_info
        is_border = (~np.isnan(border_scores_arr)) & (
            border_scores_arr >= min_border_score
        )
        is_grid = (~np.isnan(grid_scores_arr)) & (grid_scores_arr >= min_grid_score)

        # Priority: grid > border > place > unclassified (assign in reverse so
        # higher-priority labels overwrite lower ones).
        labels[is_place] = "place"
        labels[is_border] = "border"
        labels[is_grid] = "grid"

        return labels

    def detect_cell_types(
        self,
        min_spatial_info: float = 0.5,
        min_grid_score: float = 0.4,
        min_border_score: float = 0.5,
    ) -> NDArray[np.str_]:
        """Deprecated alias for :meth:`label_cell_types`.

        .. deprecated:: 0.6
            ``detect_cell_types`` is deprecated since 0.6; use
            :meth:`label_cell_types` instead. Removed in 0.7.

        Parameters
        ----------
        min_spatial_info : float, default 0.5
            Minimum spatial information (bits/spike).
        min_grid_score : float, default 0.4
            Minimum grid score for a grid cell.
        min_border_score : float, default 0.5
            Minimum border score for a border cell.

        Returns
        -------
        ndarray, shape (n_neurons,)
            String labels: ``"grid"``/``"border"``/``"place"``/
            ``"unclassified"``.
        """
        warnings.warn(
            "detect_cell_types is deprecated since 0.6, use label_cell_types; "
            "removed in 0.7",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.label_cell_types(
            min_spatial_info=min_spatial_info,
            min_grid_score=min_grid_score,
            min_border_score=min_border_score,
        )

    def classify(self, *, min_spatial_info: float = 0.5) -> NDArray[np.bool_]:
        """Classify neurons as place cells (single-type boolean predicate).

        A neuron is classified as a place cell if its spatial information
        meets the minimum threshold. This is the single-type boolean
        predicate ("is this a place cell") sibling of
        :meth:`EgocentricRatesResult.classify` and
        :meth:`ViewRatesResult.classify`.

        For multi-class labels (``"place"``/``"grid"``/``"border"``/
        ``"unclassified"``) use :meth:`label_cell_types` instead.

        .. note::
           This batch predicate uses a **spatial-information threshold**,
           whereas the single-neuron
           :meth:`SpatialRateResult.is_place_cell` uses **place-field
           detection** (``detect_place_fields``). The two criteria can
           disagree for the same neuron (high information but no contiguous
           field, or vice versa), so ``classify()[i]`` is not guaranteed to
           equal ``result[i].is_place_cell()``. Pick the criterion that suits
           your analysis and apply it consistently.

        Parameters
        ----------
        min_spatial_info : float, default 0.5
            Minimum spatial information (bits/spike) to be classified as a
            place cell.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Boolean array where True indicates the neuron is classified as a
            place cell.

        See Also
        --------
        label_cell_types : Multi-class string labeler
        is_place_cell : Free-function single-neuron place predicate

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> is_place = result.classify()
        >>> is_place.shape
        (3,)
        >>> is_place.dtype == bool
        True
        """
        spatial_info = np.asarray(self.spatial_information())
        return spatial_info >= min_spatial_info

    def summary_table(
        self,
        unit_ids: Sequence[str | int] | None = None,
        include_classification: bool = True,
    ) -> pd.DataFrame:
        """Per-unit scalar summary: one row per unit, ``unit_id``-indexed.

        Computes all spatial metrics and returns one row per unit, indexed by
        ``unit_id``, with scalar metric columns. This is the per-unit summary a
        many-neuron user wants for filtering, sorting, and population tables.
        For the dense per-bin frame (one row per ``(unit, bin)``) use
        :meth:`to_dataframe` instead.

        This is a host-only method; all metrics are computed as NumPy arrays
        (not JAX).

        Parameters
        ----------
        unit_ids : sequence of str or int, optional
            Identity labels for the index, one per unit. If ``None``, the
            result's own :attr:`unit_ids` are used.
        include_classification : bool, default True
            Whether to include the ``cell_type`` column with classification
            labels from ``label_cell_types()``.

        Returns
        -------
        pd.DataFrame
            One row per unit, indexed by ``unit_id``, with columns:

            - peak_x: x-coordinate of peak firing location
            - peak_y: y-coordinate of peak firing location (NaN for 1D)
            - peak_rate: maximum firing rate (Hz)
            - spatial_info: spatial information (bits/spike)
            - sparsity: sparsity measure (0-1)
            - grid_score: grid score (hexagonal periodicity)
            - border_score: border score (boundary proximity tuning)
            - cell_type: classification label (if include_classification=True)
            - method: estimator used to compute the rate map

        Notes
        -----
        This method computes all metrics at once, which may be slow for
        large populations. For selective metric computation, use the
        individual methods (``spatial_information()``, ``grid_scores()``, etc.).

        **Common pandas workflows**:

        - Filter: ``df[df["cell_type"] == "place"]``
        - Sort: ``df.sort_values("spatial_info", ascending=False)``
        - Top-N: ``df.nlargest(10, "peak_rate")``

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> df = result.summary_table()
        >>> "cell_type" in df.columns
        True
        >>> len(df)
        3
        >>> df.index.name
        'unit_id'

        >>> # Filter for place cells
        >>> place_cells = df[df["cell_type"] == "place"]

        >>> # Sort by spatial information
        >>> top_cells = df.sort_values("spatial_info", ascending=False)

        >>> # Custom unit identifiers
        >>> df = result.summary_table(unit_ids=["unit_0", "unit_1", "unit_2"])
        >>> df.index.tolist()
        ['unit_0', 'unit_1', 'unit_2']

        See Also
        --------
        to_dataframe : Dense per-bin frame (one row per (unit, bin)).
        label_cell_types : Cell type classification
        spatial_information : Batch spatial information computation
        grid_scores : Batch grid score computation
        border_scores : Batch border score computation
        """
        import pandas as pd

        n_neurons = len(self)

        if unit_ids is None:
            index_ids: list[str | int] = list(np.asarray(self.unit_ids))
        else:
            index_ids = list(unit_ids)
            if len(index_ids) != n_neurons:
                raise ValueError(
                    f"unit_ids has {len(index_ids)} elements but "
                    f"result contains {n_neurons} units"
                )

        # Compute peak locations
        peaks = self.peak_location()
        n_dims = peaks.shape[1] if peaks.ndim > 1 else 1

        # Compute grid/border scores ONCE and reuse them for both the score
        # columns and label_cell_types(), avoiding a double recompute (the
        # expensive batch_grid_scores/batch_border_scores each run once).
        grid_scores_arr = self.grid_scores().scores
        border_scores_arr = self.border_scores().scores

        # Build data dictionary
        data: dict[str, Any] = {
            "peak_x": peaks[:, 0],
            "peak_y": peaks[:, 1] if n_dims > 1 else np.full(n_neurons, np.nan),
            "peak_rate": self.peak_firing_rate(),
            "spatial_info": self.spatial_information(),
            "sparsity": self.sparsity(),
            "grid_score": grid_scores_arr,
            "border_score": border_scores_arr,
        }

        if include_classification:
            data["cell_type"] = self.label_cell_types(
                grid_scores=grid_scores_arr,
                border_scores=border_scores_arr,
            )

        data["method"] = self.method

        # GAM columns, only for ``method="glm"``. ``deviance`` is per-unit; the
        # batch-scalar diagnostics broadcast to every row. ``penalty`` /
        # ``reml_objective`` / ``reml_at_boundary`` broadcast when scalar and
        # become per-unit columns when they are ``(n_units,)`` vectors
        # (``pooled=False``). Keyed on ``method`` (the single "is this glm?"
        # discriminant, matching the NWB writer), not on a GAM field's None-ness.
        if self.method == "glm":
            data["penalty"] = self.penalty
            data["rank"] = self.rank
            data["deviance"] = np.asarray(self.deviance)
            data["converged"] = self.converged
            data["n_iter"] = self.n_iter
            data["reml_objective"] = self.reml_objective
            data["reml_at_boundary"] = self.reml_at_boundary
            data["pooled"] = self.pooled
            # Per-unit provenance mask only exists for the per-unit (pooled=False)
            # path; skip it (rather than write an all-None column) otherwise.
            if self.penalty_selected_by_reml is not None:
                data["penalty_selected_by_reml"] = np.asarray(
                    self.penalty_selected_by_reml
                )

        return pd.DataFrame(data, index=pd.Index(index_ids, name="unit_id"))


# ==============================================================================
# Compute Functions
# ==============================================================================


def _fill_nan(rates: ArrayLike, fill_value: float) -> ArrayLike:
    """Replace NaN entries of a rate map with ``fill_value``.

    Works for both NumPy and JAX arrays. The masked/low-occupancy bins set
    to NaN by ``min_occupancy`` are the only NaN entries in a rate map, so
    this targets exactly those bins.

    Parameters
    ----------
    rates : ArrayLike
        Firing rate map (NumPy or JAX array), any shape.
    fill_value : float
        Value substituted wherever ``rates`` is NaN.

    Returns
    -------
    ArrayLike
        A new array of the same type/shape with NaN replaced by ``fill_value``.
    """
    # JAX arrays expose .at/.dtype but are not numpy ndarrays; dispatch on
    # whether the object is a numpy array. Both backends support np.isnan via
    # the array's own namespace, but jnp.where keeps the result on-device.
    if isinstance(rates, np.ndarray):
        return np.where(np.isnan(rates), fill_value, rates)
    import jax.numpy as jnp

    rates_jax = cast("Any", rates)
    return cast("ArrayLike", jnp.where(jnp.isnan(rates_jax), fill_value, rates_jax))


def _compute_glm_spatial_rates(
    env: Environment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    penalty: float | None,
    rank: int | None,
    pooled: bool = True,
    resolved_backend: Literal["numpy", "jax"],
    dtype: type[np.float32] | type[np.float64],
) -> tuple[ArrayLike, Any]:
    """Fit ``method="glm"`` and assemble the full active-bin firing-rate array.

    Orchestrates the ``method="glm"`` estimator: it owns the **unit-major <->
    bin-major** boundary, the float64 core / dtype-at-the-boundary split, and the
    two-concern backend handling. It composes the reduced-rank penalty basis
    (:meth:`Environment._mrf_basis`) and the penalized-Poisson fit
    (:func:`neurospatial.encoding._glm.fit_mrf_gam`) without modifying either.

    The encoding side is **unit-major** ``(n_units, n_bins)``; the fit is
    **bin-major** ``(n_live_bins, n_units)``. The transpose + the restriction to
    ``basis.live_bins`` happen here (the only live-bin restriction; the fit never
    re-slices), and the bin-major ``log_rate`` is scattered back into a
    ``_RATE_FLOOR``-filled ``(n_units, n_bins)`` array whose rows are units.

    Parameters
    ----------
    env : Environment
        Fitted environment; supplies the diffusion geometry and the reduced-rank
        MRF penalty basis.
    spike_counts : NDArray[np.float64], shape (n_units, n_bins)
        Per-unit binned spike counts in active-bin order (unit-major).
    occupancy : NDArray[np.float64], shape (n_bins,)
        Dwell time per active bin (seconds); the Poisson log-offset.
    penalty : float or None, keyword-only
        Fixed ``lambda`` (echoed on the fit) or ``None`` to select by REML.
    rank : int or None, keyword-only
        Requested basis rank cap; ``None`` resolves to the module default. The
        effective rank is reported via ``MRFFit.rank`` after clamping.
    resolved_backend : {"numpy", "jax"}, keyword-only
        The already-resolved backend (via ``get_backend_name``). Forwarded to
        ``fit_mrf_gam``, which routes the fit compute to the float32 JAX mirror
        when this is ``"jax"`` and JAX is installed (else the float64 core); either
        way ``MRFFit`` arrays come back NumPy float64. The returned rate array is
        converted to a JAX array when this is ``"jax"``, matching the ratio path's
        return contract.
    dtype : {np.float32, np.float64}, keyword-only
        Storage dtype of the returned rate array (applied at the boundary; the
        glm core stays float64).

    Returns
    -------
    firing_rates : ArrayLike, shape (n_units, n_bins)
        Assembled firing-rate map(s) (dtype-cast, JAX-converted when needed).
        ``max(exp(eta), _RATE_FLOOR)`` on live bins, ``_RATE_FLOOR`` elsewhere.
    fit : MRFFit
        The raw (always-NumPy, float64) fit result; the caller reads the GAM
        result fields off it.
    """
    from neurospatial.encoding._backend import is_jax_available
    from neurospatial.encoding._glm import _RATE_FLOOR, fit_mrf_gam

    spike_counts = np.asarray(spike_counts, dtype=np.float64)  # (n_units, n_bins)
    occupancy = np.asarray(occupancy, dtype=np.float64)  # (n_bins,)
    n_units, n_bins = spike_counts.shape

    # Reduced-rank penalty basis from occupancy (live-bin order). Cast to the
    # protocol for the ``self: SelfEnv`` bound (same pattern as ``env.diffuse`` in
    # ``_smoothing.py``).
    basis = cast("EnvironmentProtocol", env)._mrf_basis(occupancy, rank=rank)
    live_bins = np.asarray(basis.live_bins, dtype=np.intp)

    # Dead-component warning is owned here (not by ``fit_mrf_gam``): it needs env's
    # TOTAL component count, which the basis (live-only) does not carry. Warn only
    # when there ARE units and there is at least one live component but fewer than
    # the total: the fully-dead (zero total occupancy) case is already covered by
    # ``fit_mrf_gam``'s own warning, and a no-neuron call has no rates to floor.
    n_components = int(env._diffusion_geometry.n_components)
    if n_units and 0 < basis.n_live_components < n_components:
        n_dead = n_components - basis.n_live_components
        warnings.warn(
            f"MRF-GAM fit: {n_dead} of {n_components} environment components were "
            f"never occupied (dead); their bins are set to _RATE_FLOOR "
            f"({_RATE_FLOOR:.0e} Hz). Fit runs on the live bins only.",
            UserWarning,
            # 3 frames out: warn -> _compute_glm_spatial_rates -> the public
            # compute_spatial_rate(s), so it points at the caller's line.
            stacklevel=3,
        )

    # Boundary in: unit-major (n_units, n_bins) -> bin-major (n_live_bins,
    # n_units), restricted to live bins (the ONLY live-bin restriction).
    counts_fit = spike_counts.T[live_bins, :]  # (n_live_bins, n_units)
    occ_fit = occupancy[live_bins]  # (n_live_bins,)

    # Fit at the chosen penalty. ``pooled`` selects shared (``True``) vs per-unit
    # (``False``) lambda; it is forwarded from the public estimator, validated
    # there. ``backend`` dispatches the fit compute in ``fit_mrf_gam`` (the float32
    # JAX mirror when resolved to ``"jax"`` and available, else the float64 core);
    # MRFFit arrays come back NumPy float64 either way.
    fit = fit_mrf_gam(
        basis,
        counts_fit,
        occ_fit,
        penalty=penalty,
        pooled=pooled,
        backend=resolved_backend,
    )

    # Boundary out: floor-fill, then scatter max(exp(eta), _RATE_FLOOR).T into the
    # live-bin columns so rows are units.
    firing_rates = np.full((n_units, n_bins), _RATE_FLOOR, dtype=np.float64)
    if live_bins.size and n_units:
        rate_live = np.maximum(np.exp(np.asarray(fit.log_rate)), _RATE_FLOOR)
        firing_rates[:, live_bins] = rate_live.T  # (n_units, n_live_bins)

    # dtype at the result boundary (the core stayed float64), like the ratio path.
    firing_rates_out: ArrayLike = firing_rates.astype(dtype, copy=False)

    # Return-array-type contract: convert to a JAX array iff the resolved backend
    # is "jax" (resolved via get_backend_name upstream -- NOT a raw backend check),
    # matching what the ratio path returns for backend="jax".
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        jnp_dtype = jnp.float32 if dtype is np.float32 else jnp.float64
        firing_rates_out = jnp.asarray(firing_rates, dtype=jnp_dtype)

    return firing_rates_out, fit


def compute_spatial_rate(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64] | PositionLike,
    positions: NDArray[np.float64] | None = None,
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float | None = None,
    fill_value: float | None = None,
    penalty: float | None = None,
    rank: int | None = None,
    pooled: bool = True,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
    warn_on_drop: bool = True,
) -> SpatialRateResult:
    """Compute spatial firing rate map for one neuron.

    This function computes a smoothed firing rate map from spike times
    and trajectory data. The result is a SpatialRateResult object containing
    the firing rate map, occupancy, and metadata.

    Computing rate maps for many neurons? Use
    :func:`compute_spatial_rates` (plural) instead — it shares the occupancy
    and smoothing-kernel computation across the whole population in one call
    rather than recomputing it per neuron.

    Parameters
    ----------
    env : Environment
        The spatial environment defining the bin structure. Must be fitted
        (e.g., created via ``Environment.from_samples()``).
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,), or PositionLike
        Timestamps of trajectory samples in seconds. May instead be a single
        ``PositionLike`` object (exposing ``.t`` and ``.values``, e.g. a
        pynapple ``Tsd`` / ``TsdFrame``) carrying both times and positions, in
        which case ``positions`` must be omitted.
    positions : ndarray, shape (n_samples, n_dims), optional
        Position coordinates at each time sample. NaN values are treated as
        missing data and excluded from occupancy and firing-rate computation;
        callers do not need to pre-filter tracking dropouts. Omit only when
        ``times`` is a ``PositionLike`` object carrying the positions.
    method : {"diffusion_kde", "gaussian_kde", "binned", "glm"}, \
default="diffusion_kde"
        Estimator to use:

        - **diffusion_kde** (recommended): Graph-based boundary-aware KDE.
          Respects environment boundaries (walls, obstacles). Uses diffusion
          kernel computed from environment graph.
        - **gaussian_kde**: Standard Euclidean KDE. Uses Gaussian kernel based
          on Euclidean distance between bin centers. Ignores boundaries (mass
          can "bleed through" walls).
        - **binned**: Bin-then-smooth method. Computes raw rate first, then
          smooths. Can introduce discretization artifacts.
        - **glm**: Penalized-Poisson GAM. Occupancy enters as a **log-offset**
          (never a denominator) and the smoothness penalty λ is chosen by REML,
          so the fit returns **finite rates everywhere** -- including
          low-occupancy and unvisited bins where the ratio estimators NaN. Tuned
          with ``penalty`` and ``rank`` (not ``bandwidth`` / ``min_occupancy`` /
          ``fill_value``, which are mutually exclusive with ``method="glm"``).

        Note: ``diffusion_kde`` and ``binned`` smooth matrix-free via the cached
        finite-volume eigenbasis (O(n_bins·rank) per neuron) — they never build a
        dense kernel and scale to large/fine grids. ``glm`` also avoids a dense
        O(n_bins²) kernel (it fits on the rank-``r`` basis), but its
        penalized-Poisson fit is heavier and **not** linear in ``rank``: each
        Newton step builds a per-unit (r, r) Hessian ``Bᵀ diag(μ) B`` (≈
        O(n_units·n_bins·rank²)) and solves it (≈ O(n_units·rank³)), and this is
        repeated across Newton iterations and REML λ candidates. Keep ``rank``
        modest for large populations. Only ``gaussian_kde`` builds a dense
        O(n_bins²) matrix; for very large grids prefer ``diffusion_kde``, or
        increase ``bin_size``.

    bandwidth : float | None, default=None
        (Ratio methods only.) Smoothing bandwidth in the same units as bin_size;
        larger values produce more smoothing. ``None`` resolves to ``5.0``.
        Mutually exclusive with ``method="glm"``.
    min_occupancy : float | None, default=None
        (Ratio methods only.) Minimum occupancy (seconds) for a bin to be
        included; bins below the threshold are set to NaN. ``None`` resolves to
        ``0.0`` (no masking). Mutually exclusive with ``method="glm"``.
    fill_value : float | None, default=None
        (Ratio methods only.) Value used to replace NaN bins (masked/low-occupancy
        bins produced by ``min_occupancy``). When ``None`` (the default), NaN is
        preserved so existing callers see no behavior change. Pass
        ``fill_value=0.0`` for the recommended decoding golden path: a zero-rate
        map composes directly with
        :func:`~neurospatial.decoding.posterior.decode_position` without manual
        NaN scrubbing. ``occupancy`` is unaffected, so callers can still recover
        which bins were masked via ``result.occupancy < min_occupancy``.
        Mutually exclusive with ``method="glm"`` (glm rates are already finite).
    penalty : float | None, default=None
        (``method="glm"`` only.) Fixed smoothness penalty ``λ`` (≥ 0; ``0`` = no
        penalty). ``None`` (the default) selects ``λ`` by REML; on pathologically
        under-sampled data where no ``λ`` yields a converged, positive-definite
        fit, REML raises ``ValueError`` (supply a fixed ``penalty``, coarsen the
        grid, or reduce ``rank``). Mutually exclusive with the ratio methods.
    rank : int | None, default=None
        (``method="glm"`` only.) Requested rank of the reduced-rank penalty basis
        (≥ 1). ``None`` uses the module default cap. An out-of-range value is
        **clamped** (never rejected) to the effective rank
        ``max(n_live_components, min(n_live_bins, rank))``, reported via
        ``result.rank``. Mutually exclusive with the ratio methods.
    pooled : bool, default=True
        (``method="glm"`` only; strict ``bool``.) Whether REML selects **one
        shared** smoothing penalty ``λ`` for the population (``True``, the
        default) or an **independent per-unit** ``λ`` (``False``). ``pooled=False``
        runs the REML search once per unit, so the fit costs roughly one REML per
        neuron; ``result.penalty`` / ``reml_objective`` / ``reml_at_boundary``
        then hold that unit's own value (a single-unit call unwraps them to
        scalars). Zero-spike units (whose ``λ`` is unidentified) fall back to the
        pooled ``λ`` over the informative units, flagged
        ``penalty_selected_by_reml=False`` with ``reml_objective=nan``. A supplied
        fixed ``penalty`` beats ``pooled`` (REML is skipped and one scalar ``λ``
        is recorded); ``pooled`` is likewise a no-op at ``penalty_rank == 0`` (a
        shared-basis property) or when no unit spikes. A boundary warning means
        the selected ``λ`` sits on the search bound -- ``λ`` itself is weakly
        identified even though the fitted field is stable. Passing
        ``pooled=False`` with a ratio method raises ``ValueError``.
    speed : ndarray, shape (n_samples,), optional
        Precomputed instantaneous speed at each trajectory sample (physical
        units / second). Only used when ``min_speed`` is set. When
        ``min_speed`` is set and ``speed`` is ``None``, speed is auto-derived
        from the trajectory (see ``min_speed``). Pass your own ``speed`` for
        geodesic / linearized-track environments where the Euclidean
        auto-default is not appropriate. A wrong-length array raises
        ``ValueError``.
    min_speed : float, optional
        Minimum speed threshold (physical units / second). When set, low-speed
        periods are excluded from BOTH the spike numerator AND the occupancy
        denominator using ONE shared per-interval speed gate, so the firing
        rate stays correct (gating only one side would bias the rate). When
        ``None`` (the default) NO speed filtering is applied and the output is
        byte-for-byte identical to before.

        **Auto-speed convention.** When ``min_speed`` is set and ``speed`` is
        ``None``, speed is derived with a FORWARD difference to match the
        occupancy interval semantics (``time_allocation="start"``):
        ``speed[k] = ||positions[k+1] - positions[k]||_2 / (times[k+1] -
        times[k])`` for ``k = 0 .. n-2``, with ``speed[n-1] = speed[n-2]`` (the
        last sample starts no occupancy interval). This Euclidean default is
        simple; for geodesic / track environments pass an explicit ``speed``.
    max_gap : float | None, default=0.5
        Maximum trajectory time gap in seconds. Intervals with
        ``dt > max_gap`` (large tracking gaps) are excluded from BOTH the
        spike numerator AND the occupancy denominator using ONE shared
        per-interval mask, so the firing rate stays correct. This default
        matches ``env.occupancy``'s own default, so occupancy behavior is
        unchanged.

        .. note::
           **Behavior change (correctness fix).** Spikes occurring inside
           intervals longer than ``max_gap`` (large tracking gaps) or inside
           intervals whose start sample is out of bounds are now excluded from
           spike counts, matching occupancy. Previously such spikes were
           counted while their time was excluded from the denominator,
           inflating the rate. This changes firing-rate maps for sessions
           with large tracking gaps or out-of-bounds excursions. Pass
           ``max_gap=None`` to disable gap gating on BOTH sides (restoring the
           pre-fix, no-gap-gating behavior while keeping the two sides
           aligned).
    backend : {"numpy", "jax", "auto"}, default="numpy"
        Computation backend for rate map smoothing:

        - ``"numpy"``: Use NumPy for all computations. Works everywhere.
        - ``"jax"``: Use JAX for rate computation. Requires JAX installation.
          Enables GPU acceleration and JAX transformations (jit, grad).
        - ``"auto"``: Use JAX if available, otherwise NumPy.

        Note: Binning operations (spike counting, occupancy) always use NumPy.
        Only the smoothing/rate computation uses the selected backend. For
        ``method="glm"``, a resolved ``jax`` backend runs the penalized-Poisson
        fit + REML through an optional **float32** JAX mirror of the NumPy/SciPy
        core (``backend="jax"`` requires the ``jax`` extra, like the ratio
        methods; ``"auto"`` uses it when available and otherwise the NumPy core).
        The float32 mirror matches the float64 core to ~1e-6 at a fixed penalty (a
        touch looser under automatic REML, which picks a slightly different
        ``lambda``) and is markedly faster on populations. The returned
        diagnostics stay float64 either way.
    warn_on_drop : bool, default=True
        If ``True`` (the default), emit a ``UserWarning`` when a large
        fraction of spikes are silently dropped — either because they
        fall outside the position time window or because they map to
        inactive/out-of-environment bins.  A warning is always emitted
        when **all** spikes are dropped (regardless of threshold).  This
        guards against common unit mismatches (e.g. spike_times in
        milliseconds while times is in seconds).  Set to ``False`` to
        suppress all drop-related warnings. Speed-excluded spikes (via
        ``min_speed``) are intentional exclusions and do NOT trigger this
        warning.

    Returns
    -------
    SpatialRateResult
        Result object containing:

        - ``firing_rate``: Firing rate map in Hz, shape (n_bins,)
        - ``occupancy``: Time in each bin in seconds, shape (n_bins,)
        - ``env``: The environment used
        - ``method``: Method used for smoothing
        - ``bandwidth``: Bandwidth used for smoothing

    See Also
    --------
    compute_spatial_rates : Batch version for multiple neurons
    SpatialRateResult : Result class with convenience methods

    Notes
    -----
    The function uses the binning layer (``_binning.py``) to convert spike
    times to spike counts, then the smoothing layer (``_smoothing.py``) to
    compute the smoothed firing rate.

    **Algorithm**:

    1. Map trajectory positions to spatial bins
    2. Interpolate spike positions from trajectory using spike times
    3. Count spikes in each spatial bin
    4. Compute occupancy (time spent in each bin)
    5. Apply smoothing (method-dependent, see ``_smoothing.py``)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_spatial_rate

    >>> # Create environment from a seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create trajectory timestamps and (sorted) spike times
    >>> times = np.linspace(0, 10, 1000)
    >>> spike_times = np.array([1.0, 2.5, 4.0, 7.5, 8.2])

    >>> # Compute spatial rate
    >>> result = compute_spatial_rate(
    ...     env,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     method="diffusion_kde",
    ...     bandwidth=10.0,
    ... )

    >>> # Access results
    >>> result.firing_rate.shape == (env.n_bins,)
    True
    >>> bool(result.peak_firing_rate() >= 0.0)
    True
    >>> result.peak_location().shape
    (2,)

    >>> # Penalized-Poisson GAM: occupancy as a log-offset, lambda by REML.
    >>> # Returns finite rates everywhere (no NaN in low-occupancy bins).
    >>> glm = compute_spatial_rate(env, spike_times, times, positions, method="glm")
    >>> bool(np.all(np.isfinite(glm.firing_rate)))
    True
    >>> glm.bandwidth is None  # ratio-only param; glm uses penalty/rank instead
    True
    """
    from neurospatial.encoding._backend import (
        SUPPORTED_BACKENDS,
        get_backend_name,
        is_jax_available,
    )
    from neurospatial.encoding._binning import (
        _emit_all_excluded_intervals_warning,
        _resolve_interval_mask,
        bin_spike_train,
        compute_occupancy,
        resolve_speed,
    )
    from neurospatial.encoding._smoothing import (
        _validate_smoothing_parameters,
        smooth_rate_map,
    )
    from neurospatial.encoding._validation import (
        validate_env_fitted,
        validate_spike_times,
        validate_trajectory,
    )

    validate_env_fitted(env, context="compute_spatial_rate")

    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    # Resolve backend (handles "auto" → "numpy" or "jax")
    # This raises ImportError if backend="jax" and JAX is unavailable
    resolved_backend = get_backend_name(backend)

    # Method-specific validation (mutual exclusivity + value domains), in the
    # contract order, then resolve the ratio defaults. For glm the ratio params
    # stay unset (None); for ratio methods bandwidth/min_occupancy resolve to
    # their historical defaults so existing behavior is byte-identical.
    from neurospatial.encoding._smoothing import (
        validate_pooled,
        validate_spatial_method_params,
    )

    penalty, rank = validate_spatial_method_params(
        method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        fill_value=fill_value,
        penalty=penalty,
        rank=rank,
    )
    # ``pooled`` (glm-only, strict bool): shared vs per-unit lambda. pooled=True
    # with a ratio method is the harmless default; pooled=False with one raises.
    validate_pooled(pooled, method)
    # Resolve the ratio defaults. glm ignores bandwidth/min_occupancy (its branch
    # returns before any smoothing and stamps bandwidth=None on the result); for
    # ratio methods this restores the historical defaults byte-for-byte.
    bandwidth = 5.0 if bandwidth is None else bandwidth
    min_occupancy = 0.0 if min_occupancy is None else min_occupancy
    if method != "glm":
        _validate_smoothing_parameters(method, bandwidth)

    # Boundary adapter: accept EITHER a PositionLike (e.g. a pynapple
    # Tsd/TsdFrame exposing .t/.values) OR explicit (times, positions) arrays,
    # normalizing to plain float64 arrays here at the public entry. The array
    # path is unchanged byte-for-byte (plain arrays pass straight through).
    from neurospatial._typing import as_times_positions

    times, positions = as_times_positions(times, positions)

    # Convert inputs to arrays
    spike_times = np.asarray(spike_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    validate_trajectory(times, positions=positions, context="compute_spatial_rate")
    validate_spike_times(spike_times, context="compute_spatial_rate")

    # Resolve the speed gate ONCE so the SAME concrete array feeds both the
    # spike numerator and the occupancy denominator (numerator/denominator
    # alignment by construction). When min_speed is None this returns None and
    # nothing speed-related changes downstream (byte-for-byte unchanged).
    resolved_speed = resolve_speed(times, positions, speed, min_speed)

    # Resolve the FULL interval-valid mask once (max_gap ∪ out-of-bounds-start ∪
    # min_speed) so we can warn ONCE if EVERY interval is excluded (empty rate
    # map), regardless of WHICH gate caused it. Gated by warn_on_drop. Reshape
    # 1-D positions so _resolve_interval_mask sees the canonical 2-D shape.
    if warn_on_drop:
        _positions_2d = positions.reshape(-1, 1) if positions.ndim == 1 else positions
        _interval_mask = _resolve_interval_mask(
            env,
            times,
            _positions_2d,
            speed=resolved_speed,
            min_speed=min_speed,
            max_gap=max_gap,
        )
        _emit_all_excluded_intervals_warning(
            _interval_mask, max_gap=max_gap, min_speed=min_speed, stacklevel=2
        )

    # Bin spike train into spatial bins (always NumPy - CPU/joblib)
    spike_counts = bin_spike_train(
        env,
        spike_times,
        times,
        positions,
        speed=resolved_speed,
        min_speed=min_speed,
        max_gap=max_gap,
        context="compute_spatial_rate",
        warn_on_drop=warn_on_drop,
    )

    # Compute occupancy (always NumPy) using the SAME resolved speed gate and
    # the SAME max_gap, so the numerator and denominator drop identical
    # intervals.
    occupancy = compute_occupancy(
        env,
        times,
        positions,
        speed=resolved_speed,
        min_speed=min_speed,
        max_gap=max_gap,
        context="compute_spatial_rate",
    )

    # method="glm": fit the penalized-Poisson GAM (occupancy as a log-offset) and
    # return finite rates everywhere -- no ratio smoothing / min_occupancy / NaN.
    # The single-neuron counts (n_bins,) become a 1-unit unit-major batch.
    if method == "glm":
        glm_firing_rates, fit = _compute_glm_spatial_rates(
            env,
            spike_counts[np.newaxis, :],
            occupancy,
            penalty=penalty,
            rank=rank,
            pooled=pooled,
            resolved_backend=resolved_backend,
            dtype=np.float64,
        )
        # ``glm_firing_rates`` already has the right array type from the helper
        # (JAX for backend="jax", NumPy otherwise); slice row 0 once, preserving
        # that type -- no redundant round-trip. Use dedicated ``ArrayLike`` output
        # variables so the JAX branch does not reassign the NumPy-typed
        # ``occupancy`` (which would not type-check under a JAX-present run).
        single_firing_rate: ArrayLike
        single_occupancy: ArrayLike = occupancy
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            single_firing_rate = jnp.asarray(glm_firing_rates)[0]
            single_occupancy = jnp.asarray(occupancy, dtype=jnp.float64)
        else:
            single_firing_rate = np.asarray(glm_firing_rates)[0]
        # Singular result: UNWRAP the one-element per-unit vectors (pooled=False)
        # to scalars so ``compute_spatial_rate(pooled=False)`` equals
        # ``compute_spatial_rates([spikes], pooled=False)[0]`` field-for-field.
        # ``pooled=True`` fields are already scalars and pass through untouched.
        return SpatialRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            env=env,
            method=method,
            bandwidth=None,
            coefficients=np.asarray(fit.coefficients)[:, 0],
            penalty=_index_per_unit(fit.penalty, 0),
            penalty_weights=fit.penalty_weights,
            rank=fit.rank,
            deviance=fit.deviance[0],
            converged=fit.converged,
            n_iter=fit.n_iter,
            reml_objective=_index_per_unit(fit.reml_objective, 0),
            reml_at_boundary=_index_per_unit(fit.reml_at_boundary, 0),
            penalty_selected_by_reml=_index_per_unit(fit.penalty_selected_by_reml, 0),
            pooled=fit.pooled,
        )

    # Apply smoothing to compute firing rate
    # When backend="jax", uses JAX for the core rate computation
    firing_rate = smooth_rate_map(
        env,
        spike_counts,
        occupancy,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=resolved_backend,
    )

    # Replace masked/low-occupancy NaN bins with fill_value when requested.
    # Default (None) preserves NaN so existing callers see no behavior change.
    if fill_value is not None:
        firing_rate = _fill_nan(firing_rate, fill_value)

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rate is already JAX from smooth_rate_map)
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        occupancy = jnp.asarray(occupancy, dtype=jnp.float64)

    # Return result
    return SpatialRateResult(
        firing_rate=firing_rate,
        occupancy=occupancy,
        env=env,
        method=method,
        bandwidth=bandwidth,
    )


def compute_spatial_rates(
    env: Environment,
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64] | SpikeTrainsLike,
    times: NDArray[np.float64] | PositionLike,
    positions: NDArray[np.float64] | None = None,
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float | None = None,
    fill_value: float | None = None,
    penalty: float | None = None,
    rank: int | None = None,
    pooled: bool = True,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
    warn_on_drop: bool = True,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> SpatialRatesResult:
    """Compute spatial firing rate maps for multiple neurons.

    This is the batch version of ``compute_spatial_rate()`` that efficiently
    processes multiple neurons with shared trajectory data. It precomputes
    shared quantities (occupancy, position bins, diffusion kernel) once and
    optionally parallelizes spike counting with joblib.

    Parameters
    ----------
    env : Environment
        The spatial environment defining the bin structure. Must be fitted
        (e.g., created via ``Environment.from_samples()``).
    spike_times : sequence of arrays or 2D array
        Spike times for each neuron. Accepted formats:

        - List/tuple of 1D arrays: ``[spikes_0, spikes_1, ...]`` (canonical)
        - 2D array with NaN padding: shape ``(n_neurons, max_spikes)``
        - 1D array (single neuron): wrapped in list automatically
        - A ``SpikeTrainsLike`` group (e.g. a pynapple-``TsGroup``-like object
          iterating per-unit trains and carrying an ``.index`` of unit ids)

        All formats are coerced to per-neuron spike trains via ``as_spike_trains()``.
        When a group carries unit ids and ``unit_ids`` is not passed, those ids
        are threaded into the result's ``unit_ids``.
    times : ndarray, shape (n_samples,), or PositionLike
        Timestamps of trajectory samples in seconds. May instead be a single
        ``PositionLike`` object (exposing ``.t`` and ``.values``, e.g. a
        pynapple ``Tsd`` / ``TsdFrame``) carrying both times and positions, in
        which case ``positions`` must be omitted.
    positions : ndarray, shape (n_samples, n_dims), optional
        Position coordinates at each time sample. NaN values are treated as
        missing data and excluded from occupancy and firing-rate computation;
        callers do not need to pre-filter tracking dropouts. Omit only when
        ``times`` is a ``PositionLike`` object carrying the positions.
    method : {"diffusion_kde", "gaussian_kde", "binned", "glm"}, \
default="diffusion_kde"
        Estimator to use. See ``compute_spatial_rate()`` for details. In addition
        to the three ratio methods, ``method="glm"`` fits a penalized-Poisson GAM
        (occupancy as a log-offset, ``λ`` by REML) and returns finite rates
        everywhere; it is tuned with ``penalty`` / ``rank`` (mutually exclusive
        with ``bandwidth`` / ``min_occupancy`` / ``fill_value``). This function is
        the batched entry point the decoder consumes; ``method="glm"`` flows
        through both the decoder (``decode_session`` / ``BayesianDecoder``) and
        NWB persistence (``write_spatial_rates`` round-trips the GAM diagnostics).
        ``diffusion_kde`` and ``binned`` are
        matrix-free (O(n_bins·rank) per neuron); ``glm`` also avoids a dense
        O(n_bins²) kernel but its penalized-Poisson fit is **not** linear in
        ``rank`` — each Newton step builds and solves a per-unit (r, r) Hessian
        (≈ O(n_units·n_bins·rank²) + O(n_units·rank³)), repeated across Newton
        iterations and REML λ candidates, so keep ``rank`` modest for large
        populations (see ``compute_spatial_rate`` for details). Only
        ``gaussian_kde`` builds a dense O(n_bins²) kernel.
    bandwidth : float | None, default=None
        (Ratio methods only.) Smoothing bandwidth in the same units as bin_size;
        ``None`` resolves to ``5.0``. Mutually exclusive with ``method="glm"``.
    min_occupancy : float | None, default=None
        (Ratio methods only.) Minimum occupancy (seconds) for a bin to be
        included; ``None`` resolves to ``0.0`` (no masking). Mutually exclusive
        with ``method="glm"``.
    fill_value : float | None, default=None
        (Ratio methods only.) Value used to replace NaN bins (masked/low-occupancy
        bins produced by ``min_occupancy``). When ``None`` (the default), NaN is
        preserved so existing callers see no behavior change. Pass
        ``fill_value=0.0`` for the recommended decoding golden path: zero-rate maps
        compose directly with
        :func:`~neurospatial.decoding.posterior.decode_position` without manual NaN
        scrubbing. ``occupancy`` is unaffected, so callers can still recover which
        bins were masked via ``result.occupancy < min_occupancy``. Mutually
        exclusive with ``method="glm"`` (glm rates are already finite).
    penalty : float | None, default=None
        (``method="glm"`` only.) Fixed smoothness penalty ``λ`` (≥ 0). ``None``
        selects ``λ`` by REML, which raises ``ValueError`` on pathologically
        under-sampled data where no ``λ`` yields a converged fit (see
        ``compute_spatial_rate``). Mutually exclusive with the ratio methods.
    rank : int | None, default=None
        (``method="glm"`` only.) Requested rank of the reduced-rank penalty basis
        (≥ 1). ``None`` uses the module default cap; an out-of-range value is
        clamped (never rejected) to the effective rank reported via
        ``result.rank``. Mutually exclusive with the ratio methods.
    pooled : bool, default=True
        (``method="glm"`` only; strict ``bool``.) One **shared** smoothing penalty
        ``λ`` for the whole population (``True``, the default) or an **independent
        per-unit** ``λ`` (``False``). Under ``pooled=False`` the REML search runs
        once per informative unit (cost ~ one REML per neuron), so
        ``result.penalty`` / ``reml_objective`` / ``reml_at_boundary`` become
        ``(n_units,)`` vectors and ``penalty_selected_by_reml`` a per-unit mask;
        zero-spike units fall back to the pooled ``λ`` over the informative units
        (``penalty_selected_by_reml=False``, ``reml_objective=nan``). A fixed
        ``penalty`` beats ``pooled`` (scalar ``λ``); ``pooled`` is a no-op at
        ``penalty_rank == 0`` or when no unit spikes. ``pooled=False`` with a ratio
        method raises ``ValueError``.
    speed : ndarray, shape (n_samples,), optional
        Precomputed instantaneous speed at each trajectory sample (physical
        units / second). Only used when ``min_speed`` is set; auto-derived from
        the trajectory when ``None``. See ``min_speed``. A wrong-length array
        raises ``ValueError``.
    min_speed : float, optional
        Minimum speed threshold (physical units / second). When set, low-speed
        periods are excluded from BOTH the (shared) occupancy denominator AND
        every per-neuron spike numerator using ONE shared per-interval speed
        gate, so firing rates stay correct. When ``None`` (default) NO speed
        filtering is applied and the output is byte-for-byte identical to
        before. Auto-speed uses a FORWARD difference (matching the
        ``time_allocation="start"`` occupancy semantics):
        ``speed[k] = ||positions[k+1] - positions[k]||_2 / (times[k+1] -
        times[k])`` with ``speed[n-1] = speed[n-2]``; pass an explicit ``speed``
        for geodesic / linearized-track environments.
    max_gap : float | None, default=0.5
        Maximum trajectory time gap in seconds. Intervals with
        ``dt > max_gap`` (large tracking gaps) are excluded from BOTH the
        shared occupancy denominator AND every per-neuron spike numerator
        using ONE shared per-interval mask. This default matches
        ``env.occupancy``'s own default, so occupancy is unchanged.

        .. note::
           **Behavior change (correctness fix).** Spikes inside intervals
           longer than ``max_gap`` or inside intervals whose start sample is
           out of bounds are now excluded from spike counts, matching
           occupancy. Previously such spikes were counted while their time was
           excluded from the denominator, inflating the rate. This changes
           rate maps for sessions with large tracking gaps / out-of-bounds
           excursions. Pass ``max_gap=None`` to disable gap gating on BOTH
           sides (pre-fix behavior, still aligned).
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    backend : {"numpy", "jax", "auto"}, default="numpy"
        Computation backend for rate map smoothing:

        - ``"numpy"``: Use NumPy for all computations. Works everywhere.
        - ``"jax"``: Use JAX for rate computation. Requires JAX installation.
          Enables GPU acceleration and JAX transformations (jit, grad).
        - ``"auto"``: Use JAX if available, otherwise NumPy.

        Note: Binning operations (spike counting, occupancy) always use NumPy.
        Only the smoothing/rate computation uses the selected backend. For
        ``method="glm"``, a resolved ``jax`` backend runs the penalized-Poisson
        fit + REML through an optional **float32** JAX mirror of the NumPy/SciPy
        core (``backend="jax"`` requires the ``jax`` extra, like the ratio
        methods; ``"auto"`` uses it when available and otherwise the NumPy core).
        The float32 mirror matches the float64 core to ~1e-6 at a fixed penalty (a
        touch looser under automatic REML, which picks a slightly different
        ``lambda``) and is markedly faster on populations. The returned
        diagnostics stay float64 either way.
    warn_on_drop : bool, default=True
        If ``True`` (the default), emit a single ``UserWarning`` (per drop
        cause) when a large fraction of spikes are silently dropped across
        all neurons.  The warning is computed in the main process from
        aggregate statistics, so it fires exactly once even when
        ``n_jobs != 1`` (joblib worker warnings are commonly swallowed).
        Set to ``False`` to suppress all drop-related warnings.
    dtype : {np.float32, np.float64}, default=np.float64
        Storage dtype of the returned ``(n_units, n_bins)`` rate-map array.
        ``np.float32`` halves the stored rate-map array. The rate computation
        is still performed in float64 and only the final result is cast, so
        float32 values match float64 within float32 tolerance.
        ``decode_session`` / ``decode_session_summary`` now accept their own
        ``dtype`` parameter (default float64) that honors float32 end-to-end --
        the encoding-model working set AND the posterior -- so
        ``decode_session(dtype=np.float32)`` halves the decode working set on
        the golden path. Default ``np.float64`` leaves every existing caller
        byte-for-byte unchanged. Any other dtype raises ``ValueError``.
    unit_ids : ndarray or sequence, optional
        Per-unit identity labels (integers or strings), one per neuron in
        the same order as ``spike_times``. Stored on the result's
        ``unit_ids`` field and stamped onto each child's ``unit_id`` when
        indexing/iterating. Defaults to ``np.arange(n_neurons)``. A
        wrong-length value raises ``ValueError``.

    Returns
    -------
    SpatialRatesResult
        Result object containing:

        - ``firing_rates``: Firing rate maps, shape ``(n_neurons, n_bins)``
        - ``occupancy``: Time in each bin in seconds, shape ``(n_bins,)``
        - ``env``: The environment used
        - ``method``: Method used for smoothing
        - ``bandwidth``: Bandwidth used for smoothing

        The result supports iteration: ``for single in result: ...``
        and indexing: ``single = result[0]``.

    See Also
    --------
    compute_spatial_rate : Single-neuron version
    SpatialRatesResult : Result class with batch methods

    Notes
    -----
    **Efficiency advantages over calling ``compute_spatial_rate()`` in a loop**:

    1. Occupancy is computed once and shared across all neurons
    2. Diffusion kernel (for ``diffusion_kde`` method) is computed once
    3. Position-to-bin mapping is done once
    4. Spike binning can be parallelized with joblib

    **When to use batch vs single**:

    - **Batch** (this function): Processing 3+ neurons, or any case where
      efficiency matters. The overhead of precomputing shared quantities
      is amortized over multiple neurons.
    - **Single** (``compute_spatial_rate``): Processing 1-2 neurons, or when
      you need fine-grained control over individual neurons.

    **Memory (``dtype``).** Passing ``dtype=np.float32`` halves the stored
    ``(n_units, n_bins)`` rate-map array. The rate computation (GEMM /
    division) is still done in float64 and only the final result is cast to
    ``dtype``, so values match the float64 default within float32 tolerance.
    ``decode_session`` / ``decode_session_summary`` now accept their own
    ``dtype`` parameter (default float64) that honors float32 end-to-end --
    the encoding-model working set AND the posterior -- so
    ``decode_session(dtype=np.float32)`` halves the decode working set on the
    golden path.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_spatial_rates

    >>> # Create environment from a seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create trajectory timestamps
    >>> times = np.linspace(0, 10, 1000)

    >>> # Spike times for 3 neurons (each sorted ascending)
    >>> spike_times = [
    ...     np.array([1.0, 2.5, 4.0]),  # Neuron 0
    ...     np.array([0.5, 1.5, 2.5, 3.5]),  # Neuron 1
    ...     np.array([5.0, 8.0]),  # Neuron 2
    ... ]

    >>> # Compute spatial rates for all neurons
    >>> result = compute_spatial_rates(
    ...     env,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     method="diffusion_kde",
    ...     bandwidth=10.0,
    ... )

    >>> # Access results
    >>> len(result)
    3
    >>> result.firing_rates.shape == (3, env.n_bins)
    True

    >>> # Iterate over neurons
    >>> peaks = [round(float(single.peak_firing_rate()), 2) for single in result]
    >>> len(peaks)
    3

    >>> # Per-unit scalar summary (one row per unit)
    >>> summary = result.summary_table()
    >>> len(summary)
    3
    >>> # Dense per-bin frame (one row per (unit, bin))
    >>> df = result.to_dataframe()
    >>> len(df) == 3 * env.n_bins
    True

    >>> # Use 2D array with NaN padding
    >>> spike_times_2d = np.array(
    ...     [
    ...         [0.1, 0.5, 1.0, np.nan],
    ...         [0.2, 0.3, 0.8, 1.2],
    ...     ]
    ... )
    >>> result2 = compute_spatial_rates(env, spike_times_2d, times, positions)
    >>> len(result2)
    2
    """
    from neurospatial.encoding._backend import (
        SUPPORTED_BACKENDS,
        get_backend_name,
        is_jax_available,
    )
    from neurospatial.encoding._binning import (
        _emit_all_excluded_intervals_warning,
        _resolve_interval_mask,
        bin_spike_trains,
        resolve_speed,
    )
    from neurospatial.encoding._smoothing import (
        _validate_smoothing_parameters,
        smooth_rate_maps_batch,
    )
    from neurospatial.encoding._spikes import as_spike_trains_with_ids
    from neurospatial.encoding._validation import (
        validate_env_fitted,
        validate_spike_times,
        validate_trajectory,
    )

    validate_env_fitted(env, context="compute_spatial_rates")

    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    # Resolve backend (handles "auto" → "numpy" or "jax")
    # This raises ImportError if backend="jax" and JAX is unavailable
    resolved_backend = get_backend_name(backend)

    # Validate dtype: only single/double precision rate maps are supported.
    # Wrap the parse so an unparseable dtype string (e.g. "bogus") raises this
    # clean ValueError naming `dtype`, not a raw NumPy
    # ``TypeError: data type 'bogus' not understood``.
    _dtype_msg = (
        f"dtype must be np.float32 or np.float64, got {dtype!r}. "
        "Only single- and double-precision rate maps are supported "
        "(float32 halves the (n_units, n_bins) storage and the downstream "
        "decode working set)."
    )
    try:
        _resolved_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(_dtype_msg) from exc
    if _resolved_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(_dtype_msg)
    # Normalize to the canonical numpy scalar type for downstream casts.
    dtype = np.float32 if _resolved_dtype == np.dtype(np.float32) else np.float64

    # Method-specific validation (mutual exclusivity + value domains), then
    # resolve the ratio defaults. glm ignores bandwidth/min_occupancy (its branch
    # below returns before any smoothing and stamps bandwidth=None on the result);
    # for ratio methods this restores the historical defaults byte-for-byte.
    from neurospatial.encoding._smoothing import (
        validate_pooled,
        validate_spatial_method_params,
    )

    penalty, rank = validate_spatial_method_params(
        method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        fill_value=fill_value,
        penalty=penalty,
        rank=rank,
    )
    # ``pooled`` (glm-only, strict bool): shared vs per-unit lambda. pooled=True
    # with a ratio method is the harmless default; pooled=False with one raises.
    validate_pooled(pooled, method)
    bandwidth = 5.0 if bandwidth is None else bandwidth
    min_occupancy = 0.0 if min_occupancy is None else min_occupancy
    if method != "glm":
        _validate_smoothing_parameters(method, bandwidth)

    # Normalize spike times to canonical list-of-arrays format, surfacing any
    # unit ids the spikes object carries (a SpikeTrainsLike group, e.g. a
    # pynapple-TsGroup-like). Plain-array/sequence inputs return `None` ids, so
    # the trains are byte-for-byte identical to `as_spike_trains(spike_times)`.
    spike_times_list, extracted_unit_ids = as_spike_trains_with_ids(spike_times)
    n_neurons = len(spike_times_list)

    # Resolve and validate per-unit identity labels (defaults to arange). An
    # explicit `unit_ids=` always wins; otherwise the ids extracted from the
    # spikes object are threaded through so identity is not silently dropped.
    from neurospatial._results import resolve_unit_ids

    effective_unit_ids = unit_ids if unit_ids is not None else extracted_unit_ids
    resolved_unit_ids = resolve_unit_ids(
        effective_unit_ids, n_neurons, context="compute_spatial_rates"
    )

    # Boundary adapter: accept EITHER a PositionLike (e.g. a pynapple
    # Tsd/TsdFrame) OR explicit (times, positions) arrays. Array path unchanged.
    from neurospatial._typing import as_times_positions

    times, positions = as_times_positions(times, positions)

    # Convert inputs to arrays
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    validate_trajectory(times, positions=positions, context="compute_spatial_rates")
    for i, st in enumerate(spike_times_list):
        validate_spike_times(st, context=f"compute_spatial_rates (neuron {i})")

    # Resolve the shared speed gate ONCE here (raises on speed-without-min_speed,
    # mirroring env.occupancy). Resolving once also lets us warn a single time
    # if min_speed excludes ALL intervals — instead of per-neuron in the batch
    # path below. The resolved array is forwarded so bin_spike_trains does not
    # re-derive it.
    resolved_speed = resolve_speed(times, positions, speed, min_speed)

    # Resolve the FULL interval-valid mask once (max_gap ∪ out-of-bounds-start ∪
    # min_speed) so we can warn ONCE for the whole batch if EVERY interval is
    # excluded (empty rate maps), regardless of WHICH gate caused it — not once
    # per neuron. Gated by warn_on_drop. Reshape 1-D positions to canonical 2-D.
    if warn_on_drop:
        _positions_2d = positions.reshape(-1, 1) if positions.ndim == 1 else positions
        _interval_mask = _resolve_interval_mask(
            env,
            times,
            _positions_2d,
            speed=resolved_speed,
            min_speed=min_speed,
            max_gap=max_gap,
        )
        _emit_all_excluded_intervals_warning(
            _interval_mask, max_gap=max_gap, min_speed=min_speed, stacklevel=2
        )

    # method="glm": fit the penalized-Poisson GAM (occupancy as a log-offset).
    # Handles the no-neurons case too (fit_mrf_gam returns an (r_eff, 0) fit), so
    # it sits before the ratio no-neurons short-circuit and returns early, leaving
    # the ratio path below untouched.
    if method == "glm":
        from neurospatial.encoding._binning import (
            bin_spike_trains as _bin_spike_trains,
        )
        from neurospatial.encoding._binning import (
            compute_occupancy as _compute_occupancy,
        )

        if n_neurons == 0:
            spike_counts = np.zeros((0, env.n_bins), dtype=np.float64)
            occupancy = _compute_occupancy(
                env,
                times,
                positions,
                speed=resolved_speed,
                min_speed=min_speed,
                max_gap=max_gap,
                context="compute_spatial_rates",
            )
        else:
            spike_counts, occupancy = _bin_spike_trains(
                env,
                spike_times_list,
                times,
                positions,
                speed=resolved_speed,
                min_speed=min_speed,
                max_gap=max_gap,
                n_jobs=n_jobs,
                warn_on_drop=warn_on_drop,
            )

        glm_firing_rates, fit = _compute_glm_spatial_rates(
            env,
            spike_counts,
            occupancy,
            penalty=penalty,
            rank=rank,
            pooled=pooled,
            resolved_backend=resolved_backend,
            dtype=dtype,
        )
        # Dedicated ArrayLike output var so the JAX branch does not reassign the
        # NumPy-typed ``occupancy`` (which would not type-check JAX-present).
        batch_occupancy: ArrayLike = occupancy
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            batch_occupancy = jnp.asarray(occupancy, dtype=jnp.float64)
        # The per-unit lambda fields (``penalty`` / ``reml_objective`` /
        # ``reml_at_boundary`` vectors, ``penalty_selected_by_reml`` mask) carry
        # straight through from the fit; they are scalar/None under pooled=True.
        return SpatialRatesResult(
            firing_rates=glm_firing_rates,
            occupancy=batch_occupancy,
            env=env,
            method=method,
            bandwidth=None,
            unit_ids=resolved_unit_ids,
            # dtype governs the (n_units, n_bins) rate-map storage only. The GLM
            # diagnostics are the float64 fit result and are kept float64 -- so
            # they do not lose precision (deviance/coefficients) and rates[i]
            # matches compute_spatial_rate, whose diagnostics are always float64.
            coefficients=fit.coefficients,
            penalty=fit.penalty,
            penalty_weights=fit.penalty_weights,
            rank=fit.rank,
            deviance=fit.deviance,
            converged=fit.converged,
            n_iter=fit.n_iter,
            reml_objective=fit.reml_objective,
            reml_at_boundary=fit.reml_at_boundary,
            penalty_selected_by_reml=fit.penalty_selected_by_reml,
            pooled=fit.pooled,
        )

    # Handle edge case: no neurons
    # Still compute occupancy from trajectory (occupancy is independent of neural data)
    if n_neurons == 0:
        from neurospatial.encoding._binning import compute_occupancy

        # Use compute_occupancy which handles 1D position reshaping
        occupancy = compute_occupancy(
            env,
            times,
            positions,
            speed=resolved_speed,
            min_speed=min_speed,
            max_gap=max_gap,
            context="compute_spatial_rates",
        )

        # Convert to JAX if needed
        firing_rates_result: ArrayLike = np.empty((0, env.n_bins), dtype=dtype)
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            jnp_dtype = jnp.float32 if dtype is np.float32 else jnp.float64
            firing_rates_result = jnp.asarray(firing_rates_result, dtype=jnp_dtype)
            occupancy = jnp.asarray(occupancy, dtype=jnp.float64)

        return SpatialRatesResult(
            firing_rates=firing_rates_result,
            occupancy=occupancy,
            env=env,
            method=method,
            bandwidth=bandwidth,
            unit_ids=resolved_unit_ids,
        )

    # Bin spike trains and compute occupancy (always NumPy - CPU/joblib)
    # bin_spike_trains returns (spike_counts, occupancy)
    spike_counts, occupancy = bin_spike_trains(
        env,
        spike_times_list,
        times,
        positions,
        speed=resolved_speed,
        min_speed=min_speed,
        max_gap=max_gap,
        n_jobs=n_jobs,
        warn_on_drop=warn_on_drop,
    )

    # Apply batch smoothing to compute firing rates
    # When backend="jax", uses JAX for the core rate computation
    firing_rates = smooth_rate_maps_batch(
        env,
        spike_counts,
        occupancy,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=resolved_backend,
        dtype=dtype,
    )

    # Replace masked/low-occupancy NaN bins with fill_value when requested.
    # Default (None) preserves NaN so existing callers see no behavior change.
    if fill_value is not None:
        firing_rates = _fill_nan(firing_rates, fill_value)

    # Defensive final cast to the requested dtype. Under NEP 50,
    # np.where(mask, 0.0, f32_array) keeps float32, but this guarantees the
    # stored dtype regardless of the fill_value / backend path.
    if isinstance(firing_rates, np.ndarray):
        firing_rates = firing_rates.astype(dtype, copy=False)
    elif is_jax_available():
        import jax.numpy as jnp

        jnp_dtype = jnp.float32 if dtype is np.float32 else jnp.float64
        firing_rates = jnp.asarray(firing_rates, dtype=jnp_dtype)

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rates is already JAX from smooth_rate_maps_batch)
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        occupancy = jnp.asarray(occupancy, dtype=jnp.float64)

    # Return result
    return SpatialRatesResult(
        firing_rates=firing_rates,
        occupancy=occupancy,
        env=env,
        method=method,
        bandwidth=bandwidth,
        unit_ids=resolved_unit_ids,
    )


# ==============================================================================
# Directional Place Fields
# ==============================================================================


@dataclass(frozen=True, repr=False)
class DirectionalPlaceFields(ResultMixin):
    """Container for direction-conditioned place field results.

    Stores firing rate maps computed separately for different movement
    directions or trial types. This enables analysis of directional
    tuning in place cells.

    Attributes
    ----------
    firing_rates : Mapping[str, NDArray[np.float64]]
        Dictionary mapping direction labels (e.g., "A→B", "forward") to
        firing rate arrays. Each array has shape (n_bins,) matching the
        environment's bin structure.
    occupancy : Mapping[str, NDArray[np.float64]]
        Per-direction occupancy (time spent in each bin) in seconds.
        Shape (n_bins,). Same keys as ``firing_rates``.
    env : Environment
        Spatial environment used to compute the per-direction fields.
        Shared across all labels (the per-direction split is over time,
        not over space).
    labels : tuple[str, ...]
        Tuple of direction labels in iteration order. Preserves the order
        in which directions were processed, enabling reproducible iteration.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> env = Environment.from_samples(
    ...     np.linspace(0, 10, 11)[:, None], bin_size=1.0
    ... )  # doctest: +SKIP
    >>> firing_rates = {
    ...     "home→goal": np.array([1.0, 2.0, 3.0]),
    ...     "goal→home": np.array([3.0, 2.0, 1.0]),
    ... }
    >>> occupancy = {
    ...     "home→goal": np.array([1.0, 1.0, 1.0]),
    ...     "goal→home": np.array([1.0, 1.0, 1.0]),
    ... }
    >>> result = DirectionalPlaceFields(  # doctest: +SKIP
    ...     firing_rates=firing_rates,
    ...     occupancy=occupancy,
    ...     env=env,
    ...     labels=("home→goal", "goal→home"),
    ... )

    See Also
    --------
    compute_directional_place_fields : Compute directional place fields from spike data.
    """

    firing_rates: Mapping[str, NDArray[np.float64]]
    occupancy: Mapping[str, NDArray[np.float64]]
    env: Environment
    labels: tuple[str, ...]

    def correlation(self, label_a: str, label_b: str) -> float:
        """Pearson correlation between two directions' rate maps.

        Quantifies how similar the place-field map is between two movement
        directions (or trial types). A correlation near ``1.0`` means the cell
        fires in the same locations regardless of direction; values near
        ``0`` (or negative) indicate direction-specific tuning.

        Bins where either map is NaN are excluded pairwise before the
        correlation is computed.

        Parameters
        ----------
        label_a, label_b : str
            Direction labels to compare. Must be present in ``labels``.

        Returns
        -------
        float
            Pearson correlation coefficient in ``[-1, 1]``. Returns ``nan``
            if fewer than two finite overlapping bins exist or if either map
            has zero variance over the overlap.

        Raises
        ------
        KeyError
            If ``label_a`` or ``label_b`` is not a known direction label.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> rate = np.linspace(1.0, 5.0, n)
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={"fwd": rate, "rev": rate.copy()},
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ...     labels=("fwd", "rev"),
        ... )
        >>> bool(np.isclose(result.correlation("fwd", "rev"), 1.0))
        True
        """
        for label in (label_a, label_b):
            if label not in self.firing_rates:
                raise KeyError(
                    f"Unknown direction label {label!r}. "
                    f"Known labels: {tuple(self.firing_rates)}."
                )
        a = _to_numpy(self.firing_rates[label_a]).ravel()
        b = _to_numpy(self.firing_rates[label_b]).ravel()
        finite = np.isfinite(a) & np.isfinite(b)
        if int(finite.sum()) < 2:
            return float("nan")
        a, b = a[finite], b[finite]
        if a.std() == 0.0 or b.std() == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def directionality_index(self, label_a: str, label_b: str) -> float:
        """Per-bin directionality index between two directions.

        Returns the mean absolute normalized rate difference between the two
        directions across bins:

        .. math::

            \\mathrm{DI} = \\mathrm{mean}_i
                \\frac{|r^a_i - r^b_i|}{r^a_i + r^b_i}

        Values near ``0`` indicate direction-independent firing; values near
        ``1`` indicate strongly direction-selective firing. Bins where either
        map is NaN, or where both rates are zero, are excluded.

        Parameters
        ----------
        label_a, label_b : str
            Direction labels to compare. Must be present in ``labels``.

        Returns
        -------
        float
            Mean directionality index in ``[0, 1]``. Returns ``nan`` if no
            bin has a positive summed rate.

        Raises
        ------
        KeyError
            If ``label_a`` or ``label_b`` is not a known direction label.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> rate = np.linspace(1.0, 5.0, n)
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={"fwd": rate, "rev": rate.copy()},
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ...     labels=("fwd", "rev"),
        ... )
        >>> bool(np.isclose(result.directionality_index("fwd", "rev"), 0.0))
        True
        """
        for label in (label_a, label_b):
            if label not in self.firing_rates:
                raise KeyError(
                    f"Unknown direction label {label!r}. "
                    f"Known labels: {tuple(self.firing_rates)}."
                )
        a = _to_numpy(self.firing_rates[label_a]).ravel()
        b = _to_numpy(self.firing_rates[label_b]).ravel()
        total = a + b
        valid = np.isfinite(a) & np.isfinite(b) & (total > 0.0)
        if not valid.any():
            return float("nan")
        di = np.abs(a[valid] - b[valid]) / total[valid]
        return float(np.mean(di))

    def summary(self) -> dict[str, Any]:
        """Scalar headline metrics across directions.

        Returns
        -------
        dict
            Mapping with ``n_directions`` (int), ``n_bins`` (int), and one
            ``peak_<label>`` entry per direction giving that direction's peak
            firing rate (Hz). When exactly two directions are present, a
            ``correlation`` entry (Pearson r between the two maps) is also
            included.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={
        ...         "fwd": np.linspace(1.0, 5.0, n),
        ...         "rev": np.linspace(5.0, 1.0, n),
        ...     },
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ...     labels=("fwd", "rev"),
        ... )
        >>> result.summary()["n_directions"]
        2
        """
        out: dict[str, Any] = {
            "n_directions": len(self.labels),
            "n_bins": int(self.env.n_bins),
        }
        for label in self.labels:
            rate = _to_numpy(self.firing_rates[label])
            out[f"peak_{label}"] = float(np.nanmax(rate))
        if len(self.labels) == 2:
            out["correlation"] = self.correlation(self.labels[0], self.labels[1])
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy/long-form table: one row per (direction, bin).

        Stacks the per-direction rate maps into long form with a ``direction``
        identifier column, so directional results ``pandas.concat`` cleanly
        with other tidy result tables. Bin-center coordinates are emitted as
        ``coord_0``, ``coord_1``, ... columns.

        Returns
        -------
        pandas.DataFrame
            Long-form table with columns ``direction`` (str), ``bin`` (int),
            ``coord_0`` ... (float), ``firing_rate`` (float, Hz), and
            ``occupancy`` (float, seconds).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={
        ...         "fwd": np.linspace(1.0, 5.0, n),
        ...         "rev": np.linspace(5.0, 1.0, n),
        ...     },
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ...     labels=("fwd", "rev"),
        ... )
        >>> df = result.to_dataframe()
        >>> sorted(df["direction"].unique())
        ['fwd', 'rev']
        >>> len(df) == 2 * n
        True
        """
        import pandas as pd

        bin_centers = np.asarray(self.env.bin_centers)
        if bin_centers.ndim == 1:
            bin_centers = bin_centers[:, None]
        n_bins, n_dims = bin_centers.shape

        frames: list[pd.DataFrame] = []
        for label in self.labels:
            rate = _to_numpy(self.firing_rates[label]).ravel()
            occ = _to_numpy(self.occupancy[label]).ravel()
            data: dict[str, Any] = {
                "direction": np.repeat(label, n_bins),
                "bin": np.arange(n_bins, dtype=np.int64),
            }
            for d in range(n_dims):
                data[f"coord_{d}"] = bin_centers[:, d]
            data["firing_rate"] = rate
            data["occupancy"] = occ
            frames.append(pd.DataFrame(data))
        if not frames:
            # No labelled directions (e.g. compute_directional_place_fields
            # excluded every "other" sample, leaving labels == ()). Return an
            # empty frame with the documented column schema instead of letting
            # pd.concat([]) raise "No objects to concatenate".
            empty: dict[str, NDArray[Any]] = {
                "direction": np.array([], dtype=object),
                "bin": np.array([], dtype=np.int64),
            }
            for d in range(n_dims):
                empty[f"coord_{d}"] = np.array([], dtype=np.float64)
            empty["firing_rate"] = np.array([], dtype=np.float64)
            empty["occupancy"] = np.array([], dtype=np.float64)
            return pd.DataFrame(empty)
        return pd.concat(frames, ignore_index=True)

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Overlay the per-direction firing rate maps on a single axis.

        Plots each direction's firing rate against bin index as one line,
        producing one artist per direction. This is most informative for
        1-D / linearized environments where bin index maps to track position.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure and axes are created.
        **kwargs
            Additional keyword arguments forwarded to ``ax.plot`` for each
            direction's line.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the overlay (one line per direction).

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")  # non-interactive backend for doctest
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={
        ...         "fwd": np.linspace(1.0, 5.0, n),
        ...         "rev": np.linspace(5.0, 1.0, n),
        ...     },
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ...     labels=("fwd", "rev"),
        ... )
        >>> ax = result.plot()
        >>> len(ax.get_lines())
        2
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        for label in self.labels:
            rate = _to_numpy(self.firing_rates[label]).ravel()
            ax.plot(np.arange(rate.shape[0]), rate, label=str(label), **kwargs)

        ax.set_xlabel("Bin index")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_title("Directional place fields")
        ax.legend()
        return ax


def _subset_spikes_by_time_mask(
    times: NDArray[np.float64],
    spike_times: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Subset spike times by a boolean mask over trajectory times.

    Extracts spikes that fall within the time ranges defined by contiguous
    True segments in the mask. Uses binary search (searchsorted) for
    efficient O(log n) spike slicing per segment.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_timepoints,)
        Timestamps of trajectory samples (seconds). Must be sorted.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Timestamps of spike occurrences (seconds). Must be sorted.
    mask : NDArray[np.bool_], shape (n_timepoints,)
        Boolean mask indicating which timepoints to include.
        Contiguous True segments define time ranges for spike inclusion.

    Returns
    -------
    times_sub : NDArray[np.float64]
        Subset of times where mask is True. Same as ``times[mask]``.
    spike_times_sub : NDArray[np.float64]
        Spikes that fall within the time ranges of contiguous True segments.
        Boundaries are inclusive: spikes at segment start/end are included.

    Notes
    -----
    For each contiguous segment of True values in mask:
    - ``t_start = times[segment_first_index]``
    - ``t_end = times[segment_last_index]``
    - Spikes in ``[t_start, t_end]`` (inclusive) are selected

    This function is designed for conditioning place field analysis on
    subsets of the trajectory (e.g., by movement direction, trial type).

    Examples
    --------
    >>> import numpy as np
    >>> times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> spike_times = np.array([0.5, 1.5, 2.5, 3.5])
    >>> mask = np.array([False, True, True, False, False])
    >>> times_sub, spikes_sub = _subset_spikes_by_time_mask(times, spike_times, mask)
    >>> times_sub
    array([1., 2.])
    >>> spikes_sub
    array([1.5])
    """
    # Fast path: empty mask
    if not np.any(mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Get indices where mask is True
    true_indices = np.where(mask)[0]

    # Find contiguous segments by looking for gaps > 1
    # diff > 1 indicates a break in contiguity
    if len(true_indices) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Find segment boundaries: where consecutive indices are not adjacent
    breaks = np.where(np.diff(true_indices) > 1)[0] + 1
    segment_starts = np.concatenate([[0], breaks])
    segment_ends = np.concatenate([breaks, [len(true_indices)]])

    # Fast path: empty spike train
    if len(spike_times) == 0:
        return times[mask], np.array([], dtype=np.float64)

    # Collect spikes from each segment
    spike_slices = []

    for seg_start_idx, seg_end_idx in zip(segment_starts, segment_ends, strict=True):
        # Get the actual time indices for this segment
        first_time_idx = true_indices[seg_start_idx]
        last_time_idx = true_indices[seg_end_idx - 1]

        # Get time boundaries
        t_start = times[first_time_idx]
        t_end = times[last_time_idx]

        # Use searchsorted for O(log n) spike slicing
        # side="left" for t_start: include spikes at exactly t_start
        # side="right" for t_end: include spikes at exactly t_end
        spike_start = np.searchsorted(spike_times, t_start, side="left")
        spike_end = np.searchsorted(spike_times, t_end, side="right")

        if spike_start < spike_end:
            spike_slices.append(spike_times[spike_start:spike_end])

    # Concatenate all spike slices
    if spike_slices:
        spike_times_sub = np.concatenate(spike_slices)
    else:
        spike_times_sub = np.array([], dtype=np.float64)

    return times[mask], spike_times_sub


def compute_directional_place_fields(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    direction_labels: NDArray[np.object_],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
) -> DirectionalPlaceFields:
    """Compute place fields conditioned on movement direction or trial type.

    Separates trajectory data by direction labels and computes independent
    place fields for each direction. This enables analysis of directional
    tuning in place cells, where firing rates differ based on which way
    the animal is moving through a location.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Timestamps of spike occurrences (seconds).
    times : NDArray[np.float64], shape (n_timepoints,)
        Timestamps of trajectory samples (seconds). Must be sorted.
    positions : NDArray[np.float64], shape (n_timepoints, n_dims) or (n_timepoints,)
        Position trajectory. For 1D, can be shape (n_timepoints,) or (n_timepoints, 1).
    direction_labels : NDArray[object], shape (n_timepoints,)
        Direction label for each timepoint. Each label is a hashable string
        (e.g., "A→B", "forward", "CW"). The special label "other" is excluded
        from results, allowing unlabeled periods to be ignored.
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Estimation method passed through to the place-field helper.
    bandwidth : float, default=5.0
        Smoothing bandwidth in environment units (e.g., cm).
    min_occupancy : float, default=0.0
        Minimum occupancy threshold in seconds. Bins below this threshold are
        set to NaN.

    Returns
    -------
    DirectionalPlaceFields
        Container with:
        - ``firing_rates``: Mapping from direction label to firing rate array (n_bins,)
        - ``occupancy``: Mapping from direction label to per-bin occupancy (n_bins,)
        - ``env``: Spatial environment shared across labels
        - ``labels``: Tuple of direction labels in iteration order

    Raises
    ------
    ValueError
        If ``direction_labels`` length doesn't match ``times`` length.
    ValueError
        If ``bandwidth`` is not positive.

    See Also
    --------
    compute_spatial_rate : Compute single (non-directional) spatial rate.

    Notes
    -----
    The "other" label is reserved for timepoints that should be excluded from
    analysis (e.g., inter-trial intervals, stationary periods). Any timepoints
    with label "other" are ignored when computing fields.

    For each unique non-"other" label, this function:
    1. Creates a boolean mask for timepoints with that label
    2. Extracts the trajectory and spikes within those masked periods
    3. Calls ``compute_spatial_rate`` on the subset
    4. Stores the resulting field in the output mapping

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_directional_place_fields
    >>>
    >>> # Create environment and trajectory (seeded for reproducibility)
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> times = np.linspace(0, 100, 1000)
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>>
    >>> # Create directional labels (first half forward, second half backward)
    >>> labels = np.array(["forward"] * 500 + ["backward"] * 500, dtype=object)
    >>> spike_times = np.sort(rng.uniform(0, 100, 50))  # spikes must be sorted
    >>>
    >>> # Compute directional place fields
    >>> result = compute_directional_place_fields(
    ...     env, spike_times, times, positions, labels, bandwidth=10.0
    ... )
    >>> "forward" in result.firing_rates
    True
    >>> "backward" in result.firing_rates
    True
    """
    # Validate direction_labels length matches times
    if len(direction_labels) != len(times):
        raise ValueError(
            f"direction_labels must have same length as times, "
            f"got {len(direction_labels)} and {len(times)}"
        )

    # Convert labels to array
    labels_arr = np.asarray(direction_labels, dtype=object)

    # Get unique labels, excluding "other"
    unique_labels = [label for label in np.unique(labels_arr) if label != "other"]

    # Sort labels for reproducibility
    unique_labels = sorted(unique_labels, key=str)

    # Compute place field for each direction
    firing_rates_dict: dict[str, NDArray[np.float64]] = {}
    occupancy_dict: dict[str, NDArray[np.float64]] = {}

    for label in unique_labels:
        # Build mask for this direction
        mask = labels_arr == label

        # Get subsets using our helper
        times_sub, spike_times_sub = _subset_spikes_by_time_mask(
            times, spike_times, mask
        )
        positions_sub = positions[mask]

        single = compute_spatial_rate(
            env,
            spike_times_sub,
            times_sub,
            positions_sub,
            method=method,
            bandwidth=bandwidth,
            min_occupancy=min_occupancy,
        )
        firing_rates_dict[str(label)] = np.asarray(single.firing_rate, dtype=np.float64)
        occupancy_dict[str(label)] = np.asarray(single.occupancy, dtype=np.float64)

    return DirectionalPlaceFields(
        firing_rates=firing_rates_dict,
        occupancy=occupancy_dict,
        env=env,
        labels=tuple(str(label) for label in unique_labels),
    )


# ==============================================================================
# Place Field Detection
# ==============================================================================


def detect_place_fields(
    env: Environment,
    firing_rate: NDArray[np.float64],
    *,
    threshold: float = 0.2,
    min_size: int | None = None,
    max_mean_rate: float = 10.0,
    detect_subfields: bool = True,
) -> PlaceFieldsResult:
    """Detect place fields using iterative peak-based approach (neurocode method).

    This implements the field-standard algorithm used by neurocode (AyA Lab)
    with support for subfield discrimination and interneuron exclusion.

    Parameters
    ----------
    env : Environment
        Spatial environment for binning.
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz) from neuron.
    threshold : float, default=0.2
        Fraction of peak rate for field boundary detection (0-1).
        Standard value is 0.2 (20% of peak).
    min_size : int, optional
        Minimum number of bins for a valid field. If None, defaults to 9 bins.
    max_mean_rate : float, default=10.0
        Maximum mean firing rate (Hz). Neurons exceeding this are excluded
        as putative interneurons (vandermeerlab convention).
    detect_subfields : bool, default=True
        If True, recursively detect subfields within large fields using
        higher thresholds. This discriminates coalescent place fields.

    Returns
    -------
    PlaceFieldsResult
        Container with ``fields`` (list of bin-index arrays, one per
        detected field) plus ``excluded_reason`` and ``n_excluded``
        attributes that distinguish "no detectable fields" from
        "neuron excluded by interneuron-rate filter". The result is
        iterable and indexable like the underlying list, so existing
        ``for f in result`` / ``len(result)`` / ``result[i]`` patterns
        keep working.

    Notes
    -----
    **Algorithm (neurocode approach)**:

    1. **Interneuron exclusion**: If mean rate > max_mean_rate, return no fields
    2. **Peak detection**: Find global maximum in firing rate map
    3. **Field segmentation**: Threshold at fraction of peak to define boundary
    4. **Connected component**: Extract bins above threshold connected to peak
    5. **Size filtering**: Discard fields smaller than min_size
    6. **Subfield recursion**: If detect_subfields=True, recursively apply
       higher thresholds (0.5, 0.7) to discriminate coalescent fields
    7. **Iteration**: Remove detected field bins and repeat until no peaks remain

    **Interneuron exclusion**: Following vandermeerlab convention, neurons with
    mean firing rate > 10 Hz are excluded as putative interneurons. Pyramidal
    cells (place cells) typically fire at 0.5-5 Hz.

    **Subfield detection**: When two place fields are close together, they may
    appear as a single broad field at low thresholds. Recursive thresholding
    at 0.5× and 0.7× peak discriminates true subfields.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import detect_place_fields
    >>> # Create synthetic place cell
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> firing_rate = np.zeros(env.n_bins)
    >>> # Add Gaussian place field at center
    >>> for i in range(env.n_bins):
    ...     dist = np.linalg.norm(env.bin_centers[i])
    ...     firing_rate[i] = 8.0 * np.exp(-(dist**2) / (2 * 3.0**2))
    >>> fields = detect_place_fields(env, firing_rate)
    >>> len(fields)  # doctest: +SKIP
    1

    See Also
    --------
    SpatialRateResult : Result class with rate map and metrics

    References
    ----------
    .. [1] neurocode repository (AyA Lab, Cornell): FindPlaceFields.m
    .. [2] Wilson & McNaughton (1993). Dynamics of hippocampal ensemble code
           for space. Science 261(5124).
    """
    # Validate inputs
    if firing_rate.shape[0] != env.n_bins:
        raise ValueError(
            f"firing_rate shape {firing_rate.shape} does not match "
            f"env.n_bins ({env.n_bins})"
        )

    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    # Set default min_size
    if min_size is None:
        min_size = 9  # Standard minimum (3×3 bins for 2D)

    # Interneuron exclusion. Emit a UserWarning AND surface the reason
    # in the returned PlaceFieldsResult so a caller running
    # detect_place_fields over a population can tell "this neuron has
    # no detectable place fields" (empty fields, excluded_reason=None)
    # from "this neuron was excluded as a putative interneuron"
    # (empty fields, excluded_reason="mean_rate_above_threshold").
    # The structured signal removes the need to listen for warnings.
    mean_rate = np.nanmean(firing_rate)
    if mean_rate > max_mean_rate:
        warnings.warn(
            f"detect_place_fields: neuron excluded as putative interneuron "
            f"(mean rate {float(mean_rate):.2f} Hz > max_mean_rate "
            f"{max_mean_rate} Hz). Returning empty field list. Pass a larger "
            "max_mean_rate to include fast-firing cells.",
            UserWarning,
            stacklevel=2,
        )
        return PlaceFieldsResult(
            fields=[],
            excluded_reason="mean_rate_above_threshold",
            n_excluded=1,
        )

    # Make a copy to modify during iteration
    rate_map = firing_rate.copy()
    fields = []

    # Iteratively find fields
    while True:
        # Handle all-NaN case
        if not np.any(np.isfinite(rate_map)):
            break  # No valid values remaining

        # Find peak
        peak_idx = int(np.nanargmax(rate_map))
        peak_rate = rate_map[peak_idx]

        # Check if peak is meaningful
        if peak_rate <= 0 or not np.isfinite(peak_rate):
            break

        # Threshold at fraction of peak
        threshold_rate = peak_rate * threshold

        # Find bins above threshold
        above_threshold = rate_map >= threshold_rate

        # Extract connected component containing peak
        field_bins = _extract_connected_component(peak_idx, above_threshold, env)

        # Check minimum size
        if len(field_bins) < min_size:
            # Remove this small field and continue
            rate_map[field_bins] = 0
            continue

        # Check for subfields (recursive thresholding)
        if detect_subfields and len(field_bins) > min_size * 2:
            # Try higher thresholds to discriminate subfields
            subfields = _detect_subfields(
                firing_rate[field_bins], field_bins, peak_rate, env, min_size
            )
            if len(subfields) > 1:
                # Found subfields - add them separately
                fields.extend(subfields)
            else:
                # No subfields - add as single field
                fields.append(field_bins)
        else:
            # Add field
            fields.append(field_bins)

        # Remove field bins from rate map
        rate_map[field_bins] = 0

        # Check if any meaningful peaks remain
        if np.nanmax(rate_map) < threshold_rate:
            break

    return PlaceFieldsResult(fields=fields, excluded_reason=None, n_excluded=0)


def is_place_cell(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    threshold: float = 0.2,
    min_size: int | None = None,
    max_mean_rate: float = 10.0,
    detect_subfields: bool = True,
) -> bool:
    """Quick check: Is this a place cell?

    Convenience function for fast screening of neurons. Computes the spatial
    rate map and checks whether :func:`detect_place_fields` finds at least one
    place field. Agrees with :func:`detect_place_fields`: returns ``True`` iff
    that detector finds a field on the same rate map.

    For detailed metrics, use :func:`compute_spatial_rate` and inspect the
    result's methods (``is_place_cell()``, ``spatial_information()``, etc.).

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Times of spikes.
    times : NDArray[np.float64], shape (n_time,)
        Timestamps for each behavioral sample.
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Animal positions.
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Rate map smoothing method.
    bandwidth : float, default=5.0
        Smoothing bandwidth in environment units.
    threshold : float, default=0.2
        Fraction of peak rate for field boundary detection (0-1).
    min_size : int, optional
        Minimum number of bins for a valid field. If None, defaults to 9.
    max_mean_rate : float, default=10.0
        Maximum mean firing rate (Hz). Neurons exceeding this are excluded
        as putative interneurons.
    detect_subfields : bool, default=True
        If True, recursively detect subfields within large fields.

    Returns
    -------
    bool
        True if the neuron passes place-cell criteria (has >= 1 detected
        place field).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import is_place_cell
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> times = np.linspace(0, 100, 1000)
    >>> spike_times = np.sort(rng.uniform(0, 100, 50))
    >>> result = is_place_cell(env, spike_times, times, positions)
    >>> type(result)
    <class 'bool'>

    See Also
    --------
    compute_spatial_rate : Full spatial rate computation
    detect_place_fields : Place field detection algorithm this agrees with
    SpatialRateResult.is_place_cell : Place-cell classification on a result
    """
    try:
        result = compute_spatial_rate(
            env,
            spike_times,
            times,
            positions,
            method=method,
            bandwidth=bandwidth,
        )
    except (ValueError, RuntimeError):
        return False
    return result.is_place_cell(
        threshold=threshold,
        min_size=min_size,
        max_mean_rate=max_mean_rate,
        detect_subfields=detect_subfields,
    )


def _extract_connected_component_scipy(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """Extract connected component using scipy.ndimage.label (fast path for grids).

    This is the optimized path for grid-based environments, providing ~6× speedup
    over graph-based flood-fill by leveraging scipy's optimized N-D labeling.

    Parameters
    ----------
    seed_idx : int
        Starting bin index in active bin indexing.
    mask : array, shape (n_bins,)
        Boolean mask of candidate bins (active bin indexing).
    env : Environment
        Spatial environment (must be grid-based with grid_shape and active_mask).

    Returns
    -------
    component : array
        Bin indices in connected component (active bin indexing, sorted).

    Raises
    ------
    ValueError
        If environment does not have grid_shape or active_mask attributes.

    Notes
    -----
    This function only works for grid-based environments (RegularGridLayout,
    MaskedGridLayout, etc.). For non-grid environments (1D tracks, irregular
    graphs), use _extract_connected_component_graph() instead.

    The algorithm:
    1. Reshape flat mask to N-D grid using grid_shape
    2. Apply scipy.ndimage.label to find connected components
    3. Identify which component contains the seed
    4. Convert back to flat active bin indices
    """
    from scipy import ndimage

    # Validate environment has required attributes
    if env.grid_shape is None or env.active_mask is None:
        raise ValueError("scipy path requires grid_shape and active_mask")

    # Reshape flat mask (active bin indexing) to N-D grid (original grid indexing)
    grid_mask = np.zeros(env.grid_shape, dtype=bool)
    grid_mask[env.active_mask] = mask

    # Determine connectivity structure to match graph connectivity
    # Check if environment uses diagonal neighbors
    n_dims = len(env.grid_shape)
    if hasattr(env.layout, "_build_params_used"):
        params = env.layout._build_params_used
        connect_diagonal = params.get("connect_diagonal_neighbors", False)
    else:
        # Default: no diagonal connections (4-connected in 2D, 6-connected in 3D)
        connect_diagonal = False

    # Create connectivity structure for scipy
    if connect_diagonal:
        # Full connectivity (includes diagonals): connectivity = n_dims
        structure = ndimage.generate_binary_structure(n_dims, n_dims)
    else:
        # Axial connectivity only (no diagonals): connectivity = 1
        structure = ndimage.generate_binary_structure(n_dims, 1)

    # Label connected components in N-D grid
    labeled, _n_components = ndimage.label(grid_mask, structure=structure)

    # Convert seed from active bin index to grid coordinates
    # active_mask.ravel() gives flat indices of active bins in original grid
    active_flat_indices = np.where(env.active_mask.ravel())[0]
    seed_grid_flat_idx = active_flat_indices[seed_idx]
    seed_grid_coords = np.unravel_index(seed_grid_flat_idx, env.grid_shape)

    # Get label of component containing seed
    seed_label = labeled[seed_grid_coords]

    if seed_label == 0:
        # Seed not in any component (shouldn't happen if mask[seed_idx] is True)
        return np.array([seed_idx], dtype=np.int64)

    # Extract all grid positions in this component
    component_grid_mask = labeled == seed_label

    # Convert back to flat active bin indices
    # Find which active bins correspond to this component
    component_in_active_bins = component_grid_mask.ravel() & env.active_mask.ravel()
    component_grid_flat_indices = np.where(component_in_active_bins)[0]

    # Map from original grid flat indices to active bin indices
    component_bins = np.searchsorted(active_flat_indices, component_grid_flat_indices)

    return np.array(sorted(component_bins), dtype=np.int64)


def _extract_connected_component_graph(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """Extract connected component using graph-based flood-fill (fallback path).

    This is the fallback path for non-grid environments (1D tracks, irregular
    graphs) and works for any graph structure. It uses breadth-first search
    with direct graph.neighbors() queries.

    Parameters
    ----------
    seed_idx : int
        Starting bin index.
    mask : array, shape (n_bins,)
        Boolean mask of candidate bins.
    env : Environment
        Spatial environment for connectivity.

    Returns
    -------
    component : array
        Bin indices in connected component (sorted).

    Notes
    -----
    This is the original implementation, proven to be already optimal for
    sparse connected components on arbitrary graphs. Benchmarking showed
    this is faster than NetworkX's connected_components() due to avoiding
    subgraph creation overhead.
    """
    # Flood fill using graph connectivity (BFS)
    component_set = {seed_idx}
    frontier = deque([seed_idx])

    while frontier:
        current = frontier.popleft()
        # Get neighbors from graph
        neighbors = list(env.connectivity.neighbors(current))
        for neighbor in neighbors:
            if mask[neighbor] and neighbor not in component_set:
                component_set.add(neighbor)
                frontier.append(neighbor)

    return np.array(sorted(component_set), dtype=np.int64)


def _extract_connected_component(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """Extract connected component of bins from seed (routes to optimal method).

    Automatically selects the optimal algorithm based on environment type:
    - Grid environments (2D/3D): Uses scipy.ndimage.label (~6× faster)
    - Non-grid environments: Uses graph-based flood-fill

    Parameters
    ----------
    seed_idx : int
        Starting bin index.
    mask : array, shape (n_bins,)
        Boolean mask of candidate bins.
    env : Environment
        Spatial environment for connectivity.

    Returns
    -------
    component : array
        Bin indices in connected component (sorted).

    Notes
    -----
    The routing logic checks for grid-based environments using:
    - env.grid_shape is not None
    - len(env.grid_shape) >= 2 (2D or 3D grids)
    - env.active_mask is not None

    For grid environments, uses scipy.ndimage.label for ~6× speedup.
    For non-grid environments, uses graph-based flood-fill (already optimal).
    """
    # Check if scipy fast path is applicable
    if (
        env.grid_shape is not None
        and len(env.grid_shape) >= 2
        and env.active_mask is not None
    ):
        # Fast path: scipy.ndimage.label for grid environments
        return _extract_connected_component_scipy(seed_idx, mask, env)
    else:
        # Fallback path: graph-based flood-fill for non-grid environments
        return _extract_connected_component_graph(seed_idx, mask, env)


def _detect_subfields(
    field_rates: NDArray[np.float64],
    field_bins: NDArray[np.int64],
    peak_rate: float,
    env: Environment,
    min_size: int,
) -> list[NDArray[np.int64]]:
    """Recursively detect subfields using higher thresholds.

    Parameters
    ----------
    field_rates : array
        Firing rates within field bins.
    field_bins : array
        Bin indices of field.
    peak_rate : float
        Peak firing rate in field.
    env : Environment
        Spatial environment.
    min_size : int
        Minimum field size.

    Returns
    -------
    subfields : list of arrays
        List of subfield bin indices. If only one subfield found,
        returns list with original field.
    """
    # Try thresholds: 0.5 and 0.7 of peak
    subfield_thresholds = [0.5, 0.7]

    for thresh in subfield_thresholds:
        threshold_rate = peak_rate * thresh
        above_threshold = field_rates >= threshold_rate

        # Find connected components
        subfields = []
        remaining_mask = above_threshold.copy()

        while remaining_mask.any():
            # Find a seed
            seed_local_idx = np.where(remaining_mask)[0][0]
            seed_global_idx = field_bins[seed_local_idx]

            # Build mask in global coordinates
            global_mask = np.zeros(env.n_bins, dtype=bool)
            global_mask[field_bins[above_threshold]] = True

            # Extract component
            component_global = _extract_connected_component(
                seed_global_idx, global_mask, env
            )

            if len(component_global) >= min_size:
                subfields.append(component_global)

            # Remove from remaining mask
            for bin_idx in component_global:
                # Find local index
                local_indices = np.where(field_bins == bin_idx)[0]
                if len(local_indices) > 0:
                    remaining_mask[local_indices[0]] = False

        # If found multiple subfields, return them
        if len(subfields) > 1:
            return subfields

    # No subfields found
    return [field_bins]
