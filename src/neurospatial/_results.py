"""Unified result-object surface shared across analysis result classes.

This module defines :class:`ResultMixin`, the base mixin that gives every
user-facing result object a uniform, discoverable surface so that the verbs
ending an analysis -- "summarize it", "plot it", "put it in a table" -- always
have a home rather than dead-ending to manual NumPy.

The contract is intentionally small:

- ``to_dataframe()`` -- tidy/long-form :class:`pandas.DataFrame`. Tidy/long form
  is required so that heterogeneous results (a place-field result, a directional
  result, a decoding result) concatenate without error via a single
  ``pandas.concat`` into a union schema (columns absent from a given result are
  filled with ``NaN``); it does not give every result type a shared column
  vocabulary.
- ``summary()`` -- a flat ``dict`` of scalar headline metrics suitable for a
  one-line per-result report or a population table row.
- ``plot(ax=None, **kwargs)`` -- a sensible default visualization. ``plot()`` is
  **optional per result type**: the base raises :class:`NotImplementedError`, so
  a result with no meaningful single visualization is still a valid result.
  Results that can plot override this and **return the axis** so they compose
  into multi-panel figures.

``pandas`` and ``matplotlib`` are imported lazily inside the methods that need
them, keeping both optional at import time.

Notes
-----
The base implementations of ``to_dataframe`` and ``summary`` raise
:class:`NotImplementedError`. Each concrete result class (or an intermediate
mixin such as :class:`neurospatial.encoding._base.SpatialResultMixin`) supplies
the specialization appropriate to its data. This keeps the contract uniform
without forcing a single tabular shape onto every result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


def resolve_unit_ids(
    unit_ids: NDArray[Any] | Sequence[Any] | None,
    n_units: int,
    *,
    context: str = "",
) -> NDArray[Any]:
    """Resolve and validate per-unit identity labels.

    Returns an ``ndarray`` of unit identity labels, defaulting to
    ``np.arange(n_units)`` when ``unit_ids`` is ``None``. When labels are
    provided, validates that exactly one label is supplied per unit.

    Parameters
    ----------
    unit_ids : ndarray or sequence or None
        Per-unit identity labels. May be integers or strings. When ``None``,
        defaults to ``np.arange(n_units)``.
    n_units : int
        Number of units the labels must describe.
    context : str, optional
        Caller name included in the error message on a length mismatch.

    Returns
    -------
    ndarray
        Resolved labels as a 1D array of length ``n_units``.

    Raises
    ------
    ValueError
        If ``unit_ids`` is provided and is not 1-D, or if its length does not
        equal ``n_units``.
    """
    if unit_ids is None:
        return np.arange(n_units)

    resolved = np.asarray(unit_ids)
    if resolved.ndim != 1:
        where = f" in {context}" if context else ""
        raise ValueError(
            f"unit_ids must be 1-D{where}: got shape {resolved.shape}.\n"
            "  WHY: unit_ids labels one identity per unit (row).\n"
            "  HOW: pass a 1-D sequence with one entry per unit, or omit it "
            "to default to np.arange(n_units)."
        )
    if resolved.shape[0] != n_units:
        where = f" in {context}" if context else ""
        raise ValueError(
            f"unit_ids length mismatch{where}: got {resolved.shape[0]} "
            f"label(s) but there are {n_units} unit(s).\n"
            "  WHY: each unit must have exactly one identity label.\n"
            "  HOW: pass unit_ids with one entry per unit, or omit it to "
            "default to np.arange(n_units)."
        )
    return resolved


def validate_unit_table(
    unit_table: Any,
    n_units: int,
    *,
    context: str = "",
) -> None:
    """Validate an optional per-unit metadata table against the unit count.

    Enforces the documented invariant that ``unit_table`` carries exactly one
    row per unit, aligned to ``unit_ids``. A ``None`` table is always valid
    (the field is optional). A non-``None`` table whose row count differs from
    ``n_units`` raises :class:`ValueError`, naming both lengths.

    Parameters
    ----------
    unit_table : pandas.DataFrame or None
        Optional per-unit metadata table. When ``None``, no validation occurs.
    n_units : int
        Number of units the table must describe (one row per unit).
    context : str, optional
        Caller / class name included in the error message on a mismatch.

    Raises
    ------
    ValueError
        If ``unit_table`` is not ``None`` and ``len(unit_table) != n_units``.
    """
    if unit_table is None:
        return
    n_rows = len(unit_table)
    if n_rows != n_units:
        where = f" in {context}" if context else ""
        raise ValueError(
            f"unit_table length mismatch{where}: got {n_rows} row(s) but "
            f"there are {n_units} unit(s).\n"
            "  WHY: unit_table must carry exactly one row per unit, aligned "
            "to unit_ids.\n"
            "  HOW: pass a unit_table with one row per unit, or omit it."
        )


def software_version() -> str:
    """Return the installed ``neurospatial`` version string.

    Used to stamp xarray exports with the producing library version. Falls
    back to ``"unknown"`` when the package metadata cannot be resolved (e.g.
    an editable checkout without an installed distribution).

    Returns
    -------
    str
        The installed version (e.g. ``"0.6.0"``), or ``"unknown"``.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("neurospatial")
    except PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"


def units_attr(env: Any) -> dict[str, str]:
    """Return a ``{"units": ...}`` attrs fragment, omitted when units are unset.

    Reads ``env.units`` and returns ``{"units": <value>}`` only when it is a
    meaningful non-empty string. When ``units`` is ``None`` (or empty, or the
    env lacks the attribute) an **empty** mapping is returned, so that the
    literal string ``"None"`` is never stored as a units attribute on an
    xarray export.

    Parameters
    ----------
    env : Any
        An environment exposing an optional ``units`` attribute.

    Returns
    -------
    dict
        ``{"units": value}`` when units are set, else ``{}``.
    """
    units = getattr(env, "units", None)
    if units is None:
        return {}
    units_str = str(units)
    if units_str == "":
        return {}
    return {"units": units_str}


def env_fingerprint(env: Any) -> str:
    """Return a stable, human-readable identifier for an environment.

    There is no dedicated environment hash in the codebase, so this returns a
    compact ``repr``-style identifier (name, dimensionality, bin count, layout)
    suitable for stamping into xarray ``attrs`` for provenance. It is *not* a
    content hash and must not be relied on for equality checks.

    Parameters
    ----------
    env : Any
        An ``Environment`` or ``EgocentricPolarEnvironment`` instance.

    Returns
    -------
    str
        A stable single-line identifier such as
        ``"Environment(name='', 2D, 25 bins, RegularGrid)"``.
    """
    return repr(env)


def _bin_center_coords(
    env: Any, n_bins: int
) -> dict[str, tuple[str, NDArray[np.float64]]]:
    """Build non-index ``bin``-dimension coordinates from an environment.

    Returns a mapping of coordinate name to ``("bin", values)`` tuples giving
    the per-bin center coordinates. Cartesian environments yield
    ``bin_center_x`` (1-D), plus ``bin_center_y`` (2-D) and ``bin_center_z``
    (3-D). Polar egocentric environments (whose ``bin_centers[:, 0]`` is
    distance and ``bin_centers[:, 1]`` is angle in radians) yield
    ``bin_center_distance`` and ``bin_center_angle`` instead.

    Parameters
    ----------
    env : Any
        Environment exposing ``bin_centers`` and ``n_dims``. May be ``None``,
        in which case an empty mapping is returned.
    n_bins : int
        Expected number of bins; the environment's ``bin_centers`` must agree.

    Returns
    -------
    dict
        Mapping ``name -> ("bin", values)`` for use as xarray ``coords``.

    Raises
    ------
    ValueError
        If ``env`` is given but its ``bin_centers`` is not 2-D or its length
        (``bin_centers.shape[0]``) does not equal ``n_bins``. The genuine
        no-env case (``env is None``) is a no-op and returns an empty mapping.
    """
    # Genuine no-env case (e.g. directional results pass bin_centers directly
    # and have no env): no bin-center coords to build. This is NOT a mismatch.
    if env is None:
        return {}

    from neurospatial.environment.polar import EgocentricPolarEnvironment

    bin_centers = np.asarray(env.bin_centers, dtype=np.float64)
    # bin-center data IS present (env was given); a shape disagreement here is
    # a real inconsistency between the result and its environment, not an
    # optional-coord situation. Raise loudly rather than silently dropping it.
    if bin_centers.ndim != 2:
        raise ValueError(
            "Environment bin_centers must be 2-D (n_bins, n_dims) to build "
            f"xarray bin-center coords, but env produced an array with "
            f"ndim={bin_centers.ndim} (shape {bin_centers.shape}).\n"
            "  WHY: each bin needs an (x, y, ...) center for the 'bin' "
            "coordinate.\n"
            "  HOW: ensure the environment is fitted and exposes 2-D "
            "bin_centers."
        )
    if bin_centers.shape[0] != n_bins:
        raise ValueError(
            "bin_centers length does not match the result's bin count: env "
            f"bin_centers has {bin_centers.shape[0]} row(s) but the result "
            f"has {n_bins} bin(s).\n"
            "  WHY: the 'bin' coordinate must have one center per bin; a "
            "mismatch would silently produce a structurally-incomplete or "
            "misaligned Dataset.\n"
            "  HOW: pass the same environment used to compute the result, or "
            "recompute the result against this environment."
        )

    if isinstance(env, EgocentricPolarEnvironment):
        return {
            "bin_center_distance": ("bin", bin_centers[:, 0]),
            "bin_center_angle": ("bin", bin_centers[:, 1]),
        }

    names = ["bin_center_x", "bin_center_y", "bin_center_z"]
    n_dims = bin_centers.shape[1]
    coords: dict[str, tuple[str, NDArray[np.float64]]] = {}
    for i in range(min(n_dims, 3)):
        coords[names[i]] = ("bin", bin_centers[:, i])
    # For >3-D envs, expose remaining axes as bin_center_dim_3, ...
    for i in range(3, n_dims):
        coords[f"bin_center_dim_{i}"] = ("bin", bin_centers[:, i])
    return coords


def build_population_dataset(
    firing_rates: NDArray[np.float64],
    unit_ids: NDArray[Any],
    *,
    env: Any = None,
    bin_centers: NDArray[np.float64] | None = None,
    occupancy: NDArray[np.float64] | None = None,
    attrs: dict[str, Any] | None = None,
) -> Any:
    """Build a labeled population firing-rate :class:`xarray.Dataset`.

    Shared constructor for the population ``to_xarray`` methods. Produces a
    :class:`xarray.Dataset` with dims ``("unit_id", "bin")`` where ``unit_id``
    is the index coordinate (the real per-unit identity labels) and ``bin`` is
    the integer bin index carrying non-index ``bin_center_*`` coordinates.

    Parameters
    ----------
    firing_rates : ndarray, shape (n_units, n_bins)
        Population firing-rate matrix (Hz).
    unit_ids : ndarray, shape (n_units,)
        Per-unit identity labels. **Must be unique** (label-based
        ``.sel(unit_id=...)`` requires uniqueness).
    env : Any, optional
        Environment supplying ``bin_center_*`` coords. Mutually used with
        ``bin_centers``; pass at most one. ``None`` omits Cartesian coords.
    bin_centers : ndarray, optional
        Explicit angular bin centers (radians) for results without an ``env``
        (directional). When given, a ``bin_center_angle`` coord is attached.
    occupancy : ndarray, shape (n_bins,), optional
        Per-bin occupancy (seconds); attached as an ``occupancy`` data var on
        ``("bin",)`` when provided.
    attrs : dict, optional
        Dataset-level attributes (units, bandwidth, env fingerprint, version).

    Returns
    -------
    xarray.Dataset
        Dataset with data var ``firing_rate`` (dims ``("unit_id", "bin")``),
        optional ``occupancy`` (dims ``("bin",)``), and ``bin_center_*`` coords.

    Raises
    ------
    ValueError
        If ``unit_ids`` contains duplicate labels, if ``occupancy`` is
        provided but its length does not equal ``n_bins``, or if
        ``bin_centers`` is provided but is not 1-D of length ``n_bins``.
        (Absent ``occupancy``/``bin_centers`` are simply omitted; only a
        present-but-mismatched array raises.)
    ImportError
        If ``xarray`` is not installed.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "to_xarray() requires the optional 'xarray' dependency, which "
            "is not installed. Install it with "
            "'pip install neurospatial[xarray]' or 'pip install xarray'."
        ) from exc

    rates = np.asarray(firing_rates, dtype=np.float64)
    n_bins = rates.shape[1]
    ids = np.asarray(unit_ids)

    # Duplicate unit_ids break label-based .sel(unit_id=...); reject loudly.
    unique_vals, counts = np.unique(ids, return_counts=True)
    if np.any(counts > 1):
        dups = unique_vals[counts > 1]
        raise ValueError(
            "unit_ids must be unique to build an xarray.Dataset, but these "
            f"label(s) are duplicated: {list(dups)}.\n"
            "  WHY: label-based selection .sel(unit_id=...) requires a unique "
            "index coordinate.\n"
            "  HOW: deduplicate unit_ids (e.g. when concatenating populations) "
            "before calling to_xarray()."
        )

    coords: dict[str, Any] = {"unit_id": ids, "bin": np.arange(n_bins)}
    if env is not None:
        coords.update(_bin_center_coords(env, n_bins))
    if bin_centers is not None:
        # bin_centers IS present (passed explicitly, e.g. directional angular
        # centers); a shape disagreement is a real inconsistency, not an
        # optional coord. Raise rather than silently omitting the coord.
        bc = np.asarray(bin_centers, dtype=np.float64)
        if bc.ndim != 1 or bc.shape[0] != n_bins:
            raise ValueError(
                "bin_centers length does not match the firing-rate bin "
                f"count: bin_centers has shape {bc.shape} but there are "
                f"{n_bins} bin(s) (firing_rates.shape[1]).\n"
                "  WHY: the 'bin' coordinate must have one center per bin; a "
                "mismatch would silently produce a misaligned Dataset.\n"
                "  HOW: pass a 1-D bin_centers array of length n_bins."
            )
        coords["bin_center_angle"] = ("bin", bc)

    data_vars: dict[str, Any] = {
        "firing_rate": (("unit_id", "bin"), rates),
    }
    if occupancy is not None:
        # occupancy IS present; a length disagreement is a real inconsistency,
        # not an absent-optional. Raise rather than silently dropping the var.
        occ = np.asarray(occupancy, dtype=np.float64)
        if occ.ndim != 1 or occ.shape[0] != n_bins:
            raise ValueError(
                "occupancy length does not match the firing-rate bin count: "
                f"occupancy has shape {occ.shape} but there are {n_bins} "
                "bin(s) (firing_rates.shape[1]).\n"
                "  WHY: the 'occupancy' data var is indexed by bin; a length "
                "mismatch would silently produce a structurally-incomplete "
                "Dataset.\n"
                "  HOW: pass a 1-D occupancy array of length n_bins, or omit "
                "it entirely if unavailable."
            )
        data_vars["occupancy"] = (("bin",), occ)

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs or {},
    )


class ResultMixin:
    """Uniform result-object surface: ``to_dataframe``, ``summary``, ``plot``.

    Mix this into any analysis result class to advertise (and partially
    fulfill) the three analysis-ending verbs. The base implementations of
    ``to_dataframe`` and ``summary`` raise :class:`NotImplementedError`,
    signalling that a concrete result must specialize them. ``plot`` likewise
    raises by default, because a meaningful default visualization is optional:
    a result with no single natural plot is still a valid result.

    Notes
    -----
    - ``to_dataframe()`` should return **tidy/long form** so heterogeneous
      results concatenate without error in one :func:`pandas.concat` into a
      union schema (missing columns filled with ``NaN``).
    - ``plot(ax=None, **kwargs)`` should accept an optional ``ax`` and
      **return it**, for composition into multi-panel figures.
    - ``pandas`` and ``matplotlib`` must be imported lazily inside the methods
      that use them, never at module scope.

    Examples
    --------
    >>> from neurospatial._results import ResultMixin
    >>> class Empty(ResultMixin):
    ...     pass
    >>> try:
    ...     Empty().summary()
    ... except NotImplementedError:
    ...     print("not implemented")
    not implemented
    """

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy/long-form :class:`pandas.DataFrame` of this result.

        Tidy/long form means one observation per row with explicit identifier
        columns, so that DataFrames from different result types concatenate
        without error in a single :func:`pandas.concat`. Because result types use
        different identifier columns (e.g. spatial uses ``neuron``/``coord_0``;
        directional uses ``direction``), the concatenation yields a union schema
        in which columns absent from a given result are filled with ``NaN``,
        rather than a single shared column vocabulary.

        Returns
        -------
        pandas.DataFrame
            Long-form table of the result's contents.

        Raises
        ------
        NotImplementedError
            If the concrete result class does not specialize this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement to_dataframe(). "
            f"to_dataframe() must be specialized per result type."
        )

    def summary(self) -> dict[str, Any]:
        """Return a flat dict of scalar headline metrics for this result.

        Intended for a one-line report per result or as a row in a population
        table. Values should be plain Python scalars (``float``, ``int``,
        ``str``) so the dict is trivially serializable and tabulatable.

        Returns
        -------
        dict
            Mapping from metric name to scalar value.

        Raises
        ------
        NotImplementedError
            If the concrete result class does not specialize this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement summary(). "
            f"summary() must be specialized per result type."
        )

    def plot(self, *args: Any, **kwargs: Any) -> Axes:
        """Plot this result, returning the axis for composition.

        ``plot()`` is **optional** per result type: the base implementation
        raises :class:`NotImplementedError`, so a result with no meaningful
        single visualization is still valid. Results that can plot override
        this with a concrete signature, accept an optional ``ax`` keyword, and
        **return it**. The base is deliberately permissive (``*args, **kwargs``)
        so both single-result (``plot(ax=None, ...)``) and batch-result
        (``plot(idx, ax=None, ...)``) specializations are valid overrides.

        Parameters
        ----------
        *args
            Implementation-specific positional arguments. Single-result
            overrides take an optional leading ``ax``; batch overrides take a
            leading selector (e.g. a neuron index) followed by ``ax``.
        **kwargs
            Implementation-specific plotting options (typically including
            ``ax``).

        Returns
        -------
        matplotlib.axes.Axes
            The axes the result was drawn on.

        Raises
        ------
        NotImplementedError
            If the concrete result class provides no default visualization.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement plot(). "
            f"A default visualization is optional for this result type."
        )
