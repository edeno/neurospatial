"""pynapple ingress / egress adapters (optional dependency).

These two functions are the only code in neurospatial that imports pynapple,
and they import it lazily: ``import pynapple`` runs *inside* the functions, so
the package and the array path keep working when pynapple is not installed.
Analysis functions also accept pynapple objects directly, by duck typing and
without importing pynapple: a ``TsGroup`` / ``Ts`` as spike input (a
``TsGroup`` brings its labels and metadata), a ``TsdFrame`` / ``Tsd`` as
``positions`` or ``headings`` (its ``time_support`` becomes ``epochs`` when
none are given), and an ``IntervalSet`` as ``epochs=`` / ``spike_window=``.
:func:`from_pynapple` returns each object's ``time_support`` as epochs, so
gaps between recording epochs are not lost on the array route either.

Functions
---------
from_pynapple
    Convert a pynapple ``Ts`` / ``Tsd`` / ``TsdFrame`` / ``TsGroup`` /
    ``IntervalSet`` to plain arrays plus its time support as ``(n_epochs, 2)``
    epochs.
to_pynapple
    Convert ``times`` + ``values`` or a decode result (MAP or mean track, or
    the posterior) to a pynapple ``Tsd`` / ``TsdFrame``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._pynapple import (
    interval_rows,
    refuse_permuted_spatial_columns,
)

__all__ = ["from_pynapple", "to_pynapple"]

_IMPORT_ERROR_MSG = (
    "This function requires the optional 'pynapple' extra; install it with "
    "`pip install neurospatial[pynapple]` (or `uv add neurospatial[pynapple]`)."
)

_DecodeQuantity = Literal["map_position", "mean_position", "posterior"]
_DECODE_QUANTITIES = ("map_position", "mean_position", "posterior")


def _require_pynapple() -> Any:
    """Import and return the ``pynapple`` module, or raise a clear ImportError.

    The import is lazy so importing :mod:`neurospatial.io` never requires
    pynapple. Raises a clear, actionable :class:`ImportError` naming the extra
    when pynapple is absent.
    """
    try:
        import pynapple
    except ImportError as exc:  # pragma: no cover - exercised only when absent
        raise ImportError(_IMPORT_ERROR_MSG) from exc
    return pynapple


def from_pynapple(obj: Any) -> tuple[Any, ...] | NDArray[np.float64]:
    """Convert a pynapple object to plain NumPy arrays and its epochs.

    Each time series comes back with its pynapple ``time_support`` as
    ``epochs``, a ``(n_epochs, 2)`` array of half-open ``[start, stop)`` rows
    in seconds. Pass it as ``epochs=`` (or, for spikes, ``spike_window=``) so the
    gaps between recording epochs stay out of occupancy and rates.

    Parameters
    ----------
    obj : pynapple Ts, Tsd, TsdFrame, TsGroup, or IntervalSet
        The pynapple object to convert.

    Returns
    -------
    spike_times, epochs : ndarray, shapes (n_spikes,) and (n_epochs, 2)
        For a ``Ts``: its timestamps and time support.
    times, values, epochs : ndarray, shapes (n_samples,), (n_samples,) or \
(n_samples, n_columns), and (n_epochs, 2)
        For a ``Tsd`` (1-D values) or ``TsdFrame`` (2-D values, columns in
        ``tsdframe.columns`` order).
    trains, unit_ids, epochs : list of ndarray, ndarray of shape (n_units,), \
ndarray of shape (n_epochs, 2)
        For a ``TsGroup``: one spike-time array per unit in group order, the
        group's unit labels, and the group's time support.
    epochs : ndarray, shape (n_epochs, 2)
        For an ``IntervalSet``: one ``[start, stop)`` row per interval.

    All arrays are float64 copies (``unit_ids`` keeps its label type), so
    editing them never changes the pynapple object.

    Raises
    ------
    ImportError
        If pynapple is not installed.
    TypeError
        If ``obj`` is not a pynapple ``Ts``, ``Tsd``, ``TsdFrame``, ``TsGroup``
        or ``IntervalSet`` (a ``TsdTensor`` has no spatial reading).
    ValueError
        If a ``TsdFrame``'s columns are spatial axis names (``x``, ``y``,
        ``z``) in another order, such as ``["y", "x"]``.

    Notes
    -----
    pynapple intervals ``[start, end]`` include their end, while neurospatial
    windows are half-open ``[start, stop)``. Each ``stop`` is the next float
    above ``end`` (``np.nextafter(end, np.inf)``), so the epochs hold exactly
    the timestamps pynapple holds: a sample or spike at ``end``, such as the
    last spike of a support built from first to last spike, stays inside.
    An ``IntervalSet`` passed directly as ``epochs=`` or ``spike_window=`` is
    converted the same way.

    A ``TsGroup`` built from spike times alone gets a default time support
    from its first to its last spike, which is not recording coverage. Use
    its epochs as ``spike_window=`` only when the group's time support was
    set to the acquisition intervals.

    Examples
    --------
    >>> from neurospatial.io import from_pynapple  # doctest: +SKIP
    >>> times, positions, epochs = from_pynapple(tsdframe)  # doctest: +SKIP
    >>> trains, unit_ids, recorded = from_pynapple(tsgroup)  # doctest: +SKIP
    >>> rates = compute_spatial_rates(  # doctest: +SKIP
    ...     env, trains, times, positions, epochs=epochs, spike_window=recorded,
    ...     unit_ids=unit_ids,
    ... )
    """
    nap = _require_pynapple()

    if isinstance(obj, nap.IntervalSet):
        return interval_rows(obj)
    if isinstance(obj, nap.Ts):
        return np.array(obj.t, dtype=np.float64), interval_rows(obj.time_support)
    if isinstance(obj, (nap.Tsd, nap.TsdFrame)):
        if isinstance(obj, nap.TsdFrame):
            refuse_permuted_spatial_columns(obj.columns, fix="from_pynapple({frame})")
        return (
            np.array(obj.t, dtype=np.float64),
            np.array(obj.values, dtype=np.float64),
            interval_rows(obj.time_support),
        )
    if isinstance(obj, nap.TsGroup):
        from neurospatial._spikes import _as_spike_trains_with_ids

        trains, unit_ids = _as_spike_trains_with_ids(obj)
        return (
            [np.array(train) for train in trains],
            unit_ids,
            interval_rows(obj.time_support),
        )

    raise TypeError(
        f"from_pynapple does not recognize {type(obj).__name__!r}.\n"
        "Why: it converts the pynapple types with a spatial-analysis reading.\n"
        "Fix: pass a pynapple Ts, Tsd, TsdFrame, TsGroup or IntervalSet."
    )


def to_pynapple(
    data: Any,
    values: NDArray[np.float64] | None = None,
    *,
    columns: Any = None,
    epochs: Any = None,
    quantity: _DecodeQuantity = "map_position",
) -> Any:
    """Convert neurospatial data to the matching pynapple object.

    - timestamps with ``values``: a ``Tsd`` (1-D values) or ``TsdFrame``
      (2-D values).
    - a decode result: a ``TsdFrame`` of ``quantity``, the MAP track by
      default.

    Parameters
    ----------
    data : array-like or decode result
        Finite, strictly increasing timestamps in seconds (with ``values``),
        or a :class:`~neurospatial.decoding.DecodingResult` or
        :class:`~neurospatial.decoding.DecodingSummary` (any object with
        ``.times`` and ``.map_position``). Timestamps are never sorted here:
        pynapple would sort them without reordering ``values``.
    values : NDArray[np.float64] or None, default=None
        Values sampled at ``data``, shape ``(n,)`` or ``(n, n_columns)``.
        Required with timestamps; must be ``None`` for a decode result.
    columns : sequence, optional
        Column labels of a ``TsdFrame``, one per value column. Defaults to
        the environment's axis names (``x``, ``y``, ...) for a decoded track.
        Not accepted for a posterior, whose columns are bin indices.
    epochs : (start, stop), array-like of shape (n_epochs, 2), IntervalSet, \
or None, optional
        Time support of the output, in seconds: the time over which it was observed, so gaps between
        epochs are not read as recorded time. Rows are half-open
        ``[start, stop)``, sorted and touching rows merged, as for every
        ``epochs=`` argument. Every timestamp must lie inside a row
        (pynapple would silently drop the others). Default ``None``: a
        decode result's ``time_support``, else pynapple's default, one
        interval from the first to the last timestamp. Each pynapple interval
        ends at the float just below ``stop``, so it holds the same
        timestamps.
    quantity : {"map_position", "mean_position", "posterior"}, \
default="map_position"
        Which estimate of a decode result to export: the MAP or posterior
        mean track, as a ``TsdFrame`` with one column per spatial axis, or
        the full posterior (``DecodingResult`` only), as a ``TsdFrame`` with
        one column per environment bin; see Notes.

    Returns
    -------
    pynapple.Tsd or pynapple.TsdFrame
        See the list above. Arrays are copies: editing the pynapple object
        never changes the neurospatial data.

    Raises
    ------
    ImportError
        If pynapple is not installed.
    TypeError
        If ``values`` is ``None`` and ``data`` is not a decode result,
        ``columns`` is given with a posterior, or ``quantity`` is given with
        a ``values`` array.
    ValueError
        If timestamps are not 1-D, finite and strictly increasing, ``values``
        is not 1-D or 2-D or differs in length, ``columns`` does not have one
        label per value column, ``epochs`` is not a valid set of windows or
        leaves a timestamp outside, a decode result has no ``times``,
        or ``quantity`` is unknown or ``"posterior"`` for a
        ``DecodingSummary``. These checks run before pynapple is imported.

    Notes
    -----
    **Decode results.** The time support is the result's ``time_support``,
    the windows its time bins cover, so time between decoded epochs is not
    part of the pynapple object.

    **Posterior.** ``quantity="posterior"`` returns a ``TsdFrame`` with one
    column per environment bin (column ``j`` is bin ``j``). Each value is
    the probability mass in that bin, so each row sums to 1, or is entirely
    NaN for a time bin that could not be decoded. The column metadata
    ``bin_center_x``, ``bin_center_y`` (``bin_center_z``; or
    ``bin_center_distance`` and ``bin_center_angle`` for a polar
    environment) hold each bin's center, as in
    :meth:`~neurospatial.decoding.DecodingResult.to_xarray`. A bin column
    rather than a grid-shaped ``TsdTensor`` keeps every environment
    (tracks, graphs, masked and hexagonal layouts) exact, never pads bins
    outside the environment with values that read like probabilities or
    missing data, and carries the bin coordinates with the data. To select
    bins by place, index the columns, e.g.
    ``posterior[:, posterior.bin_center_x < 50]``. On unequal bins the
    largest mass is not the MAP bin: the MAP ranks density, mass divided by
    ``env.bin_sizes``.

    pynapple's own Bayesian decoder (``nap.decode_bayes``) differs from
    neurospatial's in two ways that change posteriors near unvisited bins:
    it sums each bin's log-likelihood with ``np.nansum``, so a unit whose
    rate is NaN in a bin drops out of that bin's likelihood instead of
    excluding the bin, and its uniform prior is flat over bins rather than
    over space. neurospatial gives a bin where any rate is missing zero
    probability, and its default prior is proportional to bin size.

    Examples
    --------
    A decoded MAP track and posterior, with the decoded epochs as time
    support:

    >>> from neurospatial.io import to_pynapple  # doctest: +SKIP
    >>> track = to_pynapple(result)  # TsdFrame with columns x, y  # doctest: +SKIP
    >>> posterior = to_pynapple(result, quantity="posterior")  # doctest: +SKIP
    >>> posterior.bin_center_x  # column metadata  # doctest: +SKIP

    Plain arrays:

    >>> tsd = to_pynapple(times, linear_positions, epochs=run_epochs)  # doctest: +SKIP
    """
    if values is None:
        if not (hasattr(data, "times") and hasattr(data, "map_position")):
            raise TypeError(
                "to_pynapple(data) without values expects a decode result "
                "(e.g. a neurospatial.decoding.DecodingResult); got "
                f"{type(data).__name__!r}.\n"
                "Why: only a decode result carries its own values; timestamps "
                "need the values sampled at them.\n"
                "Fix: to convert plain arrays, pass to_pynapple(times, values) "
                "with an explicit `values` array."
            )
        return _decode_to_pynapple(data, columns, epochs, quantity)

    if quantity != "map_position":
        raise TypeError(
            f"to_pynapple got quantity={quantity!r} with a values array.\n"
            "Why: quantity selects which estimate of a decode result to "
            "export; plain arrays are exported as given.\n"
            "Fix: omit quantity=, or pass the decode result itself."
        )
    times = np.asarray(data, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    return _series_to_pynapple(times, values, columns, epochs, metadata=None)


def _decode_to_pynapple(result: Any, columns: Any, epochs: Any, quantity: str) -> Any:
    """Export one estimate of a decode result as a ``TsdFrame``."""
    from neurospatial._results import _bin_center_coords, _coord_dim_names
    from neurospatial._validation import validate_choice

    validate_choice(
        quantity, _DECODE_QUANTITIES, name="quantity", owner="to_pynapple()"
    )
    if result.times is None:
        raise ValueError(
            "The decode result has no `times`.\n"
            "Why: a pynapple Tsd/TsdFrame needs one timestamp per decoded "
            "time bin.\n"
            "Fix: decode with times=, or pass to_pynapple(bin_times, "
            "result.map_position) with the time-bin centers in seconds."
        )
    times = np.array(result.times, dtype=np.float64)
    if epochs is None:
        epochs = getattr(result, "time_support", None)
    metadata = None
    if quantity == "posterior":
        if not hasattr(result, "posterior"):
            raise ValueError(
                f"to_pynapple got quantity='posterior' for a "
                f"{type(result).__name__}, which keeps no posterior.\n"
                "Why: the summary decoders reduce each posterior block to "
                "per-time estimates and discard it.\n"
                "Fix: decode with decode_session (or decode_position) to keep "
                "the posterior, or export quantity='map_position'."
            )
        if columns is not None:
            raise TypeError(
                "to_pynapple got columns= with quantity='posterior'.\n"
                "Why: posterior column j is environment bin j; its center is "
                "in the bin_center_* column metadata.\n"
                "Fix: omit columns=."
            )
        values = np.array(result.posterior)
        columns = np.arange(values.shape[1])
        metadata = {
            name: np.array(centers)
            for name, (_, centers) in _bin_center_coords(
                result.env, values.shape[1]
            ).items()
        }
    else:
        values = np.array(getattr(result, quantity), dtype=np.float64)
        if columns is None:
            columns = _coord_dim_names(result.env)
    return _series_to_pynapple(times, values, columns, epochs, metadata=metadata)


def _checked_support(
    times: NDArray[np.float64], epochs: Any
) -> NDArray[np.float64] | None:
    """Normalize ``epochs`` and require every timestamp to lie inside a row."""
    from neurospatial._intervals import as_intervals, times_in_intervals

    support = as_intervals(epochs, name="epochs")
    if support is None:
        return None
    outside = np.flatnonzero(~times_in_intervals(support, times))
    if outside.size:
        i = int(outside[0])
        raise ValueError(
            f"{outside.size} of {len(times)} timestamp(s) lie outside "
            f"`epochs` (first at index {i}, t={float(times[i])!r}).\n"
            "Why: pynapple silently drops samples outside the time "
            "support.\n"
            "Fix: pass epochs that cover every timestamp, or select the "
            "samples first, e.g. with neurospatial.behavior.restrict(times, "
            "values, epochs=epochs)."
        )
    return support


def _interval_set(nap: Any, support: NDArray[np.float64]) -> Any:
    """A pynapple ``IntervalSet`` holding the same floats as half-open rows."""
    from neurospatial._intervals import stop_to_pynapple_end

    return nap.IntervalSet(start=support[:, 0], end=stop_to_pynapple_end(support[:, 1]))


def _series_to_pynapple(
    times: NDArray[np.float64],
    values: NDArray[Any],
    columns: Any,
    epochs: Any,
    *,
    metadata: dict[str, NDArray[Any]] | None,
) -> Any:
    """Validate a time series and build its ``Tsd`` / ``TsdFrame``."""
    # Prevalidate at this boundary so bad shapes raise an actionable
    # neurospatial ValueError rather than a raw pynapple AssertionError from
    # deep inside nap.Tsd / nap.TsdFrame.
    if times.ndim != 1:
        raise ValueError(
            f"`times` must be 1-D, got shape {times.shape}.\n"
            "  WHY: pynapple indexes a Tsd/TsdFrame by a 1-D time axis.\n"
            "  Fix: pass a 1-D array of timestamps."
        )
    if values.ndim not in (1, 2):
        raise ValueError(
            f"`values` must be 1-D or 2-D, got {values.ndim}-D (shape "
            f"{values.shape}).\n"
            "  WHY: a Tsd holds 1-D values, a TsdFrame holds 2-D "
            "(n_samples, n_columns) values.\n"
            "  Fix: pass values shaped (n,) or (n, n_columns)."
        )
    if len(times) != len(values):
        raise ValueError(
            f"`times` and `values` must have the same length, got "
            f"{len(times)} timestamps and {len(values)} value rows.\n"
            "  WHY: each value (row) is sampled at one timestamp.\n"
            "  Fix: pass times and values with matching first-axis length."
        )
    non_finite = np.flatnonzero(~np.isfinite(times))
    if non_finite.size:
        raise ValueError(
            f"`times` must be finite, but {non_finite.size} timestamp(s) are NaN "
            f"or infinite (first at index {int(non_finite[0])}).\n"
            "Why: a non-finite timestamp has no place on a pynapple time axis.\n"
            "Fix: keep = np.isfinite(times); to_pynapple(times[keep], "
            "values[keep])"
        )
    non_increasing = np.flatnonzero(times[1:] <= times[:-1])
    if non_increasing.size:
        i = int(non_increasing[0])
        raise ValueError(
            f"`times` must be strictly increasing, but timestamps at indices {i} "
            f"and {i + 1} are {times[i]!r} and {times[i + 1]!r}.\n"
            "Why: pynapple sorts timestamps without reordering the values, "
            "which would pair each value with another sample's time; repeated "
            "timestamps have no defined order.\n"
            'Fix: order = np.argsort(times, kind="stable"); '
            "to_pynapple(times[order], values[order]), after resolving any "
            "repeated timestamps."
        )
    if values.ndim == 2 and columns is not None and len(columns) != values.shape[1]:
        raise ValueError(
            f"`columns` has {len(columns)} labels but values has "
            f"{values.shape[1]} value columns.\n"
            "Why: pynapple replaces a mismatched label list with integer "
            "column names.\n"
            f"Fix: pass exactly {values.shape[1]} labels, or omit columns=."
        )
    support = _checked_support(times, epochs)

    # Import only after validation, so invalid input never reaches pynapple.
    nap = _require_pynapple()
    time_support = None if support is None else _interval_set(nap, support)
    if values.ndim == 1:
        return nap.Tsd(t=times, d=values, time_support=time_support)
    return nap.TsdFrame(
        t=times,
        d=values,
        columns=columns,
        time_support=time_support,
        metadata=metadata,
    )
