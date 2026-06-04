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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


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
