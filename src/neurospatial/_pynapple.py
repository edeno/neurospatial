"""Read pynapple objects passed straight to analysis functions.

Analysis functions accept pynapple objects by duck typing and never import
pynapple. This module holds what those inputs carry beyond plain arrays: a
``TsGroup``'s per-unit metadata, a ``TsdFrame``'s column names and an
``IntervalSet``'s inclusive ends. :mod:`neurospatial.io.pynapple` applies the
same rules to explicit conversions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import _is_pynapple_object, pynapple_end_to_stop

if TYPE_CHECKING:
    import pandas as pd

_SPATIAL_AXES = ("x", "y", "z")


def is_pynapple_type(value: Any, *names: str) -> bool:
    """Return whether ``value`` is a pynapple object of one of the named types."""
    return _is_pynapple_object(value) and type(value).__name__ in names


def refuse_permuted_spatial_columns(columns: Any, *, fix: str) -> None:
    """Refuse a TsdFrame whose columns are x/y(/z) in another order.

    Parameters
    ----------
    columns : sequence
        The frame's column labels.
    fix : str
        The corrected call, shown after ``Fix:``; ``{frame}`` is replaced by
        the reordered frame, ``tsdframe.loc[[...]]``.

    Raises
    ------
    ValueError
        If the labels are distinct spatial axis names not in x, y, z order.
    """
    labels = [str(c) for c in columns]
    lowered = [label.lower() for label in labels]
    spatial = set(lowered) <= set(_SPATIAL_AXES) and len(set(lowered)) == len(lowered)
    if not spatial or lowered == sorted(lowered):
        return
    canonical = sorted(lowered)
    frame = f"tsdframe.loc[{sorted(labels, key=str.lower)}]"
    raise ValueError(
        f"TsdFrame columns {labels} are spatial axes not in x, y, z order.\n"
        "Why: neurospatial reads positions by column position, so these values "
        f"would be used as ({', '.join(lowered)}) where every analysis expects "
        f"({', '.join(canonical)}), silently swapping the axes.\n"
        "Fix: reorder the columns first, " + fix.format(frame=frame) + "."
    )


def interval_rows(intervals: Any) -> NDArray[np.float64]:
    """Return a pynapple ``IntervalSet`` as half-open ``(n, 2)`` rows.

    Rows keep their order and are not merged. Each inclusive pynapple ``end``
    becomes the next float above it, so a row holds the same timestamps.

    Parameters
    ----------
    intervals : pynapple.IntervalSet

    Returns
    -------
    ndarray of float64, shape (n_intervals, 2)
        ``[start, stop)`` rows in seconds, an owned copy.
    """
    return np.column_stack(
        [
            np.array(intervals.start, dtype=np.float64),
            pynapple_end_to_stop(np.asarray(intervals.end, dtype=np.float64)),
        ]
    )


def tsgroup_unit_table(group: Any) -> pd.DataFrame | None:
    """Return a pynapple ``TsGroup``'s metadata as a unit table, or None.

    Rows are indexed by unit id (index named ``unit_id``). pynapple's ``rate``
    column is dropped: pynapple computes it as spike count over the group's
    time support, which is not the time an analysis observed, so it is not a
    property of the unit. A group with no other metadata gives None.

    Parameters
    ----------
    group : pynapple.TsGroup

    Returns
    -------
    pandas.DataFrame or None
        One row per unit, in group order.
    """
    metadata = getattr(group, "metadata", None)
    if metadata is None:
        return None
    table = metadata.drop(columns=["rate"], errors="ignore")
    if table.shape[1] == 0:
        return None
    table = table.copy()
    table.index = table.index.rename("unit_id")
    return table
