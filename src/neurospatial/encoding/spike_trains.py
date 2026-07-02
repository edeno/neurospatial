"""The :class:`SpikeTrains` ragged-spike-train container.

``SpikeTrains`` is the single new container Phase 3 adds. It is justified
because ragged per-unit spike times genuinely do not fit a rectangular array;
every other neurospatial surface stays array-first (or ``xarray`` for labeled
grids). The container bundles the per-unit trains with their identity labels
(``unit_ids``) and an optional per-unit metadata table (``unit_table``), giving
users label access (``st[unit_id]``), iteration, and a
metadata-driven :meth:`SpikeTrains.filter`.

Interop role
------------
``SpikeTrains`` **duck-types as a** ``SpikeTrainsLike`` **group** so it flows
straight into the batch encoding/decoding functions through the Phase 3.1
spike-input adapter (:func:`neurospatial.encoding.as_spike_trains_with_ids`).
That adapter detects a group via a non-callable ``.index`` and then extracts
trains in one of two ways:

- if the group is a :class:`collections.abc.Mapping` (a real pynapple
  ``TsGroup`` is a ``UserDict``) it indexes ``group[uid]`` (iterating a Mapping
  would yield KEYS, not trains);
- otherwise it **iterates** the object to collect the trains.

``SpikeTrains`` is designed for that *iterate* branch: it is **not** a
``Mapping``, its ``.index`` is a non-callable property returning ``unit_ids``,
and its :meth:`~SpikeTrains.__iter__` yields the per-unit **train arrays** (not
the ids). This keeps unit identity from being silently dropped and avoids the
"garbage trains" footgun that would arise if iteration yielded the ids.

Label access (``st[unit_id]``) and the adapter's iteration coexist because they
use different dunders: the adapter reads ``.index`` and iterates via
``__iter__`` (positional order), while ``__getitem__`` is user-facing **label**
access keyed by unit id.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._results import resolve_unit_ids, validate_unit_table

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["SpikeTrains"]


@dataclass(frozen=True)
class SpikeTrains:
    """Immutable container of ragged per-unit spike trains.

    Bundles per-unit spike-time arrays with their identity labels and optional
    per-unit metadata. The container is frozen: "modifying" it (e.g.
    :meth:`filter`) returns a new instance and never mutates the original.

    ``SpikeTrains`` duck-types as a ``SpikeTrainsLike`` group, so it can be
    passed directly where a batch encoding/decoding function accepts spike
    input (e.g. :func:`~neurospatial.encoding.compute_spatial_rates`); its
    :attr:`unit_ids` are then carried into the result.

    Parameters
    ----------
    trains : list of ndarray
        Ragged per-unit spike times, one 1-D array per unit. Each element is
        coerced to a 1-D ``float64`` array at construction; a non-1-D element
        raises :class:`ValueError`.
    unit_ids : ndarray or sequence, optional
        Identity label for each unit, one per train. Defaults to
        ``np.arange(len(trains))``. Must be 1-D, length ``len(trains)``, and
        **unique** (label access and downstream selection require uniqueness);
        a length mismatch or duplicate labels raise :class:`ValueError`.
    unit_table : pandas.DataFrame or None, optional
        Per-unit metadata aligned to :attr:`unit_ids` (e.g. region, quality,
        depth, inclusion), one row per unit. When provided its length must
        equal ``len(trains)`` or :class:`ValueError` is raised. Enables
        :meth:`filter`.

    Attributes
    ----------
    trains : list of ndarray
        The per-unit ``float64`` spike-time arrays. The list is a plain
        ``list`` for ergonomics, but the dataclass is frozen: do not mutate it
        in place -- build a new :class:`SpikeTrains` instead.
    unit_ids : ndarray
        Resolved identity labels, one per train.
    unit_table : pandas.DataFrame or None
        The per-unit metadata table, or ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import SpikeTrains
    >>> st = SpikeTrains(
    ...     [np.array([0.1, 0.5]), np.array([0.2, 0.3, 0.8])],
    ...     unit_ids=np.array([7, 9]),
    ... )
    >>> len(st)
    2
    >>> st[9]  # label access, by unit id (not position)
    array([0.2, 0.3, 0.8])
    """

    trains: list[NDArray[np.float64]]
    unit_ids: NDArray[Any] | None = None
    unit_table: pd.DataFrame | None = field(default=None)

    def __post_init__(self) -> None:
        """Coerce trains, resolve/validate unit ids, and validate the table."""
        coerced: list[NDArray[np.float64]] = []
        for i, train in enumerate(self.trains):
            arr = np.asarray(train, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(
                    f"Each spike train must be 1-D, but train {i} has shape "
                    f"{arr.shape}.\n"
                    "  WHY: SpikeTrains holds one 1-D spike-time array per unit.\n"
                    "  HOW: pass a 1-D array of spike times for each unit."
                )
            coerced.append(arr)
        object.__setattr__(self, "trains", coerced)

        n_units = len(coerced)
        resolved = resolve_unit_ids(self.unit_ids, n_units, context="SpikeTrains")

        # Uniqueness is required: label access (st[unit_id]) and downstream
        # label-based selection are ambiguous with duplicate ids.
        unique_vals, counts = np.unique(resolved, return_counts=True)
        if unique_vals.shape[0] != resolved.shape[0]:
            duplicated = unique_vals[counts > 1]
            raise ValueError(
                f"unit_ids must be unique in SpikeTrains: duplicated label(s) "
                f"{duplicated.tolist()}.\n"
                "  WHY: label access st[unit_id] and downstream selection "
                "require one row per label.\n"
                "  HOW: pass distinct unit_ids, or omit them to default to "
                "np.arange(n_units)."
            )
        object.__setattr__(self, "unit_ids", resolved)

        validate_unit_table(self.unit_table, n_units, context="SpikeTrains")

    @property
    def index(self) -> NDArray[Any]:
        """Unit ids, one per train (the ``SpikeTrainsLike`` group key surface).

        A non-callable property (unlike a ``list``/``tuple``'s ``.index``
        method), so the spike-input adapter's group detector recognizes this
        container and surfaces its :attr:`unit_ids`.

        Returns
        -------
        ndarray
            The :attr:`unit_ids` array.
        """
        return np.asarray(self.unit_ids)

    def __len__(self) -> int:
        """Return the number of units.

        Returns
        -------
        int
            Number of spike trains (units).
        """
        return len(self.trains)

    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        """Iterate over the per-unit train arrays in order.

        Iteration yields the **train arrays** (not the unit ids), which is what
        the spike-input adapter's iterate branch relies on to extract trains
        from a non-Mapping ``SpikeTrainsLike`` group.

        Yields
        ------
        ndarray
            Each unit's 1-D ``float64`` spike-time array, in stored order.
        """
        return iter(self.trains)

    def __getitem__(self, unit_id: Any) -> NDArray[np.float64]:
        """Return a unit's train by **label** (unit id), not by position.

        Parameters
        ----------
        unit_id : Any
            The identity label of the desired unit (an entry of
            :attr:`unit_ids`).

        Returns
        -------
        ndarray
            The 1-D ``float64`` spike-time array for that unit.

        Raises
        ------
        KeyError
            If ``unit_id`` is not present in :attr:`unit_ids`.

        Notes
        -----
        This is user-facing label access (``st[7]`` selects the unit whose id
        is ``7``). It is intentionally distinct from the adapter's iteration:
        the adapter never calls ``__getitem__``, so label-vs-position access
        does not affect interop.
        """
        ids = np.asarray(self.unit_ids)
        matches = np.flatnonzero(ids == unit_id)
        if matches.size == 0:
            raise KeyError(
                f"unit_id {unit_id!r} not found in SpikeTrains "
                f"(unit_ids={ids.tolist()})."
            )
        return self.trains[int(matches[0])]

    def filter(self, query: str) -> SpikeTrains:
        """Return a new :class:`SpikeTrains` keeping units matching ``query``.

        Selects units whose :attr:`unit_table` rows satisfy the pandas
        ``DataFrame.query`` expression, keeping :attr:`trains`, :attr:`unit_ids`,
        and :attr:`unit_table` aligned. The original container is unchanged.

        Parameters
        ----------
        query : str
            A :meth:`pandas.DataFrame.query` expression over the
            :attr:`unit_table` columns, e.g. ``"region == 'CA1'"`` or
            ``"quality > 0.5 and region == 'CA3'"``.

        Returns
        -------
        SpikeTrains
            A new container with only the matching units (trains, ids, and a
            row-subset of the table, its index reset).

        Raises
        ------
        ValueError
            If :attr:`unit_table` is ``None`` (there is nothing to filter on).

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from neurospatial import SpikeTrains
        >>> st = SpikeTrains(
        ...     [np.array([0.1]), np.array([0.2]), np.array([0.3])],
        ...     unit_ids=np.array([10, 20, 30]),
        ...     unit_table=pd.DataFrame({"region": ["CA1", "CA3", "CA1"]}),
        ... )
        >>> ca1 = st.filter("region == 'CA1'")
        >>> ca1.unit_ids
        array([10, 30])
        """
        if self.unit_table is None:
            raise ValueError(
                "Cannot filter a SpikeTrains with unit_table=None: there is no "
                "per-unit metadata to query.\n"
                "  WHY: filter() selects units by matching unit_table rows.\n"
                "  HOW: construct SpikeTrains with a unit_table (one row per "
                "unit) before calling filter()."
            )

        # Compute matched *positions* on a positionally-indexed copy so the
        # result is robust to any original index (including non-unique labels).
        positional = self.unit_table.reset_index(drop=True)
        positions = positional.query(query).index.to_numpy()

        new_trains = [self.trains[i] for i in positions]
        new_unit_ids = np.asarray(self.unit_ids)[positions]
        new_table = self.unit_table.iloc[positions].reset_index(drop=True)
        return SpikeTrains(
            trains=new_trains, unit_ids=new_unit_ids, unit_table=new_table
        )
