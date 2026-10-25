"""The :class:`SpikeTrains` ragged-spike-train container.

``SpikeTrains`` is the library's one spike container. It is justified
because ragged per-unit spike times genuinely do not fit a rectangular array;
every other neurospatial surface stays array-first (or ``xarray`` for labeled
grids). The container bundles the per-unit trains with their identity labels
(``unit_ids``) and an optional per-unit metadata table (``unit_table``), giving
users positional access (``st[i]``), label selection
(``st.sel(unit_id=...)``), iteration, and a metadata-driven
:meth:`SpikeTrains.filter`.

Interop role
------------
``SpikeTrains`` **duck-types as a spike group** so it flows
straight into the batch encoding/decoding functions through the shared
spike-input adapter (``neurospatial._spikes._as_spike_trains_with_ids``).
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

The adapter reads ``.index`` and iterates via ``__iter__``; it never calls
``__getitem__`` or ``sel``. ``st[i]`` is positional, like every other unit
container in neurospatial, so ``st[i]`` and ``rates[i]`` name the same unit.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._results import (
    UnitSelectionMixin,
    align_unit_table,
    read_only_copy,
    resolve_unit_ids,
)
from neurospatial._spikes import require_finite_spike_times

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["SpikeTrains"]


@dataclass(frozen=True, eq=False)
class SpikeTrains(UnitSelectionMixin):
    """Immutable container of ragged per-unit spike trains.

    Bundles per-unit spike-time arrays with their identity labels and optional
    per-unit metadata. The container is frozen: "modifying" it (e.g.
    :meth:`filter`) returns a new instance and never mutates the original.

    ``SpikeTrains`` duck-types as a spike group, so it can be
    passed directly where a batch encoding/decoding function accepts spike
    input (e.g. :func:`~neurospatial.encoding.compute_spatial_rates`); its
    :attr:`unit_ids` are then carried into the result.

    Parameters
    ----------
    trains : sequence of ndarray
        Ragged per-unit spike times, one 1-D array per unit (a ``list`` is the
        ergonomic input). Each element is copied to a read-only 1-D ``float64``
        array at construction, so later edits to the arrays you passed in do
        not reach the container; a non-1-D element raises
        :class:`ValueError`.
    unit_ids : ndarray or sequence, optional
        Identity label for each unit, one per train. Defaults to
        ``np.arange(len(trains))``; such generated labels do not count as
        labels when the container is passed to an encoder or decoder (an
        explicit ``unit_ids=`` there may name the units, and decoding pairs
        trains by position). Must be 1-D, length ``len(trains)``, and
        **unique** (:meth:`sel` and downstream selection require uniqueness);
        a length mismatch or duplicate labels raise :class:`ValueError`.
    unit_table : pandas.DataFrame or None, optional
        Per-unit metadata aligned to :attr:`unit_ids` (e.g. region, quality,
        depth, inclusion), one row per unit. A ``unit_id`` column or a named
        or non-default index aligns rows by identity; missing, extra or
        duplicate labels raise :class:`ValueError`. A plain unnamed
        ``RangeIndex`` without a ``unit_id`` column uses train order. The
        table is copied at construction. Enables :meth:`filter`.

    Attributes
    ----------
    trains : tuple of ndarray
        The per-unit ``float64`` spike-time arrays. Both the ``tuple`` and its
        arrays are read-only, so in-place mutation raises. Build a new
        :class:`SpikeTrains` to change the trains.
    unit_ids : ndarray
        Resolved identity labels, one per train (read-only).
    unit_table : pandas.DataFrame or None
        The per-unit metadata table, or ``None``.

    Other Parameters
    ----------------
    _unit_ids_generated : bool, default=False
        Internal identity flag carried by selection or fitting when unit labels
        were generated. Leave it at its default when supplying real unit IDs.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding import SpikeTrains
    >>> st = SpikeTrains(
    ...     [np.array([0.1, 0.5]), np.array([0.2, 0.3, 0.8])],
    ...     unit_ids=np.array([7, 9]),
    ... )
    >>> len(st)
    2
    >>> st.sel(unit_id=9)  # by label
    array([0.2, 0.3, 0.8])
    >>> st[1]  # by position: the same unit
    array([0.2, 0.3, 0.8])
    """

    trains: Sequence[NDArray[np.float64]]
    unit_ids: NDArray[Any] | None = None
    unit_table: pd.DataFrame | None = None
    # True only when ``unit_ids`` was omitted and ``arange`` labels were
    # generated; such labels are not caller-supplied (see ``.index``).
    _unit_ids_generated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Coerce trains, resolve/validate unit ids, and validate the table."""
        coerced: list[NDArray[np.float64]] = []
        for i, train in enumerate(self.trains):
            arr = read_only_copy(train, np.float64)
            if arr.ndim != 1:
                raise ValueError(
                    f"Each spike train must be 1-D, but train {i} has shape "
                    f"{arr.shape}.\n"
                    "  WHY: SpikeTrains holds one 1-D spike-time array per unit.\n"
                    "  Fix: pass a 1-D array of spike times for each unit."
                )
            coerced.append(arr)
        # Store as a tuple so in-place mutation (e.g. ``st.trains.append(...)``)
        # raises, keeping the ``len(unit_ids) == len(trains)`` invariant intact.
        object.__setattr__(self, "trains", tuple(coerced))

        n_units = len(coerced)
        if self.unit_ids is None:
            object.__setattr__(self, "_unit_ids_generated", True)
        resolved = resolve_unit_ids(self.unit_ids, n_units, context="SpikeTrains")
        require_finite_spike_times(coerced, resolved)
        object.__setattr__(self, "unit_ids", resolved)

        object.__setattr__(
            self,
            "unit_table",
            align_unit_table(self.unit_table, self.unit_ids, context="SpikeTrains"),
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle or deepcopy, keeping the arrays read-only."""
        super().__setstate__(state)
        for train in self.trains:
            train.flags.writeable = False

    @property
    def index(self) -> NDArray[Any]:
        """Unit ids, one per train (the spike-group key surface).

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
        from a non-Mapping spike group.

        Yields
        ------
        ndarray
            Each unit's 1-D ``float64`` spike-time array, in stored order.
        """
        return iter(self.trains)

    def __getitem__(self, idx: int) -> NDArray[np.float64]:
        """Return the train of the unit at position ``idx``.

        ``st[i]`` is the ``i``-th unit in stored order, matching iteration,
        :attr:`trains` and the rows of every result computed from this
        container. Select by label with :meth:`sel`.

        Parameters
        ----------
        idx : int
            Unit position (0-based; negative positions count from the end).

        Returns
        -------
        ndarray
            The unit's read-only 1-D ``float64`` spike-time array.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        TypeError
            If ``idx`` is not an integer; select labels with :meth:`sel`.
        """
        return self.trains[self._unit_position(idx)]

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
            row-subset of the table, preserving its identity index).

        Raises
        ------
        ValueError
            If :attr:`unit_table` is ``None`` (there is nothing to filter on).

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from neurospatial.encoding import SpikeTrains
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
                "  Fix: construct SpikeTrains with a unit_table (one row per "
                "unit) before calling filter()."
            )

        import pandas as pd

        # Metadata was aligned to unit_ids at construction. Query a positional
        # copy so selection preserves that alignment and the identity index.
        positional = self.unit_table.reset_index(drop=True)
        positions = positional.query(query).index.to_numpy()

        new_trains = [self.trains[i] for i in positions]
        new_unit_ids = np.asarray(self.unit_ids)[positions]
        new_table = self.unit_table.iloc[positions].copy()
        if self.unit_table.index.name is None and self.unit_table.index.equals(
            pd.RangeIndex(len(self))
        ):
            new_table = new_table.reset_index(drop=True)
        return SpikeTrains(
            trains=new_trains,
            unit_ids=new_unit_ids,
            unit_table=new_table,
            _unit_ids_generated=self._unit_ids_generated,
        )
