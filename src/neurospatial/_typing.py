"""Structural types for neurospatial's spike-input boundary.

This module names the spike inputs public functions accept, so third-party
objects (a pynapple ``TsGroup``, a dict of unit id to spike times) and
:class:`~neurospatial.encoding.SpikeTrains` type-check without the scientific
core importing or ``isinstance``-checking those libraries. It holds types
only: the adapters that convert these inputs to plain NumPy arrays, once at
the public entry point, live in :mod:`neurospatial._spikes` and
:mod:`neurospatial._validation`; everything downstream is array-only.

Design rules:

- **Arrays are the universal baseline.** The adapters return plain
  ``float64`` arrays; a plain-array caller is unchanged byte-for-byte.
- **Duck-typing, never ``isinstance`` on third-party types.** The adapters
  test for attributes (``.t`` / ``.values`` / ``.index``), not concrete types.
- **Import-light.** This module imports only ``typing`` + ``numpy`` at runtime;
  it never imports pynapple, pynwb, xarray, or any other optional/heavy
  dependency.

Types
-----
SpikeTrainsLike
    Every spike input a population function accepts, as ``ArrayLike`` names
    every array input: a sequence of per-unit arrays, a 1-D or NaN-padded 2-D
    array, a dict of unit id to spike times, a
    :class:`~neurospatial.encoding.SpikeTrains` container or a
    :class:`SpikeGroupLike` group.
SpikeGroupLike
    A per-unit spike-train collection **indexable by unit id**: exposes
    ``.index`` (the unit ids / keys) and ``obj[unit_id]`` returning a per-unit
    timestamp source with a ``.t`` attribute. A real pynapple ``TsGroup`` (a
    ``collections.UserDict`` — iterating it yields **keys**, not trains)
    satisfies this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike

    from neurospatial.encoding.spike_trains import SpikeTrains

__all__ = [
    "SpikeGroupLike",
    "SpikeTrainsLike",
]


class _HasTimes(Protocol):
    """A per-unit timestamp source exposing ``.t`` (pynapple ``Ts`` conforms)."""

    @property
    def t(self) -> NDArray[np.float64]:
        """1-D spike timestamps (seconds)."""
        ...


class SpikeGroupLike(Protocol):
    """Structural type for a per-unit spike-train collection indexable by id.

    Captures the *interesting* third-party case — a real pynapple ``TsGroup``
    (a ``collections.UserDict``: iterating it yields **unit-id keys**, not
    trains). It exposes ``.index`` (the unit ids / keys) and supports
    ``obj[unit_id]`` returning a per-unit timestamp source with a ``.t``
    attribute. :class:`~neurospatial.encoding.SpikeTrains` exposes the same
    ``.index`` but indexes by position and is read by iteration. The
    id-surfacing adapter
    :func:`neurospatial._spikes._as_spike_trains_with_ids` extracts trains by
    **indexing each id** (``obj[uid].t``); for a ``TsGroup`` it must NOT iterate
    the object, because iterating a ``UserDict`` yields its keys, not its trains.

    The spike boundary accepts more than this Protocol; :data:`SpikeTrainsLike`
    names the full set. Plain-array forms carry no unit ids; only a group
    surfaces ``unit_ids``. Normalization is done by the adapter, not by an
    ``isinstance`` check against this Protocol.
    """

    @property
    def index(self) -> Any:
        """Unit ids / keys, one per train (array-like / sequence)."""
        ...

    def __getitem__(self, unit_id: Any) -> _HasTimes:
        """Return the per-unit timestamp source for ``unit_id``."""
        ...


SpikeTrainsLike: TypeAlias = (
    "ArrayLike | Sequence[ArrayLike] | Mapping[Any, ArrayLike] | SpikeTrains"
    " | SpikeGroupLike"
)
"""Spike input accepted by every population encoder, test and decoder.

A sequence of per-unit spike-time arrays (the canonical form), a 1-D array
(one unit), a NaN-padded 2-D ``(n_units, max_spikes)`` array, a dict of unit
id to spike times (``{"ca1_01": times, ...}``), a
:class:`~neurospatial.encoding.SpikeTrains` container or a
:class:`SpikeGroupLike` group such as a pynapple ``TsGroup``. A dict, a group
and a ``SpikeTrains`` with caller-supplied labels carry their ``unit_ids``
into the result.
"""
