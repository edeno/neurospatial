"""Structural typing surface for neurospatial's public input boundary.

This module holds the small set of Protocols and boundary adapters that let
third-party objects (pynapple ``Tsd`` / ``TsdFrame`` / ``TsGroup``, and the
future :class:`SpikeTrains` container) flow into the **array-first** scientific
core *without* the core ever importing or ``isinstance``-checking those
libraries. Conversion to plain NumPy arrays happens once, at the public entry
point, via the adapters here; everything downstream is array-only.

Design rules:

- **Arrays are the universal baseline.** Adapters return plain ``float64``
  arrays; a plain-array caller is unchanged byte-for-byte.
- **Duck-typing, never ``isinstance`` on third-party types.** The adapters
  test for attributes (``.t`` / ``.values`` / ``.index``), not concrete types.
- **Import-light.** This module imports only ``typing`` + ``numpy`` and the
  first-party :class:`EnvironmentProtocol`; it never imports pynapple, pynwb,
  xarray, or any other optional/heavy dependency.

Types
-----
SpikeTrainsLike
    Every spike input a population function accepts, as ``ArrayLike`` names
    every array input: a sequence of per-unit arrays, a 1-D or NaN-padded 2-D
    array, a :class:`~neurospatial.encoding.SpikeTrains` container or a
    :class:`SpikeGroupLike` group.
SpikeGroupLike
    A per-unit spike-train collection **indexable by unit id**: exposes
    ``.index`` (the unit ids / keys) and ``obj[unit_id]`` returning a per-unit
    timestamp source with a ``.t`` attribute. A real pynapple ``TsGroup`` (a
    ``collections.UserDict`` — iterating it yields **keys**, not trains)
    satisfies this.
EnvironmentLike
    The minimal structural surface shared by
    :class:`~neurospatial.environment.Environment` and its polar sibling
    :class:`~neurospatial.environment.polar.EgocentricPolarEnvironment`
    (``bin_centers`` / ``connectivity`` / ``neighbors``) — matched to the
    runtime :func:`is_environment_like` check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    import networkx as nx
    from numpy.typing import ArrayLike

    from neurospatial.encoding.spike_trains import SpikeTrains

__all__ = [
    "EnvironmentLike",
    "SpikeGroupLike",
    "SpikeTrainsLike",
    "is_environment_like",
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
    :func:`neurospatial.encoding.as_spike_trains_with_ids` extracts trains by
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
    "ArrayLike | Sequence[ArrayLike] | SpikeTrains | SpikeGroupLike"
)
"""Spike input accepted by every population encoder, test and decoder.

A sequence of per-unit spike-time arrays (the canonical form), a 1-D array
(one unit), a NaN-padded 2-D ``(n_units, max_spikes)`` array, a
:class:`~neurospatial.encoding.SpikeTrains` container or a
:class:`SpikeGroupLike` group such as a pynapple ``TsGroup``. Groups and
``SpikeTrains`` with caller-supplied labels carry their ``unit_ids`` into the
result.
"""


class EnvironmentLike(Protocol):
    """Minimal structural surface of a neurospatial spatial environment.

    The **narrow** public contract shared by
    :class:`~neurospatial.environment.Environment` and its sibling
    :class:`~neurospatial.environment.polar.EgocentricPolarEnvironment` (which is
    NOT an ``Environment`` subclass, so ``isinstance(polar, Environment)`` is
    ``False``). Deliberately just the three members a consumer needs to treat an
    object as a spatial environment — ``bin_centers``, ``connectivity``,
    ``neighbors`` — matched exactly to the runtime :func:`is_environment_like`
    duck-check (:data:`_ENVIRONMENT_LIKE_ATTRS`).

    It intentionally does NOT republish the internal ``EnvironmentProtocol`` (the
    mixin ``self`` type), whose ~14 members are private implementation surface,
    not a public contract. Use this in annotations, and use
    :func:`is_environment_like` for the runtime duck-check.
    """

    @property
    def bin_centers(self) -> NDArray[np.float64]:
        """Bin-center coordinates, shape ``(n_bins, n_dims)``."""
        ...

    @property
    def connectivity(self) -> nx.Graph:
        """Connectivity graph over active bins."""
        ...

    def neighbors(self, bin_index: int) -> list[int]:
        """Active-bin indices adjacent to ``bin_index``."""
        ...


# Attributes shared by ``Environment`` and ``EgocentricPolarEnvironment`` that a
# consumer needs to treat an object as a spatial environment. Kept intentionally
# minimal: enough to distinguish a real environment from an arbitrary object,
# without pinning down the (large) full ``EnvironmentProtocol`` surface (a
# runtime ``isinstance`` against that Protocol misfires — even a real
# ``Environment`` lacks some of its lazily-created private attributes at the
# instance level).
#
# Deliberately excludes the ``@check_fitted``-guarded ``n_bins`` / ``n_dims``
# properties: probing those on an *unfitted* environment raises
# ``EnvironmentNotFittedError`` (not ``AttributeError``), which ``hasattr`` would
# let propagate. ``bin_centers`` / ``connectivity`` / ``neighbors`` are present
# regardless of fitted state, so an unfitted environment is still recognized as
# environment-like (and any downstream fitted check reports the real problem).
_ENVIRONMENT_LIKE_ATTRS: tuple[str, ...] = (
    "bin_centers",
    "connectivity",
    "neighbors",
)


def is_environment_like(obj: object) -> bool:
    """Return whether ``obj`` is duck-typed as a spatial environment.

    Checks for the minimal shared surface of
    :class:`~neurospatial.environment.Environment` and its sibling
    :class:`~neurospatial.environment.polar.EgocentricPolarEnvironment`
    (``bin_centers``, ``connectivity``, ``neighbors``).

    Use this in place of ``isinstance(obj, Environment)`` at internal call sites
    that must also accept the polar sibling — ``isinstance(polar, Environment)``
    is ``False`` because ``EgocentricPolarEnvironment`` is a sibling type, not a
    subclass, which would wrongly reject a legitimate environment.

    Parameters
    ----------
    obj : object
        Candidate object.

    Returns
    -------
    bool
        ``True`` if ``obj`` exposes every attribute a spatial environment must
        provide, ``False`` otherwise.
    """
    return all(hasattr(obj, attr) for attr in _ENVIRONMENT_LIKE_ATTRS)
