"""Structural typing surface for neurospatial's public input boundary.

This module holds the small set of Protocols and boundary adapters that let
third-party objects (pynapple ``Tsd`` / ``TsdFrame`` / ``TsGroup``, and the
future :class:`SpikeTrains` container) flow into the **array-first** scientific
core *without* the core ever importing or ``isinstance``-checking those
libraries. Conversion to plain NumPy arrays happens once, at the public entry
point, via the adapters here; everything downstream is array-only.

Design rules (from the Phase 3 design lock):

- **Arrays are the universal baseline.** Adapters return plain ``float64``
  arrays; a plain-array caller is unchanged byte-for-byte.
- **Duck-typing, never ``isinstance`` on third-party types.** The adapters
  test for attributes (``.t`` / ``.values`` / ``.index``), not concrete types.
- **Import-light.** This module imports only ``typing`` + ``numpy`` and the
  first-party :class:`EnvironmentProtocol`; it never imports pynapple, pynwb,
  xarray, or any other optional/heavy dependency.

Protocols
---------
PositionLike
    A position/time-series source exposing ``.t`` and ``.values`` (pynapple
    ``Tsd`` / ``TsdFrame`` conform).
SpikeTrainsLike
    A per-unit spike-train collection **indexable by unit id**: exposes
    ``.index`` (the unit ids / keys) and ``obj[unit_id]`` returning a per-unit
    timestamp source with a ``.t`` attribute. A real pynapple ``TsGroup`` (a
    ``collections.UserDict`` — iterating it yields **keys**, not trains) and the
    future :class:`SpikeTrains` container both satisfy this.
EnvironmentLike
    The minimal structural surface shared by
    :class:`~neurospatial.environment.Environment` and its polar sibling
    :class:`~neurospatial.environment.polar.EgocentricPolarEnvironment`
    (``bin_centers`` / ``connectivity`` / ``neighbors``) — matched to the
    runtime :func:`is_environment_like` check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import networkx as nx

__all__ = [
    "EnvironmentLike",
    "PositionLike",
    "SpikeTrainsLike",
    "as_times_positions",
    "is_environment_like",
]


class PositionLike(Protocol):
    """Structural type for a position / time-series source.

    An object exposing ``.t`` (1-D timestamps, seconds) and ``.values``
    (position samples, shape ``(n,)`` or ``(n, n_dims)``). pynapple ``Tsd`` and
    ``TsdFrame`` conform.

    The member types are intentionally loose (``NDArray[Any]``): pynapple's
    ``.values`` is frequently not ``float64`` (the boundary adapter coerces).
    The runtime adapter :func:`as_times_positions` additionally accepts a ``.d``
    alias for ``.values`` (pynapple's data accessor), so an object exposing
    ``.t`` + ``.d`` conforms in practice even though ``.d`` is not declared here.

    Convert an instance to plain arrays with :func:`as_times_positions`; the
    scientific core only ever sees the returned ``(times, positions)`` arrays.
    """

    @property
    def t(self) -> NDArray[Any]:
        """1-D timestamps (seconds)."""
        ...

    @property
    def values(self) -> NDArray[Any]:
        """Position samples, shape ``(n,)`` or ``(n, n_dims)``."""
        ...


class _HasTimes(Protocol):
    """A per-unit timestamp source exposing ``.t`` (pynapple ``Ts`` conforms)."""

    @property
    def t(self) -> NDArray[np.float64]:
        """1-D spike timestamps (seconds)."""
        ...


class SpikeTrainsLike(Protocol):
    """Structural type for a per-unit spike-train collection indexable by id.

    Captures the *interesting* third-party case — a real pynapple ``TsGroup``
    (a ``collections.UserDict``: iterating it yields **unit-id keys**, not
    trains), or the future :class:`~neurospatial.encoding.SpikeTrains`
    container. Both expose ``.index`` (the unit ids / keys) and support
    ``obj[unit_id]`` returning a per-unit timestamp source with a ``.t``
    attribute. The id-surfacing adapter
    :func:`neurospatial.encoding.as_spike_trains_with_ids` extracts trains by
    **indexing each id** (``obj[uid].t``); for a ``TsGroup`` it must NOT iterate
    the object, because iterating a ``UserDict`` yields its keys, not its trains.

    It is primarily a *documentation / typing* alias for the union of inputs the
    spike boundary accepts. In practice
    :func:`neurospatial.encoding.as_spike_trains` /
    :func:`neurospatial.encoding.as_spike_trains_with_ids` accept **more** than
    this Protocol: a ``Sequence[NDArray]`` (the canonical format), a 2-D
    NaN-padded ``(n_units, max_spikes)`` array, or a single 1-D array. Those
    plain-array forms carry no unit ids; only a ``SpikeTrainsLike`` object
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


def _is_position_like(obj: object) -> bool:
    """Return whether ``obj`` is a ``PositionLike`` (has ``.t`` + values).

    Duck-typed: a plain NumPy array has ``.T`` (transpose) but not lowercase
    ``.t``; a pandas ``Series`` / ``DataFrame`` has ``.values`` but not ``.t``.
    So requiring both ``.t`` and (``.values`` or ``.d``) cleanly selects only a
    pynapple ``Tsd`` / ``TsdFrame``-like object.
    """
    return hasattr(obj, "t") and (hasattr(obj, "values") or hasattr(obj, "d"))


def as_times_positions(
    obj_or_times: PositionLike | ArrayLike,
    positions: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize a position input to plain ``(times, positions)`` arrays.

    Accepts **either** a single :class:`PositionLike` object (e.g. a pynapple
    ``Tsd`` / ``TsdFrame`` exposing ``.t`` and ``.values`` / ``.d``) **or** an
    explicit ``(times, positions)`` array pair, and returns plain ``float64``
    arrays. This is the position boundary adapter: it runs once at a public
    entry point so the scientific core only ever sees arrays.

    Parameters
    ----------
    obj_or_times : PositionLike or array-like
        Either a ``PositionLike`` object carrying both timestamps and position
        samples, or a 1-D array of timestamps (in which case ``positions`` must
        be supplied).
    positions : array-like or None, default=None
        Position samples, shape ``(n_samples,)`` or ``(n_samples, n_dims)``.
        Required when ``obj_or_times`` is a timestamp array; must be ``None``
        when ``obj_or_times`` is a ``PositionLike`` object.

    Returns
    -------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps as ``float64``.
    positions : NDArray[np.float64], shape (n_samples,) or (n_samples, n_dims)
        Position samples as ``float64``.

    Raises
    ------
    ValueError
        If a ``PositionLike`` object is passed together with a non-``None``
        ``positions`` (ambiguous), or if a timestamp array is passed without
        ``positions``.
    """
    if _is_position_like(obj_or_times):
        if positions is not None:
            raise ValueError(
                "as_times_positions received both a PositionLike object (with "
                ".t/.values) and a separate `positions` array. Pass EITHER a "
                "single PositionLike object OR (times, positions) arrays, not "
                "both."
            )
        # Duck-typed extraction: prefer ``.values``, fall back to ``.d`` (the
        # pynapple data alias). No isinstance / pynapple import.
        values = getattr(obj_or_times, "values", None)
        if values is None:
            values = obj_or_times.d  # type: ignore[union-attr]
        times = np.asarray(obj_or_times.t, dtype=np.float64)  # type: ignore[union-attr]
        return times, np.asarray(values, dtype=np.float64)

    if positions is None:
        raise ValueError(
            "as_times_positions received a timestamp array but no `positions`. "
            "Pass positions alongside times, or pass a single PositionLike "
            "object (e.g. a pynapple Tsd/TsdFrame) exposing .t and .values."
        )
    times = np.asarray(obj_or_times, dtype=np.float64)
    return times, np.asarray(positions, dtype=np.float64)
