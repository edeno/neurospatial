"""The frozen :class:`Session` bundle and :func:`load_session` loader.

A :class:`Session` groups the objects an analysis session revolves around --
the spatial ``env``, the animal's ``position`` (times + coordinates), the
population ``spikes`` (with unit identity), optional ``epochs``, and free-form
``metadata`` -- into a single discoverable, immutable bundle.

Bundle, not god-object
----------------------
``Session`` is a **discoverability bundle**, not a god-object: it carries data
and exposes the raw arrays (``session.times`` / ``session.positions`` /
``session.spikes``), but it holds **no** heavy analysis methods. Compute stays
functional, taking the bundle's fields directly, e.g.::

    from neurospatial.encoding import compute_spatial_rates

    rates = compute_spatial_rates(
        session.env, session.spikes, session.times, session.positions
    )

Keeping analysis out of the bundle avoids a sprawling god-object and keeps the
arrays trivially extractable for any function in the library.

Immutability
------------
``Session`` is a frozen dataclass. "Modifying" it returns a **new** bundle and
never mutates the original: :meth:`Session.with_environment` swaps the env and
:meth:`Session.restrict` slices position and spikes to a set of epochs. The
spike restriction is **identity-preserving** -- restriction only trims spikes
per unit (it never drops units), so ``unit_ids`` and ``unit_table`` ride along
unchanged.

Array-first / optional-dep discipline
-------------------------------------
This module is array-first and **never imports pynwb or pynapple at module
load**. :meth:`Session.from_nwb` delegates all NWB file I/O to the lazy
:mod:`neurospatial.io.nwb` readers (which import pynwb only when called), so
``import neurospatial.recording`` stays cheap and dependency-free. The deep,
lazily-materialized NWB read path (``lazy=True``) is intentionally deferred to a
later phase and is **not** added here.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._typing import (
    PositionLike,
    as_times_positions,
    is_environment_like,
)

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial._typing import EnvironmentLike
    from neurospatial.encoding.spike_trains import SpikeTrains

__all__ = ["Position", "Session", "load_session"]


@dataclass(frozen=True)
class Position:
    """Minimal frozen position holder exposing ``.t`` and ``.values``.

    A tiny, immutable stand-in that lets :attr:`Session.position` expose ``.t``
    (timestamps) and ``.values`` (positions) uniformly, regardless of whether
    the session was built from plain arrays (this holder) or from a pynapple
    ``Tsd`` / ``TsdFrame`` (which already conforms). It duck-types as a
    ``PositionLike`` (see :mod:`neurospatial._typing`).

    Parameters
    ----------
    t : ndarray, shape (n,)
        Timestamps in seconds. Coerced to ``float64``.
    values : ndarray, shape (n,) or (n, n_dims)
        Position samples. Coerced to ``float64``.
    """

    t: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Coerce ``t`` and ``values`` to ``float64`` arrays."""
        object.__setattr__(self, "t", np.asarray(self.t, dtype=np.float64))
        object.__setattr__(self, "values", np.asarray(self.values, dtype=np.float64))


@dataclass(frozen=True)
class Session:
    """A frozen bundle of the data an analysis session revolves around.

    Groups the spatial environment, the animal's position, the population
    spikes (with unit identity), optional analysis epochs, and free-form
    metadata into one immutable, discoverable object. It is a **bundle, not a
    god-object**: it exposes the raw arrays (:attr:`times` / :attr:`positions` /
    :attr:`spikes`) but carries no heavy analysis methods -- compute stays as
    functions taking the bundle's fields.

    Parameters
    ----------
    env : EnvironmentLike or None
        The spatial environment (an :class:`~neurospatial.Environment` or its
        polar sibling), or ``None`` when no environment is attached yet. When
        not ``None`` it must satisfy
        :func:`~neurospatial._typing.is_environment_like`.
    position : PositionLike
        A position source exposing ``.t`` (timestamps) and ``.values``
        (positions) -- a :class:`Position` holder (from :meth:`from_arrays`) or
        a pynapple ``Tsd`` / ``TsdFrame``.
    spikes : SpikeTrains
        Population spike trains carrying ``unit_ids`` / ``unit_table``. A plain
        list / 2-D array / pynapple ``TsGroup`` passed here is coerced to a
        :class:`~neurospatial.encoding.SpikeTrains`, so :attr:`spikes` is
        **always** a ``SpikeTrains``.
    epochs : (start, end) arrays, IntervalSet-like, or None, optional
        The epochs this session is restricted to, or ``None`` for the whole
        session. Set by :meth:`restrict`.
    metadata : Mapping or None, optional
        Optional free-form provenance (e.g. session name, subject), kept out of
        the compute path.

    Attributes
    ----------
    env, position, spikes, epochs, metadata
        See parameters.

    Notes
    -----
    Frozen: :meth:`with_environment` and :meth:`restrict` return a **new**
    ``Session`` and never mutate the original.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.recording import Session
    >>> positions = np.random.default_rng(0).uniform(0, 100, (200, 2))
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>> times = np.arange(200) / 20.0
    >>> spikes = [np.array([0.1, 0.5]), np.array([0.2, 0.8])]
    >>> sess = Session.from_arrays(
    ...     env=env, times=times, positions=positions, spike_times=spikes
    ... )
    >>> sess.spikes.unit_ids
    array([0, 1])
    """

    env: EnvironmentLike | None
    position: PositionLike
    spikes: SpikeTrains
    epochs: Any | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Coerce ``spikes`` to a ``SpikeTrains`` and validate env / lengths."""
        from neurospatial.encoding import as_spike_trains_with_ids
        from neurospatial.encoding.spike_trains import SpikeTrains

        # ``spikes`` is always a SpikeTrains: coerce a plain list / 2-D array /
        # SpikeTrainsLike group (surfacing unit ids), leave an existing
        # SpikeTrains (with its unit_table) untouched. Bind to a loosely-typed
        # local so the coercion branch is reachable (the field type documents
        # the post-init invariant, not the accepted inputs).
        spikes_in: Any = self.spikes
        if not isinstance(spikes_in, SpikeTrains):
            trains, unit_ids = as_spike_trains_with_ids(spikes_in)
            object.__setattr__(self, "spikes", SpikeTrains(trains, unit_ids=unit_ids))

        # Position must have matching timestamp / sample counts.
        n_t = len(np.asarray(self.position.t))
        n_v = len(np.asarray(self.position.values))
        if n_t != n_v:
            raise ValueError(
                f"position.t and position.values must have the same length, "
                f"but len(position.t)={n_t} and len(position.values)={n_v}.\n"
                "  WHY: each position sample needs exactly one timestamp.\n"
                "  HOW: pass times and positions of equal length (one row per "
                "sample)."
            )

        # A non-None env must be a real spatial environment (or its polar
        # sibling) -- duck-typed, never isinstance on Environment.
        if self.env is not None and not is_environment_like(self.env):
            raise ValueError(
                "env must be an Environment-like object (exposing bin_centers, "
                "connectivity, neighbors) or None, got "
                f"{type(self.env).__name__}.\n"
                "  WHY: Session.env provides the spatial context for encoding / "
                "decoding.\n"
                "  HOW: pass an Environment (e.g. Environment.from_samples(...)) "
                "or None."
            )

    # -- Convenience accessors (raw arrays; no heavy compute) ---------------

    @property
    def times(self) -> NDArray[np.float64]:
        """Timestamps (seconds), i.e. ``self.position.t`` as a raw array.

        Returns
        -------
        ndarray, shape (n,)
            The position timestamps.
        """
        return np.asarray(self.position.t, dtype=np.float64)

    @property
    def positions(self) -> NDArray[np.float64]:
        """Position samples, i.e. ``self.position.values`` as a raw array.

        Returns
        -------
        ndarray, shape (n,) or (n, n_dims)
            The position coordinates.
        """
        return np.asarray(self.position.values, dtype=np.float64)

    # -- Constructors -------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        *,
        env: EnvironmentLike | None = None,
        times: NDArray[np.float64] | PositionLike,
        positions: NDArray[np.float64] | None = None,
        spike_times: Any,
        unit_ids: NDArray[Any] | Sequence[Any] | None = None,
        unit_table: pd.DataFrame | None = None,
        epochs: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Session:
        """Build a :class:`Session` from in-memory arrays.

        Parameters
        ----------
        env : EnvironmentLike or None, optional
            The spatial environment, or ``None``.
        times : ndarray or PositionLike
            Timestamps (seconds). May instead be a single ``PositionLike``
            object (e.g. a pynapple ``Tsd`` / ``TsdFrame``) carrying both times
            and positions, in which case ``positions`` must be omitted.
        positions : ndarray, shape (n,) or (n, n_dims), optional
            Position samples aligned to ``times``. Omit only when ``times`` is a
            ``PositionLike`` object carrying the positions.
        spike_times : list of ndarray, 2-D array, SpikeTrains, or TsGroup-like
            Population spike times in any form accepted by
            :func:`~neurospatial.encoding.as_spike_trains_with_ids`. Unit ids
            carried by a group (``TsGroup`` / ``SpikeTrains``) are threaded into
            the result.
        unit_ids : ndarray or sequence, optional
            Explicit identity label per unit (overrides ids inferred from a
            group). A wrong-length value raises ``ValueError`` (from
            :class:`~neurospatial.encoding.SpikeTrains`).
        unit_table : pandas.DataFrame or None, optional
            Optional per-unit metadata aligned to ``unit_ids``.
        epochs : (start, end) arrays, IntervalSet-like, or None, optional
            Epochs this session is restricted to. Default ``None`` (whole
            session).
        metadata : Mapping or None, optional
            Optional free-form provenance.

        Returns
        -------
        Session
            A new frozen session.
        """
        from neurospatial.encoding import as_spike_trains_with_ids
        from neurospatial.encoding.spike_trains import SpikeTrains

        t, pos = as_times_positions(times, positions)
        position = Position(t=t, values=pos)

        # Normalize an explicit `unit_ids` override to an array once (SpikeTrains
        # takes NDArray | None); when omitted it falls back to ids inferred from
        # the input.
        override_ids: NDArray[Any] | None = (
            None if unit_ids is None else np.asarray(unit_ids)
        )

        if isinstance(spike_times, SpikeTrains):
            # Preserve the container; honor explicit overrides when given.
            spikes = SpikeTrains(
                trains=list(spike_times),
                unit_ids=(
                    override_ids if override_ids is not None else spike_times.unit_ids
                ),
                unit_table=(
                    unit_table if unit_table is not None else spike_times.unit_table
                ),
            )
        else:
            trains, group_ids = as_spike_trains_with_ids(spike_times)
            spikes = SpikeTrains(
                trains=trains,
                unit_ids=override_ids if override_ids is not None else group_ids,
                unit_table=unit_table,
            )

        return cls(
            env=env,
            position=position,
            spikes=spikes,
            epochs=epochs,
            metadata=metadata,
        )

    @classmethod
    def from_nwb(
        cls,
        path_or_file: str | os.PathLike[str] | Any,
        *,
        unit_ids: Sequence[int] | None = None,
        **read_kwargs: Any,
    ) -> Session:
        """Build a :class:`Session` from an NWB file (path or open ``NWBFile``).

        Uses the existing :mod:`neurospatial.io.nwb` readers, which import pynwb
        **lazily**; this module never imports pynwb. Reads spikes
        (:func:`~neurospatial.io.nwb.read_units`), position
        (:func:`~neurospatial.io.nwb.read_position`), and -- if present in the
        file -- the environment
        (:func:`~neurospatial.io.nwb.read_environment`); a file with no stored
        environment yields ``env=None``.

        Parameters
        ----------
        path_or_file : str, os.PathLike, or NWBFile
            An ``.nwb`` file path (opened and closed here) or an already-open
            pynwb ``NWBFile`` (the caller owns its lifecycle).
        unit_ids : sequence of int, optional
            Subset of unit ids to read (forwarded to
            :func:`~neurospatial.io.nwb.read_units`). Default reads all units.
        **read_kwargs
            Extra keyword arguments forwarded to
            :func:`~neurospatial.io.nwb.read_position` (e.g.
            ``processing_module=``, ``position_name=``) to disambiguate the
            position source.

        Returns
        -------
        Session
            A new frozen session equivalent to :meth:`from_arrays` on the same
            underlying arrays.

        Notes
        -----
        The deep, lazily-materialized NWB read path (``lazy=True``) is
        intentionally **not** implemented here; it is deferred to a later phase
        that extends both this method and the readers together.
        """
        from neurospatial.io.nwb import (
            read_environment,
            read_position,
            read_units,
        )
        from neurospatial.io.nwb._core import open_nwbfile

        with open_nwbfile(path_or_file) as nwbfile:
            trains, ids = read_units(nwbfile, unit_ids=unit_ids)
            # read_position returns (positions, timestamps) -- positions first.
            positions, times = read_position(nwbfile, **read_kwargs)
            try:
                env: EnvironmentLike | None = read_environment(nwbfile)
            except KeyError:
                # No environment stored in the file -> None (env is optional).
                env = None

        return cls.from_arrays(
            env=env,
            times=times,
            positions=positions,
            spike_times=trains,
            unit_ids=ids,
        )

    # -- Return-new "modifiers" ---------------------------------------------

    def with_environment(self, env: EnvironmentLike) -> Session:
        """Return a NEW session with ``env`` swapped in; original unchanged.

        Parameters
        ----------
        env : EnvironmentLike
            The environment to attach. Must satisfy
            :func:`~neurospatial._typing.is_environment_like`.

        Returns
        -------
        Session
            A new session with the new environment.
        """
        if not is_environment_like(env):
            raise ValueError(
                "env must be an Environment-like object (exposing bin_centers, "
                f"connectivity, neighbors), got {type(env).__name__}.\n"
                "  WHY: with_environment attaches the spatial context.\n"
                "  HOW: pass an Environment (e.g. Environment.from_samples(...))."
            )
        return dataclasses.replace(self, env=env)

    def restrict(self, epochs: Any) -> Session:
        """Return a NEW session sliced to ``epochs``; original unchanged.

        Slices the position (by its timestamps) and the spikes (each train by
        its own timestamps) to the given epochs. The spike restriction is
        **identity-preserving**: it trims spikes per unit but never drops units,
        so ``unit_ids`` and ``unit_table`` are carried onto the restricted
        :class:`~neurospatial.encoding.SpikeTrains` unchanged.

        Parameters
        ----------
        epochs : (start, end) arrays, IntervalSet-like, or ndarray
            Epochs in any form accepted by
            :func:`~neurospatial.behavior.restrict`.

        Returns
        -------
        Session
            A new session whose ``position`` / ``spikes`` are restricted to
            ``epochs`` and whose ``epochs`` records the restriction. ``env`` and
            ``metadata`` ride along unchanged.
        """
        from neurospatial.behavior import restrict as _restrict
        from neurospatial.behavior import restrict_spike_trains
        from neurospatial.encoding.spike_trains import SpikeTrains

        times_kept, positions_kept = _restrict(
            self.times, self.positions, epochs=epochs
        )
        new_position = Position(t=times_kept, values=positions_kept)

        # restrict_spike_trains returns a plain list (type-stable); rebuild the
        # container preserving identity (ids + table) -- restriction never drops
        # units, so the labels still align one-to-one. ``list(self.spikes)``
        # yields the per-unit train arrays (SpikeTrains.__iter__).
        new_trains = restrict_spike_trains(list(self.spikes), epochs)
        new_spikes = SpikeTrains(
            trains=new_trains,
            unit_ids=self.spikes.unit_ids,
            unit_table=self.spikes.unit_table,
        )

        return dataclasses.replace(
            self, position=new_position, spikes=new_spikes, epochs=epochs
        )


def _looks_like_nwbfile(obj: object) -> bool:
    """Return whether ``obj`` duck-types as an open pynwb ``NWBFile``.

    Checks for the NWBFile container surface (``acquisition`` / ``processing`` /
    ``units``) without importing pynwb. A path is handled separately (str /
    ``os.PathLike``), so this only needs to recognize an already-open file.
    """
    return (
        hasattr(obj, "acquisition")
        and hasattr(obj, "processing")
        and hasattr(obj, "units")
    )


def load_session(source: Any, **kwargs: Any) -> Session:
    """Load a :class:`Session` from an NWB source.

    Dispatch:

    - a ``str`` / ``os.PathLike`` (e.g. an ``.nwb`` path) or an already-open
      pynwb ``NWBFile`` -> :meth:`Session.from_nwb`;
    - anything else -> a clear ``TypeError`` directing you to
      :meth:`Session.from_arrays`.

    Parameters
    ----------
    source : str, os.PathLike, or NWBFile
        The NWB file path or open ``NWBFile`` to load.
    **kwargs
        Forwarded to :meth:`Session.from_nwb` (e.g. ``unit_ids=``,
        ``processing_module=``).

    Returns
    -------
    Session
        The loaded session.

    Raises
    ------
    TypeError
        If ``source`` is neither an NWB path / ``NWBFile`` (use
        :meth:`Session.from_arrays` for in-memory arrays).
    """
    if isinstance(source, (str, os.PathLike)) or _looks_like_nwbfile(source):
        return Session.from_nwb(source, **kwargs)
    raise TypeError(
        f"load_session(source) expects an NWB file path (str / os.PathLike, "
        f"e.g. 'session.nwb') or an open pynwb NWBFile, got "
        f"{type(source).__name__}.\n"
        "  WHY: load_session is the file-loading entry point; in-memory arrays "
        "are not files.\n"
        "  HOW: for arrays use Session.from_arrays(env=..., times=..., "
        "positions=..., spike_times=...)."
    )
