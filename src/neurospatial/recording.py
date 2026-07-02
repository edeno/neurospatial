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
import warnings
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


def _validate_aligned_lengths(n_times: int, n_values: int, *, owner: str) -> None:
    """Raise if a position source's timestamp and sample counts differ.

    Shared by :class:`Position` and :class:`Session`, which both require exactly
    one timestamp per position sample. ``owner`` names the calling surface (e.g.
    ``"Position"`` or ``"position"``) so the error points at that site's ``.t`` /
    ``.values`` attributes.

    Parameters
    ----------
    n_times : int
        Number of timestamps (``len(.t)``).
    n_values : int
        Number of position samples (``len(.values)``).
    owner : str
        The attribute-owner name interpolated into the message.

    Raises
    ------
    ValueError
        If ``n_times != n_values``.
    """
    if n_times != n_values:
        raise ValueError(
            f"{owner}.t and {owner}.values must have the same length, but "
            f"len({owner}.t)={n_times} and len({owner}.values)={n_values}.\n"
            "  WHY: each position sample needs exactly one timestamp.\n"
            "  HOW: pass `t` and `values` of equal length (one timestamp per "
            "sample)."
        )


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
        Timestamps in seconds. Coerced to ``float64``. Must be 1-D.
    values : ndarray, shape (n,) or (n, n_dims)
        Position samples. Coerced to ``float64``. Must have the same length as
        ``t`` (one timestamp per sample).

    Raises
    ------
    ValueError
        If ``t`` is not 1-D, or if ``len(t) != len(values)``.
    """

    t: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Coerce to ``float64`` and enforce a 1-D ``t`` matching ``values``."""
        t = np.asarray(self.t, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "values", values)

        if t.ndim != 1:
            raise ValueError(
                f"Position.t must be 1-D, but got a {t.ndim}-D array with shape "
                f"{t.shape}.\n"
                "  WHY: `t` is the 1-D timestamp axis, one entry per position "
                "sample.\n"
                "  HOW: pass a 1-D array of timestamps (seconds)."
            )
        _validate_aligned_lengths(len(t), len(values), owner="Position")


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

    Treat the bundle as **read-only**. ``metadata`` is not defensively copied
    (the mapping you pass in is stored as-is), and :attr:`times` / :attr:`positions`
    may return **views** into the underlying position arrays -- do not mutate the
    returned arrays (or the passed-in ``metadata``) in place, or you will alter
    the session's data behind its back.

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

        # ``position`` must expose the PositionLike surface (.t / .values) --
        # duck-checked (never isinstance) so a Position holder OR a pynapple
        # Tsd / TsdFrame both pass, but a raw array raises a clean ValueError
        # (not a bare AttributeError below). Unlike ``_typing._is_position_like``,
        # the ``.d`` alias is deliberately NOT accepted here: ``Session.positions``
        # reads ``.values`` directly, so a ``.d``-only source would break downstream.
        position = self.position
        if not (hasattr(position, "t") and hasattr(position, "values")):
            raise ValueError(
                "position must be a PositionLike object exposing `.t` "
                "(timestamps) and `.values` (positions), got "
                f"{type(position).__name__}.\n"
                "  WHY: Session reads the animal's trajectory from position.t "
                "and position.values.\n"
                "  HOW: build with Session.from_arrays(times=..., positions=...) "
                "(which wraps arrays into a Position holder), or pass a pynapple "
                "Tsd / TsdFrame."
            )

        # Position must have matching timestamp / sample counts.
        n_t = len(np.asarray(self.position.t))
        n_v = len(np.asarray(self.position.values))
        _validate_aligned_lengths(n_t, n_v, owner="position")

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

        # One construction for every accepted spike input: the adapter already
        # handles a SpikeTrains (surfacing its ids) as well as the array / group
        # forms. Ids precedence: explicit `unit_ids=` override, else the group's
        # ids. Table precedence: explicit `unit_table=` override, else a passed
        # SpikeTrains's own table (arrays carry none).
        trains, group_ids = as_spike_trains_with_ids(spike_times)
        table = unit_table
        if table is None and isinstance(spike_times, SpikeTrains):
            table = spike_times.unit_table
        spikes = SpikeTrains(
            trains=trains,
            unit_ids=override_ids if override_ids is not None else group_ids,
            unit_table=table,
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
        environment_name: str | None = None,
        unit_ids: Sequence[int] | None = None,
        **read_kwargs: Any,
    ) -> Session:
        """Build a :class:`Session` from an NWB file (path or open ``NWBFile``).

        Uses the existing :mod:`neurospatial.io.nwb` readers, which import pynwb
        **lazily**; this module never imports pynwb. Reads spikes
        (:func:`~neurospatial.io.nwb.read_units`), position
        (:func:`~neurospatial.io.nwb.read_position`), and -- if a scratch entry
        named ``environment_name`` is present -- the environment
        (:func:`~neurospatial.io.nwb.read_environment`).

        Environment presence is decided by **membership** (is the scratch entry
        there?), not by catching an error, so the two cases stay distinct: a
        file with **no** stored environment yields ``env=None``, while an
        environment that IS present but cannot be read (malformed / wrong schema)
        **raises** -- it is never silently turned into ``None``.

        Parameters
        ----------
        path_or_file : str, os.PathLike, or NWBFile
            An ``.nwb`` file path (opened and closed here) or an already-open
            pynwb ``NWBFile`` (the caller owns its lifecycle).
        environment_name : str or None, optional
            Name of the environment scratch entry to read. Default (``None``)
            reads the standard ``spatial_environment`` scratch entry; pass a
            custom name to select an environment written under a non-default
            name. If the named entry is absent, ``env`` is ``None``; if it is
            present but unreadable, the read error propagates (it is not swapped
            for ``None``).
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

        # Function-local so no pynwb / reader constant is imported at module
        # load (preserving the array-first import guarantee).
        from neurospatial.io.nwb._environment import DEFAULT_ENVIRONMENT_NAME

        name = (
            DEFAULT_ENVIRONMENT_NAME if environment_name is None else environment_name
        )

        with open_nwbfile(path_or_file) as nwbfile:
            trains, ids = read_units(nwbfile, unit_ids=unit_ids)
            # read_position returns (positions, timestamps) -- positions first.
            positions, times = read_position(nwbfile, **read_kwargs)
            # Gate on presence, not on catching KeyError: a genuinely-absent env
            # maps to None, while ANY error reading a present-but-malformed env
            # propagates to the caller (instead of being silenced to None).
            env: EnvironmentLike | None
            if name in nwbfile.scratch:
                env = read_environment(nwbfile, name=name)
            else:
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

        Warns
        -----
        UserWarning
            If ``epochs`` keep **zero** position samples (the epochs do not
            overlap the session at all -- often a seconds-vs-milliseconds unit
            mismatch). The empty session is still returned. Per-unit *empty spike
            trains* while some position samples survive are legitimate and do
            **not** warn.

        Notes
        -----
        ``.epochs`` records only the **most recent** restriction. Chaining
        (``s.restrict(a).restrict(b)``) restricts the data to the intersection
        of ``a`` and ``b``, but the returned session's ``.epochs == b`` (it does
        not accumulate the chain).
        """
        from neurospatial.behavior import restrict as _restrict
        from neurospatial.behavior import restrict_spike_trains
        from neurospatial.encoding.spike_trains import SpikeTrains

        times_kept, positions_kept = _restrict(
            self.times, self.positions, epochs=epochs
        )
        # Zero surviving position samples means the epochs miss the session
        # entirely -- warn (naming the likely cause) but still return the empty
        # session, since per-unit empty trains alone remain a legitimate outcome.
        if len(times_kept) == 0:
            warnings.warn(
                "restrict(epochs) kept zero position samples: the epochs do not "
                "overlap this session's time range.\n"
                "  WHY: a common cause is a seconds-vs-milliseconds unit "
                "mismatch between `epochs` and the session timestamps.\n"
                "  HOW: check that `epochs` are in the same time units (seconds) "
                "as session.times.",
                stacklevel=2,
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
