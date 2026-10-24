"""Shared result-object surface and unit-identity helpers.

:class:`ResultMixin` gives every user-facing result a ``summary()`` of scalar
headline metrics and a concise ``__repr__``/``_repr_html_`` built from it,
instead of the default dataclass repr that dumps every array. Results add the
other terminal verbs they support (``to_dataframe()``, ``summary_table()``,
``to_xarray()``, ``plot()``); a verb a result does not define is an
``AttributeError``, not a stub that raises.

The module also holds the pieces results share: read-only array ownership
(:class:`ReadOnlyArraysMixin`, :func:`read_only_copy`), owned
``analysis_settings``, unit-label resolution and selection
(:func:`resolve_unit_ids`, :class:`UnitSelectionMixin`), unit-table alignment
and the xarray export helpers.
"""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error


def read_only_copy(array: Any, dtype: Any = None) -> NDArray[Any]:
    """Return a read-only copy that no caller-held view can alias."""
    owned = np.array(array, dtype=dtype, copy=True)
    owned.flags.writeable = False
    return owned


def as_label_array(labels: Any) -> NDArray[Any]:
    """Convert unit labels to an array without changing any label's type.

    ``np.asarray([1, "u"])`` would turn the integer ``1`` into the string
    ``"1"`` (and ``[1, 2.5]`` would turn ``1`` into ``1.0``), changing the
    label a unit is reported, looked up and seeded under. Labels of more than
    one kind are kept in an object array; homogeneous labels keep their
    natural dtype.
    """
    if isinstance(labels, np.ndarray):
        return labels
    values = list(labels)
    array = np.asarray(values)
    if array.ndim == 1 and len({np.asarray(v).dtype.kind for v in values}) > 1:
        array = np.empty(len(values), dtype=object)
        array[:] = values
    return array


def require_label_sequence(labels: Any, *, name: str, call: str, example: str) -> None:
    """Refuse a lone string, bytes or scalar where one label per item is needed.

    ``list("speed")`` would split a string into characters, and a number is
    not iterable at all.

    Parameters
    ----------
    labels : Any
        The caller's labels argument (not ``None``).
    name : str
        Argument name, ``"feature_ids"`` or ``"unit_ids"``.
    call : str
        Calling function's name, for the message.
    example : str
        One plausible label, shown in the fix (e.g. ``"speed"``).
    """
    if isinstance(labels, str | bytes) or np.ndim(labels) == 0:
        item = name.removesuffix("_ids")
        raise ValueError(
            _format_error(
                f"{call} got {name}={labels!r}; it must be a sequence of "
                f"labels, one per {item}.",
                why="Why: a single string or number would be split into "
                f"characters or cannot label each {item}.",
                fix=f"Pass a list or array, e.g. {name}=[{example!r}] for one {item}.",
            )
        )


def _missing_label_positions(labels: Sequence[Any]) -> list[int]:
    """Positions of ``None`` or NaN labels."""
    return [
        i
        for i, label in enumerate(labels)
        if label is None or (isinstance(label, float | np.floating) and np.isnan(label))
    ]


def _same_labels(given: NDArray[Any], carried: NDArray[Any]) -> bool:
    """Whether two label arrays hold equal labels of the same kind, in order.

    ``1`` and ``1.0`` compare equal but are different labels: shuffle streams
    and lookups are keyed by the label itself.
    """
    if given.shape != carried.shape:
        return False
    return all(
        a == b and np.asarray(a).dtype.kind == np.asarray(b).dtype.kind
        for a, b in zip(given.tolist(), carried.tolist(), strict=True)
    )


def label_at(unit_ids: Any, index: int) -> Any:
    """Return one label as a plain Python value, whatever the array's dtype."""
    value = np.asarray(unit_ids)[index]
    return value.item() if isinstance(value, np.generic) else value


_ResultT = TypeVar("_ResultT")


def _copy_analysis_settings(cls: type[_ResultT]) -> type[_ResultT]:
    """Give a result dataclass independently owned settings on every read.

    Apply after ``@dataclass`` so its constructor and field metadata stay intact.

    Parameters
    ----------
    cls : type
        Result dataclass with an ``analysis_settings`` dictionary field.
    """

    def read_settings(result: object) -> dict[str, Any] | None:
        settings: dict[str, Any] | None = vars(result).get("_analysis_settings")
        return deepcopy(settings)

    def write_settings(result: object, settings: dict[str, Any] | None) -> None:
        vars(result)["_analysis_settings"] = deepcopy(settings)

    setattr(cls, "analysis_settings", property(read_settings, write_settings))  # noqa: B010
    return cls


class ReadOnlyArraysMixin:
    """A frozen result whose NumPy arrays are read-only copies it owns.

    A subclass calls :meth:`_own_arrays` at the end of ``__post_init__``.
    Editing an array the result was built from then cannot change it, and
    editing an array read from it raises. Copy with ``np.array(...)`` to
    modify values. JAX arrays are immutable already and are kept as given.
    """

    def _own_arrays(self) -> None:
        """Replace every NumPy array attribute with a read-only copy."""
        for name, value in list(vars(self).items()):
            if isinstance(value, np.ndarray):
                object.__setattr__(self, name, read_only_copy(value))

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle or deepcopy, keeping the arrays read-only."""
        self.__dict__.update(state)
        for value in state.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False


_Unit_co = TypeVar("_Unit_co", covariant=True)


class _UnitContainer(Protocol[_Unit_co]):
    """A labelled container whose ``[i]`` returns its ``i``-th unit."""

    @property
    def unit_ids(self) -> Any: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> _Unit_co: ...


class UnitSelectionMixin(ReadOnlyArraysMixin):
    """Position indexing with ``[i]`` and label selection with :meth:`sel`.

    Shared by every container of units (``SpikeTrains`` and the population
    results) so ``[]`` has one meaning everywhere: ``x[i]`` is the ``i``-th
    unit in stored order, the same row as every per-unit array and as
    iteration. Labels are selected explicitly with ``x.sel(unit_id=label)``,
    as in ``x.to_xarray().sel(unit_id=label)``. ``unit_ids`` stays read-only,
    including after pickling or copying, so it cannot be edited into the
    duplicate labels that would make ``sel`` return the first match.
    """

    def sel(self: _UnitContainer[_Unit_co], *, unit_id: Any) -> _Unit_co:
        """Return the unit whose label is ``unit_id``.

        Parameters
        ----------
        unit_id : Any
            An entry of ``unit_ids``.

        Returns
        -------
        object
            What ``self[i]`` returns at that unit's position: the spike train
            for :class:`~neurospatial.encoding.SpikeTrains`, the single-unit
            result for a population result.

        Raises
        ------
        KeyError
            If no unit has this label.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding import SpikeTrains
        >>> st = SpikeTrains([np.array([0.1]), np.array([0.2, 0.3])], unit_ids=[7, 9])
        >>> st.sel(unit_id=9)
        array([0.2, 0.3])
        >>> st[1]  # the same unit, by position
        array([0.2, 0.3])
        """
        labels = np.asarray(self.unit_ids).tolist()
        try:
            position = labels.index(unit_id)
        except ValueError:
            shown = ", ".join(repr(label) for label in labels[:10])
            more = f", ... ({len(labels)} units)" if len(labels) > 10 else ""
            missing = unit_id.item() if isinstance(unit_id, np.generic) else unit_id
            raise KeyError(
                f"unit_id {missing!r} is not in this {type(self).__name__} "
                f"(unit_ids: {shown}{more}). Pass one of these labels to "
                "sel(), or use [i] to select the i-th unit by position."
            ) from None
        return self[position]

    def _unit_position(self: _UnitContainer[Any], index: Any) -> int:
        """Validate a positional ``[]`` index, pointing label lookups at ``sel``."""
        name = type(self).__name__
        try:
            position = operator.index(index)
        except TypeError:
            label = index.item() if isinstance(index, np.generic) else index
            raise TypeError(
                _format_error(
                    f"{name}[{label!r}]: [] selects one unit by integer position.",
                    fix=f"use .sel(unit_id={label!r}) to select a unit by label",
                )
            ) from None
        n_units = len(self)
        if not -n_units <= position < n_units:
            raise IndexError(
                _format_error(
                    f"{name}[{position}] is out of range for {n_units} unit(s).",
                    why="Why: [] selects by position, not by unit_id.",
                    fix=f"use .sel(unit_id={position!r}) to select a unit by "
                    f"label, or a position from {-n_units} to {n_units - 1}",
                )
            )
        return position


def resolve_unit_ids(
    unit_ids: NDArray[Any] | Sequence[Any] | None,
    n_units: int,
    *,
    context: str = "",
    input_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> NDArray[Any]:
    """Resolve and validate per-unit identity labels.

    Returns a read-only ``ndarray`` of unit identity labels: ``unit_ids`` when
    given, else the labels the spike input carries (``input_ids``), else
    ``np.arange(n_units)``. Validates that there is exactly one label per unit
    and that no label repeats or is missing. The array is a copy, so editing
    the caller's labels afterwards cannot relabel or duplicate a unit.

    Parameters
    ----------
    unit_ids : ndarray or sequence or None
        Caller-supplied per-unit identity labels. May be integers or strings.
    n_units : int
        Number of units the labels must describe.
    context : str, optional
        Caller name included in error messages.
    input_ids : ndarray or sequence or None, optional
        Labels carried by the spike input itself (for example a pynapple
        ``TsGroup`` index). A labelled input names its own units, so when both
        ``unit_ids`` and ``input_ids`` are given they must be identical.

    Returns
    -------
    ndarray
        Resolved labels as a read-only 1D array of length ``n_units``.

    Raises
    ------
    ValueError
        If ``unit_ids`` is a single string or number, if ``unit_ids`` and
        ``input_ids`` are both given and differ in length, order, value or
        kind (``1`` and ``1.0`` are different labels), if the resolved labels
        are not 1-D, if their length does not equal ``n_units``, or if any
        label repeats or is ``None``/NaN.
    """
    call = context or "This call"
    if unit_ids is not None:
        require_label_sequence(unit_ids, name="unit_ids", call=call, example="u1")
    if unit_ids is not None and input_ids is not None:
        given, carried = as_label_array(unit_ids), as_label_array(input_ids)
        if not _same_labels(given, carried):
            raise ValueError(
                _format_error(
                    f"{call} got unit_ids={given.tolist()}, but the spike input "
                    f"is already labelled {carried.tolist()}.",
                    why="Why: a labelled input names its own units; a different "
                    "unit_ids would attach another unit's label to each spike "
                    "train.",
                    fix="drop unit_ids= to keep the input's labels, or relabel "
                    "the input itself before the call (pynapple: build the "
                    "TsGroup with the labels you want).",
                )
            )
    from_input = unit_ids is None and input_ids is not None
    unit_ids = unit_ids if unit_ids is not None else input_ids
    if unit_ids is None:
        return read_only_copy(np.arange(n_units))

    resolved = read_only_copy(as_label_array(unit_ids))
    where = f" in {context}" if context else ""
    if resolved.ndim != 1:
        raise ValueError(
            _format_error(
                f"unit_ids must be 1-D{where}: got shape {resolved.shape}.",
                why="Why: unit_ids labels one identity per unit (row).",
                fix="pass a 1-D sequence with one entry per unit, or omit it "
                "to default to np.arange(n_units).",
            )
        )
    if resolved.shape[0] != n_units:
        raise ValueError(
            _format_error(
                f"unit_ids length mismatch{where}: got {resolved.shape[0]} "
                f"label(s) but there are {n_units} unit(s).",
                why="Why: unit_ids must identify every row without shifting "
                "unit labels.",
                fix=f"pass one label per unit (len(unit_ids) == {n_units}), or "
                "omit unit_ids to default to np.arange(n_units)",
            )
        )
    labels = resolved.tolist()
    missing = _missing_label_positions(labels)
    if missing:
        raise ValueError(
            _format_error(
                f"unit_ids{where} has a missing label (None or NaN) at "
                f"position(s) {missing[:5]}.",
                why="Why: results select units by label, and a missing label "
                "never matches, so that unit could not be selected.",
                fix="give every unit a label, or omit unit_ids= to number the "
                "units 0..n-1.",
            )
        )
    # Count with a hash-based Counter: np.unique cannot sort a mixed int/str
    # object array.
    duplicated = [label for label, count in Counter(labels).items() if count > 1]
    if duplicated:
        fix = (
            "pass each unit once in the spike input (pynapple: check group.index)."
            if from_input
            else "pass one distinct label per unit, or omit unit_ids= to "
            "number the units 0..n-1."
        )
        raise ValueError(
            _format_error(
                f"unit_ids must be unique{where}: label(s) {duplicated} are repeated.",
                why="Why: results index units by label, so a repeated label "
                "would name two spike trains.",
                fix=fix,
            )
        )
    return resolved


def align_unit_table(unit_table: Any, unit_ids: Any, *, context: str = "") -> Any:
    """Align labelled metadata by identity; accept a plain RangeIndex by position.

    A ``unit_id`` column or a non-default/named index declares unit identity.
    Missing, extra or duplicate labels raise instead of selecting another unit.

    Parameters
    ----------
    unit_table : pandas.DataFrame or None
        Optional per-unit metadata, one row per unit.
    unit_ids : array-like, shape (n_units,)
        The units' resolved labels.
    context : str, optional
        Caller / class name included in error messages.

    Returns
    -------
    pandas.DataFrame or None
        A copy with rows in unit order, or ``None``.

    Raises
    ------
    ValueError
        If ``unit_table`` is not a DataFrame, has a row count other than
        ``len(unit_ids)``, or labels rows inconsistently, with repeated or
        missing labels, or with labels other than ``unit_ids``.
    """
    import pandas as pd

    if unit_table is None:
        return None
    ids = np.asarray(unit_ids)
    where = f" in {context}" if context else ""
    if not isinstance(unit_table, pd.DataFrame):
        raise ValueError(
            _format_error(
                f"unit_table must be a pandas DataFrame{where}, got "
                f"{type(unit_table).__name__}.",
                fix="pass pd.DataFrame(...) with one row per unit, or omit it.",
            )
        )
    if len(unit_table) != len(ids):
        raise ValueError(
            _format_error(
                f"unit_table length mismatch{where}: got {len(unit_table)} "
                f"row(s) but there are {len(ids)} unit(s).",
                why="Why: unit_table must carry exactly one row per unit, "
                "aligned to unit_ids.",
                fix="pass a unit_table with one row per unit, or omit it.",
            )
        )
    positional = (
        isinstance(unit_table.index, pd.RangeIndex)
        and unit_table.index.equals(pd.RangeIndex(len(ids)))
        and unit_table.index.name is None
    )
    if "unit_id" in unit_table.columns:
        labels = pd.Index(unit_table["unit_id"], name="unit_id")
        if not positional and not labels.equals(unit_table.index):
            raise ValueError(
                _format_error(
                    f"unit_table index and unit_id column disagree{where}.",
                    fix="use one consistent unit identity for each row.",
                )
            )
    elif positional:
        return unit_table.copy()
    else:
        labels = unit_table.index
    if not labels.is_unique or labels.hasnans:
        raise ValueError(
            _format_error(
                f"unit_table labels must be unique and non-missing{where}.",
                fix="provide one metadata row per unit_id.",
            )
        )
    order = labels.get_indexer(ids)
    if np.any(order < 0):
        raise ValueError(
            _format_error(
                f"unit_table labels do not match unit_ids={ids.tolist()}{where}.",
                fix="provide metadata for exactly these units; a plain unnamed "
                "RangeIndex is accepted only for rows already in spike order.",
            )
        )
    aligned = unit_table.iloc[order].copy()
    if positional:
        aligned = aligned.reset_index(drop=True)
    return aligned


def software_version() -> str:
    """Return the installed ``neurospatial`` version string.

    Used to stamp xarray exports with the producing library version. Falls
    back to ``"unknown"`` when the package metadata cannot be resolved (e.g.
    an editable checkout without an installed distribution).

    Returns
    -------
    str
        The installed version (e.g. ``"0.6.0"``), or ``"unknown"``.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("neurospatial")
    except PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"


def units_attr(env: Any) -> dict[str, str]:
    """Return a ``{"units": ...}`` attrs fragment, omitted when units are unset.

    Reads ``env.units`` and returns ``{"units": <value>}`` only when it is a
    meaningful non-empty string. When ``units`` is ``None`` (or empty, or the
    env lacks the attribute) an **empty** mapping is returned, so that the
    literal string ``"None"`` is never stored as a units attribute on an
    xarray export.

    Parameters
    ----------
    env : Any
        An environment exposing an optional ``units`` attribute.

    Returns
    -------
    dict
        ``{"units": value}`` when units are set, else ``{}``.
    """
    units = getattr(env, "units", None)
    if units is None:
        return {}
    units_str = str(units)
    if units_str == "":
        return {}
    return {"units": units_str}


def env_fingerprint(env: Any) -> str:
    """Return a stable, human-readable identifier for an environment.

    There is no dedicated environment hash in the codebase, so this returns a
    compact ``repr``-style identifier (name, dimensionality, bin count, layout)
    suitable for stamping into xarray ``attrs`` for provenance. It is *not* a
    content hash and must not be relied on for equality checks.

    Parameters
    ----------
    env : Any
        A Cartesian or polar ``Environment`` instance.

    Returns
    -------
    str
        A stable single-line identifier such as
        ``"Environment(name='', 2D, 25 bins, RegularGrid)"``.
    """
    return repr(env)


def _coord_dim_names(n_dims: int) -> list[str]:
    """Per-dimension coordinate column names for tidy decode dataframes.

    Returns ``["x", "y", "z"][:n_dims]`` for ``n_dims <= 3`` and
    ``["dim_0", "dim_1", ...]`` otherwise. Shared by
    :meth:`DecodingResult.to_dataframe` and
    :meth:`DecodingSummary.to_dataframe` so their coordinate column naming
    (``map_x``/``mean_x``/... vs ``map_dim_0``/...) stays identical.

    Parameters
    ----------
    n_dims : int
        Number of spatial dimensions.

    Returns
    -------
    list[str]
        Coordinate names, one per dimension.
    """
    if n_dims <= 3:
        return ["x", "y", "z"][:n_dims]
    return [f"dim_{i}" for i in range(n_dims)]


def _bin_center_coords(
    env: Any, n_bins: int
) -> dict[str, tuple[str, NDArray[np.float64]]]:
    """Build non-index ``bin``-dimension coordinates from an environment.

    Returns a mapping of coordinate name to ``("bin", values)`` tuples giving
    the per-bin center coordinates. Cartesian environments yield
    ``bin_center_x`` (1-D), plus ``bin_center_y`` (2-D) and ``bin_center_z``
    (3-D). Polar egocentric environments (whose ``bin_centers[:, 0]`` is
    distance and ``bin_centers[:, 1]`` is angle in radians) yield
    ``bin_center_distance`` and ``bin_center_angle`` instead.

    Parameters
    ----------
    env : Any
        Environment exposing ``bin_centers`` and ``n_dims``. May be ``None``,
        in which case an empty mapping is returned.
    n_bins : int
        Expected number of bins; the environment's ``bin_centers`` must agree.

    Returns
    -------
    dict
        Mapping ``name -> ("bin", values)`` for use as xarray ``coords``.

    Raises
    ------
    ValueError
        If ``env`` is given but its ``bin_centers`` is not 2-D or its length
        (``bin_centers.shape[0]``) does not equal ``n_bins``. The genuine
        no-env case (``env is None``) is a no-op and returns an empty mapping.
    """
    # Genuine no-env case (e.g. directional results pass bin_centers directly
    # and have no env): no bin-center coords to build. This is NOT a mismatch.
    if env is None:
        return {}

    bin_centers = np.asarray(env.bin_centers, dtype=np.float64)
    # bin-center data IS present (env was given); a shape disagreement here is
    # a real inconsistency between the result and its environment, not an
    # optional-coord situation. Raise loudly rather than silently dropping it.
    if bin_centers.ndim != 2:
        raise ValueError(
            "Environment bin_centers must be 2-D (n_bins, n_dims) to build "
            f"xarray bin-center coords, but env produced an array with "
            f"ndim={bin_centers.ndim} (shape {bin_centers.shape}).\n"
            "  WHY: each bin needs an (x, y, ...) center for the 'bin' "
            "coordinate.\n"
            "  Fix: pass an Environment, whose bin_centers have shape "
            "(n_bins, n_dims)."
        )
    if bin_centers.shape[0] != n_bins:
        raise ValueError(
            "bin_centers length does not match the result's bin count: env "
            f"bin_centers has {bin_centers.shape[0]} row(s) but the result "
            f"has {n_bins} bin(s).\n"
            "  WHY: the 'bin' coordinate must have one center per bin; a "
            "mismatch would silently produce a structurally-incomplete or "
            "misaligned Dataset.\n"
            "  Fix: pass the same environment used to compute the result, or "
            "recompute the result against this environment."
        )

    if getattr(env, "coordinate_kind", "cartesian") == "polar":
        return {
            "bin_center_distance": ("bin", bin_centers[:, 0]),
            "bin_center_angle": ("bin", bin_centers[:, 1]),
        }

    names = ["bin_center_x", "bin_center_y", "bin_center_z"]
    n_dims = bin_centers.shape[1]
    coords: dict[str, tuple[str, NDArray[np.float64]]] = {}
    for i in range(min(n_dims, 3)):
        coords[names[i]] = ("bin", bin_centers[:, i])
    # For >3-D envs, expose remaining axes as bin_center_dim_3, ...
    for i in range(3, n_dims):
        coords[f"bin_center_dim_{i}"] = ("bin", bin_centers[:, i])
    return coords


def build_population_dataset(
    firing_rates: NDArray[np.float64],
    unit_ids: NDArray[Any],
    *,
    env: Any = None,
    bin_centers: NDArray[np.float64] | None = None,
    occupancy: NDArray[np.float64] | None = None,
    attrs: dict[str, Any] | None = None,
) -> Any:
    """Build a labeled population firing-rate :class:`xarray.Dataset`.

    Shared constructor for the population ``to_xarray`` methods. Produces a
    :class:`xarray.Dataset` with dims ``("unit_id", "bin")`` where ``unit_id``
    is the index coordinate (the real per-unit identity labels) and ``bin`` is
    the integer bin index carrying non-index ``bin_center_*`` coordinates.

    Parameters
    ----------
    firing_rates : ndarray, shape (n_units, n_bins)
        Population firing-rate matrix (Hz).
    unit_ids : ndarray, shape (n_units,)
        Per-unit identity labels. **Must be unique** (label-based
        ``.sel(unit_id=...)`` requires uniqueness).
    env : Any, optional
        Environment supplying ``bin_center_*`` coords. Mutually used with
        ``bin_centers``; pass at most one. ``None`` omits Cartesian coords.
    bin_centers : ndarray, optional
        Explicit angular bin centers (radians) for results without an ``env``
        (directional). When given, a ``bin_center_angle`` coord is attached.
    occupancy : ndarray, shape (n_bins,), optional
        Per-bin occupancy (seconds); attached as an ``occupancy`` data var on
        ``("bin",)`` when provided.
    attrs : dict, optional
        Dataset-level attributes (units, bandwidth, env fingerprint, version).

    Returns
    -------
    xarray.Dataset
        Dataset with data var ``firing_rate`` (dims ``("unit_id", "bin")``),
        optional ``occupancy`` (dims ``("bin",)``), and ``bin_center_*`` coords.

    Raises
    ------
    ValueError
        If ``unit_ids`` contains duplicate labels, if ``occupancy`` is
        provided but its length does not equal ``n_bins``, or if
        ``bin_centers`` is provided but is not 1-D of length ``n_bins``.
        (Absent ``occupancy``/``bin_centers`` are simply omitted; only a
        present-but-mismatched array raises.)
    ImportError
        If ``xarray`` is not installed.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "to_xarray() requires the optional 'xarray' dependency, which "
            "is not installed. Install it with "
            "'pip install neurospatial[xarray]' or 'pip install xarray'."
        ) from exc

    # Copies: the Dataset is the caller's to edit, and results are read-only.
    rates = np.array(firing_rates, dtype=np.float64)
    n_bins = rates.shape[1]
    ids = np.asarray(unit_ids)

    # Duplicate unit_ids break label-based .sel(unit_id=...); reject loudly.
    # Count with a hash-based Counter: np.unique cannot sort a mixed int/str
    # object array.
    dups = [label for label, count in Counter(ids.tolist()).items() if count > 1]
    if dups:
        raise ValueError(
            _format_error(
                "unit_ids must be unique to build an xarray.Dataset, but these "
                f"label(s) are duplicated: {dups}.",
                why="Why: label-based selection .sel(unit_id=...) requires a "
                "unique index coordinate.",
                fix="deduplicate unit_ids (e.g. when concatenating populations) "
                "before calling to_xarray().",
            )
        )

    coords: dict[str, Any] = {"unit_id": ids, "bin": np.arange(n_bins)}
    coords.update(_bin_center_coords(env, n_bins))
    if bin_centers is not None:
        # bin_centers IS present (passed explicitly, e.g. directional angular
        # centers); a shape disagreement is a real inconsistency, not an
        # optional coord. Raise rather than silently omitting the coord.
        bc = np.array(bin_centers, dtype=np.float64)
        if bc.ndim != 1 or bc.shape[0] != n_bins:
            raise ValueError(
                "bin_centers length does not match the firing-rate bin "
                f"count: bin_centers has shape {bc.shape} but there are "
                f"{n_bins} bin(s) (firing_rates.shape[1]).\n"
                "  WHY: the 'bin' coordinate must have one center per bin; a "
                "mismatch would silently produce a misaligned Dataset.\n"
                "  Fix: pass a 1-D bin_centers array of length n_bins."
            )
        coords["bin_center_angle"] = ("bin", bc)

    data_vars: dict[str, Any] = {
        "firing_rate": (("unit_id", "bin"), rates),
    }
    if occupancy is not None:
        # occupancy IS present; a length disagreement is a real inconsistency,
        # not an absent-optional. Raise rather than silently dropping the var.
        occ = np.array(occupancy, dtype=np.float64)
        if occ.ndim != 1 or occ.shape[0] != n_bins:
            raise ValueError(
                "occupancy length does not match the firing-rate bin count: "
                f"occupancy has shape {occ.shape} but there are {n_bins} "
                "bin(s) (firing_rates.shape[1]).\n"
                "  WHY: the 'occupancy' data var is indexed by bin; a length "
                "mismatch would silently produce a structurally-incomplete "
                "Dataset.\n"
                "  Fix: pass a 1-D occupancy array of length n_bins, or omit "
                "it entirely if unavailable."
            )
        data_vars["occupancy"] = (("bin",), occ)

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs or {},
    )


def _format_summary_value(value: Any) -> str:
    """Format a scalar ``summary()`` value compactly for repr/HTML.

    Python and NumPy floats are rounded to 4 significant figures so a
    headline metric reads as ``peak_firing_rate=18.48`` rather than
    ``18.480991614424113``; everything else is passed through ``str``.
    ``.4g`` renders ``nan``/``inf`` correctly.
    """
    if isinstance(value, float | np.floating):
        return f"{value:.4g}"
    return str(value)


class ResultMixin:
    """Uniform result-object surface: ``summary`` and a concise repr.

    Mix this into any analysis result class. ``summary`` must be specialized;
    the repr shows the class name and its headline scalars. Results define the
    other terminal verbs they support, following the public API contract:
    ``to_dataframe()`` returns a **tidy/long-form** frame, and
    ``plot(ax=None, **kwargs)`` accepts an optional ``ax`` and **returns it**.
    ``pandas`` and ``matplotlib`` must be imported lazily inside the methods
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
    >>> Empty()
    Empty(...)
    """

    def _summary_for_repr(self) -> dict[str, Any] | str:
        """``summary()``, or a short note when it cannot be shown."""
        try:
            return self.summary()
        except NotImplementedError:
            return {}
        except Exception as exc:  # a repr must never raise
            return f"<summary() failed: {type(exc).__name__}: {exc}>"

    def __repr__(self) -> str:
        """Concise one-line repr built from :meth:`summary` scalar metrics.

        Result classes carry large arrays (firing rates, occupancy, posteriors),
        which the default dataclass repr dumps in full -- a wall of numbers when
        a user simply types the result at a REPL. This shows the class name and
        the headline scalars instead (e.g.
        ``SpatialRateResult(n_bins=121, peak_firing_rate=18.48, ...)``). A
        ``__repr__`` must never raise: a result without ``summary()`` shows
        ``ClassName(...)``, and a failing ``summary()`` names its error.
        """
        name = type(self).__name__
        summary = self._summary_for_repr()
        if isinstance(summary, str):
            return f"{name}({summary})"
        if not summary:
            return f"{name}(...)"
        parts = ", ".join(f"{k}={_format_summary_value(v)}" for k, v in summary.items())
        return f"{name}({parts})"

    def _repr_html_(self) -> str:
        """Rich HTML table of :meth:`summary` metrics for Jupyter.

        Jupyter renders this instead of ``__repr__`` when a result is a cell's
        last expression, giving a scannable metric table rather than a wall of
        array values. Falls back to the plain repr when ``summary()`` is
        unavailable or fails.
        """
        import html

        summary = self._summary_for_repr()
        if isinstance(summary, str) or not summary:
            return f"<pre>{html.escape(repr(self))}</pre>"
        rows = "".join(
            f"<tr><th style='text-align:right'>{html.escape(str(k))}</th>"
            f"<td style='text-align:left'>{html.escape(_format_summary_value(v))}</td></tr>"
            for k, v in summary.items()
        )
        return (
            f"<table><caption style='text-align:left'><b>"
            f"{html.escape(type(self).__name__)}</b></caption>{rows}</table>"
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
