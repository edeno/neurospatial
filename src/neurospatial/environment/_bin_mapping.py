"""Exact correspondence between an environment's bins and those of another.

``Environment.bin_mapping_from(parent)`` pairs bins from geometry alone, so it
works on environments read from files as well as on fresh derivations. It
reports only pairs it can prove: a child bin that is the same cell as a parent
bin (a subset, the cells a clip did not cut, an identical rebuild), and parent
cells that lie wholly inside one cell of a coarser grid drawn on the same
lattice (``rebin``). A parent cell divided between child cells is split, and
its contents cannot be divided without knowing where in the cell each
observation fell; it is reported, never paired. Nearest-center assignment
(``ops.map_probabilities``, ``ops.resample_field``) is an approximation and is
not used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import shapely
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import IncompatibleEnvironmentError, _format_error
from neurospatial._results import ReadOnlyArraysMixin
from neurospatial.environment._comparison import (
    _cell_array,
    _compares_polygons,
    _label_difference,
    _paired_cell_differences,
)

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment
    from neurospatial.geometry import Binning, RectilinearGrid

_OPERATION = "Environment.bin_mapping_from()"
# Grid edges coincide within a billionth of a cell width: the tolerance
# rebin(bin_size=...) uses to decide which new cell a bin overlaps.
_EDGE_RTOL = 1e-9


@dataclass(frozen=True, eq=False, repr=False)
class BinMapping(ReadOnlyArraysMixin):
    """Exact correspondence from the bins of a parent environment to a child's.

    Returned by :meth:`Environment.bin_mapping_from`. Every pair is proven
    from geometry: the parent bin is the same cell as the child bin, or lies
    wholly inside it. The transfer methods carry values computed on the
    parent environment to the child's bins without any nearest-center
    approximation. Arrays are read-only, including after pickling or copying.

    Attributes
    ----------
    pairs : ndarray of intp, shape (n_pairs, 2)
        Rows of ``(bin in parent, bin in child)``, sorted by child bin, then
        parent bin. Each parent bin appears at most once. For a subset, a
        clip or an identical rebuild, these are the rows
        ``compare_spatial_rates(rates_parent, rates_child, bin_mapping=...)``
        takes.
    kind : {"one_to_one", "many_to_one"}
        ``"many_to_one"`` when some child bin contains more than one parent
        bin, as after ``rebin``.
    n_parent_bins, n_child_bins : int
        Number of bins in each environment.
    unmapped_parent_bins : ndarray of intp, shape (n_unmapped,)
        Parent bins with no child bin: dropped by a subset or a clip, beyond
        a truncated rebin, or in a coarse cell whose center fell outside the
        arena. Their values are not carried to the child.
    split_parent_bins : ndarray of intp, shape (n_split,)
        Parent bins whose cell is divided between child bins, such as cells
        a clip cut. They are in no pair, and their values are not carried.
    uncovered_child_bins : ndarray of intp, shape (n_uncovered,)
        Child bins that contain no whole parent bin, such as the pieces of a
        cut cell. Transfers give them NaN (or ``missing``).
    child_coverage : ndarray of float64, shape (n_child_bins,)
        Fraction of each child bin's area (or length, volume) covered by its
        mapped parent bins: 1 for an identical cell or a coarse cell made of
        whole parent cells, less than 1 for a coarse cell that also covers
        space the parent did not bin, 0 for an uncovered child bin.

    See Also
    --------
    Environment.bin_mapping_from : Build the mapping.
    neurospatial.encoding.compare_spatial_rates : Takes ``pairs`` as
        ``bin_mapping=``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> fine = Environment.from_grid_mask(
    ...     np.ones((4, 4), dtype=bool), grid_edges=(np.arange(5.0),) * 2
    ... )
    >>> mapping = fine.rebin(2).bin_mapping_from(fine)
    >>> mapping.kind, mapping.n_parent_bins, mapping.n_child_bins
    ('many_to_one', 16, 4)
    >>> mapping.sum(np.ones(16)).tolist()  # four fine bins in each coarse bin
    [4.0, 4.0, 4.0, 4.0]
    """

    pairs: NDArray[np.intp]
    kind: Literal["one_to_one", "many_to_one"]
    n_parent_bins: int
    n_child_bins: int
    unmapped_parent_bins: NDArray[np.intp]
    split_parent_bins: NDArray[np.intp]
    uncovered_child_bins: NDArray[np.intp]
    child_coverage: NDArray[np.float64]

    def __post_init__(self) -> None:
        self._own_arrays()

    def sum(self, values: ArrayLike) -> NDArray[np.float64]:
        """Add up an extensive quantity of the parent bins in each child bin.

        Use it for quantities that add over space: spike counts, occupancy
        seconds, probability mass. Never for rates or other densities; see
        :meth:`rates`.

        Parameters
        ----------
        values : array_like, shape (n_parent_bins,) or (n_units, n_parent_bins)
            Values on the parent environment's bins.

        Returns
        -------
        ndarray of float64, shape (n_child_bins,) or (n_units, n_child_bins)
            Each child bin's sum over its mapped parent bins. NaN when any of
            them is NaN (missing is not zero, so NaN is never skipped), and
            NaN for ``uncovered_child_bins``. Values of
            ``unmapped_parent_bins`` and ``split_parent_bins`` are not
            carried, so totals over the child bins can be smaller than over
            the parent bins.

        Raises
        ------
        ValueError
            If ``values`` does not have ``n_parent_bins`` on its last axis or
            is not numeric.

        Notes
        -----
        Occupancy summed this way equals ``child.occupancy(times, positions)``
        from the same trajectory when positions in unmapped parent bins are
        excluded from both (set to NaN). Otherwise an interval that starts in
        a mapped bin and ends in an unmapped one counts on the parent, which
        has both samples in bins, but not on the child, which does not.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> fine = Environment.from_grid_mask(
        ...     np.ones((1, 4), dtype=bool), grid_edges=(np.arange(2.0), np.arange(5.0))
        ... )
        >>> mapping = fine.rebin((1, 2)).bin_mapping_from(fine)
        >>> mapping.sum([1.0, 2.0, 3.0, np.nan]).tolist()
        [3.0, nan]
        """
        array = self._parent_values(values, "values", "sum")
        totals: NDArray[np.float64] = np.full(
            (*array.shape[:-1], self.n_child_bins), np.nan
        )
        covered, starts = self._groups()
        totals[..., covered] = np.add.reduceat(
            array[..., self.pairs[:, 0]], starts, axis=-1
        )
        return totals

    def rates(self, counts: ArrayLike, occupancy: ArrayLike) -> NDArray[np.float64]:
        """Pool spike counts and occupancy into each child bin's rate.

        A child bin's rate is its summed counts over its summed occupancy.
        Rates are never averaged: rates of 0 and 100 Hz over 100 s and 1 s
        pool to 100 spikes / 101 s = 0.99 Hz, not 50 Hz.

        Parameters
        ----------
        counts : array_like, shape (n_parent_bins,) or (n_units, n_parent_bins)
            Raw spike counts per parent bin. For a rate result from
            ``method="binned", bandwidth=0`` with no ``min_occupancy`` they
            are ``np.where(occupancy > 0, firing_rates * occupancy, 0.0)``:
            an unvisited bin has a NaN rate but no spikes. A smoothed map's
            rates are not counts per bin: re-estimate it on the child
            environment instead.
        occupancy : array_like, shape (n_parent_bins,) or counts.shape
            Seconds spent in each parent bin.

        Returns
        -------
        ndarray of float64, shape counts.shape[:-1] + (n_child_bins,)
            Rates in spikes per second. NaN where the summed occupancy is
            zero, where any mapped count or occupancy is NaN, and for
            ``uncovered_child_bins``.

        Raises
        ------
        ValueError
            If either array does not have ``n_parent_bins`` on its last axis,
            is not numeric, or ``occupancy`` is neither one row nor the shape
            of ``counts``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> fine = Environment.from_grid_mask(
        ...     np.ones((1, 2), dtype=bool), grid_edges=(np.arange(2.0), np.arange(3.0))
        ... )
        >>> mapping = fine.rebin((1, 2)).bin_mapping_from(fine)
        >>> # 0 Hz for 100 s and 100 Hz for 1 s: 100 spikes in 101 s
        >>> mapping.rates([0.0, 100.0], [100.0, 1.0]).round(2).tolist()
        [0.99]
        """
        spikes = self._parent_values(counts, "counts", "rates")
        seconds = self._parent_values(occupancy, "occupancy", "rates")
        if seconds.shape not in ((self.n_parent_bins,), spikes.shape):
            raise ValueError(
                _format_error(
                    f"BinMapping.rates() got occupancy with shape {seconds.shape} "
                    f"for counts with shape {spikes.shape}.",
                    why="Why: each count is divided by the time spent in the same bin.",
                    fix=f"pass occupancy with shape ({self.n_parent_bins},), "
                    "shared by every unit, or the shape of counts",
                )
            )
        summed_counts, summed_seconds = self.sum(spikes), self.sum(seconds)
        with np.errstate(divide="ignore", invalid="ignore"):
            pooled = summed_counts / summed_seconds
        result: NDArray[np.float64] = np.where(summed_seconds > 0, pooled, np.nan)
        return result

    def category(self, labels: ArrayLike, *, missing: Any) -> NDArray[Any]:
        """Carry a label of the parent bins to each child bin they agree on.

        A child bin keeps a label only when all its mapped parent bins share
        it, so a coarse bin straddling two regions gets ``missing``.

        Parameters
        ----------
        labels : array_like, shape (n_parent_bins,) or (n_units, n_parent_bins)
            One label per parent bin: integers, strings or booleans, such as
            region names or field membership.
        missing : scalar
            The value for child bins whose parent bins disagree and for
            ``uncovered_child_bins``. It must not be one of the labels, so it
            cannot be mistaken for one.

        Returns
        -------
        ndarray, shape labels.shape[:-1] + (n_child_bins,)
            The child bins' labels. Its dtype holds both the labels and
            ``missing`` (for example ``"<U6"`` for short names and
            ``missing="mixed"``); labels and a ``missing`` of another kind,
            such as booleans with ``missing=None``, give an object array.

        Raises
        ------
        ValueError
            If ``labels`` does not have ``n_parent_bins`` on its last axis, or
            ``missing`` is one of the labels.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> fine = Environment.from_grid_mask(
        ...     np.ones((1, 4), dtype=bool), grid_edges=(np.arange(2.0), np.arange(5.0))
        ... )
        >>> mapping = fine.rebin((1, 2)).bin_mapping_from(fine)
        >>> mapping.category(["arm", "arm", "arm", "goal"], missing="mixed").tolist()
        ['arm', 'mixed']
        """
        array = self._parent_array(labels, "labels", "category")
        present = set(array.ravel().tolist())
        if missing in present:
            raise ValueError(
                _format_error(
                    f"BinMapping.category() got missing={missing!r}, which is "
                    "one of the labels.",
                    why="Why: a child bin given missing could not be told from "
                    "one that really has that label.",
                    fix="pass a value that is not a label, e.g. missing=-1 for "
                    "non-negative integer labels or missing='mixed' for names",
                )
            )
        result = np.full(
            (*array.shape[:-1], self.n_child_bins),
            missing,
            dtype=_category_dtype(array.dtype, missing),
        )
        covered, starts = self._groups()
        mapped = array[..., self.pairs[:, 0]]
        first = mapped[..., starts]
        sizes = np.diff(np.append(starts, len(self.pairs)))
        agree = np.logical_and.reduceat(
            np.asarray(mapped == np.repeat(first, sizes, axis=-1), dtype=bool),
            starts,
            axis=-1,
        )
        kept = result[..., covered]
        kept[agree] = first[agree]
        result[..., covered] = kept
        return result

    def summary(self) -> dict[str, Any]:
        """Counts of each kind of bin, as a flat dict of scalars.

        Returns
        -------
        dict
            ``kind``, ``n_parent_bins``, ``n_child_bins``, ``n_pairs``,
            ``n_mapped_parent_bins``, ``n_unmapped_parent_bins``,
            ``n_split_parent_bins``, ``n_fully_covered_child_bins``,
            ``n_partly_covered_child_bins`` and ``n_uncovered_child_bins``.
        """
        return {
            "kind": self.kind,
            "n_parent_bins": self.n_parent_bins,
            "n_child_bins": self.n_child_bins,
            "n_pairs": len(self.pairs),
            "n_mapped_parent_bins": len(self.pairs),
            "n_unmapped_parent_bins": len(self.unmapped_parent_bins),
            "n_split_parent_bins": len(self.split_parent_bins),
            "n_fully_covered_child_bins": int(
                np.count_nonzero(self.child_coverage == 1)
            ),
            "n_partly_covered_child_bins": int(
                np.count_nonzero((self.child_coverage > 0) & (self.child_coverage < 1))
            ),
            "n_uncovered_child_bins": len(self.uncovered_child_bins),
        }

    def __repr__(self) -> str:
        counts = self.summary()
        return (
            f"BinMapping(kind={self.kind!r}, {self.n_parent_bins} parent bins -> "
            f"{self.n_child_bins} child bins; parent bins: "
            f"{counts['n_mapped_parent_bins']} mapped, "
            f"{counts['n_unmapped_parent_bins']} unmapped, "
            f"{counts['n_split_parent_bins']} split; child bins: "
            f"{counts['n_fully_covered_child_bins']} fully covered, "
            f"{counts['n_partly_covered_child_bins']} partly covered, "
            f"{counts['n_uncovered_child_bins']} uncovered)"
        )

    def _groups(self) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Covered child bins and the row of ``pairs`` where each one starts."""
        children = self.pairs[:, 1]
        starts = np.flatnonzero(np.r_[True, children[1:] != children[:-1]])
        return children[starts], starts

    def _parent_array(self, values: ArrayLike, name: str, method: str) -> NDArray[Any]:
        """``values`` as an array with parent bins on its last axis."""
        array = np.asarray(values)
        if array.ndim in (1, 2) and array.shape[-1] == self.n_parent_bins:
            return array
        if array.ndim in (1, 2) and array.shape[-1] == self.n_child_bins:
            fix = (
                f"these look like values on the child's {self.n_child_bins} bins; "
                "pass the values computed on the parent environment"
            )
        elif array.ndim == 2 and array.shape[0] == self.n_parent_bins:
            fix = f"put bins on the last axis: pass {name}.T"
        else:
            fix = (
                f"pass one value per parent bin, shape ({self.n_parent_bins},), "
                f"or one row per unit, shape (n_units, {self.n_parent_bins})"
            )
        raise ValueError(
            _format_error(
                f"BinMapping.{method}() got {name} with shape {array.shape}; it "
                f"takes values on the parent environment's {self.n_parent_bins} "
                "bins.",
                why="Why: entry j of the last axis is parent bin j.",
                fix=fix,
            )
        )

    def _parent_values(
        self, values: ArrayLike, name: str, method: str
    ) -> NDArray[np.float64]:
        """Numeric ``values`` on the parent bins, as float64."""
        array = self._parent_array(values, name, method)
        if array.dtype.kind not in "biuf":
            raise ValueError(
                _format_error(
                    f"BinMapping.{method}() got {name} of dtype {array.dtype}; it "
                    "adds numbers.",
                    fix="carry labels such as region names with "
                    "BinMapping.category(labels, missing=...)",
                )
            )
        return np.asarray(array, dtype=np.float64)


def _category_dtype(labels: np.dtype[Any], missing: Any) -> np.dtype[Any]:
    """A dtype holding the labels and ``missing`` without converting either."""
    fill = np.asarray(missing)
    for family in ("b", "iu", "f", "U", "S"):
        if labels.kind in family and fill.dtype.kind in family:
            return np.result_type(labels, fill)
    return np.dtype(object)


def _bin_mapping_from(child: Environment, parent: Environment) -> BinMapping:
    """Build ``child.bin_mapping_from(parent)``; see that method."""
    from neurospatial.environment.core import Environment

    if not isinstance(parent, Environment):
        raise TypeError(
            _format_error(
                f"{_OPERATION} got parent of type {type(parent).__name__}.",
                fix="pass the Environment this one was made from, e.g. "
                "child.bin_mapping_from(env) for child = env.subset(...)",
            )
        )
    _refuse_labels(child, parent)
    _refuse_dimensions(child, parent)
    parent_bins, child_bins = parent._binning, child._binning
    lattices = _shared_lattice(parent_bins, child_bins)
    if isinstance(lattices, tuple):
        parent_grid, child_grid = lattices
        nested = _nested_pairs(parent_grid, child_grid, parent_bins, child_bins)
        if nested is None:
            raise _no_pairs_error(
                child,
                parent,
                _lattice_reason(parent_grid, child_grid),
                finer=_nests(child_grid, parent_grid) is not None,
            )
        pairs, coverage = nested
        split = np.zeros(0, dtype=np.intp)
        reason = "every parent bin lies outside this environment's bins"
    else:
        pairs, split = _identity_pairs(parent, child)
        coverage = np.zeros(child.n_bins)
        coverage[pairs[:, 1]] = 1.0
        reason = lattices or (
            f"{len(split)} parent bins are split between bins of this "
            "environment, and no bin is the same cell as a parent bin"
            if len(split)
            else "no bin of this environment is the same cell as a parent bin"
        )
    if len(pairs) == 0:
        raise _no_pairs_error(child, parent, reason, finer=False)
    return _assemble(pairs, split, coverage, parent.n_bins, child.n_bins)


def _assemble(
    pairs: NDArray[np.intp],
    split: NDArray[np.intp],
    coverage: NDArray[np.float64],
    n_parent: int,
    n_child: int,
) -> BinMapping:
    """Sort the pairs and sort every bin into its category."""
    pairs = pairs[np.lexsort((pairs[:, 0], pairs[:, 1]))]
    per_child = np.bincount(pairs[:, 1], minlength=n_child)
    mapped = np.zeros(n_parent, dtype=bool)
    mapped[pairs[:, 0]] = True
    split = np.setdiff1d(split, pairs[:, 0])
    unmapped = ~mapped
    unmapped[split] = False
    return BinMapping(
        pairs=pairs.astype(np.intp),
        kind="many_to_one" if per_child.max() > 1 else "one_to_one",
        n_parent_bins=n_parent,
        n_child_bins=n_child,
        unmapped_parent_bins=np.flatnonzero(unmapped).astype(np.intp),
        split_parent_bins=split.astype(np.intp),
        uncovered_child_bins=np.flatnonzero(per_child == 0).astype(np.intp),
        child_coverage=coverage,
    )


# --- coordinate labels and refusals ----------------------------------------


def _transform_note(child: Environment, parent: Environment) -> str:
    """Advice for a child that may be ``parent.apply_transform(...)``, or ''."""
    known = parent.derivations
    later = child.derivations[len(known) :]
    if (
        child.derivations[: len(known)] != known
        or not later
        or set(later) != {"apply_transform"}
        or child.n_bins != parent.n_bins
    ):
        return ""
    return (
        "If this environment is parent.apply_transform(...), it needs no "
        "mapping: transforms keep bin IDs, so bin i of the transformed "
        "environment is bin i of the original, and values computed on parent "
        "apply to it index for index."
    )


def _refuse_labels(child: Environment, parent: Environment) -> None:
    """Refuse environments whose coordinate labels differ."""
    reason = _label_difference(parent, child)
    if reason is None:
        return
    if reason == "coordinate systems":
        named = (
            f"coordinate systems ({parent.coordinate_system!r} for parent vs "
            f"{child.coordinate_system!r})"
        )
        remedy = "build both environments in the same coordinate system"
    else:
        named = (
            f"units or coordinate frames (parent: units {parent.units!r}, frame "
            f"{parent.frame!r}; this environment: units {child.units!r}, frame "
            f"{child.frame!r})"
        )
        remedy = (
            "build both environments from positions in the same units and "
            "frame; if only the labels are wrong, correct them with "
            "with_metadata(units=..., frame=...)"
        )
    note = _transform_note(child, parent)
    raise IncompatibleEnvironmentError(
        f"{_OPERATION}: this environment and parent differ in {named}, so the "
        "same coordinates are different places and no bin can be matched to a "
        "bin of parent.",
        fix=f"{note} Otherwise, {remedy}" if note else remedy,
        first=parent,
        second=child,
    )


def _no_pairs_error(
    child: Environment, parent: Environment, reason: str, *, finer: bool
) -> ValueError:
    """The refusal for environments with no exact pair, with how to proceed."""
    note = _transform_note(child, parent)
    fixes = [note] if note else []
    if finer:
        fixes.append(
            "To carry values the other way, from these finer bins into the "
            "coarser bins of parent, call parent.bin_mapping_from(env)."
        )
    fixes.append(
        "Re-estimate on this environment from the original observations, e.g. "
        "env.occupancy(times, positions), or compute_spatial_rates with env; "
        "or accept an approximation with neurospatial.ops.map_probabilities or "
        "neurospatial.ops.resample_field, which assign each bin by its nearest "
        "center and do not transfer values exactly."
    )
    return ValueError(
        _format_error(
            f"{_OPERATION} found no bin of this environment (env) that is a bin "
            f"of parent or contains whole parent bins: {reason}.",
            why="Why: values move exactly only between identical cells, or from "
            "whole cells into the coarser cell that contains them. A cell divided "
            "between bins cannot have its count divided without knowing where in "
            "the cell each observation fell.",
            fix=" ".join(fixes),
        )
    )


# --- nested grids -----------------------------------------------------------


def _refuse_dimensions(child: Environment, parent: Environment) -> None:
    """Refuse environments whose coordinates have different widths."""
    if child.n_dims == parent.n_dims:
        return
    raise IncompatibleEnvironmentError(
        f"{_OPERATION}: parent has {parent.n_dims}-D coordinates and this "
        f"environment {child.n_dims}-D, so no bin of one is a place in the other.",
        fix="build both environments from positions with the same number of "
        "coordinates",
        first=parent,
        second=child,
    )


def _lattice(binning: Binning) -> tuple[NDArray[np.float64], RectilinearGrid] | None:
    """The grid's map from lattice to environment coordinates, and its cells."""
    n_dims = binning.bins.n_dims
    if binning.rectilinear is not None:
        return np.eye(n_dims + 1), binning.rectilinear
    if binning.grid_frame is not None:
        return binning.grid_frame.forward, binning.grid_frame.source
    return None


def _shared_lattice(
    parent: Binning, child: Binning
) -> tuple[RectilinearGrid, RectilinearGrid] | str | None:
    """Both grids in their shared lattice coordinates, a reason, or None.

    Returns the two grids when both environments are rectangular grids drawn
    in the same lattice frame; a reason when both are grids in different
    frames; None when either is not a grid.
    """
    first, second = _lattice(parent), _lattice(child)
    if first is None or second is None:
        return None
    if not np.array_equal(first[0], second[0]):
        return (
            "both are grids, but drawn in different lattice frames (one is "
            "rotated, sheared or shifted relative to the other), so no cell of "
            "one lies exactly inside a cell of the other"
        )
    return first[1], second[1]


def _axis_fit(
    inner: NDArray[np.float64], outer: NDArray[np.float64], tolerance: float
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Which outer interval holds each inner interval along one axis.

    Returns
    -------
    position : ndarray of intp, shape (len(inner) - 1,)
        The outer interval containing each inner interval, or -1 when it
        lies outside the outer edges.
    fits : ndarray of bool, shape (len(inner) - 1,)
        False for an inner interval that an outer edge cuts.
    """
    low, high = inner[:-1], inner[1:]
    index = np.searchsorted(outer, low + tolerance, side="right") - 1
    within = (index >= 0) & (index < len(outer) - 1)
    upper = outer[np.clip(index + 1, 0, len(outer) - 1)]
    inside = within & (high - tolerance <= upper)
    outside = (high <= outer[0] + tolerance) | (low >= outer[-1] - tolerance)
    return np.where(inside, index, -1).astype(np.intp), inside | outside


def _nests(inner: RectilinearGrid, outer: RectilinearGrid) -> NDArray[np.intp] | None:
    """Each active inner cell's outer lattice cell, or None if an edge cuts one.

    Returns
    -------
    ndarray of intp, shape (n_dims, n_inner_bins), or None
        Outer lattice index along each axis for every inner bin, -1 when the
        bin lies outside the outer grid. None when an outer edge cuts an
        inner bin. Edges coincide within a billionth of the narrower cell
        width.
    """
    cells = np.nonzero(inner.active_mask)
    positions = []
    for axis, (inner_edges, outer_edges) in enumerate(
        zip(inner.edges, outer.edges, strict=True)
    ):
        width = min(np.min(np.diff(inner_edges)), np.min(np.diff(outer_edges)))
        position, fits = _axis_fit(inner_edges, outer_edges, _EDGE_RTOL * width)
        if not np.all(fits[cells[axis]]):
            return None
        positions.append(position[cells[axis]])
    return np.array(positions, dtype=np.intp).reshape(len(cells), -1)


def _nested_pairs(
    parent_grid: RectilinearGrid,
    child_grid: RectilinearGrid,
    parent: Binning,
    child: Binning,
) -> tuple[NDArray[np.intp], NDArray[np.float64]] | None:
    """Pair each parent cell with the child cell holding it, with coverage.

    Returns None when a child edge cuts a parent cell. Parent cells outside
    the child grid, or in an inactive child cell, are left unpaired.
    """
    positions = _nests(parent_grid, child_grid)
    if positions is None:
        return None
    child_of_cell = np.full(child_grid.shape, -1, dtype=np.intp)
    child_of_cell[np.nonzero(child_grid.active_mask)] = np.arange(child.bins.n_bins)
    inside = np.all(positions >= 0, axis=0)
    owner = np.full(parent.bins.n_bins, -1, dtype=np.intp)
    owner[inside] = child_of_cell[tuple(positions[:, inside])]
    parents = np.flatnonzero(owner >= 0)
    pairs = np.column_stack([parents, owner[parents]]).astype(np.intp)
    covered = np.bincount(
        owner[parents],
        weights=parent.bins.sizes[parents],
        minlength=child.bins.n_bins,
    )
    ratio = covered / child.bins.sizes
    # Edges that coincide within the tolerance leave gaps of that size; a
    # cell made of whole parent cells is fully covered.
    complete = np.abs(ratio - 1) <= 2 * len(positions) * _EDGE_RTOL
    coverage = np.where(complete, 1.0, np.minimum(ratio, 1.0))
    return pairs, coverage


def _lattice_reason(parent: RectilinearGrid, child: RectilinearGrid) -> str:
    """Why grids in one lattice frame have no exact pair."""
    if _nests(child, parent) is not None:
        return (
            "this environment's cells are smaller than parent's and lie inside "
            "them (a refinement), so every parent bin it overlaps is split"
        )
    return (
        "the grids share a lattice frame, but this environment's cell edges do "
        "not fall on parent's cell edges, so its edges cut parent cells (use a "
        "bin_size that is a whole multiple of parent's cell size from the same "
        "origin, or rebin(factor))"
    )


# --- identical cells --------------------------------------------------------


def _identity_pairs(
    parent: Environment, child: Environment
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Pairs of identical cells and the parent bins split between child bins.

    Cells are compared by ``_paired_cell_differences``, the definition of the
    same cell that ``compare_spatial_rates(bin_mapping=...)`` also checks.
    Candidates are the parent bins with exactly the child centers, or for
    polygon cells, whose centers may differ by rounding, the parent bin
    containing each child center.
    """
    parent_bins, child_bins = parent._binning, child._binning
    if not _compares_polygons(parent_bins, child_bins):
        child_rows, parent_rows = _equal_centers(
            parent_bins.bins.centers, child_bins.bins.centers
        )
        same = _paired_cell_differences(parent, child, parent_rows, child_rows) < 0
        pairs = np.column_stack([parent_rows[same], child_rows[same]])
        return pairs.astype(np.intp), np.zeros(0, dtype=np.intp)
    candidates = parent.bin_at(child.bin_centers)
    child_rows = np.flatnonzero(candidates >= 0).astype(np.intp)
    parent_rows = candidates[child_rows]
    same = _paired_cell_differences(parent, child, parent_rows, child_rows) < 0
    pairs = np.column_stack([parent_rows[same], child_rows[same]]).astype(np.intp)
    # A child cell inside a larger parent cell means that cell is divided.
    rest = ~same
    parent_cells = _cell_array(parent_bins)[parent_rows[rest]]
    child_cells = _cell_array(child_bins)[child_rows[rest]]
    child_area = shapely.area(child_cells)
    within = (
        shapely.area(shapely.intersection(parent_cells, child_cells))
        >= (1 - _EDGE_RTOL) * child_area
    )
    smaller = child_area < (1 - _EDGE_RTOL) * shapely.area(parent_cells)
    split = np.unique(parent_rows[rest][within & smaller]).astype(np.intp)
    return pairs, split


def _equal_centers(
    parent: NDArray[np.float64], child: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Child bins whose center is exactly a parent bin's center, and that bin."""
    _, first, inverse = np.unique(
        np.vstack([parent, child]), axis=0, return_index=True, return_inverse=True
    )
    owner = first[np.ravel(inverse)[len(parent) :]]
    matched = owner < len(parent)
    return np.flatnonzero(matched).astype(np.intp), owner[matched].astype(np.intp)
