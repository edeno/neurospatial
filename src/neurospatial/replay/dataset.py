"""Data-set-level replay tests: is there more replay in a data set than chance?

Several papers draw no per-event replay verdict. They ask whether a whole
data set (a session, or every event of a condition) holds more trajectory
events than data sets without sequences would, and test it in one of three
ways:

- A data-set statistic against null data sets (Monte Carlo). The share of
  events passing thresholds, at every point of a grid of thresholds, is
  compared with its value in each of ``K`` null data sets, each made by
  shuffling every event once (Silva et al. 2015's significance matrix;
  Farooq et al. 2019). :func:`dataset_values` measures the events of the
  real and the null data sets the same way, and :func:`dataset_test` turns
  them into a p-value per grid point.
- A pooled two-sample test of per-event values: the real events' values
  against the pooled values of every null event, by a Kolmogorov-Smirnov,
  rank-sum or Levene test (Stella et al. 2019; Xu et al. 2019; Grosmark &
  Buzsáki 2016, Fig. S8; Silva et al. 2015, Fig. 1d).
  :func:`distribution_test` states the side in terms of the values.
- A share of events against a chance rate (Grosmark & Buzsáki 2016, Figs.
  S5C and S7B): ``detection_rates(table, expected_rate=0.05)``.

Liu et al. (2019) compared a sleep template, each cell's mean relative
rank in its sleep frames (:func:`relative_rank_template`), with the running
template, against shuffles of the running template
(:func:`template_correlation_test`).

Spike-level surrogate data sets (rate-matched Poisson, jittered spikes) are
made with :func:`~neurospatial.replay.null_events` and decoded with
:func:`~neurospatial.replay.decode_intervals`; their ``copy_index`` names the
null data set of each surrogate event. The ``*_2015`` to ``*_2019``
:class:`DatasetTestRecipe` objects record each paper's data-set test, with
every value and the choices the paper does not state.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial.replay._compare import (
    _check_criterion_settings,
    _compare,
    _criterion_label,
)
from neurospatial.replay.correlation import weighted_correlation
from neurospatial.replay.methods import (
    FAROOQ_2019,
    GROSMARK_2016,
    SILVA_2015,
    RankOrderMethod,
    ReplayCriterion,
    ReplayMethod,
)
from neurospatial.replay.significance import _ALTERNATIVES, _DECODING_NULLS
from neurospatial.stats._utils import _entropy, _stream

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from neurospatial.decoding import DecodingResult
    from neurospatial.environment import Environment

__all__ = [
    "DATASET_RECIPES",
    "FAROOQ_2019_POISSON",
    "FAROOQ_2019_SPACE_BIN",
    "FAROOQ_2019_TIME_BIN",
    "FAROOQ_2019_WEIGHTED",
    "GROSMARK_2016_SURROGATES",
    "LIU_2019_TEMPLATES",
    "SILVA_2015_MATRIX",
    "SILVA_2015_RANK_ORDER",
    "STELLA_2019_JITTER",
    "XU_2019_ROTATION",
    "CriterionGrid",
    "DatasetTestRecipe",
    "DatasetTestResult",
    "DistributionTestResult",
    "ShuffledDataset",
    "dataset_test",
    "dataset_values",
    "distribution_test",
    "relative_rank_template",
    "shuffle_events",
    "template_correlation_test",
]

# Nulls that shuffle the decoded posterior of each event.
_POSTERIOR_NULLS = (
    "column_cycle",
    "column_cycle_weighted",
    "time_bin_permutation",
    "pseudo_event",
)
_EVENT_NULLS = _POSTERIOR_NULLS + _DECODING_NULLS
# Nulls whose one draw (a permutation of cells, a set of shifted rate maps)
# can be shared by every event of a data set.
_SHAREABLE_NULLS = ("unit_identity", "place_field_circular", "cell_order")
# Posterior nulls whose shuffles are a row order and a circular shift per row.
_ROLLING_NULLS = ("column_cycle", "column_cycle_weighted", "time_bin_permutation")
# Posterior nulls that shift rows along a track, so need a linearized track.
_COLUMN_CYCLE_NULLS = ("column_cycle", "column_cycle_weighted")
# Quantities the moment-table fast path computes; the others go through
# detect_replay.
_FAST_QUANTITIES = (
    "score",
    "n_active_cells",
    "active_fraction",
    "n_spikes",
    "n_time_bins",
    "duration",
    "track_length",
    "max_map_jump",
    "map_path_length",
    "start_to_end_map_distance",
    "map_range",
    "mean_peak_probability",
)


# --------------------------------------------------------------------------
# Thresholds over a grid
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CriterionGrid:
    """A :class:`~neurospatial.replay.ReplayCriterion` with several thresholds.

    :func:`dataset_test` counts the events passing every criterion at each
    point of the product of the grids' thresholds, such as Silva et al.'s
    (2015) 10 x 10 matrix of |weighted correlation| and maximum jump
    thresholds.

    Attributes
    ----------
    quantity : str
        The column of the per-event table that is compared, such as
        ``"score"`` or ``"max_map_jump"``.
    comparison : {">", ">=", "<", "<="}
        How each value is compared with each threshold.
    thresholds : float or NDArray[np.float64], shape (n_thresholds,)
        The thresholds, finite, in environment units for distances unless
        ``relative_to`` is set. A single number is one threshold without a
        grid axis: the statistic then has no axis for this criterion.
    absolute : bool, default=False
        Compare the absolute value, for signed quantities such as the
        weighted correlation.
    relative_to : {"track_length"} or None, default=None
        Scale each threshold by the row's ``track_length``, for thresholds
        given as a fraction of the track.

    Notes
    -----
    Each comparison follows :meth:`ReplayCriterion.evaluate
    <neurospatial.replay.ReplayCriterion.evaluate>`: a NaN value fails, and
    a value within 1e-9 (absolute plus relative) of the threshold counts as
    equal to it.

    Examples
    --------
    Silva et al.'s grid, |r| > 0, 0.1, ..., 0.9 and jumps below 0.1, ...,
    1.0 of the track:

    >>> import numpy as np
    >>> from neurospatial.replay import CriterionGrid
    >>> grid = [
    ...     CriterionGrid("score", ">", np.arange(10) / 10, absolute=True),
    ...     CriterionGrid(
    ...         "max_map_jump", "<", np.arange(1, 11) / 10, relative_to="track_length"
    ...     ),
    ... ]
    >>> grid[0].label, grid[1].label
    ('|score| >', 'max_map_jump < x track_length')
    """

    quantity: str
    comparison: Literal[">", ">=", "<", "<="]
    thresholds: NDArray[np.float64]
    absolute: bool = False
    relative_to: Literal["track_length"] | None = None

    def __post_init__(self) -> None:
        """Own a read-only copy of the thresholds and refuse invalid settings."""
        prefix = f"CriterionGrid {self.quantity!r}"
        thresholds = np.array(self.thresholds, dtype=np.float64)
        if thresholds.ndim > 1 or thresholds.size == 0:
            raise ValueError(
                f"{prefix}: thresholds must be a number or a 1-D array of at "
                f"least one threshold, got shape {thresholds.shape}."
            )
        if not np.isfinite(thresholds).all():
            raise ValueError(
                _format_error(
                    f"{prefix}: thresholds must be finite.",
                    why="every event fails a NaN threshold",
                    fix="give the thresholds as numbers, such as np.arange(10) / 10",
                )
            )
        thresholds.flags.writeable = False
        object.__setattr__(self, "thresholds", thresholds)
        _check_criterion_settings(
            prefix, self.comparison, self.relative_to, self.absolute
        )

    @property
    def label(self) -> str:
        """A readable form without the thresholds, such as ``"|score| >"``."""
        return _criterion_label(
            self.quantity,
            self.comparison,
            absolute=self.absolute,
            relative_to=self.relative_to,
        )

    def passes(self, table: pd.DataFrame) -> NDArray[np.bool_]:
        """Whether each row of ``table`` passes each threshold.

        Parameters
        ----------
        table : pandas.DataFrame
            Per-event values with a ``quantity`` column, and a
            ``track_length`` column for a grid relative to it.

        Returns
        -------
        NDArray[np.bool_], shape (n_rows, n_thresholds)
            The comparison of each row with each threshold.

        Raises
        ------
        ValueError
            If ``table`` lacks the quantity's column, or the
            ``track_length`` column for a grid relative to it.
        """
        for column in (self.quantity, *(("track_length",) if self.relative_to else ())):
            if column not in table:
                raise ValueError(
                    _format_error(
                        f"CriterionGrid {self.label!r}: the table has no "
                        f"{column!r} column.",
                        why="each row's value is compared with the thresholds"
                        if column == self.quantity
                        else "thresholds relative to the track length are "
                        "scaled by each row's track_length",
                        fix="measure the events with dataset_values(..., "
                        f"quantities=({self.quantity!r},)), or add the column",
                    )
                )
        values = table[self.quantity].to_numpy(dtype=np.float64)
        if self.absolute:
            values = np.abs(values)
        scale = (
            table["track_length"].to_numpy(dtype=np.float64)
            if self.relative_to
            else np.ones(len(values))
        )
        thresholds = np.atleast_1d(self.thresholds)[None, :] * scale[:, None]
        return _compare(values[:, None], self.comparison, thresholds)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True, repr=False, eq=False)
class DatasetTestResult(ReadOnlyArraysMixin, ResultMixin):
    """A data-set statistic tested against null data sets.

    Attributes
    ----------
    statistic : float or NDArray[np.float64]
        The statistic of the observed data set: for :func:`dataset_test`
        with ``criteria``, the share of events passing every criterion at
        each point of the grid, shape ``grid_shape``; a float for one
        threshold per criterion or a scalar statistic.
    null_statistics : NDArray[np.float64], shape (n_datasets, *grid_shape)
        The statistic of each null data set.
    p_value : float or NDArray[np.float64]
        ``(1 + k) / (1 + n_datasets)`` at each grid point, where ``k``
        counts null data sets at least as extreme as the observed one
        (``>=`` for ``"greater"``, ``<=`` for ``"less"``); twice the
        smaller side, at most 1, for ``"two-sided"``. This is Silva et
        al.'s (2015) ``(n + 1) / (r + 1)`` exactly.
    percentile : float or NDArray[np.float64]
        ``100 * #(null < observed) / n_datasets``. Papers that report a
        percentile use ``p = 1 - percentile / 100`` without the ``+ 1``:
        Farooq et al. (2019, "the number of shuffled datasets with lower
        proportions than the real sleep was divided by the total number of
        shuffles. This value was subtracted from 1") and Liu et al. (2019,
        one-tailed; twice that two-tailed). See
        :meth:`DatasetTestRecipe.paper_p_value`.
    alternative : {"greater", "less", "two-sided"}
        The side tested.
    n_datasets : int
        Number of null data sets.
    n_observed : int
        Rows of the observed data set: events (event and template pairs
        for a rank-order method with several templates), or templates for
        :func:`template_correlation_test`.
    labels : tuple of str
        The label of each grid axis, such as ``("|score| >", "max_map_jump
        < x track_length")``; empty for a statistic.
    thresholds : tuple of NDArray[np.float64]
        The thresholds along each grid axis, in the order of ``labels``.

    See Also
    --------
    dataset_test, template_correlation_test : Functions that produce this
        result.
    """

    statistic: float | NDArray[np.float64]
    null_statistics: NDArray[np.float64]
    p_value: float | NDArray[np.float64]
    percentile: float | NDArray[np.float64]
    alternative: str
    n_datasets: int
    n_observed: int
    labels: tuple[str, ...] = ()
    thresholds: tuple[NDArray[np.float64], ...] = ()

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays, including the thresholds."""
        owned = []
        for values in self.thresholds:
            array = np.array(values, dtype=np.float64)
            array.flags.writeable = False
            owned.append(array)
        object.__setattr__(self, "thresholds", tuple(owned))
        object.__setattr__(self, "labels", tuple(self.labels))
        self._own_arrays()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle or deepcopy, keeping every array read-only."""
        super().__setstate__(state)
        for values in self.thresholds:
            values.flags.writeable = False

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Shape of the grid of thresholds; ``()`` for one statistic."""
        return tuple(np.shape(self.statistic))

    @property
    def monte_carlo_error(self) -> float | NDArray[np.float64]:
        """Standard error of each p-value from the finite number of null data sets.

        ``sqrt(p (1 - p) / n_datasets)``: about 0.01 at p = 0.05 with 500
        data sets.
        """
        p = np.asarray(self.p_value, dtype=np.float64)
        error = np.sqrt(p * (1.0 - p) / self.n_datasets)
        return float(error) if error.ndim == 0 else error

    def grid_table(self) -> pd.DataFrame:
        """One row per grid point: its thresholds, statistic and p-value.

        Returns
        -------
        pandas.DataFrame
            One column per grid axis, named by its label and holding that
            axis' threshold, then ``statistic``, ``p_value``, ``percentile``
            and ``monte_carlo_error``; one row for a statistic.
        """
        import pandas as pd

        shape = self.grid_shape
        columns: dict[str, Any] = {}
        if self.labels:
            mesh = np.meshgrid(*self.thresholds, indexing="ij")
            for label, values in zip(self.labels, mesh, strict=True):
                columns[label] = values.ravel()
        size = int(np.prod(shape)) if shape else 1
        for name in ("statistic", "p_value", "percentile"):
            columns[name] = np.asarray(getattr(self, name), dtype=np.float64).reshape(
                size
            )
        columns["monte_carlo_error"] = np.asarray(
            self.monte_carlo_error, dtype=np.float64
        ).reshape(size)
        table = pd.DataFrame(columns)
        table.attrs.update(
            alternative=self.alternative,
            n_datasets=self.n_datasets,
            n_observed=self.n_observed,
        )
        return table

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result.

        For one statistic: the statistic, its p-value, percentile and Monte
        Carlo error. For a grid: the number of grid points, the smallest
        p-value and the number of points with p < 0.05, Silva et al.'s
        (2015) significance level.
        """
        values: dict[str, Any] = {
            "alternative": self.alternative,
            "n_datasets": self.n_datasets,
            "n_observed": self.n_observed,
        }
        if not self.grid_shape:
            values.update(
                statistic=float(self.statistic),
                p_value=float(self.p_value),
                percentile=float(self.percentile),
                monte_carlo_error=float(self.monte_carlo_error),
            )
            return values
        p = np.asarray(self.p_value)
        values.update(
            n_grid_points=int(p.size),
            min_p_value=float(p.min()),
            n_significant=int(np.sum(p < 0.05)),
        )
        return values

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the test: a significance matrix, a p-value curve or the null.

        A grid over two criteria is drawn as Silva et al.'s (2015, Fig. 3)
        significance matrix: the p-value of each pair of thresholds on a
        log scale, red below 0.05, white at 0.05 and blue above, with the
        first criterion down the rows and the second across the columns. A
        grid over one criterion plots the p-value against the threshold,
        and one statistic plots the null data sets' histogram with the
        observed value.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to draw on; a new figure when None.
        **kwargs
            Passed to ``imshow``, ``plot`` or ``hist``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.

        Raises
        ------
        ValueError
            For a grid over more than two criteria.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        if ax is None:
            _, ax = plt.subplots()
        shape = self.grid_shape
        if len(shape) > 2:
            raise ValueError(
                f"DatasetTestResult.plot(): a grid over {len(shape)} criteria "
                "cannot be drawn as one matrix; plot slices of grid_table()."
            )
        if not shape:
            ax.hist(self.null_statistics, bins=kwargs.pop("bins", 30), **kwargs)
            ax.axvline(float(self.statistic), color="k", linestyle="--")
            ax.set_xlabel("statistic")
            ax.set_ylabel("null data sets")
            ax.set_title(f"p = {float(self.p_value):.3g}")
            return ax
        if len(shape) == 1:
            ax.plot(self.thresholds[0], self.p_value, marker="o", **kwargs)
            ax.axhline(0.05, color="0.5", linestyle=":")
            ax.set_yscale("log")
            ax.set_xlabel(f"{self.labels[0]} threshold")
            ax.set_ylabel("p-value")
            return ax
        floor = np.log10(1.0 / (self.n_datasets + 1))
        log_p = np.log10(np.asarray(self.p_value, dtype=np.float64))
        image = ax.imshow(
            log_p,
            cmap=kwargs.pop("cmap", "RdBu"),
            norm=TwoSlopeNorm(
                vmin=min(floor, np.log10(0.05) - 1e-6),
                vcenter=np.log10(0.05),
                vmax=0.0,
            ),
            origin="upper",
            aspect="auto",
            **kwargs,
        )
        rows, columns = self.thresholds
        ax.set_yticks(range(len(rows)), [f"{value:g}" for value in rows])
        ax.set_xticks(range(len(columns)), [f"{value:g}" for value in columns])
        ax.set_ylabel(self.labels[0])
        ax.set_xlabel(self.labels[1])
        colorbar = ax.figure.colorbar(image, ax=ax)
        colorbar.set_label("log10 p-value (white at 0.05)")
        return ax


@dataclass(frozen=True, repr=False, eq=False)
class DistributionTestResult(ResultMixin):
    """Observed events' values against pooled null events' values.

    Attributes
    ----------
    column : str
        The compared column.
    test : {"ks", "rank_sum", "levene"}
        The two-sample test.
    alternative : {"greater", "less", "two-sided"}
        The side, in terms of the values: ``"greater"`` asks whether the
        observed values tend to be larger than the null values.
    statistic : float
        The test statistic: the Kolmogorov-Smirnov distance, the
        Mann-Whitney U of the observed values, or Levene's W.
    p_value : float
        The test's p-value.
    n_observed, n_null : int
        Values compared, after NaN values are dropped.
    n_observed_dropped, n_null_dropped : int
        NaN values dropped.
    median_observed, median_null : float
        Median of each sample.

    See Also
    --------
    distribution_test : Function that produces this result.
    """

    column: str
    test: str
    alternative: str
    statistic: float
    p_value: float
    n_observed: int
    n_null: int
    n_observed_dropped: int
    n_null_dropped: int
    median_observed: float
    median_null: float

    def summary(self) -> dict[str, Any]:
        """Flat dict of every scalar of this result."""
        return {
            item.name: getattr(self, item.name) for item in dataclasses.fields(self)
        }


@dataclass(frozen=True, repr=False, eq=False)
class ShuffledDataset(ResultMixin):
    """Every event of a data set, shuffled once.

    Attributes
    ----------
    events : tuple of DecodingResult
        Each event's shuffled posterior, with the event's times.
    spike_counts : tuple of NDArray[np.int64] or None
        Each event's spike counts, as shuffled (permuted with the rows by
        ``"time_bin_permutation"``, or as decoded by a null that decodes
        again); None when no counts were given. ``"pseudo_event"`` draws
        posterior rows from other events, which carry no counts, so its
        counts are the real event's, unchanged: they give the event's
        active cells and time bins without spikes, but do not decode to the
        drawn rows.
    null : str
        The null that made the data set.

    See Also
    --------
    shuffle_events : Function that produces this result.
    """

    events: tuple[DecodingResult, ...]
    spike_counts: tuple[NDArray[np.int64], ...] | None
    null: str

    def __post_init__(self) -> None:
        """Own read-only copies of the spike counts."""
        object.__setattr__(self, "events", tuple(self.events))
        if self.spike_counts is not None:
            owned = []
            for counts in self.spike_counts:
                array = np.array(counts)
                array.flags.writeable = False
                owned.append(array)
            object.__setattr__(self, "spike_counts", tuple(owned))

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle or deepcopy, keeping the counts read-only."""
        self.__dict__.update(state)
        for counts in self.spike_counts or ():
            counts.flags.writeable = False

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {"null": self.null, "n_events": len(self.events)}


# --------------------------------------------------------------------------
# Per-event values of real and null data sets
# --------------------------------------------------------------------------


def dataset_values(
    env: Environment | Mapping[str, Environment],
    events: Sequence[DecodingResult] | NDArray[np.float64],
    *,
    replay_method: ReplayMethod | RankOrderMethod,
    null: str | None = None,
    n_datasets: int = 1,
    shared: bool = False,
    quantities: Sequence[str] = (),
    null_options: Mapping[str, Mapping[str, Any]] | None = None,
    spike_counts: Sequence[NDArray[np.int64]] | None = None,
    encoding_models: Any = None,
    dt: float | None = None,
    prior: NDArray[np.float64] | None = None,
    template: str | None = None,
    pseudo_event_pool: NDArray[np.float64] | None = None,
    spike_times: Any = None,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Per-event values of a real data set, or of null data sets, by one method.

    A data-set test compares the events of the real data set with those of
    null data sets, so both must be measured the same way: "The above
    results demonstrate the importance of using matching procedures for
    actual and shuffled data" (Silva et al. 2015). With ``null=None`` this
    measures the events as they are; with a null, each of ``n_datasets``
    null data sets shuffles every event once and measures the shuffled
    events with the same method.

    Parameters
    ----------
    env : Environment or Mapping[str, Environment]
        The environment the events were decoded on; for a rank-order method
        with a mapping of rate maps, one environment or a mapping keyed
        like it.
    events : sequence of DecodingResult, or NDArray[np.float64]
        The decoded events, each with ``times``, decoded on one set of rate
        maps. A rank-order method also takes the events as intervals,
        shape (n_events, 2), start and stop in seconds; decoded events give
        the window ``times[0] - dt / 2`` to ``times[-1] + dt / 2``, as
        :func:`~neurospatial.replay.compare_replay_methods` does.
    replay_method : ReplayMethod or RankOrderMethod
        The method whose score and criteria quantities measure each event.
        Its nulls, alpha and shuffles are not used: the null is ``null``.
        A :class:`~neurospatial.replay.ReplayMethod` is applied as
        :func:`~neurospatial.replay.detect_replay` applies it without a
        shuffle test (its segmentation and time-bin rules included); a
        :class:`~neurospatial.replay.RankOrderMethod` gives the rank-order
        correlation and its active-cell quantities.
    null : str or None, default=None
        None measures the events as they are, as data set 0. Otherwise a
        null of :func:`~neurospatial.replay.replay_significance` (for a
        replay method: ``"column_cycle"``, ``"column_cycle_weighted"``,
        ``"time_bin_permutation"``, ``"pseudo_event"``, ``"unit_identity"``,
        ``"spike_train_circular"``, ``"place_field_circular"`` or
        ``"spike_time_uniform"``), or ``"cell_order"`` for a rank-order
        method: the field positions permuted among the event's active
        template cells. The column-cycle nulls shift each row along the
        track, so need a 1-D environment or a linearized track;
        ``"time_bin_permutation"`` and ``"pseudo_event"`` keep the bins as
        numbered and work on any environment, such as an open-field arena.
        Spike-level surrogate data sets (rate-matched
        Poisson, jitter) are made with
        :func:`~neurospatial.replay.null_events` instead; measure them with
        ``null=None`` and set ``dataset`` from their ``copy_index``.
    n_datasets : int, default=1
        Null data sets; each shuffles every event once.
    shared : bool, default=False
        Draw a template null once per data set and use it for every event:
        one permutation of cell identities (``"unit_identity"``, or
        ``"cell_order"``, which then permutes the rows of
        ``encoding_models``, the same permutation for every template), or
        one set of shifted rate maps (``"place_field_circular"``). Silva et
        al.'s (2015) "5,000 shuffled templates" and Grosmark & Buzsáki's
        (2016) "1,000 null templates" read this way. The events of one
        shared data set are scored against the same shuffled template, so
        their values are correlated, and a count over events varies more
        between data sets than with a draw per event; a per-event p-value
        is the same either way. Refused for the other nulls, which act on
        each event.
    quantities : sequence of str, default=()
        Further per-event quantities to report, scalars of
        :class:`~neurospatial.replay.TrajectoryMetrics` measured on the
        rows the method tests, such as ``("max_map_jump",)`` for a method
        without a jump criterion. Replay methods only.
    null_options : Mapping[str, Mapping[str, Any]] or None
        Options of ``null``, as for
        :func:`~neurospatial.replay.replay_significance`, such as
        ``{"column_cycle": {"exclude_zero_shift": True}}`` for Farooq et
        al.'s (2019) space-bin shuffle ("circularly shifted by a random
        amount between 1 and track length minus 1").
    spike_counts : sequence of arrays or None
        Each event's spike counts, (n_time_bins, n_cells): needed for
        criteria on active cells or spikes (such as ``FAROOQ_2019``'s
        ``n_active_cells >= 5``), empty time bins and the nulls that decode
        again.
    encoding_models, dt, prior
        The decoding inputs of the nulls that decode again (see
        :func:`~neurospatial.replay.replay_significance`), and the rate
        maps (a mapping per template) and window of a rank-order method.
    template : str or None
        For a rank-order method with a mapping of rate maps, the template
        to measure; None measures every template, one row each.
    pseudo_event_pool : NDArray[np.float64] or None
        Posterior rows for the ``"pseudo_event"`` and
        ``"column_cycle_weighted"`` nulls; None stacks the events'
        posteriors, as :func:`~neurospatial.replay.compare_replay_methods`
        does.
    spike_times : array, sequence of arrays, or spike group, optional
        The session's spike times, for a rank-order method.
    rng : numpy.random.Generator, int, or None
        Random number generator or seed. Each event draws from its own
        stream, derived from ``rng`` and the event's position, so the
        values of one event do not depend on the others, and a shared draw
        from a stream of its own. ``table.attrs["rng_entropy"]`` reproduces
        the table.

    Returns
    -------
    pandas.DataFrame
        One row per (dataset, event), and per template for a rank-order
        method given several: ``dataset`` (0 with ``null=None``),
        ``event``, ``template`` (rank order with a mapping), ``score``, a
        column per criterion quantity of the method with the raw value it
        is compared on (``max_map_jump`` in environment units, before the
        ``relative_to`` scale or ``absolute``), the ``quantities``,
        ``n_active_cells`` when counts or spikes are given, and
        ``track_length`` on a track. A value is NaN where the method gives
        none: an event with fewer than two time bins to score, and, for a
        method that thresholds its score, every criterion value of an event
        whose score is undefined (as
        :func:`~neurospatial.replay.compare_replay_methods` reports them).

    Raises
    ------
    ValueError
        If ``null`` is not a null this method can draw, ``shared`` is given
        for a null that acts on each event, ``n_datasets`` is not a positive
        integer, a quantity is not a scalar of TrajectoryMetrics, an event is
        not a DecodingResult with times or was decoded on an environment
        whose bins or connectivity differ from ``env``'s, ``spike_counts``
        does not have one entry per event or is missing for a method that
        counts spikes or active cells, the inputs a null needs are missing,
        a column-cycle null or a score along the track is used off a track,
        or decoding the counts does not reproduce an event's posterior.

    See Also
    --------
    dataset_test : Test a data-set statistic of these values.
    distribution_test : Compare these values with a two-sample test.
    shuffle_events : The shuffled events themselves.

    Notes
    -----
    For the column-cycle, weighted column-cycle and time-bin nulls with a
    weighted-correlation score and quantities of the MAP path
    (``max_map_jump``, ``map_path_length``, ``start_to_end_map_distance``,
    ``map_range``, ``mean_peak_probability``), each event is measured once
    for every circular shift of every time bin: the mean and variance of
    the position under each shifted row, and its MAP bin. A shuffle is
    then a gather from these tables, and its weighted correlation and MAP
    jumps follow in a few operations per time bin. These agree with
    measuring each shuffled posterior with ``detect_replay`` to the last
    rounding of the score (1e-12). Measured on a laptop (NumPy 2.3, 72
    bins, 5 to 15 time bins per event), 1,000 events and 5,000 column-cycle
    data sets of |r| and the largest jump took 1.8 s, and the 10 x 10
    significance matrix of :func:`dataset_test` 1.7 s, so Silva et al.'s
    9,818 events take about 20 s each; measuring each shuffle with
    ``detect_replay`` takes about 0.5 ms, some 7 hours at that scale. The
    table then holds 49 million rows, about 2 GB. Other nulls and methods
    measure each shuffle with ``detect_replay``.

    References
    ----------
    Silva, D., Feng, T., & Foster, D. J. (2015). Trajectory events across
    hippocampal place cells require previous experience. *Nature
    Neuroscience*, 18(12), 1772-1779.

    Grosmark, A. D., & Buzsáki, G. (2016). Diversity in neural firing
    dynamics supports both rigid and learned hippocampal sequences.
    *Science*, 351(6280), 1440-1443.

    Farooq, U., Sibille, J., Liu, K., & Dragoi, G. (2019). Strengthened
    temporal coordination within pre-existing sequential cell assemblies
    supports trajectory replay. *Neuron*, 103(4), 719-733.

    Examples
    --------
    Twenty sweeps along a 100 cm track, measured as Silva et al. (2015) did
    and in 200 column-cycle data sets:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import DecodingResult
    >>> from neurospatial.replay import dataset_values
    >>> from neurospatial.replay.methods import SILVA_2015
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
    ... )
    >>> times = np.arange(10) * 0.02
    >>> events = [
    ...     DecodingResult(
    ...         posterior=np.eye(20)[start + np.arange(10)], env=env, times=times
    ...     )
    ...     for start in range(0, 10, 2)
    ...     for _ in range(4)
    ... ]
    >>> real = dataset_values(env, events, replay_method=SILVA_2015)
    >>> real[["dataset", "event", "score", "max_map_jump"]].head(2)
       dataset  event  score  max_map_jump
    0        0      0    1.0           5.0
    1        0      1    1.0           5.0
    >>> null = dataset_values(
    ...     env,
    ...     events,
    ...     replay_method=SILVA_2015,
    ...     null="column_cycle",
    ...     n_datasets=200,
    ...     rng=0,
    ... )
    >>> len(null), int(null["dataset"].max())
    (4000, 199)
    """
    return _dataset_values(
        env,
        events,
        replay_method,
        null=null,
        n_datasets=n_datasets,
        shared=shared,
        quantities=quantities,
        null_options=null_options,
        spike_counts=spike_counts,
        encoding_models=encoding_models,
        dt=dt,
        prior=prior,
        template=template,
        pseudo_event_pool=pseudo_event_pool,
        spike_times=spike_times,
        rng=rng,
        fast=True,
    )


def _dataset_values(
    env: Any,
    events: Any,
    method: ReplayMethod | RankOrderMethod,
    *,
    null: str | None,
    n_datasets: int,
    shared: bool,
    quantities: Sequence[str],
    null_options: Mapping[str, Mapping[str, Any]] | None,
    spike_counts: Sequence[NDArray[np.int64]] | None,
    encoding_models: Any,
    dt: float | None,
    prior: Any,
    template: str | None,
    pseudo_event_pool: NDArray[np.float64] | None,
    spike_times: Any,
    rng: np.random.Generator | int | None,
    fast: bool,
) -> pd.DataFrame:
    """:func:`dataset_values`, with the moment-table fast path optional."""
    operation = "dataset_values()"
    _positive_integer(n_datasets, "n_datasets", operation)
    if null is None and (n_datasets != 1 or shared):
        raise ValueError(
            f"{operation}: n_datasets and shared apply to null data sets; with "
            "null=None the events are measured once, as data set 0."
        )
    if shared and null not in _SHAREABLE_NULLS:
        raise ValueError(
            _format_error(
                f"{operation}: shared=True cannot be used with null={null!r}.",
                why="only a template null (a permutation of cell identities or "
                "shifted rate maps) has one draw that every event can share; "
                "the other nulls shuffle each event's own posterior or spikes",
                fix=f"use one of {list(_SHAREABLE_NULLS)}, or shared=False",
            )
        )
    entropy = _entropy(rng)
    if isinstance(method, RankOrderMethod):
        table = _rank_order_values(
            env,
            events,
            method,
            null=null,
            n_datasets=n_datasets,
            shared=shared,
            quantities=quantities,
            null_options=null_options,
            encoding_models=encoding_models,
            dt=dt,
            template=template,
            spike_times=spike_times,
            entropy=entropy,
            operation=operation,
        )
    elif isinstance(method, ReplayMethod):
        table = _replay_values(
            env,
            events,
            method,
            null=null,
            n_datasets=n_datasets,
            shared=shared,
            quantities=quantities,
            null_options=null_options,
            spike_counts=spike_counts,
            encoding_models=encoding_models,
            dt=dt,
            prior=prior,
            template=template,
            pseudo_event_pool=pseudo_event_pool,
            entropy=entropy,
            fast=fast,
            operation=operation,
        )
    else:
        raise TypeError(
            _format_error(
                f"{operation}: method must be a ReplayMethod or a "
                f"RankOrderMethod, got {type(method).__name__}.",
                why="the values are a method's score and criteria quantities",
                fix="pass a decoded method such as SILVA_2015 or a rank-order "
                "method such as DIBA_BUZSAKI_2007",
            )
        )
    table.attrs["rng_entropy"] = entropy
    table.attrs["null"] = null
    table.attrs["method"] = method.name
    return table


# Spawn keys of the random streams: per event, shared draws, real events.
_EVENT_KEY = 0
_SHARED_KEY = 1
_REAL_KEY = 2


def _replay_values(
    env: Any,
    events: Any,
    method: ReplayMethod,
    *,
    null: str | None,
    n_datasets: int,
    shared: bool,
    quantities: Sequence[str],
    null_options: Mapping[str, Mapping[str, Any]] | None,
    spike_counts: Sequence[NDArray[np.int64]] | None,
    encoding_models: Any,
    dt: float | None,
    prior: Any,
    template: str | None,
    pseudo_event_pool: NDArray[np.float64] | None,
    entropy: int,
    fast: bool,
    operation: str,
) -> pd.DataFrame:
    """The table of :func:`dataset_values` for a decoded method."""
    import pandas as pd

    from neurospatial.replay.methods import _criterion_quantities

    if null is not None and null not in _EVENT_NULLS:
        raise ValueError(
            f"{operation}: null must be None or one of {list(_EVENT_NULLS)} for a "
            f"replay method, got {null!r}; 'cell_order' is a rank-order null."
        )
    if template is not None:
        raise ValueError(
            f"{operation}: template applies to a rank-order method with several "
            "rate maps; a replay method measures each event's posterior as "
            "decoded."
        )
    if isinstance(env, Mapping):
        raise ValueError(
            f"{operation}: a replay method takes the one environment the events "
            "were decoded on, not a mapping."
        )
    decoded = _decoded_events(env, events, operation)
    counts = _event_counts(spike_counts, decoded, operation)
    scalars = [
        name
        for name in _criterion_quantities()
        if name not in ("score", "n_active_cells", "active_fraction", "n_spikes")
        and not name.startswith("segment")
        and name not in ("template_fraction",)
    ]
    unknown = [name for name in quantities if name not in scalars]
    if unknown:
        raise ValueError(
            f"{operation}: quantities {unknown} are not scalars of "
            f"TrajectoryMetrics; choose from {scalars}."
        )
    variant, labels = _measuring_method(method, quantities)
    if counts is None and _counts_spikes(variant):
        raise ValueError(
            _format_error(
                f"{operation}: method {method.name!r} needs spike_counts.",
                why="it drops or zeroes time bins without spikes, counts the "
                "event's spikes, or counts the cells active in the event or in "
                "each time bin",
                fix="pass spike_counts, one (n_time_bins, n_cells) array per "
                "event, such as the counts decode_intervals returns",
            )
        )
    options_of = _null_options_of(null, null_options, operation)
    track_length = _track_length_or_nan(env)
    columns = list(dict.fromkeys(labels.values()))
    if counts is not None and "n_active_cells" not in columns:
        columns.append("n_active_cells")
    rows: dict[str, list[NDArray[Any]]] = {
        name: [] for name in ("dataset", "event", "score", *columns)
    }

    def add(index: int, datasets: NDArray[np.intp], values: dict[str, Any]) -> None:
        rows["dataset"].append(np.asarray(datasets, dtype=np.intp))
        rows["event"].append(np.full(len(datasets), index, dtype=np.intp))
        for name in ("score", *columns):
            rows[name].append(
                np.broadcast_to(
                    np.asarray(values.get(name, np.nan), dtype=np.float64),
                    (len(datasets),),
                )
            )

    if null is None:
        for index, event in enumerate(decoded):
            event_counts = None if counts is None else counts[index]
            measured = _measure(
                env,
                np.asarray(event.posterior),
                np.asarray(event.times),
                event_counts,
                variant,
                labels,
                _stream(entropy, _REAL_KEY, index),
            )
            add(index, np.zeros(1, dtype=np.intp), measured)
    else:
        context = _NullContext.build(
            env,
            decoded,
            counts,
            null=null,
            options=options_of,
            n_datasets=n_datasets,
            shared=shared,
            encoding_models=encoding_models,
            dt=dt,
            prior=prior,
            pseudo_event_pool=pseudo_event_pool,
            entropy=entropy,
            operation=operation,
        )
        use_tables = fast and _fast_path_applies(variant, null, labels)
        datasets = np.arange(n_datasets, dtype=np.intp)
        for index, event in enumerate(decoded):
            event_counts = None if counts is None else counts[index]
            posterior = np.asarray(event.posterior, dtype=np.float64)
            times = np.asarray(event.times, dtype=np.float64)
            if use_tables and np.isfinite(posterior).all():
                draws = context.rolls(index, posterior)
                values = _MomentTables(context, posterior, times, event_counts).values(
                    draws, variant, labels
                )
                add(index, datasets, values)
                continue
            measured_rows: dict[str, list[float]] = {
                name: [] for name in ("score", *columns)
            }
            stream = _stream(entropy, _EVENT_KEY, index, 1)
            for shuffled, shuffled_counts in context.shuffles(
                index, posterior, times, event_counts
            ):
                measured = _measure(
                    env, shuffled, times, shuffled_counts, variant, labels, stream
                )
                for name in measured_rows:
                    measured_rows[name].append(measured.get(name, np.nan))
            add(
                index,
                datasets,
                {name: np.asarray(values) for name, values in measured_rows.items()},
            )
    table = pd.DataFrame(
        {
            name: np.concatenate(parts) if parts else np.empty(0)
            for name, parts in rows.items()
        }
    )
    if np.isfinite(track_length):
        table["track_length"] = track_length
    return table


def _measuring_method(
    method: ReplayMethod, quantities: Sequence[str]
) -> tuple[ReplayMethod, dict[str, str]]:
    """The method without its shuffle test, and each criterion's quantity.

    The extra ``quantities`` become criteria every value meets, so that
    :func:`detect_replay` measures them on the method's rows. Returns the
    method and a mapping from criterion label to the quantity's column
    (``"score"`` criteria are reported as the score).
    """
    criteria = list(method.criteria)
    present = {criterion.quantity for criterion in criteria}
    for name in quantities:
        if name not in present:
            criteria.append(ReplayCriterion(name, "<=", float(np.finfo(float).max)))
            present.add(name)
    labels = {
        criterion.label: criterion.quantity
        for criterion in criteria
        if criterion.quantity != "score"
    }
    if not criteria and method.segmentation is None:
        # A criterion every event meets, so the method tests something.
        criteria.append(ReplayCriterion("n_time_bins", ">=", 0))
    variant = dataclasses.replace(
        method,
        nulls=(),
        null_options={},
        tested_quantities={},
        template_choice=None
        if method.template_choice in ("smallest_p", "max_score")
        else method.template_choice,
        criteria=tuple(criteria),
    )
    # Keep the first label of each quantity.
    first: dict[str, str] = {}
    for label, quantity in labels.items():
        first.setdefault(quantity, label)
    return variant, {label: quantity for quantity, label in first.items()}


def _counts_spikes(method: ReplayMethod) -> bool:
    """Whether ``detect_replay`` needs spike counts to apply ``method``."""
    segmentation = method.segmentation
    return (
        method.empty_time_bins != "keep"
        or any(
            criterion.quantity in ("n_active_cells", "active_fraction", "n_spikes")
            for criterion in method.criteria
        )
        or (
            segmentation is not None
            and segmentation.min_active_cells_per_time_bin is not None
        )
    )


def _measure(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    counts: NDArray[np.int64] | None,
    variant: ReplayMethod,
    labels: Mapping[str, str],
    rng: np.random.Generator,
) -> dict[str, float]:
    """One event's score and quantities, as compare_replay_methods measures it."""
    from neurospatial.replay.methods import detect_replay

    values: dict[str, float] = {}
    if counts is not None:
        values["n_active_cells"] = float(np.sum(np.asarray(counts).sum(axis=0) > 0))
    usable = times
    if variant.empty_time_bins != "keep" and counts is not None:
        usable = times[np.asarray(counts).sum(axis=1) > 0]
    if np.unique(usable).size < 2:
        values.pop("n_active_cells", None)
        return values
    result = detect_replay(
        env,
        posterior,
        times,
        replay_method=variant,
        spike_counts=counts,
        rng=rng,
    )
    values["score"] = float(result.score)
    for label, quantity in labels.items():
        values[quantity] = float(result.values[label])
    return values


def _track_length_or_nan(env: Environment) -> float:
    from neurospatial.environment import _track_extents

    extents = _track_extents(env)
    if extents is None:
        return np.nan
    starts, stops = extents
    return float(np.sum(np.asarray(stops) - np.asarray(starts)))


def _decoded_events(
    env: Environment | Mapping[str, Environment], events: Any, operation: str
) -> list[DecodingResult]:
    """The events, checked to be decodes on ``env``, one set of rate maps, with times.

    A mapping of environments (a rank-order method's templates) is not
    compared, as :func:`~neurospatial.replay.compare_replay_methods` does
    not compare it.
    """
    from neurospatial.decoding import DecodingResult
    from neurospatial.replay.methods import _environment_mismatch

    decoded = list(events)
    if not decoded:
        raise ValueError(f"{operation}: events is empty; pass at least one event.")
    for index, event in enumerate(decoded):
        if not isinstance(event, DecodingResult):
            raise ValueError(
                _format_error(
                    f"{operation}: event {index} is a {type(event).__name__}, "
                    "not a DecodingResult.",
                    why="a replay method's values are measured on one "
                    "posterior per event; joint decodes and separately decoded "
                    "templates choose among templates per event",
                    fix="pass each event decoded on one set of rate maps, e.g. "
                    "decode_intervals(env, spike_times, intervals, rates, dt=dt)",
                )
            )
        if event.times is None:
            raise ValueError(
                f"{operation}: event {index} has no times; decode with times=."
            )
        if not isinstance(env, Mapping):
            mismatch = _environment_mismatch(env, event.env)
            if mismatch is not None:
                raise ValueError(
                    _format_error(
                        f"{operation}: event {index} was decoded on an "
                        f"environment that differs from env in {mismatch}.",
                        why="distances, jumps and positions would be read on "
                        "bins the event was not decoded on",
                        fix="pass the environment the events were decoded on",
                    )
                )
    return decoded


def _event_counts(
    spike_counts: Sequence[NDArray[np.int64]] | None,
    events: Sequence[DecodingResult],
    operation: str,
) -> list[NDArray[np.int64]] | None:
    if spike_counts is None:
        return None
    if len(spike_counts) != len(events):
        raise ValueError(
            f"{operation}: spike_counts has {len(spike_counts)} entries but there "
            f"are {len(events)} events."
        )
    counts = []
    for index, (entry, event) in enumerate(zip(spike_counts, events, strict=True)):
        array = np.asarray(entry)
        if array.ndim != 2 or len(array) != len(event.posterior):
            raise ValueError(
                _format_error(
                    f"{operation}: spike_counts[{index}] has shape {array.shape}, "
                    f"but event {index} has {len(event.posterior)} time bins.",
                    why="each row of counts holds one time bin's spikes per cell",
                    fix="pass (n_time_bins, n_cells) counts for every event",
                )
            )
        counts.append(array)
    return counts


def _null_options_of(
    null: str | None,
    null_options: Mapping[str, Mapping[str, Any]] | None,
    operation: str,
) -> dict[str, Any]:
    """The options of ``null``, with defaults, after refusing unknown ones."""
    from neurospatial.replay.significance import _checked_null_options

    if null is None:
        if null_options:
            raise ValueError(
                f"{operation}: null_options needs a null; with null=None the "
                "events are not shuffled."
            )
        return {}
    checked = _checked_null_options(null_options, (null,), operation)
    return checked.get(null, {})


class _NullContext:
    """What every event's shuffles need: inputs, options and shared draws."""

    def __init__(self) -> None:
        self.env: Any = None
        self.null = ""
        self.options: dict[str, Any] = {}
        self.n_datasets = 0
        self.entropy = 0
        self.order: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self.probabilities: NDArray[np.float64] | None = None
        self.pool: NDArray[np.float64] | None = None
        self.decoder: Any = None
        self.shared_draws: list[Any] | None = None
        self._environment_tables: dict[str, Any] | None = None

    def environment_tables(self) -> dict[str, Any]:
        """Per-environment tables of the fast path, computed once.

        Positions along the track in bin numbering, centered on the middle
        of their range as :func:`weighted_correlation` centers them; the
        geodesic distance between every pair of bins; the track's length;
        and each bin's position along an unbranched track (None on a
        branched one).
        """
        if self._environment_tables is None:
            from neurospatial.environment import _track_extents
            from neurospatial.replay.correlation import _track_bin_centers
            from neurospatial.replay.criteria import _along_track, _bin_distance_lookup

            env = self.env
            positions = _track_bin_centers(env, "dataset_values()")
            extents = _track_extents(env)
            assert extents is not None  # _track_bin_centers refused other envs
            starts, stops = (np.asarray(values) for values in extents)
            between, _ = _bin_distance_lookup(env, np.arange(env.n_bins), "geodesic")
            self._environment_tables = {
                "positions": positions - (positions.min() + positions.max()) / 2.0,
                "distances": between,
                "track_length": float(np.sum(stops - starts)),
                "along": _along_track(env, starts, stops),
            }
        return self._environment_tables

    @classmethod
    def build(
        cls,
        env: Environment,
        events: Sequence[DecodingResult],
        counts: Sequence[NDArray[np.int64]] | None,
        *,
        null: str,
        options: dict[str, Any],
        n_datasets: int,
        shared: bool,
        encoding_models: Any,
        dt: float | None,
        prior: Any,
        pseudo_event_pool: NDArray[np.float64] | None,
        entropy: int,
        operation: str,
    ) -> _NullContext:
        from neurospatial.replay.nulls import (
            _POSTERIOR_SHUFFLE_ALTERNATIVE,
            _peak_probabilities,
            _track_order,
        )
        from neurospatial.replay.significance import (
            _EventDecoder,
            _require_contiguous_time_bins,
            _validated_pool,
        )

        context = cls()
        context.env = env
        context.null = null
        context.options = options
        context.n_datasets = n_datasets
        context.entropy = entropy
        if null in _COLUMN_CYCLE_NULLS:
            context.order = _track_order(
                env,
                np.asarray(events[0].posterior),
                operation,
                _POSTERIOR_SHUFFLE_ALTERNATIVE,
            )
        else:
            # The other nulls (permuting time bins, drawing pool rows) need
            # no order along a track: the bins keep their numbering.
            context.order = np.arange(env.n_bins)
        if null in ("pseudo_event", "column_cycle_weighted"):
            pool = (
                np.vstack([np.asarray(event.posterior) for event in events])
                if pseudo_event_pool is None
                else pseudo_event_pool
            )
            context.pool = _validated_pool(pool, env)
            if null == "column_cycle_weighted":
                context.probabilities = _peak_probabilities(
                    env,
                    context.order,
                    context.pool.mean(axis=0),
                    options.get("edge_exclusion", 0.0),
                    operation,
                )
        if null in _DECODING_NULLS:
            missing = [
                name
                for name, value in (
                    ("spike_counts", counts),
                    ("encoding_models", encoding_models),
                    ("dt", dt),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    _format_error(
                        f"{operation}: the {null} null decodes the events again "
                        f"and needs {missing}.",
                        why="it decodes shuffled spike counts or shifted rate "
                        "maps the way the events were decoded",
                        fix="pass spike_counts, encoding_models and dt (and "
                        "prior if the decode used one)",
                    )
                )
            assert dt is not None and counts is not None
            if null in ("spike_train_circular", "spike_time_uniform"):
                for event in events:
                    _require_contiguous_time_bins(
                        event.times,
                        dt,
                        [null],
                        overlap=null == "spike_time_uniform",
                    )
            context.decoder = _EventDecoder(env, encoding_models, dt, prior, None)
            if shared and options.get("active_only", False):
                raise ValueError(
                    f"{operation}: shared=True permutes every cell once per data "
                    "set, so unit_identity's active_only, which permutes each "
                    "event's active cells, cannot be used with it."
                )
            if shared:
                generator = _stream(entropy, _SHARED_KEY)
                if null == "unit_identity":
                    n_cells = np.asarray(counts[0]).shape[1]
                    context.shared_draws = [
                        generator.permutation(n_cells) for _ in range(n_datasets)
                    ]
                else:
                    context.shared_draws = list(
                        context.decoder.shifted_models(n_datasets, generator)
                    )
        return context

    def rolls(
        self, index: int, posterior: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Each null data set's row order and circular shift of each time bin.

        Returns ``(rows, shifts)``, both (n_datasets, n_time_bins): shuffle
        ``k`` puts row ``rows[k, t]`` of the event at time bin ``t``, shifted
        along the track by ``shifts[k, t]`` bins.
        """
        generator = _stream(self.entropy, _EVENT_KEY, index)
        n_time = len(posterior)
        n_bins = len(self.order)
        shape = (self.n_datasets, n_time)
        identity = np.broadcast_to(np.arange(n_time), shape)
        if self.null == "column_cycle":
            smallest = 1 if self.options.get("exclude_zero_shift", False) else 0
            shifts = generator.integers(smallest, n_bins, size=shape)
            return np.array(identity), np.asarray(shifts, dtype=np.intp)
        if self.null == "column_cycle_weighted":
            assert self.probabilities is not None
            source = np.argmax(posterior[:, self.order], axis=1)
            destination = generator.choice(n_bins, size=shape, p=self.probabilities)
            return np.array(identity), np.asarray(
                (destination - source[None, :]) % n_bins, dtype=np.intp
            )
        order = np.argsort(generator.random(shape), axis=1)
        return np.asarray(order, dtype=np.intp), np.zeros(shape, dtype=np.intp)

    def shuffles(
        self,
        index: int,
        posterior: NDArray[np.float64],
        times: NDArray[np.float64],
        counts: NDArray[np.int64] | None,
    ) -> Iterator[tuple[NDArray[np.float64], NDArray[np.int64] | None]]:
        """Each null data set's shuffle of one event, with its spike counts."""
        from neurospatial.replay.nulls import (
            shuffle_cell_identity,
            shuffle_spike_counts_circular,
            shuffle_spike_counts_uniform,
        )
        from neurospatial.replay.significance import _require_reproducible_decode

        if self.null in _ROLLING_NULLS:
            rows, shifts = self.rolls(index, posterior)
            along = posterior[:, self.order]
            n_bins = len(self.order)
            columns = np.arange(n_bins)
            for k in range(self.n_datasets):
                shuffled = np.array(posterior, dtype=np.float64)
                shuffled[:, self.order] = along[rows[k]][
                    np.arange(len(posterior))[:, None],
                    (columns[None, :] - shifts[k][:, None]) % n_bins,
                ]
                yield (
                    shuffled,
                    None if counts is None else np.asarray(counts)[rows[k]],
                )
            return
        generator = _stream(self.entropy, _EVENT_KEY, index)
        if self.null == "pseudo_event":
            assert self.pool is not None
            draws = generator.integers(
                len(self.pool), size=(self.n_datasets, len(posterior))
            )
            for rows_k in draws:
                yield self.pool[rows_k], counts
            return
        assert counts is not None
        event_counts = np.asarray(counts)
        decoder = self.decoder
        _require_reproducible_decode(decoder.decode(event_counts), posterior)
        if self.null == "unit_identity":
            if self.shared_draws is not None:
                permuted: Iterator[NDArray[np.int64]] = (
                    event_counts[:, permutation] for permutation in self.shared_draws
                )
            else:
                permuted = (
                    shuffled
                    for shuffled, _ in shuffle_cell_identity(
                        event_counts,
                        decoder.first_rates,
                        active_only=self.options.get("active_only", False),
                        n_shuffles=self.n_datasets,
                        rng=generator,
                    )
                )
            for shuffled_counts in permuted:
                yield decoder.decode(shuffled_counts), shuffled_counts
        elif self.null == "place_field_circular":
            models_of = (
                iter(self.shared_draws)
                if self.shared_draws is not None
                else decoder.shifted_models(self.n_datasets, generator)
            )
            for models in models_of:
                yield decoder.decode(event_counts, models), event_counts
        else:
            shuffle = (
                shuffle_spike_counts_circular
                if self.null == "spike_train_circular"
                else shuffle_spike_counts_uniform
            )
            for shuffled_counts in shuffle(
                event_counts, n_shuffles=self.n_datasets, rng=generator
            ):
                yield decoder.decode(shuffled_counts), shuffled_counts


def _fast_path_applies(
    variant: ReplayMethod, null: str, labels: Mapping[str, str]
) -> bool:
    """Whether every value of these shuffles follows from moment tables."""
    options = dict(variant.score_options)
    return (
        null in _ROLLING_NULLS
        and variant.score is weighted_correlation
        and set(options) <= {"row_normalization"}
        and options.get("row_normalization", "per_time_bin") == "per_time_bin"
        and variant.segmentation is None
        and variant.empty_time_bins == "keep"
        and all(quantity in _FAST_QUANTITIES for quantity in labels.values())
    )


class _MomentTables:
    """One event measured under every circular shift of every time bin.

    For time bin ``t`` shifted by ``s`` bins along the track: the row's
    mass, the mean and variance of position under it (positions centered
    as :func:`weighted_correlation` centers them), its MAP bin and the mass
    in that bin. A shuffle that puts row ``rows[t]``, shifted by
    ``shifts[t]``, at time bin ``t`` is then measured from these tables.
    The pooled variance of a shuffle adds the rows' own variances and the
    spread of their means, both sums of non-negative terms, so no
    difference of large numbers loses precision.
    """

    def __init__(
        self,
        context: _NullContext,
        posterior: NDArray[np.float64],
        times: NDArray[np.float64],
        counts: NDArray[np.int64] | None,
    ) -> None:
        from neurospatial.decoding import map_bin

        env = context.env
        tables = context.environment_tables()
        order = context.order
        n_bins = len(order)
        self.tables = tables
        self.times = times
        self.counts = counts
        # Only the shifts that can be drawn are tabulated.
        self.shifts = (
            np.zeros(1, dtype=np.intp)
            if context.null == "time_bin_permutation"
            else np.arange(n_bins)
        )
        along = posterior[:, order]  # (n_time, n_bins), in track order
        mass = along.sum(axis=1)
        self.mass = mass
        index = (np.arange(n_bins)[None, :] - self.shifts[:, None]) % n_bins
        rolled = along[:, index]  # (n_time, n_shifts, n_bins)
        positions = tables["positions"][order]
        with np.errstate(invalid="ignore", divide="ignore"):
            weights = np.where(
                mass[:, None, None] > 0, rolled / mass[:, None, None], 0.0
            )
        mean = weights @ positions
        self.mean = mean
        self.variance = np.einsum(
            "tsb,tsb->ts", weights, (positions[None, None, :] - mean[..., None]) ** 2
        )
        # The MAP bin of each shifted row, as trajectory_metrics finds it: on
        # the renormalized row, in bin numbering.
        n_time, n_shifts, _ = rolled.shape
        in_bins = np.empty((n_time * n_shifts, env.n_bins))
        in_bins[:, order] = rolled.reshape(n_time * n_shifts, n_bins)
        maps = np.asarray(
            map_bin(env, in_bins / in_bins.sum(axis=1, keepdims=True))
        ).reshape(n_time, n_shifts)
        self.maps = maps
        self.peaks = in_bins[np.arange(n_time * n_shifts), maps.ravel()].reshape(
            n_time, n_shifts
        )

    def values(
        self,
        draws: tuple[NDArray[np.intp], NDArray[np.intp]],
        variant: ReplayMethod,
        labels: Mapping[str, str],
    ) -> dict[str, NDArray[np.float64]]:
        """The score and quantities of each null data set's shuffle of the event.

        The same values :func:`_measure` gives for the shuffled posterior,
        including NaN criterion values where a thresholded score is
        undefined.
        """
        from neurospatial.replay.trajectory import _SCORE_DECIMALS

        rows, shifts = draws
        n_datasets, n_time = rows.shape
        if len(self.shifts) == 1:
            shifts = np.zeros_like(shifts)
        needed = list(dict.fromkeys(labels.values()))
        constant: dict[str, float] = {}
        if self.counts is not None:
            counts = np.asarray(self.counts)
            active = float(np.sum(counts.sum(axis=0) > 0))
            constant["n_active_cells"] = active
            constant["active_fraction"] = active / max(counts.shape[1], 1)
            constant["n_spikes"] = float(counts.sum())
        if np.unique(self.times).size < 2:
            return {}
        constant["n_time_bins"] = float(n_time)
        constant["duration"] = float(self.times[-1] - self.times[0])
        constant["track_length"] = self.tables["track_length"]

        # The weighted correlation, as _weighted_correlations computes it on
        # rows normalized per time bin: every row with mass weighs 1.
        times = self.times - (self.times.min() + self.times.max()) / 2.0
        mass = (self.mass[rows] > 0).astype(np.float64)  # (n_datasets, n_time)
        mean = self.mean[rows, shifts]
        variance = self.variance[rows, shifts]
        total = mass.sum(axis=1)
        time_mass = mass / total[:, None]
        time_dev = times[None, :] - (time_mass @ times)[:, None]
        mean_position = np.sum(mass * mean, axis=1) / total
        deviation = mean - mean_position[:, None]
        covariance = np.sum(time_dev * mass * deviation, axis=1) / total
        var_time = np.sum(time_mass * time_dev**2, axis=1)
        var_position = np.sum(mass * (variance + deviation**2), axis=1) / total
        eps = np.finfo(np.float64).eps
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = covariance / np.sqrt(var_time * var_position)
        varies = (var_time > eps * float(np.ptp(times)) ** 2) & (
            var_position > eps * float(np.ptp(self.tables["positions"])) ** 2
        )
        score = np.where(
            varies,
            np.clip(np.round(correlation, _SCORE_DECIMALS), -1.0, 1.0),
            np.nan,
        )
        values: dict[str, NDArray[np.float64]] = {"score": score}
        if "n_active_cells" in constant:
            values["n_active_cells"] = np.full(n_datasets, constant["n_active_cells"])

        maps = self.maps[rows, shifts]  # (n_datasets, n_time)
        distances = self.tables["distances"]
        jumps = distances[maps[:, :-1], maps[:, 1:]]
        for quantity in needed:
            if quantity in constant:
                values[quantity] = np.full(n_datasets, constant[quantity])
            elif quantity == "max_map_jump":
                values[quantity] = jumps.max(axis=1)
            elif quantity == "map_path_length":
                values[quantity] = jumps.sum(axis=1)
            elif quantity == "start_to_end_map_distance":
                values[quantity] = distances[maps[:, 0], maps[:, -1]]
            elif quantity == "mean_peak_probability":
                values[quantity] = self.peaks[rows, shifts].mean(axis=1)
            else:  # map_range
                along = self.tables["along"]
                values[quantity] = (
                    np.full(n_datasets, np.nan)
                    if along is None
                    else np.ptp(along[maps], axis=1)
                )
        if any(c.quantity == "score" for c in variant.criteria):
            # An undefined score gives no criterion values (detect_replay).
            undefined = ~np.isfinite(score)
            for quantity in needed:
                values[quantity] = np.where(undefined, np.nan, values[quantity])
        return values


# --------------------------------------------------------------------------
# Rank-order values
# --------------------------------------------------------------------------


def _rank_order_values(
    env: Any,
    events: Any,
    method: RankOrderMethod,
    *,
    null: str | None,
    n_datasets: int,
    shared: bool,
    quantities: Sequence[str],
    null_options: Mapping[str, Mapping[str, Any]] | None,
    encoding_models: Any,
    dt: float | None,
    template: str | None,
    spike_times: Any,
    entropy: int,
    operation: str,
) -> pd.DataFrame:
    """The table of :func:`dataset_values` for a rank-order method."""
    import pandas as pd

    from neurospatial._spikes import as_spike_trains
    from neurospatial.replay.rank_order import (
        _event_pairs,
        _field_positions,
        _result,
        _supported_rates,
    )

    if null not in (None, "cell_order"):
        raise ValueError(
            f"{operation}: a rank-order method's null is 'cell_order' (or None), "
            f"got {null!r}."
        )
    if null_options:
        raise ValueError(f"{operation}: the cell_order null takes no options.")
    if quantities:
        raise ValueError(
            f"{operation}: quantities are TrajectoryMetrics of a decoded event; a "
            "rank-order method measures spikes and reports its score and active "
            "cells."
        )
    if method.test == "pooled_ssi":
        raise ValueError(
            f"{operation}: method {method.name!r} scores the SSI against an epoch's "
            "pooled shuffles; use a rank-order method with a correlation test."
        )
    if spike_times is None or encoding_models is None:
        raise ValueError(
            _format_error(
                f"{operation}: rank-order method {method.name!r} needs spike_times "
                "and encoding_models.",
                why="it correlates each cell's firing time in the event with its "
                "field position",
                fix="pass the session's spike_times and the rate maps",
            )
        )
    windows = _event_windows(env, events, dt, operation)
    if isinstance(encoding_models, Mapping):
        names: list[str | None] = list(encoding_models)
        if template is not None:
            if template not in encoding_models:
                raise ValueError(
                    f"{operation}: template {template!r} is not one of "
                    f"{list(encoding_models)}."
                )
            names = [template]
    else:
        if template is not None:
            raise ValueError(
                f"{operation}: template={template!r} needs a mapping of rate maps."
            )
        names = [None]
    trains = as_spike_trains(spike_times)
    templates = []
    for name in names:
        template_env = env[name] if isinstance(env, Mapping) else env
        rates = encoding_models if name is None else encoding_models[name]
        _field_positions(template_env, rates, method.field_order, operation)
        masked = _supported_rates(rates)
        if method.min_peak_rate is not None:
            peak = np.where(np.isfinite(masked), masked, 0.0).max(axis=1)
            masked = np.where((peak >= method.min_peak_rate)[:, None], masked, np.nan)
        positions = _field_positions(
            template_env, masked, method.field_order, operation
        )
        templates.append((name, template_env, masked, positions))
        if len(positions) != len(trains):
            raise ValueError(
                f"{operation}: spike_times has {len(trains)} cells but the rate "
                f"maps have {len(positions)}."
            )
    permutations = None
    if shared:
        generator = _stream(entropy, _SHARED_KEY)
        permutations = np.argsort(generator.random((n_datasets, len(trains))), axis=1)
    rows: dict[str, list[Any]] = {
        "dataset": [],
        "event": [],
        "template": [],
        "score": [],
        "n_active_cells": [],
        "active_fraction": [],
    }
    for index, (start, stop) in enumerate(windows):
        in_event = [train[(train >= start) & (train < stop)] for train in trains]
        for slot, (name, template_env, masked, positions) in enumerate(templates):
            n_template = max(int(np.isfinite(positions).sum()), 1)
            if null is None:
                pairs = _event_pairs(
                    template_env,
                    trains,
                    (start, stop),
                    masked,
                    spike_statistic=method.spike_statistic,
                    field_order=method.field_order,
                    min_active_cells=method.min_active_cells,
                    operation=operation,
                )
                observed = _result(pairs, method.spike_statistic, method.field_order)
                scores = np.array([observed.score])
                active = np.array([observed.n_active_cells])
                datasets = np.zeros(1, dtype=np.intp)
            else:
                datasets = np.arange(n_datasets, dtype=np.intp)
                scores, active = _permuted_rank_scores(
                    in_event,
                    positions,
                    method,
                    permutations,
                    _stream(entropy, _EVENT_KEY, index, slot),
                    n_datasets,
                )
            rows["dataset"].append(datasets)
            rows["event"].append(np.full(len(datasets), index, dtype=np.intp))
            rows["template"].append(np.full(len(datasets), name, dtype=object))
            rows["score"].append(np.asarray(scores, dtype=np.float64))
            rows["n_active_cells"].append(np.asarray(active, dtype=np.float64))
            rows["active_fraction"].append(
                np.asarray(active, dtype=np.float64) / n_template
            )
    table = pd.DataFrame({name: np.concatenate(parts) for name, parts in rows.items()})
    if names == [None]:
        table = table.drop(columns="template")
    if not any(c.quantity == "active_fraction" for c in method.criteria):
        table = table.drop(columns="active_fraction")
    return table


def _event_windows(
    env: Environment | Mapping[str, Environment],
    events: Any,
    dt: float | None,
    operation: str,
) -> list[tuple[float, float]]:
    """Each event's (start, stop): given as intervals, or a decode's window."""
    if isinstance(events, np.ndarray) or (
        isinstance(events, Sequence)
        and len(events)
        and not hasattr(events[0], "posterior")
    ):
        from neurospatial.replay.calibration import _checked_intervals

        windows = _checked_intervals(events, operation)
        return [(float(a), float(b)) for a, b in windows]
    decoded = _decoded_events(env, events, operation)
    if dt is None:
        raise ValueError(
            _format_error(
                f"{operation}: a rank-order method scores spikes over each "
                "event's decoding windows, which need dt.",
                why="the window runs from times[0] - dt / 2 to times[-1] + dt / 2",
                fix="pass dt, or the events as (n_events, 2) intervals",
            )
        )
    return [
        (float(times[0]) - dt / 2, float(times[-1]) + dt / 2)
        for times in (np.asarray(event.times) for event in decoded)
    ]


def _permuted_rank_scores(
    in_event: list[NDArray[np.float64]],
    positions: NDArray[np.float64],
    method: RankOrderMethod,
    permutations: NDArray[np.intp] | None,
    generator: np.random.Generator,
    n_datasets: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rank-order correlations of one event under each null data set's permutation.

    Without ``permutations``, each data set permutes the field positions
    among the event's active template cells, as
    :func:`~neurospatial.replay.rank_order_significance` does. With them,
    data set ``k`` gives cell ``c`` the field position of cell
    ``permutations[k, c]``, so which spiking cells have a field, and so the
    active cells, can change.

    Returns the correlations and the active cells, each (n_datasets,).
    """
    spiking = np.array(
        [cell for cell, spikes in enumerate(in_event) if spikes.size], dtype=np.intp
    )
    if permutations is None:
        # Only the active template cells take part; their number is fixed.
        spiking = spiking[np.isfinite(positions[spiking])]
    if method.spike_statistic == "all_spikes":
        times = (
            np.concatenate([in_event[cell] for cell in spiking])
            if spiking.size
            else np.empty(0)
        )
        cell_of = np.repeat(
            np.arange(spiking.size), [in_event[cell].size for cell in spiking]
        )
    else:
        summaries: dict[str, Callable[..., Any]] = {
            "first": np.min,
            "median": np.median,
            "mean": np.mean,
        }
        summarize = summaries[method.spike_statistic]
        times = np.array([summarize(in_event[cell]) for cell in spiking], dtype=float)
        cell_of = np.arange(spiking.size)
    if permutations is None:
        order = np.argsort(generator.random((n_datasets, spiking.size)), axis=1)
        fields = positions[spiking][order]
    else:
        fields = positions[permutations[:, spiking]]
    scores = np.full(n_datasets, np.nan)
    finite = np.isfinite(fields)
    n_active = finite.sum(axis=1).astype(np.float64)
    enough = n_active >= method.min_active_cells
    if finite.all():
        if enough.any():
            scores[enough] = _spearman_rows(fields[enough][:, cell_of], times)
        return scores, n_active
    for k in np.flatnonzero(enough):
        keep = finite[k][cell_of]
        scores[k] = _spearman_rows(fields[k][cell_of][keep][None, :], times[keep])[0]
    return scores, n_active


def _spearman_rows(
    x: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Spearman correlation of each row of ``x`` with ``values``.

    Average ranks for ties; NaN where a row, or ``values``, does not vary.
    """
    from scipy import stats

    v_ranks = stats.rankdata(values)
    v_dev = v_ranks - v_ranks.mean()
    x_ranks = stats.rankdata(x, axis=1)
    x_dev = x_ranks - x_ranks.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (x_dev @ v_dev) / np.sqrt((x_dev**2).sum(axis=1) * (v_dev**2).sum())
    varies = (np.ptp(x, axis=1) > 0) & (np.ptp(values) > 0)
    return np.where(varies, rho, np.nan)


# --------------------------------------------------------------------------
# Shuffled events as data
# --------------------------------------------------------------------------


def shuffle_events(
    env: Environment,
    events: Sequence[DecodingResult],
    *,
    null: str,
    n_datasets: int = 1,
    null_options: Mapping[str, Mapping[str, Any]] | None = None,
    spike_counts: Sequence[NDArray[np.int64]] | None = None,
    encoding_models: Any = None,
    dt: float | None = None,
    prior: NDArray[np.float64] | None = None,
    pseudo_event_pool: NDArray[np.float64] | None = None,
    rng: np.random.Generator | int | None = None,
) -> list[ShuffledDataset]:
    """Null data sets as decoded events, for a control analysed like real data.

    Some control data sets are analysed with per-event shuffles of their
    own: Silva et al. (2015) ran their significance matrix on a "time
    swap" data set, and Grosmark & Buzsáki (2016, Fig. S8) tested every
    event of 100 time-bin-shuffled data sets against 200 place-field
    shuffles. This returns the shuffled events, which can be passed to
    :func:`dataset_values` or
    :func:`~neurospatial.replay.compare_replay_methods` like real events.

    Parameters
    ----------
    env : Environment
        The environment the events were decoded on.
    events : sequence of DecodingResult
        The decoded events, each with ``times``.
    null : str
        A null of :func:`~neurospatial.replay.replay_significance` that
        shuffles each event: ``"column_cycle"``,
        ``"column_cycle_weighted"``, ``"time_bin_permutation"``,
        ``"pseudo_event"``, ``"unit_identity"``,
        ``"spike_train_circular"``, ``"place_field_circular"`` or
        ``"spike_time_uniform"``. ``"time_bin_permutation"`` permutes each
        event's posterior rows and its spike-count rows together, so a
        null that decodes again afterwards decodes the permuted counts
        (Grosmark & Buzsáki's time-bin-shuffle surrogate).
        ``"pseudo_event"`` keeps each event's counts unchanged next to
        posterior rows drawn from the pool, so the counts no longer decode
        to the rows; ``dataset_values`` uses them the same way, for the
        event's active cells and time bins without spikes. Only the
        column-cycle nulls need a 1-D environment or a linearized track.
    n_datasets : int, default=1
        Null data sets.
    null_options, spike_counts, encoding_models, dt, prior, pseudo_event_pool
        As for :func:`dataset_values`.
    rng : numpy.random.Generator, int, or None
        Random number generator or seed; each event draws from its own
        stream, as in :func:`dataset_values`, so ``dataset_values(...,
        null=null, rng=rng)`` measures these same shuffles.

    Returns
    -------
    list of ShuffledDataset
        ``n_datasets`` data sets, each with every event shuffled once.

    Raises
    ------
    ValueError
        As :func:`dataset_values`.

    See Also
    --------
    dataset_values : Measure the events of real and null data sets.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import DecodingResult
    >>> from neurospatial.replay import shuffle_events
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> event = DecodingResult(
    ...     posterior=np.eye(10)[:5], env=env, times=np.arange(5) * 0.02
    ... )
    >>> (swapped,) = shuffle_events(env, [event], null="time_bin_permutation", rng=0)
    >>> sorted(swapped.events[0].map_bin.tolist())
    [0, 1, 2, 3, 4]
    """
    operation = "shuffle_events()"
    _positive_integer(n_datasets, "n_datasets", operation)
    if null not in _EVENT_NULLS:
        raise ValueError(
            f"{operation}: null must be one of {list(_EVENT_NULLS)}, got {null!r}."
        )
    from neurospatial.decoding import DecodingResult

    decoded = _decoded_events(env, events, operation)
    counts = _event_counts(spike_counts, decoded, operation)
    options = _null_options_of(null, null_options, operation)
    entropy = _entropy(rng)
    context = _NullContext.build(
        env,
        decoded,
        counts,
        null=null,
        options=options,
        n_datasets=n_datasets,
        shared=False,
        encoding_models=encoding_models,
        dt=dt,
        prior=prior,
        pseudo_event_pool=pseudo_event_pool,
        entropy=entropy,
        operation=operation,
    )
    posteriors: list[list[DecodingResult]] = [[] for _ in range(n_datasets)]
    shuffled_counts: list[list[NDArray[np.int64]]] = [[] for _ in range(n_datasets)]
    for index, event in enumerate(decoded):
        event_counts = None if counts is None else counts[index]
        for k, (posterior, new_counts) in enumerate(
            context.shuffles(
                index,
                np.asarray(event.posterior, dtype=np.float64),
                np.asarray(event.times),
                event_counts,
            )
        ):
            posteriors[k].append(
                DecodingResult(posterior=posterior, env=env, times=event.times)
            )
            if new_counts is not None:
                shuffled_counts[k].append(np.asarray(new_counts))
    return [
        ShuffledDataset(
            events=tuple(posteriors[k]),
            spike_counts=None if counts is None else tuple(shuffled_counts[k]),
            null=null,
        )
        for k in range(n_datasets)
    ]


# --------------------------------------------------------------------------
# Tests of a data set
# --------------------------------------------------------------------------


def dataset_test(
    observed: pd.DataFrame,
    null_table: pd.DataFrame,
    *,
    criteria: Sequence[CriterionGrid] | None = None,
    statistic: Callable[[pd.DataFrame], Any] | None = None,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
) -> DatasetTestResult:
    """Test a data-set statistic against null data sets (a Monte Carlo test).

    The statistic is computed on the observed data set and on each null
    data set; the p-value is the share of null data sets at least as
    extreme, ``(1 + k) / (1 + n_datasets)``.

    Parameters
    ----------
    observed : pandas.DataFrame
        One row per event of the observed data set, such as from
        :func:`dataset_values` with ``null=None``. A ``dataset`` column, if
        present, must hold one value.
    null_table : pandas.DataFrame
        One row per event of each null data set, with a ``dataset`` column
        naming the data set, such as from :func:`dataset_values` with a
        null, or the measured events of
        :func:`~neurospatial.replay.null_events` with ``dataset`` set from
        their ``copy_index``.
    criteria : sequence of CriterionGrid or None
        The statistic is the share of events passing every criterion, at
        each point of the product of the criteria's thresholds: a matrix
        for two criteria, such as Silva et al.'s (2015) significance
        matrix. With as many events in every data set (true of every null
        here), the share's test is the count's test.
    statistic : callable or None
        Instead of ``criteria``, any statistic of one data set's table,
        such as ``lambda table: table["score"].abs().mean()``; it may
        return an array, tested elementwise.
    alternative : {"greater", "less", "two-sided"}, default="greater"
        ``"greater"`` asks whether the observed statistic is larger than
        the null data sets'; ``"two-sided"`` doubles the smaller side.

    Returns
    -------
    DatasetTestResult
        The statistic, the null data sets' statistics, the p-value and
        percentile of each grid point, and the grid.

    Raises
    ------
    ValueError
        If both or neither of ``criteria`` and ``statistic`` are given; if
        ``null`` lacks a ``dataset`` column, has rows without one, or
        ``observed`` holds several data sets; if a table lacks a criterion's
        column, or a data set has no events; if ``statistic`` is not finite
        on the observed data set or on a null data set; or if
        ``alternative`` is invalid.

    See Also
    --------
    dataset_values : The per-event values.
    DatasetTestRecipe.paper_p_value : A paper's own p-value rule.

    Notes
    -----
    Silva et al. (2015): "The Monte Carlo P-value was calculated as (n + 1)/
    (r + 1), where r is the total number of shuffles and n is the number of
    shuffles that produce greater than or equal to the number of trajectory
    events that meet the criteria compared to the actual data", which is
    ``p_value``; their Supplementary Table 1 reports p = 0.0002 = 1/5,001.
    Farooq et al. (2019) and Liu et al. (2019) used ``1 - percentile /
    100``, which leaves out the ``+ 1``: with 500 data sets it is smaller by
    ``(1 - p) / 501``, at most 0.002.

    The comparison of shares is exact: shares of data sets of different
    sizes are compared as integer counts, so a tie is a tie.

    Examples
    --------
    A data set in which 8 of 10 events pass, against 99 null data sets in
    which a binomial share does:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from neurospatial.replay import CriterionGrid, dataset_test
    >>> rng = np.random.default_rng(0)
    >>> observed = pd.DataFrame({"score": [0.9] * 8 + [0.1] * 2})
    >>> null = pd.DataFrame(
    ...     {
    ...         "dataset": np.repeat(np.arange(99), 10),
    ...         "score": np.where(rng.random(990) < 0.3, 0.9, 0.1),
    ...     }
    ... )
    >>> result = dataset_test(
    ...     observed, null, criteria=[CriterionGrid("score", ">", 0.5)]
    ... )
    >>> print(result.statistic, result.p_value)
    0.8 0.01
    """
    operation = "dataset_test()"
    if (criteria is None) == (statistic is None):
        raise ValueError(
            _format_error(
                f"{operation}: pass exactly one of criteria and statistic.",
                why="the tested statistic is either the share of events passing "
                "the criteria or the statistic function's value",
                fix="pass criteria=[CriterionGrid(...)] or statistic=lambda table: ...",
            )
        )
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"{operation}: alternative must be one of {_ALTERNATIVES}, got "
            f"{alternative!r}."
        )
    if "dataset" not in null_table:
        raise ValueError(
            _format_error(
                f"{operation}: null_table has no 'dataset' column.",
                why="each null data set's statistic is computed on its own rows",
                fix="pass the table of dataset_values(..., null=...), or set "
                "dataset from a NullSpikeData's copy_index",
            )
        )
    if "dataset" in observed and observed["dataset"].nunique(dropna=False) > 1:
        raise ValueError(
            f"{operation}: observed holds {observed['dataset'].nunique()} data "
            "sets; pass the one observed data set (dataset_values with null=None)."
        )
    if len(observed) == 0:
        raise ValueError(f"{operation}: observed has no events.")
    n_unlabeled = int(null_table["dataset"].isna().sum())
    if n_unlabeled:
        raise ValueError(
            _format_error(
                f"{operation}: {n_unlabeled} rows of null_table have no dataset.",
                why="each row must belong to one null data set; rows without "
                "one cannot be counted in any data set's statistic",
                fix="set dataset on every row (for null_events copies, from "
                "copy_index), or drop the rows that belong to no data set",
            )
        )
    import pandas as pd

    codes, names = pd.factorize(null_table["dataset"], sort=True)
    n_datasets = len(names)
    if n_datasets == 0:
        raise ValueError(f"{operation}: null_table has no data sets.")
    if criteria is not None:
        grids = list(criteria)
        if not grids or not all(isinstance(c, CriterionGrid) for c in grids):
            raise ValueError(
                f"{operation}: criteria must be one or more CriterionGrid, got "
                f"{criteria!r}."
            )
        observed_count = _grid_counts(
            observed, np.zeros(len(observed), dtype=np.intp), 1, grids
        )[0]
        null_counts = _grid_counts(null_table, codes, n_datasets, grids)
        sizes = np.bincount(codes, minlength=n_datasets)
        n_observed = len(observed)
        # Compare shares exactly, as integers: c_k / n_k vs c / n.
        axes = [grid for grid in grids if grid.thresholds.ndim == 1]
        scale = (n_datasets,) + (1,) * len(axes)
        left = null_counts * n_observed
        right = observed_count[None, ...] * sizes.reshape(scale)
        greater = np.sum(left >= right, axis=0)
        less = np.sum(left <= right, axis=0)
        below = np.sum(left < right, axis=0)
        observed_statistic: Any = observed_count / n_observed
        null_statistics = null_counts / sizes.reshape(scale)
        labels = tuple(grid.label for grid in axes)
        thresholds = tuple(grid.thresholds for grid in axes)
    else:
        assert statistic is not None
        observed_statistic = np.asarray(statistic(observed), dtype=np.float64)
        null_statistics = np.stack(
            [
                np.asarray(statistic(frame), dtype=np.float64)
                for _, frame in null_table.groupby(codes, sort=True)
            ]
        )
        if not np.isfinite(observed_statistic).all():
            raise ValueError(
                f"{operation}: the observed statistic is not finite "
                f"({observed_statistic}); it cannot be compared with the null."
            )
        undefined = ~np.isfinite(null_statistics).reshape(n_datasets, -1).all(axis=1)
        if undefined.any():
            raise ValueError(
                _format_error(
                    f"{operation}: the statistic is not finite for "
                    f"{int(undefined.sum())} of {n_datasets} null data sets "
                    f"(first: dataset {names[np.flatnonzero(undefined)[0]]}).",
                    why="a null data set without a statistic is neither at "
                    "least as extreme as the observed one nor less, and "
                    "leaving it out would make p too small",
                    fix="make the statistic defined on every data set, such "
                    "as a mean that skips events without a score "
                    "(table['score'].abs().mean() does), or decide how such "
                    "events count before testing",
                )
            )
        greater = np.sum(null_statistics >= observed_statistic, axis=0)
        less = np.sum(null_statistics <= observed_statistic, axis=0)
        below = np.sum(null_statistics < observed_statistic, axis=0)
        n_observed = len(observed)
        labels, thresholds = (), ()
    p_greater = (1.0 + greater) / (1.0 + n_datasets)
    p_less = (1.0 + less) / (1.0 + n_datasets)
    p_value = (
        p_greater
        if alternative == "greater"
        else p_less
        if alternative == "less"
        else np.minimum(2.0 * np.minimum(p_greater, p_less), 1.0)
    )
    percentile = 100.0 * below / n_datasets

    def scalar_or_array(values: Any) -> Any:
        array = np.asarray(values, dtype=np.float64)
        return float(array) if array.ndim == 0 else array

    return DatasetTestResult(
        statistic=scalar_or_array(observed_statistic),
        null_statistics=np.asarray(null_statistics, dtype=np.float64),
        p_value=scalar_or_array(p_value),
        percentile=scalar_or_array(percentile),
        alternative=alternative,
        n_datasets=n_datasets,
        n_observed=n_observed,
        labels=labels,
        thresholds=thresholds,
    )


# Rows times grid points of the pass matrix built at once.
_GRID_BLOCK = 20_000_000


def _grid_counts(
    table: pd.DataFrame,
    codes: NDArray[np.intp],
    n_datasets: int,
    grids: Sequence[CriterionGrid],
) -> NDArray[np.int64]:
    """Events of each data set passing every criterion at each grid point.

    Returns an integer array of shape (n_datasets, *grid_shape).
    """
    from scipy import sparse

    shape = tuple(grid.thresholds.size for grid in grids if grid.thresholds.ndim)
    size = int(np.prod(shape))
    passes = [grid.passes(table) for grid in grids]
    counts = np.zeros((n_datasets, size), dtype=np.int64)
    n_rows = len(table)
    block = max(1, _GRID_BLOCK // size)
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        joint = passes[0][start:stop]
        for more in passes[1:]:
            joint = (joint[:, :, None] & more[start:stop, None, :]).reshape(
                stop - start, -1
            )
        membership = sparse.csr_matrix(
            (
                np.ones(stop - start, dtype=np.int64),
                (codes[start:stop], np.arange(stop - start)),
            ),
            shape=(n_datasets, stop - start),
        )
        counts += np.asarray(membership @ joint.astype(np.int64))
    return counts.reshape((n_datasets, *shape))


def distribution_test(
    observed: pd.DataFrame,
    null_table: pd.DataFrame,
    *,
    column: str,
    test: Literal["ks", "rank_sum", "levene"] = "ks",
    alternative: Literal["greater", "less", "two-sided"] = "two-sided",
) -> DistributionTestResult:
    """Compare observed events' values with the pooled values of null events.

    Stella et al. (2019), Xu et al. (2019), Grosmark & Buzsáki (2016, Fig.
    S8) and Silva et al. (2015, Fig. 1d) compared a value of every real
    event with the same value of every null event pooled over the null
    copies, by a Kolmogorov-Smirnov, rank-sum or Levene test.

    Parameters
    ----------
    observed, null_table : pandas.DataFrame
        Per-event values of the real events and of the null events, such
        as from :func:`dataset_values`,
        :func:`~neurospatial.replay.compare_replay_methods` or
        :func:`~neurospatial.replay.event_properties`.
    column : str
        The compared column, in both tables.
    test : {"ks", "rank_sum", "levene"}, default="ks"
        ``"ks"``: the two-sample Kolmogorov-Smirnov test,
        ``scipy.stats.ks_2samp``. ``"rank_sum"``: the Wilcoxon rank-sum
        (Mann-Whitney U) test, ``scipy.stats.mannwhitneyu``. ``"levene"``:
        Levene's test of equal variances, ``scipy.stats.levene`` with
        ``center="mean"`` (Levene's original test; scipy's default, the
        median, is the Brown-Forsythe variant).
    alternative : {"greater", "less", "two-sided"}, default="two-sided"
        Stated in terms of the values: ``"greater"`` asks whether the
        observed values tend to be larger than the null values. For the KS
        test this is scipy's ``alternative="less"``, since scipy names the
        side by the empirical distribution functions: larger values have a
        lower CDF. Levene's test has no side and takes only
        ``"two-sided"``.

    Returns
    -------
    DistributionTestResult
        The statistic, p-value, sample sizes, NaN values dropped and
        medians.

    Raises
    ------
    ValueError
        If ``column`` is missing from a table, a sample has fewer than two
        finite values, ``test`` or ``alternative`` is invalid, or Levene's
        test is given a side.

    See Also
    --------
    dataset_test : A Monte Carlo test over whole null data sets.

    Notes
    -----
    Pooled copies of one event are not independent of each other: with
    many copies per event, the pooled null estimates the null distribution
    well, and the number of observed events is what the test's power
    rests on. A small p-value with tens of thousands of null values can
    reflect a small shift; report the medians alongside it. NaN values,
    such as events with an undefined score, are dropped and counted. To
    test per session or per task, call once per group.

    Examples
    --------
    Observed values shifted above the null values give a small p-value
    only for ``alternative="greater"``:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from neurospatial.replay import distribution_test
    >>> rng = np.random.default_rng(0)
    >>> observed = pd.DataFrame({"score": rng.normal(1.0, 1.0, 200)})
    >>> null = pd.DataFrame({"score": rng.normal(0.0, 1.0, 2000)})
    >>> greater = distribution_test(
    ...     observed, null, column="score", alternative="greater"
    ... )
    >>> less = distribution_test(observed, null, column="score", alternative="less")
    >>> bool(greater.p_value < 1e-10), bool(less.p_value > 0.5)
    (True, True)
    """
    from scipy import stats

    operation = "distribution_test()"
    tests = ("ks", "rank_sum", "levene")
    if test not in tests:
        raise ValueError(f"{operation}: test must be one of {tests}, got {test!r}.")
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"{operation}: alternative must be one of {_ALTERNATIVES}, got "
            f"{alternative!r}."
        )
    if test == "levene" and alternative != "two-sided":
        raise ValueError(
            _format_error(
                f"{operation}: Levene's test has no side, got "
                f"alternative={alternative!r}.",
                why="it tests whether the two samples' spreads differ",
                fix="pass alternative='two-sided'",
            )
        )
    samples = []
    dropped = []
    for name, table in (("observed", observed), ("null_table", null_table)):
        if column not in table:
            raise ValueError(f"{operation}: {name} has no {column!r} column.")
        values = np.asarray(table[column], dtype=np.float64)
        finite = np.isfinite(values)
        if finite.sum() < 2:
            raise ValueError(
                f"{operation}: {name} has {int(finite.sum())} finite values of "
                f"{column!r}; a two-sample test needs at least two."
            )
        samples.append(values[finite])
        dropped.append(int((~finite).sum()))
    first, second = samples
    if test == "ks":
        scipy_side = {"greater": "less", "less": "greater", "two-sided": "two-sided"}
        result = stats.ks_2samp(first, second, alternative=scipy_side[alternative])
    elif test == "rank_sum":
        result = stats.mannwhitneyu(first, second, alternative=alternative)
    else:
        result = stats.levene(first, second, center="mean")
    return DistributionTestResult(
        column=column,
        test=test,
        alternative=alternative,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n_observed=int(first.size),
        n_null=int(second.size),
        n_observed_dropped=dropped[0],
        n_null_dropped=dropped[1],
        median_observed=float(np.median(first)),
        median_null=float(np.median(second)),
    )


# --------------------------------------------------------------------------
# Template correlation (Liu et al. 2019)
# --------------------------------------------------------------------------


def relative_rank_template(
    spike_times: Any,
    intervals: NDArray[np.float64],
    *,
    spike_statistic: Literal["mean", "first", "median"] = "mean",
    min_active_cells: int = 2,
) -> NDArray[np.float64]:
    """Each cell's mean relative rank over a session's frames (Liu et al. 2019).

    In each frame (candidate event), the active cells are ranked by when
    they fired, and cell ``i`` gets the relative rank ``(k_i - 1) / (n -
    1)`` of its rank ``k_i`` among the ``n`` active cells: 0 for the first,
    1 for the last, 0.5 in the middle. A cell's template value is its mean
    relative rank over the frames it was active in; ordering cells by it
    gives the sleep template.

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        The session's spike times, one train per cell, in seconds.
    intervals : NDArray[np.float64], shape (n_frames, 2)
        Start and stop of each frame, in seconds; spikes at ``start <= t <
        stop`` count.
    spike_statistic : {"mean", "first", "median"}, default="mean"
        When each cell fired in a frame: its mean spike time ("the spike
        time center of mass", Liu et al.), first spike or median spike.
    min_active_cells : int, default=2
        Frames with fewer active cells are left out. Liu et al.'s frames
        had at least 4 pyramidal cells.

    Returns
    -------
    NDArray[np.float64], shape (n_cells,)
        Each cell's mean relative rank; NaN for a cell active in no frame
        that is kept.

    Raises
    ------
    ValueError
        If ``intervals`` is not a list of finite ``(start, stop)`` rows,
        ``spike_statistic`` is not one of the options, or
        ``min_active_cells`` is not an integer of at least 2.

    See Also
    --------
    template_correlation_test : Compare the template with the running template.

    Notes
    -----
    Cells firing at the same time share their average rank.

    References
    ----------
    Liu, K., Sibille, J., & Dragoi, G. (2019). Preconfigured patterns are
    the primary driver of offline multi-neuronal sequence replay.
    *Hippocampus*, 29(3), 275-283.

    Examples
    --------
    Three cells fire in the order 0, 1, 2 in one frame and 0, 2, 1 in
    another:

    >>> import numpy as np
    >>> from neurospatial.replay import relative_rank_template
    >>> spikes = [
    ...     np.array([0.01, 1.01]),
    ...     np.array([0.02, 1.03]),
    ...     np.array([0.03, 1.02]),
    ... ]
    >>> relative_rank_template(spikes, np.array([[0.0, 0.1], [1.0, 1.1]])).tolist()
    [0.0, 0.75, 0.75]
    """
    from scipy import stats

    from neurospatial._spikes import as_spike_trains
    from neurospatial.replay.calibration import _checked_intervals

    operation = "relative_rank_template()"
    statistics: dict[str, Callable[..., Any]] = {
        "mean": np.mean,
        "first": np.min,
        "median": np.median,
    }
    if spike_statistic not in statistics:
        raise ValueError(
            f"{operation}: spike_statistic must be one of {list(statistics)}, got "
            f"{spike_statistic!r}."
        )
    if (
        isinstance(min_active_cells, bool)
        or not isinstance(min_active_cells, (int, np.integer))
        or min_active_cells < 2
    ):
        raise ValueError(
            f"{operation}: min_active_cells must be an integer of at least 2, got "
            f"{min_active_cells!r}; a relative rank needs two active cells."
        )
    windows = _checked_intervals(intervals, operation)
    trains = [np.sort(train) for train in as_spike_trains(spike_times)]
    # When each cell fired in each frame, (n_cells, n_frames); NaN if silent.
    firing = np.full((len(trains), len(windows)), np.nan)
    for cell, train in enumerate(trains):
        first = np.searchsorted(train, windows[:, 0], side="left")
        last = np.searchsorted(train, windows[:, 1], side="left")
        active = np.flatnonzero(last > first)
        if spike_statistic == "first":
            firing[cell, active] = train[first[active]]
        else:
            summarize = statistics[spike_statistic]
            firing[cell, active] = [
                summarize(train[first[f] : last[f]]) for f in active
            ]
    n_active = np.sum(np.isfinite(firing), axis=0)
    kept = n_active >= min_active_cells
    ranks = stats.rankdata(firing[:, kept], axis=0, nan_policy="omit")
    relative = (ranks - 1.0) / (n_active[kept] - 1.0)
    counted = np.isfinite(relative)
    frames = counted.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        template: NDArray[np.float64] = np.where(
            frames > 0, np.where(counted, relative, 0.0).sum(axis=1) / frames, np.nan
        )
    return template


def template_correlation_test(
    templates: Sequence[NDArray[np.float64]],
    reference_templates: Sequence[NDArray[np.float64]],
    *,
    n_shuffles: int = 1000,
    use_absolute: bool = False,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    rng: np.random.Generator | int | None = None,
) -> DatasetTestResult:
    """Test how well templates match reference templates, against cell-order shuffles.

    Liu et al. (2019) asked whether the order of cells in sleep (each
    cell's mean relative rank, :func:`relative_rank_template`) follows
    their order along the track while running. The statistic is the mean,
    over animals and running directions, of the Spearman correlation
    between the sleep template and the running template; each shuffle
    permutes every running template's cells once and takes the mean
    again: "one shuffled sequence was generated for each run template,
    Spearman's rank order correlation was calculated between the shuffled
    run templates and sleep templates and the average of correlation
    values across all six animals was taken. The shuffling and averaging
    steps were repeated 1,000 times".

    Parameters
    ----------
    templates : sequence of NDArray[np.float64]
        One template per unit (animal and running direction): a value per
        cell, such as its mean relative rank in sleep; NaN for cells
        without one.
    reference_templates : sequence of NDArray[np.float64]
        The reference template of each unit, with the same cells: such as
        each place cell's field peak along the running direction ("always
        forward"); NaN for cells not in the template.
    n_shuffles : int, default=1000
        Shuffles; each permutes the cells of every reference template.
    use_absolute : bool, default=False
        Average the absolute correlations, Liu et al.'s second test.
    alternative : {"greater", "less", "two-sided"}, default="greater"
        The side tested.
    rng : numpy.random.Generator, int, or None
        Random number generator or seed.

    Returns
    -------
    DatasetTestResult
        The mean correlation as ``statistic``, each shuffle's mean, the
        p-value ``(1 + k) / (1 + n_shuffles)`` and the percentile. Liu et
        al.'s one-tailed p is ``1 - percentile / 100`` and their
        two-tailed p twice that (``LIU_2019_TEMPLATES.paper_p_value``).

    Raises
    ------
    ValueError
        If the two sequences differ in length, a pair of templates differs
        in its number of cells or shares fewer than three cells with values
        in both, a template or its reference is constant over the shared
        cells (no rank correlation), or ``n_shuffles`` is not a positive
        integer.

    See Also
    --------
    relative_rank_template : The sleep template.

    Notes
    -----
    Each correlation is over the cells with a value in both templates (the
    running template holds place cells, the sleep template the cells
    active in sleep). A shuffle permutes the reference values among the
    cells of the reference template, a random order of its cells, and the
    correlation is again over the cells in both. A shuffle that puts tied
    reference values on every shared cell has no correlation; it counts as
    at least as extreme as the observed statistic on either side, and is
    NaN in ``null_statistics``. Liu et al. also mention a
    Wilcoxon signed-rank test of the real correlation against the shuffled
    ones; its result is not reported, and a one-sample test of 1,000
    shuffles against one value does not test their hypothesis, so it is
    left out.

    References
    ----------
    Liu, K., Sibille, J., & Dragoi, G. (2019). Preconfigured patterns are
    the primary driver of offline multi-neuronal sequence replay.
    *Hippocampus*, 29(3), 275-283.

    Examples
    --------
    Two animals whose sleep order follows the running order:

    >>> import numpy as np
    >>> from neurospatial.replay import template_correlation_test
    >>> run = [np.arange(8.0), np.arange(10.0)]
    >>> sleep = [np.arange(8.0) / 7, np.r_[np.arange(10.0)[:8], 9.0, 8.0] / 9]
    >>> result = template_correlation_test(sleep, run, n_shuffles=999, rng=0)
    >>> print(f"{result.statistic:.3f}, p = {result.p_value:.3f}")
    0.994, p = 0.001
    """

    operation = "template_correlation_test()"
    _positive_integer(n_shuffles, "n_shuffles", operation)
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"{operation}: alternative must be one of {_ALTERNATIVES}, got "
            f"{alternative!r}."
        )
    if len(templates) != len(reference_templates) or not len(templates):
        raise ValueError(
            f"{operation}: templates and reference_templates must pair one to one "
            f"and not be empty, got {len(templates)} and "
            f"{len(reference_templates)}."
        )
    generator = np.random.default_rng(rng)
    observed = []
    shuffled = np.empty((n_shuffles, len(templates)))
    for unit, (template, reference) in enumerate(
        zip(templates, reference_templates, strict=True)
    ):
        x = np.asarray(template, dtype=np.float64)
        y = np.asarray(reference, dtype=np.float64)
        if x.shape != y.shape or x.ndim != 1:
            raise ValueError(
                f"{operation}: template {unit} has shape {x.shape} and its "
                f"reference {y.shape}; pass one value per cell, the same cells "
                "in both."
            )
        in_reference = np.flatnonzero(np.isfinite(y))
        both = np.isfinite(x) & np.isfinite(y)
        if both.sum() < 3:
            raise ValueError(
                f"{operation}: template {unit} shares {int(both.sum())} cells with "
                "its reference; a rank correlation needs at least three."
            )
        correlation = _spearman_rows(y[both][None, :], x[both])[0]
        if not np.isfinite(correlation):
            raise ValueError(
                _format_error(
                    f"{operation}: the correlation of template {unit} with its "
                    "reference is not defined.",
                    why="one of the two templates has the same value on every "
                    "cell they share, so the cells have no rank order",
                    fix="leave this unit out, or give templates whose shared "
                    "cells differ in value",
                )
            )
        observed.append(correlation)
        order = np.argsort(generator.random((n_shuffles, in_reference.size)), axis=1)
        permuted = np.tile(y, (n_shuffles, 1))
        permuted[:, in_reference] = y[in_reference][order]
        shuffled[:, unit] = _spearman_rows(permuted[:, both], x[both])
    values = np.asarray(observed)
    if use_absolute:
        values, shuffled = np.abs(values), np.abs(shuffled)
    statistic = float(np.mean(values))
    null = shuffled.mean(axis=1)
    # A shuffle without a correlation counts as at least as extreme.
    undefined = np.isnan(null)
    greater = int(np.sum(undefined | (null >= statistic)))
    less = int(np.sum(undefined | (null <= statistic)))
    p_greater = (1.0 + greater) / (1.0 + n_shuffles)
    p_less = (1.0 + less) / (1.0 + n_shuffles)
    p_value = (
        p_greater
        if alternative == "greater"
        else p_less
        if alternative == "less"
        else min(2.0 * min(p_greater, p_less), 1.0)
    )
    return DatasetTestResult(
        statistic=statistic,
        null_statistics=null,
        p_value=float(p_value),
        percentile=float(100.0 * np.sum(null < statistic) / n_shuffles),
        alternative=alternative,
        n_datasets=n_shuffles,
        n_observed=len(templates),
    )


def _positive_integer(value: Any, name: str, operation: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(
            f"{operation}: {name} must be a positive integer, got {value!r}."
        )


# --------------------------------------------------------------------------
# Published data-set tests
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetTestRecipe:
    """A published data-set-level replay test, with every value it uses.

    A recipe records the paper's choices; the documented calls in each
    recipe's ``calls`` note and in the user guide run it with
    :func:`dataset_values`, :func:`dataset_test`, :func:`distribution_test`
    or :func:`template_correlation_test`.

    Attributes
    ----------
    name : str
        Short name, such as ``"silva_2015_matrix"``.
    reference : str
        The paper.
    test : {"monte_carlo", "distribution", "template_correlation"}
        The kind of test: a data-set statistic against null data sets, a
        pooled two-sample test of per-event values, or Liu et al.'s
        template test.
    method : ReplayMethod, RankOrderMethod or None
        The per-event measure, such as ``SILVA_2015`` from
        :mod:`neurospatial.replay.methods`; None when the paper's measure
        is not a library method.
    null : str or None
        The null: a ``dataset_values`` null (``null_source="shuffle"``), a
        :func:`~neurospatial.replay.null_events` kind (``"spikes"``), or
        None for Liu et al.'s cell-order shuffle of templates.
    null_source : {"shuffle", "spikes", "templates"}
        What makes the null data sets: :func:`dataset_values` (or
        :func:`shuffle_events`) shuffling decoded events,
        :func:`~neurospatial.replay.null_events` copies decoded with
        :func:`~neurospatial.replay.decode_intervals`, or
        :func:`template_correlation_test`.
    n_datasets : int
        Null data sets (or, for a pooled test, null copies per event; for
        the template test, shuffles).
    shared : bool
        Whether one template draw serves every event of a data set.
    null_options : Mapping[str, Mapping[str, Any]]
        Options of the null, as ``dataset_values`` takes them.
    criteria : tuple of CriterionGrid
        The threshold grid of a Monte Carlo test.
    quantities : tuple of str
        Quantities to measure besides the method's criteria.
    columns : tuple of str
        The per-event values a pooled test compares. ``"abs_score"`` is
        ``|score|``, made with ``table.assign(abs_score=table["score"].abs())``;
        the notes say how to make any other column the library's tables do
        not hold.
    distribution : tuple of {"ks", "rank_sum", "levene"}
        The pooled tests.
    alternative : {"greater", "less", "two-sided"}
        The side, in terms of the values or the statistic.
    alpha : float
        The significance level the paper used or implies.
    p_value_rule : {"monte_carlo", "percentile", "percentile_two_tailed", "test"}
        How the paper computed p: ``(n + 1) / (r + 1)`` (``p_value``), ``1 -
        percentile / 100``, twice that, or the two-sample test's p.
    decoding : Mapping[str, Any]
        The paper's decoding settings, for reference.
    notes : tuple of str
        The documented calls, choices the paper does not state, and
        readings of them.
    """

    name: str
    reference: str
    test: Literal["monte_carlo", "distribution", "template_correlation"]
    method: ReplayMethod | RankOrderMethod | None = None
    null: str | None = None
    null_source: Literal["shuffle", "spikes", "templates"] = "shuffle"
    n_datasets: int = 1
    shared: bool = False
    null_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    criteria: tuple[CriterionGrid, ...] = ()
    quantities: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    distribution: tuple[Literal["ks", "rank_sum", "levene"], ...] = ()
    alternative: Literal["greater", "less", "two-sided"] = "greater"
    alpha: float = 0.05
    p_value_rule: Literal[
        "monte_carlo", "percentile", "percentile_two_tailed", "test"
    ] = "monte_carlo"
    decoding: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the mappings and sequences."""
        object.__setattr__(
            self,
            "null_options",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(options))
                    for name, options in self.null_options.items()
                }
            ),
        )
        object.__setattr__(self, "decoding", MappingProxyType(dict(self.decoding)))
        for name in ("criteria", "quantities", "columns", "distribution", "notes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def __hash__(self) -> int:
        """Hash by name and reference; the mappings are not hashable."""
        return hash((self.name, self.reference))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mappings cannot pickle."""
        from neurospatial._results import reduce_through_constructor

        rebuild, (cls, state) = reduce_through_constructor(self)
        state["null_options"] = {
            name: dict(options) for name, options in self.null_options.items()
        }
        return rebuild, (cls, state)

    def paper_p_value(self, result: DatasetTestResult) -> float | NDArray[np.float64]:
        """The p-value by the paper's own rule.

        Parameters
        ----------
        result : DatasetTestResult
            The test's result.

        Returns
        -------
        float or NDArray[np.float64]
            ``result.p_value`` for ``(n + 1) / (r + 1)``; ``1 - percentile /
            100`` for a percentile rule, twice that (at most 1) for a
            two-tailed one.

        Raises
        ------
        ValueError
            For a recipe whose p-value is a two-sample test's.
        """
        if self.p_value_rule == "monte_carlo":
            return result.p_value
        if self.p_value_rule == "test":
            raise ValueError(
                f"{self.name}: the paper's p-value is the two-sample test's; read "
                "it from distribution_test()."
            )
        one_tailed = 1.0 - np.asarray(result.percentile, dtype=np.float64) / 100.0
        if self.p_value_rule == "percentile_two_tailed":
            one_tailed = np.minimum(2.0 * one_tailed, 1.0)
        return float(one_tailed) if one_tailed.ndim == 0 else one_tailed


def _silva_grid() -> tuple[CriterionGrid, ...]:
    return (
        CriterionGrid(
            "score", ">", np.round(np.arange(0.0, 1.0, 0.1), 1), absolute=True
        ),
        CriterionGrid(
            "max_map_jump",
            "<",
            np.round(np.arange(0.1, 1.01, 0.1), 1),
            relative_to="track_length",
        ),
    )


_SILVA = (
    "Silva, D., Feng, T., & Foster, D. J. (2015). Trajectory events across "
    "hippocampal place cells require previous experience. Nature "
    "Neuroscience, 18(12), 1772-1779."
)
_FAROOQ = (
    "Farooq, U., Sibille, J., Liu, K., & Dragoi, G. (2019). Strengthened "
    "temporal coordination within pre-existing sequential cell assemblies "
    "supports trajectory replay. Neuron, 103(4), 719-733."
)
_FAROOQ_NOTES = (
    "Thresholds: |weighted correlation| > 0 to > 0.9 and maximum jump < 0.1 "
    "to < 1 in steps of 0.1 (p. e3), 'as in (Silva et al., 2015)'. The jump "
    "is read as a fraction of the track's length (the paper does not say; "
    "Silva normalizes it).",
    "p = 1 - #(null data sets with a lower share than the real sleep) / 500 "
    "(p. e3), without the + 1: 1 - percentile / 100 (paper_p_value). The "
    "library's p_value is larger by (1 - p) / 501, at most 0.002.",
    "No alpha is stated; the colour bar runs from 0 to 0.1 with a 'No data' "
    "category, read as grid points no frame passes. alpha = 0.05 is assumed.",
    "Frames pooled over animals (n = 38,820 in Fig. 1, 38,220 on p. e7), "
    "each running direction decoded and analysed on its own (p. e3); whether "
    "a frame counts once per direction in the pooled matrix is not stated.",
    "Frames are events of 100 to 800 ms 'that contained activation of at "
    "least 5 distinct neurons' (p. e2). dataset_values reports FAROOQ_2019's "
    "n_active_cells but dataset_test does not apply it, so pass only the "
    "frames: keep = [np.sum(c.sum(axis=0) > 0) >= 5 for c in counts], with "
    "the counts of the cells that reading counts (FAROOQ_2019's notes), and "
    "make surrogate data sets from these frames' intervals.",
)

SILVA_2015_MATRIX = DatasetTestRecipe(
    name="silva_2015_matrix",
    reference=_SILVA,
    test="monte_carlo",
    method=SILVA_2015,
    null="column_cycle",
    n_datasets=5000,
    criteria=_silva_grid(),
    alternative="greater",
    alpha=0.05,
    p_value_rule="monte_carlo",
    decoding={"time_bin": 0.02},
    notes=(
        "Calls: real = dataset_values(env, events, replay_method=SILVA_2015); shuffled = "
        "dataset_values(env, events, replay_method=SILVA_2015, null='column_cycle', "
        "n_datasets=5000, rng=0); dataset_test(real, shuffled, "
        "criteria=SILVA_2015_MATRIX.criteria). Significant cells: p_value < "
        "0.05; trajectory events are the lower-left quadrant (high |r|, small "
        "jumps). The trio of Supplementary Table 1: (|r| > 0.6, jump < 0.4), "
        "(0.7, 0.4), (0.7, 0.3).",
        "Statistic (p. 1775): the number of candidate events with |weighted "
        "correlation| above one threshold and maximum jump between peak "
        "decoded positions, as a fraction of the track, below the other; with "
        "data sets of equal size the share tested here is the count.",
        "p = (n + 1) / (r + 1) with n the shuffled data sets at least as large "
        "(Online Methods): the library's p_value exactly; Supplementary Table 1 "
        "reports 0.0002 = 1 / 5,001.",
        "Grid read from Fig. 3a (ticks >0, >0.3, >0.6, >0.9 and <0.2, <0.5, "
        "<0.8): |r| > 0, 0.1, ..., 0.9 and jump < 0.1, ..., 1.0; the text "
        "does not list it, Farooq et al. (2019) state the same grid 'as in "
        "(Silva et al., 2015)'.",
        "Column cycle: each time bin's posterior shifted circularly by a random "
        "distance; whether a shift of 0 is allowed is not stated (the default "
        "allows it; null_options={'column_cycle': {'exclude_zero_shift': "
        "True}} does not).",
        "Events pooled over the sessions of a condition (9,818 for e1s1, the "
        "sum of Supplementary Table 2). Controls (Fig. 3b): a Poisson data set "
        "(null_events(kind='rate_poisson', n_copies=1, cells=place_cells), "
        "decoded) and a time-swap data set (shuffle_events(env, events, "
        "null='time_bin_permutation')), each run through the same matrix "
        "against its own 5,000 column-cycle data sets; their comparison with "
        "the real matrix is visual.",
        "Runtime: the moment-table path of dataset_values takes about 20 s for "
        "9,818 events and 5,000 data sets, dataset_test about as long, and "
        "the table about 2 GB.",
    ),
)

SILVA_2015_RANK_ORDER = DatasetTestRecipe(
    name="silva_2015_rank_order",
    reference=_SILVA,
    test="monte_carlo",
    method=RankOrderMethod(
        name="silva_2015_rank",
        reference=_SILVA,
        spike_statistic="all_spikes",
        min_active_cells=3,
        test="cell_order",
        n_shuffles=5000,
        notes=(
            "Spearman correlation of every spike's time with its cell's field "
            "peak; the minimum number of cells is not stated, 3 is the "
            "library's floor.",
        ),
    ),
    null="cell_order",
    n_datasets=5000,
    shared=True,
    criteria=(CriterionGrid("score", ">", np.array([0.5, 0.6, 0.7]), absolute=True),),
    columns=("abs_score",),
    distribution=("levene", "rank_sum", "ks"),
    alternative="greater",
    p_value_rule="monte_carlo",
    notes=(
        "Calls: maps = {'right': right_maps, 'left': left_maps}; real = "
        "dataset_values(env, events, replay_method=recipe.method, spike_times=spikes, "
        "encoding_models=maps, dt=0.02); null = dataset_values(env, events, replay_method="
        "recipe.method, null='cell_order', shared=True, n_datasets=5000, "
        "spike_times=spikes, encoding_models=maps, dt=0.02, rng=0); "
        "dataset_test(real, null, criteria=recipe.criteria) for both templates "
        "pooled (Fig. 1d, top); for the larger |r| of the two templates "
        "(bottom), group each table by (dataset, event) and take the maximum "
        "|score| first. The pooled tests compare absolute correlations ('The "
        "absolute values of spike correlations between original and shuffled "
        "data sets were compared', Online Methods): distribution_test(real"
        ".assign(abs_score=real.score.abs()), null.assign(abs_score=null.score"
        ".abs()), column='abs_score', test='levene', 'rank_sum' or 'ks', "
        "alternative='greater' for the one-tailed tests), or the maximum "
        "abs_score per (dataset, event).",
        "'5,000 shuffled templates' is read as one permutation of cell "
        "identities per shuffled data set, shared by every event (shared=True); "
        "a per-event p-value is the same either way, a count over events is "
        "not. Which cells are permuted (all place cells, or each event's "
        "active cells) is not stated: pass the place cells only.",
        "Reported (p. 1773): one-tailed KS p = 0.122; Monte Carlo p > 0.1 at "
        "|r| > 0.5, 0.6 and 0.7. Levene's test here is Levene's original, "
        "centered on the mean.",
    ),
)

GROSMARK_2016_SURROGATES = DatasetTestRecipe(
    name="grosmark_2016_surrogates",
    reference=(
        "Grosmark, A. D., & Buzsáki, G. (2016). Diversity in neural firing "
        "dynamics supports both rigid and learned hippocampal sequences. "
        "Science, 351(6280), 1440-1443."
    ),
    test="distribution",
    method=dataclasses.replace(GROSMARK_2016, n_shuffles=200),
    null="rate_poisson",
    null_source="spikes",
    n_datasets=100,
    columns=("z_score[place_field_circular]", "abs_score", "quality"),
    distribution=("rank_sum",),
    alternative="two-sided",
    p_value_rule="test",
    decoding={"time_bin": 0.02, "position_bin": 2.0, "prior": "uniform"},
    notes=(
        "Fig. S8: observed events against 100 Poisson surrogate data sets "
        "(null_events(kind='rate_poisson', n_copies=100) per epoch and running "
        "direction, decoded with that direction's maps), each event tested "
        "against 200 circular place-field shuffles: compare_replay_methods("
        "env, events, replay_methods=[replace(GROSMARK_2016, n_shuffles=200)], "
        "...) for the observed and the surrogate events.",
        "Compared values: rZ (the table's z_score[place_field_circular], |r| "
        "z-scored against its shuffles; np.std, ddof = 0, where MATLAB's std "
        "has ddof = 1, a factor sqrt(200 / 199)), |r| and the reconstruction "
        "quality, the median over time bins of the largest posterior value "
        "(event_properties gives the mean): table.assign(abs_score=table.score"
        ".abs(), quality=[np.median(e.posterior.max(axis=1)) for e in events]) "
        "on each compare_replay_methods table, rows in event order.",
        "The test is not named in the legend; a rank-sum test of observed "
        "events against the pooled surrogate events is a reading (the "
        "supplement's other between-condition tests are rank-sum tests).",
        "Time-bin-shuffle surrogates (Fig. S8B): shuffle_events(env, events, "
        "null='time_bin_permutation', n_datasets=100, spike_counts=counts), "
        "each set's events and counts passed to compare_replay_methods. Figs. "
        "S8C-D use 200 time-bin permutations as the per-event null.",
        "Figs. S5C and S7B (share of p < 0.05 events against chance, "
        "chi-square): detection_rates(table.assign(is_replay=table.p_value < "
        "0.05), expected_rate=0.05), an exact binomial test; whether S5C "
        "compares with 5% or PRE with POST is ambiguous.",
        "Runtime: 200 place-field shuffles take about 35 ms per event, so 100 "
        "data sets of 1,000 events take about an hour per class and epoch; run "
        "data sets in parallel with seeds from np.random.SeedSequence(seed)"
        ".spawn(100).",
    ),
)

FAROOQ_2019_TIME_BIN = DatasetTestRecipe(
    name="farooq_2019_time_bin",
    reference=_FAROOQ,
    test="monte_carlo",
    method=FAROOQ_2019,
    null="time_bin_permutation",
    n_datasets=500,
    criteria=_silva_grid(),
    quantities=("max_map_jump",),
    alternative="greater",
    p_value_rule="percentile",
    decoding={"time_bin": 0.02, "prior": "uniform"},
    notes=(
        "Fig. 1E (Pre-Run vs tPre-Run): 500 data sets with each frame's "
        "decoded time bins permuted. Calls: pre = dataset_values(env, frames, replay_method="
        "FAROOQ_2019, spike_counts=counts, quantities=('max_map_jump',)); "
        "dataset_test(pre, dataset_values(env, frames, replay_method=FAROOQ_2019, "
        "null='time_bin_permutation', n_datasets=500, spike_counts=counts, "
        "quantities=('max_map_jump',), rng=0), criteria=recipe.criteria); "
        "recipe.paper_p_value(result).",
        "Fig. 1G (Poisson vs tPoisson): one Poisson data set, then this test on it.",
        *_FAROOQ_NOTES,
    ),
)

FAROOQ_2019_POISSON = dataclasses.replace(
    FAROOQ_2019_TIME_BIN,
    name="farooq_2019_poisson",
    null="rate_poisson",
    null_source="spikes",
    notes=(
        "Fig. 1F (Pre-Run vs 500 rate-matched Poisson data sets): each cell "
        "fires as a Poisson process at its mean rate over every frame of the "
        "sleep (p. e4). Calls, one data set at a time: tables = []; for k, "
        "seed in enumerate(np.random.SeedSequence(1).spawn(500)): pois = "
        "null_events(spikes, frame_intervals, kind='rate_poisson', "
        "rng=np.random.default_rng(seed)); pe, pc = decode_intervals(env, "
        "pois.spike_times, pois.intervals, maps, dt=0.02); tables.append("
        "dataset_values(env, pe, replay_method=FAROOQ_2019, spike_counts=pc, "
        "quantities=('max_map_jump',)).assign(dataset=k)); then "
        "dataset_test(pre, pd.concat(tables), criteria=recipe.criteria).",
        "Fig. 1H compares the mean |r| of Pre-Run, tPre-Run, Poisson and "
        "tPoisson frames by rank-sum tests; the SEMs suggest one shuffled data "
        "set (n = 38,820), not 500 pooled (a reading): distribution_test(pre"
        ".assign(abs_score=pre.score.abs()), one_set.assign(abs_score="
        "one_set.score.abs()), column='abs_score', test='rank_sum').",
        "Memory and runtime: decode_intervals holds every time bin's posterior "
        "and counts at once, so decoding all 500 x 38,820 surrogate frames of "
        "10 time bins in one call would need about 270 GB (72 bins, 100 "
        "cells); "
        "one data set at a time needs about 0.5 GB. Decoding all 500 takes about 10 minutes "
        "(0.16 s per 5,000 frames measured, one matrix product per block of "
        "50,000 time bins) rather than the 12 hours one decode per frame "
        "takes; dataset_values measures each frame with detect_replay, about "
        "0.5 ms per frame, so about 3 hours; null_events takes about 0.2 ms "
        "per frame, about an hour. The data sets are independent and can run "
        "in parallel.",
        *_FAROOQ_NOTES,
    ),
)

FAROOQ_2019_SPACE_BIN = dataclasses.replace(
    FAROOQ_2019_TIME_BIN,
    name="farooq_2019_space_bin",
    null="column_cycle",
    null_options={"column_cycle": {"exclude_zero_shift": True}},
    notes=(
        "Fig. S3A/C (sPre-Run, sPoisson): each time bin's posterior "
        "'circularly shifted by a random amount between 1 and track length "
        "minus 1' (p. e3): the column cycle without the zero shift.",
        "Fig. S3D compares the sPre-Run and sPoisson families with each other; "
        "which plays the observed data set is not stated.",
        *_FAROOQ_NOTES,
    ),
)

FAROOQ_2019_WEIGHTED = dataclasses.replace(
    FAROOQ_2019_TIME_BIN,
    name="farooq_2019_weighted",
    null="column_cycle_weighted",
    null_options={"column_cycle_weighted": {"edge_exclusion": 10.0}},
    notes=(
        "Fig. S3F (csPre-Run, csPoisson): each time bin's peak moved to a "
        "position drawn from the sleep's mean decoded posterior, kept 10 cm "
        "from the track's ends (p. e4). Pass pseudo_event_pool = the sleep's "
        "frames' posteriors (the default stacks the events given), in cm.",
        *_FAROOQ_NOTES,
    ),
)

LIU_2019_TEMPLATES = DatasetTestRecipe(
    name="liu_2019_templates",
    reference=(
        "Liu, K., Sibille, J., & Dragoi, G. (2019). Preconfigured patterns are "
        "the primary driver of offline multi-neuronal sequence replay. "
        "Hippocampus, 29(3), 275-283."
    ),
    test="template_correlation",
    null_source="templates",
    null=None,
    n_datasets=1000,
    alternative="greater",
    p_value_rule="percentile",
    notes=(
        "Calls: sleep = [relative_rank_template(spikes[a], frames[a], "
        "spike_statistic='mean', min_active_cells=4) for each animal and "
        "direction]; run = each direction's field peaks in running order "
        "('always forward', the leftward direction's positions negated); "
        "result = template_correlation_test(sleep, run, n_shuffles=1000, "
        "rng=0); result.percentile and recipe.paper_p_value(result). Liu et "
        "al. report percentiles 97.6 (pre-run) and 95.8 (post-run), one-tailed "
        "p = .024 and .042, two-tailed .048 and .084: p = 1 - percentile / 100 "
        "and twice that (replace(recipe, p_value_rule='percentile_two_tailed'))"
        "; use_absolute=True gives their absolute-correlation test.",
        "The statistic is the mean over animals and both running templates of "
        "each animal of Spearman(sleep template, run template); with two "
        "templates per animal it equals the mean of per-animal means.",
        "The correlation is over cells in both templates (a reading: the run "
        "template holds place cells, the sleep template the pyramidal cells "
        "active in sleep).",
        "The Wilcoxon signed-rank test 'between the real correlation and the "
        "distribution of control shuffled correlations' is not reported and "
        "is left out.",
    ),
)

STELLA_2019_JITTER = DatasetTestRecipe(
    name="stella_2019_jitter",
    reference=(
        "Stella, F., Baracskay, P., O'Neill, J., & Csicsvari, J. (2019). "
        "Hippocampal reactivation of random trajectories resembling Brownian "
        "diffusion. Neuron, 102(2), 450-461."
    ),
    test="distribution",
    null="spike_time_uniform",
    null_source="spikes",
    n_datasets=500,
    columns=("mean_peak_probability", "step"),
    distribution=("ks",),
    alternative="two-sided",
    p_value_rule="test",
    decoding={"time_bin": 0.008},
    notes=(
        "Calls (per session, 8 ms windows), one copy at a time so that only "
        "one copy's posteriors are held: nul = []; for seed in "
        "np.random.SeedSequence(0).spawn(500): jit = null_events(spikes, "
        "selected_intervals, kind='spike_time_uniform', "
        "rng=np.random.default_rng(seed)); je, _ = decode_intervals(env, "
        "jit.spike_times, jit.intervals, maps, dt=0.008); "
        "nul.append(event_properties(env, je)); then obs = "
        "event_properties(env, selected_events); distribution_test(obs, "
        "pd.concat(nul), column='mean_peak_probability', test='ks'); the mean "
        "step per table is t.assign(step=t.map_path_length / (t.n_time_bins - "
        "1)). Decoding all 279,000 copies in one call would hold every 8 ms "
        "bin's posterior at once.",
        "500 copies per event is inferred from 279,000 shuffled and 558 "
        "original events (Fig. 2B-C); the KS side is not stated (two-sided "
        "here).",
        "Only events passing the quality criteria are jittered, and shuffles "
        "are not filtered (p. e5). Within events Stella kept only windows "
        "whose peak passes 0.1 when averaging steps; that filter is not "
        "applied. The 15-spike windows cannot be made by decode_intervals.",
        "Fig. S5E (share passing selection, binomial test) is read as the "
        "original count against Binomial(n, shuffled share).",
    ),
)

XU_2019_ROTATION = DatasetTestRecipe(
    name="xu_2019_rotation",
    reference=(
        "Xu, H., Baracskay, P., O'Neill, J., & Csicsvari, J. (2019). Assembly "
        "responses of hippocampal CA1 place cells predict learned behavior in "
        "goal-directed spatial tasks on the radial eight-arm maze. Neuron, "
        "101(1), 119-132."
    ),
    test="distribution",
    null="place_field_circular",
    null_source="shuffle",
    n_datasets=100,
    columns=("score",),
    distribution=("ks",),
    alternative="greater",
    p_value_rule="test",
    decoding={"time_bin": 0.02, "position_bins_per_arm": 70},
    notes=(
        "Score: summed posterior probability along the path that maximizes it, "
        "with jumps of at most 20 bins between windows, over the number of "
        "windows. No library score computes this jump-constrained path; give "
        "it as a ReplayMethod score to run the calls.",
        "Calls (per task): obs = dataset_values(env, events, replay_method=XU); shuf = "
        "dataset_values(env, events, replay_method=XU, null='place_field_circular', "
        "n_datasets=100, spike_counts=counts, encoding_models=maps, dt=0.02, "
        "rng=0); distribution_test(obs, shuf, column='score', test='ks', "
        "alternative='greater'), the one-tailed KS of 'significantly weaker "
        "probabilities' for rotated maps.",
        "Null: each cell's field rotated on each arm independently, wrapping "
        "within the arm; place_field_circular wraps over the whole track. Decode "
        "each arm on its own environment to wrap within it.",
        "No per-event threshold: z = (score - mean of its shuffles) / their "
        "standard deviation is reported and tested at the data-set level, "
        "#(z > 0) against Binomial(n, 0.5), one-sided (a reading of 'binomial "
        "test').",
        "Per-event criteria (p. e3): at least 4 windows, at least 20 bins "
        "covered and a mean speed above 2 bins per window.",
    ),
)


DATASET_RECIPES: tuple[DatasetTestRecipe, ...] = (
    SILVA_2015_MATRIX,
    SILVA_2015_RANK_ORDER,
    GROSMARK_2016_SURROGATES,
    FAROOQ_2019_TIME_BIN,
    FAROOQ_2019_POISSON,
    FAROOQ_2019_SPACE_BIN,
    FAROOQ_2019_WEIGHTED,
    LIU_2019_TEMPLATES,
    STELLA_2019_JITTER,
    XU_2019_ROTATION,
)
"""Every published data-set-level test recipe in this module."""
