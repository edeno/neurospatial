"""Immutable ``BayesianDecoder`` fit/predict wrapper over the decode core.

Provides :class:`BayesianDecoder`, a thin **frozen** convenience layer over the
existing functional decoders (:func:`~neurospatial.decoding.decode_session`,
:func:`~neurospatial.decoding.decode_session_summary`). The wrapper does **not**
re-implement decoding: ``fit`` reuses the same internal encoder
(``decode_session``'s ``_build_encoding_model``) that ``decode_session`` runs
internally, and ``predict``/``predict_summary`` delegate straight to the
functional core with the fitted encoding models. Reusing that internal encoder
is what guarantees the encoding step is byte-identical to ``decode_session``'s
(same ``fill_value=0.0``, ``max_gap`` masking, dtype, KDE parameters), so a
fitted decoder's posterior is byte-for-byte equal to ``decode_session`` on the
same inputs.

Unlike pynapple's ``decode_1d`` / ``decode_2d``, decoding runs through an
:class:`~neurospatial.environment.Environment`, so geodesic / linearized-track /
graph-based decoding works: the same fit/predict flow decodes a 1-D linearized
track or a masked open field, not just a rectangular grid.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from neurospatial._typing import SpikeTrainsLike
    from neurospatial.decoding._result import DecodingResult, DecodingSummary
    from neurospatial.environment import Environment

__all__ = ["BayesianDecoder"]


@dataclass(frozen=True)
class BayesianDecoder:
    """Immutable Bayesian position decoder over the functional decode core.

    A thin, frozen wrapper that pairs an :class:`~neurospatial.environment.Environment`
    and a set of encoding parameters with (once fitted) a population of encoding
    models. :meth:`fit` builds the encoding models and returns a **new** fitted
    decoder; :meth:`predict` / :meth:`predict_summary` decode fresh spikes
    against those models; :meth:`score` reports decode error against ground
    truth. The class is frozen, so "fitting" never mutates the original decoder.

    The wrapper delegates to the functional core rather than re-implementing it:
    a fitted decoder's :meth:`predict` reproduces
    :func:`~neurospatial.decoding.decode_session` byte-for-byte on the same
    inputs and parameters.

    Because decoding runs through the ``Environment``, geodesic / linearized /
    graph-based decoding works (unlike pynapple ``decode_1d`` / ``decode_2d``):
    the same fit/predict flow handles a 1-D linearized track or a masked open
    field, not only a rectangular grid.

    Parameters
    ----------
    env : Environment
        Fitted spatial environment defining the bin layout and connectivity.
    dt : float, default=0.025
        Decoding time-bin width in seconds.
    bandwidth : float or None, default=None
        Smoothing bandwidth (position units) for the ratio-method encoding step.
        ``None`` resolves to the encoder default (5.0); a ratio-only param, so it
        must stay ``None`` when ``method="glm"`` (validated at construction).
    method : str, default="diffusion_kde"
        Estimator for the encoding step. One of ``"diffusion_kde"``,
        ``"gaussian_kde"``, ``"binned"``, or ``"glm"`` (penalized-Poisson GAM,
        tuned with ``penalty`` / ``rank``).
    min_occupancy : float or None, default=None
        Minimum occupancy (seconds) for a bin to enter the ratio-method encoding
        model. Low-occupancy bins are filled with ``0.0`` Hz (``fill_value=0.0``),
        never ``NaN``. ``None`` resolves to the encoder default (0.0); a
        ratio-only param, so it must stay ``None`` when ``method="glm"``.
    penalty : float or None, default=None
        ``method="glm"`` smoothness penalty ``lambda``; ``None`` chooses it by
        REML. Mutually exclusive with the ratio params (``bandwidth`` /
        ``min_occupancy``); validated at construction.
    rank : int or None, default=None
        ``method="glm"`` requested basis rank cap; ``None`` uses the encoder
        default.
    max_gap : float or None, default=0.5
        Maximum trajectory time gap (seconds) forwarded to the encoding step.
        Intervals longer than ``max_gap`` are dropped from both the spike
        numerator and the occupancy denominator. Matches
        :func:`~neurospatial.decoding.decode_session`'s default. Pass ``None`` to
        count all intervals regardless of gap.
    dtype : {np.float32, np.float64}, default=np.float64
        Working dtype for the encoding models and the posterior. ``np.float32``
        halves the working set; ``np.float64`` (default) is byte-for-byte the
        reference decode.
    warn_on_drop : bool, default=True
        Forwarded to the encode step (:meth:`fit`) and the decode step
        (:meth:`predict` / :meth:`predict_summary`). When ``True`` (default) a
        single ``UserWarning`` fires if a large fraction of spikes fall outside
        the decode/encode time window (the seconds-vs-milliseconds footgun).
        Set ``False`` to silence those warnings on a genuinely sparse session;
        a single knob covers both the fit and predict passes (the default would
        otherwise warn once per pass).
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins) or None
        Fitted encoding-model firing-rate maps. ``None`` (default) marks the
        decoder **unfitted**; :meth:`predict` / :meth:`predict_summary` /
        :meth:`score` raise until :meth:`fit` populates it. Set only via
        :meth:`fit`.
    unit_ids : NDArray or None, default=None
        Identity label per encoding model. Populated by :meth:`fit` from a
        labelled spike input such as a pynapple ``TsGroup`` (its index), or
        ``arange(n_neurons)`` when the input carries no labels. Labels passed
        here directly, or captured by :meth:`fit` from a labelled input, are
        used to pair spike trains with encoding models: when both they and the
        spike input given to :meth:`predict` / :meth:`predict_summary` /
        :meth:`score` are labelled, trains are matched to models by label. In
        every other case (including the generated ``arange`` labels) trains are
        paired by position, one per fitted unit, in fit order.

    Attributes
    ----------
    env, dt, bandwidth, method, min_occupancy, penalty, rank, max_gap, dtype, \
warn_on_drop
        The configuration passed at construction (immutable).
    encoding_models : NDArray[np.float64] or None
        The fitted encoding models, or ``None`` when unfitted.
    unit_ids : NDArray or None
        Per-model identity labels, or ``None`` when unfitted.
    is_fitted : bool
        Read-only property: ``True`` once :meth:`fit` has populated the
        encoding models, ``False`` otherwise.

    Raises
    ------
    ValueError
        At construction, if ``dt`` is not a finite ``> 0`` number, if ``dtype``
        is not ``np.float32`` / ``np.float64``, if the method-specific params are
        invalid (an unknown ``method``; a ratio param with ``method="glm"`` or a
        glm param with a ratio method; an out-of-domain ``penalty`` / ``rank``) --
        these mirror ``compute_spatial_rate`` -- or (when ``encoding_models`` is
        injected directly) if the fitted state is inconsistent (missing
        ``unit_ids``, wrong ndim, or a bin/unit-count mismatch against ``env``).
    RuntimeError
        From :meth:`predict` / :meth:`predict_summary` / :meth:`score` if the
        decoder is unfitted.

    Other Parameters
    ----------------
    _unit_ids_generated : bool, default=False
        Internal identity flag carried by selection or fitting when unit labels
        were generated. Leave it at its default when supplying real unit IDs.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import BayesianDecoder
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 50, (500, 2))  # doctest: +SKIP
    >>> env = Environment.from_samples(positions, bin_size=5.0)  # doctest: +SKIP
    >>> decoder = BayesianDecoder(env, dt=0.1).fit(  # doctest: +SKIP
    ...     spike_times, times, positions
    ... )
    >>> result = decoder.predict(spike_times, times)  # doctest: +SKIP
    >>> result.map_position.shape  # doctest: +SKIP
    (n_time_bins, 2)

    See Also
    --------
    neurospatial.decoding.decode_session : Functional encode->bin->decode core.
    neurospatial.decoding.decode_session_summary : Memory-safe streamed sibling.
    """

    env: Environment
    dt: float = 0.025
    bandwidth: float | None = None
    method: str = "diffusion_kde"
    min_occupancy: float | None = None
    penalty: float | None = None
    rank: int | None = None
    max_gap: float | None = 0.5
    dtype: type[np.float32] | type[np.float64] = np.float64
    warn_on_drop: bool = True
    # Fitted state (private; ``None`` => unfitted). Set only via :meth:`fit`.
    encoding_models: NDArray[np.float64] | None = None
    unit_ids: NDArray[Any] | None = None
    # True only when ``fit`` generated ``arange`` labels for an unlabelled
    # input; such labels never drive label-based pairing.
    _unit_ids_generated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate config domain and (if injected) fitted-state coupling.

        Runs at construction and again on every :func:`~dataclasses.replace`
        (so :meth:`fit`'s returned decoder is validated too). The dataclass is
        frozen, so this only raises -- it never assigns fields. Reuses the core
        :func:`~neurospatial.decoding._binning.validate_dt` so ``dt`` semantics
        cannot drift from the decode entry points, and the encoder's
        :func:`~neurospatial.encoding._smoothing.validate_spatial_method_params`
        so ``method`` / ``bandwidth`` / ``min_occupancy`` / ``penalty`` / ``rank``
        errors mirror ``compute_spatial_rate`` exactly (one validator, no
        duplicate allow-set).
        """
        from neurospatial.decoding._binning import validate_dt
        from neurospatial.encoding._smoothing import validate_spatial_method_params

        # Config domain: dt (shared core validator) and dtype.
        validate_dt(self.dt)
        if self.dtype not in (np.float32, np.float64):
            raise ValueError(
                f"dtype must be np.float32 or np.float64, got {self.dtype!r}."
            )

        # Method-specific validation, reusing the encoder's validator so the
        # mutual-exclusivity + value-domain errors are byte-identical to
        # compute_spatial_rate's. fill_value is not a decoder param (pass None).
        validate_spatial_method_params(
            self.method,
            bandwidth=self.bandwidth,
            min_occupancy=self.min_occupancy,
            fill_value=None,
            penalty=self.penalty,
            rank=self.rank,
        )

        # Fitted-state coupling: only meaningful when encoding_models is set.
        # This closes the direct-injection backdoor
        # (``BayesianDecoder(env, encoding_models=arr)``): a hand-built fitted
        # decoder is now checked for the same invariants ``fit`` guarantees.
        if self.encoding_models is not None:
            if self.unit_ids is None:
                raise ValueError(
                    "encoding_models is set but unit_ids is None; a fitted "
                    "BayesianDecoder must carry one unit_id per encoding model. "
                    "Build fitted decoders via `.fit(...)` (which populates "
                    "unit_ids), or pass a matching `unit_ids` array."
                )
            models = np.asarray(self.encoding_models)
            if models.ndim != 2:
                raise ValueError(
                    f"encoding_models must be 2-D (n_units, n_bins), got "
                    f"{models.ndim}-D with shape {models.shape}."
                )
            if models.shape[1] != self.env.n_bins:
                raise ValueError(
                    f"encoding_models has {models.shape[1]} bins but env.n_bins "
                    f"is {self.env.n_bins}; the encoding model's bin axis must "
                    f"match the environment it decodes over."
                )
            # One distinct label per encoding model: label pairing in predict
            # relies on it. Runs for constructor labels and, via replace, for
            # the labels fit captures.
            from neurospatial._results import resolve_unit_ids

            resolve_unit_ids(self.unit_ids, models.shape[0], context="BayesianDecoder")

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has populated the encoding models.

        Returns
        -------
        bool
            ``True`` once :meth:`fit` has built the encoding models, ``False``
            for a freshly constructed (unfitted) decoder. Lets callers branch
            without catching the :class:`RuntimeError` that
            :meth:`predict` / :meth:`predict_summary` / :meth:`score` raise when
            unfitted.
        """
        return self.encoding_models is not None

    def _check_fitted(self) -> NDArray[np.float64]:
        """Return the fitted encoding models, or raise if unfitted.

        Returns
        -------
        NDArray[np.float64], shape (n_neurons, n_bins)
            The fitted encoding models.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called (``encoding_models is None``).
        """
        if self.encoding_models is None:
            raise RuntimeError(
                "BayesianDecoder is not fitted; call "
                "`.fit(spike_times, times, positions)` first."
            )
        return self.encoding_models

    def fit(
        self,
        spike_times: SpikeTrainsLike,
        times: ArrayLike,
        positions: NDArray[np.float64],
        *,
        unit_ids: NDArray[Any] | Sequence[Any] | None = None,
        speed: NDArray[np.float64] | None = None,
        min_speed: float | None = None,
        epochs: Any = None,
        spike_window: Any = None,
    ) -> BayesianDecoder:
        """Build encoding models from training data; return a new fitted decoder.

        Encodes the population's spatial rate maps using the same internal
        encoder :func:`~neurospatial.decoding.decode_session` runs
        (``_build_encoding_model``), so the models are byte-identical to
        ``decode_session``'s internal encode step. Does **not** mutate ``self``:
        the original decoder stays unfitted and a new frozen decoder carrying the
        fitted ``encoding_models`` + ``unit_ids`` is returned.

        Parameters
        ----------
        spike_times : SpikeTrainsLike
            Spike times for one or more units. Accepts the canonical array forms,
            a :class:`~neurospatial.encoding.SpikeTrains` container, or a pynapple
            ``TsGroup``-like group. A group's index becomes ``unit_ids`` and is
            later used to match predict-time spike trains to these models by
            label; an unlabelled input gets ``arange(n_units)``, and later inputs
            are then paired by position.
        times : array-like, shape (n_frames,)
            Timestamp array in seconds. Decode bins tile each run whose gaps
            are no longer than ``max_gap``. For spans without tracking, pass
            ``times=np.arange(t0, t1, dt)``. For pynapple, pass ``tsd.t``.
        positions : NDArray[np.float64], shape (n_frames, n_dims)
            Required sample-aligned coordinates. For pynapple, pass ``tsd.values``.
        unit_ids : ndarray or sequence, optional
            One distinct label per spike train. If the spike group carries labels,
            these must match exactly in the same order. Caller-supplied labels
            enable label alignment for labelled prediction inputs; generated
            ``arange`` labels pair by position.
        speed : NDArray[np.float64], shape (n_frames,), optional
            Precomputed speed, forwarded to the encoder. Only used when
            ``min_speed`` is set; auto-derived when ``None``.
        min_speed : float, optional
            Minimum speed threshold (position units / second). When set,
            low-speed samples are excluded from both the spike numerator and the
            occupancy denominator of the encoding model.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). An interval counts only if it lies entirely
            inside one window. ``None`` (default) means unrestricted.
        spike_window : same forms as ``epochs``, or None
            When the electrophysiology was recording. Intervals outside it are
            excluded from occupancy (and their spikes are not counted). ``None``
            (default) assumes spikes were recorded whenever position was; this is an
            assumption, not something the function checks. Pass it when tracking
            started before, or continued after, the spike recording.

        Returns
        -------
        BayesianDecoder
            A new fitted decoder with the same configuration plus
            ``encoding_models`` and ``unit_ids`` populated. The original is
            unchanged.

        Notes
        -----
        The default ``min_occupancy=None`` (resolving to ``0.0``, paired with the
        ratio decode golden path's ``fill_value=0.0``) means a small epoch or
        sparse training data can build a **degenerate low-coverage** encoding
        model *without erroring* -- most bins fall back to ``0.0`` Hz, so the fit
        "succeeds" but decodes poorly. For short epochs, raise ``min_occupancy``
        (ratio methods) or check the fitted model's spatial coverage before
        trusting a decode. ``method="glm"`` has no such knob -- occupancy enters
        as a log-offset, so every bin gets a finite rate.

        When predicting with the fitted decoder, decode time bins are formed
        separately within each run of samples whose
        gaps are no longer than ``max_gap`` and that lie inside ``epochs`` and
        ``spike_window``; no bin spans a pause, and spikes between runs are not
        counted. ``result.times`` may therefore be non-contiguous.

        Examples
        --------
        >>> decoder = BayesianDecoder(env).fit(  # doctest: +SKIP
        ...     spike_times, times, positions, epochs=(0.0, 60.0)
        ... )
        """
        from neurospatial._intervals import resolve_time_windows
        from neurospatial._results import resolve_unit_ids
        from neurospatial.decoding.session import _build_encoding_model
        from neurospatial.encoding._spikes import as_spike_trains_with_ids

        # Capture unit identity once, from the ORIGINAL spike input (temporal
        # restriction never changes which units exist, only their spike counts).
        trains, extracted_ids = as_spike_trains_with_ids(spike_times)
        resolved_ids = resolve_unit_ids(
            unit_ids,
            len(trains),
            input_ids=extracted_ids,
            context="BayesianDecoder.fit",
        )

        resolved_epochs, resolved_spike_window = resolve_time_windows(
            epochs, spike_window
        )

        firing_rates = _build_encoding_model(
            self.env,
            trains,
            times,
            positions,
            dt=self.dt,
            bandwidth=self.bandwidth,
            method=self.method,
            min_occupancy=self.min_occupancy,
            penalty=self.penalty,
            rank=self.rank,
            speed=speed,
            min_speed=min_speed,
            max_gap=self.max_gap,
            warn_on_drop=self.warn_on_drop,
            dtype=self.dtype,
            context="BayesianDecoder.fit",
            epochs=resolved_epochs,
            spike_window=resolved_spike_window,
        )[1]

        return replace(
            self,
            encoding_models=firing_rates,
            unit_ids=resolved_ids,
            _unit_ids_generated=unit_ids is None and extracted_ids is None,
        )

    def _align_to_fitted_units(
        self, spike_times: SpikeTrainsLike, caller: str
    ) -> list[NDArray[np.float64]]:
        """Pair each spike train with its encoding model.

        By label when both ``fit`` and this input carried caller-supplied labels;
        otherwise by position, which requires one train per fitted unit.
        """
        from neurospatial._results import resolve_unit_ids
        from neurospatial.encoding._spikes import as_spike_trains_with_ids

        trains, input_ids = as_spike_trains_with_ids(spike_times)
        n_models = self._check_fitted().shape[0]
        if input_ids is not None:
            labels = resolve_unit_ids(
                None,
                len(trains),
                input_ids=input_ids,
                context=f"BayesianDecoder.{caller}",
            ).tolist()
            if self.unit_ids is not None and not self._unit_ids_generated:
                fitted = np.asarray(self.unit_ids).tolist()
                row = {u: i for i, u in enumerate(labels)}
                fitted_set = set(fitted)
                missing = [u for u in fitted if u not in row]
                unexpected = [u for u in labels if u not in fitted_set]
                if missing or unexpected:
                    raise ValueError(
                        "Spike input unit labels do not match the decoder's fitted "
                        f"unit_ids (missing: {missing}, unexpected: {unexpected}).\n"
                        "Decoding would pair spike trains with the wrong encoding "
                        "models.\n"
                        "Fix: pass spikes for exactly the fitted units (pynapple: "
                        "group[list(decoder.unit_ids)]), or refit on this input."
                    )
                return [trains[row[u]] for u in fitted]
        if len(trains) != n_models:
            raise ValueError(
                f"Got {len(trains)} spike trains but the decoder was fitted with "
                f"{n_models} units. Without caller-supplied labels on both the fit "
                "and the predict input, trains are paired with encoding models by "
                "position.\n"
                "Fix: pass one spike train per fitted unit in the order used for "
                "fit, or fit and predict with the same labelled TsGroup."
            )
        return trains

    def predict(
        self,
        spike_times: SpikeTrainsLike,
        times: ArrayLike,
        *,
        epochs: Any = None,
        spike_window: Any = None,
    ) -> DecodingResult:
        """Decode the full posterior for new spikes against the fitted models.

        Uses the fitted rate maps with the same per-run binning and posterior
        calculation as :func:`~neurospatial.decoding.decode_session`.

        Parameters
        ----------
        spike_times : SpikeTrainsLike
            Spike times to decode. Same accepted forms as :meth:`fit`. When both
            this input and the fit carried caller-supplied labels (a labelled
            group, or ``unit_ids`` given at construction), trains are matched
            to encoding models by label, in any order. Otherwise they are paired
            by position: one train per fitted unit, in fit order.
        times : array-like, shape (n_frames,)
            Timestamp array in seconds. Decode bins tile each run whose gaps
            are no longer than ``max_gap``. For spans without tracking, pass
            ``times=np.arange(t0, t1, dt)``. For pynapple, pass ``tsd.t``.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). An interval counts only if it lies entirely
            inside one window. ``None`` (default) means unrestricted.
        spike_window : same forms as ``epochs``, or None
            When the electrophysiology was recording. Intervals outside it are
            excluded from occupancy (and their spikes are not counted). ``None``
            (default) assumes spikes were recorded whenever position was; this is an
            assumption, not something the function checks. Pass it when tracking
            started before, or continued after, the spike recording. The result
            records the window applied (``result.spike_window``) and whether it was
            assumed (``result.spike_window_assumed``).

        Returns
        -------
        DecodingResult
            Full posterior over positions per decode time bin (shape
            ``(n_time_bins, n_bins)``) plus MAP / mean / entropy accessors.

        Raises
        ------
        RuntimeError
            If the decoder is unfitted.
        ValueError
            If the spike input repeats a unit label; if label matching applies
            and the input's labels differ from ``unit_ids`` (the message lists
            the missing and unexpected labels); or if trains are paired by
            position and their number differs from the number of fitted units.

        Notes
        -----
        Decode time bins are formed separately within each run of samples whose
        gaps are no longer than ``max_gap`` and that lie inside ``epochs`` and
        ``spike_window``; no bin spans a pause, and spikes between runs are not
        counted. ``result.times`` may therefore be non-contiguous.
        """
        from neurospatial.decoding.session import _decode_with_models

        encoding_models = self._check_fitted()
        return _decode_with_models(
            self.env,
            self._align_to_fitted_units(spike_times, "predict"),
            times,
            encoding_models,
            dt=self.dt,
            warn_on_drop=self.warn_on_drop,
            dtype=self.dtype,
            max_gap=self.max_gap,
            epochs=epochs,
            spike_window=spike_window,
        )

    def predict_summary(
        self,
        spike_times: SpikeTrainsLike,
        times: ArrayLike,
        *,
        epochs: Any = None,
        spike_window: Any = None,
        time_chunk: int = 1024,
    ) -> DecodingSummary:
        """Decode memory-safe per-time reductions for new spikes.

        Uses the fitted rate maps and streams time-binning, reducing the
        posterior block-by-block, so the full ``(n_time, n_bins)`` posterior is
        never materialized. The MAP estimate equals :meth:`predict`'s.

        Parameters
        ----------
        spike_times : SpikeTrainsLike
            Spike times to decode. Same accepted forms as :meth:`fit`. When both
            this input and the fit carried caller-supplied labels (a labelled
            group, or ``unit_ids`` given at construction), trains are matched
            to encoding models by label, in any order. Otherwise they are paired
            by position: one train per fitted unit, in fit order.
        times : array-like, shape (n_frames,)
            Timestamp array in seconds. Decode bins tile each run whose gaps
            are no longer than ``max_gap``. For spans without tracking, pass
            ``times=np.arange(t0, t1, dt)``. For pynapple, pass ``tsd.t``.
        time_chunk : int, default=1024
            Streaming block size (number of time bins per block). Must be a
            positive integer.

        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). An interval counts only if it lies entirely
            inside one window. ``None`` (default) means unrestricted.
        spike_window : same forms as ``epochs``, or None
            When the electrophysiology was recording. Intervals outside it are
            excluded from occupancy (and their spikes are not counted). ``None``
            (default) assumes spikes were recorded whenever position was; this is an
            assumption, not something the function checks. Pass it when tracking
            started before, or continued after, the spike recording. The result
            records the window applied (``result.spike_window``) and whether it was
            assumed (``result.spike_window_assumed``).

        Returns
        -------
        DecodingSummary
            Per-time reductions (MAP position/bin, mean position, entropy, peak
            probability) plus ``times`` and ``env``.

        Raises
        ------
        RuntimeError
            If the decoder is unfitted.
        ValueError
            If the spike input repeats a unit label; if label matching applies
            and the input's labels differ from ``unit_ids`` (the message lists
            the missing and unexpected labels); or if trains are paired by
            position and their number differs from the number of fitted units.

        Notes
        -----
        Decode time bins are formed separately within each run of samples whose
        gaps are no longer than ``max_gap`` and that lie inside ``epochs`` and
        ``spike_window``; no bin spans a pause, and spikes between runs are not
        counted. ``result.times`` may therefore be non-contiguous.
        """
        from neurospatial.decoding.session import _decode_with_models_summary

        encoding_models = self._check_fitted()
        return _decode_with_models_summary(
            self.env,
            self._align_to_fitted_units(spike_times, "predict_summary"),
            times,
            encoding_models,
            dt=self.dt,
            warn_on_drop=self.warn_on_drop,
            dtype=self.dtype,
            max_gap=self.max_gap,
            epochs=epochs,
            spike_window=spike_window,
            time_chunk=time_chunk,
        )

    def score(
        self,
        spike_times: SpikeTrainsLike,
        times: ArrayLike,
        positions: NDArray[np.float64],
        *,
        epochs: Any = None,
        spike_window: Any = None,
        metric: str = "median_error",
        distance: str = "euclidean",
    ) -> float:
        """Decode and report position error against ground truth (lower is better).

        Decodes ``spike_times`` over ``times`` (via :meth:`predict`), aligns the
        MAP estimate to the ground-truth track (``times``, ``positions``) with
        :meth:`~neurospatial.decoding.DecodingResult.error_against`, and reduces
        the per-time-bin errors to a single scalar. **Lower is better.**

        Undecodable decode time bins -- those whose posterior row is entirely
        non-finite -- are stamped ``nan`` by ``error_against`` and are
        **excluded** from the reduction. Because ``nanmedian`` / ``nanmean``
        silently ignore them, a decoder that decodes only part of the session
        could otherwise report a misleadingly good score; this method therefore
        **warns** (naming the dropped fraction) when any bin is dropped and
        **raises** (rather than returning ``nan``) when *no* bin is decodable.
        The all-decodable path stays warning-free.

        Parameters
        ----------
        spike_times : SpikeTrainsLike
            Spike times to decode. Same accepted forms as :meth:`fit`. When both
            this input and the fit carried caller-supplied labels (a labelled
            group, or ``unit_ids`` given at construction), trains are matched
            to encoding models by label, in any order. Otherwise they are paired
            by position: one train per fitted unit, in fit order.
        times : array-like, shape (n_frames,)
            Timestamp array in seconds. Decode bins tile each run whose gaps
            are no longer than ``max_gap``. For spans without tracking, pass
            ``times=np.arange(t0, t1, dt)``. For pynapple, pass ``tsd.t``.
        positions : NDArray[np.float64], shape (n_frames, n_dims)
            Required sample-aligned coordinates. For pynapple, pass ``tsd.values``.
        metric : {"median_error", "mean_error"}, default="median_error"
            Reduction over the per-time-bin errors. ``"median_error"`` ->
            ``nanmedian``; ``"mean_error"`` -> ``nanmean``. Lower is better.
        distance : {"euclidean", "geodesic"}, default="euclidean"
            Distance metric forwarded to ``error_against`` for the per-time-bin
            error. ``"geodesic"`` uses the environment's connectivity graph
            (shortest-path along the track / masked field), the differentiator
            over straight-line euclidean error; ``"euclidean"`` is the default.

        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). An interval counts only if it lies entirely
            inside one window. ``None`` (default) means unrestricted.
        spike_window : same forms as ``epochs``, or None
            When the electrophysiology was recording. Intervals outside it are
            excluded from occupancy (and their spikes are not counted). ``None``
            (default) assumes spikes were recorded whenever position was; this is an
            assumption, not something the function checks. Pass it when tracking
            started before, or continued after, the spike recording. The result
            records the window applied (``result.spike_window``) and whether it was
            assumed (``result.spike_window_assumed``).

        Returns
        -------
        float
            The reduced decode error (environment units, e.g. cm). Lower is
            better.

        Raises
        ------
        RuntimeError
            If the decoder is unfitted.
        ValueError
            If ``metric`` is not ``"median_error"`` / ``"mean_error"``, if
            ``distance`` is not ``"euclidean"`` / ``"geodesic"`` (both checked
            **before** decoding, so a typo does not cost a full decode), or if
            **no** decode time bin was decodable (every posterior row was
            non-finite) -- likely a degenerate/empty encoding model from too few
            training samples, or a spikes/times unit mismatch (seconds vs
            milliseconds). Also raised when the spike input cannot be paired
            with the fitted units (see :meth:`predict`).

        Notes
        -----
        Decode time bins are formed separately within each run of samples whose
        gaps are no longer than ``max_gap`` and that lie inside ``epochs`` and
        ``spike_window``; no bin spans a pause, and spikes between runs are not
        counted. ``result.times`` may therefore be non-contiguous.
        """
        from neurospatial._validation import validate_times_positions

        # Validate the reduction (`metric`) and error metric (`distance`) up
        # front, BEFORE any decode -- a typo should raise cheaply, not after a
        # full decode pass.
        if metric not in ("median_error", "mean_error"):
            raise ValueError(
                f"Unknown metric {metric!r}; `metric` selects the reduction and "
                f"must be one of 'median_error', 'mean_error'."
            )
        if distance not in ("euclidean", "geodesic"):
            raise ValueError(
                f"Unknown distance {distance!r}; `distance` selects the error "
                f"metric and must be one of 'euclidean', 'geodesic'."
            )

        self._check_fitted()

        times_arr, positions_arr = validate_times_positions(
            times, positions, call="BayesianDecoder.score"
        )

        result = self.predict(
            spike_times, times_arr, epochs=epochs, spike_window=spike_window
        )
        # `distance` is validated above, so narrowing it to the Literal the
        # DecodingResult expects is safe.
        errors = result.error_against(
            times_arr,
            positions_arr,
            metric=cast("Literal['euclidean', 'geodesic']", distance),
        )

        # Guard against silently reducing over dropped bins. error_against stamps
        # nan for every undecodable decode time bin; nanmedian/nanmean would
        # ignore those without a trace, biasing model selection toward decoders
        # that decode fewer bins.
        n = int(errors.size)
        n_bad = int(np.isnan(errors).sum())
        if n == 0 or n_bad == n:
            raise ValueError(
                "score() could not decode any time bin: every decode time bin's "
                "posterior was entirely non-finite (undecodable), so there is no "
                "error to reduce. Likely causes: a degenerate/empty encoding "
                "model built from too few training samples, or a spikes/times "
                "unit mismatch (e.g. spike_times in milliseconds while times is "
                "in seconds). Check the encoding-model coverage and that "
                "spike_times and times share units (both seconds)."
            )
        if n_bad > 0:
            warnings.warn(
                f"{n_bad}/{n} decode time bins were undecodable and excluded "
                f"from the score.",
                UserWarning,
                stacklevel=2,
            )

        if metric == "median_error":
            return float(np.nanmedian(errors))
        return float(np.nanmean(errors))
