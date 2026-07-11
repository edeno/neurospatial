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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from neurospatial._typing import PositionLike, SpikeTrainsLike
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
    bandwidth : float, default=5.0
        KDE smoothing bandwidth (position units) for the encoding step.
    method : str, default="diffusion_kde"
        KDE method for the encoding step. One of ``"diffusion_kde"``,
        ``"gaussian_kde"``, ``"binned"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to enter the encoding model.
        Low-occupancy bins are filled with ``0.0`` Hz (``fill_value=0.0``), never
        ``NaN``.
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
        Identity label per encoding model (introspection only; the posterior
        carries no unit axis). Populated by :meth:`fit` from the spike input
        (``arange(n_neurons)`` when the input carries no ids).

    Attributes
    ----------
    env, dt, bandwidth, method, min_occupancy, max_gap, dtype, \
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
        is not ``np.float32`` / ``np.float64``, or (when ``encoding_models`` is
        injected directly) if the fitted state is inconsistent (missing
        ``unit_ids``, wrong ndim, or a bin/unit-count mismatch against ``env``).
    RuntimeError
        From :meth:`predict` / :meth:`predict_summary` / :meth:`score` if the
        decoder is unfitted.

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
    bandwidth: float = 5.0
    method: str = "diffusion_kde"
    min_occupancy: float = 0.0
    max_gap: float | None = 0.5
    dtype: type[np.float32] | type[np.float64] = np.float64
    warn_on_drop: bool = True
    # Fitted state (private; ``None`` => unfitted). Set only via :meth:`fit`.
    encoding_models: NDArray[np.float64] | None = None
    unit_ids: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        """Validate config domain and (if injected) fitted-state coupling.

        Runs at construction and again on every :func:`~dataclasses.replace`
        (so :meth:`fit`'s returned decoder is validated too). The dataclass is
        frozen, so this only raises -- it never assigns fields. Reuses the core
        :func:`~neurospatial.decoding._binning.validate_dt` so ``dt`` semantics
        cannot drift from the decode entry points; ``method`` is left
        to :meth:`fit` / ``compute_spatial_rates`` (no duplicate allow-set).
        """
        from neurospatial.decoding._binning import validate_dt

        # Config domain: dt (shared core validator) and dtype.
        validate_dt(self.dt)
        if self.dtype not in (np.float32, np.float64):
            raise ValueError(
                f"dtype must be np.float32 or np.float64, got {self.dtype!r}."
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
            if models.shape[0] != len(self.unit_ids):
                raise ValueError(
                    f"encoding_models has {models.shape[0]} units but unit_ids "
                    f"has length {len(self.unit_ids)}; there must be exactly one "
                    f"unit_id per encoding model."
                )

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
        times: ArrayLike | PositionLike,
        positions: NDArray[np.float64] | None = None,
        *,
        speed: NDArray[np.float64] | None = None,
        min_speed: float | None = None,
        epoch: Any = None,
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
            ``TsGroup``-like group (its ``unit_ids`` are captured).
        times : array-like, shape (n_frames,), or PositionLike
            Training timestamps (seconds), or a ``PositionLike`` object carrying
            both times and positions (then ``positions`` must be omitted).
        positions : NDArray[np.float64], shape (n_frames, n_dims), optional
            Training positions. Omit only when ``times`` is a ``PositionLike``.
        speed : NDArray[np.float64], shape (n_frames,), optional
            Precomputed speed, forwarded to the encoder. Only used when
            ``min_speed`` is set; auto-derived when ``None``.
        min_speed : float, optional
            Minimum speed threshold (position units / second). When set,
            low-speed samples are excluded from both the spike numerator and the
            occupancy denominator of the encoding model.
        epoch : IntervalSet-like, tuple, list, or ndarray, optional
            When given, the training data is restricted to these epochs *first*
            (via :func:`neurospatial.behavior.restrict` for the position track
            and :func:`neurospatial.behavior.restrict_spike_trains` for spikes),
            then encoded. Enables train/test splits.

        Returns
        -------
        BayesianDecoder
            A new fitted decoder with the same configuration plus
            ``encoding_models`` and ``unit_ids`` populated. The original is
            unchanged.

        Notes
        -----
        The default ``min_occupancy=0.0`` (paired with the decode golden path's
        ``fill_value=0.0``) means a small ``epoch`` or sparse training data can
        build a **degenerate low-coverage** encoding model *without erroring* --
        most bins fall back to ``0.0`` Hz, so the fit "succeeds" but decodes
        poorly. For short epochs, raise ``min_occupancy`` or check the fitted
        model's spatial coverage before trusting a decode.

        Examples
        --------
        >>> decoder = BayesianDecoder(env).fit(  # doctest: +SKIP
        ...     spike_times, times, positions, epoch=(0.0, 60.0)
        ... )
        """
        from neurospatial._typing import as_times_positions
        from neurospatial.decoding.session import _build_encoding_model
        from neurospatial.encoding import as_spike_trains_with_ids

        # Capture unit identity once, from the ORIGINAL spike input (temporal
        # restriction never changes which units exist, only their spike counts).
        trains, unit_ids = as_spike_trains_with_ids(spike_times)

        if epoch is not None:
            from neurospatial.behavior import restrict, restrict_spike_trains

            # Restrict the training data to the epoch BEFORE encoding. Normalize
            # the position track to arrays first (restrict needs a shared time
            # axis); restrict the already-normalized trains (never the raw group
            # -- iterating a TsGroup would yield ids, not trains). ``speed`` (if
            # given) is time-aligned to ``times``, so it must be sliced by the
            # SAME epoch mask -- otherwise the full-length speed reaches the
            # encoder with the restricted times and raises a length mismatch.
            times, positions = as_times_positions(times, positions)
            if speed is not None:
                speed_arr = np.asarray(speed, dtype=np.float64)
                times, positions, speed = restrict(
                    times, positions, speed_arr, epochs=epoch
                )
            else:
                times, positions = restrict(times, positions, epochs=epoch)
            spike_input: Any = restrict_spike_trains(trains, epoch)
        else:
            # No epoch: reuse the trains already coerced above rather than
            # re-coercing the raw input inside the encoder. Re-coercing a
            # list[float64 array] is a no-op, so the encode stays byte-for-byte
            # identical to decode_session's (and to the epoch branch above).
            spike_input = trains

        firing_rates = _build_encoding_model(
            self.env,
            spike_input,
            times,
            positions,
            dt=self.dt,
            bandwidth=self.bandwidth,
            method=self.method,
            min_occupancy=self.min_occupancy,
            speed=speed,
            min_speed=min_speed,
            max_gap=self.max_gap,
            encoding_models=None,
            warn_on_drop=self.warn_on_drop,
            dtype=self.dtype,
            context="BayesianDecoder.fit",
        )[1]

        if unit_ids is None:
            unit_ids = np.arange(firing_rates.shape[0])

        return replace(self, encoding_models=firing_rates, unit_ids=unit_ids)

    def predict(
        self,
        spike_times: SpikeTrainsLike,
        times: ArrayLike | PositionLike,
    ) -> DecodingResult:
        """Decode the full posterior for new spikes against the fitted models.

        Delegates to :func:`~neurospatial.decoding.decode_session` with the
        fitted ``encoding_models``, so the encode step is skipped and the
        posterior is computed directly from those models.

        Parameters
        ----------
        spike_times : SpikeTrainsLike
            Spike times to decode. Same accepted forms as :meth:`fit`.
        times : array-like, shape (n_frames,), or PositionLike
            Timestamps defining the decode window ``[min, max]``; a
            ``PositionLike`` is accepted (its positions are ignored for decode).

        Returns
        -------
        DecodingResult
            Full posterior over positions per decode time bin (shape
            ``(n_time_bins, n_bins)``) plus MAP / mean / entropy accessors.

        Raises
        ------
        RuntimeError
            If the decoder is unfitted.
        """
        from neurospatial.decoding.session import decode_session

        encoding_models = self._check_fitted()
        return decode_session(
            self.env,
            spike_times,
            times,
            positions=None,
            dt=self.dt,
            encoding_models=encoding_models,
            warn_on_drop=self.warn_on_drop,
            dtype=self.dtype,
        )

    def predict_summary(
        self,
        spike_times: SpikeTrainsLike,
        times: ArrayLike | PositionLike,
        *,
        time_chunk: int = 1024,
    ) -> DecodingSummary:
        """Decode memory-safe per-time reductions for new spikes.

        Delegates to :func:`~neurospatial.decoding.decode_session_summary` with
        the fitted ``encoding_models``. Streams the time-binning and reduces the
        posterior block-by-block, so the full ``(n_time, n_bins)`` posterior is
        never materialized. The MAP estimate equals :meth:`predict`'s.

        Parameters
        ----------
        spike_times : SpikeTrainsLike
            Spike times to decode. Same accepted forms as :meth:`fit`.
        times : array-like, shape (n_frames,), or PositionLike
            Timestamps defining the decode window.
        time_chunk : int, default=1024
            Streaming block size (number of time bins per block). Must be a
            positive integer.

        Returns
        -------
        DecodingSummary
            Per-time reductions (MAP position/bin, mean position, entropy, peak
            probability) plus ``times`` and ``env``.

        Raises
        ------
        RuntimeError
            If the decoder is unfitted.
        """
        from neurospatial.decoding.session import decode_session_summary

        encoding_models = self._check_fitted()
        return decode_session_summary(
            self.env,
            spike_times,
            times,
            positions=None,
            dt=self.dt,
            encoding_models=encoding_models,
            warn_on_drop=self.warn_on_drop,
            dtype=self.dtype,
            time_chunk=time_chunk,
        )

    def score(
        self,
        spike_times: SpikeTrainsLike,
        times: ArrayLike | PositionLike,
        positions: NDArray[np.float64] | None = None,
        *,
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
            Spike times to decode. Same accepted forms as :meth:`fit`.
        times : array-like, shape (n_frames,), or PositionLike
            Ground-truth timestamps (seconds), or a ``PositionLike`` carrying
            both times and positions (then ``positions`` must be omitted).
        positions : NDArray[np.float64], shape (n_frames, n_dims), optional
            Ground-truth positions to score against. Omit only when ``times`` is
            a ``PositionLike``.
        metric : {"median_error", "mean_error"}, default="median_error"
            Reduction over the per-time-bin errors. ``"median_error"`` ->
            ``nanmedian``; ``"mean_error"`` -> ``nanmean``. Lower is better.
        distance : {"euclidean", "geodesic"}, default="euclidean"
            Distance metric forwarded to ``error_against`` for the per-time-bin
            error. ``"geodesic"`` uses the environment's connectivity graph
            (shortest-path along the track / masked field), the differentiator
            over straight-line euclidean error; ``"euclidean"`` is the default.

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
            milliseconds).
        """
        from neurospatial._typing import as_times_positions

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

        # Normalize the ground-truth track to arrays so a PositionLike scores
        # like the explicit (times, positions) pair.
        times_arr, positions_arr = as_times_positions(times, positions)

        result = self.predict(spike_times, times_arr)
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
