"""Immutable ``BayesianDecoder`` fit/predict wrapper over the decode core.

Provides :class:`BayesianDecoder`, a thin **frozen** convenience layer over the
existing functional decoders (:func:`~neurospatial.decoding.decode_session`,
:func:`~neurospatial.decoding.decode_session_summary`). The wrapper does **not**
re-implement decoding: ``fit`` reuses the same internal encoder
(``decode_session``'s ``_fit_encoding_model``) that ``decode_session`` runs
internally, and ``predict``/``predict_summary`` delegate straight to the
functional core with the fitted encoding models. Reusing that internal encoder
is what guarantees the encoding step is byte-identical to ``decode_session``'s
(same ``max_gap`` masking, dtype, KDE parameters), so a
fitted decoder's posterior is byte-for-byte equal to ``decode_session`` on the
same inputs.

Unlike pynapple's ``decode_1d`` / ``decode_2d``, decoding runs through an
:class:`~neurospatial.environment.Environment`, so geodesic / linearized-track /
graph-based decoding works: the same fit/predict flow decodes a 1-D linearized
track or a masked open field, not just a rectangular grid.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import DEFAULT_MAX_GAP

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from neurospatial._typing import SpikeTrainsLike
    from neurospatial.decoding._result import DecodingResult, DecodingSummary
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment

__all__ = ["BayesianDecoder"]

_ENCODING_FIELDS = ("method", "bandwidth", "min_occupancy", "penalty", "rank")
_EncodingConfig = tuple[str, float | None, float | None, float | None, int | None]


@dataclass(frozen=True, eq=False)
class _FittedState:
    """A fitted decoder's encoding models with their identity and provenance.

    Holds read-only copies, so editing the arrays a decoder was built from
    cannot change it. Built by :meth:`BayesianDecoder.fit`,
    :meth:`BayesianDecoder.from_rates` and :meth:`BayesianDecoder.from_models`;
    the decoder refuses an environment whose bins differ from ``env``.
    """

    encoding_models: NDArray[np.float64]
    unit_ids: NDArray[Any] | Sequence[Any] | None
    # True when ``arange`` labels were generated for an unlabelled input; such
    # labels never drive label-based pairing.
    unit_ids_generated: bool
    # Environment the models are defined on; a fitted decoder keeps it.
    env: Environment
    spatial_support: NDArray[np.bool_] | None = None
    spike_window: NDArray[np.float64] | None = None
    analysis_settings: dict[str, Any] | None = None
    encoding_config: _EncodingConfig = ("diffusion_kde", None, None, None, None)

    def __post_init__(self) -> None:
        """Snapshot the arrays and validate the models and their labels."""
        from neurospatial._results import read_only_copy, resolve_unit_ids

        models = read_only_copy(self.encoding_models)
        if models.ndim != 2:
            raise ValueError(
                f"encoding_models must be 2-D (n_units, n_bins), got "
                f"{models.ndim}-D with shape {models.shape}."
            )
        object.__setattr__(self, "encoding_models", models)
        # One distinct label per encoding model: label pairing relies on it.
        object.__setattr__(
            self,
            "unit_ids",
            resolve_unit_ids(self.unit_ids, models.shape[0], context="BayesianDecoder"),
        )
        if self.spatial_support is not None:
            object.__setattr__(
                self, "spatial_support", read_only_copy(self.spatial_support)
            )
        if self.spike_window is not None:
            object.__setattr__(
                self, "spike_window", read_only_copy(self.spike_window, np.float64)
            )
        object.__setattr__(
            self, "analysis_settings", copy.deepcopy(self.analysis_settings)
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle or deepcopy, keeping the arrays read-only."""
        self.__dict__.update(state)
        for value in state.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False


@dataclass(frozen=True)
class BayesianDecoder:
    """Immutable Bayesian position decoder over the functional decode core.

    A thin, frozen wrapper that pairs an :class:`~neurospatial.environment.Environment`
    and a set of encoding parameters with (once fitted) a population of encoding
    models. The constructor takes configuration only. :meth:`fit` builds the
    encoding models and returns a **new** fitted decoder;
    :meth:`from_rates` wraps rates from :func:`compute_spatial_rates`, and
    :meth:`from_models` wraps hand-built rate maps such as a simulation's
    ground truth. :meth:`predict` / :meth:`predict_summary` decode fresh spikes
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
        Spatial environment defining the bin layout and connectivity.
    dt : float, default=0.025
        Decoding time-bin width in seconds.
    bandwidth : float or None, default=None
        Smoothing bandwidth (position units) for the ratio-method encoding step.
        ``None`` uses 5 cm in declared metric units, otherwise 2.5 median bin
        spacings; a ratio-only param, so it
        must stay ``None`` when ``method="glm"`` (validated at construction).
    method : str, default="diffusion_kde"
        Estimator for the encoding step. One of ``"diffusion_kde"``,
        ``"gaussian_kde"``, ``"binned"``, or ``"glm"`` (penalized-Poisson GAM,
        tuned with ``penalty`` / ``rank``).
    min_occupancy : float or None, default=None
        Minimum occupancy (seconds) for a bin to enter the ratio-method encoding
        model. Unoccupied bins and bins below this threshold are excluded
        from the posterior by ``spatial_support``. ``None`` resolves to 0.0; a
        ratio-only param, so it must stay ``None`` when ``method="glm"``.
    penalty : float or None, default=None
        ``method="glm"`` smoothness penalty ``lambda``; ``None`` chooses it by
        REML. Mutually exclusive with the ratio params (``bandwidth`` /
        ``min_occupancy``); validated at construction.
    rank : int or None, default=None
        ``method="glm"`` requested basis rank cap; ``None`` uses the encoder
        default.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    dtype : {np.float32, np.float64}, default=np.float64
        Working dtype for the encoding models and the posterior. ``np.float32``
        halves the working set; ``np.float64`` (default) is byte-for-byte the
        reference decode.
        Replacing this on a fitted decoder changes prediction precision;
        the stored encoding models retain their fitted precision.
    warn_on_drop : bool, default=True
        Forwarded to the encode step (:meth:`fit`) and the decode step
        (:meth:`predict` / :meth:`predict_summary`). When ``True`` (default) a
        single ``UserWarning`` fires if a large fraction of spikes fall outside
        the decode/encode time window (the seconds-vs-milliseconds footgun),
        and prediction warns when more than 10% of the observed time falls in
        run remainders shorter than ``dt``, which are not decoded.
        Set ``False`` to silence those warnings on a genuinely sparse session;
        a single knob covers both the fit and predict passes (the default would
        otherwise warn once per pass).

    Attributes
    ----------
    env, dt, bandwidth, method, min_occupancy, penalty, rank, max_gap, dtype, \
warn_on_drop
        The configuration passed at construction (immutable).
        On a fitted decoder, ``method``, ``bandwidth``, ``min_occupancy``,
        ``penalty`` and ``rank`` cannot change via ``dataclasses.replace``,
        and ``env`` can only be replaced by an equivalent copy. Construct a
        new unfitted decoder to fit with different encoding settings or on
        another environment. Prediction defaults (``dt``, ``max_gap``, ``dtype`` and
        ``warn_on_drop``) can change; training provenance stays unchanged.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins), or None
        The fitted firing-rate maps (a read-only copy), or ``None`` when
        unfitted; :meth:`predict` / :meth:`predict_summary` / :meth:`score`
        raise until the decoder is fitted.
    unit_ids : NDArray or None
        Identity label per encoding model (read-only), or ``None`` when
        unfitted. :meth:`fit` takes them from a labelled spike input such as a
        pynapple ``TsGroup`` (its index) or from ``fit(unit_ids=...)``, and
        generates ``arange(n_neurons)`` otherwise. Caller-supplied labels
        require a labelled spike input in :meth:`predict` /
        :meth:`predict_summary` / :meth:`score`, matched by label; generated
        labels require plain trains, paired by position in fit order.
    spatial_support : ndarray of bool or None
        Training bins allowed in the posterior, from observed occupancy and the
        occupancy threshold. ``None`` when unfitted, and for
        :meth:`from_models` decoders built without one.
    analysis_settings : dict or None
        Recorded training settings, including observed intervals and gates.
    spike_window : ndarray or None
        Spike-recording windows carried from training or precomputed rates as
        provenance. Prediction takes its own explicit observation window.
    is_fitted : bool
        ``True`` once the decoder holds encoding models.

    Raises
    ------
    ValueError
        At construction, if ``dt`` is not a finite ``> 0`` number, if ``dtype``
        is not ``np.float32`` / ``np.float64``, or if the method-specific
        params are invalid (an unknown ``method``; a ratio param with
        ``method="glm"`` or a glm param with a ratio method; an out-of-domain
        ``penalty`` / ``rank``) -- these mirror ``compute_spatial_rate``.
        Also if encoding settings change on a fitted decoder via ``replace``.
    IncompatibleEnvironmentError
        If ``env`` is replaced on a fitted decoder by an environment whose
        bins or connectivity differ from the training environment.
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
    bandwidth: float | None = None
    method: str = "diffusion_kde"
    min_occupancy: float | None = None
    penalty: float | None = None
    rank: int | None = None
    max_gap: float | None = DEFAULT_MAX_GAP
    dtype: type[np.float32] | type[np.float64] = np.float64
    warn_on_drop: bool = True
    # Fitted state; ``None`` marks an unfitted decoder. Built by fit,
    # from_rates and from_models, and carried through dataclasses.replace.
    _fit: _FittedState | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        """Validate the configuration and any fitted state against ``env``.

        Runs at construction and again on every :func:`~dataclasses.replace`
        (so :meth:`fit`'s returned decoder is validated too). Fitted state is
        stored as read-only copies, so editing the arrays a decoder was built
        from (for example ``rates.firing_rates`` after :meth:`from_rates`)
        cannot change its predictions. Reuses the core
        :func:`~neurospatial.decoding._time_bins.validate_dt` so ``dt`` semantics
        cannot drift from the decode entry points, and the encoder's
        :func:`~neurospatial.encoding._validate_spatial_method_params`
        so ``method`` / ``bandwidth`` / ``min_occupancy`` / ``penalty`` / ``rank``
        errors mirror ``compute_spatial_rate`` exactly (one validator, no
        duplicate allow-set).
        """
        from neurospatial.decoding._time_bins import validate_dt
        from neurospatial.encoding import _validate_spatial_method_params

        # Config domain: dt (shared core validator) and dtype.
        validate_dt(self.dt)
        if self.dtype not in (np.float32, np.float64):
            raise ValueError(
                f"dtype must be np.float32 or np.float64, got {self.dtype!r}."
            )

        if self._fit is not None:
            for name, fitted_value, current_value in zip(
                _ENCODING_FIELDS,
                self._fit.encoding_config,
                self._encoding_config(),
                strict=True,
            ):
                if current_value != fitted_value:
                    raise ValueError(
                        f"{name} cannot change on a fitted decoder: changing "
                        "encoding settings would not refit the stored models. "
                        "Fix: construct a new BayesianDecoder with the desired "
                        "encoding settings and call fit()."
                    )

        # Method-specific validation, reusing the encoder's validator so the
        # mutual-exclusivity + value-domain errors are byte-identical to
        # compute_spatial_rate's.
        _validate_spatial_method_params(
            self.method,
            bandwidth=self.bandwidth,
            min_occupancy=self.min_occupancy,
            penalty=self.penalty,
            rank=self.rank,
        )

        # The models' bin axis must match the environment they decode over.
        if self._fit is not None:
            self._check_fitted_environment()
            n_model_bins = self._fit.encoding_models.shape[1]
            if n_model_bins != self.env.n_bins:
                raise ValueError(
                    f"encoding_models has {n_model_bins} bins but env.n_bins "
                    f"is {self.env.n_bins}; the encoding model's bin axis must "
                    f"match the environment it decodes over."
                )
            from neurospatial.decoding._support import validate_support

            if self._fit.spatial_support is not None:
                validate_support(self._fit.spatial_support, self.env.n_bins)

    def _check_fitted_environment(self) -> None:
        """Refuse an environment whose bins differ from the fitted one."""
        assert self._fit is not None
        if self.env is self._fit.env:
            return
        from neurospatial._exceptions import IncompatibleEnvironmentError
        from neurospatial.environment import _environment_differences

        bins, topology = _environment_differences(self._fit.env, self.env)
        mismatch = bins or topology
        if mismatch is not None:
            raise IncompatibleEnvironmentError(
                "env cannot change on a fitted decoder: the encoding models "
                f"were fitted on an environment that differs in {mismatch}.",
                fix="fit a new BayesianDecoder(new_env, ...) on the training "
                "data, or use BayesianDecoder.from_models(new_env, models) for "
                "rate maps defined on new_env.",
                first=self._fit.env,
                second=self.env,
            )

    def _encoding_config(self) -> _EncodingConfig:
        """Snapshot requested encoding settings, before defaults are resolved."""
        return self.method, self.bandwidth, self.min_occupancy, self.penalty, self.rank

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore fitted configuration snapshots, including legacy pickles."""
        self.__dict__.update(state)
        if self._fit is not None and "encoding_config" not in vars(self._fit):
            object.__setattr__(self._fit, "encoding_config", self._encoding_config())

    @property
    def encoding_models(self) -> NDArray[np.float64] | None:
        """Fitted ``(n_neurons, n_bins)`` rate maps, or ``None`` when unfitted."""
        return None if self._fit is None else self._fit.encoding_models

    @property
    def unit_ids(self) -> NDArray[Any] | None:
        """Identity label per encoding model, or ``None`` when unfitted."""
        return None if self._fit is None else np.asarray(self._fit.unit_ids)

    @property
    def spatial_support(self) -> NDArray[np.bool_] | None:
        """Training bins allowed in the posterior, or ``None``."""
        return None if self._fit is None else self._fit.spatial_support

    @property
    def spike_window(self) -> NDArray[np.float64] | None:
        """Training spike-recording windows (provenance), or ``None``."""
        return None if self._fit is None else self._fit.spike_window

    @property
    def analysis_settings(self) -> dict[str, Any] | None:
        """An independent copy of recorded training settings, or ``None``."""
        return None if self._fit is None else copy.deepcopy(self._fit.analysis_settings)

    @property
    def _unit_ids_generated(self) -> bool:
        """Whether the labels were generated, so pairing stays positional."""
        return self._fit is not None and self._fit.unit_ids_generated

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

    @classmethod
    def from_rates(
        cls, rates: SpatialRatesResult, *, dt: float = 0.025
    ) -> BayesianDecoder:
        """Create a fitted decoder from existing spatial population rate maps.

        Parameters
        ----------
        rates : SpatialRatesResult
            Result from ``compute_spatial_rates``. Carries the environment,
            rate maps, unit labels and recorded spike-observation windows.
        dt : float, default=0.025
            Decode bin width in seconds.

        Returns
        -------
        BayesianDecoder
            Frozen fitted decoder; call ``predict`` or ``predict_summary`` with
            spikes and timestamp arrays, without supplying tracking positions.

        Raises
        ------
        TypeError
            If ``rates`` is not a ``SpatialRatesResult``.
        ValueError
            If ``dt`` is not a valid positive finite bin width.

        Notes
        -----
        Caller-supplied labels require labelled prediction inputs, matched by
        identity; generated row numbers require plain trains in the same order. The spike window is
        training provenance; pass the prediction recording's ``spike_window``
        explicitly when decoding. Observed occupancy, the recorded occupancy
        threshold and finite rates determine ``spatial_support``. Unsupported
        bins have zero posterior probability; maps are not recomputed.
        Encoding configuration comes from the rates' method, bandwidth and
        recorded settings. ``analysis_settings`` retains the complete training
        choices; unavailable legacy settings remain ``None``. For GLM rates,
        ``penalty=None`` denotes automatic selection, whose fitted penalties
        remain on the rate result.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment, compute_spatial_rates
        >>> times = np.arange(600) / 30.0
        >>> positions = np.c_[10 + 5 * np.sin(times), 10 + 5 * np.cos(times)]
        >>> env = Environment.from_samples(positions, bin_size=2.0)
        >>> spikes = [times[::10], times[::15]]
        >>> train, test = (0.0, 10.0), (10.0, 19.9)
        >>> rates = compute_spatial_rates(env, spikes, times, positions, epochs=train)
        >>> result = BayesianDecoder.from_rates(rates).predict(
        ...     spikes, times, epochs=test
        ... )
        >>> bool(np.all(result.times >= 10.0))
        True
        """
        from neurospatial.encoding.spatial import SpatialRatesResult

        if not isinstance(rates, SpatialRatesResult):
            raise TypeError(
                f"BayesianDecoder.from_rates expects SpatialRatesResult, got {type(rates).__name__}.\n"
                "Why: the decoder needs spatial rate maps, their environment and unit identities.\n"
                "Fix: call rates = compute_spatial_rates(env, spike_times, times, positions), "
                "then BayesianDecoder.from_rates(rates)."
            )
        models = np.asarray(rates.firing_rates)
        dtype = np.float32 if models.dtype == np.float32 else np.float64
        settings = rates.analysis_settings or {}
        config: _EncodingConfig = (
            rates.method,
            rates.bandwidth,
            None if rates.method == "glm" else rates.min_occupancy,
            settings.get("penalty"),
            settings.get("rank", rates.rank),
        )
        return cls(
            rates.env,
            dt=dt,
            dtype=dtype,
            method=config[0],
            bandwidth=config[1],
            min_occupancy=config[2],
            penalty=config[3],
            rank=config[4],
            _fit=_FittedState(
                models,
                rates.unit_ids,
                unit_ids_generated=rates._unit_ids_generated,
                env=rates.env,
                spatial_support=rates.spatial_support,
                spike_window=rates.spike_window,
                analysis_settings=rates.analysis_settings,
                encoding_config=config,
            ),
        )

    @classmethod
    def from_models(
        cls,
        env: Environment,
        encoding_models: ArrayLike,
        *,
        unit_ids: NDArray[Any] | Sequence[Any] | None = None,
        spatial_support: ArrayLike | None = None,
        dt: float = 0.025,
    ) -> BayesianDecoder:
        """Create a fitted decoder from hand-built encoding models.

        Use this for rate maps that do not come from
        :func:`~neurospatial.compute_spatial_rates`, such as a simulation's
        ground-truth rates. For rates computed by neurospatial, prefer
        :meth:`from_rates`, which also carries their training support, unit
        labels and settings.

        Parameters
        ----------
        env : Environment
            Environment whose bins the models are defined on.
        encoding_models : array-like, shape (n_neurons, n_bins)
            Expected firing rate (Hz) of each unit in each bin. Copied.
        unit_ids : ndarray or sequence, optional
            One distinct label per model. With labels, prediction needs a
            labelled spike group, matched by label; without them, prediction
            needs plain trains, paired by position.
        spatial_support : ndarray of bool, shape (n_bins,), optional
            Bins allowed in the posterior. Every unit needs a finite rate in
            every allowed bin; bins outside the support may hold NaN.
            ``None`` allows every bin.
        dt : float, default=0.025
            Decode bin width in seconds.

        Returns
        -------
        BayesianDecoder
            Frozen fitted decoder; call ``predict`` or ``predict_summary``.

        Raises
        ------
        ValueError
            If the models are not 2-D or their bin axis does not match ``env``,
            if a rate is negative, or NaN or infinite in a supported bin, if
            ``unit_ids`` has the wrong length or repeats a label, or if
            ``spatial_support`` is not a boolean ``(n_bins,)`` mask with at
            least one supported bin.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> corners = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        >>> env = Environment.from_samples(corners, bin_size=5.0)
        >>> models = np.full((2, env.n_bins), 5.0)
        >>> decoder = BayesianDecoder.from_models(env, models, unit_ids=[3, 8])
        >>> decoder.is_fitted, decoder.unit_ids.tolist()
        (True, [3, 8])
        """
        from neurospatial._exceptions import _format_error

        models = np.asarray(encoding_models)
        dtype = np.float32 if models.dtype == np.float32 else np.float64
        if models.ndim == 2 and np.issubdtype(models.dtype, np.number):
            n_negative = int(np.sum(models < 0))
            if n_negative:
                raise ValueError(
                    _format_error(
                        f"BayesianDecoder.from_models: encoding_models has "
                        f"{n_negative} negative rate(s).",
                        why="a firing rate cannot be negative.",
                        fix="fix the rate maps; mark bins without an estimate "
                        "with NaN and exclude them with spatial_support.",
                    )
                )
            support = (
                np.ones(models.shape[1], dtype=bool)
                if spatial_support is None
                else np.asarray(spatial_support)
            )
            if support.shape == (models.shape[1],) and support.dtype == np.bool_:
                n_missing = int(np.sum(~np.isfinite(models[:, support])))
                if n_missing:
                    raise ValueError(
                        _format_error(
                            f"BayesianDecoder.from_models: encoding_models has "
                            f"{n_missing} NaN or Inf rate(s) in bins allowed by "
                            f"spatial_support"
                            f"{' (None allows every bin)' if spatial_support is None else ''}.",
                            why="every bin in the posterior needs a rate "
                            "estimate for every unit.",
                            fix="pass spatial_support=np.isfinite(models).all(axis=0) "
                            "to exclude bins without estimates, or recompute the "
                            "maps there.",
                        )
                    )
        return cls(
            env,
            dt=dt,
            dtype=dtype,
            _fit=_FittedState(
                models.astype(dtype, copy=False),
                unit_ids,
                unit_ids_generated=unit_ids is None,
                env=env,
                spatial_support=(
                    None if spatial_support is None else np.asarray(spatial_support)
                ),
            ),
        )

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
                "BayesianDecoder is not fitted.\n"
                "Why: fit() returns a new fitted decoder and leaves the original "
                "unchanged, so calling decoder.fit(...) without keeping the result "
                "does not fit decoder.\n"
                "Fix: decoder = decoder.fit(spike_times, times, positions), or "
                "decoder = BayesianDecoder.from_rates(rates) for precomputed rate maps."
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
        break_before: ArrayLike | None = None,
        spike_window: Any = None,
    ) -> BayesianDecoder:
        """Build encoding models from training data; return a new fitted decoder.

        Encodes the population's spatial rate maps using the same internal
        encoder :func:`~neurospatial.decoding.decode_session` runs
        (``_fit_encoding_model``), so the models are byte-identical to
        ``decode_session``'s internal encode step. Does **not** mutate ``self``:
        the original decoder stays unfitted and a new frozen decoder carrying the
        fitted ``encoding_models`` + ``unit_ids`` is returned.

        Parameters
        ----------
        spike_times : SpikeTrainsLike
            Spike times for one or more units. Accepts the canonical array forms,
            a :class:`~neurospatial.encoding.SpikeTrains` container, or a pynapple
            ``TsGroup``-like group. A group's index becomes ``unit_ids`` and
            prediction then needs a labelled group, matched by label; an
            unlabelled input gets ``arange(n_units)`` and prediction then needs
            plain trains in the same order.
        times : array-like, shape (n_frames,)
            Timestamp array in seconds. Decode bins tile each run whose gaps
            are no longer than ``max_gap``. For spans without tracking, pass
            ``times=np.arange(t0, t1, dt)``. For pynapple, pass ``tsd.t``.
        positions : NDArray[np.float64], shape (n_frames, n_dims)
            Required sample-aligned coordinates. For pynapple, pass ``tsd.values``.
        unit_ids : ndarray or sequence, optional
            One distinct label per spike train. If the spike group carries labels,
            these must match exactly in the same order. Prediction then needs
            a labelled spike group, matched by label.
        speed : NDArray[np.float64], shape (n_frames,), optional
            Precomputed speed, forwarded to the encoder. Only used when
            ``min_speed`` is set; auto-derived when ``None``.
        min_speed : float, optional
            Minimum speed threshold (position units / second). When set,
            low-speed samples are excluded from both the spike numerator and the
            occupancy denominator of the encoding model.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). The interval between two consecutive samples
            counts only if it lies entirely inside one window. ``None`` (default)
            means unrestricted.
        break_before : ArrayLike of bool, shape (n_frames,), optional
            ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
            ``k`` (for example where a new recording starts), whatever the elapsed
            time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
            is excluded from the encoding models' occupancy and spike counts, as
            for a gap longer than ``max_gap``, and the speed derived for
            ``min_speed`` is not smoothed across it.
            ``analysis_settings["break_times"]`` records ``times[k]`` of each
            break.
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
        ``spatial_support`` includes only occupied bins meeting the occupancy
        threshold with finite model rates. Predictions assign zero probability
        elsewhere, including during silence. Inspect this mask alongside errors:
        a short training epoch may cover little of the test trajectory. A fit
        with no supported bin raises an actionable ValueError.

        Training runs may be shorter than ``dt``: fitting requires supported
        position intervals, and decoding time bins are formed during prediction.

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
        from neurospatial._spikes import as_labelled_spike_trains
        from neurospatial.decoding.session import _fit_encoding_model

        # Capture unit identity once, from the ORIGINAL spike input (temporal
        # restriction never changes which units exist, only their spike counts).
        trains, resolved_ids, labels_supplied = as_labelled_spike_trains(
            spike_times, unit_ids, context="BayesianDecoder.fit"
        )

        resolved_epochs, resolved_spike_window = resolve_time_windows(
            epochs, spike_window
        )

        _, firing_rates, rates_result = _fit_encoding_model(
            self.env,
            trains,
            times,
            positions,
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
            break_before=break_before,
            spike_window=resolved_spike_window,
        )
        return replace(
            self,
            _fit=_FittedState(
                firing_rates,
                resolved_ids,
                unit_ids_generated=not labels_supplied,
                env=self.env,
                spatial_support=rates_result.spatial_support,
                spike_window=resolved_spike_window,
                analysis_settings=rates_result.analysis_settings,
                encoding_config=self._encoding_config(),
            ),
        )

    def _align_to_fitted_units(
        self, spike_times: SpikeTrainsLike, caller: str
    ) -> list[NDArray[np.float64]]:
        """Pair each spike train with its encoding model.

        By label when both the fitted models and this input carry
        caller-supplied labels; by position, one train per fitted unit, when
        neither does. Labels on only one side raise.
        """
        from neurospatial._exceptions import _format_error
        from neurospatial._results import resolve_unit_ids
        from neurospatial._spikes import _as_spike_trains_with_ids

        trains, input_ids = _as_spike_trains_with_ids(spike_times)
        n_models = self._check_fitted().shape[0]
        fit_labelled = not self._unit_ids_generated
        if fit_labelled != (input_ids is not None):
            given, missing = (
                ("the fitted models", f"the {caller} input")
                if fit_labelled
                else (f"the {caller} input", "the fitted models")
            )
            raise ValueError(
                _format_error(
                    f"BayesianDecoder.{caller}: {given} carry unit labels but "
                    f"{missing} do not.",
                    why="Why: the spike trains would be paired with encoding "
                    "models by position, so a group whose units are listed in "
                    "another order would silently decode with the wrong models.",
                    fix="label both (fit and predict with labelled groups such "
                    "as a TsGroup, or fit(unit_ids=...) / from_models(unit_ids=...) "
                    "with a labelled prediction group), or pass plain lists of "
                    "spike trains in the same order to both",
                )
            )
        if input_ids is None:
            if len(trains) != n_models:
                raise ValueError(
                    f"Got {len(trains)} spike trains but the decoder was fitted "
                    f"with {n_models} units. Unlabelled trains are paired with "
                    "encoding models by position.\n"
                    "Fix: pass one spike train per fitted unit in the order used "
                    "for fit."
                )
            return trains
        labels = resolve_unit_ids(
            None,
            len(trains),
            input_ids=input_ids,
            context=f"BayesianDecoder.{caller}",
        ).tolist()
        fitted = np.asarray(self.unit_ids).tolist()
        row = {u: i for i, u in enumerate(labels)}
        fitted_set = set(fitted)
        missing_ids = [u for u in fitted if u not in row]
        unexpected = [u for u in labels if u not in fitted_set]
        if missing_ids or unexpected:
            raise ValueError(
                "Spike input unit labels do not match the decoder's fitted "
                f"unit_ids (missing: {missing_ids}, unexpected: {unexpected}).\n"
                "Decoding would pair spike trains with the wrong encoding "
                "models.\n"
                "Fix: pass spikes for exactly the fitted units (pynapple: "
                "group[list(decoder.unit_ids)]), or refit on this input."
            )
        return [trains[row[u]] for u in fitted]

    def predict(
        self,
        spike_times: SpikeTrainsLike,
        times: ArrayLike,
        *,
        epochs: Any = None,
        break_before: ArrayLike | None = None,
        spike_window: Any = None,
    ) -> DecodingResult:
        """Decode the full posterior for new spikes against the fitted models.

        Uses the fitted rate maps with the same per-run binning and posterior
        calculation as :func:`~neurospatial.decoding.decode_session`.

        Parameters
        ----------
        spike_times : SpikeTrainsLike
            Spike times to decode. Same accepted forms as :meth:`fit`. When the
            fitted models carry caller-supplied labels (from a labelled group
            or ``unit_ids=`` in :meth:`fit` or :meth:`from_models`), pass a
            labelled group: trains are matched to models by label, in any
            order. When they do not, pass one plain train per fitted unit, in
            fit order. Labels on only one side raise ``ValueError``.
        times : array-like, shape (n_frames,)
            Timestamp array in seconds. Decode bins tile each run whose gaps
            are no longer than ``max_gap``. For spans without tracking, pass
            ``times=np.arange(t0, t1, dt)``. For pynapple, pass ``tsd.t``.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). The interval between two consecutive samples
            counts only if it lies entirely inside one window. ``None`` (default)
            means unrestricted.
        break_before : ArrayLike of bool, shape (n_frames,), optional
            ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
            ``k`` (for example where a new recording starts), whatever the elapsed
            time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
            is excluded from decoding, as for a gap longer than ``max_gap``: no
            decode time bin spans it and spikes inside it are not counted.
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
            If only one of the fit input and this input carried unit labels;
            if the input repeats a unit label; if its labels differ from
            ``unit_ids`` (the message lists the missing and unexpected labels);
            or if plain trains are paired by position and their number differs
            from the number of fitted units.

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
            spatial_support=self.spatial_support,
            dt=self.dt,
            warn_spikes_outside=self.warn_on_drop,
            warn_partial_bins=self.warn_on_drop,
            dtype=self.dtype,
            max_gap=self.max_gap,
            epochs=epochs,
            break_before=break_before,
            spike_window=spike_window,
            context="BayesianDecoder.predict",
        )

    def predict_summary(
        self,
        spike_times: SpikeTrainsLike,
        times: ArrayLike,
        *,
        epochs: Any = None,
        break_before: ArrayLike | None = None,
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
            Spike times to decode. Same accepted forms as :meth:`fit`. When the
            fitted models carry caller-supplied labels (from a labelled group
            or ``unit_ids=`` in :meth:`fit` or :meth:`from_models`), pass a
            labelled group: trains are matched to models by label, in any
            order. When they do not, pass one plain train per fitted unit, in
            fit order. Labels on only one side raise ``ValueError``.
        times : array-like, shape (n_frames,)
            Timestamp array in seconds. Decode bins tile each run whose gaps
            are no longer than ``max_gap``. For spans without tracking, pass
            ``times=np.arange(t0, t1, dt)``. For pynapple, pass ``tsd.t``.
        time_chunk : int, default=1024
            Streaming block size (number of time bins per block). Must be a
            positive integer.

        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). The interval between two consecutive samples
            counts only if it lies entirely inside one window. ``None`` (default)
            means unrestricted.
        break_before : ArrayLike of bool, shape (n_frames,), optional
            ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
            ``k`` (for example where a new recording starts), whatever the elapsed
            time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
            is excluded from decoding, as for a gap longer than ``max_gap``: no
            decode time bin spans it and spikes inside it are not counted.
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
            If only one of the fit input and this input carried unit labels;
            if the input repeats a unit label; if its labels differ from
            ``unit_ids`` (the message lists the missing and unexpected labels);
            or if plain trains are paired by position and their number differs
            from the number of fitted units.

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
            spatial_support=self.spatial_support,
            dt=self.dt,
            warn_spikes_outside=self.warn_on_drop,
            warn_partial_bins=self.warn_on_drop,
            dtype=self.dtype,
            max_gap=self.max_gap,
            epochs=epochs,
            break_before=break_before,
            spike_window=spike_window,
            context="BayesianDecoder.predict_summary",
            time_chunk=time_chunk,
        )

    def score(
        self,
        spike_times: SpikeTrainsLike,
        times: ArrayLike,
        positions: NDArray[np.float64],
        *,
        epochs: Any = None,
        break_before: ArrayLike | None = None,
        spike_window: Any = None,
        metric: str = "median_error",
        distance: str = "geodesic",
    ) -> float:
        """Decode and report position error against ground truth (lower is better).

        Decodes ``spike_times`` over ``times`` (via :meth:`predict`), aligns the
        MAP estimate to the ground-truth track (``times``, ``positions``) with
        :meth:`~neurospatial.decoding.DecodingResult.error_against`, and reduces
        the per-time-bin errors to a single scalar. **Lower is better.**

        Undecodable decode time bins -- those whose posterior row is entirely
        non-finite -- and bins without observed ground truth are stamped
        ``nan`` by ``error_against`` and are
        **excluded** from the reduction. Because ``nanmedian`` / ``nanmean``
        silently ignore them, a decoder that decodes only part of the session
        could otherwise report a misleadingly good score; this method therefore
        **warns** (naming the dropped fraction) when any bin is dropped and
        **raises** (rather than returning ``nan``) when *no* bin is decodable.
        The all-decodable path stays warning-free.

        Parameters
        ----------
        spike_times : SpikeTrainsLike
            Spike times to decode. Same accepted forms as :meth:`fit`. When the
            fitted models carry caller-supplied labels (from a labelled group
            or ``unit_ids=`` in :meth:`fit` or :meth:`from_models`), pass a
            labelled group: trains are matched to models by label, in any
            order. When they do not, pass one plain train per fitted unit, in
            fit order. Labels on only one side raise ``ValueError``.
        times : array-like, shape (n_frames,)
            Timestamp array in seconds. Decode bins tile each run whose gaps
            are no longer than ``max_gap``. For spans without tracking, pass
            ``times=np.arange(t0, t1, dt)``. For pynapple, pass ``tsd.t``.
        positions : NDArray[np.float64], shape (n_frames, n_dims)
            Required sample-aligned coordinates. For pynapple, pass ``tsd.values``.
            NaN marks missing tracking. Ground-truth alignment respects this
            decoder's ``max_gap`` and never extrapolates outside tracking.
        metric : {"median_error", "mean_error"}, default="median_error"
            Reduction over the per-time-bin errors. ``"median_error"`` ->
            ``nanmedian``; ``"mean_error"`` -> ``nanmean``. Lower is better.
        distance : {"geodesic", "euclidean"}, default="geodesic"
            Distance metric forwarded to ``error_against`` for the per-time-bin
            error. ``"geodesic"`` measures along the environment's connectivity
            graph (along the track, around walls); ``"euclidean"`` gives the
            straight-line distance.

        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). The interval between two consecutive samples
            counts only if it lies entirely inside one window. ``None`` (default)
            means unrestricted.
        break_before : ArrayLike of bool, shape (n_frames,), optional
            ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
            ``k`` (for example where a new recording starts), whatever the elapsed
            time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
            is excluded from decoding, as for a gap longer than ``max_gap``: no
            decode time bin spans it and spikes inside it are not counted. Ground
            truth is never interpolated across a break.
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
        from neurospatial._observation import validate_break_before
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
        breaks = validate_break_before(
            break_before, times_arr.size, call="BayesianDecoder.score"
        )

        result = self.predict(
            spike_times,
            times_arr,
            epochs=epochs,
            break_before=breaks,
            spike_window=spike_window,
        )
        # `distance` is validated above, so narrowing it to the Literal the
        # DecodingResult expects is safe.
        errors = result.error_against(
            times_arr,
            positions_arr,
            metric=cast("Literal['euclidean', 'geodesic']", distance),
            max_gap=self.max_gap,
            break_before=breaks,
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
                "posterior was entirely non-finite (undecodable), or no observed "
                "ground truth was available, so there is no "
                "error to reduce. Likely causes: a degenerate/empty encoding "
                "model built from too few training samples, or a spikes/times "
                "unit mismatch (e.g. spike_times in milliseconds while times is "
                "in seconds). Check the encoding-model coverage and that "
                "spike_times and times share units (both seconds)."
            )
        if n_bad > 0:
            warnings.warn(
                f"{n_bad}/{n} decode time bins were undecodable or lacked observed "
                f"ground truth and were excluded "
                f"from the score.",
                UserWarning,
                stacklevel=2,
            )

        if metric == "median_error":
            return float(np.nanmedian(errors))
        return float(np.nanmean(errors))
