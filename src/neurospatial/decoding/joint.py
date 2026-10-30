"""Decode position jointly over several sets of rate maps.

Replay analyses often compare rate maps from different behavior: one set for
each running direction on a track, or one set for each of two tracks. Decoding
each set separately gives posteriors that each sum to 1, so it cannot say
which set the spikes match better. :func:`decode_jointly` decodes all sets in
one posterior that sums to 1 over every bin of every set, so the probability
of each set ("template") is comparable across sets and time bins.

This is the decode used by Ambrose et al. (2016) and Mallory et al. (2025),
who concatenate the two running-direction maps, and by Tirole et al. (2022),
who normalize the posterior across two tracks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin, read_only_copy
from neurospatial.decoding._result import DecodingResult
from neurospatial.decoding._time_bins import validate_dt

if TYPE_CHECKING:
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment

_OPERATION = "decode_jointly()"


@dataclass(frozen=True, repr=False, eq=False)
class JointDecodingResult(ReadOnlyArraysMixin, ResultMixin):
    """Posterior over the bins of several rate-map sets, decoded together.

    Parameters
    ----------
    template_names : tuple of str
        Name of each rate-map set ("template"), in the order of the columns
        of ``template_probability``.
    template_probability : NDArray[np.float64], shape (n_time_bins, n_templates)
        Probability that each template generated the spikes in each time
        bin. Each row is non-negative and sums to 1.
    conditional : Mapping[str, DecodingResult]
        For each template, the posterior over its own bins given that it
        generated the spikes: each row sums to 1 over that template's bins.

    Attributes
    ----------
    n_time_bins : int
        Number of decoded time bins.

    Notes
    -----
    The joint posterior of a template is its conditional posterior times
    its template probability, ``joint(name)``. Its rows sum, over all
    templates, to 1.

    The arrays are read-only copies the result owns, and ``conditional`` is
    a read-only mapping. Results compare equal only to themselves.

    See Also
    --------
    decode_jointly : Create this result from spike counts.
    """

    template_names: tuple[str, ...]
    template_probability: NDArray[np.float64]
    conditional: Mapping[str, DecodingResult]

    def __post_init__(self) -> None:
        """Own the arrays and the mapping, and check their agreement."""
        object.__setattr__(self, "template_names", tuple(self.template_names))
        object.__setattr__(
            self,
            "template_probability",
            read_only_copy(self.template_probability, np.float64),
        )
        object.__setattr__(
            self, "conditional", MappingProxyType(dict(self.conditional))
        )
        probability = self.template_probability
        problems = []
        if tuple(self.conditional) != self.template_names:
            problems.append(
                f"conditional has templates {tuple(self.conditional)} but "
                f"template_names is {self.template_names}"
            )
        if probability.ndim != 2 or probability.shape[1] != len(self.template_names):
            problems.append(
                "template_probability must have shape (n_time_bins, "
                f"{len(self.template_names)}), got {probability.shape}"
            )
        else:
            n_time = probability.shape[0]
            for name, result in self.conditional.items():
                if result.n_time_bins != n_time:
                    problems.append(
                        f"conditional[{name!r}] has {result.n_time_bins} time "
                        f"bins but template_probability has {n_time}"
                    )
            if (
                not np.isfinite(probability).all()
                or np.any(probability < 0)
                or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6)
            ):
                problems.append(
                    "each template_probability row must be finite, "
                    "non-negative and sum to 1"
                )
        if problems:
            raise ValueError(
                _format_error(
                    "JointDecodingResult: " + "; ".join(problems) + ".",
                    why="row i of template_probability gives, for time bin i, "
                    "the probability of each template, and conditional holds "
                    "one posterior per template over the same time bins",
                    fix="build the result with decode_jointly()",
                )
            )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mapping cannot pickle."""
        return (
            type(self),
            (
                self.template_names,
                np.array(self.template_probability),
                dict(self.conditional),
            ),
        )

    @property
    def n_time_bins(self) -> int:
        """Number of decoded time bins."""
        return int(self.template_probability.shape[0])

    def joint(self, name: str) -> NDArray[np.float64]:
        """Joint posterior over one template's bins.

        Parameters
        ----------
        name : str
            Template name.

        Returns
        -------
        NDArray[np.float64], shape (n_time_bins, n_bins)
            Probability that the spikes in each time bin came from template
            ``name`` at each of its bins. Summed over the bins of every
            template, each row is 1. The array is read-only.

        Raises
        ------
        KeyError
            If ``name`` is not a template of this result.
        """
        if name not in self.conditional:
            raise KeyError(
                f"no template named {name!r}; the templates are "
                f"{list(self.template_names)}."
            )
        column = self.template_names.index(name)
        joint = (
            self.conditional[name].posterior
            * self.template_probability[:, column, np.newaxis]
        )
        joint.flags.writeable = False
        return joint

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        mean = self.template_probability.mean(axis=0)
        return {
            "n_time_bins": self.n_time_bins,
            "n_templates": len(self.template_names),
            **{
                f"mean_probability[{name}]": float(value)
                for name, value in zip(self.template_names, mean, strict=True)
            },
        }


def decode_jointly(
    env: Environment | Mapping[str, Environment],
    spike_counts: NDArray[np.int64],
    encoding_models: Mapping[str, NDArray[np.float64] | SpatialRatesResult],
    *,
    dt: float,
    prior: Mapping[str, NDArray[np.float64]] | None = None,
    times: NDArray[np.float64] | None = None,
    validate: bool = True,
) -> JointDecodingResult:
    """Decode position jointly over several sets of rate maps.

    Each set of rate maps (a "template") describes the same neurons in a
    different behavior, for example running in each direction on a track or
    running on each of two tracks. The posterior covers every bin of every
    template and sums to 1, so the probability that each template generated
    the spikes can be compared between templates.

    Parameters
    ----------
    env : Environment or Mapping[str, Environment]
        The environment the rate maps are on. Pass one environment when every
        template is on the same bins (running directions), or one per
        template, keyed like ``encoding_models`` (separate tracks).
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per time bin, one column per neuron, in the same neuron
        order as every template's rate maps.
    encoding_models : Mapping[str, array or SpatialRatesResult]
        Rate maps for each template, keyed by template name: an
        ``(n_neurons, n_bins)`` array of rates in Hz, or the rate result from
        :func:`~neurospatial.encoding.compute_spatial_rates`, whose
        ``spatial_support`` excludes bins without training data. At least two
        templates.
    dt : float
        Time-bin duration in seconds.
    prior : Mapping[str, array] or None, default=None
        Non-negative prior weight of each bin of each template, shape
        ``(n_bins,)`` or ``(n_time_bins, n_bins)``, keyed like
        ``encoding_models``. The weights share one scale across templates:
        a template's total weight is its prior probability. ``None`` gives
        the prior uniform over space across all templates, each bin weighted
        by its length, area or volume, so a template's prior probability is
        proportional to the space its environment covers.
    times : NDArray[np.float64] or None, default=None
        Center time of each time bin in seconds, stored on each conditional
        posterior.
    validate : bool, default=True
        Run the input checks of :func:`decode_position` for each template.

    Returns
    -------
    JointDecodingResult
        The probability of each template in each time bin, and each
        template's posterior over its own bins given that template.

    Raises
    ------
    ValueError
        If fewer than two templates are given, if a template name is not a
        string, if ``env`` or ``prior`` is a mapping whose keys differ from
        ``encoding_models``, if ``prior`` is not a mapping, if the default
        prior is used with separate environments that have no units or whose
        bins are measured in different units or by different measures (length
        and area), if a template's prior has no weight in some time bin, or
        for any input :func:`decode_position` refuses.

    See Also
    --------
    decode_position : Decode one set of rate maps.

    Notes
    -----
    With template prior weights :math:`w_k(x)` and Poisson likelihood
    :math:`L_k(x)` of the spike counts under template :math:`k` at bin
    :math:`x`, the joint posterior is

    .. math::

        P(k, x \\mid n) = \\frac{w_k(x) L_k(x)}
        {\\sum_{k'} \\sum_{x'} w_{k'}(x') L_{k'}(x')}.

    The template probability is its sum over :math:`x`, and the conditional
    posterior is :math:`P(x \\mid k, n) = P(k, x \\mid n) / P(k \\mid n)`.
    Both are computed in log space, so a template with very small
    probability still has an accurate conditional posterior.

    With one environment and the default prior, every template gets prior
    probability ``1 / n_templates``, as when the direction maps are
    concatenated with a prior flat over bins (Ambrose et al. 2016; Mallory et
    al. 2025). With separate track environments it is proportional to track
    length, which matches normalizing a posterior uniform over bins of one
    size across the tracks (Tirole et al. 2022).

    Bins excluded from a template (outside its ``spatial_support``, or where
    a neuron's rate is NaN) get zero probability in that template only. Their
    prior weight moves to the template's other bins, so the template's prior
    probability does not depend on how much of its environment the training
    data covered: two directions with different coverage still start equal.

    References
    ----------
    Ambrose, R. E., Pfeiffer, B. E., & Foster, D. J. (2016). Reverse replay of
    hippocampal place cells is uniquely modulated by changing reward.
    *Neuron*, 91(5), 1124-1136.

    Mallory, C. S., Widloski, J., & Foster, D. J. (2025). The time course and
    organization of hippocampal replay. *Science*, 387(6733), 541-548.

    Tirole, M., Huelin Gorriz, M., Takigawa, M., Kukovska, L., & Bendor, D.
    (2022). Experience-driven rate modulation is reinstated during
    hippocampal replay. *eLife*, 11, e79031.

    Examples
    --------
    Two running directions on one track, with the same place fields held by
    different cells. The spikes come from the outbound maps, so the outbound
    template takes most of the probability; a time bin with few spikes can
    stay ambiguous:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import decode_jointly
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> centers = np.linspace(5.0, 95.0, 8)
    >>> x = env.bin_centers[:, 0]
    >>> outbound = 1.0 + 20.0 * np.exp(-0.5 * ((x - centers[:, None]) / 8.0) ** 2)
    >>> inbound = outbound[[0, 5, 2, 7, 4, 1, 6, 3]]  # same fields, other cells
    >>> rng = np.random.default_rng(0)
    >>> counts = rng.poisson(outbound[:, 3] * 0.1, size=(5, 8))
    >>> result = decode_jointly(
    ...     env, counts, {"outbound": outbound, "inbound": inbound}, dt=0.1
    ... )
    >>> result.template_names
    ('outbound', 'inbound')
    >>> result.template_probability.mean(axis=0).round(2)
    array([0.81, 0.19])
    >>> int(counts[0].sum()), result.template_probability[0].round(2)
    (1, array([0.48, 0.52]))
    """
    names = tuple(encoding_models)
    if len(names) < 2:
        raise ValueError(
            _format_error(
                f"{_OPERATION} needs at least two templates, got {len(names)}.",
                why="a joint decode compares sets of rate maps",
                fix="decode one set of rate maps with decode_position()",
            )
        )
    bad_names = [name for name in names if not isinstance(name, str)]
    if bad_names:
        raise ValueError(
            f"{_OPERATION}: template names must be strings, got {bad_names!r}."
        )
    envs = _per_template(env, names, "env")
    if prior is not None and not isinstance(prior, Mapping):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: prior must be a mapping from template name to "
                f"weights, got {type(prior).__name__}.",
                why="each template has its own bins, and the weights of all "
                "templates share one scale",
                fix='pass prior={"outbound": weights_out, "inbound": weights_in}',
            )
        )
    priors = None if prior is None else _per_template(prior, names, "prior")
    dt = validate_dt(dt)
    if priors is None:
        _require_comparable_measures(envs)

    from neurospatial.decoding.posterior import (
        _posterior_blocks,
        _prepare_decode_inputs,
        _validate_spike_counts,
    )

    spike_counts = np.asarray(spike_counts)
    if validate:
        _validate_spike_counts(spike_counts)
    n_time = len(spike_counts)
    log_evidence = []
    conditional = {}
    for name in names:
        context = f"{_OPERATION} template {name!r}"
        weights = _bin_measures(envs[name], context) if priors is None else priors[name]
        total = np.asarray(weights, dtype=np.float64).sum(axis=-1)
        rates, weights = _prepare_decode_inputs(
            envs[name],
            encoding_models[name],
            weights,
            n_time,
            dt=dt,
            validate=validate,
            context=context,
        )
        posterior = np.empty((n_time, envs[name].n_bins))
        # One block: the normalizer of the posterior under the template's
        # normalized prior is its evidence relative to its total weight.
        # Excluded bins lose their weight; normalizing spreads it over the
        # template's other bins, so its prior probability stays its full
        # total.
        [(_, _, _, log_norm)] = _posterior_blocks(
            lambda start, stop: spike_counts[start:stop],
            n_time,
            rates,
            weights,
            time_chunk=max(n_time, 1),
            dtype=np.float64,
            out=posterior,
        )
        log_evidence.append(log_norm + np.log(total))
        conditional[name] = DecodingResult._from_owned_posterior(
            posterior, env=envs[name], times=times, spike_window=None
        )

    log_evidence_array = np.column_stack(log_evidence)
    template_probability = np.exp(
        log_evidence_array - logsumexp(log_evidence_array, axis=1, keepdims=True)
    )
    return JointDecodingResult(
        template_names=names,
        template_probability=template_probability,
        conditional=conditional,
    )


def _bin_measures(env: Environment, context: str) -> NDArray[np.float64]:
    """Each bin's length, area or volume, for the default prior."""
    try:
        return np.asarray(env.bin_sizes, dtype=np.float64)
    except ValueError as err:
        raise ValueError(
            _format_error(
                f"{context}: the default prior needs each bin's length, area or "
                f"volume, but {err}",
                why="the default prior weights each bin by the space it covers",
                fix="pass prior= with weights on one scale for every template",
            )
        ) from err


def _per_template(value: Any, names: tuple[str, ...], argument: str) -> dict[str, Any]:
    """One ``value`` per template: a mapping keyed like the templates, or shared."""
    if not isinstance(value, Mapping):
        return dict.fromkeys(names, value)
    if set(value) != set(names):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: {argument} has keys {sorted(value)} but "
                f"encoding_models has {sorted(names)}.",
                why=f"each template needs its own {argument}",
                fix=f"key {argument} by the same template names as encoding_models",
            )
        )
    return {name: value[name] for name in names}


def _require_comparable_measures(envs: dict[str, Environment]) -> None:
    """Refuse the uniform-over-space prior across incomparable bin measures."""
    if len({id(env) for env in envs.values()}) == 1:
        return  # one environment: every template is measured alike
    units = {name: env.units for name, env in envs.items()}
    if any(value is None for value in units.values()):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: the templates' environments do not all have "
                f"units ({units}).",
                why="the default prior compares templates by the space their "
                "bins cover, which needs every environment's units to be known",
                fix="build the environments with units= (or "
                "env.with_metadata(units='cm')), or pass prior= with weights on "
                "one scale for every template",
            )
        )
    measures = {
        name: tuple(np.unique(env.binning.bins.measure_dims))
        for name, env in envs.items()
    }
    if len(set(units.values())) > 1 or len(set(measures.values())) > 1:
        described = ", ".join(
            f"{name}: units {units[name]!r}, measured by "
            f"{'/'.join({1: 'length', 2: 'area', 3: 'volume'}.get(int(d), f'{d}-D') for d in measures[name])}"
            for name in envs
        )
        raise ValueError(
            _format_error(
                f"{_OPERATION}: the templates' environments are not measured "
                f"alike ({described}).",
                why="the default prior weights each bin by its length, area or "
                "volume, which compares templates only when every bin is "
                "measured in the same unit and by the same measure",
                fix="pass prior= with weights on one scale for every template, "
                "or build the environments in the same units",
            )
        )
