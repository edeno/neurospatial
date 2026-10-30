"""Full-matrix activation strength: z(t)^T C z(t) with a zero diagonal."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.decoding import bin_spikes_in_time
from neurospatial.reactivation import (
    AssemblyPattern,
    assembly_activation,
    pairwise_activation,
    pairwise_correlations,
    simulated_reactivation_session,
)


def test_hand_calculation_on_three_neurons() -> None:
    # Pairs (0, 1), (0, 2), (1, 2).
    template = np.array([0.5, -0.2, 0.1])
    z = np.array([[1.0, 2.0, -1.0], [3.0, 0.0, 0.0], [-1.0, -1.0, 2.0]])
    activation = pairwise_activation(z, template, z_score_input=False)
    # 2 * (0.5 z0 z1 - 0.2 z0 z2 + 0.1 z1 z2), bin by bin:
    # bin 0: 2 * (1.0 + 0.2 - 0.2) = 2.0
    # bin 1: one neuron alone deviates, nothing else: 0
    # bin 2: 2 * (0.5 + 0.4 - 0.2) = 1.4
    np.testing.assert_allclose(activation, [2.0, 0.0, 1.4])


def test_time_average_is_the_inner_product_of_pair_correlations() -> None:
    """Supplementary material of Tingley & Peyrache (2020): the time average
    of the coherogram is the period's variance explained by the template
    correlations, sum_{i != j} C_template_ij C_period_ij."""
    rng = np.random.default_rng(0)
    shared = rng.poisson(1.0, (3000, 1))
    template = rng.poisson(1.5, (3000, 8)) + shared * [1, 1, 1, 0, 0, 1, 0, 0]
    period = rng.poisson(1.5, (1000, 8)) + shared[:1000] * [1, 1, 0, 0, 0, 1, 0, 1]
    activation = pairwise_activation(
        period.astype(float), pairwise_correlations(template.astype(float))
    )
    c_template = np.corrcoef(template, rowvar=False)
    c_period = np.corrcoef(period, rowvar=False)
    off_diagonal = ~np.eye(8, dtype=bool)
    expected = np.sum(c_template[off_diagonal] * c_period[off_diagonal])
    np.testing.assert_allclose(activation.mean(), expected, rtol=1e-12)


def test_rank_one_template_is_the_assembly_projector() -> None:
    """With template 'correlations' w_i w_j, the full-matrix projector is the
    assembly projector of pattern w (assembly_activation)."""
    weights = np.array([0.6, 0.5, -0.4, 0.3, 0.37])
    weights /= np.linalg.norm(weights)
    first, second = np.triu_indices(5, k=1)
    counts = np.random.default_rng(1).poisson(2.0, (400, 5)).astype(float)
    pattern = AssemblyPattern(weights, np.array([0, 1, 2]), 0.5)
    np.testing.assert_allclose(
        pairwise_activation(counts, weights[first] * weights[second]),
        assembly_activation(counts, pattern),
        atol=1e-12,
    )


def test_labels_match_columns_in_any_order() -> None:
    rng = np.random.default_rng(2)
    template = pairwise_correlations(rng.poisson(2.0, (500, 4)).astype(float))
    counts = rng.poisson(2.0, (200, 4)).astype(float)
    labels = np.array([10, 20, 30, 40])
    order = np.array([2, 0, 3, 1])
    by_label = pairwise_activation(
        counts[:, order], template, unit_ids=labels[order], template_unit_ids=labels
    )
    np.testing.assert_allclose(by_label, pairwise_activation(counts, template))


def test_undefined_template_pairs_get_weight_zero() -> None:
    rng = np.random.default_rng(3)
    counts = rng.poisson(2.0, (200, 4)).astype(float)
    template = np.array([0.4, np.nan, 0.2, -0.1, np.nan, 0.3])
    with pytest.warns(UserWarning, match="2 of 6 template pairs are NaN"):
        activation = pairwise_activation(counts, template)
    np.testing.assert_allclose(
        activation, pairwise_activation(counts, np.nan_to_num(template))
    )


def test_constant_neuron_contributes_nothing_and_warns() -> None:
    counts = np.random.default_rng(4).poisson(2.0, (200, 3)).astype(float)
    counts[:, 2] = 1.0
    with pytest.warns(UserWarning, match=r"columns \[2\]"):
        activation = pairwise_activation(counts, np.array([0.3, 0.5, 0.5]))
    np.testing.assert_allclose(
        activation, pairwise_activation(counts, np.array([0.3, 0.0, 0.0]))
    )


def test_constant_neuron_warning_names_the_callers_column_and_label() -> None:
    """Matching by label reorders the columns; the warning still names the
    column the caller passed and its label."""
    rng = np.random.default_rng(5)
    template = pairwise_correlations(rng.poisson(2.0, (500, 4)).astype(float))
    counts = rng.poisson(2.0, (200, 4)).astype(float)
    counts[:, 0] = 1.0
    with pytest.warns(UserWarning) as record:
        pairwise_activation(
            counts,
            template,
            unit_ids=["a", "b", "c", "d"],
            template_unit_ids=["d", "c", "b", "a"],
        )
    (message,) = [str(warning.message) for warning in record]
    assert "neuron columns [0] (units ['a'])" in message


@pytest.mark.parametrize(
    ("template", "options", "message"),
    [
        (np.zeros(2), {}, "3 pairs"),
        (np.array([0.1, 1.5, 0.0]), {}, r"\[-1, 1\]"),
        (np.array([0.1, np.inf, 0.0]), {}, r"\[-1, 1\]"),
        (np.zeros(3), {"unit_ids": [1, 2, 3]}, "together"),
        (
            np.zeros(3),
            {"unit_ids": [1, 2, 3], "template_unit_ids": [1, 2, 4]},
            "template's units",
        ),
    ],
)
def test_refuses_inputs_it_cannot_pair(template, options, message: str) -> None:
    counts = np.ones((5, 3)) + np.arange(5)[:, None]
    with pytest.raises(ValueError, match=message):
        pairwise_activation(counts, template, **options)


def test_rest_after_the_task_resembles_the_template_only_with_reactivation() -> None:
    """Mean activation after the task minus before it: the assemblies'
    co-firing returns only in the reactivated session."""
    differences = {}
    for name, reactivation in (("reactivated", 1.0), ("none", 0.0)):
        session = simulated_reactivation_session(
            duration=600.0, reactivation=reactivation, rng=0
        )

        def counts(epochs, session=session):
            return bin_spikes_in_time(session.spike_times, dt=0.1, epochs=epochs)[0]

        template = pairwise_correlations(counts(session.template_epochs))
        before = pairwise_activation(counts(session.control_epochs), template)
        after = pairwise_activation(counts(session.match_epochs), template)
        differences[name] = after.mean() - before.mean()
    # Without reactivation the difference varies from session to session by
    # about 0.15 (the number of population bursts sets each rest period's
    # overall correlation); with it, it is about 11.
    assert differences["reactivated"] > 5.0
    assert abs(differences["none"]) < 1.0
