"""Unit identity of detected assemblies and pairwise reactivation.

A pattern's weight ``i`` belongs to one cell. When the pattern and new counts
both carry the sorter's labels, the counts' columns are matched to the weights
by label, so the answer cannot depend on the order the units were listed in.
Likewise, labelled pair correlations of different periods are compared pair by
pair of the same two cells. Without labels everything pairs by position, as
before.
"""

from __future__ import annotations

import copy
import pickle
import re

import numpy as np
import pytest

from neurospatial.reactivation.assemblies import (
    AssemblyDetectionResult,
    AssemblyPattern,
    assembly_activation,
    assembly_reactivation,
    detect_assemblies,
    explained_variance_reactivation,
    pairwise_correlations,
)

LABELS = np.array(
    [40, 12, 33, 7, 21, 58, 3, 16, 9, 27, 61, 66, 72, 5, 81, 90, 14, 52, 38, 45]
)
N_UNITS = LABELS.size
MEMBERS = [1, 4, 8]
# Reordering columns changes floating-point summation order only.
ROUNDING = {"rtol": 1e-12, "atol": 1e-12}


def _counts(seed: int, n_time: int, drive: float) -> np.ndarray:
    """Poisson counts in which units ``MEMBERS`` share a common drive."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(1.0, (n_time, N_UNITS)).astype(float)
    counts[:, MEMBERS] += 3 * rng.poisson(drive, (n_time, 1))
    return counts


@pytest.fixture(scope="module")
def run_counts() -> np.ndarray:
    return _counts(seed=1, n_time=3000, drive=0.3)


@pytest.fixture(scope="module")
def rest_counts() -> np.ndarray:
    return _counts(seed=2, n_time=1500, drive=0.5)


@pytest.fixture(scope="module")
def labelled(run_counts) -> AssemblyDetectionResult:
    return detect_assemblies(run_counts, unit_ids=LABELS, rng=0)


@pytest.fixture(scope="module")
def unlabelled(run_counts) -> AssemblyDetectionResult:
    return detect_assemblies(run_counts, rng=0)


# ---------------------------------------------------------------------------
# Detection stores labels
# ---------------------------------------------------------------------------


def test_default_labels_are_generated_column_positions(unlabelled) -> None:
    np.testing.assert_array_equal(unlabelled.unit_ids, np.arange(N_UNITS))
    assert unlabelled._unit_ids_generated
    for pattern in unlabelled.patterns:
        np.testing.assert_array_equal(pattern.unit_ids, np.arange(N_UNITS))
        assert pattern._unit_ids_generated
        np.testing.assert_array_equal(pattern.member_unit_ids, pattern.member_indices)


def test_labels_name_the_planted_members(labelled) -> None:
    assert not labelled._unit_ids_generated
    np.testing.assert_array_equal(labelled.unit_ids, LABELS)
    pattern = labelled.patterns[0]
    assert not pattern._unit_ids_generated
    np.testing.assert_array_equal(pattern.unit_ids, LABELS)
    np.testing.assert_array_equal(pattern.member_indices, MEMBERS)
    assert pattern.member_unit_ids.tolist() == LABELS[MEMBERS].tolist()


def test_labels_do_not_change_detection(labelled, unlabelled) -> None:
    np.testing.assert_array_equal(labelled.activations, unlabelled.activations)
    np.testing.assert_array_equal(labelled.eigenvalues, unlabelled.eigenvalues)
    for with_labels, without in zip(
        labelled.patterns, unlabelled.patterns, strict=True
    ):
        np.testing.assert_array_equal(with_labels.weights, without.weights)
        np.testing.assert_array_equal(
            with_labels.member_indices, without.member_indices
        )


def test_stored_labels_are_owned_and_read_only(run_counts) -> None:
    labels = LABELS.copy()
    result = detect_assemblies(run_counts, unit_ids=labels, rng=0)
    labels[0] = -1
    assert result.unit_ids[0] == LABELS[0]
    assert result.patterns[0].unit_ids[0] == LABELS[0]
    for owner in (result, result.patterns[0]):
        with pytest.raises(ValueError, match="read-only"):
            owner.unit_ids[0] = 0
    for restored in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
        assert not restored.unit_ids.flags.writeable
        assert not restored._unit_ids_generated
        assert not restored.patterns[0].unit_ids.flags.writeable


@pytest.mark.parametrize(
    ("unit_ids", "match"),
    [(LABELS[:-1], "length mismatch"), (np.r_[LABELS[:-1], 40], "unique")],
    ids=["wrong-length", "duplicate"],
)
def test_invalid_labels_are_refused(run_counts, unit_ids, match) -> None:
    with pytest.raises(ValueError, match=match) as caught:
        detect_assemblies(run_counts, unit_ids=unit_ids, rng=0)
    assert "detect_assemblies()" in str(caught.value)


# ---------------------------------------------------------------------------
# Applying a pattern to new counts
# ---------------------------------------------------------------------------

ORDERS = (
    np.arange(N_UNITS)[::-1],
    np.random.default_rng(9).permutation(N_UNITS),
)


@pytest.mark.parametrize("z_score_input", [True, False])
@pytest.mark.parametrize("order", ORDERS)
def test_activation_is_invariant_to_column_order_with_labels(
    labelled, rest_counts, order, z_score_input
) -> None:
    pattern = labelled.patterns[0]
    expected = assembly_activation(rest_counts, pattern, z_score_input=z_score_input)

    activation = assembly_activation(
        rest_counts[:, order],
        pattern,
        unit_ids=LABELS[order],
        z_score_input=z_score_input,
    )

    np.testing.assert_allclose(activation, expected, **ROUNDING)


def test_positional_activation_of_reordered_columns_is_wrong(
    labelled, rest_counts
) -> None:
    """Why identity matters: by position, the weights meet other cells."""
    pattern = labelled.patterns[0]
    order = ORDERS[1]
    right = assembly_activation(rest_counts, pattern)
    wrong = assembly_activation(rest_counts[:, order], pattern)
    assert np.corrcoef(right, wrong)[0, 1] < 0.5
    # The bins where the members fire together stand out only when paired.
    together = (rest_counts[:, MEMBERS] >= 3).all(axis=1)
    assert right[together].mean() > 2 * right[~together].mean() > 0
    assert abs(wrong[together].mean()) < 0.5


def test_label_free_activation_is_unchanged_by_labels_on_the_pattern(
    labelled, unlabelled, rest_counts
) -> None:
    np.testing.assert_array_equal(
        assembly_activation(rest_counts, labelled.patterns[0]),
        assembly_activation(rest_counts, unlabelled.patterns[0]),
    )


@pytest.mark.parametrize("order", ORDERS)
def test_assembly_reactivation_is_invariant_to_column_order_with_labels(
    labelled, run_counts, rest_counts, order
) -> None:
    pattern = labelled.patterns[0]
    expected = assembly_reactivation(run_counts, rest_counts, pattern)
    # The members burst more strongly in rest_counts than in run_counts.
    assert expected.mean_activation_difference > 0.1

    result = assembly_reactivation(
        run_counts[:, order],
        rest_counts[:, order],
        pattern,
        control_unit_ids=LABELS[order],
        match_unit_ids=LABELS[order],
    )

    np.testing.assert_allclose(
        list(result.summary().values()),
        list(expected.summary().values()),
        **ROUNDING,
    )


def test_assembly_reactivation_matches_each_period_by_its_own_labels(
    labelled, run_counts, rest_counts
) -> None:
    """Periods sorted or binned in different unit orders give the same answer."""
    pattern = labelled.patterns[0]
    expected = assembly_reactivation(run_counts, rest_counts, pattern)
    run_order, rest_order = ORDERS

    result = assembly_reactivation(
        run_counts[:, run_order],
        rest_counts[:, rest_order],
        pattern,
        control_unit_ids=LABELS[run_order],
        match_unit_ids=LABELS[rest_order],
    )

    np.testing.assert_allclose(
        list(result.summary().values()),
        list(expected.summary().values()),
        **ROUNDING,
    )
    # Why identity matters: by position, the rest period's members meet the
    # weights of other cells, and the burst disappears.
    wrong = assembly_reactivation(
        run_counts[:, run_order],
        rest_counts[:, rest_order],
        _relabelled(pattern, run_order),
    )
    assert wrong.match_mean_activation < 0.5 * expected.match_mean_activation


def _relabelled(pattern: AssemblyPattern, order: np.ndarray) -> AssemblyPattern:
    """The pattern with its weights in ``order``, as if detected on reordered counts."""
    return AssemblyPattern(
        pattern.weights[order],
        np.flatnonzero(np.isin(order, pattern.member_indices)),
        pattern.explained_variance_ratio,
        unit_ids=pattern.unit_ids[order],
    )


@pytest.mark.parametrize("given", ["control_unit_ids", "match_unit_ids"])
def test_assembly_reactivation_refuses_labels_for_one_period(
    labelled, run_counts, rest_counts, given
) -> None:
    missing = ({"control_unit_ids", "match_unit_ids"} - {given}).pop()
    with pytest.raises(
        ValueError, match=re.escape(f"got {given}= but not {missing}=")
    ) as caught:
        assembly_reactivation(
            run_counts, rest_counts, labelled.patterns[0], **{given: LABELS}
        )
    assert "or neither to pair columns by position" in str(caught.value)


def test_assembly_reactivation_names_missing_labels_of_each_period(
    labelled, run_counts, rest_counts
) -> None:
    pattern = labelled.patterns[0]
    labels = LABELS.copy()
    labels[3] = 100  # unit 7 replaced by 100
    with pytest.raises(
        ValueError,
        match=re.escape("only in the pattern: [7], only in match_counts: [100]"),
    ) as caught:
        assembly_reactivation(
            run_counts,
            rest_counts,
            pattern,
            control_unit_ids=LABELS,
            match_unit_ids=labels,
        )
    assert "np.isin(match_unit_ids, pattern.unit_ids)" in str(caught.value)
    with pytest.raises(ValueError, match="unique") as caught:
        assembly_reactivation(
            run_counts,
            rest_counts,
            pattern,
            control_unit_ids=np.r_[LABELS[:-1], LABELS[0]],
            match_unit_ids=LABELS,
        )
    assert "for control_counts" in str(caught.value)


def test_missing_and_extra_labels_are_named(labelled, rest_counts) -> None:
    pattern = labelled.patterns[0]
    labels = LABELS.copy()
    labels[[0, 5]] = [100, 200]  # units 40 and 58 replaced by 100 and 200

    with pytest.raises(
        ValueError,
        match=re.escape(
            "only in the pattern: [40, 58], only in spike_counts: [100, 200]"
        ),
    ) as caught:
        assembly_activation(rest_counts, pattern, unit_ids=labels)
    message = str(caught.value)
    assert message.startswith("assembly_activation():")
    assert "np.isin(unit_ids, pattern.unit_ids)" in message

    with pytest.raises(
        ValueError,
        match=re.escape("only in the pattern: [45], only in spike_counts: []"),
    ):
        assembly_activation(rest_counts[:, :-1], pattern, unit_ids=LABELS[:-1])


def test_assembly_reactivation_names_the_period_with_wrong_labels(
    labelled, run_counts, rest_counts
) -> None:
    pattern = labelled.patterns[0]
    with pytest.raises(ValueError, match="for match_counts") as caught:
        assembly_reactivation(
            run_counts,
            rest_counts[:, :-1],
            pattern,
            control_unit_ids=LABELS,
            match_unit_ids=LABELS,
        )
    assert "length mismatch" in str(caught.value)


def test_labels_against_a_pattern_without_labels_are_refused(
    unlabelled, rest_counts
) -> None:
    with pytest.raises(ValueError, match="the pattern has no unit_ids") as caught:
        assembly_activation(rest_counts, unlabelled.patterns[0], unit_ids=LABELS)
    assert "unit_ids= to detect_assemblies()" in str(caught.value)
    with pytest.raises(ValueError, match="the pattern has no unit_ids"):
        assembly_reactivation(
            rest_counts,
            rest_counts,
            unlabelled.patterns[0],
            control_unit_ids=LABELS,
            match_unit_ids=LABELS,
        )


def test_string_labels_match_across_periods() -> None:
    weights = np.array([0.8, 0.6, 0.0])
    pattern = AssemblyPattern(
        weights, np.array([0, 1]), 0.4, unit_ids=["t1c1", "t1c2", "t2c1"]
    )
    counts = np.random.default_rng(4).poisson(2.0, (80, 3)).astype(float)
    np.testing.assert_allclose(
        assembly_activation(
            counts[:, [2, 0, 1]], pattern, unit_ids=["t2c1", "t1c1", "t1c2"]
        ),
        assembly_activation(counts, pattern),
        **ROUNDING,
    )
    assert pattern.member_unit_ids.tolist() == ["t1c1", "t1c2"]


def test_labelled_counts_must_be_two_dimensional(labelled, rest_counts) -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        assembly_reactivation(
            rest_counts,
            rest_counts[:, 0],
            labelled.patterns[0],
            control_unit_ids=LABELS,
            match_unit_ids=LABELS,
        )


# ---------------------------------------------------------------------------
# Explained variance: pairs of the same two cells in every period
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def period_counts(run_counts) -> dict[str, np.ndarray]:
    """Pre-task rest without the assembly, running and post-task rest with it."""
    return {
        "control": _counts(seed=3, n_time=3000, drive=0.0),
        "template": run_counts,
        "match": _counts(seed=4, n_time=3000, drive=0.3),
    }


def _ev(period_counts, orders=None, *, labelled=False, control=True):
    """EV of the periods, with each period's columns in ``orders[period]``."""
    orders = orders or {}
    correlations, labels = {}, {}
    for period, counts in period_counts.items():
        order = orders.get(period, np.arange(N_UNITS))
        correlations[period] = pairwise_correlations(counts[:, order])
        labels[f"{period}_unit_ids"] = LABELS[order]
    if not control:
        del correlations["control"], labels["control_unit_ids"]
    return explained_variance_reactivation(
        correlations["template"],
        correlations["match"],
        control_correlations=correlations.get("control"),
        **(labels if labelled else {}),
    )


def _values(result) -> list[float]:
    return [
        result.explained_variance,
        result.reversed_explained_variance,
        result.partial_correlation,
    ]


PERMUTED = {
    "match": {"match": ORDERS[1]},
    "control": {"control": ORDERS[1]},
    "every-period": {"template": ORDERS[0], "match": ORDERS[1], "control": ORDERS[0]},
}


@pytest.mark.parametrize("permuted", PERMUTED.values(), ids=PERMUTED.keys())
def test_ev_is_invariant_to_unit_order_with_labels(period_counts, permuted) -> None:
    expected = _ev(period_counts)
    assert expected.explained_variance > 0.5 > expected.reversed_explained_variance

    result = _ev(period_counts, permuted, labelled=True)

    np.testing.assert_allclose(_values(result), _values(expected), **ROUNDING)
    assert result.n_pairs == expected.n_pairs == N_UNITS * (N_UNITS - 1) // 2


def test_ev_without_control_is_invariant_to_unit_order_with_labels(
    period_counts,
) -> None:
    expected = _ev(period_counts, control=False)
    result = _ev(period_counts, PERMUTED["match"], labelled=True, control=False)
    np.testing.assert_allclose(_values(result), _values(expected), **ROUNDING)


def test_positional_ev_of_reordered_units_is_wrong(period_counts) -> None:
    """Why identity matters: by position, each rest pair meets another pair."""
    right = _ev(period_counts)
    wrong = _ev(period_counts, PERMUTED["match"])
    assert right.explained_variance > 0.5
    assert wrong.explained_variance < 0.1


def test_positional_control_of_reordered_units_fakes_reactivation(
    period_counts,
) -> None:
    """A mispaired baseline hides correlations that were there before the task."""
    already_correlated = {
        **period_counts,
        "control": _counts(seed=3, n_time=3000, drive=0.3),
    }
    right = _ev(already_correlated)
    assert (
        right.explained_variance < right.reversed_explained_variance
    )  # no evidence of replay

    wrong = _ev(already_correlated, PERMUTED["control"])
    assert wrong.explained_variance > 0.5 > 0.1 > wrong.reversed_explained_variance

    labelled = _ev(already_correlated, PERMUTED["control"], labelled=True)
    np.testing.assert_allclose(_values(labelled), _values(right), **ROUNDING)


def test_labels_in_the_same_order_give_the_label_free_answer(period_counts) -> None:
    unlabelled = _ev(period_counts)
    labelled = _ev(period_counts, labelled=True)
    assert _values(labelled) == _values(unlabelled)
    assert labelled.n_pairs == unlabelled.n_pairs


@pytest.fixture(scope="module")
def period_correlations(period_counts) -> dict[str, np.ndarray]:
    return {
        period: pairwise_correlations(counts)
        for period, counts in period_counts.items()
    }


@pytest.mark.parametrize(
    ("given", "with_control", "missing"),
    [
        (["template_unit_ids"], False, "but not match_unit_ids="),
        (["match_unit_ids"], False, "but not template_unit_ids="),
        (["template_unit_ids", "match_unit_ids"], True, "but not control_unit_ids="),
        (["control_unit_ids"], True, "but not template_unit_ids= or match_unit_ids="),
    ],
    ids=["template-only", "match-only", "control-missing", "control-only"],
)
def test_ev_refuses_labels_for_some_periods(
    period_correlations, given, with_control, missing
) -> None:
    with pytest.raises(ValueError, match=re.escape(missing)) as caught:
        explained_variance_reactivation(
            period_correlations["template"],
            period_correlations["match"],
            control_correlations=(
                period_correlations["control"] if with_control else None
            ),
            **dict.fromkeys(given, LABELS),
        )
    assert "or none of them to pair by position" in str(caught.value)


def test_ev_refuses_control_labels_without_control(period_correlations) -> None:
    with pytest.raises(ValueError, match="but no control_correlations="):
        explained_variance_reactivation(
            period_correlations["template"],
            period_correlations["match"],
            template_unit_ids=LABELS,
            match_unit_ids=LABELS,
            control_unit_ids=LABELS,
        )


def _ev_with_labels(period_correlations, **overrides):
    """Labelled EV of the three periods, with some arguments replaced."""
    arguments = {
        "template": period_correlations["template"],
        "match": period_correlations["match"],
        "control_correlations": period_correlations["control"],
        "template_unit_ids": LABELS,
        "match_unit_ids": LABELS,
        "control_unit_ids": LABELS,
    }
    arguments.update(overrides)
    return explained_variance_reactivation(
        arguments.pop("template"), arguments.pop("match"), **arguments
    )


def test_ev_names_units_missing_from_a_period(
    period_counts, period_correlations
) -> None:
    """A unit lost in one period is named, not shifted onto another's pairs."""
    keep = np.arange(N_UNITS) != 3  # unit 7 not recorded in the match period
    with pytest.raises(
        ValueError,
        match=re.escape("only in template_unit_ids: [7], only in match_unit_ids: []"),
    ) as caught:
        _ev_with_labels(
            period_correlations,
            match=pairwise_correlations(period_counts["match"][:, keep]),
            match_unit_ids=LABELS[keep],
        )
    message = str(caught.value)
    assert message.startswith("explained_variance_reactivation(): match_correlations")
    assert "compute pairwise_correlations() again" in message

    labels = LABELS.copy()
    labels[0] = 100
    with pytest.raises(
        ValueError,
        match=re.escape(
            "only in template_unit_ids: [40], only in control_unit_ids: [100]"
        ),
    ):
        _ev_with_labels(period_correlations, control_unit_ids=labels)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"match_unit_ids": np.r_[LABELS[:-1], 40]}, "unique"),
        ({"control_unit_ids": LABELS[:-1]}, "length mismatch"),
        ({"match": np.zeros(189)}, re.escape("not n * (n - 1) / 2 pairs")),
        ({"template": np.zeros((N_UNITS, N_UNITS))}, "must be 1-D"),
    ],
    ids=["duplicate", "wrong-length", "not-all-pairs", "matrix"],
)
def test_ev_refuses_invalid_labels(period_correlations, overrides, match) -> None:
    with pytest.raises(ValueError, match=match) as caught:
        _ev_with_labels(period_correlations, **overrides)
    assert "explained_variance_reactivation()" in str(caught.value)
