"""Unit identity of detected assemblies.

A pattern's weight ``i`` belongs to one cell. When the pattern and new counts
both carry the sorter's labels, the counts' columns are matched to the weights
by label, so the answer cannot depend on the order the units were listed in.
Without labels the columns pair by position, as before.
"""

from __future__ import annotations

import copy
import pickle
import re

import numpy as np
import pytest

from neurospatial.decoding.assemblies import (
    AssemblyDetectionResult,
    AssemblyPattern,
    assembly_activation,
    detect_assemblies,
    reactivation_strength,
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
    drive = rest_counts[:, MEMBERS].sum(axis=1)
    assert np.corrcoef(right, drive)[0, 1] > 0.9
    assert np.corrcoef(wrong, drive)[0, 1] < 0.5


def test_label_free_activation_is_unchanged_by_labels_on_the_pattern(
    labelled, unlabelled, rest_counts
) -> None:
    np.testing.assert_array_equal(
        assembly_activation(rest_counts, labelled.patterns[0]),
        assembly_activation(rest_counts, unlabelled.patterns[0]),
    )


@pytest.mark.parametrize("order", ORDERS)
def test_reactivation_strength_is_invariant_to_column_order_with_labels(
    labelled, run_counts, rest_counts, order
) -> None:
    pattern = labelled.patterns[0]
    expected = reactivation_strength(run_counts, rest_counts, pattern)
    assert expected > 1.0  # the members burst more strongly at rest

    strength = reactivation_strength(
        run_counts[:, order], rest_counts[:, order], pattern, unit_ids=LABELS[order]
    )

    assert strength == pytest.approx(expected, rel=1e-12)


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


def test_reactivation_strength_names_the_period_with_wrong_labels(
    labelled, run_counts, rest_counts
) -> None:
    pattern = labelled.patterns[0]
    with pytest.raises(ValueError, match="for match_counts") as caught:
        reactivation_strength(run_counts, rest_counts[:, :-1], pattern, unit_ids=LABELS)
    assert "length mismatch" in str(caught.value)


def test_labels_against_a_pattern_without_labels_are_refused(
    unlabelled, rest_counts
) -> None:
    with pytest.raises(ValueError, match="the pattern has no unit_ids") as caught:
        assembly_activation(rest_counts, unlabelled.patterns[0], unit_ids=LABELS)
    assert "unit_ids= to detect_assemblies()" in str(caught.value)
    with pytest.raises(ValueError, match="the pattern has no unit_ids"):
        reactivation_strength(
            rest_counts, rest_counts, unlabelled.patterns[0], unit_ids=LABELS
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
        reactivation_strength(
            rest_counts, rest_counts[:, 0], labelled.patterns[0], unit_ids=LABELS
        )
