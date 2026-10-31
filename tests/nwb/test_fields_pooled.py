"""NWB persistence of the per-neuron-lambda (``pooled=False``) diagnostics.

The GAM metadata blob stores ``pooled`` and a scalar ``reml_at_boundary``; when
``penalty`` / ``reml_objective`` / ``reml_at_boundary`` are per-unit vectors,
those plus ``penalty_selected_by_reml`` are stored as per-unit table columns.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# Skip every test here when pynwb is absent (HAS_PYNWB gate).
pytest.importorskip("pynwb")

from neurospatial.io.nwb._fields import (
    DEFAULT_SPATIAL_RATES_NAME,
)


def _make_per_unit_glm_result(
    env, *, n_units: int = 3, rank: int = 5, seed: int = 0, unit_table=None
):
    """A ``pooled=False`` glm result with per-unit vectors (one fallback unit).

    Unit ``n_units - 1`` is a zero-spike fallback: ``penalty_selected_by_reml``
    is ``False`` and ``reml_objective`` is ``nan`` (the documented sentinel).
    """
    from neurospatial.encoding.spatial import SpatialRatesResult

    rng = np.random.default_rng(seed)
    penalty = rng.uniform(0.1, 5.0, n_units)
    reml_obj = rng.uniform(-500.0, 500.0, n_units)
    reml_obj[-1] = np.nan  # zero-spike fallback sentinel
    boundary = np.array([False] * (n_units - 1) + [True], dtype=bool)
    selected = np.array([True] * (n_units - 1) + [False], dtype=bool)
    return SpatialRatesResult(
        firing_rates=rng.uniform(0.01, 10.0, (n_units, env.n_bins)),
        occupancy=rng.uniform(0.0, 5.0, env.n_bins),
        env=env,
        method="glm",
        bandwidth=None,
        coefficients=rng.standard_normal((rank, n_units)),
        penalty=penalty,
        penalty_weights=rng.uniform(0.0, 2.0, rank),
        rank=rank,
        deviance=rng.uniform(0.0, 100.0, n_units),
        converged=True,
        n_iter=9,
        reml_objective=reml_obj,
        reml_at_boundary=boundary,
        penalty_selected_by_reml=selected,
        pooled=False,
        unit_table=unit_table,
    )


def _make_pooled_glm_result(
    env,
    *,
    penalty=0.75,
    reml_objective=-12.3,
    reml_at_boundary=False,
    seed=0,
    unit_table=None,
):
    """A ``pooled=True`` glm result (scalar diagnostics)."""
    from neurospatial.encoding.spatial import SpatialRatesResult

    rng = np.random.default_rng(seed)
    rank = 5
    return SpatialRatesResult(
        firing_rates=rng.uniform(0.01, 10.0, (3, env.n_bins)),
        occupancy=rng.uniform(0.0, 5.0, env.n_bins),
        env=env,
        method="glm",
        bandwidth=None,
        coefficients=rng.standard_normal((rank, 3)),
        penalty=penalty,
        penalty_weights=rng.uniform(0.0, 2.0, rank),
        rank=rank,
        deviance=rng.uniform(0.0, 100.0, 3),
        converged=True,
        n_iter=7,
        reml_objective=reml_objective,
        reml_at_boundary=reml_at_boundary,
        pooled=True,
        unit_table=unit_table,
    )


def test_pooled_false_nwb_roundtrip(empty_nwb, sample_environment):
    """A pooled=False result round-trips penalty / reml_objective /
    reml_at_boundary / penalty_selected_by_reml as per-unit columns; scalar/vector
    shapes are preserved and the nan fallback sentinel survives."""
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    env = sample_environment
    result = _make_per_unit_glm_result(env, n_units=3, rank=5, seed=1)

    write_spatial_rates(empty_nwb, result)
    back = read_spatial_rates(empty_nwb, env=env)

    assert back.pooled is False
    np.testing.assert_allclose(np.asarray(back.penalty), np.asarray(result.penalty))
    np.testing.assert_allclose(
        np.asarray(back.reml_objective),
        np.asarray(result.reml_objective),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        np.asarray(back.reml_at_boundary), np.asarray(result.reml_at_boundary)
    )
    np.testing.assert_array_equal(
        np.asarray(back.penalty_selected_by_reml),
        np.asarray(result.penalty_selected_by_reml),
    )
    # The per-unit fields are vectors (not scalars) round-tripped.
    assert np.asarray(back.penalty).shape == (3,)
    assert np.asarray(back.penalty_selected_by_reml).dtype == np.bool_


def test_pooled_true_nwb_roundtrip_scalar_boundary(empty_nwb, sample_environment):
    """A pooled automatic-REML result stores a scalar reml_at_boundary in the blob
    (no per-unit columns, penalty_selected_by_reml None on read)."""
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    env = sample_environment
    result = _make_pooled_glm_result(env, reml_at_boundary=True)

    write_spatial_rates(empty_nwb, result)
    back = read_spatial_rates(empty_nwb, env=env)

    assert back.pooled is True
    assert back.reml_at_boundary is True
    assert np.isscalar(back.penalty) or isinstance(back.penalty, float)
    assert back.penalty_selected_by_reml is None


def test_fixed_penalty_nwb_boundary_none(empty_nwb, sample_environment):
    """A fixed-penalty glm result (REML skipped) keeps reml_at_boundary None."""
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    env = sample_environment
    result = _make_pooled_glm_result(
        env, penalty=2.5, reml_objective=None, reml_at_boundary=None
    )

    write_spatial_rates(empty_nwb, result)
    back = read_spatial_rates(empty_nwb, env=env)

    assert back.pooled is True
    assert back.reml_at_boundary is None
    assert back.reml_objective is None
    assert float(back.penalty) == pytest.approx(2.5)


@pytest.mark.parametrize(
    "key",
    [
        "rank",
        "n_iter",
        "converged",
        "penalty_weights",
        "pooled",
        "per_unit",
        "penalty",
        "reml_objective",
        "reml_at_boundary",
    ],
)
def test_missing_gam_metadata_key_is_refused(empty_nwb, sample_environment, key):
    """A GAM key the writer always stores is never defaulted on read."""
    import re

    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    env = sample_environment
    write_spatial_rates(empty_nwb, _make_pooled_glm_result(env))
    table = empty_nwb.analysis[DEFAULT_SPATIAL_RATES_NAME]
    meta = json.loads(table.description)
    meta["gam"].pop(key)
    table.fields["description"] = json.dumps(meta)

    message = (
        f"Corrupt spatial-rates record '{DEFAULT_SPATIAL_RATES_NAME}': missing "
        f"metadata key 'gam.{key}'."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        read_spatial_rates(empty_nwb, env=env)


def test_glm_record_without_gam_metadata_is_refused(empty_nwb, sample_environment):
    """A glm record whose GAM blob is gone cannot be reconstructed."""
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    env = sample_environment
    write_spatial_rates(empty_nwb, _make_pooled_glm_result(env))
    table = empty_nwb.analysis[DEFAULT_SPATIAL_RATES_NAME]
    meta = json.loads(table.description)
    del meta["gam"]
    table.fields["description"] = json.dumps(meta)

    with pytest.raises(ValueError, match="missing metadata key 'gam'"):
        read_spatial_rates(empty_nwb, env=env)


def test_pooled_survives_nwb_and_indexing(empty_nwb, sample_environment):
    """.pooled survives an NWB round-trip and singular indexing."""
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates

    env = sample_environment
    result = _make_per_unit_glm_result(env, n_units=3, rank=5, seed=2)

    write_spatial_rates(empty_nwb, result)
    back = read_spatial_rates(empty_nwb, env=env)
    assert back.pooled is False
    assert back[0].pooled is False


def test_pooled_true_write_allows_per_unit_column_name_in_unit_table(
    empty_nwb, sample_environment
):
    """A pooled=True glm write does NOT reserve the per-unit-lambda column names
    (they are only written for pooled=False), so a unit_table column literally
    named 'penalty' is accepted -- while a pooled=False write still rejects it."""
    import pandas as pd

    from neurospatial.io.nwb import write_spatial_rates

    env = sample_environment
    # pooled=True: 'penalty' unit_table column is fine (no penalty column emitted).
    ok = _make_pooled_glm_result(env, unit_table=pd.DataFrame({"penalty": [1, 2, 3]}))
    write_spatial_rates(empty_nwb, ok, name="pooled_ok")

    # pooled=False: 'penalty' IS a written column -> still reserved -> reject.
    per_unit = _make_per_unit_glm_result(
        env, n_units=3, rank=5, seed=3, unit_table=pd.DataFrame({"penalty": [1, 2, 3]})
    )
    with pytest.raises(ValueError, match="reserved"):
        write_spatial_rates(empty_nwb, per_unit, name="pooled_false_clash")


def test_truncated_per_unit_file_reads_with_clear_error(empty_nwb, sample_environment):
    """A per_unit=True record missing a per-unit column reads with a schema-aware
    ValueError naming the table + missing column, not a raw hdmf KeyError."""
    from neurospatial.io.nwb import read_spatial_rates, write_spatial_rates
    from neurospatial.io.nwb._fields import (
        DEFAULT_SPATIAL_RATES_NAME,
    )

    env = sample_environment
    # Write a pooled=True result (no per-unit columns), then lie in the blob that
    # it is per_unit -- exactly a truncated / corrupt per-unit record.
    write_spatial_rates(empty_nwb, _make_pooled_glm_result(env))
    table = empty_nwb.analysis[DEFAULT_SPATIAL_RATES_NAME]
    meta = json.loads(table.description)
    meta["gam"]["per_unit"] = True
    table.fields["description"] = json.dumps(meta)

    with pytest.raises(ValueError, match=r"per-unit column"):
        read_spatial_rates(empty_nwb, env=env)
