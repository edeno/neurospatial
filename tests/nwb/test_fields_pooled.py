"""NWB persistence of the per-neuron-lambda (``pooled=False``) diagnostics.

Schema 2.2 adds ``pooled`` and a scalar ``reml_at_boundary`` to the GAM metadata
blob, and -- when ``penalty`` / ``reml_objective`` / ``reml_at_boundary`` are
per-unit vectors -- persists those plus ``penalty_selected_by_reml`` as per-unit
table columns. A schema-2.1 file has none of these keys and reads back
method-conditionally: ``pooled=True`` for glm, ``pooled=None`` for ratio.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# Skip every test here when pynwb is absent (HAS_PYNWB gate).
pytest.importorskip("pynwb")

from neurospatial.io.nwb._fields import (
    DEFAULT_ANALYSIS_MODULE,
    DEFAULT_SPATIAL_RATES_NAME,
)


def _make_per_unit_glm_result(env, *, n_units: int = 3, rank: int = 5, seed: int = 0):
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
    )


def _make_pooled_glm_result(
    env, *, penalty=0.75, reml_objective=-12.3, reml_at_boundary=False, seed=0
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
    )


def test_pooled_false_nwb_roundtrip(empty_nwb, sample_environment):
    """A pooled=False result round-trips penalty / reml_objective /
    reml_at_boundary / penalty_selected_by_reml as per-unit columns; scalar/vector
    shapes are preserved and the nan fallback sentinel survives."""
    from neurospatial.io.nwb import read_place_field, write_spatial_rates

    env = sample_environment
    result = _make_per_unit_glm_result(env, n_units=3, rank=5, seed=1)

    write_spatial_rates(empty_nwb, result)
    back = read_place_field(empty_nwb, env=env)

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
    from neurospatial.io.nwb import read_place_field, write_spatial_rates

    env = sample_environment
    result = _make_pooled_glm_result(env, reml_at_boundary=True)

    write_spatial_rates(empty_nwb, result)
    back = read_place_field(empty_nwb, env=env)

    assert back.pooled is True
    assert back.reml_at_boundary is True
    assert np.isscalar(back.penalty) or isinstance(back.penalty, float)
    assert back.penalty_selected_by_reml is None


def test_fixed_penalty_nwb_boundary_none(empty_nwb, sample_environment):
    """A fixed-penalty glm result (REML skipped) keeps reml_at_boundary None."""
    from neurospatial.io.nwb import read_place_field, write_spatial_rates

    env = sample_environment
    result = _make_pooled_glm_result(
        env, penalty=2.5, reml_objective=None, reml_at_boundary=None
    )

    write_spatial_rates(empty_nwb, result)
    back = read_place_field(empty_nwb, env=env)

    assert back.pooled is True
    assert back.reml_at_boundary is None
    assert back.reml_objective is None
    assert float(back.penalty) == pytest.approx(2.5)


def _write_then_downgrade_to_schema_2_1(nwbfile, result):
    """Write a glm result, then rewrite its description to the schema-2.1 shape --
    drop the schema-2.2 gam keys and downgrade schema_version."""
    from neurospatial.io.nwb import write_spatial_rates

    write_spatial_rates(nwbfile, result)
    table = nwbfile.processing[DEFAULT_ANALYSIS_MODULE][DEFAULT_SPATIAL_RATES_NAME]
    meta = json.loads(table.description)
    meta["schema_version"] = "2.1"
    gam = meta.get("gam")
    if isinstance(gam, dict):
        for key in ("pooled", "per_unit", "reml_at_boundary"):
            gam.pop(key, None)
    table.fields["description"] = json.dumps(meta)


def test_nwb_schema_2_1_glm_reads_pooled_true(empty_nwb, sample_environment):
    """A schema-2.1 glm file (no schema-2.2 keys) reads back as pooled=True,
    scalar penalty/reml_objective, reml_at_boundary None, and
    penalty_selected_by_reml None (method-conditional default)."""
    from neurospatial.io.nwb import read_place_field

    env = sample_environment
    result = _make_pooled_glm_result(env, penalty=0.5, reml_objective=-3.2)
    _write_then_downgrade_to_schema_2_1(empty_nwb, result)

    back = read_place_field(empty_nwb, env=env)
    assert back.pooled is True  # method-conditional default for glm
    assert np.isscalar(back.penalty) or isinstance(back.penalty, float)
    assert float(back.penalty) == pytest.approx(0.5)
    assert back.reml_at_boundary is None
    assert back.penalty_selected_by_reml is None


def test_nwb_legacy_ratio_pooled_none(empty_nwb, sample_environment):
    """A schema-2.1 ratio file (no pooled key) reads back as pooled=None -- pooled
    is meaningless for a ratio result (method-conditional default)."""
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.io.nwb import read_place_field, write_spatial_rates

    env = sample_environment
    rng = np.random.default_rng(0)
    result = SpatialRatesResult(
        firing_rates=rng.uniform(0, 10, (3, env.n_bins)),
        occupancy=rng.uniform(0, 5, env.n_bins),
        env=env,
        method="diffusion_kde",
        bandwidth=5.0,
    )
    write_spatial_rates(empty_nwb, result)
    # Downgrade the schema to 2.1; a ratio table never had a pooled key.
    table = empty_nwb.processing[DEFAULT_ANALYSIS_MODULE][DEFAULT_SPATIAL_RATES_NAME]
    meta = json.loads(table.description)
    meta["schema_version"] = "2.1"
    table.fields["description"] = json.dumps(meta)

    back = read_place_field(empty_nwb, env=env)
    assert back.pooled is None  # not True -- pooled is meaningless for ratio
    assert back.method == "diffusion_kde"


def test_pooled_survives_nwb_and_indexing(empty_nwb, sample_environment):
    """.pooled survives an NWB round-trip and singular indexing."""
    from neurospatial.io.nwb import read_place_field, write_spatial_rates

    env = sample_environment
    result = _make_per_unit_glm_result(env, n_units=3, rank=5, seed=2)

    write_spatial_rates(empty_nwb, result)
    back = read_place_field(empty_nwb, env=env)
    assert back.pooled is False
    assert back[0].pooled is False
