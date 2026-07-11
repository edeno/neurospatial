"""Public contract for the ``method`` estimator-axis parameter.

The smoothing encoders expose their estimator choice as a single, uniformly
named keyword ``method`` (and result field ``.method``). These tests lock that
contract:

- every smoothing encoder accepts ``method=`` and, where the result carries it,
  records the value (and the explicit-default call is byte-identical to the
  default path, so the keyword is a pure name change, not a behavior change);
- the previous ``smoothing_method=`` keyword is rejected (hard rename, no alias);
- rate result classes expose ``.method`` and no longer ``.smoothing_method``;
- the egocentric encoders keep their ``"binned"`` default (the others default to
  ``"diffusion_kde"``).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from neurospatial.encoding.egocentric import (
    compute_egocentric_rate,
    compute_egocentric_rates,
)
from neurospatial.encoding.spatial import (
    compute_directional_place_fields,
    compute_spatial_rate,
    compute_spatial_rates,
    is_place_cell,
)
from neurospatial.encoding.view import compute_view_rate, compute_view_rates


# --- Call wrappers over the shared ``ovc_session`` fixture -------------------
# Each takes ``method`` (None => omit the keyword, exercising the default path)
# and returns the encoder's result. Plural encoders get a 2-unit spike list.
def _spatial_rate(session, method):
    env, st, t, pos, _hd, _obj = session
    kw = {} if method is None else {"method": method}
    return compute_spatial_rate(env, st, t, pos, **kw)


def _spatial_rates(session, method):
    env, st, t, pos, _hd, _obj = session
    kw = {} if method is None else {"method": method}
    return compute_spatial_rates(env, [st, st], t, pos, **kw)


def _view_rate(session, method):
    env, st, t, pos, hd, _obj = session
    kw = {} if method is None else {"method": method}
    return compute_view_rate(env, st, t, pos, hd, **kw)


def _view_rates(session, method):
    env, st, t, pos, hd, _obj = session
    kw = {} if method is None else {"method": method}
    return compute_view_rates(env, [st, st], t, pos, hd, **kw)


def _ego_rate(session, method):
    env, st, t, pos, hd, obj = session
    kw = {} if method is None else {"method": method}
    return compute_egocentric_rate(env, st, t, pos, hd, obj, **kw)


def _ego_rates(session, method):
    env, st, t, pos, hd, obj = session
    kw = {} if method is None else {"method": method}
    return compute_egocentric_rates(env, [st, st], t, pos, hd, obj, **kw)


def _is_place_cell(session, method):
    env, st, t, pos, _hd, _obj = session
    kw = {} if method is None else {"method": method}
    return is_place_cell(env, st, t, pos, **kw)


# (id, call, default_method): encoders whose explicit-default call must match the
# default path exactly, and (where a ``.method`` field exists) record the method.
KWARG_CASES = [
    ("compute_spatial_rate", _spatial_rate, "diffusion_kde"),
    ("compute_spatial_rates", _spatial_rates, "diffusion_kde"),
    ("compute_view_rate", _view_rate, "diffusion_kde"),
    ("compute_view_rates", _view_rates, "diffusion_kde"),
    ("compute_egocentric_rate", _ego_rate, "binned"),
    ("compute_egocentric_rates", _ego_rates, "binned"),
    ("is_place_cell", _is_place_cell, "diffusion_kde"),
]


def _firing(result):
    """Canonical numeric output for behavior-preserving comparison."""
    if hasattr(result, "firing_rates"):
        return np.asarray(result.firing_rates)
    if hasattr(result, "firing_rate"):
        return np.asarray(result.firing_rate)
    return np.asarray(result)  # is_place_cell -> bool


@pytest.mark.parametrize(
    "call, default_method",
    [(c[1], c[2]) for c in KWARG_CASES],
    ids=[c[0] for c in KWARG_CASES],
)
def test_method_kwarg_each_encoder(ovc_session, call, default_method):
    """``method=`` is accepted and is a pure rename of the default path."""
    r_explicit = call(ovc_session, default_method)
    r_default = call(ovc_session, None)

    # Passing the default value explicitly reproduces the default path exactly:
    # the renamed keyword is a pure name change with no behavior shift. (That the
    # keyword actually drives the estimator -- rather than being silently ignored
    # -- is shown by test_egocentric_default_preserved, where the "binned" and
    # "diffusion_kde" maps differ.)
    np.testing.assert_array_equal(_firing(r_explicit), _firing(r_default))

    # Where the result carries the estimator, it records the requested method.
    if hasattr(r_explicit, "method"):
        assert r_explicit.method == default_method


# Old-kwarg rejection covers every function that gained a ``method`` param,
# including compute_directional_place_fields (which needs direction_labels and
# whose result carries no ``.method`` field, so it is only covered here).
def _old_kwarg_calls(session):
    env, st, t, pos, hd, obj = session
    direction_labels = np.array(["N"] * len(t))
    return {
        "compute_spatial_rate": lambda: compute_spatial_rate(
            env, st, t, pos, smoothing_method="diffusion_kde"
        ),
        "compute_spatial_rates": lambda: compute_spatial_rates(
            env, [st], t, pos, smoothing_method="diffusion_kde"
        ),
        "compute_view_rate": lambda: compute_view_rate(
            env, st, t, pos, hd, smoothing_method="diffusion_kde"
        ),
        "compute_view_rates": lambda: compute_view_rates(
            env, [st], t, pos, hd, smoothing_method="diffusion_kde"
        ),
        "compute_egocentric_rate": lambda: compute_egocentric_rate(
            env, st, t, pos, hd, obj, smoothing_method="binned"
        ),
        "compute_egocentric_rates": lambda: compute_egocentric_rates(
            env, [st], t, pos, hd, obj, smoothing_method="binned"
        ),
        "is_place_cell": lambda: is_place_cell(
            env, st, t, pos, smoothing_method="diffusion_kde"
        ),
        "compute_directional_place_fields": lambda: compute_directional_place_fields(
            env, st, t, pos, direction_labels, smoothing_method="diffusion_kde"
        ),
    }


@pytest.mark.parametrize(
    "encoder_id",
    [
        "compute_spatial_rate",
        "compute_spatial_rates",
        "compute_view_rate",
        "compute_view_rates",
        "compute_egocentric_rate",
        "compute_egocentric_rates",
        "is_place_cell",
        "compute_directional_place_fields",
    ],
)
def test_old_kwarg_rejected(ovc_session, encoder_id):
    """The legacy ``smoothing_method=`` keyword no longer exists (no alias)."""
    call = _old_kwarg_calls(ovc_session)[encoder_id]
    with pytest.raises(TypeError, match="smoothing_method"):
        call()


# Only the spatial/view rate results persist the estimator as ``.method``; the
# egocentric results do not carry it (they never stored ``smoothing_method``
# either), so they are excluded from the field-rename check.
_FIELD_CASES = [
    c
    for c in KWARG_CASES
    if c[0]
    in {
        "compute_spatial_rate",
        "compute_spatial_rates",
        "compute_view_rate",
        "compute_view_rates",
    }
]


@pytest.mark.parametrize(
    "call, default_method",
    [(c[1], c[2]) for c in _FIELD_CASES],
    ids=[c[0] for c in _FIELD_CASES],
)
def test_result_field_renamed(ovc_session, call, default_method):
    """Rate results expose ``.method``; the old ``.smoothing_method`` is gone."""
    result = call(ovc_session, default_method)
    assert result.method == default_method
    assert not hasattr(result, "smoothing_method")


def test_egocentric_default_preserved(ovc_session):
    """Egocentric encoders default to ``"binned"`` (not ``"diffusion_kde"``)."""
    env, st, t, pos, hd, obj = ovc_session

    for func in (compute_egocentric_rate, compute_egocentric_rates):
        assert inspect.signature(func).parameters["method"].default == "binned"

    # Behaviorally, omitting ``method`` matches ``method="binned"`` and differs
    # from ``method="diffusion_kde"`` -- the default is binned, not KDE.
    default = np.asarray(compute_egocentric_rate(env, st, t, pos, hd, obj).firing_rate)
    binned = np.asarray(
        compute_egocentric_rate(env, st, t, pos, hd, obj, method="binned").firing_rate
    )
    diffusion = np.asarray(
        compute_egocentric_rate(
            env, st, t, pos, hd, obj, method="diffusion_kde"
        ).firing_rate
    )
    np.testing.assert_array_equal(default, binned)
    assert not np.array_equal(default, diffusion)
