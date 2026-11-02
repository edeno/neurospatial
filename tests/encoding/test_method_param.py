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
    has_place_field,
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
    return compute_view_rate(env, st, t, pos, hd, **kw, boundary_source="analysis")


def _view_rates(session, method):
    env, st, t, pos, hd, _obj = session
    kw = {} if method is None else {"method": method}
    return compute_view_rates(
        env, [st, st], t, pos, hd, **kw, boundary_source="analysis"
    )


def _ego_rate(session, method):
    env, st, t, pos, hd, obj = session
    kw = {} if method is None else {"method": method}
    return compute_egocentric_rate(env, st, t, pos, hd, obj, **kw)


def _ego_rates(session, method):
    env, st, t, pos, hd, obj = session
    kw = {} if method is None else {"method": method}
    return compute_egocentric_rates(env, [st, st], t, pos, hd, obj, **kw)


def _has_place_field(session, method):
    env, st, t, pos, _hd, _obj = session
    kw = {} if method is None else {"method": method}
    return has_place_field(env, st, t, pos, **kw)


# (id, call, default_method): encoders whose explicit-default call must match the
# default path exactly, and (where a ``.method`` field exists) record the method.
KWARG_CASES = [
    ("compute_spatial_rate", _spatial_rate, "diffusion_kde"),
    ("compute_spatial_rates", _spatial_rates, "diffusion_kde"),
    ("compute_view_rate", _view_rate, "diffusion_kde"),
    ("compute_view_rates", _view_rates, "diffusion_kde"),
    ("compute_egocentric_rate", _ego_rate, "binned"),
    ("compute_egocentric_rates", _ego_rates, "binned"),
    ("has_place_field", _has_place_field, "diffusion_kde"),
]


def _firing(result):
    """Canonical numeric output for behavior-preserving comparison."""
    if hasattr(result, "firing_rates"):
        return np.asarray(result.firing_rates)
    if hasattr(result, "firing_rate"):
        return np.asarray(result.firing_rate)
    return np.asarray(result)  # has_place_field -> bool


@pytest.mark.parametrize(
    ("call", "default_method"),
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
            env,
            st,
            t,
            pos,
            hd,
            smoothing_method="diffusion_kde",
            boundary_source="analysis",
        ),
        "compute_view_rates": lambda: compute_view_rates(
            env,
            [st],
            t,
            pos,
            hd,
            smoothing_method="diffusion_kde",
            boundary_source="analysis",
        ),
        "compute_egocentric_rate": lambda: compute_egocentric_rate(
            env, st, t, pos, hd, obj, smoothing_method="binned"
        ),
        "compute_egocentric_rates": lambda: compute_egocentric_rates(
            env, [st], t, pos, hd, obj, smoothing_method="binned"
        ),
        "has_place_field": lambda: has_place_field(
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
        "has_place_field",
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
    ("call", "default_method"),
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


# --- method carried into the terminal-verb projections (acceptance criterion) --
# The rate results must surface the estimator in every projection, not just as a
# field, so downstream tables/datasets record which estimator produced them.
def test_method_in_projections(ovc_session):
    """Plural rate results carry `method` in summary / to_dataframe / summary_table."""
    env, st, t, pos, hd, _obj = ovc_session
    cases = [
        compute_spatial_rates(env, [st, st], t, pos, method="binned"),
        compute_view_rates(
            env, [st, st], t, pos, hd, method="binned", boundary_source="analysis"
        ),
    ]
    for result in cases:
        assert result.summary()["method"] == "binned"
        df = result.to_dataframe()
        assert "method" in df.columns
        assert (df["method"] == "binned").all()
        table = result.summary_table()
        assert "method" not in table.columns
        assert table.attrs["method"] == "binned"


def test_method_in_xarray_attrs(ovc_session):
    """Plural rate results carry `method` in to_xarray() dataset attrs."""
    pytest.importorskip("xarray")
    env, st, t, pos, hd, _obj = ovc_session
    for result in (
        compute_spatial_rates(env, [st, st], t, pos, method="binned"),
        compute_view_rates(
            env, [st, st], t, pos, hd, method="binned", boundary_source="analysis"
        ),
    ):
        assert result.to_xarray().attrs["method"] == "binned"


def test_method_in_singular_projections(ovc_session):
    """Single-unit rate results carry `method` in summary / to_dataframe."""
    env, st, t, pos, hd, _obj = ovc_session
    for result in (
        compute_spatial_rate(env, st, t, pos, method="binned"),
        compute_view_rate(
            env, st, t, pos, hd, method="binned", boundary_source="analysis"
        ),
    ):
        assert result.summary()["method"] == "binned"
        df = result.to_dataframe()
        assert "method" in df.columns
        assert (df["method"] == "binned").all()


# --- Invalid-method error text is stable -------------------------------------
_INVALID_METHOD_SCRIPT = """
import numpy as np
from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rates
from neurospatial.encoding.view import compute_view_rates

rng = np.random.default_rng(0)
t = np.linspace(0, 10, 200)
pos = rng.uniform(0, 10, (200, 2))
env = Environment.from_samples(pos, bin_size=5.0)
for call in (
    lambda: compute_spatial_rates(env, [], t, pos, method="invalid"),
    lambda: compute_view_rates(env, [], t, pos, np.zeros(200), method="invalid"),
):
    try:
        call()
    except ValueError as err:
        print(str(err).splitlines()[0])
"""


def test_invalid_method_message_lists_options_in_declared_order():
    import os
    import subprocess
    import sys

    outputs = []
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run(
            [sys.executable, "-c", _INVALID_METHOD_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(done.stdout)
    assert len(set(outputs)) == 1
    spatial, view = outputs[0].strip().splitlines()
    assert spatial == (
        "method must be one of ['diffusion_kde', 'gaussian_kde', 'binned', 'glm'], "
        "got 'invalid'"
    )
    assert view == (
        "Rate estimation got method='invalid'; method must be one of "
        "'diffusion_kde', 'gaussian_kde', 'binned'."
    )


@pytest.fixture(scope="module")
def binned_recording():
    """A 5-minute random walk in a 100 cm box, 10 cm bins, and 900 spikes."""
    from neurospatial import Environment

    rng = np.random.default_rng(0)
    times = np.arange(0.0, 300.0, 1 / 30)
    positions = np.cumsum(rng.normal(0.0, 1.0, (times.size, 2)), axis=0) % 100.0
    env = Environment.from_samples(positions, bin_size=10.0, units="cm")
    spike_times = np.sort(np.random.default_rng(1).uniform(0.0, 300.0, 900))
    return env, spike_times, times, positions


def _histogram_rate(env, spike_times, times, positions):
    """Spike count / occupancy per bin, NaN where unvisited, from first principles.

    Each interval between samples k and k + 1 counts toward the bin of sample
    k, as do the spikes inside it.
    """
    bins = env.bin_at(positions)
    occupancy = np.bincount(bins[:-1], weights=np.diff(times), minlength=env.n_bins)
    interval = np.searchsorted(times, spike_times, side="right") - 1
    inside = (interval >= 0) & (interval < times.size - 1)
    counts = np.bincount(bins[interval[inside]], minlength=env.n_bins)
    visited = occupancy > 0
    return np.where(visited, counts / np.where(visited, occupancy, 1.0), np.nan)


@pytest.mark.filterwarnings("error::UserWarning")
def test_binned_defaults_to_no_smoothing(binned_recording):
    env, spike_times, times, positions = binned_recording
    expected = _histogram_rate(env, spike_times, times, positions)
    assert np.isfinite(expected).sum() > 0.9 * env.n_bins
    single = compute_spatial_rate(
        env, spike_times, times, positions, method="binned", max_gap=None
    )
    plural = compute_spatial_rates(
        env, [spike_times], times, positions, method="binned", max_gap=None
    )
    for rate in (single.firing_rate, plural.firing_rates[0]):
        np.testing.assert_array_equal(np.isnan(rate), np.isnan(expected))
        np.testing.assert_allclose(rate, expected, rtol=1e-12, equal_nan=True)
    assert single.bandwidth == 0.0
    assert single.analysis_settings["bandwidth"] == 0.0
    # An explicit bandwidth still smooths a binned map.
    smoothed = compute_spatial_rate(
        env,
        spike_times,
        times,
        positions,
        method="binned",
        bandwidth=5.0,
        max_gap=None,
    )
    assert smoothed.bandwidth == 5.0
    finite = np.isfinite(expected)
    assert np.max(np.abs(smoothed.firing_rate[finite] - expected[finite])) > 0.5
    assert np.nanstd(smoothed.firing_rate) < np.nanstd(expected)


@pytest.mark.filterwarnings("error::UserWarning")
def test_binned_view_defaults_to_no_smoothing(ovc_session):
    env, spike_times, times, positions, headings, _ = ovc_session
    kwargs = {"method": "binned", "boundary_source": "analysis"}
    default = compute_view_rate(env, spike_times, times, positions, headings, **kwargs)
    unsmoothed = compute_view_rate(
        env, spike_times, times, positions, headings, bandwidth=0.0, **kwargs
    )
    np.testing.assert_array_equal(default.firing_rate, unsmoothed.firing_rate)
    smoothed = compute_view_rate(
        env, spike_times, times, positions, headings, bandwidth=5.0, **kwargs
    )
    assert not np.allclose(smoothed.firing_rate, default.firing_rate, equal_nan=True)


def _polar_calls(session):
    """Every object-vector and boundary-vector entry point, given ``bandwidth``."""
    from shapely.geometry import box

    from neurospatial import Environment
    from neurospatial.encoding import (
        compute_boundary_vector_rate,
        compute_boundary_vector_rates,
        compute_object_vector_rate,
        compute_object_vector_rates,
        is_egocentric_object_vector_cell,
        object_vector_cell_significance,
    )

    _env, st, t, pos, hd, obj = session
    grid = {"distance_range": (0.0, 50.0), "distance_bin_size": 5.0}
    arena = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")
    walk = np.random.default_rng(0).uniform(5.0, 95.0, (t.size, 2))
    return {
        "compute_egocentric_rate": lambda **k: compute_egocentric_rate(
            None, st, t, pos, hd, obj, **grid, **k
        ),
        "compute_egocentric_rates": lambda **k: compute_egocentric_rates(
            None, [st], t, pos, hd, obj, **grid, **k
        ),
        "compute_object_vector_rate": lambda **k: compute_object_vector_rate(
            None, st, t, pos, obj, **grid, **k
        ),
        "compute_object_vector_rates": lambda **k: compute_object_vector_rates(
            None, [st], t, pos, obj, **grid, **k
        ),
        "object_vector_cell_significance": lambda **k: (
            object_vector_cell_significance(
                None, [st], t, pos, obj, n_shuffles=2, min_shift=1.0, rng=0, **grid, **k
            )
        ),
        "is_egocentric_object_vector_cell": lambda **k: (
            is_egocentric_object_vector_cell(
                None, st, t, pos, hd, obj, criterion="threshold", **grid, **k
            )
        ),
        "compute_boundary_vector_rate": lambda **k: compute_boundary_vector_rate(
            arena, st, t, walk, direction_frame="allocentric", **k
        ),
        "compute_boundary_vector_rates": lambda **k: compute_boundary_vector_rates(
            arena, [st], t, walk, direction_frame="allocentric", **k
        ),
    }


def test_binned_polar_maps_refuse_bandwidth(ovc_session):
    """A binned polar map is never smoothed, so a bandwidth is refused, not ignored."""
    for name, call in _polar_calls(ovc_session).items():
        with pytest.raises(
            ValueError, match=r"method='binned' got bandwidth=5\.0"
        ) as e:
            call(method="binned", bandwidth=5.0)
        message = str(e.value)
        assert "Why:" in message, name
        assert "Fix: use method='diffusion_kde'" in message, name
        assert "omit bandwidth" in message, name
        # No bandwidth and an explicit zero are the same unsmoothed map.
        call(method="binned")
        call(method="binned", bandwidth=0.0)


def test_min_occupancy_none_is_refused(ovc_session):
    """No encoder or decoder takes ``min_occupancy=None``; 0.0 is no masking."""
    from neurospatial.decoding import BayesianDecoder, decode_session
    from neurospatial.encoding import compute_object_vector_rate

    env, st, t, pos, hd, obj = ovc_session
    calls = {
        "compute_spatial_rate": lambda: compute_spatial_rate(
            env, st, t, pos, min_occupancy=None
        ),
        "compute_view_rate": lambda: compute_view_rate(
            env, st, t, pos, hd, boundary_source="analysis", min_occupancy=None
        ),
        "compute_object_vector_rate": lambda: compute_object_vector_rate(
            None,
            st,
            t,
            pos,
            obj,
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
            min_occupancy=None,
        ),
        "BayesianDecoder": lambda: BayesianDecoder(env, dt=0.25, min_occupancy=None),
        "decode_session": lambda: decode_session(
            env, [st], t, pos, dt=0.25, min_occupancy=None
        ),
        "decode_session(glm)": lambda: decode_session(
            env, [st], t, pos, dt=0.25, method="glm", min_occupancy=None
        ),
    }
    for name, call in calls.items():
        with pytest.raises(ValueError, match="min_occupancy=None") as caught:
            call()
        assert "Fix: pass min_occupancy=0.0 for no masking" in str(caught.value), name
