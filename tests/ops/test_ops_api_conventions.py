"""Signature/naming-convention tests for ops public functions.

Pins the keyword-only separators and the `random_state` -> `rng` rename on the
basis-function public surface, using ``inspect.signature`` for structural
assertions.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops.basis import (
    chebyshev_filter_basis,
    geodesic_rbf_basis,
    heat_kernel_wavelet_basis,
    select_basis_centers,
    spatial_basis,
)


@pytest.fixture
def simple_2d_env() -> Environment:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 10, size=(200, 2))
    return Environment.from_samples(positions, bin_size=1.0)


def test_basis_rng_keyword_deterministic(simple_2d_env):
    """Basis functions take a keyword-only `rng`; results are reproducible."""
    # All five public functions expose `rng`, not `random_state`, keyword-only.
    for fn in (
        select_basis_centers,
        geodesic_rbf_basis,
        heat_kernel_wavelet_basis,
        chebyshev_filter_basis,
        spatial_basis,
    ):
        sig = inspect.signature(fn)
        assert "rng" in sig.parameters, fn.__name__
        assert "random_state" not in sig.parameters, fn.__name__
        assert sig.parameters["rng"].kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__

    # `method` is keyword-only on select_basis_centers (n_centers is the only
    # positional-or-keyword argument after env).
    sig = inspect.signature(select_basis_centers)
    assert sig.parameters["method"].kind is inspect.Parameter.KEYWORD_ONLY

    # Reproducible across two calls with the same seed.
    c1 = select_basis_centers(simple_2d_env, 8, rng=0)
    c2 = select_basis_centers(simple_2d_env, 8, rng=0)
    np.testing.assert_array_equal(c1, c2)

    b1 = geodesic_rbf_basis(simple_2d_env, n_centers=8, rng=0)
    b2 = geodesic_rbf_basis(simple_2d_env, n_centers=8, rng=0)
    np.testing.assert_array_equal(b1, b2)

    # `method`/`rng` cannot be passed positionally any more.
    with pytest.raises(TypeError):
        select_basis_centers(simple_2d_env, 8, "kmeans")
    with pytest.raises(TypeError):
        select_basis_centers(simple_2d_env, 8, "kmeans", 0)

    # The removed keyword raises TypeError.
    with pytest.raises(TypeError):
        select_basis_centers(simple_2d_env, 8, random_state=0)


def test_dir_surfaces_lazily_exported_names():
    """dir(neurospatial.ops) includes lazily-exported names for autocomplete.

    Regression: the package uses module-level __getattr__ lazy loading, so
    lazily-exported ops (e.g. visibility) were absent from dir() -- weakening
    tab-completion -- until a __dir__ unioning the globals with __all__ was
    added.
    """
    import neurospatial.ops as ops

    names = set(dir(ops))
    assert len(ops.__all__) > 0
    assert set(ops.__all__).issubset(names)
