"""Bound and unbound calls explain the same Cartesian-only capability."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial._exceptions import CartesianOnlyError


@pytest.mark.parametrize("operation", ["to_linear", "linear_to_nd", "rebin", "subset"])
def test_grid_operation_reason_matches_bound_and_unbound_calls(operation):
    env = Environment.from_polar_egocentric((0, 2), (-np.pi, np.pi), 1, np.pi / 2)
    arguments = {
        "to_linear": ((np.array([[1.0, 0.0]]),), {}),
        "linear_to_nd": ((np.array([1.0]),), {}),
        "rebin": ((1,), {}),
        "subset": ((), {"bins": np.ones(env.n_bins, dtype=bool)}),
    }
    assert not hasattr(env, operation)
    with pytest.raises(CartesianOnlyError) as bound:
        getattr(env, operation)
    args, kwargs = arguments[operation]
    with pytest.raises(CartesianOnlyError) as unbound:
        getattr(Environment, operation)(env, *args, **kwargs)
    assert "(Cartesian-grid operation)" in str(unbound.value)
    assert str(unbound.value) == str(bound.value)
