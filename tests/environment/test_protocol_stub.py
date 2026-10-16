"""Narrow scientific consumer signatures agree with public entry points."""

from __future__ import annotations

import inspect

import pytest

from neurospatial import Environment
from neurospatial.environment._protocols import (
    CartesianLookupConsumer,
    TrajectoryBinningConsumer,
)


@pytest.mark.parametrize(
    "consumer,method",
    [
        (CartesianLookupConsumer, "bin_at"),
        (CartesianLookupConsumer, "bin_sequence"),
        (TrajectoryBinningConsumer, "compute_kernel"),
    ],
)
def test_consumer_call_contract_matches_public_method(consumer, method):
    declared = inspect.signature(getattr(consumer, method)).parameters
    actual = inspect.signature(getattr(Environment, method)).parameters
    assert list(declared) == list(actual)
    for name, parameter in declared.items():
        assert parameter.kind == actual[name].kind
        assert parameter.default == actual[name].default
