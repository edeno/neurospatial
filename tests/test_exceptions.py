"""Tests for the public exception hierarchy.

The package exposes a small set of custom exception classes from
``neurospatial._exceptions`` (and re-exports them from
``neurospatial``). Each one inherits from a stdlib base so existing
broader ``except`` blocks keep working.
"""

from __future__ import annotations

import pytest

from neurospatial._exceptions import (
    BinIndexOutOfRangeError,
    EnvironmentNotFittedError,
    GraphValidationError,
    IncompatibleEnvironmentError,
    LayoutNotBuiltError,
    RegionNotFoundError,
)


class TestRegionNotFoundError:
    def test_inherits_from_key_error(self):
        exc = RegionNotFoundError("goal")
        assert isinstance(exc, KeyError)

    def test_message_includes_region_name(self):
        exc = RegionNotFoundError("goal")
        assert "goal" in str(exc)

    def test_message_lists_available_when_provided(self):
        exc = RegionNotFoundError("goal", available=["start", "feeder"])
        msg = str(exc)
        assert "goal" in msg
        assert "start" in msg
        assert "feeder" in msg

    def test_attributes(self):
        exc = RegionNotFoundError("goal", available=["start"])
        assert exc.region_name == "goal"
        assert exc.available == ["start"]


class TestBinIndexOutOfRangeError:
    def test_inherits_from_value_error(self):
        exc = BinIndexOutOfRangeError(99, n_bins=42)
        assert isinstance(exc, ValueError)

    def test_message_contains_index_and_range(self):
        exc = BinIndexOutOfRangeError(99, n_bins=42)
        msg = str(exc)
        assert "99" in msg
        assert "42" in msg
        assert "[0, 42)" in msg

    def test_attributes(self):
        exc = BinIndexOutOfRangeError(99, n_bins=42)
        assert exc.index == 99
        assert exc.n_bins == 42


class TestIncompatibleEnvironmentError:
    def test_inherits_from_value_error(self):
        exc = IncompatibleEnvironmentError("envs disagree on n_dims")
        assert isinstance(exc, ValueError)

    def test_attributes_default_to_none(self):
        exc = IncompatibleEnvironmentError("envs disagree on n_dims")
        assert exc.first is None
        assert exc.second is None

    def test_attributes_keep_supplied_objects(self):
        a, b = object(), object()
        exc = IncompatibleEnvironmentError("disagree", first=a, second=b)
        assert exc.first is a
        assert exc.second is b


class TestLayoutNotBuiltError:
    def test_inherits_from_runtime_error(self):
        exc = LayoutNotBuiltError("RegularGridLayout", "connectivity")
        assert isinstance(exc, RuntimeError)

    def test_message_includes_layout_and_attribute(self):
        exc = LayoutNotBuiltError("RegularGridLayout", "connectivity")
        msg = str(exc)
        assert "RegularGridLayout.connectivity" in msg
        assert "build()" in msg

    def test_attributes(self):
        exc = LayoutNotBuiltError("RegularGridLayout", "connectivity")
        assert exc.layout_name == "RegularGridLayout"
        assert exc.attribute == "connectivity"


class TestPublicReExports:
    """The exception classes must be importable from the top-level ``neurospatial`` package."""

    @pytest.mark.parametrize(
        "name",
        [
            "BinIndexOutOfRangeError",
            "EnvironmentNotFittedError",
            "GraphValidationError",
            "IncompatibleEnvironmentError",
            "LayoutNotBuiltError",
            "RegionNotFoundError",
        ],
    )
    def test_top_level_exports(self, name):
        import neurospatial

        assert hasattr(neurospatial, name), (
            f"{name} should be importable from `neurospatial`"
        )
        assert name in neurospatial.__all__

    def test_environment_not_fitted_error_is_same_class(self):
        from neurospatial.environment.decorators import (
            EnvironmentNotFittedError as DecoratorEnvErr,
        )

        assert EnvironmentNotFittedError is DecoratorEnvErr

    def test_graph_validation_error_is_same_class(self):
        from neurospatial.layout.validation import (
            GraphValidationError as LayoutGraphErr,
        )

        assert GraphValidationError is LayoutGraphErr
