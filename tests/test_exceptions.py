"""Tests for the public exception hierarchy.

The package exposes a small set of custom exception classes from
``neurospatial._exceptions`` (and re-exports them from
``neurospatial``). Each one inherits from a stdlib base so existing
broader ``except`` blocks keep working.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
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


@pytest.mark.parametrize(
    "name,builtin",
    [
        ("NeurospatialError", Exception),
        ("RegionNotFoundError", KeyError),
        ("BinIndexOutOfRangeError", ValueError),
        ("IncompatibleEnvironmentError", ValueError),
        ("LayoutNotBuiltError", RuntimeError),
        ("EnvironmentNotFittedError", RuntimeError),
        ("GraphValidationError", ValueError),
    ],
)
def test_every_library_error_is_a_neurospatial_error(name, builtin):
    import neurospatial

    cls = getattr(neurospatial, name)
    assert issubclass(cls, neurospatial.NeurospatialError)
    assert cls.__mro__[1] is builtin
    assert name in neurospatial.__all__


def test_region_not_found_prints_fix_unquoted():
    exc = RegionNotFoundError("hom", available=["home"])
    assert "\nFix: pass region_name='home'" in str(exc)
    assert not str(exc).startswith("'")
    assert isinstance(exc, (KeyError, ValueError))
    assert isinstance(exc, ValueError)


def test_region_not_found_names_the_argument():
    exc = RegionNotFoundError("home", available=[], argument="start_region")
    assert "This environment has no regions" in str(exc)
    assert "env.regions.add('home'" in str(exc)
    assert "start_region='home'" in str(exc)


def test_region_not_found_handles_missing_name():
    exc = RegionNotFoundError(None, available=["home"], argument="start_region")
    assert isinstance(exc, ValueError)
    assert "start_region=None" in str(exc)
    assert "Fix: pass start_region='home'" in str(exc)


@pytest.mark.parametrize("name,available", [("hom", ["home"]), ("home", [])])
def test_region_not_found_formats_end_regions_as_a_list(name, available):
    exc = RegionNotFoundError(name, available=available, argument="end_regions")
    assert "end_regions=['home']" in str(exc)


def test_no_import_cycle():
    source = Path(__file__).parents[1] / "src" / "neurospatial"
    tree = ast.parse((source / "_exceptions.py").read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.startswith("neurospatial") for name in imports)
    queries = ast.parse((source / "environment" / "queries.py").read_text())
    for node in ast.walk(queries):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert not any(
                isinstance(child, ast.ImportFrom)
                and child.module == "neurospatial._exceptions"
                for child in ast.walk(node)
            )


@pytest.mark.parametrize("engine", ["grid", "mesh"])
def test_layout_not_built_sites(engine):
    from neurospatial.layout.engines.regular_grid import RegularGridLayout
    from neurospatial.layout.engines.triangular_mesh import TriangularMeshLayout

    layout = RegularGridLayout() if engine == "grid" else TriangularMeshLayout()
    with pytest.raises(LayoutNotBuiltError) as caught:
        layout.point_to_bin_index(np.array([[0.0, 0.0]]))
    assert isinstance(caught.value, RuntimeError)
    assert str(caught.value).splitlines()[-1].startswith("Fix: ")


def test_bin_index_out_of_range_site():
    from neurospatial import Environment

    env = Environment.from_samples(np.arange(10.0)[:, None], bin_size=2.0)
    with pytest.raises(BinIndexOutOfRangeError) as caught:
        env.neighbors(env.n_bins)
    assert caught.value.index == env.n_bins
    assert "Fix:" in str(caught.value)


def test_incompatible_environment_site():
    from neurospatial import Environment
    from neurospatial.composite import CompositeEnvironment

    points = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
    a = Environment.from_samples(points[:, :1], bin_size=2.0)
    b = Environment.from_samples(points, bin_size=2.0)
    with pytest.raises(IncompatibleEnvironmentError) as caught:
        CompositeEnvironment([a, b])
    assert "[E1003]" in str(caught.value)
    assert "Fix:" in str(caught.value)
