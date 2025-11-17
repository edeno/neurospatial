"""Tests for Environment import paths after modularization.

This test module verifies that the refactoring from a monolithic environment.py
to a modular environment/ package maintains backward compatibility.

Tests verify:
1. Primary import path works: from neurospatial import Environment
2. Direct import path works: from neurospatial.environment import Environment
3. Both paths return the same class object
4. check_fitted decorator can be imported
5. Mixins are plain classes (not dataclasses)
6. Environment itself IS a dataclass
7. Factory methods return Environment type (not mixin types)
8. Method Resolution Order (MRO) is correct
"""

import numpy as np


def test_primary_import():
    """Test that primary import path still works.

    This is the main import path that users rely on and must
    remain functional after refactoring.
    """
    from neurospatial import Environment

    assert Environment is not None
    assert hasattr(Environment, "from_samples")
    assert hasattr(Environment, "bin_at")


def test_direct_import():
    """Test that direct import path also works.

    After refactoring, users should be able to import directly
    from the environment subpackage.
    """
    from neurospatial.environment import Environment

    assert Environment is not None
    assert hasattr(Environment, "from_samples")
    assert hasattr(Environment, "bin_at")


def test_imports_are_same():
    """Test that both import paths return the same class.

    Both import paths must resolve to the exact same class object
    to ensure consistent behavior and isinstance() checks.
    """
    from neurospatial import Environment as Env1
    from neurospatial.environment import Environment as Env2

    assert Env1 is Env2, "Both import paths should return the same class object"


def test_check_fitted_import():
    """Test that decorator can be imported from environment package."""
    from neurospatial.environment import check_fitted

    assert check_fitted is not None
    assert callable(check_fitted)


def test_mixins_are_not_dataclasses():
    """Ensure mixins are plain classes, not dataclasses.

    This is critical: only Environment should be a dataclass.
    Mixins must be plain classes to avoid MRO conflicts with
    dataclass field inheritance.
    """
    from neurospatial.environment.factories import EnvironmentFactories
    from neurospatial.environment.fields import EnvironmentFields
    from neurospatial.environment.metrics import EnvironmentMetrics
    from neurospatial.environment.queries import EnvironmentQueries
    from neurospatial.environment.regions import EnvironmentRegions
    from neurospatial.environment.serialization import EnvironmentSerialization
    from neurospatial.environment.trajectory import EnvironmentTrajectory
    from neurospatial.environment.transforms import EnvironmentTransforms
    from neurospatial.environment.visualization import EnvironmentVisualization

    mixins = [
        EnvironmentFactories,
        EnvironmentQueries,
        EnvironmentSerialization,
        EnvironmentRegions,
        EnvironmentVisualization,
        EnvironmentMetrics,
        EnvironmentFields,
        EnvironmentTrajectory,
        EnvironmentTransforms,
    ]

    for mixin in mixins:
        assert not hasattr(mixin, "__dataclass_fields__"), (
            f"{mixin.__name__} should NOT be a dataclass"
        )


def test_environment_is_dataclass():
    """Ensure Environment itself IS a dataclass.

    Environment must be a dataclass to provide the data storage
    for all the mixin methods.
    """
    from neurospatial.environment import Environment

    assert hasattr(Environment, "__dataclass_fields__"), (
        "Environment should be a dataclass"
    )


def test_factory_methods_return_environment_type():
    """Verify factory classmethods return Environment, not mixin type.

    When calling Environment.from_samples(), the result should be
    an instance of Environment, not one of the mixin classes.
    """
    from neurospatial.environment import Environment

    data = np.random.rand(100, 2)
    env = Environment.from_samples(data, bin_size=2.0)

    assert isinstance(env, Environment)
    assert type(env).__name__ == "Environment"


def test_mro_order():
    """Verify Method Resolution Order is correct.

    Environment should be first in MRO, followed by mixins.
    This ensures proper method resolution and attribute access.
    """
    from neurospatial.environment import Environment
    from neurospatial.environment.factories import EnvironmentFactories
    from neurospatial.environment.fields import EnvironmentFields
    from neurospatial.environment.metrics import EnvironmentMetrics
    from neurospatial.environment.queries import EnvironmentQueries
    from neurospatial.environment.regions import EnvironmentRegions
    from neurospatial.environment.serialization import EnvironmentSerialization
    from neurospatial.environment.trajectory import EnvironmentTrajectory
    from neurospatial.environment.transforms import EnvironmentTransforms
    from neurospatial.environment.visualization import EnvironmentVisualization

    mro = Environment.__mro__

    # Environment should be first
    assert mro[0] is Environment

    # All mixins should be in MRO
    assert EnvironmentFactories in mro
    assert EnvironmentQueries in mro
    assert EnvironmentSerialization in mro
    assert EnvironmentRegions in mro
    assert EnvironmentVisualization in mro
    assert EnvironmentMetrics in mro
    assert EnvironmentFields in mro
    assert EnvironmentTrajectory in mro
    assert EnvironmentTransforms in mro


def test_all_public_methods_present():
    """Verify that all expected public methods are present on Environment.

    This ensures the mixin inheritance successfully combines all
    functionality from the separate modules.
    """
    from neurospatial.environment import Environment

    # Factory methods
    factory_methods = [
        "from_samples",
        "from_graph",
        "from_polygon",
        "from_mask",
        "from_image",
        "from_layout",
    ]

    # Query methods
    query_methods = [
        "bin_at",
        "contains",
        "neighbors",
        "distance_between",
        "bin_center_of",
        "path_between",
        "bin_sizes",
    ]

    # Serialization methods
    serialization_methods = [
        "to_file",
        "from_file",
        "to_dict",
        "from_dict",
        "save",
        "load",
    ]

    # Region methods
    region_methods = ["bins_in_region", "mask_for_region"]

    # Visualization methods
    visualization_methods = ["plot", "plot_1d"]

    # Analysis methods
    analysis_methods = [
        "boundary_bins",
        "bin_attributes",
        "edge_attributes",
        "to_linear",
        "linear_to_nd",
        "linearization_properties",
    ]

    # Core methods
    core_methods = ["info", "copy"]

    # Properties
    properties = ["n_dims", "n_bins", "layout_type", "layout_parameters", "is_1d"]

    all_expected = (
        factory_methods
        + query_methods
        + serialization_methods
        + region_methods
        + visualization_methods
        + analysis_methods
        + core_methods
        + properties
    )

    for method in all_expected:
        assert hasattr(Environment, method), (
            f"Environment should have method/property '{method}'"
        )
