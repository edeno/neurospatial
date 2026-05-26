"""Pin invariants of the modular Environment package.

After the move from a monolithic ``environment.py`` to an
``environment/`` package, two properties are worth pinning:

- The mixin classes must remain plain classes (not dataclasses).
  Mixing dataclasses into a dataclass produces broken MRO and field
  inheritance bugs that are very hard to debug.
- The public method/property surface that users rely on must remain on
  ``Environment``. If a mixin is renamed or accidentally dropped from
  the inheritance chain, this fails loudly.
"""


def test_mixins_are_not_dataclasses():
    """Only ``Environment`` itself is a dataclass; the mixins must be plain.

    Mixing dataclasses into another dataclass produces broken MRO and
    silently-wrong field inheritance. This is the architectural invariant
    that lets the mixin pattern work.
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
            f"{mixin.__name__} must not be a dataclass."
        )


def test_all_public_methods_present():
    """Every documented public method / property is on ``Environment``.

    Catches the case where the mixin chain drops a class on the floor or
    a method is renamed without updating the inheritance order.
    """
    from neurospatial.environment import Environment

    expected = [
        # Factories
        "from_samples",
        "from_graph",
        "from_polygon",
        "from_grid_mask",
        "from_pixel_mask",
        "from_layout",
        # Queries
        "bin_at",
        "contains",
        "neighbors",
        "distance_between",
        "bin_center_of",
        "path_between",
        "bin_sizes",
        # Serialization (M5.9 removed pickle ``save``/``load`` — only
        # JSON+npz remains)
        "to_file",
        "from_file",
        "to_dict",
        "from_dict",
        # Regions
        "bins_in_region",
        "region_mask",
        # Visualization
        "plot",
        "plot_1d",
        # Metrics / fields (M5.6 converted ``bin_attributes`` and
        # ``edge_attributes`` from @cached_property to method form)
        "boundary_bins",
        "get_bin_attributes",
        "get_edge_attributes",
        "to_linear",
        "linear_to_nd",
        "linearization_properties",
        # Core
        "info",
        "copy",
        # Properties
        "n_dims",
        "n_bins",
        "layout_type",
        "layout_parameters",
        "is_linearized_track",
    ]
    missing = [name for name in expected if not hasattr(Environment, name)]
    assert not missing, f"Environment is missing public surface: {missing}"
