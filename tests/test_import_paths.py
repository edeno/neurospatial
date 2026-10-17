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


def test_environment_has_no_mixin_hosts():
    from typing import Generic

    from neurospatial import Environment

    assert Environment.__bases__ == (Generic,)


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
        # Serialization (pickle ``save``/``load`` removed — only
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
        # Metrics / fields (``bin_attributes`` and
        # ``edge_attributes`` converted from @cached_property to method form)
        "boundary_bins",
        "get_bin_attributes",
        "get_edge_attributes",
        "to_linear",
        "linear_to_nd",
        "linearization_properties",
        # Core
        "info",
        # Properties
        "n_dims",
        "n_bins",
        "layout_type",
        "layout_parameters",
        "is_linearized_track",
    ]
    missing = [name for name in expected if not hasattr(Environment, name)]
    assert not missing, f"Environment is missing public surface: {missing}"
