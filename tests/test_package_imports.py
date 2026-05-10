"""Cross-module import sanity checks.

Two properties worth asserting at the package boundary:

1. Every name listed in a submodule's ``__all__`` actually resolves on the
   module — catches typos that would silently break ``from x import *``.
2. The classes re-exported from the top-level package are the same objects
   as their canonical submodule sources — catches accidental shadowing
   when ``neurospatial/__init__.py`` is reorganized.

The earlier version of this file enumerated ~40 import statements per
submodule and then iterated ``__all__`` once per submodule. The
parametrized pair below covers both properties with two test bodies.
"""

import importlib

import pytest

ALL_BEARING_MODULES = [
    "neurospatial",
    "neurospatial.encoding",
    "neurospatial.encoding.spatial",
    "neurospatial.encoding.grid",
    "neurospatial.encoding.directional",
    "neurospatial.encoding.border",
    "neurospatial.encoding.egocentric",
    "neurospatial.encoding.view",
    "neurospatial.encoding.phase_precession",
    "neurospatial.encoding.population",
    "neurospatial.decoding",
    "neurospatial.behavior",
    "neurospatial.behavior.navigation",
    "neurospatial.behavior.segmentation",
    "neurospatial.behavior.trajectory",
    "neurospatial.behavior.decisions",
    "neurospatial.behavior.vte",
    "neurospatial.behavior.reward",
    "neurospatial.ops",
    "neurospatial.ops.binning",
    "neurospatial.ops.distance",
    "neurospatial.ops.normalize",
    "neurospatial.ops.smoothing",
    "neurospatial.ops.graph",
    "neurospatial.ops.calculus",
    "neurospatial.ops.transforms",
    "neurospatial.ops.alignment",
    "neurospatial.ops.egocentric",
    "neurospatial.ops.visibility",
    "neurospatial.ops.basis",
    "neurospatial.stats",
    "neurospatial.stats.circular",
    "neurospatial.stats.surrogates",
    "neurospatial.events",
    "neurospatial.io",
    "neurospatial.animation",
    "neurospatial.simulation",
    "neurospatial.layout",
    "neurospatial.regions",
]


@pytest.mark.parametrize("module_name", ALL_BEARING_MODULES)
def test_all_names_resolve(module_name):
    """Every name in ``<module>.__all__`` is an attribute of ``<module>``."""
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "__all__"):
        pytest.skip(f"{module_name} has no __all__")
    for name in mod.__all__:
        assert hasattr(mod, name), (
            f"'{name}' is in {module_name}.__all__ but not on the module."
        )


@pytest.mark.parametrize(
    "top_name,submodule_name",
    [
        ("Environment", "neurospatial.environment"),
        ("Region", "neurospatial.regions"),
        ("Regions", "neurospatial.regions"),
        ("CompositeEnvironment", "neurospatial.composite"),
    ],
)
def test_top_level_is_same_object_as_submodule(top_name, submodule_name):
    """``neurospatial.X`` is the same object as ``<submodule>.X``."""
    import neurospatial

    submodule = importlib.import_module(submodule_name)
    assert getattr(neurospatial, top_name) is getattr(submodule, top_name)
