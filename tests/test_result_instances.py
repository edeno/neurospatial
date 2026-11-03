"""Every public result class has a factory in ``tests/result_instances.py``."""

from __future__ import annotations

import pytest
from result_instances import RESULT_FACTORIES, RESULT_VARIANTS, public_result_classes

CLASSES = public_result_classes()


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_factory_builds_each_public_result_class(name):
    assert name in RESULT_FACTORIES, f"add a factory for {name} to result_instances.py"
    assert isinstance(RESULT_FACTORIES[name](), CLASSES[name])


@pytest.mark.parametrize("name", sorted(RESULT_VARIANTS))
def test_variant_builds_its_public_result_class(name):
    """A ``"ClassName[form]"`` variant builds an instance of ``ClassName``."""
    assert isinstance(RESULT_VARIANTS[name](), CLASSES[name.partition("[")[0]])
