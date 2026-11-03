"""Every public result class has a factory in ``tests/result_instances.py``."""

from __future__ import annotations

import pytest
from result_instances import RESULT_FACTORIES, public_result_classes

CLASSES = public_result_classes()


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_factory_builds_each_public_result_class(name):
    assert name in RESULT_FACTORIES, f"add a factory for {name} to result_instances.py"
    assert isinstance(RESULT_FACTORIES[name](), CLASSES[name])
