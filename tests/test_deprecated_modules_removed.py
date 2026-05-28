"""Guard tests that assert deprecated module paths no longer resolve."""

from __future__ import annotations

import importlib
import sys

import pytest


class TestOldFilesDeleted:
    """Tests that old backward-compat files are properly deleted."""

    def test_behavioral_py_deleted(self) -> None:
        """Verify behavioral.py backward-compat wrapper is deleted.

        The old neurospatial.behavioral module was a re-export wrapper.
        Users should now import from neurospatial.behavior.navigation
        or neurospatial.behavior.trajectory.
        """
        # Clear any cached import
        if "neurospatial.behavioral" in sys.modules:
            del sys.modules["neurospatial.behavioral"]

        with pytest.raises(ModuleNotFoundError, match="behavioral"):
            importlib.import_module("neurospatial.behavioral")

    def test_metrics_package_deleted(self) -> None:
        """Verify entire metrics package has been deleted.

        The metrics package was a re-export layer that duplicated the API.
        Users should now import from canonical locations:
        - neurospatial.encoding.spatial (spatial rate metrics)
        - neurospatial.encoding.directional (HD cell metrics)
        - neurospatial.encoding.egocentric (OVC metrics)
        - neurospatial.behavior.navigation (navigation metrics)
        - neurospatial.behavior.vte (VTE metrics)
        """
        # Clear any cached imports
        for module_name in list(sys.modules.keys()):
            if module_name.startswith("neurospatial.metrics"):
                del sys.modules[module_name]

        with pytest.raises(ModuleNotFoundError, match="metrics"):
            importlib.import_module("neurospatial.metrics")

    @pytest.mark.parametrize(
        "module_name",
        [
            "neurospatial.encoding.place",
            "neurospatial.encoding.head_direction",
            "neurospatial.encoding.object_vector",
            "neurospatial.encoding.spatial_view",
        ],
    )
    def test_old_encoding_modules_deleted(self, module_name: str) -> None:
        """Verify old field-level encoding modules are deleted."""
        if module_name in sys.modules:
            del sys.modules[module_name]

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)
