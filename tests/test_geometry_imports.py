"""Owned geometry imports without builders, analysis or rendering."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "neurospatial",
        "neurospatial.binning",
        "neurospatial.arena",
        "neurospatial.geometry._validation",
        "neurospatial.geometry._explicit",
    ],
)
def test_geometry_imports_without_builders_analysis_or_rendering(module):
    code = f"""
import importlib
import sys
importlib.import_module({module!r})
for prefix in (
    'neurospatial.environment', 'neurospatial.layout', 'neurospatial.ops',
    'neurospatial.encoding', 'neurospatial.decoding', 'matplotlib',
    'pynwb', 'napari', 'xarray', 'pynapple',
):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
