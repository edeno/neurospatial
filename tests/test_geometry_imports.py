"""Owned geometry imports and compatibility aliases have one implementation."""

import copy
import importlib
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
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


@pytest.mark.parametrize(
    "old,new,name",
    [
        ("layout._binning", "geometry._binning", "Binning"),
        ("layout._locators", "geometry._locators", "SubsetLocator"),
        ("layout._diffusion", "geometry._diffusion", "FaceGeometry"),
        ("layout._polygon", "geometry._polygon", "ClippedLocator"),
        ("layout._composition", "geometry._composition", "CombinedLocator"),
        ("environment._coordinates", "geometry._coordinates", "CartesianSystem"),
        ("arena", "geometry._arena", "_TransformedArena"),
    ],
)
def test_legacy_pickle_modules_resolve_the_canonical_class(old, new, name):
    previous = importlib.import_module(f"neurospatial.{old}")
    current = importlib.import_module(f"neurospatial.{new}")
    assert getattr(previous, name) is getattr(current, name)


def test_pre_move_pickle_retains_geometry_and_independent_array_headers():
    """Trusted fixture written from the pre-move builder-free commit 3999fb28."""
    from neurospatial.binning import Binning

    fixture = Path(__file__).parent / "data" / "geometry_before_package_move.pkl"
    restored = pickle.loads(fixture.read_bytes())
    env, binning = restored["environment"], restored["binning"]
    assert type(binning) is Binning
    centers = np.array([[0.5], [1.5]])
    np.testing.assert_array_equal(env.bin_centers, centers)
    np.testing.assert_array_equal(env.bin_at(centers), [0, 1])
    np.testing.assert_array_equal(env.bin_sizes, [1.0, 1.0])
    np.testing.assert_allclose(env.smooth(np.ones(2), 0.7), 1.0, atol=1e-14)
    header = env.bin_centers
    header.shape = (2,)
    np.testing.assert_array_equal(env.bin_centers, centers)
    with pytest.raises(ValueError):
        header.setflags(write=True)
    detached = copy.deepcopy(binning)
    detached.bins.centers.shape = (2,)
    np.testing.assert_array_equal(binning.bins.centers, centers)
    assert not binning.bins.centers.flags.writeable
    assert not np.shares_memory(detached.bins.centers, binning.bins.centers)
