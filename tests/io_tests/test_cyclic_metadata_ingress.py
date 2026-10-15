"""Cyclic dictionaries are rejected before recursive component decoders run."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.layout import _codecs


@pytest.mark.parametrize("field", ["layout_parameters", "binning", "arena"])
def test_circular_metadata_is_checked_before_recursive_decode(field, monkeypatch):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    data = env.to_dict()
    container = data.setdefault(field, {})
    container["cycle"] = container

    def recursive_decoder_must_not_run(*args, **kwargs):
        pytest.fail("cyclic metadata reached a recursive decoder")

    monkeypatch.setattr(_codecs, "decode", recursive_decoder_must_not_run)
    with pytest.raises(
        ValueError, match=r"Invalid environment data.*dictionary"
    ) as error:
        Environment.from_dict(data)
    assert "Fix:" in str(error.value)


def test_shared_acyclic_metadata_is_valid():
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    data = env.to_dict()
    shared = {"recording": ["one", "two"]}
    data["layout_parameters"].update(first=shared, second=shared)
    restored = Environment.from_dict(data)
    np.testing.assert_array_equal(
        restored.bin_at(env.bin_centers), np.arange(env.n_bins)
    )
