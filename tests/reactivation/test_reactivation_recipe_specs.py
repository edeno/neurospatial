"""Every published reactivation recipe, field by field, against the papers.

Each value below was checked against the recipe's ``notes`` and the paper's
text. A change to a recipe must change this table too, so a bin size, a
window or a test cannot drift silently.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from neurospatial.reactivation import (
    PUBLISHED_REACTIVATION_METHODS,
    ReactivationMethod,
    ReactivationWindows,
)

DEFAULTS: dict[str, Any] = {
    "pairs": "all",
    "control_windows": ReactivationWindows(),
    "match_windows": ReactivationWindows(),
    "detection": None,
    "smoothing_bandwidth": None,
    "activation_threshold": None,
    "statistic": "explained_variance",
    "test": None,
    "percentile": None,
    "n_shuffles": 1000,
    "alpha": 0.05,
    "n_rate_subsamples": None,
}

SPEC: dict[str, dict[str, Any]] = {
    # Kudrimoti et al. (1999): 100 ms bins, different tetrodes, last 10 min
    # of PRE SWS as control, three 10-min POST SWS blocks, partial-r test.
    "kudrimoti_1999": {
        "measure": "explained_variance",
        "bin_size": 0.1,
        "pairs": "different_tetrodes",
        "control_windows": ReactivationWindows(length=600.0, anchor="end", count=1),
        "match_windows": ReactivationWindows(length=600.0, count=3),
        "test": "partial_correlation_t",
    },
    # Pennartz et al. (2004): 50 ms, all pairs, last 10 min of REST1 and
    # 10-min blocks of REST2 on the clock, EV - REV across sessions.
    "pennartz_2004": {
        "measure": "explained_variance",
        "bin_size": 0.05,
        "control_windows": ReactivationWindows(
            length=600.0, anchor="end", count=1, clock="wall"
        ),
        "match_windows": ReactivationWindows(length=600.0, clock="wall"),
        "statistic": "ev_minus_rev",
    },
    # Tatsuno et al. (2006): 250 ms, different tetrodes, 15-min segments,
    # EV averaged over every PRE segment.
    "tatsuno_2006": {
        "measure": "explained_variance",
        "bin_size": 0.25,
        "pairs": "different_tetrodes",
        "control_windows": ReactivationWindows(
            length=900.0, anchor="end", clock="wall"
        ),
        "match_windows": ReactivationWindows(length=900.0, clock="wall"),
    },
    # Lansink et al. (2009): 50 ms, cross-region pairs, 20-min blocks of
    # concatenated rest, 10,000 pair-triplet resamples, 2.5th percentile.
    "lansink_2009": {
        "measure": "explained_variance",
        "bin_size": 0.05,
        "pairs": "different_regions",
        "control_windows": ReactivationWindows(length=1200.0, anchor="end", count=1),
        "match_windows": ReactivationWindows(length=1200.0),
        "statistic": "ev_minus_rev",
        "test": "pair_bootstrap",
        "n_shuffles": 10_000,
        "alpha": 0.05,
    },
    # Peyrache et al. (2010): 100 ms, PCA, POST - PRE mean R, PRE 99th
    # percentile, 1000 per-bin cell-identity shuffles at the 99th.
    "peyrache_2010": {
        "measure": "assembly",
        "bin_size": 0.1,
        "detection": "pca",
        "statistic": "mean_activation_difference",
        "test": "cell_identity_shuffle",
        "percentile": 99.0,
        "n_shuffles": 1000,
    },
    # van de Ven et al. (2016): 25 ms, ICA, Gaussian sigma = 25 ms / sqrt(12),
    # activations above 5, rest after - rest before.
    "van_de_ven_2016": {
        "measure": "assembly",
        "bin_size": 0.025,
        "detection": "ica",
        "smoothing_bandwidth": 0.025 / np.sqrt(12.0),
        "activation_threshold": 5.0,
        "statistic": "mean_activation_difference",
    },
    # Giri et al. (2019): 250 ms, different shanks, 15-min windows sliding
    # in 5-min steps, EV averaged over PRE windows. The rate control (100
    # rate-equalizing subsamples) checked a pooled whole-period comparison,
    # not this main analysis, so n_rate_subsamples stays None.
    "giri_2019": {
        "measure": "explained_variance",
        "bin_size": 0.25,
        "pairs": "different_shanks",
        "control_windows": ReactivationWindows(length=900.0, step=300.0, clock="wall"),
        "match_windows": ReactivationWindows(length=900.0, step=300.0, clock="wall"),
    },
    # Giri et al. (2024): as 2019, with pairs chosen by waveform similarity
    # (pair_mask) instead of shanks.
    "giri_2024": {
        "measure": "explained_variance",
        "bin_size": 0.25,
        "control_windows": ReactivationWindows(length=900.0, step=300.0, clock="wall"),
        "match_windows": ReactivationWindows(length=900.0, step=300.0, clock="wall"),
    },
}

_TEXT_FIELDS = ("name", "reference", "notes")


def _fields(method: ReactivationMethod) -> dict[str, Any]:
    return {
        f.name: getattr(method, f.name)
        for f in dataclasses.fields(method)
        if f.name not in _TEXT_FIELDS
    }


def test_every_published_recipe_has_a_spec() -> None:
    assert sorted(method.name for method in PUBLISHED_REACTIVATION_METHODS) == sorted(
        SPEC
    )


@pytest.mark.parametrize(
    "method", PUBLISHED_REACTIVATION_METHODS, ids=lambda method: method.name
)
def test_recipe_fields_match_the_papers(method: ReactivationMethod) -> None:
    expected = {**DEFAULTS, **SPEC[method.name]}
    actual = _fields(method)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name] == value, f"{method.name}.{name}"
    assert method.notes, method.name
    assert method.reference


def test_class_defaults() -> None:
    defaults = {
        f.name: f.default
        for f in dataclasses.fields(ReactivationMethod)
        if f.default is not dataclasses.MISSING and f.name not in _TEXT_FIELDS
    }
    assert defaults == DEFAULTS


def test_recipes_are_ordered_by_year() -> None:
    years = [int(method.name[-4:]) for method in PUBLISHED_REACTIVATION_METHODS]
    assert years == sorted(years)
