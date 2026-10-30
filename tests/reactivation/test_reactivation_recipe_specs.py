"""Every published reactivation recipe, field by field, against the papers.

Each value below was checked against the recipe's ``notes`` and the paper's
text. A change to a recipe must change this table too, so a bin size, a
window or a test cannot drift silently.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import numpy as np
import pytest

from neurospatial.reactivation import (
    PUBLISHED_REACTIVATION_METHODS,
    EventResponse,
    ReactivationMethod,
    ReactivationWindows,
)

DEFAULTS: dict[str, Any] = {
    "pairs": "all",
    "control_windows": ReactivationWindows(),
    "match_windows": ReactivationWindows(),
    "template_bins": "fixed",
    "rest_bins": "fixed",
    "min_events": None,
    "detection": None,
    "member_rule": "abs_z",
    "member_threshold": 2.0,
    "detection_units": "all",
    "smoothing_bandwidth": None,
    "activation_step": None,
    "zscore_over": "period",
    "activation_threshold": None,
    "event_response": None,
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
    # Oliva et al. (2020): Kudrimoti's EV over all pairs of interest, 100 ms
    # task bins, one bin per SWR of pre- and post-sleep (all SWRs pooled),
    # EV - REV across mice.
    "oliva_2020_ev": {
        "measure": "explained_variance",
        "bin_size": 0.1,
        "rest_bins": "events",
        "statistic": "ev_minus_rev",
    },
    # Oliva et al. (2020): 20 ms bins, Gaussian SD 10 ms, threshold 5, mean
    # expression post - pre.
    "oliva_2020_assembly": {
        "measure": "assembly",
        "bin_size": 0.02,
        "detection": "ica",
        "smoothing_bandwidth": 0.01,
        "activation_threshold": 5.0,
        "statistic": "mean_activation_difference",
    },
    # El-Gaby et al. (2021): 25 ms ICA patterns, signed members above mean +
    # 2 SD, member-pair r with one bin per SWR, sessions with >= 200 SWRs.
    "el_gaby_2021": {
        "measure": "member_cofiring",
        "bin_size": 0.025,
        "rest_bins": "events",
        "min_events": 200,
        "detection": "ica",
        "member_rule": "positive_z",
        "member_threshold": 2.0,
        "statistic": "mean_member_correlation_difference",
    },
    # Harvey et al. (2023): 25 ms ICA, Otsu members, sigma = 25 / sqrt(12) ms
    # on 2 ms steps, whole-session z-score, first NREM hour, SWR-triggered
    # peak 0-50 ms, activations above 5.
    "harvey_2023_assembly": {
        "measure": "assembly",
        "bin_size": 0.025,
        "control_windows": ReactivationWindows(
            length=3600.0, count=1, clock="wall", allow_shorter=True
        ),
        "match_windows": ReactivationWindows(
            length=3600.0, count=1, clock="wall", allow_shorter=True
        ),
        "detection": "ica",
        "member_rule": "otsu",
        "member_threshold": None,
        "smoothing_bandwidth": 0.025 / np.sqrt(12.0),
        "activation_step": 0.002,
        "zscore_over": "all_periods",
        "activation_threshold": 5.0,
        "event_response": EventResponse(
            window=(-0.5, 0.5), response_window=(0.0, 0.05), reduce="peak"
        ),
        "statistic": "event_response_difference",
    },
    # Liu et al. (2023): EV with one bin per theta cycle (task) and per SWR
    # (sleep).
    "liu_2023_ev": {
        "measure": "explained_variance",
        "bin_size": None,
        "template_bins": "events",
        "rest_bins": "events",
    },
    # Liu et al. (2023): task correlations against the residual of post on
    # pre, the semipartial correlation.
    "liu_2023_pairwise": {
        "measure": "explained_variance",
        "bin_size": None,
        "template_bins": "events",
        "rest_bins": "events",
        "statistic": "semipartial_correlation",
    },
    # Liu et al. (2023): 25 ms ICA, members above mean + 2 SD, last/first
    # hour of NREM, joint z-score, mean R 0-100 ms after SWR onset.
    "liu_2023_assembly": {
        "measure": "assembly",
        "bin_size": 0.025,
        "control_windows": ReactivationWindows(
            length=3600.0, anchor="end", count=1, allow_shorter=True
        ),
        "match_windows": ReactivationWindows(
            length=3600.0, count=1, allow_shorter=True
        ),
        "detection": "ica",
        "member_rule": "positive_z",
        "member_threshold": 2.0,
        "zscore_over": "control_and_match",
        "activation_threshold": 5.0,
        "event_response": EventResponse(window=(-0.5, 0.5), response_window=(0.0, 0.1)),
        "statistic": "event_response_difference",
    },
    # Chang et al. (2025): 50 ms ICA in 10 ms steps, members above mean +
    # 3 SD, (POST - PRE) / PRE of the 0-100 ms SWR response, signed-rank
    # against -500 to -100 ms.
    "chang_2025_assembly": {
        "measure": "assembly",
        "bin_size": 0.05,
        "detection": "ica",
        "member_rule": "positive_z",
        "member_threshold": 3.0,
        "activation_step": 0.01,
        "zscore_over": "control_and_match",
        "event_response": EventResponse(
            window=(-0.5, 0.5),
            response_window=(0.0, 0.1),
            baseline_window=(-0.5, -0.1),
        ),
        "statistic": "event_response_ratio",
        "test": "event_modulation",
        "alpha": 0.05,
    },
    # Robinson et al. (2026): 100 ms task bins, one bin per SWR, the
    # correlation of residuals on baseline sleep = the partial correlation.
    "robinson_2026_pairwise": {
        "measure": "explained_variance",
        "bin_size": 0.1,
        "rest_bins": "events",
        "statistic": "partial_correlation",
    },
    # Robinson et al. (2026): 50 ms ICA per region and across regions,
    # members above 1/sqrt(N), mean R 0-100 ms after SWR onset, per-SWR
    # z above 1.96.
    "robinson_2026_assembly": {
        "measure": "assembly",
        "bin_size": 0.05,
        "detection": "ica",
        "member_rule": "inverse_sqrt_n",
        "member_threshold": None,
        "detection_units": "each_region_and_cross_region",
        "activation_step": 0.01,
        "zscore_over": "control_and_match",
        "event_response": EventResponse(significant_z=1.96),
        "statistic": "event_response_difference",
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
    years = [
        int(re.search(r"_(\d{4})(_|$)", method.name).group(1))  # type: ignore[union-attr]
        for method in PUBLISHED_REACTIVATION_METHODS
    ]
    assert years == sorted(years)
