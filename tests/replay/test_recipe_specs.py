"""Every published replay recipe, field by field, against the papers' values.

Each value below was checked against the recipe's ``notes`` and reference
text. A change to a recipe must change this table too, so a threshold, a
shuffle count or a score cannot drift silently.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from neurospatial.replay import (
    ArmReplayMethod,
    PairwiseOrderMethod,
    RankOrderMethod,
    ReplayMethod,
    TemplateMatchMethod,
)
from neurospatial.replay.methods import PUBLISHED_METHODS

# Class defaults: a recipe's fields are these, updated by its entry in SPEC.
DEFAULTS: dict[type, dict[str, Any]] = {
    ReplayMethod: {
        "score_options": {},
        "nulls": (),
        "null_options": {},
        "n_shuffles": 1000,
        "alpha": 0.05,
        "alternative": "greater",
        "use_absolute": None,
        "tested_quantities": {},
        "empty_time_bins": "keep",
        "criteria": (),
        "segmentation": None,
        "units": None,
        "template_choice": None,
        "exclude_end_distance": 0.0,
        "decoding": {},
    },
    RankOrderMethod: {
        "spike_statistic": "median",
        "field_order": "peak",
        "min_peak_rate": None,
        "min_active_cells": 5,
        "min_active_fraction": None,
        "test": "cell_order",
        "alternative": "two-sided",
        "n_shuffles": 1000,
        "alpha": 0.05,
        "min_epoch_events": None,
        "template_choice": None,
    },
    # Gupta et al. (2010): fields above 5% of the population's highest rate,
    # merged across one-pixel gaps; spike pairs less than 80 ms apart; left
    # and right laps forward and backward; 300 spike-time and 300
    # cell-identity shuffles, beat 95% of both; at least 3 active cells.
    PairwiseOrderMethod: {
        "field_threshold": 0.05,
        "merge_gap_bins": 1,
        "field_center": "rate_weighted",
        "max_time_lag": 0.08,
        "directions": ("forward", "backward"),
        "nulls": ("spike_time_permutation", "cell_identity"),
        "n_shuffles": 300,
        "alpha": 0.05,
        "min_active_cells": 3,
    },
    # Wu & Foster (2014): peak above 5x chance, 10 ms smoothing, 50 ms,
    # more than 50% of the arm, |weighted correlation| > 0.5.
    ArmReplayMethod: {
        "peak_threshold": 5.0,
        "smoothing_sigma": 0.01,
        "min_duration": 0.05,
        "min_coverage": 0.5,
        "min_abs_correlation": 0.5,
        "joint_min_abs_correlation": 0.5,
        "subregions": "largest_peak",
        "merge_max_gap": None,
        "coverage": "represented",
        "joint": "only_pair",
        "joint_same_sign": True,
        "arm_fallback": False,
        "decoding": {},
    },
    TemplateMatchMethod: {
        "field_order": "peak",
        "min_peak_rate": None,
        "min_template_cells": None,
        "max_isi": None,
        "max_gap": None,
        "match_order": "interruptions_first",
        "trial_probability": None,
        "exact_max_letters": 10,
        "n_samples": 100_000,
        "bandwidth": None,
        "min_active_cells": 2,
        "alpha": 0.05,
        "p_comparison": "<",
        "template_choice": None,
        "n_shuffles": 1000,
    },
}


def _segmentation(
    options: dict[str, Any],
    relative_to: dict[str, str] | None = None,
    choose: str = "longest",
    min_active_cells_per_time_bin: int | None = None,
    rows: str = "segment",
) -> dict[str, Any]:
    return {
        "options": options,
        "relative_to": relative_to or {},
        "choose": choose,
        "min_active_cells_per_time_bin": min_active_cells_per_time_bin,
        "rows": rows,
    }


# Criteria are (quantity, comparison, threshold, relative_to, absolute).
SPEC: dict[str, tuple[type, dict[str, Any]]] = {
    # Lee & Wilson (2002): 50 ms bursts, 100 ms word gaps, p <= 1/24; the
    # number of template shuffles (1000) is not stated in the paper.
    "lee_wilson_2002": (
        TemplateMatchMethod,
        {
            "statistic": "best_match",
            "max_isi": 0.05,
            "max_gap": 0.1,
            "trial_probability": 1 / 24,
            "alpha": 1 / 24,
            "p_comparison": "<=",
        },
    ),
    # Ji & Wilson (2007): sigma 180 ms (CA1), >= 4 active cells, templates of
    # >= 5 cells, p < 0.05, the higher matching index wins, 1000 shuffles.
    "ji_wilson_2007": (
        TemplateMatchMethod,
        {
            "statistic": "matching_index",
            "min_template_cells": 5,
            "bandwidth": 0.18,
            "min_active_cells": 4,
            "template_choice": "largest_score",
        },
    ),
    "foster_wilson_2006": (
        RankOrderMethod,
        {
            "spike_statistic": "all_spikes",
            "min_peak_rate": 5.0,
            "min_active_cells": 3,
            "min_active_fraction": 1 / 3,
            "test": "t_distribution",
        },
    ),
    "diba_buzsaki_2007": (
        RankOrderMethod,
        {
            "spike_statistic": "first",
            "min_peak_rate": 5.0,
            "min_active_fraction": 0.3,
            "n_shuffles": 500,
            "alpha": 0.1,
            "template_choice": "most_active",
        },
    ),
    "davidson_2009": (
        ReplayMethod,
        {
            "score": "detect_trajectory_radon",
            "nulls": ("column_cycle", "unit_identity", "pseudo_event"),
            "n_shuffles": 1500,
            "alpha": 0.01,
            "decoding": {"time_bin": 0.02},
        },
    ),
    "karlsson_frank_2009": (
        ReplayMethod,
        {
            "score": "fit_linear_trajectory",
            "score_options": {"method": "sample", "n_samples_per_time_bin": 10_000},
            "nulls": ("time_bin_permutation",),
            "n_shuffles": 10_000,
            "empty_time_bins": "drop",
            "criteria": (("n_active_cells", ">=", 5, None, False),),
            "decoding": {"time_bin": 0.015, "position_bin": 2.0, "prior": "uniform"},
        },
    ),
    "pfeiffer_foster_2013": (
        ReplayMethod,
        {
            "score": None,
            "criteria": (("segment_start_to_end_distance", ">=", 40.0, None, False),),
            "segmentation": _segmentation(
                {"max_jump": 20.0, "method": "map", "min_time_bins": 11}
            ),
            "units": "cm",
            "decoding": {"time_bin": 0.02, "time_step": 0.005},
        },
    ),
    "gupta_2010": (PairwiseOrderMethod, {}),
    "wu_foster_2014": (
        ArmReplayMethod,
        {"decoding": {"time_bin": 0.01, "position_bin": 1.85}},
    ),
    "pfeiffer_foster_2015": (
        ReplayMethod,
        {
            "score": None,
            "criteria": (("segment_start_to_end_distance", ">=", 80.0, None, False),),
            "segmentation": _segmentation(
                {"max_jump": 50.0, "method": "expected", "min_time_bins": 11}
            ),
            "units": "cm",
            "decoding": {"time_bin": 0.02, "time_step": 0.005},
        },
    ),
    "silva_2015": (
        ReplayMethod,
        {
            "score": "weighted_correlation",
            "criteria": (
                ("score", ">", 0.6, None, True),
                ("max_map_jump", "<", 0.4, "track_length", False),
            ),
            "decoding": {"time_bin": 0.02},
        },
    ),
    "ambrose_2016": (
        ReplayMethod,
        {
            "score": "weighted_correlation",
            "criteria": (("score", ">", 0.6, None, True),),
            "decoding": {
                "time_bin": 0.02,
                "time_step": 0.01,
                "position_bin": 1.8,
                "prior": "uniform",
            },
        },
    ),
    "grosmark_2016": (
        ReplayMethod,
        {
            "score": "weighted_correlation",
            "nulls": ("place_field_circular",),
            "use_absolute": True,
            "criteria": (
                ("n_active_cells", ">=", 5, None, False),
                ("active_fraction", ">=", 0.1, None, False),
            ),
            "decoding": {"time_bin": 0.02, "position_bin": 2.0, "prior": "uniform"},
        },
    ),
    "wu_2017": (
        ReplayMethod,
        {
            "score": "fit_linear_trajectory",
            "score_options": {"method": "map"},
            "nulls": ("time_bin_permutation",),
            "empty_time_bins": "drop",
            "criteria": (("n_active_cells", ">=", 4, None, False),),
            "decoding": {"time_bin": 0.02, "time_step": 0.01},
        },
    ),
    "olafsdottir_2017": (
        ReplayMethod,
        {
            "score": "detect_trajectory_radon",
            "score_options": {"band": 30.0, "max_speed": 5000.0, "min_speed": 200.0},
            "nulls": ("place_field_circular",),
            "n_shuffles": 100,
            "alpha": 0.025,
            "empty_time_bins": "drop",
            "criteria": (
                ("n_active_cells", ">", 5, None, False),
                ("active_fraction", ">=", 0.15, None, False),
            ),
            "units": "cm",
            "decoding": {"time_bin": 0.01, "position_bin": 2.0},
        },
    ),
    "carey_2019": (
        ReplayMethod,
        {
            "score": None,
            "criteria": (("segment_duration", ">=", 0.05, None, False),),
            "segmentation": _segmentation(
                {"max_jump": 40.0, "method": "map", "min_time_bins": 10},
                choose="any",
                min_active_cells_per_time_bin=4,
            ),
            "units": "cm",
            "decoding": {"time_bin": 0.025, "time_step": 0.005, "position_bin": 3.0},
        },
    ),
    "chenani_2019": (
        RankOrderMethod,
        {
            "spike_statistic": "mean",
            "field_order": "center",
            "test": "pooled_ssi",
            "alternative": "greater",
            "n_shuffles": 100,
            "min_epoch_events": 20,
        },
    ),
    "michon_2019": (
        ReplayMethod,
        {
            "score": "fit_isotonic_trajectory",
            "score_options": {"method": "map"},
            "nulls": ("pseudo_event",),
            "n_shuffles": 500,
            "alpha": 0.01,
            "decoding": {"time_bin": 0.01},
        },
    ),
    "igata_2021": (
        ReplayMethod,
        {
            "score": None,
            "nulls": ("place_field_circular",),
            "criteria": (("segment_path_length", ">", 20.0, None, False),),
            "segmentation": _segmentation(
                {
                    "max_jump": 40.0,
                    "method": "map",
                    "max_gap_bins": 1,
                    "min_peak_probability": 5.0,
                    "min_time_bins": 5,
                },
                relative_to={"min_peak_probability": "chance"},
            ),
            "units": "cm",
            "decoding": {"time_bin": 0.02, "time_step": 0.01},
        },
    ),
    "berners_lee_2021": (
        ArmReplayMethod,
        {
            "peak_threshold": 4.0,
            "smoothing_sigma": 0.005,
            "min_duration": 0.055,
            "min_abs_correlation": 0.3,
            "joint_min_abs_correlation": 0.3,
            "subregions": "runs",
            "merge_max_gap": 0.05,
            "coverage": "mean_range",
            "joint": "strongest_pair",
            "joint_same_sign": False,
            "arm_fallback": True,
            "decoding": {"time_bin": 0.02, "time_step": 0.005, "position_bin": 2.0},
        },
    ),
    "tirole_2022": (
        ReplayMethod,
        {
            "score": "weighted_correlation",
            "score_options": {"row_normalization": "none"},
            "nulls": ("spike_train_circular", "place_field_circular", "column_cycle"),
            "use_absolute": True,
            "empty_time_bins": "zero",
            "criteria": (("n_active_cells", ">=", 5, None, False),),
            "decoding": {"time_bin": 0.02, "position_bin": 10.0},
        },
    ),
    "berners_lee_2022": (
        ReplayMethod,
        {
            "score": "weighted_correlation",
            "criteria": (
                ("score", ">", 0.6, None, True),
                ("max_map_jump", "<", 0.4, "track_length", False),
            ),
            "template_choice": "sum",
            "decoding": {
                "time_bin": 0.02,
                "time_step": 0.005,
                "position_bin": 2.5,
                "prior": "uniform",
                "templates": "position x running direction, decoded jointly",
            },
        },
    ),
    "widloski_foster_2022": (
        ReplayMethod,
        {
            "score": None,
            "criteria": (
                ("segment_duration", ">", 0.1, None, False),
                ("segment_dispersion", ">", 12.0, None, False),
            ),
            "segmentation": _segmentation(
                {
                    "max_jump": 20.0,
                    "method": "expected",
                    "jump_rule": "drop_earlier",
                    "max_spread": 10.0,
                    "merge_max_time_gap": 0.05,
                    "merge_max_distance": 20.0,
                },
                choose="any",
            ),
            "units": "cm",
            "decoding": {"time_bin": 0.08, "time_step": 0.005, "position_bin": 2.0},
        },
    ),
    "mallory_2025": (
        ReplayMethod,
        {
            "score": "weighted_correlation",
            "score_options": {"row_normalization": "none"},
            "criteria": (
                ("segment_template_fraction", ">=", 0.5, None, False),
                ("template_fraction", ">", 0.665, None, False),
                ("score", ">", 0.6, None, True),
                ("max_map_jump", "<", 0.4, "track_length", False),
                ("map_range", ">", 0.2, "track_length", False),
                ("n_active_cells", ">=", 10, None, False),
            ),
            "segmentation": _segmentation(
                {
                    "max_jump": 0.4,
                    "method": "map",
                    "min_peak_probability": 5.0,
                    "merge_max_time_gap": 0.05,
                    "merge_max_distance": 0.4,
                    "min_duration": 0.03,
                },
                relative_to={
                    "max_jump": "track_length",
                    "merge_max_distance": "track_length",
                    "min_peak_probability": "chance",
                },
                rows="trajectory",
            ),
            "units": "cm",
            "template_choice": "most_posterior",
            "exclude_end_distance": 20.0,
            "decoding": {
                "time_bin": 0.02,
                "time_step": 0.005,
                "position_bin": 2.0,
                "templates": "the two running directions, decoded jointly",
            },
        },
    ),
    "widloski_foster_2025": (
        ReplayMethod,
        {
            "score": None,
            "nulls": ("unit_identity",),
            "n_shuffles": 100,
            "alpha": 0.05,
            "tested_quantities": {"mean_spread": "less", "mean_mean_jump": "less"},
            "criteria": (
                ("segment_duration", ">", 0.1, None, False),
                ("segment_dispersion", ">", 12.0, None, False),
            ),
            "segmentation": _segmentation(
                {
                    "max_jump": 20.0,
                    "method": "expected",
                    "jump_rule": "drop_earlier",
                    "max_spread": 10.0,
                    "merge_max_time_gap": 0.05,
                    "merge_max_distance": 20.0,
                },
                choose="any",
            ),
            "units": "cm",
            "decoding": {
                "time_bin": 0.08,
                "time_step": 0.005,
                "position_bin": 2.0,
                "rate_map_smoothing_sd": 8.0,
                "prior": "uniform",
            },
        },
    ),
    "wikenheiser_redish_2013": (
        ReplayMethod,
        {
            "score": "fit_circular_sequence",
            "nulls": ("time_bin_permutation",),
            "n_shuffles": 10_000,
            "empty_time_bins": "drop",
            "criteria": (
                ("n_active_cells", ">=", 3, None, False),
                ("n_spikes", ">=", 5, None, False),
            ),
            "decoding": {"time_bin": 0.01, "position_bin": 4.0, "prior": "uniform"},
        },
    ),
    "olafsdottir_2015": (
        ReplayMethod,
        {
            "score": "detect_trajectory_radon",
            "score_options": {"band": 20.0, "max_speed": 5000.0, "min_speed": 200.0},
            "nulls": ("unit_identity",),
            "criteria": (("n_active_cells", ">=", 7, None, False),),
            "units": "cm",
            "decoding": {
                "time_bin": 0.005,
                "position_bin": 1.0,
                "rate_map_smoothing_sd": 2.0,
                "prior": "uniform",
            },
        },
    ),
    "jadhav_2016": (
        ReplayMethod,
        {
            "score": "fit_linear_trajectory",
            "score_options": {"method": "sample", "n_samples_per_time_bin": 10_000},
            "nulls": ("time_bin_permutation",),
            "n_shuffles": 500,
            "empty_time_bins": "drop",
            "criteria": (("n_active_cells", ">=", 4, None, False),),
            "template_choice": "smallest_p",
            "decoding": {
                "time_bin": 0.015,
                "templates": "the four W-track trajectories",
            },
        },
    ),
    "olafsdottir_2016": (
        ReplayMethod,
        {
            "score": "detect_trajectory_radon",
            "score_options": {"band": 30.0, "max_speed": 5000.0, "min_speed": 200.0},
            "nulls": ("place_field_circular",),
            "n_shuffles": 100,
            "alpha": 0.2,
            "criteria": (("active_fraction", ">=", 0.15, None, False),),
            "units": "cm",
            "template_choice": "smallest_p",
            "decoding": {
                "time_bin": 0.01,
                "position_bin": 2.0,
                "templates": "inbound and outbound rate maps, decoded separately",
            },
        },
    ),
    "tang_2017": (
        ReplayMethod,
        {
            "score": "fit_linear_trajectory",
            "score_options": {"method": "sample", "n_samples_per_time_bin": 10_000},
            "nulls": ("time_bin_permutation",),
            "n_shuffles": 10_000,
            "empty_time_bins": "drop",
            "criteria": (("n_active_cells", ">=", 5, None, False),),
            "template_choice": "smallest_p",
            "decoding": {
                "time_bin": 0.01,
                "position_bin": 2.0,
                "prior": "uniform",
                "templates": "the four W-track trajectories",
            },
        },
    ),
    "yamamoto_tonegawa_2017": (
        ReplayMethod,
        {
            "score": "detect_trajectory_radon",
            "nulls": ("column_cycle",),
            "n_shuffles": 2000,
            "decoding": {"prior": "uniform"},
        },
    ),
    "farooq_2019": (
        ReplayMethod,
        {
            "score": "weighted_correlation",
            "nulls": ("time_bin_permutation",),
            "n_shuffles": 500,
            "alternative": "two-sided",
            "use_absolute": False,
            "criteria": (("n_active_cells", ">=", 5, None, False),),
            "decoding": {"time_bin": 0.02, "prior": "uniform"},
        },
    ),
    "muessig_2019": (
        ReplayMethod,
        {
            "score": "detect_trajectory_radon",
            "score_options": {
                "band": 12.5,
                "max_speed": 2500.0,
                "min_speed": 0.0,
                "wrap": True,
            },
            "nulls": ("unit_identity",),
            "n_shuffles": 500,
            "units": "cm",
            "template_choice": "max_score",
            "decoding": {
                "time_bin": 0.02,
                "time_step": 0.01,
                "position_bin": 2.5,
                "rate_map_smoothing_sd": 5.0,
                "templates": (
                    "clockwise and counterclockwise rate maps, decoded separately"
                ),
            },
        },
    ),
    "shin_2019": (
        ReplayMethod,
        {
            "score": "fit_linear_trajectory",
            "score_options": {"method": "sample", "n_samples_per_time_bin": 10_000},
            "nulls": ("time_bin_permutation",),
            "n_shuffles": 1500,
            "empty_time_bins": "drop",
            "criteria": (("n_active_cells", ">=", 5, None, False),),
            "template_choice": "smallest_p",
            "decoding": {
                "time_bin": 0.01,
                "position_bin": 2.0,
                "rate_map_smoothing_sd": 4.0,
                "prior": "uniform over position and trajectory",
                "templates": "the four W-track trajectories",
            },
        },
    ),
    "bush_2022": (
        ReplayMethod,
        {
            "score": "detect_trajectory_radon",
            "score_options": {"band": 15.0, "max_speed": 5000.0, "min_speed": 100.0},
            "nulls": ("unit_identity",),
            "null_options": {"unit_identity": {"active_only": True}},
            "n_shuffles": 100,
            "empty_time_bins": "drop",
            "criteria": (
                ("n_active_cells", ">=", 5, None, False),
                ("active_fraction", ">=", 0.15, None, False),
            ),
            "units": "cm",
            "decoding": {
                "time_bin": 0.01,
                "position_bin": 2.0,
                "rate_map_smoothing_sd": 10.0,
                "templates": "outbound and inbound rate maps",
            },
        },
    ),
    "harvey_2023": (
        ReplayMethod,
        {
            "score": "weighted_correlation",
            "nulls": ("column_cycle",),
            "use_absolute": True,
            "empty_time_bins": "zero",
            "criteria": (
                ("n_active_cells", ">=", 5, None, False),
                ("empty_fraction", "<=", 0.5, None, False),
            ),
            "decoding": {
                "time_bin": 0.02,
                "position_bin": 3.0,
                "rate_map_smoothing_sd": 3.0,
                "prior": "uniform",
            },
        },
    ),
    # Harvey et al. (2023), "alternative approach", and Soula et al. (2023):
    # posterior within a 21 cm bandwidth (half-width 10.5 cm) of the line
    # fit to the posterior; 1,000 column cycles; p < 0.05; >= 5 active
    # cells; > 50% inactive time bins discarded; 20 ms and 3 cm bins.
    "harvey_2023_line": (
        ReplayMethod,
        {
            "score": "regression_line_score",
            "score_options": {"band": 10.5},
            "nulls": ("column_cycle",),
            "empty_time_bins": "zero",
            "criteria": (
                ("n_active_cells", ">=", 5, None, False),
                ("empty_fraction", "<=", 0.5, None, False),
            ),
            "units": "cm",
            "decoding": {
                "time_bin": 0.02,
                "position_bin": 3.0,
                "rate_map_smoothing_sd": 3.0,
                "prior": "uniform",
            },
        },
    ),
    "soula_2023": (
        ReplayMethod,
        {
            "score": "regression_line_score",
            "score_options": {"band": 10.5},
            "nulls": ("column_cycle",),
            "empty_time_bins": "zero",
            "criteria": (
                ("n_active_cells", ">=", 5, None, False),
                ("empty_fraction", "<=", 0.5, None, False),
            ),
            "units": "cm",
            "decoding": {"time_bin": 0.02, "position_bin": 3.0, "prior": "uniform"},
        },
    ),
    # Chang et al. (2025): weighted correlation of each trajectory's share
    # of a joint decode in 10 ms bins; 1,000 column cycles; above the 97.5th
    # or below the 2.5th percentile; lowest p-value over the trajectories;
    # >= 5 place cells.
    "chang_2025": (
        ReplayMethod,
        {
            "score": "weighted_correlation",
            "score_options": {"row_normalization": "none"},
            "nulls": ("column_cycle",),
            "alternative": "two-sided",
            "use_absolute": False,
            "empty_time_bins": "zero",
            "criteria": (("n_active_cells", ">=", 5, None, False),),
            "template_choice": "smallest_p",
            "decoding": {
                "time_bin": 0.01,
                "prior": "uniform over position and trajectory",
                "templates": "left and right T-maze trajectories, decoded jointly",
                "rate_map_speed": 2.0,
            },
        },
    ),
    "maboudi_2018": (
        ReplayMethod,
        {
            "score": "detect_trajectory_radon",
            "score_options": {
                "band": 3.0,
                "off_track": "bin_median",
                "zero_rows": "line_median",
                "n_lines": 35000,
            },
            "nulls": ("column_cycle",),
            "n_shuffles": 5000,
            "alpha": 0.01,
            "empty_time_bins": "zero",
            "criteria": (
                ("n_active_cells", ">=", 4, None, False),
                ("n_time_bins", ">=", 4, None, False),
            ),
            "units": "cm",
            "decoding": {
                "time_bin": 0.02,
                "prior": "uniform",
                "rate_map_time_bin": 0.1,
            },
        },
    ),
    "liu_2023": (
        ReplayMethod,
        {
            "score": "detect_trajectory_radon",
            "score_options": {
                "band": 22.5,
                "max_speed": None,
                "line_ends": "on_track",
            },
            "nulls": ("column_cycle",),
            "n_shuffles": 500,
            "units": "cm",
            "decoding": {
                "time_bin": 0.02,
                "position_bin": 1.5,
                "templates": (
                    "OFF and ON running directions, decoded and tested separately"
                ),
            },
        },
    ),
}

_TEXT_FIELDS = ("name", "reference", "notes")


def _plain(name: str, value: Any) -> Any:
    """A field's value as plain Python, comparable with the table above."""
    if name == "score":
        return None if value is None else value.__name__
    if name == "criteria":
        return tuple(
            (c.quantity, c.comparison, c.threshold, c.relative_to, c.absolute)
            for c in value
        )
    if name == "segmentation" and value is not None:
        return {
            "options": dict(value.options),
            "relative_to": dict(value.relative_to),
            "choose": value.choose,
            "min_active_cells_per_time_bin": value.min_active_cells_per_time_bin,
            "rows": value.rows,
        }
    if name in ("score_options", "decoding", "tested_quantities"):
        return dict(value)
    if name == "null_options":
        return {null: dict(options) for null, options in value.items()}
    return value


def _fields(method: Any) -> dict[str, Any]:
    return {
        f.name: _plain(f.name, getattr(method, f.name))
        for f in dataclasses.fields(method)
        if f.name not in _TEXT_FIELDS
    }


def test_every_published_recipe_has_a_spec():
    assert sorted(method.name for method in PUBLISHED_METHODS) == sorted(SPEC)


@pytest.mark.parametrize("method", PUBLISHED_METHODS, ids=lambda m: m.name)
def test_recipe_fields_match_the_papers(method):
    cls, overrides = SPEC[method.name]
    assert type(method) is cls
    expected = {**DEFAULTS[cls], **overrides}
    actual = _fields(method)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name] == value, f"{method.name}.{name}"


@pytest.mark.parametrize("cls", list(DEFAULTS), ids=lambda c: c.__name__)
def test_class_defaults(cls):
    defaults = {}
    for f in dataclasses.fields(cls):
        if f.default_factory is not dataclasses.MISSING:
            defaults[f.name] = _plain(f.name, f.default_factory())
        elif f.default is not dataclasses.MISSING:
            defaults[f.name] = _plain(f.name, f.default)
    for text in _TEXT_FIELDS:
        defaults.pop(text, None)
    assert defaults == DEFAULTS[cls]
