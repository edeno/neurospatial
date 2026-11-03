from __future__ import annotations

from dataclasses import replace

import numpy as np
from _record import validate_recording
from operations import egocentric_boundary

from conftest import EBC_DIRECTION, EBC_DISTANCE_CM


def test_egocentric_boundary_operation_finds_simulated_cell(
    egocentric_boundary_recording,
):
    rec = egocentric_boundary_recording
    validate_recording(rec)
    cache: dict = {}
    outcome = egocentric_boundary(rec, cache)
    assert outcome.status == "ok"
    rates = cache["egocentric_boundary"]
    row = rates.summary_table().iloc[0]
    ring = (rates.distance_range[1] - rates.distance_range[0]) / rates.n_distance_bins
    sector = 2 * np.pi / rates.n_direction_bins
    assert abs(row["preferred_distance"] - EBC_DISTANCE_CM) <= ring
    error = np.angle(np.exp(1j * (row["preferred_direction"] - EBC_DIRECTION)))
    assert abs(error) <= sector
    assert outcome.summary["frac_peak_on_left"] == 1.0


def test_egocentric_boundary_mirrored_headings_move_the_peak_right(
    egocentric_boundary_recording,
):
    # Headings reflected (h -> -h) put the same wall on the other side.
    rec = egocentric_boundary_recording
    mirrored = replace(rec, headings=-rec.headings, positions=rec.positions * [1, -1])
    outcome = egocentric_boundary(mirrored, {})
    assert outcome.summary["frac_peak_on_left"] == 0.0


def test_egocentric_boundary_needs_two_dimensions(egocentric_boundary_recording):
    rec = egocentric_boundary_recording
    flat = replace(rec, positions=rec.positions[:, :1])
    assert egocentric_boundary(flat, {}).status == "n/a"
