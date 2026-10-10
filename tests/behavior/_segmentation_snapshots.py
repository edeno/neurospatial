"""Shared real-data calls and structural serialization for trajectory goldens."""

import dataclasses

import numpy as np
import scipy.sparse


def capture_segmentation_outputs(pause, laps):
    """Exercise every public segmentation/window/sequence entry point."""
    from neurospatial.behavior import (
        Trial,
        detect_goal_directed_runs,
        detect_laps,
        detect_region_crossings,
        detect_runs_between_regions,
        running_direction_labels,
        segment_by_velocity,
        segment_trials,
    )
    from neurospatial.behavior.decisions import (
        compute_decision_analysis,
        compute_pre_decision_metrics,
        detect_boundary_crossings,
        extract_pre_decision_window,
    )
    from neurospatial.behavior.vte import compute_vte_session, compute_vte_trial

    p = pause
    lap_recording = laps
    labels = np.where(p.env.bin_centers[:, 0] < 50, 0, 1)
    reference = lap_recording.position_bins[100:501]
    trials = [Trial(0.0, p.times[-1], "source", "target", True)]
    return {
        "detect_region_crossings": detect_region_crossings(
            p.position_bins, p.times, p.env, region_name="target"
        ),
        "detect_runs_between_regions": detect_runs_between_regions(
            p.position_bins,
            p.times,
            p.env,
            source="source",
            target="target",
            max_duration=2000,
        ),
        "segment_by_velocity": segment_by_velocity(
            p.times, p.positions, min_speed=5.0, min_duration=0.1
        ),
        "detect_laps_region": detect_laps(
            lap_recording.position_bins,
            lap_recording.times,
            lap_recording.env,
            method="region",
            start_region="start",
        ),
        "detect_laps_auto": detect_laps(
            lap_recording.position_bins,
            lap_recording.times,
            lap_recording.env,
            method="auto",
        ),
        "detect_laps_reference": detect_laps(
            lap_recording.position_bins,
            lap_recording.times,
            lap_recording.env,
            method="reference",
            reference_lap=reference,
        ),
        "segment_trials": segment_trials(
            p.position_bins,
            p.times,
            p.env,
            start_region="source",
            end_regions=["target"],
            max_duration=2000,
        ),
        "detect_goal_directed_runs": detect_goal_directed_runs(
            p.position_bins, p.times, p.env, goal_region="target"
        ),
        "running_direction_labels": running_direction_labels(
            p.position_bins,
            p.times,
            p.env,
            start_region="source",
            end_regions="target",
            max_duration=2000,
        ),
        "detect_boundary_crossings": detect_boundary_crossings(
            p.position_bins, labels, p.times
        ),
        "extract_pre_decision_window": extract_pre_decision_window(
            p.times, p.positions, entry_time=100.5, window_duration=2.0
        ),
        "compute_pre_decision_metrics": compute_pre_decision_metrics(
            p.times, p.positions, entry_time=100.5, window_duration=2.0
        ),
        "compute_decision_analysis": compute_decision_analysis(
            p.env,
            p.times,
            p.positions,
            decision_region="target",
            goal_regions=["source", "target"],
            pre_window=2.0,
        ),
        "compute_vte_trial": compute_vte_trial(
            p.times, p.positions, entry_time=100.5, window_duration=2.0
        ),
        "compute_vte_session": compute_vte_session(
            p.env,
            p.times,
            p.positions,
            decision_region="target",
            trials=trials,
            window_duration=2.0,
        ),
        "bin_sequence": p.env.bin_sequence(p.times, p.positions),
        "bin_sequence_per_sample": p.env.bin_sequence(
            p.times, p.positions, dedup=False
        ),
        "bin_sequence_with_runs": p.env.bin_sequence_with_runs(p.times, p.positions),
        "transitions": p.env.transitions(
            times=p.times, positions=p.positions, allow_teleports=True, normalize=False
        ),
        "transitions_normalized": p.env.transitions(
            times=p.times, positions=p.positions, allow_teleports=True
        ),
    }


def structural_snapshot(value):
    """Keep array types/shapes and sparse indices without pickled objects."""
    if dataclasses.is_dataclass(value):
        return {
            "dataclass": type(value).__name__,
            "fields": {
                f.name: structural_snapshot(getattr(value, f.name))
                for f in dataclasses.fields(value)
            },
        }
    if scipy.sparse.issparse(value):
        return {
            "sparse": value.format,
            "shape": list(value.shape),
            "indptr": structural_snapshot(value.indptr),
            "indices": structural_snapshot(value.indices),
            "data": structural_snapshot(value.data),
        }
    if isinstance(value, np.ndarray):
        return {
            "array": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"tuple": [structural_snapshot(item) for item in value]}
    if isinstance(value, list):
        return [structural_snapshot(item) for item in value]
    if isinstance(value, dict):
        return {key: structural_snapshot(item) for key, item in value.items()}
    return value
