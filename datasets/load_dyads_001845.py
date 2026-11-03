"""Jadhav lab rat dyads cooperating on paired W-mazes, pose keypoints, behavior only (DANDI 001845)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _record import Range, Recording

DATASET = {
    "key": "dyads_001845",
    "title": "Shukla, Jadhav et al., rat FXM110 (with FXM101), cooperative W-maze, pose keypoints",
    "source": "https://dandiarchive.org/dandiset/001845/0.260526.1807",
    "license": "CC0-1.0",
    "download_size": "1.4 MB",
}

# Pre-registered from the dataset catalog (pose keypoints, 28-37 % NaN,
# behavior only) and the file's metadata (DeepLabCut PoseEstimation of six
# keypoints in pixels; processed positions in centimeters with NaN where
# confidence was low; video at about 30 Hz) before the first run of this
# loader.
EXPECTATIONS: dict[str, Any] = {
    "behavior_only": lambda rec: rec.spike_times is None,
    "n_dims": 2,
    "sampling_rate_hz": Range(29.0, 31.0),
    "position_nan_fraction": Range(0.28, 0.40),
    "centimetres": lambda rec: rec.position_units == "cm",
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
