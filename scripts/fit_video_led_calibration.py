#!/usr/bin/env python
"""Fit a session video's pixel-to-cm calibration from the LED it shows.

For a dataset whose file states no pixel size and stores no pixel tracking
(DANDI 000568), the tracked LED is the one landmark the video shows in
every frame. This script loads the session, fetches its video, detects the
LED in every ``--step``-th frame (the brightest point of the frame's summed
RGB after a Gaussian blur of ``BLUR_PX``), and fits a robust affine from
those pixels to the file's positions of the same samples
(:func:`neurospatial.datasets._video.fit_pixel_affine`: a Huber fit with a
``HUBER_CM`` cut, then least squares over the detections within
``OUTLIER_CM`` of it). Frame k is paired with position sample k: the
session's video and position series must hold the same frames (its frame
times are the position times). The same fit is repeated with frame k
paired with sample k + lag for each ``--lags``, to show how closely the
pairing holds.

The fit, the frame size, the method, the frames used, the residuals
(median and 90th percentile, cm and px), the outlier share and the lag scan
are written into the session's video entry in ``registry/<dataset>.json``
as ``led_fit``, keyed to the video's SHA-256, so loading the session reads
them without the video. The loader accepts the fit only when its median
residual is at most one bin.

Usage
-----
    uv run --extra datasets --extra video python -I \
        scripts/fit_video_led_calibration.py valero_2022 \
        sub-fCamk2_ses-fCamk2-201014-sess3 \
        --lags -677 -30 -3 -1 0 1 3 30 677

(677 frames is the 22.6 s by which DANDI 000568's video clock leads.)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "src" / "neurospatial" / "datasets" / "registry"
#: Gaussian blur of the summed RGB frame before its brightest point is taken,
#: px: wider than a pixel's noise, narrower than the LED's glow.
BLUR_PX = 3.0
#: Every this many frames is used by default (one a second at 30 Hz).
STEP = 30
#: The Huber cut, cm: about one bin, the residual the loader accepts.
HUBER_CM = 2.0
#: Detections farther than this from the Huber fit are outliers, cm: twice
#: the bin, well beyond the head's sway about the LED.
OUTLIER_CM = 5.0
#: Pairings scanned by default, frames.
LAGS = (-30, -3, -1, 0, 1, 3, 30)
METHOD = (
    "the LED as the brightest point of each frame's summed RGB after a "
    f"Gaussian blur of {BLUR_PX:g} px (OpenCV), against the file's position of "
    f"the same sample; a Huber affine from pixels to cm (cut {HUBER_CM:g} cm, "
    "iteratively reweighted least squares), then least squares over the "
    f"detections within {OUTLIER_CM:g} cm of it"
)


def detect_led(frame: NDArray[np.uint8]) -> tuple[float, float]:
    """The LED's pixel ``(column, row)`` in one frame, shape (height, width, 3)."""
    import cv2

    total = frame.astype(np.float64).sum(axis=2)
    blurred = cv2.GaussianBlur(total, (0, 0), BLUR_PX)
    row, column = np.unravel_index(int(np.argmax(blurred)), blurred.shape)
    return float(column), float(row)


def fit(rec: Any, step: int, lags: tuple[int, ...] = LAGS) -> dict[str, Any]:
    """The ``led_fit`` registry entry for ``rec``'s video."""
    from neurospatial._video_io import _VideoReader
    from neurospatial.datasets._video import fit_pixel_affine

    video = rec.video
    if video is None or not np.array_equal(video.frame_times, rec.times):
        msg = (
            f"{rec.session}: the video's frame times are not the position times, "
            "so frame k is not position sample k"
        )
        raise ValueError(msg)
    frames = np.arange(0, len(rec.times), step)
    with _VideoReader(video.fetch(), cache_size=1) as reader:
        if reader.n_frames != len(rec.times):
            msg = f"the video decodes {reader.n_frames} frames, not {len(rec.times)}"
            raise ValueError(msg)
        pixels = np.array([detect_led(np.asarray(reader[int(k)])) for k in frames])
        size = reader.original_size_px
    robust = {"huber_cm": HUBER_CM, "outlier_cm": OUTLIER_CM}
    transform, in_cm, in_px = fit_pixel_affine(pixels, rec.positions[frames], **robust)
    scan = {}
    for lag in lags:
        paired = frames + lag
        inside = (paired >= 0) & (paired < len(rec.times))
        _, lagged, _ = fit_pixel_affine(
            pixels[inside], rec.positions[paired[inside]], **robust
        )
        scan[str(lag)] = round(float(np.median(lagged)), 4)
    script = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "video_sha256": video.sha256,
        "method": METHOD,
        "matrix": np.asarray(transform.A).round(10).tolist(),
        "frame_size_px": list(size),
        "frames": f"every {step}th frame from 0 ({len(frames)} frames)",
        "n_frames": int(in_cm.size),
        "residual_cm": {
            "median": round(float(np.median(in_cm)), 4),
            "p90": round(float(np.percentile(in_cm, 90)), 4),
        },
        "residual_px": {
            "median": round(float(np.median(in_px)), 4),
            "p90": round(float(np.percentile(in_px, 90)), 4),
        },
        "outliers": {
            "cut_cm": OUTLIER_CM,
            "share": round(float(np.mean(in_cm > OUTLIER_CM)), 4),
        },
        "lag_scan_median_cm": scan,
        "fitted_by": (
            f"scripts/fit_video_led_calibration.py with SHA-256 {script[:16]}, "
            f"{datetime.date.today().isoformat()}"
        ),
    }


def main() -> None:
    """Fit one session's video and write the fit into its registry."""
    from neurospatial.datasets import load
    from neurospatial.datasets._catalog import sessions

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset")
    parser.add_argument("session")
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--lags", type=int, nargs="+", default=list(LAGS))
    args = parser.parse_args()
    rec = load(args.dataset, args.session)
    entry = next(e for e in sessions(args.dataset) if e.session == args.session)
    assert entry.asset_path is not None
    asset_path = str(Path(entry.asset_path).parent / rec.video.external_file)
    led_fit = fit(rec, args.step, tuple(args.lags))
    path = REGISTRY / f"{args.dataset}.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    (row,) = [r for r in registry["sessions"] if r["session"] == args.session]
    (video,) = [v for v in row["videos"] if v["asset_path"] == asset_path]
    video["led_fit"] = led_fit
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(led_fit, indent=2))


if __name__ == "__main__":
    main()
