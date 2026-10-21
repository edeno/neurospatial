"""Compare two prediction recipes and two polygon recipes on identical RGB data.

Run headlessly from the feature checkout::

    PYTHONPATH=src uv run --no-sync python examples/compare_annotation_styles.py \
        --output-dir /tmp/annotation-cpu
    PYTHONPATH=src uv run --no-sync python examples/compare_annotation_styles.py \
        --provider sam2 --video data/annotation_demo/j16_02_r1.mp4 \
        --inside 410 400 --outside 25 25 --output-dir /tmp/annotation-large

The four panels separate prediction changes from polygon conversion changes.
The existing prediction recipe processes each inside click, then each outside
click, with previous-logit feedback. The CVAT-style recipe requests one mask
from all clicks without feedback. Both use the SAME Meta SAM 2 model, weights,
RGB frame, and ordered prompts. CPU mode reuses one region-growing prediction
and therefore compares polygon conversion only. No ground truth is supplied;
areas, holes, vertices, and pairwise overlap describe differences, not accuracy.

The CVAT polygon recipe follows the pinned implementation at
https://github.com/cvat-ai/cvat/blob/172ea51a2ffc11adb08b1c9fb757282d142cd810/cvat-core/src/opencv/opencv-interface.ts
It uses dilation, the longest external contour, and the default accuracy 9.
This experimental comparison uses private helpers; it does not reproduce CVAT Online's
unknown deployed model/weights. It never downloads weights or opens a viewer.
The single-mask request follows
https://github.com/cvat-ai/cvat-models/blob/d2ebbc7f37d9c002d74fa8d392e9f3f8cb3aa419/interactor/sam2/func.py
while retaining the local Meta backend. The private adapter disables Meta's
dynamic multimask stability fallback for that request when the flag is exposed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from shapely.geometry import MultiPolygon, mapping

from neurospatial.annotation import (
    RegionGrowProvider,
    SegmentAnythingProvider,
    SegmentationPrompt,
    SegmentationProposal,
)
from neurospatial.annotation._cvat_polygon import (
    CVAT_UPSTREAM_REF,
    cvat_contour,
    cvat_tolerance,
    simplify_cvat_contour,
)
from neurospatial.annotation._mask_cleanup import remove_unprompted_fragments
from neurospatial.annotation._polygon_outline import (
    PolygonOutline,
    polygon_outline,
    simplify_polygon_outline,
)


def mask_metrics(mask: NDArray[np.bool_]) -> dict[str, int]:
    """Describe support with explicit foreground/background connectivity.

    Four-connected foreground matches the topology-preserving polygon recipe.
    Eight-connected foreground and four-connected background match OpenCV's
    external contour interpretation. Padding makes border-connected background
    unambiguously exterior, including for masks that fill the image.
    """
    four = ndimage.generate_binary_structure(2, 1)
    eight = ndimage.generate_binary_structure(2, 2)
    background = ~np.pad(mask, 1, constant_values=False)
    return {
        "area_px": int(np.count_nonzero(mask)),
        "components_foreground_4": int(ndimage.label(mask, four)[1]),
        "components_foreground_8": int(ndimage.label(mask, eight)[1]),
        "holes_background_8": int(ndimage.label(background, eight)[1] - 1),
        "holes_background_4": int(ndimage.label(background, four)[1] - 1),
    }


def polygon_metrics(outline: PolygonOutline) -> dict[str, int | float]:
    """Describe complete polygon geometry, including every interior ring."""
    return {
        "area_px2": float(sum(polygon.area for polygon in outline.polygons)),
        "components": len(outline.polygons),
        "holes": sum(len(polygon.interiors) for polygon in outline.polygons),
        "vertices": outline.vertex_count,
        "tolerance_px": outline.simplify_tolerance_px,
    }


def compare_polygons(
    proposal: SegmentationProposal, prompt: SegmentationPrompt
) -> tuple[dict[str, Any], dict[str, PolygonOutline]]:
    """Apply both polygon recipes to one unchanged raw prediction.

    Fragment cleanup belongs to the existing polygon recipe and is reported
    separately. A rejected CVAT contour remains an explicit failed panel.
    """
    records: dict[str, Any] = {}
    outlines = {}
    for style in ("preserve", "cvat"):
        start = perf_counter()
        record: dict[str, Any] = {
            "requested_tolerance_px": 1.4 if style == "preserve" else cvat_tolerance(9),
            "accuracy": 9 if style == "cvat" else None,
        }
        try:
            if style == "preserve":
                cleaned = remove_unprompted_fragments(proposal, prompt)
                record["cleanup"] = {
                    "removed_components": cleaned.removed_components,
                    "missed_inside_points": cleaned.missed_inside_points,
                }
                outline = simplify_polygon_outline(
                    polygon_outline(cleaned.proposal), 1.4
                )
            else:
                outline = simplify_cvat_contour(cvat_contour(proposal), accuracy=9)
            outlines[style] = outline
            record.update(
                polygon=polygon_metrics(outline),
                conversion_mask=mask_metrics(outline.proposal.mask),
                metadata=outline.proposal.metadata,
                error=None,
            )
        except (ValueError, ImportError, RuntimeError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["conversion_seconds"] = perf_counter() - start
        records[style] = record
    return records, outlines


def _predict(
    image: NDArray[np.uint8],
    prompt: SegmentationPrompt,
    provider: RegionGrowProvider | SegmentAnythingProvider,
) -> tuple[dict[str, SegmentationProposal], dict[str, Any]]:
    if isinstance(provider, RegionGrowProvider):
        start = perf_counter()
        proposal = provider.predict(image, prompt)
        seconds = perf_counter() - start
        return {"refinement": proposal, "single_mask": proposal}, {
            "refinement": {"seconds": seconds, "calls": 1},
            "single_mask": {"seconds": 0.0, "calls": 0, "reused_from": "refinement"},
            "note": "CPU predictions are identical: one result reused; only polygon recipes differ.",
        }
    previous = None
    start = perf_counter()
    step_times = []
    for stop in range(1, len(prompt.points) + 1):
        step_start = perf_counter()
        previous = provider._predict_with_refinement(
            image,
            SegmentationPrompt(prompt.points[:stop], prompt.labels[:stop]),
            previous,
        )
        step_times.append(perf_counter() - step_start)
    assert previous is not None
    refinement_seconds = perf_counter() - start
    start = perf_counter()
    single = provider._predict_single_mask(image, prompt)
    return {"refinement": previous.proposal, "single_mask": single}, {
        "refinement": {
            "seconds": refinement_seconds,
            "calls": len(step_times),
            "step_seconds": step_times,
        },
        "single_mask": {"seconds": perf_counter() - start, "calls": 1},
        "note": "Refinement runs first and includes image embedding; single-mask reuses that embedding. Times are not an unbiased speed benchmark.",
    }


def _overlay(
    path: Path,
    image: NDArray[np.uint8],
    prompt: SegmentationPrompt,
    raw: dict[str, SegmentationProposal],
    panels: dict[str, Any],
    outlines: dict[str, PolygonOutline],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    for row, prediction in enumerate(("refinement", "single_mask")):
        for col, conversion in enumerate(("preserve", "cvat")):
            key = f"{prediction}__{conversion}"
            axis = axes[row, col]
            axis.imshow(image)
            tint = np.zeros((*image.shape[:2], 4))
            tint[raw[prediction].mask] = [0.2, 0.7, 1.0, 0.25]
            axis.imshow(tint)
            for label, color, marker in ((1, "lime", "+"), (0, "red", "x")):
                points = prompt.points[prompt.labels == label]
                axis.scatter(points[:, 0], points[:, 1], c=color, marker=marker, s=65)
            record = panels[key]
            if key in outlines:
                for polygon in outlines[key].polygons:
                    x, y = polygon.exterior.xy
                    axis.plot(x, y, color="yellow", linewidth=1.3)
                    for ring in polygon.interiors:
                        x, y = ring.xy
                        axis.plot(x, y, color="magenta", linewidth=1.3)
                metrics = record["polygon"]
                detail = (
                    f"area {metrics['area_px2']:,.1f} px²; "
                    f"{metrics['components']} components; {metrics['holes']} holes; "
                    f"{metrics['vertices']} vertices"
                )
            else:
                detail = "Conversion rejected (see metrics.json)"
                axis.text(
                    0.5,
                    0.5,
                    record["error"],
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="white",
                    wrap=True,
                    bbox={"facecolor": "black", "alpha": 0.8},
                    fontsize=8,
                )
            axis.set_title(
                f"{prediction} prediction / {conversion} polygon\n{detail}", fontsize=10
            )
            axis.set_xlim(-0.5, image.shape[1] - 0.5)
            axis.set_ylim(image.shape[0] - 0.5, -0.5)
            axis.set_aspect("equal")
            axis.set_xlabel("x (pixel centers)")
            axis.set_ylabel("y (pixel centers)")
    figure.suptitle(
        "Same RGB + prompts + weights | blue: raw mask | yellow: exterior | magenta: holes\nDescriptive comparison; no ground-truth accuracy measured",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in (
        "neurospatial",
        "numpy",
        "scipy",
        "shapely",
        "opencv-python",
        "matplotlib",
        "torch",
        "SAM-2",
    ):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("cpu", "sam2"), default="cpu")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path)
    source.add_argument("--image", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--inside", type=float, nargs=2, action="append", metavar=("X", "Y")
    )
    parser.add_argument(
        "--outside", type=float, nargs=2, action="append", metavar=("X", "Y")
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory; existing paths are rejected.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(
            "--output-dir already exists; choose a new directory to preserve previous results."
        )
    if args.frame_index < 0:
        parser.error("--frame-index must be nonnegative.")
    if (args.video is not None or args.image is not None) and not args.inside:
        parser.error("Supply at least one --inside X Y for your image/video.")
    if args.video is not None:
        from neurospatial.annotation.core import _load_video_frame

        image = _load_video_frame(args.video.expanduser(), args.frame_index)
        source_info = {
            "kind": "video",
            "path": str(args.video.expanduser().resolve()),
            "frame_index": args.frame_index,
        }
    elif args.image is not None:
        from PIL import Image

        with Image.open(args.image.expanduser()) as opened:
            image = np.array(opened.convert("RGB"))
        source_info = {"kind": "image", "path": str(args.image.expanduser().resolve())}
    else:
        from try_annotation import sample_arena

        image = sample_arena()
        source_info = {
            "kind": "synthetic",
            "recipe": "examples/try_annotation.py:sample_arena",
        }
    inside = args.inside or [[160.0, 160.0]]
    outside = args.outside or []
    if (
        args.inside is None
        and args.outside is None
        and source_info["kind"] == "synthetic"
    ):
        outside = [[25.0, 25.0]]
    prompt = SegmentationPrompt(
        np.array(inside + outside), np.array([1] * len(inside) + [0] * len(outside))
    )
    if (prompt.points < 0).any() or (
        prompt.points >= [image.shape[1], image.shape[0]]
    ).any():
        parser.error("Every click must be finite and inside the RGB image bounds.")
    checkpoint = args.checkpoint or (
        Path(__file__).resolve().parents[1]
        / "data/annotation_demo/sam2.1_hiera_large.pt"
    )
    load_start = perf_counter()
    provider: RegionGrowProvider | SegmentAnythingProvider
    if args.provider == "cpu":
        provider = RegionGrowProvider(tolerance=0.1)
    else:
        print(
            f"Loading {args.model_config} on {args.device} from {checkpoint}…",
            flush=True,
        )
        provider = SegmentAnythingProvider.from_sam2(
            args.model_config, checkpoint, device=args.device
        )
    load_seconds = perf_counter() - load_start
    print("Predicting identical RGB data and ordered clicks…", flush=True)
    raw, prediction_info = _predict(image, prompt, provider)
    panels = {}
    outlines = {}
    for prediction, proposal in raw.items():
        records, converted = compare_polygons(proposal, prompt)
        for conversion, record in records.items():
            panels[f"{prediction}__{conversion}"] = record
        for conversion, outline in converted.items():
            outlines[f"{prediction}__{conversion}"] = outline
    union = np.count_nonzero(raw["refinement"].mask | raw["single_mask"].mask)
    metrics = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "versions": _versions(),
        "source": {
            **source_info,
            "rgb_shape": list(image.shape),
            "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        },
        "provider": args.provider,
        "provider_parameters": {"tolerance": 0.1} if args.provider == "cpu" else {},
        "model": None
        if args.provider == "cpu"
        else {
            "backend": "Meta SAM 2",
            "config": args.model_config,
            "checkpoint": str(checkpoint.expanduser().resolve()),
            "checkpoint_bytes": checkpoint.expanduser().stat().st_size,
            "device": args.device,
        },
        "load_seconds": load_seconds,
        "ordered_prompts": [
            {"x": float(point[0]), "y": float(point[1]), "label": int(label)}
            for point, label in zip(prompt.points, prompt.labels, strict=True)
        ],
        "prompt_order": "All inside clicks in supplied order, then all outside clicks in supplied order.",
        "prediction_recipes": prediction_info,
        "raw_predictions": {
            name: {
                **mask_metrics(proposal.mask),
                "score": proposal.score,
                "metadata": proposal.metadata,
            }
            for name, proposal in raw.items()
        },
        "raw_mask_iou": float(
            np.count_nonzero(raw["refinement"].mask & raw["single_mask"].mask) / union
        ),
        "panels": panels,
        "reference": {
            "cvat_ref": CVAT_UPSTREAM_REF,
            "url": f"https://github.com/cvat-ai/cvat/tree/{CVAT_UPSTREAM_REF}",
            "sam_interactor_url": "https://github.com/cvat-ai/cvat-models/blob/d2ebbc7f37d9c002d74fa8d392e9f3f8cb3aa419/interactor/sam2/func.py",
            "cvat_accuracy": 9,
            "preserve_tolerance_px": 1.4,
        },
        "limitations": [
            "No ground truth; these are descriptive geometry differences, not accuracy scores.",
            "CVAT-style prediction uses the same local Meta SAM backend; CVAT Online deployed weights and backend may differ.",
            "Prediction timings share cached embeddings and are not unbiased speed comparisons.",
            "Coordinates are image pixels; GeoJSON has no geographic CRS.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "raw_masks.npz",
        rgb=image,
        **{name: proposal.mask for name, proposal in raw.items()},
    )
    features = [
        {
            "type": "Feature",
            "id": name,
            "properties": {"panel": name, **polygon_metrics(outline)},
            "geometry": mapping(MultiPolygon(outline.polygons)),
        }
        for name, outline in outlines.items()
    ]
    (args.output_dir / "polygons.geojson").write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    _overlay(args.output_dir / "comparison.png", image, prompt, raw, panels, outlines)
    print(
        f"Raw-mask overlap (IoU): {metrics['raw_mask_iou']:.4f}; no ground-truth accuracy measured."
    )
    for name, record in panels.items():
        if record["error"]:
            print(f"  {name}: {record['error']}")
        else:
            values = record["polygon"]
            print(
                f"  {name}: {values['area_px2']:,.1f} px², {values['components']} components, {values['holes']} holes, {values['vertices']} vertices"
            )
    print(
        f"Saved overlay, metrics, RGB/raw masks, and full polygons to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
