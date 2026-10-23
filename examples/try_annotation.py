"""Try CPU or SAM 2 assisted annotation on a sample arena or your own recording.

Run from the annotation feature checkout with its optional viewers installed::

    uv run --no-sync python examples/try_annotation.py --provider cpu
    uv run --no-sync python examples/try_annotation.py --provider sam2
    uv run --no-sync python examples/try_annotation.py --provider cpu --video movie.mp4

SAM 2 uses Meta's SAM 2.1 Large, which needs its optional dependencies and the
downloaded checkpoint, and runs on a CUDA GPU or Apple silicon when available.
See the video annotation guide for setup. This script never downloads a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from neurospatial.annotation import (
    AnnotationProject,
    RegionGrowProvider,
    SegmentAnythingProvider,
    SegmentationPrompt,
    SegmentationProvider,
    annotate_project,
    annotate_video,
)


def sample_arena() -> NDArray[np.uint8]:
    """Return a synthetic RGB arena with a central obstacle and a reward area."""
    height, width = 512, 768
    rows, columns = np.mgrid[:height, :width]
    frame = np.full((height, width, 3), 32, dtype=np.uint8)
    frame[55:457, 75:693] = [105, 108, 112]
    floor = (rows >= 70) & (rows < 442) & (columns >= 90) & (columns < 678)
    shade = 8.0 * columns / width + 4.0 * rows / height
    colors = np.array([178, 166, 142]) + shade[..., None]
    frame[floor] = colors[floor].astype(np.uint8)
    obstacle = (columns - 384) ** 2 + (rows - 256) ** 2 <= 48**2
    frame[obstacle] = [45, 49, 53]
    reward = (columns - 600) ** 2 + (rows - 370) ** 2 <= 22**2
    frame[reward] = [80, 160, 100]
    return frame


def main() -> None:
    """Choose a provider, then open a frame in the annotation editor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("cpu", "sam2"), default="cpu")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="Use a frame from your video.")
    source.add_argument("--image", type=Path, help="Use a photo or extracted frame.")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--bin-size", type=float, default=8.0, help="Output grid size in pixels."
    )
    parser.add_argument(
        "--save-project", type=Path, help="Optional project/autosave destination."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="SAM 2.1 Large checkpoint (default: data/annotation_demo/sam2.1_hiera_large.pt).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="SAM 2 device (default: CUDA, then Apple silicon, then CPU).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a sample prediction without opening a viewer.",
    )
    args = parser.parse_args()
    if args.check and (args.image is not None or args.video is not None):
        parser.error("--check uses the sample arena; omit --image and --video.")
    if args.save_project is not None and args.save_project.exists():
        parser.error(
            "Project already exists. Resume it with annotate_project(path), or choose a new path."
        )

    provider: SegmentationProvider
    if args.provider == "cpu":
        provider = RegionGrowProvider(tolerance=0.1)
    else:
        checkpoint = args.checkpoint or (
            Path(__file__).resolve().parents[1]
            / "data/annotation_demo/sam2.1_hiera_large.pt"
        )
        device = None if args.device == "auto" else args.device
        print(
            f"Loading SAM 2.1 Large on {device or 'the fastest available device'} "
            f"from {checkpoint}…",
            flush=True,
        )
        provider = SegmentAnythingProvider.from_sam2(checkpoint, device=device)

    if args.check:
        prompt = SegmentationPrompt(
            np.array([[160.0, 160.0], [25.0, 25.0]]),
            np.array([1, 0]),
        )
        start = perf_counter()
        proposal = provider.predict(sample_arena(), prompt)
        print(
            f"{args.provider}: {int(proposal.mask.sum()):,} foreground pixels; "
            f"score={proposal.score}; prediction took {perf_counter() - start:.2f}s"
        )
        return

    print(
        "In Project and Assistance: Mark points → left-click the tan arena "
        "(or your target area) → refine the live outline → Accept outline.\n"
        "Right-click to exclude areas; Alt-click to remove a prompt.\n"
        "Polygon detail starts at 1.4 px; set it to 0 for the exact mask boundary.\n"
        "Use Paint / erase for manual corrections.\n"
        "Choose the annotation role/name in Annotation Controls; click Done there to finish.",
        flush=True,
    )
    if args.video is not None:
        result = annotate_video(
            args.video,
            frame_index=args.frame_index,
            bin_size=args.bin_size,
            project_path=args.save_project,
            segmentation_provider=provider,
        )
    else:
        if args.image is None:
            frame = sample_arena()
        else:
            from PIL import Image

            with Image.open(args.image) as image:
                frame = np.array(image.convert("RGB"))
        result = annotate_project(
            AnnotationProject(frame, bin_size=args.bin_size),
            project_path=args.save_project,
            segmentation_provider=provider,
        )
    bins = 0 if result.environment is None else result.environment.n_bins
    print(f"Finished: {bins:,} environment bins, {len(result.regions)} named regions.")


if __name__ == "__main__":
    main()
