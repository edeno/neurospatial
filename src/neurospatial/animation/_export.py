"""Physical-clock sampling and atomic video writing shared by scene and views."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class _ExportPlan:
    path: Path
    times: NDArray[np.float64]
    fps: float
    overwrite: bool
    codec: str | None

    def write(self, render_frame: Callable[[float], NDArray[np.uint8]]) -> Path:
        """Encode frames without publishing a partial file or replacing a race winner."""
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError(
                "Animation export requires imageio. Install video support with: "
                "pip install 'neurospatial[video]'."
            ) from exc

        descriptor, temporary = tempfile.mkstemp(
            suffix=self.path.suffix, prefix=".animation-", dir=self.path.parent
        )
        os.close(descriptor)
        try:
            options: dict[str, Any] = (
                {"mode": "I", "duration": 1000 / self.fps, "loop": 0}
                if self.path.suffix.lower() == ".gif"
                else {
                    "fps": self.fps,
                    "codec": self.codec or "libx264",
                    "macro_block_size": 2,
                }
            )
            shape = None
            with cast("Any", imageio.get_writer(temporary, **options)) as writer:
                for time in self.times:
                    frame = np.asarray(render_frame(float(time)))
                    if (
                        frame.dtype != np.uint8
                        or frame.ndim != 3
                        or frame.shape[-1] not in (3, 4)
                        or min(frame.shape[:2]) == 0
                    ):
                        raise ValueError(
                            "The renderer must return a nonempty uint8 RGB or RGBA image."
                        )
                    frame = frame[..., :3]
                    if shape is None:
                        shape = frame.shape
                    elif frame.shape != shape:
                        raise ValueError(
                            "The rendered image size changed during export. "
                            "Keep the viewer window size fixed until export completes."
                        )
                    writer.append_data(frame)
            if self.overwrite:
                Path(temporary).replace(self.path)
            else:
                _publish_exclusive(Path(temporary), self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return self.path


def _publish_exclusive(temporary: Path, path: Path) -> None:
    """Publish ``temporary`` at ``path`` without replacing a file that appeared.

    A different writer may create the destination after prepare_export checked
    it. Hard links publish complete files exclusively; filesystems without them
    (exFAT, FAT32, some network mounts) use an exclusive create and copy.
    """
    try:
        os.link(temporary, path)
        return
    except FileExistsError:
        raise
    except OSError:
        pass
    with path.open("xb") as destination:
        try:
            with temporary.open("rb") as source:
                shutil.copyfileobj(source, destination)
        except BaseException:
            destination.close()
            path.unlink(missing_ok=True)
            raise


def prepare_export(
    path: str | Path,
    source_times: NDArray[np.float64],
    *,
    time_range: Any = None,
    fps: float = 30,
    speed: float = 1,
    overwrite: bool = False,
    codec: str | None = None,
    sample_times: Iterable[NDArray[np.float64]] = (),
) -> _ExportPlan:
    """Validate export before touching a viewer and select ``[start, stop)``.

    Correct arithmetic roundoff to the nearest observation within four ULPs.
    Exact queries stay exact; ties prefer the scene clock, then earlier clocks
    in ``sample_times``. Distinct nearby source floats cannot all be exact at
    one shared export time.
    """
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(speed) or speed <= 0:
        raise ValueError("fps and speed must be positive finite numbers.")
    path = Path(path)
    if path.suffix.lower() not in (".mp4", ".gif"):
        raise ValueError("Animation export supports .mp4 and .gif paths.")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it."
        )
    bounds = np.asarray(
        (source_times[0], source_times[-1]) if time_range is None else time_range,
        dtype=float,
    )
    if (
        bounds.shape != (2,)
        or not np.isfinite(bounds).all()
        or bounds[1] < bounds[0]
        or (time_range is not None and bounds[1] == bounds[0])
    ):
        raise ValueError(
            "time_range must be a finite half-open (start, stop) pair with start < stop."
        )
    if bounds[0] < source_times[0] or bounds[1] > source_times[-1]:
        raise ValueError("time_range must lie inside the scene's physical clock.")
    steps = float((bounds[1] - bounds[0]) * fps / speed)
    nearest = round(steps)
    if np.isclose(steps, nearest, rtol=1e-12, atol=1e-12):
        steps = float(nearest)
    count = max(1, int(np.ceil(steps)))
    times = bounds[0] + np.arange(count) * speed / fps
    # Adding a decimal frame interval to an absolute recording time can land
    # one ULP before an observation (100.1 + 0.1 < 100.2). That must not hold
    # the preceding field or hide an exact sample beside a recording break.
    # Keep genuinely intermediate times unchanged, and consider each stream's
    # clock even when the caller chose a separate scene frame clock.
    snapped = times.copy()
    best_distance = np.full(times.shape, np.inf)
    tolerance = 4 * np.abs(np.spacing(times))
    for clock in chain((source_times,), sample_times):
        if not len(clock):
            continue
        right = np.searchsorted(clock, times)
        for index in (np.maximum(right - 1, 0), np.minimum(right, len(clock) - 1)):
            candidate = clock[index]
            distance = np.abs(candidate - times)
            valid = (candidate >= bounds[0]) & (candidate <= bounds[1])
            nearer = valid & (distance <= tolerance) & (distance < best_distance)
            snapped[nearer] = candidate[nearer]
            best_distance[nearer] = distance[nearer]
    times = snapped
    if bounds[1] > bounds[0]:
        times = times[times < bounds[1]]
    times.flags.writeable = False
    return _ExportPlan(path, times, float(fps), overwrite, codec)
