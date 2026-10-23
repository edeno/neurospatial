"""Render scene frames in worker processes, delivered in order to one encoder.

Each worker rebuilds an equivalent :class:`Animation` from its construction
arguments: the scene itself holds locks and open video readers, while file
paths, arrays and environments travel by pickle. Memory-mapped arrays are
sent as a reference to their file and reopened read-only, so a recording is
not copied into every worker. Workers apply the parent's Matplotlib settings,
so frames are identical to rendering in the parent.
"""

from __future__ import annotations

import io
import mmap
import pickle
import warnings
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Frames per task and tasks queued per worker. Rendered frames waiting for
# the encoder are bounded by _CHUNK * _IN_FLIGHT_PER_WORKER * n_workers.
_CHUNK = 4
_IN_FLIGHT_PER_WORKER = 2


def _reopen_memmap(
    filename: str,
    offset: int,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    strides: tuple[int, ...],
) -> NDArray[Any]:
    buffer = np.memmap(filename, dtype=np.uint8, mode="r")
    return np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset, strides=strides)


class _ScenePickler(pickle.Pickler):
    """Pickle file-backed memory maps, including views, by reference."""

    def reducer_override(self, obj: Any) -> Any:
        mapped = getattr(obj, "_mmap", None)
        # Copy-on-write maps may hold changes that are not in the file.
        if (
            isinstance(obj, np.memmap)
            and mapped is not None
            and obj.filename is not None
            and obj.mode != "c"
        ):
            base = np.frombuffer(mapped, dtype=np.uint8).ctypes.data
            # np.memmap maps the file from an allocation-granularity boundary.
            start = obj.offset - obj.offset % mmap.ALLOCATIONGRANULARITY
            offset = start + obj.ctypes.data - base
            return _reopen_memmap, (
                obj.filename,
                offset,
                obj.shape,
                obj.dtype,
                obj.strides,
            )
        return NotImplemented


def _dumps(value: Any) -> bytes:
    buffer = io.BytesIO()
    _ScenePickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return buffer.getvalue()


def _pickled_scene(scene: Any, n_workers: int) -> dict[str, Any]:
    """Pickle each construction argument, naming one that cannot be sent."""
    arguments = scene._construction_arguments()
    parts: list[tuple[str, str, Any]] = [
        ("the environment", "env", arguments.pop("env")),
        *(
            (f"field {name!r}", f"field:{name}", field)
            for name, field in arguments.pop("fields").items()
        ),
        *(
            (f"overlay {index} ({type(overlay).__name__})", f"overlay:{index}", overlay)
            for index, overlay in enumerate(arguments.pop("overlays"))
        ),
        *((name.replace("_", " "), name, value) for name, value in arguments.items()),
    ]
    pickled = {}
    for label, key, value in parts:
        try:
            pickled[key] = _dumps(value)
        except Exception as error:
            raise ValueError(
                f"Parallel export (n_workers={n_workers}) sends the scene to "
                f"worker processes, but {label} cannot be pickled "
                f"({type(error).__name__}: {error}). Pass n_workers=1 to render "
                "in this process, or replace what cannot be pickled with plain "
                "data: NumPy arrays, numbers, strings, and file paths instead of "
                "open readers, locks, lambdas or local functions."
            ) from error
    return pickled


_worker: dict[str, Any] = {}


def _initialize(
    pickled: dict[str, bytes],
    rc_params: dict[str, Any],
    dpi: int,
    figsize: tuple[float, float] | None,
) -> None:
    # Build lazily in the first task, so a failure reaches the parent with
    # its own exception rather than as a broken pool.
    _worker.update(pickled=pickled, rc_params=rc_params, dpi=dpi, figsize=figsize)


def _renderer() -> Any:
    if "renderer" not in _worker:
        from multiprocessing.util import Finalize

        import matplotlib

        from neurospatial.animation._scene_rendering import SceneRenderer
        from neurospatial.animation.scene import Animation

        matplotlib.rcParams.update(_worker.pop("rc_params"))
        arguments: dict[str, Any] = {"fields": {}, "overlays": []}
        for key, data in _worker.pop("pickled").items():
            value = pickle.loads(data)
            kind, _, name = key.partition(":")
            if kind == "field":
                arguments["fields"][name] = value
            elif kind == "overlay":
                arguments["overlays"].append(value)
            else:
                arguments[key] = value
        # The parent already reported construction warnings for this scene.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene = Animation(**arguments)
        # Release video readers when the worker exits.
        Finalize(None, scene.close, exitpriority=0)
        _worker["renderer"] = SceneRenderer(
            scene, dpi=_worker["dpi"], figsize=_worker["figsize"]
        )
    return _worker["renderer"]


def _render_chunk(times: tuple[float, ...]) -> list[NDArray[np.uint8]]:
    renderer = _renderer()
    return [renderer.render(time) for time in times]


def render_in_workers(
    scene: Any,
    times: NDArray[np.float64],
    *,
    dpi: int,
    figsize: tuple[float, float] | None,
    n_workers: int,
) -> Iterator[NDArray[np.uint8]]:
    """Frames at ``times``, rendered by worker processes, in order.

    Pickling is checked before any worker starts. Contiguous chunks keep
    each worker's renderer moving forward in time.
    """
    import matplotlib

    pickled = _pickled_scene(scene, n_workers)
    rc_params = {
        # Reading "backend" through rcParams would select and import one.
        key: dict.__getitem__(matplotlib.rcParams, key)
        for key in matplotlib.rcParams
        if key != "backend"
    }
    chunks = [
        tuple(float(time) for time in times[start : start + _CHUNK])
        for start in range(0, len(times), _CHUNK)
    ]
    n_workers = min(n_workers, len(chunks))
    return _ordered_frames(chunks, (pickled, rc_params, dpi, figsize), n_workers)


def _ordered_frames(
    chunks: list[tuple[float, ...]], initargs: tuple[Any, ...], n_workers: int
) -> Iterator[NDArray[np.uint8]]:
    remaining = iter(chunks)
    pending: deque[Future[list[NDArray[np.uint8]]]] = deque()
    with ProcessPoolExecutor(
        n_workers, initializer=_initialize, initargs=initargs
    ) as pool:
        try:
            for chunk in remaining:
                pending.append(pool.submit(_render_chunk, chunk))
                if len(pending) == n_workers * _IN_FLIGHT_PER_WORKER:
                    break
            while pending:
                frames = pending.popleft().result()
                following: tuple[float, ...] | None = next(remaining, None)
                if following is not None:
                    pending.append(pool.submit(_render_chunk, following))
                yield from frames
        finally:
            # A failed encoder or render stops the remaining work promptly.
            pool.shutdown(cancel_futures=True)
