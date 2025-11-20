# Animation Overlays — Merged Plan (v0.4.0)

**Status:** Ready for implementation
**Target:** v0.4.0

## 1) Public API (dataclasses + protocol update)

### 1.1 Overlay config dataclasses (user-facing)

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray


@dataclass
class PositionOverlay:
    """Single trajectory with optional trail."""
    data: NDArray[np.float64]        # (n_samples, n_dims)
    times: NDArray[np.float64] | None = None
    color: str = "red"
    size: float = 10.0
    trail_length: int | None = None   # in frames after alignment


@dataclass
class BodypartOverlay:
    """Multi-keypoint pose with optional skeleton."""
    data: dict[str, NDArray[np.float64]]       # {part_name: (n_samples, n_dims)}
    times: NDArray[np.float64] | None = None
    skeleton: list[tuple[str, str]] | None = None
    colors: dict[str, str] | None = None       # per-part color
    skeleton_color: str = "white"
    skeleton_width: float = 2.0


@dataclass
class HeadDirectionOverlay:
    """Heading as angles (rad) or unit vectors; rendered as arrows."""
    data: NDArray[np.float64]                  # (n_samples,) or (n_samples, n_dims)
    times: NDArray[np.float64] | None = None
    color: str = "yellow"
    length: float = 20.0                       # arrow length in env units
```

### 1.2 Environment protocol change (mandatory)

Update `EnvironmentProtocol.animate_fields()` to accept overlays, `frame_times`, and region controls:

```python
def animate_fields(
    self,
    fields: Any,
    *,
    backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
    save_path: str | None = None,
    fps: int = 30,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: Any = None,
    title: str = "Spatial Field Animation",
    dpi: int = 100,
    codec: str = "h264",
    bitrate: int = 5000,
    n_workers: int | None = None,
    dry_run: bool = False,
    image_format: Literal["png", "jpeg"] = "png",
    max_html_frames: int = 500,
    contrast_limits: tuple[float, float] | None = None,
    show_colorbar: bool = False,
    colorbar_label: str = "",
    overlays: list[PositionOverlay | BodypartOverlay | HeadDirectionOverlay] | None = None,
    frame_times: NDArray[np.float64] | None = None,  # (n_frames,)
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
) -> Any: ...
```

*Multi-animal support:* pass multiple instances of the same overlay type (distinct colors/sizes).

---

## 2) Internal data model & timeline

### 2.1 Private timeline (hidden helper)

* Construct frame times either from `frame_times` (preferred, strict) or from `fps` and `n_frames`.
* Provide vectorized interpolation:

  * `interp_linear(t_src, x_src) → x_frame`
  * `interp_nearest(t_src, x_src) → x_frame`
* Handle edge fill: extrapolate as NaN; validation decides whether to warn or fail.

### 2.2 Internal container given to backends

```python
from dataclasses import dataclass, field

@dataclass
class PositionData:
    data: NDArray[np.float64]     # (n_frames, n_dims)
    color: str
    size: float
    trail_length: int | None


@dataclass
class BodypartData:
    bodyparts: dict[str, NDArray[np.float64]]  # name → (n_frames, n_dims)
    skeleton: list[tuple[str, str]] | None
    colors: dict[str, str] | None
    skeleton_color: str
    skeleton_width: float


@dataclass
class HeadDirectionData:
    data: NDArray[np.float64]     # (n_frames,) or (n_frames, n_dims)
    color: str
    length: float


@dataclass
class OverlayData:
    positions: list[PositionData] = field(default_factory=list)
    bodypart_sets: list[BodypartData] = field(default_factory=list)
    head_directions: list[HeadDirectionData] = field(default_factory=list)
    regions: list[str] | dict[int, list[str]] | None = None  # optional, unchanged for v0.4.0

    def __post_init__(self) -> None:
        # Pickle-ability check (no lambdas/closures); raise ValueError with fix hints
        ...
```

### 2.3 Conversion funnel

`_convert_overlays_to_data(overlays, frame_times, n_frames, env) -> OverlayData`

* Align each overlay to `n_frames` using private Timeline.
* For `BodypartOverlay`, align each keypoint separately.
* Ensure coordinate dimension equals `env.n_dims`.
* Return one pickle-safe `OverlayData` used by all backends.

---

## 3) Validation (WHAT / WHY / HOW)

Run once in conversion:

1. **Monotonic time**

   * *WHAT:* Non-monotonic `times` detected.
   * *WHY:* Interpolation requires increasing timestamps.
   * *HOW:* Sort or call `fix_monotonic_timestamps()`.

2. **Finite values**

   * *WHAT:* Found NaN/Inf in overlay arrays (count and first index).
   * *WHY:* Rendering cannot place invalid coordinates.
   * *HOW:* Clean or mask; consider interpolation over gaps.

3. **Shape**

   * *WHAT:* Shape mismatch (expected `(n_samples, n_dims)`, got `...`).
   * *WHY:* Coordinate dimensionality must match environment.
   * *HOW:* Project/reformat to `(…, 2)` or set consistent `env.n_dims`.

4. **Temporal alignment**

   * *WHAT:* No overlap between `overlay.times` and `frame_times`.

   * *WHY:* Interpolation domain is disjoint.

   * *HOW:* Provide overlapping time ranges or resample source data.

   * *WARN:* Partial overlap <50% (report percentage).

5. **Bounds**

   * *WARN:* >X% points outside `env.dimension_ranges` (show min/max vs env).
   * *HOW:* Confirm coordinate system and units.

6. **Skeleton consistency**

   * *WHAT:* Skeleton references missing part(s).
   * *WHY:* Cannot draw edges without endpoints.
   * *HOW:* Fix names; suggest nearest matches.

7. **Pickle-ability**

   * *WHAT:* OverlayData not pickle-able (show offending attribute).
   * *WHY:* Parallel video rendering requires pickling.
   * *HOW:* Remove unpickleable obj or run with `n_workers=1`; `env.clear_cache()` first.

---

## 4) Module & call flow

```
src/neurospatial/animation/
├── overlays.py        # dataclasses, validation, conversion, timeline helpers
├── core.py            # calls conversion, routes to backend
├── rendering.py       # existing field imagery helpers (unchanged)
└── backends/
    ├── napari_backend.py  # full overlays
    ├── video_backend.py   # full overlays (matplotlib)
    ├── html_backend.py    # partial overlays (positions + regions)
    └── widget_backend.py  # reuse video’s renderer/caching
```

**Dispatcher (core):**

1. Compute `n_frames` from `fields`.
2. Build/verify `frame_times` (or synthesize from `fps`).
3. `overlay_data = _convert_overlays_to_data(...)`
4. Route with `overlay_data` to selected backend.

---

## 5) Backend contracts

### 5.1 Napari (full)

* Layers:

  * Positions: `add_tracks` for trail, `add_points` for current position.
  * Bodyparts: `add_points` per part set; Skeleton: `add_shapes` with `shape_type="line"`.
  * Head direction: `add_vectors`.
  * Regions: `add_shapes` polygons if `show_regions`.
* One callback bound to `viewer.dims.events.current_step` updates all layers in a single batch to minimize latency.
* Axis order: convert `(x, y)` → `(y, x)`.

### 5.2 Video (full, matplotlib)

* Render overlays into each frame:

  * Trails as polylines with decaying alpha (ring buffer by `trail_length`).
  * Skeleton via `LineCollection` (single call per frame).
  * Head direction via arrows (vectorized).
  * Regions via `PathPatch`.
* Parallel-safe: validate `OverlayData` pickle-ability; call `env.clear_cache()` when `n_workers > 1`.

### 5.3 HTML (partial)

* Client-side canvas draws **positions + regions** only.
* Serialize compact overlay JSON; enforce `max_html_frames`; emit capability warnings when bodyparts/head direction are supplied.
* Keep file size guardrails; auto-disable oversized overlays with a user-facing warning string.

### 5.4 Widget

* Reuse video renderer to produce PNG frames (with overlays).
* LRU cache frames for responsive scrubbing.

---

## 6) Performance notes

* Prefer single allocations and artist reuse per frame.
* Trails: build coordinates once per frame; avoid many scatter points.
* Skeleton: always `LineCollection`, not per-edge loops.
* Keep Napari update < 50 ms/frame with batched layer updates.
* Maintain parallel video encode; avoid capturing closures in worker payloads.

---

## 7) Testing matrix (focused)

**Unit (tests/animation/test_overlays.py)**

* Time monotonicity; finite checks; shape checks.
* Temporal overlap (error) and partial overlap (warning).
* Skeleton name diffs.
* Pickle-ability failures (actionable message).

**Integration (tests/animation/test_animation_with_overlays.py)**

* Napari: layers created, batch update works, axis swap correct.
* Video: LineCollection used; trails render; `n_workers > 1` succeeds.
* HTML: positions + regions only; size warnings trigger properly.
* Cross-backend parity for same overlay config.

**Performance (tests/animation/test_performance.py)**

* Napari update < 50 ms on realistic pose + trail.
* Video overhead < 2× vs no overlays.

**Visual regression (pytest-mpl)**

* Golden images: trail, skeleton, head direction, regions.

---

## 8) Documentation & examples (lean)

* `docs/animation_overlays.md`:

  * Quickstart with three overlays.
  * Multi-animal by passing multiple instances.
  * Backend capability table + HTML limits.
  * Common errors (WHAT/WHY/HOW) and fixes.
* `examples/08_animation_with_overlays.ipynb`:

  * Trajectory + trail, pose + skeleton, head direction, multi-animal, regions.
  * Mixed-rate alignment using `frame_times`.

---

## 9) Phased delivery (short)

1. **Core & Validation (week 1–2)**
   Dataclasses, Timeline, conversion, validation, tests.

2. **Napari (week 3–4)**
   Layers + batched updates + profiling.

3. **Video (week 5–6)**
   Matplotlib renderer, LineCollection, parallel path.

4. **HTML & Widget (week 7)**
   Canvas (positions/regions), widget reuse, warnings.

5. **Docs, Examples, Benchmarks (week 8)**
   Tight docs, gallery notebook, regression baselines.

---

## 10) Notes for implementers

* Keep all new public types in `overlays.py`; backends import only `OverlayData`.
* Do **not** introduce another public overlay protocol; the dataclasses are the user API.
* Internally, the private Timeline and conversion funnel are the single source of truth for alignment and validation.
* Treat regions as a flag in v0.4.0; design a future `RegionsOverlay` so the backend contracts won’t change later.

This is internally consistent, narrow enough to fit v0.4.0, and directly actionable by the codebase.
