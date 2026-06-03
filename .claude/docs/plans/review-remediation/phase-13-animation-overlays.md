# Phase 13 — Animation: overlay correctness

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

This phase fixes four overlay-rendering correctness bugs in
`src/neurospatial/animation/` that silently misplace, mis-orient, or drop
overlay data. None of them raise — they all produce a plausible-but-wrong
animation, which is why they shipped. Each fix is local and independently
testable:

1. The napari coordinate transform truncates fractional pixel coordinates to
   `int` whenever the input overlay coords have an integer dtype (because
   `np.empty_like(coords)` inherits that dtype), misplacing pixel/DLC overlays
   by up to ~1 bin.
2. `HeadDirectionOverlay` linearly interpolates 1-D heading **angles**, which
   sweeps the wrong way across the ±π wrap and can point the arrow ~180° off.
3. `_validate_bounds` prints `nan` for the data-range diagnostic when aligned
   overlay data has NaN gaps.
4. The **default** reuse-artists video render path never renders `EventOverlay`
   markers, so events silently vanish from exported video.

**Inputs to read first:**

- [src/neurospatial/animation/transforms.py:320-323](../../../../src/neurospatial/animation/transforms.py) — `transform_coords_for_napari` computes `row`/`col` as float64, then writes them into `result = np.empty_like(coords)` (line 320). `np.empty_like` copies the **dtype** of `coords`; if `coords` is integer-dtyped (e.g. pixel-mask overlay coordinates, or DLC keypoints stored as int), the float `row`/`col` are truncated on assignment at lines 321-322. Fix: allocate `np.empty(coords.shape, dtype=np.float64)`.
- [src/neurospatial/animation/transforms.py:377-379](../../../../src/neurospatial/animation/transforms.py) — same bug in the **fallback** branch of `transform_direction_for_napari`: `result = np.empty_like(direction)` (line 377) before writing `-dy`/`dx`.
- [src/neurospatial/animation/transforms.py:388-390](../../../../src/neurospatial/animation/transforms.py) — same bug in the **scaled** branch of `transform_direction_for_napari`: `result = np.empty_like(direction)` (line 388) before writing `dr`/`dc`.
- [src/neurospatial/animation/overlays.py:641-687](../../../../src/neurospatial/animation/overlays.py) — `HeadDirectionOverlay.convert_to_data`. When `self.headings` is a **1-D angle array** (the documented angle form; the `(n, 2)` unit-vector form is at lines 609-614 in the docstring), it passes the raw angles to `_align_to_frame_times` (line 673). For 1-D data this routes to `np.interp`, which interpolates angles **linearly** — across the ±π discontinuity this gives a value far from either neighbor (e.g. interpolating between `+3.0` and `-3.0` rad yields ~`0.0`, pointing the arrow the opposite way). The fix lives here, not in the shared interpolator.
- [src/neurospatial/animation/overlays.py:639](../../../../src/neurospatial/animation/overlays.py) — the `interp: Literal["linear", "nearest"]` field on `HeadDirectionOverlay`. The circular fix only applies to the `"linear"` path; `"nearest"` already picks an actual sample and is correct.
- [src/neurospatial/animation/overlays.py:3776-3864](../../../../src/neurospatial/animation/overlays.py) — `_interp_linear`. For `x_src.ndim == 1` it calls `np.interp` (line 3858); for N-D it interpolates each column independently (lines 3861-3862). This function is shared by **every** overlay (positions, bodyparts, rates, …), so it must **not** be changed to assume circularity. The 2-D unit-vector heading path through this function is correct (interpolating Cartesian components, then re-normalizing downstream is fine). Only the 1-D-angle case is wrong, and only `HeadDirectionOverlay` produces 1-D angle data.
- [src/neurospatial/animation/overlays.py:3344-3414](../../../../src/neurospatial/animation/overlays.py) — `_validate_bounds`. Line 3389-3390 compute `data.min(axis=0)` / `data.max(axis=0)` for the warning's "Data ranges" line. After temporal alignment, out-of-range frames are filled with NaN by `_interp_linear` (line 3843), so `data` routinely contains NaN rows; `np.ndarray.min`/`max` then return `nan`, printing `[nan, nan]`. The outside-bounds **count** (lines 3381-3385) is unaffected because NaN comparisons are `False`. Fix: `np.nanmin`/`np.nanmax` on lines 3389-3390. (Guard for the all-NaN column is below — see Task 3.)
- [src/neurospatial/animation/_parallel.py:825-865](../../../../src/neurospatial/animation/_parallel.py) — `OverlayArtistManager.initialize`. It creates persistent artists for regions, positions, bodyparts, and head directions — but **not** events. There is no event artist list and no event branch.
- [src/neurospatial/animation/_parallel.py:1066-1104](../../../../src/neurospatial/animation/_parallel.py) — `OverlayArtistManager.update_frame`. Updates positions, bodyparts, and head directions per frame — again **no** event handling.
- [src/neurospatial/animation/_parallel.py:1194-1251](../../../../src/neurospatial/animation/_parallel.py) — `_update_head_direction` (lines 1194-1211) and `clear` (lines 1213-1251). The head-direction quiver is *recreated* each frame (remove old artist, build new) — this is the exact pattern Task 4 reuses for events, since event visibility (cumulative / instant / decay) changes every frame. `clear` removes every tracked artist list; the new event list must be torn down there too.
- [src/neurospatial/animation/_parallel.py:135-211](../../../../src/neurospatial/animation/_parallel.py) — `_render_event_overlay_matplotlib`. Computes per-frame visible events (cumulative / instant / decay modes, lines 168-178) and draws each as `ax.scatter(...)` at `zorder=104` (lines 202-211). It does **not** return the created artists, so the reuse path cannot track them as written; Task 4 captures and returns them.
- [src/neurospatial/animation/_parallel.py:1334-1336](../../../../src/neurospatial/animation/_parallel.py) — inside `_render_all_overlays`: the **legacy / non-reuse** path *does* render events (`for event_data in overlay_data.events: _render_event_overlay_matplotlib(...)`). This is the behavior the reuse path must match.
- [src/neurospatial/animation/_parallel.py:1684-1737](../../../../src/neurospatial/animation/_parallel.py) — the render driver. When `reuse_artists` is True and there are overlays, it builds an `OverlayArtistManager` and calls `initialize(0)` (line 1693) for frame 0 and `overlay_manager.update_frame(local_idx)` (line 1731) for frames 1+. `reuse_artists` defaults to True (line 1612: `task.get("reuse_artists", True)`). So with events present, the **default** export path emits frames with no event markers. The `else` fallback (lines 1732-1737) and the `reuse_artists is False` branch (lines 1748-1779) both call `_render_all_overlays`, which renders events — confirming the bug is specific to the manager path.
- [src/neurospatial/animation/overlays.py:2610-2677](../../../../src/neurospatial/animation/overlays.py) — `EventData` dataclass: `event_positions: dict[str, NDArray]`, `event_frame_indices: dict[str, NDArray[np.int_]]`, `colors`, `markers`, `size`, `decay_frames`, `border_color`, `border_width`, `opacity`. This is what `OverlayData.events` (overlays.py:3104, `events: list[EventData]`) holds, and what `_render_event_overlay_matplotlib` consumes.
- [tests/animation/test_overlay_artist_manager.py:1-40](../../../../tests/animation/test_overlay_artist_manager.py) — existing `OverlayArtistManager` lifecycle tests, `simple_env` fixture, and import conventions (`from neurospatial.animation._parallel import OverlayArtistManager`, `from neurospatial.animation.overlays import ..., OverlayData, EventData`). New manager/event tests go here.
- [tests/animation/test_transforms.py](../../../../tests/animation/test_transforms.py) — existing transform tests; the integer-dtype regression test goes here. (No napari/ffmpeg dependency — these run unconditionally.)
- [tests/animation/test_event_overlay.py](../../../../tests/animation/test_event_overlay.py) — existing event-overlay tests for fixture/style conventions.
- [tests/animation/conftest.py](../../../../tests/animation/conftest.py) — shared fixtures and the marker/skip conventions (napari and ffmpeg gating) used across the animation suite.

**Contracts referenced:** none. (This phase touches no `_validation.py` helper and no result-object surface; it is purely animation-internal.)

## Tasks

### 1. Allocate transform outputs as float64 (`animation/transforms.py`)

The three `np.empty_like(...)` allocations inherit the input dtype and truncate
float results when the input is integer-typed. Replace each with an explicit
`np.float64` allocation. The computed `row`/`col`/`dr`/`dc` are already float64,
so this is the only change needed.

In `transform_coords_for_napari`, change line 320 from:

```python
        # Return in napari (row, col) order
        result = np.empty_like(coords)
```

to:

```python
        # Return in napari (row, col) order. Allocate float64 explicitly:
        # np.empty_like would inherit an integer dtype from integer-typed input
        # (e.g. pixel-mask or DLC coords) and silently truncate the fractional
        # pixel position.
        result = np.empty(coords.shape, dtype=np.float64)
```

In `transform_direction_for_napari`, change the fallback-branch allocation at
line 377 from:

```python
            result = np.empty_like(direction)
            result[..., 0] = -dy  # Y inverted (environment Y up, napari row down)
            result[..., 1] = dx
```

to:

```python
            result = np.empty(direction.shape, dtype=np.float64)
            result[..., 0] = -dy  # Y inverted (environment Y up, napari row down)
            result[..., 1] = dx
```

and the scaled-branch allocation at line 388 from:

```python
        result = np.empty_like(direction)
        result[..., 0] = dr
        result[..., 1] = dc
```

to:

```python
        result = np.empty(direction.shape, dtype=np.float64)
        result[..., 0] = dr
        result[..., 1] = dc
```

No docstring change is needed — both functions already document a float pixel
result; the int truncation was an undocumented bug.

### 2. Circular interpolation for 1-D head-direction angles (`animation/overlays.py`)

The fix is scoped to `HeadDirectionOverlay.convert_to_data` (overlays.py:641-687).
**Do not** change the shared `_interp_linear` (overlays.py:3776) — it is correct
for every other overlay and for the 2-D unit-vector heading form.

When the heading data is 1-D **and** `interp == "linear"` **and** timestamps are
provided, interpolate on the unit circle: convert angles to `(cos, sin)`
components, interpolate the two components through the existing
`_align_to_frame_times` machinery (which keeps the NaN-extrapolation and
temporal-alignment validation), then recover angles with `arctan2`. This takes
the geodesic ("short way") between samples and is correct across the ±π wrap.

The `"nearest"` path and the 2-D vector path already pick/preserve real samples,
so leave them on the existing `_align_to_frame_times` call.

Replace the alignment block (overlays.py:672-680):

```python
        # Align to frame times (validates times if provided)
        aligned_data = _align_to_frame_times(
            self.headings,
            self.times,
            frame_times,
            n_frames,
            self.interp,
            name="HeadDirectionOverlay",
        )
```

with:

```python
        # Align to frame times (validates times if provided).
        #
        # For 1-D angle data interpolated linearly, plain linear interpolation
        # of the angle is wrong across the +/-pi wrap (e.g. interpolating
        # between +3.0 and -3.0 rad yields ~0.0, pointing the arrow the opposite
        # way). Interpolate on the unit circle instead: split into (cos, sin),
        # interpolate the components, then recover the angle via arctan2. This
        # always takes the short way around the circle. The (n, 2) unit-vector
        # form and the "nearest" method already preserve real samples and need
        # no special handling.
        if (
            self.headings.ndim == 1
            and self.interp == "linear"
            and self.times is not None
        ):
            components = np.column_stack(
                [np.cos(self.headings), np.sin(self.headings)]
            )
            aligned_components = _align_to_frame_times(
                components,
                self.times,
                frame_times,
                n_frames,
                self.interp,
                name="HeadDirectionOverlay",
            )
            # arctan2 of NaN components stays NaN, preserving extrapolation gaps.
            aligned_data = np.arctan2(
                aligned_components[:, 1], aligned_components[:, 0]
            )
        else:
            aligned_data = _align_to_frame_times(
                self.headings,
                self.times,
                frame_times,
                n_frames,
                self.interp,
                name="HeadDirectionOverlay",
            )
```

Notes:
- `_align_to_frame_times` with `times is None` requires `len(data) == n_frames`
  and returns the data unchanged (overlays.py:3766-3773); the `self.times is
  None` branch above falls through to the plain call, so no-timestamps behavior
  is unchanged.
- `np.arctan2(nan, nan)` is `nan`, so frames extrapolated to NaN by
  `_interp_linear` (overlays.py:3843) stay NaN — the downstream renderers
  (`_create_head_direction_quiver`, _parallel.py:1037-1040) already skip NaN
  angles. Behavior on gaps is preserved.
- The resulting angle lies in `(-pi, pi]`, which is what `np.cos`/`np.sin`
  downstream consume; no caller depends on the original `[0, 2*pi)` range.

### 3. NaN-safe data-range diagnostic in `_validate_bounds` (`animation/overlays.py`)

Aligned overlay data can contain NaN rows (extrapolated frames). The
warning's data-range line then prints `[nan, nan]`. Use the NaN-aware
reductions, and guard the all-NaN column case so the reduction does not warn or
emit `nan` itself.

Replace overlays.py:3388-3390:

```python
        # Calculate actual data ranges
        data_mins = data.min(axis=0)
        data_maxs = data.max(axis=0)
```

with:

```python
        # Calculate actual data ranges, ignoring NaN gaps from temporal
        # extrapolation (which would otherwise make every range read [nan, nan]).
        with warnings.catch_warnings():
            # An all-NaN column yields nan from nanmin/nanmax and a RuntimeWarning;
            # suppress it here (this is a diagnostic message, not a computation).
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data_mins = np.nanmin(data, axis=0)
            data_maxs = np.nanmax(data, axis=0)
```

(`warnings` is already imported at the top of `_validate_bounds` —
overlays.py:3371 — so it is in scope here.)

The outside-bounds count and percentage (overlays.py:3381-3385) are
intentionally left unchanged: NaN coordinates compare `False` against both
bounds, so they are correctly **not** counted as out-of-bounds.

### 4. Render events in the reuse-artists path (`animation/_parallel.py`)

`OverlayArtistManager` does not handle `overlay_data.events`, so the default
(`reuse_artists=True`) export path drops all event markers. Events are
frame-dependent (cumulative/instant/decay) and `_render_event_overlay_matplotlib`
already produces scatter artists per frame; mirror the head-direction quiver
pattern (recreate each frame) so the manager renders events on `initialize` and
`update_frame`, and tears them down on `clear`.

**4a. Make `_render_event_overlay_matplotlib` return its created artists.**
Today it returns `None` (`_parallel.py:135-211`). Capture each `ax.scatter`
return value so the manager can remove them next frame. Change the signature and
the scatter call, and append a `return`:

Change the signature (line 135) from:

```python
def _render_event_overlay_matplotlib(ax: Any, event_data: Any, frame_idx: int) -> None:
```

to:

```python
def _render_event_overlay_matplotlib(
    ax: Any, event_data: Any, frame_idx: int
) -> list[Any]:
```

Update the docstring Returns section to document the returned artist list, then
collect the scatters. Replace the render loop tail (lines 196-211):

```python
        # Render each event with its computed alpha
        base_rgba = to_rgba(color)

        for pos, alpha in zip(active_positions, alphas, strict=True):
            if np.any(np.isnan(pos)):
                continue
            ax.scatter(
                pos[0],
                pos[1],
                c=[(*base_rgba[:3], alpha)],
                s=event_data.size**2,
                marker=marker,
                zorder=104,
                edgecolors=event_data.border_color,
                linewidths=event_data.border_width,
            )
```

with:

```python
        # Render each event with its computed alpha
        base_rgba = to_rgba(color)

        for pos, alpha in zip(active_positions, alphas, strict=True):
            if np.any(np.isnan(pos)):
                continue
            scatter = ax.scatter(
                pos[0],
                pos[1],
                c=[(*base_rgba[:3], alpha)],
                s=event_data.size**2,
                marker=marker,
                zorder=104,
                edgecolors=event_data.border_color,
                linewidths=event_data.border_width,
            )
            artists.append(scatter)
```

and initialize `artists` before the outer loop (insert just before line 159's
`for event_name, positions in event_data.event_positions.items():`):

```python
    artists: list[Any] = []
```

then add `return artists` as the final statement of the function. The legacy
callers at `_parallel.py:1335-1336` ignore the return value, so they are
unaffected.

**4b. Add an event-artist list to `OverlayArtistManager`.** After the
`_head_direction_quivers` field (`_parallel.py:821`), add:

```python
    _event_artists: list[Any] = field(default_factory=list)
```

(Update the class Attributes docstring to list `_event_artists` alongside the
other artist lists.)

**4c. Render events on `initialize`.** At the end of
`OverlayArtistManager.initialize`, after the head-direction loop and before
`self._initialized = True` (`_parallel.py:863-865`), add:

```python
        # Initialize event overlays. Events change visibility per frame
        # (cumulative / instant / decay), so they are recreated each frame like
        # head-direction quivers rather than updated in place.
        for event_data in self.overlay_data.events:
            self._event_artists.extend(
                _render_event_overlay_matplotlib(self.ax, event_data, frame_idx)
            )
```

**4d. Re-render events on `update_frame`.** At the end of
`OverlayArtistManager.update_frame`, after the head-direction loop
(`_parallel.py:1103-1104`), add:

```python
        # Re-render event overlays. Remove last frame's event artists, then
        # redraw the events visible at this frame.
        for artist in self._event_artists:
            artist.remove()
        self._event_artists.clear()
        for event_data in self.overlay_data.events:
            self._event_artists.extend(
                _render_event_overlay_matplotlib(self.ax, event_data, frame_idx)
            )
```

(`update_frame` early-returns when `self.overlay_data is None` at
_parallel.py:1078-1079, so `self.overlay_data.events` is safe here.)

**4e. Tear down events in `clear`.** In `OverlayArtistManager.clear`, after the
head-direction quiver removal block (`_parallel.py:1240-1244`) and before the
region-patch block, add:

```python
        # Remove event artists
        for artist in self._event_artists:
            artist.remove()
        self._event_artists.clear()
```

### 5. Documentation

No README / CHANGELOG / QUICKSTART changes. These are bug fixes to existing
behavior; the only doc edits are the in-code docstring updates for
`_render_event_overlay_matplotlib` (Returns section, Task 4a) and the
`OverlayArtistManager` Attributes section (Task 4b). Do **not** touch `.claude/`
docs.

## Deliberately not in this phase

- **The removed-kwarg doc examples** `PositionOverlay(data=...)` and
  `BodypartOverlay(skeleton=[...])` in docstrings/examples that no longer match
  the current constructors → **phase 23** (overlay constructor/doc cleanup).
  This phase changes only overlay *behavior*, not constructor signatures or
  their examples. Do not "while I'm here" rewrite those examples.
- **The `field_to_rgb_for_napari` orientation regression test** (verifying the
  RGB image flip matches the coordinate transform) → **phase 25**. Task 1 fixes
  the *coordinate* transform dtype; it does not add or revisit the
  image-orientation test.
- **The napari backend's own overlay rendering** (the live viewer path) — out
  of scope. This phase fixes the matplotlib/video render path
  (`_parallel.py`), the shared coordinate transform, and the overlay data
  alignment. The napari layer-data pipeline is reviewed separately.
- **Generalizing circular interpolation into `_interp_linear`** — explicitly
  not done. `_interp_linear` stays linear for all overlays; circularity is
  handled only in `HeadDirectionOverlay.convert_to_data` (Task 2), which is the
  only producer of 1-D angle data.
- **Changing event visibility semantics** (cumulative / instant / decay) — Task
  4 makes the reuse path *render* events using the existing
  `_render_event_overlay_matplotlib` logic unchanged; it does not alter what
  events are visible on a given frame.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_transform_coords_integer_dtype_keeps_fraction` | `transform_coords_for_napari(np.array([[5, 5]], dtype=np.int64), env)` (env where a unit step in x is < 1 pixel apart, or grid finer than coord spacing) returns a float64 array whose values are **not** integer-truncated — assert `result.dtype == np.float64` and `result` differs from the int-truncated value for a coord that maps to a fractional pixel. **Fails before** (returns int64, truncated). |
| `test_transform_direction_integer_dtype_float_result` | `transform_direction_for_napari(np.array([[1, 1]], dtype=np.int64), env)` returns a float64 array with the correctly scaled `dr`/`dc` (non-integer when `x_scale`/`y_scale` are non-integer). **Fails before** (int dtype truncates the scaled components). |
| `test_transform_direction_fallback_integer_dtype` | `transform_direction_for_napari(np.array([[1, 1]], dtype=np.int64), None)` (fallback branch, no scale) returns float64 `[-1.0, 1.0]`. Guards the second `np.empty_like` site. |
| `test_head_direction_interp_crosses_pi_short_way` | `HeadDirectionOverlay(headings=np.array([3.0, -3.0]), times=np.array([0.0, 1.0]))` converted at `frame_times=[0.5]` yields an angle near `±pi` (short way: `\|angle\|` close to `pi`), **not** near `0`. **Fails before** (linear interp gives ~`0.0`). |
| `test_head_direction_interp_nowrap_unchanged` | A heading sequence with no wrap (e.g. `[0.1, 0.2]`) interpolated at the midpoint gives ~`0.15` — circular path agrees with linear when no wrap is crossed (no regression). |
| `test_head_direction_interp_nan_gap_preserved` | A `frame_times` value outside the source time range yields `NaN` (extrapolation gap preserved through the cos/sin → arctan2 path). |
| `test_head_direction_vector_form_unchanged` | A `(n, 2)` unit-vector heading overlay interpolates component-wise as before (circular branch not taken; behavior identical to pre-fix). |
| `test_validate_bounds_data_range_ignores_nan` | `_validate_bounds` on `(n, 2)` data with some NaN rows and >10% out-of-bounds points emits a warning whose message contains finite "Data ranges" numbers (no `nan` substring). **Fails before** (message contains `nan`). |
| `test_validate_bounds_all_nan_column_no_crash` | `_validate_bounds` on data with an entirely-NaN column does not raise and emits no `RuntimeWarning` (the all-NaN reduction is suppressed). |
| `test_event_overlay_renders_in_reuse_path` | An `OverlayArtistManager` built with an `EventData` (instant or cumulative mode), `initialize(0)` then `update_frame(k)` for a frame where an event is visible, leaves `len(manager._event_artists) > 0` and a scatter artist present on the axes at the event position. **Fails before** (`_event_artists` does not exist / no event artist created). |
| `test_event_overlay_cleared_between_frames` | After `update_frame` advances past a decay window (or from a frame with an event to one without in instant mode), the prior frame's event artists are removed (count drops; no stale artists accumulate). |
| `test_render_event_overlay_returns_artists` | `_render_event_overlay_matplotlib(ax, event_data, frame_idx)` returns a non-empty `list` of artists for a frame with visible events, and `[]` for a frame with none. **Fails before** (returns `None`). |
| `test_overlay_manager_clear_removes_events` | After rendering events, `manager.clear()` empties `_event_artists` and removes the artists from the axes. |
| `test_parallel_render_events_with_reuse[reuse_artists=True]` *(integration; requires ffmpeg/matplotlib Agg)* | Running the reuse-artists frame render with an `EventOverlay` present produces frames containing the event marker (compare a region of the rendered frame to a no-event baseline, or assert a non-background pixel at the event's transformed location). **Fails before** (event absent from reuse-path frames). |

Mark `test_parallel_render_events_with_reuse` and any test invoking the full
video pipeline `@pytest.mark.integration`; gate ffmpeg-dependent assertions
behind the suite's existing ffmpeg-availability skip (see
`tests/animation/conftest.py`). The transform, interpolation, `_validate_bounds`,
and `OverlayArtistManager` unit tests use the matplotlib **Agg** backend only
(no display, no ffmpeg) and run unconditionally.

## Fixtures

- **Transforms** (`tests/animation/test_transforms.py`): reuse the existing
  `env` / `simple_env` fixture there. Construct integer-dtyped coord/direction
  inputs inline (`np.array([[5, 5]], dtype=np.int64)`). No new fixture needed.
- **Head-direction interpolation** (`tests/animation/test_overlays.py` or
  `test_event_overlay.py` sibling): build `HeadDirectionOverlay` inline with a
  small `simple_env` (the 4-corner `Environment.from_samples` fixture already in
  `test_overlay_artist_manager.py`). Call `convert_to_data(frame_times, n_frames,
  env)` directly — no rendering needed.
- **`_validate_bounds`** (`tests/animation/test_overlays.py`): synthesize a small
  `(n, 2)` array with injected NaN rows and a fraction of points outside a tiny
  `dim_ranges` to trip the threshold; assert on `pytest.warns(UserWarning)`
  message text.
- **`OverlayArtistManager` + events** (`tests/animation/test_overlay_artist_manager.py`):
  reuse the module's `simple_env` fixture and import `EventData`/`OverlayData`
  from `neurospatial.animation.overlays`. Build an `EventData` directly with a
  single event type, explicit `event_positions`/`event_frame_indices`,
  `decay_frames=0` (instant) or `None` (cumulative). Use a real
  `matplotlib.pyplot.subplots()` axes (Agg backend) — these tests already
  construct real axes for artist-count assertions.
- **Parallel render integration**: reuse the existing video-render fixtures /
  `tmp_path` pattern from `tests/animation/test_video_*`; add an `EventOverlay`
  to the overlays list and assert on the rendered frame. Gate on ffmpeg.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:

- Every task is implemented as specified: float64 allocation at **all three**
  `np.empty_like` sites in `transforms.py`; circular interpolation **only** in
  `HeadDirectionOverlay.convert_to_data` (not in shared `_interp_linear`);
  `np.nanmin`/`np.nanmax` with the all-NaN guard in `_validate_bounds`; events
  rendered+cleared in `OverlayArtistManager` and `_render_event_overlay_matplotlib`
  returning its artist list.
- The "Deliberately not in this phase" list is honored — no overlay-constructor
  or doc-example edits (phase 23), no `field_to_rgb_for_napari` orientation test
  (phase 25), no change to `_interp_linear` semantics, no change to event
  visibility rules.
- The scope is strictly `src/neurospatial/animation/` — no edits outside that
  package.
- Validation-slice tests pass. Tests that exercise the full video pipeline /
  ffmpeg are marked `@pytest.mark.integration` and gated behind the suite's
  ffmpeg/napari availability skips; the transform, interpolation,
  `_validate_bounds`, and manager unit tests run headless (Agg) and
  unconditionally.
- The fail-before/pass-after tests genuinely fail on the pre-fix code
  (spot-check by stashing each fix): integer-dtype coords truncate, the ±π
  crossing interpolates to ~0, `_validate_bounds` prints `nan`, and reuse-path
  frames lack event markers. Tests are not tautologies; shared setup lives in
  fixtures.
- Docstrings, test names, and module names do not reference this plan or its
  phase number.
- `uv run pytest tests/animation -q`, `uv run ruff check . && uv run ruff
  format .`, and `uv run mypy src/neurospatial/animation/` all pass. (The new
  `list[Any]` return annotation on `_render_event_overlay_matplotlib` and the
  `_event_artists` field must type-check cleanly.)
