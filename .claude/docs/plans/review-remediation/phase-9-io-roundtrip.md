# Phase 9 — IO: round-trip fidelity & validation

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

This phase hardens the serialization surface in `src/neurospatial/io/` so that
**every** layout factory survives `to_file()`/`from_file()`, polar environments
survive NWB write/read without flipping to Cartesian, and the NWB readers reject
(rather than silently propagate) malformed time axes, mismatched array lengths,
and convention-violating head-direction data.

**Inputs to read first:**

- [src/neurospatial/io/files.py:212-256](../../../../src/neurospatial/io/files.py) — `to_file()` builds the metadata dict and serializes `env.layout_parameters` **verbatim** at line 222, then JSON-encodes it. `_convert_arrays_to_lists` (lines 44-69) only handles arrays/dicts/lists/tuples/np-scalars and passes everything else through unchanged (`else: return obj`), so a `networkx.Graph` (Graph layout) or `shapely.Polygon` (Polygon layout) reaches `json.dump` and raises `TypeError: Object of type Graph is not JSON serializable`. **Verified reproducible** for both `Environment.from_graph(...)` and `Environment.from_polygon(...)`.
- [src/neurospatial/io/files.py:387-434](../../../../src/neurospatial/io/files.py) — the `from_file()` read path. Note that geometry is restored from the npz (`bin_centers`, `active_mask`, `grid_edges_*`) and the node-link `graph`, and that `bin_centers`/`connectivity`/`grid_*` are **overridden after** `from_layout(...)` (lines 410-418). So `layout_parameters` is effectively introspection metadata on the round-trip, not load-bearing geometry — it only needs to be JSON-safe and to round-trip back to an equivalent dict.
- [src/neurospatial/io/files.py:439-487](../../../../src/neurospatial/io/files.py) — `to_dict()` has the **same** `layout_parameters` verbatim serialization (line 485). Must get the same JSON-safe treatment so the in-memory handoff path doesn't crash either.
- [src/neurospatial/io/nwb/_environment.py:228-243](../../../../src/neurospatial/io/nwb/_environment.py) — `write_environment` builds the metadata JSON. `coordinate_kind` is **absent** from this dict, so it is never persisted.
- [src/neurospatial/io/nwb/_environment.py:517-591](../../../../src/neurospatial/io/nwb/_environment.py) — `read_environment` parses metadata and reconstructs, but **never reads or restores `coordinate_kind`**, so a polar environment comes back with the field default `"cartesian"`. The `EnvironmentMetadata` TypedDict (lines 33-50) also lacks the field.
- [src/neurospatial/io/nwb/_environment.py:475-482](../../../../src/neurospatial/io/nwb/_environment.py) — docstring claims Hexagonal is "fully reconstructed from stored grid_edges and active_mask." It is **not**: `_extract_grid_data` returns `None` when `len(grid_edges) == 0` (lines 388-393, comment explicitly names "Hexagonal and TriangularMesh have empty grid_edges tuple"), so Hexagonal has `has_grid_data=False` and goes through the `_ReconstructedLayout` KDTree fallback (lines 692-701). The docstring overstates fidelity.
- [src/neurospatial/io/nwb/_adapters.py:57-70](../../../../src/neurospatial/io/nwb/_adapters.py) — `timestamps_from_series` computes `np.arange(n) / rate + starting_time` with no guard. `rate <= 0` produces Inf/`-Inf`/NaN timestamps; a non-finite `rate` propagates silently.
- [src/neurospatial/io/nwb/_behavior.py:84-88](../../../../src/neurospatial/io/nwb/_behavior.py) — `read_position` returns `(positions, timestamps)` with no length-agreement check.
- [src/neurospatial/io/nwb/_behavior.py:242-301](../../../../src/neurospatial/io/nwb/_behavior.py) — `read_head_direction`: line 298 does `np.asarray(data).ravel()` unconditionally (corrupts a `(n, 2)` heading-vector series into a length-`2n` 1-D array); it ignores the series `unit` attribute (deg vs rad); it does not document or enforce the allocentric 0=East convention; and (line 297-301) returns angles+timestamps with no length check.
- [src/neurospatial/io/nwb/_pose.py:82-96](../../../../src/neurospatial/io/nwb/_pose.py) — `read_pose` reads each bodypart's `data` and takes timestamps from the first series only; per-bodypart lengths are never checked against the shared `timestamps`.
- [tests/nwb/conftest.py](../../../../tests/nwb/conftest.py) — existing NWB fixtures (`empty_nwb`, `sample_environment`, `sample_nwb_with_position`, `sample_nwb_with_head_direction`, `sample_nwb_with_pose`) and the `pytest.importorskip("pynwb")` / `pytest.mark.integration` pattern this phase's tests must follow.
- [tests/io_tests/test_io.py:32-316](../../../../tests/io_tests/test_io.py) — existing file round-trip tests cover only `simple_env` (RegularGrid). Graph/Polygon/Hexagonal/etc. are uncovered — this is why the crash shipped.

**Contracts referenced:**

- [Input-validation helpers](shared-contracts.md#input-validation-helpers) — this phase consumes `validate_lengths` from `src/neurospatial/_validation.py` for all NWB read length checks. `validate_lengths` compares lengths **only** (length-1 is a mismatch, not a broadcast) and raises naming the offending arguments and their lengths. Do not weaken. If `_validation.py` already exists when this PR is cut (created by an earlier phase — phase 4 is the scheduled first creator), import from it; do **not** redefine it.

## Tasks

### 1. Make `layout_parameters` JSON-safe in `to_file()` and `to_dict()` (`io/files.py`)

`env.layout_parameters` can contain non-JSON-serializable objects: a
`networkx.Graph` (Graph layout, key `graph_definition`) and a `shapely`
geometry (Polygon layout, key `polygon`). Add a dedicated serializer that
converts these to JSON-safe forms, and route `layout_parameters` through it in
**both** write paths.

Add this helper alongside `_convert_arrays_to_lists` in `io/files.py`:

```python
def _jsonsafe_layout_parameters(params: dict[str, Any] | None) -> dict[str, Any]:
    """Return a JSON-serializable copy of layout-engine parameters.

    ``env.layout_parameters`` may contain objects that ``json`` cannot encode:
    a ``networkx.Graph`` (Graph layout) or a Shapely geometry (Polygon layout).
    These are converted to portable, self-describing forms so the metadata JSON
    round-trips. The geometry itself is restored on read from the stored
    ``bin_centers``/``active_mask``/``grid_edges`` arrays and the node-link
    ``graph`` (see ``from_file``); these serialized parameters are introspection
    metadata, not the source of truth for reconstruction.

    Parameters
    ----------
    params : dict or None
        The ``env.layout_parameters`` dict (or None).

    Returns
    -------
    dict
        A new dict with every value JSON-serializable.
    """
    if not params:
        return {}

    def _encode(value: Any) -> Any:
        # networkx graph -> node-link dict (same form used for connectivity)
        if isinstance(value, nx.Graph):
            return {"__nx_graph__": nx.node_link_data(value, edges="links")}
        # shapely geometry -> WKT string (round-trippable, human-readable)
        wkt = getattr(value, "wkt", None)
        geom_type = getattr(value, "geom_type", None)
        if wkt is not None and geom_type is not None:
            return {"__shapely_wkt__": wkt}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, dict):
            return {k: _encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_encode(v) for v in value]
        return value

    return {k: _encode(v) for k, v in params.items()}
```

Then in `to_file()` change line 222 from:

```python
        "layout_parameters": env.layout_parameters,
```

to:

```python
        "layout_parameters": _jsonsafe_layout_parameters(env.layout_parameters),
```

and make the identical change at the `to_dict()` metadata dict (line 485).

Because `from_file()` overrides `bin_centers`, `connectivity`, `grid_edges`,
`grid_shape`, and `active_mask` after `from_layout(...)` (files.py:410-418), the
decoded sentinel dicts (`{"__nx_graph__": ...}`, `{"__shapely_wkt__": ...}`)
that survive into `layout_params` do not break reconstruction for the layouts in
scope. To keep `from_file()` robust, decode the sentinels back so round-tripped
`layout_parameters` is semantically equivalent. After
`layout_params = _convert_lists_to_arrays(layout_params)` (files.py:407), add:

```python
    # Decode any JSON-safe sentinels written by _jsonsafe_layout_parameters so
    # round-tripped layout_parameters is equivalent to the original.
    layout_params = _decode_layout_parameters(layout_params)
```

and define the decoder next to the encoder:

```python
def _decode_layout_parameters(params: Any) -> Any:
    """Inverse of ``_jsonsafe_layout_parameters`` for the sentinel dicts."""
    if isinstance(params, dict):
        if set(params) == {"__nx_graph__"}:
            return nx.node_link_graph(params["__nx_graph__"], edges="links")
        if set(params) == {"__shapely_wkt__"}:
            try:
                from shapely import wkt as _wkt

                return _wkt.loads(params["__shapely_wkt__"])
            except ImportError:
                # shapely absent at load time: leave the WKT string in place
                # rather than fail; geometry is reconstructed from arrays anyway.
                return params["__shapely_wkt__"]
        return {k: _decode_layout_parameters(v) for k, v in params.items()}
    if isinstance(params, list):
        return [_decode_layout_parameters(v) for v in params]
    return params
```

Note that `_convert_lists_to_arrays` leaves string-only dicts/lists untouched
(it only converts numeric lists; files.py:86-99), so the sentinel string values
survive to the decoder intact.

### 2. Persist and restore `coordinate_kind` across the NWB round-trip (`io/nwb/_environment.py`)

- Add `coordinate_kind: str` to the `EnvironmentMetadata` TypedDict (after `is_linearized_track`, line 49).
- In `write_environment`, add the field to the metadata dict (lines 230-243). Read it defensively so non-polar/older envs are unaffected:

```python
    metadata = json.dumps(
        {
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "name": env.name,
            "units": units,
            "frame": frame,
            "n_dims": n_dims,
            "layout_type": layout_type,
            "n_bins": env.n_bins,
            "n_edges": len(edges),
            "is_linearized_track": env.is_linearized_track,
            "has_grid_data": grid_data is not None,
            "coordinate_kind": getattr(env, "coordinate_kind", "cartesian"),
        }
    )
```

- In `read_environment`, after the env is reconstructed and `units`/`frame` are restored (around lines 576-582), restore `coordinate_kind`. A missing key (older files) falls back to the field default `"cartesian"`:

```python
    # Restore coordinate_kind. Older NWB files (written before this field was
    # persisted) lack the key and fall back to the env field default
    # "cartesian", matching the on-disk behavior of those files.
    coordinate_kind = metadata.get("coordinate_kind", "cartesian")
    if coordinate_kind != "cartesian":
        env.coordinate_kind = coordinate_kind
```

(`env.coordinate_kind` is a real field on `Environment` — `environment/core.py:257`, `Literal["cartesian", "polar"]`. `from_polar_egocentric` sets it to `"polar"` — `factories.py:992`.)

### 3. Correct the Hexagonal fidelity docstring (`io/nwb/_environment.py:475-482`)

The Notes section overstates fidelity. Move Hexagonal into the non-grid /
KDTree-fallback group, since it has empty `grid_edges` and `_extract_grid_data`
returns `None` for it:

```python
    Notes
    -----
    For grid-based layouts (RegularGrid, MaskedGrid, ImageMask, ShapelyPolygon),
    the layout is fully reconstructed from stored grid_edges and active_mask,
    enabling proper point_to_bin_index functionality.

    For layouts without a rectangular grid (Graph, Hexagonal, TriangularMesh),
    a KDTree-based layout is used. This provides nearest-neighbor point mapping
    over the stored bin centers and connectivity, but does not reconstruct the
    original layout engine's bin geometry exactly.
```

### 4. Guard against non-finite / non-positive rate (`io/nwb/_adapters.py:57-70`)

Replace the unconditional `rate = float(rate)` / `arange / rate` block with a
finite-and-positive guard. This protects every reader that derives timestamps
from a rate (`read_position`, `read_head_direction`, `read_pose`).

```python
    rate = getattr(series, "rate", None)
    if rate is None:
        raise ValueError(
            "Series has neither 'timestamps' nor 'rate' attribute; cannot derive "
            "time axis."
        )

    rate = float(rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(
            f"Series 'rate' must be a finite positive value to derive a time "
            f"axis, got {rate!r}. Provide an explicit 'timestamps' array instead."
        )

    n_samples = len(series.data)
    starting_time = float(
        getattr(series, "starting_time", DEFAULT_STARTING_TIME) or DEFAULT_STARTING_TIME
    )
    timestamps = np.arange(n_samples, dtype=np.float64) / rate + starting_time
    return np.asarray(timestamps, dtype=np.float64)
```

Add a `ValueError` entry to the Raises section of the docstring noting the
finite-positive-rate requirement.

### 5. Length-agreement checks on the NWB readers (`io/nwb/_behavior.py`, `io/nwb/_pose.py`)

Import the shared helper and assert data/timestamp length agreement.

In `read_position` (after line 86, before `return`):

```python
    from neurospatial._validation import validate_lengths

    validate_lengths({"positions": positions, "timestamps": timestamps})
    return positions, timestamps
```

In `read_pose`, after the loop and the `timestamps is None` guard (after line
91), check **every** bodypart against the shared timestamps:

```python
    from neurospatial._validation import validate_lengths

    for bp_name, bp_data in bodyparts.items():
        validate_lengths({f"bodypart[{bp_name}]": bp_data, "timestamps": timestamps})
```

(`validate_lengths` uses `len(...)`, which for a 2-D `(n_samples, n_dims)` array
is `n_samples` — the intended axis. Do not flatten.)

### 6. Head-direction: unit handling, `(n, 2)` vectors, convention, length check (`io/nwb/_behavior.py:242-301`)

Rewrite the extraction block (lines 297-301) so it:

1. Reads the series `unit` attribute and converts degrees → radians when the
   unit string indicates degrees.
2. Handles a `(n, 2)` heading-vector series with `arctan2` instead of an
   unconditional `.ravel()` (which would corrupt it).
3. Documents the allocentric 0=East, counterclockwise-positive convention in the
   Returns section.
4. Validates length agreement after extraction.

Replace lines 297-301 with:

```python
    # Extract angle data and timestamps. CompassDirection SpatialSeries store
    # head direction as either a 1-D angle series or an (n, 2) unit-vector
    # series; handle both. Angles are returned in the allocentric convention
    # (0 = East, increasing counterclockwise).
    raw = np.asarray(spatial_series.data[:], dtype=np.float64)
    timestamps = _get_timestamps(spatial_series)

    if raw.ndim == 2 and raw.shape[1] == 2:
        # (n, 2) heading vectors -> angle via arctan2 (already radians, 0=East).
        angles = np.arctan2(raw[:, 1], raw[:, 0])
    else:
        angles = raw.reshape(-1)
        # Honor the stored unit: convert degrees to radians. NWB SpatialSeries
        # carry a free-text 'unit'; treat anything starting with "deg" as
        # degrees (e.g. "deg", "degree", "degrees").
        unit = getattr(spatial_series, "unit", None)
        if isinstance(unit, str) and unit.strip().lower().startswith("deg"):
            angles = np.deg2rad(angles)

    from neurospatial._validation import validate_lengths

    validate_lengths({"angles": angles, "timestamps": timestamps})

    return angles, timestamps
```

Update the `read_head_direction` Returns section docstring (lines 262-267) to
state the convention explicitly:

```python
    Returns
    -------
    angles : NDArray[np.float64], shape (n_samples,)
        Head direction angles in radians, allocentric convention
        (0 = East, increasing counterclockwise; range as stored, typically
        (-pi, pi] from arctan2 or [0, 2*pi) from a wrapped angle series).
        If the source SpatialSeries reports a degree unit, values are
        converted to radians on read.
    timestamps : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
```

### 7. Documentation

The above docstring edits (tasks 3, 4, 6) are the only user-facing doc changes.
No README/CHANGELOG/QUICKSTART updates are required: these are bug fixes to
existing public functions, not new API. Do **not** touch `.claude/` docs.

## Deliberately not in this phase

- **New `read_units` / `read_spikes` readers** — phase 15. This phase adds no new readers; it only hardens the existing `read_position` / `read_head_direction` / `read_pose` / `read_environment` / `timestamps_from_series` surface.
- **Redesigning the polar-environment representation** — phase 19. This phase only *preserves the existing `coordinate_kind` flag* across the NWB round-trip; it does not change what "polar" means, add new polar geometry, or alter `from_polar_egocentric`.
- **True (non-KDTree) Hexagonal / TriangularMesh / Graph reconstruction from NWB** — out of scope. Task 3 only *corrects the docstring* to match current behavior; it does not implement exact layout reconstruction. (The on-disk `to_file`/`from_file` path already round-trips these via stored arrays + node-link graph; the NWB path's KDTree fallback is a known, documented limitation.)
- **Changing `_convert_arrays_to_lists` / `_convert_lists_to_arrays` semantics** — task 1 adds a *new* `layout_parameters`-specific serializer and leaves the generic array converters untouched (the string-list handling at files.py:86-99 is relied upon, not modified).
- **Validation in the NWB *write* path** — this phase validates on read. Write-side guards are not in scope.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_to_file_roundtrip_all_layouts[Graph]` | `from_graph(...)` env survives `to_file`→`from_file` (no `TypeError`); `n_bins`, `n_dims`, `bin_centers`, `is_linearized_track`, edge set preserved. **Fails before** (raises `TypeError: Object of type Graph is not JSON serializable`). |
| `test_to_file_roundtrip_all_layouts[Polygon]` | `from_polygon(...)` env survives round-trip; bin_centers + active_mask preserved. **Fails before** (`TypeError: ... Polygon ...`). |
| `test_to_file_roundtrip_all_layouts[Hexagonal]` | `from_samples(..., layout="hexagonal")` (or equivalent) env survives round-trip. (Parametrized.) |
| `test_to_file_roundtrip_all_layouts[Masked]` | `from_grid_mask(...)` env survives round-trip. |
| `test_to_file_roundtrip_all_layouts[ImageMask]` | `from_pixel_mask(...)` env survives round-trip. |
| `test_to_file_roundtrip_all_layouts[3D]` | A 3-D `from_samples(...)` env survives round-trip (`n_dims == 3`). |
| `test_to_dict_roundtrip_graph_layout` | `to_dict(graph_env)` is `json.dumps`-able and `from_dict` reconstructs equivalent env. **Fails before** (same `TypeError`). |
| `test_to_file_graph_layout_parameters_roundtrip` | After round-trip, `loaded.layout_parameters` decodes the `graph_definition` back to an `nx.Graph` (not a sentinel dict). |
| `test_nwb_roundtrip_preserves_coordinate_kind` *(integration; importorskip pynwb)* | `from_polar_egocentric(...)` env → `write_environment` → `read_environment` yields `loaded.coordinate_kind == "polar"` and `loaded.is_linearized_track` preserved. **Fails before** (loads as `"cartesian"`). |
| `test_nwb_roundtrip_cartesian_unaffected` *(integration; importorskip pynwb)* | A Cartesian env round-trips with `coordinate_kind == "cartesian"` (no regression; field absent → default). |
| `test_read_environment_old_file_without_coordinate_kind` *(integration; importorskip pynwb)* | A written env with `coordinate_kind` manually stripped from the stored metadata JSON still reads, defaulting to `"cartesian"` (backward compat). |
| `test_timestamps_from_series_rejects_zero_rate` | A series stub with `rate=0.0`, no `timestamps`, raises `ValueError` matching "finite positive". **Fails before** (returns Inf timestamps). |
| `test_timestamps_from_series_rejects_negative_rate` | `rate=-30.0` raises `ValueError`. |
| `test_timestamps_from_series_rejects_nonfinite_rate` | `rate=float("inf")` and `rate=float("nan")` each raise `ValueError`. **Fails before** (returns NaN/Inf). |
| `test_timestamps_from_series_positive_rate_ok` | `rate=30.0` returns the expected `arange(n)/30` axis (no regression). |
| `test_read_position_length_check` *(integration; importorskip pynwb)* | A position SpatialSeries whose `data` length ≠ `timestamps` length raises `ValueError` "Length mismatch". |
| `test_read_pose_length_check` *(integration; importorskip pynwb)* | A pose container with one bodypart series shorter than the shared timestamps raises `ValueError`. |
| `test_read_head_direction_degrees_to_radians` *(integration; importorskip pynwb)* | A CompassDirection SpatialSeries with `unit="degrees"` and data `[0, 90, 180]` reads back `[0, pi/2, pi]` (within tolerance). **Fails before** (returns raw degrees). |
| `test_read_head_direction_vector_form` *(integration; importorskip pynwb)* | An `(n, 2)` unit-vector series reads back `n` angles via `arctan2` (length `n`, not `2n`). **Fails before** (`.ravel()` yields length `2n`). |
| `test_read_head_direction_radians_passthrough` *(integration; importorskip pynwb)* | A `unit="radians"` 1-D series is returned unchanged (no regression; covered by `sample_nwb_with_head_direction`). |
| `test_read_head_direction_length_check` *(integration; importorskip pynwb)* | data/timestamps length mismatch raises `ValueError`. |

Tests touching `write_environment`/`read_environment`/`read_position`/
`read_head_direction`/`read_pose` (the `pynwb`/`ndx-pose` surface) are
`pytest.mark.integration` and gated with `pytest.importorskip("pynwb")` (and
`"ndx_pose"` for pose), matching `tests/nwb/conftest.py`. The pure-file
`to_file`/`to_dict` and `timestamps_from_series` tests have **no** pynwb
dependency and run unconditionally.

## Fixtures

- **File round-trip (`tests/io_tests/`)**: add a `parametrize`-driven factory map producing one env per layout (Graph, Polygon, Hexagonal, Masked, ImageMask, 3D). Synthesize inline with seeded RNG (`np.random.default_rng(0)`) — no checked-in data. For Polygon, guard the parameter with `pytest.importorskip("shapely")`; for ImageMask, a small synthetic boolean image. Reuse `tmp_path` (already used throughout `test_io.py`).
- **NWB coordinate_kind (`tests/nwb/`)**: add a `sample_polar_environment` fixture (`Environment.from_polar_egocentric(...)`) to `tests/nwb/conftest.py`, mirroring the existing `sample_environment`. Reuse `empty_nwb`.
- **`timestamps_from_series` (`tests/nwb/test_adapters.py`)**: a tiny stub object (or `types.SimpleNamespace`) exposing `data`, `rate`, `starting_time`, and `timestamps=None` — no pynwb needed, so these tests are not gated.
- **Head-direction unit / vector form**: extend `tests/nwb/conftest.py` with `sample_nwb_with_head_direction_degrees` (`unit="degrees"`, data in degrees) and `sample_nwb_with_head_direction_vectors` (`(n, 2)` unit vectors). Model on the existing `sample_nwb_with_head_direction` fixture.
- **Length-mismatch fixtures**: build the mismatched series in-test from `empty_nwb` (a SpatialSeries with `data` length ≠ explicit `timestamps` length).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:

- Every task is implemented as specified; the round-trip works for **all**
  layout factories in the validation slice (not just RegularGrid).
- The "Deliberately not in this phase" list is honored — no new readers
  (`read_units`/`read_spikes`), no polar-representation redesign, no exact
  Hexagonal reconstruction, no changes to the generic `_convert_*` helpers.
- `_validation.py` is **imported**, not re-implemented; `validate_lengths`
  semantics are not weakened (length-1 is a mismatch, not a broadcast).
- Validation-slice tests pass. NWB-dependent tests are marked
  `pytest.mark.integration` and gated with `pytest.importorskip` so the suite
  passes whether or not `pynwb`/`ndx-pose`/`shapely` are installed.
- The fail-before/pass-after tests genuinely fail on the pre-fix code (spot-check
  by stashing the fix) — they are not tautologies. Shared setup lives in
  `conftest.py` fixtures, not copy-pasted.
- The Hexagonal docstring now matches actual behavior (KDTree fallback), and the
  `read_head_direction` Returns docstring states the 0=East convention and the
  degree→radian conversion.
- Docstrings, test names, and module names do not reference this plan or its
  phase number.
- `uv run pytest tests/io_tests tests/nwb -q`, `uv run ruff check . && uv run
  ruff format .`, and `uv run mypy src/neurospatial/io/` all pass.
