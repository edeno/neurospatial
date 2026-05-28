# Phase 6 — NWB disk round-trips

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Add `NWBHDF5IO` write→close→reopen tests for every NWB writer in the codebase except `Environment` (which already has one — see [tests/nwb/test_environment.py:657-749](../../../tests/nwb/test_environment.py) as the template). Currently `write_laps`, `write_trials`, `write_region_crossings`, `write_events`, `write_fields` are tested only in-memory against the live `NWBFile` object — a pynwb major-version bump or ndx-events serialization regression would slip through silently.

**Inputs to read first:**

- [tests/nwb/test_environment.py:657-749](../../../tests/nwb/test_environment.py) — the canonical example. Read this fully; mirror its structure (`tmp_path` fixture, `NWBHDF5IO("w")` write, close, `NWBHDF5IO("r")` reopen, assert attributes).
- [tests/nwb/conftest.py:71-361](../../../tests/nwb/conftest.py) — existing fixtures producing real `pynwb` and `ndx-events` containers. Reuse rather than re-build.
- [src/neurospatial/io/nwb/](../../../src/neurospatial/io/nwb/) — list of NWB writer modules. Expected: `_events.py`, `_fields.py`, `_environment.py`. The reader-only modules `_behavior.py` and `_pose.py` are not in scope.
- [tests/nwb/test_events.py:425-591](../../../tests/nwb/test_events.py) — existing in-memory tests for `write_laps`, `write_trials`, `write_region_crossings`, `write_events`. Each currently round-trips through the live NWBFile object only.
- [tests/nwb/test_fields.py:17-446](../../../tests/nwb/test_fields.py) — existing in-memory tests for `write_fields`.

## Tasks

### 1. Disk round-trip for `write_laps`

In [tests/nwb/test_events.py](../../../tests/nwb/test_events.py), add `TestWriteLapsDiskRoundTrip` (place near the existing `TestWriteLaps` class):

- `test_laps_survive_disk_roundtrip`: build a lap dataframe with at least 3 laps (start/end times, lap number, direction). Call `write_laps(nwbfile, laps_df)`. Write to `tmp_path / "laps.nwb"` with `NWBHDF5IO("w")`; close; reopen with `NWBHDF5IO("r")`; call the corresponding reader (`read_laps` if present, else access the `TimeIntervals` table directly). Assert that the recovered dataframe equals the input element-wise (`pd.testing.assert_frame_equal`).
- `test_laps_metadata_survives_disk_roundtrip`: extra metadata columns (e.g., `direction`, `lap_number`) preserved.

### 2. Disk round-trip for `write_trials`

Same pattern in `tests/nwb/test_events.py`: `TestWriteTrialsDiskRoundTrip`:
- `test_trials_survive_disk_roundtrip`: trial start/stop times, trial outcome, condition labels — write to disk, reopen, compare.
- `test_trials_with_optional_columns_survives`: only some optional metadata columns populated; missing columns reload as expected (NaN or absent).

### 3. Disk round-trip for `write_region_crossings`

`TestWriteRegionCrossingsDiskRoundTrip`:
- `test_region_crossings_survive_disk_roundtrip`: build a region-crossing dataframe (timestamp, region name, enter/exit). Write to disk, reopen, compare frame equality.

### 4. Disk round-trip for `write_events`

`TestWriteEventsDiskRoundTrip`:
- `test_events_with_ndx_events_survive_disk_roundtrip`: gated on `pytest.importorskip("ndx_events")`. Build a small event series, write to disk via the `ndx-events` extension, reopen, assert event timestamps and labels survive.
- `test_events_without_ndx_events_uses_fallback`: if the codebase has a fallback path when `ndx-events` is unavailable, exercise it. Read the source to determine if such a path exists; if not, skip this test.

### 5. Disk round-trip for `write_fields`

In [tests/nwb/test_fields.py](../../../tests/nwb/test_fields.py), add `TestWriteFieldsDiskRoundTrip`:
- `test_fields_survive_disk_roundtrip`: build a small `SpatialRateResult` (via `compute_spatial_rate` on synthetic data, see Phase 2 fixtures). Call `write_fields(nwbfile, result)`. Write to disk, reopen, read back, assert `np.allclose(reopened.firing_rate, original.firing_rate, atol=1e-10)` and metadata preserved (bin centers, occupancy).
- `test_multiple_fields_in_one_file`: write fields for 5 cells, reopen, assert all 5 survive in order.

### 6. Environment polygon round-trip on disk

In [tests/nwb/test_environment.py](../../../tests/nwb/test_environment.py), add `test_environment_with_polygon_regions_survives_disk_roundtrip` (placement: in the existing `TestRoundtrip` class around line 700):

- Build an `Environment` with at least one `ShapelyPolygon` region (use `env.regions.add('arm', point=..., polygon=shapely.Polygon([...]))`). Write to disk, reopen, assert the polygon geometry is preserved (`shapely.equals(reopened.regions['arm'].polygon, original.regions['arm'].polygon)`).
- Audit flagged this as a likely silent-loss spot: Shapely-to-HDF5 serialization is non-trivial and not currently exercised on disk.

### 7. Verify all new round-trip tests pass on a pynwb version bump

Mark all 6 new round-trip test classes with `@pytest.mark.slow` (each writes/reads disk). Add a CI note in the PR description: this PR significantly improves resilience to pynwb / ndx-events version bumps. If the codebase has a pinned pynwb version in `pyproject.toml`, run the new tests against `pynwb>=2.0` (latest) and the pinned version; report any divergence.

## Deliberately not in this phase

- **No new NWB writers.** Phase 6 is purely additive test coverage for existing writers.
- **No removal of in-memory tests.** Existing in-memory round-trip tests (cited in audit as `test_events.py:575-591`, etc.) stay alongside the new disk round-trips; in-memory is faster for everyday CI.
- **No fix for `MockTimeSeries`/`MockEventsTable`.** Phase 8 handles mock removal.
- **No `monkeypatch(builtins.__import__)` test rewrite.** Phase 8.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/nwb/test_events.py::TestWriteLapsDiskRoundTrip::test_laps_survive_disk_roundtrip` | Lap dataframe survives `NWBHDF5IO` write→close→reopen. |
| `tests/nwb/test_events.py::TestWriteTrialsDiskRoundTrip::test_trials_survive_disk_roundtrip` | Trial table survives disk round-trip with all metadata columns. |
| `tests/nwb/test_events.py::TestWriteRegionCrossingsDiskRoundTrip::test_region_crossings_survive_disk_roundtrip` | Region-crossing events survive disk round-trip. |
| `tests/nwb/test_events.py::TestWriteEventsDiskRoundTrip::test_events_with_ndx_events_survive_disk_roundtrip` | Events via ndx-events survive disk round-trip. **`importorskip("ndx_events")`.** |
| `tests/nwb/test_fields.py::TestWriteFieldsDiskRoundTrip` (2 tests) | Firing-rate fields and metadata survive disk round-trip. |
| `tests/nwb/test_environment.py::test_environment_with_polygon_regions_survives_disk_roundtrip` | Shapely polygon geometry survives disk round-trip. |

All marked `@pytest.mark.slow`.

## Fixtures

In `tests/nwb/conftest.py` (extend existing):
- `simple_laps_dataframe`: 3-lap dataframe with start_time, stop_time, lap_number, direction.
- `simple_trials_dataframe`: 5-trial table with start_time, stop_time, outcome, condition.
- `simple_region_crossings_dataframe`: 8 enter/exit events across 2 regions.
- `nwbfile_with_subject`: helper that returns a minimal `pynwb.NWBFile` with required fields populated. Reuse if already present in `conftest.py`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into Phase 8 (mock removal).
- Validation slice tests pass; all marked `@pytest.mark.slow`.
- Tests aren't trivial — each round-trip test uses `pd.testing.assert_frame_equal` or `np.allclose` against the input data, not just "doesn't raise". (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (none).
- User-facing documentation listed as tasks is updated, not deferred (none in this phase).
