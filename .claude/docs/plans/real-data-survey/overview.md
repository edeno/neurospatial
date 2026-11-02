# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

The harness lives outside `src/` and only *calls* the public API. No phase
changes `src/neurospatial/`; fixes the gap log motivates ship as their own PRs
after phase 4.

- `data/load_bandit_data.py:1-60` — existing download-and-verify pattern (SHA-256,
  temp file, refuse to overwrite a mismatched file). `datasets/_fetch.py`
  generalizes it; `data/load_bandit_data.py` itself is left alone (the J16
  tutorial depends on it).
- `.gitignore` — `data/` is ignored except `README.md` and
  `load_bandit_data.py`, so the default cache `data/datasets/` is already ignored.
  `docs/research/` (survey notes and report) is ignored too.
- `pyproject.toml:131` — `testpaths = ["tests"]`: harness tests under
  `datasets/tests/` are not collected by the default suite; run them explicitly.
- `pyproject.toml:175-232` — ruff runs on the whole repo and selects `T20`
  (print). Add a per-file ignore `"datasets/**/*.py" = ["T201"]` so the
  standalone loader scripts can print summaries.
- `pyproject.toml:446` — `[dependency-groups]`: add a non-default `datasets`
  group (see Dependency policy).
- Public API called (signatures verified in `tests/data/public_api.txt` and source):
  - `src/neurospatial/environment/core.py:3288` `Environment.from_samples(positions, *, bin_size, units=, frame=, dimension_ranges=, infer_active_bins=, ...)`
  - `src/neurospatial/environment/core.py:3944` `Environment.from_graph(graph, edge_order, *, bin_size, ...)`
  - `src/neurospatial/environment/core.py:1903` `Environment.occupancy(times, positions, *, speed=, min_speed=, ...)`
  - `src/neurospatial/encoding/spatial.py:4129` `compute_spatial_rates(env, spike_times, times, positions, *, method="diffusion_kde"|"gaussian_kde"|"binned"|"glm", ...)` → `SpatialRatesResult` (`spatial.py:1940`; fields `firing_rates`, `occupancy`; property `spatial_information` bits/spike at `spatial.py:2380`; `classify()`, `label_cell_types()`, `summary_table()`)
  - `neurospatial.encoding.spatial_autocorrelation(env, firing_rate)`, `grid_score(autocorr_2d)`, `border_score(env, firing_rate, ...)`
  - `src/neurospatial/encoding/directional.py:2585` `compute_directional_rates(spike_times, times, headings, *, bin_size=, angle_unit="rad", ...)`
  - `src/neurospatial/decoding/estimator.py:95` `BayesianDecoder(env).fit(spike_times, times, positions, *, epochs=...)` (`:647`) and `.predict(spike_times, times, *, epochs=...)` (`:866`) → `DecodingResult` (`decoding/_result.py:337`; fields `posterior`, `times`; property `map_position`)
  - `src/neurospatial/decoding/metrics.py:290` `median_decoding_error(env, decoded_positions, actual_positions, *, metric="geodesic")`
  - `neurospatial.io.nwb.read_position / read_head_direction / read_units / read_intervals / read_pose / write_environment / write_spatial_rates / read_spatial_rates`
  - `neurospatial.replay.detect_trajectory_radon(env, posterior, times, ...)`, `neurospatial.behavior.segment_by_velocity / detect_laps / detect_region_visits / graph_turn_sequence`

## Scope and dependency policy

### Goals

- One standalone, runnable loader script per dataset that returns the
  [`Recording`](shared-contracts.md#recording) record — doubling as an example
  of bringing your own recording into neurospatial.
- A driver that runs a fixed operation registry on every loaded recording and
  writes a dataset × operation table with status, runtime and peak memory.
- Pre-registered expectations per dataset, committed before the first run.
- Independent reference checks where another tool publishes or computes the
  same quantity (pynapple spatial information and head-direction decoding,
  opexebo grid score).
- Warning calibration in both directions: every warning neurospatial emits on
  correctly loaded real data gets a verdict (warranted or false alarm), and each
  recording is re-run with known defects re-introduced (headings in degrees,
  missing positions as 0, wrong time or length units, the dataset's own quirks)
  to see whether neurospatial raises, warns or stays silent.
- A gap log with evidence, and (phase 4) a ranked list of proposed changes.

### Non-Goals

- Changing `src/neurospatial/` in phases 1–3. A loader may *work around* a gap;
  it must not patch the library. Fixes come after phase 4, one PR per decision.
- A general-purpose dataset package or a `Session` object for neurospatial.
  The `Recording` record is harness-internal (the package stays raw-array).
- Covering the whole catalog. Later batches are chosen in phase 4 from the
  empty cells of the table.
- CI integration. The harness needs network and large downloads; it runs locally.
- Promoting scripts to `examples/` or tutorials (later, after phase 4 picks which).

### Dependency policy

Add a non-default dependency group in `pyproject.toml`:

```toml
datasets = [
    "fsspec[http]>=2024.6", # byte-range reads of zip members (bat archive)
    "mat-io>=0.5",        # MATLAB `table` objects (bat spike tables) -> pandas
    "opexebo>=0.7",       # reference grid score
]
```

Run the harness with
`uv run --group datasets --extra nwb --extra pynapple python datasets/run.py ...`.
pynwb and pynapple come from the existing `nwb` and `pynapple` extras. Check each
floor against the version `uv lock` resolves and set it to that version's major/
minor; if `opexebo` cannot be installed on the project's Python, drop it from the
group and log the grid-score cross-check as unavailable (do not vendor it).
Commit `uv.lock` changes in the same commit as the group.

## Metrics

- Every dataset in the batch loads, passes `validate_recording`, and has a
  committed expectations block that was committed before its first run.
- Every (dataset, applicable operation) cell has a status; every `workaround`,
  `missing` or `error` cell links to a gap-log entry.
- Reference checks report a numeric agreement for each comparison, with the
  convention differences written down when they don't agree.
- Every warning template and hazard probe in `results/warnings.md` has a verdict.
- Phase 4 output: each proposed change cites ≥2 datasets or one reference
  disagreement as evidence.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Scope creep into 40 loaders and a parallel I/O layer | Fixed batches; phase 4 gate before more datasets; a pattern is promoted to neurospatial only after it recurs across datasets |
| Loader silently "fixes" a quirk and hides the gap | Every transformation is listed in `Recording.conversions` and gets a gap entry; loaders don't hide quirks, they record them |
| Confirmation bias (outputs look plausible, so "ok") | Expectations committed before the first run; reference checks against independent tools |
| Comparing tools with different conventions and calling it a bug | Reference checks fix conventions explicitly (bins, range, smoothing, occupancy handling) and report the difference before judging |
| NWB data in nonstandard locations misread as missing | Loaders walk the whole HDF5 tree by `neurodata_type` before declaring something absent (see `nwb-nonstandard-locations` memory) |
| License / redistribution | No data is committed; the cache is git-ignored; loaders record license per dataset |
| Untrusted downloaded content | Never unpickle; `np.load(..., allow_pickle=False)`; read MATLAB/HDF5 with data readers only |
| Long downloads stall a session | Each loader states its download size; the bat archive is read by member (byte ranges), not downloaded whole |

## Rollout Strategy

Additive, one PR per phase into `datasets/`. Nothing in the installed package
changes. Phase 4 is docs-only.

## Open Questions

1. Does the gap log become GitHub issues, or stay a file? — Best answer: stay a
   file through phase 4; phase 4 proposes which entries become issues/PRs.
2. Should loaders later move into `neurospatial.datasets` (pooch-style fetchers
   like NeMoS/MNE)? — Deferred to phase 4; decide from how much loader code
   turned out to be generic.

## Estimated Effort

Phase 1 ≈ 900–1,200 LOC (harness ~500, two loaders ~150 each, reference checks
~200, tests ~200). Phases 2 and 3 ≈ 150–250 LOC per loader plus new operations.
Phase 4 is prose.
