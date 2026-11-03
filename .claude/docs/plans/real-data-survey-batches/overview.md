# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

Everything this plan touches lives in `datasets/` (tracked, not installed, not
run by CI). The harness contract is documented in `datasets/README.md`
("Add a dataset") and in code:

- `datasets/_record.py:21` — `Recording` dataclass of plain arrays; `params`
  documents its keys at `:52` (`nwb_path` among them). Extended by one key:
  `nwb_path` becomes `nwb_source` (a local path or a URL; see
  [designs.md#streamed-nwb](designs.md#streamed-nwb)).
- `datasets/_fetch.py:75` `fetch`, `:90` `fetch_zip_member`, `:111`
  `dandi_asset_url` — pinned downloads. Gains `open_nwb` (local or streamed).
- `datasets/run.py:33` — `LOADERS` registry; one line per new loader.
- `datasets/operations.py:682` — `OPERATIONS` registry (19 ops). New columns
  are added only where a batch brings a new input kind (pose, egocentric
  boundaries, objects). `:487` `nwb_readers` opens `nwb_path` at `:503`;
  switches to `open_nwb`. `:149` `N_SHIFTS = 4`, `:153`
  `information_vs_shifts` — unchanged (the null study is a separate script
  with many more shifts).
- `datasets/reference_checks.py:220` `grid_score_check` — today run only on
  000582 (`:355`); phase 1 runs it on the grid populations too. `:298` opens
  `nwb_path`; switches to `open_nwb`.
- `datasets/load_miniscope_002030.py:108-115` — hazard that reopens
  `nwb_path`; switches to `open_nwb`.
- `datasets/load_rosenberg_labyrinth.py:117` `_TrajUnpickler` — the existing
  restricted unpickler pattern; Gardner's `.npz` reuses the idea (see
  [designs.md#safe-pickles](designs.md#safe-pickles)).
- `datasets/_calibration.py` — hazard probes; new loaders add `HAZARDS` and
  the driver probes them unchanged.
- `datasets/GAPS.md` (G-001…G-036), `datasets/verdicts.json`,
  `datasets/results/`, `datasets/FINDINGS.md` — updated by every phase.
- `pyproject.toml:495-501` — the `datasets` dependency group.
- `src/neurospatial/` — **untouched** by every phase.
- `docs/research/reports/Open spatial neuroscience datasets.md` — the catalog
  the batches are drawn from (rows cited by dataset name in
  [appendix.md](appendix.md)).

## Scope and dependency policy

### Goals

- Load most of the downloadable catalog through the harness, so that every
  operation in the matrix has been exercised on several recordings and every
  kind of input (2-D, 1-D, 3-D, loops, graphs, pixels, pose, head direction,
  objects, sleep) has real data behind it.
- Produce the evidence the maintainer's three gated decisions wait on
  (`datasets/FINDINGS.md`, "Decisions for the maintainer", items 1–4):
  grid score (Gardner and Vollan populations), the origin-share warning (more
  recordings, including linear tracks whose 0 is a real place), the
  heading-coverage warning (a linear track with tracked head direction), and
  the spatial-information bias (shuffle-null study).
- Keep every loader usable on its own as an example of bringing that dataset
  into neurospatial.

### Non-Goals

- Any change under `src/neurospatial/`. The agreed library changes (the
  radians-labelled-degrees reader warning, the fewer-than-5-bins warning, NWB
  reader coverage, a loop-track factory `Environment.from_loop`) get their own
  plan. Gaps found here are logged with evidence, never fixed here.
- Calcium / continuous-signal rate maps and heading decoding (deferred by the
  maintainer). Imaging datasets in the catalog are not in these batches.
- A `neurospatial.datasets` fetcher or promoting a loader to `examples/`
  (tutorial deferred).
- Datasets whose smallest usable file is above about 15 GB and cannot be read
  by byte range (e.g. DANDI 000978, the Frank-lab raw dandisets, Spyglass
  exports). They are listed as out of reach in `datasets/FINDINGS.md`, not
  loaded.
- Choosing between the decision options. Phase 1 reports evidence and a
  recommendation; the maintainer decides.

### Dependency policy

New packages go into the `datasets` dependency group only
(`uv add --group datasets <pkg>`), never into the package's runtime
dependencies or extras. Expected additions: `h5py` access over HTTP comes with
`fsspec[http]` already; `inflate64` (Deflate64 zips, phase 3); an Axona reader
only if parsing `.pos`/`.cut` by hand is unreasonable (phase 3 decides, prefers
hand parsing). After any `uv add`, check that `uv.lock` changed only for the
added package (memory: an earlier `dandi` addition bumped `wcwidth`).

## Metrics

- Recordings loaded: 8 → about 35 by the end of phase 4.
- Every non-`ok` matrix cell has a gap id; every new `results/warnings.md` row
  has a verdict (`run.py` already fails on `unreviewed`).
- No new neurospatial warning on a clean recording without a gap entry
  explaining it.
- Phase 1 alone: the three gated decisions each get a table in
  `datasets/FINDINGS.md` with the numbers the decision needs (see
  [designs.md#gated-evidence](designs.md#gated-evidence)).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Streaming a 5–9 GB NWB file reads far more than needed (lazy datasets touched by pynwb) | Read with h5py paths for the few datasets needed; pynwb only for `io.nwb_readers`. Measure bytes and seconds in the smoke test before the full run ([designs.md#streamed-nwb](designs.md#streamed-nwb)) |
| A streamed file changes under a fixed asset id | Pin DANDI's published `dandi:sha2-256` digest and compare it to the API's metadata at load time |
| A pickled `.npz` executes code on load | Inspect opcodes with `pickletools` without executing, then unpickle with an allow-list ([designs.md#safe-pickles](designs.md#safe-pickles)) |
| Expectations tuned to the data | Each phase commits `DATASET` + `EXPECTATIONS` before the first load, as in the first survey; PRs merge with a merge commit (not squash) so that order is visible |
| Long runs (hours of sleep, 100+ units, 100 shifts) | Smoke-test one dataset per phase, measure, extrapolate; run in tmux and record the session name |
| Account-gated sites (CRCNS) | Credentials only through `CRCNS_USERNAME` / `CRCNS_PASSWORD` by name, never printed; never work around bot checks — log a site that blocks as out of reach |
| A shared checkout switching branches under the run | Work in a dedicated worktree; check the current branch before each commit |

## Rollout Strategy

One PR per phase, merge commit, on top of current `main`. Phases 2–4 do not
depend on each other and may run in any order after phase 1 has added
`open_nwb` (phase 2 needs it; phases 3–4 do not). Results are regenerated
on the `main` of the day, so a library change merged mid-plan shows up as a
changed cell; record it in the phase's FINDINGS update.

## Open Questions

1. Laps in open arenas (`datasets/FINDINGS.md` decision 5) is still open; the
   new open-field recordings add `behavior.laps` rows that bear on it. Report
   them, do not decide.
2. Which Vollan open-field session has the most grid units — answer by reading
   the file metadata before writing that loader's expectations (appendix).

## Estimated Effort

About 25–30 new loaders of 80–150 lines each, one streamed-opener helper
(~60 lines), one study script (~250 lines), three new operations (~40 lines
each), tests (~400 lines), and Markdown updates. No time estimate.
