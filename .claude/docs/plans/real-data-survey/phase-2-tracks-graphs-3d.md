# Phase 2 — 1-D track with replay, graph maze, and 3-D flight

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Prerequisite: phase 1 merged (harness, registry, driver exist).

**Inputs to read first:**

- `datasets/` as merged in phase 1 — `_record.py`, `operations.py`, `run.py`, `GAPS.md`, `results/matrix.md`.
- `src/neurospatial/environment/core.py:3944` (`Environment.from_graph(graph, edge_order, *, bin_size, edge_spacing=, track_width=, ...)`) and `docs/user-guide/environments.md` (track and graph environments).
- `docs/user-guide/replay-methods.md` — event decoding (`decode_intervals`, lines ~100-130), `compare_replay_methods` (~172-200), "Using your own recording" (~903-925).
- `neurospatial.replay` names in `tests/data/public_api.txt:539-659` (`decode_intervals`, `compare_replay_methods`, `detection_rates`, `replay.methods.DAVIDSON_2009`, `GROSMARK_2016`).
- `docs/user-guide/trajectory-and-behavioral-analysis.md` and `neurospatial.behavior` signatures in `tests/data/public_api.txt:87-148` (`segment_by_velocity`, `detect_laps`, `detect_region_visits`, `graph_turn_sequence`).
- Dataset specifics: [figure8_000552](appendix.md#figure8_000552-phase-2), [rosenberg_labyrinth](appendix.md#rosenberg_labyrinth-phase-2), [yartsev_bat_demo](appendix.md#yartsev_bat_demo-phase-2).

**Contracts referenced:** [Recording](shared-contracts.md#recording), [Loader module](shared-contracts.md#loader-module), [Operation and Outcome](shared-contracts.md#operation-and-outcome), [Gap-log entry](shared-contracts.md#gap-log-entry) — unchanged; do not weaken.

**Designs referenced:** [loader recipe](designs.md#loader-recipe), [fetch](designs.md#fetch-and-cache) (`fetch_zip_member` for the bat archive), [registry](designs.md#operation-registry).

## Tasks

- **Expectations first (own commit).** Add `datasets/load_figure8_000552.py`, `load_rosenberg_labyrinth.py`, `load_yartsev_bat_demo.py` with `DATASET` + `EXPECTATIONS` only (seeds in the appendix) and a `load` raising `NotImplementedError`. Register the keys in `run.py:LOADERS`. Commit.
- **figure8_000552 loader.** Linearized position as `positions` (n, 1) in cm; 2-D position too if the file has it (walk the tree), kept in `geometry["positions_2d"]` with its own units. `events["ripples"]` from the Ripples table (n, 2); sleep states as `epochs`. `params`: `bin_size` (justify, e.g. 2 cm on a 222 cm track), `min_speed` for rate maps, `train_epochs`/`test_epochs` from running laps, `replay_dt` 0.02 s (cite the recipe's published value from `GROSMARK_2016`/`DAVIDSON_2009`).
- **rosenberg_labyrinth loader.** Read the repo's tf format by re-implementing the repo's reader (MIT; attribute it in the module docstring) — no pickle. `positions` = nose x, y in the repo's units; `geometry["graph"]` = the 127-node maze graph with node coordinates, built from the repo's maze definition. No neurons (`spike_times=None`).
- **yartsev_bat_demo loader.** Fetch the behavior and three spike files with `fetch_zip_member`; read behavior with `scipy.io.loadmat`, spike tables with `matio.load_from_mat`; spike times = `spikeTimes_usec / 1e6` (global clock); apply the duplicate-unit merges from `merges_Dataset_*.mat` following `NP_Bat_Demo.m`. `positions` = `r` (m); `epochs["flight"]` from `f_smp` / `t`; `geometry["bounds"]` = `r_lim`. `params`: `bin_size` (justify, e.g. 0.2 m → ~29 × 26 × 12 bins, and record the bin count), `analysis_epochs` = flight. Record the outside-`r_lim` samples as a `data-quirk` gap.
- **New operations** (append to `operations.py`, ordered after the phase-1 ones):
  - `env.from_graph` (requires `positions`, `geometry["graph"]` — return `n/a` from inside the op when the graph is absent): `Environment.from_graph(graph, edge_order, bin_size=...)`; summary `n_bins`. Decide `edge_order` from the graph (document how).
  - `behavior.segment_by_velocity` (requires `positions`): `segment_by_velocity(times, positions, min_speed=params["min_speed"])`; summary: number of segments, total running time.
  - `behavior.laps` (requires `positions`): `detect_laps(env, times, positions)`; summary: n laps by direction.
  - `behavior.turn_sequence` (requires `positions`, graph env): `graph_turn_sequence(graph_env, times, positions)`; summary: n turns, fraction left.
  - `replay.compare_methods` (requires `spike_times`, `events["ripples"]`): fit rates on running epochs (`compute_spatial_rates(..., epochs=train_epochs, min_speed=...)`), `decode_intervals(env, spike_times, ripples, rates, dt=replay_dt)`, then `compare_replay_methods(env, events, replay_methods=[DAVIDSON_2009, GROSMARK_2016], spike_counts=counts, spike_times=spike_times, encoding_models=rates, dt=replay_dt, rng=0)` and `detection_rates(table)`; summary: per-method detection rate with CI. The bat ripples (`RPL_probe*`) are optional — include only if the file is under ~100 MB for the chosen probe.
  - The phase-1 ops already cover 3-D rates, information, decoding; the 3-D run is the test of whether they work in 3-D. Grid and border scores on 3-D maps: let them run; an error or meaningless value is the finding.
- **Run and log.** Run the driver on all five datasets so far; commit results; give every non-`ok` cell a gap entry. Record runtime and peak memory for the 3-D decode and the replay comparison in the gap log if they're slow (> 60 s) or large (> 4 GB).
- **README.** Add the three datasets to `datasets/README.md`'s dataset list with their download sizes.

## Deliberately not in this phase

- Changes under `src/neurospatial/` — log them.
- Imaging data and the quirky-NWB datasets (phase 3).
- Replay methods beyond DAVIDSON_2009 and GROSMARK_2016; null-data calibration (`replay-methods.md` "False positives on null data") — useful later, but this phase only checks the workflow runs on real events.
- Downloading the bat LFP or the full zip.
- Theta sequences / phase precession (needs LFP; later batch).

## Validation slice

| Test | Asserts |
| --- | --- |
| `datasets/tests/test_operations_tracks.py::test_graph_env_on_simulated_track` | `env.from_graph` op on a simulated linear-track recording with a 2-node graph returns `ok` and the expected bin count for the track length / bin size |
| `test_operations_tracks.py::test_replay_op_detects_simulated_sweeps` | on a seeded `neurospatial.simulation.linear_track_session` + `simulate_replay_event` sweeps (as in `docs/user-guide/replay-methods.md:69-107`), `replay.compare_methods` reports a detection rate for sweeps above that for stationary events (ground truth known) |
| `test_operations_tracks.py::test_ops_na_without_graph_or_ripples` | ops return `n/a` with an explanatory note when `geometry["graph"]` or `events["ripples"]` is absent |
| `datasets/tests/test_record.py::test_three_dimensional_recording_validates` | a simulated 3-D recording passes validation and `n_dims` measures 3 |
| `uv run ruff check datasets && uv run ruff format --check datasets` | clean |
| Real run (manual, network) | the three loaders validate; expectations results committed; every non-`ok` cell has a gap id |

## Fixtures

- Simulated linear track and replay events from `neurospatial.simulation` (seeded), reused from the replay guide's construction.
- A simulated 3-D recording: random walk in a box with 3-D Gaussian place fields (build in `conftest.py`, seeded).
- Real data only from the cache.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Expectations were committed before the first real run of each new loader.
- The vendored labyrinth reader carries the MIT attribution.
