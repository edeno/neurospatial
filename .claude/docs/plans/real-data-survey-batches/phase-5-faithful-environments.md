# Phase 5 — Faithful environments from the stated apparatus

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [appendix](appendix.md)

Prerequisite: the environment-fidelity study (`datasets/studies/environment_fidelity.py`,
`datasets/ENVIRONMENT_FIDELITY.md`, gaps G-110–G-117) and the CRCNS batch merged.

Goal: every loader hands the harness the apparatus as its source states it
(outline, track graph, bars, room), so the driver builds the environment from
declared geometry; samples only place it. Where the source states no geometry,
the environment stays `from_samples` and is marked inferred.

**Inputs to read first:**

- `datasets/ENVIRONMENT_FIDELITY.md` and `datasets/results/studies/environment_fidelity.md` — per-recording verdicts, the stated geometry and its sources.
- `datasets/operations.py` `_env` and `_graph_env` — how environments are built today; `_graph_env` passes `nx.dfs_edges`, which drops every non-tree edge (G-112).
- `datasets/GAPS.md` G-110 (labyrinth walls), G-111/G-112 (lattice; `from_graph` drops edges silently), G-113 (000447 two W-tracks), G-114 (Vollan round samples in a square), G-015 (open-arena holes), G-021 (loops), G-025 (bat room).
- The public factories in `tests/data/public_api.txt`: `Environment.from_polygon`, `Environment.open_field`, `Environment.from_graph`, `Environment.linear_track`, `Environment.maze`, and the arena classes; read their docstrings before use.

## Tasks

- **Harness: declared environments.** Add `rec.geometry["environment"]` as the
  contract: a zero-argument callable (or a description the harness turns into
  one) returning the declared `Environment`. `_env` uses it when present and
  falls back to `from_samples`; every result records `env_kind` (`declared`,
  `inferred`). Keep the samples-built environment available as
  `env.from_samples` so both can be compared.
- **Harness: G-112 workaround.** `_graph_env` (and any declared graph) passes
  every edge of the graph in `edge_order` (an order that visits all edges, not
  `dfs_edges`), and refuses when `env` has fewer bin links than the graph
  implies. Test on a 100 cm square loop: four edges kept, no point on the
  closing edge assigned to a bin more than half a bin away.
- **Loaders, arenas and 3-D** (one agent): 000582, 002030, 000405, Gardner,
  Vollan OF, 000987, 001880 (if stated), Kanter, Hafting, watermaze, hc-2 —
  stated outline and size, placement fitted (centre, rotation; size fixed) with
  the fit residual in `conversions`; Grieves — the 147 bars as a 3-D graph with
  every edge; bat — the room (Forli et al. 2025: 5.6 × 5.2 × 2.5 m) as declared
  bounds. Gap ids G-130–G-149.
- **Loaders, tracks and mazes** (another agent): figure-8, 000044 linear and
  circular, Vollan LT, 000568, 000165, 001695, 000447 (split by session: two
  W-tracks), J16 (authors' graph), Takigawa (centrelines), labyrinth (graph),
  Agarwal 3-arm, Shipley 8-arm, Ólafsdóttir Z-track, hc-3, hc-28, hc-31, hc-18,
  001701 X maze — a track graph from the authors' graph or linearization where
  the file has one, else from the stated layout fitted to the samples, with the
  stated track width. Loops as cyclic graphs. Gap ids G-150–G-169.
- **Compare.** Run the driver with both environments where a declared one
  exists; a study script reports, per operation, what changes (decoding error,
  laps, border scores, information, occupancy) and the fidelity metrics of the
  declared environment.
- **Findings.** A FINDINGS section: which recordings now have faithful
  environments, what changed, and which library changes would make the
  declared path the easy one.

## Deliberately not in this phase

- Changes under `src/`, including the G-112 fix (logged; ships in the source
  plan) and `Environment.from_loop` (decided; in the source plan).
- Guessing geometry no source states: such recordings stay `inferred`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `datasets/tests/test_operations_env.py::test_declared_environment_is_used` | a recording with `geometry["environment"]` runs every op on it and records `env_kind="declared"` |
| `test_operations_env.py::test_graph_env_keeps_every_edge` | a square loop graph keeps four edges; points on the closing edge map within half a bin |
| `datasets/tests/test_placement.py::test_fitted_square_recovers_pose` | samples inside a known square (rotated, shifted) give back centre and rotation within a stated tolerance |
| `uv run ruff check datasets && uv run ruff format --check datasets` | clean |
| Real run (manual) | every loader with stated geometry returns a declared environment; fidelity study rerun shows the declared environments match |

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- User-facing documentation listed as tasks is updated, not deferred.
- Every stated dimension cites a source that was read; fitted placements record their residuals.
