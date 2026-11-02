# Phase 7 — A first-timer (or an LLM) can find the right function

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [src/neurospatial/__init__.py](../../../../src/neurospatial/__init__.py) — the top-level docstring (the most useful single aid in the LLM probe) and the lazy exports. Lines 95 and 173-188 teach private paths.
- [src/neurospatial/decoding/__init__.py](../../../../src/neurospatial/decoding/__init__.py) — the "Start Here" block to copy.
- `src/neurospatial/encoding/__init__.py`, `src/neurospatial/behavior/__init__.py` — they list only private submodules.
- `src/neurospatial/io/__init__.py`, `src/neurospatial/io/nwb/__init__.py` — the lazy `nwb` subpackage is missing from `dir()`.
- `src/neurospatial/replay/__init__.py` — re-exports names that are also public in `replay.{calibration,dataset,methods,pairwise_order,arms}`.
- [docs/glossary.md](../../../../docs/glossary.md), [docs/user-guide/index.md](../../../../docs/user-guide/index.md), [docs/user-guide/neuroscience-metrics.md](../../../../docs/user-guide/neuroscience-metrics.md) (cell-classification guidance around lines 1041-1086, with no heading).
- [tests/test_public_api_snapshot.py](../../../../tests/test_public_api_snapshot.py) — `NAMESPACES`.

**Contracts referenced:**

- [Argument vocabulary](shared-contracts.md#argument-vocabulary) — the names the docs teach.
- [Verdict vs screen](shared-contracts.md#verdict-vs-screen) — the decision table's wording.

## Tasks

- **Write the `compute_` naming rule.**
  - Add to `.claude/rules/public-api.md` and `docs/contributing.md`:
    > `compute_*` takes raw recordings (spike times, times, positions,
    > headings) and returns a result object. A bare noun (`spatial_information`,
    > `border_score`, `goal_bias`) computes one metric from an existing map,
    > result or array. `detect_*` returns events/intervals; `is_*`/`has_*`
    > return bool verdicts; `plot_*` draws.
  - List the public names that break the rule:
    `grep -E "^neurospatial\.[a-z_.]+\.(compute_|[a-z])" tests/data/public_api.txt`,
    then classify each. Expected outliers include:
    - `compute_field_emd`, `compute_region_coverage`,
      `compute_shuffle_pvalue`, `compute_shuffle_zscore`,
      `compute_vte_index`, `compute_step_lengths`,
      `compute_egocentric_bearing`, `compute_egocentric_distance`
      (derived inputs with a `compute_` prefix → drop the prefix);
    - `traveled_path_length`, `mean_square_displacement`, `goal_bias`,
      `angular_efficiency`, `approach_rate` (raw `times, positions` with no
      prefix).

    Decide each in the PR description, then rename.
  - Add `test_compute_prefix_rule` to `tests/test_argument_conventions.py`.
    It checks that every `compute_*` has `times` or `spike_times` among its
    positional parameters (allowlist any justified exception).
- **Start Here blocks.**
  - In the `encoding` and `behavior` package docstrings, model the "Start
    Here" block on `decoding/__init__.py`. Name the 5–8 functions a
    first-timer needs, grouped by question, using public paths only.
  - Remove the private submodule listings.
- **Place-cell decision table.**
  - Add a `## Which place-cell function?` heading in
    `docs/user-guide/neuroscience-metrics.md` (around line 1041). It answers
    each question with its function:
    - Rate map → `compute_spatial_rate(s)`.
    - Does it have a field → `has_place_field` / `detect_place_fields`.
    - Is it a place cell (verdict) → `is_place_cell` (shuffle default) /
      `place_cell_significance` (population).
    - Fast screen → `result.is_place_cell()`, `.classify()`.
    - All cell types → `spatial_cell_type_significance`,
      `.label_cell_types()`.
  - Link the table from `docs/user-guide/index.md:13` and from the
    `is_place_cell` and `has_place_field` docstrings.
- **Root decoder.**
  - Export `decode_session` at the top level (lazy export plus
    `__all__`).
  - The `decode_position` docstring summary becomes "Decode position from a
    precomputed spike-count matrix and rate maps", and it points to
    `decode_session` for spike times.
  - Fix the "Main entry point" wording at `decoding/posterior.py:372`.
- **`__dir__` on lazy modules.** `io` lists `nwb`. `io.nwb`, and every
  package using lazy `__getattr__`, defines `__dir__` returning
  `sorted(__all__)`. Find them with `grep -rln "def __getattr__" src/neurospatial --include=__init__.py`.
  `dir(neurospatial.encoding)` stops listing private submodules: return
  `__all__` only.
- **One public home per name.**
  - For the 50 objects public at two paths (mostly `replay.X` and
    `replay.<submodule>.X`, plus `ScaleBarConfig` in both `animation` and
    `layout`), keep the shorter path.
  - Remove the submodule from `NAMESPACES` and drop the duplicate entries
    from the snapshot. Keep a submodule public only if it holds names that
    are not re-exported, such as recipe constants. In that case document
    the split in its docstring.
  - Update docs that import recipe constants from submodules
    (`docs/user-guide/replay-methods.md` around lines 50, 60, 526-527).
- **Private paths in docs.** Replace them in the README (around lines
  200-201), the top-level `__init__` docstring (lines 95, 173-188) and
  `docs/glossary.md:16`.
- **Glossary.**
  - "Place field" means a thresholded region of the rate map. The README
    and quickstart say "rate map" where they mean the whole map.
  - Fix the PSTH's "spike-triggered" placement (it is event-triggered).
  - Add entries: epoch, interval, trial, lap, `max_gap`, `spike_window`,
    posterior, PSTH, `dt`.
  - Rename "Cell (graph)" to "Bin cell" or similar, to avoid the clash with
    neurons.
- **Docs home.** `docs/index.md` leads with a scientific question and a
  first-figure example. Move layout-engine and protocol material below it,
  and use a first example that does not print "Bin 2 has 0 neighbors".
- **`llms.txt` (optional, small; drop it if the phase is already large).**
  - Generate `docs/llms.txt` with a script in `scripts/`, from
    `tests/data/public_api.txt` plus each public function's first docstring
    line, followed by a short conventions header: argument order, units, and
    NaN vs zero. Copy the header from `CLAUDE.md` "Scientific conventions"
    and "Canonical argument order".
  - Add a sync test that regenerates it and diffs.
  - Have mkdocs serve it at the site root.
- **CHANGELOG.** **Breaking**: removed duplicate public paths and renamed
  `compute_` outliers (old → new table). "Added": root `decode_session`,
  `llms.txt`.

## Deliberately not in this phase

- LLM redirect aliases for non-existent names (rejected as overkill).
- Notebook content rewrites beyond import paths and renamed functions.
- A clean outside-the-repo LLM rerun. It is a measurement to run after
  merge, not code.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_compute_prefix_rule` | Every public `compute_*` takes raw recordings, apart from the allowlist. |
| `tests/test_public_api_snapshot.py` | Only intended removals and renames. |
| `test_lazy_modules_dir_lists_all` | `set(__all__) <= set(dir(mod))` and no private submodule names, for each lazy package. `"nwb" in dir(neurospatial.io)`. |
| `test_decode_session_at_root` | `neurospatial.decode_session is neurospatial.decoding.decode_session`. |
| `test_docs_teach_public_paths` | No ``neurospatial.<pkg>.<private_submodule>`` import appears in README, `docs/**/*.md` or any `__init__` docstring. Build the private-module list from the package tree minus `NAMESPACES`. |
| `uv run pytest tests/docs/test_executable_docs.py -n 0` | Docs run. |
| `test_llms_txt_in_sync` (if shipped) | The regenerated file equals the committed file. |

## Fixtures

None.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
