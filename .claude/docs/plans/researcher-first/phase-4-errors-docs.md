# Phase 4 — Errors that teach, complete docstrings, and examples that run in CI

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#error-message-contract)

**Line numbers** below were read on `main` at `da631a47`. Phases 1–3 land first and shift them, so find each site by the quoted message text, not by the number alone.

**Inputs to read first:**

- [src/neurospatial/_exceptions.py](../../../../src/neurospatial/_exceptions.py) — four public classes. It imports `EnvironmentNotFittedError` (`environment/decorators.py:19`) and `GraphValidationError` (`layout/validation.py:28`) to re-export them. That import is the cycle worked around at `environment/queries.py:583-590`. `BinIndexOutOfRangeError`, `IncompatibleEnvironmentError` and `LayoutNotBuiltError` are exported but never raised anywhere in `src/`.
- `src/neurospatial/environment/core.py:339-353` — `[E1006]`, the model message. Even it has no `Fix:` line.
- `src/neurospatial/encoding/_validation.py:40-247` — `validate_env_fitted`, `validate_times` (its `context` defaults to `"encoding"`, used by decoding too) and `validate_trajectory`.
- `scripts/test_doc_snippets.py`, `docs/snippets.yml`, `tests/test_doc_snippets_helper.py` and `.github/workflows/test_docs.yml` — the current snippet runner. It is a 27-entry manifest keyed by block *index*, so inserting a block silently retargets an entry. All 25 non-skipped entries pass on `main`.
- `pytest.ini` — this overrides `pyproject.toml`'s pytest table. `addopts` deselects `slow`, and **no CI job runs `slow` tests**, so a docs test marked `slow` is never enforced.
- **Files earlier phases already changed** (re-locate every site by its quoted message text):
  - `.github/workflows/test_docs.yml` and `test_nwb.yml`: Phase 1 Task 1 added `feat/researcher-first` to their triggers, so CI already runs on this branch. This phase edits only their steps.
  - `environment/core.py`, `environment/factories.py`, `layout/engines/graph.py`: Phase 2 Tasks 1 and 5 (calculus operator cache, `edge_id` numbering).
  - `decoding/posterior.py`, `decoding/estimator.py`: Phase 1 Tasks 2 and 6; Phase 3b (per-run bins).
  - `events/alignment.py`: Phase 2 Task 8 and Phase 3b (`_keep_observed_events`). The four window checks in 4.3 sit next to that code.
  - `behavior/segmentation.py`: Phase 3c split each detector into a public wrapper and `_<name>_contiguous`. Raise `RegionNotFoundError` in the public wrapper, before the per-run loop.
  - `io/nwb/_behavior.py`: Phase 2 Task 3 (`data_from_series`, lazy refusal).
  - `CLAUDE.md` and `docs/snippets.yml`: Phase 3c changed the `heading_from_velocity` calls (pattern 7). `CHANGELOG.md`: append after the earlier phases' sections.

**Contracts referenced:**

- [Error-message contract](shared-contracts.md#error-message-contract) — implemented here. Rules 1–4 apply to every site this phase touches.
- [Time-window semantics](shared-contracts.md#time-window-semantics) — the docs examples exercise Phase 3's defaults. This phase does not change them.

**Designs referenced:** none.

## Tasks

### 4.1 Exception base class and hierarchy

- In `_exceptions.py`, add the base class and the shared formatter, and **move** the `EnvironmentNotFittedError` and `GraphValidationError` class bodies in unchanged. `environment/decorators.py` and `layout/validation.py` then import them *from* `_exceptions` (`_exceptions` imports nothing internal). This removes the cycle, so replace the local import at `queries.py:583-590` with a top-level import.

  ```python
  class NeurospatialError(Exception):
      """Base class for every exception neurospatial defines.

      Each concrete error also inherits a built-in type, listed first, so
      ``except ValueError`` keeps working. ``except NeurospatialError`` catches
      only problems that neurospatial itself detected.
      """


  def _format_error(what: str, *, fix: str, why: str | None = None) -> str:
      """Return ``what``, an optional ``why``, and a final ``Fix:`` line."""
      lines = [what.strip()] + ([why.strip()] if why else []) + [f"Fix: {fix.strip()}"]
      return "\n".join(lines)
  ```

- Bases:
  - `RegionNotFoundError(KeyError, ValueError, NeurospatialError)`. Adding `ValueError` (the precedent is NumPy's `AxisError(ValueError, IndexError)`) lets this one class replace the eight `ValueError("... not found ...")` sites in 4.3 with no change of catch type.
  - `BinIndexOutOfRangeError(ValueError, NeurospatialError)`.
  - `IncompatibleEnvironmentError(ValueError, NeurospatialError)`.
  - `LayoutNotBuiltError(RuntimeError, NeurospatialError)`.
  - `EnvironmentNotFittedError(RuntimeError, NeurospatialError)`.
  - `GraphValidationError(ValueError, NeurospatialError)`.

  `RegionNotFoundError` must also define `__str__` returning `str(self.args[0])`. `KeyError.__str__` comes first in its MRO, so it quotes the message and prints the `Fix:` line as a literal `\n`; verified: an override on the base class is shadowed. Each `__init__` builds its message with `_format_error`. `RegionNotFoundError.__init__(name, *, available=None, argument="region_name")` suggests the closest match with `difflib.get_close_matches(name, available, n=1)`. Its fix is either "pass `{argument}='{match}'`" or "add it first: `env.regions.add('{name}', point=(x, y))` (or `polygon=...`)".
- Export `NeurospatialError` from `neurospatial/__init__.py` (import block at `:223-229`, `__all__` at `:297-305`).
- Give the unused classes real raise sites:
  - `BinIndexOutOfRangeError` replaces the `IndexError` at `environment/queries.py:51`. Update the `Raises` sections of `neighbors` and the other callers of `_resolve_point_or_index`.
  - `IncompatibleEnvironmentError` is raised at `composite.py:104` (`[E1003]`) and `decoding/posterior.py:1636` (encoding-model bins ≠ `env.n_bins`).
  - `LayoutNotBuiltError` replaces the eight `RuntimeError("Layout not built…")` sites: `layout/mixins.py:293, :358, :436`; `layout/engines/hexagonal.py:364`; `graph.py:386, :422`; `shapely_polygon.py:191`; `triangular_mesh.py:208`.

### 4.2 Mechanical relabel and the domain word

- Rename the label `HOW:` to `Fix:` in the existing WHAT/WHY/HOW messages: 156 occurrences across 22 files, `events/` and `animation/` mostly (`grep -rn "HOW:" src/neurospatial`). This is text only.
- Make `context` keyword-required, with no `"encoding"` default, in `validate_times`, `validate_spike_times` and `validate_trajectory` (`encoding/_validation.py:76, :116, :187`). Then every caller, including `decoding/session.py:445`, names its own function (contract rule 3).
- `validate_trajectory` gains `n_dims: int | None = None`. It collects every problem into a list and raises once (rule 4), covering:
  - `times` not 1-D;
  - length mismatches;
  - `positions.shape[1] != n_dims`;
  - 1-D `positions` with `n_dims > 1`;
  - **swap detection**: `times.ndim == 2` and `positions.ndim == 1` gives "did you pass positions before times?".

  Pass `n_dims=env.n_dims` from every encoding entry point that has an env. This replaces the deep `layout/helpers/regular_grid.py:682` "Dimensionality mismatch … grid_edges" message that users hit today.

### 4.3 Rewrite the first-run raise sites

Of 454 raise sites in the flagship modules, 1 has a `Fix:` line and 122 give no guidance at all (an AST scan of `raise X("...")`). Rewrite only the sites below, chosen because they are on first-run paths or were found by probes. All use `_format_error`.

| Site (`da631a47`) | Change |
| --- | --- |
| `environment/core.py:339`, `_exceptions.py` E1004 text (from `decorators.py:97-103`) | Add `Fix: env = Environment.from_samples(positions, bin_size=2.0)`. |
| `environment/factories.py:420` (`positions must be a 2D array`) | Fix: `positions[:, None]` for 1-D data; `positions.T` when the shape is `(n_dims, n_samples)`. |
| `environment/factories.py:189` (`Unknown maze kind`) | List the valid kinds. |
| `environment/queries.py:51, :56, :63` | See 4.4 (length-1 array). |
| `layout/helpers/utils.py:331` (`All 'positions' are NaN`) | Report "N of M rows contain NaN; check for tracking dropouts", with a fix. |
| `layout/helpers/regular_grid.py:378-390` (`[E1002]`) | Fix gives a value, e.g. `bin_size=2.0` (same units as `positions`). |
| `encoding/_validation.py:73` | See 4.4 (non-Environment first argument). |
| `encoding/_validation.py:95, :155, :219-245`; `encoding/_binning.py:813` | Fix lines; covered by the 4.2 rewrite. |
| `encoding/spatial.py:2306`, `directional.py:1603`, `view.py:1063`, `egocentric.py:1183` (`unit_ids has … elements`) | Fix: "pass one label per unit (`len(unit_ids) == {n}`)". |
| `encoding/spatial.py:3751, :3817` (unknown direction label) | Fix names the known labels. |
| `decoding/posterior.py:1519`, `likelihood.py:153` (neuron-count mismatch) | Fix: build counts and models from the same unit list, in the same order. |
| `decoding/posterior.py:216, :1636`; `decoding/_binning.py:192` | Fix: `handle_degenerate='uniform'`; `IncompatibleEnvironmentError` (4.1); "use `dt <= {span}`". |
| `events/alignment.py:131, :282, :437, :586` (four copies of the window check) | Replace with one `_validate_window(window, *, context)` (see 4.4). |
| `behavior/segmentation.py:462, :645, :651, :1244, :1849, :1860, :2339`; `events/regressors.py:963`; `environment/queries.py:592` | `raise RegionNotFoundError(name, available=..., argument="start_region")`, with the right argument name each time. |
| `behavior/segmentation.py:1238, :1242, :1855` | Fix shows the missing argument, e.g. `start_region='home'`. |
| `animation/core.py:328, :333, :508, :556` | Fix: `save_path='out.mp4'`; list the valid backends. |
| `io/files.py:299, :442, :444` | Fix line. Note that `env.to_file(path)` writes both `.json` and `.npz`. |
| `io/nwb/_units.py:171`; `io/nwb/_behavior.py:183, :199, :208` | Fix names the reader argument (`unit_ids=`, `processing_module=`). These stay bare `KeyError`, whose `str()` escapes newlines, so the fix is the final sentence (`… Fix: pass processing_module='behavior'.`), not a separate line. |

Representative before/after messages (each "after" message is asserted in 4.4's tests):

```text
# compute_spatial_rate(spike_times, times, positions)        -- env forgotten
before: EnvironmentNotFittedError: [E1004] compute_spatial_rate() requires the environment to be fully initialized. Ensure it was created with a factory method. …
after:  TypeError: compute_spatial_rate() expects an Environment as its first argument, got ndarray with shape (200,).
        Fix: build one with env = Environment.from_samples(positions, bin_size=2.0), then call compute_spatial_rate(env, spike_times, times, positions).

# compute_spatial_rate(env, spike_times, times, positions[:, 0])   -- 2-D env, 1-D positions
before: ValueError: Dimensionality mismatch: points have 1 dimension(s), but grid_edges has 2 and grid_shape has 2. …
after:  ValueError: compute_spatial_rate: positions has shape (1800,) but env is 2-D, so positions must have shape (n_samples, 2).
        Fix: pass both coordinates, e.g. np.column_stack([x, y]); for a 1-D track, build env from 1-D data (positions[:, None]) or Environment.linear_track(...).

# detect_laps(..., start_region="home") on an env with no regions
before: ValueError: start_region 'home' not in env.regions. Available regions: []
after:  RegionNotFoundError: Region 'home' not found. This environment has no regions.
        Fix: add it first: env.regions.add('home', point=(x, y)) (or polygon=...), then pass start_region='home'.
```

### 4.4 Detect the silent and weak first-run mistakes

| Mistake (probed on `main`) | Today | Change |
| --- | --- | --- |
| `from_samples(positions, bin_size=500)` on 80 cm data | Succeeds silently with a 4-bin env. | In `from_samples` (`factories.py`, just before `cls.from_layout` at `:506`): if `np.all(bin_size >= np.ptp(finite_positions, axis=0))` and some extent is > 0, `warnings.warn(UserWarning)`. The message gives the value, the per-axis extent, and `Fix: bin_size=<max extent / 50>`. Run the check only when `bin_size` is a scalar or has length `n_dims`. A 2-D linear track (200 × 5 cm, `bin_size=5`) must not warn. |
| `bin_size=0.01`, ~99M bins | `ResourceWarning`, which default filters hide (`layout/helpers/utils.py:1181-1188`). | **Decision:** make it a `UserWarning` and add a `Fix:` line ("increase bin_size; this grid has {n_bins:,} bins"). The existing 8 GiB hard ceiling (`regular_grid.py:45, :517`) stays. |
| `env.neighbors(env.bin_at([x, y]))` | Fails deep inside with "points have 1 dimension(s)… grid_edges". This is CLAUDE.md pattern 1. | In `_resolve_point_or_index` (`queries.py:53`): a shape-`(1,)` input in an env with `n_dims > 1` raises `ValueError`. The message explains that `bin_at` returns an array, with `Fix: env.neighbors(int(bin_idx[0]))` or pass the point. |
| Environment forgotten | Misleading E1004, shown above. | `validate_env_fitted` (`_validation.py:40`) raises `TypeError` when `env` has no `_is_fitted` attribute. The real E1004 stays for half-built envs. |
| `peri_event_histogram(..., window=(-500, 1000))`, a window in ms | Silently returns 60 000 bins. | `_validate_window` warns (`UserWarning`) when `stop - start > 60` s, with `Fix: window=(-0.5, 1.0) if you meant milliseconds`. **Assumption:** 60 s is wider than any realistic peri-event window. |
| `from_samples(positions)` (no `bin_size`); `animate_fields(fields)` (no `frame_times`) | Python's own `TypeError`, which names the mixin (`EnvironmentFactories`, `EnvironmentVisualization`). | **Decision: leave as is.** The message already names the missing argument. Rewriting the mixin `__qualname__` would also mislabel `EgocentricPolarEnvironment`, which shares those mixins (MRO verified). A `=None` sentinel that then raises would hide that the argument is required. |

### 4.5 Docstring completeness

Most docstrings are already complete. Measured over every `__all__` function in the twelve snapshot namespaces, plus the `Environment` factories and flagship methods, counting both "Parameters" and "Other Parameters":

- Every `neurospatial.encoding` and `neurospatial.simulation` function has Parameters, Returns and Examples. (The archive branch's "no Parameters section" finding does not apply to `main`.)
- **No Parameters section:** `Environment.from_polar_egocentric` (it delegates to `EgocentricPolarEnvironment.create`). Copy that method's Parameters section in.
- **Undocumented parameters:**
  - `Environment.from_samples` `**layout_specific_kwargs`;
  - `behavior.detect_region_crossings` `arg3`/`arg4`. Delete the old-order compatibility dispatch (`segmentation.py:411-457`, plus the docstring note at `:340-346`; it warns that it will be "removed in 0.7"). The signature becomes `(position_bins, times, env, *, region_name, direction="both")`, per decision 1. In `tests/behavior/test_detect_region_crossings_argorder.py`, delete `test_old_positional_order_warns` and `test_old_and_new_order_identical`; keep the canonical-order tests.
- **No Examples section** (`load_session` also lacks one, but Phase 6 deletes it, so skip it here):
  - `decode_session_summary`;
  - `decode_position_summary`;
  - `behavior.angular_efficiency`;
  - `behavior.subgoal_efficiency`.
- **No Returns section:** `events.validate_events_dataframe`. Out of scope, but listed for the test: `stats.shuffle_*` and `generate_*` use Yields-style returns, and 8 of 8 `neurospatial.regions` functions lack Examples. The test covers only the root and flagship set (4.6), so those remain follow-ups.
- New `tests/docs/test_docstring_sections.py`: for every root-exported callable and every entry of `FLAGSHIP` (4.6), assert that Parameters (when the signature has parameters), Returns and Examples sections are present, and that every signature parameter except `self`/`cls` is named under Parameters or Other Parameters. Parse with a 15-line regex section splitter, not `numpydoc`: that is only installed transitively through napari, which CI's `--extra dev` does not install.

### 4.6 Flagship docstring examples run in the default test job

`tests/docs/test_flagship_docstrings.py` runs each flagship object's examples with `doctest`, in that object's module globals, in a temporary working directory. It also forbids `+SKIP`. Today none of the 44 docstrings listed below fail, but about 130 of their examples are `+SKIP` and never run: `decode_session` 10, `detect_laps` 18, `detect_runs_between_regions` 16, `Environment.animate_fields` 24, `plot_field` 11, `occupancy` 6, `PeriEventResult` 3, the package docstring 26, and so on. Remove those `+SKIP`s by making each example self-contained, with ≤ 60 s of simulated data.

```python
FLAGSHIP = (
    "neurospatial",  # package docstring
    *(f"neurospatial.Environment.{m}" for m in (
        "from_samples", "open_field", "linear_track", "maze", "from_graph", "from_polygon",
        "occupancy", "bin_at", "neighbors", "to_file", "from_file", "plot_field", "animate_fields")),
    *(f"neurospatial.encoding.{f}" for f in (
        "compute_spatial_rate", "compute_spatial_rates", "compute_directional_rate",
        "compute_directional_rates", "compute_view_rate", "compute_view_rates",
        "compute_egocentric_rate", "compute_egocentric_rates", "is_place_cell",
        "is_head_direction_cell", "is_object_vector_cell", "is_spatial_view_cell",
        "detect_place_fields")),
    *(f"neurospatial.decoding.{f}" for f in (
        "decode_position", "decode_session", "bin_spikes_in_time", "decoding_error", "DecodingResult")),
    *(f"neurospatial.events.{f}" for f in (
        "peri_event_histogram", "population_peri_event_histogram", "PeriEventResult",
        "PopulationPeriEventResult")),
    *(f"neurospatial.behavior.{f}" for f in (
        "detect_laps", "segment_trials", "detect_region_crossings", "detect_runs_between_regions")),
)
NWB_FLAGSHIP = tuple(f"neurospatial.io.nwb.{f}" for f in (  # run in test_nwb.yml (see below)
    "read_position", "read_units", "write_environment", "read_environment"))
# +SKIP is allowed only where execution needs a display or ffmpeg.
_ALLOWED_SKIP = re.compile(r"""backend\s*=\s*["'](napari|video)["']""")


def _resolve(dotted: str) -> object:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(dotted)


def _run_examples(dotted: str) -> None:
    obj = _resolve(dotted)
    module = sys.modules[obj.__name__ if inspect.ismodule(obj) else obj.__module__]
    tests = doctest.DocTestFinder(recurse=False).find(obj, dotted, globs=dict(vars(module)))
    examples = [ex for t in tests for ex in t.examples]
    assert examples, f"{dotted} has no Examples to run"
    skipped = [ex.source for ex in examples
               if ex.options.get(doctest.SKIP) and not _ALLOWED_SKIP.search(ex.source)]
    assert not skipped, f"{dotted}: {len(skipped)} example(s) marked +SKIP never run:\n{skipped[0]}"
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    report: list[str] = []
    for test in tests:
        runner.run(test, out=report.append)
    plt.close("all")
    assert runner.failures == 0, "".join(report)


@pytest.mark.parametrize("dotted", FLAGSHIP)
def test_flagship_docstring_examples_run(dotted, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _run_examples(dotted)


@pytest.mark.nwb
@pytest.mark.parametrize("dotted", NWB_FLAGSHIP)
def test_nwb_flagship_docstring_examples_run(dotted, monkeypatch, tmp_path):
    pytest.importorskip("pynwb")
    monkeypatch.chdir(tmp_path)
    _run_examples(dotted)
```

- **CI for the NWB examples.** Append `tests/docs/test_flagship_docstrings.py -m nwb` to the `pytest` command in `.github/workflows/test_nwb.yml`. The default `test.yml` job installs only `--extra dev`, so `NWB_FLAGSHIP` would otherwise be skipped by `importorskip` and never enforced.

### 4.7 Executable documentation (replaces the snippet manifest)

New file `tests/docs/test_executable_docs.py`:

- **Collection.** It collects ```` ```python ```` blocks itself, so block indices disappear. Markdown carries the markers, on the line directly above a fence:
  - `<!-- docs-test: skip <reason> -->` (the reason is required);
  - `<!-- docs-test: raises <ExceptionName> -->` (used for "❌ Wrong" gotcha examples);
  - `<!-- docs-test: run [setup=<name>] -->` (opt-in files only).
- **File modes.**
  - `RUN_ALL` files execute every block in order, in one namespace per file, the way a reader would.
  - `OPT_IN` files execute only `run` blocks. These are the pages the manifest covered.
- **Fixtures.** README and the getting-started quickstart get **no** pre-seeded names: they must be self-contained, because users copy them. CLAUDE.md and the opt-in pages get `_fixture_namespace()` (see Fixtures below).
- **Tracebacks.** Code is compiled with `"\n" * (line - 1) + code` against the real file path, so a traceback points at the Markdown line.

```python
RUN_ALL = {"README.md": False, "docs/getting-started/quickstart.md": False, "CLAUDE.md": True}
OPT_IN = (".claude/QUICKSTART.md", "docs/user-guide/alignment.md", "docs/user-guide/animation.md",
          "docs/user-guide/interoperability.md", "docs/user-guide/trajectory-and-behavioral-analysis.md",
          "docs/user-guide/video-annotation.md", "docs/user-guide/workflows.md", "docs/migration/v0.6.md")
SETUPS = {  # named preludes migrated verbatim from docs/snippets.yml `setup:` entries
    "animation_stub": "...",       # snippets.yml docs_animation_quick_start
    "annotation_result": "...",    # snippets.yml docs_video_annotation_use_results
}
_FENCE = re.compile(r"^(?P<indent>[ \t]*)```python[ \t]*\n(?P<body>.*?)^(?P=indent)```[ \t]*$", re.M | re.S)
_MARKER = re.compile(r"<!--\s*docs-test:\s*(?P<kind>run|skip|raises)\b(?P<arg>[^>]*?)\s*-->")


@dataclass(frozen=True)
class DocBlock:
    path: str
    line: int  # 1-based line of the first code line
    code: str
    kind: str | None
    arg: str


def collect_blocks(text: str, path: str) -> list[DocBlock]:
    blocks = []
    for m in _FENCE.finditer(text):
        above = text[: m.start()].rstrip().rsplit("\n", 1)[-1]
        mark = _MARKER.search(above)
        blocks.append(DocBlock(path, text.count("\n", 0, m.start()) + 2, textwrap.dedent(m["body"]),
                               mark["kind"] if mark else None, mark["arg"].strip() if mark else ""))
    return blocks


def _run_document(path: str, *, opt_in: bool, seeded: bool) -> int:
    namespace: dict[str, object] = _fixture_namespace() if seeded else {}
    executed = 0
    for block in collect_blocks((ROOT / path).read_text(encoding="utf-8"), path):
        if block.kind == "skip":
            assert block.arg, f"{path}:{block.line}: 'docs-test: skip' needs a reason"
            continue
        if opt_in and block.kind != "run":
            continue
        if setup := re.search(r"setup=(\w+)", block.arg):
            exec(SETUPS[setup[1]], namespace)
        code = compile("\n" * (block.line - 1) + block.code, str(ROOT / path), "exec")
        try:
            if block.kind == "raises":
                name = block.arg.split()[0]
                expected = getattr(builtins, name, None) or getattr(neurospatial, name)
                with pytest.raises(expected):
                    exec(code, namespace)
            else:
                exec(code, namespace)
        finally:
            plt.close("all")
        executed += 1
    return executed


@pytest.mark.parametrize("path", sorted(RUN_ALL))
def test_documented_examples_run(path, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _run_document(path, opt_in=False, seeded=RUN_ALL[path]) > 0


@pytest.mark.parametrize("path", OPT_IN)
def test_opted_in_examples_run(path, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _run_document(path, opt_in=True, seeded=True) > 0
```

- Add the markers so that every manifest entry is reproduced: each `index: k` becomes a `run` marker on that block, and each `skip:` becomes a `skip` marker. The package-docstring entry is covered by `FLAGSHIP` (4.6).
- Then delete `scripts/test_doc_snippets.py`, `docs/snippets.yml`, `tests/test_doc_snippets_helper.py`, and the two snippet steps in `.github/workflows/test_docs.yml:57-65`. Keep its `--doctest-modules` step.
- **Do not mark these tests `slow`.** CI never runs `slow`, so the guard would be inert. Measured on `main`:
  - README blocks 0–6 take ≈ 5 s;
  - the quickstart takes ≈ 3 s;
  - CLAUDE.md takes < 3 s with fixtures;
  - README block 8 takes 134 s and is skip-marked (napari and ffmpeg).

### 4.8 Stale examples on `main` to fix

Found by running every block. `RUN_ALL` blocks were run cumulatively; CLAUDE.md and QUICKSTART were run with the fixture namespace.

| Location | Failure | Fix |
| --- | --- | --- |
| `CLAUDE.md:44` (argument-order pseudo-code) | `SyntaxError` | Fence it as `text`. |
| `CLAUDE.md:233-250` pattern 1 | `env.neighbors(bin_idx)` passes the array from `bin_at`, giving the deep dimensionality error. With 100 random points and 2 cm bins, `bin_at` also returns `-1`. | Use `rng.uniform(0, 100, (5000, 2))` and `env.neighbors(int(bin_idx[0]))`. |
| `CLAUDE.md:289` pattern 3 | Needs napari and ffmpeg (`n_workers=4` spawns processes). | `skip` marker with a reason. |
| `CLAUDE.md:459` gotcha 2 | The comment says `RuntimeError`, but the real error is `ValueError [E1006]`. | Fix the comment; add `raises ValueError`. |
| `CLAUDE.md:475, :489-495, :503-510` gotchas 3–5 | Undefined `data`, `new_point`, `position`, and no `'goal'` region. | Use `positions`; add `env.regions.add("goal", point=(50.0, 50.0))` and `new_point = (60.0, 60.0)`; use `positions[:1]`. Mark the ❌ blocks `raises TypeError` / `raises AttributeError`. |
| `CLAUDE.md` heading "Error: `RuntimeError: Environment must be fitted`" | Stale type. | Change it to `ValueError: [E1006]`. |
| `.claude/QUICKSTART.md:86` `from_graph` | Edges lack the required `distance` attribute. | Add `distance=50.0` to the edges; mark `run`. |
| `.claude/QUICKSTART.md:669` `visible_cues(observer_position=…)` | The real signature is `(env, position, heading, cue_positions, *, fov)`. | Rewrite the call; mark `run`. |
| `.claude/QUICKSTART.md:785` `env.point_in_region` | No such method exists. | Use `env.regions` / `env.bins_in_region` (check which method is current); mark `run`. |

### 4.9 User-facing documentation

- `CHANGELOG.md` `[Unreleased]`:
  - `NeurospatialError` and the hierarchy;
  - `RegionNotFoundError` now raised by segmentation, which is also a `ValueError`;
  - the `BinIndexOutOfRangeError` type change at `neighbors`;
  - the coarse-`bin_size` and window-span warnings;
  - `ResourceWarning` → `UserWarning`;
  - the removed `detect_region_crossings` old order.
- `docs/errors.md`:
  - a "Catching neurospatial errors" section, showing `except NeurospatialError`, the stdlib-first bases, and the `Fix:` convention;
  - update the E1004 and E1006 entries.
- README "Your First Place Field": drop `method="diffusion_kde"` (it is the default) so the call is the shortest correct one. Point the decoding paragraph at `decode_position` / `decode_session` as they exist after Phase 3.
- `docs/getting-started/quickstart.md`: in "Bringing your own data", add one sentence saying that recording pauses longer than `max_gap=0.5` s are excluded automatically, and that `epochs=` restricts the analysis (if Phase 3 did not already add it).
- `CLAUDE.md` and `.claude/DEVELOPMENT.md`: describe the `docs-test` markers and the `uv run pytest tests/docs` command in place of `scripts/test_doc_snippets.py`.
- **One changelog source.** `docs/changelog.md` is a hand-maintained copy that has drifted from `CHANGELOG.md`. Replace its body with the snippet include `--8<-- "CHANGELOG.md"`; `pymdownx.snippets` is already enabled in `mkdocs.yml`, with `check_paths: true`. Before replacing it, move any entries that exist only in `docs/changelog.md` into `CHANGELOG.md`. Verify the result with `uv run mkdocs build --strict`.
- `CLAUDE.md` v0.6 naming contract: delete the sentence saying the old `detect_region_crossings` positional order "remains supported as a compatibility form and warns" (it no longer exists after 4.5).

## Deliberately not in this phase

- **Rewriting all 454 raise sites.** Only the 4.3 list and the 156 relabels. A blanket rewrite stalls the phase; later phases follow the contract for code they touch.
- **The remaining `.claude/QUICKSTART.md` blocks and all other user-guide blocks.** Most of their failures are undefined fragment names (`neuron1_spikes`, `goal_position`, `trials`, …), not API drift. Of 38 QUICKSTART blocks, 3 genuine API failures are fixed in 4.8. Those files stay opt-in, and converting them is a follow-up.
- **Namespace moves, `__all__` curation and the API snapshot** (Phase 6). Phase 6 updates `FLAGSHIP` paths if it moves a name; the tests fail loudly if it doesn't.
- **Population-silence and gap warnings** (Phase 3); **classifier wording** (Phase 5); **summary/repr/overwrite fixes** (Phase 7).
- **Archive-only items** that are absent on `main` (verified): `_PopulationTypeError`, `remediation` fields, `TemporalSupport`, and `Session.from_arrays` support errors. Nothing to port.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_exceptions.py::test_every_library_error_is_a_neurospatial_error` | For all 7 classes, `issubclass(cls, NeurospatialError)`, and `cls.__mro__[1]` is the stdlib type listed in 4.1. |
| `test_exceptions.py::test_region_not_found_prints_fix_unquoted` | `str(RegionNotFoundError("hom", available=["home"]))` contains `"\nFix: pass region_name='home'"`; it does not start with `'`; `isinstance(exc, ValueError)` and `isinstance(exc, KeyError)` are both True. |
| `test_exceptions.py::test_no_import_cycle` | In a fresh subprocess, `import neurospatial.environment.queries` succeeds before `neurospatial._exceptions` is imported. |
| `test_first_run_errors.py::test_missing_env_names_the_call` | `compute_spatial_rate(spikes, times, positions)` raises `TypeError` matching `"expects an Environment"`, and its last line starts with `"Fix: "`. |
| `test_first_run_errors.py::test_1d_positions_on_2d_env` | Raises `ValueError` matching `r"shape \(1800,\).*2-D"` and `"Fix:"`, not `"grid_edges"`. |
| `test_first_run_errors.py::test_swapped_times_positions` | `compute_spatial_rate(env, s, positions, times)` raises with `"did you pass positions before times"`. Several problems in one call are listed in one message: a length mismatch plus a dimension mismatch gives 2 bullet lines. |
| `test_first_run_errors.py::test_neighbors_of_bin_at_output` | `env.neighbors(env.bin_at([[50, 50]]))` raises `ValueError` matching `r"int\(bin_idx\[0\]\)"`. |
| `test_first_run_errors.py::test_coarse_bin_size_warns` | `bin_size=500` on 80 cm data gives exactly one `UserWarning` containing `"bin_size=500"` and `"Fix:"`. `bin_size=2.0` gives none. A 200 × 5 cm track with `bin_size=5.0` gives none. |
| `test_first_run_errors.py::test_large_grid_warning_is_visible` | The grid-size warning category is `UserWarning`, so it is visible under default filters. |
| `test_first_run_errors.py::test_psth_window_in_ms_warns` | `window=(-500, 1000)` warns with `"window=(-0.5, 1.0)"`. `(-1.0, 2.0)` does not warn. `(1.0, -1.0)` raises with `"Fix:"`. |
| `test_first_run_errors.py::test_segmentation_unknown_region` | `detect_laps(..., start_region="home")` on an env without regions raises `RegionNotFoundError` containing `"env.regions.add('home'"`. |
| `test_first_run_errors.py::test_rewritten_sites_teach` (parametrized over the 4.3 table) | Each triggering call's `str(exc)` contains a line starting `"Fix: "`. The NWB `KeyError` sites are checked for `"Fix:"` anywhere. |
| `tests/docs/test_docstring_sections.py` | 0 missing sections and 0 undocumented parameters across root callables plus `FLAGSHIP`. |
| `tests/docs/test_flagship_docstrings.py` | 40 `FLAGSHIP` ids pass. There are 0 disallowed `+SKIP`s. The 4 `NWB_FLAGSHIP` ids pass under `-m nwb`. |
| `tests/docs/test_executable_docs.py` | 3 `RUN_ALL` files and 8 `OPT_IN` files pass. Every skip marker has a reason. |
| `tests/docs/test_executable_docs.py::test_collector_markers` | On a synthetic Markdown string, `collect_blocks` returns the right `kind`, `arg`, `line` and dedented code for fences that are unmarked, `skip`, `raises`, `run setup=x` or indented. A `skip` without a reason fails. (This replaces `test_doc_snippets_helper.py`.) |

No test in this phase is marked `slow`. The whole `tests/docs` directory must finish in < 60 s on one core; measure it and record the time in the PR.

## Fixtures

- `_fixture_namespace()` in `tests/docs/test_executable_docs.py` returns a fresh dict on each call. It contains:
  - `np`, `Environment`, `rng = default_rng(0)`;
  - `times`, 60 s at 30 Hz;
  - `positions = 50 + 40 * [sin(2πt/20), cos(2πt/13)]` (cm), and `env = from_samples(positions, bin_size=4.0, units="cm")`;
  - `spike_times` (300 uniform spikes);
  - `spike_times_list` (300, 150 and 450 spikes);
  - `headings` (velocity angle);
  - `object_positions = [[50, 50], [75, 25]]`;
  - `reward_times = [10, 25, 40]`;
  - `fields`, a `(20, n_bins)` uniform array, with `frame_times = arange(20) / 30` and `trajectory = positions[:20]`.

  It uses plain NumPy, not the simulator, so Phase 6's simulation changes cannot break it.
- The error tests reuse that trajectory, built in a module-level fixture in `tests/test_first_run_errors.py`. They need no real data.
- Docstring examples build their own ≤ 60 s simulated data inline.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
