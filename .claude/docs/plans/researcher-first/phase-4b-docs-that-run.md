# Phase 4b — Complete docstrings and documentation that runs in CI

**Requires:** Phase 4a merged. 4b executes examples that depend on the four-array calls and on 4a's error types (`raises` markers name them).

[← back to PLAN.md](PLAN.md) · [executing a phase](executing.md) · [overview](overview.md) · [shared contracts](shared-contracts.md) · previous: [4a](phase-4a-errors.md)

**After this phase merges,** the researcher-workflow checkpoint runs before Phase 5a starts ([overview → Rollout Strategy](overview.md#rollout-strategy)). Its journeys work only from the public documentation that this phase makes executable, so leave every doc you touch in a state a newcomer can follow without reading source.

**Line numbers** below were read on `main` at `da631a47`. Phases 1–4a land first and shift them, so find each site by its quoted text or symbol, not by the number alone.

**Inputs to read first:**

- `scripts/test_doc_snippets.py`, `docs/snippets.yml`, `tests/test_doc_snippets_helper.py` and `.github/workflows/test_docs.yml` — the current snippet runner. It is a 27-entry manifest keyed by block *index*, so inserting a block silently retargets an entry. All 25 non-skipped entries pass on `main`. **Each entry runs in its own subprocess** (`run_snippet`, `scripts/test_doc_snippets.py:236-260`) with `MPLBACKEND=Agg`, so nothing one snippet defines or monkeypatches reaches another. The in-process replacement below must keep that isolation explicitly.
- `pytest.ini` — this overrides `pyproject.toml`'s pytest table. `addopts` deselects `slow` and `napari`. An explicit `-m` replaces that selection.
- `.github/workflows/test_nwb.yml` — runs `uv run pytest tests/nwb tests/test_recording.py -n 0 -q` with the NWB extra. Only `tests/nwb` relies on `importorskip`; the `nwb` marker is used by few tests, so adding `-m nwb` to that command would deselect most of the NWB suite.
- **Files earlier phases already changed** (re-locate every site by its quoted text):
  - `.github/workflows/test_docs.yml` and `test_nwb.yml`: Phase 1 Task 1 added `feat/researcher-first` to their triggers. This phase edits only their steps.
  - `CLAUDE.md` and `docs/snippets.yml`: Phase 3e changed the `heading_from_velocity` calls (pattern 7).
  - `CHANGELOG.md`: Phases 1–4a appended under the first `## [Unreleased]` (`:3`). A second one sits at `:494`.
  - `tests/behavior/test_detect_region_crossings_argorder.py`: Phase 3d may have added time-window cases; keep them.

**Contracts referenced:**

- [Error-message contract](shared-contracts.md#error-message-contract) — the `raises` markers in CLAUDE.md's gotchas name the types 4a established.
- [Time-window semantics](shared-contracts.md#time-window-semantics) — the docs examples exercise Phase 3's defaults. This phase does not change them.

**Designs referenced:** none.

## Tasks

### 4.5 Docstring completeness

Most docstrings are already complete. Measured over every `__all__` function in the twelve snapshot namespaces, plus the `Environment` factories and flagship methods, counting both "Parameters" and "Other Parameters":

- Every `neurospatial.encoding` and `neurospatial.simulation` function has Parameters, Returns and Examples. (The archive branch's "no Parameters section" finding does not apply to `main`.)
- **No Parameters section:** `Environment.from_polar_egocentric` (it delegates to `EgocentricPolarEnvironment.create`). Copy that method's Parameters section in.
- **Undocumented parameters:**
  - `Environment.from_samples` `**layout_specific_kwargs`;
  - `Environment` constructor `regions` (the other constructor parameters are documented in `__init__`);
  - `Regions` constructor `items`;
  - `behavior.detect_region_crossings` `arg3`/`arg4`. Delete the old-order compatibility dispatch (`segmentation.py:411-457`, plus the docstring note at `:340-346`; it warns that it will be "removed in 0.7"). The signature becomes `(position_bins, times, env, *, region_name, direction="both")`, per decision 1. In `tests/behavior/test_detect_region_crossings_argorder.py`:
    - delete `test_old_positional_order_warns` and `test_old_and_new_order_identical`; keep the canonical-order tests;
    - `test_new_order_with_positional_region_name_raises_clear_type_error` matches `"keyword-only"`, which was the dispatch's own message. Without the dispatch, Python's message is `detect_region_crossings() takes 3 positional arguments but 4 were given`. Change the match to `"takes 3 positional arguments"`.
- **No Examples section** (`load_session` also lacks one, but Phase 6c deletes it, so it is exempt here):
  - `decode_session_summary`;
  - `decode_position_summary`;
  - `behavior.angular_efficiency`;
  - `behavior.subgoal_efficiency`;
  - the root classes `Environment`, `CompositeEnvironment`, `Region` and `Regions`.
- **No Returns section:** `events.validate_events_dataframe`. Out of scope, but listed for the test: `stats.shuffle_*` and `generate_*` use Yields-style returns, and 8 of 8 `neurospatial.regions` functions lack Examples. The test covers only the root and flagship set (4.6), so those remain follow-ups.
- New `tests/docs/test_docstring_sections.py`, over every root-exported callable and every entry of `FLAGSHIP` (4.6). The rules differ for functions and classes:
  - **Functions and methods:** Parameters (when the signature has parameters other than `self`/`cls`), Returns (unless the return annotation is `None`; `Yields` also counts) and Examples. Every signature parameter except `self`/`cls` is named under Parameters or Other Parameters.
  - **Classes:** Examples, and every constructor parameter named under Parameters, Other Parameters or Attributes, in the class docstring or the `__init__` docstring (resolved through the MRO). Dataclass results document their fields under Attributes. Classes are never checked for Returns.
  - **Exception classes** (subclasses of `BaseException`) are skipped; `tests/test_exceptions.py` covers them.
  - `_EXEMPT = frozenset({"neurospatial.load_session"})`. The test also asserts that every exempt name still resolves, so Phase 6c, which deletes `load_session`, must delete the entry in the same change.

  Under these rules, a probe at `da631a47` flags exactly: `Environment` (no Examples; `regions`), `CompositeEnvironment` (no Examples), `Region` (no Examples), `Regions` (no Examples; `items`), `load_session` (exempt), `Environment.from_samples` (`layout_specific_kwargs`) and `detect_region_crossings` (`arg3`, `arg4`). If your count differs after Phases 1–4a, record it in the PR. Parse with a 15-line regex section splitter, not `numpydoc`: that is only installed transitively through napari, which CI's `--extra dev` does not install.

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

- **`tests/docs/conftest.py`** calls `matplotlib.use("Agg")` at import. It replaces the manifest's `MPLBACKEND=Agg`, so a local run never opens a window (`plt.show()` appears in README and quickstart blocks).
- **CI for the NWB examples.** Add a **separate step** to `.github/workflows/test_nwb.yml`, after the existing one: `uv run pytest tests/docs/test_flagship_docstrings.py -m nwb -n 0`. Do not append to the existing command: its `-m nwb` would replace `pytest.ini`'s selection for `tests/nwb` too and deselect every unmarked NWB test. The default `test.yml` job installs only `--extra dev`, so without this step `NWB_FLAGSHIP` would be skipped by `importorskip` and never enforced.
- **Module doctests must not write into the repository.** Once the `+SKIP`s are gone, `test_docs.yml`'s `pytest --doctest-modules src/neurospatial/` runs `to_file`, `animate_fields(save_path=…)` and similar examples in the checkout. Add `src/conftest.py` (beside the package, not inside it, so it never ships in the wheel; `pyproject.toml` packages only `src/neurospatial`):

  ```python
  import pytest


  @pytest.fixture(autouse=True)
  def _doctest_in_tmp_path(request: pytest.FixtureRequest, tmp_path, monkeypatch) -> None:
      """Run each module doctest in its own temporary working directory."""
      if isinstance(request.node, pytest.DoctestItem):
          monkeypatch.chdir(tmp_path)
  ```

  Verified with a scratch package: pytest collects a `conftest.py` in the parent directory of the package under `--doctest-modules <pkg>/`, and the doctest's working directory becomes its `tmp_path`.

### 4.7 Executable documentation (replaces the snippet manifest)

New file `tests/docs/test_executable_docs.py`:

- **Collection.** It collects ```` ```python ```` blocks itself, so block indices disappear. Markdown carries the markers, on the line directly above a fence:
  - `<!-- docs-test: skip <reason> -->` (the reason is required);
  - `<!-- docs-test: raises <ExceptionName> -->` (used for "❌ Wrong" gotcha examples);
  - `<!-- docs-test: run [setup=<manifest id>] -->` (opt-in files only).
- **File modes.**
  - `RUN_ALL` files execute every block.
  - `OPT_IN` files execute only `run` blocks. These are the pages the manifest covered.
- **Namespaces.**
  - README and the getting-started quickstart get **no** pre-seeded names and run **cumulatively, one namespace per file**, the way a reader would. They must be self-contained, because users copy them. (Probe at `da631a47`: README blocks 0–6 and all four quickstart blocks pass this way.)
  - CLAUDE.md and every opt-in block run **each block in a fresh `_fixture_namespace()`**. CLAUDE.md's blocks are independent patterns, and pattern 1 reassigns `positions` (100 points) and `env`. Run cumulatively at `da631a47`, patterns 2, 4, 8 and 9 then fail with `times length (1800) must match positions length (100)` and `Fields array has shape (20, 245) but expected (20, 98)`; with a fresh namespace per block they pass. A fresh namespace per opt-in block also reproduces the manifest's one-subprocess-per-entry isolation.
- **Setups.** `SETUPS` has **one entry per manifest `setup:` that defines names, migrated verbatim and keyed by its manifest id**. That is 11 entries: `docs_animation_quick_start`, `docs_video_annotation_use_results`, `docs_trajectory_region_crossings`, `quickstart_vte_session`, `quickstart_ovc_classify_single`, `quickstart_view_classify`, `quickstart_circular_basis_metrics`, `quickstart_overlay_block`, `quickstart_events_glm_regressors`, `workflows_decode_session_summary_streaming` and `workflows_batch_processing_compute_spatial_rates`. The other 11 manifest setups only set `MPLBACKEND=Agg`, which `tests/docs/conftest.py` now does. The block's marker names its setup, for example `<!-- docs-test: run setup=quickstart_vte_session -->`.
- **Monkeypatches stay inside one test.** Two setups (`docs_animation_quick_start`, `quickstart_overlay_block`) assign `Environment.animate_fields = <stub>`. In a subprocess that was harmless; in-process it would leak into every later test on the same xdist worker, including the `animate_fields` flagship doctest. Both test functions therefore start with `monkeypatch.setattr(Environment, "animate_fields", Environment.animate_fields)`, which registers the original for restoration at teardown.
- **Tracebacks.** Code is compiled with `"\n" * (line - 1) + code` against the real file path, so a traceback points at the Markdown line.

```python
RUN_ALL = {"README.md": False, "docs/getting-started/quickstart.md": False, "CLAUDE.md": True}
OPT_IN = (".claude/QUICKSTART.md", "docs/user-guide/alignment.md", "docs/user-guide/animation.md",
          "docs/user-guide/interoperability.md", "docs/user-guide/trajectory-and-behavioral-analysis.md",
          "docs/user-guide/video-annotation.md", "docs/user-guide/workflows.md", "docs/migration/v0.6.md")
SETUPS = {  # one per docs/snippets.yml `setup:` that defines names; verbatim, keyed by manifest id
    "docs_animation_quick_start": "...",
    "docs_video_annotation_use_results": "...",
    # ... the other 9 ids listed above
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
    shared: dict[str, object] = {}  # unseeded files run cumulatively, like a reader
    executed = 0
    for block in collect_blocks((ROOT / path).read_text(encoding="utf-8"), path):
        if block.kind == "skip":
            assert block.arg, f"{path}:{block.line}: 'docs-test: skip' needs a reason"
            continue
        if opt_in and block.kind != "run":
            continue
        namespace = _fixture_namespace() if seeded else shared  # seeded: fresh per block
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
    monkeypatch.setattr(Environment, "animate_fields", Environment.animate_fields)
    monkeypatch.chdir(tmp_path)
    assert _run_document(path, opt_in=False, seeded=RUN_ALL[path]) > 0


@pytest.mark.parametrize("path", OPT_IN)
def test_opted_in_examples_run(path, monkeypatch, tmp_path):
    monkeypatch.setattr(Environment, "animate_fields", Environment.animate_fields)
    monkeypatch.chdir(tmp_path)
    assert _run_document(path, opt_in=True, seeded=True) > 0
```

- Add the markers so that every manifest entry is reproduced: each `index: k` becomes a `run` marker on that block (with `setup=<id>` where the entry had a name-defining setup), and each `skip:` becomes a `skip` marker. The package-docstring entry is covered by `FLAGSHIP` (4.6).
- Then delete `scripts/test_doc_snippets.py`, `docs/snippets.yml`, `tests/test_doc_snippets_helper.py`, and the two snippet steps in `.github/workflows/test_docs.yml:57-65`. Keep its `--doctest-modules` step. `tests/docs` runs in the default `test.yml` job with the rest of `tests/`.
- **Do not mark these tests `slow`.** The guard must run on every PR. Measured on `main`:
  - README blocks 0–6 take ≈ 5 s;
  - the quickstart takes ≈ 3 s;
  - CLAUDE.md takes < 3 s with fixtures, once pattern 3 is skip-marked (its video export alone took 7.6 s and needs ffmpeg);
  - README block 8 takes 134 s and is skip-marked (napari and ffmpeg).

### 4.8 Stale examples on `main` to fix

Found by running every block: README and the quickstart cumulatively, CLAUDE.md and QUICKSTART with a fresh fixture namespace per block.

| Location | Failure | Fix |
| --- | --- | --- |
| `CLAUDE.md:44` (argument-order pseudo-code) | `SyntaxError` | Fence it as `text`. |
| `CLAUDE.md:233-250` pattern 1 | `env.neighbors(bin_idx)` passes the array from `bin_at`; since 4a this raises the `int(bin_idx[0])` error. With 100 random points and 2 cm bins, `bin_at` also returns `-1`. | Use `rng.uniform(0, 100, (5000, 2))` and `env.neighbors(int(bin_idx[0]))`. |
| `CLAUDE.md:289` pattern 3 | Needs napari and ffmpeg (`n_workers=4` spawns processes). | `skip` marker with a reason. |
| `CLAUDE.md:459` gotcha 2 | The comment says `RuntimeError`, but the real error is `ValueError [E1006]`. | Fix the comment; add `raises ValueError`. |
| `CLAUDE.md:475, :489-495, :503-510` gotchas 3–5 | Undefined `data`, `new_point`, `position`, and no `'goal'` region. | Use `positions`; add `env.regions.add("goal", point=(50.0, 50.0))` and `new_point = (60.0, 60.0)`; use `positions[:1]`. Mark the ❌ blocks `raises TypeError` / `raises AttributeError`. |
| `CLAUDE.md` heading "Error: `RuntimeError: Environment must be fitted`" | Stale type. | Change it to `ValueError: [E1006]`. |
| `README.md` "Simulation > Quick Example" (block 6, `:413-457`) | Runs, but prints a wrong answer: "Detected peak: [14, 70]" for a true center of `[50, 75]`. `bin_size=2.0` on 2000 uniform points leaves holes, the OU walk is trapped (61 of 1381 bins visited), 1308 bins are NaN, and `detected_field.argmax()` is meaningless on NaN. A longer duration does not fix it (300 s: 1216 NaN bins, peak `[32, 76]`). | Use `bin_size=5.0` and print `result.peak_location()`. Measured at 120 s: 0 NaN bins, peak `[50, 75]`. |
| `.claude/QUICKSTART.md:86` `from_graph` | Edges lack the required `distance` attribute. | Add `distance=50.0` to the edges; mark `run`. |
| `.claude/QUICKSTART.md:669` `visible_cues(observer_position=…)` | The real signature is `(env, position, heading, cue_positions, *, fov)`. | Rewrite the call; mark `run`. |
| `.claude/QUICKSTART.md:785` `env.point_in_region` | No such method exists. | Use `env.regions` / `env.bins_in_region` (check which method is current); mark `run`. |

### 4.9b User-facing documentation

Per [executing.md](executing.md#while-you-work), each commit appends its own `CHANGELOG.md` bullet; this task checks they are present and does the rest.

- `CHANGELOG.md` `[Unreleased]`: the removed `detect_region_crossings` old positional order (`Removed`).
- README "Your First Place Field": drop `method="diffusion_kde"` (it is the default) so the call is the shortest correct one. Point the decoding paragraph at `decode_position` / `decode_session` as they exist after Phase 3c.
- `docs/getting-started/quickstart.md`: in "Bringing your own data", add one sentence saying that recording pauses longer than `max_gap=0.5` s are excluded automatically, and that `epochs=` restricts the analysis (if Phase 3e did not already add it).
- `CLAUDE.md` and `.claude/DEVELOPMENT.md`: describe the `docs-test` markers (including `setup=<manifest id>` and that seeded blocks run in a fresh namespace) and the `uv run pytest tests/docs` command in place of `scripts/test_doc_snippets.py`.
- **One changelog source.** `docs/changelog.md` is a hand-maintained copy that has drifted from `CHANGELOG.md`. Replace its body with the snippet include `--8<-- "CHANGELOG.md"`; `pymdownx.snippets` is already enabled in `mkdocs.yml`, with `check_paths: true`. Before replacing it:
  - move any entries that exist only in `docs/changelog.md` into `CHANGELOG.md`;
  - merge the two `## [Unreleased]` headings (`:3` and `:494`) into one, and the two 0.6.0 headings (`:355` `[v0.6.0]`, `:496` `[0.6.0]`) likewise if they describe the same release;
  - make relative links absolute. `CHANGELOG.md:1458` links `](docs/glossary.md)`, which resolves from `docs/changelog.md` to the non-existent `docs/docs/glossary.md` and fails the strict build. Use the published URL (`https://edeno.github.io/neurospatial/glossary/`), which also works when `CHANGELOG.md` is read on GitHub.

  Verify with `uv run --extra docs mkdocs build --strict`.
- `CLAUDE.md` v0.6 naming contract: delete the sentence saying the old `detect_region_crossings` positional order "remains supported as a compatibility form and warns" (it no longer exists after 4.5).

## Deliberately not in this phase

- **Error types, messages and warnings** — Phase 4a.
- **The remaining `.claude/QUICKSTART.md` blocks and all other user-guide blocks.** Most of their failures are undefined fragment names (`neuron1_spikes`, `goal_position`, `trials`, …), not API drift. Of 38 QUICKSTART blocks, 3 genuine API failures are fixed in 4.8. Those files stay opt-in, and converting them is a follow-up.
- **Namespace moves, `__all__` curation and the API snapshot** (Phase 6a). Phase 6a updates `FLAGSHIP` paths if it moves a name; the tests fail loudly if it doesn't.
- **Running the researcher-workflow checkpoint.** It follows this PR; it is not part of it.
- **Population-silence and gap warnings** (Phase 3a); **classifier wording** (Phase 5b); **summary/repr/overwrite fixes** (Phase 7).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/docs/test_docstring_sections.py` | 0 missing sections and 0 undocumented parameters across root callables plus `FLAGSHIP`, under the function and class rules in 4.5. Every `_EXEMPT` name resolves. |
| `tests/docs/test_flagship_docstrings.py` | 40 `FLAGSHIP` ids pass. There are 0 disallowed `+SKIP`s. The 4 `NWB_FLAGSHIP` ids pass under `-m nwb` (the new `test_nwb.yml` step). |
| `tests/docs/test_executable_docs.py` | 3 `RUN_ALL` files and 8 `OPT_IN` files pass. Every skip marker has a reason. Every `setup=` name is a `SETUPS` key, and every `SETUPS` key is used. |
| `tests/docs/test_executable_docs.py::test_collector_markers` | On a synthetic Markdown string, `collect_blocks` returns the right `kind`, `arg`, `line` and dedented code for fences that are unmarked, `skip`, `raises`, `run setup=x` or indented. A `skip` without a reason fails. (This replaces `test_doc_snippets_helper.py`.) |
| `tests/docs/test_executable_docs.py::test_stubs_do_not_leak` | `original = Environment.animate_fields`; inside `with pytest.MonkeyPatch.context() as mp:` apply the same guard and `mp.chdir(tmp_path)`, then run `_run_document("docs/user-guide/animation.md", opt_in=True, seeded=True)`; inside the block `Environment.animate_fields is not original` (the setup's stub was installed), and after it `Environment.animate_fields is original`. |
| `tests/behavior/test_detect_region_crossings_argorder.py` | The two old-order tests are gone; the positional-`region_name` test matches `"takes 3 positional arguments"`. |

No test in this phase is marked `slow`. The whole `tests/docs` directory must finish in < 60 s on one core; measure it and record the time in the PR.

## Fixtures

- `_fixture_namespace()` in `tests/docs/test_executable_docs.py` returns a fresh dict on each call. It contains:
  - `np`, `Environment`, `rng = default_rng(0)`;
  - `times`, 60 s at 30 Hz;
  - `positions = 50 + 40 * [sin(2πt/20), cos(2πt/13)]` (cm), and `env = from_samples(positions, bin_size=4.0, units="cm")` (the same trajectory as 4a's error-test fixture);
  - `spike_times` (300 uniform spikes);
  - `spike_times_list` (300, 150 and 450 spikes);
  - `headings` (velocity angle);
  - `object_positions = [[50, 50], [75, 25]]`;
  - `reward_times = [10, 25, 40]`;
  - `fields`, a `(20, n_bins)` uniform array, with `frame_times = arange(20) / 30` and `trajectory = positions[:20]`.

  It uses plain NumPy, not the simulator, so Phase 6c's simulation changes cannot break it.
- Docstring examples build their own ≤ 60 s simulated data inline.

## Definition of done

Everything in [executing.md → Definition of done](executing.md#definition-of-done) (from this phase on, executable documentation is `uv run pytest tests/docs -n 4`), plus:

- `uv run pytest tests/docs -n 0` under 60 s; record the time in the PR.
- `uv run pytest tests/docs/test_flagship_docstrings.py -m nwb -n 0` with the NWB extra: 4 passed, 0 skipped.
- After `uv run pytest --doctest-modules src/neurospatial/ -n 0` with the `test_docs.yml` ignores, `git status --porcelain` lists no new files.
- `uv run --extra docs mkdocs build --strict`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the snippet runner, manifest, helper test and workflow steps; the `detect_region_crossings` dispatch.
- User-facing documentation listed as tasks is updated, not deferred.
