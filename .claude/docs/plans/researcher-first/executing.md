# Executing a phase

[← back to PLAN.md](PLAN.md)

Read this before starting **any** phase. It is the single definition of how a phase is carried out and when it is done. Phase files contain only what is specific to that phase.

## Before you start

1. **Check status.** Read the `Status:` line in [PLAN.md](PLAN.md). Confirm that every phase your phase lists under "Requires" is marked done.
2. **Check for added tasks.** After Phase 4b, the researcher-workflow checkpoint ([overview → Rollout Strategy](overview.md#rollout-strategy)) may have added tasks to later phases. Look for a "Checkpoint additions" section at the top of your phase file.
3. **Branch.**
   ```
   git switch feat/researcher-first && git pull
   git switch -c phase-<id>-<slug>
   ```
   For example, `phase-3a-time-windows-core`.
4. **Environment.** Run `uv sync --all-extras`. Tests for optional integrations (NWB, xarray, pynapple, JAX) **skip silently** without their extras, which gives a false green.

## While you work

- **Commits.** One commit per task, using Conventional Commits (`fix(scope):`, `feat(scope):`, `test:`, `docs:`, `refactor(scope):`).
- **Regression tests first.** For a bug fix, write the regression test first. Run it against the unmodified code and record the failure in the commit body, then fix.
- **CHANGELOG.** Each commit that changes user-visible behavior appends its own bullet under `## [Unreleased]` in `CHANGELOG.md`, in `### Fixed`, `### Changed`, `### Added` or `### Removed`. The first commit that needs a section creates it. A phase's final "documentation" task is a check that every bullet is present, not a separate batch.
- **Line numbers drift.** References are from `da631a47` unless a phase says otherwise. Earlier phases move code, so re-locate by symbol or by message text, not by line number.
- **Probe numbers are evidence, not targets.** If a number quoted in the plan does not reproduce, stop. Record the measured value in the commit body and the PR description. Do not tune a test to match the plan's number.
- **When the plan and reality disagree.** That covers a snippet contradicting a test or contract, a referenced symbol that does not exist, and a task that would change a contract or a scientific default.
  - Stop, write down the conflict, and ask the maintainer.
  - Do not silently pick one interpretation.
  - Mechanical drift you resolve yourself, such as a renamed file or shifted lines, goes in the PR description.
- **Test runs.** Use `-n 4`, not `-n auto`: other jobs may share the machine, and CPU starvation looks like a hang.
  - The default selection in `pytest.ini` is `-m "not slow and not napari"`.
  - Any explicit `-m` replaces it, so always include `and not napari` yourself.
- **Running examples or docstrings by hand.** Run them from a temporary directory. Several of them write files (videos, HTML, `.dat`) into the current directory.

## Definition of done

All of these pass locally before you open the PR.

**Lint and format:**

```
uv run ruff check . && uv run ruff format --check .
```

**Type checking.** CI runs mypy on several platforms:

```
uv run --extra dev mypy src/neurospatial/
uv run --extra dev mypy src/neurospatial/ --platform win32
```

**Default test suite.** Inspect the skip summary from `-rs`. Every skip must be expected, never a missing extra.

```
uv run pytest -n 4 -rs
```

**Slow tests.** Phase 1 adds the CI job that runs these.

```
uv run pytest -m "slow and not napari" -n 4
```

**Doctests:**

```
uv run pytest --doctest-modules src/neurospatial/ -n 0
```

**Executable documentation:**
- before Phase 4b: `uv run python scripts/test_doc_snippets.py`;
- from Phase 4b on: `uv run pytest tests/docs -n 4`.

**Docs build.** Only when docs, docstrings or the changelog changed:

```
uv run --extra docs mkdocs build --strict
```

**API snapshot.** From Phase 6a on, `uv run pytest tests/test_public_api_snapshot.py`. If the API change is intended, regenerate the snapshot as [API snapshot](shared-contracts.md#api-snapshot) describes, and make sure the diff appears in the PR.

**Independent review.** Follow the phase's **Review** block (dispatch `code-reviewer` against the diff) and address its findings.

## Opening the PR

This plan authorizes the executor to:
- push the phase branch;
- open a PR into `feat/researcher-first` with `gh pr create --base feat/researcher-first`.

It does **not** authorize:
- pushing to `main`;
- force-pushing a shared branch;
- merging. The maintainer merges.

Verify CI with `gh pr checks <number>`. PR runs are keyed by the head branch, so `gh run list --branch feat/researcher-first` does not show them.

The PR description lists:
- the tasks done;
- deviations from the plan, with the measured numbers;
- stale tests that were fixed or marked `xfail(strict=True)`, each with a reason;
- anything deferred.

## After merge

Update the `Status:` line in [PLAN.md](PLAN.md) to `Phase <id> done (<merge sha>, PR #<n>); next: Phase <id>`, and commit it on `feat/researcher-first` (`docs(plan): mark phase <id> done`).
