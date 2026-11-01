---
paths:
  - "tests/**"
---

# Tests

- Register new markers in `[tool.pytest.ini_options]` in `pyproject.toml`
  (`--strict-markers`). The default run uses `-n auto` and deselects `slow` and
  `napari`.
- In simulation tests, seed the stimulus RNG (positions, headings) and
  `generate_poisson_spikes(..., seed=...)` differently. A shared seed correlates
  spike acceptance with the stimulus and silently corrupts the spike train.
- In a git worktree, `uv run pytest` can import the main checkout's `src/`
  through its `.venv`. Run
  `PYTHONPATH=<worktree>/src <main>/.venv/bin/python -m pytest ...` and confirm
  `neurospatial.__file__` points into the worktree before trusting a result.
