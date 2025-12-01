# Development Guide

Commands and workflows for developing neurospatial.

**IMPORTANT: All commands MUST be prefixed with `uv run`. Run from project root.**

---

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test module
uv run pytest tests/test_environment.py

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=src/neurospatial

# Run specific test function
uv run pytest tests/test_environment.py::test_function_name -v

# Run doctests (validate docstring examples)
uv run pytest --doctest-modules src/neurospatial/

# Run performance benchmarks (slow tests)
uv run pytest -m slow -v -s

# Skip performance benchmarks
uv run pytest -m "not slow"

# Run NWB tests (requires nwb-full extra)
uv run pytest tests/nwb/ -v
```

---

## Code Quality

```bash
# Run ruff linter
uv run ruff check .

# Run ruff formatter
uv run ruff format .

# Both check and format
uv run ruff check . && uv run ruff format .

# Type checking with mypy
uv run mypy src/neurospatial/

# Check specific file
uv run mypy --ignore-missing-imports src/neurospatial/environment/fields.py
```

---

## Environment Setup

```bash
# Sync dependencies (uv handles this automatically)
uv sync

# Add dependency
uv add package-name

# Add dev dependency
uv add --dev package-name
```

---

## Running Code

```bash
# Run script
uv run python path/to/script.py

# Interactive Python
uv run python

# IPython (recommended for exploration)
uv run ipython
```

---

## Git Workflow

### Committing Changes

**Only create commits when requested by user.**

```bash
# Check status
uv run git status

# Check diff
uv run git diff

# Check recent commits (for message style)
uv run git log --oneline -5

# Stage files
git add <files>

# Commit with conventional commit message
git commit -m "feat(scope): description

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Conventional Commit Format:**

- `feat(scope): description` - New features
- `fix(scope): description` - Bug fixes
- `docs(scope): description` - Documentation changes
- `test(scope): description` - Test additions/fixes
- `chore(scope): description` - Maintenance tasks

### Creating Pull Requests

**Only when requested by user.**

```bash
# Check branch status
git status

# View all commits in this branch
git log main..HEAD --oneline

# View diff from main
git diff main...HEAD

# Push to remote (if needed)
git push -u origin branch-name

# Create PR with gh CLI
gh pr create --title "Title" --body "$(cat <<'EOF'
## Summary
- Bullet point 1
- Bullet point 2

## Test plan
- [ ] Test item 1
- [ ] Test item 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Documentation Style

### NumPy Docstring Format (REQUIRED)

**All docstrings MUST follow NumPy format.**

**Structure:**

```python
def function_name(param1, param2):
    """
    Short one-line summary ending with a period.

    Optional longer description providing more context about what the
    function does, its behavior, and any important implementation details.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2. Default is None.

    Returns
    -------
    return_type
        Description of return value.

    Raises
    ------
    ValueError
        Description of when ValueError is raised.

    See Also
    --------
    related_function : Brief description of relation.

    Notes
    -----
    Additional technical information or mathematical details.

    Examples
    --------
    >>> result = function_name(arg1, arg2)
    >>> print(result)
    expected_output
    """
```

**Key guidelines:**

1. **Section headers**: Use underlines with dashes matching section name length
2. **Type annotations**: Include types after parameter names with colon separator
3. **Section order**: Parameters → Returns → Raises → See Also → Notes → Examples
4. **Blank lines**: One blank line between sections
5. **Examples**: Use `>>>` for interactive examples, show expected output
6. **Cross-references**: Use backticks for code elements: `Environment`, `bin_centers`
7. **Arrays**: Specify shape in type: `NDArray[np.float64], shape (n_samples, n_dims)`

**Resources:**

- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [numpydoc package](https://github.com/numpy/numpydoc) for validation

---

## Performance Profiling

### Napari Performance Monitoring

Profile animation backends with napari's perfmon:

```bash
# Enable perfmon with environment variable
NAPARI_PERFMON=1 uv run python your_script.py

# Output trace to file
NAPARI_PERFMON=/tmp/perfmon.json uv run python your_script.py
```

**View traces:**

- Chrome: `chrome://tracing` and drag-drop JSON
- Speedscope: <https://www.speedscope.app/>

**Programmatic timing:**

```python
from napari.utils.perf import perf_timer, add_instant_event

with perf_timer("my_operation"):
    expensive_function()

add_instant_event("checkpoint_reached")
```

**Key hotspots in this codebase:**

- `_build_skeleton_vectors` - skeleton overlay construction
- `_render_bodypart_overlay` - bodypart rendering per frame
- `_render_head_direction_overlay` - head direction arrows
- Layer update callbacks during playback

See: <https://napari.org/stable/howtos/perfmon.html>

---

## Pre-commit Hooks

**Pre-commit hooks run automatically on commit.**

**If hooks fail:**

```bash
# Option 1: Let hooks auto-fix, then commit again
git commit -m "message"  # Hooks auto-fix
git add .  # Stage fixes
git commit -m "message"  # Commit again

# Option 2: Manually fix before committing
uv run ruff check . && uv run ruff format .
git add .
git commit -m "message"
```

**Hook configuration:** `.pre-commit-config.yaml`

---

## Type Checking with Mypy

**Mypy runs in pre-commit hooks and should pass without errors.**

**Configuration:** `pyproject.toml` - `[tool.mypy]` section

**Guidelines:**

1. **Prefer proper typing over suppressions** - Add type hints when possible
2. **Mixin methods should be typed** - Use `self: "Environment"` annotation
3. **Avoid skipping mypy** - Let pre-commit run mypy normally

**Run mypy manually:**

```bash
# Check all files
uv run mypy src/neurospatial/

# Check specific file
uv run mypy --ignore-missing-imports src/neurospatial/environment/fields.py
```

**For mixin type annotation patterns, see [PATTERNS.md - Mypy Type Checking](PATTERNS.md#mypy-type-checking-requirements).**

---

## Creating New Layout Engines

1. Implement the `LayoutEngine` protocol in `src/neurospatial/layout/engines/`
2. Populate required attributes in your `build()` method
3. Add to `_LAYOUT_MAP` in [layout/factories.py](../src/neurospatial/layout/factories.py)
4. Ensure graph nodes/edges have mandatory metadata (see [PATTERNS.md - Graph Metadata](PATTERNS.md#graph-metadata-requirements))
5. Test boundary detection works with your layout
6. Add tests in `tests/layout/` following existing patterns

---

## Memory Profiling

For large grid memory issues:

```python
# Check estimated memory usage
from neurospatial.layout.engines.regular_grid import _estimate_grid_memory_mb

estimated_mb = _estimate_grid_memory_mb(grid_shape, dtype=np.float64)
print(f"Estimated memory: {estimated_mb:.1f} MB")
```

**Memory warnings trigger at 100MB.** See [TROUBLESHOOTING.md - ResourceWarning](TROUBLESHOOTING.md#resourcewarning-creating-large-grid-v021) for fixes.
