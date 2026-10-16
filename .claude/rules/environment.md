---
paths:
  - "src/neurospatial/environment/**"
  - "src/neurospatial/layout/**"
  - "src/neurospatial/geometry/**"
  - "src/neurospatial/ops/**"
---

# Environment, geometry, layout and ops code

The design intent is in the "Environment architecture" section of
`docs/contributing.md`. These steps fail silently when skipped:

- A new built-in layout engine needs a `LayoutType` member and a
  `_LAYOUT_MAP` entry in `layout/factories.py`, an export from
  `layout/__init__.py`, a `_data_parameters` class attribute naming its data
  arguments, and an entry in `MINIMAL_BUILD_PARAMS` in
  `tests/layout/test_layout_engine.py`. The protocol-adherence test skips
  engines missing from that dict instead of failing.
- A new `Environment` factory needs `@keyword_hint()` (from
  `neurospatial._keyword_hint`, under `@classmethod`) and an entry in
  `_DATA_ARGUMENTS` in `tests/environment/test_factory_arguments.py`; a
  factory missing from it is not checked, and one without the decorator
  answers an option passed by position with Python's generic error.
- Before and after a refactor of environment behavior, record and compare:
  `uv run python scripts/environment_baseline.py record|compare` (see
  `docs/contributing.md` for the full command).
