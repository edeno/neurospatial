---
paths:
  - "src/neurospatial/animation/**"
  - "tests/animation/**"
  - "scripts/*napari*"
---

# Animation code

- Overlay data is in environment `(x, y)` coordinates. Never swap axes or flip y
  for napari; the backends convert.
- Every backend must draw each overlay type or warn explicitly that it cannot
  (the HTML backend shows the pattern), and overlay defaults must be valid in
  every backend.
- Napari tests need a display and serial execution:
  `uv run pytest -n 0 -m "napari and not slow" tests/animation`.
- `NAPARI_PERFMON` takes `1` or the path of a config file such as
  `scripts/perfmon_config.json`, not an output path, despite older examples in
  this repository.
