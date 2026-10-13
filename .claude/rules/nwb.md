---
paths:
  - "src/neurospatial/io/nwb/**"
  - "tests/nwb/**"
  - "docs/user-guide/interoperability.md"
---

# NWB code

- Install with `uv sync --extra dev --extra nwb`. CI runs NWB tests serially:
  `uv run pytest tests/nwb -n 0`.
- Import `ndx_events` and `ndx_pose` before opening an `NWBHDF5IO`. Otherwise
  pynwb builds generic classes for those types, and, for example, `read_events`
  raises "'rewards' is not an EventsTable (got EventsTable)".
- Environments read back from NWB only partly round-trip. Non-grid layouts are
  rebuilt as `_ReconstructedLayout`, which maps points far outside the arena to
  the nearest bin instead of -1. Check behavior after read-back rather than
  assuming it matches the original layout.
