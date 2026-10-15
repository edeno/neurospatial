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
- Environment records use schema `2.0`: exact binning, lookup, faces, components,
  graph attributes and coordinate metadata are saved. Loading uses
  `BinningLayout` and never runs a layout builder. Schemas `1.0` and `1.1` are
  refused with recreation instructions. Standard one-row ragged columns hold
  arrays; one description JSON attribute maps canonical keys to columns,
  shapes and dtypes. Check scientific outputs against the original after read-back.
