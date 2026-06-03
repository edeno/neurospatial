# Phase 15 — NWB spike-unit readers

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Adds `read_units` / `read_spikes` to the NWB reader family (DESIGN-REVIEW High #2). Spikes are the one neural data type with no reader, so every multi-cell workflow currently begins in bare `pynwb`. This reader mirrors `read_position`'s tuple contract and handles the ragged `units` `DynamicTable`.

**Inputs to read first:**

- [io/nwb/_behavior.py:30](../../../../src/neurospatial/io/nwb/_behavior.py#L30) — `read_position`: copy its tuple-return contract, optional-arg, and validation style.
- [io/nwb/__init__.py:95](../../../../src/neurospatial/io/nwb/__init__.py#L95) — `_LAZY_IMPORTS` dict and `__all__` (line 135): the new readers register here.
- A `pynwb` `units` table: `nwbfile.units` is a `DynamicTable` whose `spike_times` is a ragged (VectorData + index) column; `nwbfile.units[i, "spike_times"]` returns one neuron's spike-time array.

**Contracts referenced:**

- [Input-validation helpers](shared-contracts.md#input-validation-helpers) — only if validating `unit_ids` membership.

## Tasks

- Create `src/neurospatial/io/nwb/_units.py` with:

  ```python
  def read_units(nwbfile, *, unit_ids=None):
      """Read spike-time arrays from an NWB ``units`` table.

      Parameters
      ----------
      nwbfile : pynwb.NWBFile
      unit_ids : sequence of int, optional
          Subset of units to read, given as the table's ``id`` values (the
          identifiers in ``units.id``), not row indices. Each requested id is
          matched against ``units.id``; any id not present raises ``ValueError``
          naming the missing id(s). Default reads all units in table order.

      Returns
      -------
      spike_trains : list of NDArray[np.float64]
          One sorted 1-D array of spike times (seconds) per unit.
      unit_ids : NDArray
          The unit identifiers, aligned with ``spike_trains``.
      """
      units = nwbfile.units
      if units is None:
          raise ValueError("NWBFile has no `units` table.")
      ids = np.asarray(units.id.data[:])
      if unit_ids is None:
          rows = range(len(ids))
          out_ids = ids
      else:
          # `unit_ids` are matched against the table's id values ONLY. Resolve
          # each to its row; an id with no match is a hard error (do not fall
          # back to interpreting it as a row index, which would silently return
          # the wrong unit on a typo).
          rows = []
          missing = []
          for u in unit_ids:
              match = np.flatnonzero(ids == u)
              if match.size == 0:
                  missing.append(u)
              else:
                  rows.append(int(match[0]))
          if missing:
              raise ValueError(
                  f"unit_ids not found in the units table: {missing}. "
                  f"Available ids: {ids.tolist()}."
              )
          out_ids = ids[rows]
      spike_trains = [np.asarray(units[i, "spike_times"], dtype=np.float64) for i in rows]
      return spike_trains, out_ids
  ```

  (Read the real `pynwb` units accessor before finalizing — confirm `units[i, "spike_times"]` vs `units["spike_times"][i]`; use whichever the installed API exposes. Keep `read_spikes` as a thin alias of `read_units` if the review prefers that name, or omit.)
- Register in `io/nwb/__init__.py` `_LAZY_IMPORTS` (`"read_units": "neurospatial.io.nwb._units:read_units"`) and add to `__all__`.
- Docstring with a runnable Example (registered in phase 23): `read_units(nwbfile)` → `bin_spikes_in_time` → `decode_position`.
- CHANGELOG entry.

## Deliberately not in this phase

- Temporal binning of the returned trains — phase 14 (`bin_spikes_in_time`).
- Any change to `units`-table *writing* — out of scope; this is a reader.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_read_units_roundtrip` (integration, `importorskip("pynwb")`) | A synthesized NWB with 3 units and known ragged spike times reads back the exact arrays and ids. |
| `test_read_units_subset` | `unit_ids=[<id0>, <id2>]` (table `id` values, not row indices) returns those two trains in order with matching ids. |
| `test_read_units_unknown_id_raises` | `unit_ids` containing an id absent from `units.id` raises a clear `ValueError` naming the missing id (no silent row-index fallback). |
| `test_read_units_no_table_raises` | An NWBFile without `units` raises a clear `ValueError`. |
| `test_read_units_lazy_export` | `from neurospatial.io.nwb import read_units` works via the lazy `__getattr__`. |

## Fixtures

A synthesized NWB file written in `conftest` (or per-test) with a `units` table of 3 neurons; gate on `pynwb` availability with `pytest.importorskip`.

## Review

Dispatch `code-reviewer` against the diff. Confirm: tuple contract matches `read_position`; ragged `units` access uses the real installed `pynwb` API (verified, not guessed); lazy registration works; tests are `pynwb`-gated and exercise a real round-trip; CHANGELOG updated; no plan/phase references in code/test names.
