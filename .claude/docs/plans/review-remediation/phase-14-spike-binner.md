# Phase 14 — Public temporal spike-binner

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Adds the single most-requested missing primitive (DESIGN-REVIEW High #1): a public function that bins spike-time arrays into a time × neuron (or neuron × time) count matrix, owning the time grid and `dt/2` bin-center construction. This repairs the spike→time-bin→decode seam present in 4 of 5 user journeys and defuses the silent transpose footgun between `decode_position` and the assembly functions by making the axis orientation an explicit argument.

**Inputs to read first:**

- [decoding/posterior.py:265](../../../../src/neurospatial/decoding/posterior.py#L265) — `decode_position`: confirm it expects spike counts shaped `(n_time_bins, n_neurons)`; the binner's default `orient` must match.
- [decoding/assemblies.py](../../../../src/neurospatial/decoding/assemblies.py) — `detect_assemblies`/reactivation expect `(n_neurons, n_time_bins)`; this is the opposite axis order, which is exactly the footgun `orient=` removes.
- [src/neurospatial/__init__.py:232](../../../../src/neurospatial/__init__.py#L232) — top-level `__all__` for the re-export.

**Contracts referenced:**

- [Input-validation helpers](shared-contracts.md#input-validation-helpers) — `validate_finite` for `dt`/bounds (reject non-finite, `dt <= 0`).

## Tasks

- Add `bin_spikes_in_time` to `src/neurospatial/decoding/` (new `_binning.py` or into an existing decoding util module — read the package layout and follow it). Complete signature and body:

  ```python
  def bin_spikes_in_time(
      spike_trains,           # Sequence[NDArray]: one 1-D array of spike times per neuron
      dt,                     # float: bin width, same time units as the spike times
      t_start=None,           # float | None: defaults to min spike time across neurons
      t_stop=None,            # float | None: defaults to max spike time across neurons
      *,
      orient="time_x_neuron", # "time_x_neuron" (decode_position) | "neuron_x_time" (assemblies)
  ):
      """Bin per-neuron spike times into a count matrix on a regular time grid.

      Returns
      -------
      counts : NDArray[np.int64]
          Shape ``(n_time_bins, n_neurons)`` if ``orient="time_x_neuron"`` (the
          shape ``decode_position`` expects), else ``(n_neurons, n_time_bins)``
          (the shape the assembly functions expect).
      bin_centers : NDArray[np.float64]
          Shape ``(n_time_bins,)``; bin left edge + ``dt / 2``.
      """
      if dt <= 0 or not np.isfinite(dt):
          raise ValueError(f"dt must be finite and > 0, got {dt!r}.")
      trains = [np.asarray(s, dtype=np.float64) for s in spike_trains]
      if t_start is None:
          t_start = min((s.min() for s in trains if s.size), default=0.0)
      if t_stop is None:
          t_stop = max((s.max() for s in trains if s.size), default=t_start + dt)
      if t_stop <= t_start:
          raise ValueError(f"t_stop ({t_stop}) must be > t_start ({t_start}).")
      edges = np.arange(t_start, t_stop + dt, dt)
      counts = np.stack(
          [np.histogram(s, bins=edges)[0] for s in trains], axis=1
      ).astype(np.int64)  # (n_time_bins, n_neurons)
      bin_centers = edges[:-1] + dt / 2.0
      if orient == "neuron_x_time":
          counts = counts.T
      elif orient != "time_x_neuron":
          raise ValueError(
              f"orient must be 'time_x_neuron' or 'neuron_x_time', got {orient!r}."
          )
      return counts, bin_centers
  ```

- Export it from `neurospatial.decoding` (`decoding/__init__.py` `__all__`) **and** the top-level package (`src/neurospatial/__init__.py` `__all__`) — it is a primary entry point, so it should be importable as `import neurospatial as ns; ns.bin_spikes_in_time(...)`.
- Docstring: NumPy style, with a runnable Example that bins two neurons and feeds the result straight into `decode_position` (`orient="time_x_neuron"`) — this example is registered with the docs CI in phase 23.
- CHANGELOG: add under a new-features section.

## Deliberately not in this phase

- The `read_units` NWB loader that produces `spike_trains` — phase 15. (This function takes already-loaded arrays.)
- The end-to-end worked example wiring `read_units` → `bin_spikes_in_time` → `decode_position` → `animate_fields` — phase 23, once phases 14 and 15 have both landed.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_bin_spikes_counts_hand_computed` | A 2-neuron fixture with known spike times yields the exact expected count matrix and `bin_centers == edges[:-1] + dt/2`. |
| `test_bin_spikes_orient_transposes` | `orient="neuron_x_time"` is the transpose of `orient="time_x_neuron"`; shapes are `(n_neuron, n_time)` vs `(n_time, n_neuron)`. |
| `test_bin_spikes_rejects_bad_dt` | `dt <= 0` and non-finite `dt` raise `ValueError`; `t_stop <= t_start` raises. |
| `test_bin_spikes_feeds_decode_position` | Output with default orient is accepted by `decode_position` without reshaping (integration). |
| `test_bin_spikes_empty_neuron` | A neuron with no spikes yields an all-zero column, not a crash. |

## Fixtures

Synthesized in-test: two short spike-time arrays with hand-countable bins; reuse the existing decoding `conftest` env/encoding-model fixtures for the integration test.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- The function is implemented as specified and exported at both `neurospatial.decoding` and top level.
- `orient` default matches `decode_position`'s expected shape (verify by reading `decode_position`, not by assumption).
- "Deliberately not in this phase" is honored (no NWB reader here).
- Validation tests pass and exercise real behavior (hand-computed counts), not tautologies.
- Docstring example runs; CHANGELOG updated.
- No references to this plan or phase numbers in code/test names.
