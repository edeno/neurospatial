# Phase 3 — Simulation ground-truth correctness

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

This phase fixes four silent-correctness bugs in `src/neurospatial/simulation/`. Simulated ground truth is the oracle that the encoding/decoding test suites compare against, so a wrong place-field shape or a wrong spike train poisons every downstream "we recovered the planted cell" assertion. Scope is **strictly** `src/neurospatial/simulation/` (plus its tests). One PR.

**Inputs to read first:**

- [../../../../src/neurospatial/simulation/models/place_cells.py:313-335](../../../../src/neurospatial/simulation/models/place_cells.py) — the Gaussian evaluation. Lines 317-324 build the anisotropic effective distance (`normalized_diff = diff / width`, then `distances = norm(normalized_diff)`), which is **already in sigma units**. Lines 333-334 then divide that result *again* by `width_for_clip = mean(width)` inside `exp(-0.5 * (distances / width_for_clip) ** 2)`. That is the double normalization producing near-flat fields. The isotropic branch (line 327, `width_for_clip = float(width)`) is correct and must keep working unchanged.
- [../../../../src/neurospatial/simulation/models/place_cells.py:193-201](../../../../src/neurospatial/simulation/models/place_cells.py) — where `self.width` is assigned in `__init__` (default `3.0 * mean(bin_sizes)` or the user value, passed through with no positivity check). This is the construction site where the new validator belongs. `PlaceCellModel` is a **plain class**, not a dataclass — there is no `__post_init__`; validation goes inline in `__init__` after `self.width` is set.
- [../../../../src/neurospatial/simulation/models/place_cells.py:240-342](../../../../src/neurospatial/simulation/models/place_cells.py) — `firing_rate`. With `width == 0`, `distances / width_for_clip` is `0/0 → NaN` at the center, so `gaussian` is NaN and the rate is NaN, which yields zero candidate spikes silently downstream. Rejecting non-positive width at construction (above) is the fix; no per-call guard is needed once construction rejects it.
- [../../../../src/neurospatial/simulation/spikes.py:125-150](../../../../src/neurospatial/simulation/spikes.py) — `generate_poisson_spikes`. Line 132 computes `dt = times[1] - times[0]` and uses that single scalar for **every** bin (the docstring at 30-32 even promises "must be uniformly spaced" but never enforces it). Non-ascending or non-uniform `times` silently produce wrong per-bin spike probabilities. Line 141-143 computes `1 - exp(-max(rate,0)*dt)`: a NaN rate floors-through `np.maximum` (NaN survives), `exp` gives NaN, `np.clip(NaN,0,1)` stays NaN, and `random < NaN` is `False` — so NaN rates silently drop all spikes instead of erroring.
- [../../../../src/neurospatial/simulation/spikes.py:357-394](../../../../src/neurospatial/simulation/spikes.py) — `_compute_model_firing_rate`, the population path that feeds `generate_poisson_spikes`. Already length-checks rates vs times; do **not** duplicate that check. The new finite/timestamp guards live inside `generate_poisson_spikes` so both the single-cell and population entry points are covered.
- [../../../../src/neurospatial/encoding/_validation.py:73-110](../../../../src/neurospatial/encoding/_validation.py) — `validate_times`: the existing house style for a finite + monotonic timestamp check (message format, `np.diff`, `np.where` index reporting). Mirror this style; do **not** import it (it is encoding-scoped and allows equal adjacent samples, which is too weak for a per-bin `dt`).
- [../../../../tests/simulation/test_models.py:94-113](../../../../tests/simulation/test_models.py) — `test_gaussian_falloff` already asserts the **isotropic** 1σ → `max_rate * exp(-0.5)` behavior. The new regression test must cover the **anisotropic** case (the bug), not re-assert isotropic.
- [../../../../tests/simulation/conftest.py:9-26](../../../../tests/simulation/conftest.py) — `simple_2d_env` fixture (100×100 cm arena, 2 cm bins) used by the model tests.
- [../../../../tests/simulation/test_spikes.py](../../../../tests/simulation/test_spikes.py) — existing `generate_poisson_spikes` tests; new validation tests go here.

**Contracts referenced:**

- [Input-validation helpers](shared-contracts.md#input-validation-helpers) — use `validate_finite` (from the new top-level `src/neurospatial/_validation.py`) for the spike-rate finiteness guard. Semantics are fixed: raises on Inf always, raises on NaN, names the argument and first offending index. Do **not** weaken to a silent NaN→0 coercion.

> **Cross-phase note on `src/neurospatial/_validation.py`.** The shared-contracts home is a *new* top-level module also consumed by phases 4–9. If this phase lands before that module exists, this phase **creates** `src/neurospatial/_validation.py` containing exactly the `validate_finite` (and `validate_lengths`) implementation from [shared-contracts.md](shared-contracts.md#input-validation-helpers), verbatim, and imports `validate_finite` from it. Later phases then reuse it rather than redefining. If the module already exists when this phase is implemented, import from it and do not redefine. Either way the implementation must match the contract byte-for-byte; do not fork a simulation-local copy.

## Tasks

### Task 1 — Reject non-positive place-field width at construction

In `PlaceCellModel.__init__` (`place_cells.py`), immediately after `self.width` is assigned (the `if width is None / else` block ending at line 200), add a positivity validator. This makes the `width=0` NaN bug structurally impossible and also rejects negative widths.

Insert after line 200 (`self.width = width`), before the geodesic-compatibility check at line 202:

```python
        # Reject non-positive width: width is the Gaussian standard deviation,
        # so width <= 0 makes the field undefined (0/0 -> NaN at the center,
        # silently producing zero spikes downstream).
        width_check = np.asarray(self.width, dtype=np.float64)
        if not np.all(np.isfinite(width_check)):
            msg = (
                f"width must be finite and positive, got {self.width!r}. "
                "width is the Gaussian standard deviation of the place field."
            )
            raise ValueError(msg)
        if np.any(width_check <= 0.0):
            msg = (
                f"width must be strictly positive, got {self.width!r}. "
                "width is the Gaussian standard deviation of the place field; "
                "a zero or negative width gives an undefined (0/0) firing rate "
                "at the field center."
            )
            raise ValueError(msg)
```

This runs for both scalar and array (anisotropic) widths because `np.asarray(...).<=0` and `np.isfinite` are elementwise. The default-width branch (`3.0 * mean(bin_sizes)`) is always positive, so default construction is unaffected.

### Task 2 — Fix anisotropic double-normalization in `firing_rate`

In `PlaceCellModel.firing_rate` (`place_cells.py`), the anisotropic branch already produces `distances` in **sigma units** (each dimension divided by its own width). The subsequent `exp(-0.5 * (distances / width_for_clip) ** 2)` must therefore use a **unit divisor of 1.0** for the anisotropic case, while clipping still uses a representative length scale.

Replace the anisotropic/isotropic block (lines 313-334) with the following. Note the new `sigma_divisor` variable: it is the value that converts `distances` back into raw sigma units inside the `exp`. For isotropic, `distances` are raw Euclidean and `sigma_divisor = width`. For anisotropic, `distances` are *already* in sigma units, so `sigma_divisor = 1.0`.

```python
        # Convert width to array for anisotropic case
        width = np.asarray(self.width)

        if width.ndim > 0 and len(width) > 1:
            # Anisotropic: each dimension has its own width. Build an
            # effective distance already expressed in standard-deviation
            # units: d_sigma = sqrt(sum_i ((x_i - c_i) / width_i)^2).
            diff = positions - self.center
            normalized_diff = diff / width
            distances = np.linalg.norm(normalized_diff, axis=1)
            # `distances` is now in sigma units, so the Gaussian divisor is 1.0.
            # Use the mean width only as a length scale for the clip threshold,
            # expressed back in sigma units (5 sigma).
            sigma_divisor = 1.0
            clip_threshold = 5.0  # 5 sigma, since `distances` is in sigma units
        else:
            # Isotropic: `distances` (set above) is a raw Euclidean distance,
            # so divide by the scalar width to convert to sigma units.
            scalar_width = float(width)
            sigma_divisor = scalar_width
            clip_threshold = 5.0 * scalar_width  # 5 sigma in raw distance units

        # Clip for numerical stability (Gaussian < 1e-6 beyond 5 sigma).
        distances = np.minimum(distances, clip_threshold)

        # Compute Gaussian firing rate:
        #   rate = baseline + (max - baseline) * exp(-0.5 * (d / sigma)^2)
        gaussian = np.exp(-0.5 * (distances / sigma_divisor) ** 2)
        rates = self.baseline_rate + (self.max_rate - self.baseline_rate) * gaussian
```

Behavior after the fix: at exactly 1 sigma from center along any axis (e.g. anisotropic `width=[wx, wy]`, position offset `wx` in x only), `d_sigma == 1.0` and `gaussian == exp(-0.5)`, so the rate is `baseline + (max - baseline) * exp(-0.5)` — matching the isotropic branch. The isotropic numbers are byte-for-byte unchanged (`sigma_divisor == width`, `clip_threshold == 5*width`).

> The geodesic branch (lines 282-309) always sets `self.width` to a scalar (anisotropic geodesic is rejected in `__init__`, line 203-212), so it flows through the isotropic branch unchanged. No geodesic change needed.

### Task 3 — Validate timestamps in `generate_poisson_spikes`

In `generate_poisson_spikes` (`spikes.py`), replace the silent `dt = times[1] - times[0]` (line 132) with full timestamp validation and a **per-bin** `dt`. Times must be equal-length to `firing_rate`, finite, and strictly increasing (strictly — a flat `dt=0` bin has zero spike probability and a decreasing step is nonsense for a forward-time process).

Replace lines 128-132:

```python
    # Compute time step (assume uniform sampling)
    if len(times) < 2:
        return np.array([], dtype=np.float64)

    dt = times[1] - times[0]
```

with:

```python
    times = np.asarray(times, dtype=np.float64)
    firing_rate = np.asarray(firing_rate, dtype=np.float64)

    if len(firing_rate) != len(times):
        raise ValueError(
            f"firing_rate and times must have the same length, got "
            f"firing_rate={len(firing_rate)}, times={len(times)}."
        )

    if len(times) < 2:
        return np.array([], dtype=np.float64)

    if not np.all(np.isfinite(times)):
        n_bad = int(np.sum(~np.isfinite(times)))
        raise ValueError(
            f"times must be finite, got {n_bad} NaN/inf entr(y/ies)."
        )

    # Per-bin dt: the inhomogeneous Poisson probability uses the actual
    # spacing of each interval, so non-uniform sampling is handled correctly.
    # Require strictly increasing times: a zero-width interval has no spike
    # probability and a decreasing interval is meaningless for a forward-time
    # point process.
    bin_dt = np.diff(times)
    if np.any(bin_dt <= 0.0):
        bad_idx = np.where(bin_dt <= 0.0)[0]
        raise ValueError(
            "times must be strictly increasing. Found "
            f"{len(bad_idx)} non-increasing interval(s) at indices: "
            f"{bad_idx.tolist()[:5]}"
            + (" ..." if len(bad_idx) > 5 else "")
        )

    # Pair each time point with the width of the interval that follows it.
    # The final sample has no following interval; reuse the last dt so the
    # `firing_rate` and `dt` arrays stay aligned and length-n.
    dt = np.empty(len(times), dtype=np.float64)
    dt[:-1] = bin_dt
    dt[-1] = bin_dt[-1]
```

`dt` is now a length-`n` array, so the existing elementwise computation at line 141-143 (`1.0 - np.exp(-np.maximum(firing_rate, 0.0) * dt)`) broadcasts per bin with no further change.

Update the `times` parameter docstring (lines 30-32) to drop "must be uniformly spaced" and state that non-uniform spacing is supported but times must be strictly increasing. Update the Notes algorithm step (lines 49-53) reference to `dt` to read "the width of the interval following time `i`".

### Task 4 — Validate finite firing rates in `generate_poisson_spikes`

Still in `generate_poisson_spikes`, guard against NaN/Inf rates using the shared `validate_finite` helper (see the cross-phase note above on `src/neurospatial/_validation.py`). Insert the import at the top of `spikes.py` (after the existing imports, line 10):

```python
from neurospatial._validation import validate_finite
```

Then, after the length check added in Task 3 and before the per-bin `dt` is used, validate rates. Place this immediately after the `firing_rate = np.asarray(...)` line from Task 3:

```python
    firing_rate = validate_finite(firing_rate, name="firing_rate")
```

`validate_finite` raises on any NaN or Inf, naming `firing_rate` and the first offending index — replacing the previous silent drop-all-spikes behavior. (Negative-but-finite rates remain permitted and are floored at 0 by the existing `np.maximum(firing_rate, 0.0)`; only non-finite rates are rejected.)

> If `src/neurospatial/_validation.py` does not yet exist in the tree when this phase is implemented, create it with the exact `validate_finite` and `validate_lengths` bodies from [shared-contracts.md](shared-contracts.md#input-validation-helpers) (verbatim) as part of this PR.

## Deliberately not in this phase

- **RNG re-coupling / independence audit.** A separate review note suggests revisiting how `generate_population_spikes` derives per-cell seeds (`base_seed + i`) for stream independence. That is a reproducibility-design change, not a correctness bug, and belongs to the later RNG/reproducibility phase. Do not touch `generate_population_spikes` seeding here.
- **Object-vector angle-convention note.** The review flagged an egocentric/allocentric angle-convention concern in `simulation/models/object_vector_cells.py`. That is its own correctness item handled in the egocentric-convention phase (phase 8 territory). Out of scope here — do not edit `object_vector_cells.py`.
- **`validate_lengths` adoption across other modules.** This phase only *creates* (if absent) and *uses* `validate_finite`. Wiring `validate_lengths` into encoding/stats/io is phases 4–9.
- **`generate_population_spikes` rate-length check.** It already validates `len(rates) == len(times)` in `_compute_model_firing_rate`; do not duplicate or relocate it.

## Validation slice

All tests in `tests/simulation/`. Each is a fail-before / pass-after for one fix.

| Test | Asserts |
| --- | --- |
| `test_anisotropic_one_sigma_rate` (test_models.py) | Anisotropic `width=[10.0, 5.0]`, `max_rate=20.0`, `baseline_rate=0.0`; position offset 10.0 in x only from center → rate ≈ `20.0 * exp(-0.5)` (≈12.13). **Fails before** (double-normalization makes rate ≈ `20.0 * exp(-0.5 * (1/mean_width)^2·…)`, near-flat / far too high). |
| `test_anisotropic_one_sigma_rate_y_axis` (test_models.py) | Same model, offset 5.0 in y only → rate ≈ `20.0 * exp(-0.5)`. Confirms per-axis sigma, not a coincidence on x. |
| `test_isotropic_one_sigma_unchanged` (test_models.py) | Isotropic `width=10.0`, offset 10.0 → rate ≈ `20.0 * exp(-0.5)` (regression guard that the isotropic path is byte-for-byte unchanged). Mirrors existing `test_gaussian_falloff`. |
| `test_width_zero_raises` (test_models.py) | `PlaceCellModel(env, center=..., width=0.0)` raises `ValueError` mentioning "positive". **Fails before** (constructs fine, NaN at center). |
| `test_width_negative_raises` (test_models.py) | `width=-5.0` raises `ValueError`. |
| `test_anisotropic_width_zero_component_raises` (test_models.py) | `width=[10.0, 0.0]` raises `ValueError` (elementwise positivity). |
| `test_width_nan_raises` (test_models.py) | `width=float("nan")` raises `ValueError` mentioning "finite". |
| `test_nonincreasing_times_raises` (test_spikes.py) | `times` with a decreasing or duplicate step → `generate_poisson_spikes` raises `ValueError` mentioning "strictly increasing". **Fails before** (silently uses `times[1]-times[0]`). |
| `test_nonuniform_times_uses_per_bin_dt` (test_spikes.py) | With deterministically non-uniform but strictly increasing `times` and a constant high rate, spike probability tracks each bin's width (e.g. a 10× wider bin yields ~10× the per-bin spike probability for small `rate*dt`); assert via expected-count bounds with fixed seed. **Fails before** (uniform `dt` from first interval misestimates wide/narrow bins). |
| `test_length_mismatch_raises` (test_spikes.py) | `len(firing_rate) != len(times)` raises `ValueError` mentioning both lengths. |
| `test_nan_firing_rate_raises` (test_spikes.py) | `firing_rate` containing a NaN raises `ValueError` naming `firing_rate`. **Fails before** (NaN silently drops all spikes; returns empty array). |
| `test_inf_firing_rate_raises` (test_spikes.py) | `firing_rate` containing `+inf` raises `ValueError`. |
| `test_finite_rate_negative_still_floored` (test_spikes.py) | A finite negative rate does **not** raise and produces zero spikes for that bin (confirms we only reject non-finite, not negative). |

None of these are slow; no `pytest.mark.slow` needed. Run the scoped suite:

```bash
uv run pytest tests/simulation/test_models.py tests/simulation/test_spikes.py -v
```

Also confirm no regression in the broader simulation + encoding suites that consume this ground truth:

```bash
uv run pytest tests/simulation/ tests/encoding/ -q
```

## Fixtures

Reuse `simple_2d_env` from `tests/simulation/conftest.py` (100×100 cm arena, 2 cm bins) for all place-cell tests; the field center used in tests is `[50.0, 50.0]`. For spike tests, synthesize `times`/`firing_rate` arrays inline (small, deterministic, no environment needed) — e.g. `times = np.array([0.0, 0.1, 0.2, ...])` and constant or hand-built rate arrays — with a fixed `seed` passed to `generate_poisson_spikes`. No new shared fixtures and no checked-in data are required; non-uniform-`dt` arrays are constructed inline in the one test that needs them.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:

- Every task in this phase is implemented as specified: width validator in `__init__`, anisotropic divisor fixed to 1.0 with isotropic path unchanged, per-bin strictly-increasing-`dt` timestamp validation, and `validate_finite` rate guard.
- The "Deliberately not in this phase" list is honored — no edits to `generate_population_spikes` seeding and no edits to `object_vector_cells.py`.
- Validation slice tests pass; the anisotropic 1σ test and the `width=0` raise test in particular fail on the pre-fix code (verify by stashing the fix).
- Tests aren't trivial — the 1σ tests assert the numeric `exp(-0.5)` value (not just "rate > 0"), and the per-bin-`dt` test asserts a quantitative count relationship, not a tautology. Shared setup uses the `simple_2d_env` fixture, not copy-pasted env construction. (`testing-anti-patterns` covers the failure modes.)
- Docstrings, test names, and module names don't reference this plan, "phase 3", or "review-remediation".
- The `times` docstring in `generate_poisson_spikes` no longer claims "must be uniformly spaced" and now documents the strictly-increasing requirement.
- `src/neurospatial/_validation.py` matches the shared-contracts body verbatim (created here only if absent); no simulation-local fork of `validate_finite`.
- Scope is strictly `src/neurospatial/simulation/` plus `src/neurospatial/_validation.py` (if newly created) and `tests/simulation/`.
