# Phase 4 — Stats: weighted-circular & shuffle correctness

[← back to overview](overview.md) · [shared contracts](shared-contracts.md#input-validation-helpers)

One PR, scoped strictly to `src/neurospatial/stats/`. Fixes six silent-correctness
bugs in the weighted circular statistics (`circular.py`) and the shuffle p-value /
cell-identity helpers (`shuffle.py`). Every fix turns a silently-wrong result into
either a correct result or a raised `ValueError`/emitted warning.

**Inputs to read first:**

- [../../../../src/neurospatial/stats/circular.py:181](../../../../src/neurospatial/stats/circular.py) — `_mean_resultant_length` (lines 181–237). The weighted branch (218–237) normalizes `weights` by `sum(weights)` with **no length check**: a length-1 `weights` broadcasts against `cos_angles`/`sin_angles`, and `weight_sum` is then a single scalar, silently producing an out-of-range `R`. This helper is called by `mean_resultant_length`, `circular_variance`, `circular_mean` (its own copy of the same logic, lines 973–983), and `rayleigh_test`.
- [../../../../src/neurospatial/stats/circular.py:240](../../../../src/neurospatial/stats/circular.py) — `_validate_circular_input` (lines 240–333). Drops NaN angles via `nan_mask` (line 274, `angles = angles[~nan_mask]` at 297) and wraps out-of-range values. It takes **only `angles`** today, so when `rayleigh_test` co-passes `weights`, the weights are never filtered by the same mask → misalignment.
- [../../../../src/neurospatial/stats/circular.py:485](../../../../src/neurospatial/stats/circular.py) — `rayleigh_test` (lines 485–612). Calls `_validate_circular_input(angles, ...)` at 562–564 (drops NaN angles only), then computes `r_mean = _mean_resultant_length(angles, weights=weights)` at 569, then `n_eff = float(np.sum(weights))` at 587–589, `z = n_eff * r_mean**2` at 594, and the finite-sample correction `if n_eff < 50:` at 603–607 which divides by `n_eff` (bare `ZeroDivisionError` when `n_eff == 0`; `z < 0` when weights are negative).
- [../../../../src/neurospatial/stats/circular.py:932](../../../../src/neurospatial/stats/circular.py) — `circular_mean` (932–985) and `circular_variance` (988–1029) and `mean_resultant_length` (1032–1077): the three other weighted public entry points that must validate weights.
- [../../../../src/neurospatial/stats/shuffle.py:927](../../../../src/neurospatial/stats/shuffle.py) — `compute_shuffle_pvalue` (927–1023). `n = len(null_scores)` at 1000; the three tails (1002–1018) count `null_scores >= observed` etc. A NaN null score satisfies neither `>=` nor `<=`, so it is silently excluded from `k` but **still counted in `n`**, biasing every tail toward significance.
- [../../../../src/neurospatial/stats/shuffle.py:227](../../../../src/neurospatial/stats/shuffle.py) — `shuffle_cell_identity` (signature ~227–235; body 306–313). `n_neurons = spike_counts.shape[1]` at 307; it permutes that many columns but never checks `encoding_models.shape[0] == n_neurons`, so a mismatched model array silently yields wrong decodes.

**Contracts referenced:**

- [Input-validation helpers](shared-contracts.md#input-validation-helpers) — this phase is the **first** consumer of the new `src/neurospatial/_validation.py`. Note the load-bearing semantic: `validate_lengths` compares lengths only and **a length-1 array is a mismatch, not a broadcast** — that is exactly the weighted-circular bug here. `validate_finite` raises on Inf always and on NaN unless `allow_nan=True`. Do not weaken either.

## Tasks

### Task 1 — Create the shared validation module (if not already present)

This phase is the first scheduled consumer of `_validation.py` (overview §"Input-validation helpers"). If a sibling phase has not already created it, create it now exactly as specified in the contract (verbatim `validate_finite` + `validate_lengths` from [shared-contracts.md](shared-contracts.md#input-validation-helpers)). If it already exists when this PR is cut, **do not** redefine it — import from it. Either way, `circular.py` consumes only `validate_lengths` (for the equal-length check); the non-negativity check below is stats-local.

### Task 2 — Add a shared `_validate_weights` helper in `circular.py`

Add this helper in the "Internal Helper Functions" block of `circular.py` (immediately after `_to_radians`, before `_mean_resultant_length`, ~line 180). It is the single entry point every weighted code path calls.

```python
def _validate_weights(
    angles: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    name: str = "weights",
) -> NDArray[np.float64]:
    """
    Validate per-angle weights for weighted circular statistics.

    Weights must be the same length as ``angles`` (a length-1 array is a
    mismatch, never broadcast) and must be non-negative. This guards the
    weighted resultant-length / Rayleigh paths against silently producing
    out-of-range statistics.

    Parameters
    ----------
    angles : ndarray of shape (n,)
        Angles the weights apply to (already raveled to 1-D).
    weights : ndarray of shape (n,)
        Per-angle weights, interpreted as counts/frequencies.
    name : str, default='weights'
        Argument name for error messages.

    Returns
    -------
    ndarray of shape (n,)
        Validated weights as a float64 array.

    Raises
    ------
    ValueError
        If lengths differ or any weight is negative.
    """
    from neurospatial._validation import validate_lengths

    weights = np.asarray(weights, dtype=np.float64).ravel()
    validate_lengths({"angles": angles, name: weights})
    if np.any(weights < 0):
        n_neg = int(np.sum(weights < 0))
        first = int(np.argmax(weights < 0))
        raise ValueError(
            f"{name} must be non-negative (interpreted as counts/frequencies). "
            f"Found {n_neg} negative value(s) (first at index {first}: "
            f"{weights[first]!r}). "
            f"Fix: pass spike counts or occupancy times, not signed quantities."
        )
    return weights
```

Notes for the executor:
- `validate_lengths` raises with `"Length mismatch: angles=N, weights=M. These must agree."`. The length-1 case (`weights=1`) is therefore rejected, not broadcast — that is the intended behavior.
- Keep the `np.asarray(..., dtype=np.float64).ravel()` here so callers can pass lists.

### Task 3 — Route `_mean_resultant_length` through `_validate_weights`

In `_mean_resultant_length` (circular.py:181–237), the weighted branch begins at line 218 (`if weights is not None:`). Replace the unvalidated `weights = np.asarray(weights, dtype=np.float64)` (line 220) with a validated call. The `angles` passed in are already raveled by every caller, but call `np.ravel` defensively to match `_validate_weights`'s shape contract.

Replace lines 218–224 (the `if weights is not None:` block down to `weights_norm = weights / weight_sum`) with:

```python
    if weights is not None:
        # Validate (equal length, non-negative); reject length-1 broadcast.
        weights = _validate_weights(np.ravel(angles), weights)
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return np.nan
        weights_norm = weights / weight_sum
```

The `weight_sum == 0 → np.nan` guard (existing lines 221–223) is preserved.

### Task 4 — Validate weights in `circular_mean`'s own copy of the logic

`circular_mean` (circular.py:932–985) does **not** call `_mean_resultant_length`; it has its own weighted block at lines 973–983. Replace its `weights = np.asarray(weights, dtype=np.float64)` (line 974) so it validates too:

```python
    if weights is not None:
        weights = _validate_weights(angles, weights)
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return np.nan
        weights_norm = weights / weight_sum
        mean_cos = np.sum(weights_norm * cos_angles)
        mean_sin = np.sum(weights_norm * sin_angles)
```

`mean_resultant_length` (1032–1077) and `circular_variance` (988–1029) both delegate to `_mean_resultant_length`, so Task 3 already covers them — no edit needed in those two beyond confirming they pass `weights` through unchanged.

### Task 5 — Co-filter weights with the NaN mask in `_validate_circular_input`

`rayleigh_test` is the only caller that pairs `weights` with `angles` through `_validate_circular_input`. Extend that helper to optionally accept and co-filter `weights`, so the same NaN mask (and the same out-of-range wrapping, which does not drop elements) keeps the two arrays aligned.

Change the signature (circular.py:240–246) to add an optional `weights` parameter and a paired return:

```python
def _validate_circular_input(
    angles: NDArray[np.float64],
    name: str = "angles",
    *,
    min_samples: int = 3,
    check_range: bool = True,
    weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
```

Update the docstring `Returns` block to document the tuple `(angles, weights)` (weights is `None` if none were passed). Then, inside the body:

1. **Before** the NaN handling, validate length/non-negativity if weights were supplied, so a mismatch is reported up front:

   ```python
   angles = np.asarray(angles, dtype=np.float64).ravel()
   if weights is not None:
       weights = _validate_weights(angles, weights, name=f"{name}_weights")
   ```

2. In the NaN branch (currently `angles = angles[~nan_mask]` at line 297), filter weights with the **same mask**:

   ```python
       angles = angles[~nan_mask]
       if weights is not None:
           weights = weights[~nan_mask]
       warnings.warn(...)  # existing warning unchanged
   ```

   (The out-of-range wrapping at lines 322–331 uses modulo and does not drop elements, so weights need no change there.)

3. Change the single `return angles` (line 333) to:

   ```python
   return angles, weights
   ```

Update the **other** callers of `_validate_circular_input` to unpack the tuple and ignore the second element. The only other caller is `rayleigh_test` itself (line 562) — but grep to confirm before committing:

```bash
grep -rn "_validate_circular_input" src/neurospatial/
```

Any caller not passing `weights` receives `(angles, None)` and writes `angles, _ = _validate_circular_input(...)`.

### Task 6 — Rework `rayleigh_test` validation, n_eff guard, and weights flow

In `rayleigh_test` (circular.py:485–612), replace the validation + n_eff computation (lines 557–594) so weights are co-filtered and `n_eff <= 0` is guarded. The body from the `# Convert to radians` comment through the `z = ...` line becomes:

```python
    # Convert to radians if needed
    angles = np.asarray(angles, dtype=np.float64)
    angles = _to_radians(angles, angle_unit)

    # Validate input (handles NaN drop + Inf + min samples) and co-filter
    # weights so angle/weight alignment survives NaN removal. Weight
    # non-negativity and equal-length are enforced inside the validator.
    angles, weights = _validate_circular_input(
        angles, "angles", min_samples=3, check_range=False, weights=weights
    )

    n = len(angles)

    # Weighted mean resultant length (weights already validated above).
    r_mean = _mean_resultant_length(angles, weights=weights)

    # Effective count for the z-statistic and finite-sample correction.
    # Count-based (grouped-data) weighted Rayleigh (Mardia & Jupp 2000,
    # Section 5.3.5): n_eff = sum(weights); for unit weights, n_eff = n.
    if weights is not None:
        n_eff: float = float(np.sum(weights))
    else:
        n_eff = float(n)

    # Guard: all-zero (or empty) weights give n_eff == 0. The weighted R
    # is already NaN in that case; surface an actionable error instead of a
    # bare ZeroDivisionError in the finite-sample correction below.
    if n_eff <= 0:
        raise ValueError(
            "Total weight (sum of weights) is zero; cannot run the weighted "
            "Rayleigh test. At least one angle must carry positive weight.\n"
            "Fix: check that your per-angle counts/occupancy are not all zero."
        )

    # Rayleigh z-statistic: z = n_eff * R^2.
    z = n_eff * r_mean**2
```

The remainder of the function (p-value, finite-sample correction `if n_eff < 50:` at the old lines 603–607, the `np.clip`, the `return`) is unchanged and now safe because `n_eff > 0`.

Because `_mean_resultant_length` no longer re-validates length (it now trusts the already-validated `weights`), and `_validate_weights` already ran inside `_validate_circular_input`, there is no double-error risk: the first failure is the length/negativity check, reported once.

Drop the now-redundant re-cast `weights = np.asarray(weights, dtype=np.float64)` that lived at old lines 588.

### Task 7 — Drop non-finite null scores in `compute_shuffle_pvalue`

In `compute_shuffle_pvalue` (shuffle.py:927–1023), replace `n = len(null_scores)` (line 1000) with a finite-filter that warns and recomputes `n` from the finite count, so NaN/Inf nulls cannot silently inflate significance. Insert before the `if tail == ...` ladder:

```python
    null_scores = np.asarray(null_scores, dtype=np.float64).ravel()
    finite = np.isfinite(null_scores)
    n_total = null_scores.size
    if not finite.all():
        n_dropped = int(n_total - finite.sum())
        warnings.warn(
            f"Dropped {n_dropped} non-finite null score(s) out of {n_total} "
            f"before computing the p-value. Using {int(finite.sum())} finite "
            f"null scores; non-finite nulls would otherwise bias the p-value "
            f"toward significance.",
            category=UserWarning,
            stacklevel=2,
        )
        null_scores = null_scores[finite]

    n = len(null_scores)
    if n == 0:
        raise ValueError(
            "null_scores contains no finite values; cannot compute a "
            "shuffle p-value. Check that the shuffle produced valid scores."
        )
```

Confirm `import warnings` is present at the top of `shuffle.py`; add it if absent (grep first). The three tail branches below are unchanged — they now operate on the filtered array and corrected `n`.

### Task 8 — Neuron-count check in `shuffle_cell_identity`

In `shuffle_cell_identity` (shuffle.py body at 306–313), after `n_neurons = spike_counts.shape[1]` (line 307), add a shape-agreement check against `encoding_models`:

```python
    generator = _ensure_rng(rng)
    n_neurons = spike_counts.shape[1]

    if encoding_models.shape[0] != n_neurons:
        raise ValueError(
            f"spike_counts has {n_neurons} neuron(s) (columns) but "
            f"encoding_models has {encoding_models.shape[0]} (rows). These must "
            f"match: row i of encoding_models is the place field for neuron i "
            f"(column i of spike_counts).\n"
            f"Fix: pass spike_counts shaped (n_time_bins, n_neurons) and "
            f"encoding_models shaped (n_neurons, n_bins)."
        )
```

This raises before the generator loop, so a mismatch fails fast rather than yielding silently-wrong decodes.

### Task 9 — Docstring touch-ups for behavior changes

Update the docstrings of the touched public functions to state the new guarantees (no plan/milestone references in prose):

- `rayleigh_test`: in `Raises`, add that a `ValueError` is raised when total weight is zero, and that `weights` must be non-negative and the same length as `angles` (length-1 is rejected, not broadcast).
- `circular_mean` / `circular_variance` / `mean_resultant_length`: in each `weights` parameter description, note "Must be the same length as `angles` and non-negative."
- `compute_shuffle_pvalue`: in `Notes`, add that non-finite null scores are dropped (with a warning) and excluded from `n`, and that an empty finite null raises.
- `shuffle_cell_identity`: in `Raises` (add the section), document the neuron-count mismatch `ValueError`.

Add one CHANGELOG entry summarizing: weighted circular functions now validate weight length/sign; `rayleigh_test` co-filters weights on NaN drop and raises on zero total weight; `compute_shuffle_pvalue` drops non-finite nulls; `shuffle_cell_identity` validates neuron-count agreement.

## Deliberately not in this phase

- **Compass-vs-math plotting convention** in `plot_circular_basis_tuning` (the `set_theta_zero_location("N")` / `set_theta_direction(-1)` choice, circular.py:1681–1682) — this is a documentation/convention clarification, deferred to the docs phase (phase 23). Do not change plotting orientation here.
- **`circular_basis` docstring "Properties" → use of a real `Properties` heading and the unit-handling note** — wording-only suggestions; bundle into phase 23, not this correctness PR.
- **Polar-environment redesign** (any `from_polar_egocentric` / egocentric polar binning) — phase 19. This phase does not touch polar paths even though they consume circular stats.
- **The other shuffle helpers** (`shuffle_time_bins*`, `shuffle_place_fields_circular*`, `shuffle_posterior_*`, `shuffle_spikes_isi`, `ShuffleTestResult`) — only `compute_shuffle_pvalue` and `shuffle_cell_identity` are in scope. Do not "while I'm here" them.
- **`surrogates.py`** — listed in the module touch-map for the stats subsystem but has no finding in this phase; leave it unedited unless a regression test forces a change.

## Validation slice

All tests live under `tests/stats/` (add to `test_stats_circular.py` and `test_stats_shuffle.py`; do not create plan-named files). Each must **fail on the current code and pass after the fix**.

| Test | Asserts |
| --- | --- |
| `test_rayleigh_test_length1_weights_raises` | `rayleigh_test(angles, weights=np.array([2.0]))` with `len(angles)==20` raises `ValueError` (length mismatch). Today it broadcasts and returns a finite `z`. |
| `test_rayleigh_test_negative_weights_raises` | `rayleigh_test(angles, weights=w)` with one negative weight raises `ValueError`. Today it can yield `z < 0`. |
| `test_rayleigh_test_all_zero_weights_raises` | `rayleigh_test(angles, weights=np.zeros(n))` raises `ValueError` (not `ZeroDivisionError`). |
| `test_rayleigh_test_nan_cofilters_weights` | With one `np.nan` angle and matched `weights`, the returned `z` equals `rayleigh_test` run on the manually NaN-dropped (angle, weight) pair (both filtered). Today the un-filtered weights misalign → different `z`. Assert via `assert_allclose`. |
| `test_rayleigh_test_integer_weights_match_replication` | `rayleigh_test(angles, weights=counts)` returns the same `(z, p)` (within `assert_allclose`, `rtol=1e-9`) as `rayleigh_test(np.repeat(angles, counts))` for small integer `counts` — pins the count-based semantics survive the refactor. |
| `test_mean_resultant_length_length1_weights_raises` | `mean_resultant_length(angles, weights=np.array([1.0]))` raises `ValueError`. |
| `test_circular_mean_negative_weights_raises` | `circular_mean(angles, weights=w_with_negative)` raises `ValueError`. |
| `test_circular_variance_weights_validates` | `circular_variance(angles, weights=mismatched_len)` raises `ValueError`. |
| `test_compute_shuffle_pvalue_drops_nan_nulls` | With `null=[1,2,nan,4]` and `observed=10`, `tail="greater"`: warns `UserWarning` and returns `(0+1)/(3+1)=0.25` (n reduced to 3), not `1/5=0.2`. Use `pytest.warns`. |
| `test_compute_shuffle_pvalue_all_nan_raises` | `compute_shuffle_pvalue(5.0, np.full(4, np.nan))` raises `ValueError`. |
| `test_compute_shuffle_pvalue_finite_unchanged` | All-finite null reproduces the existing doctest value `0.2` for `observed=10, null=[1,2,3,4]` — regression guard that finite path is untouched. |
| `test_shuffle_cell_identity_neuron_count_mismatch_raises` | `spike_counts` shape `(5, 3)`, `encoding_models` shape `(4, 7)` → `ValueError` mentioning 3 vs 4. |
| `test_shuffle_cell_identity_matched_ok` | Matched shapes still yield `n_shuffles` arrays of the right shape with `models is encoding_models` — regression guard. |

Mark none of these `slow`; all are sub-millisecond.

## Fixtures

No checked-in data or real-data slice needed. Synthesize inline in each test with a seeded `np.random.default_rng(0)`:

- A concentrated angle sample (`rng.vonmises(mu=0.5, kappa=4.0, size=20)` wrapped to `[0, 2π)`) for the weighted Rayleigh tests.
- Small integer `counts` (e.g. `np.array([1, 2, 1, 3, 1])` against 5 angles) for the replication-equivalence test.
- Plain Python lists / small `np.array`s for the shuffle-p-value and cell-identity tests (no RNG needed).

If a shared concentrated-angles helper is reused across ≥3 tests, lift it into a `conftest.py` fixture under `tests/stats/` rather than copy-pasting (there is no `tests/stats/conftest.py` today — create one only if the duplication warrants it).

## Review

Before opening the PR, dispatch an independent reviewer (`scientific-code-change-audit` lens — these are scientific-number changes) against the diff. Confirm:

- All six findings are fixed: weight length/sign validation (Tasks 2–4), NaN co-filter (Task 5), `n_eff <= 0` guard (Task 6), shuffle non-finite null drop (Task 7), neuron-count check (Task 8).
- The "Deliberately not in this phase" list is honored — no edits to plotting convention, polar paths, `surrogates.py`, or the other shuffle helpers.
- `validate_lengths` is **used** for the equal-length check (length-1 rejected, not broadcast) and `_validation.py` is imported, not re-implemented, if it already exists.
- Every validation-slice test fails on `main` and passes on the branch (run each against pre-fix `circular.py`/`shuffle.py` to confirm it is not a tautology). The integer-weights replication test and the finite-null regression test pin existing correct numbers so the refactor does not silently move them.
- `uv run pytest tests/stats/ -q`, `uv run ruff check src/neurospatial/stats/ && uv run ruff format --check src/neurospatial/stats/`, and `uv run mypy src/neurospatial/stats/` all pass. Note `_validate_circular_input` now returns a tuple — mypy will flag any unupdated caller.
- Docstrings, test names, and the new helper name reference no plan/milestone (`grep -ri "phase 4\|milestone\|review-remediation" src/neurospatial/stats/ tests/stats/` is empty).
- The CHANGELOG entry is present and names the new raised-error behaviors (per pre-1.0 policy: silently-wrong → raised is intended).
