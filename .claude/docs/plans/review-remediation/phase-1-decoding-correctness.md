# Phase 1 — Decoding correctness

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

One shippable PR scoped entirely to `src/neurospatial/decoding/`. Fixes five scientific-correctness defects: an inert reactivation control (REV==EV), a control-ignoring explained variance, a magnitude-blind reactivation strength, an unvalidated bin-count/timestamp mismatch in the decoder, a NaN bin silently entering the HPD credible region, NaN posterior rows corrupting the confusion matrix, and a 1-D `spike_counts` that silently collapses the time axis.

**Inputs to read first:**
- [decoding/assemblies.py:910-972](../../../../src/neurospatial/decoding/assemblies.py) — `reactivation_strength`; the double-z-score makes it magnitude-blind (finding 1c).
- [decoding/assemblies.py:975-1122](../../../../src/neurospatial/decoding/assemblies.py) — `explained_variance_reactivation`; REV hardcoded `= ev` at line 1099, and EV ignores `control_correlations` (findings 1a, 1b).
- [decoding/assemblies.py:746-835](../../../../src/neurospatial/decoding/assemblies.py) — `assembly_activation`, with `z_score_input` keyword; needed to add the non-double-z path used by `reactivation_strength`.
- [decoding/posterior.py:265-399](../../../../src/neurospatial/decoding/posterior.py) — `decode_position`; never checks `encoding_models.shape[1]` vs `env.n_bins`, never checks `len(times)` vs `n_time_bins` (finding 2).
- [decoding/posterior.py:402-474](../../../../src/neurospatial/decoding/posterior.py) — `_validate_inputs`; existing rich-diagnostic validation pattern to extend.
- [decoding/estimates.py:333-371](../../../../src/neurospatial/decoding/estimates.py) — `credible_region`; descending argsort puts NaN bins first, so a NaN bin is reported as the highest-density member of the HPD set (finding 3).
- [decoding/metrics.py:336-350](../../../../src/neurospatial/decoding/metrics.py) — `confusion_matrix`; `np.argmax` over a NaN row returns bin 0, and the `"expected"` branch adds NaN mass (finding 4).
- [decoding/likelihood.py:40-181](../../../../src/neurospatial/decoding/likelihood.py) — `log_poisson_likelihood`; a 1-D `spike_counts` (n_neurons,) passes `np.asarray` and is matmul'd as `(n_neurons,) @ (n_neurons, n_bins) -> (n_bins,)`, silently collapsing the time axis (finding 5).
- [decoding/likelihood.py:184-256](../../../../src/neurospatial/decoding/likelihood.py) — `poisson_likelihood`; thin wrapper that calls `log_poisson_likelihood`, so the 2-D guard added there covers it, but its `.max(axis=1)` assumes 2-D and must not regress.

**Contracts referenced:**
- [Input-validation helpers](shared-contracts.md#input-validation-helpers) — use `validate_lengths` for the `times`/`n_time_bins` agreement check in `decode_position`; do not weaken (length-only, no broadcast). The bin-count and ndim checks are local raises since they compare a shape axis against a scalar, not two array lengths. If `src/neurospatial/_validation.py` does not yet exist when this phase executes, inline the `times`/`n_time_bins` length raise with the same message shape (`Length mismatch: ...`) rather than blocking on the shared module.

> Verification note: all candidate line numbers from the prior review were confirmed accurate against the current files; no drift. The reactivation EV/REV block is at lines 1096–1122 (REV at 1099), `reactivation_strength` double-z at 960–972, `decode_position` body at 365–399, `credible_region` at 337–371, `confusion_matrix` at 336–350, `log_poisson_likelihood` matmul at 119–181.

## Tasks

### Task 1 — `log_poisson_likelihood`: reject non-2-D `spike_counts` (finding 5)

In `decoding/likelihood.py`, inside `log_poisson_likelihood`, immediately after the `np.asarray` coercions (currently lines 119–121) add an ndim/orientation guard. A 1-D `spike_counts` of shape `(n_neurons,)` currently matmuls to `(n_bins,)`, silently dropping the time axis and producing a 1-D "likelihood" that downstream code treats as a single time bin.

Insert after line 121 (`encoding_models = np.asarray(encoding_models)`):

```python
    # spike_counts must be 2-D (n_time_bins, n_neurons). A 1-D array of
    # shape (n_neurons,) would matmul against the (n_neurons, n_bins)
    # encoding models to give (n_bins,), silently collapsing the time
    # axis and yielding a "likelihood" that downstream code mistakes for
    # a single time bin. Require an explicit time axis instead.
    if spike_counts.ndim != 2:
        raise ValueError(
            f"spike_counts must be 2-D with shape (n_time_bins, n_neurons), "
            f"got ndim={spike_counts.ndim} with shape {spike_counts.shape}. "
            f"For a single time bin pass shape (1, n_neurons), e.g. "
            f"`spike_counts[np.newaxis, :]`."
        )
    if encoding_models.ndim != 2:
        raise ValueError(
            f"encoding_models must be 2-D with shape (n_neurons, n_bins), "
            f"got ndim={encoding_models.ndim} with shape {encoding_models.shape}."
        )
    if spike_counts.shape[1] != encoding_models.shape[0]:
        raise ValueError(
            f"Neuron-count mismatch: spike_counts has {spike_counts.shape[1]} "
            f"neurons (axis 1) but encoding_models has {encoding_models.shape[0]} "
            f"neurons (axis 0). These must agree for the Poisson likelihood."
        )
```

`poisson_likelihood` (line 184) needs no change: it delegates to `log_poisson_likelihood` (line 248), so the guard fires before its own `.max(axis=1)`. Verify the existing doctest at likelihood.py:234-240 still passes (it uses a 2-D `spike_counts`, so it does).

### Task 2 — `decode_position`: validate bin count and timestamp length (finding 2)

In `decoding/posterior.py`, extend `_validate_inputs` to also check `encoding_models` bin count against `env.n_bins`, and add a `times` length check in `decode_position`. Currently `_validate_inputs(spike_counts, encoding_models, prior)` (called at line 382) has no access to `env`, so thread `env` (and `times`) through.

Change the call site (line 382) and signature (line 402). New `decode_position` body region (replacing lines 380–399):

```python
    # Validate inputs if requested
    if validate:
        _validate_inputs(spike_counts, encoding_models, prior, env)

    # Bin-count agreement is a *correctness* check, not an opt-in one:
    # an encoding model with the wrong number of bins produces a
    # well-formed posterior over the WRONG positions with no error.
    # Enforce it even when validate=False.
    if encoding_models.ndim != 2:
        raise ValueError(
            f"encoding_models must be 2-D (n_neurons, n_bins), got shape "
            f"{encoding_models.shape}."
        )
    if encoding_models.shape[1] != env.n_bins:
        raise ValueError(
            f"encoding_models has {encoding_models.shape[1]} bins (axis 1) "
            f"but env has {env.n_bins} active bins. The encoding models must "
            f"be defined on the same environment used for decoding."
        )

    # Compute log-likelihood using Poisson model
    log_ll = log_poisson_likelihood(spike_counts, encoding_models, dt=dt)

    # Normalize to posterior
    posterior = normalize_to_posterior(log_ll, prior=prior)

    # Validate output if requested
    if validate:
        _validate_output(posterior)

    # Handle times
    if times is not None:
        times = np.asarray(times, dtype=np.float64)
        n_time_bins = posterior.shape[0]
        if times.ndim != 1 or len(times) != n_time_bins:
            raise ValueError(
                f"Length mismatch: times has {times.shape} but posterior has "
                f"{n_time_bins} time bins. `times` must be a 1-D array of bin "
                f"centers, one per row of spike_counts."
            )

    # Return DecodingResult
    return DecodingResult(posterior=posterior, env=env, times=times)
```

Update `_validate_inputs` signature (line 402) to accept `env` and add the bin-count check there as well so the rich-diagnostics path covers it; the unconditional check above is the floor that runs even with `validate=False`. New signature and added block:

```python
def _validate_inputs(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    prior: NDArray[np.float64] | None,
    env: Environment,
) -> None:
```

and, after the existing `encoding_models` negativity check (currently ends at line 474), append:

```python
    # Encoding models must be defined on the decoding environment.
    if encoding_models.ndim == 2 and encoding_models.shape[1] != env.n_bins:
        raise ValueError(
            f"encoding_models has {encoding_models.shape[1]} bins (axis 1) "
            f"but env has {env.n_bins} active bins. Recompute the place "
            f"fields on this environment before decoding."
        )
```

`Environment` is already imported in posterior.py (used in the signature at line 266); confirm the import is module-level and add it to `_validate_inputs`'s type annotations only (no new import needed).

### Task 3 — `credible_region`: exclude NaN bins from the HPD set (finding 3)

In `decoding/estimates.py`, `credible_region` sorts each posterior row in descending order (line 342). NumPy's `argsort` places NaN at the **end** ascending, hence at the **front** after `[:, ::-1]`, so a NaN bin becomes the reported highest-density member. Replace NaN with `-inf` before sorting so NaN bins sink to the back and never enter the HPD region, and renormalize-safe the cumsum against the finite mass.

Replace lines 340–350 (from the `# Vectorized sorting` comment through the `cumsum` assignment):

```python
    # NaN bins (e.g. masked/off-track positions) must not be reported as
    # high-density members of the HPD region. argsort puts NaN last in
    # ascending order, i.e. FIRST after the descending flip — which would
    # name a NaN bin as the single highest-density bin. Sink them with
    # -inf so they sort to the back and are excluded by the cumulative
    # mass threshold.
    safe_posterior = np.where(np.isfinite(posterior), posterior, -np.inf)

    # Vectorized sorting: sort all rows at once (descending order).
    # argsort returns ascending, so we use [:, ::-1] to reverse.
    sorted_indices = np.argsort(safe_posterior, axis=1)[:, ::-1]

    # Gather sorted probabilities using advanced indexing.
    # sorted_probs[t, i] = safe_posterior[t, sorted_indices[t, i]]
    row_indices = np.arange(n_time_bins)[:, np.newaxis]
    sorted_probs = safe_posterior[row_indices, sorted_indices]

    # Treat the -inf placeholders as zero mass for the cumulative sum so
    # they can never push cumsum over `level`.
    sorted_probs = np.where(np.isfinite(sorted_probs), sorted_probs, 0.0)

    # Vectorized cumulative sum across bins.
    cumsum = np.cumsum(sorted_probs, axis=1)
```

The downstream `reached_level`, `n_bins_needed`, `never_reached`, and result-loop (lines 356–369) are unchanged. With NaN bins contributing 0 mass and sorted last, a row that is all-NaN (no finite mass) hits `never_reached` and returns all bins as before — but no NaN bin is ever the *leading* HPD member. (Optional, document only: if every bin is NaN the region is meaningless; this is out of scope and left as the existing all-bins fallback.)

### Task 4 — `confusion_matrix`: handle NaN posterior rows (finding 4)

In `decoding/metrics.py`, `confusion_matrix` uses `np.argmax(posterior, axis=1)` (line 340) for the `"map"` branch — `argmax` over a row containing NaN returns the index of the first NaN (or 0), silently misattributing mass to bin 0. The `"expected"` branch (line 348) adds NaN posterior rows directly into the matrix, poisoning entire rows. Skip rows whose posterior is not finite.

Replace the dispatch block (lines 338–348) with:

```python
    # Rows with any non-finite posterior entry are undecodable (e.g. a
    # time bin with no spikes against a masked environment). argmax over
    # a NaN row returns bin 0, and accumulating a NaN row into the
    # "expected" matrix poisons that actual-bin's entire row. Drop such
    # rows from both summaries and surface how many were excluded.
    finite_rows = np.isfinite(posterior).all(axis=1)
    n_dropped = int(np.sum(~finite_rows))
    if n_dropped > 0:
        warnings.warn(
            f"confusion_matrix: skipped {n_dropped} of {n_time_bins} time "
            f"bins whose posterior contained NaN/Inf. The returned matrix is "
            f"built from the remaining {n_time_bins - n_dropped} bins.",
            UserWarning,
            stacklevel=2,
        )

    if summary_method == "map":
        # Use argmax to get decoded bin for each (finite) time step.
        decoded_bins = np.argmax(posterior[finite_rows], axis=1)
        # Vectorized counting using np.add.at.
        np.add.at(cm, (actual_bins[finite_rows], decoded_bins), 1.0)

    else:  # summary_method == "expected"
        # Accumulate posterior mass only from finite rows.
        np.add.at(cm, actual_bins[finite_rows], posterior[finite_rows])
```

Confirm `import warnings` is present at the top of `decoding/metrics.py`; if absent, add it (it is a stdlib import, place it with the other stdlib imports). `n_time_bins` is already bound at line 303.

### Task 5 — `explained_variance_reactivation`: real REV and control-aware EV (findings 1a, 1b)

In `decoding/assemblies.py`, the EV/REV/partial block (lines 1091–1122) currently sets `rev = ev` (line 1099, inert control) and computes `ev = r_tm**2` ignoring `control_correlations` (the control only feeds `partial_corr`). Implement the Kudrimoti (1999) role-swapped REV and make EV the squared **partial** correlation when a control is supplied.

The Kudrimoti EV/REV definitions, with t=template, m=match (test), c=control (pre):

- EV = (partial correlation of template↔match, controlling for control)² = `r_partial(t,m | c)²`
- REV = same formula with the **roles of template and control swapped**: `r_partial(c,m | t)²`. For true reactivation EV > REV.
- Partial correlation: `r(a,b|c) = (r_ab − r_ac·r_bc) / sqrt((1−r_ac²)(1−r_bc²))`.

When no control is supplied, fall back to the simple symmetric definition where EV = `r_tm²` and REV is computed against a degenerate (zero) control, i.e. REV = `r_tm²` as well — but rather than hardcode equality, compute it through the same partial-correlation path with `control = 0`-correlations so the two are *structurally* derived, and document that EV==REV without a control is expected (the control is what breaks the symmetry).

Replace lines 1091–1122 (from `# Compute correlation between template and match` through the `return ExplainedVarianceResult(...)`):

```python
    # Pairwise correlation between template and match correlation vectors.
    r_tm = np.nan_to_num(np.corrcoef(template_valid, match_valid)[0, 1], nan=0.0)

    if control_correlations is not None:
        control_valid = control_correlations[valid_mask]
        r_tc = np.nan_to_num(np.corrcoef(template_valid, control_valid)[0, 1], nan=0.0)
        r_mc = np.nan_to_num(np.corrcoef(match_valid, control_valid)[0, 1], nan=0.0)

        # EV: variance of match correlations explained by template, after
        # partialling out the control (pre-behavior) baseline.
        #   r_partial(template, match | control)
        denom_ev = np.sqrt((1.0 - r_tc**2) * (1.0 - r_mc**2))
        partial_corr = (r_tm - r_tc * r_mc) / denom_ev if denom_ev > 1e-10 else 0.0
        ev = partial_corr**2

        # REV (Kudrimoti 1999): swap the roles of template and control.
        # The match's correlation with the control, partialled for the
        # template, is the reverse-direction "explained variance". For
        # genuine reactivation EV should exceed REV.
        #   r_partial(control, match | template)
        denom_rev = np.sqrt((1.0 - r_tc**2) * (1.0 - r_tm**2))
        rev_partial = (r_mc - r_tc * r_tm) / denom_rev if denom_rev > 1e-10 else 0.0
        rev = rev_partial**2
    else:
        # No control: EV is the plain squared correlation. REV reduces to
        # the same value because there is no asymmetry to break — the
        # control is what makes EV != REV meaningful. This is intentional,
        # not the old hardcoded `rev = ev` shortcut.
        partial_corr = r_tm
        ev = r_tm**2
        rev = r_tm**2

    return ExplainedVarianceResult(
        explained_variance=float(ev),
        reversed_ev=float(rev),
        partial_correlation=float(partial_corr),
        n_pairs=int(np.sum(valid_mask)),
    )
```

This removes the old `ev = r_tm**2` at line 1096, the inert `rev = ev` at line 1099, and the separate `if control_correlations is not None:` partial-correlation block at lines 1102–1115 (folded into the unified branch above). The validation/NaN-mask block at lines 1055–1089 is unchanged.

### Task 6 — `reactivation_strength`: magnitude-sensitive path (finding 1c)

In `decoding/assemblies.py`, `reactivation_strength` (lines 910–972) projects both periods through `assembly_activation` with its default `z_score_input=True` (line 832), so each period's activity is independently z-scored to mean 0 / unit scale *before* the magnitude comparison. The ratio `mean(|act_match|)/mean(|act_template|)` is then ≈1 regardless of how much stronger the match-period activity actually is — the metric is structurally blind to magnitude.

Fix: compute activations **without** the inner per-period z-score so absolute magnitude survives. Use a common normalization (z-score against the *template* statistics) so a match period with larger projected activity yields strength > 1. Replace lines 960–972 (`# Compute activations` through the final `else: return 0.0`):

```python
    # Compute activations WITHOUT the inner per-period z-score. The default
    # assembly_activation(..., z_score_input=True) standardizes each period
    # to unit scale independently, which erases the very magnitude
    # difference this metric is meant to detect (strength would sit at ~1
    # no matter how much stronger the match activity is). Normalize both
    # periods against a *single* baseline (template mean/std) so larger
    # match activity produces strength > 1.
    template_counts = np.asarray(template_counts, dtype=np.float64)
    match_counts = np.asarray(match_counts, dtype=np.float64)

    mean = template_counts.mean(axis=1, keepdims=True)
    std = template_counts.std(axis=1, keepdims=True)
    std = np.where(std > 1e-10, std, 1.0)

    template_norm = (template_counts - mean) / std
    match_norm = (match_counts - mean) / std

    act_template = assembly_activation(
        template_norm, pattern, z_score_input=False
    )
    act_match = assembly_activation(match_norm, pattern, z_score_input=False)

    # Compare mean absolute activation magnitudes on the shared scale.
    mean_template = np.mean(np.abs(act_template))
    mean_match = np.mean(np.abs(act_match))

    if mean_template > 1e-10:
        return float(mean_match / mean_template)
    else:
        return 0.0
```

This relies on `assembly_activation` already supporting `z_score_input=False` (confirmed, line 750/835); no change to `assembly_activation` is required.

### Task 7 — Docstring updates (behavior changes)

- `explained_variance_reactivation` (assemblies.py:981-1053): update the **Reversed EV (REV)** Notes block (lines 1021–1024) to state REV is the role-swapped partial correlation `r_partial(control, match | template)²` per Kudrimoti 1999, and that EV is the *partial*-correlation-squared when a control is given (lines 1010–1019). Note explicitly that without a control EV == REV by construction.
- `reactivation_strength` (assemblies.py:915-959): update Notes (lines 938–950) to state both periods are normalized against template statistics (shared scale, no per-period z-score) so strength > 1 means genuinely larger match-period activation; remove any implication of independent standardization.
- `decode_position` (posterior.py:276-365): in **Raises** (lines 324–328) add that a `ValueError` is raised when `encoding_models` bin count != `env.n_bins` (always, even with `validate=False`) and when `times` length != number of time bins.
- `log_poisson_likelihood` / `poisson_likelihood` (likelihood.py): in **Parameters**/**Raises**, document that `spike_counts` must be 2-D `(n_time_bins, n_neurons)` and a 1-D array now raises rather than collapsing the time axis; show the `[np.newaxis, :]` remedy.
- `credible_region` (estimates.py:300-333) and `confusion_matrix` (metrics.py docstring): add a Notes sentence that NaN/Inf posterior bins are excluded (credible_region: never reported as HPD members; confusion_matrix: such time bins are dropped with a warning).

## Deliberately not in this phase
- Decoding slope-to-speed **units** bug (line-fitting velocity scaling) → phase 22.
- xarray-typed posterior / `to_dataframe()` result-object surface for `DecodingResult` → phase 20 (and the result-mixin contract, phase 17).
- `normalize_to_posterior` prior-handling or float32 memory mode → out of scope (perf, not correctness).
- Generalizing `validate_lengths`/`validate_finite` into a shared `_validation.py` module is owned by the shared-contracts phases; this phase only *uses* `validate_lengths` semantics and may inline the one length raise if the module is not present yet.
- `gaussian`/`clusterless` likelihood models — not implemented; untouched.

## Validation slice

All tests live under `tests/decoding/`. Run with `uv run pytest tests/decoding/ -v`. Each test must FAIL on current `main` and PASS after the corresponding task.

| Test | Asserts |
|------|---------|
| `tests/decoding/test_assemblies.py::test_explained_variance_rev_differs_from_ev_with_control` | On a fixture where template↔match correlation and control↔match correlation differ, `result.reversed_ev != result.explained_variance` (and `result.explained_variance > result.reversed_ev` for the reactivation-positive fixture). FAILS now (REV hardcoded == EV). |
| `tests/decoding/test_assemblies.py::test_explained_variance_uses_control` | With a `control_correlations` strongly correlated to `match`, `result.explained_variance` (partial²) is strictly smaller than the no-control EV (`r_tm²`) on the same template/match. FAILS now (control ignored for EV). |
| `tests/decoding/test_assemblies.py::test_explained_variance_no_control_ev_equals_rev` | Without a control, `ev == rev` and both `== r_tm**2` (guards the documented symmetric fallback; ensures the fix didn't break the no-control case). |
| `tests/decoding/test_assemblies.py::test_reactivation_strength_detects_magnitude` | Build `match_counts = 3 * template_counts` on the assembly's member neurons; assert `reactivation_strength(...) > 1.5`. FAILS now (returns ≈1 due to double z-score). |
| `tests/decoding/test_assemblies.py::test_reactivation_strength_weak_match_below_one` | `match_counts` with much smaller member-neuron activity than template → `strength < 1`. Confirms directionality both ways. |
| `tests/decoding/test_posterior.py::test_decode_position_rejects_bin_count_mismatch` | `encoding_models` with `n_bins != env.n_bins` raises `ValueError` mentioning bins — both with `validate=True` and `validate=False`. FAILS now (silent wrong positions). |
| `tests/decoding/test_posterior.py::test_decode_position_rejects_times_length_mismatch` | `times` of wrong length raises `ValueError` ("Length mismatch"). FAILS now (stored unvalidated). |
| `tests/decoding/test_estimates.py::test_credible_region_excludes_nan_bin` | Posterior row with one NaN bin and a clear finite mode → the returned HPD index array contains the finite mode and **not** the NaN bin index. FAILS now (NaN bin returned as leading HPD member). |
| `tests/decoding/test_metrics.py::test_confusion_matrix_skips_nan_rows_map` | One NaN posterior row among finite rows; `summary_method="map"` → that time bin contributes nothing to `cm` (bin-0 column not inflated) and a `UserWarning` is emitted. FAILS now (argmax→0). |
| `tests/decoding/test_metrics.py::test_confusion_matrix_skips_nan_rows_expected` | Same fixture, `summary_method="expected"` → `cm` is finite (no NaN entries) and the NaN row's `actual_bin` row is unpoisoned. FAILS now (NaN added in). |
| `tests/decoding/test_likelihood.py::test_log_poisson_likelihood_rejects_1d_spike_counts` | 1-D `spike_counts` of shape `(n_neurons,)` raises `ValueError` mentioning 2-D / `(n_time_bins, n_neurons)`. FAILS now (collapses to `(n_bins,)`). |
| `tests/decoding/test_likelihood.py::test_log_poisson_likelihood_neuron_mismatch` | `spike_counts` neuron axis != `encoding_models` neuron axis raises `ValueError`. FAILS now (NumPy matmul error or silent broadcast). |
| `tests/decoding/test_likelihood.py::test_log_poisson_likelihood_2d_single_timebin_ok` | Shape `(1, n_neurons)` still works and returns `(1, n_bins)` (guards the remedy path). Passes before and after — non-regression. |

None are slow; all are unit-level (small synthesized arrays). No integration/real-data test required for this phase.

## Fixtures

- **Synthesized (in-test or `tests/decoding/conftest.py`).** No `tests/decoding/conftest.py` exists yet; if more than two tests share a fixture, add one with:
  - `small_env`: `Environment.from_samples(np.random.default_rng(0).uniform(0, 100, (500, 2)), bin_size=10.0)` for the decode/credible/confusion tests (gives a known `env.n_bins`).
  - `assembly_fixture`: deterministic `template_counts`/`match_counts` (shape `(n_neurons, n_time_bins)`) plus an `AssemblyPattern` whose weights load one clear assembly, using `np.random.default_rng(seed)` for reproducibility. The reactivation-magnitude test derives `match_counts` from `template_counts` by a known scalar so the expected `strength` is pinnable.
  - `corr_triplet`: three correlation vectors (`template`, `match`, `control`) constructed so `r(template,match)`, `r(control,match)` are known and distinct, to pin EV != REV.
- **Real-data slice:** none required; all defects are reproducible with tiny synthesized arrays and pinned scientific values.

## Review

Standard review checklist — dispatch the `code-reviewer` agent against the phase diff and confirm:

- [ ] Every task (1–7) implemented; diff is scoped entirely to `src/neurospatial/decoding/` and `tests/decoding/`.
- [ ] The old code paths are removed, not merely shadowed: `rev = ev` (assemblies.py:1099) gone; the standalone partial-correlation `if` block (assemblies.py:1102-1115) folded into the unified branch; the double-z `reactivation_strength` body (assemblies.py:960-972) replaced; `_validate_inputs` call site updated to pass `env`.
- [ ] Each correctness fix has a regression test that **fails on `main`** and **passes** after — verified by running the named tests on both states.
- [ ] Regression tests pin corrected scientific values (REV != EV and EV > REV on the reactivation fixture; control lowers EV; `strength > 1.5` for 3× match, `< 1` for weak match), not just "no exception".
- [ ] "Deliberately not in this phase" honored — no slope-units, xarray, or shared-`_validation` module work crept in.
- [ ] No code, test, fixture, or module name references "phase 1", "remediation", or this plan.
- [ ] Docstrings updated wherever behavior changed (Task 7 items all present); NumPy docstring style preserved.
- [ ] `uv run pytest tests/decoding/ -v` green; `uv run ruff check src/neurospatial/decoding/ tests/decoding/ && uv run ruff format --check src/neurospatial/decoding/ tests/decoding/` clean; `uv run mypy src/neurospatial/decoding/` clean (note the new `env: Environment` annotation on `_validate_inputs`).
