# Phase 3 — Decoding & stats correctness

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Add closed-form and analytic-reference tests where the current decoding/stats suite asserts only that outputs are normalized or have the right sign. Closes specific named gaps in posterior math, trajectory detection, shuffle nulls, circular statistics, and long-trajectory numerical stability.

**Inputs to read first:**

- [src/neurospatial/decoding/likelihood.py](../../../src/neurospatial/decoding/likelihood.py) — `log_poisson_likelihood(spike_counts, firing_rates, dt)`. Returns log-likelihood per time step per state.
- [src/neurospatial/decoding/posterior.py](../../../src/neurospatial/decoding/posterior.py) — `normalize_to_posterior(log_likelihood, prior)`. Returns posterior probabilities summing to 1 per row.
- [src/neurospatial/decoding/trajectory.py](../../../src/neurospatial/decoding/trajectory.py) — `detect_trajectory_radon(posterior, ...)` returns `(score, angle_degrees)`. Used to detect replay-like trajectories in posteriors.
- [src/neurospatial/stats/shuffle.py](../../../src/neurospatial/stats/shuffle.py) — `shuffle_cell_identity`, `shuffle_spikes_isi`, etc. Audit found these test multiset preservation but not pairing destruction.
- [src/neurospatial/stats/circular.py](../../../src/neurospatial/stats/circular.py) — Rayleigh test, circular-circular correlation. Audit found qualitative `p < 0.001` thresholds with no analytic reference.
- [src/neurospatial/stats/surrogates.py:139-239](../../../src/neurospatial/stats/surrogates.py) — `generate_inhomogeneous_poisson_surrogates`. Audit found rate preservation never tested.
- [tests/decoding/test_posterior.py:96-136](../../../tests/decoding/test_posterior.py) — existing row-sum tests (the pattern to extend).
- [tests/decoding/test_likelihood.py:133-149](../../../tests/decoding/test_likelihood.py) — `test_formula_correctness` is the existing model of how to do analytic-reference testing.
- [tests/decoding/test_trajectory.py:529-587](../../../tests/decoding/test_trajectory.py) — current Radon tests assert only `score > 0`.

## Tasks

### 1. Closed-form posterior on a toy problem

In [tests/decoding/test_posterior.py](../../../tests/decoding/test_posterior.py), add `TestPosteriorClosedForm`:

- `test_two_bin_one_neuron_bayes_closed_form`: 1 neuron, 2 spatial bins, 1 time bin. Firing rates `[λ_0, λ_1] = [5.0, 10.0]` Hz, dt = 0.1 s, observed spike count = 2, uniform prior `[0.5, 0.5]`. Hand-compute the Poisson likelihoods, normalize, and assert `np.testing.assert_allclose(posterior, expected, atol=1e-12)`. The expected values:
  ```python
  # log L_k = n * log(λ_k * dt) - λ_k * dt - log(n!)
  # ll_0 = 2 * log(0.5) - 0.5 - log(2) = -1.886...
  # ll_1 = 2 * log(1.0) - 1.0 - log(2) = -1.693...
  # post = softmax(ll + log_prior); compute analytically and pin to ~1e-12.
  ```
- `test_three_bin_uniform_likelihood_returns_prior`: 1 time bin, 3 bins, all firing rates identical → posterior equals prior exactly (up to floating-point).
- `test_zero_prior_bin_stays_zero`: any likelihood, prior with one zero entry → that posterior bin is exactly 0.

These three tests pin Bayes' rule independent of how the function decomposes log-space arithmetic.

### 2. Long-trajectory underflow stability

In [tests/decoding/test_likelihood.py](../../../tests/decoding/test_likelihood.py) and [tests/decoding/test_posterior.py](../../../tests/decoding/test_posterior.py), add `TestLongTrajectoryStability` (one class per file, sharing the same fixture):

- `test_100k_timebins_50_neurons_finite_posterior`: simulate `n_time=100_000`, `n_neurons=50`, `n_bins=80`. Firing rates `~ U[0.5, 30] Hz`, `dt=0.02 s`. Spike counts ~ Poisson. Compute log-likelihood, then posterior. Assert `np.isfinite(posterior).all()` and `np.allclose(posterior.sum(axis=1), 1.0, atol=1e-10)`. Mark `@pytest.mark.slow`.
- `test_extreme_firing_rates_no_overflow`: `lambda * dt` reaching 200 (very high rate × big timestep). Assert no `inf` or `nan` in log-likelihood.

### 3. Radon trajectory angle recovery

In [tests/decoding/test_trajectory.py](../../../tests/decoding/test_trajectory.py), replace the toothless `score > 0` assertions in the existing diagonal-trajectory tests (lines 529-587) with angle-recovery assertions. Add `TestRadonAngleRecovery`:

- `test_45_degree_trajectory_recovered`: construct a posterior where the argmax traces a 45° line across `(time, bin)` (e.g. `posterior[t, t]` is sharply peaked for each `t`). Assert `abs(result.angle_degrees - 45.0) < 5.0`.
- `test_horizontal_stationary_trajectory_angle_zero`: posterior peaked at the same bin for all time. Assert `abs(result.angle_degrees) < 5.0`.
- `test_negative_slope_trajectory_recovered`: argmax decreases over time. Assert `abs(result.angle_degrees - (-45.0)) < 5.0` (or whatever the sign convention is — read the docstring; if the convention is `[0, 180)` then assert `135` instead and document).
- `test_uniform_posterior_returns_low_score`: posterior is uniform `1/n_bins` everywhere. Assert `score < 0.1` (well below the threshold for any meaningful detection).

Keep the existing tests but add the new ones. Phase 7 may remove the old `score > 0` lines after these are in.

### 4. Shuffle pairing destruction

In [tests/stats/test_stats_shuffle.py](../../../tests/stats/test_stats_shuffle.py), add `TestShufflePairingDestruction`:

For each shuffle function (`shuffle_cell_identity`, `shuffle_spikes_isi`, plus any others present), add two paired assertions per call:

- **Marginal preserved**: e.g. for `shuffle_cell_identity` on a `(n_time, n_neurons)` spike count matrix, the per-neuron total `spike_counts.sum(axis=0)` is preserved as a multiset across the shuffle (matches `np.sort` element-wise).
- **Pairing destroyed**: the position of each per-neuron column changes with high probability. Run the shuffle 200 times with different seeds; for each, record a per-column hash (e.g. `hash(tuple(col))`). Assert that for at least 90% of seeds, the original column-0 hash is not at column-0 in the shuffle.

This is the test that **actually justifies the shuffle as a null distribution** — the audit found that current tests verify what is preserved but never that the relevant correlation structure is broken.

### 5. Rayleigh test and circular-linear correlation pinned to analytic values

In [tests/stats/test_circular_metrics.py](../../../tests/stats/test_circular_metrics.py), strengthen the existing qualitative tests:

- For `rayleigh_test`: construct a von Mises sample with `kappa=2.0, mu=0.0, n=100` (`scipy.stats.vonmises.rvs(kappa=2.0, loc=0.0, size=100, random_state=42)`). The expected mean resultant length `R̄ ≈ I_1(2)/I_0(2) ≈ 0.6977`. Expected Rayleigh `z = n * R̄² ≈ 48.7`. Expected `p ≈ 8.5e-22`. Pin: `assert_allclose(R_bar, 0.69, atol=0.05)` (sampling noise budget); `assert_allclose(z, 48, atol=3)`; `assert p < 1e-15`.
- For `circular_linear_correlation`: hand-construct paired angle/linear data with a known correlation (`np.cos(linear)` vs `linear` gives `r=1`; add noise for known `r ≈ 0.7`). Pin against a re-derived expected value within sampling tolerance.

### 6. Inhomogeneous-Poisson surrogate rate preservation

In [tests/stats/test_stats_surrogates.py](../../../tests/stats/test_stats_surrogates.py), add `TestInhomogeneousPoissonRatePreservation`:

- `test_average_surrogate_matches_input_rate`: define a ramping rate `λ(t) = 1.0 + 9.0 * (t / T)` Hz. Generate spike counts from this rate. Run `generate_inhomogeneous_poisson_surrogates(spike_counts, n_surrogates=1000, seed=42)`. Average across surrogates; smooth with a Gaussian kernel (`sigma = 5 bins`). Assert the smoothed average matches the smoothed input rate within Poisson noise (`assert_allclose(mean, expected, rtol=0.15)` — noise budget for n=1000 surrogates).
- `test_constant_rate_surrogate_matches_constant`: special-case the constant-rate input → surrogate should look statistically identical to a fresh draw.

### 7. Add ISI re-pairing assertion for `shuffle_spikes_isi`

In [tests/stats/test_stats_shuffle.py](../../../tests/stats/test_stats_shuffle.py), `TestShuffleSpikesIsi`:

- `test_isi_order_is_destroyed`: starting from a spike train where ISIs are monotonically increasing (`event_times = np.cumsum(np.arange(1, n+1))`), shuffle 200 times. For each shuffle, compute the correlation between original ISI sequence and shuffled ISI sequence; assert mean absolute correlation across seeds is `< 0.1`. (Multiset preservation already tested at lines 223-238 per audit; this complements.)

### 8. Add `align_events` zero-relative-time edge case

In [tests/events/test_alignment.py](../../../tests/events/test_alignment.py), after line 833, add `test_align_events_at_reference_time`: events exactly equal to the reference time. Assert `relative_time == 0` is preserved (not dropped by a `>0` filter).

## Deliberately not in this phase

- **No new GLM regressor tests beyond what Phase 1 added for `exponential_kernel`.** The audit flagged design-matrix conditioning gaps (`test_stats_circular.py:37-46` orthogonality, `test_regressors.py` rank); deferred to a future plan. Phase 3 stays focused on decoding/null/statistics correctness.
- **No `_compute_r_squared` ss_tot=0 branch tests.** Audit cited [trajectory.py:354](../../../src/neurospatial/decoding/trajectory.py) — interesting but a corner case; deferred.
- **No closed-form `circular_basis_metrics` p-value test against chi-squared.** Useful but out of scope; Phase 3 is already ~400 LOC.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/decoding/test_posterior.py::TestPosteriorClosedForm` (3 tests) | Bayes' rule pinned at atol=1e-12 on a toy 1-neuron 2-bin problem. |
| `tests/decoding/test_likelihood.py::TestLongTrajectoryStability::test_100k_timebins_*` | No NaN/Inf on 100k timesteps × 50 neurons. **Mark `@pytest.mark.slow`.** |
| `tests/decoding/test_trajectory.py::TestRadonAngleRecovery` (4 tests) | Radon recovers diagonal angle within ±5°; uniform posterior gives low score. |
| `tests/stats/test_stats_shuffle.py::TestShufflePairingDestruction` (2+ tests, one per shuffle fn) | Marginal preserved AND pairing destroyed in ≥90% of seeds. |
| `tests/stats/test_circular_metrics.py::test_rayleigh_vonmises_kappa2_pinned` | Rayleigh R̄, z, p pinned to analytic values within sampling tolerance. |
| `tests/stats/test_stats_surrogates.py::TestInhomogeneousPoissonRatePreservation` (2 tests) | Average of 1000 surrogates matches input rate within 15%. **Mark `@pytest.mark.slow`.** |
| `tests/stats/test_stats_shuffle.py::test_isi_order_is_destroyed` | Shuffled ISIs uncorrelated with original (mean \|r\| < 0.1 over 200 seeds). |
| `tests/events/test_alignment.py::test_align_events_at_reference_time` | Event exactly at t=0 is not dropped. |

## Fixtures

In `tests/decoding/conftest.py` (extend or create):
- `long_trajectory_session`: `n_time=100_000, n_neurons=50, n_bins=80`, seeded. Memoize across tests with `@pytest.fixture(scope="module")`.
- `closed_form_posterior_inputs`: returns the 1-neuron 2-bin firing rates / dt / spike-count / prior tuple used by the closed-form test.

In `tests/stats/conftest.py`:
- `vonmises_sample_kappa2_n100`: deterministic von Mises sample for Rayleigh pinning.
- `ramping_rate_spike_counts`: spike counts under `λ(t) = 1 + 9*(t/T)` for surrogate rate-preservation test.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — closed-form tests use hand-computed expected values, not values pulled from the function under test. Shuffle pairing-destruction tests use ≥100 seeds to make the 90%-broken assertion meaningful. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (none — Phase 7 removes the old `score > 0` lines).
- User-facing documentation listed as tasks is updated, not deferred (none in this phase).
