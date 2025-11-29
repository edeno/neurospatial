# Decoding Subpackage Implementation Tasks

Implementation tasks for `neurospatial.decoding` based on [DECODING_PLAN.md](DECODING_PLAN.md).

**Target**: Bayesian position decoding from population neural activity.

---

## Milestone 1: Core Decoding Infrastructure

**Goal**: Minimal viable decoding pipeline - from spike counts to posterior.

### 1.1 Package Setup

- [x] Create `src/neurospatial/decoding/` directory
- [x] Create `__init__.py` with placeholder exports
- [x] Add `decoding` to main package `__init__.py` imports

**Success**: `from neurospatial.decoding import decode_position` imports without error (even if not implemented). ✅

### 1.2 DecodingResult Container (`_result.py`)

- [x] Create `_result.py` with `DecodingResult` dataclass
- [x] Implement fields: `posterior`, `env`, `times`
- [x] Implement `@cached_property` for `map_estimate` (argmax)
- [x] Implement `@cached_property` for `map_position` (env.bin_centers lookup)
- [x] Implement `@cached_property` for `mean_position` (posterior @ bin_centers)
- [x] Implement `@cached_property` for `uncertainty` (mask-based entropy)
- [x] Implement `n_time_bins` property
- [x] Add `plot()` method stub (heatmap visualization)
- [x] Add `to_dataframe()` method stub

**Success**:

```python
result = DecodingResult(posterior=np.eye(10), env=env, times=np.arange(10))
assert result.map_estimate.shape == (10,)
assert result.uncertainty.shape == (10,)
```

**Dependencies**: None

### 1.3 Likelihood Functions (`likelihood.py`)

- [x] Implement `log_poisson_likelihood(spike_counts, encoding_models, dt, *, min_rate=1e-10)`
  - Shape: `(n_time_bins, n_neurons)` x `(n_neurons, n_bins)` → `(n_time_bins, n_bins)`
  - Formula: `sum_i [n_i * log(lambda_i * dt) - lambda_i * dt]`
  - Clip rates to `[min_rate, inf)` before log
  - Omit `-log(n_i!)` term (constant across positions)
- [x] Implement `poisson_likelihood()` as thin wrapper
  - Call `log_poisson_likelihood`, then `np.exp(ll - ll.max(axis=1, keepdims=True))`
  - Add docstring warning about underflow

**Success**:

```python
spike_counts = np.array([[0, 1], [2, 0]])  # (2 time bins, 2 neurons)
encoding = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])  # (2 neurons, 3 bins)
ll = log_poisson_likelihood(spike_counts, encoding, dt=0.025)
assert ll.shape == (2, 3)
assert np.isfinite(ll).all()
```

**Dependencies**: None

### 1.4 Posterior Normalization (`posterior.py`)

- [x] Implement `normalize_to_posterior(log_likelihood, *, prior=None, axis=-1, handle_degenerate="uniform")`
  - Log-sum-exp implementation: `ll -= ll.max(keepdims=True); post = exp(ll); post /= post.sum()`
  - Prior handling: normalize to sum=1, apply as `log_likelihood + log(prior)`
  - Degenerate handling: detect all-inf rows, apply `handle_degenerate` strategy
- [x] Implement `decode_position(env, spike_counts, encoding_models, dt, *, prior=None, method="poisson", times=None, validate=False)`
  - Call `log_poisson_likelihood` → `normalize_to_posterior` → `DecodingResult`
  - Implement `validate=True` checks (row sums, NaN/Inf, extreme values)

**Success**:

```python
posterior = normalize_to_posterior(log_likelihood)
assert np.allclose(posterior.sum(axis=1), 1.0)

result = decode_position(env, spike_counts, encoding_models, dt=0.025)
assert isinstance(result, DecodingResult)
assert result.posterior.shape == (n_time_bins, env.n_bins)
```

**Dependencies**: 1.2, 1.3

### 1.5 Estimate Functions (`estimates.py`)

- [x] Implement `map_estimate(posterior)` → bin indices
- [x] Implement `map_position(env, posterior)` → coordinates
- [x] Implement `mean_position(env, posterior)` → coordinates
- [x] Implement `entropy(posterior)` → bits (mask-based)
- [x] Implement `credible_region(env, posterior, level=0.95)` → HPD bin indices

**Success**:

```python
bins = map_estimate(posterior)
pos = map_position(env, posterior)
ent = entropy(posterior)
assert bins.shape == (n_time_bins,)
assert pos.shape == (n_time_bins, env.n_dims)
assert ent.shape == (n_time_bins,)
assert (ent >= 0).all() and (ent <= np.log2(env.n_bins)).all()
```

**Dependencies**: None

### 1.6 Phase 1 Tests

- [x] `tests/decoding/test_result.py`
  - Test DecodingResult properties compute correctly
  - Test cached_property caching behavior
  - Test with edge cases (uniform posterior, delta posterior)
- [x] `tests/decoding/test_likelihood.py`
  - Test log_poisson_likelihood shape and dtype
  - Test min_rate clipping prevents -inf
  - Test against reference implementation (replay_trajectory_classification)
  - Test numerical stability with extreme rates
- [x] `tests/decoding/test_posterior.py`
  - Test normalize_to_posterior sums to 1
  - Test prior application
  - Test handle_degenerate modes
  - Test decode_position end-to-end
- [x] `tests/decoding/test_estimates.py`
  - Test each estimate function
  - Test consistency with DecodingResult properties
  - Test entropy bounds

**Success**: `uv run pytest tests/decoding/ -v` passes.

**Dependencies**: 1.2-1.5

---

## Milestone 2: Quality Metrics

**Goal**: Evaluate decoding accuracy with standard metrics.

### 2.1 Error Metrics (`metrics.py` - Part 1)

- [x] Implement `decoding_error(decoded_positions, actual_positions, *, metric="euclidean", env=None)`
  - Euclidean: `np.linalg.norm(decoded - actual, axis=1)`
  - Graph: Use `env.distance_between()` for shortest-path distance
  - Handle NaN propagation
- [x] Implement `median_decoding_error(decoded_positions, actual_positions)`
  - `np.nanmedian(decoding_error(...))`

**Success**:

```python
errors = decoding_error(decoded, actual)
assert errors.shape == (n_time_bins,)
median = median_decoding_error(decoded, actual)
assert isinstance(median, float)
```

**Dependencies**: Milestone 1

### 2.2 Confusion Matrix (`metrics.py` - Part 2)

- [x] Implement `confusion_matrix(env, posterior, actual_bins, *, method="map")`
  - method="map": Count argmax predictions
  - method="expected": Accumulate full posterior mass
  - Return shape `(n_bins, n_bins)`

**Success**:

```python
cm = confusion_matrix(env, posterior, actual_bins)
assert cm.shape == (env.n_bins, env.n_bins)
assert cm.sum() == n_time_bins  # for method="map"
```

**Dependencies**: Milestone 1

### 2.3 Correlation Metric (`metrics.py` - Part 3)

- [x] Implement `decoding_correlation(decoded_positions, actual_positions, weights=None)`
  - Weighted Pearson correlation with numerically stable formula
  - Use centered computation: `np.average()` for weighted means
  - Handle multi-dimensional positions (mean across dims)
  - Exclude NaN and zero-weight bins

**Success**:

```python
r = decoding_correlation(decoded, actual)
assert -1 <= r <= 1
r_weighted = decoding_correlation(decoded, actual, weights=certainty)
assert -1 <= r_weighted <= 1
```

**Dependencies**: Milestone 1

### 2.4 Phase 2 Tests

- [x] `tests/decoding/test_metrics.py`
  - Test decoding_error with known positions
  - Test graph metric requires env
  - Test confusion_matrix shapes and sums
  - Test decoding_correlation range and symmetry
  - Test NaN handling in all functions
  - Test weighted correlation against unweighted

**Success**: `uv run pytest tests/decoding/test_metrics.py -v` passes. ✅ (64 tests)

**Dependencies**: 2.1-2.3

---

## Milestone 3: Trajectory Analysis

**Goal**: Fit and detect trajectories in posterior sequences.

### 3.1 Result Dataclasses (`trajectory.py` - Part 1)

- [x] Implement `IsotonicFitResult` frozen dataclass
  - Fields: `fitted_positions`, `r_squared`, `direction`, `residuals`
- [x] Implement `LinearFitResult` frozen dataclass
  - Fields: `slope`, `intercept`, `r_squared`, `slope_std`
- [x] Implement `RadonDetectionResult` frozen dataclass
  - Fields: `angle_degrees`, `score`, `offset`, `sinogram`

**Success**: All result dataclasses created and exported. ✅ (12 tests)

**Dependencies**: None

### 3.2 Isotonic Regression (`trajectory.py` - Part 2)

- [x] Implement `fit_isotonic_trajectory(posterior, times, *, increasing=None, method="expected")`
  - Extract positions: `argmax` (method="map") or weighted mean (method="expected")
  - Use `sklearn.isotonic.IsotonicRegression`
  - If `increasing=None`, try both directions, return better R²
  - Return `IsotonicFitResult`

**Success**: ✅ (15 new tests, 27 total in test_trajectory.py)

```python
result = fit_isotonic_trajectory(posterior, times)
assert result.fitted_positions.shape == (n_time_bins,)
assert 0 <= result.r_squared <= 1
assert result.direction in ("increasing", "decreasing")
```

**Dependencies**: 3.1

### 3.3 Linear Regression (`trajectory.py` - Part 3)

- [x] Implement `fit_linear_trajectory(env, posterior, times, *, n_samples=1000, method="sample", rng=None)`
  - method="map": Simple linear regression on argmax positions
  - method="sample": Monte Carlo sampling from posterior
    - Use cumulative-sum sampling for numerical stability
    - Fit line to each sample, average coefficients
  - Handle `rng` parameter for reproducibility
  - Return `LinearFitResult`

**Success**: ✅ (13 new tests, 44 total in test_trajectory.py)

```python
result = fit_linear_trajectory(env, posterior, times, rng=42)
result2 = fit_linear_trajectory(env, posterior, times, rng=42)
assert result.slope == result2.slope  # Reproducible
assert result.slope_std is not None  # Only for method="sample"
```

**Dependencies**: 3.1

### 3.4 Radon Transform Detection (`trajectory.py` - Part 4)

- [x] Add scikit-image optional dependency check
- [x] Implement `detect_trajectory_radon(posterior, *, theta_range=(-90, 90), theta_step=1.0)`
  - Import guard for `skimage.transform.radon`
  - Clear ImportError message with install instructions
  - Compute Radon transform of posterior image
  - Find peak in sinogram
  - Return `RadonDetectionResult`

**Success**: ✅ (21 tests, 20 passed, 1 skipped for import guard)

```python
# With scikit-image installed:
result = detect_trajectory_radon(posterior)
assert -90 <= result.angle_degrees <= 90
assert result.sinogram.ndim == 2

# Without scikit-image:
# ImportError with clear message
```

**Dependencies**: 3.1, optional `scikit-image`

### 3.5 Phase 3 Tests

- [x] `tests/decoding/test_trajectory.py`
  - Test isotonic regression on synthetic monotonic data
  - Test linear regression reproducibility with rng
  - Test Radon detection on diagonal posterior pattern
  - Test result dataclass fields
  - Skip Radon tests if scikit-image not installed

**Success**: `uv run pytest tests/decoding/test_trajectory.py -v` passes.

**Dependencies**: 3.2-3.4

---

## Milestone 4: Shuffle-Based Significance Testing

**Goal**: Statistical shuffling procedures for establishing null distributions.

### 4.1 Core Temporal Shuffles (`shuffle.py` - Part 1)

- [x] Implement `shuffle_time_bins(spike_counts, *, n_shuffles=1000, rng=None)`
  - Generator that yields shuffled versions
  - Randomly permute rows (time bins) of spike_counts
  - Preserve spike counts per neuron per time bin
- [x] Implement `shuffle_time_bins_coherent()` (time-swap shuffle)
  - Same permutation applied to all neurons
  - Preserves instantaneous population vectors

**Success**:

```python
for i, shuffled in enumerate(shuffle_time_bins(spikes, n_shuffles=100, rng=42)):
    assert shuffled.shape == spikes.shape
    assert shuffled.sum() == spikes.sum()  # Same total spikes
    if i >= 99:
        break
```

**Dependencies**: None

### 4.2 Cell Identity Shuffles (`shuffle.py` - Part 2)

- [x] Implement `shuffle_cell_identity(spike_counts, encoding_models, *, n_shuffles=1000, rng=None)`
  - Permute columns (neuron axis) of spike_counts
  - Yield (shuffled_counts, original_models) tuples
- [x] Implement `shuffle_place_fields_circular(encoding_models, *, n_shuffles=1000, rng=None)`
  - Circularly shift each place field by random amount
  - Each neuron shifted independently
- [x] Implement `shuffle_place_fields_circular_2d(encoding_models, env, ...)` for 2D environments

**Success**:

```python
for shuffled_spikes, models in shuffle_cell_identity(spikes, encoding, n_shuffles=10, rng=42):
    assert shuffled_spikes.shape == spikes.shape
    assert models is encoding  # Original models unchanged
```

**Dependencies**: None

### 4.3 Posterior Shuffles (`shuffle.py` - Part 3)

- [x] Implement `shuffle_posterior_circular(posterior, *, n_shuffles=1000, rng=None)`
  - Circularly shift each row (time bin) by random amount
  - Each time bin shifted independently
- [x] Implement `shuffle_posterior_weighted_circular()` with edge effect mitigation

**Success**: ✅ (32 tests for posterior shuffles)

```python
for shuffled_post in shuffle_posterior_circular(posterior, n_shuffles=10, rng=42):
    assert shuffled_post.shape == posterior.shape
    assert np.allclose(shuffled_post.sum(axis=1), 1.0)  # Still normalized
```

**Dependencies**: None

### 4.4 Surrogate Generation (`shuffle.py` - Part 4)

- [x] Implement `generate_poisson_surrogates(spike_counts, dt, *, n_surrogates=1000, rng=None)`
  - Compute mean firing rate per neuron
  - Generate spike counts from Poisson(rate * dt)
- [x] Implement `generate_inhomogeneous_poisson_surrogates()` with smoothed rates

**Success**:

```python
for surrogate in generate_poisson_surrogates(spikes, dt=0.025, n_surrogates=10, rng=42):
    assert surrogate.shape == spikes.shape
    assert surrogate.dtype == np.int64
```

**Dependencies**: None

### 4.5 Significance Testing Functions (`shuffle.py` - Part 5)

- [x] Implement `compute_shuffle_pvalue(observed, null_scores, *, tail="greater")`
  - Monte Carlo p-value with correction: (k + 1) / (n + 1)
  - Support "greater", "less", "two-sided" tails
- [x] Implement `compute_shuffle_zscore(observed, null_scores)`
  - Z-score: (observed - null_mean) / null_std
- [x] Implement `ShuffleTestResult` frozen dataclass
  - Fields: observed_score, null_scores, p_value, z_score, shuffle_type, n_shuffles
  - Property: is_significant (p < 0.05)
  - Method: plot() for null distribution visualization

**Success**: ✅ (31 new tests for significance testing)

```python
p = compute_shuffle_pvalue(5.0, np.array([1, 2, 3, 4]))
assert 0 < p <= 1.0
assert p == (0 + 1) / (4 + 1)  # 0 null values >= 5.0

z = compute_shuffle_zscore(5.0, np.array([1, 2, 3, 4]))
assert np.isfinite(z)
```

**Dependencies**: None

### 4.6 Phase 4 Tests

- [x] `tests/decoding/test_shuffle.py`
  - Test each shuffle function preserves invariants
  - Test reproducibility with rng parameter
  - Test p-value calculation edge cases (observed > all null, observed < all null)
  - Test z-score with zero variance (returns NaN)
  - Test ShuffleTestResult properties

**Success**: `uv run pytest tests/decoding/test_shuffle.py -v` passes.

**Dependencies**: 4.1-4.5

---

## Milestone 5: Integration & Documentation

**Goal**: Polish API and add documentation.

### 5.1 Public API Finalization

- [x] Update `decoding/__init__.py` with all exports per plan
- [x] Update main `neurospatial/__init__.py` with top-level exports
- [x] Verify all `__all__` lists are complete

**Success**: All public symbols importable from documented locations. ✅ (19 import tests)

### 5.2 Visualization

- [ ] Implement `DecodingResult.plot()` - posterior heatmap
- [ ] Implement `DecodingResult.to_dataframe()` - pandas export

**Success**:

```python
result.plot()  # Shows matplotlib figure
df = result.to_dataframe()
assert "time" in df.columns or result.times is None
```

**Dependencies**: Milestone 1

### 5.3 Documentation

- [ ] Add decoding quick reference to CLAUDE.md
- [ ] Create `examples/bayesian_decoding.ipynb` notebook
  - Show end-to-end workflow
  - Demonstrate all main functions
  - Include visualization examples

**Success**: Example notebook runs without error.

**Dependencies**: Milestones 1-3

### 5.4 Optional Dependency Setup

- [ ] Add `[trajectory]` extra to `pyproject.toml` for scikit-image
- [ ] Document optional dependencies in README/docs

---

## Milestone 6: Future Extensions (Low Priority)

These are tracked for future work, not immediate implementation.

### 6.1 Alternative Likelihood Models

- [ ] Gaussian likelihood model (`method="gaussian"`)
- [ ] Clusterless/mark-based likelihood

### 6.2 State-Space Smoothing

- [ ] Kalman filter integration
- [ ] Forward-backward algorithm

### 6.3 Performance

- [ ] GPU acceleration (JAX/CuPy)
- [ ] float32 dtype option for memory reduction
- [ ] Chunked processing for long recordings

### 6.4 NWB Integration

- [ ] `write_decoded_position()` to NWB analysis module
- [ ] `read_encoding_models()` from NWB

---

## Task Dependencies Graph

```
Milestone 1 (Core)
├── 1.1 Package Setup
├── 1.2 DecodingResult ──────────────────┐
├── 1.3 Likelihood ──────────────────────┤
├── 1.4 Posterior ───────────────────────┼── 1.6 Tests
├── 1.5 Estimates ───────────────────────┘
│
├── Milestone 2 (Metrics)
│   ├── 2.1 Error Metrics
│   ├── 2.2 Confusion Matrix ────────────── 2.4 Tests
│   └── 2.3 Correlation
│
├── Milestone 3 (Trajectory)
│   ├── 3.1 Result Dataclasses
│   ├── 3.2 Isotonic ────────────────────┐
│   ├── 3.3 Linear ──────────────────────┼── 3.5 Tests
│   └── 3.4 Radon (optional) ────────────┘
│
├── Milestone 4 (Shuffle)
│   ├── 4.1 Temporal Shuffles
│   ├── 4.2 Cell Identity Shuffles
│   ├── 4.3 Posterior Shuffles ──────────── 4.6 Tests
│   ├── 4.4 Surrogate Generation
│   └── 4.5 Significance Testing
│
└── Milestone 5 (Integration)
    ├── 5.1 Public API
    ├── 5.2 Visualization
    ├── 5.3 Documentation
    └── 5.4 Optional Deps
```

---

## Running Tests

```bash
# All decoding tests
uv run pytest tests/decoding/ -v

# Specific milestone
uv run pytest tests/decoding/test_likelihood.py tests/decoding/test_posterior.py -v

# With coverage
uv run pytest tests/decoding/ --cov=src/neurospatial/decoding
```

---

## Checklist Summary

| Milestone | Tasks | Priority |
|-----------|-------|----------|
| 1. Core Decoding | 6 sections | High |
| 2. Quality Metrics | 4 sections | Medium |
| 3. Trajectory Analysis | 5 sections | Medium |
| 4. Shuffle-Based Significance | 6 sections | Medium |
| 5. Integration & Docs | 4 sections | Medium |
| 6. Future Extensions | 4 sections | Low |
