# Phase 1 — Red-table real bugs

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Ships the five concrete bugs / silent-contract violations the audit surfaced. Each source change is paired with a regression test that would have caught it. The fifth item (CLAUDE.md Tier 4 drift) is doc-only. The sixth (OVC convention test) is test-only.

**Inputs to read first:**

- [src/neurospatial/ops/egocentric.py:540-590](../../../src/neurospatial/ops/egocentric.py) — `compute_egocentric_distance` geodesic branch. The bug is on line 569: `for i, target in enumerate(targets_3d[0])` with an inline comment `# Assume same targets over time`. The Euclidean branch above handles time-varying targets correctly via broadcasting; the geodesic branch was written by hand and only iterates the first timestep.
- [src/neurospatial/events/regressors.py:1-30](../../../src/neurospatial/events/regressors.py) — module docstring lists `exponential_kernel` as a "Temporal regressor", but no definition follows in the file.
- [src/neurospatial/encoding/egocentric.py:1610-1750](../../../src/neurospatial/encoding/egocentric.py) — `object_vector_score(tuning_curve, ...)` definition. Returns a scalar OVC score in `[0, 1]` quantifying how concentrated a tuning curve is in distance-direction space. Currently exported but never tested directly.
- [tests/simulation/models/test_object_vector_cells.py:498-516](../../../tests/simulation/models/test_object_vector_cells.py) — `test_increasing_heading_decreases_rate`. Inline comment says "heading=pi/2 (North), object is to the right". The CLAUDE.md egocentric convention is π/2 = **left**, -π/2 = right. The test passes either way because it only asserts monotonic decrease.
- [CLAUDE.md](../../../CLAUDE.md) "Tier 4 - Domains" section under "Architecture Overview". Lists `encoding/place.py`, `head_direction.py`, `object_vector.py`, `spatial_view.py` — none exist. Actual files: `spatial.py`, `directional.py`, `egocentric.py`, `view.py`.
- [tests/test_m10_old_files_deleted.py:55-70](../../../tests/test_m10_old_files_deleted.py) — already asserts the old module paths raise `ModuleNotFoundError`. Use this list to confirm what the canonical-and-deleted paths are.

## Tasks

### 1. Fix `compute_egocentric_distance` geodesic time-varying targets

[src/neurospatial/ops/egocentric.py:569](../../../src/neurospatial/ops/egocentric.py): replace the `targets_3d[0]` static-target loop with a per-timestep loop that recomputes the target bin when the target moves. The Euclidean branch (above) uses broadcasting; the geodesic branch needs an explicit double loop because `compute_distance_field` is per-source-bin.

Replacement structure (algorithm — replaces the `for i, target in enumerate(targets_3d[0]):` block and inner `dist_field[pos_bins]` lookup):

```python
# Cache distance fields by target bin to avoid recomputing for repeated targets.
distance_field_cache: dict[int, NDArray[np.float64]] = {}

for t in range(n_time):
    pos_bin = int(pos_bins[t])
    for i in range(n_targets):
        target = targets_3d[t, i]  # shape (2,)
        target_bins = env.bin_at(target.reshape(1, -1))
        target_bin = int(target_bins[0])
        if target_bin < 0:
            continue  # distances[t, i] stays NaN
        if target_bin not in distance_field_cache:
            distance_field_cache[target_bin] = compute_distance_field(
                env.connectivity, sources=[target_bin]
            )
        distances[t, i] = distance_field_cache[target_bin][pos_bin]
```

The cache keeps the static-target case cheap (one `compute_distance_field` per unique target). The time-varying case pays for as many fields as there are unique target bins.

Remove the `# Assume same targets over time` comment.

### 2. Add regression test for time-varying geodesic targets

In [tests/ops/test_reference_frames.py](../../../tests/ops/test_reference_frames.py), add a class `TestGeodesicDistanceTimeVarying` with two tests:

- `test_moving_target_returns_per_timestep_distance`: construct a 1D linear env (`Environment.from_samples` with `bin_size=1.0` on positions ranging 0–100), animal stationary at position 50 for 3 timesteps, target moves through `[(10,0), (50,0), (90,0)]`. Assert `result[0,0] ≈ 40` (graph distance 50→10), `result[1,0] ≈ 0` (animal on target), `result[2,0] ≈ 40` (graph distance 50→90). Tolerance `< 1.5 * bin_size`.
- `test_static_target_matches_previous_behavior`: animal moving, single target broadcast-to-3D via `np.tile`. Assert geodesic distances match the same calculation done with `metric="euclidean"` within graph-vs-euclidean tolerance on a simple convex environment (open square).

### 3. `exponential_kernel` — implement or remove

First, **executor confirms with `grep -rn exponential_kernel src/ tests/ examples/ docs/`**.

**If zero callers (expected — verified at plan time):** remove the bullet `- exponential_kernel: Convolve events with exponential kernel` from the docstring at [src/neurospatial/events/regressors.py:13](../../../src/neurospatial/events/regressors.py). No test change needed. Add an entry to `CHANGELOG.md` (if one exists at the repo root) under an unreleased section: `Documentation: dropped unimplemented exponential_kernel from events.regressors module docstring.`

**If callers found:** implement the function in `src/neurospatial/events/regressors.py`. Signature and behavior:

```python
def exponential_kernel(
    times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    tau: float,
) -> NDArray[np.float64]:
    """
    Convolve events with a causal exponential kernel.

    For each sample time t, returns sum over events at e <= t of exp(-(t - e) / tau).

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Sample times.
    event_times : ndarray, shape (n_events,)
        Event times.
    tau : float
        Time constant. Must be positive.

    Returns
    -------
    kernel_value : ndarray, shape (n_samples,)
        Convolved signal at each sample time.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    # Vectorized: time differences (n_samples, n_events), mask future events.
    dt = times[:, None] - event_times[None, :]
    mask = dt >= 0
    return np.where(mask, np.exp(-dt / tau), 0.0).sum(axis=1)
```

Add tests in [tests/events/test_regressors.py](../../../tests/events/test_regressors.py) (new class `TestExponentialKernel`):
- `test_single_event_decays`: one event at t=0, kernel at t=tau ≈ 1/e, at t=2*tau ≈ 1/e².
- `test_future_events_ignored`: event at t=10, query at t=5 → 0.
- `test_multiple_events_sum`: two events at t=0 and t=tau, query at t=tau → `1 + 1/e`.
- `test_negative_tau_raises`: `pytest.raises(ValueError, match="positive")`.

### 4. Add tests for `object_vector_score`

Create [tests/encoding/test_object_vector_score.py](../../../tests/encoding/test_object_vector_score.py) (new file). Cover behavioral contract — score is high for concentrated tuning, low for diffuse tuning, monotonic in concentration. Use the existing `polar_egocentric` environment factory for fixture construction.

Test cases (all driving `object_vector_score`, not constructing results by hand):
- `test_perfectly_concentrated_tuning_high_score`: tuning curve with a single non-zero bin → score in top decile (`> 0.9`).
- `test_uniform_tuning_low_score`: tuning curve = constant array → score in bottom decile (`< 0.1`).
- `test_score_monotonic_in_concentration`: parametrize over a Gaussian bump with widths `[1.0, 3.0, 10.0]` bins; assert `score(width=1) > score(width=3) > score(width=10)`.
- `test_score_matches_hand_computed_intermediate`: the score implementation at [src/neurospatial/encoding/egocentric.py:1700-1710](../../../src/neurospatial/encoding/egocentric.py) computes `score = normalized_dist_sel * direction_selectivity` where `direction_selectivity` is the mean resultant length of the direction marginal of the tuning curve, and `normalized_dist_sel` is the distance-selectivity component (read the source for its exact formula). Construct a tuning curve where direction is uniform across one distance bin (so `direction_selectivity = 0`) and assert `score == 0.0` exactly. Construct another where direction is a delta at one bin and distance is a delta at one bin and assert `score == 1.0 * 1.0 = 1.0` within `atol=1e-10`. This pins the multiplicative factorization, not just the `np.clip` boundary (the function ends with `return float(np.clip(score, 0.0, 1.0))` — a bound check is structural, not behavioral).
- `test_all_zero_tuning_curve`: pin the documented behavior — read the docstring at egocentric.py:1610 first. If it raises, assert raise; if it returns NaN, assert NaN; if it returns 0, assert 0.

### 5. Fix CLAUDE.md Tier 4 drift

In [CLAUDE.md](../../../CLAUDE.md), locate "Tier 4 - Domains" under "Architecture Overview". Replace the encoding sub-bullet that reads:

```
  - `place.py`, `grid.py`, `head_direction.py`, `border.py`
  - `object_vector.py`, `spatial_view.py`, `phase_precession.py`, `population.py`
```

with the actual file inventory (verify by `ls src/neurospatial/encoding/`):

```
  - `spatial.py`, `grid.py`, `directional.py`, `border.py`
  - `egocentric.py`, `view.py`, `phase_precession.py`, `population.py`
```

(Confirm `grid.py`, `border.py`, `phase_precession.py`, `population.py` still exist — they do per the `__init__.py` exports.)

### 6. Fix OVC convention test and add a left/right symmetry assertion

In [tests/simulation/models/test_object_vector_cells.py:498-516](../../../tests/simulation/models/test_object_vector_cells.py), `test_increasing_heading_decreases_rate`:

- Fix the inline comment: `# heading=pi/2 (North), object is to the right` → `# heading=pi/2 means the animal is facing North; object due-East is to the animal's right (egocentric bearing = -pi/2)`. Per the CLAUDE.md convention (π/2 = left, -π/2 = right).
- Strengthen the test by adding an explicit left/right symmetry assertion: at `preferred_direction=0` (object straight ahead), the firing rate at `heading=+pi/4` must equal the firing rate at `heading=-pi/4` within Poisson noise. This catches a sign-flip in `compute_egocentric_bearing` that the current monotonic-decrease assertion would miss.

```python
# new sub-test in the existing class
def test_left_right_symmetry_for_ahead_target(self, ovc_model):
    # Object straight ahead → response should be symmetric in heading sign.
    ovc_model.preferred_direction = 0.0
    rate_plus = ovc_model.firing_rate(position=[0, 0], heading=np.pi / 4, object_pos=[0, 10])
    rate_minus = ovc_model.firing_rate(position=[0, 0], heading=-np.pi / 4, object_pos=[0, 10])
    assert np.isclose(rate_plus, rate_minus, rtol=1e-6)
```

(Adjust call signature to whatever `ObjectVectorCellModel.firing_rate` expects — read the source at `src/neurospatial/simulation/models/object_vector_cells.py`.)

## Deliberately not in this phase

- **No end-to-end recovery test for `compute_egocentric_rate`** (place an animal at origin, object ahead, recover preferred direction near 0). That's Phase 2.
- **No NumPy↔JAX parity for `compute_directional_rate`**. Phase 2.
- **No threshold tightening on the existing monotonic-decrease test** beyond adding the symmetry sub-test. Phase 7 sweeps thresholds.
- **No refactor of the geodesic distance algorithm.** The per-timestep loop is fine — the bug is correctness, not performance.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/ops/test_reference_frames.py::TestGeodesicDistanceTimeVarying::test_moving_target_returns_per_timestep_distance` | Geodesic distances reflect target position at each timestep, not always t=0. Would fail today. |
| `tests/ops/test_reference_frames.py::TestGeodesicDistanceTimeVarying::test_static_target_matches_previous_behavior` | Backward-compatible: static-target callers see no behavior change. |
| `tests/events/test_regressors.py::TestExponentialKernel::test_*` (4 tests, if implementing) | Causality, decay, summation, validation. **Skip these tasks entirely if executor confirms zero callers and removes docstring entry.** |
| `tests/encoding/test_object_vector_score.py` (5 tests) | OVC score is high for concentrated tuning, low for diffuse, monotonic in concentration, in [0,1], well-defined for degenerate input. |
| `tests/simulation/models/test_object_vector_cells.py::test_left_right_symmetry_for_ahead_target` | Sign-flip in `compute_egocentric_bearing` would fail this test. |
| Existing test suite (`uv run pytest`) | No regression. The geodesic fix is the only behavior change; existing callers using static (broadcast) targets see identical output. |

## Fixtures

No new fixtures needed. Phase 1 reuses existing `polar_egocentric` env, `ObjectVectorCellModel`, and ad-hoc 1D linear envs constructed inline.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (the `# Assume same targets over time` comment and either the docstring entry for `exponential_kernel` or the placeholder, depending on the decision).
- User-facing documentation listed as tasks is updated, not deferred (CLAUDE.md Tier 4 fix and CHANGELOG entry if removing the docstring).
