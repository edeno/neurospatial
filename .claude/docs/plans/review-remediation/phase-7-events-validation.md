# Phase 7 — Events: validation correctness

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#input-validation-helpers)

One PR, scoped strictly to `src/neurospatial/events/`. Fixes three silent-correctness
bugs where the events subsystem accepts malformed input and returns a plausible-but-wrong
DataFrame/array instead of raising:

1. `events_to_intervals(match_by=...)` silently emits a **Cartesian product** when the
   match key is non-unique (its "unmatched values" check only compares the *sets* of key
   values, so duplicate keys pass and `DataFrame.merge(..., how="inner")` cross-joins them).
2. `distance_to_reward` silently produces wrong distances when `reward_times`/`positions`
   contain **NaN/Inf**, and infers reward positions with `np.interp(..., times, ...)` which
   **requires `times` ascending** — unsorted `times` yield silently-wrong interpolation.
3. `add_positions` silently returns **all-NaN / Inf positions** for degenerate trajectories
   (a single position sample, or all-identical `times`) because `scipy.interpolate.interp1d`
   cannot interpolate from them.

Every fix turns a silently-wrong result into a raised `ValueError` with an actionable
message (matching the project's rich-diagnostic `WHY/HOW` bar, e.g. the `E1001`
"no active bins" message).

**Inputs to read first:**

- [../../../../src/neurospatial/events/intervals.py:204](../../../../src/neurospatial/events/intervals.py) — `events_to_intervals` (signature 204–210; `match_by` branch 328–375). The "unmatched values" guard at **343–356** builds `start_values = set(...)`, `stop_values = set(...)` and only rejects values present on one side. Duplicate keys on **both** sides pass this check; the `start_df.merge(stop_df, on=match_by, how="inner")` at **363** then cross-joins them. Verified: 3 starts + 3 stops with key counts `{1:2, 2:1}` on each side yield **5** rows, not 3.
- [../../../../src/neurospatial/events/regressors.py:487](../../../../src/neurospatial/events/regressors.py) — `distance_to_reward` (signature 487–496; body 614–779). Input cast at **617–619** (`positions`, `times`, `reward_times`), no finite guard. The reward-position interpolation branch is **645–672**; the load-bearing call is `np.interp(clipped_reward_times, times, positions[:, dim])` at **669** inside the `np.column_stack` at **667–672**. `np.interp` requires its `xp` argument (`times`) to be **ascending**; `times` is never sorted here (unlike `add_positions`), so unsorted timestamps silently corrupt the inferred reward positions.
- [../../../../src/neurospatial/events/regressors.py:111](../../../../src/neurospatial/events/regressors.py) — `time_to_nearest_event` NaN/Inf guard (**116–143**): the existing in-module pattern this phase copies for `distance_to_reward` (`np.isnan`/`np.isinf` → `ValueError` with `WHY/HOW`). Reuse the **shared** `validate_finite` instead (see contract), but match the message tone.
- [../../../../src/neurospatial/events/detection.py:22](../../../../src/neurospatial/events/detection.py) — `add_positions` (signature 22–28; body 93–172). Length check at **114–120**. The interpolation block is **141–162**: `sort_idx = np.argsort(times)` (142), then a per-dim `interp1d(sorted_times, sorted_positions[:, dim], kind="linear", fill_value="extrapolate")` at **152–157**. Verified degenerate behavior: a single `times` sample → `interp1d(...)` returns **NaN**; all-identical `times` (duplicate `xp`) → returns **NaN/Inf**. The NaN-timestamp branch at **160–162** is intentional and stays. `detection.py` has **no `warnings` import** today (imports are 15–19) — not needed by this phase, which raises rather than warns.

**Contracts referenced:**

- [Input-validation helpers](shared-contracts.md#input-validation-helpers) — `validate_finite(a, *, name, allow_nan=False)` is the NaN/Inf guard for `distance_to_reward`'s `times`/`positions`/`reward_times` (finding 2) and `add_positions`' degeneracy preconditions (finding 3). **Do not weaken:** `validate_finite` raises on Inf always and on NaN unless `allow_nan=True`; it never coerces NaN→0 or drops values. For `add_positions`, `times`/`positions` may legitimately need NaN dropped by the caller, but the *event* timestamps' NaN→NaN-position behavior is a separate, documented path that is preserved. If `src/neurospatial/_validation.py` already exists when this PR is cut, **import** it; do not re-implement.

## Tasks

### Task 1 — Ensure the shared validation module exists

`src/neurospatial/_validation.py` is introduced in phase 4 (the first scheduled consumer).
If it is not yet present when this PR is cut, create it **verbatim** from
[shared-contracts.md](shared-contracts.md#input-validation-helpers) (`validate_finite` +
`validate_lengths`). If it already exists, import from it — do **not** redefine. This phase
consumes only `validate_finite`.

### Task 2 — Reject non-unique `match_by` keys in `events_to_intervals`

In `events_to_intervals` (`intervals.py`), the `match_by` branch validates *set difference*
(unmatched values) at lines **343–356** but not *per-key multiplicity*. Add a one-to-one
uniqueness check **immediately after** the unmatched-values block (after line 356, before the
`# Merge on match_by column` comment at line 358), so a duplicate key fails fast instead of
cross-joining:

```python
        # Each match_by value must appear at most once on each side. Without
        # this, DataFrame.merge(how="inner") forms a Cartesian product for any
        # duplicated key, silently inflating the interval count (e.g. two
        # starts and two stops sharing a key yield four intervals, not two).
        start_dups = start_events[match_by][
            start_events[match_by].duplicated(keep=False)
        ].unique()
        stop_dups = stop_events[match_by][
            stop_events[match_by].duplicated(keep=False)
        ].unique()
        if len(start_dups) > 0 or len(stop_dups) > 0:
            raise ValueError(
                f"Duplicate values in '{match_by}' column prevent one-to-one "
                f"start/stop pairing.\n"
                f"  Duplicated in start_events: {sorted(start_dups.tolist())}\n"
                f"  Duplicated in stop_events: {sorted(stop_dups.tolist())}\n"
                "  WHY: Matching by a non-unique key would cross-join "
                "(Cartesian product) every repeated value, producing more "
                "intervals than real start/stop pairs.\n"
                "  HOW: De-duplicate so each value of "
                f"'{match_by}' appears once per side, or use match_by=None "
                "for sequential pairing of equal-length start/stop events."
            )
```

Notes for the executor:
- This runs **after** the existing unmatched-values check, so a key that is both duplicated
  and unmatched reports the unmatched error first (existing behavior). That ordering is fine —
  both raise `ValueError`.
- `.duplicated(keep=False)` flags **all** members of any group with count > 1; `.unique()`
  collapses them to the distinct offending key values for the message. `sorted(... .tolist())`
  keeps the message deterministic for both numeric and string keys.
- After this guard, the `merge(..., how="inner")` at line 363 is guaranteed one-to-one.

Update the `events_to_intervals` `Raises` docstring (intervals.py:241–247) to add: a
`ValueError` is raised when the `match_by` column contains duplicate values in either
DataFrame (one-to-one pairing required).

### Task 3 — Guard `distance_to_reward` against non-finite input and unsorted `times`

In `distance_to_reward` (`regressors.py`), replace the bare casts at lines **617–619** with
validated casts, and assert `times` is ascending before the `np.interp` at line 669.

Replace lines **616–619**:

```python
    # Input validation
    positions = np.asarray(positions, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    reward_times = np.asarray(reward_times, dtype=np.float64)
```

with:

```python
    from neurospatial._validation import validate_finite

    # Input validation: reject NaN/Inf in the spatial + temporal inputs.
    # A non-finite time or position silently corrupts interpolation,
    # searchsorted ordering, and the resulting distance-to-reward regressor.
    times = validate_finite(times, name="times")
    reward_times = validate_finite(reward_times, name="reward_times")
    positions = validate_finite(np.asarray(positions, dtype=np.float64), name="positions")
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)
```

Notes:
- `validate_finite` returns a `float64` array, so the explicit `dtype=np.float64` casts are
  subsumed. `positions` is wrapped in `np.asarray` first only to give `validate_finite` an
  ndarray to reshape-test; keeping the `ndim == 1 → reshape(-1, 1)` here is defensive — the
  function already assumes 2-D `positions` at line 669 (`positions[:, dim]`) and line 675
  (`env.bin_at(positions)`). Verify no later code re-flattens it.
- `reward_positions`, when supplied (the `else` branch is skipped), is *not* forced finite by
  this task — a NaN reward position maps to an invalid bin and the existing
  `target_bins < 0 → NaN` path (lines 776–777) already handles it. Leave that path; this task
  guards only the **interpolated** path's preconditions.

Then, **inside** the `else:` interpolation branch (the one starting at line 645), guard the
`np.interp` sortedness precondition. Insert immediately before `t_min = times.min()`
(line 653):

```python
        # np.interp requires its sample points (times) to be ascending;
        # unsorted times produce silently-wrong interpolated reward positions.
        if np.any(np.diff(times) < 0):
            raise ValueError(
                "times must be sorted in ascending order to interpolate "
                "reward positions.\n"
                "  WHY: linear interpolation of reward locations assumes "
                "monotonically increasing timestamps; unsorted times yield "
                "wrong reward positions and a corrupted distance regressor.\n"
                "  HOW: sort (times, positions) by time before calling, or "
                "pass explicit reward_positions to skip interpolation."
            )
```

Notes:
- Place the sortedness check **inside** the `else` (interpolation) branch: when the caller
  passes explicit `reward_positions`, `times` ordering is irrelevant to the interpolation, and
  the downstream `np.searchsorted(sorted_reward_times, times, ...)` (lines 689, 716, 723) sorts
  *rewards*, not `times`, so sample ordering does not need to be monotone for that path. Only
  the `np.interp` at 669 has the ascending-`times` precondition. Keeping the guard scoped to
  the interpolation branch avoids rejecting valid explicit-position calls.
- `np.diff(times) < 0` (strict) permits equal consecutive timestamps, which `np.interp`
  tolerates; only a true descending step is rejected.

Update `distance_to_reward`'s `Raises` docstring (regressors.py:546–551) to add: `ValueError`
if `times`, `positions`, or `reward_times` contain non-finite values, or if `times` is not
ascending when reward positions are inferred (no `reward_positions` given).

### Task 4 — Reject degenerate trajectories in `add_positions`

In `add_positions` (`detection.py`), the cast/length checks are at lines **110–120** and the
`interp1d` block at **141–162**. A trajectory with a **single** sample, or with **all-identical
`times`**, cannot define a linear interpolant: `interp1d(..., fill_value="extrapolate")`
returns NaN (single sample) or NaN/Inf (duplicate `xp`) for every event — silently producing
an all-NaN/Inf position column.

Add a degeneracy guard **after** the length check (after line 120, before the
`# Ensure positions is 2D` comment at line 122). Also validate `times` is finite here, since a
NaN/Inf trajectory timestamp poisons `np.argsort` + `interp1d`:

```python
    from neurospatial._validation import validate_finite

    # Trajectory timestamps must be finite to define the interpolant; a NaN/Inf
    # time silently maps every event to NaN/Inf. (Event timestamps may still be
    # NaN -> NaN position, handled separately below.)
    times = validate_finite(times, name="times")

    # A linear interpolant needs at least two samples spanning a non-zero time
    # range. A single sample, or all-identical times, leaves interp1d undefined
    # and would return all-NaN/Inf positions silently.
    if len(times) < 2:
        raise ValueError(
            f"add_positions needs at least 2 trajectory samples to "
            f"interpolate, got {len(times)}.\n"
            "  WHY: linear interpolation is undefined for a single sample and "
            "would return NaN for every event position.\n"
            "  HOW: pass a trajectory with >= 2 samples spanning the event times."
        )
    if np.ptp(times) == 0:
        raise ValueError(
            "add_positions trajectory times are all identical "
            f"(every sample at t={times[0]:g}).\n"
            "  WHY: interpolation needs a non-zero time span; duplicate sample "
            "times leave the interpolant undefined (NaN/Inf positions).\n"
            "  HOW: pass a trajectory whose timestamps vary."
        )
```

Notes for the executor:
- Place this **after** the existing `len(times) != len(positions)` check (line 115) so a length
  mismatch is still reported first.
- `np.ptp` (peak-to-peak) is `times.max() - times.min()`; `== 0` catches all-identical times
  regardless of count. The `len(times) < 2` check is kept separately because a single sample is
  a distinct, common user error worth its own message (and `np.ptp` of one element is `0`, so it
  would otherwise be caught by the second branch with a less precise message).
- This guard is about the **trajectory** (`times`/`positions`), not the **events**. The
  existing NaN-event-timestamp → NaN-position path (detection.py:160–162) is **unchanged** and
  is still exercised by its doctest/tests. Do **not** add a finite check on `event_times`.
- `times` is reused below at `np.argsort(times)` (line 142); the `validate_finite` return
  (float64) is fine there.

Update the `add_positions` `Raises` docstring (detection.py:56–62) to add: `ValueError` if the
trajectory has fewer than 2 samples, if all trajectory `times` are identical, or if trajectory
`times` contain non-finite values. The existing "Events with NaN timestamps will have NaN
positions" note (line 73) stays — that is the *event* path, still supported.

### Task 5 — CHANGELOG entry

Add one entry under `## Unreleased` → `### Bug fixes` in
[CHANGELOG.md](../../../../CHANGELOG.md) summarizing:

- `events_to_intervals(match_by=...)` now raises `ValueError` on duplicate match keys instead
  of silently forming a Cartesian product (more intervals than real pairs).
- `distance_to_reward` now raises on non-finite `times`/`positions`/`reward_times` and on
  unsorted `times` when inferring reward positions, instead of returning silently-wrong
  distances.
- `add_positions` now raises on degenerate trajectories (single sample, all-identical times, or
  non-finite trajectory times) instead of returning all-NaN/Inf positions.

## Deliberately not in this phase

- **`window` scalar-vs-tuple inconsistency across the GLM regressors** — `event_count_in_window`
  and `event_indicator` take `window: tuple[float, float]` while `time_to_nearest_event` uses a
  scalar `max_time` for the *same* peri-event concept (regressors.py:214/350 vs 28). Harmonizing
  this signature is an **API-naming** change → **phase 22**, not this correctness PR. Do not
  touch the `window`/`max_time` parameter shapes here.
- **The `event_indicator` tuple-window doc example** (and any docstring/QUICKSTART example that
  shows the windowed regressors) — these are runnable-example fixes that land with the docs CI
  in **phase 23**. This phase only touches the `Raises` sections of the three functions it
  actually modifies (Tasks 2–4) and the CHANGELOG.
- **`reward_positions` finiteness when explicitly supplied** — an explicit NaN reward position
  already routes through the existing `target_bins < 0 → NaN` path (regressors.py:776–777). This
  phase guards only the *interpolated* reward-position path. Do not add a `validate_finite` call
  on the user-supplied `reward_positions` branch.
- **Out-of-range reward-time clipping warning** (regressors.py:655–665) — that behavior is
  intentional and already warns; leave it. This phase adds the *sortedness* and *finiteness*
  guards around it, not a change to the clipping policy.
- **Other `events/` modules** (`alignment.py`, `core.py`, `nwb.py`) and the other regressors
  (`time_to_nearest_event`, `event_count_in_window`, `event_indicator`, `distance_to_boundary`)
  — no finding here. Do not "while I'm in here" them.

## Validation slice

All tests live under `tests/events/` (add to `test_intervals.py`, `test_regressors.py`,
`test_detection.py`; do **not** create plan-named files). Each must **fail on current code and
pass after the fix**.

| Test | Asserts |
| --- | --- |
| `test_events_to_intervals_duplicate_match_key_raises` | `events_to_intervals` with `start`/`stop` each having `trial_id=[1, 1, 2]` and `match_by="trial_id"` raises `ValueError` mentioning duplicates. Today it returns **5** rows (Cartesian product) instead of 3. |
| `test_events_to_intervals_duplicate_start_only_raises` | Duplicate key on the **start** side only (`start trial_id=[1,1,2]`, `stop trial_id=[1,2]`) raises `ValueError` (the unmatched check would not catch this — sets are equal). |
| `test_events_to_intervals_unique_match_key_unchanged` | The existing matched example (`starts trial_id=[1,2,3]`, out-of-order `stops`) still returns the 3 documented intervals — regression guard that the unique path is untouched. |
| `test_distance_to_reward_nan_reward_time_raises` | `reward_times` with one `np.nan` raises `ValueError` naming `reward_times`. Today it interpolates/searchsorts through the NaN and returns silently-wrong distances. |
| `test_distance_to_reward_nan_position_raises` | `positions` with one `np.nan` raises `ValueError` naming `positions`. |
| `test_distance_to_reward_unsorted_times_raises` | Unsorted `times` (descending step) with **no** `reward_positions` raises `ValueError` mentioning ascending/sorted. Today `np.interp` returns wrong reward positions silently. |
| `test_distance_to_reward_unsorted_times_ok_with_explicit_positions` | Unsorted `times` **with** explicit `reward_positions` does **not** raise (the sortedness guard is scoped to the interpolation branch) and returns finite distances for valid bins. |
| `test_distance_to_reward_finite_inputs_unchanged` | A small all-finite, sorted example returns the same distances as on `main` (e.g. assert a known value / monotonic decrease toward the reward bin) — regression guard. |
| `test_add_positions_single_sample_raises` | 1-sample trajectory (`times=[1.0]`, `positions=[[0,0]]`) raises `ValueError` mentioning at least 2 samples. Today returns an all-NaN `x`/`y` column. |
| `test_add_positions_identical_times_raises` | `times=[2.0, 2.0, 2.0]` raises `ValueError` mentioning identical/non-zero span. Today returns NaN/Inf positions. |
| `test_add_positions_nan_trajectory_time_raises` | A `np.nan` in **trajectory** `times` raises `ValueError` naming `times`. |
| `test_add_positions_nan_event_timestamp_still_nan` | An **event** with NaN timestamp still yields NaN `x`/`y` (existing behavior) while finite events get correct positions — regression guard that the event-NaN path is preserved (use a valid >=2-sample, varying-time trajectory). |
| `test_add_positions_valid_unchanged` | The module doctest example (`rewards` at `[1.5, 3.5]`) still returns `[[3,3],[7,7]]` — regression guard. |

Mark none `slow`; all are sub-millisecond.

## Fixtures

No checked-in data or real-data slice needed. Synthesize inline:

- **intervals tests:** small `pd.DataFrame`s with explicit `timestamp` + `trial_id` columns
  (no RNG).
- **distance_to_reward tests:** a tiny 2-D `Environment.from_samples(positions, bin_size=...)`
  with a short, sorted `(times, positions)` trajectory and 1–2 `reward_times`. A seeded
  `np.random.default_rng(0)` may seed the trajectory, but the failing cases inject the
  `np.nan`/descending step explicitly. Reuse any existing `tests/events/` env fixture if one is
  already defined in `tests/events/conftest.py` / `test_regressors.py`; only lift a new fixture
  into `conftest.py` if the same env is built in ≥3 tests.
- **add_positions tests:** plain `np.array` trajectories and a 1–2 row events `pd.DataFrame`; the
  degenerate cases are constructed directly (single sample, repeated times, NaN time).

## Review

Before opening the PR, dispatch an independent reviewer (`scientific-code-change-audit` lens —
these guard scientific quantities: interval counts, interpolated positions, distance
regressors). Confirm:

- All three findings are fixed: duplicate-key Cartesian product (Task 2), `distance_to_reward`
  non-finite + unsorted-`times` guards (Task 3), `add_positions` degenerate-trajectory guard
  (Task 4).
- The "Deliberately not in this phase" list is honored — no `window`/`max_time` signature
  changes (phase 22), no doc-example edits beyond `Raises` sections (phase 23), no guard added
  to the explicit-`reward_positions` branch, no edits to other `events/` modules/regressors.
- `validate_finite` is **imported** from `src/neurospatial/_validation.py`, not re-implemented,
  and is used (not bypassed by ad-hoc `np.isnan` in the new code).
- Every validation-slice test **fails on `main`** and passes on the branch (run each against the
  pre-fix files to confirm it is not a tautology). The four regression guards
  (`*_unchanged`, `*_ok_with_explicit_positions`, `*_event_timestamp_still_nan`, `*_valid_unchanged`)
  pin existing correct behavior so the guards do not over-reject.
- Tests aren't trivial — they assert the raised `ValueError` *and* (for regression guards) the
  numeric output; shared env setup is in a fixture, not copy-pasted. (`testing-anti-patterns`.)
- `uv run pytest tests/events/ -q`, `uv run ruff check src/neurospatial/events/ && uv run ruff format --check src/neurospatial/events/`, and `uv run mypy src/neurospatial/events/` all pass.
- Docstrings, test names, and any new code reference no plan/milestone
  (`grep -ri "phase 7\|milestone\|review-remediation" src/neurospatial/events/ tests/events/`
  is empty).
- The CHANGELOG entry is present and names the new raised-error behaviors (per pre-1.0 policy:
  silently-wrong → raised is intended).
