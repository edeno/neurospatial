# Phase 8 — Ops: numerical robustness

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#input-validation-helpers)

Five low-level numerical-robustness defects in the primitives layer, all scoped to `src/neurospatial/ops/`. Each one silently corrupts a result — a single out-of-source NaN zeroing an entire resampled field, an unguarded `dt` producing NaN or 180°-wrong headings, KDTree failures returning an all-zero probability map, an eigensolver fallback masked behind a broad `except`, and a Kabsch reflection bug that returns an improper "rotation" for reflected/coplanar landmark sets. None of these crashes; each ships with a fail-before / pass-after regression test.

**Soft dependency on phase 4.** This phase imports `validate_finite` / `validate_lengths` from `src/neurospatial/_validation.py`, the new top-level module created in **phase 4** (see [shared contracts](shared-contracts.md#input-validation-helpers)). Land phase 4 first, or — if this phase lands first — create that module here per the contract verbatim and let phase 4 adopt it. Do **not** fork a private copy inside `ops/`.

---

**Inputs to read first:**

- [../../../../src/neurospatial/ops/binning.py:780](../../../../src/neurospatial/ops/binning.py) — `resample_field` pullback + diffuse step (lines 780–806). Line 788 sets `resampled[outside_source] = np.nan`; the diffuse branch (lines 791–804) then calls `apply_kernel(resampled, kernel, mode="forward")` at line 804. A single NaN propagates through the kernel matmul to **every** reachable destination bin, so one out-of-source bin can zero/NaN the whole field. This is the bug to fix.
- [../../../../src/neurospatial/ops/binning.py:735](../../../../src/neurospatial/ops/binning.py) — `resample_field` docstring "Diffuse method" Notes (lines 692–694, doctest at 723–733): confirms diffuse smoothing is expected to interpolate, not annihilate; the doctest plants a single non-zero bin and only asserts shape, so it does not currently catch the NaN-propagation regression.
- [../../../../src/neurospatial/ops/egocentric.py:635](../../../../src/neurospatial/ops/egocentric.py) — `heading_from_velocity` (lines 635–754). `velocity = np.diff(positions, axis=0) / dt` at line 722 divides by `dt` with no validation; the only guard is `len(positions) < 2` (line 710). `dt == 0` → inf/NaN velocity; `dt < 0` (e.g. caller passed `times[0] - times[1]`) silently negates velocity → headings rotated by π. The docstring (lines 668–671) only documents the `< 2 samples` `ValueError`.
- [../../../../src/neurospatial/ops/egocentric.py:493](../../../../src/neurospatial/ops/egocentric.py) — `compute_egocentric_distance` (lines 493–632). The 3D-targets branch at lines 577–581: when `targets.ndim == 3` the code takes `targets_3d = targets` directly and reads `n_targets = targets_3d.shape[1]` (line 583) **without checking** `targets.shape[0] == n_time` (`n_time = len(positions)`, line 575). A mismatched time axis broadcasts/indexes wrong or raises a cryptic shape error in the `diff` at line 591.
- [../../../../src/neurospatial/ops/egocentric.py:419](../../../../src/neurospatial/ops/egocentric.py) — `compute_egocentric_bearing` (lines 419–474). `arctan2` at line 469 already returns `(-π, π]`; line 472 then calls `_wrap_angle(bearing)`. Docstring (line 442) promises range `(-pi, pi]`.
- [../../../../src/neurospatial/ops/egocentric.py:477](../../../../src/neurospatial/ops/egocentric.py) — `_wrap_angle` (lines 477–490). Body `(angle + np.pi) % (2 * np.pi) - np.pi` maps an exact `+π` (the antipode / directly-behind target) to **`-π`**, contradicting the documented half-open `(-π, π]`. This is the only caller (`grep` confirms lines 472 and 477 are the sole references).
- [../../../../src/neurospatial/ops/alignment.py:374](../../../../src/neurospatial/ops/alignment.py) — `_map_nearest_neighbor` (lines 347–385): `try: tree.query(...) except Exception: warn + return zeros` at lines 374–382.
- [../../../../src/neurospatial/ops/alignment.py:424](../../../../src/neurospatial/ops/alignment.py) — `_map_inverse_distance_weighted` (lines 388–452): identical broad `except Exception` at lines 424–432.
- [../../../../src/neurospatial/ops/alignment.py:578](../../../../src/neurospatial/ops/alignment.py) — `map_probabilities` KDTree construction (lines 577–586): `try: cKDTree(...) except Exception: warn + return zeros`. A genuine error (NaN bin centers, wrong-dim target env) is swallowed into an all-zero probability map that downstream code treats as a valid (lossy) mapping.
- [../../../../src/neurospatial/ops/alignment.py:456](../../../../src/neurospatial/ops/alignment.py) — `map_probabilities` public signature (lines 456–467) and `Returns`/`Raises` docstring (lines 500–513): documents `ValueError` on shape/dim mismatch — so a malformed env *should* raise, not return zeros.
- [../../../../src/neurospatial/ops/basis.py:961](../../../../src/neurospatial/ops/basis.py) — `_estimate_spectral_radius` (lines 961–1002): `try: eigsh(...) except Exception: warn + max-degree fallback` at lines 988–1002. The bound `2 * max(degree)` is a valid upper bound, but the broad `except` hides real failures (singular Laplacian, malformed sparse matrix) and silently inflates the spectral radius, over-smoothing every basis built on top.
- [../../../../src/neurospatial/ops/transforms.py:1146](../../../../src/neurospatial/ops/transforms.py) — `estimate_transform` signature (line 1146) and the Procrustes rigid/similarity branch (lines 1253–1309).
- [../../../../src/neurospatial/ops/transforms.py:1269](../../../../src/neurospatial/ops/transforms.py) — the reflection-correction (lines 1269–1276). After `R = R_proc.T` (line 1270), the guard at line 1274 does `if det(R) < 0: R[:, -1] = -R[:, -1]`. Flipping a column of the **already-formed** orthogonal matrix yields a different orthogonal matrix that is *no longer the optimal rotation* — the correct Kabsch fix folds the sign into the SVD factors *before* reassembling R. For coplanar/reflected landmark sets this returns a worse fit (and, in 3D, can still be improper). Doctests at lines 1199–1216 only exercise proper rotations, so they pass today.

**Contracts referenced:**

- [Input-validation helpers](shared-contracts.md#input-validation-helpers) — `validate_finite(a, *, name, allow_nan=False)` and `validate_lengths(name_to_array)` from `src/neurospatial/_validation.py`. Task 2 uses `validate_finite(positions, name="positions")` and a `dt <= 0` guard at the `heading_from_velocity` entry. **Do not weaken**: `validate_finite` raises on Inf always and never silently coerces; the guards must run so genuine bad input surfaces as a `ValueError`, not a NaN array.

**Designs referenced:** none (each fix is local; the algorithm corrections are inlined below).

---

## Tasks

### Task 1 — `resample_field(method="diffuse")`: don't let one out-of-source NaN annihilate the field

The pullback (binning.py:780–788) correctly marks destination bins that fall **outside** the source as NaN. But the diffuse branch then runs `apply_kernel(resampled, kernel, mode="forward")` (line 804) directly on the NaN-containing array. A forward diffusion kernel is a dense-ish row-stochastic matmul, so any single NaN poisons every destination bin reachable from it — the documented "interpolate and smooth" behavior collapses to NaN/zero across the whole output.

Fix: smooth with the outside-source bins **zero-filled** (a missing source contributes no mass — the correct neutral element for a forward diffusion), then **re-impose NaN** on the bins that are genuinely outside the source after smoothing, so callers still see the un-covered region as missing rather than as spuriously-smoothed zero.

Replace the diffuse branch (binning.py:791–804) with:

```python
    # Step 3: Optionally apply smoothing for diffuse method
    if method == "diffuse":
        # Import here to avoid circular dependency
        from neurospatial.ops.smoothing import apply_kernel

        # Type narrowing: bandwidth is guaranteed to be float at this point
        assert bandwidth is not None  # Already validated above

        # Compute diffusion kernel on destination environment
        kernel = cast("EnvironmentProtocol", dst_env).compute_kernel(
            bandwidth=bandwidth, mode="transition"
        )

        # The pullback marked out-of-source destination bins as NaN (Step 2).
        # apply_kernel is a forward diffusion (a row-stochastic matmul), so a
        # single NaN would propagate to every reachable bin and wipe out the
        # whole field. An un-covered source bin should contribute *no mass* to
        # the smoothing, which is exactly a zero — so zero-fill the NaNs before
        # smoothing, then re-impose NaN on the genuinely-outside bins afterward
        # so the un-covered region stays marked as missing (not a smoothed zero).
        to_smooth = np.where(outside_source, 0.0, resampled)
        smoothed = apply_kernel(to_smooth, kernel, mode="forward")
        smoothed[outside_source] = np.nan
        resampled = smoothed

    return resampled
```

`outside_source` is already in scope from Step 2 (binning.py:785). No other change to the nearest branch.

> Note: this preserves the existing "outside-source bins are NaN" contract for *both* methods. A destination bin that is **inside** the source but happens to be NaN-valued for another reason is out of scope here — the only NaNs `resample_field` itself introduces are the outside-source ones.

### Task 2 — `heading_from_velocity`: validate `positions` finite and `dt > 0`

`heading_from_velocity` divides by `dt` (egocentric.py:722) with no guard. `dt == 0` gives inf/NaN; `dt < 0` (a caller who computed `times[0] - times[1]` from descending timestamps) silently flips the velocity vector, rotating every heading by π — a 180°-wrong head direction that no downstream code can detect. Non-finite `positions` likewise propagate to NaN headings.

Add guards at the top of the function body, immediately after the `np.asarray` at egocentric.py:708 and **before** the `len(positions) < 2` check (so a bad `dt` is reported even for a 2-point trajectory):

```python
    from neurospatial._validation import validate_finite

    positions = np.asarray(positions, dtype=np.float64)

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            f"Cannot compute heading: dt must be a positive, finite time step "
            f"(got {dt!r}).\n\n"
            f"WHAT: dt is the seconds between consecutive position samples\n"
            f"WHY: velocity = diff(positions) / dt; dt <= 0 negates or NaNs the "
            f"velocity, rotating every heading by pi (180 deg)\n\n"
            f"HOW to fix:\n"
            f"1. Pass dt = times[1] - times[0] from ASCENDING timestamps\n"
            f"2. Sort your timestamps before differencing"
        )

    validate_finite(positions, name="positions")
```

(The existing `if len(positions) < 2:` block at egocentric.py:710–719 stays directly after this.)

Update the `Raises` section of the docstring (egocentric.py:668–671) to add:

```
    ValueError
        If positions has fewer than 2 samples, contains non-finite values,
        or if dt is not a positive finite number.
```

**Task 2b — `compute_egocentric_distance`: validate 3D targets' time axis.** In the `targets.ndim != 2` branch (egocentric.py:577–581), the code currently trusts that a 3D `targets` array has `shape[0] == n_time`. Add the check (right after `n_time = len(positions)` at line 575 is computed and the `targets.ndim == 2` branch):

```python
    n_time = len(positions)

    # Expand targets to 3D if needed
    if targets.ndim == 2:
        targets_3d = np.broadcast_to(targets, (n_time, targets.shape[0], 2))
    elif targets.ndim == 3:
        if targets.shape[0] != n_time:
            raise ValueError(
                f"targets time axis {targets.shape[0]} does not match positions "
                f"length {n_time}.\n\n"
                f"WHAT: a 3D targets array must have shape (n_time, n_targets, 2)\n"
                f"WHY: each timepoint's distance is computed against that "
                f"timepoint's targets\n\n"
                f"HOW to fix:\n"
                f"1. Pass static targets as a 2D (n_targets, 2) array, or\n"
                f"2. Make targets.shape[0] equal len(positions)"
            )
        targets_3d = targets
    else:
        raise ValueError(
            f"targets must be 2D (n_targets, 2) or 3D (n_time, n_targets, 2), "
            f"got shape {targets.shape}"
        )
```

This replaces the original two-branch `if targets.ndim == 2: ... else: targets_3d = targets` (egocentric.py:578–581).

**Task 2c — `_wrap_angle`: keep the antipode at `+π`, per the documented `(-π, π]`.** The current `(angle + π) % (2π) - π` sends an exact `+π` to `-π`. Make the wrap half-open on the documented side:

```python
def _wrap_angle(angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap angles to (-pi, pi].

    Parameters
    ----------
    angle : NDArray
        Angles in radians.

    Returns
    -------
    NDArray
        Angles wrapped to the half-open interval (-pi, pi] (a target directly
        behind the animal returns +pi, never -pi).
    """
    # ((angle - pi) % (2*pi)) - (-pi)  shifts so the open end is at -pi and the
    # closed end at +pi, giving the documented (-pi, pi] half-open interval.
    wrapped = (angle - np.pi) % (-2 * np.pi) + np.pi
    return wrapped
```

`(angle - np.pi) % (-2 * np.pi)` is in `(-2π, 0]` (NumPy's modulo takes the sign of the divisor), so adding `π` yields `(-π, π]` with `+π` retained. Confirm with a doctest-style check that `_wrap_angle(np.array([np.pi, -np.pi, 3*np.pi]))` all equal `+π` (within float tol) and that interior values are unchanged. No change needed at the `compute_egocentric_bearing` call site (egocentric.py:472) — its docstring already promises `(-pi, pi]`.

### Task 3 — `map_probabilities` (alignment.py): narrow the swallowed exceptions

Three `except Exception` sites turn any failure into an all-zero probability map that silently looks like a valid (total-mass-losing) result. The only failure that *should* degrade to a warning-plus-zeros is a `cKDTree` query that legitimately returns no neighbors — and `cKDTree.query` does not raise for that; it returns `inf` distance / out-of-range index, handled downstream. There is no benign exception to catch here, so **remove the broad `try/except`** and let real errors (NaN coordinates, dimension mismatch, malformed env) propagate as the documented `ValueError` / library error.

**3a. `_map_nearest_neighbor`** (alignment.py:372–385) — replace lines 374–384 with:

```python
    target_probs = np.zeros(n_tgt, dtype=float)

    # cKDTree.query does not raise on "no neighbor" (it returns inf distance /
    # out-of-range index). A raised exception here means genuinely bad input
    # (e.g. non-finite coordinates) and must propagate rather than degrade to a
    # silent all-zero map.
    _, idxs = tree.query(src_centers, k=1)

    np.add.at(target_probs, idxs, source_probs)
    return target_probs
```

**3b. `_map_inverse_distance_weighted`** (alignment.py:419–432) — drop the `try/except` around `tree.query` (lines 424–432); call it directly:

```python
    target_probs = np.zeros(n_tgt, dtype=float)

    # Clamp n_neighbors if larger than number of target bins
    k_eff = min(n_neighbors, n_tgt)

    # See _map_nearest_neighbor: a raised exception is bad input, not a benign
    # "no neighbor" — let it propagate.
    dists, idxs = tree.query(src_centers, k=k_eff)
```

(The rest of the function — the `k_eff == 1` reshape at lines 434–437 onward — is unchanged.)

**3c. `map_probabilities` KDTree construction** (alignment.py:577–586) — drop the `try/except` around `cKDTree(...)`:

```python
    # Build KDTree on target bin centers. A construction error here means the
    # target env has malformed (e.g. non-finite or wrong-dimension) bin centers;
    # surface it instead of returning a misleading all-zero map.
    tree = cKDTree(target_env.bin_centers, leafsize=KDTREE_LEAF_SIZE)
```

The legitimate empty-environment early-return (alignment.py:560–567, `if n_src == 0 or n_tgt == 0`) stays — that is a real, documented zero-output case and is **not** an error path.

If `import warnings` becomes unused after these removals, drop it (run `ruff check`); leave it if other code in the module still uses it (the empty-env `UserWarning` at line 562 does — so it stays).

### Task 4 — `_estimate_spectral_radius` (basis.py): narrow the eigsh fallback

`_estimate_spectral_radius` falls back to the `2 * max(degree)` bound on **any** exception (basis.py:988–1002). The max-degree bound is a legitimate fallback only for the documented `eigsh` convergence/`ArpackNoConvergence` failure — not for a malformed Laplacian or a programming error, which the broad `except` hides while silently over-estimating the spectral radius (and thus over-smoothing). Narrow the catch to the ARPACK convergence failure.

Replace lines 988–1002 with:

```python
    from scipy.sparse.linalg import ArpackError, ArpackNoConvergence

    # Use eigsh with which='LM' (largest magnitude)
    try:
        eigenvalues = eigsh(laplacian, k=1, which="LM", return_eigenvectors=False)
        return float(eigenvalues[0])
    except (ArpackNoConvergence, ArpackError) as e:
        # ARPACK failed to converge on the largest eigenvalue. Fall back to the
        # exact upper bound lambda_max <= 2 * max_degree (diagonal of the graph
        # Laplacian holds node degrees). Any *other* exception (malformed matrix,
        # wrong dtype) is a real bug and propagates.
        max_degree_bound = 2.0 * float(np.max(laplacian.diagonal()))
        warnings.warn(
            f"eigsh did not converge ({e}); using max-degree bound "
            f"{max_degree_bound:.2f}. This is an exact upper bound but may be "
            f"loose for highly irregular graphs.",
            stacklevel=3,
            category=UserWarning,
        )
        return max_degree_bound
```

`ArpackError`/`ArpackNoConvergence` live in `scipy.sparse.linalg`. Confirm the import path during implementation (`from scipy.sparse.linalg import ArpackError, ArpackNoConvergence`); if a SciPy version pins them under `scipy.sparse.linalg.eigen.arpack`, import from the public `scipy.sparse.linalg` re-export. The module already imports `warnings` locally at basis.py:978 — keep that.

### Task 5 — `estimate_transform` (transforms.py): correct Kabsch reflection handling

The reflection guard (transforms.py:1274–1276) flips the last **column of the already-formed orthogonal matrix** `R`. That produces a different orthogonal matrix, but not the *optimal proper rotation* — the standard Kabsch correction folds the sign `d = sign(det(V·Uᵀ))` into the SVD factors **before** reassembling `R`, which keeps the result the closest proper rotation to the data. With coplanar or reflected landmark sets the current code can return a fit that is both non-optimal and (in 3D) still improper.

`orthogonal_procrustes` already returns the optimal *orthogonal* `R_proc` (possibly a reflection). To get the optimal proper rotation, redo the final assembly via SVD with the determinant sign-correction. Replace lines 1265–1276 with:

```python
        # Estimate rotation via SVD with the Kabsch determinant correction.
        # orthogonal_procrustes minimizes ||src_centered @ R - dst_centered||,
        # but its R may be a reflection (det = -1) for coplanar/reflected point
        # sets. The standard fix folds d = sign(det(V @ U.T)) into the SVD
        # factors BEFORE reassembling R, yielding the closest PROPER rotation
        # (det = +1). Flipping a column of the finished matrix instead would
        # give a non-optimal (and possibly still improper) result.
        #
        # H = src_centered.T @ dst_centered ;  H = U @ S @ Vt
        # R_fit = V @ diag(1, ..., 1, d) @ U.T   with d = sign(det(V @ U.T))
        H = src_centered.T @ dst_centered
        U, _S, Vt = np.linalg.svd(H)
        V = Vt.T
        d = np.sign(np.linalg.det(V @ U.T))
        D = np.eye(n_dims)
        D[-1, -1] = d
        R = V @ D @ U.T
```

This `R` is the rotation in the convention `T(x) = R @ x + t` (matching the existing `t = dst_mean - R @ src_mean` at line 1282 and `scale * R` at lines 1304/1307), so the downstream rigid/similarity assembly (transforms.py:1278–1309) is unchanged. The `orthogonal_procrustes` import at transforms.py:1235 becomes unused — **remove it** (no orphan import; the affine branch below uses `lstsq`, not Procrustes — confirm during implementation and drop the import if so).

Add a short note to the `estimate_transform` `Notes` section (transforms.py:1218–1225): "Rigid and similarity fits use the Kabsch/SVD solution with a determinant sign-correction, so the returned rotation is always proper (det = +1) even when the landmark points are coplanar or reflected."

---

## Deliberately not in this phase

- **IDW-mode `map_probabilities` test backfill** → **phase 25**. Task 3 narrows the exception handling in the `inverse-distance-weighted` path but does **not** add the broader IDW correctness/round-trip test coverage that phase 25 owns. Only add the narrow "bad input now raises instead of returning zeros" regression here.
- **`compute_viewshed` hardcoded 200-unit ray cap** (`ops/visibility.py:830`, `max_distance = 200.0`) is a **separate units concern**, not a numerical-robustness bug, and is **out of scope** here. `compute_viewshed` hardcodes the ray length to 200 (unitless) regardless of the environment's `units`, while the sibling gaze helpers (`visibility.py:494`, `:589`) take a `max_distance` parameter. Surfacing this as a parameter / units-aware default belongs to the units/visibility phase — note it, do not touch `visibility.py` in this PR.
- **Generalizing `validate_finite` / `validate_lengths`** — these are defined in phase 4 per [shared contracts](shared-contracts.md#input-validation-helpers); this phase only *consumes* them. Do not edit `_validation.py` here beyond creating it verbatim if phase 4 has not landed.
- **Egocentric heading convention / `heading_from_body_orientation`** — only `heading_from_velocity` gets the `dt`/finiteness guard; the body-orientation path and the allocentric↔egocentric transforms are unchanged.
- **No edits outside `src/neurospatial/ops/`.** No encoding, decoding, environment, or stats changes.

---

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_resample_diffuse_single_out_of_source_nan_preserves_field` | Build `src_env` / `dst_env` where a few destination bins fall outside the source (so the pullback NaNs them). `resample_field(field, src_env, dst_env, method="diffuse", bandwidth=...)` returns a result whose **inside-source** bins are finite and close to the nearest-method values (within smoothing tolerance); only the genuinely-outside bins are NaN. **Fail-before**: the whole field is NaN/zero because one NaN propagated through `apply_kernel`. |
| `test_resample_diffuse_outside_bins_remain_nan` | The bins flagged `outside_source` are NaN in the diffuse output (the re-impose step works), i.e. zero-filling for smoothing does not leak a spurious smoothed-zero into the un-covered region. |
| `test_resample_diffuse_matches_nearest_when_fully_covered` | When every destination bin is inside the source (no NaNs), diffuse output has no NaNs and total mass is finite — smoke that the zero-fill path is a no-op when there is nothing to fill. |
| `test_heading_from_velocity_rejects_nonpositive_dt` | `heading_from_velocity(positions, dt=0.0)` and `dt=-0.1` each **raise `ValueError`**. **Fail-before**: `dt=0` returns an all-NaN/inf array; `dt<0` returns headings rotated by π. |
| `test_heading_from_velocity_rejects_nonfinite_positions` | A `positions` array containing `np.nan` or `np.inf` **raises `ValueError`** (via `validate_finite`). |
| `test_heading_from_velocity_negative_dt_would_flip` *(documents the motivation)* | For an eastward trajectory, the heading computed with a correct positive `dt` is ≈0; confirm the guard now blocks the `dt<0` call that would otherwise return ≈π. |
| `test_egocentric_distance_3d_targets_time_mismatch_raises` | `compute_egocentric_distance(positions, None, targets)` with `targets.shape == (n_time+1, k, 2)` **raises `ValueError`** naming the time-axis mismatch. **Fail-before**: cryptic broadcast error or wrong distances. |
| `test_egocentric_distance_3d_targets_valid_roundtrip` | A correctly-shaped `(n_time, k, 2)` targets array returns `(n_time, k)` distances equal to the per-timepoint Euclidean distance (smoke that the new branch did not break the valid path). |
| `test_wrap_angle_antipode_is_plus_pi` | `_wrap_angle(np.array([np.pi, -np.pi, 3*np.pi, -3*np.pi]))` all equal `+π` within float tol; interior values (e.g. `0.3`, `-1.2`) are returned unchanged. **Fail-before**: `+π` wraps to `-π`. |
| `test_egocentric_bearing_directly_behind_is_plus_pi` | A target directly behind the animal (heading East, target to the West) returns bearing `+π`, not `-π`, matching the documented `(-pi, pi]`. |
| `test_map_probabilities_nan_centers_raises` | `map_probabilities` on a source/target env whose bin centers contain a NaN (or mismatched dims that make `cKDTree` raise) **raises** instead of returning an all-zero array. **Fail-before**: returns `np.zeros(n_tgt)` with only a `RuntimeWarning`. |
| `test_map_probabilities_empty_env_still_returns_zeros` | The legitimate `n_tgt == 0` path still returns an empty/zeros array with a `UserWarning` (the narrowing did not remove the real benign case). |
| `test_estimate_spectral_radius_propagates_non_arpack_error` | Feeding `_estimate_spectral_radius` a malformed input that makes `eigsh` raise a **non-ARPACK** error propagates that error (no silent max-degree fallback). A genuine `ArpackNoConvergence` still falls back with a warning. (If a non-ARPACK trigger is hard to construct, assert via a monkeypatched `eigsh` raising `ValueError` that it propagates, and `ArpackNoConvergence` that it warns-and-bounds.) |
| `test_estimate_transform_reflected_points_proper_rotation` | Build `src` and a **reflected** copy `dst` (apply a reflection, e.g. negate x, then a small rotation/translation). `estimate_transform(src, dst, kind="rigid")` returns a transform whose linear block has `det ≈ +1` (a proper rotation, not a reflection). **Fail-before**: the naive column-flip yields a non-optimal / improper matrix. |
| `test_estimate_transform_coplanar_3d_proper_rotation` | A 3D set of **coplanar** landmark points (all z=0) related by a known rotation+translation recovers a transform with `det(R) ≈ +1` and `transform(src) ≈ dst`. **Fail-before**: degenerate SVD direction handled wrong by the column-flip. |
| `test_estimate_transform_pure_rotation_unchanged` | The existing proper-rotation doctest cases (2D 45°, 3D z-rotation) still recover the planted transform exactly — regression guard that the SVD rewrite matches the old behavior on non-reflected data. |

All tests are fast; none need `pytest.mark.slow`.

## Fixtures

Synthesized and seeded — no checked-in data. Use the existing ops test modules as homes:

- `tests/ops/test_binning.py` — resample tests. Provide a small helper (or local fixture) building a coarse `src_env` and a finer `dst_env` that extends slightly **beyond** the source extent so some destination bins are out-of-source (giving the NaN pullback). Reuse the `Environment.from_samples(..., bin_size=...)` pattern already in this module.
- `tests/ops/test_reference_frames.py` — `heading_from_velocity`, `compute_egocentric_distance`, `_wrap_angle`, and `compute_egocentric_bearing` tests (this is the existing home for egocentric ops; `grep` confirms it imports these functions). Build short eastward/northward trajectories inline with a fixed `dt`.
- `tests/ops/test_ops_alignment.py` — `map_probabilities` tests. Reuse the env-construction helpers already there; build a NaN-center env (or wrong-dim target) to trigger the now-propagating error, and keep the empty-env zeros case.
- `tests/ops/test_basis.py` — `_estimate_spectral_radius` tests. Monkeypatch `scipy.sparse.linalg.eigsh` (via `monkeypatch.setattr`) to raise `ArpackNoConvergence` (asserts warn+bound) and a plain `ValueError` (asserts propagate), against a small graph Laplacian built from an existing fixture env.
- `tests/ops/test_transforms.py` — Kabsch tests. Build reflected/coplanar point sets with `np.random.default_rng(seed)`; assert on `np.linalg.det` of the recovered linear block and on `transform(src) ≈ dst` for the proper (non-reflected) cases.

Extend these existing modules rather than adding new test files.

## Review

Before opening the PR, dispatch `code-reviewer` (or `scientific-code-change-audit`, given these are scientific-quantity changes) against the diff. Confirm:

- Every task (1–5) is implemented as specified; scope stays strictly inside `src/neurospatial/ops/` — **no** edits to `visibility.py`, `_validation.py` (beyond verbatim creation if phase 4 has not landed), or any non-ops module.
- The "Deliberately not in this phase" list is honored: the `compute_viewshed` 200-unit cap is **noted, not touched**; no broad IDW test backfill (phase 25).
- Task 1 zero-fills outside-source NaNs **before** `apply_kernel` and **re-imposes** NaN on those bins after — verify the re-impose step is present so the un-covered region stays NaN, not a smoothed zero.
- Task 2's `dt <= 0` / non-finite guards run at function entry and **raise** (not warn); `validate_finite` is imported from the shared `_validation.py`, not re-implemented; `_wrap_angle`'s new formula keeps `+π` (spot-check `+π`, `-π`, and an interior value).
- Task 3 removed the broad `except Exception` at all three sites and the empty-env `UserWarning` early-return is **kept**; any now-unused `import warnings` is removed (or kept if still used).
- Task 4 catches only `ArpackError`/`ArpackNoConvergence`; an unrelated exception propagates. Confirm the SciPy import path resolves on the pinned version.
- Task 5 uses the SVD determinant sign-correction (`d = sign(det(V @ U.T))` folded into `D` *before* `R = V @ D @ U.T`), **not** a post-hoc column flip; the unused `orthogonal_procrustes` import is removed; recovered `R` has `det ≈ +1` on reflected/coplanar inputs and the proper-rotation doctests still pass.
- Validation slice passes, including the fail-before assertions (reviewer reverts at least the resample NaN fix, the `dt` guard, the `_wrap_angle` formula, the alignment narrowing, and the Kabsch fix locally to confirm each test goes red).
- Tests aren't trivial — the resample, reflected-Kabsch, and dt-guard tests exercise the asserted behavior, not tautologies; shared env setup lives in fixtures/helpers, not copy-pasted (`testing-anti-patterns`).
- `uv run pytest tests/ops/ -q`, `uv run pytest --doctest-modules src/neurospatial/ops/binning.py src/neurospatial/ops/egocentric.py src/neurospatial/ops/transforms.py`, `uv run ruff check . && uv run ruff format --check .`, and `uv run mypy src/neurospatial/ops/` all pass.
- Docstrings, test names, and module names do **not** reference this plan, "phase 8", or any milestone.
- No orphaned code: the old `resampled = apply_kernel(...)`-on-NaN line, the old `_wrap_angle` formula, the three broad `except Exception` blocks, the old column-flip reflection guard, and the now-unused `orthogonal_procrustes` import are removed, not left dead alongside the new paths.
