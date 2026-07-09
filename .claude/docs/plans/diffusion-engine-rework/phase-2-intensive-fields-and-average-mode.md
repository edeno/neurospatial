# Phase 2 — Public `mode="average"` + intensive-field callers

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md) · [contracts](shared-contracts.md)

Ship the public row-stochastic `mode="average"` (`H`) intensive-field smoother and route the
two internal intensive-field callers — `smooth_rate_map(method="binned")` and
`resample_field(method="diffuse")` — through the masked `H`-average, removing their
volume/zero-fill bias on non-uniform `M`. Depends only on Phase 1's `H` operator.

**Inputs to read first:**

- [shared-contracts.md](shared-contracts.md) — C2 (`average` row = `H`; intensive vs extensive).
- [designs.md#d5-masked-h-average-binned--resample](designs.md#d5-masked-h-average-binned--resample) — the Nadaraya-Watson masked pattern.
- Mode surface to thread `"average"` through: [fields.py:48](../../../src/neurospatial/environment/fields.py) (`compute_kernel` Literal), [fields.py:155](../../../src/neurospatial/environment/fields.py) (`smooth` Literal), [fields.py:285](../../../src/neurospatial/environment/fields.py) (`valid_modes`), [_protocols.py:50](../../../src/neurospatial/environment/_protocols.py) (cache-key type), [_protocols.py:307](../../../src/neurospatial/environment/_protocols.py) (protocol Literal).
- `binned` callers: [_smoothing.py:665-726](../../../src/neurospatial/encoding/_smoothing.py) (`_binned`), [:789-839](../../../src/neurospatial/encoding/_smoothing.py) (`_binned_batch`), [:869-902](../../../src/neurospatial/encoding/_smoothing.py) + [:965-1005](../../../src/neurospatial/encoding/_smoothing.py) (JAX binned).
- `resample`: [binning.py:790-815](../../../src/neurospatial/ops/binning.py) (diffuse branch, currently zero-fill + single `mode="transition"` smooth).
- `ops/diffusion.py` from Phase 1 — `heat_kernel_from_W` already returns `H` for `mode="average"` (D4); this phase only exposes it.

**Contracts referenced:** [C2](shared-contracts.md#c2-kernel-modes--orientation) — `average` is for intensive quantities only; discrete probability *mass* stays on `transition`.

**Designs referenced:** [D5](designs.md#d5-masked-h-average-binned--resample).

## Tasks

- **Expose `mode="average"`** end to end: add `"average"` to the `Literal[...]` types
  ([fields.py:48,155](../../../src/neurospatial/environment/fields.py),
  [_protocols.py:307](../../../src/neurospatial/environment/_protocols.py)), the `valid_modes`
  set ([fields.py:285](../../../src/neurospatial/environment/fields.py)), the `(bandwidth,
  mode)` cache-key type ([_protocols.py:50](../../../src/neurospatial/environment/_protocols.py)),
  and `compute_kernel`'s dispatch (already routes to `diffusion_kernel`, which handles
  `"average"`). Confirm the low-level `compute_diffusion_kernels(..., mode="average")` returns
  `H` (Phase 1 D4 already implements it).
- **Route `binned` through `H`** (D5): switch the two `env.smooth(...)` calls in `_binned`
  ([_smoothing.py:711,714](../../../src/neurospatial/encoding/_smoothing.py)), `_binned_batch`
  ([:829,830](../../../src/neurospatial/encoding/_smoothing.py)), and both JAX binned paths
  from the default `density` to `mode="average"`. The existing `smooth(rate·mask)/smooth(mask)`
  structure is already the masked average — only the mode changes. Preserves valid-bin
  semantics; removes the volume bias on non-uniform `M`.
- **Route `resample_field(method="diffuse")` through masked `H`** (D5): replace the
  zero-fill-then-single-smooth ([binning.py:810-813](../../../src/neurospatial/ops/binning.py))
  with `num = H@(v·valid); den = H@valid; out = where(den>0, num/den, nan)` using
  `dst_env.compute_kernel(mode="average")`, `valid = (~outside_source) & np.isfinite(resampled)`,
  and `v` **zero-filled where `~valid`** (an un-zeroed source `NaN` poisons every reachable
  bin). Re-impose `NaN` on `outside_source` (and where `den == 0`).
- **Docs (ship with this phase):** `fields.py::smooth`/`compute_kernel` docstrings —
  document the three modes' input types (transition = extensive/mass-conserving, density =
  extensive→density, average = intensive averaging) and that `env.smooth`'s **default stays
  `density`** (deliberate, backward-compatible; `average` recommended for rate maps).
  CHANGELOG entry for the new `mode="average"`.

## Deliberately not in this phase

- **Flipping `env.smooth`'s default to `average`** — stays `density` (would silently change
  every mode-less call). Tracked follow-up only.
- **Operator / geometry / low-level API** — all Phase 1.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_average_is_row_stochastic` | `compute_kernel(mode="average")` rows sum to 1 (uniform + polar) |
| `test_average_differs_from_density_nonuniform` | on polar, `average` kernel ≠ `density` kernel |
| `test_binned_unbiased_on_nonuniform_M` | `binned` on polar: smoothed rate is the valid-bin-normalized `H` average (matches `H@(r·mask)/H@mask`), **not** volume-biased (`≠` the `density`-path result); uniform grid unchanged |
| `test_resample_diffuse_masked_not_biased_down` | covered bins adjacent to uncovered region are **not** pulled toward 0 (vs. the old zero-fill single-smooth); `outside_source` stays `NaN` |
| `test_resample_diffuse_source_nan_no_propagation` | a `NaN` in the source field does **not** propagate across reachable bins (masked out via `isfinite` + zero-fill); only the originating bin stays `NaN` |
| `test_average_smooths_intensive_field` | a flat rate map smooths to itself under `average` (row-stochastic ⇒ constant preserved) on non-uniform `M` |

## Fixtures

Reuse Phase 1's `conftest.py` (polar env, masked grid, resample src/dst env pair). Add a
src→dst env pair with a partially-covered destination for the masked-resample test.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- `"average"` threads every mode surface listed; no site left on the two-mode Literal.
- `binned` and `resample` both routed through `H`-masked-average; the `density` mode is no
  longer used for these intensive fields.
- `env.smooth` default is still `density` (not silently flipped).
- Validation slice passes; tests assert de-biasing behavior, not tautologies.
- No plan/phase references in docstrings, test names, module names.
- Docstrings + CHANGELOG for `mode="average"` shipped in this PR.
