# Shared contracts

[← back to PLAN.md](PLAN.md)

Contracts referenced by ≥2 phases. Each lives here once; phases link in by anchor and must not weaken these.

## Index

- [Input-validation helpers](#input-validation-helpers) — generic finite/length/range guards (phases 4, 5, 6, 7, 8, 9).
- [Result-object contract](#result-object-contract) — the `ResultMixin` surface (phases 17, 20).

---

## Input-validation helpers

Several correctness phases add the same shape of guard: reject non-finite values, reject mismatched array lengths, with an actionable message naming the offending argument. To avoid each module growing its own ad-hoc version, add **one** generic home and reuse it.

**Home:** `src/neurospatial/_validation.py` (new top-level private module). Domain-specific validators (`encoding/_validation.py`, `layout/validation.py`, etc.) stay where they are and may call these.

```python
# src/neurospatial/_validation.py
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def validate_finite(a: NDArray, *, name: str, allow_nan: bool = False) -> NDArray:
    """Return ``a`` as float64, raising ValueError on non-finite values.

    Parameters
    ----------
    a : array-like
        Values to check.
    name : str
        Argument name, used in the error message.
    allow_nan : bool, optional
        If True, NaN is permitted (but Inf is not). Default False.
    """
    arr = np.asarray(a, dtype=np.float64)
    bad = ~np.isfinite(arr)
    if allow_nan:
        bad &= ~np.isnan(arr)
    if bad.any():
        n = int(bad.sum())
        first = int(np.argmax(bad))
        raise ValueError(
            f"{name} contains {n} non-finite value(s) "
            f"(first at index {first}: {arr.flat[first]!r}). "
            f"Remove or mask them before calling."
        )
    return arr


def validate_lengths(name_to_array: dict[str, NDArray]) -> None:
    """Raise ValueError if the named 1-D arrays do not share a length.

    Example
    -------
    >>> validate_lengths({"spike_times": s, "times": t, "positions": p})
    """
    lengths = {k: len(np.asarray(v)) for k, v in name_to_array.items()}
    if len(set(lengths.values())) > 1:
        pairs = ", ".join(f"{k}={n}" for k, n in lengths.items())
        raise ValueError(f"Length mismatch: {pairs}. These must agree.")
```

**Semantics (do not weaken):**
- `validate_finite` **raises** on Inf always; raises on NaN unless `allow_nan=True`. It never silently coerces NaN→0 or drops values — callers that legitimately need NaN-dropping do it explicitly and visibly.
- `validate_lengths` compares lengths only; it does not reshape or broadcast. Length-1 "broadcastable" arrays are a mismatch, not a convenience (this is the exact stats weighted-circular bug — phase 4).
- Error messages name the argument and the first offending index/value. This matches the project's rich-diagnostic bar (the `E1001` "no active bins" message is the model).

Phases using these: 4 (weighted-circular weights), 5 (headings, classifier inputs), 6 (dt/timestamps), 7 (`distance_to_reward` reward times), 8 (`heading_from_velocity` dt, egocentric targets), 9 (NWB read length agreement).

---

## Result-object contract

`src/neurospatial/encoding/_base.py:158` already defines `SpatialResultMixin`, used by all `*RateResult`/`*RatesResult` encoding classes. The design gap (DESIGN-REVIEW Med; SUMMARY Theme 7) is that several user-facing result objects are **bare dataclasses** that do not share it — notably `DirectionalPlaceFields` and `PlaceFieldsResult` (`encoding/spatial.py`) — so the analysis-ending verbs ("compare them", "score it", "to a table") dead-end to manual NumPy.

**Contract established in phase 17** — a result mixin guaranteeing a uniform surface:

```python
class ResultMixin:
    def to_dataframe(self) -> "pandas.DataFrame": ...   # tidy long form; pandas imported lazily
    def summary(self) -> dict[str, float]: ...           # scalar headline metrics
    def plot(self, ax=None, **kwargs): ...               # a sensible default visualization
```

**Invariants (do not weaken):**
- **Additive only.** Introducing the mixin must not remove or rename any existing attribute/accessor on classes that already have them (e.g. `SpatialRateResult.firing_rate`, `.occupancy`, `.spatial_information()`). Pre-1.0 lets us rename elsewhere, but result-object accessors are load-bearing in the journeys (DESIGN-REVIEW "keep").
- `to_dataframe()` returns tidy/long form so heterogeneous results compose in one `pd.concat`.
- `plot()` accepts an optional `ax` and returns it, for composition into multi-panel figures.
- Phase 17 decides (Open Q2) whether `SpatialResultMixin` is generalized into / re-parented under `ResultMixin`, or `ResultMixin` is a new base both adopt. Either way the existing `SpatialResultMixin` accessors are preserved.

**Extended in phase 20:** `to_xarray()` is added to the array-shaped results (`DecodingResult` with `('time','bin')`, `SpatialRatesResult` with `('neuron','bin')`), importing `xarray` lazily so it stays an optional dependency.

Phases using this: 17 (defines + backfills bare dataclasses), 20 (`to_xarray` extension).
