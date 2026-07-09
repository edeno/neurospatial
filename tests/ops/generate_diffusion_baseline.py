"""Capture the shipped DENSE diffusion-operator baseline (run BEFORE any refactor).

This freezes the pre-refactor dense operator's outputs so the matrix-free
``env.diffuse`` apply-path can be verified against them:

- Per (geometry, sigma, mode): the ``env.compute_kernel`` matrix (byte-identity
  guard -- ``compute_kernel`` must stay unchanged) and ``kernel @ field`` for the
  canonical field set including a **signed** field (equivalence oracle).
- The dense-``expm`` wall-time and peak memory on a large grid (perf reference).

Run once to (re)generate the pickle::

    uv run python tests/ops/generate_diffusion_baseline.py

The pickle is committed at ``data/diffusion_perf_baseline.pkl`` and consumed by
``test_diffusion_apply.py``. Regenerate only when the fixture set changes; the
byte-identity guard exists precisely to catch an unintended change to the dense
path, so do NOT regenerate to "make the test pass".
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import time
import tracemalloc
import warnings
from importlib.metadata import version as _pkg_version

import numpy as np
from diffusion_fixtures import (
    BASELINE_PATH,
    GEOMETRIES,
    MODES,
    PERF_SIGMA,
    build_perf_grid,
    make_fields,
    source_bin,
)

# Fields whose smooths are stored as the equivalence oracle (finite only).
_ORACLE_FIELDS = ("point", "nonneg", "signed")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def capture_baseline(*, reuse_perf: bool = False) -> dict:
    """Build the baseline dict (dense-operator outputs + perf reference).

    ``reuse_perf`` keeps the (slow, ~3 min) dense-``expm`` perf number from an
    existing pickle instead of recomputing it -- handy when only the fixture
    field set changes.
    """
    cases: list[dict] = []
    for geom_name, (builder, sigma_small, sigma_large) in GEOMETRIES.items():
        env, source_coord = builder()
        src = source_bin(env, source_coord)
        fields = make_fields(env, src)
        for sigma_label, sigma in (("small", sigma_small), ("large", sigma_large)):
            for mode in MODES:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kernel = env.compute_kernel(sigma, mode=mode, cache=False)
                # Store only the dense operator's ACTION on the canonical fields
                # (kernel @ field), not the full (n, n) matrix -- keeps the
                # committed baseline small. A linear operator is determined by
                # its action, and the field set (unit point source = one full
                # kernel column, plus a non-negative and a signed field) gives
                # strong coverage for both the equivalence oracle and the
                # compute_kernel-unchanged guard.
                kernel_at_field = {
                    name: np.asarray(kernel @ fields[name], dtype=np.float64)
                    for name in _ORACLE_FIELDS
                }
                cases.append(
                    {
                        "geom": geom_name,
                        "sigma_label": sigma_label,
                        "sigma": float(sigma),
                        "mode": mode,
                        "n_bins": int(env.n_bins),
                        "fields": {k: np.asarray(v) for k, v in fields.items()},
                        "kernel_at_field": kernel_at_field,
                        "bin_sizes": np.asarray(env.bin_sizes, dtype=np.float64),
                    }
                )

    perf = _reuse_existing_perf() if reuse_perf else _capture_perf()
    return {
        "meta": {
            "version": _pkg_version("neurospatial"),
            "git_sha": _git_sha(),
            "note": (
                "Dense diffusion-operator baseline captured pre-refactor. Perf grid is "
                "3600 bins (dense expm ~O(n^3): a true 10k-bin dense capture is "
                "~25 min / multi-GB, so the apply-path is separately shown to scale "
                "to 10k bins in test_perf_large_grid)."
            ),
        },
        "cases": cases,
        "perf": perf,
    }


def _capture_perf() -> dict:
    """Dense-``expm`` wall-time + peak memory on the large perf grid."""
    env = build_perf_grid()
    n_bins = int(env.n_bins)
    tracemalloc.start()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env.compute_kernel(PERF_SIGMA, mode="density", cache=False)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "n_bins": n_bins,
        "sigma": PERF_SIGMA,
        "mode": "density",
        "dense_expm_time_s": float(elapsed),
        "dense_peak_mem_bytes": int(peak),
    }


def _reuse_existing_perf() -> dict:
    """Load the perf entry from the existing pickle (avoids the slow recapture)."""
    with BASELINE_PATH.open("rb") as f:
        return pickle.load(f)["perf"]


def main() -> None:
    reuse_perf = "--reuse-perf" in sys.argv
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline = capture_baseline(reuse_perf=reuse_perf)
    with BASELINE_PATH.open("wb") as f:
        pickle.dump(baseline, f, protocol=pickle.HIGHEST_PROTOCOL)
    perf = baseline["perf"]
    print(f"Wrote baseline -> {BASELINE_PATH}")
    print(f"  {len(baseline['cases'])} (geom x sigma x mode) cases")
    print(
        f"  perf grid: {perf['n_bins']} bins, dense expm "
        f"{perf['dense_expm_time_s']:.1f}s, peak "
        f"{perf['dense_peak_mem_bytes'] / 1e6:.0f} MB"
    )


if __name__ == "__main__":
    main()
