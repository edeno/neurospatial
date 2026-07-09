# Designs — diffusion engine rework

[← back to PLAN.md](PLAN.md) · [shared contracts](shared-contracts.md) · [spec](design-correctness.md)

Algorithmic detail too large for the phase task blocks. Phases reference these by anchor.

## Index

- [D0. Module layout & operator assembly](#d0-module-layout--operator-assembly)
- [D1. Per-geometry finite-volume builders](#d1-per-geometry-finite-volume-builders)
- [D2. Graph junction contraction](#d2-graph-junction-contraction)
- [D3. Mesh skew guard](#d3-mesh-skew-guard)
- [D4. heat_kernel + per-component renormalization](#d4-heat_kernel--per-component-renormalization)
- [D5. Masked H-average (binned + resample)](#d5-masked-h-average-binned--resample)

---

## D0. Module layout & operator assembly

New module `src/neurospatial/ops/diffusion.py`. Public entry `diffusion_kernel(env, sigma,
*, mode)`; it (1) dispatches geometry (C3) to a builder that returns a **working graph**
carrying the `"A"` edge attribute plus a node-ordered `volumes` array, then (2) calls the
low-level `compute_diffusion_kernels`.

```python
def diffusion_kernel(env, sigma, *, mode="density"):
    graph_with_A, volumes = _finite_volume_geometry(env)   # dispatch, C3 / D1
    return compute_diffusion_kernels(graph_with_A, volumes=volumes, sigma=sigma, mode=mode)
```

`_finite_volume_geometry(env)` builds on a **copy** of `env.connectivity` (never mutate the
env's graph) and stamps `"A"` on every edge. Dispatch:

```python
def _finite_volume_geometry(env):
    if getattr(env, "_POLAR", False):          # env-level, NOT engine (C3)
        return _polar_fv(env)
    engine = type(env.layout).__name__          # ACTUAL class names (verified):
    builders = {
        "RegularGridLayout": _cartesian_fv, "MaskedGridLayout": _cartesian_fv,
        "ImageMaskLayout": _cartesian_fv, "ShapelyPolygonLayout": _cartesian_fv,
        "HexagonalLayout": _hex_fv, "GraphLayout": _graph_fv,
        "TriangularMeshLayout": _mesh_fv,
    }
    try:
        return builders[engine](env)
    except KeyError:
        raise NotImplementedError(f"diffusion kernel unsupported for layout {engine!r}")
```

The four `*GridLayout`/`ImageMaskLayout`/`ShapelyPolygonLayout` classes all subclass
`_GridMixin`, so `isinstance(env.layout, _GridMixin) → _cartesian_fv` is an equally valid
(and more future-proof) cartesian test. Use exact class names or the `_GridMixin` isinstance
check — do not paraphrase the names.

Low-level `compute_diffusion_kernels(graph, *, volumes, sigma, mode)` (rewrite of
`ops/smoothing.py`) assembles `L` and calls [D4](#d4-heat_kernel--per-component-renormalization):

```python
def compute_diffusion_kernels(graph, *, volumes, sigma, mode):
    n = graph.number_of_nodes()
    rows, cols, vals = [], [], []
    for u, v, data in graph.edges(data=True):
        if "A" not in data:                       # C6: missing A raises
            raise ValueError(f"edge ({u},{v}) has no 'A' (face measure) attribute")
        A = data["A"]; d = data["distance"]
        if not d > 0:
            raise ValueError(f"edge ({u},{v}) has non-positive distance {d}")
        if A == 0.0:                              # C6: explicit A=0 => no diffusion edge
            continue
        w = A / d                                 # finite-volume flux weight (see note)
        rows += [u, v]; cols += [v, u]; vals += [w, w]
    W = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return heat_kernel_from_W(W, np.asarray(volumes, float), sigma, mode=mode)
```

> **Note on `w = A/d`:** the finite-volume flux weight is `A/d` (face measure ÷ center
> distance). On a *uniform Cartesian* grid this equals the finite-difference `1/d²` the
> evidence script uses (face `A = h^(n−1)`, `d = h`, `w = h^(n−1)/h`; with `M = h^n` the
> operator `M⁻¹(D−W)` matches `1/d²`-no-mass — see spec §3.A). The σ-recovery tests (a
> priori tolerances, [design §4](design-correctness.md)) are the gate that this is right per
> geometry; if a geometry misses, it is an operator/spec correction, not a tolerance bump.

## D1. Per-geometry finite-volume builders

Each returns `(graph_copy_with_A, volumes)`. `volumes = env.bin_sizes` for every geometry
(C4). `d` is the existing edge `"distance"` unless noted. `A` per edge:

| geometry | `A[i,j]` (shared face) | notes |
| --- | --- | --- |
| cartesian | offset nonzero in exactly 1 axis → `∏` other-axis bin widths; else `A=0` | drops 8-conn diagonals; uniform grid → `A = h^(n_dims−1)` |
| hex | hex side length `s` (shared edge) | K-orthogonal; `d` = center distance (`√3·s` regular); no hand constant |
| polar | radial nbr → `r_face · Δθ` (arc); angular/seam nbr → `Δr` (radial segment) | K-orthogonal; `d` = existing polar `"distance"` |
| graph | `1.0` (unit cross-section) | `d` = along-track distance; junction edges via [D2](#d2-graph-junction-contraction) |
| mesh | shared triangle-edge length | `d` = centroid distance (existing); skew-guard [D3](#d3-mesh-skew-guard) |

**Cartesian** `A` (the one with real subtlety — masks/holes just drop nodes, geometry is
unchanged):

```python
def _cartesian_fv(env):
    g = env.connectivity.copy()
    centers = env.bin_centers                     # (n_bins, n_dims)
    widths = _per_axis_bin_widths(env)            # from env.edges_ / grid_edges, per axis
    for u, v, data in g.edges(data=True):
        offset = centers[u] - centers[v]
        moved = np.abs(offset) > 1e-9
        if moved.sum() != 1:                      # diagonal / Moore edge
            data["A"] = 0.0
            continue
        axis = int(np.flatnonzero(moved)[0])
        data["A"] = float(np.prod([widths[d_] for d_ in range(centers.shape[1]) if d_ != axis]))
    return g, env.bin_sizes
```

**Polar** `A` (edge type from whether the two bins share a distance-bin or an angle-bin;
derive `r`/`θ` indices from `place_bin` layout or `bin_centers[:,0]=r`, `[:,1]=θ`):

```python
def _polar_fv(env):
    g = env.connectivity.copy()
    r = env.bin_centers[:, 0]; theta = env.bin_centers[:, 1]
    for u, v, data in g.edges(data=True):
        dr = abs(r[u] - r[v]) > 1e-9
        # Angular delta must account for the ±π seam (wrap) before deciding pure-angular.
        dth = _angular_delta(theta[u], theta[v], env) > 1e-9
        if dr and not dth:                         # PURE radial: face = arc at boundary r
            r_face = 0.5 * (r[u] + r[v])
            data["A"] = float(r_face * _angular_bin_width(env))
        elif dth and not dr:                       # PURE angular / seam: face = radial segment
            data["A"] = float(_radial_bin_width(env, u))   # Δr of that ring
        else:                                      # DIAGONAL (both differ): corner touch, no face
            data["A"] = 0.0                        # mirror Cartesian diagonals — do NOT leak
    return g, env.bin_sizes
```

Hex, graph, mesh builders follow the table; `_hex_fv` derives `s` from the hex geometry
(adjacent-center distance / √3, or the layout's stored side length), `_graph_fv` sets
`A=1.0` and fixes junction `d` per D2, `_mesh_fv` sets `A =` shared-edge length (from the
Delaunay triangulation the layout already holds,
[triangular_mesh.py:166](../../../src/neurospatial/layout/engines/triangular_mesh.py)) and
emits the skew warning.

## D2. Graph junction contraction

Current linear-track junction (inter-segment) edges carry a Euclidean **chord** distance
([graph.py:296](../../../src/neurospatial/layout/helpers/graph.py)), which understates
along-track distance around bends and oversmooths. Replace `d` for junction edges with the
true along-track distance via Dijkstra over the substrate graph (bin-center, bin-edge, and
junction nodes), treating other bin-centers as sinks (record but don't expand through).
Mirror NLD's `_neighbor_centers` ([appendix](appendix.md#nld-junction-contraction)). Intra-
segment edges ([graph.py:261](../../../src/neurospatial/layout/helpers/graph.py)) already
carry correct along-track spacing — unchanged. A straight track has no junction edges, so
this is a no-op there.

## D3. Mesh skew guard

TPFA on triangle **centroids** (not circumcenters) is exact only as the mesh approaches
K-orthogonality. Compute, over interior (shared) edges, the non-orthogonality angle between
the centroid-connection vector `c_j − c_i` and the shared-edge normal. **A-priori
threshold**: if the fraction of interior edges with angle `> 30°` exceeds `5%`, emit a
`UserWarning` naming the fraction and that σ is approximate for this mesh. σ-recovery is
asserted (D-tests) only within the well-shaped regime; the warning makes the approximation
self-reporting, never silent.

## D4. heat_kernel + per-component renormalization

```python
def _raw_heat_operator(W, volumes, sigma):
    """Dense exp(-t L), L = M^-1(D-W). The M-self-adjoint operator of C1, PRE clip/normalize.
    Exposed as a seam so C1's raw invariants (H·1=1; M_i H_ij == M_j H_ji) are directly
    testable — the mode outputs are normalized and no longer expose them."""
    degree = np.asarray(W.sum(axis=1)).ravel()
    L = scipy.sparse.diags(1.0 / volumes) @ (scipy.sparse.diags(degree) - W)  # M^-1 (D-W)
    H = scipy.sparse.linalg.expm(-(sigma**2 / 2.0) * L)
    return np.asarray(H.todense()) if hasattr(H, "todense") else np.asarray(H)


def heat_kernel_from_W(W, volumes, sigma, *, mode):
    H = np.clip(_raw_heat_operator(W, volumes, sigma), 0.0, None)  # round-off; real lobes under PR2
    # Normalize EACH mode to ITS OWN contract (C1/C2). Do NOT row-normalize once and reuse:
    # row-normalization preserves row sums but not the M-weighted column sum, so `density`
    # would not integrate to 1 after clipping. Each branch below enforces its exact invariant.
    if mode == "average":                          # row-stochastic: kernel @ intensive rate
        s = H.sum(axis=1, keepdims=True)
        return H / np.where(s > 0, s, 1.0)
    if mode == "transition":                        # column-stochastic: (row-normalized H).T
        s = H.sum(axis=1, keepdims=True)
        return (H / np.where(s > 0, s, 1.0)).T
    if mode == "density":                           # M-weighted columns integrate to 1
        col_mass = volumes @ H                       # (n,): col_mass[j] = Σ_i M_i H[i,j]
        return H / np.where(col_mass > 0, col_mass, 1.0)[np.newaxis, :]
    raise ValueError(f"unknown mode {mode!r}")
```

**Components (C5) at full rank are automatic:** a disconnected `W` (masked wall, corner-only
`A=0`) makes `expm(−tL)` **block-diagonal**, and clipping adds no cross-block entries, so the
per-mode normalization is inherently within-component — no explicit component loop needed in
Phase 1. `_components_from_W` (`scipy.sparse.csgraph.connected_components(W, directed=False)`,
**not** `env.connectivity`) is used by the corner-split test and becomes load-bearing only
under **PR2** truncation, where clipping removes real lobes and can leak across blocks. PR2
replaces the `expm` line with the cached truncated eigenbasis of
`S = M^(−1/2)(D−W)M^(−1/2)`; keep the three mode outputs as the only mode-facing surface so
that swap stays internal.

## D5. Masked H-average (binned + resample)

Intensive fields with missing/invalid bins must be averaged, not mass-propagated, and the
missing bins must contribute **no weight** (not a real 0). Nadaraya-Watson with `H`:

```python
# H = compute_kernel(..., mode="average")   (row-stochastic)
num = H @ (value * valid)      # valid in {0,1}; value 0-filled where invalid
den = H @ valid
out = np.where(den > 0, num / den, np.nan)
out[outside_or_invalid] = np.nan               # re-impose missingness
```

- **binned** ([_smoothing.py:665](../../../src/neurospatial/encoding/_smoothing.py), plus
  `_binned_batch`:789 and the JAX paths :847/:934): already smooths `rate_filled` and a
  `{0,1}` weight mask, then divides — switch both `env.smooth(...)` calls from the default
  `density` to `mode="average"`. Semantics preserved (valid-bin-normalized), volume bias on
  non-uniform `M` removed.
- **resample** ([binning.py:790-813](../../../src/neurospatial/ops/binning.py)): currently
  zero-fills then single-smooths with `mode="transition"`. Replace with the masked average
  above using `mode="average"`, with `valid = (~outside_source) & np.isfinite(resampled)` and
  the value array **zero-filled where `~valid`** (a source `NaN` must contribute *no weight*,
  not propagate — `H @ (v·valid)` with an un-zeroed `NaN` still poisons every reachable bin).
  Re-impose `NaN` on `outside_source` (and where `den == 0`) at the end.
