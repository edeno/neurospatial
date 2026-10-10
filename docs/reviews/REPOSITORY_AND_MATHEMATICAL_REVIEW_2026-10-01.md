# Repository, Architectural, and Mathematical Review — 2026-10-01

**Repository:** `edeno/neurospatial`  
**Current Branch:** `feat/public-api-curation`  
**Package Version:** `0.8.0`  
**Date:** 2026-10-01  
**Scope:** Full repository audit covering architectural design, API contracts, scientific data integrity, testing/CI health, and an in-depth mathematical and numerical verification of all core algorithms.

---

## 1. Executive Summary

**neurospatial** is an advanced scientific Python library designed to discretize continuous $N$-dimensional spatial environments into graph-connected bins/nodes for systems neuroscience (place fields, grid cells, head direction cells, border cells, object-vector cells, spatial view cells, Bayesian decoding, and behavioral segmentation).

The repository demonstrates extraordinary engineering discipline, scientific rigor, and architectural maturity:
- **Boundary-Aware Finite-Volume Diffusion:** Replaces unconstrained Euclidean Gaussian smoothing with graph-based diffusion that respects barriers, walls, and non-convex boundaries.
- **Scientific Data Integrity:** Implements SHA-256 canonical JSON structural identity and revision tracking (`EnvironmentRef`), preventing accidental cross-environment analyses.
- **Clock-Qualified Temporal Support:** Uses explicit half-open interval sets (`TemporalSupport`) to prevent implicit bridging across recording pauses or trial intervals.
- **Joined Unit Identity:** Treats unit identity as a checked join key (`unit_ids`, `unit_table`) rather than an arbitrary positional row index.
- **Array-First & High Performance:** Provides NumPy baselines with optional GPU acceleration via JAX and standardized result objects (`ResultMixin`).

Overall, the mathematical foundations of the library are robust and principled. However, this audit identifies **one significant mathematical inconsistency in `ops/calculus.py`**, **four subtle numerical nuances/edge cases**, and **two test-suite linter errors that currently block CI**.

---

## 2. Mathematical and Numerical Audit

### 2.1 Significant Mathematical Inconsistency: `ops/calculus.py` (Metric Distance vs. Graph Affinity)

#### Location
- [`src/neurospatial/ops/calculus.py:43-156`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/ops/calculus.py#L43-L156) (`compute_differential_operator`)
- [`src/neurospatial/ops/calculus.py:159-263`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/ops/calculus.py#L159-L263) (`gradient`)
- [`src/neurospatial/ops/calculus.py:265-350`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/ops/calculus.py#L265-L350) (`divergence`)

#### The Formulation
In `compute_differential_operator`:
```python
distance = edge_data["distance"]
sqrt_weight = np.sqrt(distance)

# Source node gets negative weight
data_values[idx] = -sqrt_weight
...
# Destination node gets positive weight
data_values[idx] = sqrt_weight
```
And in `gradient(f) = D.T @ f`:
$$\text{grad}(f)_e = \sqrt{d_e} \cdot (f_j - f_i)$$
The associated Laplacian is:
$$\text{Lap}(f)_i = (D D^T f)_i = \sum_{j \sim i} d_{ij} (f_i - f_j)$$

#### Mathematical Analysis & Issue
1. **Inverse vs. Direct Scaling with Distance:**
   In physical continuous space and discrete calculus, the directional derivative along an edge of length $d$ between node $i$ and node $j$ is:
   $$\frac{\partial f}{\partial s} \approx \frac{f_j - f_i}{d}$$
   The rate of change is **inversely proportional** to the distance $d$. In `calculus.py`, the gradient is **directly proportional** to $\sqrt{d}$. An edge that is 10 times longer will report a $\sqrt{10} \approx 3.16\times$ larger gradient for the exact same difference in field values, rather than a $10\times$ smaller gradient.
2. **Conflict with [`ops/diffusion.py`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/ops/diffusion.py):**
   In `diffusion.py`, the finite-volume Laplacian $L = M^{-1}(D - W)$ correctly discretizes $-\nabla^2$ with $W_{ij} = A_{ij} / d_{ij}$ (dividing by distance). In `calculus.py`, $W_{ij} = d_{ij}$ (multiplying by distance).
3. **Dimensionality Error:**
   If $f$ has units of firing rate $[\text{Hz}]$ and distance is in $[\text{cm}]$:
   - Physical gradient: $[\text{Hz} / \text{cm}]$.
   - `calculus.py` gradient: $[\text{Hz} \cdot \sqrt{\text{cm}}]$.
   - Continuous Laplacian: $[\text{Hz} / \text{cm}^2]$.
   - `calculus.py` Laplacian: $[\text{Hz} \cdot \text{cm}]$.
4. **Why It Passed Existing Tests:**
   Unit tests and doctests (e.g. line 90) use `data = np.array([[0.0], [1.0], [2.0], [3.0]])` with `bin_size = 1.0`. When $d \equiv 1.0$, $\sqrt{1.0} = 1.0 / 1.0 = 1.0$, hiding the inverse relationship. As soon as $d \neq 1.0$, diagonal edges ($\sqrt{2}$) are present, or irregular meshes are analyzed, the gradient scales in the wrong direction.
5. **Root Cause:**
   In graph signal processing (PyGSP), edge weights $W_{ij}$ represent *affinity* or *similarity* (where larger means closer/more coupled). In NetworkX, `nx.laplacian_matrix(G, weight="distance")` treats the attribute `"distance"` directly as $W_{ij}$, conflating metric distance with graph affinity.

---

### 2.2 Numerical Nuances & Edge Cases

#### A. Zero-Prior Clipping to $10^{-10}$ in Bayesian Decoding
- **Location:** [`src/neurospatial/decoding/posterior.py:415-417`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/decoding/posterior.py#L415-L417)
- **Code:**
  ```python
  prior_clipped = np.clip(prior_arr, 1e-10, 1.0)
  log_prior = np.log(prior_clipped)
  ```
- **Analysis:** When a user passes an explicit structural prior with exact $0.0$ probability (e.g. for obstacle bins or unvisited zones), clipping forces the prior probability to $10^{-10}$. In time bins with zero spikes or uninformative likelihoods, those impossible bins receive a small but non-zero posterior probability rather than exactly $0.0$.
- **Recommendation:** Allow exact zeros in the prior by setting $\log(0) = -\infty$. In the log-sum-exp normalization, $-\infty$ bins naturally evaluate to $\exp(-\infty) = 0.0$ in the posterior without causing numerical instability.

#### B. Catastrophic Cancellation in Circular-Linear Correlation P-Values
- **Location:** [`src/neurospatial/stats/circular.py:833`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/stats/circular.py#L833)
- **Code:**
  ```python
  chi2_stat = n * r_squared
  pval = float(1.0 - chi2.cdf(chi2_stat, df=2))
  ```
- **Analysis:** For large sample sizes or strong correlations where $\chi^2 > 700$, `chi2.cdf` evaluates to `1.0` in IEEE 754 float64 arithmetic. The expression `1.0 - chi2.cdf(...)` cancels catastrophically to `0.0`, discarding extreme tail p-values.
- **Recommendation:** Use the survival function `chi2.sf(chi2_stat, df=2)`, which evaluates extreme tail probabilities accurately down to $\sim 10^{-300}$.

#### C. Antipodal Singularity in Circular Heading Interpolation
- **Location:** [`src/neurospatial/ops/egocentric.py:855-860`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/ops/egocentric.py#L855-L860)
- **Code:**
  ```python
  cos_interp = np.interp(invalid_indices, valid_indices, cos_h[valid_indices])
  sin_interp = np.interp(invalid_indices, valid_indices, sin_h[valid_indices])
  result[mask] = np.arctan2(sin_interp, cos_interp)
  ```
- **Analysis:** If an animal turns $180^\circ$ ($\Delta \theta = \pi$) across an unobserved gap, linear interpolation between $(\cos \theta, \sin \theta) = (1, 0)$ and $(-1, 0)$ passes directly through $(0, 0)$. At the midpoint, `np.arctan2(0.0, 0.0)` evaluates to `0.0` (East), causing an abrupt directional discontinuity.
- **Recommendation:** Detect when $\cos^2 + \sin^2 \approx 0$ and choose an angular geodesic path (or spherical linear interpolation, SLERP) across the circle.

#### D. Non-Uniform Grid Spacing Assumption in Cartesian Face Measures
- **Location:** [`src/neurospatial/ops/diffusion.py:1288-1300`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/ops/diffusion.py#L1288-L1300)
- **Code:**
  ```python
  def _per_axis_bin_widths(env: EnvironmentProtocol) -> NDArray[np.float64]:
      grid_edges = cast("Any", env.layout).grid_edges
      return np.array([float(np.diff(edges)[0]) for edges in grid_edges])
  ```
- **Analysis:** Taking the first cell spacing `np.diff(edges)[0]` assumes uniform bin widths along each axis. For custom layouts with variable grid spacing, the face measure $A$ treats all cells as having the first cell's width, while cell volumes $M$ vary per cell, breaking the physical $\sigma$ diffusion equivalence.

---

### 2.3 Confirmed Mathematically Correct Implementations

The following key algorithms were audited and verified to be strictly mathematically correct:
1. **Finite-Volume Diffusion ([`diffusion.py`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/ops/diffusion.py)):**
   - $H(\sigma) = \exp(-t L)$ with $t = \sigma^2 / 2$ and $L = M^{-1}(D - W)$ where $W_{ij} = A_{ij}/d_{ij}$.
   - Mass conservation holds in `"transition"` mode ($H^T$ column-stochastic).
   - Density integration holds in `"density"` mode ($H M^{-1}$ column-integrates to 1 under cell volume measure).
   - Intensive averaging holds in `"average"` mode ($H$ row-stochastic).
   - Component-decomposed eigensolves prevent spurious mode-leakage across barriers.
2. **Log Poisson Likelihood & Bayesian Decoding ([`likelihood.py`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/decoding/likelihood.py), [`posterior.py`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/decoding/posterior.py)):**
   - Implements $\sum_i [n_i \log(\lambda_i \Delta t) - \lambda_i \Delta t]$ with log-sum-exp numerical stabilization.
3. **Penalized Poisson GAM ([`_glm.py`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/encoding/_glm.py), [`_glm_numpy.py`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/encoding/_glm_numpy.py)):**
   - Correctly models counts with log-occupancy exposure offsets ($\log \mu = \log(\text{occ}) + B \gamma$), eliminating low-occupancy division artifacts.
4. **Spatial Information & Sparsity ([`_core_numpy.py`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/encoding/_core_numpy.py)):**
   - Skaggs spatial information: $\sum_i p_i \frac{r_i}{\bar{r}} \log_2 \left(\frac{r_i}{\bar{r}}\right)$ with proper $\lim_{r \to 0} r \log r = 0$ handling.
   - Treves-Rolls spatial sparsity: $\frac{(\sum p_i r_i)^2}{\sum p_i r_i^2}$.
5. **Rayleigh Test Finite-Sample Correction ([`circular.py`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/stats/circular.py#L696-L701)):**
   - Correctly reproduces Mardia & Jupp (2000) Eq. 5.3.6:
     $$p \approx e^{-Z} \left( 1 + \frac{2Z - Z^2}{4n} - \frac{24Z - 132Z^2 + 76Z^3 - 9Z^4}{288n^2} \right)$$
6. **Earth Mover's Distance ([`_field_metrics.py`](file:///Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/encoding/_field_metrics.py#L1060-L1335)):**
   - Formulates exact optimal transport via linear programming (HiGHS solver) with Euclidean and geodesic distance matrices.

---

## 3. Code Quality & CI Health Findings

Running `ruff check .` on the current working tree identified **two linting errors in the test suite that will block CI on push/PR**:

1. **`tests/nwb/test_fields_pooled.py:205:42`**  
   - **Rule:** `RUF043` (Pattern passed to `match=` contains metacharacters but is neither escaped nor raw).  
   - **Code:** `with pytest.raises(ValueError, match="predates schema 3.0"):`  
   - **Cause:** The unescaped `.` in `3.0` acts as a regex wildcard.  
   - **Remediation:** Change to `match=r"predates schema 3\.0"`.

2. **`tests/test_session_nwb_roundtrip_contract.py:293:5`**  
   - **Rule:** `B018` (Useless attribute access).  
   - **Code:** `sess.read_session(os.fspath(destination)).env  # no raise`  
   - **Cause:** Standalone attribute access without assignment or assertion.  
   - **Remediation:** Change to `assert sess.read_session(os.fspath(destination)).env is not None`.

---

## 4. Prioritized Action Items

| Priority | Component | Issue | Recommended Fix |
|---|---|---|---|
| **P0** | Tests | 2 linter failures blocking CI (`RUF043`, `B018`) | Escape regex dot in `test_fields_pooled.py`; add assert in `test_session_nwb_roundtrip_contract.py`. |
| **P1** | `ops/calculus.py` | Distance-weighted gradient & Laplacian scale as $\sqrt{d}$ rather than $1/d$ | Redefine edge differential operator with inverse-distance weights $1/\sqrt{d_e}$ or divide gradient by $d_e$. |
| **P2** | `decoding/posterior.py` | Prior probability clipped to $10^{-10}$ | Allow $\log(0) = -\infty$ for exact zero-probability bins under log-sum-exp. |
| **P2** | `stats/circular.py` | Potential cancellation in `1.0 - chi2.cdf` | Use `chi2.sf(chi2_stat, df=2)` for extreme tail precision. |
| **P3** | `ops/egocentric.py` | $180^\circ$ turn interpolation midpoint hits $(0, 0)$ | Add antipodal check in `_interpolate_heading_circular`. |
| **P3** | `ops/diffusion.py` | Uniform grid spacing assumed in `_per_axis_bin_widths` | Support per-edge non-uniform Cartesian face measures. |
