# Domain-Specific Metrics vs Primitives

## The Question

User asks: "What about things like Skaggs information?"

This raises a key architectural question: **Should neurospatial include domain-specific neuroscience metrics, or focus on providing primitives that enable users to compute them?**

## Current State

### What Exists

Neurospatial currently provides **primitives** but **NOT domain-specific metrics**:

```python
# Primitives that exist
env.occupancy(times, positions)      # Time spent in each bin → p(x)
env.smooth(field, bandwidth)         # Gaussian smoothing
env.bin_at(points)                   # Map points to bins
env.distance_to(targets)             # Distance fields

# Domain-specific metrics do NOT exist
skaggs_information()                 # ❌ Not in library
sparsity()                           # ❌ Not in library
coherence()                          # ❌ Not in library
grid_score()                         # ❌ Not in library
border_score()                       # ❌ Not in library
```

### How Users Currently Compute Skaggs Information

From `examples/08_complete_workflow.ipynb`:

```python
def compute_spatial_information(firing_rate, occupancy_time):
    """
    Compute spatial information in bits per spike.

    Skaggs et al. (1993) metric:
    I = Σ p(x) * (r(x) / r_mean) * log2(r(x) / r_mean)

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate in each bin (Hz)
    occupancy_time : array, shape (n_bins,)
        Time spent in each bin (seconds)
    """
    # Occupancy probability
    total_time = occupancy_time.sum()
    p_x = occupancy_time / total_time

    # Valid bins (non-zero rate and occupancy)
    valid = (firing_rate > 0) & (p_x > 0)

    # Mean firing rate
    r_mean = (firing_rate * p_x).sum()

    # Spatial information
    info = 0.0
    for i in np.where(valid)[0]:
        r_i = firing_rate[i]
        p_i = p_x[i]
        info += p_i * (r_i / r_mean) * np.log2(r_i / r_mean)

    return info

# Usage
occupancy_time = env.occupancy(times, positions)
firing_rate = spike_counts / occupancy_time  # User computes
spatial_info = compute_spatial_information(firing_rate, occupancy_time)
```

**Key observation**: The user implements this themselves using existing primitives.

## Standard Neuroscience Metrics

### Tier 1: Commonly Used (High Priority)

These are standard metrics in every neuroscience lab:

#### 1. Skaggs Information (Spatial Information Content)

**Formula**: `I = Σ p(x) * (r(x) / r_mean) * log2(r(x) / r_mean)`

**Primitives needed**:
- ✅ Occupancy probability `p(x)` - **EXISTS**: `env.occupancy()`
- ✅ Firing rate map `r(x)` - User computes from spike counts
- ✅ Mean rate - NumPy
- ✅ Element-wise operations - NumPy

**Status**: All primitives exist, users can implement easily

#### 2. Sparsity

**Formula**: `S = (1 - (Σ p(x) * r(x))^2 / Σ p(x) * r(x)^2) / (1 - 1/N)`

**Primitives needed**:
- ✅ Occupancy probability - EXISTS
- ✅ Element-wise operations - NumPy

**Status**: All primitives exist

#### 3. Coherence (Spatial Smoothness)

**Formula**: Correlation between each bin and average of neighbors

**Primitives needed**:
- ✅ Firing rate map - User computes
- 🔶 **Neighbor aggregation** - MISSING (need `neighbor_reduce`)
- ✅ Correlation - NumPy/SciPy

**Status**: Requires `neighbor_reduce` primitive (proposed in PRIMITIVES_PROPOSAL.md)

#### 4. Grid Score

**Formula**: Spatial autocorrelation at 60° vs 30° and 90°

**Primitives needed**:
- ✅ Firing rate map - User computes
- ❌ **Spatial autocorrelation** - MISSING (critical gap!)
- ✅ Peak detection - SciPy

**Status**: **Requires `spatial_autocorrelation` primitive** (proposed in SPATIAL_OPERATORS.md)

#### 5. Border Score

**Formula**: Coverage and mean firing rate along boundaries

**Primitives needed**:
- ✅ Boundary bins - EXISTS: `env.boundary_bins()`
- ✅ Firing rate map - User computes
- ✅ Geometric calculations - NumPy

**Status**: All primitives exist

### Tier 2: Advanced Metrics

These are more specialized but still common:

#### 6. Place Field Properties

**Metrics**: Peak rate, field size, center of mass, number of fields

**Primitives needed**:
- ✅ Firing rate map - User computes
- ✅ Connected components - NetworkX
- ✅ Geometric properties - NumPy
- 🔶 **Contour detection** - Could benefit from better support

**Status**: Mostly exists, could be improved

#### 7. Head Direction Tuning

**Formula**: Mean vector length, preferred direction, Rayleigh z-score

**Primitives needed**:
- ✅ Circular statistics - User computes with NumPy
- ✅ Binning by angle - User implements

**Status**: All primitives exist (not spatial graph operations)

#### 8. Phase Precession

**Formula**: Correlation between position in field and spike phase

**Primitives needed**:
- ✅ Position in field (normalized distance) - User computes with `env.distance_to()`
- ✅ Circular-linear correlation - User implements

**Status**: All primitives exist

## Architectural Decision

### Option 1: Focus on Primitives Only (Current Approach)

**Pros**:
- Clean separation: neurospatial = spatial graph operations
- Users maintain flexibility
- No commitment to specific formulas or conventions
- Smaller, more focused package

**Cons**:
- Users must reimplement standard metrics
- Code duplication across labs
- Potential for errors in reimplementation
- Learning curve for new users

### Option 2: Add `neurospatial.metrics` Module

**Pros**:
- Convenience for common analyses
- Standardized implementations reduce errors
- Lower barrier to entry
- Citations and formulas documented

**Cons**:
- Package scope creep (where do we stop?)
- Maintenance burden
- Formula disagreements (e.g., smoothing before or after?)
- Different conventions across labs

### Option 3: Hybrid Approach (Recommended)

Provide primitives + optional metrics module:

**Structure**:
```python
# Core package: primitives only
from neurospatial import Environment
from neurospatial import neighbor_reduce, spatial_autocorrelation, gradient

# Optional metrics module (separate import)
from neurospatial.metrics import (
    skaggs_information,
    sparsity,
    coherence,
    grid_score,
    border_score,
)
```

**Guidelines for inclusion in `metrics`**:
1. **Widely standardized** (cited >100 times)
2. **Clear consensus formula** (not multiple competing definitions)
3. **Depends on spatial graph operations** (not just NumPy)
4. **Documented assumptions** (e.g., smoothing parameters)

**What this means**:
- ✅ Include: `skaggs_information`, `sparsity`, `grid_score`, `coherence`, `border_score`
- ❌ Exclude: `phase_precession`, `head_direction_tuning` (not spatial graph operations)
- ❌ Exclude: Place field detection (too many competing methods)
- 🔶 Maybe: `place_field_properties` (if simple, well-defined)

## Missing Primitives Block Some Metrics

### Critical Finding

**Some standard metrics CANNOT be computed without missing primitives**:

1. **Coherence** → Requires `neighbor_reduce` (proposed)
2. **Grid score** → Requires `spatial_autocorrelation` (**ESSENTIAL**, proposed)

**This strengthens the case for implementing proposed primitives:**
- Not just for RL/replay analysis
- **Needed for standard neuroscience metrics**

## Recommendation

### Immediate Action

1. **Implement missing primitives first** (PRIMITIVES_PROPOSAL.md + SPATIAL_OPERATORS.md):
   - `neighbor_reduce` (enables coherence)
   - `spatial_autocorrelation` (enables grid score) ← **CRITICAL**
   - `gradient` (general differential operator)
   - `divergence` (general differential operator)
   - `convolve` (custom kernel filtering)

2. **After primitives exist**, create `neurospatial.metrics` module with:
   ```python
   neurospatial/metrics.py

   def skaggs_information(firing_rate, occupancy, *, base=2):
       """Skaggs et al. (1993) spatial information content."""

   def sparsity(firing_rate, occupancy):
       """Skaggs et al. (1996) sparsity measure."""

   def coherence(firing_rate, env, *, op='mean'):
       """Muller & Kubie (1989) spatial coherence."""
       # Uses neighbor_reduce primitive

   def grid_score(firing_rate, env, *, method='sargolini'):
       """Sargolini et al. (2006) grid score."""
       # Uses spatial_autocorrelation primitive

   def border_score(firing_rate, env):
       """Solstad et al. (2008) border score."""
   ```

### Documentation Strategy

Each metric function should include:
- Formula with math notation
- Citation (paper, year, journal)
- Parameter assumptions
- Example usage
- Link to primitive operations used

Example:
```python
def grid_score(
    firing_rate: NDArray[np.float64],
    env: Environment,
    *,
    method: Literal['sargolini', 'langston'] = 'sargolini',
) -> float:
    """
    Compute grid score (gridness metric).

    Grid score quantifies hexagonal firing pattern periodicity by comparing
    spatial autocorrelation at 60° rotations (grid alignment) vs 30° and 90°
    rotations (control angles).

    Formula (Sargolini et al. 2006):
        grid_score = min(r60, r120) - max(r30, r90, r150)

    where r_θ is the peak autocorrelation after rotating by θ degrees.

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz). Should be smoothed before calling.
    env : Environment
        Spatial environment defining bin layout.
    method : {'sargolini', 'langston'}, default='sargolini'
        Grid score computation method:
        - 'sargolini': Original formulation (Sargolini et al. 2006)
        - 'langston': Corrected version (Langston et al. 2010)

    Returns
    -------
    score : float
        Grid score. Typical values:
        - Grid cells: > 0.4
        - Spatial cells (non-grid): -0.2 to 0.2
        - Non-spatial: < -0.2

    References
    ----------
    .. [1] Sargolini et al. (2006). Conjunctive Representation of Position,
           Direction, and Velocity in Entorhinal Cortex. Science 312(5774).
    .. [2] Langston et al. (2010). Development of the Spatial Representation
           System in the Rat. Science 328(5985).

    See Also
    --------
    spatial_autocorrelation : Primitive used for computation

    Examples
    --------
    >>> # Compute grid score for a neuron
    >>> firing_rate = spike_counts / occupancy
    >>> firing_rate_smooth = env.smooth(firing_rate, bandwidth=3.0)
    >>> score = grid_score(firing_rate_smooth, env)
    >>> print(f"Grid score: {score:.3f}")
    Grid score: 0.523

    Notes
    -----
    **Important**: Firing rate should be smoothed before computing grid score
    to reduce noise. Typical smoothing bandwidth: 3-5 cm.

    This function requires the spatial_autocorrelation primitive, which
    computes autocorrelation on irregular spatial graphs.
    """
    # Implementation using spatial_autocorrelation primitive
    autocorr = spatial_autocorrelation(firing_rate, env)
    # ... rest of computation
```

## Comparison: Metrics vs Primitives

| Metric | Type | Primitives Needed | All Exist? |
|--------|------|-------------------|------------|
| Skaggs information | Domain | occupancy, element-wise ops | ✅ Yes |
| Sparsity | Domain | occupancy, element-wise ops | ✅ Yes |
| **Coherence** | Domain | occupancy, **neighbor_reduce** | ❌ No |
| **Grid score** | Domain | firing_rate, **spatial_autocorrelation** | ❌ No |
| Border score | Domain | boundary_bins, firing_rate | ✅ Yes |
| neighbor_reduce | **Primitive** | graph neighbors | ✅ Exists (as loop) |
| spatial_autocorrelation | **Primitive** | distance_field, correlations | ❌ No |
| gradient | **Primitive** | differential operator D | ❌ No |

**Key insight**: Implementing primitives **unblocks multiple domain-specific metrics**.

## Summary

**Answer to "What about Skaggs information?"**:

1. **Currently**: Users implement it themselves using existing primitives (see examples/08_complete_workflow.ipynb)

2. **Missing primitives block some metrics**:
   - Coherence needs `neighbor_reduce`
   - Grid score needs `spatial_autocorrelation` ← **CRITICAL**

3. **Recommendation**:
   - **First**: Implement primitives (especially `spatial_autocorrelation`)
   - **Then**: Add optional `neurospatial.metrics` module with standard metrics

4. **Architectural guideline**:
   - Core package = spatial graph primitives
   - Optional metrics module = domain-specific convenience functions
   - Users can always implement custom metrics using primitives

**The primitives we proposed (neighbor_reduce, spatial_autocorrelation, gradient, etc.) are even MORE important than we thought - they're required for standard neuroscience analyses, not just RL/replay!**
