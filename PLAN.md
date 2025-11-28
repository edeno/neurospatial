# Bayesian Decoding Subpackage Implementation Plan

## Overview

Create a new `decoding` subpackage within neurospatial for population-level neural analysis, starting with Bayesian decoding. This will enable researchers to decode spatial position from population neural activity using standard Bayesian methods.

## Reference Implementation Analysis

The [replay_trajectory_classification](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification) `standard_decoder.py` provides these core capabilities:

1. **Likelihood Computation**
   - `poisson_mark_log_likelihood()` - Log likelihood for marked point process
   - `predict_mark_likelihood()` - Likelihood from clusterless marks
   - `predict_poisson_likelihood()` - Likelihood from place field encoding models

2. **Posterior Estimation**
   - `normalize_to_posterior()` - Bayes rule normalization with optional priors

3. **Position Decoding**
   - `map_estimate()` - Maximum a posteriori position
   - `weighted_correlation()` - Decoded vs actual correlation

4. **Trajectory Analysis**
   - `isotonic_regression()` - Monotonic trajectory fitting
   - `linear_regression()` - Linear trajectory fitting
   - `detect_line_with_radon()` - Radon transform for trajectory detection

## Proposed Architecture

### Package Structure

```
src/neurospatial/decoding/
├── __init__.py              # Public API exports
├── _result.py               # DecodingResult dataclass
├── likelihood.py            # Likelihood computation functions
├── posterior.py             # Posterior estimation and normalization
├── estimates.py             # Point estimates (MAP, mean, entropy)
├── trajectory.py            # Trajectory fitting (isotonic, linear, radon)
├── metrics.py               # Decoding quality metrics
└── shuffle.py               # Statistical shuffling procedures for significance testing
```

**Note**: `_result.py` (not `_base.py`) since it contains only `DecodingResult`.
If common utilities are needed later, add `_utils.py`.

### Core Design Decisions

#### 1. Stateless Functions (Not Classes)

Following neurospatial's pattern (like `compute_place_field`, metrics functions), use **stateless functions** rather than decoder classes:

```python
# Preferred: stateless function
posterior = decode_position(
    env, spike_counts, encoding_models, times,
    prior=uniform_prior,
)

# NOT: stateful class
decoder = BayesianDecoder(env, encoding_models)
posterior = decoder.decode(spike_counts)
```

**Rationale**:

- Consistent with existing neurospatial API (spike_field.py, metrics/)
- Easier to test and reason about
- Natural fit for vectorized operations
- Avoids hidden state issues

#### 2. Environment Integration

Decoding functions take `Environment` as first argument:

- Provides spatial discretization (`bin_centers`, `n_bins`)
- Enables graph-aware operations (distance fields for priors)
- Consistent with existing API patterns

#### 3. Encoding Model Representation

Place fields as encoding models:

```python
# Shape: (n_neurons, n_bins) - firing rate at each bin
encoding_models: NDArray[np.float64]
```

This aligns with `compute_place_field()` output - a (n_bins,) array per neuron.

#### 4. Time Binning Strategy

Two approaches supported:

1. **Pre-binned**: User provides spike counts per time bin
2. **Raw spikes**: User provides spike times, function bins internally

```python
# Pre-binned (most common)
posterior = decode_position(
    env, spike_counts,  # (n_time_bins, n_neurons)
    encoding_models,    # (n_neurons, n_bins)
    dt=0.025,           # Time bin width in seconds
)

# Raw spikes (convenience)
posterior = decode_position_from_spikes(
    env, spike_times_list, encoding_models, time_bins,
)
```

#### 5. Shape/Dimension Conventions

All functions follow consistent array shape conventions:

| Array | Shape | Description |
|-------|-------|-------------|
| `env.bin_centers` | `(n_bins, n_dims)` | Bin center coordinates |
| `spike_counts` | `(n_time_bins, n_neurons)` | Spike count matrix |
| `encoding_models` | `(n_neurons, n_bins)` | Firing rate maps (place fields) |
| `posterior` | `(n_time_bins, n_bins)` | Posterior probability distribution |
| `decoded_positions` | `(n_time_bins, n_dims)` | Decoded position estimates |
| `actual_positions` | `(n_time_bins, n_dims)` | Ground truth positions |
| `times` | `(n_time_bins,)` | Time bin centers |

**Key invariants**:

- `n_bins = env.n_bins` (from Environment)
- `n_dims = env.n_dims` (typically 1, 2, or 3)
- `n_neurons` = number of units in encoding model
- `n_time_bins` = number of decoding time windows

#### 6. DecodingResult Computation Strategy

**Decision: Eager computation with cached properties**

The `DecodingResult` dataclass stores only `posterior` and `env` as primary data. Derived quantities (`map_estimate`, `map_position`, `mean_position`, `uncertainty`) are computed **lazily** via `@cached_property`:

```python
@dataclass
class DecodingResult:
    """Container for Bayesian decoding results."""
    posterior: NDArray[np.float64]  # (n_time_bins, n_bins) - primary data
    env: Environment                # Reference for coordinate transforms
    times: NDArray[np.float64] | None = None

    @cached_property
    def map_estimate(self) -> NDArray[np.int64]:
        """Bin index of maximum posterior probability."""
        return np.argmax(self.posterior, axis=1)

    @cached_property
    def map_position(self) -> NDArray[np.float64]:
        """MAP position in environment coordinates."""
        return self.env.bin_centers[self.map_estimate]

    @cached_property
    def mean_position(self) -> NDArray[np.float64]:
        """Posterior mean position (expected value)."""
        return self.posterior @ self.env.bin_centers

    @cached_property
    def uncertainty(self) -> NDArray[np.float64]:
        """Posterior entropy (bits)."""
        p = np.clip(self.posterior, 1e-10, 1.0)
        return -np.sum(p * np.log2(p), axis=1)
```

**Rationale**:

- Avoids computing unused quantities (e.g., if user only needs posterior)
- `@cached_property` ensures each is computed at most once
- Posterior is the primary output; everything else derives from it
- Memory-efficient: no redundant storage of derived arrays

**Note**: Using `@cached_property` requires Python 3.8+ and makes the class non-frozen. Alternative: use `__slots__` + manual caching if frozen semantics are required.

### Detailed Function Specifications

#### Module: `likelihood.py`

**Design note on `poisson_likelihood` vs `log_poisson_likelihood`:**

The primary function is `log_poisson_likelihood` - this is numerically stable and should
be used in nearly all cases. The `poisson_likelihood` function is provided as a thin
wrapper for users who need probability-space likelihoods, but directly computing raw
likelihoods with `np.exp` is prone to underflow/overflow and rarely needed in practice.

**Recommendation**: Use `log_poisson_likelihood` + `normalize_to_posterior` for decoding.
Reserve `poisson_likelihood` for visualization or when explicitly needed.

```python
def log_poisson_likelihood(
    spike_counts: NDArray[np.int64],      # (n_time_bins, n_neurons)
    encoding_models: NDArray[np.float64], # (n_neurons, n_bins)
    dt: float,                            # Time bin width (seconds)
    *,
    min_rate: float = 1e-10,              # Numerical stability floor
) -> NDArray[np.float64]:
    """Compute log Poisson likelihood (numerically stable).

    This is the **primary likelihood function** and should be used for all
    decoding pipelines.

    Computes the log-likelihood up to an additive constant:

        log P(spikes | position) ∝ sum_i [n_i * log(lambda_i * dt) - lambda_i * dt]

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) in Hz. Typical values: 0-50 Hz.
        Values are clipped to [min_rate, inf) internally.
    dt : float
        Time bin width in seconds. Typical values: 0.001-0.1s.
        Note: lambda*dt should typically be in [0, 5] for numerical stability.
    min_rate : float, default=1e-10
        Minimum firing rate floor to avoid log(0).

    Returns
    -------
    log_likelihood : NDArray[np.float64], shape (n_time_bins, n_bins)
        Log-likelihood up to an additive constant per time bin.

    Notes
    -----
    The -log(n_i!) term is **omitted** because it is constant across positions
    for each time bin. Since we normalize to posterior via softmax, this constant
    cancels out and omitting it saves O(n_neurons) log-gamma evaluations per
    time bin. This optimization is especially beneficial for large populations.

    The returned values are log-likelihoods up to an additive constant per time
    bin. They are suitable for `normalize_to_posterior()` but NOT for model
    comparison across different spike patterns.

    For extremely high firing rates (lambda*dt >> 10), consider increasing dt
    or normalizing encoding_models to avoid large exponents.
    """

def poisson_likelihood(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    dt: float,
    *,
    min_rate: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute Poisson likelihood in probability space (thin wrapper).

    **Warning**: This function can underflow/overflow for realistic spike
    trains with large populations. Prefer `log_poisson_likelihood` +
    `normalize_to_posterior` for decoding.

    This is implemented as:
        log_ll = log_poisson_likelihood(spike_counts, encoding_models, dt, min_rate)
        return np.exp(log_ll - log_ll.max(axis=1, keepdims=True))

    The likelihoods are normalized per row to prevent underflow, but this
    means they are NOT true probabilities and should only be used for
    visualization or when probability-space is explicitly required.

    Returns
    -------
    likelihood : NDArray[np.float64], shape (n_time_bins, n_bins)
        Likelihood ratios (normalized per time bin to prevent underflow).
    """
```

#### Module: `posterior.py`

```python
def normalize_to_posterior(
    log_likelihood: NDArray[np.float64],  # (n_time_bins, n_bins)
    *,
    prior: NDArray[np.float64] | None = None,  # (n_bins,) or (n_time_bins, n_bins)
    axis: int = -1,
    handle_degenerate: Literal["uniform", "nan", "raise"] = "uniform",
) -> NDArray[np.float64]:
    """Convert log-likelihood to posterior using Bayes' rule.

    P(position | spikes) = P(spikes | position) * P(position) / P(spikes)

    Parameters
    ----------
    log_likelihood : NDArray[np.float64], shape (n_time_bins, n_bins)
        Log-likelihood from `log_poisson_likelihood` or similar.
    prior : NDArray[np.float64] | None, default=None
        Prior probability over positions. If None, uses uniform prior.
        Shape (n_bins,) for stationary prior, (n_time_bins, n_bins) for
        time-varying prior.

        **Note**: Priors are treated as **probability distributions** (not
        unnormalized weights). They are normalized internally to sum to 1.0
        along the position axis before applying.
    axis : int, default=-1
        Axis along which to normalize.
    handle_degenerate : {"uniform", "nan", "raise"}, default="uniform"
        How to handle degenerate rows (all -inf or NaN):
        - "uniform": Return uniform distribution (1/n_bins per bin)
        - "nan": Return NaN for degenerate rows
        - "raise": Raise ValueError if any row is degenerate

    Returns
    -------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution. Each row sums to 1.0.

    Notes
    -----
    Implementation uses numerically stable log-sum-exp:

    ```python
    # Add log-prior to log-likelihood
    if prior is not None:
        prior = prior / prior.sum(axis=axis, keepdims=True)  # Normalize
        log_prior = np.log(np.clip(prior, 1e-10, 1.0))
        ll = log_likelihood + log_prior
    else:
        ll = log_likelihood

    # Log-sum-exp normalization (stable softmax)
    ll_max = ll.max(axis=axis, keepdims=True)
    ll_shifted = ll - ll_max                    # Shift to prevent overflow
    posterior = np.exp(ll_shifted)
    posterior /= posterior.sum(axis=axis, keepdims=True)
    ```

    For rows where all entries are -inf (e.g., no spikes and flat encoding):
    - ll_max will be -inf, ll_shifted will be NaN
    - These are detected and handled according to `handle_degenerate`
    """

def decode_position(
    env: Environment,
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    dt: float,
    *,
    prior: NDArray[np.float64] | None = None,
    method: Literal["poisson"] = "poisson",
    times: NDArray[np.float64] | None = None,
    validate: bool = False,
) -> DecodingResult:
    """Decode position from population spike counts.

    Main entry point for Bayesian decoding. Computes posterior probability
    distribution over positions for each time bin.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) for each neuron.
        Expected units: Hz (spikes/second). Typical values: 0-50 Hz.
        Very high rates (>100 Hz) may cause numerical issues.
    dt : float
        Time bin width in seconds. Typical values: 0.001-0.1s.
        Note: For typical firing rates, lambda*dt should be in [0, 5].
    prior : NDArray[np.float64] | None, default=None
        Prior probability over positions. If None, uses uniform prior.
        Shape (n_bins,) for stationary prior, (n_time_bins, n_bins) for
        time-varying prior. Normalized internally to sum to 1.0.
    method : {"poisson"}, default="poisson"
        Likelihood model. Currently only Poisson supported.
        Future: "gaussian", "clusterless".
    times : NDArray[np.float64] | None, default=None
        Time bin centers (seconds). If provided, stored in DecodingResult.
    validate : bool, default=False
        If True, run extra validation checks:
        - Verify posterior rows sum to 1.0 (within atol=1e-6)
        - Check for NaN/Inf in inputs and outputs
        - Warn if priors aren't properly normalized
        - Check encoding_models for extreme values
        This adds overhead but is useful for debugging.

    Returns
    -------
    DecodingResult
        Container with posterior, estimates, and metadata.

    Raises
    ------
    ValueError
        If validate=True and validation checks fail.

    Notes
    -----
    Memory usage: The posterior array is shape (n_time_bins, n_bins) and
    stored as float64. For long recordings (e.g., 1 hour at 25ms bins =
    144,000 time bins) with fine spatial resolution (e.g., 1000 bins),
    this requires ~1.1 GB. Consider processing in chunks for very long
    recordings, or using float32 dtype in future versions.
    """
```

#### Module: `_result.py`

```python
from functools import cached_property

@dataclass
class DecodingResult:
    """Container for Bayesian decoding results.

    Primary data (stored):
    - posterior: The full posterior distribution
    - env: Reference to environment for coordinate transforms
    - times: Optional time bin centers

    Derived properties (computed lazily on first access):
    - map_estimate: Bin index of maximum posterior
    - map_position: MAP position in environment coordinates
    - mean_position: Posterior mean position
    - uncertainty: Posterior entropy in bits

    Attributes
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over positions.
        Each row sums to 1.0 (normalized probability).
    env : Environment
        Reference to environment used for decoding.
    times : NDArray[np.float64] | None
        Time bin centers (seconds), if provided.

    Examples
    --------
    >>> result = decode_position(env, spike_counts, encoding_models, dt)
    >>> result.posterior.shape
    (1000, 400)  # 1000 time bins, 400 spatial bins
    >>> result.map_position[:5]  # First 5 MAP positions (lazy computed)
    array([[23.5, 45.2], [24.1, 44.8], ...])
    """
    posterior: NDArray[np.float64]
    env: Environment
    times: NDArray[np.float64] | None = None

    @cached_property
    def map_estimate(self) -> NDArray[np.int64]:
        """Maximum a posteriori bin index for each time bin."""
        return np.argmax(self.posterior, axis=1)

    @cached_property
    def map_position(self) -> NDArray[np.float64]:
        """MAP position in environment coordinates, shape (n_time_bins, n_dims)."""
        return self.env.bin_centers[self.map_estimate]

    @cached_property
    def mean_position(self) -> NDArray[np.float64]:
        """Posterior mean position (expected value), shape (n_time_bins, n_dims)."""
        return self.posterior @ self.env.bin_centers

    @cached_property
    def uncertainty(self) -> NDArray[np.float64]:
        """Posterior entropy in bits, shape (n_time_bins,).

        Higher values indicate more uncertain (spread out) posterior.
        Maximum is log2(n_bins) for uniform posterior.
        Minimum is 0 for delta distribution (all mass on one bin).

        Notes
        -----
        Uses mask-based computation to avoid bias from exact zeros:

        ```python
        p = np.clip(self.posterior, 0.0, 1.0)
        entropy = np.zeros(self.n_time_bins)
        for i, row in enumerate(p):
            mask = row > 0
            if mask.any():
                entropy[i] = -np.sum(row[mask] * np.log2(row[mask]))
        ```

        This is more accurate than global clipping to [1e-10, 1] which
        can slightly bias entropy upward when many exact zeros occur.
        """
        p = np.clip(self.posterior, 0.0, 1.0)
        # Vectorized mask-based entropy
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p = np.where(p > 0, np.log2(p), 0.0)
        return -np.sum(p * log_p, axis=1)

    @property
    def n_time_bins(self) -> int:
        """Number of time bins."""
        return self.posterior.shape[0]

    def plot(self, ax=None, **kwargs):
        """Plot posterior probability over time as heatmap."""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with times and position estimates."""
```

**Note**: `@dataclass` (not frozen) is used to allow `@cached_property`. The class is still effectively immutable since `posterior` and `env` are the only stored fields and modifying them would invalidate cached properties.

#### Module: `estimates.py`

**Naming convention**: Function names mirror `DecodingResult` property names for consistency.
Users can use either the result properties or these standalone functions interchangeably.

```python
def map_estimate(
    posterior: NDArray[np.float64],  # (n_time_bins, n_bins)
) -> NDArray[np.int64]:
    """Maximum a posteriori bin index for each time.

    Mirrors: DecodingResult.map_estimate
    """

def map_position(
    env: Environment,
    posterior: NDArray[np.float64],
) -> NDArray[np.float64]:
    """MAP position in environment coordinates.

    Mirrors: DecodingResult.map_position
    """

def mean_position(
    env: Environment,
    posterior: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Posterior mean position in environment coordinates.

    Mirrors: DecodingResult.mean_position
    """

def entropy(
    posterior: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Posterior entropy in bits (uncertainty measure).

    Mirrors: DecodingResult.uncertainty

    Note: Named `entropy` (not `uncertainty`) to be precise about the
    statistic being computed. DecodingResult uses `uncertainty` as a
    more user-friendly property name.
    """

def credible_region(
    env: Environment,
    posterior: NDArray[np.float64],
    level: float = 0.95,
) -> list[NDArray[np.int64]]:
    """Highest posterior density region containing specified probability mass."""
```

#### Module: `trajectory.py`

**Design decision**: Trajectory functions operate in **bin index space** (not environment coordinates) because:

1. The posterior is defined over discrete bins
2. For 1D linearized tracks, bin index directly corresponds to linear position
3. For 2D environments, bin-space trajectory analysis is more interpretable

For environment-coordinate results, users can map bin indices through `env.bin_centers[bin_indices]`.

```python
def fit_isotonic_trajectory(
    posterior: NDArray[np.float64],  # (n_time_bins, n_bins)
    times: NDArray[np.float64],      # (n_time_bins,)
    *,
    increasing: bool | None = None,
    method: Literal["map", "expected"] = "expected",
) -> IsotonicFitResult:
    """Fit monotonic trajectory to posterior using isotonic regression.

    Parameters
    ----------
    posterior : NDArray, shape (n_time_bins, n_bins)
        Posterior probability distribution.
    times : NDArray, shape (n_time_bins,)
        Time bin centers. Need not be uniformly spaced.
    increasing : bool, optional
        If True, fit increasing trajectory. If False, fit decreasing.
        If None (default), try both and return better fit.
    method : {"map", "expected"}, default="expected"
        How to extract position from posterior:
        - "map": Use argmax bin index
        - "expected": Use posterior-weighted mean bin index (smoother)

    Returns
    -------
    result : IsotonicFitResult
        Dataclass with:
        - fitted_positions: NDArray, shape (n_time_bins,) - bin indices
        - r_squared: float - coefficient of determination
        - direction: Literal["increasing", "decreasing"]
        - residuals: NDArray - for diagnostics

    Notes
    -----
    Uses `sklearn.isotonic.IsotonicRegression` internally.
    Time bins are NOT assumed uniformly spaced.
    """

def fit_linear_trajectory(
    env: Environment,
    posterior: NDArray[np.float64],  # (n_time_bins, n_bins)
    times: NDArray[np.float64],      # (n_time_bins,)
    *,
    n_samples: int = 1000,
    method: Literal["map", "sample"] = "sample",
    rng: np.random.Generator | int | None = None,
) -> LinearFitResult:
    """Fit linear trajectory to posterior.

    Parameters
    ----------
    env : Environment
        Spatial environment (for bin_centers if coordinate output needed).
    posterior : NDArray, shape (n_time_bins, n_bins)
        Posterior probability distribution.
    times : NDArray, shape (n_time_bins,)
        Time bin centers. Need not be uniformly spaced.
    n_samples : int, default=1000
        Number of Monte Carlo samples from posterior (for method="sample").
    method : {"map", "sample"}, default="sample"
        - "map": Fit line to MAP bin indices (fast, ignores uncertainty)
        - "sample": Sample positions from posterior, fit line, average
          coefficients (accounts for posterior uncertainty)
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility (method="sample" only).
        - If Generator: Use directly
        - If int: Seed for np.random.default_rng()
        - If None: Use default RNG (not reproducible)

    Returns
    -------
    result : LinearFitResult
        Dataclass with:
        - slope: float - bin indices per second
        - intercept: float - bin index at t=0
        - r_squared: float - coefficient of determination
        - slope_std: float - standard error of slope (method="sample" only)

    Notes
    -----
    Slope units are "bin indices per second". To convert to environment units:
        speed_cm_per_s = slope * env.bin_size  # for regular grids

    **Sampling implementation** (for method="sample"):

    Sampling is performed in cumulative-sum space to handle extremely
    peaky posteriors safely:

    ```python
    # Safe sampling even for very peaky posteriors
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    cumsum = np.cumsum(posterior, axis=1)
    u = rng.random((n_samples, n_time_bins, 1))
    samples = np.argmax(cumsum >= u, axis=-1)  # Shape: (n_samples, n_time_bins)
    ```

    This avoids numerical issues with np.random.choice on posteriors
    that have probabilities very close to 0 or 1.
    """

def detect_trajectory_radon(
    posterior: NDArray[np.float64],  # (n_time_bins, n_bins)
    *,
    theta_range: tuple[float, float] = (-90, 90),
    theta_step: float = 1.0,
) -> RadonDetectionResult:
    """Detect linear trajectory using Radon transform.

    Treats the posterior as a 2D image (time × bin index) and finds
    the line with maximum integrated probability mass.

    Parameters
    ----------
    posterior : NDArray, shape (n_time_bins, n_bins)
        Posterior probability distribution, treated as 2D image.
        Rows = time bins, Columns = spatial bins.
    theta_range : tuple of float, default=(-90, 90)
        Range of angles to search (degrees). 0° = horizontal (constant position),
        90° = vertical (instantaneous jump).
    theta_step : float, default=1.0
        Angular resolution in degrees.

    Returns
    -------
    result : RadonDetectionResult
        Dataclass with:
        - angle_degrees: float - detected trajectory angle
        - score: float - Radon transform value at peak (higher = better line fit)
        - offset: float - perpendicular offset from origin
        - sinogram: NDArray - full Radon transform (for visualization)

    Raises
    ------
    ImportError
        If scikit-image is not installed. Install with:
        `pip install scikit-image` or `pip install neurospatial[trajectory]`

    Notes
    -----
    The Radon transform integrates the posterior along all lines at each angle.
    The angle with highest integrated mass indicates the dominant trajectory.

    This is particularly useful for replay detection where the posterior
    forms a diagonal stripe pattern.

    Time is assumed uniformly spaced for image interpretation. For non-uniform
    times, consider interpolating the posterior first.
    """
```

**Result dataclasses**:

```python
@dataclass(frozen=True)
class IsotonicFitResult:
    fitted_positions: NDArray[np.float64]  # (n_time_bins,) bin indices
    r_squared: float
    direction: Literal["increasing", "decreasing"]
    residuals: NDArray[np.float64]

@dataclass(frozen=True)
class LinearFitResult:
    slope: float           # bin indices per second
    intercept: float       # bin index at t=0
    r_squared: float
    slope_std: float | None  # None for method="map"

@dataclass(frozen=True)
class RadonDetectionResult:
    angle_degrees: float
    score: float
    offset: float
    sinogram: NDArray[np.float64]  # (n_angles, n_offsets)
```

#### Module: `metrics.py`

```python
def decoding_error(
    decoded_positions: NDArray[np.float64],  # (n_time_bins, n_dims)
    actual_positions: NDArray[np.float64],   # (n_time_bins, n_dims)
    *,
    metric: Literal["euclidean", "graph"] = "euclidean",
    env: Environment | None = None,
) -> NDArray[np.float64]:
    """Compute position error for each time bin.

    Parameters
    ----------
    decoded_positions : NDArray, shape (n_time_bins, n_dims)
        Decoded position estimates (e.g., MAP or mean positions).
    actual_positions : NDArray, shape (n_time_bins, n_dims)
        Ground truth positions.
    metric : {"euclidean", "graph"}, default="euclidean"
        Distance metric:
        - "euclidean": Straight-line Euclidean distance
        - "graph": Shortest-path distance along environment graph
          (requires `env` parameter). Useful for mazes where Euclidean
          distance is misleading.
    env : Environment, optional
        Required when metric="graph". Used to compute graph distances.

    Returns
    -------
    errors : NDArray, shape (n_time_bins,)
        Distance between decoded and actual position at each time bin.
        Units match environment units (e.g., cm).

    Raises
    ------
    ValueError
        If metric="graph" but env is None.

    Notes
    -----
    NaN values in either input propagate to NaN in output.
    """

def median_decoding_error(
    decoded_positions: NDArray[np.float64],
    actual_positions: NDArray[np.float64],
) -> float:
    """Median Euclidean decoding error (ignoring NaN).

    Convenience function equivalent to:
        np.nanmedian(decoding_error(decoded, actual))

    Named to clearly indicate it's a specialization of `decoding_error`.
    """

def confusion_matrix(
    env: Environment,
    posterior: NDArray[np.float64],  # (n_time_bins, n_bins)
    actual_bins: NDArray[np.int64],  # (n_time_bins,)
    *,
    method: Literal["map", "expected"] = "map",
) -> NDArray[np.float64]:
    """Confusion matrix between decoded and actual bin indices.

    Parameters
    ----------
    env : Environment
        Spatial environment (used for n_bins).
    posterior : NDArray, shape (n_time_bins, n_bins)
        Posterior probability distribution.
    actual_bins : NDArray, shape (n_time_bins,)
        Ground truth bin indices.
    method : {"map", "expected"}, default="map"
        How to summarize the posterior for each time bin:
        - "map": Use argmax (most likely bin). Returns integer counts.
        - "expected": Accumulate full posterior mass. Cell (i, j) contains
          sum of P(decoded=j | actual=i) across all time bins where
          actual_bins == i. Rows sum to count of actual bin occurrences.

    Returns
    -------
    cm : NDArray, shape (n_bins, n_bins)
        Confusion matrix. Rows are actual bins, columns are decoded bins.
    """

def decoding_correlation(
    decoded_positions: NDArray[np.float64],  # (n_time_bins, n_dims)
    actual_positions: NDArray[np.float64],   # (n_time_bins, n_dims)
    weights: NDArray[np.float64] | None = None,  # (n_time_bins,)
) -> float:
    """Weighted Pearson correlation between decoded and actual positions.

    Named `decoding_correlation` (not generic `weighted_correlation`) to
    clarify its purpose in the decoding context.

    Parameters
    ----------
    decoded_positions : NDArray, shape (n_time_bins, n_dims)
        Decoded position estimates.
    actual_positions : NDArray, shape (n_time_bins, n_dims)
        Ground truth positions.
    weights : NDArray, shape (n_time_bins,), optional
        Per-time-bin weights. If None, uniform weights (standard correlation).
        Typical use: weight by posterior certainty (1 - entropy) to
        down-weight uncertain time bins.

    Returns
    -------
    r : float
        Weighted Pearson correlation coefficient.
        For n_dims > 1, returns mean correlation across dimensions.

    Notes
    -----
    Time bins with NaN in positions or zero weight are excluded.
    Returns NaN if fewer than 2 valid time bins remain.

    **Numerically stable implementation**:

    Uses centered formulas with np.average to reduce catastrophic
    cancellation, especially important for long time series with
    large weights:

    ```python
    # Normalize weights
    w = weights / weights.sum()

    # Weighted means (stable)
    mean_x = np.average(x, weights=w)
    mean_y = np.average(y, weights=w)

    # Center the data (reduces cancellation)
    x_centered = x - mean_x
    y_centered = y - mean_y

    # Weighted covariance and variances
    cov_xy = np.sum(w * x_centered * y_centered)
    var_x = np.sum(w * x_centered ** 2)
    var_y = np.sum(w * y_centered ** 2)

    # Correlation with safety check
    denom = np.sqrt(var_x * var_y)
    r = cov_xy / denom if denom > 0 else np.nan
    ```

    This approach is more stable than the direct formula
    (sum(wx*y) - n*mean_x*mean_y) / ... which can suffer from
    cancellation when sums are large.
    """
```

#### Module: `shuffle.py`

**Purpose**: Statistical shuffling procedures to establish null distributions and test
the significance of decoded sequences. These methods are essential for replay analysis
to rule out non-specific factors like firing rate biases.

**Design principles**:

1. **Generators, not accumulators**: Functions yield shuffled data one at a time for memory efficiency
2. **Reproducibility**: All functions accept `rng` parameter for reproducible results
3. **Clear naming**: Function names indicate what is being shuffled
4. **Composable**: Shuffles can be combined in analysis pipelines

**Shuffle Categories**:

| Category | Null Hypothesis Tested |
|----------|----------------------|
| **Temporal** | Sequential structure is not significant |
| **Cell Identity** | Spatial code coherence is not significant |
| **Posterior** | Trajectory detection is not biased |
| **Surrogate** | Structure exceeds rate-based expectations |

```python
# =============================================================================
# I. Temporal Order Shuffles - Test sequential structure within events
# =============================================================================

def shuffle_time_bins(
    spike_counts: NDArray[np.int64],  # (n_time_bins, n_neurons)
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.int64], None, None]:
    """Shuffle temporal order of time bins within an event.

    **Primary test for sequential structure.** Disrupts temporal order while
    preserving instantaneous firing characteristics.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts for a single candidate event (PBE/SWR).
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

    Yields
    ------
    shuffled_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts with time bins in random order.

    Notes
    -----
    - Preserves spike counts per neuron per time bin
    - Preserves total spikes per neuron across event
    - Destroys temporal sequence information
    - Underpowered with very few time bins (<5)

    Example
    -------
    >>> for shuffled in shuffle_time_bins(spike_counts, n_shuffles=100, rng=42):
    ...     result = decode_position(env, shuffled, encoding_models, dt)
    ...     null_scores.append(compute_sequence_score(result))
    """

def shuffle_time_bins_coherent(
    spike_counts: NDArray[np.int64],  # (n_time_bins, n_neurons)
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.int64], None, None]:
    """Shuffle time bins coherently across all neurons (time-swap shuffle).

    Preserves population co-activation structure while disrupting temporal order.
    Less conservative than `shuffle_time_bins` because pairwise correlations
    are maintained.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts for a single candidate event.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

    Yields
    ------
    shuffled_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts with rows (time bins) permuted coherently.

    Notes
    -----
    - All neurons see the same temporal permutation
    - Preserves instantaneous population vectors
    - Destroys temporal progression but keeps co-firing structure
    """

def jitter_spike_times(
    spike_times: list[NDArray[np.float64]],  # List of spike time arrays per neuron
    event_start: float,
    event_end: float,
    *,
    jitter_range: tuple[float, float] = (0.005, None),  # (min_jitter, max_jitter)
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[list[NDArray[np.float64]], None, None]:
    """Jitter spike times within event window.

    Disrupts precise spike timing while preserving event window and spike count.

    Parameters
    ----------
    spike_times : list of NDArray[np.float64]
        Spike times for each neuron (n_neurons arrays).
    event_start : float
        Event start time (seconds).
    event_end : float
        Event end time (seconds).
    jitter_range : tuple of float, default=(0.005, None)
        (min_jitter, max_jitter) in seconds. If max_jitter is None,
        defaults to (event_end - event_start - min_jitter).
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

    Yields
    ------
    jittered_spikes : list of NDArray[np.float64]
        Spike times with each spike independently jittered.

    Notes
    -----
    - Each spike is independently reassigned within [event_start, event_end]
    - Preserves spike count per neuron
    - Destroys precise temporal relationships
    """


# =============================================================================
# II. Cell Identity Shuffles - Test spatial code coherence
# =============================================================================

def shuffle_cell_identity(
    spike_counts: NDArray[np.int64],      # (n_time_bins, n_neurons)
    encoding_models: NDArray[np.float64], # (n_neurons, n_bins)
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[tuple[NDArray[np.int64], NDArray[np.float64]], None, None]:
    """Shuffle mapping between spike trains and place fields.

    **Primary test for spatial code coherence.** Disrupts the learned relationship
    between a neuron's activity and its encoded spatial location.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) for each neuron.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

    Yields
    ------
    shuffled_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts with neuron identities permuted.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Original encoding models (unchanged).

    Notes
    -----
    - Randomly permutes columns of spike_counts (neuron axis)
    - Encoding models remain fixed
    - Equivalent to randomly reassigning which place field goes with which spikes
    - Caution: Can introduce noise if firing rates differ greatly between cells

    Alternative implementation (equivalent):
        Instead of shuffling spike_counts columns, shuffle encoding_models rows.
        This yields the same decoded posteriors.
    """

def shuffle_place_fields_circular(
    encoding_models: NDArray[np.float64],  # (n_neurons, n_bins)
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Circularly shift each place field by a random amount.

    **Conservative null hypothesis.** Preserves individual cell spiking properties
    and local place field structure while disrupting spatial relationships.

    Parameters
    ----------
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) for each neuron.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

    Yields
    ------
    shuffled_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Place fields with each row circularly shifted by random amount.

    Notes
    -----
    - Each neuron's place field is shifted independently
    - Preserves the shape of each place field
    - Destroys spatial relationships between neurons
    - More conservative than cell identity shuffle
    - For 2D environments: consider `shuffle_place_fields_circular_2d`
    """

def shuffle_place_fields_circular_2d(
    encoding_models: NDArray[np.float64],  # (n_neurons, n_bins)
    env: Environment,
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Circularly shift 2D place fields in both dimensions.

    For 2D environments, shifts place fields in both x and y dimensions.

    Parameters
    ----------
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) for each neuron.
    env : Environment
        2D environment with grid layout (provides grid_shape).
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

    Yields
    ------
    shuffled_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Place fields with 2D circular shifts applied.

    Raises
    ------
    ValueError
        If environment is not 2D or doesn't have grid layout.
    """


# =============================================================================
# III. Posterior/Position Shuffles - Test trajectory detection
# =============================================================================

def shuffle_posterior_circular(
    posterior: NDArray[np.float64],  # (n_time_bins, n_bins)
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Circularly shift posterior at each time bin independently.

    Controls for chance linear alignment of position estimates by disrupting
    trajectory progression while preserving local smoothness.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution from decoding.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

    Yields
    ------
    shuffled_posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior with each row circularly shifted by random amount.

    Notes
    -----
    - Each time bin is shifted independently
    - Preserves the shape of each instantaneous posterior
    - Destroys temporal continuity of decoded positions
    - Caution: Can generate position representations that don't exist in
      original data (edge effects near track boundaries)
    """

def shuffle_posterior_weighted_circular(
    posterior: NDArray[np.float64],  # (n_time_bins, n_bins)
    *,
    edge_buffer: int = 5,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Weighted circular shift with edge effect mitigation.

    Refined version of posterior shuffle that maintains non-uniformity and
    reduces edge effects by restricting shifts near track boundaries.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution from decoding.
    edge_buffer : int, default=5
        Number of bins from track ends where shifts are restricted.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

    Yields
    ------
    shuffled_posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior with weighted circular shifts applied.

    Notes
    -----
    - More conservative than standard circular shuffle
    - Shifts are weighted by mean decoded posteriors across track
    - Restricted shifting near track ends mitigates edge effects
    """


# =============================================================================
# IV. Surrogate Data Generation - Test against rate-based null
# =============================================================================

def generate_poisson_surrogates(
    spike_counts: NDArray[np.int64],  # (n_time_bins, n_neurons)
    dt: float,
    *,
    n_surrogates: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.int64], None, None]:
    """Generate Poisson surrogate spike trains based on mean firing rates.

    Creates null distribution based on firing rates, removing all temporal
    structure beyond that expected by random chance (homogeneous Poisson).

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Original spike counts.
    dt : float
        Time bin width in seconds.
    n_surrogates : int, default=1000
        Number of surrogate versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

    Yields
    ------
    surrogate_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Poisson-generated spike counts based on mean rates.

    Notes
    -----
    - Computes mean firing rate per neuron across event
    - Generates new spike counts from Poisson(rate * dt) distribution
    - Preserves mean rates but removes all temporal structure
    - Tests null: sequential structure is solely due to firing rate inhomogeneity
    """

def generate_inhomogeneous_poisson_surrogates(
    spike_counts: NDArray[np.int64],  # (n_time_bins, n_neurons)
    dt: float,
    *,
    smoothing_window: int = 3,
    n_surrogates: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.int64], None, None]:
    """Generate inhomogeneous Poisson surrogates preserving rate modulation.

    More conservative than homogeneous Poisson - preserves slow rate
    fluctuations while removing fine temporal structure.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Original spike counts.
    dt : float
        Time bin width in seconds.
    smoothing_window : int, default=3
        Window size for smoothing instantaneous rates.
    n_surrogates : int, default=1000
        Number of surrogate versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

    Yields
    ------
    surrogate_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Poisson-generated spike counts with time-varying rates.
    """


# =============================================================================
# V. Cross-Event Shuffles - Test event-specific structure
# =============================================================================

def shuffle_across_events(
    posteriors: list[NDArray[np.float64]],  # List of (n_time_bins_i, n_bins) arrays
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[list[NDArray[np.float64]], None, None]:
    """Create pseudoevents by drawing time bins from different events.

    Controls for bias toward particular locations by the decoding estimator.
    Constructs artificial events by randomly drawing posterior time bins from
    the pool of all candidate events.

    Parameters
    ----------
    posteriors : list of NDArray[np.float64]
        List of posterior distributions, one per candidate event.
        Each array has shape (n_time_bins_i, n_bins).
    n_shuffles : int, default=1000
        Number of shuffled sets to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

    Yields
    ------
    shuffled_posteriors : list of NDArray[np.float64]
        Pseudoevents with same structure but mixed time bins.

    Notes
    -----
    - Preserves number of events and time bins per event
    - Destroys within-event sequential structure
    - Tests if observed patterns are specific to events or general to dataset
    """


# =============================================================================
# Convenience Functions for Significance Testing
# =============================================================================

def compute_shuffle_pvalue(
    observed_score: float,
    null_scores: NDArray[np.float64],
    *,
    tail: Literal["greater", "less", "two-sided"] = "greater",
) -> float:
    """Compute p-value from observed score and null distribution.

    Parameters
    ----------
    observed_score : float
        Score computed from actual data.
    null_scores : NDArray[np.float64], shape (n_shuffles,)
        Scores computed from shuffled data.
    tail : {"greater", "less", "two-sided"}, default="greater"
        - "greater": P(null >= observed), for testing if observed is high
        - "less": P(null <= observed), for testing if observed is low
        - "two-sided": 2 * min(P(null >= obs), P(null <= obs))

    Returns
    -------
    p_value : float
        Proportion of null scores at least as extreme as observed.
        Minimum value is 1/(n_shuffles + 1) for Monte Carlo correction.

    Notes
    -----
    Uses Monte Carlo correction: p = (k + 1) / (n + 1) where k is the
    number of null scores at least as extreme as observed, and n is the
    total number of shuffles. This ensures p > 0 even if observed exceeds
    all null values.
    """

def compute_shuffle_zscore(
    observed_score: float,
    null_scores: NDArray[np.float64],
) -> float:
    """Compute z-score of observed relative to null distribution.

    Parameters
    ----------
    observed_score : float
        Score computed from actual data.
    null_scores : NDArray[np.float64], shape (n_shuffles,)
        Scores computed from shuffled data.

    Returns
    -------
    z_score : float
        (observed - null_mean) / null_std.
        Returns NaN if null_std == 0.
    """


@dataclass(frozen=True)
class ShuffleTestResult:
    """Result of a shuffle-based significance test.

    Attributes
    ----------
    observed_score : float
        Score computed from actual data.
    null_scores : NDArray[np.float64]
        Scores from shuffled data.
    p_value : float
        P-value (proportion of null >= observed).
    z_score : float
        Z-score relative to null distribution.
    shuffle_type : str
        Name of shuffle method used.
    n_shuffles : int
        Number of shuffles performed.
    """
    observed_score: float
    null_scores: NDArray[np.float64]
    p_value: float
    z_score: float
    shuffle_type: str
    n_shuffles: int

    @property
    def is_significant(self) -> bool:
        """Whether p < 0.05."""
        return self.p_value < 0.05

    def plot(self, ax=None, **kwargs):
        """Plot null distribution with observed score marked."""
```

**Design Notes**:

1. **Generator pattern**: All shuffle functions are generators that yield one shuffled
   version at a time. This is memory-efficient for large n_shuffles and allows early
   stopping if significance is already clear.

2. **Reproducibility**: All functions accept `rng` parameter. Pass an integer seed or
   `np.random.Generator` for reproducible results.

3. **Composable workflow**:

   ```python
   # Typical usage pattern
   observed = compute_sequence_score(decode_position(env, spikes, models, dt))

   null_scores = []
   for shuffled_spikes in shuffle_time_bins(spikes, n_shuffles=1000, rng=42):
       result = decode_position(env, shuffled_spikes, models, dt)
       null_scores.append(compute_sequence_score(result))

   p_value = compute_shuffle_pvalue(observed, np.array(null_scores))
   ```

4. **Choosing the right shuffle**:

   | Question | Shuffle to Use |
   |----------|---------------|
   | Is there significant sequence structure? | `shuffle_time_bins` |
   | Is the spatial code coherent? | `shuffle_cell_identity` |
   | Is trajectory detection biased? | `shuffle_posterior_circular` |
   | Does structure exceed rate expectations? | `generate_poisson_surrogates` |
   | Are patterns event-specific? | `shuffle_across_events` |

### Public API (decoding/**init**.py)

```python
from neurospatial.decoding._result import DecodingResult
from neurospatial.decoding.likelihood import (
    log_poisson_likelihood,
    poisson_likelihood,
)
from neurospatial.decoding.posterior import (
    decode_position,
    normalize_to_posterior,
)
from neurospatial.decoding.estimates import (
    credible_region,
    entropy,
    map_estimate,
    map_position,
    mean_position,
)
from neurospatial.decoding.trajectory import (
    IsotonicFitResult,
    LinearFitResult,
    RadonDetectionResult,
    detect_trajectory_radon,
    fit_isotonic_trajectory,
    fit_linear_trajectory,
)
from neurospatial.decoding.metrics import (
    confusion_matrix,
    decoding_correlation,
    decoding_error,
    median_decoding_error,
)
from neurospatial.decoding.shuffle import (
    # Temporal shuffles
    shuffle_time_bins,
    shuffle_time_bins_coherent,
    jitter_spike_times,
    # Cell identity shuffles
    shuffle_cell_identity,
    shuffle_place_fields_circular,
    shuffle_place_fields_circular_2d,
    # Posterior shuffles
    shuffle_posterior_circular,
    shuffle_posterior_weighted_circular,
    # Surrogates
    generate_poisson_surrogates,
    generate_inhomogeneous_poisson_surrogates,
    # Cross-event shuffles
    shuffle_across_events,
    # Significance testing
    compute_shuffle_pvalue,
    compute_shuffle_zscore,
    ShuffleTestResult,
)

__all__ = [
    # Result containers
    "DecodingResult",
    "IsotonicFitResult",
    "LinearFitResult",
    "RadonDetectionResult",
    # Main entry point
    "decode_position",
    # Likelihood
    "log_poisson_likelihood",
    "poisson_likelihood",
    # Posterior
    "normalize_to_posterior",
    # Estimates (names mirror DecodingResult properties)
    "credible_region",
    "entropy",
    "map_estimate",
    "map_position",
    "mean_position",
    # Trajectory
    "detect_trajectory_radon",
    "fit_isotonic_trajectory",
    "fit_linear_trajectory",
    # Metrics
    "confusion_matrix",
    "decoding_correlation",
    "decoding_error",
    "median_decoding_error",
    # Shuffles - Temporal
    "shuffle_time_bins",
    "shuffle_time_bins_coherent",
    "jitter_spike_times",
    # Shuffles - Cell Identity
    "shuffle_cell_identity",
    "shuffle_place_fields_circular",
    "shuffle_place_fields_circular_2d",
    # Shuffles - Posterior
    "shuffle_posterior_circular",
    "shuffle_posterior_weighted_circular",
    # Shuffles - Surrogates
    "generate_poisson_surrogates",
    "generate_inhomogeneous_poisson_surrogates",
    # Shuffles - Cross-event
    "shuffle_across_events",
    # Shuffles - Significance
    "compute_shuffle_pvalue",
    "compute_shuffle_zscore",
    "ShuffleTestResult",
]
```

### Integration with Main Package

Add to `src/neurospatial/__init__.py`:

```python
# Decoding (Bayesian population analysis)
from neurospatial.decoding import (
    DecodingResult,
    decode_position,
    decoding_error,
    median_decoding_error,
)
```

Export only the most common functions at the top level; power users can import from `neurospatial.decoding` directly.

## Implementation Order

### Phase 1: Core Decoding (Priority: High)

- `_result.py` - DecodingResult dataclass
- `likelihood.py` - Poisson likelihood functions
- `posterior.py` - normalize_to_posterior, decode_position
- `estimates.py` - MAP, mean, entropy functions
- Tests for Phase 1

### Phase 2: Quality Metrics (Priority: Medium)

- `metrics.py` - decoding_error, confusion_matrix, decoding_correlation
- Tests for Phase 2

### Phase 3: Trajectory Analysis (Priority: Medium)

- `trajectory.py` - isotonic/linear regression, Radon transform
- Tests for Phase 3

### Phase 4: Shuffle-Based Significance Testing (Priority: Medium)

- `shuffle.py` - Core temporal shuffles (shuffle_time_bins, shuffle_time_bins_coherent)
- `shuffle.py` - Cell identity shuffles (shuffle_cell_identity, shuffle_place_fields_circular)
- `shuffle.py` - Posterior shuffles (shuffle_posterior_circular)
- `shuffle.py` - Surrogate generation (generate_poisson_surrogates)
- `shuffle.py` - Convenience functions (compute_shuffle_pvalue, ShuffleTestResult)
- Tests for Phase 4

### Phase 5: Extensions (Priority: Low, Future)

- Clusterless decoding (mark-based likelihood)
- Gaussian likelihood model
- State-space smoothing (Kalman filter)
- GPU acceleration (JAX/CuPy)
- Additional shuffles (jitter_spike_times, shuffle_across_events, inhomogeneous Poisson)

## Example Usage

```python
import numpy as np
from neurospatial import Environment, compute_place_field, decode_position

# Setup
positions = np.random.uniform(0, 100, (10000, 2))
times = np.linspace(0, 100, 10000)
env = Environment.from_samples(positions, bin_size=5.0)

# Compute encoding models (place fields)
spike_times_list = [np.random.uniform(0, 100, 50) for _ in range(20)]
encoding_models = np.array([
    compute_place_field(env, spikes, times, positions, bandwidth=8.0)
    for spikes in spike_times_list
])

# Bin spikes for decoding
dt = 0.025  # 25 ms bins
time_bins = np.arange(0, 100, dt)
spike_counts = np.zeros((len(time_bins) - 1, len(spike_times_list)), dtype=np.int64)
for i, spikes in enumerate(spike_times_list):
    spike_counts[:, i], _ = np.histogram(spikes, bins=time_bins)

# Decode position
result = decode_position(
    env, spike_counts, encoding_models, dt,
    prior=None,  # Uniform prior
)

# Access results
print(f"Posterior shape: {result.posterior.shape}")  # (n_time_bins, n_bins)
print(f"MAP position: {result.map_position[:5]}")    # First 5 decoded positions
print(f"Mean uncertainty: {result.uncertainty.mean():.2f} bits")

# Visualize
result.plot()

# Compute decoding error
from neurospatial.decoding import decoding_error, median_decoding_error
actual_positions = positions[::int(dt * 100)]  # Subsample to match time bins
errors = decoding_error(result.map_position, actual_positions[:len(result.map_position)])
print(f"Median error: {median_decoding_error(result.map_position, actual_positions[:len(result.map_position)]):.1f} cm")

# Use standalone estimate functions (equivalent to result properties)
from neurospatial.decoding import map_position, mean_position, entropy
map_pos = map_position(env, result.posterior)  # Same as result.map_position
mean_pos = mean_position(env, result.posterior)  # Same as result.mean_position
ent = entropy(result.posterior)  # Same as result.uncertainty
```

## Dependencies

### Required (already in neurospatial)

- `numpy` - Array operations
- `scipy` - Isotonic regression (`scipy.optimize`), sparse matrices
- `scikit-learn` - Isotonic regression alternative (`sklearn.isotonic`)

### Optional Dependencies

| Feature | Package | Install Extra |
|---------|---------|---------------|
| Radon transform | `scikit-image` | `pip install neurospatial[trajectory]` |

### Error Handling Strategy for Optional Dependencies

Use **import-time guards with clear error messages** (consistent with neurospatial patterns):

```python
# In trajectory.py
_SKIMAGE_AVAILABLE = False
try:
    from skimage.transform import radon
    _SKIMAGE_AVAILABLE = True
except ImportError:
    pass


def detect_trajectory_radon(posterior, **kwargs):
    """Detect linear trajectory using Radon transform."""
    if not _SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for Radon transform trajectory detection. "
            "Install with: pip install scikit-image\n"
            "Or install neurospatial with trajectory extras: "
            "pip install neurospatial[trajectory]"
        )
    # ... implementation
```

**Rationale**:

- Clear error message at point of use (not import time)
- Tells user exactly how to fix it
- Module still importable for other functions
- Consistent with how neurospatial handles NWB optional dependencies

## Testing Strategy

1. **Unit Tests**: Each function tested independently
2. **Integration Tests**: Full decode_position pipeline
3. **Regression Tests**: Compare output to replay_trajectory_classification
4. **Property Tests**: Posterior sums to 1, entropy bounds, etc.
5. **Performance Tests**: Benchmark against reference implementation

## Documentation

1. NumPy docstrings for all public functions
2. Example notebook: `examples/bayesian_decoding.ipynb`
3. API reference in docs
4. CLAUDE.md updates with decoding quick reference

## Numerical Stability Considerations

This section summarizes the numerical stability measures implemented throughout the decoding subpackage.

### Log-Domain Computation

All likelihood computations are performed in log-domain to prevent underflow:

| Operation | Implementation | Why |
|-----------|---------------|-----|
| Poisson likelihood | `log_poisson_likelihood()` (primary) | Prevents underflow with large populations |
| Prior application | `log_likelihood + log_prior` | Additive in log-domain |
| Normalization | Log-sum-exp trick | Stable softmax |
| Entropy | Mask-based with `np.where` | Avoids log(0) |

### Degenerate Case Handling

| Situation | Location | Handling |
|-----------|----------|----------|
| All log-likelihoods = -inf | `normalize_to_posterior` | Configurable: uniform/nan/raise |
| Zero probability bins | Entropy calculation | Mask-based: exclude from sum |
| No spikes in time bin | `log_poisson_likelihood` | Returns valid log-likelihood (rate-only term) |
| Extreme firing rates | Documentation | Warn if lambda*dt >> 10 |

### Reproducibility

| Function | Parameter | Usage |
|----------|-----------|-------|
| `fit_linear_trajectory` | `rng` | Pass `np.random.Generator` or seed |
| `fit_isotonic_trajectory` | N/A | Deterministic |
| `detect_trajectory_radon` | N/A | Deterministic |

### Memory Considerations

The posterior array is the primary memory consumer:

```python
# Memory = n_time_bins × n_bins × 8 bytes (float64)
# Example: 1 hour at 25ms bins, 1000 spatial bins
# = 144,000 × 1000 × 8 = 1.15 GB
```

**Recommendations**:

- For very long recordings, process in chunks
- Consider float32 for memory-critical applications (future enhancement)
- Use `validate=False` (default) for production to avoid overhead

### Validation Mode

When `validate=True` is passed to `decode_position`:

```python
# Checks performed:
1. Posterior rows sum to 1.0 ± atol=1e-6
2. No NaN/Inf in inputs (spike_counts, encoding_models, prior)
3. No NaN/Inf in outputs (posterior)
4. Prior is properly normalized (warns if not)
5. encoding_models values are in reasonable range (warns if >100 Hz)
```

This is useful for debugging but adds overhead; use `validate=False` for production.

### Summary of Numerical Patterns

**Always do**:

- Use `log_poisson_likelihood` + `normalize_to_posterior` (not raw likelihoods)
- Normalize priors before applying (done internally)
- Use mask-based entropy computation
- Provide `rng` parameter for reproducible sampling

**Avoid**:

- Direct `np.exp` of log-likelihoods
- Global clipping that biases results (e.g., `np.clip(p, 1e-10, 1)` for entropy)
- Non-centered weighted correlation formulas
- Assuming posteriors sum exactly to 1.0 (check with tolerance)

## Open Questions

1. **Should DecodingResult be frozen?** Frozen is consistent with DirectionalPlaceFields, prevents accidental mutation. Recommend: Yes.

2. **Radon transform dependency?** scikit-image required for Radon. Options:
   - Make trajectory.py optional (import guard)
   - Implement simplified Radon ourselves
   - Recommend: Optional dependency with graceful fallback

3. **NWB integration?** Should we add write_decoded_position() to nwb module?
   - Recommend: Yes, in follow-up PR after core decoding

4. **Clusterless decoding priority?** Mark-based likelihood is complex.
   - Recommend: Phase 4, after sorted spike decoding is stable

## Summary

This plan creates a `neurospatial.decoding` subpackage following existing patterns:

- Stateless functions (not classes)
- Environment as first argument
- NumPy docstrings
- Immutable result container
- Gradual complexity (core → metrics → trajectory → extensions)

The implementation leverages existing infrastructure (Environment, compute_place_field) while adding population-level analysis capabilities for Bayesian neural decoding.
