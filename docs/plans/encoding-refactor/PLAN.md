# Encoding Module Refactor Plan

**Committed**: 2025-12-05

> **Superseded for current API vocabulary.** This historical plan predates the
> v0.6 naming contract. Do not treat names such as `neuron_id`,
> `to_dataframe(neuron_ids=...)`, `detect_hd_cells`, `detect_view_cells`, or
> `peak_view_x` / `peak_view_y` as current guidance. Use
> `.claude/docs/plans/ux-v0.6/api-contract.md` for the active API contract.

## Overview

Refactor the `neurospatial.encoding` module to provide a consistent, user-friendly API optimized for batch processing of neural populations.

### Design Principles

1. **Consistent naming**: `compute_*_rate` pattern across all cell types
2. **Rich results**: Return dataclasses with metadata and convenience methods
3. **Batch-first**: Plural functions (`compute_spatial_rates`) for populations
4. **Metrics on results**: Grid score, border score, etc. are methods on spatial rate results
5. **Domain-appropriate**: Head direction doesn't force a spatial `Environment`
6. **Backend-flexible core**: Separate binning/IO from pure array math so we can support both NumPy (default) and JAX backends
7. **JAX-ready math**: Core rate and metric operations use array-only computations with shapes friendly to `vmap` (e.g., `(n_neurons, n_bins)`)
8. **Optional acceleration**: JAX is an optional backend (Linux/macOS only); NumPy remains the default and works everywhere, including Windows

---

## New API

### Function Naming

| Cell Type | Single Neuron | Multiple Neurons |
|-----------|---------------|------------------|
| Place/Grid/Border | `compute_spatial_rate()` | `compute_spatial_rates()` |
| Head direction | `compute_directional_rate()` | `compute_directional_rates()` |
| Spatial view | `compute_view_rate()` | `compute_view_rates()` |
| Object vector | `compute_egocentric_rate()` | `compute_egocentric_rates()` |

### Result Classes

All result classes use consistent field naming: `.firing_rate` for single neurons, `.firing_rates` for populations.

| Function | Returns | Key Fields |
|----------|---------|------------|
| `compute_spatial_rate` | `SpatialRateResult` | `.firing_rate`, `.occupancy`, `.env` |
| `compute_spatial_rates` | `SpatialRatesResult` | `.firing_rates`, `.occupancy`, `.env` |
| `compute_directional_rate` | `DirectionalRateResult` | `.firing_rate`, `.occupancy`, `.bin_centers` |
| `compute_directional_rates` | `DirectionalRatesResult` | `.firing_rates`, `.occupancy`, `.bin_centers` |
| `compute_view_rate` | `ViewRateResult` | `.firing_rate`, `.view_occupancy`, `.env` |
| `compute_view_rates` | `ViewRatesResult` | `.firing_rates`, `.view_occupancy`, `.env` |
| `compute_egocentric_rate` | `EgocentricRateResult` | `.firing_rate`, `.occupancy`, `.ego_env` |
| `compute_egocentric_rates` | `EgocentricRatesResult` | `.firing_rates`, `.occupancy`, `.ego_env` |

**Note**: Current code uses `.field` (e.g., `ObjectVectorFieldResult.field`). This refactor standardizes to `.firing_rate` for clarity.

---

## Backends and Performance

We separate encoding into two layers:

### 1. Binning Layer (CPU, joblib)

- **Input**: Spike trains as list of 1D arrays (one per neuron), `times`, positions/headings, `Environment`
- **Output**: Dense arrays: `spike_counts` with shape `(n_neurons, n_bins)` and `occupancy` with shape `(n_bins,)`
- Parallelized with joblib over neurons and/or time chunks

**Note on `n_bins`**: For all result types, `n_bins` refers to the **flattened** number of active bins in the environment. For egocentric polar environments, the underlying grid may be `(n_distance_bins, n_direction_bins)`, but arrays are always flattened to `(n_bins,)` for consistency. The `ego_env` stores the shape information needed for reshaping/plotting.

### 2. Core Rate/Metrics Layer (NumPy now, JAX later)

- **Input**: `(spike_counts, occupancy, env / bin_centers, ...)` as dense arrays
- **Output**: Rate maps and metrics, all implemented as pure array ops on shapes like `(n_neurons, n_bins)`
- **Backend argument**: `backend={"numpy", "jax", "auto"}` with:
  - `"numpy"` (default) → works everywhere, including Windows
  - `"jax"` → requires JAX; raises clear error if not available
  - `"auto"` → use JAX if installed and supported; on Windows, `"auto"` silently uses NumPy

Result objects (`SpatialRateResult`, `SpatialRatesResult`, etc.) hold array-like data that may be NumPy or JAX arrays. `ArrayLike` fields are treated opaquely—we avoid Python-side branching on array type. All result class methods use `_to_numpy()` to convert arrays transparently before operations that require NumPy (plotting, argmax, etc.).

### 3. Array Conversion and Method Classification

Result class methods fall into two categories:

**Host-only methods** (convenience, post-hoc analysis):

- `plot()`, `to_dataframe()`, `peak_location()`, `interpretation()`
- These explicitly convert to NumPy via `_to_numpy()` and are not JAX-traced
- Documented as "returns to host" in docstrings

**Backend-aware methods** (core math, JAX-friendly):

- `spatial_information()`, `sparsity()`, `grid_scores()`, `border_scores()`, `classify()`
- Route through `_metrics.py` / `_core_*` modules that dispatch to NumPy or JAX
- Preserve array type (NumPy in → NumPy out, JAX in → JAX out)
- Compatible with `jax.vmap` when using JAX backend
- Use `xp = _get_array_module(firing_rates)` then `xp.nanmax(...)` for backend dispatch

**Static data in JAX context**:

- `env.bin_centers` is always NumPy (part of Environment, not computed per-call)
- When JAX-tracing, treat `env.bin_centers` as a static constant (OK for indexing results)
- Methods that index into `bin_centers` (like `peak_locations`) are host-only anyway

```python
# In encoding/_base.py
def _to_numpy(arr: ArrayLike) -> NDArray:
    """Convert array to NumPy for host-only operations.

    Use this for plotting, DataFrame export, and other convenience methods
    that are not part of the JAX-traced compute graph.
    """
    return np.asarray(arr)


def _get_array_module(arr: ArrayLike):
    """Get the array module (numpy or jax.numpy) for backend-aware operations."""
    if hasattr(arr, '__jax_array__'):
        import jax.numpy as jnp
        return jnp
    return np
```

---

## Detailed Design

### 1. Spatial Rate (Place/Grid/Border Cells)

**File**: `encoding/spatial.py`

```python
@dataclass(frozen=True)
class SpatialRateResult(SpatialResultMixin):
    """Result of spatial rate computation for single neuron.

    Inherits from SpatialResultMixin for peak_locations() and peak_firing_rates().
    """

    firing_rate: ArrayLike              # (n_bins,) firing rate in Hz, NumPy or JAX
    occupancy: ArrayLike                # (n_bins,) seconds in bin, NumPy or JAX
    env: Environment
    smoothing_method: str
    bandwidth: float

    # Convenience methods
    def plot(self, ax=None, **kwargs):
        """Plot the rate map."""
        return self.env.plot_field(self.firing_rate, ax=ax, **kwargs)

    # NOTE: peak_locations() inherited from SpatialResultMixin (returns n_dims for single)
    # NOTE: peak_firing_rates() inherited from SpatialResultMixin (returns scalar for single)

    # Alias for backwards compatibility with single-neuron API
    def peak_location(self) -> NDArray[np.float64]:
        """Coordinates of peak firing rate. Alias for peak_locations()."""
        return self.peak_locations()

    def spatial_information(self) -> float:
        """Skaggs spatial information (bits/spike)."""
        ...

    def sparsity(self) -> float:
        """Sparsity measure (0-1)."""
        ...

    # Grid cell metrics - delegate to grid module
    def grid_score(self) -> float:
        """Grid score (hexagonal periodicity).

        Computes spatial autocorrelation and extracts grid score.
        Delegates to `neurospatial.encoding.grid.grid_score()`.
        """
        from neurospatial.encoding.grid import grid_score, spatial_autocorrelation
        autocorr = spatial_autocorrelation(self.env, self.firing_rate)
        return grid_score(autocorr, bin_size=self.env.bin_size)

    def grid_properties(self) -> "GridProperties":
        """Full grid cell metrics (score, scale, orientation).

        Returns GridProperties dataclass with score, scale, orientation,
        orientation_std, peak_coords, and n_peaks.
        Delegates to `neurospatial.encoding.grid.grid_properties()`.
        """
        from neurospatial.encoding.grid import grid_properties, spatial_autocorrelation
        autocorr = spatial_autocorrelation(self.env, self.firing_rate)
        return grid_properties(autocorr, bin_size=self.env.bin_size)

    # Border cell metrics - delegate to border module
    def border_score(
        self,
        threshold: float = 0.3,
        distance_metric: Literal["geodesic", "euclidean"] = "geodesic",
    ) -> float:
        """Border score (boundary proximity tuning).

        Quantifies alignment with environmental boundaries.
        Range [-1, 1]: +1 = perfect border cell, -1 = center-preferring.
        Delegates to `neurospatial.encoding.border.border_score()`.
        """
        from neurospatial.encoding.border import border_score
        return border_score(
            self.env, self.firing_rate,
            threshold=threshold, distance_metric=distance_metric
        )

    def region_coverage(
        self,
        threshold: float = 0.3,
        regions: list[str] | None = None,
    ) -> dict[str, float]:
        """Coverage of each region by the firing field.

        Returns fraction of each region's bins covered by the field
        (bins where firing_rate >= threshold * peak).
        Delegates to `neurospatial.encoding.border.compute_region_coverage()`.
        """
        from neurospatial.encoding._base import _to_numpy
        from neurospatial.encoding.border import compute_region_coverage
        # Threshold field at fraction of peak
        rate = _to_numpy(self.firing_rate)
        field_mask = rate >= threshold * np.nanmax(rate)
        return compute_region_coverage(field_mask, self.env, regions=regions)


@dataclass(frozen=True)
class SpatialRatesResult(SpatialResultMixin):
    """Result of spatial rate computation for multiple neurons.

    Inherits from SpatialResultMixin for peak_locations() and peak_firing_rates().
    """

    firing_rates: ArrayLike             # (n_neurons, n_bins), NumPy or JAX
    occupancy: ArrayLike                # (n_bins,) shared, NumPy or JAX
    env: Environment
    smoothing_method: str
    bandwidth: float

    def __len__(self) -> int:
        return len(self.firing_rates)

    def __getitem__(self, idx: int) -> SpatialRateResult:
        """Get single-neuron result. Enables iteration."""
        return SpatialRateResult(
            firing_rate=self.firing_rates[idx],
            occupancy=self.occupancy,
            env=self.env,
            smoothing_method=self.smoothing_method,
            bandwidth=self.bandwidth,
        )

    def __iter__(self) -> Iterator[SpatialRateResult]:
        """Iterate over single-neuron results."""
        for i in range(len(self)):
            yield self[i]

    # Batch methods - return arrays
    def plot(self, idx: int, ax=None, **kwargs):
        """Plot rate map for neuron idx."""
        return self.env.plot_field(self.firing_rates[idx], ax=ax, **kwargs)

    # NOTE: peak_locations() inherited from SpatialResultMixin
    # NOTE: peak_firing_rates() inherited from SpatialResultMixin

    def spatial_information(self) -> NDArray[np.float64]:
        """Spatial information for all neurons. Returns (n_neurons,)."""
        ...

    def sparsity(self) -> NDArray[np.float64]:
        """Sparsity for all neurons. Returns (n_neurons,)."""
        ...

    # Grid cell metrics - vectorized over neurons (backend-aware)
    def grid_scores(self) -> ArrayLike:
        """Grid scores for all neurons. Returns (n_neurons,).

        Computes spatial autocorrelation for each neuron and extracts grid score.
        Delegates to `neurospatial.encoding._metrics.batch_grid_scores()`.

        This is a backend-aware method: returns NumPy array if input was NumPy,
        JAX array if input was JAX. Compatible with jax.vmap.
        """
        from neurospatial.encoding._metrics import batch_grid_scores
        return batch_grid_scores(self.env, self.firing_rates)

    # Border cell metrics - vectorized over neurons (backend-aware)
    def border_scores(
        self,
        threshold: float = 0.3,
        distance_metric: Literal["geodesic", "euclidean"] = "geodesic",
    ) -> ArrayLike:
        """Border scores for all neurons. Returns (n_neurons,).

        Delegates to `neurospatial.encoding._metrics.batch_border_scores()`.

        This is a backend-aware method: returns NumPy array if input was NumPy,
        JAX array if input was JAX. Compatible with jax.vmap.
        """
        from neurospatial.encoding._metrics import batch_border_scores
        return batch_border_scores(
            self.env, self.firing_rates,
            threshold=threshold, distance_metric=distance_metric
        )

    def classify(
        self,
        min_spatial_info: float = 0.5,
        min_grid_score: float = 0.3,
        min_border_score: float = 0.5,
    ) -> NDArray[np.str_]:
        """Classify all neurons. Returns (n_neurons,) of labels."""
        ...

    def to_dataframe(
        self,
        neuron_ids: Sequence[str] | None = None,
        include_classification: bool = True,
    ) -> "pd.DataFrame":
        """Export metrics to DataFrame for exploratory analysis.

        Parameters
        ----------
        neuron_ids : sequence of str, optional
            Identifiers for each neuron. If None, uses integer indices.
        include_classification : bool, default True
            Whether to include cell type classification.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - neuron_id: identifier
            - peak_x, peak_y: peak firing location
            - peak_rate: maximum firing rate (Hz)
            - spatial_info: spatial information (bits/spike)
            - sparsity: sparsity measure (0-1)
            - grid_score: grid score
            - border_score: border score
            - cell_type: classification label (if include_classification)
        """
        import pandas as pd

        n_neurons = len(self)
        if neuron_ids is None:
            neuron_ids = list(range(n_neurons))

        peaks = self.peak_locations()  # From SpatialResultMixin
        data = {
            "neuron_id": neuron_ids,
            "peak_x": peaks[:, 0],
            "peak_y": peaks[:, 1] if peaks.shape[1] > 1 else np.nan,
            "peak_rate": self.peak_firing_rates(),  # From SpatialResultMixin
            "spatial_info": self.spatial_information(),
            "sparsity": self.sparsity(),
            "grid_score": self.grid_scores(),
            "border_score": self.border_scores(),
        }
        if include_classification:
            data["cell_type"] = self.classify()

        return pd.DataFrame(data)


def compute_spatial_rate(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> SpatialRateResult:
    """Compute spatial firing rate map for one neuron.

    Parameters
    ----------
    backend : {"numpy", "jax", "auto"}, default "numpy"
        Computation backend. "numpy" works everywhere. "jax" requires JAX
        installation (Linux/macOS only). "auto" uses JAX if available,
        falls back to NumPy silently on Windows or if JAX not installed.
    """
    ...


def compute_spatial_rates(
    env: Environment,
    spike_times_list: Sequence[NDArray[np.float64]],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> SpatialRatesResult:
    """Compute spatial firing rate maps for multiple neurons.

    Efficiently precomputes shared quantities (occupancy, kernel, position bins)
    and reuses across all neurons.

    Parameters
    ----------
    backend : {"numpy", "jax", "auto"}, default "numpy"
        Computation backend. "numpy" works everywhere. "jax" requires JAX
        installation (Linux/macOS only). "auto" uses JAX if available,
        falls back to NumPy silently on Windows or if JAX not installed.
    """
    ...
```

### 2. Directional Rate (Head Direction Cells)

**File**: `encoding/directional.py`

**Design note**: The current `HeadDirectionMetrics` is a data-heavy class with pre-computed fields.
The new design uses **lazy computation** - metrics are computed on-demand via methods.
This is more flexible (allows different thresholds) and avoids upfront cost for unused metrics.

```python
@dataclass(frozen=True)
class DirectionalRateResult:
    """Result of directional rate computation for single neuron.

    Replaces the current tuple return from `compute_head_direction_tuning_curve`
    and integrates functionality from `HeadDirectionMetrics`.
    """

    firing_rate: ArrayLike              # (n_bins,) firing rate by direction in Hz, NumPy or JAX
    occupancy: ArrayLike                # (n_bins,) seconds at direction, NumPy or JAX
    bin_centers: ArrayLike              # (n_bins,) direction in radians, NumPy or JAX
    bin_size: float                     # radians
    smoothing_sigma: float | None       # smoothing bandwidth in radians, None if unsmoothed

    def plot(self, ax=None, polar: bool = True, **kwargs):
        """Plot tuning curve."""
        ...

    def preferred_direction(self) -> float:
        """Preferred direction (circular mean) in radians [0, 2π]."""
        from neurospatial.stats.circular import circular_mean
        return circular_mean(self.bin_centers, weights=self.firing_rate)

    def preferred_direction_deg(self) -> float:
        """Preferred direction in degrees [0, 360]."""
        return np.degrees(self.preferred_direction())

    def mean_vector_length(self) -> float:
        """Rayleigh mean vector length (0-1). Higher = sharper tuning."""
        from neurospatial.stats.circular import mean_resultant_length
        return mean_resultant_length(self.bin_centers, weights=self.firing_rate)

    def tuning_width(self) -> float:
        """Half-width at half-maximum in radians."""
        ...

    def tuning_width_deg(self) -> float:
        """Half-width at half-maximum in degrees."""
        return np.degrees(self.tuning_width())

    def peak_firing_rate(self) -> float:
        """Maximum firing rate (Hz)."""
        return float(np.nanmax(self.firing_rate))

    def rayleigh_pvalue(self) -> float:
        """P-value from Rayleigh test for non-uniformity."""
        from neurospatial.stats.circular import rayleigh_test
        return rayleigh_test(self.bin_centers, weights=self.firing_rate)

    def is_hd_cell(self, min_mvl: float = 0.4, alpha: float = 0.05) -> bool:
        """Classify as head direction cell.

        Criteria (Taube et al., 1990):
        - Mean vector length > min_mvl (default 0.4)
        - Rayleigh test p-value < alpha (default 0.05)
        """
        return (
            self.mean_vector_length() > min_mvl
            and self.rayleigh_pvalue() < alpha
        )

    def interpretation(self, min_mvl: float = 0.4) -> str:
        """Human-readable interpretation of head direction metrics.

        Provides the same diagnostic output as current HeadDirectionMetrics.__str__().
        """
        ...


@dataclass(frozen=True)
class DirectionalRatesResult:
    """Result of directional rate computation for multiple neurons."""

    firing_rates: ArrayLike             # (n_neurons, n_bins), NumPy or JAX
    occupancy: ArrayLike                # (n_bins,) shared, NumPy or JAX
    bin_centers: ArrayLike              # (n_bins,), NumPy or JAX
    bin_size: float
    smoothing_sigma: float | None       # smoothing bandwidth in radians, None if unsmoothed

    def __len__(self) -> int:
        return len(self.firing_rates)

    def __getitem__(self, idx: int) -> DirectionalRateResult:
        """Get single-neuron result. Enables iteration."""
        return DirectionalRateResult(
            firing_rate=self.firing_rates[idx],
            occupancy=self.occupancy,
            bin_centers=self.bin_centers,
            bin_size=self.bin_size,
            smoothing_sigma=self.smoothing_sigma,
        )

    def __iter__(self) -> Iterator[DirectionalRateResult]:
        """Iterate over single-neuron results."""
        for i in range(len(self)):
            yield self[i]

    def plot(self, idx: int, ax=None, polar: bool = True, **kwargs):
        """Plot tuning curve for neuron idx."""
        ...

    def preferred_directions(self) -> NDArray[np.float64]:
        """Preferred directions for all neurons. Returns (n_neurons,)."""
        ...

    def mean_vector_lengths(self) -> NDArray[np.float64]:
        """MVL for all neurons. Returns (n_neurons,)."""
        ...

    def tuning_widths(self) -> NDArray[np.float64]:
        """Tuning widths for all neurons. Returns (n_neurons,)."""
        ...

    def detect_hd_cells(self, min_mvl: float = 0.4, alpha: float = 0.05) -> NDArray[np.bool_]:
        """Classify neurons as HD cells. Returns (n_neurons,) bool."""
        ...

    def to_dataframe(
        self,
        neuron_ids: Sequence[str] | None = None,
    ) -> "pd.DataFrame":
        """Export metrics to DataFrame for exploratory analysis.

        Parameters
        ----------
        neuron_ids : sequence of str, optional
            Identifiers for each neuron. If None, uses integer indices.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - neuron_id: identifier
            - preferred_direction: preferred direction (radians)
            - preferred_direction_deg: preferred direction (degrees)
            - mean_vector_length: Rayleigh MVL (0-1)
            - tuning_width: half-width at half-max (radians)
            - tuning_width_deg: half-width at half-max (degrees)
            - peak_rate: maximum firing rate (Hz)
            - is_hd_cell: boolean classification
        """
        import pandas as pd

        n_neurons = len(self)
        if neuron_ids is None:
            neuron_ids = list(range(n_neurons))

        pref_dir = self.preferred_directions()
        tuning = self.tuning_widths()
        data = {
            "neuron_id": neuron_ids,
            "preferred_direction": pref_dir,
            "preferred_direction_deg": np.degrees(pref_dir),
            "mean_vector_length": self.mean_vector_lengths(),
            "tuning_width": tuning,
            "tuning_width_deg": np.degrees(tuning),
            "peak_rate": np.nanmax(self.firing_rates, axis=1),
            "is_hd_cell": self.detect_hd_cells(),
        }

        return pd.DataFrame(data)


def compute_directional_rate(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    bin_size: float = np.pi / 30,  # 6 degrees
    smoothing_sigma: float | None = None,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> DirectionalRateResult:
    """Compute directional firing rate for one neuron.

    Parameters
    ----------
    headings : array
        Head direction at each time point.
    bin_size : float, default π/30
        Bin size for direction histogram. Interpreted according to angle_unit.
    smoothing_sigma : float, optional
        Gaussian smoothing bandwidth. Interpreted according to angle_unit.
    angle_unit : {"rad", "deg"}, default "rad"
        Unit for headings, bin_size, and smoothing_sigma.
        If "deg", all angular values are converted to radians internally.
        Result bin_centers are always in radians.

    Note: Does not require Environment - head direction is not spatial.
    This is an intentional deviation from the canonical argument order
    (which puts env first for spatial functions).
    """
    ...


def compute_directional_rates(
    spike_times_list: Sequence[NDArray[np.float64]],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    bin_size: float = np.pi / 30,
    smoothing_sigma: float | None = None,
    angle_unit: Literal["rad", "deg"] = "rad",
    n_jobs: int = 1,
) -> DirectionalRatesResult:
    """Compute directional firing rates for multiple neurons.

    See compute_directional_rate for parameter details.
    """
    ...


# Migration shim for backwards compatibility
def compute_head_direction_tuning_curve(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    bin_size: float = 6.0,  # Legacy default was degrees
    smoothing_sigma: float | None = None,
    angle_unit: Literal["rad", "deg"] = "deg",  # Legacy default was degrees
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Legacy API - use compute_directional_rate instead.

    .. deprecated:: 0.4.0
        Use :func:`compute_directional_rate` instead.

    Returns (bin_centers, firing_rate) tuple for backwards compatibility.
    """
    import warnings
    warnings.warn(
        "compute_head_direction_tuning_curve is deprecated. "
        "Use compute_directional_rate instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = compute_directional_rate(
        spike_times, times, headings,
        bin_size=bin_size, smoothing_sigma=smoothing_sigma,
        angle_unit=angle_unit,
    )
    return (result.bin_centers, result.firing_rate)
```

### 3. View Rate (Spatial View Cells)

**File**: `encoding/view.py`

```python
@dataclass(frozen=True)
class ViewRateResult:
    """Result of view rate computation for single neuron."""

    firing_rate: ArrayLike                  # (n_bins,) rate by viewed location in Hz, NumPy or JAX
    view_occupancy: ArrayLike               # (n_bins,) time viewing each bin, NumPy or JAX
    env: Environment
    gaze_model: str
    view_distance: float
    smoothing_method: str
    bandwidth: float

    def plot(self, ax=None, **kwargs):
        """Plot view field."""
        return self.env.plot_field(self.firing_rate, ax=ax, **kwargs)

    def peak_view_location(self) -> NDArray[np.float64]:
        """Location of peak view response."""
        return self.env.bin_centers[np.nanargmax(self.firing_rate)]

    def view_spatial_information(self) -> float:
        """Spatial information based on view occupancy."""
        ...

    def is_view_cell(self, min_info: float = 0.5) -> bool:
        """Classify as spatial view cell."""
        ...


@dataclass(frozen=True)
class ViewRatesResult:
    """Result of view rate computation for multiple neurons."""

    firing_rates: ArrayLike                 # (n_neurons, n_bins), NumPy or JAX
    view_occupancy: ArrayLike               # (n_bins,) shared, NumPy or JAX
    env: Environment
    gaze_model: str
    view_distance: float
    smoothing_method: str
    bandwidth: float

    def __len__(self) -> int:
        return len(self.firing_rates)

    def __getitem__(self, idx: int) -> ViewRateResult:
        """Get single-neuron result. Enables iteration."""
        return ViewRateResult(
            firing_rate=self.firing_rates[idx],
            view_occupancy=self.view_occupancy,
            env=self.env,
            gaze_model=self.gaze_model,
            view_distance=self.view_distance,
            smoothing_method=self.smoothing_method,
            bandwidth=self.bandwidth,
        )

    def __iter__(self) -> Iterator[ViewRateResult]:
        """Iterate over single-neuron results."""
        for i in range(len(self)):
            yield self[i]

    def plot(self, idx: int, ax=None, **kwargs):
        """Plot view field for neuron idx."""
        return self.env.plot_field(self.firing_rates[idx], ax=ax, **kwargs)

    def peak_view_locations(self) -> NDArray[np.float64]:
        """Peak view locations. Returns (n_neurons, n_dims)."""
        ...

    def view_spatial_information(self) -> NDArray[np.float64]:
        """View spatial info for all. Returns (n_neurons,)."""
        ...

    def detect_view_cells(self, min_info: float = 0.5) -> NDArray[np.bool_]:
        """Classify as view cells. Returns (n_neurons,) bool."""
        ...

    def to_dataframe(
        self,
        neuron_ids: Sequence[str] | None = None,
    ) -> "pd.DataFrame":
        """Export metrics to DataFrame for exploratory analysis.

        Parameters
        ----------
        neuron_ids : sequence of str, optional
            Identifiers for each neuron. If None, uses integer indices.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - neuron_id: identifier
            - peak_view_x, peak_view_y: peak view response location
            - peak_rate: maximum firing rate (Hz)
            - view_spatial_info: spatial information based on view occupancy
            - is_view_cell: boolean classification
        """
        import pandas as pd

        n_neurons = len(self)
        if neuron_ids is None:
            neuron_ids = list(range(n_neurons))

        peaks = self.peak_view_locations()
        data = {
            "neuron_id": neuron_ids,
            "peak_view_x": peaks[:, 0],
            "peak_view_y": peaks[:, 1] if peaks.shape[1] > 1 else np.nan,
            "peak_rate": np.nanmax(self.firing_rates, axis=1),
            "view_spatial_info": self.view_spatial_information(),
            "is_view_cell": self.detect_view_cells(),
        }

        return pd.DataFrame(data)


def compute_view_rate(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
) -> ViewRateResult:
    """Compute view field for one neuron."""
    ...


def compute_view_rates(
    env: Environment,
    spike_times_list: Sequence[NDArray[np.float64]],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    n_jobs: int = 1,
) -> ViewRatesResult:
    """Compute view fields for multiple neurons."""
    ...
```

### 4. Egocentric Rate (Object Vector Cells)

**File**: `encoding/egocentric.py`

```python
@dataclass(frozen=True)
class EgocentricRateResult:
    """Result of egocentric rate computation for single neuron."""

    firing_rate: ArrayLike              # (n_bins,) rate in ego polar coords in Hz, NumPy or JAX
    occupancy: ArrayLike                # (n_bins,) time in ego bin, NumPy or JAX
    ego_env: Environment                # Egocentric polar environment
    distance_range: tuple[float, float]
    n_distance_bins: int
    n_direction_bins: int

    def plot(self, ax=None, **kwargs):
        """Plot egocentric rate map."""
        return self.ego_env.plot_field(self.firing_rate, ax=ax, **kwargs)

    def preferred_distance(self) -> float:
        """Preferred distance to object."""
        peak_bin = np.nanargmax(self.firing_rate)
        return self.ego_env.bin_centers[peak_bin, 0]

    def preferred_direction(self) -> float:
        """Preferred direction to object (radians, 0=ahead)."""
        peak_bin = np.nanargmax(self.firing_rate)
        return self.ego_env.bin_centers[peak_bin, 1]

    def is_ovc(self, min_info: float = 0.3) -> bool:
        """Classify as object vector cell."""
        ...


@dataclass(frozen=True)
class EgocentricRatesResult:
    """Result of egocentric rate computation for multiple neurons."""

    firing_rates: ArrayLike             # (n_neurons, n_bins), NumPy or JAX
    occupancy: ArrayLike                # (n_bins,) shared, NumPy or JAX
    ego_env: Environment
    distance_range: tuple[float, float]
    n_distance_bins: int
    n_direction_bins: int

    def __len__(self) -> int:
        return len(self.firing_rates)

    def __getitem__(self, idx: int) -> EgocentricRateResult:
        """Get single-neuron result. Enables iteration."""
        return EgocentricRateResult(
            firing_rate=self.firing_rates[idx],
            occupancy=self.occupancy,
            ego_env=self.ego_env,
            distance_range=self.distance_range,
            n_distance_bins=self.n_distance_bins,
            n_direction_bins=self.n_direction_bins,
        )

    def __iter__(self) -> Iterator[EgocentricRateResult]:
        """Iterate over single-neuron results."""
        for i in range(len(self)):
            yield self[i]

    def plot(self, idx: int, ax=None, **kwargs):
        """Plot egocentric rate map for neuron idx."""
        return self.ego_env.plot_field(self.firing_rates[idx], ax=ax, **kwargs)

    def preferred_distances(self) -> NDArray[np.float64]:
        """Preferred distances. Returns (n_neurons,)."""
        ...

    def preferred_directions(self) -> NDArray[np.float64]:
        """Preferred directions. Returns (n_neurons,)."""
        ...

    def detect_ovcs(self, min_info: float = 0.3) -> NDArray[np.bool_]:
        """Classify as OVCs. Returns (n_neurons,) bool."""
        ...

    def to_dataframe(
        self,
        neuron_ids: Sequence[str] | None = None,
    ) -> "pd.DataFrame":
        """Export metrics to DataFrame for exploratory analysis.

        Parameters
        ----------
        neuron_ids : sequence of str, optional
            Identifiers for each neuron. If None, uses integer indices.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - neuron_id: identifier
            - preferred_distance: preferred distance to object (cm)
            - preferred_direction: preferred direction to object (radians, 0=ahead)
            - preferred_direction_deg: preferred direction (degrees)
            - peak_rate: maximum firing rate (Hz)
            - is_ovc: boolean classification
        """
        import pandas as pd

        n_neurons = len(self)
        if neuron_ids is None:
            neuron_ids = list(range(n_neurons))

        pref_dir = self.preferred_directions()
        data = {
            "neuron_id": neuron_ids,
            "preferred_distance": self.preferred_distances(),
            "preferred_direction": pref_dir,
            "preferred_direction_deg": np.degrees(pref_dir),
            "peak_rate": np.nanmax(self.firing_rates, axis=1),
            "is_ovc": self.detect_ovcs(),
        }

        return pd.DataFrame(data)


def compute_egocentric_rate(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    *,
    env: Environment | None = None,
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    distance_metric: Literal["euclidean", "geodesic"] = "euclidean",
    smoothing_method: Literal["binned", "diffusion_kde"] = "binned",
    bandwidth: float = 5.0,
) -> EgocentricRateResult:
    """Compute egocentric (object-vector) rate map for one neuron.

    Parameters
    ----------
    env : Environment, optional
        Required if distance_metric="geodesic". The environment is used to
        compute shortest-path distances around obstacles.

    Raises
    ------
    ValueError
        If distance_metric="geodesic" but env is None.
    """
    ...


def compute_egocentric_rates(
    spike_times_list: Sequence[NDArray[np.float64]],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    *,
    env: Environment | None = None,
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    distance_metric: Literal["euclidean", "geodesic"] = "euclidean",
    smoothing_method: Literal["binned", "diffusion_kde"] = "binned",
    bandwidth: float = 5.0,
    n_jobs: int = 1,
) -> EgocentricRatesResult:
    """Compute egocentric rate maps for multiple neurons.

    Parameters
    ----------
    env : Environment, optional
        Required if distance_metric="geodesic". The environment is used to
        compute shortest-path distances around obstacles.

    Raises
    ------
    ValueError
        If distance_metric="geodesic" but env is None.
    """
    ...
```

---

## File Structure

```
src/neurospatial/encoding/
├── __init__.py              # Re-exports all public API
├── spatial.py               # compute_spatial_rate(s), SpatialRate(s)Result
├── directional.py           # compute_directional_rate(s), DirectionalRate(s)Result
├── view.py                  # compute_view_rate(s), ViewRate(s)Result
├── egocentric.py            # compute_egocentric_rate(s), EgocentricRate(s)Result
├── _backend.py              # Backend selection: get_backend(), platform detection
├── _core_numpy.py           # NumPy implementations of core array operations
├── _core_jax.py             # JAX implementations (optional, Linux/macOS only)
├── _metrics.py              # Shared metric implementations (spatial_info, etc.)
├── _smoothing.py            # Shared smoothing implementations
├── _base.py                 # Shared protocols/mixins, array helpers (see below)
├── _spikes.py               # Spike format normalization helpers (see below)
├── border.py                # RETAINED: border_score(), compute_region_coverage()
├── grid.py                  # RETAINED: grid_score, grid_scale, grid_properties, etc.
├── population.py            # RETAINED: population-level analysis (unchanged)
└── phase_precession.py      # RETAINED: phase precession (unchanged)
```

### `_base.py` - Shared Protocols, Mixins, and Array Helpers

This module provides shared infrastructure for result classes:

```python
"""Shared protocols, mixins, and array helpers for encoding result classes."""

from typing import Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray, ArrayLike


def _to_numpy(arr: ArrayLike) -> NDArray:
    """Convert array to NumPy for host-only operations.

    Use this for plotting, DataFrame export, and other convenience methods
    that are not part of the JAX-traced compute graph.
    """
    return np.asarray(arr)


def _get_array_module(arr: ArrayLike):
    """Get the array module (numpy or jax.numpy) for backend-aware operations."""
    if hasattr(arr, '__jax_array__'):
        import jax.numpy as jnp
        return jnp
    return np


@runtime_checkable
class HasOccupancy(Protocol):
    """Protocol for result classes with occupancy data."""
    occupancy: ArrayLike


@runtime_checkable
class HasEnvironment(Protocol):
    """Protocol for result classes with spatial environment."""
    env: "Environment"


class SpatialResultMixin:
    """Mixin providing common spatial result methods.

    Requires: self.firing_rate or self.firing_rates, self.occupancy, self.env

    Result classes should inherit this mixin to get consistent implementations
    of peak_locations() and other shared methods. Do NOT reimplement these
    methods in subclasses.
    """

    def _get_rates(self) -> ArrayLike:
        """Get firing rate(s), handling both single and batch results."""
        if hasattr(self, 'firing_rates'):
            return self.firing_rates
        return self.firing_rate

    def peak_locations(self) -> NDArray[np.float64]:
        """Peak firing locations. Returns (n_dims,) for single, (n_neurons, n_dims) for batch.

        This is a host-only method: always returns NumPy arrays.
        Uses _to_numpy() internally for JAX compatibility.
        """
        rates = _to_numpy(self._get_rates())
        if rates.ndim == 1:
            return self.env.bin_centers[np.nanargmax(rates)]
        peaks = np.nanargmax(rates, axis=1)
        return self.env.bin_centers[peaks]

    def peak_firing_rates(self) -> NDArray[np.float64]:
        """Peak firing rates. Returns scalar for single, (n_neurons,) for batch.

        This is a host-only method: always returns NumPy arrays.
        """
        rates = _to_numpy(self._get_rates())
        if rates.ndim == 1:
            return float(np.nanmax(rates))
        return np.nanmax(rates, axis=1)
```

### `_spikes.py` - Spike Format Normalization

This module provides helpers to normalize various spike input formats to the canonical internal representation.

```python
"""Spike format normalization for encoding functions."""

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray


def normalize_spike_times(
    spike_times: NDArray[np.float64] | Sequence[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    """Normalize spike times to canonical list-of-arrays format.

    Parameters
    ----------
    spike_times : array or sequence of arrays
        Spike times in one of these formats:

        - 1D array (single neuron) → wrapped in list
        - 2D array (n_neurons, max_spikes) → split along axis 0, NaN-padding removed
        - List/tuple of 1D arrays (canonical format) → each element converted to array

    Returns
    -------
    list[NDArray[np.float64]]
        List of 1D spike time arrays, one per neuron.

    Raises
    ------
    ValueError
        If input is a ragged object array or has unexpected shape.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._spikes import normalize_spike_times

    >>> # Single neuron (1D array)
    >>> spikes = np.array([0.1, 0.5, 1.2])
    >>> normalized = normalize_spike_times(spikes)
    >>> len(normalized)
    1

    >>> # Multiple neurons (list of arrays)
    >>> spikes = [np.array([0.1, 0.5]), np.array([0.2, 0.3, 0.8])]
    >>> normalized = normalize_spike_times(spikes)
    >>> len(normalized)
    2

    >>> # Tuple of arrays also works
    >>> spikes = (np.array([0.1]), np.array([0.2, 0.3]))
    >>> normalized = normalize_spike_times(spikes)
    >>> len(normalized)
    2
    """
    # Handle Sequence types (list, tuple, etc.) but not ndarray
    if isinstance(spike_times, Sequence) and not isinstance(spike_times, np.ndarray):
        # Convert each element to 1D float array
        result = []
        for i, row in enumerate(spike_times):
            arr = np.asarray(row, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(
                    f"Each spike train must be 1D, but element {i} has shape {arr.shape}"
                )
            result.append(arr)
        return result

    # Convert to array for shape inspection
    arr = np.asarray(spike_times)

    # Reject object arrays (ragged input passed as array)
    if arr.dtype == object:
        raise ValueError(
            "Received ragged array (dtype=object). Pass spike times as a list of "
            "1D arrays instead, e.g., [np.array([0.1, 0.2]), np.array([0.3])]"
        )

    # 1D array: single neuron
    if arr.ndim == 1:
        return [arr.astype(np.float64)]

    # 2D array: split along axis 0, remove NaN padding
    if arr.ndim == 2:
        return [row[~np.isnan(row)].astype(np.float64) for row in arr]

    raise ValueError(
        f"spike_times must be 1D array, 2D array, or sequence of arrays, "
        f"got shape {arr.shape}"
    )
```

**Integration with compute functions:**

```python
def compute_spatial_rates(
    env: Environment,
    spike_times: NDArray | Sequence[NDArray],  # Accept multiple formats
    ...
) -> SpatialRatesResult:
    # Normalize at entry point
    spike_times_list = normalize_spike_times(spike_times)
    # ... rest of implementation uses spike_times_list
```

### `__init__.py` Exports

```python
from neurospatial.encoding.spatial import (
    compute_spatial_rate,
    compute_spatial_rates,
    SpatialRateResult,
    SpatialRatesResult,
)
from neurospatial.encoding.directional import (
    compute_directional_rate,
    compute_directional_rates,
    DirectionalRateResult,
    DirectionalRatesResult,
    # Deprecated, kept for backwards compatibility
    compute_head_direction_tuning_curve,
)
from neurospatial.encoding.view import (
    compute_view_rate,
    compute_view_rates,
    ViewRateResult,
    ViewRatesResult,
)
from neurospatial.encoding.egocentric import (
    compute_egocentric_rate,
    compute_egocentric_rates,
    EgocentricRateResult,
    EgocentricRatesResult,
)
from neurospatial.encoding.grid import (
    grid_score,
    grid_scale,
    grid_orientation,
    GridProperties,
)

__all__ = [
    # Spatial (place/grid/border)
    "compute_spatial_rate",
    "compute_spatial_rates",
    "SpatialRateResult",
    "SpatialRatesResult",
    # Directional (head direction)
    "compute_directional_rate",
    "compute_directional_rates",
    "DirectionalRateResult",
    "DirectionalRatesResult",
    # View (spatial view cells)
    "compute_view_rate",
    "compute_view_rates",
    "ViewRateResult",
    "ViewRatesResult",
    # Egocentric (object vector)
    "compute_egocentric_rate",
    "compute_egocentric_rates",
    "EgocentricRateResult",
    "EgocentricRatesResult",
    # Grid analysis
    "grid_score",
    "grid_scale",
    "grid_orientation",
    "GridProperties",
]
```

---

## Migration Path

### What Gets Removed

- `place.py` → merged into `spatial.py`
- `head_direction.py` → becomes `directional.py`
  - `compute_head_direction_tuning_curve` → deprecated shim in `directional.py`, warns and delegates to `compute_directional_rate`
- `spatial_view.py` → becomes `view.py`
- `object_vector.py` → becomes `egocentric.py`
- `HeadDirectionMetrics` → functionality absorbed into `DirectionalRateResult`
- `ObjectVectorMetrics` → functionality absorbed into `EgocentricRateResult`
- `SpatialViewMetrics` → functionality absorbed into `ViewRateResult`

### What Stays (Retained Files)

| File | Reason |
|------|--------|
| `border.py` | `border_score()` and `compute_region_coverage()` are standalone utilities. Result classes delegate to these functions. |
| `grid.py` | `grid_score()`, `grid_properties()`, `spatial_autocorrelation()` are reusable utilities. `SpatialRateResult.grid_score()` delegates to `grid.grid_score()`. |
| `population.py` | Population-level analysis (coverage, overlap) is orthogonal to single-cell encoding. |
| `phase_precession.py` | Specialized theta phase analysis, not part of rate map computation. |

### Grid/Border Integration Details

The `grid.py` and `border.py` modules are **retained as standalone utilities**.
Result class methods **delegate** to these modules rather than duplicating code:

```python
# SpatialRateResult.grid_score() implementation
def grid_score(self) -> float:
    from neurospatial.encoding.grid import grid_score, spatial_autocorrelation
    autocorr = spatial_autocorrelation(self.env, self.firing_rate)
    return grid_score(autocorr, bin_size=self.env.bin_size)

# SpatialRateResult.border_score() implementation
def border_score(self, threshold: float = 0.3, ...) -> float:
    from neurospatial.encoding.border import border_score
    return border_score(self.env, self.firing_rate, threshold=threshold, ...)
```

This design:

1. Avoids code duplication
2. Allows standalone use of `grid_score(autocorr, ...)` for custom workflows
3. Keeps result classes focused on convenience, not reimplementing algorithms

---

## Implementation Order

### Phase 0: Core Shapes and Backend Interface

1. Decide canonical shapes for counts/occupancy: `(n_neurons, n_bins)` and `(n_bins,)`
2. Define shared base protocols/mixins (`HasOccupancy`, `HasEnv`) for result classes to reduce duplication
3. Define minimal backend interface in `encoding/_backend.py`:
   - `get_backend(name: str)` → returns NumPy or JAX implementation module
   - Platform detection for JAX availability
4. Create `encoding/_core_numpy.py` with core array operations (stubs initially)
5. Create `encoding/_core_jax.py` with stubs (implemented in Phase 6)

### Phase 1: NumPy Core

1. Create `encoding/_metrics.py` with shared metric implementations (spatial_info, sparsity, etc.)
2. Create `encoding/_smoothing.py` with shared smoothing code
3. Implement all rate and metric computations on dense arrays in `_core_numpy.py`
4. Define spike format normalization helpers (1D → list, 2D → list of rows)

### Phase 2: Spatial Rate + Binning Layer

1. Create `encoding/spatial.py`
2. Implement binning layer that converts spike trains → `(spike_counts, occupancy)` using joblib
   - Ensure output shapes match canonical `(n_neurons, n_bins)` and `(n_bins,)` from Phase 1
3. Implement `SpatialRateResult` with all methods (using shared mixins)
4. Implement `SpatialRatesResult` with batch methods
5. Implement `compute_spatial_rate` (refactor from `compute_place_field`)
6. Implement `compute_spatial_rates` with precomputation optimization
7. Add grid_score, border_score as methods
8. Write tests

### Phase 3: Directional Rate

1. Create `encoding/directional.py`
2. Implement result classes (minimal Environment coupling—only bin_centers needed)
3. Implement binning layer + NumPy core
   - Ensure output shapes match canonical `(n_neurons, n_bins)` and `(n_bins,)`
4. Implement compute functions (refactor from `compute_head_direction_tuning_curve`)
5. Write tests

### Phase 4: View Rate

1. Create `encoding/view.py`
2. Implement result classes
3. Implement binning layer + NumPy core
   - Ensure output shapes match canonical `(n_neurons, n_bins)` and `(n_bins,)`
4. Implement compute functions (refactor from `compute_spatial_view_field`)
5. Write tests

### Phase 5: Egocentric Rate

1. Create `encoding/egocentric.py`
2. Implement result classes (minimal Environment coupling)
3. Implement binning layer + NumPy core
   - Ensure output shapes match canonical `(n_neurons, n_bins)` and `(n_bins,)`
4. Implement compute functions (refactor from `compute_object_vector_field`)
5. Write tests

### Phase 6: Optional JAX Backend

1. Implement JAX versions of core functions in `_core_jax.py` where beneficial:
   - Smoothing operations
   - Batch metric computations (spatial_info, grid_score, etc.)
   - Classification vectorized over neurons
2. Add backend selection logic and platform checks (Linux/macOS only)
3. Add `backend` parameter to compute functions (`"numpy"`, `"jax"`, `"auto"`)
4. Ensure plotting methods convert JAX arrays to NumPy transparently
5. Write backend-specific tests

### Phase 7: Cleanup and Documentation

1. Update `encoding/__init__.py` with new exports
2. Remove old files (`place.py`, `head_direction.py`, `spatial_view.py`, `object_vector.py`)
   - Note: `border.py` and `grid.py` are RETAINED as standalone utilities (see "What Stays" section)
3. Update all examples and documentation
4. Update CLAUDE.md with new API and backend documentation

---

## Example Usage After Refactor

```python
from neurospatial import Environment
from neurospatial.encoding import (
    compute_spatial_rates,
    compute_directional_rates,
)

# Load data
env = Environment.from_samples(positions, bin_size=5.0)
all_spike_times = [...]  # List of spike time arrays for each neuron

# Compute spatial rate maps for all neurons
spatial = compute_spatial_rates(env, all_spike_times, times, positions, n_jobs=-1)

# Analyze
info = spatial.spatial_information()        # (n_neurons,)
grid = spatial.grid_scores()                # (n_neurons,)
border = spatial.border_scores()            # (n_neurons,)
labels = spatial.classify()                 # (n_neurons,) ["place", "grid", "border", ...]

print(f"Place cells: {(labels == 'place').sum()}")
print(f"Grid cells: {(labels == 'grid').sum()}")
print(f"Border cells: {(labels == 'border').sum()}")

# Plot top place cells
place_mask = labels == "place"
top_place = np.argsort(info * place_mask)[-9:]
fig, axes = plt.subplots(3, 3)
for ax, idx in zip(axes.flat, top_place):
    spatial.plot(idx, ax=ax)
    ax.set_title(f"Neuron {idx}: {info[idx]:.2f} bits")

# Head direction analysis
hd = compute_directional_rates(all_spike_times, times, headings, n_jobs=-1)
mvl = hd.mean_vector_lengths()              # (n_neurons,)
preferred = hd.preferred_directions()       # (n_neurons,)
is_hd = hd.detect_hd_cells()                # (n_neurons,) bool

print(f"HD cells: {is_hd.sum()}")

# Export to DataFrame for exploratory analysis
df = spatial.to_dataframe(neuron_ids=[f"unit_{i}" for i in range(len(spatial))])
print(df.head())
#    neuron_id  peak_x  peak_y  peak_rate  spatial_info  sparsity  grid_score  border_score  cell_type
# 0    unit_0   45.2    32.1      12.5         1.23      0.45       0.12         0.08      place
# 1    unit_1   78.3    56.7       8.2         0.89      0.38       0.67         0.15       grid
# ...

# Filter and sort with pandas
place_cells = df[df["cell_type"] == "place"].sort_values("spatial_info", ascending=False)
print(f"Top 5 place cells by spatial info:\n{place_cells.head()}")

# Head direction DataFrame
hd_df = hd.to_dataframe()
hd_cells = hd_df[hd_df["is_hd_cell"]]
print(f"HD cells tuned to {np.degrees(hd_cells['preferred_direction'].mean()):.1f}° on average")
```

---

## Resolved Design Decisions

1. **Single vs batch result types**: Separate types. `SpatialRateResult` for single neuron, `SpatialRatesResult` for batch. Batch `__getitem__` returns single-neuron result for clean iteration.

2. **Spike format support**: Implemented in `_spikes.py`. Accept 1D array, 2D array, or list of arrays. Normalize to list of 1D arrays at entry point.

3. **classify() return type**: Return `NDArray[np.str_]` for simplicity. Add `.classify_enum()` later if needed.

4. **Field naming**: Use `.firing_rate` (single) and `.firing_rates` (batch) consistently across all result types. Replaces inconsistent `.field` naming in current code.

5. **Grid/border integration**: Result methods delegate to standalone `grid.py` and `border.py` functions. No code duplication.

6. **HeadDirectionMetrics replacement**: `DirectionalRateResult` uses lazy computation (methods) instead of eager pre-computation (dataclass fields). More flexible for varying thresholds.

7. **Mixin for shared methods**: `SpatialResultMixin` provides `peak_locations()` and `peak_firing_rates()` for both single and batch result classes. Single-neuron classes provide `peak_location()` alias for convenience. All mixin methods use `_to_numpy()` for JAX compatibility.

8. **Directional API backwards compatibility**: `compute_head_direction_tuning_curve` kept as deprecated shim that returns legacy `(bin_centers, firing_rate)` tuple. New API uses `angle_unit` parameter (default "rad") for explicit unit handling.

## Open Questions

1. **JAX integration details**
   - Which core operations should have JAX implementations initially? (smoothing, metrics, classify)
   - ~~What should `"auto"` do on Windows?~~ → **Resolved**: Force `"numpy"` with no warning
   - ~~Expose backend choice per-function via `backend=` parameter~~ → **Resolved**: Yes, add `backend=` parameter to all compute functions
   - Global config option for default backend? (e.g., `neurospatial.set_backend("jax")`)

2. **Result class duplication**
   - ~~Use shared base protocols (`HasOccupancy`, `HasEnv`) or mixins for common methods~~ → **Resolved**: Yes, define in `_base.py`
   - ~~Non-spatial results (directional, egocentric) should only depend on what they need~~ → **Resolved**: Directional uses only `bin_centers`, egocentric uses `ego_env`
   - ~~JAX array handling in methods~~ → **Resolved**: Methods classified as host-only vs backend-aware; host-only use `_to_numpy()`, backend-aware route through `_metrics.py`

3. **Batched operations for scalability**
   - ~~grid_scores/border_scores loop over neurons~~ → **Resolved**: Route through `batch_grid_scores()` / `batch_border_scores()` in `_metrics.py`
   - Consider batched `spatial_autocorrelation(env, firing_rates)` that handles `(n_neurons, n_bins)` directly
   - JAX backend can use `vmap` over the single-neuron function for automatic vectorization
