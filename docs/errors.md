# Error Reference

**Last Updated**: 2026-10-07

This reference explains common errors and how to correct their inputs. Environment errors use the codes below; other validation errors name the calling function and offending argument or shape.

## Catching neurospatial errors

Import `NeurospatialError` from the package root to catch library-specific exceptions:

```python
from neurospatial import NeurospatialError

try:
    distances = env.distance_to(region="hom")
except NeurospatialError as exc:
    print(exc)
```

If the environment contains a region named `home`, the error suggests `Fix: pass region='home'`. If the region is missing, build a replacement collection:

```python
from neurospatial import Regions

regions = Regions(env.regions.values())
regions.add("home", point=(x, y))  # Or pass polygon=... instead.
env = env.with_regions(regions)
```

Concrete exceptions inherit their standard Python type first, followed by `NeurospatialError`, so existing catch blocks continue to work:

| Exception | Standard Python bases | Use |
|-----------|-----------------------|-----|
| `RegionNotFoundError` | `KeyError`, `ValueError` | Unknown region name anywhere a region is looked up by name, including `env.regions[name]`; suggests the closest existing name |
| `BinIndexOutOfRangeError` | `ValueError` | Graph-query bin index outside `[0, env.n_bins)` |
| `IncompatibleEnvironmentError` | `ValueError` | Incompatible dimensions or decoding model bins |
| `LayoutNotBuiltError` | `RuntimeError` | Access to layout geometry before `build()` |
| `CartesianOnlyError` | `TypeError` | An operation that reads positions as Cartesian `(x, y)` called on a polar environment |

Two kinds of environment cannot run some analyses, and each kind has one error type:

- A **polar** environment (`from_polar_egocentric`) raises `CartesianOnlyError` for every operation that reads its `(distance, angle)` bins as Cartesian positions, for example `subset`, `interpolate`, wall-based border scores or `time_allocation="linear"`. Containment lookups such as `bin_at` and `contains` read `(distance, angle)` rows and work on polar environments.
- An environment whose **geometry lacks what the computation needs** raises `NotImplementedError`: for example the 2D `spatial_autocorrelation` and grid scores on a hexagonal, mesh, track or 3-D environment, wall geometry outside 2-D, `bin_at()`, `contains()` and the other position lookups on a track built without `track_width`, or `rebin` on bins that are not a rectangular grid. The message names an alternative that works on that environment.

Graph-query indices previously raised `IndexError`; catch `BinIndexOutOfRangeError` or `ValueError` for those calls. Ordinary input validation can also raise standard `TypeError` or `ValueError` directly.

Diagnostics describe what is wrong, why it matters and a corrected call using the `Fix:` label (formerly `HOW:`). Shared trajectory validation reports shape, length and timestamp problems together. `RegionNotFoundError` prints its message without `KeyError`'s quotes; bare `KeyError` messages include their fix as a final sentence. Missing required arguments keep Python's own `TypeError` and truthful signatures.

Warnings also use concrete fixes. A `bin_size` that covers every data axis warns about a possible units mismatch, and peri-event windows wider than 60 seconds suggest `window=(-0.5, 1.0)` when milliseconds were intended. Large-grid warnings are visible `UserWarning`s; the existing hard memory ceiling still applies.

---

## Quick Index

| Code | Error | Severity | Section |
|------|-------|----------|---------|
| [E1001](#e1001-no-active-bins-found) | No active bins found | Critical | [Environment Creation](#environment-creation-errors) |
| [E1002](#e1002-invalid-bin_size) | Invalid bin_size | Critical | [Environment Creation](#environment-creation-errors) |
| [E1003](#e1003-dimension-mismatch) | Dimension mismatch | Critical | [Environment Compatibility](#environment-compatibility-errors) |
| [E1005](#e1005-path-traversal-detected) | Path traversal detected | Critical | [I/O Errors](#io-errors) |
| [E1006](#e1006-environment-constructed-directly) | Environment constructed directly | Critical | [Environment Creation](#environment-creation-errors) |

---

## Environment Creation Errors

### E1001: No active bins found

**Error Message Example:**
```
ValueError: [E1001] No active bins found after filtering.

Diagnostics:
  Data range: [(0.0, 100.0), (0.0, 100.0)]
  Data extent: [100.0, 100.0]
  Number of samples: 500
  bin_size: 200.0
  Grid shape: (1, 1)
  bin_count_threshold: 1

Possible solutions:
  1. Reduce bin_size (try bin_size=10.0 for this data range)
  2. Lower bin_count_threshold (try bin_count_threshold=0)
  3. Enable morphological operations (dilate=True, fill_holes=True)
  4. Check that dimension_ranges match your data
```

**What This Means:**

The environment creation process couldn't find any spatial bins that meet the filtering criteria. This happens when:

1. **bin_size is too large** - The bin_size creates too few bins and they get filtered out
2. **bin_count_threshold is too high** - Bins need more samples than available to be "active"
3. **Data is too sparse** - Not enough samples to activate bins across the space
4. **Morphological filtering is too aggressive** - Holes/gaps leave no active bins

**Solutions:**

#### Solution 1: Reduce bin_size (Most Common)

```python
# ❌ Too large - creates only 1 bin
env = Environment.from_samples(positions, bin_size=200.0)

# ✓ Better - creates ~100 bins
env = Environment.from_samples(positions, bin_size=10.0, units='cm')
```

**Rule of thumb**: For data spanning 0-100 in each dimension, try `bin_size = data_range / 20` as a starting point.

#### Solution 2: Lower bin_count_threshold

```python
# ❌ Too strict - requires many samples per bin
env = Environment.from_samples(
    positions,
    bin_size=5.0,
    bin_count_threshold=10  # Need 10+ samples per bin
)

# ✓ More lenient - accept bins with fewer samples
env = Environment.from_samples(
    positions,
    bin_size=5.0,
    bin_count_threshold=1,  # Accept bins with 1+ sample
    units='cm'
)
```

#### Solution 3: Enable morphological operations

```python
# ❌ Gaps in coverage leave disconnected regions
env = Environment.from_samples(positions, bin_size=5.0)

# ✓ Fill in gaps and holes
env = Environment.from_samples(
    positions,
    bin_size=5.0,
    dilate=True,        # Expand active regions
    fill_holes=True,    # Fill enclosed gaps
    units='cm'
)
```

#### Solution 4: Check your data

```python
# Inspect data range
print(f"X range: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f}")
print(f"Y range: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f}")

# Look for NaN values
nan_count = np.isnan(positions).sum()
if nan_count > 0:
    print(f"Warning: {nan_count} NaN values in positions")
    positions = positions[~np.any(np.isnan(positions), axis=1)]
```

**See Also:**
- [Environment.from_samples() documentation](api/neurospatial/environment/core.md#neurospatial.environment.core.Environment.from_samples)
- [Quickstart Guide](getting-started/quickstart.md)

---

### E1002: Invalid bin_size

**Error Message Example:**
```
ValueError: [E1002] bin_size must be positive and finite, got -5.0.
Why: cell widths must be finite and positive in every spatial dimension.
Fix: pass e.g. bin_size=2.0, in the same units as the positions
```

**What This Means:**

The cell size has an invalid value. Every layout checks its cell size the
same way: `bin_size`, `hexagon_width`, `point_spacing` and `pixel_size`.
Valid values must be:

1. **Positive numbers** - Negative or zero sizes are meaningless
2. **Finite** - Not NaN or infinity
3. **Numeric** - A single number (a NumPy scalar works) or one number per
   dimension; strings and booleans are refused with `TypeError`

**Solutions:**

#### Solution 1: Use positive bin_size

```python
# ❌ Negative bin_size
env = Environment.from_samples(positions, bin_size=-5.0)

# ✓ Positive bin_size
env = Environment.from_samples(positions, bin_size=5.0, units='cm')
```

#### Solution 2: Check for NaN/inf

```python
# ❌ Invalid numeric value
bin_size = np.nan  # or np.inf
env = Environment.from_samples(positions, bin_size=bin_size)

# ✓ Valid numeric value
bin_size = 5.0
env = Environment.from_samples(positions, bin_size=bin_size, units='cm')
```

#### Solution 3: Use correct sequence format

```python
# ❌ Wrong type
env = Environment.from_samples(positions, bin_size="5.0")

# ✓ Single numeric value
env = Environment.from_samples(positions, bin_size=5.0, units='cm')

# ✓ Sequence for per-dimension bin sizes
env = Environment.from_samples(
    positions,
    bin_size=[5.0, 10.0],  # Different size per dimension
    units='cm'
)
```

**Common Mistakes:**

- Using string instead of number: `bin_size="5"` → use `bin_size=5.0`
- Forgetting units: Always specify `units='cm'` or appropriate unit
- Mixing up bin_size with number of bins: bin_size is spatial extent, not count

**See Also:**
- [Environment.from_samples() documentation](api/neurospatial/environment/core.md#neurospatial.environment.core.Environment.from_samples)
- [Layout validation module](api/neurospatial/layout/validation.md#neurospatial.layout.validation)

---

### E1006: Environment constructed directly

**Error Message:**
```
ValueError: [E1006] Environment cannot be constructed directly — use a factory method.
Fix: env = Environment.from_samples(positions, bin_size=2.0)
```
**Cause:** `Environment()` was called directly. `Environment` is always built through a factory that constructs the underlying layout/connectivity.

**Solution:** Use the factory matching your data:
```python
env = Environment.from_samples(positions, bin_size=2.0)   # position samples
# or from_polygon / from_graph / from_grid_mask / from_pixel_mask
```

---

## Environment Compatibility Errors

### E1003: Dimension mismatch

**Error Message Example:**
```
ValueError: [E1003] All sub-environments must share the same n_dims.
Env 0 has 2 dimensions, Env 1 has 3 dimensions.

Cannot combine environments with incompatible dimensionality.

Suggested fixes:
  1. Ensure all environments use the same dimensional data
  2. Verify each environment's n_dims property before creating the composite
```

**What This Means:**

You're trying to combine or compare environments with different dimensionality (e.g., mixing 2D and 3D environments). This error commonly occurs when:

1. **Calling Environment.combine** - All sub-environments must have same n_dims
2. **Comparing environments** - Alignment/comparison requires matching dimensions
3. **Stacking environments** - Cannot stack environments of different dimensionality

**Solutions:**

#### Solution 1: Check dimensions before combining

```python
# ❌ Mixing 2D and 3D environments
env_2d = Environment.from_samples(positions_2d, bin_size=5.0)  # shape (N, 2)
env_3d = Environment.from_samples(positions_3d, bin_size=5.0)  # shape (N, 3)

composite = Environment.combine({"first": env_2d, "second": env_3d})  # Error!

# ✓ Check dimensions first
print(f"env_2d dimensions: {env_2d.n_dims}")  # 2
print(f"env_3d dimensions: {env_3d.n_dims}")  # 3

# Only combine disjoint environments with matching dimensions, units and frame
composite = Environment.combine({"first": env_2d, "second": env_2d_another})
```

#### Solution 2: Project 3D to 2D

```python
# If you need to combine 3D with 2D, project first
positions_3d_projected = positions_3d[:, :2]  # Take only x, y (drop z)

env_2d_projected = Environment.from_samples(
    positions_3d_projected,
    bin_size=5.0,
    units='cm'
)

# Now can combine with other 2D environments
composite = Environment.combine({"first": env_2d, "second": env_2d_projected})
```

#### Solution 3: Create separate composites per dimensionality

```python
# Group environments by dimensionality
envs_2d = [env1, env2, env3]  # All 2D
envs_3d = [env4, env5]  # All 3D

# Create separate composites
composite_2d = Environment.combine({str(i): env for i, env in enumerate(envs_2d)})
composite_3d = Environment.combine({str(i): env for i, env in enumerate(envs_3d)})
```

**See Also:**
- [Combining environments](user-guide/composite-environments.md)
- [3D Environment Support](dimensionality_support.md)

---

## I/O Errors

### E1005: Path traversal detected

**Error Message Example:**
```
ValueError: [E1005] Path traversal detected in path: ../shared/my_environment.
Environment save and load paths may not contain '..' components; name the
location directly, for example with an absolute path.

For more information, see: https://edeno.github.io/neurospatial/errors/#e1005-path-traversal-detected
```

**What This Means:**

`to_file` and `from_file` (and `Environment.to_file` / `Environment.from_file`)
refuse a path that contains a `..` (parent directory) component.

**Why This Is Refused:**

This is input hygiene: a path written with `..` depends on the current working
directory and hides where the files go. It is **not** a security boundary. Only
literal `..` components are refused; an absolute path or a symbolic link can
still name any location. If your code saves or loads paths supplied by someone
else, check that they stay inside the directory you intend, for example by
comparing `Path(p).resolve()` with your base directory using
`Path.is_relative_to`.

**Solutions:**

#### Solution 1: Use a relative path without '..'

```python
# Refused
env.to_file("../shared/my_environment")

# Accepted
env.to_file("my_environment")
env.to_file("outputs/my_environment")
```

#### Solution 2: Use an absolute path

```python
from pathlib import Path

save_dir = Path("/Users/researcher/data/environments")
env.to_file(save_dir / "my_environment")
```

**See Also:**
- [I/O Module Documentation](api/neurospatial/io/index.md#neurospatial.io)
- [Environment serialization API](api/neurospatial/environment/core.md#neurospatial.environment.core.Environment.to_file)

---

## Getting Help

### If you encounter an error not listed here:

1. **Check the full error message** - Modern errors include diagnostic information
2. **Search the documentation** - Use the search bar at the top
3. **Check GitHub Issues** - Someone may have encountered this before: [neurospatial/issues](https://github.com/edeno/neurospatial/issues)
4. **Ask for help** - Create a new issue with:
   - Full error message
   - Minimal code to reproduce
   - Data shapes and types
   - Expected vs actual behavior

### Reporting New Errors

If you find an error that should be documented here:

1. Open an issue on GitHub
2. Tag it with `documentation` and `error-reference`
3. Include the error code (if present) and full message
4. Describe what you were trying to do
5. Suggest a solution if you found one

---

## Error Code Guidelines (For Developers)

### Adding New Error Codes

When adding new error codes to the codebase:

1. **Choose the next available code** in the appropriate range:
   - E1xxx: Environment creation/initialization errors
   - E2xxx: Data validation errors
   - E3xxx: Operation/method call errors
   - E4xxx: I/O and serialization errors
   - E5xxx: Layout engine errors

2. **Format error messages**:
   ```python
   raise ValueError(
       f"[E1001] Brief error description.\n\n"
       f"Diagnostic information here...\n\n"
       f"For more information, see: "
       f"https://edeno.github.io/neurospatial/errors/#e1001"
   )
   ```

3. **Document the error** in this file with:
   - Error code and title
   - What it means
   - Common causes
   - At least 3 solutions with code examples
   - Links to relevant documentation

4. **Add test coverage**:
   ```python
   def test_error_code_appears_in_message():
       with pytest.raises(ValueError, match=r"\[E1001\]"):
           # Code that triggers error
   ```

---

**Last Updated**: 2025-11-14
**Version**: 1.0
**Feedback**: [GitHub Issues](https://github.com/edeno/neurospatial/issues)
