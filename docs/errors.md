# Error Reference

**Last Updated**: 2025-11-14

This document provides detailed explanations and solutions for common errors in neurospatial. Each error has a unique error code for easy reference and troubleshooting.

---

## Quick Index

| Code | Error | Severity | Section |
|------|-------|----------|---------|
| [E1001](#e1001-no-active-bins-found) | No active bins found | Critical | [Environment Creation](#environment-creation-errors) |
| [E1002](#e1002-invalid-bin_size) | Invalid bin_size | Critical | [Environment Creation](#environment-creation-errors) |
| [E1003](#e1003-dimension-mismatch) | Dimension mismatch | Critical | [Environment Compatibility](#environment-compatibility-errors) |
| [E1004](#e1004-environment-not-fitted) | Environment not fitted | Critical | [Usage Errors](#usage-errors) |
| [E1005](#e1005-path-traversal-detected) | Path traversal detected | Security | [I/O Errors](#io-errors) |

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
- [Environment.from_samples() documentation](../api/environment/#neurospatial.environment.Environment.from_samples)
- [Quickstart Guide](../getting-started/quickstart.md)

---

### E1002: Invalid bin_size

**Error Message Example:**
```
ValueError: [E1002] bin_size must be positive (got -5.0).
```

**What This Means:**

The `bin_size` parameter has an invalid value. Valid bin_size values must be:

1. **Positive numbers** - Negative or zero bin_size is meaningless
2. **Finite** - Not NaN or infinity
3. **Numeric** - Either a single number or sequence of numbers

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
- [Environment.from_samples() documentation](../api/environment/#neurospatial.environment.Environment.from_samples)
- [Layout validation module](../api/layout/#neurospatial.layout.validation)

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

1. **Creating CompositeEnvironment** - All sub-environments must have same n_dims
2. **Comparing environments** - Alignment/comparison requires matching dimensions
3. **Stacking environments** - Cannot stack environments of different dimensionality

**Solutions:**

#### Solution 1: Check dimensions before combining

```python
# ❌ Mixing 2D and 3D environments
env_2d = Environment.from_samples(positions_2d, bin_size=5.0)  # shape (N, 2)
env_3d = Environment.from_samples(positions_3d, bin_size=5.0)  # shape (N, 3)

composite = CompositeEnvironment.from_environments([env_2d, env_3d])  # Error!

# ✓ Check dimensions first
print(f"env_2d dimensions: {env_2d.n_dims}")  # 2
print(f"env_3d dimensions: {env_3d.n_dims}")  # 3

# Only combine environments with matching dimensions
composite = CompositeEnvironment.from_environments([env_2d, env_2d_another])
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
composite = CompositeEnvironment.from_environments([env_2d, env_2d_projected])
```

#### Solution 3: Create separate composites per dimensionality

```python
# Group environments by dimensionality
envs_2d = [env1, env2, env3]  # All 2D
envs_3d = [env4, env5]  # All 3D

# Create separate composites
composite_2d = CompositeEnvironment.from_environments(envs_2d)
composite_3d = CompositeEnvironment.from_environments(envs_3d)
```

**See Also:**
- [CompositeEnvironment documentation](../api/composite/#neurospatial.composite.CompositeEnvironment)
- [3D Environment Support](../dimensionality_support.md)

---

## Usage Errors

### E1004: Environment not fitted

**Error Message Example:**
```
RuntimeError: [E1004] Environment.bin_at() requires the environment to be fully initialized.
Ensure it was created with a factory method.

Example (correct usage):
    env = Environment.from_samples(data, bin_size=2.0)
    result = env.bin_at(points)

Avoid:
    env = Environment()  # This will not work!

For more information, see: https://neurospatial.readthedocs.io/errors/#e1004
```

**What This Means:**

You're calling a method on an Environment that hasn't been properly initialized. The Environment class requires initialization through a **factory method** (like `from_samples()`), not through direct instantiation.

**Why This Happens:**

The Environment uses the "fitted" pattern common in scientific Python libraries (similar to scikit-learn). Direct instantiation with `Environment()` creates an empty, unfitted object that can't perform spatial operations.

**Solutions:**

#### Solution 1: Use factory methods (Recommended)

```python
# ❌ Direct instantiation doesn't work
env = Environment()
env.bin_at([10.0, 20.0])  # RuntimeError: E1004

# ✓ Use factory method
env = Environment.from_samples(positions, bin_size=5.0, units='cm')
env.bin_at([10.0, 20.0])  # Works!
```

#### Solution 2: Use the right factory for your data

```python
# For sample data
env = Environment.from_samples(positions, bin_size=5.0, units='cm')

# For 1D track graphs
env = Environment.from_graph(graph, bin_size=1.0, units='cm')

# For polygon boundaries
env = Environment.from_polygon(polygon, bin_size=5.0, units='cm')

# For boolean masks
env = Environment.from_mask(mask, dimension_ranges=ranges, units='cm')

# For image masks
env = Environment.from_image(image_path, bin_size=1.0, units='pixels')
```

#### Solution 3: Load from file

```python
# ✓ Loading from file is also a factory method
env = Environment.from_file("my_environment")
env.bin_at([10.0, 20.0])  # Works!
```

**Factory Methods Available:**
- `Environment.from_samples()` - Most common, discretizes sample data
- `Environment.from_graph()` - For 1D track linearization
- `Environment.from_polygon()` - For polygon-bounded regions
- `Environment.from_mask()` - For pre-defined boolean masks
- `Environment.from_image()` - For binary image masks
- `Environment.from_layout()` - For custom layout engines
- `Environment.from_file()` - Load from saved file

**See Also:**
- [Environment Factory Methods](../api/environment/#factory-methods)
- [Quickstart Guide](../getting-started/quickstart.md)

---

## I/O Errors

### E1005: Path traversal detected

**Error Message Example:**
```
ValueError: [E1005] Path traversal detected in path: ../../../etc/passwd.
Use absolute paths or paths without '..' components.
This restriction helps prevent security vulnerabilities.

For more information, see: https://neurospatial.readthedocs.io/errors/#e1005
```

**What This Means:**

You're trying to save or load an Environment with a path that contains `..` (parent directory references). This is blocked as a security measure to prevent path traversal attacks.

**Why This Is Blocked:**

Path traversal is a security vulnerability where an attacker could:
- Read sensitive files outside the intended directory
- Overwrite critical system files
- Access data from other users/projects

**Solutions:**

#### Solution 1: Use relative paths without '..'

```python
# ❌ Path traversal attempt
env.to_file("../../../sensitive_data")

# ✓ Use simple relative path
env.to_file("my_environment")

# ✓ Use subdirectory
env.to_file("outputs/my_environment")
```

#### Solution 2: Use absolute paths

```python
from pathlib import Path

# ✓ Absolute path is fine
save_dir = Path("/Users/researcher/data/environments")
env.to_file(save_dir / "my_environment")

# ✓ Or using os.path
import os
save_path = os.path.join(os.getcwd(), "outputs", "my_environment")
env.to_file(save_path)
```

#### Solution 3: Use Path.resolve() for user input

```python
from pathlib import Path

# When accepting user input, resolve to absolute path first
user_input = "my_environment"  # From user
safe_path = Path(user_input).resolve()

# This is now an absolute path without '..'
env.to_file(safe_path)
```

**Security Note:**

This restriction is intentional and cannot be disabled. If you have a legitimate use case requiring parent directory access, use absolute paths instead.

**See Also:**
- [I/O Module Documentation](../api/io/#neurospatial.io)
- [Environment Serialization Guide](../user-guide/workflows.md#saving-and-loading)

---

## Getting Help

### If you encounter an error not listed here:

1. **Check the full error message** - Modern errors include diagnostic information
2. **Search the documentation** - Use the search bar at the top
3. **Check GitHub Issues** - Someone may have encountered this before: [neurospatial/issues](https://github.com/yourusername/neurospatial/issues)
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
       f"https://neurospatial.readthedocs.io/errors/#e1001"
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
**Feedback**: [GitHub Issues](https://github.com/yourusername/neurospatial/issues)
