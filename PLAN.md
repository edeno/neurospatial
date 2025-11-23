# NWB Integration Plan for Neurospatial

**Status**: Planning
**Date**: 2025-11-23
**Version**: Draft 1.1

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background](#background)
3. [Design Principles](#design-principles)
4. [Data Structure Mapping](#data-structure-mapping)
5. [Integration Architecture](#integration-architecture)
6. [Container Discovery Rules](#container-discovery-rules)
7. [Implementation Plan](#implementation-plan)
8. [API Design](#api-design)
9. [Dependencies](#dependencies)
10. [Testing Strategy](#testing-strategy)
11. [Future Extension](#future-extension)

---

## Executive Summary

This document outlines a plan for integrating neurospatial with the Neurodata Without Borders (NWB) format. The integration will enable:

1. **Reading** spatial/behavioral data from NWB files into neurospatial structures
2. **Writing** neurospatial analysis results back to NWB files
3. **Round-trip** preservation of metadata and provenance

**Key design decisions:**

- Use **standard NWB types only** (no custom extension initially)
- Store data in **appropriate standard locations** (`behavior`, `analysis`, `scratch`)
- Use **edge list** as canonical graph representation (not CSR)
- Design on-disk layout so a future extension is a **thin typed wrapper**
- Any future `ndx-spatial-environment` extension lives in a **separate repository**

---

## Background

### What is NWB?

Neurodata Without Borders (NWB) is a standardized data format for neurophysiology, built on HDF5. Key features:

- **Standardized containers** for common data types (time series, electrodes, behavior)
- **Extension mechanism** for domain-specific data via ndx-* packages
- **Rich metadata** for experimental provenance
- **Cross-platform** compatibility via HDF5

### Why NWB Integration?

1. **Data sharing**: NWB is the standard for DANDI Archive and many neuroscience labs
2. **Interoperability**: Connect with other NWB-aware tools (suite2p, CaImAn, etc.)
3. **Provenance**: Track analysis pipeline from raw data to results
4. **Spyglass compatibility**: Enable seamless integration with Spyglass pipelines

### Relevant NWB Extensions

| Extension | Purpose | Neurospatial Relevance |
|-----------|---------|----------------------|
| **ndx-pose** | Pose estimation (DeepLabCut, SLEAP) | Skeleton, BodypartOverlay |
| **ndx-events** | Timestamped discrete events | Laps, region crossings, trials |
| **pynwb.behavior** | Position tracking, compass direction | PositionOverlay, HeadDirectionOverlay |
| **ndx-structured-behavior** | Behavioral task structure | Region definitions, trial structure |

---

## Design Principles

### 1. Standard Types First

Use only standard NWB types (groups, datasets, DynamicTable) initially. This ensures:

- **Immediate usability** without installing custom extensions
- **Broad compatibility** with existing NWB tools
- **Lower barrier** for adoption

### 2. Appropriate Standard Locations

Don't create a single "neurospatial" processing module. Instead, use NWB's standard organization:

| Data Type | NWB Location | Rationale |
|-----------|--------------|-----------|
| Lap events | `processing/behavior/` | Behavioral segmentation |
| Region crossings | `processing/behavior/` | Behavioral events |
| Place fields | `analysis/` | Derived neural analysis |
| Occupancy maps | `analysis/` | Derived behavioral analysis |
| Environment (no standard place) | `scratch/` | Temporary/custom data |

### 3. Edge List as Canonical Graph Format

For NWB (multi-language, archival), use edge list as the canonical representation:

```
# Dataset: edges, shape (n_edges, 2), dtype int64
# Dataset: edge_weights, shape (n_edges,), dtype float64 (optional)
# Attribute: directed = False
# Attribute: weight_units = "cm"
```

**Why edge list over CSR:**

- Simple, self-explanatory, easy to consume outside Python
- Works with any language (MATLAB, Julia, R)
- CSR can be reconstructed on load (cost is negligible vs. pipeline)

**Optional CSR storage:** If needed for very large graphs, store as additional datasets with clear `encoding="csr"` attribute, but never as the only representation.

### 4. Extension-Ready Design

Design the Python API and on-disk layout so that a future `ndx-spatial-environment` extension is:

- A **thin typed wrapper** around what we already store
- Not a completely different schema requiring migration
- Developed in a **separate repository** (not bundled with neurospatial)

---

## Data Structure Mapping

### Neurospatial → NWB Mapping

| Neurospatial Type | NWB Type | Location in NWB | Notes |
|-------------------|----------|-----------------|-------|
| `Environment` | Group + Datasets | `scratch/spatial_environment/` | Standard types, no extension needed |
| `Environment.bin_centers` | Dataset | `scratch/.../bin_centers` | Shape: (n_bins, n_dims), float64 |
| `Environment.connectivity` | Dataset (edge list) | `scratch/.../edges` | Shape: (n_edges, 2), int64 |
| Edge weights | Dataset | `scratch/.../edge_weights` | Shape: (n_edges,), float64 |
| `Region` (point) | DynamicTable row | `scratch/.../regions` | Columns: name, kind, x, y, z |
| `Region` (polygon) | DynamicTable row | `scratch/.../regions` | Columns: name, kind, vertices (ragged) |
| `Skeleton` | `ndx_pose.Skeleton` | `behavior/Skeletons` | Direct mapping (existing support) |
| Place field | `TimeSeries` | `analysis/place_fields/` | Shape: (n_bins,) or (n_time, n_bins) |
| Occupancy | `TimeSeries` | `analysis/occupancy/` | Shape: (n_bins,), with bin_centers ref |
| Lap events | `ndx_events.EventsTable` | `processing/behavior/laps` | Columns: timestamp, direction, duration |
| Region crossings | `ndx_events.EventsTable` | `processing/behavior/region_crossings` | Columns: timestamp, region, event_type |

### NWB → Neurospatial Mapping

| NWB Type | Neurospatial Type | Conversion Notes |
|----------|-------------------|------------------|
| `pynwb.behavior.Position` | `PositionOverlay.data` | Extract SpatialSeries data + timestamps |
| `pynwb.behavior.CompassDirection` | `HeadDirectionOverlay.data` | Extract theta values |
| `ndx_pose.PoseEstimation` | `BodypartOverlay` | Convert PoseEstimationSeries → dict |
| `ndx_pose.Skeleton` | `Skeleton` | Already supported via `Skeleton.from_ndx_pose()` |
| `SpatialSeries` (position) | Trajectory array | Shape: (n_samples, n_dims) |

---

## Integration Architecture

### Module Structure

```
src/neurospatial/
├── nwb/                          # New NWB integration module
│   ├── __init__.py               # Public API exports
│   ├── _core.py                  # Core utilities (discovery, validation)
│   ├── _environment.py           # Environment ↔ scratch (read/write)
│   ├── _pose.py                  # Pose/Skeleton ↔ ndx-pose
│   ├── _behavior.py              # Position/Head direction ↔ pynwb.behavior
│   ├── _events.py                # Events ↔ ndx-events
│   ├── _fields.py                # Spatial fields (place fields, occupancy)
│   └── _overlays.py              # Overlay factory functions (*_from_nwb)
```

Note: Any future `ndx-spatial-environment` extension lives in a **separate repository**, not bundled with neurospatial.

### Dependency Strategy

```python
# Lazy imports to keep NWB optional
def read_position_from_nwb(nwbfile):
    """Read position data from NWB file."""
    try:
        import pynwb
    except ImportError:
        raise ImportError(
            "pynwb is required for NWB integration. "
            "Install with: pip install pynwb"
        )
    # ... implementation
```

### Two-Phase Approach

#### Phase 1: Read/Write with standard types

- Read existing NWB data into neurospatial structures
- Write analysis results using standard NWB types
- Store Environment in `scratch/` for round-trip capability

#### Phase 2: Optional extension (separate repository)

- Create `ndx-spatial-environment` in its own repo (if community interest)
- Thin wrapper around existing on-disk layout
- `Environment.to_nwb()` auto-detects extension availability

---

## Container Discovery Rules

### Explicit Rules for Reading

When multiple containers of the same type exist, reading functions need deterministic behavior. These rules define discovery order and failure modes.

### Type-Based Search Pattern

Use `isinstance()` checks for robust discovery that doesn't depend on naming conventions:

```python
from pynwb.behavior import Position, SpatialSeries, CompassDirection

def _find_containers_by_type(nwbfile: NWBFile, target_type: type) -> list[tuple[str, Any]]:
    """
    Find all containers of a given type anywhere in the NWB file.

    Returns list of (path, container) tuples, sorted alphabetically by path.
    """
    found = []

    # Search processing modules
    for mod_name, module in nwbfile.processing.items():
        for obj_name, obj in module.data_interfaces.items():
            if isinstance(obj, target_type):
                found.append((f"processing/{mod_name}/{obj_name}", obj))

    # Search acquisition
    for obj_name, obj in nwbfile.acquisition.items():
        if isinstance(obj, target_type):
            found.append((f"acquisition/{obj_name}", obj))

    return sorted(found, key=lambda x: x[0])
```

#### Position Discovery

```python
def read_position(
    nwbfile: NWBFile,
    processing_module: str | None = None,  # None = auto-discover
    position_name: str | None = None,       # None = first found
) -> tuple[NDArray, NDArray]:
    """
    Discovery strategy:
    1. If processing_module specified: look only there
    2. Otherwise: type-based search for Position containers
    3. Priority order: processing/behavior > processing/* > acquisition

    When position_name is None:
    - If Position contains one SpatialSeries: use it
    - If multiple: use first alphabetically, emit INFO log

    Failure modes:
    - KeyError("No Position found in {searched_locations}")
    - KeyError("Position '{name}' not found. Available: {list}")
    """
```

#### Pose Estimation Discovery

```python
def read_pose(
    nwbfile: NWBFile,
    pose_estimation_name: str | None = None,  # None = auto-discover
) -> tuple[dict[str, NDArray], NDArray, Skeleton]:
    """
    Discovery strategy:
    1. Type-based search: find all PoseEstimation instances via isinstance()
    2. Priority order: processing/behavior > processing/*

    When multiple PoseEstimation exist and name is None:
    - Use first alphabetically, emit INFO log
    - Log message includes all available names

    Failure modes:
    - KeyError("No PoseEstimation found")
    - KeyError("PoseEstimation '{name}' not found. Available: {list}")
    - ImportError("ndx-pose required for pose data")
    """
```

#### Events Discovery

```python
def read_events(
    nwbfile: NWBFile,
    table_name: str,  # Required - no auto-discovery for events
    processing_module: str = "behavior",
) -> DataFrame:
    """
    Events are explicitly named (no auto-discovery) because:
    - Multiple event tables are common
    - Semantic meaning varies (laps vs stimuli vs rewards)

    Failure modes:
    - KeyError("EventsTable '{name}' not found in {module}")
    - ImportError("ndx-events required for EventsTable")
    """
```

### Logging Strategy

```python
import logging

logger = logging.getLogger("neurospatial.nwb")

# INFO: Successful auto-discovery with implicit choices
logger.info("Using Position from processing/behavior (first of 2 found)")

# WARNING: Potentially ambiguous situations
logger.warning("Multiple PoseEstimation found: %s. Using '%s'", names, chosen)

# DEBUG: Discovery process details
logger.debug("Searching for Position in %s", searched_locations)
```

### Validation on Write

```python
def write_place_field(
    nwbfile: NWBFile,
    env: Environment,
    field: NDArray,
    name: str,
    *,
    overwrite: bool = False,  # Explicit overwrite required
) -> None:
    """
    Failure modes:
    - ValueError("Place field '{name}' already exists. Use overwrite=True")
    - ValueError("Field shape {shape} doesn't match env.n_bins={n}")
    """
```

---

## Implementation Plan

### Phase 1: Core NWB Reading (Priority: High)

#### 1.1 Position/Trajectory Reading

```python
def read_position(
    nwbfile: NWBFile,
    processing_module: str = "behavior",
    position_name: str | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Read position data from NWB file.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    timestamps : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    """
```

#### 1.2 Pose/Skeleton Reading

```python
def read_pose(
    nwbfile: NWBFile,
    pose_estimation_name: str | None = None,
) -> tuple[dict[str, NDArray], NDArray[np.float64], Skeleton]:
    """
    Read pose estimation data from NWB file.

    Returns
    -------
    bodyparts : dict[str, NDArray], each shape (n_samples, n_dims)
        Mapping from bodypart name to coordinates.
    timestamps : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    skeleton : Skeleton
        Skeleton definition from ndx-pose.
    """
```

#### 1.3 Events Reading

```python
def read_events(
    nwbfile: NWBFile,
    table_name: str,
) -> DataFrame:
    """
    Read events table from NWB file.

    Returns pandas DataFrame with timestamp column and any additional columns.
    """
```

### Phase 2: NWB Writing (Priority: Medium)

#### 2.1 Writing Spatial Fields

```python
def write_place_field(
    nwbfile: NWBFile,
    env: Environment,
    field: NDArray[np.float64],
    name: str = "place_field",
    description: str = "",
) -> None:
    """
    Write spatial field to NWB file.

    Stores field values aligned with environment bin centers.
    """
```

#### 2.2 Writing Events

```python
def write_laps(
    nwbfile: NWBFile,
    lap_times: NDArray[np.float64],
    lap_types: NDArray[np.int_] | None = None,
    description: str = "Detected lap events",
) -> None:
    """Write lap detection results to NWB EventsTable."""
```

### Phase 3: Environment Round-trip (Priority: Medium)

#### 3.1 Writing Environment to Scratch

```python
def write_environment(
    nwbfile: NWBFile,
    env: Environment,
    name: str = "spatial_environment",
) -> None:
    """
    Write Environment to NWB scratch space using standard types.

    Creates structure:
        scratch/{name}/
            bin_centers       # Dataset (n_bins, n_dims)
            edges             # Dataset (n_edges, 2) - edge list
            edge_weights      # Dataset (n_edges,) - optional
            dimension_ranges  # Dataset (n_dims, 2)
            regions           # DynamicTable with point/polygon data
            metadata.json     # Dataset (string) - JSON blob for extras

    Attributes on group:
        units, frame, n_dims, layout_type
    """
```

#### 3.2 Reading Environment from Scratch

```python
def read_environment(
    nwbfile: NWBFile,
    name: str = "spatial_environment",
) -> Environment:
    """
    Read Environment from NWB scratch space.

    Reconstructs Environment from stored bin_centers and edge list.
    Rebuilds connectivity graph and regions.
    """
```

---

## API Design

### Public API

```python
# neurospatial/nwb/__init__.py

from neurospatial.nwb import (
    # === Reading Functions ===

    # Position/trajectory
    read_position,               # Position → (positions, timestamps)
    read_head_direction,         # CompassDirection → (angles, timestamps)

    # Pose estimation (requires ndx-pose)
    read_pose,                   # PoseEstimation → (bodyparts, timestamps, skeleton)

    # Events (requires ndx-events)
    read_events,                 # EventsTable → DataFrame

    # Environment (from scratch space)
    read_environment,            # scratch/env → Environment

    # === Writing Functions ===

    # Spatial fields → analysis/
    write_place_field,           # Write place field to analysis/place_fields/
    write_occupancy,             # Write occupancy map to analysis/occupancy/

    # Events → processing/behavior/
    write_laps,                  # Write lap events
    write_region_crossings,      # Write region crossing events

    # Environment → scratch/ (standard types, no extension needed)
    write_environment,           # Write Environment to scratch space

    # === Factory Functions ===

    # Create Environment from NWB data
    environment_from_position,   # Create Environment from Position + bin_size

    # Create overlays for animation
    position_overlay_from_nwb,   # PositionOverlay from NWB Position
    bodypart_overlay_from_nwb,   # BodypartOverlay from ndx-pose
    head_direction_overlay_from_nwb,  # HeadDirectionOverlay from CompassDirection
)
```

### Usage Examples

#### Reading Position Data

```python
from pynwb import NWBHDF5IO
from neurospatial import Environment
from neurospatial.nwb import read_position, environment_from_position

# Open NWB file
with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()

    # Read position data
    positions, timestamps = read_position(nwbfile)

    # Create environment from position data
    env = environment_from_position(
        nwbfile,
        bin_size=2.0,
        units="cm",
    )

    # Or use existing Environment factory with NWB data
    env = Environment.from_samples(positions, bin_size=2.0)
```

#### Reading Pose Data for Animation

```python
from neurospatial.nwb import bodypart_overlay_from_nwb, position_overlay_from_nwb

with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()

    # Create overlays directly from NWB
    position_overlay = position_overlay_from_nwb(
        nwbfile,
        color="red",
        trail_length=10,
    )

    bodypart_overlay = bodypart_overlay_from_nwb(
        nwbfile,
        pose_estimation_name="PoseEstimation",
        skeleton_color="white",
    )

# Use in animation
env.animate_fields(
    fields,
    overlays=[position_overlay, bodypart_overlay],
    backend="napari",
)
```

#### Writing Analysis Results

```python
from pynwb import NWBHDF5IO
from neurospatial import compute_place_field, detect_laps
from neurospatial.nwb import write_place_field, write_laps, write_environment

# Compute analysis results
place_field = compute_place_field(env, spike_times, timestamps, positions)
laps = detect_laps(env, timestamps, positions, region_name="start")

# Write to NWB
with NWBHDF5IO("session.nwb", "r+") as io:
    nwbfile = io.read()

    # Write place field to analysis/ (standard location for derived results)
    write_place_field(
        nwbfile,
        env,
        place_field,
        name="cell_001",
        description="Place field for cell 001",
    )

    # Write laps to processing/behavior/ (standard location for behavioral events)
    write_laps(
        nwbfile,
        laps["start_time"].values,
        lap_types=laps["direction"].values,
    )

    # Write environment to scratch/ (for round-trip capability)
    write_environment(nwbfile, env, name="linear_track")

    io.write(nwbfile)
```

### Environment Method Integration

```python
# Add NWB methods directly to Environment class
class Environment:
    # ... existing methods ...

    @classmethod
    def from_nwb(
        cls,
        nwbfile: NWBFile,
        *,
        # Option 1: Create from position data
        bin_size: float | None = None,
        position_name: str | None = None,
        # Option 2: Load from scratch (round-trip)
        scratch_name: str | None = None,
        **kwargs,
    ) -> Environment:
        """
        Create Environment from NWB file.

        Two modes:
        1. From position data: provide bin_size to discretize Position data
        2. From scratch: provide scratch_name to load previously saved Environment
        """
        from neurospatial.nwb import environment_from_position, read_environment

        if scratch_name is not None:
            return read_environment(nwbfile, name=scratch_name)
        elif bin_size is not None:
            return environment_from_position(
                nwbfile, bin_size=bin_size,
                position_name=position_name, **kwargs
            )
        else:
            raise ValueError("Provide either bin_size (to create from position) "
                           "or scratch_name (to load saved environment)")

    def to_nwb(
        self,
        nwbfile: NWBFile,
        name: str = "spatial_environment",
    ) -> None:
        """Write Environment to NWB scratch space."""
        from neurospatial.nwb import write_environment
        write_environment(nwbfile, self, name=name)
```

---

## Dependencies

### Required (for NWB functionality)

```toml
# pyproject.toml
[project.optional-dependencies]
nwb = [
    "pynwb>=2.5.0",
    "hdmf>=3.0.0",
]

# For pose data support
nwb-pose = [
    "pynwb>=2.5.0",
    "ndx-pose>=0.2.0",
]

# For events support
nwb-events = [
    "pynwb>=2.5.0",
    "ndx-events>=0.4.0",
]

# Full NWB support
nwb-full = [
    "pynwb>=2.5.0",
    "hdmf>=3.0.0",
    "ndx-pose>=0.2.0",
    "ndx-events>=0.4.0",
]
```

### Installation

```bash
# Basic NWB support
pip install neurospatial[nwb]

# Full NWB support with all extensions
pip install neurospatial[nwb-full]
```

---

## Testing Strategy

### Unit Tests

```python
# tests/nwb/test_position.py
def test_read_position_basic(sample_nwb_with_position):
    """Test reading position from NWB file."""
    positions, timestamps = read_position(sample_nwb_with_position)
    assert positions.shape == (1000, 2)
    assert timestamps.shape == (1000,)

def test_read_position_missing_raises():
    """Test appropriate error when position data missing."""
    nwbfile = create_empty_nwb()
    with pytest.raises(KeyError, match="No Position data found"):
        read_position(nwbfile)
```

### Integration Tests

```python
# tests/nwb/test_roundtrip.py
def test_environment_roundtrip(tmp_path, sample_environment):
    """Test Environment survives NWB round-trip."""
    nwb_path = tmp_path / "test.nwb"

    # Write
    with NWBHDF5IO(nwb_path, "w") as io:
        nwbfile = create_nwb_file()
        sample_environment.to_nwb(nwbfile)
        io.write(nwbfile)

    # Read
    with NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        loaded = Environment.from_nwb(nwbfile)

    assert loaded.n_bins == sample_environment.n_bins
    np.testing.assert_array_almost_equal(
        loaded.bin_centers, sample_environment.bin_centers
    )
```

### Test Fixtures

```python
# tests/nwb/conftest.py
@pytest.fixture
def sample_nwb_with_position():
    """Create NWB file with Position data for testing."""
    from pynwb import NWBFile
    from pynwb.behavior import Position, SpatialSeries
    from datetime import datetime

    nwbfile = NWBFile(
        session_description="Test session",
        identifier="test_001",
        session_start_time=datetime.now(),
    )

    # Add position data
    position = Position(name="Position")
    spatial_series = SpatialSeries(
        name="position",
        data=np.random.uniform(0, 100, (1000, 2)),
        timestamps=np.arange(1000) / 30.0,
        reference_frame="arena corner",
        unit="cm",
    )
    position.add_spatial_series(spatial_series)

    behavior = nwbfile.create_processing_module(
        name="behavior", description="Behavioral data"
    )
    behavior.add(position)

    return nwbfile
```

---

## Future Extension

### ndx-spatial-environment (Separate Repository)

If community interest warrants a formal extension, it should:

1. **Live in a separate repository** (e.g., `rly/ndx-spatial-environment` or similar)
2. **Wrap the existing on-disk layout** defined in this document
3. **Add typed Python classes** for IDE autocomplete and validation
4. **Not change the underlying structure** - migration should be trivial

### Extension Schema (Draft)

```yaml
# Would live in ndx-spatial-environment repo, NOT in neurospatial
groups:
  - neurodata_type_def: SpatialEnvironment
    neurodata_type_inc: NWBDataInterface
    doc: Discretized spatial environment for neural analysis.
    attributes:
      - name: units
        dtype: text
        doc: Spatial units (cm, m, pixels)
      - name: frame
        dtype: text
        required: false
        doc: Coordinate frame reference
      - name: n_dims
        dtype: int32
        doc: Number of spatial dimensions
    datasets:
      - name: bin_centers
        dtype: float64
        dims: [n_bins, n_dims]
        doc: Center coordinates of each spatial bin
      - name: edges
        dtype: int64
        dims: [n_edges, 2]
        doc: Edge list for connectivity graph (node index pairs)
      - name: edge_weights
        dtype: float64
        dims: [n_edges]
        quantity: "?"
        doc: Optional edge weights (typically distances)
      - name: dimension_ranges
        dtype: float64
        dims: [n_dims, 2]
        doc: Min/max extent in each dimension
    groups:
      - name: regions
        neurodata_type_inc: DynamicTable
        doc: Named regions of interest
        quantity: "?"
```

### Community Coordination

Before creating an extension:

1. Check NWB Slack/GitHub for existing spatial discretization work
2. Coordinate with Spyglass team for compatibility requirements
3. Review with NWB maintainers for schema best practices

### Future Enhancements

1. **Streaming support**: Handle very large NWB files that don't fit in memory
2. **Cloud storage**: Direct reading from DANDI Archive URLs via fsspec
3. **Parallel I/O**: Multi-threaded reading for large datasets
4. **Zarr backend**: Support NWB-Zarr for cloud-native storage
5. **Lazy loading**: Load data on access rather than eagerly

---

## Implementation Timeline

| Phase | Scope | Effort | Dependencies |
|-------|-------|--------|--------------|
| 1.1 | Position reading | Small | pynwb |
| 1.2 | Pose/Skeleton reading | Small | pynwb, ndx-pose |
| 1.3 | Events reading | Small | pynwb, ndx-events |
| 1.4 | Overlay helpers (`*_from_nwb`) | Small | Phase 1.1-1.3 |
| 2.1 | Place field writing | Medium | pynwb |
| 2.2 | Events writing (laps, crossings) | Small | pynwb, ndx-events |
| 3.1 | Environment to scratch | Medium | pynwb |
| 3.2 | Environment from scratch | Medium | Phase 3.1 |
| 3.3 | Environment.from_nwb/to_nwb | Small | Phase 3.1-3.2 |
| 4.1 | ndx-spatial-environment (separate repo) | Large | Community feedback |

---

## References

- [PyNWB Documentation](https://pynwb.readthedocs.io/en/stable/)
- [ndx-pose Repository](https://github.com/rly/ndx-pose)
- [ndx-events Repository](https://github.com/rly/ndx-events)
- [NWB Schema Language](https://schema-language.readthedocs.io/)
- [DANDI Archive](https://dandiarchive.org/)
- [NWB Best Practices](https://www.nwb.org/best-practices/)
