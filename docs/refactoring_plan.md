# Neurospatial Refactoring Plan

**Created**: 2025-10-31
**Based on**: Code Review 2025-10-31
**Target Completion**: 2-3 weeks (estimated)
**Priority**: Structured incremental improvement

---

## Table of Contents

1. [Overview & Philosophy](#overview--philosophy)
2. [Phase 1: Critical Fixes (Week 1)](#phase-1-critical-fixes-week-1)
3. [Phase 2: Test Coverage Enhancement (Week 1-2)](#phase-2-test-coverage-enhancement-week-1-2)
4. [Phase 3: Complexity Reduction (Week 2-3)](#phase-3-complexity-reduction-week-2-3)
5. [Phase 4: Documentation Polish (Week 3)](#phase-4-documentation-polish-week-3)
6. [Success Metrics](#success-metrics)
7. [Risk Mitigation](#risk-mitigation)

---

## Overview & Philosophy

### Refactoring Principles

1. **No Breaking Changes**: All refactoring maintains backward compatibility
2. **Test-Driven**: Write tests before refactoring complex functions
3. **Incremental**: Small, reviewable changes
4. **Documented**: Update docs as code changes
5. **Measured**: Track metrics to verify improvements

### Current State vs. Goals

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| Overall Test Coverage | 61% | 80%+ | High |
| Core Module Coverage | 81% | 90%+ | Medium |
| Utility Module Coverage | 0-23% | 70%+ | Critical |
| Type Hint Completeness | 85-90% | 95%+ | High |
| Functions >50 lines | 60 (32%) | <20 (11%) | Medium |
| Functions complexity >10 | 23 (12%) | <10 (5%) | High |
| Docstring Completeness | 75-85% | 90%+ | Medium |

---

## Phase 1: Critical Fixes (Week 1)

**Goal**: Fix blocking issues that affect code quality and maintainability
**Estimated Time**: 12-16 hours
**Priority**: CRITICAL

### 1.1 Type Hints Completion [2-3 hours]

#### Fix Missing Type Hints

**File**: `src/neurospatial/layout/engines/hexagonal.py`

```python
# Line 242 - BEFORE
def point_to_bin_index(self, points):
    """Map continuous N-D points to discrete active bin indices."""
    # ...

# Line 242 - AFTER
def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
    """Map continuous N-D points to discrete active bin indices."""
    # ...
```

**File**: `src/neurospatial/layout/engines/graph.py`

```python
# Line 329 - BEFORE
def linear_point_to_bin_ind(self, data_points):
    """Map 1D linear coordinates to bin indices."""
    # ...

# Line 329 - AFTER
def linear_point_to_bin_ind(
    self, data_points: NDArray[np.float64]
) -> NDArray[np.int_]:
    """Map 1D linear coordinates to bin indices."""
    # ...
```

**File**: `src/neurospatial/layout/helpers/graph.py`

```python
# Line 416 - BEFORE
def _find_bin_for_linear_position(
    linear_position: np.ndarray,
    bin_centers_linear: np.ndarray,
    active_mask: np.ndarray = None,
) -> np.ndarray:
    # ...

# Line 416 - AFTER
def _find_bin_for_linear_position(
    linear_position: NDArray[np.float64],
    bin_centers_linear: NDArray[np.float64],
    active_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.int_]:
    # ...
```

**File**: `src/neurospatial/layout/helpers/utils.py`

```python
# Multiple locations - Replace np.ndarray with NDArray[dtype]
# Line 687, 836 and others

# BEFORE
active_mask: np.ndarray = None

# AFTER
active_mask: NDArray[np.bool_] | None = None
```

**Testing**:

```bash
# Verify no runtime changes
uv run pytest

# Optional: Add mypy configuration
uv run mypy src/neurospatial --check-untyped-defs
```

**Success Criteria**:

- [ ] All functions have complete type hints
- [ ] No runtime behavior changes
- [ ] Tests still pass

---

### 1.2 Docstring Fixes [2-3 hours]

#### Fix Critical Docstring Issues

**1. Fix Parameter Name Mismatch**

**File**: `src/neurospatial/layout/mixins.py`

```python
# Line 244 - BEFORE in docstring
"""
Parameters
----------
...
draw_connectivity_graph : bool, optional
    Whether to draw connectivity edges.
...
"""

# Line 244 - AFTER in docstring
"""
Parameters
----------
...
show_connectivity : bool, optional
    Whether to draw connectivity edges.
...
"""
```

**2. Fix Incorrect Parameter Description**

**File**: `src/neurospatial/alignment.py`

```python
# Line 248 - BEFORE in docstring
"""
source_rotation_matrix : NDArray[np.float64], optional
    Rotation matrix to apply to source bin centers before mapping.
    Must be a length-n_dims array. If None, no rotation is applied.
    Defaults to None.
"""

# Line 248 - AFTER in docstring
"""
source_rotation_matrix : NDArray[np.float64], optional
    Rotation matrix to apply to source bin centers before mapping.
    Must be a 2x2 rotation matrix (shape (2, 2)) for 2D environments.
    If None, no rotation is applied. Defaults to None.
"""
```

**3. Add Missing Attributes Sections**

**File**: `src/neurospatial/composite.py`

```python
# Line 30 - ADD to class docstring
"""
Represents a composite spatial environment made from multiple sub-environments.

Attributes
----------
environments : List[Environment]
    List of constituent Environment instances that make up the composite.
name : str
    Name for the composite environment.
layout : LayoutEngine
    Combined layout engine representing the merged space.
bin_centers : NDArray[np.float64]
    Combined bin centers from all environments.
connectivity : nx.Graph
    Combined connectivity graph with bridge edges between environments.
bridges : List[Tuple[int, int, Dict]]
    List of bridge edges connecting different environments.
    Each tuple is (source_bin, target_bin, edge_attributes).
dimension_ranges : Sequence[Tuple[float, float]]
    Combined dimension ranges across all environments.
_environment_bin_ranges : Dict[str, Tuple[int, int]]
    Mapping of environment names to their bin index ranges in the composite.
"""
```

**File**: `src/neurospatial/transforms.py`

```python
# Line 36 - ADD to class docstring
"""
2D affine transformation with composable operations.

Attributes
----------
A : NDArray[np.float64], shape (3, 3)
    Homogeneous transformation matrix representing the affine transformation.
    The matrix encodes rotation, scaling, translation, and shear operations.

Examples
--------
>>> transform = Affine2D.identity().translate(10, 20).scale(2.0)
>>> points = np.array([[0, 0], [1, 1]])
>>> transformed = transform(points)
"""
```

**Testing**:

```bash
# Verify docstrings are correct
uv run python -m pydoc neurospatial.alignment
uv run python -m pydoc neurospatial.transforms
uv run python -m pydoc neurospatial.composite
```

**Success Criteria**:

- [ ] All parameter names match implementation
- [ ] All descriptions are accurate
- [ ] Attributes sections complete

---

### 1.3 Zero-Coverage Modules - Add Basic Tests [6-8 hours]

#### 1.3.1 Add Tests for calibration.py

**Create**: `tests/test_calibration.py`

```python
"""Tests for calibration module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.calibration import simple_scale


class TestSimpleScale:
    """Tests for simple_scale function."""

    def test_basic_scaling(self):
        """Test basic 1:1 scaling."""
        positions = np.array([[0, 0], [1, 1], [2, 2]])
        scale_factor = 2.0

        result = simple_scale(positions, scale_factor)
        expected = np.array([[0, 0], [2, 2], [4, 4]])

        assert_allclose(result, expected)

    def test_2d_array_scaling(self):
        """Test scaling with 2D array of positions."""
        positions = np.array([[10, 20], [30, 40]])
        scale_factor = 0.5

        result = simple_scale(positions, scale_factor)
        expected = np.array([[5, 10], [15, 20]])

        assert_allclose(result, expected)

    def test_identity_scaling(self):
        """Test that scale factor of 1.0 returns unchanged positions."""
        positions = np.array([[1.5, 2.5], [3.5, 4.5]])

        result = simple_scale(positions, 1.0)

        assert_allclose(result, positions)

    def test_zero_positions(self):
        """Test scaling of zero positions."""
        positions = np.array([[0, 0], [0, 0]])

        result = simple_scale(positions, 5.0)

        assert_allclose(result, positions)

    def test_negative_positions(self):
        """Test scaling with negative coordinates."""
        positions = np.array([[-1, -2], [3, 4]])
        scale_factor = 2.0

        result = simple_scale(positions, scale_factor)
        expected = np.array([[-2, -4], [6, 8]])

        assert_allclose(result, expected)

    def test_invalid_scale_factor(self):
        """Test that zero or negative scale factors raise error."""
        positions = np.array([[1, 2]])

        with pytest.raises(ValueError, match="scale_factor must be positive"):
            simple_scale(positions, 0.0)

        with pytest.raises(ValueError, match="scale_factor must be positive"):
            simple_scale(positions, -1.0)

    def test_1d_positions(self):
        """Test that 1D positions raise appropriate error."""
        positions = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="positions must be 2D"):
            simple_scale(positions, 2.0)

    def test_empty_positions(self):
        """Test scaling with empty array."""
        positions = np.empty((0, 2))

        result = simple_scale(positions, 2.0)

        assert result.shape == (0, 2)
```

**Estimated time**: 2-3 hours

---

#### 1.3.2 Add Tests for distance.py

**Create**: `tests/test_distance.py`

```python
"""Tests for distance module."""

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.distance import euclidean_distance_matrix, geodesic_distance_matrix


class TestEuclideanDistanceMatrix:
    """Tests for euclidean_distance_matrix function."""

    def test_basic_2d_distances(self):
        """Test basic 2D Euclidean distance calculation."""
        points = np.array([[0, 0], [3, 4], [6, 8]])

        result = euclidean_distance_matrix(points)

        # Distance from [0,0] to [3,4] is 5.0
        # Distance from [0,0] to [6,8] is 10.0
        # Distance from [3,4] to [6,8] is 5.0
        expected = np.array([
            [0.0, 5.0, 10.0],
            [5.0, 0.0, 5.0],
            [10.0, 5.0, 0.0]
        ])

        assert_allclose(result, expected)

    def test_single_point(self):
        """Test distance matrix for single point."""
        points = np.array([[1, 2]])

        result = euclidean_distance_matrix(points)
        expected = np.array([[0.0]])

        assert_allclose(result, expected)

    def test_3d_distances(self):
        """Test 3D Euclidean distance calculation."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        result = euclidean_distance_matrix(points)

        # All off-diagonal distances should be 1.0
        assert_allclose(np.diag(result), 0.0)
        assert_allclose(result[0, 1], 1.0)
        assert_allclose(result[0, 2], 1.0)
        assert_allclose(result[0, 3], 1.0)

    def test_symmetry(self):
        """Test that distance matrix is symmetric."""
        points = np.random.rand(5, 2)

        result = euclidean_distance_matrix(points)

        assert_allclose(result, result.T)

    def test_empty_points(self):
        """Test with empty points array."""
        points = np.empty((0, 2))

        result = euclidean_distance_matrix(points)

        assert result.shape == (0, 0)


class TestGeodesicDistanceMatrix:
    """Tests for geodesic_distance_matrix function."""

    def test_basic_graph_distances(self):
        """Test basic geodesic distances on simple graph."""
        # Create simple line graph: 0--1--2--3
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(1, 2, distance=1.0)
        graph.add_edge(2, 3, distance=1.0)

        result = geodesic_distance_matrix(graph, weight="distance")

        expected = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0]
        ])

        assert_allclose(result, expected)

    def test_disconnected_graph(self):
        """Test geodesic distances with disconnected components."""
        # Create two disconnected nodes: 0--1  2--3
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(2, 3, distance=1.0)

        result = geodesic_distance_matrix(graph, weight="distance")

        # Disconnected nodes should have inf distance
        assert result[0, 2] == np.inf
        assert result[0, 3] == np.inf
        assert result[1, 2] == np.inf
        assert result[1, 3] == np.inf

        # Connected nodes should have finite distance
        assert result[0, 1] == 1.0
        assert result[2, 3] == 1.0

    def test_weighted_edges(self):
        """Test geodesic distances with different edge weights."""
        # Triangle with different weights
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(1, 2, distance=1.0)
        graph.add_edge(0, 2, distance=5.0)  # Longer direct path

        result = geodesic_distance_matrix(graph, weight="distance")

        # Shortest path from 0 to 2 should be via node 1 (distance 2.0)
        assert_allclose(result[0, 2], 2.0)

    def test_single_node_graph(self):
        """Test with single isolated node."""
        graph = nx.Graph()
        graph.add_node(0)

        result = geodesic_distance_matrix(graph, weight="distance")
        expected = np.array([[0.0]])

        assert_allclose(result, expected)

    def test_empty_graph(self):
        """Test with empty graph."""
        graph = nx.Graph()

        result = geodesic_distance_matrix(graph)

        assert result.shape == (0, 0)

    def test_unweighted_graph(self):
        """Test geodesic distances without explicit weights (hop count)."""
        graph = nx.Graph()
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)

        result = geodesic_distance_matrix(graph, weight=None)

        # Distance is hop count when unweighted
        expected = np.array([
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0]
        ])

        assert_allclose(result, expected)
```

**Estimated time**: 3-4 hours

---

**Phase 1 Success Criteria**:

- [ ] All type hints added and verified
- [ ] All docstring issues fixed
- [ ] `calibration.py` has 80%+ coverage
- [ ] `distance.py` has 80%+ coverage
- [ ] All existing tests still pass
- [ ] No breaking changes to API

---

## Phase 2: Test Coverage Enhancement (Week 1-2)

**Goal**: Increase overall test coverage from 61% to 75%+
**Estimated Time**: 16-20 hours
**Priority**: HIGH

### 2.1 Regions I/O Module Testing [6-8 hours]

**Goal**: Increase `regions/io.py` from 8% to 70%+ coverage

#### Create Test Data Fixtures

**Create**: `tests/regions/fixtures/` directory

**File**: `tests/regions/fixtures/sample_labelme.json`

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "reward_zone",
      "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    },
    {
      "label": "start_point",
      "points": [[50, 50]],
      "group_id": null,
      "shape_type": "point",
      "flags": {}
    }
  ],
  "imagePath": "test_image.png",
  "imageData": null,
  "imageHeight": 300,
  "imageWidth": 300
}
```

**File**: `tests/regions/fixtures/sample_cvat.xml`

```xml
<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <name>Test Task</name>
      <size>1</size>
    </task>
  </meta>
  <image id="0" name="test_image.png" width="640" height="480">
    <polygon label="arena" points="100.0,100.0;200.0,100.0;200.0,200.0;100.0,200.0" />
    <points label="feeder" points="150.0,150.0" />
  </image>
</annotations>
```

#### Create Test Suite

**Create**: `tests/regions/test_io.py`

```python
"""Tests for regions I/O module."""

from pathlib import Path

import numpy as np
import pytest

from neurospatial.regions import Regions
from neurospatial.regions.io import (
    load_cvat_xml,
    load_labelme_json,
    mask_to_region,
    regions_from_binary_masks,
)

# Use Path to fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestLoadLabelmeJson:
    """Tests for load_labelme_json function."""

    def test_load_basic_labelme(self, tmp_path):
        """Test loading basic LabelMe JSON file."""
        json_file = FIXTURES_DIR / "sample_labelme.json"

        regions = load_labelme_json(json_file)

        assert isinstance(regions, Regions)
        assert "reward_zone" in regions
        assert "start_point" in regions
        assert regions["reward_zone"].kind == "polygon"
        assert regions["start_point"].kind == "point"

    def test_load_nonexistent_file(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_labelme_json("nonexistent.json")

    def test_load_malformed_json(self, tmp_path):
        """Test handling of malformed JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ invalid json")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_labelme_json(bad_file)

    def test_empty_shapes(self, tmp_path):
        """Test loading JSON with no shapes."""
        empty_json = tmp_path / "empty.json"
        empty_json.write_text('{"shapes": [], "version": "5.0.1"}')

        regions = load_labelme_json(empty_json)

        assert len(regions) == 0


class TestLoadCvatXml:
    """Tests for load_cvat_xml function."""

    def test_load_basic_cvat(self):
        """Test loading basic CVAT XML file."""
        xml_file = FIXTURES_DIR / "sample_cvat.xml"

        regions = load_cvat_xml(xml_file)

        assert isinstance(regions, Regions)
        assert "arena" in regions
        assert "feeder" in regions

    def test_load_nonexistent_xml(self):
        """Test that loading non-existent XML raises error."""
        with pytest.raises(FileNotFoundError):
            load_cvat_xml("nonexistent.xml")

    def test_malformed_xml(self, tmp_path):
        """Test handling of malformed XML."""
        bad_xml = tmp_path / "bad.xml"
        bad_xml.write_text("<unclosed>")

        with pytest.raises(ValueError, match="Invalid XML"):
            load_cvat_xml(bad_xml)


class TestMaskToRegion:
    """Tests for mask_to_region function."""

    def test_simple_square_mask(self):
        """Test converting simple square mask to region."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True

        region = mask_to_region(mask, "test_region")

        assert region.name == "test_region"
        assert region.kind == "polygon"

    def test_empty_mask(self):
        """Test that empty mask raises error."""
        mask = np.zeros((10, 10), dtype=bool)

        with pytest.raises(ValueError, match="empty mask"):
            mask_to_region(mask, "empty")

    def test_full_mask(self):
        """Test mask that is entirely True."""
        mask = np.ones((5, 5), dtype=bool)

        region = mask_to_region(mask, "full")

        assert region.kind == "polygon"

    def test_mask_with_hole(self):
        """Test mask with hole creates polygon with hole."""
        mask = np.ones((10, 10), dtype=bool)
        mask[4:6, 4:6] = False  # Create hole

        region = mask_to_region(mask, "donut")

        assert region.kind == "polygon"
        # Verify polygon has interior (hole)
```

**Estimated time**: 6-8 hours (including fixture creation)

---

### 2.2 Regions Operations Module Testing [3-4 hours]

**Create**: Expand `tests/regions/test_ops.py`

```python
"""Tests for regions operations module."""

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from neurospatial import Environment
from neurospatial.regions import Region, Regions
from neurospatial.regions.ops import (
    get_points_in_region,
    get_region_bins,
    points_in_region_mask,
)


class TestGetPointsInRegion:
    """Tests for get_points_in_region function."""

    def test_points_in_polygon_region(self):
        """Test getting points inside polygon region."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        region = Region(name="box", data=polygon, kind="polygon")

        points = np.array([
            [5, 5],    # Inside
            [15, 15],  # Outside
            [0, 0],    # On boundary
            [5, 5],    # Inside (duplicate)
        ])

        result = get_points_in_region(points, region)

        assert result.shape[0] <= 3  # At most 3 points inside
        assert np.all(result[:, 0] <= 10)
        assert np.all(result[:, 1] <= 10)

    def test_points_in_point_region(self):
        """Test that point regions work correctly."""
        region = Region(name="center", data=(5.0, 5.0), kind="point")

        points = np.array([[5, 5], [5.1, 5.1], [10, 10]])

        # Should return points within some tolerance of region point
        result = get_points_in_region(points, region, tolerance=0.2)

        assert len(result) >= 1  # At least the exact match

    def test_empty_points_array(self):
        """Test with empty points array."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        region = Region(name="box", data=polygon, kind="polygon")

        points = np.empty((0, 2))

        result = get_points_in_region(points, region)

        assert result.shape == (0, 2)


class TestPointsInRegionMask:
    """Tests for points_in_region_mask function."""

    def test_mask_for_polygon_region(self):
        """Test creating boolean mask for polygon region."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        region = Region(name="box", data=polygon, kind="polygon")

        points = np.array([
            [5, 5],    # Inside
            [15, 15],  # Outside
            [5, 5],    # Inside (duplicate)
        ])

        mask = points_in_region_mask(points, region)

        assert mask.shape == (3,)
        assert mask[0] == True   # First point inside
        assert mask[1] == False  # Second point outside
        assert mask[2] == True   # Third point inside

    def test_mask_all_outside(self):
        """Test mask when all points are outside."""
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        region = Region(name="small", data=polygon, kind="polygon")

        points = np.array([[10, 10], [20, 20], [30, 30]])

        mask = points_in_region_mask(points, region)

        assert np.all(~mask)  # All False

    def test_mask_all_inside(self):
        """Test mask when all points are inside."""
        polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        region = Region(name="large", data=polygon, kind="polygon")

        points = np.array([[10, 10], [20, 20], [30, 30]])

        mask = points_in_region_mask(points, region)

        assert np.all(mask)  # All True


class TestGetRegionBins:
    """Tests for get_region_bins function."""

    def test_get_bins_in_polygon(self):
        """Test getting bins that fall within a polygon region."""
        # Create simple 2D environment
        env = Environment.from_samples(
            data_samples=np.array([[0, 0], [10, 10]]),
            bin_size=1.0,
            infer_active_bins=False,
        )

        # Create polygon region covering part of environment
        polygon = Polygon([(2, 2), (5, 2), (5, 5), (2, 5)])
        region = Region(name="subset", data=polygon, kind="polygon")

        bin_indices = get_region_bins(env, region)

        # Should get some bins, but not all
        assert len(bin_indices) > 0
        assert len(bin_indices) < env.n_bins

        # All returned bins should have centers inside polygon
        for idx in bin_indices:
            center = env.bin_centers[idx]
            assert polygon.contains(Point(center))

    def test_no_bins_in_region(self):
        """Test when no bins fall within region."""
        env = Environment.from_samples(
            data_samples=np.array([[0, 0], [10, 10]]),
            bin_size=1.0,
            infer_active_bins=False,
        )

        # Polygon far from environment
        polygon = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
        region = Region(name="far", data=polygon, kind="polygon")

        bin_indices = get_region_bins(env, region)

        assert len(bin_indices) == 0
```

**Estimated time**: 3-4 hours

---

### 2.3 Triangular Mesh Layout Testing [4-5 hours]

**Create**: `tests/layout/test_triangular_mesh_full.py`

```python
"""Comprehensive tests for TriangularMeshLayout engine."""

import numpy as np
import pytest

from neurospatial.layout.engines.triangular_mesh import TriangularMeshLayout


class TestTriangularMeshLayoutBuild:
    """Tests for TriangularMeshLayout build method."""

    def test_basic_2d_triangulation(self):
        """Test basic 2D triangular mesh creation."""
        data_samples = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]
        ])

        layout = TriangularMeshLayout()
        layout.build(data_samples=data_samples)

        assert layout.bin_centers.shape[1] == 2
        assert layout.connectivity is not None
        assert layout.connectivity.number_of_nodes() > 0

    def test_3d_triangulation(self):
        """Test 3D triangular mesh creation."""
        # Create simple 3D point cloud
        data_samples = np.random.rand(20, 3)

        layout = TriangularMeshLayout()
        layout.build(data_samples=data_samples)

        assert layout.bin_centers.shape[1] == 3

    def test_minimum_points(self):
        """Test triangulation with minimum number of points."""
        data_samples = np.array([[0, 0], [1, 0], [0, 1]])  # 3 points

        layout = TriangularMeshLayout()
        layout.build(data_samples=data_samples)

        assert layout.bin_centers.shape[0] == 3

    def test_insufficient_points(self):
        """Test that insufficient points raises error."""
        data_samples = np.array([[0, 0], [1, 0]])  # Only 2 points

        layout = TriangularMeshLayout()

        with pytest.raises(ValueError, match="at least 3 points"):
            layout.build(data_samples=data_samples)


class TestTriangularMeshPointToBin:
    """Tests for point_to_bin_index method."""

    @pytest.fixture
    def simple_mesh(self):
        """Create a simple triangular mesh for testing."""
        data_samples = np.array([
            [0, 0], [2, 0], [2, 2], [0, 2], [1, 1]
        ])
        layout = TriangularMeshLayout()
        layout.build(data_samples=data_samples)
        return layout

    def test_point_at_vertex(self, simple_mesh):
        """Test mapping point exactly at a vertex."""
        points = np.array([[1, 1]])

        indices = simple_mesh.point_to_bin_index(points)

        assert indices[0] >= 0

    def test_point_inside_mesh(self, simple_mesh):
        """Test mapping point inside mesh bounds."""
        points = np.array([[0.5, 0.5], [1.5, 1.5]])

        indices = simple_mesh.point_to_bin_index(points)

        assert np.all(indices >= 0)

    def test_point_outside_mesh(self, simple_mesh):
        """Test mapping point outside mesh bounds."""
        points = np.array([[10, 10]])

        indices = simple_mesh.point_to_bin_index(points)

        # Should return -1 or map to nearest
        assert len(indices) == 1


class TestTriangularMeshBinSizes:
    """Tests for bin_sizes method."""

    def test_bin_sizes_2d(self):
        """Test bin size calculation for 2D mesh."""
        data_samples = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ])

        layout = TriangularMeshLayout()
        layout.build(data_samples=data_samples)

        sizes = layout.bin_sizes()

        assert len(sizes) == layout.bin_centers.shape[0]
        assert np.all(sizes > 0)  # All bins should have positive size

    def test_bin_sizes_3d(self):
        """Test bin size calculation for 3D mesh."""
        data_samples = np.random.rand(10, 3)

        layout = TriangularMeshLayout()
        layout.build(data_samples=data_samples)

        sizes = layout.bin_sizes()

        assert len(sizes) == layout.bin_centers.shape[0]
        assert np.all(sizes > 0)


class TestTriangularMeshPlot:
    """Tests for plotting methods."""

    def test_plot_2d_mesh(self):
        """Test 2D mesh plotting doesn't raise errors."""
        data_samples = np.random.rand(10, 2)

        layout = TriangularMeshLayout()
        layout.build(data_samples=data_samples)

        # Should not raise
        ax = layout.plot()
        assert ax is not None

    def test_plot_3d_mesh(self):
        """Test 3D mesh plotting doesn't raise errors."""
        data_samples = np.random.rand(10, 3)

        layout = TriangularMeshLayout()
        layout.build(data_samples=data_samples)

        # Should not raise
        ax = layout.plot()
        assert ax is not None
```

**Estimated time**: 4-5 hours

---

**Phase 2 Success Criteria**:

- [ ] `regions/io.py` coverage > 70%
- [ ] `regions/ops.py` coverage > 70%
- [ ] `regions/plot.py` coverage > 40%
- [ ] `triangular_mesh.py` coverage > 70%
- [ ] Overall coverage > 70%
- [ ] All new tests pass

---

## Phase 3: Complexity Reduction (Week 2-3)

**Goal**: Reduce high-complexity functions to manageable size
**Estimated Time**: 16-24 hours
**Priority**: MEDIUM-HIGH

### 3.1 Refactor `load_cvat_xml()` [6-8 hours]

**Current**: 319 lines, complexity 45, nesting 5
**Target**: <100 lines per function, complexity <10

**Strategy**: Extract XML parsing into focused helper functions

#### Step 1: Write Tests First [2 hours]

Ensure existing tests cover current behavior (done in Phase 2).

#### Step 2: Extract Helper Functions [4-6 hours]

**File**: `src/neurospatial/regions/io.py`

**New structure**:

```python
def _parse_cvat_image_tag(image_elem) -> Dict[str, Any]:
    """
    Extract image metadata from CVAT image XML element.

    Parameters
    ----------
    image_elem : xml.etree.ElementTree.Element
        XML element representing an image with annotations.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: 'id', 'name', 'width', 'height'
    """
    return {
        'id': image_elem.get('id'),
        'name': image_elem.get('name'),
        'width': int(image_elem.get('width')),
        'height': int(image_elem.get('height')),
    }


def _parse_cvat_polygon(polygon_elem) -> Tuple[str, Polygon]:
    """
    Parse CVAT polygon element into Shapely Polygon.

    Parameters
    ----------
    polygon_elem : xml.etree.ElementTree.Element
        XML element representing a polygon.

    Returns
    -------
    Tuple[str, Polygon]
        Label name and Shapely Polygon object.
    """
    label = polygon_elem.get('label')
    points_str = polygon_elem.get('points')

    # Parse "x1,y1;x2,y2;x3,y3" format
    points = []
    for point_str in points_str.split(';'):
        x, y = map(float, point_str.split(','))
        points.append((x, y))

    polygon = Polygon(points)
    return label, polygon


def _parse_cvat_points(points_elem) -> Tuple[str, Tuple[float, float]]:
    """
    Parse CVAT points element into point coordinates.

    Parameters
    ----------
    points_elem : xml.etree.ElementTree.Element
        XML element representing point annotation.

    Returns
    -------
    Tuple[str, Tuple[float, float]]
        Label name and (x, y) coordinates.
    """
    label = points_elem.get('label')
    points_str = points_elem.get('points')

    x, y = map(float, points_str.split(','))
    return label, (x, y)


def _parse_cvat_image(image_elem) -> Dict[str, Region]:
    """
    Parse all annotations from a single CVAT image element.

    Parameters
    ----------
    image_elem : xml.etree.ElementTree.Element
        XML element for a single image with annotations.

    Returns
    -------
    Dict[str, Region]
        Dictionary mapping region names to Region objects.
    """
    regions = {}

    # Parse polygons
    for polygon_elem in image_elem.findall('polygon'):
        label, polygon = _parse_cvat_polygon(polygon_elem)
        regions[label] = Region(name=label, data=polygon, kind='polygon')

    # Parse points
    for points_elem in image_elem.findall('points'):
        label, coords = _parse_cvat_points(points_elem)
        regions[label] = Region(name=label, data=coords, kind='point')

    # Parse other shapes as needed...

    return regions


def load_cvat_xml(filepath: str | Path) -> Regions:
    """
    Load regions from CVAT XML annotation file.

    This function is now much simpler - delegates to helpers.

    Parameters
    ----------
    filepath : str or Path
        Path to CVAT XML file.

    Returns
    -------
    Regions
        Regions object containing all annotations.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If XML is malformed or invalid.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML: {e}")

    all_regions = Regions()

    # Iterate through images
    for image_elem in root.findall('.//image'):
        image_regions = _parse_cvat_image(image_elem)

        # Add to collection
        for name, region in image_regions.items():
            all_regions.add(region)

    return all_regions
```

**Testing**:

```bash
# Run existing tests to ensure no regression
uv run pytest tests/regions/test_io.py -v

# Verify complexity reduction
python complexity_analysis.py  # Should show lower complexity
```

**Success Criteria**:

- [ ] Main function < 50 lines
- [ ] Cyclomatic complexity < 10
- [ ] Each helper < 30 lines
- [ ] All tests pass
- [ ] No behavior changes

**Estimated time**: 6-8 hours

---

### 3.2 Refactor `map_probabilities_to_nearest_target_bin()` [4-6 hours]

**Current**: 188 lines, complexity 25
**Target**: <80 lines, complexity <12

**Strategy**: Extract validation, transformation, and mapping logic

**File**: `src/neurospatial/alignment.py`

```python
def _validate_alignment_inputs(
    source_env: Environment,
    target_env: Environment,
    source_probabilities: NDArray[np.float64],
) -> None:
    """
    Validate inputs for probability alignment.

    Parameters
    ----------
    source_env : Environment
        Source environment.
    target_env : Environment
        Target environment.
    source_probabilities : NDArray[np.float64]
        Probability array for source environment.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If environments are not fitted.
    """
    if not source_env._is_fitted:
        raise RuntimeError("Source environment is not fitted")
    if not target_env._is_fitted:
        raise RuntimeError("Target environment is not fitted")

    if source_probabilities.shape[0] != source_env.n_bins:
        raise ValueError(
            f"source_probabilities length ({source_probabilities.shape[0]}) "
            f"does not match source_env bins ({source_env.n_bins})"
        )

    if source_env.n_dims != target_env.n_dims:
        raise ValueError(
            f"Environment dimensions must match: "
            f"source={source_env.n_dims}, target={target_env.n_dims}"
        )


def _transform_source_bins(
    source_centers: NDArray[np.float64],
    rotation_matrix: NDArray[np.float64] | None,
    translation_vector: NDArray[np.float64] | None,
    scale_factor: float,
) -> NDArray[np.float64]:
    """
    Apply geometric transformations to source bin centers.

    Parameters
    ----------
    source_centers : NDArray[np.float64]
        Source bin centers to transform.
    rotation_matrix : NDArray[np.float64] | None
        2x2 rotation matrix. If None, no rotation.
    translation_vector : NDArray[np.float64] | None
        Translation to apply. If None, no translation.
    scale_factor : float
        Scaling factor.

    Returns
    -------
    NDArray[np.float64]
        Transformed bin centers.
    """
    transformed = source_centers.copy()

    # Apply rotation
    if rotation_matrix is not None:
        transformed = transformed @ rotation_matrix.T

    # Apply scaling
    transformed = transformed * scale_factor

    # Apply translation
    if translation_vector is not None:
        transformed = transformed + translation_vector

    return transformed


def _map_to_nearest_bins(
    source_centers: NDArray[np.float64],
    target_centers: NDArray[np.float64],
    source_probabilities: NDArray[np.float64],
    n_target_bins: int,
) -> NDArray[np.float64]:
    """
    Map source probabilities to nearest target bins.

    Parameters
    ----------
    source_centers : NDArray[np.float64]
        Transformed source bin centers.
    target_centers : NDArray[np.float64]
        Target bin centers.
    source_probabilities : NDArray[np.float64]
        Probabilities for source bins.
    n_target_bins : int
        Number of target bins.

    Returns
    -------
    NDArray[np.float64]
        Probabilities mapped to target bins.
    """
    from sklearn.neighbors import NearestNeighbors

    # Find nearest target bin for each source bin
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(target_centers)
    distances, indices = nn.kneighbors(source_centers)

    # Initialize target probabilities
    target_probabilities = np.zeros(n_target_bins)

    # Accumulate probabilities for bins with multiple source bins mapping to them
    for source_idx, target_idx in enumerate(indices.flatten()):
        target_probabilities[target_idx] += source_probabilities[source_idx]

    # Normalize if needed
    total = target_probabilities.sum()
    if total > 0:
        target_probabilities /= total

    return target_probabilities


def map_probabilities_to_nearest_target_bin(
    source_env: Environment,
    target_env: Environment,
    source_probabilities: NDArray[np.float64],
    source_rotation_matrix: NDArray[np.float64] | None = None,
    source_translation_vector: NDArray[np.float64] | None = None,
    source_scale_factor: float = 1.0,
) -> NDArray[np.float64]:
    """
    Map probabilities from source to target environment bins (REFACTORED).

    Much simpler main function - delegates to helpers.

    [Keep existing comprehensive docstring]
    """
    # Validate inputs
    _validate_alignment_inputs(source_env, target_env, source_probabilities)

    # Handle empty cases
    if source_env.n_bins == 0 or target_env.n_bins == 0:
        warnings.warn("One of the environments has zero bins; returning zeros.")
        return np.zeros(target_env.n_bins)

    # Transform source bin centers
    transformed_source_centers = _transform_source_bins(
        source_env.bin_centers,
        source_rotation_matrix,
        source_translation_vector,
        source_scale_factor,
    )

    # Map to nearest target bins
    target_probabilities = _map_to_nearest_bins(
        transformed_source_centers,
        target_env.bin_centers,
        source_probabilities,
        target_env.n_bins,
    )

    return target_probabilities
```

**Estimated time**: 4-6 hours

---

### 3.3 Refactor `find_boundary_nodes()` [4-6 hours]

**Current**: 147 lines, complexity 28, nesting 5
**Target**: <60 lines, complexity <12

**Strategy**: Use strategy pattern for layout-specific logic

**File**: `src/neurospatial/layout/helpers/utils.py`

```python
def _find_boundary_nodes_grid(
    graph: nx.Graph,
    grid_shape: Tuple[int, ...],
    active_mask: NDArray[np.bool_],
) -> NDArray[np.int_]:
    """
    Find boundary nodes for grid-based layouts.

    Parameters
    ----------
    graph : nx.Graph
        Connectivity graph.
    grid_shape : Tuple[int, ...]
        Shape of the grid.
    active_mask : NDArray[np.bool_]
        Mask of active bins.

    Returns
    -------
    NDArray[np.int_]
        Indices of boundary nodes.
    """
    boundary_nodes = []

    for node_id, data in graph.nodes(data=True):
        # A node is a boundary if it has fewer neighbors than expected
        # for an interior node, or if it's next to an inactive bin

        neighbors = list(graph.neighbors(node_id))
        max_neighbors = 2 * len(grid_shape)  # For orthogonal connectivity

        if len(neighbors) < max_neighbors:
            boundary_nodes.append(node_id)
        else:
            # Check if adjacent to inactive bin
            nd_idx = data.get('original_grid_nd_index')
            if nd_idx and _is_adjacent_to_inactive(nd_idx, active_mask):
                boundary_nodes.append(node_id)

    return np.array(boundary_nodes, dtype=int)


def _find_boundary_nodes_graph(graph: nx.Graph) -> NDArray[np.int_]:
    """
    Find boundary nodes for graph-based (1D) layouts.

    Parameters
    ----------
    graph : nx.Graph
        Connectivity graph.

    Returns
    -------
    NDArray[np.int_]
        Indices of boundary nodes (degree 1 or 0).
    """
    boundary_nodes = [
        node for node, degree in graph.degree()
        if degree <= 1
    ]
    return np.array(boundary_nodes, dtype=int)


def _find_boundary_nodes_mesh(graph: nx.Graph) -> NDArray[np.int_]:
    """
    Find boundary nodes for mesh-based layouts (triangular, etc.).

    Parameters
    ----------
    graph : nx.Graph
        Connectivity graph.

    Returns
    -------
    NDArray[np.int_]
        Indices of boundary nodes.
    """
    # For meshes, use convex hull or similar approach
    # Simplified version:
    boundary_nodes = []

    for node_id in graph.nodes():
        neighbors = list(graph.neighbors(node_id))
        # Heuristic: boundary nodes have asymmetric neighborhood
        if len(neighbors) < 6:  # Typical interior mesh node has 6 neighbors
            boundary_nodes.append(node_id)

    return np.array(boundary_nodes, dtype=int)


def find_boundary_nodes(
    graph: nx.Graph,
    grid_shape: Tuple[int, ...] | None = None,
    active_mask: NDArray[np.bool_] | None = None,
    layout_kind: str | None = None,
) -> NDArray[np.int_]:
    """
    Find boundary nodes in the connectivity graph (REFACTORED).

    Now uses strategy pattern - delegates to layout-specific functions.

    [Keep existing comprehensive docstring]
    """
    if graph.number_of_nodes() == 0:
        return np.array([], dtype=int)

    # Detect layout type from graph properties or explicit parameter
    if layout_kind is None:
        layout_kind = _infer_layout_kind(graph, grid_shape)

    # Dispatch to appropriate strategy
    if layout_kind in ('RegularGrid', 'MaskedGrid', 'Hexagonal', 'ImageMask'):
        if grid_shape is None or active_mask is None:
            raise ValueError("Grid-based layouts require grid_shape and active_mask")
        return _find_boundary_nodes_grid(graph, grid_shape, active_mask)

    elif layout_kind == 'Graph':
        return _find_boundary_nodes_graph(graph)

    elif layout_kind in ('TriangularMesh', 'ShapelyPolygon'):
        return _find_boundary_nodes_mesh(graph)

    else:
        # Generic fallback
        return _find_boundary_nodes_generic(graph)
```

**Estimated time**: 4-6 hours

---

**Phase 3 Success Criteria**:

- [ ] `load_cvat_xml()` < 100 lines, complexity < 12
- [ ] `map_probabilities_to_nearest_target_bin()` < 80 lines, complexity < 12
- [ ] `find_boundary_nodes()` < 60 lines, complexity < 12
- [ ] All tests pass
- [ ] Code coverage maintained or improved
- [ ] No breaking API changes

---

## Phase 4: Documentation Polish (Week 3)

**Goal**: Complete missing documentation
**Estimated Time**: 8-12 hours
**Priority**: MEDIUM

### 4.1 Add Missing Class Docstrings [2-3 hours]

#### Mixins

**File**: `src/neurospatial/layout/mixins.py`

```python
class _KDTreeMixin:
    """
    Mixin providing KDTree-based spatial indexing for layout engines.

    This mixin adds efficient point-to-bin mapping using scipy's KDTree
    for spatial queries. It's suitable for layouts where bins are represented
    as discrete points in space (not grids).

    The mixin assumes the layout has a `bin_centers` attribute and provides:
    - Lazy KDTree construction (built on first use)
    - Cached tree for subsequent queries
    - Fast nearest-neighbor point-to-bin mapping

    Notes
    -----
    Classes using this mixin should ensure `bin_centers` is populated
    before calling any mixin methods.

    See Also
    --------
    _GridMixin : Alternative mixin for grid-based layouts.

    Examples
    --------
    >>> class MyLayout(_KDTreeMixin):
    ...     def build(self, points):
    ...         self.bin_centers = points
    ...         self._build_kdtree()
    """
    # ... existing code ...


class _GridMixin:
    """
    Mixin providing grid-based functionality for regular layout engines.

    This mixin adds grid-specific operations including:
    - Point-to-bin mapping using grid edges
    - Bin size calculation
    - Grid plotting with connectivity visualization

    The mixin assumes the layout has:
    - `grid_edges`: Tuple of 1D arrays defining bin edges
    - `grid_shape`: Tuple of ints defining grid dimensions
    - `active_mask`: Boolean array indicating active bins
    - `bin_centers`: N-D array of bin center coordinates

    Notes
    -----
    This mixin is designed for layouts with regular grid structure
    (RegularGridLayout, MaskedGridLayout, HexagonalLayout, ImageMaskLayout).

    See Also
    --------
    _KDTreeMixin : Alternative mixin for point-based layouts.

    Examples
    --------
    >>> class MyGridLayout(_GridMixin):
    ...     def build(self, grid_edges):
    ...         self.grid_edges = grid_edges
    ...         self.grid_shape = tuple(len(e)-1 for e in grid_edges)
    ...         # ... rest of build logic
    """
    # ... existing code ...
```

**Estimated time**: 1 hour

---

#### Affine2D Methods

**File**: `src/neurospatial/transforms.py`

```python
class Affine2D:
    """
    [Existing class docstring with Attributes added]
    """

    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply transformation to points.

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_points, 2)
            2D points to transform.

        Returns
        -------
        NDArray[np.float64], shape (n_points, 2)
            Transformed points.

        Examples
        --------
        >>> transform = Affine2D.identity().translate(10, 20)
        >>> points = np.array([[0, 0], [1, 1]])
        >>> transform(points)
        array([[10., 20.],
               [11., 21.]])
        """
        # ... existing code ...

    def inverse(self) -> Affine2D:
        """
        Compute the inverse transformation.

        Returns
        -------
        Affine2D
            New Affine2D representing the inverse transformation.

        Raises
        ------
        np.linalg.LinAlgError
            If transformation matrix is singular (non-invertible).

        Examples
        --------
        >>> transform = Affine2D.identity().translate(10, 20)
        >>> inv = transform.inverse()
        >>> points = np.array([[10, 20]])
        >>> inv(points)
        array([[0., 0.]])
        """
        # ... existing code ...

    def compose(self, other: Affine2D) -> Affine2D:
        """
        Compose this transformation with another.

        Parameters
        ----------
        other : Affine2D
            Transformation to compose with (applied first).

        Returns
        -------
        Affine2D
            New transformation representing self ∘ other.

        Notes
        -----
        The resulting transformation applies `other` first, then `self`.
        Composition order matters: `a.compose(b)` ≠ `b.compose(a)` in general.

        Examples
        --------
        >>> t1 = Affine2D.identity().translate(10, 0)
        >>> t2 = Affine2D.identity().translate(0, 20)
        >>> combined = t1.compose(t2)
        >>> points = np.array([[0, 0]])
        >>> combined(points)
        array([[10., 20.]])
        """
        # ... existing code ...

    def __matmul__(self, other: Affine2D) -> Affine2D:
        """
        Compose transformations using @ operator.

        Parameters
        ----------
        other : Affine2D
            Transformation to compose with.

        Returns
        -------
        Affine2D
            Composed transformation (self @ other).

        See Also
        --------
        compose : Functional composition interface.

        Examples
        --------
        >>> t1 = Affine2D.identity().translate(10, 0)
        >>> t2 = Affine2D.identity().scale(2.0)
        >>> combined = t1 @ t2
        """
        # ... existing code ...
```

**Estimated time**: 2 hours

---

### 4.2 Add Examples Sections to Key Functions [6-8 hours]

#### Factory Methods

**File**: `src/neurospatial/environment.py`

Add Examples sections to:

- `from_samples()`
- `from_graph()`
- `from_polygon()`
- `from_mask()`
- `from_image()`

**Example template**:

```python
@classmethod
def from_samples(cls, ...):
    """
    [Existing parameters docstring]

    Examples
    --------
    Create a simple 2D environment from position data:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>>
    >>> # Simulate animal position data
    >>> positions = np.random.rand(1000, 2) * 100  # 100x100 arena
    >>>
    >>> # Create environment with 5cm bins
    >>> env = Environment.from_samples(
    ...     data_samples=positions,
    ...     bin_size=5.0,
    ...     name="arena"
    ... )
    >>>
    >>> # Query environment properties
    >>> env.n_bins
    400
    >>> env.n_dims
    2

    Create environment with morphological operations:

    >>> env = Environment.from_samples(
    ...     data_samples=positions,
    ...     bin_size=5.0,
    ...     bin_count_threshold=10,  # Require 10 samples per bin
    ...     dilate=True,              # Expand active region
    ...     fill_holes=True,          # Fill interior holes
    ... )
    """
```

**Estimated time**: 4-5 hours

---

#### Utility Functions

**Files**: `alignment.py`, `transforms.py`, `regions/io.py`

Add Examples sections following NumPy format.

**Estimated time**: 2-3 hours

---

**Phase 4 Success Criteria**:

- [ ] All public classes have complete docstrings
- [ ] All public methods have docstrings with Parameters/Returns
- [ ] Key API entry points have Examples sections
- [ ] Docstring quality > 90%

---

## Success Metrics

### Quantitative Targets

| Metric | Before | After Phase 1 | After Phase 2 | After Phase 3 | After Phase 4 |
|--------|--------|---------------|---------------|---------------|---------------|
| Overall Coverage | 61% | 68% | 75% | 75% | 75% |
| Type Completeness | 85% | 95% | 95% | 95% | 95% |
| Functions >50 lines | 60 | 60 | 60 | <40 | <40 |
| Complexity >10 | 23 | 23 | 23 | <15 | <15 |
| Docstring Quality | 75% | 80% | 80% | 80% | 90% |

### Qualitative Goals

- [ ] All critical modules have tests
- [ ] Complex functions refactored and maintainable
- [ ] Documentation complete for public API
- [ ] No breaking changes to existing API
- [ ] Code review ready for v1.0 release

---

## Risk Mitigation

### Risks & Mitigation Strategies

1. **Refactoring introduces bugs**
   - **Mitigation**: Test-driven approach - write tests first
   - **Strategy**: Run full test suite after each change
   - **Rollback**: Keep each refactoring in separate commit

2. **Breaking API changes**
   - **Mitigation**: Maintain backward compatibility
   - **Strategy**: Add deprecation warnings before removals
   - **Testing**: Integration tests with existing code

3. **Time overruns**
   - **Mitigation**: Prioritize phases - can stop after any phase
   - **Strategy**: Track time spent vs. estimated
   - **Adjustment**: Skip Phase 4 if needed, focus on quality

4. **Merge conflicts**
   - **Mitigation**: Work in feature branches
   - **Strategy**: Rebase frequently on main
   - **Communication**: Coordinate with team

5. **Test maintenance burden**
   - **Mitigation**: Keep tests simple and focused
   - **Strategy**: Use fixtures to reduce duplication
   - **Documentation**: Comment complex test setups

---

## Implementation Timeline

### Week 1

**Days 1-2**: Phase 1 Critical Fixes (12-16 hours)

- Day 1 AM: Type hints fixes
- Day 1 PM: Docstring fixes
- Day 2: Add tests for calibration.py and distance.py
- Day 2 EOD: Review and verify

**Days 3-5**: Phase 2 Test Coverage (16-20 hours)

- Day 3: Regions I/O tests
- Day 4: Regions ops tests + triangular mesh tests (part 1)
- Day 5: Triangular mesh tests (part 2) + review

### Week 2

**Days 1-3**: Phase 3 Complexity Reduction (16-24 hours)

- Day 1: Refactor load_cvat_xml()
- Day 2: Refactor map_probabilities_to_nearest_target_bin()
- Day 3: Refactor find_boundary_nodes() + other complex functions

**Days 4-5**: Phase 3 Continued + Buffer

- Day 4: Refactor additional high-complexity functions
- Day 5: Testing, integration, cleanup

### Week 3

**Days 1-2**: Phase 4 Documentation (8-12 hours)

- Day 1: Add class docstrings, method docstrings
- Day 2: Add Examples sections to key functions

**Days 3-5**: Final Review & Polish

- Day 3: Full test suite run, coverage check
- Day 4: Code review, documentation review
- Day 5: Final cleanup, prepare for release

---

## Checkpoints & Reviews

### After Each Phase

1. Run full test suite: `uv run pytest`
2. Check coverage: `uv run pytest --cov=src/neurospatial`
3. Run linters: `uv run ruff check src/ && uv run ruff format --check src/`
4. Review complexity: Run complexity analysis
5. Manual spot checks of refactored code
6. Update documentation if needed

### Final Review Checklist

- [ ] All tests pass (160+ tests)
- [ ] Coverage > 75%
- [ ] No ruff violations
- [ ] Type hints > 95% complete
- [ ] Functions with complexity >10: <15
- [ ] Docstrings > 90% complete
- [ ] No breaking API changes
- [ ] CHANGELOG.md updated
- [ ] Documentation updated
- [ ] Ready for v1.0 release

---

## Notes

- Each phase is independently valuable - can stop after any phase
- Commits should be atomic and well-described
- Consider creating PRs for each phase for easier review
- Update CLAUDE.md with any new patterns or conventions
- Document any new testing patterns in conftest.py

---

**End of Refactoring Plan**

*This plan is a living document - adjust as needed based on progress and discoveries during implementation.*
