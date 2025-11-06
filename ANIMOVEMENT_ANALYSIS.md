# animovement Analysis: R Equivalent of movement

**Package**: [animovement/animovement](https://github.com/animovement/animovement)
**Language**: R
**Status**: Experimental (created March 2024)
**Stars**: 10, **Forks**: 1, **Issues**: 18 open
**Last updated**: October 18, 2025

---

## 1. Package Overview

**animovement** is "An R toolbox for analysing animal movement across space and time" - the **R equivalent** of the Python `movement` package.

**Key relationship**: Actively collaborates with the Python `movement` package developers to maintain compatible data standards and workflows across languages.

**Target users**: R users in neuroscience, ethology, biomechanics analyzing pose tracking or centroid data.

---

## 2. Core Capabilities

### Same as Python movement package:

| Capability | animovement (R) | movement (Python) |
|-----------|-----------------|-------------------|
| **Load pose data** | ✅ DLC, SLEAP, LP, Anipose | ✅ DLC, SLEAP, LP, Anipose |
| **Load centroid data** | ✅ TRex, idtracker.ai, AnimalTA | ⚠️ Limited |
| **Trackball data** | ✅ Mouse sensors, FicTrac | ❌ |
| **Data cleaning** | ✅ Filtering, smoothing | ✅ Filtering, smoothing |
| **Kinematics** | ✅ Velocity, acceleration | ✅ Velocity, acceleration |
| **Summary statistics** | ✅ | ⚠️ Basic |
| **Spatial discretization** | ❌ | ❌ |
| **Behavioral segmentation** | ❌ | ❌ |
| **Neural integration** | ❌ | ❌ |

**Conclusion**: Same scope as Python movement - tracking data cleaning and kinematics, but no spatial analysis.

---

## 3. Unique Features (vs Python movement)

### Trackball Support ⭐

**animovement provides**:
```r
# Read trackball data (unique to animovement)
trackball_data <- read_trackball(
  file = "mouse_sensor.csv",
  source = "optical_mouse"  # or "fictrac"
)
```

**Why unique**: Trackballs are common in rodent head-fixed experiments but Python movement doesn't support them.

### Centroid Tracking Software

**Better support for centroid trackers**:
- TRex
- idtracker.ai
- AnimalTA

**Python movement**: Focuses on pose estimation (DLC, SLEAP), limited centroid support.

---

## 4. What animovement DOES NOT Provide

**Same gaps as Python movement**:
- ❌ No spatial discretization (bins, graphs)
- ❌ No environment abstraction
- ❌ No behavioral segmentation (runs, laps, trials)
- ❌ No spatial metrics (Skaggs info, grid score, place fields)
- ❌ No neural data integration
- ❌ No irregular graph support

**Reason**: animovement is for **tracking data processing**, not **spatial neuroscience analysis**.

---

## 5. Relevance to neurospatial

### Not Directly Relevant ⚠️

**Why**:
1. **Different language** - animovement is R, neurospatial is Python
2. **Same scope as movement** - Already analyzed Python movement
3. **No new spatial capabilities** - Same gaps
4. **Experimental status** - Still under active development

### Ecosystem Position

```
R workflow:
  DLC/SLEAP → animovement (R) → ??? (no R spatial analysis package)

Python workflow:
  DLC/SLEAP → movement (Python) → neurospatial (Python) → pynapple → user packages
```

**Gap**: R users lack spatial analysis tools (no neurospatial equivalent in R).

---

## 6. Key Insights

### Validates Python movement Architecture

**animovement confirms**:
- ✅ Tracking data cleaning is a **distinct domain** (separate from spatial analysis)
- ✅ Python `movement` package is the **standard** (R version copies it)
- ✅ Cross-language collaboration is valuable (shared data standards)

### Highlights neurospatial's Unique Value

**R users have NO equivalent to neurospatial**:
- No spatial discretization in R
- No graph-based spatial primitives in R
- No behavioral segmentation in R

**Python advantage**: neurospatial + movement + pynapple = complete stack

---

## 7. What We Can Learn

### 1. Cross-Language Collaboration Works ⭐⭐

**animovement + movement collaboration**:
- Shared data standards
- Compatible workflows
- Users can switch languages

**For neurospatial**: Could document interoperability with R
- Export environments to R-compatible formats
- Share data standards with potential R spatial package

### 2. Trackball Support is Niche but Valuable ⭐

**animovement supports trackballs** (movement doesn't)

**For neurospatial**: Trackball data is still 2D position → already supported
```python
# Trackball data is just (x, y) position time-series
trackball_positions = load_trackball_data()  # External loader

# Works with neurospatial
env = Environment.from_samples(trackball_positions, bin_size=2.0)
```

**No action needed** - neurospatial already handles any position data.

### 3. Tidyverse-Friendly API is Popular ⭐

**animovement emphasizes "tidyverse-friendly syntax"**

**For neurospatial**: Python equivalent is **pandas-friendly** or **xarray-friendly**
- Consider `to_dataframe()` methods
- Consider pandas integration for trajectory data

**Example**:
```python
# Potential: pandas-friendly output
import pandas as pd

runs_df = pd.DataFrame([
    {"start_time": r.start_time, "end_time": r.end_time, "success": r.success}
    for r in runs
])
# Easy to filter, group, visualize with pandas/seaborn
```

**Recommendation**: ⚠️ Low priority, consider in Phase 5 documentation

---

## 8. Impact on neurospatial Implementation Plan

### No Changes Needed ✅

**animovement analysis confirms**:
1. ✅ Tracking data cleaning is separate from spatial analysis (already knew from movement)
2. ✅ neurospatial fills unique gap (no R equivalent exists)
3. ✅ Python has complete stack (movement → neurospatial → pynapple)
4. ✅ Focus on spatial primitives (don't replicate tracking tools)

**Only insight**: Trackball support is niche (but neurospatial already handles it).

---

## 9. Comparison Summary

### Three-Package Comparison

| Aspect | animovement (R) | movement (Python) | neurospatial (Python) |
|--------|-----------------|-------------------|----------------------|
| **Language** | R | Python | Python |
| **Status** | Experimental | Stable | Stable |
| **Stars** | 10 | 211 | ~200 (estimate) |
| **Load tracking** | ✅ DLC, SLEAP, centroids | ✅ DLC, SLEAP, LP | ❌ (use movement) |
| **Trackballs** | ✅ Unique | ❌ | ⚠️ (data compatible) |
| **Kinematics** | ✅ | ✅ | ❌ (use movement) |
| **Spatial discretization** | ❌ | ❌ | ✅ Unique |
| **Behavioral segmentation** | ❌ | ❌ | ✅ Unique |
| **Spatial metrics** | ❌ | ❌ | ✅ Planned |
| **Neural integration** | ❌ | ❌ | ✅ Core feature |

**Ecosystem gaps**:
- **R users**: Have tracking (animovement), but NO spatial analysis
- **Python users**: Have complete stack (movement → neurospatial → pynapple)

---

## 10. Recommendations

### For neurospatial

**1. Document R interoperability** (Phase 5)
```markdown
# docs/r_users.md

neurospatial is Python-only, but R users can:
1. Use animovement (R) for tracking data cleaning
2. Export position data to CSV
3. Load in Python neurospatial for spatial analysis
4. Export results back to R for visualization
```

**2. No code changes needed**
- animovement doesn't add new requirements
- Same scope as movement (already analyzed)

**3. Consider pandas output** (optional)
```python
# Make results pandas-friendly for R users
runs_df = detect_runs_between_regions(..., return_dataframe=True)
runs_df.to_csv("runs.csv")  # R users can import
```

**Effort**: 1 day (documentation only)

---

## 11. Key Takeaways

### What animovement Validates

✅ **Tracking data cleaning is a distinct domain** - Separate package for separate concern
✅ **Cross-language collaboration works** - Python movement + R animovement share standards
✅ **tidyverse/pandas-friendly APIs are popular** - Users want familiar syntax

### What animovement Highlights

❌ **R lacks spatial analysis tools** - No neurospatial equivalent
❌ **R neuroscience lags Python** - More mature Python ecosystem
✅ **Python has complete stack** - movement + neurospatial + pynapple

### Strategic Position Confirmed

**neurospatial fills a gap that exists in BOTH languages**:
- Python: Had movement (tracking), lacked spatial analysis → neurospatial fills gap
- R: Has animovement (tracking), lacks spatial analysis → STILL GAP

**Python advantage**: Complete spatial neuroscience stack exists

---

## 12. Conclusion

**animovement is the R equivalent of movement** with the same scope and limitations:
- ✅ Tracking data cleaning
- ✅ Kinematics
- ✅ Unique: Trackball support
- ❌ No spatial analysis
- ❌ No behavioral segmentation

**Not directly relevant to neurospatial** because:
1. Different language (R vs Python)
2. Same scope as movement (already analyzed)
3. No new spatial capabilities
4. Experimental status

**Main insight**: Confirms neurospatial's unique value - even R users lack equivalent spatial analysis tools.

**Action items**: None (documentation only, Phase 5)

---

## 13. Package Metadata

| Property | Value |
|----------|-------|
| **GitHub** | animovement/animovement |
| **Language** | R |
| **Status** | Experimental (lifecycle badge) |
| **Created** | March 17, 2024 |
| **Stars** | 10 |
| **Forks** | 1 |
| **Issues** | 18 open |
| **Last updated** | October 18, 2025 |
| **Size** | 17.4 MB |
| **Collaborates with** | Python movement package |

---

**Package comparison summary** (16 packages analyzed):

| Package | Language | Platform | Unique Value | Limitation |
|---------|----------|----------|--------------|------------|
| **animovement** | R | R-only | Trackball support, tidyverse-friendly | No spatial analysis |
| **movement** ⭐ | Python | Cross | Track cleaning, kinematics | No spatial discretization |
| **neurospatial** | Python | Cross | Any graph, spatial primitives | No tracking (use movement) |
| **pynapple** | Python | Cross | Time-series excellence | No spatial discretization |
| **opexebo** | Python | Cross | Nobel validation, metrics | Regular grids only |

**Final count**: 16 packages analyzed, ecosystem survey complete ✅
