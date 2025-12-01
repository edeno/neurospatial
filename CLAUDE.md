# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

**Last Updated**: 2025-11-30 (Modular documentation structure)

---

## 📚 Documentation Index

This documentation is organized into focused modules. **Start with QUICKSTART.md**, then reference other guides as needed.

### Start Here

- **[QUICKSTART.md](.claude/QUICKSTART.md)** - Essential patterns and your first environment (~200 lines)
  - Critical rules (always use `uv run`, factory methods, bin_size)
  - Common patterns (create environment, neural analysis, animation)
  - Quick command reference

### Reference Guides

- **[ARCHITECTURE.md](.claude/ARCHITECTURE.md)** - Core architecture and design (~300 lines)
  - Three-layer design (Layout → Environment → Regions)
  - Mixin pattern for Environment
  - Animation system architecture
  - Testing structure

- **[API_REFERENCE.md](.claude/API_REFERENCE.md)** - Import patterns organized by feature (~150 lines)
  - Spatial analysis, neural analysis, decoding
  - Behavioral analysis, animation, visualization
  - NWB integration, serialization

- **[DEVELOPMENT.md](.claude/DEVELOPMENT.md)** - Development commands and workflow (~250 lines)
  - Testing, code quality, type checking
  - Git workflow, commit messages
  - NumPy docstring format (required)
  - Performance profiling

### Implementation Guides

- **[PATTERNS.md](.claude/PATTERNS.md)** - Design patterns you must follow (~400 lines)
  - Graph metadata requirements
  - Mixin pattern for Environment
  - Mypy type checking requirements
  - Protocol-based design
  - Fitted state pattern
  - Regions are immutable

### Problem Solving

- **[TROUBLESHOOTING.md](.claude/TROUBLESHOOTING.md)** - Common errors and fixes (~400 lines)
  - 13 common gotchas with ✅/❌ examples
  - Error messages and solutions
  - Performance issues
  - NWB-specific issues

### Advanced Topics

- **[ADVANCED.md](.claude/ADVANCED.md)** - Advanced features and integrations (~400 lines)
  - NWB integration (full reference)
  - Video overlay and calibration
  - Video annotation workflow
  - Track graph annotation (1D environments)
  - Large session optimization
  - Coordinate system details

---

## 🚀 Quick Start (30 seconds)

**If you just want to get started immediately:**

### Critical Rules

1. **ALWAYS use `uv run`** before Python commands
2. **NEVER modify bare `Environment()`** - use factory methods
3. **bin_size is required** for all Environment creation
4. **NumPy docstring format** for all documentation

### Essential Commands

```bash
uv run pytest                           # Run tests
uv run ruff check . && uv run ruff format .  # Lint and format
uv run mypy src/neurospatial/          # Type check
```

### Your First Environment

```python
from neurospatial import Environment
import numpy as np

# Generate sample data
positions = np.random.rand(100, 2) * 100

# Create environment (bin_size is REQUIRED)
env = Environment.from_samples(positions, bin_size=2.0)
env.units = "cm"
env.frame = "session1"

# Query
bin_idx = env.bin_at([50.0, 50.0])
neighbors = env.neighbors(bin_idx)
```

**👉 For more examples, see [QUICKSTART.md](.claude/QUICKSTART.md)**

---

## 📖 Project Overview

**neurospatial** is a Python library for discretizing continuous N-dimensional spatial environments into bins/nodes with connectivity graphs.

### Key Features

- **Flexible discretization**: Regular grids, hexagonal, triangular, masked, polygon-bounded
- **1D linearization**: Track-based environments (T-maze, linear track)
- **Neural analysis**: Place fields, Bayesian decoding, trajectory analysis
- **Visualization**: Interactive animation with napari, video export
- **NWB integration**: Read/write NeurodataWithoutBorders files (optional)

### Package Management

**CRITICAL: This project uses `uv` for package management.**

- Python version: 3.13
- **ALWAYS** use `uv run` to execute Python commands
- **NEVER** use bare `python`, `pip`, or `pytest` commands

**Why uv?** Automatically manages the virtual environment without manual activation.

---

## 🗺️ Navigation Guide

### I want to

**Get started quickly**
→ [QUICKSTART.md](.claude/QUICKSTART.md)

**Understand the codebase architecture**
→ [ARCHITECTURE.md](.claude/ARCHITECTURE.md)

**Find import statements for a feature**
→ [API_REFERENCE.md](.claude/API_REFERENCE.md)

**Run tests or commit code**
→ [DEVELOPMENT.md](.claude/DEVELOPMENT.md)

**Extend the codebase (new layout engines, mixins)**
→ [PATTERNS.md](.claude/PATTERNS.md)

**Fix an error or warning**
→ [TROUBLESHOOTING.md](.claude/TROUBLESHOOTING.md)

**Work with NWB files or video overlays**
→ [ADVANCED.md](.claude/ADVANCED.md)

---

## 🎯 Common Tasks Quick Links

### Neural Analysis

- **Place fields**: [QUICKSTART.md - Place fields](. claude/QUICKSTART.md#neural-analysis)
- **Bayesian decoding**: [QUICKSTART.md - Bayesian decoding](.claude/QUICKSTART.md#neural-analysis)
- **Trajectory analysis**: [QUICKSTART.md - Trajectory analysis](.claude/QUICKSTART.md#neural-analysis)

### Visualization

- **Animate fields**: [QUICKSTART.md - Animate spatial fields](.claude/QUICKSTART.md#visualization--animation)
- **Add overlays**: [QUICKSTART.md - Add trajectory overlays](.claude/QUICKSTART.md#visualization--animation)
- **Video overlay**: [ADVANCED.md - Video Overlay](.claude/ADVANCED.md#video-overlay-v050)

### Data Integration

- **NWB read/write**: [ADVANCED.md - NWB Integration](.claude/ADVANCED.md#nwb-integration-v070)
- **Video annotation**: [ADVANCED.md - Video Annotation](.claude/ADVANCED.md#video-annotation-v060)
- **Track graphs**: [ADVANCED.md - Track Graph Annotation](.claude/ADVANCED.md#track-graph-annotation-v090)

### Development

- **Run tests**: [DEVELOPMENT.md - Testing](.claude/DEVELOPMENT.md#testing)
- **Type checking**: [DEVELOPMENT.md - Type Checking](.claude/DEVELOPMENT.md#type-checking-with-mypy)
- **Create layout engines**: [PATTERNS.md - Protocol-Based Design](.claude/PATTERNS.md#protocol-based-design)
- **Mixin pattern**: [PATTERNS.md - Mixin Pattern](.claude/PATTERNS.md#mixin-pattern-for-environment)

### Troubleshooting

- **Common gotchas**: [TROUBLESHOOTING.md - Common Gotchas](.claude/TROUBLESHOOTING.md#common-gotchas)
- **Error messages**: [TROUBLESHOOTING.md - Error Messages](.claude/TROUBLESHOOTING.md#error-messages)
- **Performance issues**: [TROUBLESHOOTING.md - Performance Issues](.claude/TROUBLESHOOTING.md#performance-issues)

---

## 📝 Commit Message Format

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scope): description
fix(scope): description
docs(scope): description
test(scope): description
chore(scope): description
```

**Examples:**

- `feat(M3): add .info() method`
- `fix: correct version reference`
- `docs(M8): update CLAUDE.md with speed-based animation API`

---

## 🔍 Finding What You Need

### By Task Type

| Task | Document |
|------|----------|
| Create first environment | [QUICKSTART.md](.claude/QUICKSTART.md) |
| Find import statement | [API_REFERENCE.md](.claude/API_REFERENCE.md) |
| Run tests | [DEVELOPMENT.md](.claude/DEVELOPMENT.md#testing) |
| Fix error | [TROUBLESHOOTING.md](.claude/TROUBLESHOOTING.md) |
| Understand mixins | [PATTERNS.md](.claude/PATTERNS.md#mixin-pattern-for-environment) |
| NWB integration | [ADVANCED.md](.claude/ADVANCED.md#nwb-integration-v070) |

### By Expertise Level

**Beginner** (first time using neurospatial):

1. [QUICKSTART.md](.claude/QUICKSTART.md) - Read "Your First Environment"
2. [QUICKSTART.md](.claude/QUICKSTART.md) - Try common patterns
3. [TROUBLESHOOTING.md](.claude/TROUBLESHOOTING.md) - Reference when stuck

**Intermediate** (extending or modifying code):

1. [ARCHITECTURE.md](.claude/ARCHITECTURE.md) - Understand design
2. [PATTERNS.md](.claude/PATTERNS.md) - Learn constraints
3. [DEVELOPMENT.md](.claude/DEVELOPMENT.md) - Development workflow

**Advanced** (architecting features):

1. [PATTERNS.md](.claude/PATTERNS.md) - Master all patterns
2. [ADVANCED.md](.claude/ADVANCED.md) - Advanced integrations
3. [DEVELOPMENT.md](.claude/DEVELOPMENT.md) - Full dev workflow

---

## 📦 File Structure Summary

```
.claude/
├── QUICKSTART.md         # Start here - essential patterns
├── ARCHITECTURE.md       # Core design and architecture
├── API_REFERENCE.md      # Import patterns by feature
├── DEVELOPMENT.md        # Commands and workflow
├── PATTERNS.md           # Design patterns (must follow)
├── TROUBLESHOOTING.md    # Errors and fixes
└── ADVANCED.md           # NWB, video overlays, advanced topics
```

**Total documentation:** ~2,100 lines (down from 1,750 in single file)
**Main entry point:** ~300 lines (this file)
**Average file size:** ~300 lines per topic

---

## ❓ Still Can't Find It?

1. **Search across files**: Use Ctrl+F in your editor across `.claude/` directory
2. **Check table of contents**: Each file has detailed TOC at top
3. **Follow cross-references**: Documents link to related sections
4. **Ask specific questions**: The modular structure helps Claude Code load only relevant context

---

**For questions or issues**: <https://github.com/anthropics/claude-code/issues>
