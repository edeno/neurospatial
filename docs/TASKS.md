# neurospatial Package UX Refactoring Tasks

**Based on**: UX Review (2025-11-01) and Implementation Plan
**Goal**: Refactor neurospatial package from NEEDS_POLISH to USER_READY
**Total Estimated Effort**: 5-6 days

---

## Milestone 1: Critical Fixes - Make Library Discoverable 🔴

**Goal**: Fix blocking issues preventing new user adoption
**Duration**: 1-2 days
**Success Metric**: New user can install and create first Environment in < 10 minutes

### Documentation

- [x] **Write comprehensive README.md**
  - [x] Project overview (what, who, why - 2-3 paragraphs)
  - [x] Key features (bulleted list)
  - [x] Installation instructions (pip and development setup)
  - [x] Quickstart example (copy-pasteable code)
  - [x] Core concepts section (bins, active bins, connectivity, layout engines)
  - [x] Common use cases with links
  - [x] Documentation links section
  - [x] Citation information
  - [x] Verify all examples run without modification

### Critical Bug Fixes

- [x] **Fix Regions.update() documentation bug** (`src/neurospatial/regions/core.py`)
  - [x] Implement `update_region()` method with full docstring
  - [x] Update `__setitem__` error message to reference correct methods
  - [x] Add tests for `update_region()` behavior
  - [x] Add tests for error conditions (KeyError when region doesn't exist)
  - [x] Fix metadata preservation when not explicitly provided
  - [x] Rename method to avoid MutableMapping.update() conflict

### Error Message Improvements

- [x] **Improve "No active bins found" error** (`src/neurospatial/layout/engines/regular_grid.py:143-196`)
  - [x] Add diagnostic information (data range, bin_size, thresholds)
  - [x] Explain WHAT went wrong (no bins after filtering)
  - [x] Explain WHY (3 common causes)
  - [x] Explain HOW to fix (specific suggestions)
  - [x] Add test that triggers error and validates message content

### Terminology & Concepts

- [x] **Add "active bins" definition** (`src/neurospatial/environment.py`)
  - [x] Add Terminology section to Environment class docstring
  - [x] Define "active bins" with examples
  - [x] Explain scientific motivation
  - [x] Cross-reference `infer_active_bins` parameter

- [x] **Add bin_size units clarification** (all factory methods)
  - [x] Update `from_samples()` docstring
  - [x] Update `from_polygon()` docstring
  - [x] Update `from_mask()` docstring (grid_edges)
  - [x] Update `from_image()` docstring
  - [x] Update `from_graph()` docstring
  - [x] Add warning about Hexagonal vs RegularGrid interpretation (already present)
  - [x] Update examples to show units in comments

---

## Milestone 2: Clarity & Discoverability 🟠

**Goal**: Reduce confusion and improve method/feature discovery
**Duration**: 2-3 days
**Success Metric**: 80% reduction in "which method should I use?" questions

### Navigation & Discovery

- [x] **Add Factory Method Selection Guide** (`src/neurospatial/environment.py`)
  - [x] Add guide to Environment class docstring (after class summary)
  - [x] Include decision tree for all 6 factory methods
  - [x] Order by frequency of use (most common first)
  - [x] Add brief use case description for each method
  - [x] Cross-reference individual method docstrings

- [x] **Add "See Also" cross-references** (all factory methods)
  - [x] Add to `from_samples()` → reference polygon, mask, image, graph, layout
  - [x] Add to `from_polygon()` → reference samples, mask, image
  - [x] Add to `from_mask()` → reference samples, polygon, image
  - [x] Add to `from_image()` → reference mask, polygon, samples
  - [x] Add to `from_graph()` → reference samples, layout
  - [x] Add to `from_layout()` → reference all specialized methods
  - [x] Ensure bidirectional references

### Scientific Terminology

- [x] **Define scientific terms for non-experts**
  - [x] "Place fields" in `src/neurospatial/alignment.py:27`
  - [x] "Geodesic distance" in `src/neurospatial/environment.py:1237`
  - [x] "Linearization" in `src/neurospatial/environment.py:1130`
  - [x] Add brief parenthetical definitions
  - [x] Standardize format using parentheses for consistency

### Error Message Standardization

- [x] **Standardize error messages to show actual values**
  - [x] `layout/helpers/utils.py:96` - bin_size validation
  - [x] `layout/helpers/utils.py:148` - bin_count_threshold validation
  - [x] `calibration.py:48-50` - offset_px validation
  - [x] `layout/helpers/regular_grid.py` - parameter validations
  - [x] `layout/helpers/hexagonal.py` - parameter validations
  - [x] `layout/helpers/graph.py` - parameter validations
  - [x] `layout/mixins.py` - state validation errors (already had good messages)
  - [x] Apply pattern: `f"{param} must be {constraint} (got {actual_value})"`
  - [x] Include type information where helpful
  - [x] Truncate long values (NumPy handles automatically)

- [x] **Enhance @check_fitted error with examples** (`src/neurospatial/environment.py:64-73`)
  - [x] Add example of correct usage (Environment.from_samples)
  - [x] Add example of incorrect usage to avoid (Environment())
  - [x] Keep error message concise but informative

- [x] **Improve CompositeEnvironment dimension mismatch error** (`src/neurospatial/composite.py:114-117`)
  - [x] Add "To fix" guidance section
  - [x] Explain common cause (mixed 2D/3D environments)
  - [x] Suggest checking data_samples dimensionality

### API Consistency

- [x] **Audit and standardize bin_size defaults**
  - [x] Review all factory methods for consistency
  - [x] Decide: Option A (all required) vs Option B (all have defaults)
  - [x] Update method signatures if changing defaults
  - [x] Update all docstrings to reflect decision
  - [x] Update all examples to include explicit bin_size
  - [x] Update all tests for any breaking changes
  - [x] Document breaking changes if applicable

---

## Milestone 3: Polish & User Experience 🟡

**Goal**: Professional finish and convenience features
**Duration**: 1-2 days
**Success Metric**: Zero "how do I debug this?" questions on common errors

### Proactive Guidance

- [x] **Add "Common Pitfalls" sections**
  - [x] Add to `from_samples()` docstring
    - [x] bin_size too large pitfall
    - [x] bin_count_threshold too high pitfall
    - [x] Mismatched units pitfall
    - [x] Missing morphological operations pitfall
  - [x] Add to `CompositeEnvironment.__init__()` docstring
    - [x] Dimension mismatch pitfall
    - [x] No bridge edges pitfall
    - [x] Overlapping bins pitfall

### Enhanced Error Handling

- [x] **Improve type validation for sequences**
  - [x] Add try-except for bin_size conversion in `layout/helpers/utils.py`
  - [x] Add try-except for dimension_ranges in `layout/helpers/regular_grid.py`
  - [x] Add try-except for data_samples in `environment.py`
  - [x] Provide helpful error messages for type errors
  - [x] Validate NaN/Inf separately with specific errors
  - [x] Preserve original exception with `from e`
  - [x] Add tests for invalid inputs (strings, mixed types, NaN)

### Convenience Features

- [x] **Add custom __repr__ for Environment** (`src/neurospatial/environment.py`)
  - [x] Implement concise single-line representation
  - [x] Show: name, n_dims, n_bins, layout type
  - [x] Handle edge cases (empty name, etc.)
  - [x] Add docstring with examples
  - [x] Add tests for various configurations
  - [x] Add _repr_html_() for rich Jupyter notebook display

- [x] **Add .info() method for diagnostics** (`src/neurospatial/environment.py`)
  - [x] Implement multi-line detailed summary
  - [x] Show: name, dimensions, bins, layout, extent, bin sizes, linearization, regions
  - [x] Format output for readability
  - [x] Handle edge cases (no regions, variable bin sizes, etc.)
  - [x] Add docstring with examples
  - [x] Add tests for various configurations

### Validation Warnings (Optional)

- [ ] **Add validation warnings for unusual parameters** (`from_samples()`)
  - [ ] Warn if bin_size > 20% of data range
  - [ ] Warn if bin_count_threshold > n_samples / 10
  - [ ] Warn about sparse data with no morphological operations
  - [ ] Use `stacklevel=2` to point to user code
  - [ ] Make warnings informative, not alarming
  - [ ] Add tests to verify warnings issued correctly
  - [ ] Consider adding `suppress_warnings` parameter

### Visual Documentation (Lower Priority)

- [ ] **Create morphological operations visual guide**
  - [ ] Create Jupyter notebook in `docs/examples/`
  - [ ] Setup example data with holes and gaps
  - [ ] Create figures for each operation (original, dilate, fill_holes, close_gaps, combined)
  - [ ] Add explanatory text for each operation
  - [ ] Link notebook from `from_samples()` docstring
  - [ ] Test that notebook runs without errors

---

## Milestone 4: Testing & Quality Assurance ✅

**Goal**: Ensure all improvements work correctly and don't break existing functionality
**Duration**: Concurrent with implementation
**Success Metric**: All tests pass, >95% coverage of new code

### Unit Tests

- [x] **Test new functionality**
  - [x] Regions.update() method tests
  - [x] Error message content validation tests
  - [x] __repr__ output format tests
  - [x] .info() output format tests
  - [ ] Type validation error tests
  - [ ] Warning emission tests

- [x] **Test documentation**
  - [x] Run doctests: `uv run pytest --doctest-modules`
  - [x] Verify all examples are syntactically correct
  - [x] Verify examples produce expected output
  - [x] Check docstring rendering with `help()`

### Integration Tests

- [ ] **Create UX integration test suite** (`tests/test_ux_improvements.py`)
  - [ ] Test first-run experience (README workflow)
  - [ ] Test error messages follow WHAT/WHY/HOW pattern
  - [ ] Test factory method discovery (selection guide exists)
  - [ ] Test that scientific terms are defined
  - [ ] Test cross-references work

### Manual QA

- [ ] **Sprint 1 QA Checklist**
  - [ ] Install in fresh environment and follow README
  - [ ] Time first Environment creation (should be < 10 minutes)
  - [ ] Verify README example runs without modification
  - [ ] Trigger "no active bins" error and verify message helpful
  - [ ] Try to update a region and verify it works

- [ ] **Sprint 2 QA Checklist**
  - [ ] Read factory method guide, verify clarity
  - [ ] Check that scientific terms are defined at first use
  - [ ] Trigger 5 different errors, verify all show actual values
  - [ ] Use `help(Environment.from_samples)`, verify "See Also" section

- [ ] **Sprint 3 QA Checklist**
  - [ ] Read Common Pitfalls, verify actionable
  - [ ] Create environment and check __repr__ output
  - [ ] Call env.info() and verify readable
  - [ ] Provide invalid bin_size and verify error clear

### Regression Testing

- [ ] **Ensure no breaking changes**
  - [ ] Run full test suite: `uv run pytest`
  - [ ] Run with coverage: `uv run pytest --cov=src/neurospatial`
  - [ ] Check coverage report for gaps
  - [ ] Run type checking: `uv run mypy src/neurospatial`
  - [ ] Run linting: `uv run ruff check .`
  - [ ] Run formatting: `uv run ruff format . --check`

---

## Milestone 5: Documentation & Release Prep 📚

**Goal**: Prepare for public release with complete documentation
**Duration**: 1 day
**Success Metric**: Documentation is comprehensive and error-free

### Documentation Review

- [ ] **Verify documentation completeness**
  - [ ] All public methods have docstrings
  - [ ] All parameters documented
  - [ ] All return types documented
  - [ ] All exceptions documented
  - [ ] Examples provided for key methods

- [ ] **Check rendering**
  - [ ] README renders correctly on GitHub
  - [ ] Docstrings render correctly in `help()`
  - [ ] Code blocks are properly formatted
  - [ ] Links are valid
  - [ ] No Markdown syntax errors

### Change Documentation

- [ ] **Update CHANGELOG.md** (or create if doesn't exist)
  - [ ] Document all UX improvements
  - [ ] List breaking changes (if bin_size defaults removed)
  - [ ] List new features (update(), info(), __repr__)
  - [ ] List bug fixes (documentation bugs, error messages)
  - [ ] List documentation improvements

- [ ] **Update CLAUDE.md if needed**
  - [ ] Document new patterns to follow
  - [ ] Update common gotchas if applicable
  - [ ] Add new terminology to glossary

### Release Preparation

- [ ] **Version bump**
  - [ ] Decide on version number (suggest v0.2.0 for breaking changes)
  - [ ] Update version in `pyproject.toml`
  - [ ] Update version references in documentation

- [ ] **Create release notes**
  - [ ] Highlight key UX improvements
  - [ ] Provide migration guide for breaking changes
  - [ ] Thank contributors
  - [ ] Link to detailed CHANGELOG

---

## Success Metrics & Validation

### Quantitative Targets

- [ ] **Time to first successful Environment creation**: < 10 minutes (Sprint 1), < 5 minutes (Sprint 2), < 3 minutes (Sprint 3)
- [ ] **Error messages with diagnostic info**: > 60% (Sprint 1), > 80% (Sprint 2), > 90% (Sprint 3)
- [ ] **Methods with "See Also" sections**: 100% of factory methods
- [ ] **Scientific terms defined**: 100%
- [ ] **Test coverage**: > 95% of new code
- [ ] **All doctests passing**: 100%

### Qualitative Targets

- [ ] **User feedback**: "How easy was it to get started?" → "Very easy"
- [ ] **Error helpfulness**: "When you encountered errors, were they helpful?" → "Yes, I knew what to fix"
- [ ] **Method discovery**: "Could you find the right method to use?" → "Yes, the guide was clear"
- [ ] **Understanding**: "Did you understand what the library does?" → "Yes, README explained it well"

---

## Optional Enhancements (Future Work)

These items are not critical for USER_READY status but would further improve the library:

- [ ] Interactive tutorial (Jupyter notebook walkthrough)
- [ ] Examples gallery with real neuroscience data
- [ ] Video walkthrough / screencast
- [ ] API reference documentation (Sphinx)
- [ ] Contribution guide for custom layout engines
- [ ] Performance benchmarks documentation
- [ ] Migration guides from similar libraries
- [ ] Integration examples with common neuroscience packages (NWB, etc.)

---

## Notes

**Priority**: Complete milestones in order (1 → 2 → 3 → 4 → 5)

**Flexibility**: Milestones 2-3 can be split across multiple PRs if needed

**Testing**: Add tests concurrent with implementation (don't defer to Milestone 4)

**Review**: Get code review after each milestone before proceeding

**Documentation**: Keep TASKS.md updated as work progresses (check off completed items)

**Git Strategy**:
- Create feature branch: `feature/ux-improvements`
- Commit frequently with clear messages
- Consider separate PRs per milestone for easier review
- Squash commits before merging to main

**Risk Management**:
- If time-constrained, prioritize Milestone 1 (Critical Fixes)
- Milestones 2-3 can be deferred but significantly improve UX
- Milestone 4 (Testing) is mandatory for quality
- Milestone 5 (Documentation) is mandatory for release

---

**Last Updated**: 2025-11-01
**Status**: Planning phase - ready to begin implementation
**Next Action**: Begin Milestone 1, Task 1 (Write README.md)
