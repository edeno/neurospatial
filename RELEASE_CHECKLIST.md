# Release Checklist for neurospatial v0.1.0

## Pre-Release Verification (Completed)

### Code Quality
- [x] All 676 tests passing (5 skipped, 7 warnings - all expected)
- [x] Ruff linting passes with no errors
- [x] Code formatting verified (80 files formatted correctly)
- [x] MyPy type checking passes (2 expected warnings for matplotlib 3D stubs)
- [x] Test file naming conflict fixed (renamed `tests/regions/test_io.py` to `test_serialization.py`)

### Documentation
- [x] README.md reviewed and current
- [x] CHANGELOG.md up to date (v0.1.0 entry complete)
- [x] CLAUDE.md reviewed and accurate
- [x] All docstrings follow NumPy format
- [x] Examples README citation year corrected (2024 → 2025)
- [x] Documentation site builds successfully in strict mode
- [x] API documentation complete and auto-generated

### Package Build
- [x] Package builds successfully
  - Source distribution: `dist/neurospatial-0.1.0.tar.gz`
  - Wheel: `dist/neurospatial-0.1.0-py3-none-any.whl`
- [x] Version is 0.1.0 across all files
- [x] Dependencies specified correctly in pyproject.toml
- [x] License file present (MIT)

### Project Metadata
- [x] Version: 0.1.0
- [x] Python support: 3.10+
- [x] License: MIT
- [x] Author: Eric Denovellis <eric.denovellis@ucsf.edu>
- [x] Repository: https://github.com/edeno/neurospatial
- [x] Status: Alpha

## Release Steps

### 1. Final Code Freeze
```bash
# Ensure working directory is clean
git status

# Verify all changes are committed
git log --oneline -5
```

### 2. Tag the Release
```bash
# Create annotated tag for v0.1.0
git tag -a v0.1.0 -m "Release version 0.1.0

Initial alpha release of neurospatial with:
- Multiple layout engines (grid, hexagonal, graph, masked, polygon, triangular)
- CompositeEnvironment with API parity
- KDTree-optimized spatial queries
- Comprehensive documentation and examples
- 676 passing tests"

# Verify tag
git tag -n9 v0.1.0

# Push tag to trigger release workflow
git push origin v0.1.0
```

### 3. GitHub Release (Automatic via CI/CD)
The release workflow will:
- Auto-generate changelog from conventional commits
- Create GitHub release with notes
- Update CHANGELOG.md automatically

### 4. PyPI Publication (Automatic via CI/CD)
The publish workflow will:
- Trigger on GitHub release creation
- Build distributions with uv
- Publish to PyPI using trusted publishing (OIDC)
- Publish to TestPyPI for prereleases

### 5. Documentation Deployment (Manual if needed)
```bash
# If needed, manually deploy docs
uv run mkdocs gh-deploy --clean --strict

# Verify at: https://edeno.github.io/neurospatial/
```

### 6. Post-Release Verification
- [ ] Verify GitHub release created: https://github.com/edeno/neurospatial/releases/tag/v0.1.0
- [ ] Verify PyPI package: https://pypi.org/project/neurospatial/0.1.0/
- [ ] Test installation from PyPI: `pip install neurospatial==0.1.0`
- [ ] Verify documentation site: https://edeno.github.io/neurospatial/
- [ ] Test import: `python -c "from neurospatial import Environment; print(Environment.__module__)"`

## Post-Release Tasks

### 1. Announce Release
- [ ] Update GitHub repository description
- [ ] Post announcement (if applicable)
- [ ] Update any external documentation links

### 2. Prepare for Next Version
```bash
# Optionally bump to 0.1.1-dev for development
# Update pyproject.toml version = "0.1.1-dev"
# Update CHANGELOG.md with [Unreleased] section
```

## Rollback Plan

If issues are discovered after release:

### Option 1: Yank Release (for critical bugs)
```bash
# Yank from PyPI (keeps files but discourages installation)
uv publish --yank 0.1.0
```

### Option 2: Hotfix Release
```bash
# Create hotfix branch
git checkout -b hotfix/0.1.1 v0.1.0

# Make fixes, commit with conventional commits
git commit -m "fix: critical bug description"

# Tag and release 0.1.1
git tag -a v0.1.1 -m "Hotfix release 0.1.1"
git push origin v0.1.1
```

## Notes

### Tested Configuration
neurospatial v0.1.0 tested with:
- Python: 3.13.5
- numpy: 2.3.4
- pandas: 2.3.3
- matplotlib: 3.10.7
- networkx: 3.5
- scipy: 1.16.3
- scikit-learn: 1.7.2
- shapely: 2.1.2
- track-linearization: 2.4.0

### Known Issues/Limitations
- matplotlib 3D tools missing type stubs (expected, not a runtime issue)
- Status is Alpha - API may change in future versions
- Some notebooks generate div warnings during build (cosmetic, not functional)

### CI/CD Workflows
The project has three GitHub Actions workflows:
1. **test.yml**: Runs on push/PR, tests across Python 3.10-3.13 and multiple OS
2. **publish.yml**: Triggers on release, publishes to PyPI
3. **release.yml**: Triggers on version tags, creates GitHub releases

### Required GitHub Secrets/Settings
- PyPI trusted publishing configured (OIDC token)
- GitHub Pages enabled from gh-pages branch
- (Optional) CODECOV_TOKEN for coverage reports

## Checklist Summary

**Pre-release**: ✅ All 9 verification items completed
**Release**: Ready to tag and push v0.1.0
**Post-release**: Verification and announcement tasks pending

---

**Release Manager**: Claude Code
**Date Prepared**: 2025-11-03
**Package Version**: 0.1.0
**Release Type**: Alpha
