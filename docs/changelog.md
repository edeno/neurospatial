# Changelog

All notable changes to neurospatial will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Complete MkDocs documentation with Material theme
- API reference generated with mkdocstrings
- User guide with detailed examples
- Getting started tutorials

## [0.1.0] - 2024-11-03

### Added

- Initial release of neurospatial
- Core `Environment` class with factory methods
- Multiple layout engines (regular grid, hexagonal, triangular, graph-based)
- Region support for defining ROIs
- Composite environment functionality
- Alignment and transformation tools
- Comprehensive test suite
- NumPy-style docstrings throughout

### Features

- Automatic active bin detection from data samples
- NetworkX-based connectivity graphs
- 1D linearization for track-based experiments
- Spatial queries (bin_at, neighbors, shortest_path, distance_between)
- Visualization with matplotlib
- Morphological operations (dilation, closing, hole filling)

[Unreleased]: https://github.com/edeno/neurospatial/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/edeno/neurospatial/releases/tag/v0.1.0
