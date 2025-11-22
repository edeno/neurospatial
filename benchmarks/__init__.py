"""Benchmark scripts for animation backend performance testing.

This package provides CLI tools to measure baseline performance metrics
for each animation backend (napari, video, widget).

Usage
-----
Run individual benchmarks from the project root:

    uv run python benchmarks/bench_napari.py
    uv run python benchmarks/bench_video.py
    uv run python benchmarks/bench_widget.py

Results are printed to stdout and should be copied to BASELINE.md.
"""
