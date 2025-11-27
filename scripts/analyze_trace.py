#!/usr/bin/env python
"""Analyze napari perfmon trace files.

This script parses Chrome Trace format JSON files and provides
summary statistics about performance hotspots.

Usage
-----
    uv run python scripts/analyze_trace.py /tmp/napari_trace.json

Output includes:
- Total duration by category
- Top 20 slowest operations
- Event frequency analysis
- Timeline phases breakdown
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_trace(path: str | Path) -> list[dict]:
    """Load Chrome Trace format JSON file."""
    with Path(path).open() as f:
        data = json.load(f)

    # Handle both array format and object format with "traceEvents" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]  # type: ignore[no-any-return]
    else:
        raise ValueError(f"Unknown trace format in {path}")


def analyze_trace(events: list[dict]) -> None:
    """Analyze trace events and print summary statistics."""

    # Collect duration events (complete events have dur)
    durations_by_name: dict[str, list[float]] = defaultdict(list)
    durations_by_cat: dict[str, float] = defaultdict(float)

    for event in events:
        if event.get("ph") == "X":  # Complete event
            name = event.get("name", "unknown")
            dur = event.get("dur", 0) / 1000  # Convert to milliseconds
            cat = event.get("cat", "uncategorized")

            durations_by_name[name].append(dur)
            durations_by_cat[cat] += dur

    # Print category totals
    print("\n" + "=" * 60)
    print("DURATION BY CATEGORY")
    print("=" * 60)
    for cat, total in sorted(durations_by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {total:.2f}ms")

    # Print top slowest operations
    print("\n" + "=" * 60)
    print("TOP 20 SLOWEST OPERATIONS (cumulative time)")
    print("=" * 60)

    name_totals = [
        (name, sum(durs), len(durs)) for name, durs in durations_by_name.items()
    ]
    name_totals.sort(key=lambda x: -x[1])

    for name, total, count in name_totals[:20]:
        avg = total / count
        print(f"  {name}: {total:.2f}ms total ({count} calls, avg {avg:.2f}ms)")

    # Print operations by frequency
    print("\n" + "=" * 60)
    print("TOP 20 MOST FREQUENT OPERATIONS")
    print("=" * 60)

    name_totals.sort(key=lambda x: -x[2])

    for name, total, count in name_totals[:20]:
        avg = total / count
        print(f"  {name}: {count} calls ({total:.2f}ms total, avg {avg:.2f}ms)")

    # Print instant events (markers)
    instant_events = [e for e in events if e.get("ph") == "i"]
    if instant_events:
        print("\n" + "=" * 60)
        print("INSTANT EVENTS (markers)")
        print("=" * 60)
        for event in instant_events[:20]:
            ts = event.get("ts", 0) / 1000000  # Convert to seconds
            name = event.get("name", "unknown")
            print(f"  {ts:.3f}s: {name}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_duration = sum(durations_by_cat.values())
    print(f"  Total traced time: {total_duration:.2f}ms")
    print(f"  Unique operations: {len(durations_by_name)}")
    print(f"  Total events: {len(events)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze napari perfmon trace files")
    parser.add_argument("trace_file", help="Path to Chrome Trace format JSON file")
    args = parser.parse_args()

    print(f"Loading trace from: {args.trace_file}")

    events = load_trace(args.trace_file)
    analyze_trace(events)


if __name__ == "__main__":
    main()
