"""Draw every simulated maze for the docs, from the current code.

mkdocs-gen-files runs this at build time and writes
``user-guide/images/simulated-mazes.png``, so the picture always shows the
mazes as the package builds them.
"""

import io
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mkdocs_gen_files
import numpy as np
from matplotlib.lines import Line2D

import neurospatial.simulation.mazes as mazes

# Mazes grouped by their panel in Wijnen et al. 2024, Fig. 1.
GROUPS = {
    "a": ["t_maze", "y_maze", "w_maze", "small_hex_maze"],
    "b": ["repeated_y_maze", "repeated_t_maze", "hampton_court_maze"],
    "c": ["radial_arm_maze", "barnes_maze", "cheeseboard_maze", "watermaze"],
    "d": ["crossword_maze", "honeycomb_maze", "hamlet_maze"],
    "e": ["rat_hexmaze"],
    "not in figure": ["linear_track", "plus_maze", "square_maze", "sungod_maze"],
}
ROLE_STYLE = {
    "reward_port": {"marker": "o", "color": "#d62728"},
    "hole": {"marker": "o", "color": "#7f7f7f", "mfc": "none"},
    "escape_hole": {"marker": "o", "color": "#2ca02c"},
    "platform": {"marker": "s", "color": "#1f77b4"},
    "start_box": {"marker": "^", "color": "#ff7f0e"},
    "goal_box": {"marker": "*", "color": "#9467bd"},
    "cue": {"marker": "D", "color": "#8c564b"},
}
# Polygon regions drawn as outlines; module, bridge and blind-alley regions
# would only repeat the floor.
SKIP_OUTLINES = ("module_", "bridge_", "blind_")


def draw_maze(ax, name, panel):
    maze = getattr(mazes, f"make_{name}")()
    env = maze.env_2d
    floor = env.arena.polygon
    for part in getattr(floor, "geoms", [floor]):
        ax.fill(*part.exterior.xy, fc="0.85", ec="0.5", lw=0.5)
        for hole in part.interiors:
            ax.fill(*hole.xy, fc="white", ec="0.5", lw=0.5)
    if maze.env_track is not None:
        track = maze.env_track.bin_centers
        ax.scatter(track[:, 0], track[:, 1], s=0.5, c="#1f77b4", lw=0)
    for region in env.regions.values():
        if region.kind == "polygon" and not region.name.startswith(SKIP_OUTLINES):
            for part in getattr(region.data, "geoms", [region.data]):
                ax.plot(*part.exterior.xy, lw=0.8, c="k")
    size = 3 if len(env.landmarks) > 50 else 6
    for landmark in env.landmarks.values():
        point = (
            np.asarray(landmark.data)
            if landmark.kind == "point"
            else np.asarray(landmark.data.centroid.coords[0])
        )
        style = ROLE_STYLE[landmark.metadata["role"]]
        ax.plot(*point, ms=size, ls="none", **style)
    lo, hi = env.bin_centers.min(0), env.bin_centers.max(0)
    width, height = hi - lo
    ax.set_title(f"({panel}) {name}\n{width:.0f} x {height:.0f} cm", fontsize=11)
    span = max(width, height)
    ax.plot([lo[0], lo[0] + 100.0], [lo[1] - 0.04 * span] * 2, "k-", lw=3)
    ax.text(lo[0] + 50.0, lo[1] - 0.09 * span, "1 m", ha="center", va="top")
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    names = [(name, panel) for panel, group in GROUPS.items() for name in group]
    n_cols = 4
    n_rows = -(-len(names) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    with warnings.catch_warnings():
        # The 9 x 5 m rat HexMaze warns about its grid size at 2 cm bins.
        warnings.simplefilter("ignore", UserWarning)
        # The grid has more panels than mazes; the spares are hidden below.
        for ax, (name, panel) in zip(axes.flat, names, strict=False):
            draw_maze(ax, name, panel)
    for ax in axes.flat[len(names) :]:
        ax.axis("off")
    handles = [
        Line2D([], [], ls="none", ms=8, label=role, **style)
        for role, style in ROLE_STYLE.items()
        if role != "cue"
    ]
    handles += [
        Line2D([], [], ls="none", marker="s", color="0.85", ms=10, label="floor"),
        Line2D([], [], ls="none", marker="o", color="#1f77b4", label="track"),
        Line2D([], [], color="k", lw=0.8, label="region outline"),
    ]
    fig.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        fontsize=12,
        frameon=False,
    )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=80)
    plt.close(fig)
    with mkdocs_gen_files.open("user-guide/images/simulated-mazes.png", "wb") as f:
        f.write(buffer.getvalue())


main()
