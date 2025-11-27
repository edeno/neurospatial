"""Visualization script for all 17 simulation maze environments.

This script creates visualizations of all implemented mazes from the
neurospatial.simulation.mazes module, showing both the 2D polygon
environments and the track graph structures where applicable.

Based on Wijnen et al. 2024 (Brain Structure & Function), Figure 1.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from neurospatial.simulation.mazes import (
    make_barnes_maze,
    make_cheeseboard_maze,
    make_crossword_maze,
    make_hamlet_maze,
    make_hampton_court_maze,
    make_honeycomb_maze,
    make_linear_track,
    make_radial_arm_maze,
    make_rat_hexmaze,
    make_repeated_t_maze,
    make_repeated_y_maze,
    make_small_hex_maze,
    make_sungod_maze,
    make_t_maze,
    make_w_maze,
    make_watermaze,
    make_y_maze,
)


def plot_maze_2d(ax, maze, title, max_regions=10):
    """Plot the 2D environment with bins and regions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    maze : MazeEnvironments
        The maze to plot.
    title : str
        Title for the plot.
    max_regions : int
        Maximum number of regions to label (to avoid clutter).
    """
    env = maze.env_2d

    # Plot bin centers
    bin_centers = env.bin_centers
    ax.scatter(
        bin_centers[:, 0],
        bin_centers[:, 1],
        c="lightblue",
        s=5,
        alpha=0.5,
    )

    # Separate regions by type for special handling
    hole_regions = [(n, r) for n, r in env.regions.items() if n.startswith("hole_")]
    reward_regions = [(n, r) for n, r in env.regions.items() if n.startswith("reward_")]
    well_regions = [(n, r) for n, r in env.regions.items() if n.startswith("well_")]
    platform_regions = [
        (n, r) for n, r in env.regions.items() if n.startswith("platform_")
    ]
    other_regions = [
        (n, r)
        for n, r in env.regions.items()
        if not n.startswith("hole_")
        and not n.startswith("reward_")
        and not n.startswith("well_")
        and not n.startswith("platform_")
    ]

    # Plot ALL hole regions (Barnes maze) - small circles around perimeter
    if hole_regions:
        hole_positions = np.array(
            [r.data for n, r in hole_regions if r.kind == "point"]
        )
        if len(hole_positions) > 0:
            ax.scatter(
                hole_positions[:, 0],
                hole_positions[:, 1],
                c="darkgray",
                s=60,
                marker="o",
                edgecolors="black",
                linewidth=1.0,
                zorder=5,
                label=f"holes ({len(hole_positions)})",
            )

    # Plot ALL reward regions (Cheeseboard) - small dots across surface
    if reward_regions:
        reward_positions = np.array(
            [r.data for n, r in reward_regions if r.kind == "point"]
        )
        if len(reward_positions) > 0:
            ax.scatter(
                reward_positions[:, 0],
                reward_positions[:, 1],
                c="gold",
                s=25,
                marker="o",
                edgecolors="orange",
                linewidth=0.5,
                zorder=5,
                label=f"rewards ({len(reward_positions)})",
            )

    # Plot ALL well regions (W-maze, etc.) - small dots
    if well_regions:
        well_positions = np.array(
            [r.data for n, r in well_regions if r.kind == "point"]
        )
        if len(well_positions) > 0:
            ax.scatter(
                well_positions[:, 0],
                well_positions[:, 1],
                c="gold",
                s=40,
                marker="o",
                edgecolors="orange",
                linewidth=0.5,
                zorder=5,
                label=f"wells ({len(well_positions)})",
            )

    # Plot ALL platform regions (Honeycomb) - small hexagons
    if platform_regions:
        platform_positions = np.array(
            [r.data for n, r in platform_regions if r.kind == "point"]
        )
        if len(platform_positions) > 0:
            ax.scatter(
                platform_positions[:, 0],
                platform_positions[:, 1],
                c="lightcoral",
                s=40,
                marker="h",
                edgecolors="red",
                linewidth=0.5,
                zorder=5,
                label=f"platforms ({len(platform_positions)})",
            )

    # Plot other key regions (start, goal, junction, etc.) - limited number
    region_colors = plt.cm.Set1(np.linspace(0, 1, max_regions))
    plotted = 0
    for _i, (name, region) in enumerate(other_regions):
        if plotted >= max_regions:
            break
        if region.kind == "point":
            color = region_colors[plotted % len(region_colors)]
            ax.scatter(
                region.data[0],
                region.data[1],
                c=[color],
                s=80,
                marker="*",
                edgecolors="black",
                linewidth=0.5,
                zorder=6,
                label=name if plotted < 6 else None,
            )
            plotted += 1

    ax.set_title(f"{title}\n({env.n_bins} bins)", fontsize=9)
    ax.set_xlabel("X (cm)", fontsize=8)
    ax.set_ylabel("Y (cm)", fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    # Show legend if we have meaningful items
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=5, ncol=2 if len(handles) > 4 else 1)
    ax.grid(True, alpha=0.3)


def plot_track_graph(ax, maze, title):
    """Plot the track graph structure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    maze : MazeEnvironments
        The maze to plot.
    title : str
        Title for the plot.
    """
    if maze.env_track is None:
        ax.text(
            0.5,
            0.5,
            "No track graph\n(open field)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_title(f"{title}\nTrack Graph", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    env = maze.env_track
    graph = env.connectivity

    # Get positions from node attributes
    pos = {node: graph.nodes[node]["pos"] for node in graph.nodes()}

    # Draw edges
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.5, width=1, edge_color="gray")

    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color="lightcoral",
        node_size=20,
        alpha=0.7,
    )

    ax.set_title(
        f"{title}\n({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)",
        fontsize=9,
    )
    ax.set_xlabel("X (cm)", fontsize=8)
    ax.set_ylabel("Y (cm)", fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


def create_all_mazes():
    """Create all 17 mazes with default dimensions.

    Returns
    -------
    dict
        Dictionary mapping maze names to (maze, category) tuples.
    """
    mazes = {}

    # Simple Corridor Mazes
    print("Creating Simple Corridor Mazes...")
    mazes["Linear Track"] = (make_linear_track(bin_size=2.0), "Corridor")
    mazes["T-Maze"] = (make_t_maze(bin_size=2.0), "Corridor")
    mazes["Y-Maze"] = (make_y_maze(bin_size=2.0), "Corridor")
    mazes["W-Maze"] = (make_w_maze(bin_size=2.0), "Corridor")
    mazes["Small Hex"] = (make_small_hex_maze(bin_size=2.0), "Corridor")

    # Open-Field Mazes
    print("Creating Open-Field Mazes...")
    mazes["Watermaze"] = (make_watermaze(bin_size=3.0), "Open-Field")
    mazes["Barnes"] = (make_barnes_maze(bin_size=3.0), "Open-Field")
    mazes["Cheeseboard"] = (make_cheeseboard_maze(bin_size=3.0), "Open-Field")
    mazes["Radial Arm"] = (make_radial_arm_maze(bin_size=2.0), "Open-Field")
    mazes["Sungod"] = (make_sungod_maze(bin_size=2.0), "Open-Field")

    # Repeated Alleyway Mazes
    print("Creating Repeated Alleyway Mazes...")
    mazes["Repeated Y"] = (make_repeated_y_maze(bin_size=2.0), "Repeated")
    mazes["Repeated T"] = (make_repeated_t_maze(bin_size=2.0), "Repeated")
    mazes["Hampton Court"] = (make_hampton_court_maze(bin_size=4.0), "Repeated")

    # Structured Lattice Mazes
    print("Creating Structured Lattice Mazes...")
    mazes["Crossword"] = (make_crossword_maze(bin_size=2.0), "Lattice")
    mazes["Honeycomb"] = (make_honeycomb_maze(bin_size=3.0), "Lattice")
    mazes["Hamlet"] = (make_hamlet_maze(bin_size=2.0), "Lattice")

    # Complex Mazes
    print("Creating Complex Mazes...")
    mazes["Rat HexMaze"] = (make_rat_hexmaze(bin_size=4.0), "Complex")

    return mazes


def visualize_all_2d(mazes, save_path="scripts/maze_visualization.png"):
    """Create a grid visualization of all maze 2D environments.

    Parameters
    ----------
    mazes : dict
        Dictionary from create_all_mazes().
    save_path : str
        Path to save the figure.
    """
    n_mazes = len(mazes)
    n_cols = 4
    n_rows = (n_mazes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, (name, (maze, _category)) in enumerate(mazes.items()):
        plot_maze_2d(axes[i], maze, name)

    # Hide unused axes
    for i in range(n_mazes, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    fig.suptitle(
        "Simulation Mazes: 2D Environments (Wijnen et al. 2024)",
        fontsize=14,
        y=1.01,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n2D visualization saved to: {save_path}")

    return fig


def visualize_track_graphs(mazes, save_path="scripts/maze_track_graphs.png"):
    """Create a grid visualization of all maze track graphs.

    Parameters
    ----------
    mazes : dict
        Dictionary from create_all_mazes().
    save_path : str
        Path to save the figure.
    """
    # Filter to mazes with track graphs
    track_mazes = {k: v for k, v in mazes.items() if v[0].env_track is not None}
    n_mazes = len(track_mazes)

    if n_mazes == 0:
        print("No mazes have track graphs.")
        return None

    n_cols = 4
    n_rows = (n_mazes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    axes = np.array(axes).flatten()

    for i, (name, (maze, _category)) in enumerate(track_mazes.items()):
        plot_track_graph(axes[i], maze, name)

    # Hide unused axes
    for i in range(n_mazes, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    fig.suptitle(
        "Simulation Mazes: Track Graphs (1D Linearized)",
        fontsize=14,
        y=1.01,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Track graph visualization saved to: {save_path}")

    return fig


def print_maze_summary(mazes):
    """Print summary statistics for all mazes.

    Parameters
    ----------
    mazes : dict
        Dictionary from create_all_mazes().
    """
    print("\n" + "=" * 60)
    print("MAZE SUMMARY")
    print("=" * 60)

    for name, (maze, category) in mazes.items():
        env_2d = maze.env_2d
        env_track = maze.env_track
        n_regions = len(env_2d.regions)
        track_info = (
            f"{env_track.connectivity.number_of_nodes()} nodes" if env_track else "None"
        )
        print(
            f"{name:15s} | {category:10s} | {env_2d.n_bins:5d} bins | {n_regions:3d} regions | Track: {track_info}"
        )

    print("=" * 60)


def main():
    """Create and display all maze visualizations."""
    print("Creating all 17 simulation mazes...")
    print("-" * 40)

    mazes = create_all_mazes()

    print_maze_summary(mazes)

    print("\nGenerating visualizations...")
    visualize_all_2d(mazes)
    visualize_track_graphs(mazes)

    print("\nDone! Showing plots...")
    plt.show()


if __name__ == "__main__":
    main()
