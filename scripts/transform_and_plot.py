"""
Reproduce Fig. 4 of the paper: F1/F2 transformation into 3D multinodal
graphs for all five datasets. Also prints the per-dataset transformation
diagnostics from the original notebook (F1 values, selected points,
F2 digits, and per-node connection counts).

Run from the repository root:
    python scripts/02_transform_and_plot.py
"""

import numpy as np
import pandas as pd

from proteinoid_spikes import config
from proteinoid_spikes.data_loading import load_all_datasets
from proteinoid_spikes.plotting import plot_multinodal_graph
from proteinoid_spikes.transformations import (
    F1, F2, find_point_less_than, find_points_with_same_digit,
)


def print_transformation_diagnostics(dataframes) -> None:
    """Replicates the verbose per-dataset printout from the notebook."""
    for i, df in enumerate(dataframes, 1):
        print(f"\nDataset {i} Transformation Analysis:")
        t_values = np.arange(config.F1_T_START, config.F1_T_STOP)
        x_t_values = F1(t_values)
        selected_points = [find_point_less_than(df, x) for x in x_t_values]

        print("F1 values (x(t)):")
        print(", ".join(f"{x:.2f}" for x in x_t_values))

        print("\nSelected points from dataset:")
        print(", ".join(
            f"{x:.6f}" if pd.notna(x) else "N/A" for x in selected_points
        ))

        print("\nFirst significant digits after decimal point (d):")
        d_values = [F2(x) if pd.notna(x) else "N/A" for x in selected_points]
        print(", ".join(
            str(int(d)) if pd.notna(d) and d != "N/A" else "N/A" for d in d_values
        ))

        print("\nNumber of connections for each selected point:")
        for point, d in zip(selected_points, d_values):
            if pd.notna(point) and d != "N/A":
                connected_points = find_points_with_same_digit(df, point, d)
                print(f"Point {point:.6f} (d={int(d)}): {len(connected_points)} connections")


def main() -> None:
    dataframes = load_all_datasets()

    save_path = config.FIGURES_DIR / "fig4_multinodal_graphs.png"
    plot_multinodal_graph(dataframes, save_path=save_path)
    print(f"Saved: {save_path}")

    print_transformation_diagnostics(dataframes)


if __name__ == "__main__":
    main()