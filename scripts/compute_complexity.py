"""
Reproduce Table III and Fig. 5 of the paper: eight complexity metrics
plus meta-metric across the five datasets.

Run from the repository root:
    python scripts/03_compute_complexity.py

Outputs:
    results/tables/table3_complexity_metrics.csv
    results/figures/fig5_complexity_metrics.png
"""

import pandas as pd

from proteinoid_spikes import config
from proteinoid_spikes.complexity import (
    calculate_complexity_metrics, calculate_meta_metric,
)
from proteinoid_spikes.data_loading import load_all_datasets
from proteinoid_spikes.plotting import plot_complexity_metrics
from proteinoid_spikes.transformations import build_graph


def main() -> None:
    dataframes = load_all_datasets()

    all_metrics = []
    for i, df in enumerate(dataframes, 1):
        G = build_graph(df, with_positions=False)
        metrics = calculate_complexity_metrics(G)
        metrics["Dataset"] = f"Dataset {i}"
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)

    meta_metric = calculate_meta_metric(metrics_df.drop("Dataset", axis=1))
    metrics_df["meta_metric"] = meta_metric

    metrics_df = metrics_df.sort_values("meta_metric", ascending=False)

    # Save Table III as CSV
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_path = config.TABLES_DIR / "table3_complexity_metrics.csv"
    metrics_df.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")

    # Pretty-print the table to stdout, matching the original notebook
    display_df = metrics_df.copy()
    for col in display_df.columns:
        if col != "Dataset":
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, float) else x
            )
    display_df["Rank"] = range(1, len(display_df) + 1)
    display_df = display_df.set_index("Rank")
    cols = ["Dataset"] + [c for c in display_df.columns if c != "Dataset"]
    display_df = display_df[cols]
    print("\nComprehensive Complexity Metrics Table:")
    print(display_df.to_string())

    # Save Fig. 5
    fig_path = config.FIGURES_DIR / "fig5_complexity_metrics.png"
    plot_complexity_metrics(metrics_df, save_path=fig_path)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()