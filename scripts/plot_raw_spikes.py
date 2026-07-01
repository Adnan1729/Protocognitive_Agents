"""
Reproduce Fig. 3 of the paper: raw spike trains for all five datasets.

Run from the repository root:
    python scripts/01_plot_raw_spikes.py
"""

from proteinoid_spikes import config
from proteinoid_spikes.data_loading import load_all_datasets, print_dataset_statistics
from proteinoid_spikes.plotting import plot_spike_trains


def main() -> None:
    dataframes = load_all_datasets()

    save_path = config.FIGURES_DIR / "fig3_raw_spike_trains.png"
    plot_spike_trains(dataframes, save_path=save_path)
    print(f"Saved: {save_path}")

    print_dataset_statistics(dataframes)


if __name__ == "__main__":
    main()