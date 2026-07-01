"""
Load and parse the raw VSD spike train CSV.

The raw file ``meta_data.csv`` stores five datasets as paired
(Time, Spike) columns under the headers Data1 ... Data5. Each pair has
its own length (recording durations differ), so the file has trailing NaNs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def process_dataset(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Extract a single (Time, Spike) dataset from the wide CSV.

    Parameters
    ----------
    df : pd.DataFrame
        The full wide DataFrame loaded from meta_data.csv.
    column_name : str
        One of 'Data1', ..., 'Data5'. The function reads this column and
        the column immediately to its right as (Time, Spike).

    Returns
    -------
    pd.DataFrame
        Columns: Time (float), Spike (float, 0/1), Dataset (str).
    """
    data = df[[column_name, df.columns[df.columns.get_loc(column_name) + 1]]].copy()
    data.columns = ["Time", "Spike"]
    data = data.iloc[1:]  # Remove the header row
    data["Time"] = pd.to_numeric(data["Time"], errors="coerce")
    data["Spike"] = pd.to_numeric(data["Spike"], errors="coerce")
    data = data.dropna()
    data["Dataset"] = column_name
    return data


def load_all_datasets(
    file_path: Path | str | None = None,
    columns: list[str] | None = None,
) -> list[pd.DataFrame]:
    """Load all five spike train datasets from the raw CSV."""
    file_path = Path(file_path) if file_path is not None else config.RAW_DATA_FILE
    columns = columns if columns is not None else config.DATASET_COLUMNS

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {file_path}. "
            "Place meta_data.csv at data/raw/meta_data.csv "
            "(see data/README.md for format)."
        )

    spike_data = pd.read_csv(file_path)

    dataframes: list[pd.DataFrame] = []
    for column in columns:
        if column in spike_data.columns:
            dataframes.append(process_dataset(spike_data, column))
        else:
            print(f"Column {column} not found in the dataset")

    return dataframes


def print_dataset_statistics(dataframes: list[pd.DataFrame]) -> None:
    """Print basic per-dataset stats (spike count, duration, mean ISI)."""
    for df in dataframes:
        print(f"\n{df['Dataset'].iloc[0]} Statistics:")
        print(f"Number of spikes: {df['Spike'].sum()}")
        print(f"Total duration: {df['Time'].max() - df['Time'].min():.2f} seconds")
        print(f"Average inter-spike interval: {df['Time'].diff().mean():.4f} seconds")