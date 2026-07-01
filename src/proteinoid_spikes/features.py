"""
Section V.B of the paper: 16-dimensional feature vector (Eq. 6).

Temporal:  time, time difference, ISI, CV of ISI
Statistical: rolling mean & std over windows of 3, 5, 10
Spectral:    sin & cos Fourier components for periods 5, 10, 20
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the 16-dim feature columns to a (Time, Spike) DataFrame.

    Returns the same DataFrame with extra columns:
        Time_Diff, ISI, CV_ISI,
        Rolling_Mean_3, Rolling_Std_3, Rolling_Mean_5, Rolling_Std_5,
        Rolling_Mean_10, Rolling_Std_10,
        Sin_5, Cos_5, Sin_10, Cos_10, Sin_20, Cos_20
    """
    df = df.sort_values("Time")
    df["Time_Diff"] = df["Time"].diff().fillna(0)

    # Inter-spike intervals
    df["ISI"] = df["Time_Diff"]

    # Coefficient of variation of ISIs
    window_size = config.CV_ISI_WINDOW_SIZE
    df["CV_ISI"] = (
        df["ISI"].rolling(window=window_size).std()
        / df["ISI"].rolling(window=window_size).mean()
    )

    # Rolling window features
    for size in config.ROLLING_WINDOW_SIZES:
        df[f"Rolling_Mean_{size}"] = df["Spike"].rolling(window=size).mean()
        df[f"Rolling_Std_{size}"] = df["Spike"].rolling(window=size).std()

    # Fourier features
    for period in config.FOURIER_PERIODS:
        df[f"Sin_{period}"] = np.sin(2 * np.pi * df["Time"] / period)
        df[f"Cos_{period}"] = np.cos(2 * np.pi * df["Time"] / period)

    return df.fillna(0)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of the 16 feature column names."""
    return ["Time", "Time_Diff", "ISI", "CV_ISI"] + [
        col for col in df.columns
        if col.startswith(("Rolling_", "Sin_", "Cos_"))
    ]