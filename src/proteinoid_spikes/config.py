"""
Central configuration.

All paths, seeds, and hyperparameters live here so a reader can audit the
exact constants used to produce the paper's results without grepping
through the codebase.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# config.py lives at <repo>/src/proteinoid_spikes/config.py, so three
# .parent calls take us back to the repo root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
RAW_DATA_FILE: Path = RAW_DATA_DIR / "meta_data.csv"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

# The raw CSV has paired (Time, Spike) columns under these headers.
DATASET_COLUMNS: list[str] = ["Data1", "Data2", "Data3", "Data4", "Data5"]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# The published paper was produced from an unseeded run, so re-executing
# this code will NOT reproduce 70.41% accuracy bit-exact. The seed below
# makes the pipeline deterministic going forward. See README -> "On exact
# reproduction of paper numbers".
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# F1 spiral sampling
# ---------------------------------------------------------------------------

# Paper (Eq. 1) writes t in {0, 2, 3, ..., 19, 20}.
# The original notebook uses np.arange(1, 21) == {1, 2, ..., 20}.
# We preserve the notebook's choice so numerical output matches the
# published figures and tables.
F1_T_START: int = 1
F1_T_STOP: int = 21   # exclusive upper bound for np.arange

# ---------------------------------------------------------------------------
# Complexity meta-metric weights (paper ref. [21])
# ---------------------------------------------------------------------------

META_METRIC_WEIGHTS: dict[str, float] = {
    "num_nodes":              0.05,
    "num_edges":              0.05,
    "avg_degree":             0.10,
    "clustering_coefficient": 0.10,
    "density":                0.20,
    "degree_entropy":         0.20,
    "num_components":         0.10,
    "avg_resistance":         0.20,
}

# Order in which the 8 metrics are plotted in Fig. 5.
METRICS_PLOT_ORDER: list[str] = [
    "num_nodes",
    "num_edges",
    "avg_degree",
    "clustering_coefficient",
    "density",
    "degree_entropy",
    "num_components",
    "avg_resistance",
]

# ---------------------------------------------------------------------------
# Feature engineering (Section V.B, Eq. 6)
# ---------------------------------------------------------------------------

ROLLING_WINDOW_SIZES: list[int] = [3, 5, 10]
CV_ISI_WINDOW_SIZE: int = 10
FOURIER_PERIODS: list[int] = [5, 10, 20]

# ---------------------------------------------------------------------------
# Classifier architecture & training (Section V.A)
# ---------------------------------------------------------------------------

HIDDEN_LAYER_SIZES: list[int] = [128, 64, 32, 16]
DROPOUT_RATE: float = 0.2

LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 16
EPOCHS: int = 100
VALIDATION_SPLIT: float = 0.2

EARLY_STOPPING_PATIENCE: int = 10
LR_REDUCE_FACTOR: float = 0.5
LR_REDUCE_PATIENCE: int = 5

CLASSIFICATION_THRESHOLD: float = 0.5

# Train on Datasets 1-4 concatenated, hold out Dataset 5 as the test set.
TRAIN_DATASETS: list[str] = ["Data1", "Data2", "Data3", "Data4"]
TEST_DATASETS: list[str] = ["Data5"]