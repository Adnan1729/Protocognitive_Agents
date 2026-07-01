"""
All paper figures (Figs. 3, 4, 5, 7, 8, 9) as standalone functions.

Styling preserves the original notebook's choices: black ink on white,
red/black ROC and training curves, greyscale confusion matrix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .transformations import build_graph


def _save_or_show(fig, save_path: Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Fig. 3 -- raw spike trains
# ---------------------------------------------------------------------------

def plot_spike_trains(dataframes, save_path: Path | None = None) -> None:
    """Fig. 3: raw spike trains for all five datasets in one row."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 3), sharex=True, sharey=True)

    for i, (df, ax) in enumerate(zip(dataframes, axes)):
        ax.scatter(df["Time"], df["Spike"], color="black", marker="|", s=100)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_title(f"Dataset {i+1}", color="black", fontsize=12)
        ax.set_xlabel("Time (s)", color="black", fontsize=10)
        if i == 0:
            ax.set_ylabel("Spike", color="black", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("gray")
        ax.spines["bottom"].set_color("gray")
        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")
        ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Fig. 4 -- multinodal graphs
# ---------------------------------------------------------------------------

def plot_multinodal_graph(dataframes, save_path: Path | None = None) -> None:
    """Fig. 4: 3D multinodal graph transformation for all five datasets."""
    fig, axes = plt.subplots(
        1, 5, figsize=(25, 5), subplot_kw={"projection": "3d"}
    )

    for i, (df, ax) in enumerate(zip(dataframes, axes)):
        G = build_graph(df, with_positions=True)
        pos = nx.get_node_attributes(G, "pos")

        # Edges
        for edge in G.edges():
            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            z = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x, y, z, c="gray", alpha=0.5)

        # Nodes
        x = [pos[node][0] for node in G.nodes()]
        y = [pos[node][1] for node in G.nodes()]
        z = [pos[node][2] for node in G.nodes()]
        ax.scatter(x, y, z, c="red", s=20)

        ax.set_title(f"Dataset {i+1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("F1 Index")
        ax.set_zlabel("Layer")
        ax.set_xlim(0, df["Time"].max())
        ax.set_ylim(0, 20)
        ax.set_zlim(0, 1)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Fig. 5 -- complexity metrics bar charts
# ---------------------------------------------------------------------------

def plot_complexity_metrics(metrics_df: pd.DataFrame,
                            save_path: Path | None = None) -> None:
    """Fig. 5: bar charts of the 8 complexity metrics across datasets."""
    metrics_to_plot = config.METRICS_PLOT_ORDER
    fig, axes = plt.subplots(1, 8, figsize=(24, 4))

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        metrics_df.set_index("Dataset")[metric].plot(
            kind="bar", ax=ax, color="black", width=0.7
        )
        ax.set_title(metric, color="black", fontsize=10)
        ax.set_xlabel("Dataset", color="black", fontsize=8)
        ax.set_ylabel("Value", color="black", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors="black", labelsize=8)
        ax.grid(False)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha="right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Fig. 7 -- confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, save_path: Path | None = None) -> None:
    """Fig. 7: confusion matrix heatmap (greyscale)."""
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greys",
        cbar=True, linecolor="black", linewidths=0.5,
    )
    plt.title("Confusion Matrix", color="black")
    plt.xlabel("Predicted", color="black")
    plt.ylabel("Actual", color="black")
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Fig. 8 -- training & validation accuracy / loss
# ---------------------------------------------------------------------------

def plot_training_history(history, save_path: Path | None = None) -> None:
    """Fig. 8: training and validation accuracy/loss curves."""
    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], color="red", label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], color="black", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], color="red", label="Training Loss")
    plt.plot(history.history["val_loss"], color="black", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Fig. 9 -- ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(fpr, tpr, roc_auc, save_path: Path | None = None) -> None:
    """Fig. 9: ROC curve with AUC."""
    fig = plt.figure()
    plt.plot(fpr, tpr, color="red", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    _save_or_show(fig, save_path)