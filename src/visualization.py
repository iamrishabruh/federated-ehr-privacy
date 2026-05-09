"""Plotting helpers for benchmark outputs (uses measured metrics only)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_vs_epsilon(
    epsilons: Sequence[float | None],
    accuracies: Sequence[float],
    out_path: str | Path,
    title: str = "Validation accuracy vs. reported DP-SGD ε (approximate)",
) -> None:
    """Scatter plot; skips points where epsilon is None."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = []
    ys = []
    for e, a in zip(epsilons, accuracies):
        if e is not None and not np.isnan(a):
            xs.append(float(e))
            ys.append(float(a))
    plt.figure(figsize=(8, 5))
    if xs:
        plt.scatter(xs, ys, s=80, c="#1f77b4", label="runs")
        for x, y in zip(xs, ys):
            plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("ε (TensorFlow Privacy DP-SGD accountant, approximate)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_federated_rounds(history: Sequence[dict], out_path: str | Path, metric_key: str = "val_accuracy") -> None:
    """Line plot of a metric per federated round."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rounds = [h["round"] for h in history if metric_key in h]
    vals = [float(h[metric_key]) for h in history if metric_key in h]
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, vals, marker="o")
    plt.xlabel("Global round")
    plt.ylabel(metric_key)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc_binary(fpr: np.ndarray, tpr: np.ndarray, auc_score: float, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC (hold-out test)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
