"""Metrics for EHR classification and experiment reporting."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn


def classification_metrics(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """Accuracy and one-vs-rest AUC for binary logits."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device)).cpu().numpy()
    y_pred = np.argmax(logits, axis=1)
    acc = float(accuracy_score(y, y_pred))
    if logits.shape[1] >= 2:
        try:
            if len(np.unique(y)) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y, logits[:, 1]))
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")
    return {"accuracy": acc, "auc": auc}


def build_results_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize a list of run summaries into a table-friendly list of dicts.

    Used by tests and optional CSV export; does not fabricate numeric results.
    """
    table = []
    for row in rows:
        table.append(dict(row))
    return table


def summarize_history(history: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Aggregate simple stats from federated round history."""
    if not history:
        return {}
    last = history[-1]
    keys = [k for k in last if k.startswith("val_") and k.endswith("accuracy")]
    out = {"final_round": float(last.get("round", len(history)))}
    for k in keys:
        out[k.replace("val_", "final_")] = float(last[k])
    return out
