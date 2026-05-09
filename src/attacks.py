"""Privacy leakage probes (not a full adversarial pipeline)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F


def loss_based_membership_probe(
    model: nn.Module,
    X_member: np.ndarray,
    y_member: np.ndarray,
    X_nonmember: np.ndarray,
    y_nonmember: np.ndarray,
    *,
    device: torch.device,
    n_samples: int = 512,
    seed: int = 0,
) -> dict[str, float | int]:
    """
    Simple membership inference **probe** using per-example cross-entropy loss.

    Lower loss is treated as evidence of membership (common heuristic). This is a coarse
    indicator, not an optimal attack; it exists to make privacy–utility tradeoffs discussable
    in the benchmark.

    Returns attack AUC and accuracy at a threshold chosen on the **combined** set (optimistic;
    documented in README as a limitation).
    """
    rng = np.random.default_rng(seed)
    n_m = min(n_samples // 2, len(y_member))
    n_nm = min(n_samples - n_m, len(y_nonmember))
    if n_m < 2 or n_nm < 2:
        return {
            "mia_auc": float("nan"),
            "mia_accuracy": float("nan"),
            "n_member": int(n_m),
            "n_nonmember": int(n_nm),
            "note": "Insufficient samples for MIA probe.",
        }

    idx_m = rng.choice(len(y_member), size=n_m, replace=False)
    idx_nm = rng.choice(len(y_nonmember), size=n_nm, replace=False)
    Xm, ym = X_member[idx_m], y_member[idx_m]
    Xnm, ynm = X_nonmember[idx_nm], y_nonmember[idx_nm]

    model.eval()
    with torch.no_grad():
        lm = _per_example_ce(model, Xm, ym, device)
        lnm = _per_example_ce(model, Xnm, ynm, device)

    # Score: negative loss (higher => predicted member)
    scores = np.concatenate([-lm, -lnm])
    labels = np.concatenate([np.ones(n_m), np.zeros(n_nm)])
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = float("nan")

    thr = float(np.median(scores))
    preds = (scores >= thr).astype(int)
    acc = float(np.mean(preds == labels))

    return {
        "mia_auc": auc,
        "mia_accuracy": acc,
        "n_member": int(n_m),
        "n_nonmember": int(n_nm),
        "threshold_note": "median_threshold_on_pooled_scores",
    }


def _per_example_ce(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> np.ndarray:
    xb = torch.from_numpy(X).to(device)
    yb = torch.from_numpy(y).to(device)
    logits = model(xb)
    loss = F.cross_entropy(logits, yb, reduction="none")
    return loss.detach().cpu().numpy()
