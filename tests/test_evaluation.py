from src.evaluation import build_results_table, classification_metrics, summarize_history
from src.model import EHRMLP

import numpy as np
import torch


def test_classification_metrics_keys():
    X = np.random.randn(40, 6).astype(np.float32)
    y = np.array([0, 1] * 20)
    m = EHRMLP(6, 2, hidden1=8, hidden2=4, dropout1=0.0, dropout2=0.0)
    m.eval()
    out = classification_metrics(m, X, y, torch.device("cpu"))
    assert "accuracy" in out and "auc" in out


def test_build_results_table_roundtrip():
    rows = [{"name": "a", "accuracy": 0.9}, {"name": "b", "accuracy": 0.8}]
    t = build_results_table(rows)
    assert t == rows


def test_summarize_history_reads_last_round():
    hist = [{"round": 1, "val_accuracy": 0.5}, {"round": 2, "val_accuracy": 0.7}]
    s = summarize_history(hist)
    assert s["final_accuracy"] == 0.7
