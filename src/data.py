"""EHR-style tabular data loading and client partitioning (IID and non-IID)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CONTINUOUS_COLUMNS = [
    "HAEMATOCRIT",
    "HAEMOGLOBINS",
    "ERYTHROCYTE",
    "LEUCOCYTE",
    "THROMBOCYTE",
    "MCH",
    "MCHC",
    "MCV",
    "AGE",
]

PartitionMode = Literal["iid", "dirichlet", "label_skew"]


def synthetic_fixture_path() -> Path:
    """Path to the small synthetic CSV used in tests and quick demos."""
    return Path(__file__).resolve().parent.parent / "data" / "synthetic_ehr_fixture.csv"


def load_ehr_data(
    file_path: str | Path,
    *,
    fit_scaler: StandardScaler | None = None,
    return_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Load EHR-like tabular data: impute, encode categoricals, scale continuous features.

    Expects columns including CONTINUOUS_COLUMNS, SEX (F/M), and SOURCE (in/out) as label.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df = df.ffill()

    if "SEX" in df.columns:
        df["SEX"] = df["SEX"].map({"F": 0, "M": 1})

    missing_cols = [c for c in CONTINUOUS_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required feature columns: {missing_cols}")

    scaler = fit_scaler if fit_scaler is not None else StandardScaler()
    if fit_scaler is None:
        df[CONTINUOUS_COLUMNS] = scaler.fit_transform(df[CONTINUOUS_COLUMNS])
    else:
        df[CONTINUOUS_COLUMNS] = scaler.transform(df[CONTINUOUS_COLUMNS])

    if "SOURCE" not in df.columns:
        raise ValueError("CSV must contain SOURCE column for labels (e.g. in/out).")
    df["SOURCE"] = df["SOURCE"].map({"in": 1, "out": 0})

    if df["SOURCE"].isna().any():
        raise ValueError("SOURCE column must be 'in' or 'out' for all rows.")

    X = df.drop("SOURCE", axis=1).values.astype(np.float32)
    y = df["SOURCE"].values.astype(np.int64)
    if return_scaler:
        return X, y, scaler
    return X, y


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    val_size: float,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Hold out test, then split remaining into train / validation."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=rel_val, random_state=seed, stratify=y_trainval
    )
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def partition_clients_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffle and split examples evenly across clients (last clients may differ by at most one)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    X_shuf, y_shuf = X[perm], y[perm]
    splits = np.array_split(np.arange(n), num_clients)
    return [(X_shuf[idx], y_shuf[idx]) for idx in splits]


def partition_clients_dirichlet(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Non-IID label imbalance via Dirichlet allocation per class (Hsu et al. style).

    Smaller ``alpha`` yields stronger heterogeneity across clients.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        if len(idx_c) == 0:
            continue
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        parts = np.split(idx_c, proportions)
        for i, part in enumerate(parts):
            client_indices[i].extend(part.tolist())

    for i in range(num_clients):
        if not client_indices[i]:
            # Extremely skewed draw can empty a client; steal one index from the largest client.
            donor = int(np.argmax([len(ci) for ci in client_indices]))
            if len(client_indices[donor]) < 2:
                continue
            move = client_indices[donor].pop()
            client_indices[i].append(move)

    partitions: list[tuple[np.ndarray, np.ndarray]] = []
    for indices in client_indices:
        idx = np.array(indices, dtype=int)
        rng.shuffle(idx)
        partitions.append((X[idx], y[idx]))
    return partitions


def partition_clients_label_skew(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    *,
    major_class: int = 0,
    major_fraction: float = 0.85,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Simple non-IID split: client 0 is dominated by ``major_class``; remaining clients split the rest evenly.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    n_per = max(1, n // num_clients)

    idx_major = np.where(y == major_class)[0]
    idx_rest = np.where(y != major_class)[0]
    rng.shuffle(idx_major)
    rng.shuffle(idx_rest)

    n_major_wanted = int(min(len(idx_major), max(1, round(major_fraction * n_per))))
    take_major = idx_major[:n_major_wanted]
    need_fill = n_per - len(take_major)
    take_rest = idx_rest[: max(0, need_fill)] if len(idx_rest) else np.array([], dtype=int)
    client0 = np.concatenate([take_major, take_rest])
    used = set(client0.tolist())

    pool = np.array([i for i in range(n) if i not in used], dtype=int)
    rng.shuffle(pool)

    partitions: list[tuple[np.ndarray, np.ndarray]] = []
    rng.shuffle(client0)
    partitions.append((X[client0], y[client0]))

    cursor = 0
    for _ in range(1, num_clients):
        take = pool[cursor : cursor + n_per]
        cursor += len(take)
        if len(take) == 0 and cursor < len(pool):
            take = pool[cursor : cursor + 1]
            cursor += 1
        if len(take) == 0:
            partitions.append((X[:0], y[:0]))
        else:
            partitions.append((X[take], y[take]))

    return partitions


def build_client_partitions(
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_clients: int,
    mode: PartitionMode,
    seed: int,
    dirichlet_alpha: float = 0.5,
    label_skew_major_fraction: float = 0.85,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if mode == "iid":
        return partition_clients_iid(X, y, num_clients, seed)
    if mode == "dirichlet":
        return partition_clients_dirichlet(X, y, num_clients, dirichlet_alpha, seed)
    if mode == "label_skew":
        return partition_clients_label_skew(
            X, y, num_clients, major_fraction=label_skew_major_fraction, seed=seed
        )
    raise ValueError(f"Unknown partition mode: {mode}")


def partitions_to_data_config(partitions: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """Summarize partition sizes and label rates for logging."""
    summary = []
    for i, (_, yi) in enumerate(partitions):
        if len(yi) == 0:
            summary.append({"client": i, "n": 0, "pos_rate": float("nan")})
        else:
            summary.append(
                {
                    "client": i,
                    "n": int(len(yi)),
                    "pos_rate": float(np.mean(yi)),
                }
            )
    return {"clients": summary}
