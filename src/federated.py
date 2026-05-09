"""Federated averaging simulation with optional local DP-SGD-style updates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import PartitionMode, build_client_partitions, split_train_val_test
from src.data import load_ehr_data
from src.evaluation import classification_metrics
from src.model import EHRMLP
from src.privacy import (
    apply_gradients,
    compute_per_example_gradients,
    dp_aggregate_clipped_noisy_gradients,
    federated_epsilon_report,
)
from src.attacks import loss_based_membership_probe


@dataclass
class FederatedConfig:
    seed: int = 42
    data_path: str = "data/patient_treatment.csv"
    test_size: float = 0.2
    val_size: float = 0.1
    partition: PartitionMode = "iid"
    dirichlet_alpha: float = 0.5
    label_skew_major_fraction: float = 0.85
    num_clients: int = 5
    num_global_rounds: int = 2
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.05
    global_lr: float = 1.0
    dp_enabled: bool = True
    l2_clip: float = 1.0
    noise_multiplier: float = 0.8
    hidden1: int = 128
    hidden2: int = 64
    dropout1: float = 0.3
    dropout2: float = 0.2
    device: str = "cpu"
    run_mia_probe: bool = True
    mia_n_samples: int = 256


@dataclass
class FederatedResult:
    history: list[dict[str, Any]]
    final_metrics: dict[str, Any]
    privacy: dict[str, Any]
    partition_info: dict[str, Any]
    mia: dict[str, Any] | None


def _set_seed(seed: int) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator(device="cpu").manual_seed(seed)


def _batch_tensors(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=gen,
    )


def local_train_one_epoch(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float,
    batch_size: int,
    dp_enabled: bool,
    l2_clip: float,
    noise_multiplier: float,
    device: torch.device,
    generator: torch.Generator,
    epoch_seed: int,
) -> float:
    """One pass over client data; returns mean training accuracy."""
    model.train()
    loader = _batch_tensors(X, y, batch_size, shuffle=True, seed=epoch_seed)
    criterion = nn.CrossEntropyLoss(reduction="none")
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if not dp_enabled:
            model.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb).mean()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-lr)
        else:
            model.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_vec = criterion(logits, yb)
            per_ex = compute_per_example_gradients(model, loss_vec)
            noisy_avg = dp_aggregate_clipped_noisy_gradients(
                per_ex,
                l2_clip=l2_clip,
                noise_multiplier=noise_multiplier,
                batch_size=int(xb.shape[0]),
                generator=generator,
            )
            apply_gradients(model, noisy_avg, lr)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
    return correct / max(1, total)


def fedavg_aggregate(
    client_states: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Weighted average of state dicts (weights sum to 1)."""
    out: dict[str, torch.Tensor] = {}
    keys = client_states[0].keys()
    for k in keys:
        acc = None
        for w, s in zip(weights, client_states):
            t = s[k]
            acc = w * t.clone() if acc is None else acc + w * t
        out[k] = acc  # type: ignore[assignment]
    return out


def run_federated_simulation(cfg: FederatedConfig) -> FederatedResult:
    device = torch.device(cfg.device)
    gen = _set_seed(cfg.seed)

    X, y = load_ehr_data(cfg.data_path)
    splits = split_train_val_test(
        X, y, test_size=cfg.test_size, val_size=cfg.val_size, seed=cfg.seed
    )
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    clients = build_client_partitions(
        X_train,
        y_train,
        num_clients=cfg.num_clients,
        mode=cfg.partition,
        seed=cfg.seed,
        dirichlet_alpha=cfg.dirichlet_alpha,
        label_skew_major_fraction=cfg.label_skew_major_fraction,
    )
    client_ns = [len(c[1]) for c in clients]

    num_classes = int(np.max(y_train)) + 1
    global_model = EHRMLP(
        X_train.shape[1],
        num_classes,
        hidden1=cfg.hidden1,
        hidden2=cfg.hidden2,
        dropout1=cfg.dropout1,
        dropout2=cfg.dropout2,
    ).to(device)

    history: list[dict[str, Any]] = []
    total_weight = float(sum(client_ns))

    for rnd in range(cfg.num_global_rounds):
        round_seed = cfg.seed + 1000 * rnd
        client_states = []
        train_accs = []
        for cid, (Xc, yc) in enumerate(clients):
            local = deepcopy(global_model).to(device)
            local.load_state_dict(global_model.state_dict())
            for e in range(cfg.local_epochs):
                acc = local_train_one_epoch(
                    local,
                    Xc,
                    yc,
                    lr=cfg.learning_rate,
                    batch_size=min(cfg.batch_size, max(1, len(yc))),
                    dp_enabled=cfg.dp_enabled,
                    l2_clip=cfg.l2_clip,
                    noise_multiplier=cfg.noise_multiplier,
                    device=device,
                    generator=gen,
                    epoch_seed=round_seed + cid * 10_000 + e,
                )
                train_accs.append(acc)
            client_states.append({k: v.detach().cpu() for k, v in local.state_dict().items()})

        weights = [n / total_weight for n in client_ns]
        new_state = fedavg_aggregate(client_states, weights)
        sd = global_model.state_dict()
        with torch.no_grad():
            for k in new_state:
                tgt = sd[k]
                src = new_state[k].to(device=device, dtype=tgt.dtype)
                if cfg.global_lr == 1.0:
                    tgt.copy_(src)
                else:
                    tgt.mul_(1.0 - cfg.global_lr).add_(src, alpha=cfg.global_lr)

        val_metrics = classification_metrics(global_model, X_val, y_val, device)
        history.append(
            {
                "round": rnd + 1,
                "mean_client_train_acc": float(np.mean(train_accs)) if train_accs else 0.0,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

    test_metrics = classification_metrics(global_model, X_test, y_test, device)
    privacy: dict[str, Any] = {
        "dp_enabled": cfg.dp_enabled,
        "l2_clip": cfg.l2_clip,
        "noise_multiplier": cfg.noise_multiplier,
        "accounting": None,
    }
    if cfg.dp_enabled:
        privacy["accounting"] = federated_epsilon_report(
            client_ns=client_ns,
            batch_size=cfg.batch_size,
            noise_multiplier=cfg.noise_multiplier,
            local_epochs_per_round=cfg.local_epochs,
            num_global_rounds=cfg.num_global_rounds,
        )

    mia_out = None
    if cfg.run_mia_probe:
        mia_out = loss_based_membership_probe(
            global_model,
            X_train,
            y_train,
            X_test,
            y_test,
            device=device,
            n_samples=cfg.mia_n_samples,
            seed=cfg.seed,
        )

    partition_info = {
        "mode": cfg.partition,
        "num_clients": cfg.num_clients,
        "client_ns": client_ns,
    }

    final_metrics = {
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "val_accuracy": history[-1].get("val_accuracy", float("nan")) if history else float("nan"),
    }

    return FederatedResult(
        history=history,
        final_metrics=final_metrics,
        privacy=privacy,
        partition_info=partition_info,
        mia=mia_out,
    )


def config_from_dict(d: dict[str, Any]) -> FederatedConfig:
    """Build FederatedConfig from a nested dict (e.g. loaded YAML)."""
    flat = FederatedConfig()
    if "seed" in d:
        flat.seed = int(d["seed"])
    data = d.get("data", {})
    if "path" in data:
        flat.data_path = str(data["path"])
    if "test_size" in data:
        flat.test_size = float(data["test_size"])
    if "val_size" in data:
        flat.val_size = float(data["val_size"])
    if "partition" in data:
        flat.partition = str(data["partition"])  # type: ignore[assignment]
    if "dirichlet_alpha" in data:
        flat.dirichlet_alpha = float(data["dirichlet_alpha"])
    if "label_skew_major_fraction" in data:
        flat.label_skew_major_fraction = float(data["label_skew_major_fraction"])

    fed = d.get("federated", {})
    if "num_clients" in fed:
        flat.num_clients = int(fed["num_clients"])
    if "num_global_rounds" in fed:
        flat.num_global_rounds = int(fed["num_global_rounds"])
    if "local_epochs" in fed:
        flat.local_epochs = int(fed["local_epochs"])
    if "batch_size" in fed:
        flat.batch_size = int(fed["batch_size"])
    if "learning_rate" in fed:
        flat.learning_rate = float(fed["learning_rate"])
    if "global_lr" in fed:
        flat.global_lr = float(fed["global_lr"])

    dp = d.get("dp", {})
    if "enabled" in dp:
        flat.dp_enabled = bool(dp["enabled"])
    if "l2_clip" in dp:
        flat.l2_clip = float(dp["l2_clip"])
    if "noise_multiplier" in dp:
        flat.noise_multiplier = float(dp["noise_multiplier"])

    model = d.get("model", {})
    if "hidden1" in model:
        flat.hidden1 = int(model["hidden1"])
    if "hidden2" in model:
        flat.hidden2 = int(model["hidden2"])
    if "dropout1" in model:
        flat.dropout1 = float(model["dropout1"])
    if "dropout2" in model:
        flat.dropout2 = float(model["dropout2"])

    if "device" in d:
        flat.device = str(d["device"])

    eval_ = d.get("evaluation", {})
    if "run_mia_probe" in eval_:
        flat.run_mia_probe = bool(eval_["run_mia_probe"])
    if "mia_n_samples" in eval_:
        flat.mia_n_samples = int(eval_["mia_n_samples"])

    return flat
