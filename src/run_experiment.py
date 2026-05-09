"""CLI entrypoint: load YAML config, run federated simulation, write metrics and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import math

import numpy as np
import yaml

from src.federated import config_from_dict, run_federated_simulation
from src.visualization import plot_federated_rounds


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.generic):
        v = obj.item()
        return _to_jsonable(v)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Federated EHR classification benchmark (simulation).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    raw = load_yaml_config(cfg_path)
    out = raw.get("output", {})
    plots_dir = Path(out.get("plots_dir", "plots"))
    results_path = Path(out.get("results_json", "results/last_run.json"))

    fed_cfg = config_from_dict(raw)
    result = run_federated_simulation(fed_cfg)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(
        {
            "config_path": str(cfg_path),
            "history": result.history,
            "final_metrics": result.final_metrics,
            "privacy": result.privacy,
            "partition_info": result.partition_info,
            "mia": result.mia,
        }
    )
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_federated_rounds(result.history, plots_dir / "federated_val_accuracy.png", "val_accuracy")

    print(json.dumps(payload["final_metrics"], indent=2))


if __name__ == "__main__":
    main()
