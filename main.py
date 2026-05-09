"""
Legacy entrypoint. Prefer:

    python -m src.run_experiment --config configs/baseline.yaml
"""

from __future__ import annotations

import sys

from src.run_experiment import main as run_main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--config", "configs/baseline.yaml"])
    run_main()
