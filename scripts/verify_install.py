#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys

CHECKS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("requests", "requests"),
    ("tqdm", "tqdm"),
    ("pandapower", "pandapower"),
    ("cvxpy", "cvxpy"),
    ("cvxpylayers", "cvxpylayers"),
    ("pyomo", "pyomo"),
    ("highspy", "highspy"),
    ("torch", "torch"),
    ("pytorch_forecasting", "pytorch-forecasting"),
    ("pytorch_lightning", "pytorch-lightning"),
    ("pyepo", "pyepo"),
    ("citylearn", "citylearn"),
    ("entsoe", "entsoe-py"),
    ("sklearn", "scikit-learn"),
]


def main() -> int:
    print(f"Python: {sys.version}")
    failed = []
    for module, package in CHECKS:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "OK")
            print(f"  [OK]   {package:<22} {version}")
        except ImportError:
            print(f"  [MISS] {package:<22} non installe")
            failed.append(package)

    try:
        import cvxpy as cp

        print(f"  [OK]   cvxpy solvers: {cp.installed_solvers()}")
    except Exception:
        pass

    try:
        import torch

        print(f"  [OK]   torch CUDA available: {torch.cuda.is_available()}")
    except Exception:
        pass

    if failed:
        print(f"\nManquants ({len(failed)}): {', '.join(failed)}")
        return 1
    print("\nToutes les librairies sont installees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
