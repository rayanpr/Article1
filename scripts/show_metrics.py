"""Affiche resumee des metriques experimentales (resultats fusionnes)."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/experiment_results.json")
r = json.loads(path.read_text(encoding="utf-8"))
print(f"--- {path} ---")
for k, v in r.get("metrics", {}).items():
    op = v.get("operational")
    if not op:
        continue
    crps = v.get("net", {}).get("crps", float("nan"))
    cov = v.get("net", {}).get("coverage_80", float("nan"))
    print(f"{k:22s} crps_net={crps:.3f} cov80={cov*100:5.1f}% "
          f"regret={op['regret_pct']:6.3f}%  cost_true={op['cost_true_price']:.2f} "
          f"oracle={op['cost_oracle']:.2f}")

exact = json.loads(Path("results/spo_exact_results.json").read_text(encoding="utf-8"))
print(f"\n--- phase 3 : lambda retenu = {exact['chosen_lambda']} "
      f"(val {exact['best_val_regret_pct']:.3f} %, test {exact['test']['operational']['regret_pct']:.3f} %) ---")
print(f"{'lambda':>7} {'regret_val':>11} {'regret_test':>12}")
for row in exact["grid"]:
    mark = " <-- retenu" if abs(row["lambda"] - exact["chosen_lambda"]) < 1e-12 else ""
    print(f"{row['lambda']:>7g} {row['best_val_regret_pct']:>10.3f}% {row['test_regret_pct']:>11.3f}%{mark}")
