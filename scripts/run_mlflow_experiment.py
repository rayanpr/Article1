#!/usr/bin/env python3
"""
Script d'exécution d'expériences avec suivi MLflow.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ajouter le répertoire src au chemin Python
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.optimization.mlflow_integration import run_mlflow_experiment


def main():
    """Fonction principale d'exécution."""
    parser = argparse.ArgumentParser(description="Exécution d'expériences avec suivi MLflow")
    parser.add_argument("--output-dir", type=str, default="results", help="Répertoire de sortie")
    parser.add_argument("--experiment-name", type=str, default="EMS_Experiments", help="Nom de l'expérience MLflow")
    parser.add_argument("--models", type=str, nargs="+", default=["TFT", "XGBoost", "LightGBM"], 
                       help="Modèles à tester")
    parser.add_argument("--horizon", type=int, default=24, help="Horizon de prévision")
    parser.add_argument("--n-epochs", type=int, default=20, help="Nombre d'époques")
    parser.add_argument("--n-runs", type=int, default=5, help="Nombre d'exécutions")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Taux d'apprentissage")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille du batch")
    
    args = parser.parse_args()
    
    # Configuration de base
    base_config = {
        "horizon": args.horizon,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "models": args.models,
        "n_runs": args.n_runs,
    }
    
    # Définir les fonctions d'entraînement et d'évaluation
    def train_func(config):
        """Fonction d'entraînement simulée."""
        # Ici, vous implémenteriez votre vraie fonction d'entraînement
        print(f"Entraînement avec config: {config}")
        
        # Simuler l'entraînement
        train_loss = 1.0 / (config.get("epoch", 1) + 1)
        val_loss = train_loss * 1.1
        
        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": config.get("epoch", 1),
        }
    
    def evaluate_func(model, config):
        """Fonction d'évaluation simulée."""
        # Ici, vous implémenteriez votre vraie fonction d'évaluation
        print(f"Évaluation du modèle: {model} avec config: {config}")
        
        # Simuler l'évaluation
        metrics = {
            "crps": 0.15 + (hash(model) % 100) / 1000,
            "regret": 0.03 + (hash(model) % 100) / 1000,
            "coverage": 0.8 + (hash(model) % 20) / 100,
        }
        
        return metrics
    
    # Exécuter l'expérience MLflow
    results = run_mlflow_experiment(
        config=base_config,
        train_func=train_func,
        evaluate_func=evaluate_func,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        run_name_suffix="_experiment"
    )
    
    # Sauvegarder les résultats
    output_path = Path(args.output_dir) / "mlflow_experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Résultats sauvegardés dans: {output_path}")
    print("Pour visualiser les résultats, exécutez: mlflow ui")


if __name__ == "__main__":
    main()