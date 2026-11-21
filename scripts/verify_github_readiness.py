#!/usr/bin/env python3
"""
Script de vérification de la préparation du projet pour GitHub.
Vérifie que tous les fichiers nécessaires sont présents et bien configurés.
"""

import os
import json
import hashlib
from pathlib import Path

def check_file_exists(path, description):
    """Vérifie si un fichier existe."""
    if os.path.exists(path):
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} - MANQUANT")
        return False

def check_file_not_empty(path, description):
    """Vérifie si un fichier n'est pas vide."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"✅ {description}: {path} (non vide)")
        return True
    else:
        print(f"❌ {description}: {path} - VIDE ou MANQUANT")
        return False

def check_json_valid(path, description):
    """Vérifie si un fichier JSON est valide."""
    try:
        with open(path, 'r') as f:
            json.load(f)
        print(f"✅ {description}: {path} (JSON valide)")
        return True
    except Exception as e:
        print(f"❌ {description}: {path} - JSON invalide: {e}")
        return False

def check_python_syntax(path, description):
    """Vérifie la syntaxe d'un fichier Python."""
    try:
        with open(path, 'r') as f:
            compile(f.read(), path, 'exec')
        print(f"✅ {description}: {path} (syntaxe valide)")
        return True
    except SyntaxError as e:
        print(f"❌ {description}: {path} - Erreur de syntaxe: {e}")
        return False

def main():
    """Fonction principale de vérification."""
    print("=== Vérification de la préparation pour GitHub ===\n")
    
    # Configuration
    project_root = Path(__file__).parent.parent
    required_files = [
        # Fichiers principaux
        (project_root / "README.md", "README principal"),
        (project_root / "README_AMELIORATIONS.md", "README des améliorations"),
        (project_root / "outils_mathematiques_article1_ems.md", "Outils mathématiques"),
        (project_root / "requirements.txt", "Requirements principales"),
        (project_root / "requirements_ameliorations.txt", "Requirements améliorations"),
        
        # Structure de répertoires
        (project_root / "src", "Source code"),
        (project_root / "src/optimization", "Optimisation"),
        (project_root / "src/forecasting", "Prévision"),
        (project_root / "src/data", "Données"),
        (project_root / "scripts", "Scripts"),
        (project_root / "tests", "Tests"),
        (project_root / "results", "Résultats"),
        (project_root / "figures", "Figures"),
        (project_root / "docs", "Documentation"),
        (project_root / "config", "Configuration"),
        (project_root / "data", "Données brutes"),
        
        # Fichiers de configuration
        (project_root / "config/paths.py", "Configuration des chemins"),
        (project_root / ".gitignore", "Git ignore"),
        
        # Scripts principaux
        (project_root / "scripts/download_all.py", "Script de téléchargement"),
        (project_root / "scripts/build_dataset.py", "Script de construction dataset"),
        (project_root / "scripts/verify_install.py", "Script de vérification installation"),
        (project_root / "scripts/check_data.py", "Script de vérification données"),
        (project_root / "scripts/step3_preprocess.py", "Script de prétraitement"),
        (project_root / "scripts/train_evaluate.py", "Script d'entraînement"),
        (project_root / "scripts/train_spo_exact.py", "Script SPO+ exact"),
        (project_root / "scripts/step7_report.py", "Script de génération rapport"),
        
        # Scripts des améliorations
        (project_root / "scripts/robustness_budget_sweep.py", "Analyse de robustesse"),
        (project_root / "scripts/build_professional_figures.py", "Génération figures"),
        (project_root / "scripts/benchmark_decision_latency.py", "Benchmark latence"),
        (project_root / "scripts/run_benchmark.py", "Benchmark complet"),
        (project_root / "scripts/run_visualization.py", "Génération visualisations"),
        (project_root / "scripts/run_mlflow_experiment.py", "Exécution MLflow"),
        
        # Fichiers source des améliorations
        (project_root / "src/optimization/spo_optimized.py", "SPO+ optimisé"),
        (project_root / "src/forecasting/robust_models.py", "Modèles robustes"),
        (project_root / "src/optimization/robustness_analysis.py", "Analyse robustesse"),
        (project_root / "src/optimization/visualization.py", "Outils visualisation"),
        (project_root / "src/optimization/benchmark.py", "Benchmarking"),
        (project_root / "src/optimization/mlflow_integration.py", "Intégration MLflow"),
        
        # Tests
        (project_root / "tests/test_spo_optimization.py", "Tests SPO+"),
        (project_root / "tests/test_robust_models.py", "Tests modèles robustes"),
        (project_root / "tests/test_robustness_analysis.py", "Tests analyse robustesse"),
        (project_root / "tests/test_visualization.py", "Tests visualisation"),
        (project_root / "tests/test_benchmark.py", "Tests benchmarking"),
        (project_root / "tests/test_mlflow_integration.py", "Tests MLflow"),
        (project_root / "tests/test_advanced_features.py", "Tests fonctionnalités avancées"),
        (project_root / "tests/test_smoke.py", "Tests de base"),
        
        # Résultats
        (project_root / "results/experiment_results.json", "Résultats expériences"),
        (project_root / "results/spo_exact_results.json", "Résultats SPO+"),
        (project_root / "results/robustness_budget_sweep.json", "Analyse robustesse"),
        (project_root / "results/decision_latency_benchmark.json", "Benchmark latence"),
        
        # Documentation
        (project_root / "docs/formulations_mathematiques.md", "Formulations mathématiques"),
    ]
    
    # Fichiers optionnels (non critiques)
    optional_files = [
        (project_root / "models", "Modèles entraînés"),
        (project_root / "figures", "Figures générées"),
        (project_root / "data/processed", "Données traitées"),
    ]
    
    # Vérification des fichiers requis
    print("=== Vérification des fichiers requis ===")
    required_count = 0
    total_required = len(required_files)
    
    for file_path, description in required_files:
        if file_path.is_dir():
            if check_file_exists(file_path, description):
                required_count += 1
        else:
            if check_file_not_empty(file_path, description):
                required_count += 1
    
    print(f"\nFichiers requis: {required_count}/{total_required}")
    
    # Vérification des fichiers optionnels
    print("\n=== Vérification des fichiers optionnels ===")
    optional_count = 0
    for file_path, description in optional_files:
        if file_path.exists():
            print(f"✅ {description}: {file_path}")
            optional_count += 1
        else:
            print(f"⚠️  {description}: {file_path} (optionnel)")
    
    # Vérification des fichiers JSON
    print("\n=== Vérification des fichiers JSON ===")
    json_files = [
        (project_root / "results/experiment_results.json", "Résultats expériences"),
        (project_root / "results/spo_exact_results.json", "Résultats SPO+"),
        (project_root / "results/robustness_budget_sweep.json", "Analyse robustesse"),
        (project_root / "results/decision_latency_benchmark.json", "Benchmark latence"),
    ]
    
    json_count = 0
    for file_path, description in json_files:
        if check_json_valid(file_path, description):
            json_count += 1
    
    print(f"Fichiers JSON valides: {json_count}/{len(json_files)}")
    
    # Vérification des fichiers Python
    print("\n=== Vérification des fichiers Python ===")
    python_files = [
        (project_root / "scripts/verify_github_readiness.py", "Script de vérification"),
        (project_root / "src/optimization/spo_optimized.py", "SPO+ optimisé"),
        (project_root / "src/forecasting/robust_models.py", "Modèles robustes"),
        (project_root / "src/optimization/robustness_analysis.py", "Analyse robustesse"),
        (project_root / "src/optimization/visualization.py", "Outils visualisation"),
        (project_root / "src/optimization/benchmark.py", "Benchmarking"),
        (project_root / "src/optimization/mlflow_integration.py", "Intégration MLflow"),
    ]
    
    python_count = 0
    for file_path, description in python_files:
        if check_python_syntax(file_path, description):
            python_count += 1
    
    print(f"Fichiers Python valides: {python_count}/{len(python_files)}")
    
    # Génération du rapport
    print("\n=== Rapport de préparation ===")
    print(f"Fichiers requis: {required_count}/{total_required}")
    print(f"Fichiers optionnels: {optional_count}/{len(optional_files)}")
    print(f"Fichiers JSON valides: {json_count}/{len(json_files)}")
    print(f"Fichiers Python valides: {python_count}/{len(python_files)}")
    
    if required_count == total_required:
        print("\n🎉 Le projet est prêt pour GitHub!")
        print("Actions recommandées:")
        print("1. Créer un dépôt GitHub")
        print("2. Ajouter un fichier .gitignore approprié")
        print("3. Écrire un fichier README.md détaillé")
        print("4. Ajouter une licence (ex: MIT)")
        print("5. Pousser le code sur GitHub")
    else:
        print(f"\n⚠️  {total_required - required_count} fichiers requis sont manquants.")
        print("Veuillez corriger les problèmes avant de pousser sur GitHub.")

if __name__ == "__main__":
    main()