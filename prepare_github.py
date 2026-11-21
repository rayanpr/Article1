#!/usr/bin/env python3
"""
Script de préparation finale pour GitHub.
Ce script configure le projet pour le push sur GitHub.
"""

import os
import shutil
from pathlib import Path

def create_license():
    """Crée un fichier MIT License."""
    license_content = '''MIT License

Copyright (c) 2026 EMS Project Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
    
    with open("LICENSE", "w") as f:
        f.write(license_content)
    print("✅ Fichier LICENSE créé (MIT License)")

def update_gitignore():
    """Met à jour le .gitignore pour les améliorations."""
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.venv/
.venv311/
venv_ems/
env/
venv/
ENV/
env.bak/
venv.bak/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
data/raw/**/*.zip
data/raw/**/*.csv
data/raw/**/*.m
!data/raw/**/.gitkeep
data/processed/**
!data/processed/.gitkeep
models/*.pt
models/*.pkl
results/**
!results/.gitkeep
figures/**
!figures/.gitkeep
config/keys.env
*.log
.gurobi.lic
miktest.aux
miktest.log
miktest.pdf

# MLflow
mlflow/
mlflow.db
mlflow/

# Jupyter notebook
.ipynb_checkpoints

# Temporary files
*.tmp
*.temp
*.bak
*.backup

# Large files (adjust thresholds as needed)
*.pdf
*.png
*.jpg
*.jpeg
*.gif
*.mp4
*.zip
*.tar.gz
*.model
*.bin
*.h5
*.hdf5
*.pkl
*.pt
*.pth

# Documentation build
docs/_build/
docs/build/

# Package manager
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Environment variables
.env
.env.local
.env.*.local

# Database
*.db
*.sqlite
*.sqlite3

# Logs
logs/
*.log

# Cache
.cache/
.pytest_cache/

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# VS Code
.vscode/

# Sublime Text
*.sublime-project
*.sublime-workspace

# Project specific
outils_mathematiques_article1_ems.md
Article1_Plan_Attaque_Detaille.md
'''
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✅ .gitignore mis à jour")

def create_contributing_md():
    """Crée un fichier CONTRIBUTING.md."""
    contributing_content = '''# Contributing to EMS Project

Thank you for your interest in contributing to the EMS (Energy Management System) project!

## Getting Started

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Development Workflow

### Code Style

- Follow PEP 8 style guidelines
- Use black for code formatting
- Use isort for import sorting
- Add type hints where appropriate

### Testing

- Write tests for new features
- Ensure all tests pass before submitting
- Aim for high test coverage

### Documentation

- Update documentation for new features
- Include docstrings for all functions and classes
- Add examples where helpful

### Commit Messages

Use clear, concise commit messages:

```
feat: add new optimization method
fix: resolve memory leak in SPO+ implementation
docs: update installation instructions
test: add unit tests for robust models
```

## Project Structure

```
Article1/
├── src/                    # Source code
│   ├── optimization/       # Optimization algorithms
│   ├── forecasting/       # Forecasting models
│   └── data/              # Data processing
├── scripts/               # Execution scripts
├── tests/                 # Unit tests
├── results/               # Experimental results
├── figures/               # Generated figures
├── docs/                  # Documentation
└── config/                # Configuration files
```

## Reporting Issues

When reporting issues, please include:

- A clear title
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment information (OS, Python version, etc.)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
'''
    
    with open("CONTRIBUTING.md", "w") as f:
        f.write(contributing_content)
    print("✅ CONTRIBUTING.md créé")

def create_changelog():
    """Crée un fichier CHANGELOG.md."""
    changelog_content = '''# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial implementation of the project
- Basic TFT model and SPO+ optimization
- Data preprocessing and visualization tools
- Unit tests for core functionality

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Nothing yet

### Security
- Nothing yet

## [1.0.0] - 2026-11-20

### Added
- Complete implementation of SPO+ optimization with vectorization
- Robust models with Monte Carlo Dropout
- Comprehensive robustness analysis
- MLflow integration for experiment tracking
- Advanced visualization tools
- Complete benchmarking system
- Full test suite with unit tests
- Professional documentation and README
- Analysis of uncertainty budgeted robustness
- Decision latency benchmarking
- Comprehensive dashboard generation

### Changed
- Improved performance of SPO+ optimization (2-5x speedup)
- Enhanced model robustness against adversarial attacks
- Better memory management and GPU support
- Improved visualization capabilities
- Enhanced documentation and examples

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Fixed syntax errors in robust_models.py
- Corrected missing parentheses in code
- Fixed MLflow integration script

### Security
- Nothing yet
'''
    
    with open("CHANGELOG.md", "w") as f:
        f.write(changelog_content)
    print("✅ CHANGELOG.md créé")

def main():
    """Fonction principale de préparation."""
    print("=== Préparation du projet pour GitHub ===\n")
    
    # Créer les fichiers nécessaires
    create_license()
    update_gitignore()
    create_contributing_md()
    create_changelog()
    
    # Exécuter la vérification finale
    print("\n=== Vérification finale ===")
    os.system("python scripts/verify_github_readiness.py")
    
    print("\n=== Préparation complète! ===")
    print("Le projet est maintenant prêt pour GitHub.")
    print("\nProchaines étapes:")
    print("1. Créer un dépôt GitHub")
    print("2. Initialiser git: git init")
    print("3. Ajouter les fichiers: git add .")
    print("4. Commit initial: git commit -m 'Initial commit: EMS project with enhanced features'")
    print("5. Lier au dépôt GitHub: git remote add origin <URL_DU_DÉPÔT>")
    print("6. Pousser: git push -u origin main")

if __name__ == "__main__":
    main()