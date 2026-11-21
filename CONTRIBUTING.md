# Contributing to EMS Project

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
