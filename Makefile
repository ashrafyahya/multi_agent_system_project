.PHONY: help install install-dev lint format type-check test test-cov clean pre-commit-install pre-commit-run

# Default target
help:
	@echo "Available commands:"
	@echo "  make install          - Install production dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make lint             - Run all linters (ruff, bandit)"
	@echo "  make format           - Format code with black and ruff"
	@echo "  make type-check       - Run mypy type checking"
	@echo "  make test             - Run tests"
	@echo "  make test-cov         - Run tests with coverage report"
	@echo "  make clean            - Clean build artifacts and caches"
	@echo "  make pre-commit-install - Install pre-commit hooks"
	@echo "  make pre-commit-run  - Run pre-commit hooks on all files"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pre-commit
	pre-commit install

# Linting
lint:
	@echo "Running ruff linter..."
	ruff check src tests
	@echo "Running bandit security scanner..."
	bandit -r src -c pyproject.toml

# Formatting
format:
	@echo "Formatting with black..."
	black src tests
	@echo "Formatting with ruff..."
	ruff format src tests

# Type checking
type-check:
	@echo "Running mypy type checker..."
	mypy src

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Cleaning
clean:
	@echo "Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	@echo "Cleaning test artifacts..."
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf dist
	rm -rf build
	@echo "Clean complete!"

# Pre-commit hooks
pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

