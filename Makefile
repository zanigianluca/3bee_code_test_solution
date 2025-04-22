# Check if uv is installed
UV := $(shell command -v uv 2> /dev/null)
ifeq ($(UV),)
$(error "uv command not found in PATH. Please install uv: https://github.com/astral-sh/uv")
endif

.ONESHELL:
ENV_NAME="pollinator_abundance" # You can keep this for descriptive messages

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

.PHONY: venv
venv: ## Create the .venv virtual environment using uv if it doesn't exist.
	@if [ ! -d .venv ]; then \
		echo "Creating virtual environment '.venv' using uv..."; \
		uv venv; \
		echo "Virtual environment created. Install dependencies with 'make env_requirements'"; \
	else \
		echo "Virtual environment '.venv' already exists."; \
	fi

.PHONY: show
show: venv ## Show details about the current uv-managed environment.
	@echo "Current $(ENV_NAME) environment (managed by uv):"
	@echo "uv version:"
	@uv --version
	@echo "Python version:"
	@uv run python -V
	@echo "Installed packages:"
	@uv pip list

.PHONY: fmt
fmt: venv ## Format, lint, and type-check code using uv run.
	@echo "Running formatters, linters, and type checkers via 'uv run'..."
	@echo "--- Formatting (ruff format) ---"
	@uv run ruff format src/pollinator_abundance/
	@echo "--- Linting & Autofixing (ruff check) ---"
	@uv run ruff check --fix src/pollinator_abundance/
	@echo "--- Type Checking (mypy) ---"
	@uv run mypy src/pollinator_abundance/
	@echo "Formatting, linting, and type checking complete."

.PHONY: clean
clean: ## Remove the .venv directory.
	@echo "Removing .venv directory..."
	@rm -rf .venv
