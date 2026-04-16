# GitHub Actions CI/CD

This directory contains the GitHub Actions workflows for continuous integration and deployment.

## Workflows

### CI (`ci.yml`)

The main CI workflow runs on push and pull requests to main/master/develop branches.

**Jobs:**

1. **Lint** - Code quality checks
   - Ruff formatting check (`ruff format --check`)
   - Ruff linting (`ruff check`)
   - MyPy type checking (`mypy kingmaker`)
     - Currently set to `continue-on-error: true` (won't fail CI)
     - Once types are added, remove this flag

2. **Test** - Unit tests
   - Runs on Python 3.9, 3.10, 3.11, 3.12
   - Executes pytest with coverage
   - Uploads coverage to Codecov (Python 3.11 only)

3. **Build** - Package building
   - Builds source and wheel distributions
   - Validates package with `twine check`

## Local Development

To run the same checks locally:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run ruff formatting
ruff format .

# Run ruff linting
ruff check .

# Run type checking
mypy kingmaker

# Run tests
pytest tests/ --cov=kingmaker
```

## Adding Type Hints

When ready to enforce type checking:

1. Add type hints to the codebase
2. In `.github/workflows/ci.yml`, remove `continue-on-error: true` from the mypy step
3. In `pyproject.toml`, gradually tighten mypy settings:
   - Set `check_untyped_defs = true`
   - Set `disallow_incomplete_defs = true`
   - Eventually set `disallow_untyped_defs = true` for full strictness
