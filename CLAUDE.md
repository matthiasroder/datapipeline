# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Lint/Test Commands
- Build: `python setup.py build`
- Lint: `flake8 datapipeline tests`
- Test (all): `pytest tests/`
- Test (single): `pytest tests/test_file.py::test_function_name -v`
- Type check: `mypy datapipeline/`

## Code Style Guidelines
- Follow PEP 8 for Python code style
- Maximum line length: 100 characters
- Use type hints for all function parameters and return values
- Class names: PascalCase (e.g., `DataPipeline`)
- Functions/variables: snake_case (e.g., `process_data`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_BATCH_SIZE`)
- Imports order: standard library, third-party, local
- Error handling: Use specific exceptions, include helpful error messages
- Document public APIs with docstrings (Google style)