# AGENTS.md

Guidelines for AI coding agents working in this repository.

## Project Overview

Govio is a data governance knowledge graph library built on NetworkX. It provides metadata management, data standard recommendation, and graph database integration capabilities.

- **Language**: Python 3.13+
- **Package Manager**: uv
- **Build Backend**: hatchling

## Build/Lint/Test Commands

### Installation

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --group dev
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_relationship.py

# Run a specific test
uv run pytest tests/test_relationship.py::test_load_json_success

# Run tests with verbose output
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=src/govio
```

### Linting and Formatting

```bash
# Check code with ruff
uv run ruff check src/ tests/

# Format code with ruff
uv run ruff format src/ tests/

# Fix linting issues automatically
uv run ruff check --fix src/ tests/
```

### Build

```bash
# Build package
uv build

# Build wheel only
uv build --wheel
```

### Type Checking

```bash
# Run pyright (if available)
uv run pyright src/
```

## Code Style Guidelines

### Imports

Order imports in three groups, separated by blank lines:

1. Standard library imports (alphabetical)
2. Third-party imports (alphabetical)
3. Local imports (alphabetical)

```python
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .application import AppInfoLoader
from .database import DatabaseLoader
```

### Type Hints

Use modern Python 3.13+ type hint syntax:

```python
# Preferred
def load_tables(self, schema_limits: list[str] | None) -> pd.DataFrame:
    ...

# Avoid (old style)
from typing import List, Optional
def load_tables(self, schema_limits: Optional[List[str]]) -> pd.DataFrame:
    ...
```

Use union types with `|` for optional parameters. Use `Any` sparingly.

### Naming Conventions

- **Modules**: snake_case (`recommender.py`, `relationship.py`)
- **Classes**: PascalCase (`StandardRecommender`, `RelationshipLoader`)
- **Functions/Methods**: snake_case (`load_relationships`, `find_k_neighbors`)
- **Private methods**: prefix with underscore (`_preprocess_std_data`, `_validate_inputs`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_WEIGHTS`, `MIN_SIMILARITY`)
- **Properties**: snake_case with `@property` decorator (`PhysicalTable`, `Col`)

### Docstrings

Use Chinese docstrings for Chinese-language projects. Include Args and Returns sections:

```python
def validate_relationship(self, rel: dict[str, Any], index: int) -> bool:
    """验证单个关系的有效性

    Args:
        rel: 关系字典
        index: 关系索引（用于错误消息）

    Returns:
        bool: 是否有效
    """
```

### Classes

- Use `__init__` for initialization with type-annotated parameters
- Use `@property` for computed attributes that don't require parameters
- Use factory functions for complex object creation

```python
class DatabaseLoader:
    def __init__(self, db: str, workspace_uuid: str, schema_limits: list[str] | None = None) -> None:
        self.engine = create_engine(db)
        self.workspace_uuid = workspace_uuid

    @property
    def PhysicalTable(self) -> pd.DataFrame:
        return self.load_tables()
```

### Error Handling

- Raise descriptive exceptions with context
- Use logging module for warnings (not print statements)
- Validate inputs early in public methods

```python
def _validate_inputs(self):
    if not self.json_path.exists():
        raise FileNotFoundError(f"关系文件不存在: {self.json_path}")

    if self.df_tables.empty:
        raise ValueError("PhysicalTable DataFrame 为空")
```

### Comments

- Write comments in Chinese for this codebase
- Avoid inline comments that restate the obvious
- Use module-level docstrings to explain purpose

### File Organization

```
src/govio/
├── __init__.py          # Public exports only
├── graph/               # Graph database implementations
│   ├── __init__.py
│   ├── networkx_graph.py
│   └── falkordb_graph.py
└── metadata/            # Metadata loading and processing
    ├── __init__.py
    ├── database.py
    ├── recommender.py
    └── relationship.py
```

### Constants and Configuration

Define module-level constants at the top of the file:

```python
DEFAULT_WEIGHTS = {
    'table': 0.20,
    'name': 0.26,
    'comment': 0.22,
    'type': 0.22,
    'numeric': 0.10
}

DEFAULT_K_NEIGHBORS = 5
MIN_SIMILARITY = 0.7
```

### Avoid

- Adding comments that restate code
- Using `print()` for logging (use `logging` module)
- Mutable default arguments
- Bare `except` clauses
- Star imports (`from module import *`)

## Project-Specific Notes

### Entry Points

The package defines CLI entry points in `pyproject.toml`:

- `metadata` - Generate metadata CSV files
- `gml_generate` - Generate GML graph files

### Environment Variables

Load environment variables using `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()
db = os.getenv("KUNDB_URL", "")
```

### Testing

Tests use pytest with fixtures. Place fixtures at module level or in conftest.py:

```python
@pytest.fixture
def sample_tables():
    return pd.DataFrame({
        "full_table_name": ["db.schema.table1"],
    })
```