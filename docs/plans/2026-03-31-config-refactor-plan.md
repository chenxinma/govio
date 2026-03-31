# 数据源配置重构实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 重构数据源配置支持 URL 格式，并增加 DuckDB 目录数据源支持

**Architecture:** 简化 DataSourceConfig 为 url + connect_args，DatabaseManager 区分 SQLAlchemy 和 DuckDB 两种引擎类型

**Tech Stack:** Python 3.13+, SQLAlchemy, DuckDB, pytest

---

### Task 1: 添加 DuckDB 依赖

**Files:**
- Modify: `pyproject.toml`

**Step 1: 添加 duckdb 依赖**

在 `dependencies` 中添加 `duckdb`：

```toml
dependencies = [
    # ... 现有依赖
    "duckdb>=0.10.0",
]
```

**Step 2: 安装依赖**

Run: `uv sync`

**Step 3: 验证安装**

Run: `uv run python -c "import duckdb; print(duckdb.__version__)"`

Expected: 输出 DuckDB 版本号

---

### Task 2: 重构 DataSourceConfig 类

**Files:**
- Modify: `src/govio/mcp/config.py`
- Create: `tests/mcp/test_config.py`

**Step 1: 创建测试文件**

创建 `tests/mcp/test_config.py`：

```python
import pytest
from pathlib import Path
import tempfile
import json

from govio.mcp.config import DataSourceConfig, Config, load_config


def test_datasource_config_url():
    config = DataSourceConfig(url="trino://user:pass@host:8080/db")
    assert config.url == "trino://user:pass@host:8080/db"
    assert config.connect_args == {}


def test_datasource_config_with_connect_args():
    config = DataSourceConfig(
        url="trino://user:pass@host:8080/db",
        connect_args={"http_scheme": "https"}
    )
    assert config.url == "trino://user:pass@host:8080/db"
    assert config.connect_args == {"http_scheme": "https"}


def test_load_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "ds1": {
                "url": "trino://user:pass@host:8080/db",
                "connect_args": {"http_scheme": "https"}
            },
            "ds2": {
                "url": "duckdb:///data/warehouse"
            }
        }, f)
        f.flush()
        
        config = load_config(Path(f.name))
        
        assert "ds1" in config.datasources
        assert "ds2" in config.datasources
        assert config.datasources["ds1"].url == "trino://user:pass@host:8080/db"
        assert config.datasources["ds1"].connect_args == {"http_scheme": "https"}
        assert config.datasources["ds2"].url == "duckdb:///data/warehouse"
        assert config.datasources["ds2"].connect_args == {}


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.json"))
```

**Step 2: 运行测试确认失败**

Run: `uv run pytest tests/mcp/test_config.py -v`

Expected: FAIL - DataSourceConfig 没有 url 属性

**Step 3: 重构 DataSourceConfig**

修改 `src/govio/mcp/config.py`：

```python
"""数据源配置加载"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataSourceConfig:
    """数据源配置"""

    url: str
    connect_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """配置"""

    datasources: dict[str, DataSourceConfig]


def load_config(path: Path) -> Config:
    """加载配置文件"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasources = {}
    for name, ds_data in data.items():
        datasources[name] = DataSourceConfig(
            url=ds_data.get("url", ""),
            connect_args=ds_data.get("connect_args", {}),
        )

    return Config(datasources=datasources)
```

**Step 4: 运行测试确认通过**

Run: `uv run pytest tests/mcp/test_config.py -v`

Expected: PASS

**Step 5: 运行代码检查**

Run: `uv run ruff check src/govio/mcp/config.py tests/mcp/test_config.py`

Expected: 无错误

**Step 6: 提交**

```bash
git add src/govio/mcp/config.py tests/mcp/test_config.py
git commit -m "refactor: simplify DataSourceConfig to url + connect_args"
```

---

### Task 3: 重构 DatabaseManager 支持 DuckDB

**Files:**
- Modify: `src/govio/mcp/core/database.py`
- Create: `tests/mcp/core/test_database.py`

**Step 1: 创建测试文件**

创建 `tests/mcp/core/test_database.py`：

```python
import pytest
import tempfile
from pathlib import Path

from govio.mcp.config import DataSourceConfig
from govio.mcp.core.database import DatabaseManager


def test_database_manager_sqlalchemy():
    configs = {
        "sqlite_db": DataSourceConfig(url="sqlite:///:memory:")
    }
    manager = DatabaseManager(configs)
    
    engine = manager.get_engine("sqlite_db")
    assert engine is not None


def test_database_manager_duckdb():
    with tempfile.TemporaryDirectory() as tmpdir:
        configs = {
            "local_data": DataSourceConfig(url=f"duckdb://{tmpdir}")
        }
        manager = DatabaseManager(configs)
        
        df = manager.execute_sql("local_data", "SELECT 1 as a")
        assert df is not None
        assert list(df.columns) == ["a"]


def test_database_manager_duckdb_invalid_dir():
    configs = {
        "invalid": DataSourceConfig(url="duckdb:///nonexistent/path")
    }
    
    with pytest.raises(RuntimeError, match="初始化数据源"):
        manager = DatabaseManager(configs)


def test_database_manager_unknown_datasource():
    configs = {
        "sqlite_db": DataSourceConfig(url="sqlite:///:memory:")
    }
    manager = DatabaseManager(configs)
    
    with pytest.raises(ValueError, match="数据源不存在"):
        manager.get_engine("unknown")
```

**Step 2: 运行测试确认失败**

Run: `uv run pytest tests/mcp/core/test_database.py -v`

Expected: FAIL - DatabaseManager 不支持 DuckDB

**Step 3: 重构 DatabaseManager**

修改 `src/govio/mcp/core/database.py`：

```python
"""数据库连接管理"""

from pathlib import Path

import duckdb
import pandas as pd
from duckdb import DuckDBPyConnection
from sqlalchemy import Engine, create_engine, text

from govio.mcp.config import DataSourceConfig


class DatabaseManager:
    """数据库连接管理器"""

    def __init__(self, datasources: dict[str, DataSourceConfig]) -> None:
        self._datasources = datasources
        self._engines: dict[str, Engine] = {}
        self._duckdb_conns: dict[str, DuckDBPyConnection] = {}
        self._duckdb_dirs: dict[str, str] = {}
        self._init_connections()

    def _init_connections(self) -> None:
        """初始化所有数据源连接"""
        for name, config in self._datasources.items():
            try:
                if config.url.startswith("duckdb://"):
                    dir_path = config.url[9:]
                    if not Path(dir_path).exists():
                        raise ValueError(f"目录不存在: {dir_path}")
                    conn = duckdb.connect(":memory:")
                    self._duckdb_conns[name] = conn
                    self._duckdb_dirs[name] = dir_path
                else:
                    self._engines[name] = create_engine(
                        config.url, connect_args=config.connect_args
                    )
            except Exception as e:
                raise RuntimeError(f"初始化数据源 '{name}' 失败: {e}") from e

    def get_engine(self, datasource: str) -> Engine:
        """获取数据源引擎（仅用于 SQLAlchemy 数据源）"""
        if datasource not in self._engines:
            raise ValueError(f"数据源不存在或不是 SQLAlchemy 类型: {datasource}")
        return self._engines[datasource]

    def execute_sql(self, datasource: str, sql: str) -> pd.DataFrame:
        """执行 SQL 并返回 DataFrame"""
        if datasource in self._duckdb_conns:
            conn = self._duckdb_conns[datasource]
            return conn.execute(sql).df()
        elif datasource in self._engines:
            engine = self._engines[datasource]
            with engine.connect() as conn:
                return pd.read_sql(text(sql), conn)
        else:
            raise ValueError(f"数据源不存在: {datasource}")
```

**Step 4: 运行测试确认通过**

Run: `uv run pytest tests/mcp/core/test_database.py -v`

Expected: PASS

**Step 5: 运行代码检查**

Run: `uv run ruff check src/govio/mcp/core/database.py tests/mcp/core/test_database.py`

Expected: 无错误

**Step 6: 提交**

```bash
git add src/govio/mcp/core/database.py tests/mcp/core/test_database.py
git commit -m "feat: add DuckDB support to DatabaseManager"
```

---

### Task 4: 更新现有测试

**Files:**
- Modify: `tests/mcp/test_*.py` (如有使用旧配置格式的测试)

**Step 1: 搜索使用旧配置的测试**

Run: `grep -r "driver=" tests/` 或 `grep -r "DataSourceConfig(" tests/`

**Step 2: 更新测试使用新格式**

如果找到使用旧格式的测试，更新为：

```python
DataSourceConfig(url="sqlite:///:memory:")
```

**Step 3: 运行所有测试**

Run: `uv run pytest tests/ -v`

Expected: PASS

**Step 4: 提交（如有变更）**

```bash
git add tests/
git commit -m "test: update tests for new config format"
```

---

### Task 5: 更新文档

**Files:**
- Modify: `docs/connect.json` (示例配置)

**Step 1: 更新示例配置**

确保 `docs/connect.json` 使用新格式（已符合）：

```json
{
  "ds1": {
    "url": "trino://user:{password}@host:port/database",
    "connect_args": {
      "http_scheme": "https",
      "timezone": "Asia/Shanghai"
    }
  }
}
```

**Step 2: 提交**

```bash
git add docs/connect.json
git commit -m "docs: update connect.json example"
```

---

### Task 6: 最终验证

**Step 1: 运行所有测试**

Run: `uv run pytest tests/ -v`

Expected: PASS

**Step 2: 运行代码检查**

Run: `uv run ruff check src/ tests/`

Expected: 无错误

**Step 3: 运行格式化**

Run: `uv run ruff format src/ tests/`

Expected: 无变更或格式化完成

**Step 4: 最终提交（如有变更）**

```bash
git add .
git commit -m "chore: final cleanup and formatting"
```
