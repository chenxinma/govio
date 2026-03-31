# MCP 数据探查服务实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 创建一个基于 Streamable HTTP 的 MCP 服务，提供数据表比对和数据表关系探查功能。

**Architecture:** 服务通过 SQLAlchemy 连接数据库，将查询结果存储为内存 DataFrame，基于 datacompy 进行表比对，基于列名相似性和数据内容推断关系，使用 networkx 生成关系图谱。

**Tech Stack:** Python 3.13+, MCP 2025-03-26, SQLAlchemy, pandas, duckdb, datacompy, networkx

---

## Task 1: 添加依赖项

**Files:**
- Modify: `pyproject.toml`

**Step 1: 添加 MCP 和相关依赖**

```toml
dependencies = [
    # 现有依赖...
    "duckdb>=0.10.0",
    "datacompy>=0.10.0",
]
```

**Step 2: 添加脚本入口**

在 `[project.scripts]` 部分添加：

```toml
mcp-server = "govio.mcp.server:main"
```

**Step 3: 安装依赖**

Run: `uv sync`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(mcp): add mcp dependencies and entry point"
```

---

## Task 2: 创建 MCP 包结构

**Files:**
- Create: `src/govio/mcp/__init__.py`
- Create: `src/govio/mcp/tools/__init__.py`
- Create: `src/govio/mcp/core/__init__.py`

**Step 1: 创建目录结构**

```bash
mkdir -p src/govio/mcp/tools src/govio/mcp/core
```

**Step 2: 创建 `src/govio/mcp/__init__.py`**

```python
"""MCP 数据探查服务"""
```

**Step 3: 创建 `src/govio/mcp/tools/__init__.py`**

```python
"""MCP 工具模块"""
```

**Step 4: 创建 `src/govio/mcp/core/__init__.py`**

```python
"""MCP 核心模块"""
```

**Step 5: Commit**

```bash
git add src/govio/mcp/
git commit -m "feat(mcp): create mcp package structure"
```

---

## Task 3: 实现数据源配置模块

**Files:**
- Create: `src/govio/mcp/config.py`
- Create: `tests/test_mcp_config.py`

**Step 1: 编写测试**

```python
"""测试数据源配置"""
import json
import tempfile
from pathlib import Path

import pytest

from govio.mcp.config import DataSourceConfig, load_config


def test_load_config_success():
    config_data = {
        "datasources": {
            "testdb": {
                "driver": "sqlite",
                "database": ":memory:"
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        f.flush()
        config = load_config(Path(f.name))
    
    assert "testdb" in config.datasources
    assert config.datasources["testdb"].driver == "sqlite"


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.json"))


def test_datasource_config_to_url():
    config = DataSourceConfig(
        driver="mysql+pymysql",
        host="localhost",
        port=3306,
        database="testdb",
        username="user",
        password="pass"
    )
    
    url = config.to_url()
    assert url == "mysql+pymysql://user:pass@localhost:3306/testdb"


def test_datasource_config_to_url_sqlite():
    config = DataSourceConfig(
        driver="sqlite",
        database="/path/to/db.sqlite"
    )
    
    url = config.to_url()
    assert url == "sqlite:///path/to/db.sqlite"
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_config.py -v`
Expected: FAIL (模块不存在)

**Step 3: 实现配置模块**

```python
"""数据源配置加载"""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataSourceConfig:
    """数据源配置"""
    driver: str
    host: str = ""
    port: int = 0
    database: str = ""
    username: str = ""
    password: str = ""

    def to_url(self) -> str:
        """转换为 SQLAlchemy 连接 URL"""
        if self.driver.startswith("sqlite"):
            return f"{self.driver}:///{self.database}"
        
        auth = ""
        if self.username:
            auth = self.username
            if self.password:
                auth += f":{self.password}"
            auth += "@"
        
        port_str = f":{self.port}" if self.port else ""
        return f"{self.driver}://{auth}{self.host}{port_str}/{self.database}"


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
    for name, ds_data in data.get("datasources", {}).items():
        datasources[name] = DataSourceConfig(
            driver=ds_data.get("driver", ""),
            host=ds_data.get("host", ""),
            port=ds_data.get("port", 0),
            database=ds_data.get("database", ""),
            username=ds_data.get("username", ""),
            password=ds_data.get("password", ""),
        )
    
    return Config(datasources=datasources)
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/config.py tests/test_mcp_config.py
git commit -m "feat(mcp): add datasource config module"
```

---

## Task 4: 实现数据库连接管理

**Files:**
- Create: `src/govio/mcp/core/database.py`
- Create: `tests/test_mcp_database.py`

**Step 1: 编写测试**

```python
"""测试数据库连接管理"""
import pytest
from sqlalchemy import create_engine

from govio.mcp.config import DataSourceConfig
from govio.mcp.core.database import DatabaseManager


@pytest.fixture
def sqlite_config():
    return DataSourceConfig(
        driver="sqlite",
        database=":memory:"
    )


def test_database_manager_init(sqlite_config):
    manager = DatabaseManager({"test": sqlite_config})
    assert "test" in manager._engines


def test_get_engine(sqlite_config):
    manager = DatabaseManager({"test": sqlite_config})
    engine = manager.get_engine("test")
    assert engine is not None


def test_get_engine_not_found(sqlite_config):
    manager = DatabaseManager({"test": sqlite_config})
    with pytest.raises(ValueError, match="数据源不存在"):
        manager.get_engine("nonexistent")


def test_execute_sql(sqlite_config):
    manager = DatabaseManager({"test": sqlite_config})
    
    df = manager.execute_sql("test", "SELECT 1 as a, 2 as b")
    
    assert len(df) == 1
    assert list(df.columns) == ["a", "b"]
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_database.py -v`
Expected: FAIL

**Step 3: 实现数据库管理模块**

```python
"""数据库连接管理"""
import pandas as pd
from sqlalchemy import Engine, create_engine, text

from govio.mcp.config import DataSourceConfig


class DatabaseManager:
    """数据库连接管理器"""

    def __init__(self, datasources: dict[str, DataSourceConfig]) -> None:
        self._datasources = datasources
        self._engines: dict[str, Engine] = {}
        self._init_engines()

    def _init_engines(self) -> None:
        """初始化所有数据源连接"""
        for name, config in self._datasources.items():
            url = config.to_url()
            self._engines[name] = create_engine(url)

    def get_engine(self, datasource: str) -> Engine:
        """获取数据源引擎"""
        if datasource not in self._engines:
            raise ValueError(f"数据源不存在: {datasource}")
        return self._engines[datasource]

    def execute_sql(self, datasource: str, sql: str) -> pd.DataFrame:
        """执行 SQL 并返回 DataFrame"""
        engine = self.get_engine(datasource)
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_database.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/core/database.py tests/test_mcp_database.py
git commit -m "feat(mcp): add database manager module"
```

---

## Task 5: 实现 DataFrame 存储管理

**Files:**
- Create: `src/govio/mcp/core/dataframe_store.py`
- Create: `tests/test_mcp_dataframe_store.py`

**Step 1: 编写测试**

```python
"""测试 DataFrame 存储"""
import pandas as pd
import pytest

from govio.mcp.core.dataframe_store import DataFrameInfo, DataFrameStore


@pytest.fixture
def store():
    return DataFrameStore()


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_store_dataframe(store, sample_df):
    info = store.store("test_df", sample_df)
    
    assert info.name == "test_df"
    assert info.rows == 3
    assert info.columns == 2


def test_get_dataframe(store, sample_df):
    store.store("test_df", sample_df)
    
    df = store.get("test_df")
    
    assert df is not None
    assert len(df) == 3


def test_get_dataframe_not_found(store):
    df = store.get("nonexistent")
    assert df is None


def test_list_dataframes(store, sample_df):
    store.store("df1", sample_df)
    store.store("df2", sample_df)
    
    infos = store.list()
    
    assert len(infos) == 2
    names = [info.name for info in infos]
    assert "df1" in names
    assert "df2" in names


def test_release_dataframe(store, sample_df):
    store.store("test_df", sample_df)
    
    result = store.release("test_df")
    
    assert result is True
    assert store.get("test_df") is None


def test_release_dataframe_not_found(store):
    result = store.release("nonexistent")
    assert result is False
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_dataframe_store.py -v`
Expected: FAIL

**Step 3: 实现 DataFrame 存储模块**

```python
"""DataFrame 内存存储管理"""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DataFrameInfo:
    """DataFrame 信息"""
    name: str
    rows: int
    columns: int
    column_info: list[dict] = field(default_factory=list)


class DataFrameStore:
    """DataFrame 内存存储"""
    _instance = None
    _dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __new__(cls) -> "DataFrameStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._dataframes = {}
        return cls._instance

    def store(self, name: str, df: pd.DataFrame) -> DataFrameInfo:
        """存储 DataFrame"""
        self._dataframes[name] = df
        
        column_info = [
            {"name": col, "dtype": str(df[col].dtype)}
            for col in df.columns
        ]
        
        return DataFrameInfo(
            name=name,
            rows=len(df),
            columns=len(df.columns),
            column_info=column_info
        )

    def get(self, name: str) -> pd.DataFrame | None:
        """获取 DataFrame"""
        return self._dataframes.get(name)

    def list(self) -> list[DataFrameInfo]:
        """列出所有 DataFrame"""
        infos = []
        for name, df in self._dataframes.items():
            column_info = [
                {"name": col, "dtype": str(df[col].dtype)}
                for col in df.columns
            ]
            infos.append(DataFrameInfo(
                name=name,
                rows=len(df),
                columns=len(df.columns),
                column_info=column_info
            ))
        return infos

    def release(self, name: str) -> bool:
        """释放 DataFrame"""
        if name in self._dataframes:
            del self._dataframes[name]
            return True
        return False
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_dataframe_store.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/core/dataframe_store.py tests/test_mcp_dataframe_store.py
git commit -m "feat(mcp): add dataframe store module"
```

---

## Task 6: 实现表比对模块

**Files:**
- Create: `src/govio/mcp/core/comparator.py`
- Create: `tests/test_mcp_comparator.py`

**Step 1: 编写测试**

```python
"""测试表比对"""
import pandas as pd
import pytest

from govio.mcp.core.comparator import CompareResult, TableComparator


@pytest.fixture
def comparator():
    return TableComparator()


@pytest.fixture
def source_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
        "value": [100, 200, 300]
    })


@pytest.fixture
def target_df():
    return pd.DataFrame({
        "id": [1, 2, 4],
        "name": ["a", "b", "d"],
        "value": [100, 250, 400]
    })


def test_compare_schema(comparator, source_df, target_df):
    result = comparator.compare_schema(source_df, target_df)
    
    assert result["match"] is True
    assert len(result["source_columns"]) == 3
    assert len(result["target_columns"]) == 3


def test_compare_schema_different_columns(comparator, source_df):
    target_df = pd.DataFrame({
        "id": [1, 2],
        "name": ["a", "b"],
        "extra": [1, 2]
    })
    
    result = comparator.compare_schema(source_df, target_df)
    
    assert result["match"] is False
    assert "value" in result["source_only"]
    assert "extra" in result["target_only"]


def test_compare_data(comparator, source_df, target_df):
    result = comparator.compare_data(source_df, target_df, join_columns=["id"])
    
    assert result["match_rate"] < 1.0
    assert result["rows_matched"] == 1


def test_compare_full(comparator, source_df, target_df):
    result = comparator.compare(source_df, target_df, join_columns=["id"])
    
    assert "schema" in result
    assert "data" in result
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_comparator.py -v`
Expected: FAIL

**Step 3: 实现表比对模块**

```python
"""表比对核心逻辑"""
from typing import Any

import datacompy
import pandas as pd


class TableComparator:
    """表比对器"""

    def compare_schema(
        self, source: pd.DataFrame, target: pd.DataFrame
    ) -> dict[str, Any]:
        """比对表结构"""
        source_cols = set(source.columns)
        target_cols = set(target.columns)
        
        common_cols = source_cols & target_cols
        source_only = source_cols - target_cols
        target_only = target_cols - source_cols
        
        return {
            "match": len(source_only) == 0 and len(target_only) == 0,
            "source_columns": sorted(list(source_cols)),
            "target_columns": sorted(list(target_cols)),
            "common_columns": sorted(list(common_cols)),
            "source_only": sorted(list(source_only)),
            "target_only": sorted(list(target_only)),
        }

    def compare_data(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        join_columns: list[str]
    ) -> dict[str, Any]:
        """比对数据"""
        compare = datacompy.Compare(
            df1=source,
            df2=target,
            join_columns=join_columns
        )
        
        return {
            "match_rate": compare.percent_match,
            "rows_matched": compare.count_matching_rows(),
            "rows_in_source": compare.df1_rows,
            "rows_in_target": compare.df2_rows,
            "rows_only_in_source": compare.df1_unq_rows,
            "rows_only_in_target": compare.df2_unq_rows,
        }

    def compare(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        join_columns: list[str]
    ) -> dict[str, Any]:
        """完整比对"""
        schema_result = self.compare_schema(source, target)
        data_result = self.compare_data(source, target, join_columns)
        
        return {
            "schema": schema_result,
            "data": data_result,
        }
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_comparator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/core/comparator.py tests/test_mcp_comparator.py
git commit -m "feat(mcp): add table comparator module"
```

---

## Task 7: 实现关系探查模块

**Files:**
- Create: `src/govio/mcp/core/explorer.py`
- Create: `tests/test_mcp_explorer.py`

**Step 1: 编写测试**

```python
"""测试关系探查"""
import pandas as pd
import pytest

from govio.mcp.core.explorer import RelationExplorer


@pytest.fixture
def explorer():
    return RelationExplorer()


@pytest.fixture
def orders_df():
    return pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer_id": [100, 101, 100],
        "amount": [1000, 2000, 1500]
    })


@pytest.fixture
def customers_df():
    return pd.DataFrame({
        "customer_id": [100, 101, 102],
        "name": ["Alice", "Bob", "Charlie"]
    })


def test_find_column_similarity(explorer, orders_df, customers_df):
    similarities = explorer.find_column_similarity(orders_df, customers_df)
    
    customer_id_match = next(
        (s for s in similarities if s["column"] == "customer_id"),
        None
    )
    assert customer_id_match is not None
    assert customer_id_match["similarity"] > 0.9


def test_infer_foreign_keys(explorer, orders_df, customers_df):
    relations = explorer.infer_foreign_keys(
        orders_df, "orders",
        customers_df, "customers"
    )
    
    assert len(relations) > 0
    fk = relations[0]
    assert fk["source_table"] == "orders"
    assert fk["source_column"] == "customer_id"
    assert fk["target_table"] == "customers"
    assert fk["target_column"] == "customer_id"


def test_explore(explorer, orders_df, customers_df):
    dataframes = {
        "orders": orders_df,
        "customers": customers_df
    }
    
    relations = explorer.explore(dataframes)
    
    assert len(relations) > 0
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_explorer.py -v`
Expected: FAIL

**Step 3: 实现关系探查模块**

```python
"""关系探查核心逻辑"""
from difflib import SequenceMatcher
from typing import Any

import pandas as pd


class RelationExplorer:
    """关系探查器"""

    def find_column_similarity(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """查找列名相似的列"""
        similarities = []
        
        for col1 in df1.columns:
            for col2 in df2.columns:
                ratio = SequenceMatcher(None, col1.lower(), col2.lower()).ratio()
                if ratio > 0.7:
                    similarities.append({
                        "column": col1,
                        "match_column": col2,
                        "similarity": ratio
                    })
        
        return sorted(similarities, key=lambda x: x["similarity"], reverse=True)

    def infer_foreign_keys(
        self,
        source_df: pd.DataFrame,
        source_name: str,
        target_df: pd.DataFrame,
        target_name: str
    ) -> list[dict[str, Any]]:
        """推断外键关系"""
        relations = []
        
        for col in source_df.columns:
            if col.endswith("_id") or col.endswith("Id"):
                target_col = col.replace("_id", "").replace("Id", "") + "_id"
                if target_col not in target_df.columns:
                    target_col = col
                
                if col in target_df.columns:
                    source_values = set(source_df[col].dropna().unique())
                    target_values = set(target_df[col].dropna().unique())
                    
                    if source_values and target_values:
                        overlap = len(source_values & target_values)
                        ratio = overlap / len(source_values) if source_values else 0
                        
                        if ratio > 0.5:
                            relations.append({
                                "source_table": source_name,
                                "source_column": col,
                                "target_table": target_name,
                                "target_column": col,
                                "confidence": ratio
                            })
        
        return relations

    def explore(
        self, dataframes: dict[str, pd.DataFrame]
    ) -> list[dict[str, Any]]:
        """探查所有 DataFrame 之间的关系"""
        all_relations = []
        names = list(dataframes.keys())
        
        for i, name1 in enumerate(names):
            for name2 in names[i + 1:]:
                df1 = dataframes[name1]
                df2 = dataframes[name2]
                
                relations = self.infer_foreign_keys(df1, name1, df2, name2)
                all_relations.extend(relations)
                
                relations = self.infer_foreign_keys(df2, name2, df1, name1)
                all_relations.extend(relations)
        
        return all_relations
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_explorer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/core/explorer.py tests/test_mcp_explorer.py
git commit -m "feat(mcp): add relation explorer module"
```

---

## Task 8: 实现关系可视化模块

**Files:**
- Create: `src/govio/mcp/core/visualizer.py`
- Create: `tests/test_mcp_visualizer.py`

**Step 1: 编写测试**

```python
"""测试关系可视化"""
import pytest

from govio.mcp.core.visualizer import RelationVisualizer


@pytest.fixture
def visualizer():
    return RelationVisualizer()


@pytest.fixture
def relations():
    return [
        {
            "source_table": "orders",
            "source_column": "customer_id",
            "target_table": "customers",
            "target_column": "customer_id",
            "confidence": 0.9
        }
    ]


def test_to_networkx(visualizer, relations):
    graph = visualizer.to_networkx(relations)
    
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1


def test_to_json(visualizer, relations):
    result = visualizer.to_json(relations)
    
    assert "nodes" in result
    assert "edges" in result
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1


def test_visualize(visualizer, relations):
    result = visualizer.visualize(relations)
    
    assert "nodes" in result
    assert "edges" in result
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_visualizer.py -v`
Expected: FAIL

**Step 3: 实现关系可视化模块**

```python
"""关系可视化"""
from typing import Any

import networkx as nx


class RelationVisualizer:
    """关系可视化器"""

    def to_networkx(
        self, relations: list[dict[str, Any]]
    ) -> nx.DiGraph:
        """转换为 NetworkX 图"""
        graph = nx.DiGraph()
        
        for rel in relations:
            source = rel["source_table"]
            target = rel["target_table"]
            
            if not graph.has_node(source):
                graph.add_node(source, type="table")
            if not graph.has_node(target):
                graph.add_node(target, type="table")
            
            graph.add_edge(
                source, target,
                source_column=rel.get("source_column", ""),
                target_column=rel.get("target_column", ""),
                confidence=rel.get("confidence", 0)
            )
        
        return graph

    def to_json(
        self, relations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """转换为 JSON 格式"""
        graph = self.to_networkx(relations)
        
        nodes = []
        for node, data in graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": node,
                **data
            })
        
        edges = []
        for source, target, data in graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                **data
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }

    def visualize(
        self, relations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """生成可视化数据"""
        return self.to_json(relations)
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_visualizer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/core/visualizer.py tests/test_mcp_visualizer.py
git commit -m "feat(mcp): add relation visualizer module"
```

---

## Task 9: 实现 MCP 工具 - load_dataframe

**Files:**
- Create: `src/govio/mcp/tools/load_dataframe.py`
- Create: `tests/test_mcp_tools_load.py`

**Step 1: 编写测试**

```python
"""测试 load_dataframe 工具"""
import pandas as pd
import pytest
from sqlalchemy import create_engine

from govio.mcp.config import DataSourceConfig
from govio.mcp.core.database import DatabaseManager
from govio.mcp.core.dataframe_store import DataFrameStore
from govio.mcp.tools.load_dataframe import load_dataframe


@pytest.fixture
def store():
    return DataFrameStore()


@pytest.fixture
def db_manager():
    config = DataSourceConfig(driver="sqlite", database=":memory:")
    manager = DatabaseManager({"test": config})
    engine = manager.get_engine("test")
    
    with engine.connect() as conn:
        conn.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test_table VALUES (1, 'a'), (2, 'b')")
        conn.commit()
    
    return manager


def test_load_dataframe_success(store, db_manager):
    result = load_dataframe(
        store=store,
        db_manager=db_manager,
        datasource="test",
        name="my_df",
        sql="SELECT * FROM test_table"
    )
    
    assert result["success"] is True
    assert result["rows"] == 2
    assert result["columns"] == 2


def test_load_dataframe_datasource_not_found(store, db_manager):
    result = load_dataframe(
        store=store,
        db_manager=db_manager,
        datasource="nonexistent",
        name="my_df",
        sql="SELECT 1"
    )
    
    assert result["success"] is False
    assert "不存在" in result["error"]


def test_load_dataframe_sql_error(store, db_manager):
    result = load_dataframe(
        store=store,
        db_manager=db_manager,
        datasource="test",
        name="my_df",
        sql="SELECT * FROM nonexistent"
    )
    
    assert result["success"] is False
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_tools_load.py -v`
Expected: FAIL

**Step 3: 实现工具**

```python
"""load_dataframe 工具"""
from typing import Any

from govio.mcp.core.database import DatabaseManager
from govio.mcp.core.dataframe_store import DataFrameStore


def load_dataframe(
    store: DataFrameStore,
    db_manager: DatabaseManager,
    datasource: str,
    name: str,
    sql: str
) -> dict[str, Any]:
    """加载 DataFrame 到内存
    
    Args:
        store: DataFrame 存储
        db_manager: 数据库管理器
        datasource: 数据源名称
        name: DataFrame 名称
        sql: 查询 SQL
    
    Returns:
        加载结果
    """
    try:
        df = db_manager.execute_sql(datasource, sql)
        info = store.store(name, df)
        
        return {
            "success": True,
            "name": info.name,
            "rows": info.rows,
            "columns": info.columns,
            "column_info": info.column_info
        }
    except ValueError as e:
        return {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"SQL 执行错误: {str(e)}"
        }
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_tools_load.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/tools/load_dataframe.py tests/test_mcp_tools_load.py
git commit -m "feat(mcp): add load_dataframe tool"
```

---

## Task 10: 实现 MCP 工具 - list_dataframes

**Files:**
- Create: `src/govio/mcp/tools/list_dataframes.py`
- Create: `tests/test_mcp_tools_list.py`

**Step 1: 编写测试**

```python
"""测试 list_dataframes 工具"""
import pandas as pd
import pytest

from govio.mcp.core.dataframe_store import DataFrameStore
from govio.mcp.tools.list_dataframes import list_dataframes


@pytest.fixture
def store():
    return DataFrameStore()


def test_list_dataframes_empty(store):
    result = list_dataframes(store)
    
    assert result["dataframes"] == []


def test_list_dataframes_with_data(store):
    store.store("df1", pd.DataFrame({"a": [1, 2]}))
    store.store("df2", pd.DataFrame({"b": [3, 4, 5]}))
    
    result = list_dataframes(store)
    
    assert len(result["dataframes"]) == 2
    names = [df["name"] for df in result["dataframes"]]
    assert "df1" in names
    assert "df2" in names
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_tools_list.py -v`
Expected: FAIL

**Step 3: 实现工具**

```python
"""list_dataframes 工具"""
from typing import Any

from govio.mcp.core.dataframe_store import DataFrameStore


def list_dataframes(store: DataFrameStore) -> dict[str, Any]:
    """列出已加载的 DataFrame
    
    Args:
        store: DataFrame 存储
    
    Returns:
        DataFrame 清单
    """
    infos = store.list()
    
    return {
        "dataframes": [
            {
                "name": info.name,
                "rows": info.rows,
                "columns": info.columns,
                "column_info": info.column_info
            }
            for info in infos
        ]
    }
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_tools_list.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/tools/list_dataframes.py tests/test_mcp_tools_list.py
git commit -m "feat(mcp): add list_dataframes tool"
```

---

## Task 11: 实现 MCP 工具 - release_dataframe

**Files:**
- Create: `src/govio/mcp/tools/release_dataframe.py`
- Create: `tests/test_mcp_tools_release.py`

**Step 1: 编写测试**

```python
"""测试 release_dataframe 工具"""
import pandas as pd
import pytest

from govio.mcp.core.dataframe_store import DataFrameStore
from govio.mcp.tools.release_dataframe import release_dataframe


@pytest.fixture
def store():
    return DataFrameStore()


def test_release_dataframe_success(store):
    store.store("test_df", pd.DataFrame({"a": [1]}))
    
    result = release_dataframe(store, "test_df")
    
    assert result["success"] is True
    assert store.get("test_df") is None


def test_release_dataframe_not_found(store):
    result = release_dataframe(store, "nonexistent")
    
    assert result["success"] is False
    assert "不存在" in result["error"]
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_tools_release.py -v`
Expected: FAIL

**Step 3: 实现工具**

```python
"""release_dataframe 工具"""
from typing import Any

from govio.mcp.core.dataframe_store import DataFrameStore


def release_dataframe(
    store: DataFrameStore,
    name: str
) -> dict[str, Any]:
    """释放 DataFrame
    
    Args:
        store: DataFrame 存储
        name: DataFrame 名称
    
    Returns:
        释放结果
    """
    if store.get(name) is None:
        return {
            "success": False,
            "error": f"DataFrame '{name}' 不存在"
        }
    
    store.release(name)
    
    return {
        "success": True,
        "message": f"DataFrame '{name}' 已释放"
    }
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_tools_release.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/tools/release_dataframe.py tests/test_mcp_tools_release.py
git commit -m "feat(mcp): add release_dataframe tool"
```

---

## Task 12: 实现 MCP 工具 - compare_tables

**Files:**
- Create: `src/govio/mcp/tools/compare_tables.py`
- Create: `tests/test_mcp_tools_compare.py`

**Step 1: 编写测试**

```python
"""测试 compare_tables 工具"""
import pandas as pd
import pytest

from govio.mcp.core.dataframe_store import DataFrameStore
from govio.mcp.tools.compare_tables import compare_tables


@pytest.fixture
def store():
    return DataFrameStore()


def test_compare_tables_success(store):
    source_df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"]
    })
    target_df = pd.DataFrame({
        "id": [1, 2, 4],
        "name": ["a", "b", "d"]
    })
    
    store.store("source", source_df)
    store.store("target", target_df)
    
    result = compare_tables(store, "source", "target", ["id"])
    
    assert result["success"] is True
    assert "schema" in result
    assert "data" in result


def test_compare_tables_source_not_found(store):
    result = compare_tables(store, "nonexistent", "target", ["id"])
    
    assert result["success"] is False
    assert "不存在" in result["error"]
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_tools_compare.py -v`
Expected: FAIL

**Step 3: 实现工具**

```python
"""compare_tables 工具"""
from typing import Any

from govio.mcp.core.comparator import TableComparator
from govio.mcp.core.dataframe_store import DataFrameStore


def compare_tables(
    store: DataFrameStore,
    source_df: str,
    target_df: str,
    join_columns: list[str]
) -> dict[str, Any]:
    """比对两个 DataFrame
    
    Args:
        store: DataFrame 存储
        source_df: 源 DataFrame 名称
        target_df: 目标 DataFrame 名称
        join_columns: 用于比对的列
    
    Returns:
        比对结果
    """
    source = store.get(source_df)
    if source is None:
        return {
            "success": False,
            "error": f"DataFrame '{source_df}' 不存在"
        }
    
    target = store.get(target_df)
    if target is None:
        return {
            "success": False,
            "error": f"DataFrame '{target_df}' 不存在"
        }
    
    comparator = TableComparator()
    result = comparator.compare(source, target, join_columns)
    
    return {
        "success": True,
        **result
    }
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_tools_compare.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/tools/compare_tables.py tests/test_mcp_tools_compare.py
git commit -m "feat(mcp): add compare_tables tool"
```

---

## Task 13: 实现 MCP 工具 - explore_relations

**Files:**
- Create: `src/govio/mcp/tools/explore_relations.py`
- Create: `tests/test_mcp_tools_explore.py`

**Step 1: 编写测试**

```python
"""测试 explore_relations 工具"""
import pandas as pd
import pytest

from govio.mcp.core.dataframe_store import DataFrameStore
from govio.mcp.tools.explore_relations import explore_relations


@pytest.fixture
def store():
    return DataFrameStore()


def test_explore_relations_success(store):
    orders_df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer_id": [100, 101, 100]
    })
    customers_df = pd.DataFrame({
        "customer_id": [100, 101, 102],
        "name": ["Alice", "Bob", "Charlie"]
    })
    
    store.store("orders", orders_df)
    store.store("customers", customers_df)
    
    result = explore_relations(store, ["orders", "customers"])
    
    assert result["success"] is True
    assert len(result["relations"]) > 0


def test_explore_relations_all_dataframes(store):
    orders_df = pd.DataFrame({
        "order_id": [1, 2],
        "customer_id": [100, 101]
    })
    customers_df = pd.DataFrame({
        "customer_id": [100, 101],
        "name": ["Alice", "Bob"]
    })
    
    store.store("orders", orders_df)
    store.store("customers", customers_df)
    
    result = explore_relations(store)
    
    assert result["success"] is True
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_tools_explore.py -v`
Expected: FAIL

**Step 3: 实现工具**

```python
"""explore_relations 工具"""
from typing import Any

from govio.mcp.core.dataframe_store import DataFrameStore
from govio.mcp.core.explorer import RelationExplorer


def explore_relations(
    store: DataFrameStore,
    dataframes: list[str] | None = None
) -> dict[str, Any]:
    """探查 DataFrame 之间的关系
    
    Args:
        store: DataFrame 存储
        dataframes: DataFrame 名称列表，为空则探查全部
    
    Returns:
        关系列表
    """
    if dataframes is None:
        infos = store.list()
        dataframes = [info.name for info in infos]
    
    df_dict = {}
    for name in dataframes:
        df = store.get(name)
        if df is None:
            return {
                "success": False,
                "error": f"DataFrame '{name}' 不存在"
            }
        df_dict[name] = df
    
    explorer = RelationExplorer()
    relations = explorer.explore(df_dict)
    
    return {
        "success": True,
        "relations": relations
    }
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_tools_explore.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/tools/explore_relations.py tests/test_mcp_tools_explore.py
git commit -m "feat(mcp): add explore_relations tool"
```

---

## Task 14: 实现 MCP 工具 - visualize_relations

**Files:**
- Create: `src/govio/mcp/tools/visualize_relations.py`
- Create: `tests/test_mcp_tools_visualize.py`

**Step 1: 编写测试**

```python
"""测试 visualize_relations 工具"""
import pytest

from govio.mcp.tools.visualize_relations import visualize_relations


def test_visualize_relations_success():
    relations = [
        {
            "source_table": "orders",
            "source_column": "customer_id",
            "target_table": "customers",
            "target_column": "customer_id",
            "confidence": 0.9
        }
    ]
    
    result = visualize_relations(relations)
    
    assert result["success"] is True
    assert "nodes" in result
    assert "edges" in result
    assert len(result["nodes"]) == 2


def test_visualize_relations_empty():
    result = visualize_relations([])
    
    assert result["success"] is True
    assert result["nodes"] == []
    assert result["edges"] == []
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_tools_visualize.py -v`
Expected: FAIL

**Step 3: 实现工具**

```python
"""visualize_relations 工具"""
from typing import Any

from govio.mcp.core.visualizer import RelationVisualizer


def visualize_relations(
    relations: list[dict[str, Any]]
) -> dict[str, Any]:
    """生成关系图谱
    
    Args:
        relations: 关系列表
    
    Returns:
        可视化数据
    """
    visualizer = RelationVisualizer()
    result = visualizer.visualize(relations)
    
    return {
        "success": True,
        **result
    }
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_tools_visualize.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/tools/visualize_relations.py tests/test_mcp_tools_visualize.py
git commit -m "feat(mcp): add visualize_relations tool"
```

---

## Task 15: 实现 MCP 服务器

**Files:**
- Create: `src/govio/mcp/server.py`
- Create: `tests/test_mcp_server.py`

**Step 1: 编写测试**

```python
"""测试 MCP 服务器"""
import json
import tempfile
from pathlib import Path

from govio.mcp.config import load_config
from govio.mcp.server import create_server


def test_create_server():
    config_data = {
        "datasources": {
            "testdb": {
                "driver": "sqlite",
                "database": ":memory:"
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        f.flush()
        config = load_config(Path(f.name))
    
    server = create_server(config)
    
    assert server is not None
    assert server.name == "govio-data-exploration"
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: FAIL

**Step 3: 实现服务器**

```python
"""MCP 服务器入口"""
import argparse
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.streamable_http import streamable_http_server
from mcp.types import Tool, TextContent

from govio.mcp.config import Config, load_config
from govio.mcp.core.database import DatabaseManager
from govio.mcp.core.dataframe_store import DataFrameStore
from govio.mcp.tools.compare_tables import compare_tables
from govio.mcp.tools.explore_relations import explore_relations
from govio.mcp.tools.list_dataframes import list_dataframes
from govio.mcp.tools.load_dataframe import load_dataframe
from govio.mcp.tools.release_dataframe import release_dataframe
from govio.mcp.tools.visualize_relations import visualize_relations

logger = logging.getLogger(__name__)

_store = DataFrameStore()
_db_manager: DatabaseManager | None = None


def create_server(config: Config) -> Server:
    """创建 MCP 服务器"""
    global _db_manager
    _db_manager = DatabaseManager(config.datasources)
    
    server = Server("govio-data-exploration")
    
    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="load_dataframe",
                description="执行 SQL 并将结果存入内存 DataFrame",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "datasource": {"type": "string", "description": "数据源名称"},
                        "name": {"type": "string", "description": "DataFrame 名称"},
                        "sql": {"type": "string", "description": "查询 SQL"}
                    },
                    "required": ["datasource", "name", "sql"]
                }
            ),
            Tool(
                name="list_dataframes",
                description="列出已加载的 DataFrame 清单",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="release_dataframe",
                description="释放 DataFrame",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "DataFrame 名称"}
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="compare_tables",
                description="比对两个 DataFrame",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_df": {"type": "string", "description": "源 DataFrame 名称"},
                        "target_df": {"type": "string", "description": "目标 DataFrame 名称"},
                        "join_columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "用于比对的列"
                        }
                    },
                    "required": ["source_df", "target_df", "join_columns"]
                }
            ),
            Tool(
                name="explore_relations",
                description="探查 DataFrame 之间的关系",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataframes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "DataFrame 名称列表，为空则探查全部"
                        }
                    }
                }
            ),
            Tool(
                name="visualize_relations",
                description="生成关系图谱",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "relations": {
                            "type": "array",
                            "description": "关系列表"
                        }
                    },
                    "required": ["relations"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "load_dataframe":
            result = load_dataframe(
                store=_store,
                db_manager=_db_manager,
                **arguments
            )
        elif name == "list_dataframes":
            result = list_dataframes(store=_store)
        elif name == "release_dataframe":
            result = release_dataframe(store=_store, **arguments)
        elif name == "compare_tables":
            result = compare_tables(store=_store, **arguments)
        elif name == "explore_relations":
            result = explore_relations(store=_store, **arguments)
        elif name == "visualize_relations":
            result = visualize_relations(**arguments)
        else:
            result = {"error": f"未知工具: {name}"}
        
        return [TextContent(type="text", text=str(result))]
    
    return server


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="Govio MCP 数据探查服务")
    parser.add_argument(
        "--datasource-config",
        type=str,
        required=True,
        help="数据源配置文件路径"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务端口"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务地址"
    )
    
    args = parser.parse_args()
    
    config = load_config(Path(args.datasource_config))
    server = create_server(config)
    
    logging.basicConfig(level=logging.INFO)
    logger.info(f"启动 MCP 服务: http://{args.host}:{args.port}")
    
    streamable_http_server(server, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/govio/mcp/server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add mcp server with streamable http"
```

---

## Task 16: 运行完整测试套件

**Step 1: 运行所有测试**

Run: `uv run pytest tests/ -v`

**Step 2: 运行 lint 检查**

Run: `uv run ruff check src/govio/mcp/ tests/test_mcp_*.py`

**Step 3: 格式化代码**

Run: `uv run ruff format src/govio/mcp/ tests/test_mcp_*.py`

**Step 4: Commit（如有修改）**

```bash
git add -A
git commit -m "style(mcp): fix lint and format issues"
```

---

## Task 17: 更新包导出

**Files:**
- Modify: `src/govio/__init__.py`

**Step 1: 更新导出**

在 `src/govio/__init__.py` 中添加：

```python
from .mcp.server import create_server

__all__ = ['run', 'gml_generate', 'FalkorDBGraph', 'NetworkXGraph', 'create_server']
```

**Step 2: Commit**

```bash
git add src/govio/__init__.py
git commit -m "feat: export mcp server create function"
```

---

## 完成检查清单

- [ ] 所有测试通过
- [ ] Lint 检查通过
- [ ] 代码格式正确
- [ ] 文档已更新
- [ ] 依赖已添加到 pyproject.toml
- [ ] 脚本入口已添加