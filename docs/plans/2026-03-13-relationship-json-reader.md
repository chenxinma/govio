# 物理表关系JSON读取器实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 创建读取 schema_of_relationships.json 文件的程序，将其补充到本体模型图数据生成中。

**Architecture:** 创建 relationship.py 模块读取和验证 JSON 文件，修改 gen_networkx.py 支持新的边类型，修改 utility.py 集成关系生成流程。

**Tech Stack:** Python, pandas, networkx, json

---

### Task 1: 创建 relationship.py 模块基础结构

**Files:**
- Create: `src/govio/metadata/relationship.py`

**Step 1: 创建模块文件和基础导入**

创建 `src/govio/metadata/relationship.py`:

```python
"""
govio.metadata.relationship
读取 schema_of_relationships.json 文件，生成物理表之间的关系边数据
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


VALID_RELATIONSHIP_TYPES = {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}


class RelationshipLoader:
    """加载和验证表关系JSON文件"""
    
    def __init__(self, json_path: str, df_tables: pd.DataFrame, df_columns: pd.DataFrame):
        """
        Args:
            json_path: JSON文件路径
            df_tables: PhysicalTable DataFrame
            df_columns: Col DataFrame
        """
        self.json_path = Path(json_path)
        self.df_tables = df_tables
        self.df_columns = df_columns
        self._validate_inputs()
    
    def _validate_inputs(self):
        """验证输入参数"""
        if not self.json_path.exists():
            raise FileNotFoundError(f"关系文件不存在: {self.json_path}")
        
        if self.df_tables.empty:
            raise ValueError("PhysicalTable DataFrame 为空")
        
        if self.df_columns.empty:
            raise ValueError("Col DataFrame 为空")
```

**Step 2: 验证文件创建**

运行: `python -c "from govio.metadata.relationship import RelationshipLoader; print('OK')"`
预期: `OK`

**Step 3: 提交**

```bash
git add src/govio/metadata/relationship.py
git commit -m "feat: add relationship.py module structure"
```

---

### Task 2: 实现 JSON 加载和验证

**Files:**
- Modify: `src/govio/metadata/relationship.py`

**Step 1: 添加 JSON 加载方法**

在 `RelationshipLoader` 类中添加方法:

```python
    def load_json(self) -> dict[str, Any]:
        """加载JSON文件"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "version" not in data:
            raise ValueError("JSON 缺少 version 字段")
        
        if "relationships" not in data:
            raise ValueError("JSON 缺少 relationships 字段")
        
        return data
    
    def validate_relationship(self, rel: dict[str, Any], index: int) -> bool:
        """
        验证单个关系的有效性
        
        Args:
            rel: 关系字典
            index: 关系索引（用于错误消息）
        
        Returns:
            bool: 是否有效
        """
        required_fields = ["source", "target", "relationship_type"]
        for field in required_fields:
            if field not in rel:
                logger.warning(f"关系 {index} 缺少必需字段 '{field}'，跳过")
                return False
        
        if rel["relationship_type"] not in VALID_RELATIONSHIP_TYPES:
            logger.warning(
                f"关系 {index} 的 relationship_type '{rel['relationship_type']}' 无效，"
                f"有效值: {VALID_RELATIONSHIP_TYPES}，跳过"
            )
            return False
        
        if "PhysicalTable" not in rel["source"] or "Cols" not in rel["source"]:
            logger.warning(f"关系 {index} 的 source 缺少 PhysicalTable 或 Cols 字段，跳过")
            return False
        
        if "PhysicalTable" not in rel["target"] or "Cols" not in rel["target"]:
            logger.warning(f"关系 {index} 的 target 缺少 PhysicalTable 或 Cols 字段，跳过")
            return False
        
        return True
```

**Step 2: 验证模块可导入**

运行: `python -c "from govio.metadata.relationship import RelationshipLoader, VALID_RELATIONSHIP_TYPES; print(VALID_RELATIONSHIP_TYPES)"`
预期: `{'one_to_one', 'one_to_many', 'many_to_one', 'many_to_many'}`

**Step 3: 提交**

```bash
git add src/govio/metadata/relationship.py
git commit -m "feat: add JSON loading and validation methods"
```

---

### Task 3: 实现表和列验证

**Files:**
- Modify: `src/govio/metadata/relationship.py`

**Step 1: 添加表和列验证方法**

在 `RelationshipLoader` 类中添加方法:

```python
    def _validate_table_exists(self, table_name: str, context: str) -> bool:
        """
        验证表是否存在
        
        Args:
            table_name: 表名
            context: 上下文描述（用于错误消息）
        
        Returns:
            bool: 是否存在
        """
        table_names = self.df_tables.get("full_table_name", self.df_tables.get("name", pd.Series()))
        
        if table_name not in table_names.values:
            logger.warning(f"{context}: 表 '{table_name}' 不存在于 PhysicalTable 数据中，跳过")
            return False
        
        return True
    
    def _validate_column_exists(self, table_name: str, column_name: str, context: str) -> bool:
        """
        验证列是否存在
        
        Args:
            table_name: 表名
            column_name: 列名
            context: 上下文描述（用于错误消息）
        
        Returns:
            bool: 是否存在
        """
        full_col_name = f"{table_name}.{column_name}"
        
        if "column" in self.df_columns.columns:
            col_names = self.df_columns["column"]
        elif "full_column_name" in self.df_columns.columns:
            col_names = self.df_columns["full_column_name"]
        else:
            logger.warning(f"{context}: 列 '{full_col_name}' 无法验证，DataFrame 缺少列名列，跳过")
            return False
        
        if full_col_name not in col_names.values:
            logger.warning(f"{context}: 列 '{full_col_name}' 不存在于 Col 数据中，跳过")
            return False
        
        return True
    
    def validate_table_and_columns(self, rel: dict[str, Any], index: int) -> bool:
        """
        验证关系中的表和列是否存在
        
        Args:
            rel: 关系字典
            index: 关系索引
        
        Returns:
            bool: 是否有效
        """
        source_table = rel["source"]["PhysicalTable"]
        target_table = rel["target"]["PhysicalTable"]
        
        if not self._validate_table_exists(source_table, f"关系 {index} source"):
            return False
        
        if not self._validate_table_exists(target_table, f"关系 {index} target"):
            return False
        
        for col in rel["source"]["Cols"]:
            if not self._validate_column_exists(source_table, col, f"关系 {index} source"):
                return False
        
        for col in rel["target"]["Cols"]:
            if not self._validate_column_exists(target_table, col, f"关系 {index} target"):
                return False
        
        return True
```

**Step 2: 验证代码语法**

运行: `python -m py_compile src/govio/metadata/relationship.py`
预期: 无输出（编译成功）

**Step 3: 提交**

```bash
git add src/govio/metadata/relationship.py
git commit -m "feat: add table and column validation methods"
```

---

### Task 4: 实现关系数据转换

**Files:**
- Modify: `src/govio/metadata/relationship.py`

**Step 1: 添加关系转换方法**

在 `RelationshipLoader` 类中添加方法:

```python
    def _convert_to_edge_row(self, rel: dict[str, Any]) -> dict[str, Any]:
        """
        将单个关系转换为边数据行
        
        Args:
            rel: 关系字典
        
        Returns:
            dict: 边数据行
        """
        return {
            "source": rel["source"]["PhysicalTable"],
            "target": rel["target"]["PhysicalTable"],
            "relationship_type": rel["relationship_type"],
            "description": rel.get("description", ""),
            "source_columns": ",".join(rel["source"]["Cols"]),
            "target_columns": ",".join(rel["target"]["Cols"]),
        }
    
    def load_relationships(self) -> pd.DataFrame:
        """
        加载并验证所有关系，返回边数据 DataFrame
        
        Returns:
            pd.DataFrame: 包含边数据的 DataFrame
        """
        data = self.load_json()
        relationships = data["relationships"]
        
        edge_rows = []
        for idx, rel in enumerate(relationships):
            if not self.validate_relationship(rel, idx):
                continue
            
            if not self.validate_table_and_columns(rel, idx):
                continue
            
            edge_row = self._convert_to_edge_row(rel)
            edge_rows.append(edge_row)
        
        if not edge_rows:
            logger.warning("没有有效的关系数据")
            return pd.DataFrame(columns=[
                "source", "target", "relationship_type",
                "description", "source_columns", "target_columns"
            ])
        
        df = pd.DataFrame(edge_rows)
        logger.info(f"成功加载 {len(df)} 个表关系")
        
        return df


def load_relationships(json_path: str, df_tables: pd.DataFrame, df_columns: pd.DataFrame) -> pd.DataFrame:
    """
    便捷函数：加载关系JSON并返回边数据
    
    Args:
        json_path: JSON文件路径
        df_tables: PhysicalTable DataFrame
        df_columns: Col DataFrame
    
    Returns:
        pd.DataFrame: 边数据 DataFrame
    """
    loader = RelationshipLoader(json_path, df_tables, df_columns)
    return loader.load_relationships()
```

**Step 2: 验证完整模块**

运行: `python -c "from govio.metadata.relationship import load_relationships; print('OK')"`
预期: `OK`

**Step 3: 提交**

```bash
git add src/govio/metadata/relationship.py
git commit -m "feat: add relationship conversion and main loader function"
```

---

### Task 5: 修改 gen_networkx.py 支持新边类型

**Files:**
- Modify: `src/govio/metadata/gen_networkx.py:58`

**Step 1: 添加 RELATES_TO.csv 到边文件列表**

修改 `src/govio/metadata/gen_networkx.py` 第58行:

```python
def load_edges(csv_dir: str) -> pd.DataFrame:
    edge_files = ["HAS_COLUMN.csv", "USE.csv", "COMPLIES_WITH.csv", "RELATES_TO.csv"]
    edges_list = []
    for filename in edge_files:
        filepath = Path(csv_dir) / filename
        if not filepath.exists():
            continue
        df = pd.read_csv(filepath)
        src_col = df.columns[0]
        dst_col = df.columns[1]
        src_match = re.match(r":START_ID\((\w+)\)", src_col)
        dst_match = re.match(r":END_ID\((\w+)\)", dst_col)
        if not src_match or not dst_match:
            continue
        edge_type = Path(filename).stem
        df = df.rename(columns={src_col: "source", dst_col: "target"})
        df["edge_type"] = edge_type
        edges_list.append(df)
    if not edges_list:
        return pd.DataFrame(columns=["source", "target", "edge_type"])
    return pd.concat(edges_list)
```

**Step 2: 验证修改**

运行: `python -m py_compile src/govio/metadata/gen_networkx.py`
预期: 无输出（编译成功）

**Step 3: 提交**

```bash
git add src/govio/metadata/gen_networkx.py
git commit -m "feat: add RELATES_TO.csv support to gen_networkx"
```

---

### Task 6: 修改 utility.py 集成关系生成

**Files:**
- Modify: `src/govio/metadata/utility.py:90`

**Step 1: 添加 relationship 模块导入**

在 `src/govio/metadata/utility.py` 顶部添加导入（约第13行后）:

```python
from .relationship import load_relationships
```

**Step 2: 修改 make_csv 函数签名**

修改 `make_csv` 函数（约第90行）:

```python
def make_csv(output:Path, db:str, workspace_uuid:str, app_list_file: str, df_app_db_map: pd.DataFrame, relationship_file: str | None = None):
    db_loader = DatabaseLoader(db, workspace_uuid, df_app_db_map["schema"].to_list())
    app_loader = AppInfoLoader(app_list_file, df_app_db_map["name"].to_list())
    std_loader = StandardLoader(db, workspace_uuid)

    df_tables = db_loader.PhysicalTable
    df_columns = db_loader.Col
    df_apps = app_loader.Application
    df_stds = std_loader.Standard

    reorder_index([df_tables, df_columns, df_apps, df_stds])

    files = []

    df_tables.to_csv(output / "PhysicalTable.csv", index_label=":ID(PhysicalTable)")
    files.append("-n " + str(output/ "PhysicalTable.csv"))

    df_columns.to_csv(output / "Col.csv", index_label=":ID(Col)")
    files.append("-n " + str(output/ "Col.csv"))

    df_apps.to_csv(output / "Application.csv", index_label=":ID(Application)")
    files.append("-n " + str(output/ "Application.csv"))

    df_stds.to_csv(output / "Standard.csv", index_label=":ID(Standard)")
    files.append("-n " + str(output/ "Standard.csv"))

    df_has_column = pd.merge(
        df_tables[["full_table_name"]].reset_index().rename(columns={"index":":START_ID(PhysicalTable)"}), 
        df_columns[["full_table_name"]].reset_index().rename(columns={"index":":END_ID(Col)"}), 
        on="full_table_name", 
        how="inner") [[":START_ID(PhysicalTable)", ":END_ID(Col)"]]
    df_has_column.to_csv(output / "HAS_COLUMN.csv", index=False)
    files.append("-r " + str(output/ "HAS_COLUMN.csv"))

    df_app_table = pd.merge(df_app_db_map, 
                      df_tables[["schema"]].reset_index().rename(columns={"index": ":END_ID(PhysicalTable)"}), 
                    on="schema", how="inner")
    df_use = pd.merge(df_apps[["name"]].reset_index().rename(columns={"index": ":START_ID(Applicatin)"}),
                      df_app_table,
                      on="name", how="inner")[[":START_ID(Applicatin)", ":END_ID(PhysicalTable)"]]
    
    df_use.to_csv(output / "USE.csv", index=False)
    files.append("-r " + str(output/ "USE.csv"))

    if relationship_file:
        try:
            df_relates_to = load_relationships(relationship_file, df_tables, df_columns)
            df_relates_to.to_csv(output / "RELATES_TO.csv", index=False, header=[
                ":START_ID(PhysicalTable)", ":END_ID(PhysicalTable)",
                "relationship_type", "description", "source_columns", "target_columns"
            ])
            files.append("-r " + str(output / "RELATES_TO.csv"))
            print(f"成功生成 RELATES_TO.csv，包含 {len(df_relates_to)} 个关系")
        except Exception as e:
            print(f"警告: 无法加载关系文件: {e}")

    s = f"falkordb-bulk-insert {GRAPH} {"  ".join(files)}"
    print("Bulk insert usage:")
    print(s)
```

**Step 3: 修改 run 函数添加命令行参数**

修改 `run` 函数（约第33行）:

```python
def run():
    """
    1.从数据治理平台和应用清单获取基础元数据生成CSV
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(f'''\
        从元数据管理的数据库中提取元数据信息生成用于沟通图数据库的csv。
        从应用清单中获得应用信息生成用于沟通图数据库的csv。
        ''')
    )
    parser.add_argument('--kundb', type=str, help='元数据库URL')
    parser.add_argument('--app-list', type=str, help="应用清单")
    parser.add_argument('--app-map', type=str, help="应用数据库映射")
    parser.add_argument('--relationship', type=str, help="表关系JSON文件路径")
    parser.add_argument('-m', "--mode", type=Mode, choices=list(Mode), default=Mode.CSV)
    parser.add_argument('-o', '--output', type=str, default=".", help="输出目录")
    # 解析命令行参数
    args = parser.parse_args()

    load_dotenv()
    db = os.getenv("KUNDB_URL", "")
    if args.kundb:
        db = args.kundb
    
    app_list = os.getenv("APP_LIST_FILE", "")
    if args.app_list:
        app_list = args.app_list

    app_map = os.getenv("APP_MAP", "")
    if args.app_map:
        app_map = args.app_map
    
    relationship_file = args.relationship

    workspace_uuid = '82ee37374b314a938bf28170ab4db7cf'

    if len(db) == 0:
        print("元数据管理库未设置")
        sys.exit()
    
    if not os.path.exists(args.output):
        print("输出目录未找到")
        sys.exit()

    if not os.path.exists(app_map):
        print("应用和数据库映射未找到")
        sys.exit()
    
    output = Path(args.output)

    df_app_db_map = pd.read_json(app_map, orient='records')
    
    if args.mode == Mode.CSV:
        make_csv(output, db, workspace_uuid, app_list, df_app_db_map, relationship_file)
    elif args.mode == Mode.RECOMMEND:
        data_standard_recommend(output, db, workspace_uuid, df_app_db_map)
```

**Step 4: 验证修改**

运行: `python -m py_compile src/govio/metadata/utility.py`
预期: 无输出（编译成功）

**Step 5: 提交**

```bash
git add src/govio/metadata/utility.py
git commit -m "feat: integrate relationship generation into utility.py"
```

---

### Task 7: 创建测试数据并测试

**Files:**
- Create: `tests/test_relationship.py`

**Step 1: 创建测试目录结构**

运行: `mkdir -p tests`
预期: 无输出（目录创建成功）

**Step 2: 创建测试文件**

创建 `tests/test_relationship.py`:

```python
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from govio.metadata.relationship import RelationshipLoader, load_relationships


@pytest.fixture
def sample_tables():
    return pd.DataFrame({
        "full_table_name": ["db.schema.table1", "db.schema.table2"],
        "name": ["table1", "table2"],
        "schema": ["schema", "schema"]
    })


@pytest.fixture
def sample_columns():
    return pd.DataFrame({
        "column": [
            "db.schema.table1.col1",
            "db.schema.table1.col2",
            "db.schema.table2.col_a",
            "db.schema.table2.col_b"
        ]
    })


@pytest.fixture
def valid_json_data():
    return {
        "version": "1.0",
        "relationships": [
            {
                "description": "外键关联",
                "source": {
                    "PhysicalTable": "db.schema.table1",
                    "Cols": ["col1"]
                },
                "target": {
                    "PhysicalTable": "db.schema.table2",
                    "Cols": ["col_a"]
                },
                "relationship_type": "many_to_one"
            }
        ]
    }


def test_load_json_success(sample_tables, sample_columns, valid_json_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(valid_json_data, f)
        f.flush()
        
        loader = RelationshipLoader(f.name, sample_tables, sample_columns)
        data = loader.load_json()
        
        assert data["version"] == "1.0"
        assert len(data["relationships"]) == 1


def test_validate_relationship_valid(sample_tables, sample_columns):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"version": "1.0", "relationships": []}, f)
        f.flush()
        
        loader = RelationshipLoader(f.name, sample_tables, sample_columns)
        
        rel = {
            "source": {"PhysicalTable": "t1", "Cols": ["c1"]},
            "target": {"PhysicalTable": "t2", "Cols": ["c2"]},
            "relationship_type": "many_to_one"
        }
        
        assert loader.validate_relationship(rel, 0) is True


def test_validate_relationship_invalid_type(sample_tables, sample_columns):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"version": "1.0", "relationships": []}, f)
        f.flush()
        
        loader = RelationshipLoader(f.name, sample_tables, sample_columns)
        
        rel = {
            "source": {"PhysicalTable": "t1", "Cols": ["c1"]},
            "target": {"PhysicalTable": "t2", "Cols": ["c2"]},
            "relationship_type": "invalid_type"
        }
        
        assert loader.validate_relationship(rel, 0) is False


def test_load_relationships_success(sample_tables, sample_columns, valid_json_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(valid_json_data, f)
        f.flush()
        
        df = load_relationships(f.name, sample_tables, sample_columns)
        
        assert len(df) == 1
        assert df.iloc[0]["source"] == "db.schema.table1"
        assert df.iloc[0]["target"] == "db.schema.table2"
        assert df.iloc[0]["relationship_type"] == "many_to_one"
        assert df.iloc[0]["source_columns"] == "col1"
        assert df.iloc[0]["target_columns"] == "col_a"


def test_load_relationships_invalid_table(sample_tables, sample_columns):
    data = {
        "version": "1.0",
        "relationships": [
            {
                "source": {"PhysicalTable": "nonexistent", "Cols": ["c1"]},
                "target": {"PhysicalTable": "db.schema.table2", "Cols": ["c2"]},
                "relationship_type": "many_to_one"
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        f.flush()
        
        df = load_relationships(f.name, sample_tables, sample_columns)
        
        assert len(df) == 0


def test_composite_key_relationship(sample_tables, sample_columns):
    data = {
        "version": "1.0",
        "relationships": [
            {
                "description": "复合键关联",
                "source": {
                    "PhysicalTable": "db.schema.table1",
                    "Cols": ["col1", "col2"]
                },
                "target": {
                    "PhysicalTable": "db.schema.table2",
                    "Cols": ["col_a", "col_b"]
                },
                "relationship_type": "many_to_many"
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        f.flush()
        
        df = load_relationships(f.name, sample_tables, sample_columns)
        
        assert len(df) == 1
        assert df.iloc[0]["source_columns"] == "col1,col2"
        assert df.iloc[0]["target_columns"] == "col_a,col_b"
```

**Step 3: 运行测试**

运行: `pytest tests/test_relationship.py -v`
预期: 所有测试通过

**Step 4: 提交**

```bash
git add tests/test_relationship.py
git commit -m "test: add comprehensive tests for relationship module"
```

---

### Task 8: 创建示例JSON文件

**Files:**
- Create: `data/example_relationships.json`

**Step 1: 创建示例文件**

创建 `data/example_relationships.json`:

```json
{
  "version": "1.0",
  "relationships": [
    {
      "description": "用户表关联部门表",
      "source": {
        "PhysicalTable": "db.schema.users",
        "Cols": ["dept_id"]
      },
      "target": {
        "PhysicalTable": "db.schema.departments",
        "Cols": ["id"]
      },
      "relationship_type": "many_to_one"
    },
    {
      "description": "订单明细关联订单主表",
      "source": {
        "PhysicalTable": "db.schema.order_items",
        "Cols": ["order_id"]
      },
      "target": {
        "PhysicalTable": "db.schema.orders",
        "Cols": ["id"]
      },
      "relationship_type": "many_to_one"
    },
    {
      "description": "订单与商品多对多关系",
      "source": {
        "PhysicalTable": "db.schema.orders",
        "Cols": ["id"]
      },
      "target": {
        "PhysicalTable": "db.schema.products",
        "Cols": ["id"]
      },
      "relationship_type": "many_to_many"
    }
  ]
}
```

**Step 2: 提交**

```bash
git add data/example_relationships.json
git commit -m "docs: add example relationships JSON file"
```

---

### Task 9: 更新 __init__.py

**Files:**
- Modify: `src/govio/metadata/__init__.py`

**Step 1: 导出新模块**

修改 `src/govio/metadata/__init__.py`:

```python
from .relationship import RelationshipLoader, load_relationships

__all__ = [
    "RelationshipLoader",
    "load_relationships",
]
```

**Step 2: 验证导入**

运行: `python -c "from govio.metadata import load_relationships; print('OK')"`
预期: `OK`

**Step 3: 提交**

```bash
git add src/govio/metadata/__init__.py
git commit -m "feat: export relationship module in __init__.py"
```

---

### Task 10: 最终集成测试

**Files:**
- 无文件修改，运行完整流程测试

**Step 1: 运行所有测试**

运行: `pytest tests/ -v`
预期: 所有测试通过

**Step 2: 验证命令行帮助**

运行: `python -m govio.metadata.utility --help`
预期: 显示帮助信息，包含 `--relationship` 参数

**Step 3: 验证 gen_networkx 帮助**

运行: `python -m govio.metadata.gen_networkx --help`
预期: 显示帮助信息

**Step 4: 最终提交（如有修改）**

```bash
git status
# 如有未提交文件，提交它们
git add .
git commit -m "chore: final integration cleanup"
```

---

## 验收标准

- [ ] `relationship.py` 模块可以成功加载和验证 JSON 文件
- [ ] 支持 single key 和 composite key（复合键）
- [ ] 有效的表名和列名验证
- [ ] 无效关系被跳过并记录警告
- [ ] `gen_networkx.py` 支持 RELATES_TO.csv 边类型
- [ ] `utility.py` 集成关系生成功能
- [ ] 命令行参数 `--relationship` 可用
- [ ] 所有测试通过
- [ ] 示例 JSON 文件可用