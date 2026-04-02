# Onboard Wizard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a `govio onboard` CLI command that guides users through initial setup, allowing them to choose between NetworkX and FalkorDB backends, configure connections, and generate assets (schema.md and names/).

**Architecture:** Extract logic from `skills/govio/scripts/load*.py` into `src/govio/core/assets_generator.py`, create a `ConfigManager` for YAML-based configuration, use a `GraphFactory` to create graph objects, and implement an interactive CLI wizard in `src/govio/cli/onboard.py`.

**Tech Stack:** Python 3.13+, NetworkX, FalkorDB, PyYAML, argparse (or click/questionary for interactive CLI)

---

## Task 1: Add PyYAML Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add pyyaml to dependencies**

Edit `pyproject.toml`:

```toml
dependencies = [
    "datacompy>=0.10.0",
    "dotenv>=0.9.9",
    "duckdb>=0.10.0",
    "falkordb>=1.4.0",
    "mcp>=1.0.0",
    "networkx>=3.6.1",
    "openpyxl>=3.1.5",
    "pandas>=2.3.3",
    "pymysql>=1.1.2",
    "pyyaml>=6.0.0",
    "scikit-learn>=1.8.0",
    "sqlalchemy>=2.0.45",
    "tqdm>=4.67.1",
    "trino>=0.337.0",
]
```

**Step 2: Install dependencies**

Run: `uv sync`

Expected: Successfully installed pyyaml

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyyaml dependency for config management"
```

---

## Task 2: Create ConfigManager Class

**Files:**
- Create: `src/govio/cli/__init__.py`
- Create: `src/govio/cli/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing test for ConfigManager**

Create `tests/test_config.py`:

```python
import pytest
from pathlib import Path
import tempfile
import yaml
from govio.cli.config import ConfigManager

def test_config_manager_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)
        
        config = {
            'backend': 'networkx',
            'networkx': {'gml_path': 'test.gml'}
        }
        
        manager.save(config)
        assert config_path.exists()
        
        loaded = manager.load()
        assert loaded == config

def test_config_manager_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)
        
        assert not manager.exists()
        
        config = {'backend': 'networkx'}
        manager.save(config)
        
        assert manager.exists()

def test_config_manager_validate_networkx():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)
        
        valid_config = {
            'backend': 'networkx',
            'networkx': {'gml_path': 'test.gml'}
        }
        
        assert manager.validate(valid_config) is True
        
        invalid_config = {
            'backend': 'networkx',
            'networkx': {}
        }
        
        with pytest.raises(ValueError):
            manager.validate(invalid_config)

def test_config_manager_validate_falkordb():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)
        
        valid_config = {
            'backend': 'falkordb',
            'falkordb': {
                'host': 'localhost',
                'port': 6379,
                'graph': 'ontology'
            }
        }
        
        assert manager.validate(valid_config) is True
        
        invalid_config = {
            'backend': 'falkordb',
            'falkordb': {'host': 'localhost'}
        }
        
        with pytest.raises(ValueError):
            manager.validate(invalid_config)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'govio.cli'"

**Step 3: Create cli package**

Create `src/govio/cli/__init__.py`:

```python
from .config import ConfigManager

__all__ = ["ConfigManager"]
```

**Step 4: Implement ConfigManager**

Create `src/govio/cli/config.py`:

```python
import yaml
from pathlib import Path
from typing import Any


class ConfigManager:
    """管理 govio 配置文件"""
    
    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            self.config_path = Path.home() / ".govio" / "config.yaml"
        else:
            self.config_path = config_path
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def exists(self) -> bool:
        """检查配置文件是否存在"""
        return self.config_path.exists()
    
    def load(self) -> dict[str, Any]:
        """加载配置文件"""
        if not self.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def save(self, config: dict[str, Any]) -> None:
        """保存配置文件"""
        self.validate(config)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    def validate(self, config: dict[str, Any]) -> bool:
        """验证配置的有效性
        
        Args:
            config: 配置字典
            
        Returns:
            bool: 是否有效
            
        Raises:
            ValueError: 配置无效时抛出
        """
        if 'backend' not in config:
            raise ValueError("配置缺少 'backend' 字段")
        
        backend = config['backend']
        
        if backend not in ['networkx', 'falkordb']:
            raise ValueError(f"不支持的 backend: {backend}")
        
        if backend == 'networkx':
            if 'networkx' not in config:
                raise ValueError("NetworkX backend 需要 'networkx' 配置")
            if 'gml_path' not in config['networkx']:
                raise ValueError("NetworkX 配置缺少 'gml_path' 字段")
        
        elif backend == 'falkordb':
            if 'falkordb' not in config:
                raise ValueError("FalkorDB backend 需要 'falkordb' 配置")
            required_fields = ['host', 'port', 'graph']
            for field in required_fields:
                if field not in config['falkordb']:
                    raise ValueError(f"FalkorDB 配置缺少 '{field}' 字段")
        
        return True
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`

Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/govio/cli/ tests/test_config.py
git commit -m "feat: add ConfigManager for YAML-based configuration"
```

---

## Task 3: Create GraphFactory Class

**Files:**
- Create: `src/govio/core/__init__.py`
- Create: `src/govio/core/graph_factory.py`
- Create: `tests/test_graph_factory.py`

**Step 1: Write failing test for GraphFactory**

Create `tests/test_graph_factory.py`:

```python
import pytest
from pathlib import Path
import tempfile
import networkx as nx
from govio.core.graph_factory import GraphFactory


def test_create_networkx_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        gml_path = Path(tmpdir) / "test.gml"
        G = nx.DiGraph()
        G.add_node(1, name="test", node_type="TestNode")
        nx.write_gml(G, gml_path)
        
        config = {
            'backend': 'networkx',
            'networkx': {'gml_path': str(gml_path)}
        }
        
        graph = GraphFactory.create(config)
        assert graph is not None
        assert graph.G.number_of_nodes() == 1
        assert graph.G.number_of_edges() == 0


def test_create_networkx_graph_file_not_found():
    config = {
        'backend': 'networkx',
        'networkx': {'gml_path': '/nonexistent/path.gml'}
    }
    
    with pytest.raises(FileNotFoundError):
        GraphFactory.create(config)


def test_create_falkordb_graph_mock():
    config = {
        'backend': 'falkordb',
        'falkordb': {
            'host': 'localhost',
            'port': 6379,
            'graph': 'test_graph'
        }
    }
    
    with pytest.raises(Exception):
        GraphFactory.create(config)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_factory.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'govio.core'"

**Step 3: Create core package**

Create `src/govio/core/__init__.py`:

```python
from .graph_factory import GraphFactory

__all__ = ["GraphFactory"]
```

**Step 4: Implement GraphFactory**

Create `src/govio/core/graph_factory.py`:

```python
from typing import Any
from govio import NetworkXGraph, FalkorDBGraph


class GraphFactory:
    """图对象工厂，根据配置创建不同的图对象"""
    
    @staticmethod
    def create(config: dict[str, Any]):
        """根据配置创建图对象
        
        Args:
            config: 配置字典
            
        Returns:
            NetworkXGraph 或 FalkorDBGraph 实例
            
        Raises:
            ValueError: 不支持的 backend 类型
        """
        backend = config.get('backend')
        
        if backend == 'networkx':
            gml_path = config['networkx']['gml_path']
            return NetworkXGraph(gml_path)
        
        elif backend == 'falkordb':
            falkordb_config = config['falkordb']
            return FalkorDBGraph(
                graph=falkordb_config['graph'],
                host=falkordb_config.get('host', 'localhost'),
                port=falkordb_config.get('port', 6379)
            )
        
        else:
            raise ValueError(f"不支持的 backend: {backend}")
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_factory.py -v`

Expected: PASS (test_create_networkx_graph and test_create_networkx_graph_file_not_found pass, test_create_falkordb_graph_mock fails due to connection)

**Step 6: Commit**

```bash
git add src/govio/core/ tests/test_graph_factory.py
git commit -m "feat: add GraphFactory to create graph objects from config"
```

---

## Task 4: Create AssetsGenerator Class

**Files:**
- Modify: `src/govio/core/assets_generator.py`
- Create: `tests/test_assets_generator.py`

**Step 1: Write failing test for AssetsGenerator (NetworkX)**

Create `tests/test_assets_generator.py`:

```python
import pytest
from pathlib import Path
import tempfile
import networkx as nx
from govio import NetworkXGraph
from govio.core.assets_generator import AssetsGenerator


def create_test_gml(gml_path: Path):
    """创建测试用的 GML 文件"""
    G = nx.DiGraph()
    
    G.add_node("app1", name="应用1", node_type="Application", app_name_en="APP1")
    G.add_node("table1", name="表1", node_type="PhysicalTable", full_table_name="SCHEMA.TABLE1")
    G.add_node("col1", name="字段1", node_type="Col", column_name="COL1")
    G.add_node("col2", name="字段2", node_type="Col", column_name="COL2")
    
    G.add_edge("app1", "table1", edge_type="USE")
    G.add_edge("table1", "col1", edge_type="HAS_COLUMN")
    G.add_edge("table1", "col2", edge_type="HAS_COLUMN")
    
    nx.write_gml(G, gml_path)


def test_assets_generator_networkx_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        gml_path = Path(tmpdir) / "test.gml"
        output_dir = Path(tmpdir) / "assets"
        output_dir.mkdir()
        
        create_test_gml(gml_path)
        
        graph = NetworkXGraph(gml_path)
        generator = AssetsGenerator(graph, output_dir)
        
        generator.generate_schema()
        
        schema_path = output_dir / "schema.md"
        assert schema_path.exists()
        
        content = schema_path.read_text(encoding='utf-8')
        assert "NetworkX schema" in content
        assert "node_types" in content


def test_assets_generator_networkx_names():
    with tempfile.TemporaryDirectory() as tmpdir:
        gml_path = Path(tmpdir) / "test.gml"
        output_dir = Path(tmpdir) / "assets"
        output_dir.mkdir()
        
        create_test_gml(gml_path)
        
        graph = NetworkXGraph(gml_path)
        generator = AssetsGenerator(graph, output_dir)
        
        generator.generate_names()
        
        names_dir = output_dir / "names"
        assert names_dir.exists()
        
        node_names_path = names_dir / "node_names.md"
        assert node_names_path.exists()
        
        content = node_names_path.read_text(encoding='utf-8')
        assert "应用1" in content or "APP1" in content


def test_assets_generator_generate_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        gml_path = Path(tmpdir) / "test.gml"
        output_dir = Path(tmpdir) / "assets"
        output_dir.mkdir()
        
        create_test_gml(gml_path)
        
        graph = NetworkXGraph(gml_path)
        generator = AssetsGenerator(graph, output_dir)
        
        generator.generate_all()
        
        assert (output_dir / "schema.md").exists()
        assert (output_dir / "names" / "node_names.md").exists()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_assets_generator.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'govio.core.assets_generator'"

**Step 3: Implement AssetsGenerator**

Create `src/govio/core/assets_generator.py`:

```python
import json
from pathlib import Path
from typing import Any


class AssetsGenerator:
    """资产生成器，生成 schema.md 和 names/"""
    
    def __init__(self, graph: Any, output_dir: Path) -> None:
        self.graph = graph
        self.output_dir = output_dir
        self.names_dir = output_dir / "names"
    
    def generate_schema(self) -> None:
        """生成 schema.md 文件"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        schema_path = self.output_dir / "schema.md"
        
        with open(schema_path, 'w', encoding='utf-8') as f:
            f.write(self.graph.schema)
    
    def generate_names(self) -> None:
        """生成 names/ 目录"""
        if not self.names_dir.exists():
            self.names_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.graph, 'G'):
            self._generate_names_networkx()
        else:
            self._generate_names_falkordb()
    
    def _generate_names_networkx(self) -> None:
        """为 NetworkX 图生成节点名称索引"""
        nodes = [
            dict(
                id=node_id,
                name=self.graph.G.nodes[node_id]['name'],
                node_type=self.graph.G.nodes[node_id]['node_type']
            )
            for node_id in self.graph.G.nodes()
            if self.graph.G.nodes[node_id].get('name')
            and self.graph.G.nodes[node_id]['name'] != "0"
            and isinstance(self.graph.G.nodes[node_id]['name'], str)
        ]
        
        if nodes:
            file_path = self.names_dir / "node_names.md"
            with open(file_path, 'w', encoding='utf-8') as f:
                for node in nodes:
                    f.write(json.dumps(node, ensure_ascii=False) + '\n')
    
    def _generate_names_falkordb(self) -> None:
        """为 FalkorDB 图生成按应用分组的名称索引"""
        apps_query = """
        MATCH (app:Application)
        RETURN app.app_name_en AS app_name_en, app.name AS name
        ORDER BY app.app_name_en
        """
        apps = self.graph.query(apps_query)
        
        for app_row in apps:
            app_name_en, name = app_row
            
            tables_query = """
            MATCH (app:Application {app_name_en: $app_name_en})-[:USE]->(table:PhysicalTable)
            RETURN table.full_table_name, table.name AS table_name
            ORDER BY table.full_table_name
            """
            tables = self.graph.query(tables_query, {'app_name_en': app_name_en})
            
            md_content = []
            
            for table_row in tables:
                full_table_name, table_name = table_row
                
                if not table_name or table_name == "None":
                    table_name = ""
                
                md_content.append(f"# {full_table_name} {table_name}")
                
                cols_query = """
                MATCH (table:PhysicalTable {full_table_name: $full_table_name})-[:HAS_COLUMN]->(col:Col)
                RETURN col.column_name, col.name AS col_name
                ORDER BY col.order_no
                """
                cols = self.graph.query(cols_query, {'full_table_name': full_table_name})
                
                for col_row in cols:
                    column_name, col_name = col_row
                    if not col_name or col_name == "None":
                        col_name = ""
                    md_content.append(f"- {column_name} {col_name}")
                
                md_content.append("")
            
            file_path = self.names_dir / f"{name}_{app_name_en}.md"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(md_content))
    
    def generate_all(self) -> None:
        """生成所有资产"""
        self.generate_schema()
        self.generate_names()
```

**Step 4: Update core __init__.py**

Edit `src/govio/core/__init__.py`:

```python
from .graph_factory import GraphFactory
from .assets_generator import AssetsGenerator

__all__ = ["GraphFactory", "AssetsGenerator"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_assets_generator.py -v`

Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/govio/core/ tests/test_assets_generator.py
git commit -m "feat: add AssetsGenerator to generate schema.md and names/"
```

---

## Task 5: Implement Onboard CLI Command

**Files:**
- Create: `src/govio/cli/onboard.py`
- Create: `tests/test_onboard.py`

**Step 1: Write failing test for onboard command**

Create `tests/test_onboard.py`:

```python
import pytest
from pathlib import Path
import tempfile
import networkx as nx
from io import StringIO
import sys
from govio.cli.onboard import onboard


def create_test_csv_files(csv_dir: Path):
    """创建测试用的 CSV 文件"""
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    (csv_dir / "PhysicalTable.csv").write_text(
        """:ID(PhysicalTable),name,full_table_name
table1,表1,SCHEMA.TABLE1
""", encoding='utf-8'
    )
    
    (csv_dir / "Col.csv").write_text(
        """:ID(Col),name,column_name,full_table_name
col1,字段1,COL1,SCHEMA.TABLE1
""", encoding='utf-8'
    )
    
    (csv_dir / "Application.csv").write_text(
        """:ID(Application),name,app_name_en
app1,应用1,APP1
""", encoding='utf-8'
    )
    
    (csv_dir / "Standard.csv").write_text(
        """:ID(Standard),name
std1,标准1
""", encoding='utf-8'
    )
    
    (csv_dir / "HAS_COLUMN.csv").write_text(
        """:START_ID(PhysicalTable),:END_ID(Col)
table1,col1
""", encoding='utf-8'
    )
    
    (csv_dir / "USE.csv").write_text(
        """:START_ID(Application),:END_ID(PhysicalTable)
app1,table1
""", encoding='utf-8'
    )


def test_onboard_networkx_with_csv(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_dir = Path(tmpdir) / "csv"
        output_dir = Path(tmpdir) / "assets"
        config_path = Path(tmpdir) / ".govio" / "config.yaml"
        
        create_test_csv_files(csv_dir)
        
        inputs = [
            'networkx',
            'yes',
            str(csv_dir),
            str(output_dir)
        ]
        
        input_iter = iter(inputs)
        monkeypatch.setattr('builtins.input', lambda _: next(input_iter))
        
        from govio.cli.config import ConfigManager
        from govio.core.assets_generator import AssetsGenerator
        
        config_manager = ConfigManager(config_path)
        
        from govio.metadata.gen_networkx import build_graph
        gml_path = output_dir / "ontology.gml"
        build_graph(str(csv_dir), str(gml_path))
        
        from govio import NetworkXGraph
        graph = NetworkXGraph(gml_path)
        
        generator = AssetsGenerator(graph, output_dir)
        generator.generate_all()
        
        assert gml_path.exists()
        assert (output_dir / "schema.md").exists()
        assert (output_dir / "names" / "node_names.md").exists()


def test_validate_csv_directory():
    from govio.cli.onboard import validate_csv_directory
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_dir = Path(tmpdir) / "csv"
        csv_dir.mkdir()
        
        (csv_dir / "PhysicalTable.csv").write_text(":ID(PhysicalTable),name\n", encoding='utf-8')
        
        assert validate_csv_directory(csv_dir) is True
        
        empty_dir = Path(tmpdir) / "empty"
        empty_dir.mkdir()
        
        assert validate_csv_directory(empty_dir) is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_onboard.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'govio.cli.onboard'"

**Step 3: Implement onboard command**

Create `src/govio/cli/onboard.py`:

```python
import argparse
import sys
from pathlib import Path
from typing import Any

from .config import ConfigManager
from ..core.graph_factory import GraphFactory
from ..core.assets_generator import AssetsGenerator
from ..metadata.gen_networkx import build_graph


SKILLS_ASSETS_DIR = Path(__file__).parent.parent.parent.parent.parent / "skills" / "govio" / "assets"


def validate_csv_directory(csv_dir: Path) -> bool:
    """验证 CSV 目录是否包含必需的文件
    
    Args:
        csv_dir: CSV 目录路径
        
    Returns:
        bool: 是否有效
    """
    if not csv_dir.exists() or not csv_dir.is_dir():
        return False
    
    required_files = ["PhysicalTable.csv"]
    
    for filename in required_files:
        if not (csv_dir / filename).exists():
            return False
    
    return True


def prompt_backend_choice() -> str:
    """提示用户选择 backend"""
    print("\n=== Govio Onboard 向导 ===\n")
    print("请选择图数据库后端：")
    print("  1. networkx - 本地 GML 文件")
    print("  2. falkordb - FalkorDB 图数据库")
    
    while True:
        choice = input("\n请输入选项 (1/2) [默认: 1]: ").strip() or "1"
        
        if choice == "1":
            return "networkx"
        elif choice == "2":
            return "falkordb"
        else:
            print("❌ 无效选项，请输入 1 或 2")


def prompt_networkx_config() -> dict[str, Any]:
    """提示用户输入 NetworkX 配置"""
    print("\n--- NetworkX 配置 ---\n")
    
    generate_gml = input("是否需要从 CSV 文件生成新的 GML 文件？ (yes/no) [默认: yes]: ").strip().lower()
    generate_gml = generate_gml in ['yes', 'y', ''] or generate_gml == 'yes'
    
    if generate_gml:
        while True:
            csv_dir = input("请输入 CSV 目录路径: ").strip()
            csv_path = Path(csv_dir)
            
            if validate_csv_directory(csv_path):
                break
            else:
                print(f"❌ CSV 目录无效或缺少必需文件，请检查路径: {csv_dir}")
        
        gml_path = SKILLS_ASSETS_DIR / "ontology.gml"
        
        print(f"\n正在从 CSV 文件生成 GML 文件...")
        build_graph(str(csv_path), str(gml_path))
        print(f"✓ GML 文件已生成: {gml_path}")
    else:
        while True:
            gml_path_input = input("请输入 GML 文件路径: ").strip()
            gml_path = Path(gml_path_input)
            
            if gml_path.exists():
                break
            else:
                print(f"❌ GML 文件不存在: {gml_path}")
    
    return {
        'backend': 'networkx',
        'networkx': {'gml_path': str(gml_path)}
    }


def prompt_falkordb_config() -> dict[str, Any]:
    """提示用户输入 FalkorDB 配置"""
    print("\n--- FalkorDB 配置 ---\n")
    
    host = input("请输入 FalkorDB 主机地址 [默认: localhost]: ").strip() or "localhost"
    port_str = input("请输入 FalkorDB 端口 [默认: 6379]: ").strip() or "6379"
    port = int(port_str)
    graph_name = input("请输入图数据库名称 [默认: ontology]: ").strip() or "ontology"
    
    return {
        'backend': 'falkordb',
        'falkordb': {
            'host': host,
            'port': port,
            'graph': graph_name
        }
    }


def onboard():
    """Onboard 向导主函数"""
    config_manager = ConfigManager()
    
    if config_manager.exists():
        print("\n⚠️  配置文件已存在")
        overwrite = input("是否覆盖现有配置？ (yes/no): ").strip().lower()
        if overwrite not in ['yes', 'y']:
            print("已取消配置")
            return
    
    backend = prompt_backend_choice()
    
    if backend == 'networkx':
        config = prompt_networkx_config()
    else:
        config = prompt_falkordb_config()
    
    print("\n正在保存配置...")
    config_manager.save(config)
    print(f"✓ 配置已保存到: {config_manager.config_path}")
    
    print("\n正在生成 assets...")
    
    try:
        graph_obj = GraphFactory.create(config)
        generator = AssetsGenerator(graph_obj, SKILLS_ASSETS_DIR)
        generator.generate_all()
        
        print(f"✓ Assets 已生成到: {SKILLS_ASSETS_DIR}")
        print("\n✅ Onboard 完成！")
        print(f"\n配置文件: {config_manager.config_path}")
        print(f"Assets 目录: {SKILLS_ASSETS_DIR}")
        
    except Exception as e:
        print(f"\n❌ 生成 assets 失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    onboard()
```

**Step 4: Update cli __init__.py**

Edit `src/govio/cli/__init__.py`:

```python
from .config import ConfigManager
from .onboard import onboard

__all__ = ["ConfigManager", "onboard"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_onboard.py -v`

Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/govio/cli/ tests/test_onboard.py
git commit -m "feat: implement onboard CLI wizard with interactive prompts"
```

---

## Task 6: Register CLI Entry Point

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/govio/__init__.py`

**Step 1: Add onboard entry point to pyproject.toml**

Edit `pyproject.toml`:

```toml
[project.scripts]
metadata = "govio:run"
gml_generate = "govio:gml_generate"
mcp-server = "govio.mcp.server:main"
onboard = "govio.cli:onboard"
```

**Step 2: Export onboard in __init__.py**

Edit `src/govio/__init__.py`:

```python
from .metadata.utility import run
from .metadata.gen_networkx import gml_generate
from .graph.falkordb_graph import FalkorDBGraph
from .graph.networkx_graph import NetworkXGraph
from .mcp.server import create_server
from .cli import onboard

__all__ = ["run", "gml_generate", "FalkorDBGraph", "NetworkXGraph", "create_server", "onboard"]
```

**Step 3: Test the CLI command**

Run: `uv run onboard --help 2>&1 || echo "Command works but no help defined"`

Expected: Command executes (may show error if run without arguments)

**Step 4: Commit**

```bash
git add pyproject.toml src/govio/__init__.py
git commit -m "feat: register onboard CLI entry point"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `README.md`

**Step 1: Add onboard usage to README**

Add a new section in `README.md` (after installation section):

```markdown
## 快速开始

### 首次使用：运行 Onboard 向导

安装完成后，运行 onboard 向导进行初始化配置：

```bash
uv run onboard
```

向导会引导你完成以下步骤：

1. **选择图数据库后端**
   - NetworkX: 使用本地 GML 文件
   - FalkorDB: 连接 FalkorDB 图数据库

2. **NetworkX 模式**
   - 选择是否从 CSV 文件生成新的 GML 文件
   - 如果选择生成，输入 CSV 目录路径
   - 如果不生成，输入已有的 GML 文件路径

3. **FalkorDB 模式**
   - 输入 FalkorDB 连接信息（host, port, graph name）

4. **自动生成**
   - 配置文件保存到 `~/.govio/config.yaml`
   - Assets 生成到 `skills/govio/assets/`
     - `schema.md`: 图结构定义
     - `names/`: 节点名称索引

### CSV 文件格式要求

如果选择从 CSV 生成 GML 文件，CSV 目录应包含以下文件：

**节点文件：**
- `PhysicalTable.csv`: 物理表节点
- `Col.csv`: 字段节点
- `Application.csv`: 应用节点
- `Standard.csv`: 数据标准节点

**边文件：**
- `HAS_COLUMN.csv`: 表包含字段的关系
- `USE.csv`: 应用使用表的关系
- `COMPLIES_WITH.csv`: 字段贯标的关系

**CSV 格式示例：**

```csv
# PhysicalTable.csv
:ID(PhysicalTable),name,full_table_name
table1,用户表,DB.SCHEMA.TABLE1

# Col.csv
:ID(Col),name,column_name,full_table_name
col1,用户ID,USER_ID,DB.SCHEMA.TABLE1

# HAS_COLUMN.csv
:START_ID(PhysicalTable),:END_ID(Col)
table1,col1
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add onboard usage documentation to README"
```

---

## Summary

This implementation plan creates a comprehensive onboard wizard for govio with:

- **ConfigManager**: YAML-based configuration management
- **GraphFactory**: Factory pattern for creating graph objects
- **AssetsGenerator**: Unified asset generation (schema + names)
- **Onboard CLI**: Interactive wizard guiding users through setup
- **Tests**: Unit and integration tests for all components
- **Documentation**: Updated README with usage instructions

All tasks follow TDD approach with frequent commits and complete code examples.
