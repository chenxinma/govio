# Config.yaml 结构重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `~/.govio/config.yaml` 从扁平结构重构为 metadata / graph / datasources 三段式嵌套结构，支持旧格式自动迁移。

**Architecture:** ConfigManager 新增 `_migrate()` 方法检测旧格式并转换；所有消费端（onboard, query, meta_export, std_recommend, main, graph_factory）改为从嵌套路径读取配置。GraphFactory.create() 接口不变，调用方传入 `config["graph"]` 子字典。

**Tech Stack:** Python 3.13+, PyYAML, pytest

---

### Task 1: ConfigManager — 迁移逻辑与新验证规则

**Files:**
- Modify: `src/govio/cli/config.py:1-89`
- Test: `tests/test_config_manager.py`

- [ ] **Step 1: 编写迁移测试**

```python
# tests/test_config_manager.py 末尾追加

def test_migrate_old_flat_config():
    """旧扁平格式应自动迁移为嵌套结构"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        # 写入旧格式
        old_config = {
            "kundb": "mysql+pymysql://user:pass@host/db",
            "workspace_uuid": "test-uuid",
            "app_list": "app.xlsx",
            "app_map": "app.json",
            "relationship": "rel.json",
            "metric": "metric.json",
            "csv_dir": "/tmp/csv",
            "backend": "networkx",
            "networkx": {"gml_path": "test.gml"},
            "datasources": {"ds1": {"url": "mysql://localhost/db"}},
        }
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(old_config, f)

        # 加载应自动迁移
        loaded = manager.load()

        # 验证新结构
        assert "metadata" in loaded
        assert loaded["metadata"]["kundb"] == "mysql+pymysql://user:pass@host/db"
        assert loaded["metadata"]["workspace_uuid"] == "test-uuid"
        assert loaded["metadata"]["csv_dir"] == "/tmp/csv"

        assert "graph" in loaded
        assert loaded["graph"]["backend"] == "networkx"
        assert loaded["graph"]["networkx"]["gml_path"] == "test.gml"

        assert "datasources" in loaded
        assert loaded["datasources"]["ds1"]["url"] == "mysql://localhost/db"

        # 旧字段不应存在
        assert "kundb" not in loaded
        assert "backend" not in loaded

        # 备份文件应存在
        assert (Path(tmpdir) / "config.yaml.bak").exists()


def test_new_config_not_migrated():
    """新格式配置不应被迁移"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        new_config = {
            "metadata": {"kundb": "mysql://host/db", "csv_dir": "/tmp/csv"},
            "graph": {"backend": "networkx", "networkx": {"gml_path": "test.gml"}},
        }
        manager.save(new_config)

        loaded = manager.load()
        assert loaded == new_config


def test_validate_new_structure():
    """验证新嵌套结构"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(config_path)

        valid_config = {
            "graph": {"backend": "networkx", "networkx": {"gml_path": "test.gml"}},
        }
        assert manager.validate(valid_config) is True

        invalid_config = {
            "graph": {"backend": "networkx"},  # 缺少 networkx 配置
        }
        with pytest.raises(ValueError):
            manager.validate(invalid_config)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /data/home/macx/work/python/govio && uv run pytest tests/test_config_manager.py -v`
Expected: FAIL（`_migrate` 方法不存在，新结构验证不通过）

- [ ] **Step 3: 实现迁移逻辑和新验证**

```python
# src/govio/cli/config.py — 完整替换

import shutil
import yaml
from pathlib import Path
from typing import Any

# 旧格式中属于 metadata section 的字段
_METADATA_KEYS = {"kundb", "workspace_uuid", "app_list", "app_map", "relationship", "metric", "csv_dir"}
# 旧格式中属于 graph section 的字段
_GRAPH_KEYS = {"backend", "networkx", "falkordb"}


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
        """加载配置文件，自动迁移旧格式"""
        if not self.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        if self._is_old_format(config):
            config = self._migrate(config)

        return config

    def save(self, config: dict[str, Any]) -> None:
        """保存配置文件"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    def _is_old_format(self, config: dict[str, Any]) -> bool:
        """检测是否为旧的扁平格式"""
        # 旧格式特征：顶层存在 kundb 或 backend 字段
        return "kundb" in config or ("backend" in config and "graph" not in config)

    def _migrate(self, config: dict[str, Any]) -> dict[str, Any]:
        """将旧扁平格式迁移为新的嵌套格式"""
        # 备份旧文件
        backup_path = self.config_path.with_suffix(".yaml.bak")
        shutil.copy2(self.config_path, backup_path)

        new_config: dict[str, Any] = {}

        # 提取 metadata 字段
        metadata = {}
        for key in _METADATA_KEYS:
            if key in config:
                metadata[key] = config[key]
        if metadata:
            new_config["metadata"] = metadata

        # 提取 graph 字段
        graph = {}
        for key in _GRAPH_KEYS:
            if key in config:
                graph[key] = config[key]
        if graph:
            new_config["graph"] = graph

        # datasources 保持不变
        if "datasources" in config:
            new_config["datasources"] = config["datasources"]

        # 保存迁移后的配置
        self.save(new_config)

        return new_config

    def validate(self, config: dict[str, Any]) -> bool:
        """验证配置的有效性

        支持新格式（嵌套）和旧格式（扁平）的验证。
        """
        # 如果是新格式，从 graph section 取 backend
        if "graph" in config:
            graph = config["graph"]
            if "backend" not in graph:
                raise ValueError("配置缺少 'graph.backend' 字段")
            backend = graph["backend"]
            if backend not in ["networkx", "falkordb"]:
                raise ValueError(f"不支持的 backend: {backend}")
            if backend == "networkx":
                if "networkx" not in graph:
                    raise ValueError("NetworkX backend 需要 'networkx' 配置")
                if "gml_path" not in graph["networkx"]:
                    raise ValueError("NetworkX 配置缺少 'gml_path' 字段")
            elif backend == "falkordb":
                if "falkordb" not in graph:
                    raise ValueError("FalkorDB backend 需要 'falkordb' 配置")
                for field in ["host", "port", "graph"]:
                    if field not in graph["falkordb"]:
                        raise ValueError(f"FalkorDB 配置缺少 '{field}' 字段")
        elif "backend" in config:
            # 兼容旧格式验证
            backend = config["backend"]
            if backend not in ["networkx", "falkordb"]:
                raise ValueError(f"不支持的 backend: {backend}")
            if backend == "networkx":
                if "networkx" not in config:
                    raise ValueError("NetworkX backend 需要 'networkx' 配置")
                if "gml_path" not in config["networkx"]:
                    raise ValueError("NetworkX 配置缺少 'gml_path' 字段")
            elif backend == "falkordb":
                if "falkordb" not in config:
                    raise ValueError("FalkorDB backend 需要 'falkordb' 配置")
                for field in ["host", "port", "graph"]:
                    if field not in config["falkordb"]:
                        raise ValueError(f"FalkorDB 配置缺少 '{field}' 字段")

        # csv_dir 验证（新格式在 metadata 下）
        csv_dir = config.get("metadata", {}).get("csv_dir") or config.get("csv_dir")
        if csv_dir:
            csv_path = Path(csv_dir)
            if not csv_path.exists():
                raise ValueError(f"csv_dir 不存在: {csv_path}")

        # graph_dir 验证
        if "graph_dir" in config:
            graph_path = Path(config["graph_dir"])
            if not graph_path.exists():
                raise ValueError(f"graph_dir 不存在: {graph_path}")

        # datasources 验证
        datasources = config.get("datasources")
        if datasources:
            if not isinstance(datasources, dict):
                raise ValueError("datasources 必须为字典类型")
            for name, ds_data in datasources.items():
                if not isinstance(ds_data, dict):
                    raise ValueError(f"数据源 '{name}' 配置必须为字典类型")
                if "url" not in ds_data:
                    raise ValueError(f"数据源 '{name}' 缺少 'url' 字段")

        return True
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /data/home/macx/work/python/govio && uv run pytest tests/test_config_manager.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/govio/cli/config.py tests/test_config_manager.py
git commit -m "feat: config restructure — add migration and new validation to ConfigManager"
```

---

### Task 2: GraphFactory — 无需修改，调用方适配

GraphFactory.create() 接收 dict，期望 `config["backend"]`、`config["networkx"]`、`config["falkordb"]`。新结构下 `config["graph"]` 子字典恰好包含这三个 key，所以 **GraphFactory.create() 不需要修改**。

调用方（onboard.py, meta_export.py）改为传入 `config["graph"]` 即可。

无需独立 commit。此 task 仅作说明。

---

### Task 3: onboard.py — 适配新配置结构

**Files:**
- Modify: `src/govio/cli/onboard.py`
- Test: `tests/test_onboard.py`

- [ ] **Step 1: 更新 onboard 测试中的配置结构断言**

```python
# tests/test_onboard.py — 修改 test_onboard_networkx_workflow 的断言部分

# 在 "Verify outputs" 部分，将:
#     assert config_path.exists()
# 改为验证新结构:

    # Verify config uses new nested structure
    saved_config = ConfigManager(config_path).load()
    assert "metadata" in saved_config
    assert "graph" in saved_config
    assert saved_config["graph"]["backend"] == "networkx"
```

同时修改 `test_onboard_new_falkordb` 和 `test_onboard_new_networkx` 的断言：

```python
# test_onboard_new_falkordb — 将:
#     assert saved_config["backend"] == "falkordb"
#     assert saved_config["csv_dir"] == str(csv_dir)
#     assert saved_config["falkordb"]["host"] == "localhost"
#     assert saved_config["falkordb"]["graph"] == "test_graph"
# 改为:
    assert saved_config["graph"]["backend"] == "falkordb"
    assert saved_config["metadata"]["csv_dir"] == str(csv_dir)
    assert saved_config["graph"]["falkordb"]["host"] == "localhost"
    assert saved_config["graph"]["falkordb"]["graph"] == "test_graph"


# test_onboard_new_networkx — 将:
#     assert saved_config["backend"] == "networkx"
#     assert saved_config["csv_dir"] == str(csv_dir)
#     assert "gml_path" in saved_config["networkx"]
# 改为:
    assert saved_config["graph"]["backend"] == "networkx"
    assert saved_config["metadata"]["csv_dir"] == str(csv_dir)
    assert "gml_path" in saved_config["graph"]["networkx"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /data/home/macx/work/python/govio && uv run pytest tests/test_onboard.py -v`
Expected: FAIL（onboard 写入的仍是旧格式）

- [ ] **Step 3: 修改 onboard.py 生成新格式配置**

关键变更点：

**prompt_csv_config()** — 返回值不变（仍返回 metadata 字段的 dict），调用方负责嵌套。

**onboard() 函数** — 核心变更：

```python
# onboard.py — onboard() 函数中，所有组装 full_config 的地方改为嵌套结构

# === Path A: --new-networkx ===
# 原:
#     config = {"backend": "networkx", "networkx": {"gml_path": str(gml_path)}}
#     full_config = {**existing_config, "csv_dir": str(csv_dir), **config}
# 改为:
        graph_config = {"backend": "networkx", "networkx": {"gml_path": str(gml_path)}}
        full_config = {
            **existing_config,
            "metadata": {**existing_config.get("metadata", {}), "csv_dir": str(csv_dir)},
            "graph": graph_config,
        }


# === Path B: --new-falkordb ===
# 原:
#     config = prompt_falkordb_config(csv_dir)
#     full_config = {**existing_config, "csv_dir": str(csv_dir), **config}
# 改为:
        graph_config = prompt_falkordb_config(csv_dir)
        full_config = {
            **existing_config,
            "metadata": {**existing_config.get("metadata", {}), "csv_dir": str(csv_dir)},
            "graph": graph_config,
        }


# === Path C: 检测已有配置跳过 ===
# 原:
#     has_backend = "backend" in existing_config
#     ... existing_config['backend']
# 改为:
        has_backend = "graph" in existing_config and "backend" in existing_config.get("graph", {})
        if has_backend:
            print(f"\n⚠️  检测到已有配置 (backend: {existing_config['graph']['backend']})")


# === Path D: 完整 onboard ===
# 原:
#     csv_config = prompt_csv_config(config_manager)
#     ...
#     full_config = {**csv_config, "csv_dir": str(Path(csv_config["csv_dir"]).resolve())}
#     backend = prompt_backend_choice()
#     csv_dir = Path(csv_config["csv_dir"])
#     if backend == "networkx":
#         config = prompt_networkx_config()
#         full_config.update(config)
#     else:
#         config = prompt_falkordb_config(csv_dir)
#         full_config.update(config)
# 改为:
        csv_config = prompt_csv_config(config_manager)
        print("\n正在生成 CSV 文件...")
        try:
            generate_csv(csv_config)
            print(f"✓ CSV 文件已生成到: {csv_config['csv_dir']}")
        except Exception as e:
            print(f"❌ CSV 生成失败: {e}")
            sys.exit(1)

        metadata = {
            **csv_config,
            "csv_dir": str(Path(csv_config["csv_dir"]).resolve()),
        }

        backend = prompt_backend_choice()
        csv_dir = Path(csv_config["csv_dir"])

        if backend == "networkx":
            graph_config = prompt_networkx_config()
        else:
            graph_config = prompt_falkordb_config(csv_dir)

        full_config: dict[str, Any] = {"metadata": metadata, "graph": graph_config}

        print("\n正在保存配置...")
        config_manager.save(full_config)
        print(f"✓ 配置已保存到: {config_manager.config_path}")

        datasources = prompt_datasource_config()
        if datasources:
            full_config["datasources"] = datasources
            config_manager.save(full_config)

        # Assets 生成使用 graph_config（结构与 GraphFactory.create 期望的一致）
        print("\n正在生成 assets...")
        try:
            graph_obj = GraphFactory.create(graph_config)
            ...
```

同时修改 `GraphFactory.create()` 的调用：所有 `GraphFactory.create(config)` 改为 `GraphFactory.create(graph_config)` 或 `GraphFactory.create(full_config["graph"])`。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /data/home/macx/work/python/govio && uv run pytest tests/test_onboard.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/govio/cli/onboard.py tests/test_onboard.py
git commit -m "feat: config restructure — onboard writes nested config format"
```

---

### Task 4: query.py — 适配新配置结构

**Files:**
- Modify: `src/govio/cli/query.py:96-126`

- [ ] **Step 1: 修改 query() 函数读取嵌套配置**

```python
# src/govio/cli/query.py — query() 函数

def query(query_text):
    """Query CLI 主函数"""

    config_manager = ConfigManager()
    if not config_manager.exists():
        print("配置文件不存在，请先运行 govio-cli onboard")
        sys.exit(1)

    config = config_manager.load()
    graph = config.get("graph", {})
    backend = graph.get("backend")
    if not backend:
        print("配置文件缺少 'graph.backend' 字段，请重新运行 govio-cli onboard")
        sys.exit(1)

    if backend == "networkx":
        gml_path = graph.get("networkx", {}).get("gml_path")
        if not gml_path:
            print("配置文件缺少 graph.networkx.gml_path 字段")
            sys.exit(1)
        cmd_networkx(query_text, gml_path)
    elif backend == "falkordb":
        falkordb_config = graph.get("falkordb", {})
        cmd_falkordb(
            query_text,
            host=falkordb_config.get("host", "localhost"),
            port=falkordb_config.get("port", 6379),
            graph_name=falkordb_config.get("graph", "ontology"),
        )
    else:
        print(f"不支持的 backend: {backend}")
        sys.exit(1)
```

- [ ] **Step 2: 运行现有测试**

Run: `cd /data/home/macx/work/python/govio && uv run pytest tests/ -v -k "query"`
Expected: 现有 query 测试通过（如有），或无 query 测试则无影响

- [ ] **Step 3: Commit**

```bash
git add src/govio/cli/query.py
git commit -m "feat: config restructure — query reads from nested config"
```

---

### Task 5: main.py — 适配新配置结构

**Files:**
- Modify: `src/govio/cli/main.py:43-50, 83-93`

- [ ] **Step 1: 修改 main.py 中读取 backend 的两处代码**

```python
# main.py — query 子命令的 code_type 提示 (lines 43-49)
# 原:
#     backend = config.get("backend")
# 改为:
        backend = config.get("graph", {}).get("backend")

# main.py — backend 子命令 (lines 88-93)
# 原:
#     backend = config.get("backend")
# 改为:
        backend = config.get("graph", {}).get("backend")
```

- [ ] **Step 2: Commit**

```bash
git add src/govio/cli/main.py
git commit -m "feat: config restructure — main reads backend from nested config"
```

---

### Task 6: meta_export.py — 适配新配置结构

**Files:**
- Modify: `src/govio/cli/meta_export.py:30-37, 216-243`

- [ ] **Step 1: 修改 meta_export() 中的配置读取**

```python
# src/govio/cli/meta_export.py — meta_export() 函数

# --- Load config for TDS ---
config = ConfigManager().load()
metadata = config.get("metadata", {})
kundb = metadata["kundb"]
workspace_uuid = metadata.get("workspace_uuid", "82ee37374b314a938bf28170ab4db7cf")
app_list_file = metadata["app_list"]
app_map_file = metadata["app_map"]
relationship_file = metadata.get("relationship")
metric_file = metadata.get("metric")

# --- Update graph section ---
graph = config.get("graph", {})
backend = graph.get("backend")
if not backend:
    print("警告: 配置中未指定 backend，跳过图数据更新和 assets 生成")
    return

if backend == "falkordb":
    falkordb_cfg = graph.get("falkordb", {})
    host = falkordb_cfg.get("host", "localhost")
    port = falkordb_cfg.get("port", 6379)
    graph_name = falkordb_cfg.get("graph", "ontology")
    ...
elif backend == "networkx":
    networkx_cfg = graph.get("networkx", {})
    gml_path = networkx_cfg.get("gml_path", str(SKILLS_ASSETS_DIR / "ontology.gml"))
    ...

# GraphFactory.create 接收 graph 子字典
graph_obj = GraphFactory.create(graph)
```

- [ ] **Step 2: 运行测试**

Run: `cd /data/home/macx/work/python/govio && uv run pytest tests/test_meta_export.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/govio/cli/meta_export.py
git commit -m "feat: config restructure — meta_export reads from nested config"
```

---

### Task 7: std_recommend.py — 适配新配置结构

**Files:**
- Modify: `src/govio/cli/std_recommend.py:18-26`

- [ ] **Step 1: 修改 std_recommend() 中的配置读取**

```python
# src/govio/cli/std_recommend.py — std_recommend() 函数

config = config_manager.load()

metadata = config.get("metadata", {})
kundb = metadata.get("kundb", "")
workspace_uuid = metadata.get("workspace_uuid", "")
app_map = metadata.get("app_map", "")
csv_dir = metadata.get("csv_dir", "./")
```

- [ ] **Step 2: Commit**

```bash
git add src/govio/cli/std_recommend.py
git commit -m "feat: config restructure — std_recommend reads from nested config"
```

---

### Task 8: 全量回归测试

- [ ] **Step 1: 运行全部测试**

Run: `cd /data/home/macx/work/python/govio && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: 验证旧格式迁移端到端**

手动验证：创建一个旧格式 config.yaml，运行 `govio-cli backend`，确认输出正确且生成了 .bak 备份。

- [ ] **Step 3: 最终 Commit（如有遗漏修复）**

```bash
git add -A
git commit -m "chore: config restructure — final regression fixes"
```
