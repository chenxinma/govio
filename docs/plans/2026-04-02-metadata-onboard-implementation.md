# Metadata Onboard 重设计实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 CSV 元数据生成逻辑从 `utility.py` 移到 `onboard` 向导的第一步，并通过 config.yaml 管理参数。

**Architecture:** 在现有 onboard 向导前插入 CSV 生成步骤，复用 `make_csv` 函数生成 CSV，更新 ConfigManager 支持 csv_dir 字段。

**Tech Stack:** Python 3.13+, pandas, sqlalchemy, pyyaml

---

## 实施步骤

### Task 1: 修改 ConfigManager 支持 csv_dir 字段

**Files:**
- Modify: `src/govio/cli/config.py:29-32`

**Step 1: 修改 save 方法支持 csv_dir**

在 `save` 方法中，确保 `csv_dir` 和 `graph_dir` 被保留。现有代码已经使用 `yaml.dump(config, f)`，所以只需要确保 ConfigManager 的 `validate` 方法和调用方正确处理新字段。

当前 ConfigManager.save 会完整保存传入的 config dict，所以只需确保 onboarding 步骤正确传递 csv_dir。

**Step 2: 修改 validate 方法支持 csv_dir**

在 `validate` 方法中添加对 `csv_dir` 和 `graph_dir` 的可选验证：

```python
def validate(self, config: dict[str, Any]) -> bool:
    # ... existing backend validation ...
    
    # csv_dir and graph_dir are optional
    if "csv_dir" in config:
        csv_path = Path(config["csv_dir"])
        if not csv_path.exists():
            raise ValueError(f"csv_dir 不存在: {csv_path}")
    
    if "graph_dir" in config:
        graph_path = Path(config["graph_dir"])
        if not graph_path.exists():
            raise ValueError(f"graph_dir 不存在: {graph_path}")
    
    return True
```

**Step 3: Commit**

```bash
git add src/govio/cli/config.py
git commit -m "feat(cli): add csv_dir/graph_dir validation to ConfigManager"
```

---

### Task 2: 在 onboard.py 中添加 CSV 生成步骤

**Files:**
- Modify: `src/govio/cli/onboard.py:35-108`

**Step 1: 添加 prompt_csv_config 函数**

在 `prompt_backend_choice` 函数后添加：

```python
def prompt_csv_config(config_manager: ConfigManager) -> dict[str, Any]:
    """提示用户输入 CSV 生成配置"""
    print("\n=== 步骤 1: CSV 元数据生成 ===\n")
    
    existing_config = {}
    if config_manager.exists():
        try:
            existing_config = config_manager.load()
        except Exception:
            pass
    
    # kundb
    default_kundb = existing_config.get("kundb", "")
    kundb = input(f"请输入元数据库 URL [默认: {default_kundb}]: ").strip()
    kundb = kundb or default_kundb
    
    # app-list
    default_app_list = existing_config.get("app_list", "")
    app_list = input(f"请输入应用清单 Excel 文件路径 [默认: {default_app_list}]: ").strip()
    app_list = app_list or default_app_list
    
    # app-map
    default_app_map = existing_config.get("app_map", "")
    app_map = input(f"请输入应用数据库映射 JSON 文件路径 [默认: {default_app_map}]: ").strip()
    app_map = app_map or default_app_map
    
    # relationship (optional)
    default_relationship = existing_config.get("relationship", "")
    relationship = input(f"请输入表关系 JSON 文件路径（可选，直接回车跳过） [默认: {default_relationship}]: ").strip()
    relationship = relationship or default_relationship
    
    # csv-dir
    default_csv_dir = existing_config.get("csv_dir", "")
    csv_dir = input(f"请输入 CSV 输出目录 [默认: {default_csv_dir}]: ").strip()
    csv_dir = csv_dir or default_csv_dir
    
    return {
        "kundb": kundb,
        "app_list": app_list,
        "app_map": app_map,
        "relationship": relationship if relationship else None,
        "csv_dir": csv_dir,
    }
```

**Step 2: 添加 generate_csv 函数**

```python
def generate_csv(config: dict[str, Any]) -> None:
    """根据配置生成 CSV 文件"""
    from ..metadata.utility import make_csv
    from ..metadata.database import DatabaseLoader
    from ..metadata.application import AppInfoLoader
    from ..metadata.standard import StandardLoader
    import pandas as pd
    
    kundb = config["kundb"]
    app_list = config["app_list"]
    app_map = config["app_map"]
    relationship = config.get("relationship")
    csv_dir = Path(config["csv_dir"])
    workspace_uuid = "82ee37374b314a938bf28170ab4db7cf"
    
    if not csv_dir.exists():
        csv_dir.mkdir(parents=True, exist_ok=True)
    
    df_app_db_map = pd.read_json(app_map, orient="records")
    
    make_csv(
        output=csv_dir,
        db=kundb,
        workspace_uuid=workspace_uuid,
        app_list_file=app_list,
        df_app_db_map=df_app_db_map,
        relationship_file=relationship,
    )
```

**Step 3: 修改 onboard 函数**

修改 `onboard()` 函数，在 `prompt_backend_choice()` 前插入 CSV 生成步骤：

```python
def onboard():
    config_manager = ConfigManager()

    if config_manager.exists():
        print("\n⚠️  配置文件已存在")
        overwrite = input("是否覆盖现有配置？ (yes/no): ").strip().lower()
        if overwrite not in ["yes", "y"]:
            print("已取消配置")
            return

    # 步骤 1: CSV 生成
    csv_config = prompt_csv_config(config_manager)
    print("\n正在生成 CSV 文件...")
    try:
        generate_csv(csv_config)
        print(f"✓ CSV 文件已生成到: {csv_config['csv_dir']}")
    except Exception as e:
        print(f"❌ CSV 生成失败: {e}")
        sys.exit(1)

    # 合并配置
    full_config = {
        **csv_config,
        "csv_dir": str(Path(csv_config["csv_dir"]).resolve()),
    }

    # 步骤 2: 选择后端
    backend = prompt_backend_choice()
    # ... rest unchanged until save ...
```

**Step 4: Commit**

```bash
git add src/govio/cli/onboard.py
git commit -m "feat(onboard): add CSV metadata generation as step 1"
```

---

### Task 3: 从 pyproject.toml 移除 metadata 命令

**Files:**
- Modify: `pyproject.toml:44-48`

**Step 1: 移除 metadata entry point**

将：
```toml
[project.scripts]
metadata = "govio:run"
gml_generate = "govio:gml_generate"
mcp-server = "govio.mcp.server:main"
onboard = "govio.cli:onboard"
```

改为：
```toml
[project.scripts]
gml_generate = "govio:gml_generate"
mcp-server = "govio.mcp.server:main"
onboard = "govio.cli:onboard"
```

**Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: remove metadata command, use onboard instead"
```

---

### Task 4: 清理 utility.py 中不再使用的代码

**Files:**
- Modify: `src/govio/metadata/utility.py:35-92`

**Step 1: 移除 run() 函数和 make_csv 的直接调用**

保留 `make_csv` 函数（因为 onboard 需要调用），但移除 `run()` 函数和相关的 argparse 代码。

修改后的 utility.py 应该只保留：
- `Mode` enum（如还需要）
- `reorder_index` 函数
- `make_csv` 函数
- `data_standard_recommend` 函数（如需要保留）
- `eval_test` 函数（如需要保留）

移除：
- `run()` 函数
- `argparse` 相关代码
- `load_dotenv()` 调用

**Step 2: Commit**

```bash
git add src/govio/metadata/utility.py
git commit -m "chore: remove run() and argparse code from utility.py, keep make_csv"
```

---

### Task 5: 验证修改

**Step 1: 运行 lint 和 typecheck**

```bash
uv run ruff check src/govio/cli/ src/govio/metadata/utility.py
uv run pyright src/govio/cli/ src/govio/metadata/utility.py
```

**Step 2: 运行测试**

```bash
uv run pytest tests/ -v
```

**Step 3: 测试 onboard 向导（交互式）**

手动测试 `onboard` 命令是否能正常完成流程。

---

## 依赖文件

需要查看以下文件以确保 make_csv 函数签名和依赖正确：

- `src/govio/metadata/database.py` - DatabaseLoader
- `src/govio/metadata/application.py` - AppInfoLoader
- `src/govio/metadata/standard.py` - StandardLoader
- `src/govio/metadata/relationship.py` - load_relationships

## 注意事项

1. `workspace_uuid = "82ee37374b314a938bf28170ab4db7cf"` 是硬编码的，后续可能需要移到 config
2. `make_csv` 函数复用时，需要确保 DatabaseLoader 和 AppInfoLoader 的参数正确
3. 当前 onboard 向导会在 CSV 生成后询问 NetworkX 的 csv_dir，但新设计中 csv_dir 已在步骤 1 获取，需调整 NetworkX 配置步骤使用已有的 csv_dir
