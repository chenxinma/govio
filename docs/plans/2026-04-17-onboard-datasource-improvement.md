# Onboard 数据源配置优化实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 优化 onboard 命令，支持跳过 CSV/Graph 配置单独配置数据源，connect_args 改为 key-value 输入

**Architecture:** 修改 onboard.py 中的 onboard() 和 prompt_datasource_config() 函数，添加跳过逻辑和改进输入交互

**Tech Stack:** Python 3.13+, argparse, yaml

---

### Task 1: 添加 prompt_connect_args 函数

**Files:**
- Modify: `src/govio/cli/onboard.py:277-321`

**Step 1: 编写 prompt_connect_args 函数**

在 `prompt_datasource_config` 函数之前添加新函数：

```python
def prompt_connect_args(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """交互式输入连接参数（key=value 格式）

    Args:
        existing: 已有的连接参数

    Returns:
        dict: 连接参数字典
    """
    connect_args: dict[str, Any] = {}

    if existing:
        print(f"  当前连接参数: {existing}")
        keep = input("  是否保留现有参数？ (yes/no) [默认: yes]: ").strip().lower()
        if keep not in ("no", "n"):
            return existing

    print("  输入连接参数 (key=value 格式，空行结束):")
    print("  示例: ssl=true, timeout=30")

    while True:
        line = input("  > ").strip()
        if not line:
            break
        if "=" not in line:
            print("  格式错误，请使用 key=value 格式")
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        connect_args[key] = value

    return connect_args
```

**Step 2: 运行 lint 检查**

Run: `uv run ruff check src/govio/cli/onboard.py`
Expected: 无错误

**Step 3: 提交**

```bash
git add src/govio/cli/onboard.py
git commit -m "feat: add prompt_connect_args for key-value input"
```

### Task 2: 重构 prompt_datasource_config 支持管理已有数据源

**Files:**
- Modify: `src/govio/cli/onboard.py:277-321`

**Step 1: 重写 prompt_datasource_config 函数**

```python
def prompt_datasource_config(
    existing_datasources: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """提示用户配置数据源（可选）

    Args:
        existing_datasources: 已有的数据源配置

    Returns:
        dict: 数据源配置字典，None 表示无数据源
    """
    print("\n=== 数据源配置（可选）===\n")
    print("配置数据源供 observe 命令使用")
    print("可添加 MySQL、DuckDB 等数据源\n")

    datasources: dict[str, Any] = dict(existing_datasources) if existing_datasources else {}

    while True:
        if datasources:
            print("已配置的数据源:")
            for name, ds in datasources.items():
                print(f"  - {name}: {ds['url']}")
            print()

        print("操作选项:")
        print("  1. 添加数据源")
        print("  2. 删除数据源")
        print("  3. 完成配置")

        choice = input("\n请选择操作 (1/2/3) [默认: 3]: ").strip() or "3"

        if choice == "1":
            name = input("  数据源名称: ").strip()
            if not name:
                print("  名称不能为空")
                continue
            url = input("  URL (如 mysql+pymysql://user:pass@host/db): ").strip()
            if not url:
                print("  URL 不能为空")
                continue
            existing_args = datasources.get(name, {}).get("connect_args")
            connect_args = prompt_connect_args(existing_args)
            datasources[name] = {"url": url, "connect_args": connect_args}
            print(f"  已添加数据源: {name}")

        elif choice == "2":
            if not datasources:
                print("  没有可删除的数据源")
                continue
            print("  选择要删除的数据源:")
            names = list(datasources.keys())
            for i, n in enumerate(names, 1):
                print(f"    {i}. {n}")
            del_choice = input("  输入编号 (或直接回车取消): ").strip()
            if not del_choice:
                continue
            try:
                idx = int(del_choice) - 1
                if 0 <= idx < len(names):
                    removed = names[idx]
                    del datasources[removed]
                    print(f"  已删除: {removed}")
                else:
                    print("  无效编号")
            except ValueError:
                print("  请输入数字编号")

        elif choice == "3":
            break

    return datasources if datasources else None
```

**Step 2: 运行 lint 检查**

Run: `uv run ruff check src/govio/cli/onboard.py`
Expected: 无错误

**Step 3: 提交**

```bash
git add src/govio/cli/onboard.py
git commit -m "feat: refactor datasource config with add/delete support"
```

### Task 3: 修改 onboard 函数支持跳过 CSV/Graph 配置

**Files:**
- Modify: `src/govio/cli/onboard.py:324-388`

**Step 1: 重写 onboard 函数**

```python
def onboard():
    """Onboard 向导主函数"""
    config_manager = ConfigManager()

    if config_manager.exists():
        existing_config = config_manager.load()
        has_backend = "backend" in existing_config

        if has_backend:
            print(f"\n⚠️  检测到已有配置 (backend: {existing_config['backend']})")
            skip = (
                input("是否跳过 CSV/Graph 配置，仅配置数据源？ (yes/no) [默认: no]: ")
                .strip()
                .lower()
            )
            if skip in ("yes", "y"):
                full_config = existing_config
                datasources = prompt_datasource_config(full_config.get("datasources"))
                if datasources is not None:
                    full_config["datasources"] = datasources
                else:
                    full_config.pop("datasources", None)
                config_manager.save(full_config)
                print(f"\n✅ 配置已更新: {config_manager.config_path}")
                return

        print("\n⚠️  配置文件已存在")
        overwrite = input("是否覆盖现有配置？ (yes/no): ").strip().lower()
        if overwrite not in ["yes", "y"]:
            print("已取消配置")
            return

    csv_config = prompt_csv_config(config_manager)
    print("\n正在生成 CSV 文件...")
    try:
        generate_csv(csv_config)
        print(f"✓ CSV 文件已生成到: {csv_config['csv_dir']}")
    except Exception as e:
        print(f"❌ CSV 生成失败: {e}")
        sys.exit(1)

    full_config = {
        **csv_config,
        "csv_dir": str(Path(csv_config["csv_dir"]).resolve()),
    }

    backend = prompt_backend_choice()
    csv_dir = Path(csv_config["csv_dir"])

    if backend == "networkx":
        config = prompt_networkx_config()
        full_config.update(config)
    else:
        config = prompt_falkordb_config(csv_dir)
        full_config.update(config)

    print("\n正在保存配置...")
    config_manager.save(full_config)
    print(f"✓ 配置已保存到: {config_manager.config_path}")

    datasources = prompt_datasource_config()
    if datasources:
        full_config["datasources"] = datasources
        config_manager.save(full_config)

    backend_file = SKILLS_ASSETS_DIR / "backend.txt"
    backend_file.write_text(backend + "\n")
    print(f"✓ Backend 已写入: {backend_file}")

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
```

**Step 2: 运行 lint 检查**

Run: `uv run ruff check src/govio/cli/onboard.py`
Expected: 无错误

**Step 3: 运行格式化**

Run: `uv run ruff format src/govio/cli/onboard.py`
Expected: 格式化完成

**Step 4: 提交**

```bash
git add src/govio/cli/onboard.py
git commit -m "feat: add skip CSV/Graph option in onboard flow"
```

### Task 4: 添加单元测试

**Files:**
- Create: `tests/cli/test_onboard.py`

**Step 1: 创建测试文件**

```python
"""测试 onboard 命令的交互逻辑"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from govio.cli.onboard import prompt_connect_args, validate_csv_directory


class TestPromptConnectArgs:
    """测试 prompt_connect_args 函数"""

    def test_empty_input(self):
        """测试空输入返回空字典"""
        with patch("builtins.input", return_value=""):
            result = prompt_connect_args()
            assert result == {}

    def test_single_kv(self):
        """测试单个 key-value 输入"""
        responses = ["ssl=true", ""]
        with patch("builtins.input", side_effect=responses):
            result = prompt_connect_args()
            assert result == {"ssl": True}

    def test_multiple_kv(self):
        """测试多个 key-value 输入"""
        responses = ["ssl=true", "timeout=30", "name=test", ""]
        with patch("builtins.input", side_effect=responses):
            result = prompt_connect_args()
            assert result == {"ssl": True, "timeout": 30, "name": "test"}

    def test_invalid_format_then_valid(self):
        """测试格式错误后继续输入"""
        responses = ["invalid", "key=value", ""]
        with patch("builtins.input", side_effect=responses):
            result = prompt_connect_args()
            assert result == {"key": "value"}

    def test_keep_existing(self):
        """测试保留已有参数"""
        existing = {"ssl": True, "timeout": 30}
        with patch("builtins.input", return_value=""):
            result = prompt_connect_args(existing)
            assert result == existing

    def test_replace_existing(self):
        """测试替换已有参数"""
        existing = {"ssl": True}
        responses = ["no", "timeout=60", ""]
        with patch("builtins.input", side_effect=responses):
            result = prompt_connect_args(existing)
            assert result == {"timeout": 60}

    def test_float_value(self):
        """测试浮点数值"""
        responses = ["ratio=0.5", ""]
        with patch("builtins.input", side_effect=responses):
            result = prompt_connect_args()
            assert result == {"ratio": 0.5}


class TestValidateCsvDirectory:
    """测试 validate_csv_directory 函数"""

    def test_nonexistent_dir(self, tmp_path):
        """测试不存在的目录"""
        assert not validate_csv_directory(tmp_path / "nonexistent")

    def test_file_instead_of_dir(self, tmp_path):
        """测试文件而非目录"""
        file_path = tmp_path / "file.txt"
        file_path.touch()
        assert not validate_csv_directory(file_path)

    def test_missing_required_files(self, tmp_path):
        """测试缺少必需文件"""
        assert not validate_csv_directory(tmp_path)

    def test_valid_dir(self, tmp_path):
        """测试有效目录"""
        (tmp_path / "PhysicalTable.csv").touch()
        assert validate_csv_directory(tmp_path)
```

**Step 2: 运行测试**

Run: `uv run pytest tests/cli/test_onboard.py -v`
Expected: 全部通过

**Step 3: 提交**

```bash
git add tests/cli/test_onboard.py
git commit -m "test: add tests for onboard datasource improvements"
```

### Task 5: 最终检查

**Step 1: 运行完整测试套件**

Run: `uv run pytest tests/ -v`
Expected: 全部通过

**Step 2: 运行 lint 检查**

Run: `uv run ruff check src/ tests/`
Expected: 无错误

**Step 3: 运行格式化**

Run: `uv run ruff format src/ tests/`
Expected: 格式化完成
