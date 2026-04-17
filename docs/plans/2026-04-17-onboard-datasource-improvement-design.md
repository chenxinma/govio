# Onboard 数据源配置优化设计

## 概述

优化 `govio-cli onboard` 命令，支持单独配置数据源，改进 connect_args 输入交互。

## 需求

1. 支持跳过 CSV/Graph 配置，仅配置数据源
2. connect_args 改为 key-value 输入方式
3. 优化数据源管理交互（显示列表、支持删除）

## 设计

### 1. onboard 流程增加跳过选项

在 `onboard()` 函数开始处，检查 config 是否存在且包含 backend 配置：
- 若存在，显示提示并询问是否跳过 CSV/Graph 配置
- 跳过时加载已有 config，直接进入数据源配置步骤
- 不跳过时走原有流程

```python
def onboard():
    config_manager = ConfigManager()

    if config_manager.exists():
        existing_config = config_manager.load()
        has_backend = "backend" in existing_config

        if has_backend:
            print("\n⚠️  检测到已有配置（backend: {}）".format(existing_config["backend"]))
            skip = input("是否跳过 CSV/Graph 配置，仅配置数据源？ (yes/no) [默认: no]: ").strip().lower()
            if skip in ("yes", "y"):
                # 跳过流程，直接配置数据源
                full_config = existing_config
                datasources = prompt_datasource_config(existing_config.get("datasources"))
                if datasources is not None:
                    full_config["datasources"] = datasources
                    config_manager.save(full_config)
                print("\n✅ 数据源配置完成！")
                return

        print("\n⚠️  配置文件已存在")
        overwrite = input("是否覆盖现有配置？ (yes/no): ").strip().lower()
        if overwrite not in ["yes", "y"]:
            print("已取消配置")
            return

    # 原有流程...
```

### 2. connect_args 改为 key-value 输入

修改 `prompt_datasource_config()` 中的 connect_args 输入方式：

```python
def prompt_connect_args(existing: dict | None = None) -> dict:
    """交互式输入连接参数（key=value 格式）"""
    connect_args = {}
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
        # 尝试转换布尔值和数字
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

### 3. 数据源管理优化

修改 `prompt_datasource_config()` 支持显示已有数据源和删除操作：

```python
def prompt_datasource_config(existing_datasources: dict | None = None) -> dict | None:
    """提示用户配置数据源（可选）"""
    print("\n=== 数据源配置（可选）===\n")

    datasources = dict(existing_datasources) if existing_datasources else {}

    if datasources:
        print("已配置的数据源:")
        for name, ds in datasources.items():
            print(f"  - {name}: {ds['url']}")

    print("\n操作选项:")
    print("  1. 添加数据源")
    print("  2. 删除数据源")
    print("  3. 完成配置")

    while True:
        choice = input("\n请选择操作 (1/2/3) [默认: 3]: ").strip() or "3"

        if choice == "1":
            # 添加数据源逻辑
            name = input("  数据源名称: ").strip()
            url = input("  URL: ").strip()
            connect_args = prompt_connect_args()
            datasources[name] = {"url": url, "connect_args": connect_args}

        elif choice == "2":
            # 删除数据源逻辑
            if not datasources:
                print("  没有可删除的数据源")
                continue
            print("  选择要删除的数据源:")
            names = list(datasources.keys())
            for i, n in enumerate(names, 1):
                print(f"    {i}. {n}")
            del_choice = input("  输入编号: ").strip()
            try:
                idx = int(del_choice) - 1
                if 0 <= idx < len(names):
                    del datasources[names[idx]]
                    print(f"  已删除: {names[idx]}")
            except (ValueError, IndexError):
                print("  无效选择")

        elif choice == "3":
            break

    return datasources if datasources else None
```

## 文件变更

- `src/govio/cli/onboard.py`: 修改 `onboard()` 和 `prompt_datasource_config()` 函数

## 测试

- 手动测试 onboard 流程
- 验证跳过功能在有/无 config 时的行为
- 验证 connect_args key-value 输入
- 验证数据源删除功能
