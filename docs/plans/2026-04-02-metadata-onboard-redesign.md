# Metadata Onboard 重设计方案

## 目标

将 `utility.py` 中从数据治理平台和应用清单获取基础元数据生成 CSV 的逻辑移到 `onboard` 作为第一步。onboard 的 CLI 以向导交互的形式逐一问询用户相关参数并记录到 `config.yaml`。

## 整体流程

```
govio onboard
    │
    ├─ 步骤 1: CSV 元数据生成（新增）
    │    ├─ 读取 config.yaml 获取默认值
    │    ├─ 问询 kundb / app-list / app-map / relationship
    │    ├─ 生成 CSV 到 csv_dir
    │    └─ 更新 config.yaml（csv_dir）
    │
    ├─ 步骤 2: 图数据库后端选择（现有）
    │    └─ NetworkX / FalkorDB
    │
    ├─ 步骤 3: 后端配置（现有）
    │    └─ 根据后端类型询问配置
    │
    └─ 保存 config.yaml
```

## config.yaml 结构变更

```yaml
csv_dir: "/path/to/csv/output"    # CSV 生成目录
graph_dir: "/path/to/graph/output" # 图文件目录

backend: "networkx"  # 或 "falkordb"
...
```

## 问询参数（步骤 1）

| 参数 | 说明 | 默认值来源 |
|------|------|-----------|
| `kundb` | 元数据库 URL | `config.yaml` |
| `app-list` | 应用清单 Excel 文件路径 | `config.yaml` |
| `app-map` | 应用数据库映射 JSON 文件路径 | `config.yaml` |
| `relationship` | 表关系 JSON 文件路径（可选） | `config.yaml`，无则跳过 |
| `csv-dir` | CSV 输出目录 | `config.yaml` |

**不再使用环境变量 `KUNDB_URL`、`APP_LIST_FILE`、`APP_MAP`。**

## 实现要点

1. **复用 `utility.py` 中的 `make_csv` 函数** - 直接调用，不复制代码
2. **移除 `metadata` 命令** - 从 pyproject.toml 中删除 entry point
3. **config.yaml 路径** - 保持 `~/.govio/config.yaml`

## 实施步骤

1. 修改 `onboard.py`，在现有步骤前插入 CSV 生成步骤
2. 修改 `ConfigManager` 支持 `csv_dir` 字段
3. 在 onboard 步骤 1 中问询参数并调用 `make_csv`
4. 从 pyproject.toml 移除 `metadata` 命令
5. 删除 `utility.py` 中不再使用的 `run()` 函数相关代码
