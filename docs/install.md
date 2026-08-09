# Govio 安装引导

本文档指导你从零开始安装 Govio 并完成首次配置，包括导入示例元数据和配置 AI Agent 集成。

## 前置条件

- 已安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 包管理器
- Python 3.13+

验证环境：

```bash
uv --version
python --version   # 确认 >= 3.13
```

## 安装 Govio

```bash
uv tool install govio
```

确认安装成功：

```bash
govio-cli -V
```

应输出类似 `govio 0.5.2` 的版本号。

## 初始化配置

运行 onboard 向导，完成图数据库后端和数据源的初始配置：

```bash
govio-cli onboard
```

向导会引导完成两项配置：

1. **选择图数据库后端** — 推荐选择 `ladybug`（嵌入式，无需额外服务），默认数据库路径为 `~/.govio/ontology.lbdb`
2. **配置数据源（可选）** — 供 `observe` 命令使用，初次安装可跳过，后续按需配置

配置保存到 `~/.govio/config.yaml`。

## 导入示例元数据（Chocolate）

Chocolate 是内置的示例数据集，包含 DuckDB 数据库、表间关系和指标定义，可用于快速体验 Govio 的完整功能。

### 1. 下载示例数据

从 [GitHub Releases](https://github.com/chenxinma/govio/releases/download/v0.5.2/chocolate.zip) 下载 `chocolate.zip`，解压到任意目录：

```
/path/to/chocolate/
├── chocolate.db                  # DuckDB 数据文件
├── chocolate_relationships.json  # 表间关系定义
└── chocolate_metrics.json        # 指标维度定义
```

### 2. 导入元数据

进入解压目录，依次执行以下命令：

```bash
cd /path/to/chocolate

# 读取 DuckDB 中的表和字段元数据
govio-cli meta sync meta

# 读取表间关系
govio-cli meta sync rel

# 读取指标和维度定义
govio-cli meta sync metric
```

每条命令会将结果写入当前目录下的 CSV 文件（默认 `./output`）。

### 3. 构建知识图谱

将 CSV 数据写入图数据库并生成 assets：

```bash
govio-cli meta sync graph --mode rebuild
```

成功后会在当前目录生成 `skills/govio/assets/`，包含：

```
skills/govio/assets/
├── schema.md         # 图数据库模式描述
├── metrics_index.md  # 指标索引（原子/派生分组）
├── ontology.gml      # NetworkX GML 数据文件
└── names/            # 节点名称索引（按应用分文件）
```

## 配置 AI Agent 集成

Govio 通过 Skills 为 AI Agent（如 Codex、Claude Code 等）提供数据治理能力。

### 1. 下载 Skills 定义

从 [GitHub Releases](https://github.com/chenxinma/govio/releases/download/v0.5.2/govio-skills.zip) 下载 `govio-skills.zip`，解压 `skills/` 目录到以下位置之一：

| 范围 | 路径 | 说明 |
|------|------|------|
| 项目级 | `<项目目录>/.codex/skills` | 仅当前项目可用 |
| 全局级 | `~/.codex/skills` | 所有项目共享 |

> **注意**：不同 Agent 的 skills 目录可能不同，请参考对应 Agent 的文档。例如 Claude Code 使用 `.claude/skills`，Codex 使用 `.codex/skills`。

### 2. 复制 Assets

将上一步 `meta sync graph` 生成的 assets 复制到 skills 目录中：

```bash
# 项目级
cp -r skills/govio/assets <项目目录>/.codex/skills/govio/

# 或全局级
cp -r skills/govio/assets ~/.codex/skills/govio/
```

确保最终目录结构如下：

```
<skills目录>/govio/
├── SKILL.md
└── assets/
    ├── schema.md
    ├── metrics_index.md
    ├── ontology.gml
    └── names/
        └── *.md
```

完成后即可通过自然语言与 Agent 交互，例如：
- "查询有哪些应用"
- "查找所有包含'客户'的表名"
- "本月账单收入是多少"

## 配置数据源

如需使用 `observe` 命令进行数据探查，需配置数据源连接。重新运行 onboard 向导：

```bash
govio-cli onboard
```

选择跳过图后端配置，进入数据源配置。以 Chocolate 示例为例：

```
数据源名称: chocolate
URL: duckdb:///path/to/chocolate.db
```

URL 格式说明：

| 数据源类型 | URL 格式 | 示例 |
|-----------|---------|------|
| DuckDB 文件 | `duckdb:///path/to/file.db` | `duckdb:///data/chocolate.db` |
| DuckDB 目录 | `duckdb:///path/to/dir` | `duckdb:///data/` |
| MySQL | `mysql+pymysql://user:pass@host/db` | `mysql+pymysql://app:***@localhost/mydb` |

密码会自动加密存储，配置文件可安全分享。

## 常用命令速查

```bash
# 查看版本
govio-cli -V

# 查看当前图后端
govio-cli backend

# 知识图谱查询（Cypher）
govio-cli query -c "MATCH (n:PhysicalTable) RETURN n.name LIMIT 5"

# 数据探查
govio-cli observe load chocolate     # 加载数据源
govio-cli observe explore            # 探索表间关系

# 指标 SQL 组装
govio-cli sql build -f spec.json -o output.sql
```

更多用法请参考 [README](../README.md)。
