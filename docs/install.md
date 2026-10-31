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

应输出类似 `govio 0.7.0` 的版本号。

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

从 [GitHub Releases](https://github.com/chenxinma/govio/releases/download/v0.7.0/chocolate.zip) 下载 `chocolate.zip`，解压到任意目录：

```
/path/to/chocolate/
├── chocolate.db                  # DuckDB 数据文件
├── chocolate_relationships.json  # 表间关系定义
└── chocolate_metrics.json        # 指标维度定义
```

### 2. 导入元数据

进入解压目录，依次执行以下命令（各子命令独立运行，输入全部通过参数显式传入）：

```bash
cd /path/to/chocolate

# 读取 DuckDB 中的表和字段元数据（--datasource、--schemas 必填；--schemas 写源库真实 schema 名）
govio-cli meta meta --source duckdb --db ./chocolate.db --datasource chocolate --schemas chocolate --output ./output

# 读取表间关系
govio-cli meta rel --file ./chocolate_relationships.json --output ./output

# 读取指标和维度定义
govio-cli meta metric --file ./chocolate_metrics.json --output ./output
```

每条命令会将结果写入 `--output` 目录下的 CSV 文件（默认 `./output`）。该目录是子命令之间唯一的共享状态：**多次导入请复用同一目录**，已有内容会增量合并（幂等）。

> schema 名写错或源库无表时，命令会失败退出（不写任何 CSV），并在错误信息中列出该库可导入的 schema。

> 示例用 DuckDB；`meta meta --source` 还支持 `mysql` / `postgres` / `trino`（`--db` 为连接 URL，详见 [specs/cli.md](specs/cli.md)）。

### 3. 构建知识图谱

将 CSV 数据写入图数据库并生成 assets：

```bash
govio-cli meta graph --output ./output --mode rebuild
```

成功后会在当前目录生成 `.agents/skills/govio/assets/`（可用 `--assets-dir` 指定其他目录；命令结束会打印其**绝对路径**），包含：

```
.agents/skills/govio/assets/
├── schema.md         # 图数据库模式描述
├── metrics_index.md  # 指标索引（原子/派生分组）
├── ontology.gml      # NetworkX GML 数据文件
└── names/            # 节点名称索引（按应用分文件）
```

## 配置 AI Agent 集成

Govio 通过 Skills 为 AI Agent 提供数据治理能力。govio 本体不含 LLM，
可插拔任意 Agent Runtime（Codex、Claude Code、pi、DeepSeek harness 等）。
按宿主选择一种分发形态：

| 形态 | 适用宿主 | 来源 / 产物 | 说明 |
|------|---------|------------|------|
| 一：Skills 压缩包 | 任意 Agent（Codex、Claude Code 等） | `govio-skills.zip` | 通用基线，只有 skills |
| 二：pi.agent 扩展 | pi（pi-coding-agent） | `dist/govio-pi-<version>/` | skills + `govio_cli`/`govio_query` 工具 + `/govio` 诊断命令 |
| 三：DeepSeek Harness 插件 | DeepSeek harness（`dsh`） | `dist/govio-dsh-<version>/` | skills + 同一工具面契约 |

三种形态共享同一约定：

- **skills 单一真源**：6 个 skill（`govio`、`govio-meta`、`govio-query`、`govio-observe`、`govio-eda`、`govio-reconcile`）必须**直接位于 skills 根下**（不支持递归发现）；
- **assets 落位 `<skills-root>/govio/assets`**：assets 是派生数据，由 `govio-cli meta graph --assets-dir <skills-root>/govio/assets` 生成，任何安装/升级流程丢失 assets 的兜底动作都是重跑 `meta graph`。

### 形态一：Skills 压缩包（通用）

#### 1. 下载 Skills 定义

从 [GitHub Releases](https://github.com/chenxinma/govio/releases/download/v0.7.0/govio-skills.zip) 下载 `govio-skills.zip`，解压 `skills/` 目录到以下位置之一：

| 范围 | 路径 | 说明 |
|------|------|------|
| 项目级 | `<项目目录>/.agents/skills` | 仅当前项目可用（推荐，与 `meta graph` 默认输出对齐） |
| 全局级 | `~/.agents/skills` | 所有项目共享 |

> **注意**：不同 Agent 的 skills 目录可能不同，请参考对应 Agent 的文档。例如 Claude Code 使用 `.claude/skills`，Codex 使用 `.codex/skills`。目标布局必须是 `<skills目录>/govio/SKILL.md` 等 6 个目录直接平铺。

#### 2. 复制 Assets

将上一步 `meta graph` 生成的 assets 复制到 skills 目录中（两者路径不一致时以命令打印的绝对路径为准）：

> **注意**：若 assets 已经位于目标 `<skills目录>/govio/assets`（如项目级布局下 `meta graph` 用默认 `--assets-dir` 生成时），**跳过复制**——源与目标同目录执行 `cp -r` 会报 `cannot copy a directory into itself`。以下命令仅在 assets 与 skills 目录不在同一位置（如全局级安装、或 `--assets-dir` 指向了别处）时需要：

```bash
# 项目级（assets 不在 <项目目录>/.agents/skills/govio/ 下时）
cp -r <assets绝对路径> <项目目录>/.agents/skills/govio/

# 或全局级
cp -r <assets绝对路径> ~/.agents/skills/govio/
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

### 形态二：pi.agent 集成

Govio 提供 pi.agent 扩展分发包 `@govio/pi-package`：注册 `govio_cli` / `govio_query` 两个 custom tools 与 `/govio` 诊断命令，并携带 6 个 Govio skills。它是 `govio-cli` 之上的薄壳，所有操作通过 `govio-cli` 子进程完成。

#### 前置条件

- `govio-cli` ≥ 0.7.0 在 PATH（上文「安装 Govio」步骤，验证：`govio-cli -V`）；也可用环境变量 `GOVIO_CLI` 指定可执行文件路径
- pi（pi-coding-agent）已安装

#### 构建与安装

```bash
# 构建（仓库根目录）
uv run scripts/package_pi_extension.py
# 产物：dist/govio-pi-<version>/（含 skills 副本，已排除 skills/govio/assets 派生数据）

# 安装（三选一）
pi install ./dist/govio-pi-0.7.0                        # 本地目录
pi install git:github.com/chenxinma/govio@v0.7.0        # git 源（子目录包支持待实测）
pi install npm:@govio/pi-package@0.7.0                  # npm 发布后（可选）
```

安装后 `pi list` 确认包已登记；会话内运行 `/govio` 自检（govio-cli 版本 / 图后端 / assets 约定位置）。

#### skills 与 assets 落位

推荐项目级 `.agents/skills`（与 `meta graph` 默认输出对齐）：

```bash
mkdir -p .agents/skills
cp -r dist/govio-pi-0.7.0/skills/* .agents/skills/    # 目标必须是 .agents/skills/govio/SKILL.md 等 6 个目录
govio-cli meta graph --output ./output --assets-dir .agents/skills/govio/assets
# 命令打印 assets 绝对路径，请核对等于 <skills-root>/govio/assets
```

> **前置条件**：`--output` 必须指向已由 `meta meta` / `meta rel` / `meta metric` 写出 CSV 的**已存在**目录（与上文 Chocolate 流程的 `./output` 一致）。`meta graph` 对不存在的目录会报 `❌ 输出目录不存在` 并退出，不会自动创建。

零拷贝方案：直接用包内 skills（pi 自动发现），assets 生成进包内目录。npm/git 源更新时包内 assets 可能被重建，丢失后重跑 `meta graph` 即可。

#### 工具与命令

| 能力 | 名称 | 说明 |
|---|---|---|
| custom tool | `govio_cli` | `args: string[]` 转发 `govio-cli`；子命令白名单 meta/query/observe/sql/backend |
| custom tool | `govio_query` | `query: string`（可选 `language: cypher\|python` 语法提示）；等价 `govio-cli query -c <query>` |
| 诊断命令 | `/govio` | govio-cli 版本、当前图后端、assets 约定位置 |

工具执行默认超时 120 秒，尊重 pi 取消信号；退出码非 0 时返回含 stderr 摘要的错误结果。
`govio_cli query` 会在会话工作目录产生 `.govio/output-*.json` 结果文件（govio-cli 既有行为），可自行清理。

#### 升级 / 卸载

```bash
pi update --extensions          # 对账更新
pi remove <source>              # 卸载（source 与 pi list 显示一致）
rm -rf .agents/skills/govio*    # 手动清理已复制的 skills
```

升级/更新后 assets 若丢失，重跑 `meta graph`（派生数据兜底）。

#### 版本对应与已知限制

- 壳包版本 x.y 对应 govio-cli ≥ x.y；版本号由 `scripts/package_pi_extension.py` 从 `pyproject.toml` 注入。
- pi git 源对仓库子目录包（`packaging/pi`）的支持待实测；不支持时 clone 后 `pi install ./govio/packaging/pi` 或以 npm 发布为准。
- 零拷贝方案包内 assets 在 `pi update --extensions` 后的存续行为未实测；兜底为重跑 `meta graph`。

完整说明见 [packaging/pi/README.md](../packaging/pi/README.md)。

### 形态三：DeepSeek Harness 集成

Govio 提供 DeepSeek Harness（`dsh`）bundle 插件 `@dsh-external/govio-tools`，把 `govio-cli`
包装为两个 DSH 工具（`govio_cli`、`govio_query`）；skills 与 assets 沿用共享约定，不进插件包。

#### 前置条件

- `govio-cli` ≥ 0.7.0 在 PATH（上文「安装 Govio」步骤，可用 `govio-cli -V` 自检）
- Node.js ≥ 22（版本线待实测确认）与 `pnpm` 在 PATH（`dsh plugin add` 会把参数转发给 pnpm）
- 从仓库源码构建还需 TypeScript（`cd packaging/deepseek && npm install`）

#### 1. 构建插件包

```bash
# 产物: dist/govio-dsh-<version>/（lib/ 已编译；版本号默认取 pyproject.toml）
uv run python scripts/package_harness_plugin.py

# 需要配套 skills 副本时（6 个 skill 目录，排除 assets）：
uv run python scripts/package_harness_plugin.py --skills-bundle
#   -> dist/govio-dsh-skills/
```

#### 2. 安装插件

```bash
# 本地目录（开发/内网；Windows 把路径换成 D:\work\gov-io\govio\dist\govio-dsh-0.7.0）
dsh plugin --profile web add link:/path/to/govio/dist/govio-dsh-0.7.0

# npm 发布后（可选）
dsh plugin --profile web add @dsh-external/govio-tools@0.7.0
```

**bundle 变更需重启对应 profile 才生效**（重启 `dsh web` / `dsh --profile <name>`）。
挂载行由包自带 `cordis.patch.yml` 并入 profile 层叠，无需手写 profile patch。

验证挂载：

```bash
dsh web --dump-config | grep govio-tools
```

默认配置 `cliPath: govio-cli`、`timeoutMs: 120000`；如需覆盖（如 govio-cli 不在 PATH），
在用户 profile 的 `cordis.patch.yml` 按 `id: govio-tools` 整行覆盖（`config` 为整行替换语义，不是深合并）。

#### 3. skills 与 assets 落位

skills 由 DSH 从文件系统发现（不走 `dsh plugin add`）。发现根按优先级：
`<项目>/.dsh/skills` > `<项目>/.agents/skills` > customSkillDirs > `~/.dsh/skills` > `~/.agents/skills`。
推荐沿用共享约定（`~/.agents/skills/` 或项目 `.agents/skills/`）：

```bash
# 用 dist/govio-dsh-skills/ 副本（无外层 skills/ 目录）
cp -r dist/govio-dsh-skills/* ~/.agents/skills/          # POSIX
# Windows (PowerShell)
Copy-Item dist\govio-dsh-skills\* $env:USERPROFILE\.agents\skills\ -Recurse
```

布局要求：6 个 skill 目录必须**直接位于 skills 根下**（DSH 不支持递归发现）。
assets 生成与核对沿用共享约定：

```bash
govio-cli meta graph --output ./output --assets-dir .agents/skills/govio/assets
# 同形态二：--output 指向已写出 meta CSV 的已存在目录（不存在会报「输出目录不存在」并退出）
```

#### 4. 升级 / 卸载

```bash
# 升级（换版本后重启 profile）
dsh plugin --profile web update @dsh-external/govio-tools

# 卸载（挂载行随 bundle 层叠自动移出）
dsh plugin --profile web remove @dsh-external/govio-tools
rm -rf ~/.agents/skills/govio ~/.agents/skills/govio-*   # 手动清理 skills（插件卸载不回收）
```

注意：若曾手动往 profile 的 `cordis.patch.yml` 加过 govio 挂载行，卸载时需自行删除该行。
升级/卸载后 assets 若丢失：重跑 `govio-cli meta graph`（派生数据，兜底动作）。

#### 兼容性说明

- 壳包版本 x.y 对应 `govio-cli` ≥ x.y。
- DSH 处于 developer preview、可能有破坏性变更；插件把运行时差异收敛在
  `packaging/deepseek/src/index.ts` 一层 TS 胶水内，DSH 破坏性变更时只改该文件与 `cordis.patch.yml`。

完整说明见 [packaging/deepseek/README.md](../packaging/deepseek/README.md)。

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
| PostgreSQL | `postgresql+psycopg://user:pass@host:5432/db` | `postgresql+psycopg://app:***@localhost/mydb` |
| Trino | `trino://user:pass@host:8080/catalog` | `trino://app:***@trino-host:8080/hive` |

> URL scheme 大小写不敏感且自动归一：裸 `mysql://` / `postgres://` / `postgresql://`、以及 `trino+http://` / `trino+https://` 均可直接使用（分别归一到对应驱动与 HTTP/HTTPS 协议，详见 [specs/config.md](specs/config.md)）。

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
govio-cli observe load --name t1 --datasource chocolate --sql "SELECT * FROM t1"  # 加载 DataFrame
govio-cli observe explore            # 探索表间关系

# 指标 SQL 组装
govio-cli sql build -f spec.json -o output.sql
```

更多用法请参考 [README](../README.md)。
