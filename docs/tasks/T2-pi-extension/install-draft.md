# docs/install.md 增补草稿 —— pi.agent 集成（T2 产出，供 T7 整合）

> 本文是 `docs/install.md` 的「pi.agent 集成」章节草稿，按 T1 设计（docs/tasks/T1-distribution-design/design.md §2）实现内容撰写。
> T7 整合时请并入 install.md 正文并调整上下文衔接；本文件不直接改动 install.md（任务硬约束）。

## pi.agent 集成（形态 A：Pi package）

Govio 提供 pi.agent 扩展分发包 `@govio/pi-package`：注册 `govio_cli` / `govio_query` 两个 custom tools 与 `/govio` 诊断命令，并携带 6 个 Govio skills。它是 `govio-cli` 之上的薄壳，所有操作通过 `govio-cli` 子进程完成。

### 前置条件

- `govio-cli` ≥ 0.6.0 在 PATH（`uv tool install govio`，验证：`govio-cli -V`）；也可用环境变量 `GOVIO_CLI` 指定可执行文件路径。
- pi（pi-coding-agent）已安装。

### 构建与安装

```bash
# 构建（仓库根目录）
uv run scripts/package_pi_extension.py
# 产物：dist/govio-pi-<version>/（含 skills 副本，已排除 skills/govio/assets 派生数据）

# 安装（三选一）
pi install ./dist/govio-pi-0.6.0                        # 本地目录
pi install git:github.com/chenxinma/govio@v0.6.0        # git 源（子目录包支持待实测）
pi install npm:@govio/pi-package@0.6.0                  # npm 发布后（可选）
```

安装后 `pi list` 确认包已登记；会话内运行 `/govio` 自检（govio-cli 版本 / 图后端 / assets 约定位置）。

### skills 与 assets 落位

assets 是派生数据，由 `govio-cli meta graph` 生成，永远落在 `<skills-root>/govio/assets`。推荐项目级 `.agents/skills`（与 meta graph 默认输出对齐）：

```bash
mkdir -p .agents/skills
cp -r dist/govio-pi-0.6.0/skills/* .agents/skills/    # 目标必须是 .agents/skills/govio/SKILL.md 等 6 个目录
govio-cli meta graph --output ./data/meta --assets-dir .agents/skills/govio/assets
# 命令打印 assets 绝对路径，请核对等于 <skills-root>/govio/assets
```

零拷贝方案（R2）：直接用包内 skills（pi 自动发现），assets 生成进包内目录。npm/git 源更新时包内 assets 可能被重建，丢失后重跑 `meta graph` 即可。

### 工具与命令

| 能力 | 名称 | 说明 |
|---|---|---|
| custom tool | `govio_cli` | `args: string[]` 转发 `govio-cli`；子命令白名单 meta/query/observe/sql/backend |
| custom tool | `govio_query` | `query: string`（可选 `language: cypher\|python` 语法提示）；等价 `govio-cli query -c <query>` |
| 诊断命令 | `/govio` | govio-cli 版本、当前图后端、assets 约定位置 |

工具执行默认超时 120 秒，尊重 pi 取消信号；退出码非 0 时返回含 stderr 摘要的错误结果。

### 升级 / 卸载

```bash
pi update --extensions          # 对账更新
pi remove <source>              # 卸载（source 与 pi list 显示一致）
rm -rf .agents/skills/govio*    # 手动清理已复制的 skills
```

升级/更新后 assets 若丢失，重跑 `meta graph`（派生数据兜底）。

### 版本对应

壳包版本 x.y 对应 govio-cli ≥ x.y；版本号由 `scripts/package_pi_extension.py` 从 `pyproject.toml` 注入。

### 已知限制（待实测收敛）

1. pi git 源对仓库子目录包（`packaging/pi`）的支持未知；不支持时改为 clone 后 `pi install ./govio/packaging/pi` 或以 npm 发布为准。
2. R2 方案包内 assets 在 `pi update --extensions` 后的存续行为未实测；兜底为重跑 `meta graph`。
