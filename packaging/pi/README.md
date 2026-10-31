# @govio/pi-package — Govio Pi 工具与技能包

Govio 数据治理知识图谱的 [pi.agent](https://pi.dev) 扩展分发包（分发形态 A）：

- **custom tools**：`govio_cli`（转发 `govio-cli` 子命令，白名单 meta/query/observe/sql/backend）、`govio_query`（知识图谱查询）；
- **skills**：6 个 Govio 工作流技能（govio / govio-meta / govio-query / govio-observe / govio-eda / govio-reconcile），遵循 Agent Skills 规范；
- **诊断命令**：`/govio`（检查 govio-cli 版本、图后端、assets 约定位置）。

本包是 `govio-cli` 之上的薄壳，不包含 Python 核心逻辑；所有操作通过 `govio-cli` 子进程完成（argv 数组，无 shell）。

## 前置条件

1. **govio-cli ≥ 0.7.0 在 PATH**（与本包版本 x.y 对应 govio-cli ≥ x.y）：

   ```bash
   uv tool install govio
   govio-cli -V        # 应输出 govio <版本>
   ```

   也可用环境变量 `GOVIO_CLI` 指定 govio-cli 可执行文件路径。

2. **pi（pi-coding-agent）已安装**，Node 运行时可用。

## 构建产物

本包随 govio 仓库分发。仓库根目录执行：

```bash
uv run scripts/package_pi_extension.py
```

产物为 `dist/govio-pi-<version>/`（含 skills 副本，已排除 `skills/govio/assets` 派生数据），可直接安装：

```bash
pi install ./dist/govio-pi-0.7.0
```

也可以直接从源码目录取包（skills 副本由打包脚本同步生成）：

```bash
uv run scripts/package_pi_extension.py   # 生成 packaging/pi/skills/ 副本
pi install ./packaging/pi
```

## 安装（三选一）

```bash
# 本地目录（开发/内网）
pi install ./dist/govio-pi-0.7.0

# git 源（发布 tag 后；子目录包如不被 git 源支持，见下方「待实测」）
pi install git:github.com/chenxinma/govio@v0.7.0

# npm 发布后（可选）
pi install npm:@govio/pi-package@0.7.0
```

安装后用 `pi list` 确认包已登记，在 pi 会话中运行 `/govio` 验证扩展加载与 govio-cli 可用性。

### skills + assets 落位（二选一，推荐 R1）

assets 是派生数据，由 `govio-cli meta graph` 生成，永远落在 `<skills-root>/govio/assets`。

**R1（推荐）：项目级 `.agents/skills`** —— 与 `meta graph` 默认输出路径天然对齐：

```bash
mkdir -p .agents/skills
# POSIX
cp -r dist/govio-pi-0.7.0/skills/* .agents/skills/
# Windows (PowerShell)
Copy-Item -Recurse dist\govio-pi-0.7.0\skills\* .agents\skills\

# 生成 assets（默认即 .agents/skills/govio/assets，命令会打印绝对路径，请核对）
govio-cli meta graph --output ./data/meta --assets-dir .agents/skills/govio/assets
```

目标布局必须是 `.agents/skills/govio/SKILL.md` 等 6 个目录直接位于根下（不要多套一层 `skills/`）。
也可以直接解压既有 `dist/govio-skills.zip` 得到同样内容。

**R2（零拷贝）：直接用包内 skills**（pi 自动发现），assets 生成进包内：

```bash
# 具体路径以 pi 安装位置为准（本地源安装即源目录，路径待实测）
govio-cli meta graph --output ./data/meta --assets-dir <包目录>/skills/govio/assets
```

注意：npm/git 源安装的包在 `pi update --extensions` 对账更新时包内内容可能被重建，R2 生成的 assets 可能丢失；丢失后兜底动作是重跑 `meta graph`（派生数据，随时可再生）。

## 工具说明

| 能力 | 名称 | 说明 |
|---|---|---|
| custom tool | `govio_cli` | 参数 `args: string[]`，转发给 `govio-cli`；子命令限 meta/query/observe/sql/backend，越权或空参数直接拒绝 |
| custom tool | `govio_query` | 参数 `query: string`、`language?: "cypher"\|"python"`（语法提示用，实际由图后端决定）；等价 `govio-cli query -c <query>` |
| 诊断命令 | `/govio` | 显示 govio-cli 版本、当前图后端、assets 约定位置 |

高频能力（meta、observe、sql）通过 skills 指令 + `govio_cli` 工具完成。工具执行默认超时 120 秒，尊重 pi 的取消信号。

## 升级

```bash
pi update --extensions                      # 对账更新已装包（git/npm 源）
pi remove <source> && pi install <new-source>   # 换版本（source 与 pi list 显示一致）
```

升级后 skills 若已有新版本，重新执行 R1 的复制步骤；assets 若丢失，重跑 `meta graph`。

## 卸载

```bash
pi remove <source>          # <source> 与 pi list 显示一致
rm -rf .agents/skills/govio*    # 手动清理 skills（包卸载不回收已复制的 skills）
# Windows (PowerShell): Remove-Item -Recurse .agents\skills\govio*
```

## 待实测（已知残留）

1. pi git 源对仓库**子目录**包（`packaging/pi`）的支持未知；若不支持，改为「clone 仓库后 `pi install ./govio/packaging/pi`」或以 npm 发布为准。
2. R2 方案中包内 assets 在包更新后是否被清除未实测；已给兜底（重跑 `meta graph`）。

## 版本对应

本包版本 x.y 对应 govio-cli ≥ x.y（版本由 `scripts/package_pi_extension.py` 从 `pyproject.toml` 注入）。
