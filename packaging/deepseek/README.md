# @dsh-external/govio-tools — Govio DeepSeek Harness 插件

把 `govio-cli` 包装为 DeepSeek Harness（DSH）工具的 bundle 插件（Cordis 插件）：

| 工具 | 参数 | 行为 |
|---|---|---|
| `govio_cli` | `args: string[]` | 转发给 `govio-cli`，子命令白名单：`meta`/`query`/`observe`/`sql`/`backend` |
| `govio_query` | `query: string`，`language?: "cypher"\|"python"` | 等价 `govio-cli query -c <query>`，后端由 govio-cli 自适应 |

所有操作通过 `govio-cli` 子进程完成（argv 数组、无 shell、UTF-8 解码），不改动 govio Python 核心。
**skills 不进插件包**（DSH 明确 skill 不用 `dsh plugin add` 安装），按下文「skills 安装」独立落位。

## 前置条件

- `govio-cli` ≥ 0.6.0 在 PATH 中（`uv tool install govio`），可用 `govio-cli -V` 自检
- Node.js ≥ 22（版本线待实测确认）、`pnpm` 在 PATH（`dsh plugin add` 会把参数转发给 pnpm）
- 从源码构建还需 TypeScript（见下文；插件包内已含 `lib/` 编译产物时可跳过）

## 构建（仓库内打包脚本）

```bash
# 产物: dist/govio-dsh-<version>/（lib/ 已编译）
uv run python scripts/package_harness_plugin.py

# 额外生成 dist/govio-dsh-skills/（6 个 skill 目录副本，排除 assets）
uv run python scripts/package_harness_plugin.py --skills-bundle

# 版本号默认取 pyproject.toml 的 project.version，可用 --version 覆盖
uv run python scripts/package_harness_plugin.py --version 0.6.0 --output dist
```

首次从源码编译（`lib/` 缺失时脚本会自动调 `npx tsc -p packaging/deepseek`）：

```bash
cd packaging/deepseek && npm install && cd ../..
```

`npm install` 生成的 `node_modules/` 与 `package-lock.json` 已加入 `.gitignore`，
且打包脚本统一排除（不会进入 `dist/` 产物）。

## 安装

```bash
# 本地目录（开发/内网，推荐从构建产物安装）
dsh plugin --profile web add link:/path/to/govio/dist/govio-dsh-0.6.0
# 或直接从源码目录安装（该目录含 lib/ 编译产物时）
dsh plugin --profile web add link:/path/to/govio/packaging/deepseek

# npm 发布后（可选）
dsh plugin --profile web add @dsh-external/govio-tools@0.6.0
```

**bundle 变更需重启对应 profile 才生效**（重启 `dsh web` / `dsh --profile <name>`）。
安装后挂载行由包自带 `cordis.patch.yml` 并入 profile 层叠，无需手写 profile patch。

验证挂载：

```bash
dsh web --dump-config | grep govio-tools
```

插件配置默认 `cliPath: govio-cli`、`timeoutMs: 120000`；如需覆盖（例如 govio-cli 不在 PATH），
在用户 profile 的 `cordis.patch.yml` 中按行覆盖同 `id` 行（`config` 为整行替换语义，不是深合并）：

```yaml
- insert:
    - id: govio-tools
      name: '@dsh-external/govio-tools'
      inject: [tools]
      config:
        cliPath: /opt/govio/bin/govio-cli
        timeoutMs: 300000
```

## skills 安装（与插件解耦）

skills 由 DSH 从文件系统发现（不走 `dsh plugin add`）。发现根按优先级：
`<项目>/.dsh/skills` > `<项目>/.agents/skills` > customSkillDirs > `~/.dsh/skills` > `~/.agents/skills`。
推荐 `~/.agents/skills/` 或项目 `.agents/skills/`（与 pi.agent 共用一套安装说明，且
`govio-cli meta graph` 默认 assets 路径 `.agents/skills/govio/assets` 天然对齐项目级）。

```bash
# POSIX：用既有 govio-skills.zip（注意 zip 内路径带 skills/ 外层，需去掉）
unzip govio-skills.zip -d /tmp/govio-skills && cp -r /tmp/govio-skills/skills/* ~/.agents/skills/

# 或用打包脚本产出的 skills 副本（无外层 skills/ 目录）
cp -r dist/govio-dsh-skills/* ~/.agents/skills/

# Windows (PowerShell)
Expand-Archive dist\govio-skills.zip -DestinationPath $env:TEMP\govio-skills
Copy-Item $env:TEMP\govio-skills\skills\* $env:USERPROFILE\.agents\skills\ -Recurse
```

布局要求：6 个 skill 目录（`govio`、`govio-meta`、`govio-query`、`govio-observe`、`govio-eda`、
`govio-reconcile`）必须**直接位于 skills 根下**（DSH 不支持递归发现）。

生成 assets（派生数据，任何丢失的兜底动作都是重跑本命令）：

```bash
govio-cli meta graph --output ./data/meta --assets-dir .agents/skills/govio/assets
```

## 升级

```bash
dsh plugin --profile web update @dsh-external/govio-tools
# 重启对应 profile（bundle 变更的生效边界）
```

升级后 skills 若缺失：重新复制（见「skills 安装」）；assets 若丢失：重跑 `govio-cli meta graph`。

## 卸载

```bash
dsh plugin --profile web remove @dsh-external/govio-tools
rm -rf ~/.agents/skills/govio ~/.agents/skills/govio-*   # 手动清理 skills（插件卸载不回收）
```

挂载行随 bundle 层叠移出，无需手写 patch 回写；**若曾手动往 profile 的 `cordis.patch.yml`
加过 govio 行，卸载时需自行删除该行**。

## 版本与兼容性

- 壳包版本 x.y 对应 `govio-cli` ≥ x.y（本包与 `pyproject.toml` 的 `version` 同步）。
- DSH 处于 developer preview、存在破坏性变更：本插件把运行时差异收敛在 `src/index.ts`
  一层 TS 胶水内，DSH 破坏性变更时只改该文件与 `cordis.patch.yml`。
- 类型契约核对基线：`@deepseek-ai/cordis` 4.0.x、`@deepseek-ai/dsh-tools` 0.0.1-rc.x。

## 已知假设与待实测项

- `peerDependencies` 范围与发布 `files` 清单为设计推断（对照官方 workspace 约定），
  待 `dsh plugin add link:` 实测后收敛（见 docs/tasks/T1-distribution-design/research.md「Missing evidence」）。
- Node.js 版本要求（第三方称 22.19+）未获官方原文确认。
- skills frontmatter 的 `allowed-tools` 字段在 DSH 侧的行为（忽略或告警）未验证。
