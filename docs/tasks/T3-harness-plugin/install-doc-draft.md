# docs/install.md 增补草稿 —「DeepSeek Harness 集成」章节（T7 整合用）

> T3 产出。本文件是 `docs/install.md` 的增补草稿，**不是** install.md 本体（T3 无权改动共享文档）。
> T7 整合时建议作为「配置 AI Agent 集成」下的新小节（与现有「下载 Skills 定义 / 复制 Assets」并列），
> 或独立章节；命令与路径已按 Windows/POSIX 双平台给出。
> 与 `packaging/deepseek/README.md` 内容同源，整合时以本文的 install.md 语体为准，避免两处重复可删减 README 交叉引用。

## DeepSeek Harness 集成（可选）

Govio 提供 DeepSeek Harness（`dsh`）bundle 插件 `@dsh-external/govio-tools`，把 `govio-cli`
包装为两个 DSH 工具（`govio_cli`、`govio_query`）；skills 与 assets 沿用上文通用流程，不进插件包。

### 前置条件

- `govio-cli` ≥ 0.6.0 在 PATH（上文「安装 Govio」步骤，可用 `govio-cli -V` 自检）
- Node.js ≥ 22（版本线待实测确认）与 `pnpm` 在 PATH（`dsh plugin add` 会把参数转发给 pnpm）
- 从仓库源码构建还需 TypeScript（`cd packaging/deepseek && npm install`）

### 1. 构建插件包

```bash
# 产物: dist/govio-dsh-<version>/（lib/ 已编译；版本号默认取 pyproject.toml）
uv run python scripts/package_harness_plugin.py

# 需要配套 skills 副本时（6 个 skill 目录，排除 assets）：
uv run python scripts/package_harness_plugin.py --skills-bundle
#   -> dist/govio-dsh-skills/
```

### 2. 安装插件

```bash
# 本地目录（开发/内网；Windows 把路径换成 D:\work\gov-io\govio\dist\govio-dsh-0.6.0）
dsh plugin --profile web add link:/path/to/govio/dist/govio-dsh-0.6.0

# npm 发布后（可选）
dsh plugin --profile web add @dsh-external/govio-tools@0.6.0
```

**bundle 变更需重启对应 profile 才生效**（重启 `dsh web` / `dsh --profile <name>`）。
挂载行由包自带 `cordis.patch.yml` 并入 profile 层叠，无需手写 profile patch。

验证挂载：

```bash
dsh web --dump-config | grep govio-tools
```

默认配置 `cliPath: govio-cli`、`timeoutMs: 120000`；如需覆盖（如 govio-cli 不在 PATH），
在用户 profile 的 `cordis.patch.yml` 按 `id: govio-tools` 整行覆盖（`config` 为整行替换语义，不是深合并）。

### 3. skills 与 assets 落位

skills 由 DSH 从文件系统发现（不走 `dsh plugin add`）。发现根按优先级：
`<项目>/.dsh/skills` > `<项目>/.agents/skills` > customSkillDirs > `~/.dsh/skills` > `~/.agents/skills`。
推荐沿用上文通用方案（`~/.agents/skills/` 或项目 `.agents/skills/`）：

```bash
# 用 dist/govio-dsh-skills/ 副本（无外层 skills/ 目录）
cp -r dist/govio-dsh-skills/* ~/.agents/skills/          # POSIX
# Windows (PowerShell)
Copy-Item dist\govio-dsh-skills\* $env:USERPROFILE\.agents\skills\ -Recurse
```

布局要求：6 个 skill 目录（`govio`、`govio-meta`、`govio-query`、`govio-observe`、`govio-eda`、
`govio-reconcile`）必须**直接位于 skills 根下**（DSH 不支持递归发现）。

assets 生成与核对沿用上文「复制 Assets」约定：

```bash
govio-cli meta graph --output ./data/meta --assets-dir .agents/skills/govio/assets
```

### 4. 升级 / 卸载

```bash
# 升级（换版本后重启 profile）
dsh plugin --profile web update @dsh-external/govio-tools

# 卸载（挂载行随 bundle 层叠自动移出）
dsh plugin --profile web remove @dsh-external/govio-tools
rm -rf ~/.agents/skills/govio ~/.agents/skills/govio-*   # 手动清理 skills（插件卸载不回收）
```

注意：若曾手动往 profile 的 `cordis.patch.yml` 加过 govio 挂载行，卸载时需自行删除该行。
升级/卸载后 assets 若丢失：重跑 `govio-cli meta graph`（派生数据，兜底动作）。

### 兼容性说明

- 壳包版本 x.y 对应 `govio-cli` ≥ x.y。
- DSH 处于 developer preview、可能有破坏性变更；插件把运行时差异收敛在
  `packaging/deepseek/src/index.ts` 一层 TS 胶水内，DSH 破坏性变更时只改该文件与 `cordis.patch.yml`。
