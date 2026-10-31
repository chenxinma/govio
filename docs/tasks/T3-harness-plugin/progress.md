# T3 进度记录 — DeepSeek Harness 插件分发实现

## 目标

按 T1 设计（`docs/tasks/T1-distribution-design/design.md` §3/§4.3/§4.4）实现 govio 的
DeepSeek Harness（DSH）bundle 插件分发形态：插件包目录结构、可重复执行的打包脚本
（产物落 `dist/`、命名不冲突）、产物结构自检。不改动 govio Python 核心与任何共享文件。

## 改动文件清单

| 文件 | 说明 |
|---|---|
| `packaging/deepseek/package.json` | DSH bundle 插件 manifest（`dsh.bundle.patch` 声明），版本由打包脚本注入 |
| `packaging/deepseek/cordis.patch.yml` | bundle patch 挂载行（`id: govio-tools`，config: cliPath/timeoutMs） |
| `packaging/deepseek/src/index.ts` | Cordis 插件实现：`ctx.tools.register(defineTool(...))` 注册 `govio_cli`/`govio_query`，runGovioCli 子进程胶水（argv 数组、无 shell、UTF-8、signal/超时收尾） |
| `packaging/deepseek/tsconfig.json` | ES2022 + NodeNext，outDir `lib/`，声明输出 `lib/types/` |
| `packaging/deepseek/README.md` | 中文安装/构建/skills 落位/升级/卸载说明 + 假设与待实测项 |
| `packaging/deepseek/lib/index.js`、`lib/types/index.d.ts` | tsc 编译产物（保留：打包脚本免 Node 复用；`dsh plugin add link:` 可直接安装） |
| `scripts/package_harness_plugin.py` | 打包脚本：版本注入、统一排除规则、`--skills-bundle`、产物结构自检、可重复执行 |
| `docs/tasks/T3-harness-plugin/test_package_harness_plugin.py` | 打包/排除规则/版本注入/自检/端到端可重复性测试（8 个用例） |
| `docs/tasks/T3-harness-plugin/install-doc-draft.md` | docs/install.md「DeepSeek Harness 集成」章节草稿（T7 整合） |
| `docs/tasks/T3-harness-plugin/progress.md`、`status.json` | 本文件与任务状态 |

**未改动**：`src/govio/**`、`skills/**`（仅被复制）、`package_skills.py`、`pyproject.toml`、
`docs/install.md`、`docs/roadmap.md`、`docs/specs/**`、`tests/` 等共享文件全部只读；
未执行任何 git 写操作。`dist/` 为 gitignore 的构建产物目录，产出
`dist/govio-dsh-0.6.0/`（7 个文件）与 `dist/govio-dsh-skills/`（21 个文件）。

## 与 T1 设计的差异（任务所有权所致，均已在实现中对齐设计语义）

1. **路径命名**：T1 设计为 `packaging/dsh/` + 仓库根 `package_dsh.py`；本任务所有权清单指定
   `packaging/deepseek/` + `scripts/package_harness_plugin.py`，按所有权清单落位。
   产物名仍按设计用 `dist/govio-dsh-<version>/`、`dist/govio-dsh-skills/`（与
   `dist/govio-skills.zip`、`dist/govio-pi-*` 无冲突）。
2. **`@types/node` 追加进 devDependencies**（设计标注 peerDependencies/devDependencies 为
   研究员推断、待实测收敛）：`src/index.ts` 使用 `node:child_process` 与 `AbortSignal`，编译必需。
3. **tsconfig 增设 `declarationDir: lib/types`**：使声明产物正好落进设计 `files` 清单的
   `lib/types/**/*.d.ts`（该清单对照 `@deepseek-ai/dsh-tools` 自身 `lib/index.js + lib/types/index.d.ts` 布局类比）。
4. **`npx tsc` 执行细节**（设计只给了意图）：Windows 下 `npx` 实为 `npx.cmd`，须
   `shutil.which` 解析后 subprocess 才能执行；且必须以插件目录为 CWD，否则 `npx tsc`
   解析到占位 npm 包而非本地 TypeScript。两处均给出中文错误与「先 npm install」提示。

## 实现依据与假设（T1 假设性适配层的收敛情况）

T1 对 DSH 的工具注册契约是「按官方文档骨架 + 待实测」。本任务在实现时**用真实 npm 类型契约做了编译级验证**：

- 从 npm 取得 `@deepseek-ai/cordis@4.0.5`、`@deepseek-ai/dsh-tools@0.0.1-rc.1`（及其 peer 全集，
  含 `@deepseek-ai/dsh-llm`）真实 `.d.ts` 核对：`defineTool({ name, description, parameters, output: { schema, render }, execute(args, exec) })`、
  `execute` 返回规范 JSON 值、`exec.signal: AbortSignal`、`ctx.tools.register()`、
  `Plugin.Function` 的 `name`/`inject`/`apply(ctx, config)` 导出形态、`TextBlock = { type: 'text', text: string }`
  全部与 T1 骨架一致，**无需启用 T1 预留的降级方案**（raw `ToolDefinition` 注册）。
  `src/index.ts` 以 `strict: true` 编译零错误。
- 保留 T1 的假设项（标记待实测）：`peerDependencies: "*"` 范围、发布 `files` 清单、
  `dsh plugin add link:` 安装/卸载回路、Node ≥ 22 版本线、skills frontmatter `allowed-tools`
  在 DSH 侧的行为。风险详见 `packaging/deepseek/README.md`「已知假设与待实测项」与 T1 research.md「Missing evidence」。

## 验证命令与结果

| 验证项 | 命令 | 结果 |
|---|---|---|
| TypeScript 编译（真实 @deepseek-ai/* 类型，strict） | `cd packaging/deepseek && npm install --package-lock=false && npx tsc -p .` | 通过（零错误，产出 `lib/index.js` + `lib/types/index.d.ts`） |
| 打包 + skills 副本 + 自检 | `uv run python scripts/package_harness_plugin.py --skills-bundle` | 通过（`dist/govio-dsh-0.6.0/` 7 文件、`dist/govio-dsh-skills/` 21 文件、自检通过、退出码 0） |
| 可重复执行 | 连续两次运行打包脚本 | 通过（每次先清理旧产物重建，文件集一致，退出码 0） |
| CWD 无关 | 从仓库外目录执行同一脚本 | 通过（退出码 0） |
| 编译触发路径（`lib/` 缺失时自动 `npx tsc`） | 移走 `lib/` 后运行脚本 | 通过（自动编译并打包；Windows npx.cmd 问题已修复） |
| 版本校验 | `--version "bad/version"`、`--version ""` | 均退出码 1 并给出中文错误 |
| 单元/端到端测试 | `uv run python -m pytest docs/tasks/T3-harness-plugin/test_package_harness_plugin.py -v` | 8 passed |
| Lint / 格式 | `uv run ruff check` + `uv run ruff format --check`（本任务两个 Python 文件） | 通过（`logger.error` 处 1 个 `# noqa: TRY400`，理由见注释：CLI 边界友好报错不打堆栈，与 pyproject 对 BLE001/TRY003 的意图性排除一致） |
| 产物命名不冲突 | `ls dist/` | `govio-dsh-0.6.0/`、`govio-dsh-skills/` 与既有 `govio-skills.zip`、`govio-0.6.0-*.whl` 等无冲突 |

## 遗留问题

1. **`dsh plugin add link:` 实测未执行**（本任务范围为实现+打包+自检；环境未装 dsh/pnpm）。
   peerDependencies/files 契约、安装/卸载回路、profile patch 语义均待 T 系列后续按 T1 §7 实测清单收敛。
2. **DSH developer preview 破坏性变更风险**：类型基线为 cordis 4.0.x / dsh-tools 0.0.1-rc.x；
   DSH 升级若破坏 `defineTool`/patch 格式，只改 `src/index.ts` 与 `cordis.patch.yml`（壳层单点收敛）。
3. **测试文件位于任务目录**（`tests/` 为共享目录，不在本任务所有权内）；建议 T7 迁入
   `tests/test_package_harness_plugin.py` 并纳入 `uv run pytest tests/` 常规集。
4. **重新编译需要 Node 环境**：`lib/` 为已保留编译产物；改动 `src/index.ts` 后需在
   `packaging/deepseek` 下 `npm install` 再跑打包脚本（脚本检测到 `lib/` 缺失会自动编译）。
5. `docs/install.md` 与 `docs/specs/README.md` 的正式增补由 T7 按 `install-doc-draft.md` 整合
  （本任务硬约束不改共享文档）。
