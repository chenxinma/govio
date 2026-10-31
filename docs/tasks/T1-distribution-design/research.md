# 调研笔记：govio 新分发形态（pi.agent 扩展 / DeepSeek Harness 插件）

> T1 产出。供 T2（pi.agent 扩展）/ T3（DeepSeek Harness 插件）直接照做。
> 证据标注约定：**[直接证据]** 原文明确陈述；**[来源解读]** 基于原文的合理解读；**[研究员推断]** 未获直接证据的推断，仅供设计参考。

## 结论速览

1. **pi.agent 扩展形态有官方机制**：Pi Packages（`pi install`）可以把 extensions + skills + prompts + themes 打成一个包分发；扩展是 TypeScript 模块（`ExtensionAPI` 注册 custom tools / commands / MCP server），skills 遵循 Agent Skills 规范，安装/更新/卸载由 `pi install / pi update --extensions / pi remove` 管理。**[直接证据]**
2. **DeepSeek Harness（dsh）插件形态有官方机制**：DeepSeek 开源了 agent harness "DeepSeek Harness"（`dsh`，Cordis "everything-is-a-plugin" 架构，developer preview）。插件 = Cordis 插件 + `package.json` 的 `dsh` 字段；bundle 插件经 `dsh plugin --profile <name> add <包或 git 源>`（pnpm 转发）安装。**[直接证据]**
3. **skills 在两个 runtime 都不需要"注册代码"**：都是文件系统发现（SKILL.md 目录），且都原生支持 `~/.agents/skills/`、`.agents/skills/`（Agent Skills 位置）。govio 现有 6 个 skill 全部 kebab-case、Agent Skills 兼容，可近乎零改动复用。**[直接证据+来源解读]**
4. **govio 无需改动 Python 核心**：两种形态都是 govio-cli 之上的"壳"。工具层统一以「子进程调用 govio-cli」为适配层，runtime API 漂移只影响一层薄 TS 胶水。**[研究员推断，基于两 runtime 的工具契约]**

---

## 1. pi.agent 扩展机制

来源（本机安装的 pi-coding-agent 1.4.0 发行包内官方文档，路径
`C:/Users/Administrator/.pi/agent/install/releases/1.4.0/node_modules/@earendil-works/pi-coding-agent/`）：

### 1.1 扩展本体

- 扩展是 TypeScript/JavaScript 模块，导出默认工厂函数，接收 `ExtensionAPI`，在其中注册能力。jiti 直接加载 TS，无需编译。**[直接证据]**（`docs/extensions.md`）
- 加载位置：用户目录 `<agent-dir>/extensions/`（默认 `~/.pi/agent/extensions/`）、项目 `.pi/extensions/`；单文件或含 `index.ts/index.js` 的子目录；npm 依赖放附近 `package.json`。开发期可 `pi --extension ./hello.ts` 直载。**[直接证据]**（`docs/extensions.md`、`docs/configuration.md`）
- 集成点（与 govio 相关的）：`pi.registerTool()`（模型可调用工具，TypeBox 参数 schema + `execute()`，结果需 `content` + `details`）、`pi.registerCommand()`（`/` 命令）、`pi.registerMcpServer(name, config)`（MCP server，配置形状同 `mcp.json` 条目）、`pi.registerFlag()` 等。**[直接证据]**
- 工具注册最小示例见 `examples/extensions/hello.ts`：`defineTool({ name, label, description, parameters: Type.Object(...), execute(...) })`（`defineTool` 来自 `@earendil-works/pi-coding-agent`，`Type` 来自 `@earendil-works/pi-ai`），工厂内 `pi.registerTool(helloTool)`。**[直接证据]**
- 生命周期：工厂内勿起进程/定时器，长资源从 `session_start` 或命令/工具里起，`session_shutdown` 幂等清理。非交互模式（RPC/JSON/print）无完整 TUI，工具行为要与渲染解耦。**[直接证据]**

### 1.2 包分发（Pi Packages）——分发形态 A 的载体

- 包是普通目录或 npm 包，约定目录 `extensions/ skills/ prompts/ themes/`（无 manifest 时自动发现），或在 `package.json` 的 `pi` 键显式声明（数组收 glob，支持 `!` 排除）。**[直接证据]**（`docs/packages.md`）
- 安装源：`npm:@scope/name@ver`（钉版本）、`git:github.com/...@tag`、URL（按 git 处理）、本地路径（不复制、原地加载）。安装命令 `pi install <source>`；`pi list` 查看；`pi remove <source>` 卸载；`pi update --extensions` 对账更新；`pi -e <source>` 单次试用。**[直接证据]**
- 声明落点：个人安装写 `~/.pi/agent/settings.json`；`--local` 写项目 `.pi/settings.json`（需 project trust 后加载）。对象形式可按资源类型过滤（如 `skills: []` 不加载包内 skills）。**[直接证据]**
- 依赖规则：扩展 import 的运行时 npm 包放 `dependencies`；宿主供给包（`@earendil-works/pi-ai`、`pi-agent-core`、`pi-coding-agent`、`pi-tui`、`typebox`）放 `peerDependencies: {"*": ...}`，不得打进 dependencies。**[直接证据]**
- 包更新用 `pi update --extensions` 对账（git ref 不自动漂移）。**[直接证据]**

### 1.3 skills（Pi 侧）

- skill = 含 `SKILL.md` 的目录，frontmatter（name/description/allowed-tools/disable-model-invocation 等）+ 指令正文；可捆绑 scripts/references/assets。Pi 实现 Agent Skills 规范（agentskills.io）。**[直接证据]**（`docs/skills.md`）
- 发现位置：用户 `<agent-dir>/skills/`、项目 `.pi/skills/`，以及 Agent Skills 位置 `~/.agents/skills/`、`.agents/skills/`（项目级从工作目录向上发现到仓库根）；包内 `skills/` 目录经 packages 机制发现。**[直接证据]**
- 启动时只把 name + description 注入系统提示，正文按需加载；`/skill:name` 强制加载。name 建议与目录名一致（其他 Agent Skills 实现可能强制）。**[直接证据]**

## 2. DeepSeek Harness（dsh）插件机制

主要来源：GitHub 官方仓库 `deepseek-ai/deepseek-harness`（fetch 时浅克隆到本地，读取了仓库内文档原文）：
`README.md`、`docs/subsystems/skills.md`、`docs/cookbook/extension-cookbook.md`、`docs/cookbook/adding-a-tool.md`、`apps/cli/reference/README.md`、`packages/bundle/sdk-minimal/cordis.patch.yml`。

### 2.1 存在性与状态

- DeepSeek Harness（`dsh`）是 DeepSeek 开源的 agent harness，"everything-is-a-plugin"（Cordis 架构）；README 明示 **developer preview、会有破坏性变更**；npm 启动 `npx @deepseek-ai/dsh web`；官方文档站 `https://deepseek-harness.github.io/deepseek-harness/`。**[直接证据]**（README）
- 参考版本线索：仓库 tag `dsh-v0.2.1-alpha.4`；第三方文档镜像标注审计基线 `0.1.5-alpha.1`。**[直接证据（tag）/来源解读（基线）]**

### 2.2 插件形态与安装

- 插件 = Cordis 插件（`export const name`、`export const inject`、`export function apply(ctx, config)`，或 `class extends Service`）+ `package.json` 的 `dsh` 字段。bundle 形态声明 `"dsh": { "bundle": { "patch": "./cordis.patch.yml" } }`，其 patch 层在安装后并入 profile 的层叠。**[直接证据]**（`apps/cli/reference/README.md`；插件解剖细节见第三方镜像 deepseekdocs.com，与原文一致）
- 安装：`dsh plugin --profile <name> add <package-or-git-spec>`，把参数**原样转发给 pnpm**（add/remove/update 等均可用，pnpm 须在 PATH）；装完后 **bundle 变更需重启 profile** 才生效（profile 的 `cordis.patch.yml` 行内修改可热重载）。**[直接证据]**（`apps/cli/reference/README.md`）
- 对账语义：`dsh.profile.bundles` 按「已安装状态」对账——凡依赖解析到声明 `dsh.bundle` 的包即自动加入层叠（`update` 获得该声明也会激活）；移除依赖即移出层叠。**[直接证据]**
- 卸载：`dsh plugin --profile <name> remove <package>`。注意：remove 不回写 profile 的 `cordis.patch.yml`；**若把插件行写进包自带 bundle patch（而非让用户手写 profile patch），卸载即随层叠移除，无残留**。**[直接证据+研究员推断]**
- `cordis.patch.yml` 语义：顶层 YAML 数组，`insert:` 追加行（`id/name/inject/config/disabled`），或按 `id` 整行覆盖既有行；`config` 是整行替换不是深合并；`name` 不匹配→告警并跳过。应用顺序：bundle patches → profile patch → `$DSH_HOME/cordis.patch.yml` → `--patch` 覆盖。**[直接证据]**（`packages/bundle/sdk-minimal/cordis.patch.yml` 行格式；语义细节来自镜像并与 CLI 参考一致）
- 工具注册契约：`ctx.tools.register(defineTool({ name, description, parameters, output: {schema, render}, execute(args, exec) }))`，`inject = ['tools']`；参数自动校验；`execute` 返回规范 JSON 值，抛错即 `isError`；必须尊重 `exec.signal`；schema 自动进系统提示。**[直接证据]**（`docs/cookbook/adding-a-tool.md`）
- 模块解析注意：框架 vendored 为 `@deepseek-ai/*`（如 `@deepseek-ai/cordis`），裸名 `cordis`/`schemastery` 解析不到，import 必须用 `@deepseek-ai/` 前缀。**[直接证据]**（README/镜像一致）

### 2.3 skills（DSH 侧）

- **skills 不通过 `dsh plugin add` 安装**（"You do not install a Skill with `dsh plugin add`"）；skills 由 `dsh-skill-filesystem` 从文件系统根发现，注册表分层合并。**[直接证据]**（`docs/subsystems/skills.md`）
- 发现根（按优先级）：`<projectRoot>/.dsh/skills`（100）→ `<projectRoot>/.agents/skills`（200）→ `Config.customSkillDirs`（300）→ `<dshHome>/skills`（400，`~/.dsh/skills`）→ `<agentsHome>/skills`（500，`~/.agents/skills`）→ bundled（600）。projectRoot = 最近的含 `.git` 祖先。**[直接证据]**
- skill 形态：目录 bundle `<name>/SKILL.md` 或扁平 `<name>.md`；**不支持嵌套递归 `**/SKILL.md` 发现**（skill 必须直接位于根下）；name 必须 kebab-case；frontmatter 读取 `disable-model-invocation`、`user-invocable`。**[直接证据]**
- 模型经 `skill({name})` 工具按需加载正文，`resourceBase` 解析 skill 内相对引用的脚本/资源。**[直接证据]**

### 2.4 `.dsh-plugin` 仓库插件（备选形态）

- 第三方文档（deepseekdocs.com，标注"逐条对照官方源码验证"）：bundle 插件（`dsh plugin add "github:..."`，重启生效）与**仓库插件 `.dsh-plugin`**（Plugin 面板 → repository source，`github:owner/repo#ref&path:...`，即时生效）是两种安装形态；后者由第三方 `@dsh-external/plugin-console` + `$DSH_HOME/cordis.patch.yml` 实现，**不属于 DSH 内核**。`@deepseek-ai/*` 为内核包，`@dsh-external/*` 为第三方社区包。**[来源解读（第三方镜像，未经官方原文二次确认）]**
- 设计取舍：govio 采用 **bundle 插件**（内核机制、可 pin 版本），不依赖第三方 plugin-console。**[研究员推断]**

### 2.5 source_check 验证说明

对「dsh 插件安装/技能发现/工具注册」组合断言执行了 `source_check`，自动语义评估不可用（status: unclear，confidence 0.30）。按预案改为**直接核对 GitHub 原始文档**（上文 §2.2/§2.3 的仓库内文档原文），关键结论均有 primary 来源支撑；第三方镜像（deepseekdocs.com）仅用于交叉印证与 `.dsh-plugin` 补充。

## 3. govio 现状盘点（分发相关）

- **6 个 skill**：`govio`（主控路由）、`govio-meta`、`govio-query`、`govio-observe`、`govio-eda`、`govio-reconcile`（多源对账）。全部 kebab-case（DSH 命名规则合规）、frontmatter 含 name/description（Agent Skills 兼容）。**[直接证据]**（`skills/*/SKILL.md`、`docs/specs/README.md`）
- **skills 相对引用约定**：子 skill 以 `../govio/assets/...` 读取共享 assets（schema.md、metrics_index.md、ontology.gml、names/）。**[直接证据]**（`skills/govio-query/SKILL.md`）
- **assets 是派生数据**：`govio-cli meta graph` 生成到默认 `.agents/skills/govio/assets`（CWD 相对，可 `--assets-dir` 指定），命令打印绝对路径。该默认位置恰好落在 pi 与 DSH 共同的 `.agents/skills/` 发现根上。**[直接证据]**（`docs/install.md`、README 运行示例）
- **package_skills.py**：把 `skills/` 打成 `dist/govio-skills.zip`，**排除 `skills/govio/assets`**，zip 内路径形如 `skills/govio/SKILL.md`。现有安装流程 = zip 解压到 Agent skills 目录 + 手动复制 assets。**[直接证据]**（`package_skills.py`、`docs/install.md`）
- **govio-cli 是唯一执行入口**（skills 全局约束明示"所有 govio 操作只通过 govio-cli 完成"），entry point `govio.cli:main`；版本 0.6.0（pyproject）。**[直接证据]**
- **MCP 依赖之谜**：`pyproject.toml` 含 `mcp>=1.0.0`，但 AGENTS.md 文件组织与 docs/specs 模块表中均无 MCP server 实现。**[直接证据（依赖存在）+ 缺失证据（用途未知）]**

## 4. 两种 runtime 的注册契约对照（关键结论）

| 契约项 | pi.agent | DeepSeek Harness | 共性结论 |
|---|---|---|---|
| skills 装载 | 文件发现（`~/.pi/agent/skills`、`.pi/skills`、`~/.agents/skills`、`.agents/skills`、包内 `skills/`） | 文件发现（`.dsh/skills`、`.agents/skills`、`~/.dsh/skills`、`~/.agents/skills`、custom dirs） | 都不需要注册代码；**统一落地到 `<skills-root>/{govio*}`，assets 放 `<skills-root>/govio/assets`** |
| custom tools | TS 扩展 `pi.registerTool(defineTool(...))` | TS Cordis 插件 `ctx.tools.register(defineTool(...))` | 都是薄 TS 胶水包 `govio-cli` 子进程 |
| MCP | `pi.registerMcpServer()` / `mcp.json` | `@deepseek-ai/dsh-mcp-client` patch 层，一 server 一插件 | 可选路线（govio 无 MCP server 实现，列为后续） |
| 安装/更新/卸载 | `pi install/update --extensions/remove` | `dsh plugin add/update/remove`（pnpm 转发）+ 重启边界 | 都是包管理器语义、可 pin 版本 |
| 分发载体 | Pi package（npm/git/本地） | npm 包（bundle 插件）/ git 源 | 都可从 monorepo 子目录取包 |

**[表格为研究员对上文直接证据的归纳]**

## Contradictions（矛盾与分歧证据）

1. **dsh 发布时间/版本基线不一致**：第三方站点 dshdocs.com 称 harness 于 2026-08-13 公开发布；官方 README 未给日期、只说 developer preview；版本基线有 `0.1.5-alpha.1`（镜像审计基线）与 `dsh-v0.2.1-alpha.4`（仓库 tag）两个数。→ 处理：设计按「alpha、快速迭代、API 可能破坏」前提做适配层收敛，不锁死具体小版本号。
2. **第三方镜像 vs 官方原文**：deepseekdocs.com 的插件类型表（bundle vs `.dsh-plugin`）无官方原文对应页可直接核对（官方文档站内容不在该仓库 docs 内）。→ 处理：核心机制（bundle 安装/对账/重启边界）以仓库原文为准；`.dsh-plugin` 细节标注为来源解读。
3. **source_check 自动验证不可用**（见 §2.5）：已在方法上降级为直接核对原文并披露。

## Missing evidence（未验证项）

1. DSH 第三方插件**发布到 npm 的确切约束**（peerDependencies 范围、files 清单、locale/i18n 要求是否适用于社区包）：仓库内 `adding-a-package.md` 是 workspace 内部包清单（`private: true`），不适用于对外发布包。→ T3 实现时用 `dsh plugin add link:...` 本地实测收敛。
2. govio `mcp` 依赖的实际用途未证实（T1 时无 shell 工具未能 grep；V1 已全库 grep 确认 `src/` 中无 mcp 引用，依赖仅为 `pyproject.toml` 声明，仍无 MCP server 实现）。
3. pi 包（npm 安装）目录下 skills/assets 的**更新擦除行为**未实测：包更新后包内 assets 是否被清除、需要重新 `meta graph`（assets 为可再生派生数据，风险可控）。
4. DSH skill 的 `allowed-tools`（govio 主 skill 使用）在 DSH 本地 provider 的行为（忽略或告警）未验证；DSH 文档仅说明其读取 `disable-model-invocation`/`user-invocable`。
5. `npx @deepseek-ai/dsh` 的实际 Node 版本要求（第三方称 Node 22.19+）未获官方原文确认。

## Sources（来源清单）

**Kept（主证据）：**
- pi `docs/extensions.md`（本机 1.4.0 发行包）— 扩展注册/生命周期/工具契约
- pi `docs/packages.md` — 包结构、pi manifest、安装/更新/卸载
- pi `docs/skills.md` — skill 规范与发现位置
- pi `docs/configuration.md` — `<agent-dir>`/`.pi` 目录职责表
- pi `examples/extensions/hello.ts` — custom tool 最小实现
- deepseek-ai/deepseek-harness `README.md` — 开源状态、developer preview 警告
- deepseek-harness `docs/subsystems/skills.md` — skills 发现根/优先级/形态（决定 skills 安装方式）
- deepseek-harness `docs/cookbook/extension-cookbook.md` — 插件形态与特性→机制映射
- deepseek-harness `docs/cookbook/adding-a-tool.md` — 工具注册契约（execute/output/signal）
- deepseek-harness `apps/cli/reference/README.md` — `dsh plugin` pnpm 转发/对账/重启边界
- deepseek-harness `packages/bundle/sdk-minimal/cordis.patch.yml` — patch 行格式实例
- govio `README.md` / `docs/install.md` / `docs/specs/README.md` / `pyproject.toml` / `package_skills.py` / `skills/govio/SKILL.md` / `skills/govio-query/SKILL.md` — 现状盘点

**交叉印证（次级）：**
- deepseekdocs.com `en/docs/user-guide/plugins`、`en/docs/learn/core/plugin-anatomy` — 插件类型表、patch 语义、第三方包命名空间（已与仓库原文交叉一致处采用，未一致处标注）

**Rejected/deprioritized（仅作线索，不作证据）：**
- upd.dev 源码镜像、dshdocs.com（SEO 聚合站）— 二手聚合，内容与官方重复
- openma-ai/dsh-agents-plugins、Relistencode/dsh-extension-hub、Hilbert-beinghappy/seektty、frederico-kluser/dsh-plugin-dev-agent-skill、sandbaseai/deepseek-harness-handbook — 第三方/社区项目，只用于发现官方线索
- deepseek.com/en/harness 营销页 — 只佐证"开源+public preview"一句话

## Next steps（最有用的后续）

1. 用 `dsh plugin add link:packaging/deepseek` 实测第三方包 manifest/peerDependencies 要求（补 Missing #1；T3 仅完成真实 @deepseek-ai/* 类型 strict 编译验证，link: 安装回路未实测）。
2. grep 确认 govio `mcp` 依赖用途（补 Missing #2；V1 已 grep：src/ 无 mcp 引用），决定是否启用"形态 C：MCP server"（两 runtime 均有官方 MCP 承接机制）。
3. 实测 pi 包安装后 assets 落位与更新行为（补 Missing #3；T2 实测未覆盖，缺口如实保留），在 design.md 的 R1/R2 中定稿其一。
