# govio 新分发形态设计（pi.agent 扩展 / DeepSeek Harness 插件）

> T1 产出，T2（pi.agent 扩展）/ T3（DeepSeek Harness 插件）按本文直接实现。
> 证据与来源见 [research.md](./research.md)。本文中标注「（待实测）」的条目为已知残留风险，实现时按 §8 的验证清单收敛。
> 代码骨架给出签名与关键行，完整实现由 T2/T3 补全；所有新代码遵守 AGENTS.md（Python 3.13+ 类型语法、`encoding="utf-8"`、中文注释/docstring、logging 不用 print、import 三组分离、命名规范、无裸 except）。TypeScript 侧沿用各 runtime 官方示例风格（tab 缩进随官方示例）。

## 0. 设计目标与不变式

1. **不改动 govio Python 核心**（`src/govio/**` 零改动）。两种新形态都是 `govio-cli` 之上的薄壳：skills 提供工作流指令，可选工具层以子进程调用 `govio-cli`。
2. **skills 单一来源**：`skills/` 目录（6 个 skill）是唯一真源，三种分发产物（zip / pi 包 / dsh 插件配套 skills）都从它复制生成，排除 `skills/govio/assets`（assets 是派生数据）。
3. **assets 契约不变**：assets 永远落在 `<skills-root>/govio/assets/`，由 `govio-cli meta graph --assets-dir` 生成（默认 `.agents/skills/govio/assets` 已对齐该约定）。
4. **runtime API 漂移只影响一层 TS 胶水**：工具实现统一「spawn `govio-cli` + argv 数组（无 shell）+ UTF-8 解码」，不依赖任何 harness 私有行为。
5. 与现有 `dist/govio-skills.zip` + `uv tool install govio` 流程**共存不替代**（见 §6）。

## 1. 共享注册契约（两个 runtime 通用，T2/T3 必须遵守）

### 1.1 skills 布局契约

```
<skills-root>/
├── govio/
│   ├── SKILL.md
│   └── assets/                 # 派生数据，由 govio-cli meta graph 生成（打包时排除）
│       ├── schema.md
│       ├── metrics_index.md
│       ├── ontology.gml
│       └── names/*.md
├── govio-meta/SKILL.md
├── govio-query/SKILL.md        # 内部以 ../govio/assets/... 引用共享资源
├── govio-observe/SKILL.md
├── govio-eda/SKILL.md
└── govio-reconcile/SKILL.md
```

- `<skills-root>` 推荐值（两 runtime 均原生发现）：
  - 项目级：`<项目>/.agents/skills/`（pi：Agent Skills 位置；DSH：project-agents 根，优先级 200）
  - 用户级：`~/.agents/skills/`（pi：Agent Skills 位置；DSH：user-agents 根，优先级 500）
  - runtime 专属用户目录：pi `~/.pi/agent/skills/`；DSH `~/.dsh/skills/`
- **禁止嵌套**：DSH 不支持递归 `**/SKILL.md` 发现，6 个 skill 必须直接位于 `<skills-root>` 下（当前布局已满足）。
- name 全部 kebab-case（已满足 DSH 正则 `^[a-z0-9]+(-[a-z0-9]+)*$` 与 Agent Skills 约束），目录名与 frontmatter `name` 保持一致（已满足）。
- frontmatter 保持现状（`name`/`description` 必备；`allowed-tools` 为 Agent Skills 实验字段，pi 接受、DSH 忽略（待实测））。

### 1.2 assets 契约

1. `govio-cli meta graph --assets-dir <skills-root>/govio/assets`（或在含 `.agents/skills/` 约定的 CWD 下用默认值）。
2. 命令打印绝对路径；安装文档引导用户核对该路径等于 `<skills-root>/govio/assets`。
3. assets 可随时从图库重新生成（派生数据），因此任何安装/升级流程丢失 assets 的兜底动作都是「重跑 `meta graph`」。

### 1.3 govio-cli 契约（适配层边界）

- 目标环境须有可执行 `govio-cli`（`uv tool install govio`，版本 ≥ 0.6.0）。安装检查：spawn `govio-cli -V`。
- 工具层以 **argv 数组**调用（不拼 shell 字符串），超时可配（默认 120s），尊重 runtime 的取消信号（pi `signal` / DSH `exec.signal`），stdout/stderr 按 UTF-8 解码。
- 退出码非 0 ⇒ 工具返回错误结果（含 stderr 摘要），不抛裸异常。

### 1.4 工具面契约（可选增强层，A/B 两形态一致）

仅两个工具，避免与 CLI 演进产生同步负担：

| 工具 | 参数（模型可见 schema） | 行为 |
|---|---|---|
| `govio_cli` | `args: string[]`（argv，如 `["query","-c","MATCH ..."]`） | 转发给 `govio-cli`；**子命令白名单**：`meta`/`query`/`observe`/`sql`/`backend`；越权或空 args 拒绝执行 |
| `govio_query` | `query: string`，`language?: "cypher"\|"python"`（默认按后端自适应） | 等价 `govio-cli query -c <query>`，返回结构化文本 |

其余高频能力（meta、observe、sql）走 skills 指令 + `govio_cli`，不单独做 typed wrapper（保持壳层最小）。

## 2. 形态 A：pi.agent 扩展包（Pi package）

### 2.1 产物与源码结构

仓库新增（T2 实现）：

```
packaging/pi/                        # Pi package 源（也可被 pi install ./packaging/pi 原地加载）
├── package.json                     # npm 元数据 + pi manifest（见 2.2）
├── README.md                        # 中文安装/升级/卸载说明（见 2.4）
├── extensions/
│   └── govio-tools.ts               # 扩展入口：注册 govio_cli / govio_query + /govio 诊断命令
└── skills/                          # 由 package_pi.py 从仓库 skills/ 复制生成（排除 assets）
    └── (govio, govio-meta, ..., govio-reconcile)
package_pi.py                        # 打包脚本（见 §4）
dist/govio-pi-<version>/             # 构建产物（可直接 pi install，或 npm pack 发布）
```

### 2.2 `packaging/pi/package.json`（定稿）

```json
{
  "name": "@govio/pi-package",
  "version": "0.6.0",
  "description": "Govio 数据治理知识图谱 — Pi 工具与技能包",
  "keywords": ["pi-package"],
  "type": "module",
  "pi": {
    "extensions": ["extensions/govio-tools.ts"],
    "skills": ["skills"]
  },
  "peerDependencies": {
    "@earendil-works/pi-ai": "*",
    "@earendil-works/pi-agent-core": "*",
    "@earendil-works/pi-coding-agent": "*",
    "@earendil-works/pi-tui": "*",
    "typebox": "*"
  }
}
```

要点（pi `docs/packages.md` 直接证据）：
- 约定目录会被自动发现；显式 `pi` manifest 用于钉死范围（`skills` 收 `skills` 目录，排除规则如需可写 `"!skills/govio/assets/**"`，但打包时已排除 assets，双保险）。
- 宿主供给包必须放 `peerDependencies: "*"`, **不得放 `dependencies`**；本包无自有 npm 依赖，故无 `dependencies`。
- `pi-package` keyword 使包可被 pi 包画廊发现（可选）。
- `version` 与 `pyproject.toml` 的 `version` 同步（由 package_pi.py 注入）。

### 2.3 `extensions/govio-tools.ts` 骨架（T2 补全）

```ts
/**
 * Govio 工具扩展 — 将 govio-cli 包装为 Pi custom tools
 * 所有操作通过 govio-cli 子进程完成（argv 数组，无 shell），符合 skills 全局约束
 */
import { spawn } from "node:child_process";

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const CLI = process.env.GOVIO_CLI ?? "govio-cli";
const SUBCOMMANDS = new Set(["meta", "query", "observe", "sql", "backend"]);
const DEFAULT_TIMEOUT_MS = 120_000;

/** 运行 govio-cli 子进程并收集 UTF-8 输出 */
async function runGovioCli(
  args: string[],
  signal: AbortSignal,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<{ code: number | null; stdout: string; stderr: string }> {
  // spawn(CLI, args) 无 shell；signal 转发为 kill；stdout/stderr 以 utf8 拼接；
  // 超时 kill 并返回 code: null（完整实现由 T2 补全）
}

const govioCliTool = defineTool({
  name: "govio_cli",
  label: "Govio CLI",
  description:
    "执行 govio-cli 数据治理命令（子命令限 meta/query/observe/sql/backend）。" +
    "用于知识图谱维护、元数据/指标查询、数据探查与指标 SQL 组装。",
  parameters: Type.Object({
    args: Type.Array(Type.String(), { description: "govio-cli 参数数组，如 ['query','-c','MATCH ...']" }),
  }),
  async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
    const [sub, ...rest] = params.args;
    if (!sub || !SUBCOMMANDS.has(sub)) {
      throw new Error(`不允许的子命令: ${sub ?? "(空)"}；允许: ${[...SUBCOMMANDS].join(", ")}`);
    }
    const r = await runGovioCli([sub, ...rest], signal);
    if (r.code !== 0) {
      throw new Error(`govio-cli 退出码 ${r.code}: ${r.stderr.slice(0, 2000)}`);
    }
    return { content: [{ type: "text", text: r.stdout }], details: { args: params.args, code: r.code } };
  },
});

const govioQueryTool = defineTool({
  name: "govio_query",
  label: "Govio Query",
  description: "查询 govio 知识图谱（Cypher 或 Python，后端由 govio-cli 自动适配）",
  parameters: Type.Object({
    query: Type.String({ description: "Cypher（以 MATCH 开头）或 NetworkX Python 代码" }),
    language: Type.Optional(Type.Union([Type.Literal("cypher"), Type.Literal("python")])),
  }),
  async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
    // 等价 runGovioCli(["query", "-c", params.query], signal)；language 现阶段仅校验提示
  },
});

export default function (pi: ExtensionAPI) {
  pi.registerTool(govioCliTool);
  pi.registerTool(govioQueryTool);
  pi.registerCommand("govio", {
    description: "Govio 环境自检（govio-cli 版本 / 后端 / assets 路径提示）",
    handler: async (_name, ctx) => {
      // spawn govio-cli -V / govio-cli backend，ctx.ui.notify 展示；实现由 T2 补全
    },
  });
}
```

要点（pi `docs/extensions.md` + `examples/extensions/hello.ts` 直接证据）：
- `defineTool` 来自 `@earendil-works/pi-coding-agent`，`Type` 来自 `@earendil-works/pi-ai`；结果必须返回 `content` + `details`；执行失败 **throw**。
- 工厂内不得启动长驻资源；子进程在 `execute()` 内按需起、随 signal 收尾。
- 工具行为与渲染解耦（RPC/print 模式可用）；无需 TUI 专有代码。

### 2.4 安装 / 升级 / 卸载流程（写进 `packaging/pi/README.md` 与 docs/install.md 增补）

```bash
# 前置：govio-cli 在 PATH（uv tool install govio）

# 安装（三选一）
pi install ./dist/govio-pi-0.6.0          # 本地目录（开发/内网）
pi install git:github.com/chenxinma/govio@v0.6.0   # git 源（若发布到仓库子目录，见下注）
pi install npm:@govio/pi-package@0.6.0    # npm 发布后（可选）

# skills + assets 落位（两种方案，README 都写，推荐 R1）
# R1（推荐）：项目级 .agents/skills —— 与 meta graph 默认输出天然对齐
mkdir -p .agents/skills
cp -r dist/govio-pi-0.6.0/skills/* .agents/skills/     # 或解压 govio-skills.zip 得到 skills/
govio-cli meta graph --output ./data/meta --assets-dir .agents/skills/govio/assets
# R2（零拷贝）：用包内 skills（pi 自动发现），assets 同步进包内：
govio-cli meta graph --output ./data/meta --assets-dir "$(pi list | grep govio)/skills/govio/assets"  # 具体路径以 pi list 为准（待实测）

# 升级
pi update --extensions                     # 对账更新已装包
pi remove <source> && pi install <new-source>   # 换版本（npm 源可直接 install 新版本钉住）
# 升级后 assets 若丢失：重跑 meta graph（派生数据，兜底动作）

# 卸载
pi remove <source>                         # <source> 与 pi list 显示一致
rm -rf .agents/skills/govio*               # 手动清理 skills（包卸载不回收已复制的 skills）
```

注：pi 的 git 源按「整个 repo/目录」装载，若 `packaging/pi` 是仓库子目录且 pi 不支持子路径 git 源（待实测），则 git 安装方式改为「clone 后 `pi install ./govio/packaging/pi`」或以 npm 发布为准；T2 实测后在 README 定稿其一。

### 2.5 注册契约小结（pi）

| 能力 | 注册方式 |
|---|---|
| custom tools | 扩展内 `pi.registerTool()`（`govio_cli`、`govio_query`） |
| skills | 包内 `skills/`（pi manifest 声明）自动发现；或复制到 `~/.pi/agent/skills/`、`.agents/skills/` 等 |
| MCP（可选路线 C） | `pi.registerMcpServer("govio", { command: "govio-cli", args: ["mcp"] })` 或 `~/.pi/agent/mcp.json`（前提：govio 提供 MCP server，现状无，见 §8） |

## 3. 形态 B：DeepSeek Harness 插件（DSH bundle plugin）

### 3.1 产物与源码结构

仓库新增（T3 实现）：

```
packaging/dsh/                         # DSH bundle 插件源（npm 包）
├── package.json                       # dsh.bundle.patch 声明（见 3.2）
├── cordis.patch.yml                   # insert 行：挂载本插件（见 3.3）
├── src/
│   └── index.ts                       # Cordis 插件：ctx.tools.register（见 3.4）
├── tsconfig.json                      # 目标 ES2022+/NodeNext，outDir lib/
├── README.md                          # 中文安装/升级/卸载说明（见 3.5）
└── (skills/)                          # 不打进插件包！skills 走独立安装（见 3.6）
package_dsh.py                         # 打包脚本（见 §4）
dist/govio-dsh-<version>/              # 构建产物（lib/ 已编译，可 npm pack / 直接 add）
```

**skills 不进插件包**（DSH 明确「skill 不用 `dsh plugin add` 安装」）；插件包只携带工具。skills 用既有 `dist/govio-skills.zip`（或 package_dsh.py 顺带产出 `dist/govio-dsh-skills/` 目录副本）按 §3.6 安装。

### 3.2 `packaging/dsh/package.json`（定稿）

```json
{
  "name": "@dsh-external/govio-tools",
  "version": "0.6.0",
  "description": "Govio 数据治理工具 — DeepSeek Harness 插件",
  "type": "module",
  "main": "lib/index.js",
  "exports": {
    ".": "./lib/index.js",
    "./package.json": "./package.json"
  },
  "files": ["lib/index.js", "lib/types/**/*.d.ts", "cordis.patch.yml", "README.md"],
  "dsh": { "bundle": { "patch": "./cordis.patch.yml" } },
  "peerDependencies": {
    "@deepseek-ai/cordis": "*",
    "@deepseek-ai/dsh-tools": "*"
  },
  "devDependencies": {
    "@deepseek-ai/cordis": "*",
    "@deepseek-ai/dsh-tools": "*",
    "typescript": "^5.6.0"
  }
}
```

要点：
- `"dsh": { "bundle": { "patch": "./cordis.patch.yml" } }` 是 bundle 插件的唯一清单机制（没有 `dsh.plugin.json`）；安装后该 patch 层自动并入 profile 层叠，卸载自动移出——**因此插件行必须写在包自带 patch 里，不让用户手写 profile patch**。
- `@dsh-external/*` 是第三方社区包命名空间（内核为 `@deepseek-ai/*`）。
- import 必须用 `@deepseek-ai/cordis`、`@deepseek-ai/dsh-tools`（裸名 `cordis`/`schemastery` 解析不到）。
- 上述 peerDependencies 范围、发布 files 清单为**研究员推断 + 官方 workspace 约定类比**，T3 以 `dsh plugin add link:packaging/dsh` 实测定稿（待实测）。

### 3.3 `packaging/dsh/cordis.patch.yml`（定稿）

```yaml
# Govio 工具插件挂载行（bundle patch 层，安装后随包进入 profile 层叠）
- insert:
    - id: govio-tools
      name: '@dsh-external/govio-tools'
      inject: [tools]
      config:
        cliPath: govio-cli        # 可被用户 profile patch 按行覆盖（config 为整行替换语义）
        timeoutMs: 120000
```

（行格式对照官方 `packages/bundle/sdk-minimal/cordis.patch.yml`：`insert` 列表 + `id/name/inject/config/disabled`。）

### 3.4 `packaging/dsh/src/index.ts` 骨架（T3 补全）

```ts
/**
 * Govio 工具插件 — 将 govio-cli 包装为 DSH tools
 * 所有操作通过 govio-cli 子进程完成（argv 数组，无 shell）
 */
import { spawn } from 'node:child_process'

import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'govio-tools'
export const inject = ['tools']

interface GovioToolsConfig {
  cliPath?: string
  timeoutMs?: number
}

export function apply(ctx: Context, config: GovioToolsConfig) {
  const cli = config.cliPath ?? 'govio-cli'
  const timeoutMs = config.timeoutMs ?? 120000
  const subcommands = new Set(['meta', 'query', 'observe', 'sql', 'backend'])

  // runGovioCli(args, signal): spawn(cli, args)，UTF-8 收集输出，signal/超时收尾（T3 补全）

  ctx.tools.register(defineTool({
    name: 'govio_cli',
    description: '执行 govio-cli 数据治理命令（子命令限 meta/query/observe/sql/backend）',
    parameters: {
      args: { type: 'array', required: true, description: 'govio-cli 参数数组' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: typeof value === 'string' ? value : 'ok' }],
    },
    async execute({ args }, exec) {
      const [sub, ...rest] = args as string[]
      if (!sub || !subcommands.has(sub)) throw new Error(`不允许的子命令: ${sub ?? '(空)'}`)
      // return runGovioCli([sub, ...rest], exec.signal) 的 stdout；非 0 抛错
    },
  }))

  ctx.tools.register(defineTool({
    name: 'govio_query',
    description: '查询 govio 知识图谱（Cypher 或 Python）',
    parameters: {
      query: { type: 'string', required: true },
      language: { type: 'string', enum: ['cypher', 'python'] },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: typeof value === 'string' ? value : 'ok' }],
    },
    async execute({ query }, exec) {
      // return runGovioCli(['query', '-c', query], exec.signal) 的 stdout
    },
  }))
}
```

要点（`docs/cookbook/adding-a-tool.md` 直接证据）：`execute(args, exec)` 返回**规范 JSON 值**（这里用字符串）；抛错即 `isError`；必须响应 `exec.signal`；注册随插件 fiber 卸载自动撤销；schema 自动进系统提示。若 `defineTool` 导入在实测中有解析问题，降级方案为 `ctx.tools.register()` 直接收 raw JSON-Schema `ToolDefinition`（官方认可的等价路径）。

### 3.5 安装 / 升级 / 卸载流程（写进 `packaging/dsh/README.md` 与 docs/install.md 增补）

```bash
# 前置：govio-cli 在 PATH；pnpm 在 PATH；Node ≥ 22（待实测确认版本线）

# 安装（二选一）
dsh plugin --profile web add link:/path/to/govio/packaging/dsh   # 本地（开发/内网）
dsh plugin --profile web add @dsh-external/govio-tools@0.6.0     # npm 发布后
# 之后重启对应 profile（bundle 变更的生效边界）：重启 `dsh web` / `dsh --profile <name>`

# 验证挂载
dsh web --dump-config | grep govio-tools

# skills 落位（与插件解耦，按 §3.6）
unzip govio-skills.zip -d ~/.agents/skills/        # 得到 ~/.agents/skills/skills/... 时注意去掉外层 skills/ 目录
# 目标布局必须是 ~/.agents/skills/govio/SKILL.md 等 6 个目录（见 §1.1）
govio-cli meta graph --output ./data/meta --assets-dir ~/.agents/skills/govio/assets

# 升级
dsh plugin --profile web update @dsh-external/govio-tools && 重启 profile

# 卸载
dsh plugin --profile web remove @dsh-external/govio-tools   # 行随 bundle 层叠移出，无需手写 patch 回写
rm -rf ~/.agents/skills/govio*
```

注意（官方文档直接证据）：`remove` 不回写 profile 的 `cordis.patch.yml`——本设计把挂载行放进**包自带 bundle patch**，规避该坑；README 中提示用户：若曾手动往 profile patch 加过 govio 行，卸载时需自行删除。

### 3.6 skills 安装（DSH 专属说明）

- skills 根（按 DSH 优先级）：`<项目>/.dsh/skills` > `<项目>/.agents/skills` > customSkillDirs > `~/.dsh/skills` > `~/.agents/skills`。
- **推荐 `~/.agents/skills/` 或项目 `.agents/skills/`**（与 pi 共用一套安装说明，且 `meta graph` 默认 assets 路径 `.agents/skills/govio/assets` 天然对齐项目级）。
- skill 必须直接位于根下（DSH 无递归发现）；6 个 skill 已合规。
- frontmatter：DSH 读 `disable-model-invocation` / `user-invocable`；`allowed-tools` 忽略（待实测是否告警）。

### 3.7 注册契约小结（DSH）

| 能力 | 注册方式 |
|---|---|
| custom tools | Cordis 插件 `ctx.tools.register(defineTool(...))`（`govio_cli`、`govio_query`），`inject: ['tools']` |
| skills | 文件系统发现（§3.6），不走插件 |
| MCP（可选路线 C） | 每个 server 一个插件：MCP 工具发现后 `ctx.tools.register()`（DSH 自带 `@deepseek-ai/dsh-mcp-client` 供 patch 层使用，无默认启用的 server） |

## 4. 打包脚本接口（T2/T3 实现，Python，遵守 AGENTS.md）

### 4.1 现有脚本不动

`package_skills.py`（产出 `dist/govio-skills.zip`）**保持原样**，是 zip 流程的唯一入口。

### 4.2 新增 `package_pi.py`（T2）

```
用法: uv run package_pi.py [--version X.Y.Z] [--output dist]
```

- 读取 `pyproject.toml` 的 `version` 作为默认版本（`--version` 覆盖）。
- 清理并重建 `dist/govio-pi-<version>/`：
  1. 复制 `packaging/pi/**`（package.json、README.md、extensions/）；
  2. 复制 `skills/**` 到 `skills/`，**排除 `skills/govio/assets`**（复用 package_skills.py 的排除规则）；
  3. 用 `re` 或 json 重写 `package.json` 的 `version` 字段；
  4. 打印产物绝对路径与文件计数。
- 文件读写一律 `encoding="utf-8"`；输出用 `logging`（或沿用脚本既有 print 风格——`package_skills.py` 用 print，**为一致性允许打包脚本沿用 print**，但模块不得用 print 做日志）。
- 退出码：成功 0；源缺失/参数非法非 0 并给出中文错误。

### 4.3 新增 `package_dsh.py`（T3）

```
用法: uv run package_dsh.py [--version X.Y.Z] [--output dist] [--skills-bundle]
```

- 生成 `dist/govio-dsh-<version>/`：复制 `packaging/dsh/**`，重写 `package.json` 版本；
- 若 TypeScript 编译产物 `lib/` 不在源里，脚本负责调 `npx tsc -p packaging/dsh`（无 Node 环境时给出中文提示并非 0 退出）；
- `--skills-bundle`：额外生成 `dist/govio-dsh-skills/`（6 个 skill 目录副本，排除 assets），供 §3.6 安装；
- 其余约束同 4.2。

### 4.4 排除规则（两脚本统一）

排除：`skills/govio/assets/**`、`**/__pycache__/**`、`**/*.pyc`、`.DS_Store`。zip/目录内路径用 POSIX 分隔符（沿用 package_skills.py 的 `as_posix()`）。

## 5. 版本与发布策略

- 三种产物版本号与 `pyproject.toml` 的 `version` 同步（脚本注入）；govio-cli 与壳包允许小版本错位，但 README 声明「壳包版本 x.y 对应 govio-cli ≥ x.y」。
- 发布渠道：GitHub Releases（沿用现有 zip 渠道）追加 `govio-pi-<ver>.zip` / `govio-dsh-<ver>.zip`（或 npm 包）；git 安装用 tag 钉版本。
- DSH 为 alpha 快速迭代：壳包**锁定一层胶水**（§0.4），每次 DSH 破坏性变更只改 `packaging/dsh/src/index.ts` 与 patch 格式，不扩散。

## 6. 与现有 dist/govio-skills.zip + uv tool 流程的共存

| 形态 | 产物 | 安装目标 | 消费方 | govio-cli |
|---|---|---|---|---|
| 现有（保留） | `dist/govio-skills.zip` | `~/.codex/skills`、`.claude/skills` 等（手动） | Codex / Claude Code / 任意 Agent Skills runtime | uv tool install |
| A：pi 包 | `dist/govio-pi-<ver>/`（含 skills 副本） | `pi install` + skills 落位（§2.4） | pi.agent | 同上 |
| B：DSH 插件 | `dist/govio-dsh-<ver>/` + skills 副本 | `dsh plugin add` + skills 落位（§3.5/3.6） | DeepSeek Harness | 同上 |

- `start.sh` 本地部署流程不变（构建 → uv tool install → package_skills.py）；T2/T3 的新脚本追加到 `start.sh` 末尾是**可选项**，不属本设计强制范围。
- 三种产物共享同一 `skills/` 源与同一 assets 契约，互不覆盖：同一台机器可同时服务 Codex、pi、DSH。

## 7. T2 / T3 实现清单（文件级）

**T2（pi.agent 扩展）——允许改动：**
- 新增 `packaging/pi/package.json`、`packaging/pi/README.md`、`packaging/pi/extensions/govio-tools.ts`
- 新增 `package_pi.py`
- 增补 `docs/install.md`（「pi.agent 集成」章节）与 `docs/specs/README.md`（如视作分发/构建变更）
- 测试：`tests/test_package_pi.py`（打包排除规则、版本注入、产物结构断言）

**T3（DeepSeek Harness 插件）——允许改动：**
- 新增 `packaging/dsh/package.json`、`packaging/dsh/cordis.patch.yml`、`packaging/dsh/src/index.ts`、`packaging/dsh/tsconfig.json`、`packaging/dsh/README.md`
- 新增 `package_dsh.py`
- 增补 `docs/install.md`（「DeepSeek Harness 集成」章节）与 `docs/specs/README.md`
- 测试：`tests/test_package_dsh.py`（同上）
- 实测清单：`dsh plugin add link:packaging/dsh` 走通安装/卸载；补 Missing 证据（peerDependencies/files 契约）

**共同约束**：不改 `src/govio/**`、不改 `skills/**` 内容（skills 仅被复制）；不做无关重构；实现后按 AGENTS.md 文档同步规则更新对应 spec。

## 8. 风险与开放问题（残余）

1. **DSH API 破坏性变更**（developer preview 明示）：壳层单点收敛已预留；T3 需在 README 标注适配的 dsh 版本基线。
2. **pi git 子目录源 / 包内 assets 更新行为**（待实测）：R1/R2 方案 README 双写，实测后定稿。
3. **DSH 第三方发布契约**（peerDependencies/files）未经官方第三方发布文档确认（待实测）。
4. **govio `mcp` 依赖用途未知**：若确认已有/计划 MCP server，可增开「形态 C」（pi `registerMcpServer` + DSH `dsh-mcp-client` patch），两 runtime 承接机制均已调研就绪。
5. **`allowed-tools` 在 DSH 的告警行为**未知（可能仅警告，不阻断加载）。
6. shell 命令示例中的路径分隔符（Windows/POSIX）在 README 需给出双平台写法（实现时注意 UTF-8 与 cp936 陷阱，见 AGENTS.md）。
