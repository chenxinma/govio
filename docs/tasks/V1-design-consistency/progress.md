# V1 进度记录 — T1 设计-实现-文档三方一致性核验

## 目标

对照 `docs/tasks/T1-distribution-design/design.md` 与 `research.md`，逐项核对 `packaging/pi/`、`packaging/deepseek/`、`scripts/package_pi_extension.py`、`scripts/package_harness_plugin.py`、`docs/install.md` 的实际实现是否符合设计（打包产物结构、skills 接线、`govio_cli`/`govio_query` 注册契约、assets 契约 `<skills-root>/govio/assets`、安装/升级/卸载流程）；核对 T1 标注的已知缺口（pi git 子目录源、R2 assets 更新行为）是否在文档中如实保留；发现漂移以实现为准修正 `design.md`/`research.md` 描述。

约束遵守：不改代码、不改 `docs/install.md`（归 V7）；仅改任务所有权目录（`docs/tasks/T1-distribution-design/`、`docs/tasks/V1-design-consistency/`）；无任何 git 写操作。

## 改动文件清单

| 文件 | 说明 |
|---|---|
| `docs/tasks/T1-distribution-design/design.md` | 修正 23 处与实现的漂移（脚本/目录命名、pi manifest 定稿、打包脚本接口、devDependencies、§7 交付清单标注、§8 风险状态注记、§2.4 R2 措辞） |
| `docs/tasks/T1-distribution-design/research.md` | 修正 4 处（Next steps 路径 `packaging/dsh`→`packaging/deepseek` 与 T3 实测状态注记、Missing evidence #2 补 V1 grep 结果） |
| `docs/tasks/V1-design-consistency/progress.md` | 本文件（含核验表） |
| `docs/tasks/V1-design-consistency/status.json` | 任务状态 |

**未改动**：`packaging/**`、`scripts/**`、`docs/install.md`、`src/**`、`skills/**`、`docs/specs/**`、`tests/**` 全部只读。

## 核验表（设计声明 / 实际实现 / 一致与否）

| # | 设计声明 | 实际实现 | 一致？ |
|---|---|---|---|
| 1 | §2.1 pi 包结构：`packaging/pi/{package.json,README.md,extensions/govio-tools.ts,skills/}`，产物 `dist/govio-pi-<version>/`（design.md:73-81） | `packaging/pi/` 四件齐全；`skills/` 为脚本生成副本（.gitignore）；产物 `dist/govio-pi-0.6.0/` 24 文件（实测脚本自检通过） | 一致（脚本名漂移已修正：`package_pi.py`→`scripts/package_pi_extension.py`） |
| 2 | §2.2 pi manifest 定稿：`pi.extensions`+`pi.skills`、peerDependencies 五个宿主包、version 注入（design.md:84-111） | `packaging/pi/package.json`：manifest/peerDependencies 与设计一致；`skills: ["skills", "!skills/govio/assets/**"]`（设计允许的双保险排除，实现定稿含之）；version 0.6.0 由脚本注入产物 | 一致（定稿 JSON 已按实现补排除项） |
| 3 | §2.3 `govio_cli` 契约：`args: string[]`，子命令白名单 meta/query/observe/sql/backend，越权/空 args 拒绝，失败 throw 含 stderr 摘要（design.md:123-128） | `packaging/pi/extensions/govio-tools.ts`：SUBCOMMANDS 白名单同；`!sub \|\| !SUBCOMMANDS.has(sub)` 拒绝；`toToolError` 含 stderr.slice(0,2000)；T2 实测越权 `onboard` 被拒（0ms 未起子进程） | 一致 |
| 4 | §2.3 `govio_query` 契约：`query`+`language?: cypher\|python`，等价 `govio-cli query -c`（design.md:130-138） | 同签名；language 仅作提示（description 明示）；空 query 拒绝（实现附加校验，不违反契约）；T2 实测 `MATCH (n) RETURN count(n)` 全链路通过 | 一致 |
| 5 | §2.3 `/govio` 诊断命令：spawn `govio-cli -V`/`backend`，`ctx.ui.notify` 展示（design.md:152-156） | handler 实现同（诊断超时 15s）；调用面 `-V`/`backend` 已实测，handler 本身 T2 未端到端实测（-p 非交互不执行 slash 命令） | 一致（实测覆盖缺口见 T2 progress 遗留 #3） |
| 6 | §1.3 govio-cli 子进程契约：argv 数组无 shell、默认 120s 超时、尊重取消信号、UTF-8 解码、非零退出转错误（design.md:60-65） | 两 TS 均 `spawn(..., {shell:false})`、`setEncoding("utf8")`、DEFAULT_TIMEOUT_MS=120_000、AbortSignal→kill、非零→throw 中文错误 | 一致 |
| 7 | §1.2 assets 契约：`govio-cli meta graph --assets-dir <skills-root>/govio/assets`，默认 `.agents/skills/govio/assets`，打印绝对路径（design.md:53-56） | `src/govio/cli/meta.py:43` `DEFAULT_ASSETS_DIR = Path(".agents/skills/govio/assets")`；`--assets-dir` 参数在；`_generate_assets` 打印 `✓ Assets 已生成到: {resolved}` 绝对路径 | 一致 |
| 8 | §1.1/§4.4 skills 接线：单一真源 `skills/`，打包排除 `skills/govio/assets`，6 个 kebab-case skill 直接位于根下（design.md:38-50） | `scripts/package_pi_extension.py` 同步 `skills/`→`packaging/pi/skills/`（排除 govio/assets、缓存）；`scripts/package_harness_plugin.py --skills-bundle` 同规则；`diff -r skills packaging/pi/skills` 一致（SKILLS_IN_SYNC）；skill 自带资源 `govio-observe/assets/` 正确保留 | 一致 |
| 9 | §2.4 pi 安装/升级/卸载：三选一安装、R1/R2 落位、`pi update --extensions`、`pi remove`+手动清 skills（design.md:191-218） | `packaging/pi/README.md` + `docs/install.md`「形态二」同流程，另补双平台路径与 `GOVIO_CLI` 环境变量 | 一致 |
| 10 | §3.1 DSH 插件结构：`packaging/dsh/`+`package_dsh.py`，产物 `dist/govio-dsh-<version>/`（design.md:236-246） | 实为 `packaging/deepseek/`+`scripts/package_harness_plugin.py`（T3 所有权指定）；产物名 `dist/govio-dsh-<version>/` 同设计；`lib/` 编译产物保留入库（设计未写） | 命名漂移已修正；`lib/` 入库已补进设计 §3.1 |
| 11 | §3.2/§3.3 DSH manifest 定稿：`dsh.bundle.patch`、peerDeps、files、cordis.patch.yml insert 行（design.md:250-296） | `packaging/deepseek/package.json` 与 `cordis.patch.yml` 逐字段一致；devDependencies 实际多 `@types/node: "*"`（编译必需） | 一致（devDeps 已按实现补进定稿） |
| 12 | §3.4 DSH 工具注册契约：`ctx.tools.register(defineTool(...))`、`inject:['tools']`、execute 返回 JSON 值/抛错即 isError/响应 `exec.signal`（design.md:298-350） | `src/index.ts` 同契约；T3 以真实 `@deepseek-ai/cordis` 4.0.x / `dsh-tools` 0.0.1-rc.x 类型 strict 编译零错误验证，未启用降级方案 | 一致 |
| 13 | §3.5/§3.6 DSH 安装/升级/卸载 + skills 独立落位：`dsh plugin add/update/remove`、重启生效、dump-config 验证、bundle patch 卸载无残留（design.md:362-398） | `packaging/deepseek/README.md` + `docs/install.md`「形态三」同流程（含「手动 patch 需自删」提示） | 一致 |
| 14 | §4.2 pi 打包脚本接口：`--version`/`--output`、版本注入、自检、UTF-8、退出码（design.md:411-424） | `scripts/package_pi_extension.py` 参数/行为同；实现另含「同步 skills 到 packaging/pi/skills/」步骤与产物结构自检；输出用 logging（设计允许） | 一致（描述已按实现重写步骤） |
| 15 | §4.3 DSH 打包脚本接口：`--version`/`--output`/`--skills-bundle`、lib 缺失时 `npx tsc`（design.md:426-441） | `scripts/package_harness_plugin.py` 参数/行为同；`lib/` 缺失时在插件目录 `npx tsc -p .`；无 npx 中文报错非 0 退出 | 一致（路径/编译细节已按实现修正） |
| 16 | §4.4 排除规则：govio/assets、`__pycache__`、`*.pyc`、`.DS_Store`（design.md:443-445） | 两脚本同规则；harness 另排除 `node_modules`（设计未列） | 基本一致（node_modules 已补进设计） |
| 17 | §5 版本同步：壳包版本与 pyproject 同步注入，README 声明「x.y 对应 govio-cli ≥ x.y」 | 两脚本默认读 `pyproject.toml` project.version；两 README 均有版本对应声明 | 一致 |
| 18 | §6 共存：`package_skills.py`/`start.sh` 不动，产物命名无冲突 | `start.sh` 仍 5 步（未追加新脚本，设计明示可选）；产物 `govio-skills.zip`/`govio-pi-*`/`govio-dsh-*` 无冲突（实测 `ls dist/` 类产物命名） | 一致 |
| 19 | §7 T2/T3 实现清单（文件级） | 路径/命名漂移见 #1/#10；**测试交付漂移**：`tests/test_package_pi.py` 未交付（T2 以脚本自检+pi 实测替代）；`tests/test_package_dsh.py` 实际落在 `docs/tasks/T3-harness-plugin/test_package_harness_plugin.py`（8 用例，建议迁入 tests/） | 部分不一致（已在 design.md §7 如实标注，缺口保留） |
| 20 | **已知缺口保留①**：pi git 子目录源支持未知（design §2.4 注/§8.2；research Next steps） | `packaging/pi/README.md`「待实测 #1」与 `docs/install.md` L180/L223 均保留「子目录包支持待实测」+降级方案；T2 实测未覆盖（T2 progress 遗留 #1） | 一致（缺口如实保留） |
| 21 | **已知缺口保留②**：R2 包内 assets 更新存续行为未知（design §2.4 R2/§8.2；research Missing #3） | `packaging/pi/README.md`「待实测 #2」与 `docs/install.md` L197/L224 均保留「更新时可能被重建/丢失，兜底重跑 meta graph」；T2 实测未覆盖 | 一致（缺口如实保留） |
| 22 | docs/install.md 与实现/设计三方一致（只读核对） | 「形态二/三」引用的脚本名、产物名、安装/升级/卸载命令、R1/R2、待实测标注均与实现一致（install.md 归 V7，未改动） | 一致 |
| 23 | research.md 事实性陈述 | 「现状盘点」（6 skill、assets 契约、package_skills.py、govio-cli 入口）逐项复核属实；`mcp` 依赖之谜经 V1 全库 grep 收敛（src/ 无引用）；DSH/pai 机制性调研为历史证据记录，不与实现冲突 | 一致（4 处注记更新） |

## 验证命令与结果

| 命令 | 结果 |
|---|---|
| `uv run ruff check scripts/package_pi_extension.py scripts/package_harness_plugin.py` | 通过（All checks passed!） |
| `uv run python -m py_compile scripts/package_pi_extension.py scripts/package_harness_plugin.py` | 通过（COMPILE_OK） |
| `uv run python scripts/package_pi_extension.py --output <tmp>` | 通过：`✓ dist/govio-pi-0.6.0（24 个文件，其中 skills 副本 21 个文件）` + `✓ 产物结构自检通过（6 个 skill、无 assets/缓存文件、版本 0.6.0 已注入）`，退出码 0 |
| `uv run python scripts/package_harness_plugin.py --output <tmp> --skills-bundle` | 通过：`✓ govio-dsh-0.6.0（7 个文件）`、`✓ govio-dsh-skills（21 个文件）`、`产物结构自检通过`，退出码 0（复用已有 lib/ 编译产物） |
| `diff -r skills packaging/pi/skills -x assets -x __pycache__ -x "*.pyc" -x .DS_Store` | 通过（SKILLS_IN_SYNC：skills 单一真源与生成副本一致） |
| `uv run --no-sync python -m pytest docs/tasks/T3-harness-plugin/test_package_harness_plugin.py -q` | 通过（8 passed in 0.53s） |
| `grep` 核对 `src/govio/cli/meta.py` assets 契约 | `DEFAULT_ASSETS_DIR = Path(".agents/skills/govio/assets")`、`--assets-dir` 参数、打印 resolved 绝对路径均在 |
| `git status --short` / `git diff --stat` | 仅父会话既有改动（`docs/task-plan.json`、2 个编排脚本 untracked）；无 staged 文件；本次改动仅 4 个 docs 任务文件 |
| `grep -n "package_pi\.py\|package_dsh\.py\|packaging/dsh"`（design/research） | 仅剩有意保留的引用（修订说明行、§7 测试缺口标注、`tests/test_package_*` 原定文件名） |

注：打包脚本验证输出写入系统临时目录（mktemp）；`package_pi_extension.py` 执行时会重建生成目录 `packaging/pi/skills/`（gitignore 的派生副本），执行前后 `diff -r` 确认内容与 `skills/` 一致、无实质改动。

## 遗留问题

1. **`tests/test_package_pi.py` 未交付**（design §7 建议项）：T2 以脚本内置产物自检 + pi 1.4.0 实测替代；建议后续补打包测试进 `tests/`。
2. **打包测试位置漂移**：`docs/tasks/T3-harness-plugin/test_package_harness_plugin.py` 建议迁入 `tests/test_package_harness_plugin.py` 并纳入常规 pytest 集（T3 progress 遗留 #3 同建议）。
3. **pi git 子目录源、R2 assets 更新行为**仍待实测（T1 缺口，文档已如实保留，非本轮问题）。
4. **`dsh plugin add link:` 安装/卸载回路未实测**（T3 环境未装 dsh/pnpm）；peerDependencies/files 发布契约仍为推断（T3 已做类型编译验证部分收敛）。
5. T2 progress 遗留 #6：`tests/test_query_output.py::test_named_records_file_and_ascii_inline` 失败项归属待 parent 确认（与本核验无关，本任务未改 src/tests）。
