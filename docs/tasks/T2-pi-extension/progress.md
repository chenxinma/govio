# T2 进度记录 — pi.agent 扩展分发实现

## 目标

按 T1 设计（docs/tasks/T1-distribution-design/design.md §2/§4）实现 govio 的 pi.agent 扩展分发形态：
`packaging/pi/` 扩展包源（package.json + 扩展入口 + skills 接线）、可重复执行的打包脚本（产物 `dist/govio-pi-<version>/`，不与 `dist/govio-skills.zip` 冲突）、产物结构自检、安装文档草稿（T7 整合）。
任务所有权：`packaging/pi/`、`scripts/package_pi_extension.py`、`docs/tasks/T2-pi-extension/`；其余文件只读；未执行任何 git 写操作。

## 改动文件清单

| 文件 | 说明 |
|---|---|
| `packaging/pi/package.json` | Pi package 清单（pi manifest 声明 extensions/skills，宿主供给包放 peerDependencies，version 由脚本注入产物） |
| `packaging/pi/extensions/govio-tools.ts` | 扩展入口：注册 `govio_cli` / `govio_query` custom tools（子命令白名单、argv 无 shell 子进程、UTF-8、超时/取消收尾）与 `/govio` 诊断命令 |
| `packaging/pi/README.md` | 中文安装/升级/卸载说明（R1/R2 skills+assets 落位、双平台路径、待实测项） |
| `packaging/pi/skills/` | **生成目录**（打包脚本从仓库 `skills/` 同步的副本，排除 `skills/govio/assets`），供 `pi install ./packaging/pi` 原地加载 |
| `scripts/package_pi_extension.py` | 打包脚本：同步 skills 副本 → 清理重建 `dist/govio-pi-<version>/` → 版本注入 → 产物结构自检；可重复执行 |
| `docs/tasks/T2-pi-extension/install-draft.md` | docs/install.md「pi.agent 集成」章节草稿（T7 整合，本任务不改 install.md） |
| `docs/tasks/T2-pi-extension/progress.md` | 本文件 |
| `docs/tasks/T2-pi-extension/status.json` | 任务状态 |

构建产物（gitignore 范围，非源码改动）：`dist/govio-pi-0.6.0/`（24 个文件：package.json、README.md、extensions/govio-tools.ts、6 个 skill 共 21 个文件）。

**未改动**：`src/govio/**`、`skills/**`、`package_skills.py`、`pyproject.toml`、`docs/install.md`、`docs/specs/**` 等共享文件全部只读（工作树中 `pyproject.toml`、`src/govio/cli/meta.py`、`src/govio/observe_data/core/database.py` 的改动来自并行任务，非 T2）。

## 与 T1 设计的对齐与最简决策

1. **脚本位置**：任务所有权指定 `scripts/package_pi_extension.py`（设计 §4.2 写作根目录 `package_pi.py`）。接口一致（`--version`/`--output`），用法为 `uv run scripts/package_pi_extension.py [--version X.Y.Z] [--output dist]`。
2. **packaging/pi/skills/ 生成副本**（设计 §2.1 明示「由打包脚本从仓库 skills/ 复制生成」）：脚本先同步 `skills/` → `packaging/pi/skills/`，再整体复制 `packaging/pi/**` 到 dist 并注入版本（等价设计 §4.2 步骤 1+2 的产物结果）。每次运行先清理再重建，可重复执行。
3. **日志**：打包脚本用 `logging`（任务通用约束「用 logging 不用 print」优先于设计 §4.2 对 print 的豁免）。
4. **pi manifest**：在设计 §2.2 定稿基础上加 `"!skills/govio/assets/**"` 排除（设计明示允许的双保险）。
5. **未新增 `tests/test_package_pi.py`**（设计 §7 建议）：任务所有权不含 `tests/`，以脚本内置产物自检 + 实测替代；建议后续任务补打包测试。
6. **排除规则**（设计 §4.4）：`skills/govio/assets/**`（任何 `<skills-root>/govio/assets`）、`**/__pycache__/**`、`**/*.pyc`、`.DS_Store`。`skills/govio-observe/assets/visualize-relations-schema.json` 是 skill 自带资源（与 `package_skills.py` 行为一致），保留。

## 验证命令与结果

| 命令 | 结果 |
|---|---|
| `uv run python -m py_compile scripts/package_pi_extension.py` | 通过 |
| `uv run ruff check scripts/package_pi_extension.py` | 通过（All checks passed） |
| `uv run python scripts/package_pi_extension.py`（连续执行 3 次） | 均通过：`✓ dist/govio-pi-0.6.0（24 个文件，其中 skills 副本 21 个文件）` + `✓ 产物结构自检通过（6 个 skill、无 assets/缓存文件、版本 0.6.0 已注入）`；重复执行幂等（先清理后重建） |
| `uv run python scripts/package_pi_extension.py --version "bad ver/../x"` | 正确拒绝：`✗ 非法版本号`，退出码 1 |
| `node --check packaging/pi/extensions/govio-tools.ts`（Node 24 类型剥离） | 通过（TS_SYNTAX_OK） |
| 产物清单核对 `find dist/govio-pi-0.6.0 -type f` | 24 个文件；6 个 skill 目录各含 SKILL.md；无 `skills/govio/assets`、无 `__pycache__`/`*.pyc`；package.json version=0.6.0、pi manifest 与 peerDependencies 完整 |
| pi 实测（pi 1.4.0，`pi -e ./dist/govio-pi-0.6.0 -p "..." --no-session --mode json`，不写 settings、不落 session） | ① 包加载无扩展错误；② 模型可见 `govio_cli`/`govio_query` 工具 schema；③ 模型调用 `govio_cli {"args":["backend"]}` → 子进程返回 `"ladybug"`（`details:{args,code:0}`, `isError:false`, 371ms）；④ 越权 `govio_cli {"args":["onboard","--help"]}` → `isError:true`「不允许的子命令: onboard；允许: meta, query, observe, sql, backend」（0ms，未起子进程）；⑤ `govio_query {"query":"MATCH (n) RETURN count(n)","language":"cypher"}` → 返回 127 行计数（`isError:false`, 4263ms） |
| `govio-cli -V` / `govio-cli backend` | `govio 0.6.0` / `ladybug`（/govio 诊断命令调用的同一命令面） |
| `uv run --no-sync python -m pytest tests/ -q` | 237 passed, 1 skipped, **1 failed**：`tests/test_query_output.py::test_named_records_file_and_ascii_inline`（amount `822.0` vs `'822'`）。该测试对应 `src/govio/cli/query.py` 输出格式，T2 未改动任何 src/tests 文件，判定为并行任务（pyproject 依赖变更触发环境重同步）或环境差异所致，非 T2 引入，留 parent 关注 |

## 遗留问题

1. **pi git 子目录源**（T1 待实测 #2）仍未实测：`git:github.com/chenxinma/govio@tag` 能否取到 `packaging/pi` 子目录包。README/install-draft 已写降级方案（clone 后本地安装 / npm 发布）。
2. **R2 包内 assets 更新存续行为**（T1 待实测 #3）未实测；兜底已写明（重跑 `meta graph`）。
3. `/govio` 诊断命令未端到端实测（`-p` 非交互模式不执行 slash 命令）；其调用面（`govio-cli -V`/`backend`）已验证，handler 与 `ctx.ui.notify` 签名已对照 pi 1.4.0 官方文档/类型声明核实。
4. `tests/test_package_pi.py` 打包测试缺失（所有权约束，见决策 5），建议后续补。
5. 工具在会话 CWD 运行 `govio-cli query` 会在 CWD 产生 `.govio/output-*.json`（govio-cli 既有行为，非本扩展引入）；实测已清理，README 未特别提示，T7 可酌情补一句。
6. `tests/test_query_output.py` 失败项（见验证表最后一行）需 parent/并行任务确认归属与修复。
