# V3 核验 T3 DSH 插件 + 修复评审发现 — 进度

## 目标

1. **P1-2**：`docs/tasks/T3-harness-plugin/test_package_harness_plugin.py`（8 用例）游离在 `uv run pytest tests/` 常规覆盖之外，打包脚本零常规测试覆盖——迁入 `tests/test_package_harness_plugin.py`，`REPO_ROOT` 由 `parents[3]` 改为 `parents[1]`，删除原文件（git 历史可溯）。
2. **P2-e**：`packaging/deepseek` 的 `npm install` 生成物未被忽略且 `package-lock.json` 会被复制进 dist 产物——`.gitignore` 增加两条忽略规则，打包脚本排除集补 `package-lock.json`，并同步脚本自检与 README 描述。

## 改动文件清单

| 文件 | 改动 |
|---|---|
| `tests/test_package_harness_plugin.py` | 迁入的 8 用例测试（新位置）：`REPO_ROOT = parents[1]`；模块 docstring 改为说明常规覆盖与用例范围；排除规则用例补 `package-lock.json` 断言；端到端用例补 `node_modules`/`package-lock.json` 不入产物断言 |
| `docs/tasks/T3-harness-plugin/test_package_harness_plugin.py` | 删除（迁入 tests/，git 历史可溯）；同目录 `__pycache__/` 陈旧字节码一并清理 |
| `.gitignore` | 新增 `packaging/deepseek/node_modules/` 与 `packaging/deepseek/package-lock.json`（含中文注释说明 npm install 生成物不进 dist） |
| `scripts/package_harness_plugin.py` | `EXCLUDED_FILE_NAMES` 补 `package-lock.json`；模块 docstring 排除规则描述同步。自检（`check_plugin_artifact`/`check_skills_artifact`）复用 `is_excluded`，自动同步 |
| `packaging/deepseek/README.md` | 「首次从源码编译」小节补充：npm install 生成物已被 .gitignore 忽略且打包脚本排除，不进 dist 产物 |
| `docs/tasks/V3-dsh-fixes/` | 本文件与 `status.json` |

## 验证命令与结果

| 命令 | 结果 |
|---|---|
| `uv run python -m pytest tests/test_package_harness_plugin.py -q` | 通过：**8 passed**（含端到端打包用例，`lib/index.js` 已存在故未 skip） |
| `uv run python -m pytest tests/ -q` | 通过：**286 passed, 1 skipped**（skip 为无关用例），迁移后测试被常规覆盖 |
| `uv run python -m pytest tests/ -q --collect-only` | `tests/test_package_harness_plugin.py` 下 8 个用例全部被收集 |
| 打包脚本重跑自检（源目录临时放 `node_modules/` + `package-lock.json` 后执行 `uv run python scripts/package_harness_plugin.py --output dist --skills-bundle`） | 通过：`产物结构自检通过`（govio-dsh-0.6.0 共 7 个文件、govio-dsh-skills 共 21 个文件）；`find dist/govio-dsh-* \( -name node_modules -o -name package-lock.json -o -name '*.pyc' -o -name .DS_Store \)` 计数 **0**，排除项确实未进产物。验证后临时文件已删除，`git status packaging/deepseek/` 仅剩 README.md 预期改动 |
| `git check-ignore -v packaging/deepseek/node_modules/ packaging/deepseek/package-lock.json` | 两条规则命中（`.gitignore:25`、`.gitignore:26`） |
| `uv run ruff check tests/test_package_harness_plugin.py scripts/package_harness_plugin.py` | 通过：All checks passed |
| `uv run ruff format --check tests/test_package_harness_plugin.py scripts/package_harness_plugin.py` | 通过：2 files already formatted |
| `git diff --cached --name-only` | 空（无 staged 文件，全程无 git 写操作） |

## 遗留问题

1. **design.md §4.4 排除规则清单未列 `package-lock.json`**：`docs/tasks/T1-distribution-design/design.md:446` 的统一排除规则写作「（插件源另排除 `**/node_modules/**`）」，未含 `package-lock.json`；该文件不在本任务所有权清单（只读），建议设计文档所有者后续同步（V1 已同步过 `node_modules`，同理可补）。
2. **T3 历史 progress.md 引用旧路径**：`docs/tasks/T3-harness-plugin/progress.md` 仍写 `docs/tasks/T3-harness-plugin/test_package_harness_plugin.py`；该文件属 T3 历史记录且不在所有权清单，未改（git 历史可溯旧文件）。
3. **环境**：本机 `uv run pytest ...` 报 `Failed to canonicalize script path`（uv 0.12.1 环境问题，与本改动无关），验证用等价的 `uv run python -m pytest ...` 执行；AGENTS.md 文档同步规则的 docs/specs/ 对应表中无打包脚本/gitignore 行，`docs/install.md`、`docs/specs/` 无过时描述需同步（仅新增能力说明放在 `packaging/deepseek/README.md`，已在所有权清单内）。
