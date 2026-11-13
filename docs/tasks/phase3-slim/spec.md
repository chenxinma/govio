# 任务：完成 govio Phase 3「工程精简与边界收敛」全部 7 项未勾选任务

## 背景

仓库 govio（Python 3.13 + uv，`src/govio/`）。`docs/roadmap.md` 的 **Phase 3** 章节列了 9 个勾选项，其中 7 项仍是 `- [ ]` 未完成。本任务把这 7 项全部做完，并把 roadmap 对应项勾选为 `- [x]`。

先读 `AGENTS.md`（代码风格、文档同步规则表、文件组织树、`encoding="utf-8"` 要求）与 `docs/roadmap.md` Phase 3 全文，再动手。

## 待完成的 7 项（逐项验收标准）

### 1. 删除未使用的 `mcp` 依赖
- 现状：`pyproject.toml:15` 有 `"mcp>=1.0.0"`，`src/`、`tests/`、`scripts/` 全文无 `import mcp` / `from mcp`。
- 做：移除该依赖，更新 `uv.lock`（`uv lock` 或 `uv sync`）。
- 验收：`uv sync --group dev` 成功；`uv run python -c "import govio"` 成功；grep 无 mcp 引用；`docs/specs/README.md` 依赖表若列了 mcp 需同步删除。

### 2. `_parse_type` 三份实现收敛为共享实现
- 现状：`src/govio/metadata/mysql_loader.py:46`、`postgres_loader.py:72`、`trino_loader.py:13` 各有一份 `_parse_type`，返回 `(size, precision, scale)`；postgres 版本签名不同（多参数）。
- 做：抽出共享实现（建议新模块 `src/govio/metadata/type_parser.py`），保留各方言差异所需的参数化能力（如 postgres 的 `numeric_precision`/`numeric_scale` 走 pg_catalog 列值而非字符串解析），三个 loader 改为调用共享实现；删除重复代码。
- 验收：**解析结果与重构前逐例一致**（重构前先给三份实现各写/记录一批代表性输入的期望输出，重构后用同一批输入断言）；新增或补充单测覆盖 varchar(n)/decimal(p,s)/numeric/char/int/bigint/text 等分支与非法输入回退；`docs/specs/metadata.md` 同步（新模块与函数签名）。

### 3. 打包排除规则与版本注入收敛到 `scripts/` 共享模块
- 现状：`package_skills.py`（仓库根）、`scripts/package_pi_extension.py`、`scripts/package_harness_plugin.py` 各自实现排除规则与版本注入。
- 做：抽共享模块（建议 `scripts/_packaging_common.py`），三支打包脚本复用；保持各脚本的 CLI 参数与产物结构不变。
- 验收：**重构前先各跑一次三支打包脚本并记录产物逐文件哈希（含 zip 内条目清单），重构后再跑一次对比，内容必须一致**（时间戳类差异需在报告里说明）；`tests/test_package_pi_extension.py`、`tests/test_package_harness_plugin.py` 全过；打包脚本自检（脚本自带的产物校验）通过；`docs/install.md` 若描述了打包脚本内部结构则同步。

### 4. `src/govio/cli/meta.py`（1294 行）按子命令域拆文件
- 现状：单文件 1294 行，混合「管线步骤 / CLI 处理函数 / argparse 入口」三类职责。
- 做：拆成 `cli/meta/` 包或 `cli/meta_*.py` 若干模块（自选，需在报告里说明切分依据），共享辅助（如 loader 构造、公共参数）单独一模块；`cli/main.py` 的 dispatch 与 `govio-cli meta <子命令>` 入口保持不变。
- 验收：**CLI 契约零变化** —— `govio-cli meta --help` 与每个子命令 `--help` 的输出、参数、退出码（失败 1 / 成功含空结果 0，见 `docs/specs/cli.md`）、stdout/stderr JSON 结构均与重构前一致（重构前后各抓一份 `--help` 输出做 diff 附在报告里）；`tests/` 全过（含 `tests/test_exit_codes.py`、`tests/regression/`）；`docs/specs/cli.md`、`docs/specs/metadata.md`、`AGENTS.md` 文件组织树同步。

### 5. `src/govio/metadata/utility.py`（264 行）拆分「导出编排 / 推荐入口」两类职责
- 做：按职责拆为两个模块（如 `export.py` + `recommend_entry.py`，命名自定），更新所有 import（已知引用方：`src/govio/cli/std_recommend.py`、meta 相关 CLI）。
- 验收：`make_csv` / `data_standard_recommend` 等公开函数行为与签名不变（如必须改名，需在 `docs/specs/metadata.md` 说明并同步全部调用方与 skills 文档）；测试全过。

### 6. `src/govio/metadata/gen_networkx.py`（CSV→GML）迁移到 `graph/`
- 做：移动到 `src/govio/graph/`（模块名可保留 `gen_networkx.py` 或改为更贴切的名字，需在报告说明），更新全部 import（已知：`src/govio/cli/meta*.py`、`src/govio/metadata/import_report.py`、`tests/regression/conftest.py`）。
- 验收：GML 产物与迁移前一致（同一份 CSV 输入迁移前后各生成一次，文件内容 diff 为空）；测试全过；**文档同步**：`docs/specs/graph.md`（新增该模块章节）、`docs/specs/metadata.md`（移除旧章节）、`docs/specs/data-model.md`、`docs/specs/README.md` 模块表、`AGENTS.md` 文件组织树。

### 7. observe_data 的 tools/core 分层说明补进 `docs/specs/observe-data.md`
- 做：新增一节说明 `observe_data/core/`（业务内核：observe_store、database、comparator、explorer、visualizer、chart、dataframe_store）与 `observe_data/tools/`（CLI/Agent 适配薄层）的职责边界与调用方向，并与实际代码一致（不要写不存在的类名/方法名）。
- 验收：`docs/specs/observe-data.md` 有该分层说明，且与 `src/govio/observe_data/` 真实文件结构逐项对得上。

## 全局约束

- **行为契约不变**：CLI 子命令/参数/退出码/输出 JSON 结构、node_id 稳定性（10 字符确定性 ID）、CSV 格式与列名、assets 契约（`<skills-root>/govio/assets`）、三种分发形态的注册契约。本任务只做精简与搬迁，**不实现 Phase 4/5 任何新功能**。
- **文档同步规则**（`AGENTS.md`）：改到哪个模块就同步 `docs/specs/` 对应文档；`AGENTS.md` 里的「File Organization」树也要跟着改。
- 代码风格：Python 3.13 类型标注（`list[str] | None`，不用 `typing.List/Optional`）、中文 docstring（含 Args/Returns）、文件读写显式 `encoding="utf-8"`、不用 `print` 做日志、不用裸 `except`、不用星号导入。
- 提交规范：按项（或按逻辑分组）多次提交，message 沿用仓库风格。**不要 squash 成一个提交**。
- 分支：在你所在 worktree 的分支上提交；若该分支是 `main`，先 `git switch -c refactor/phase3-slim` 再提交。**不要合并回 main、不要 push**（除非协调者另行指示），结束时报告分支名与提交列表（`git log --oneline main..HEAD`）。
- 不要启动容器：真实库集成测试（`tests/test_real_*.py`）默认 skip，保持 skip 即可，不要执行 wslc/docker。
- 任何命令非 0 退出即视为该项未完成，不要静默跳过或放宽断言（禁止改测试来「让它变绿」，除非能证明是测试本身的契约错误，并在报告中单列说明理由）。

## 结束前必须自证的证据（写进 `worker_done` 报告）

1. `uv run python -m pytest tests/ -q` 输出末尾统计：**>= 435 passed, 18 skipped**（基线，只增不减；若新增测试则通过数更多）。
2. `uv run ruff check src/ tests/ scripts/ package_skills.py` 与 `uv run ruff format --check src/ tests/ scripts/` 均无错误。
3. 第 3 项的打包产物哈希对比结果（前后一致 / 差异说明）。
4. 第 4 项的 `--help` 输出 diff 结果（应为空或仅无害空白差异）。
5. 第 6 项的 GML 产物 diff 结果（应为空）。
6. 第 2 项的 `_parse_type` 代表性输入前后一致性断言结果。
7. 逐项列出：改了什么、新增/删除/移动的文件清单、对应同步的 spec 文档。
8. `docs/roadmap.md` Phase 3 的 7 项已勾选为 `- [x]`（附最终 Phase 3 章节文本）。
9. 未解决的阻塞或有意延后的子项（若有），说明原因。

## 过程记录

先把这份 spec 原文保存为 `docs/tasks/phase3-slim/spec.md`（作为验收依据留档），然后在 `docs/tasks/phase3-slim/progress.md` 写过程记录（每项一节：现状 -> 做法 -> 验证命令与输出摘要），并在 `docs/tasks/phase3-slim/status.json` 写 `{"status":"done"|"blocked","summary":"...","files":[...]}`（与仓库既有 `docs/tasks/*/status.json` 格式一致）。这两个路径归执行者所有，可自由写。
