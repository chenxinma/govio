# T7 进度记录 — 架构审视、功能精简与路线图修正

## 目标

整合收尾任务：在 govio「无 LLM、可插拔任意 Agent Runtime 的 skill+CLI」定位下完成
①架构审视与功能精简建议（design.md）②docs/roadmap.md 路线图修正 ③docs/install.md
三种分发形态安装/升级文档整合 ④docs/specs/README.md 依赖与模块变更同步 ⑤全量验证。

任务所有权：`docs/roadmap.md`、`docs/install.md`、`docs/specs/README.md`、
`docs/tasks/T7-architecture-roadmap/`；未执行任何 git 写操作（commit/push/checkout/merge 均无）。

## 改动文件清单

| 文件 | 归属 | 说明 |
|---|---|---|
| `docs/tasks/T7-architecture-roadmap/design.md` | 所有权内 | 架构审视与功能精简建议：模块边界评价、三种分发形态统一抽象、冗余/可删减清单、应停止路线、2-3 个迭代精简方向 |
| `docs/roadmap.md` | 所有权内 | 路线图修正：纳入 mysql/postgres/trino loader、查询执行统一、三种分发形态；新增 Phase 3「工程精简与边界收敛」为下一优先（原 Phase 3/4 顺延为 4/5）；标注过时项（DatabaseManager 多库、assets 归并已解决、MCP/npm/DSH 深度适配暂缓）；架构图更新 |
| `docs/install.md` | 所有权内 | 整合三种分发形态（形态一 skills zip+uv tool / 形态二 pi 扩展 / 形态三 DeepSeek Harness 插件）的安装/升级/卸载文档（T2/T3 草稿并入，共享 skills+assets 约定前置）；数据源 URL 表补 PostgreSQL/Trino 行；补 `.govio/output-*.json` 副作用提示（T2 遗留 5） |
| `docs/specs/README.md` | 所有权内 | 依赖表补 `psycopg[binary]`（T5）、`dotenv`（既有缺口）并给 `mcp` 标注精简候选；`govio.metadata` 模块描述补 5 个 loader；新增 Packaging 行（packaging/ + scripts/ 打包脚本） |
| `src/govio/cli/query.py` | **所有权外（bug 级小修）** | 修复 `output_result` 的 Decimal 序列化回归：pandas 2.3.x 把 Decimal 直接序列化为 float（822 → 822.0），违反「Decimal→str」契约（代码注释与 tests/test_query_output.py 双重声明）。修复为 to_json 往返前显式 `Decimal → str`（2 行 + import 1 行）。该失败是 T2/T4/T5/T6 均记录的既有失败，本修复使 `tests/test_query_output.py` 4 个用例全过 |
| `docs/tasks/T7-architecture-roadmap/progress.md` | 所有权内 | 本文件 |
| `docs/tasks/T7-architecture-roadmap/status.json` | 所有权内 | 任务状态 |

工作树中其余改动（`pyproject.toml`、`src/govio/cli/meta.py`、`src/govio/observe_data/core/database.py`、
`docs/specs/cli.md|metadata.md|observe-data.md`、`packaging/`、`scripts/`、`src/govio/metadata/{mysql,postgres}_loader.py`、
`tests/test_{meta_cli_sources,mysql_loader,observe_multi_db,postgres_loader}.py`、`docs/tasks/` 其余任务目录）
均为 T1-T6 产出，本轮只读未动。

## 验证命令与结果

| 命令 | 结果 |
|---|---|
| `uv run ruff check src/ tests/` | **All checks passed!** |
| `uv run pytest tests/`（等价 `uv run python -m pytest tests/ -q`） | **270 passed, 1 skipped**（修复前 269 passed / 1 failed / 1 skipped，唯一失败即 query.py Decimal 序列化，已修） |
| `uv run python -m pytest tests/test_query_output.py -q` | 4 passed（定向验证 bug 修复） |
| `uv run ruff check src/govio/cli/query.py` | All checks passed |
| `git diff --cached --name-only` | 空（无 staged 文件，未执行 git 写操作） |

注 1：本环境 `uv run pytest tests/` 直调报 "Failed to canonicalize script path"，用等价的
`uv run python -m pytest tests/ -q` 执行（T6 已记录同一现象）。

注 2：`uv run ruff format --check src/` 显示 39 个文件存在**既有**格式偏差（含 query.py 的
docstring 后空行、`fname` 换行等，全部与本次改动无关）。全仓 `ruff format` 归一属格式化大扫除，
超出「禁止重构性改动」的最小 diff 原则，未执行；T7 对 query.py 的 3 处新增行本身符合格式。

## 遗留问题

1. **`docs/specs/config.md` 的 "Datasource URL Formats" 表缺 PostgreSQL 行**（T6 遗留 2，该文件不在
   T7 所有权内）：install.md 已补 PostgreSQL/Trino 行，config.md 待后续任务同步。
2. **`tests/test_package_harness_plugin.py` 仍在 `docs/tasks/T3-harness-plugin/`**，未进
   `uv run pytest tests/` 常规集（T3 遗留 3）；迁移属移动级改动，不在 T7 所有权内，已列入
   design.md §6 迭代 1 与 roadmap Phase 3。
3. **精简建议均为设计建议未实施**（design.md §6）：`mcp` 依赖删除、`_parse_type` 三份收敛、
   打包排除规则收敛、`cli/meta.py` 拆分等留待后续迭代，本次严格只做文档 + 一处 bug 修复。
4. **format 归一未做**（见验证注 2）：39 个文件既有 `ruff format` 偏差，建议单独开格式化任务。
5. pi git 子目录源、DSH `dsh plugin add link:` 等实测缺口（T1/T2/T3 遗留）不属 T7 范围，
   install.md 对应小节已标注「待实测」。
