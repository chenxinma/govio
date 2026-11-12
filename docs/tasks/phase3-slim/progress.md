# Phase 3「工程精简与边界收敛」过程记录

- 分支：`chenxinma/phase3-slim`（worktree `C:\Users\Administrator\orca\workspaces\govio\phase3-slim`）
- 验收依据：[spec.md](spec.md)
- 环境前置（不入库）：本 worktree 缺 gitignored 产物，执行前已补齐
  - 复制主仓 `skills/govio/assets/`（`tests/test_metadata_benchmark.py` 依赖 assets 契约产物）
  - `uv.lock`（gitignored）与主仓对齐后 `uv lock`，`datacompy` 回到 `0.19.5`（代码依赖 `datacompy.core`，PyPI 最新 1.2.0 已删除该模块；见「遗留」）
- 基线（改动前，`logs/p3/before/`）：`435 passed, 18 skipped`；`ruff check` 全绿；`ruff format --check` 51 个文件待格式化（主仓 HEAD 同数，属既有漂移）
- 执行顺序：1 → 2 → 3 → 5 → 6 → 4 → 7（5/6 先做模块搬迁，4 最后拆 `cli/meta.py`，避免拆分后再改 import 的两轮返工；`--help` / GML / 打包哈希基线在改动前一次采集，末尾统一对比）

## 1. 删除未使用的 `mcp` 依赖

- 现状：`pyproject.toml` 声明 `mcp>=1.0.0`，`src/`、`tests/`、`scripts/` 全文无 `import mcp` / `from mcp`；`docs/specs/README.md` 依赖表已标注「当前无代码引用，精简候选」。
- 做法：删除该依赖行；`uv lock` 重新求解（连带移除 `mcp`/`mcp-types` 及仅由其引入的传递依赖：`pydantic`、`pydantic-core`、`pydantic-settings`、`pyjwt`、`python-multipart`、`sse-starlette`、`starlette`、`typing-inspection`、`uvicorn`）；`docs/specs/README.md` 依赖表删除 mcp 行。`uv.lock` 本身在 `.gitignore` 内，不入库。
- 验证：`rg -n "mcp" pyproject.toml uv.lock src/ tests/ scripts/ package_skills.py docs/specs/README.md` → 无匹配；`uv sync --group dev` 退出 0；`uv run python -c "import govio"` 退出 0；`pytest tests/ -q` → 435 passed, 18 skipped。
- 提交：`d9f39d3 chore(deps): 移除未使用的 mcp 依赖`（同时留档本 spec）。

## 2. `_parse_type` 三份实现收敛为共享实现

- 现状：`metadata/mysql_loader.py:46`、`metadata/postgres_loader.py:72`、`metadata/trino_loader.py:13` 各一份 `_parse_type`；mysql/trino 为同一套串式解析（mysql 多一步 `unsigned/zerofill` 剥离），postgres 走 `information_schema` 属性列，签名不同。
- 做法：新增 `src/govio/metadata/type_parser.py`
  - `parse_type_string(type_str, *, strip_numeric_modifiers=False)`：串式语法（Trino 默认，MySQL 传 `True`），mysql/trino 的 `_TYPE_RE` 与 mysql 的 `_MODIFIER_RE` 一并迁入，两个 loader 删除本地实现与 `import re`。
  - `parse_type_attributes(data_type, character_maximum_length, numeric_precision, numeric_scale)`：属性式语法（PostgreSQL），判空统一 `_int_or_zero`（`pd.notna`）。
  - 类型名集合分两套常量（`STRING_ARG_TYPES`/`DECIMAL_ARG_TYPES` 与 `LENGTH_ATTRIBUTE_TYPES`/`DECIMAL_ATTRIBUTE_TYPES`）：合并会改变行为（串式语法下 `bit(8)` 原本 `(0,0,0)`，合并后变 `(8,0,0)`），故按方言分开，只共享算法与三元组语义。
- 等价性证据（重构前采集，重构后同一批输入复算）：
  - 快照 `logs/p3/before/parse_type.json`（30 条类型串 × mysql/trino 两份 + 14 条属性组合）。
  - 差分脚本 `logs/p3/diffcheck.py`：`strings=2177 pg_cases=17303 diffs=0 golden=OK`（笛卡尔积覆盖 base×参数×修饰后缀与属性值 None/NaN/int/float/str 组合）。
  - 固化测试 `tests/test_type_parser.py`（+47 用例）：STRING_CASES 同时断言两种 flag 的输出、ATTRIBUTE_CASES、非法输入回退、NaN 缺失、整型位宽不计 precision。
  - 重构后复采 `logs/p3/after_item3/parse_type.json` 与基线逐例相同（`parse_type identical: True`）。
- 测试侧改动说明（非「改测试让它绿」）：`tests/test_mysql_loader.py`、`tests/test_real_mysql.py`、`tests/test_postgres_loader.py`、`tests/test_real_postgres.py`、`tests/test_trino_loader.py` 仅把 `_parse_type` 引用改为共享函数（MySQL 用例带 `strip_numeric_modifiers=True`），parametrize 参数表与期望元组逐字未改。
- 文档同步：`docs/specs/metadata.md`（Mysql/Postgres/Trino 三处「类型解析」改引共享函数名 + 新增 `## type_parser.py` 章节）、`AGENTS.md` 文件组织树。
- 验证：`pytest tests/ -q` → 482 passed, 18 skipped（435 + 47）；`ruff check` 全绿；新增文件 `ruff format --check` 干净。
- 提交：`dd9f82e refactor(metadata): _parse_type 三份方言实现收敛为共享 type_parser`。

## 3. 打包排除规则与版本注入收敛到 `scripts/` 共享模块

- 现状：`package_skills.py` 只有 `skills/govio/assets` 一条排除规则；`scripts/package_pi_extension.py` 自实现 `_is_excluded_dir`/`_copy_tree_filtered`/`_inject_version`/`_read_pyproject_version`；`scripts/package_harness_plugin.py` 自实现 `EXCLUDED_*` 常量/`is_excluded`/`collect_files`/`copy_files`/`rewrite_package_version`/`read_default_version`。
- 做法：新增 `scripts/_packaging_common.py`
  - 规则：`EXCLUDED_DIR_NAMES`（`__pycache__`/`node_modules`）、`EXCLUDED_FILE_NAMES`（`.DS_Store`/`package-lock.json`）、`EXCLUDED_SUFFIXES`（`.pyc`）、`has_derived_assets`/`ASSETS_EXCLUDE_PREFIX`（`<...>/govio/assets` 派生数据，任意深度）。
  - 工具：`is_excluded_dir` / `is_excluded_file` / `is_excluded(path, root, exclude_prefixes=())` / `collect_files(src, exclude_prefixes=())` / `copy_files(files, src, dst)`。
  - 版本：`read_pyproject_version(pyproject_path)`、`validate_version(version, pattern)`、`rewrite_package_version(package_json, version)`。
  - 三支脚本改为复用；按文件路径加载脚本时 `scripts/` 不在 `sys.path`，各脚本显式插入后导入共享模块。
- 契约保持：三支脚本 CLI 参数、产物结构、日志/自检文案不变；pi 与 DSH 各自的版本号校验正则作为该分发形态的 CLI 契约留在脚本内（pi 拒绝非法版本时仍打印 `✗ 非法版本号: ...` 且不打印堆栈）；`package_skills.py` 的 zip 条目顺序仍按 `sorted(Path)` 写入。
- 等价性证据：重构前/后各跑一次三支脚本（输出到 `logs/p3/{before,after_item3}/`），产物逐文件 SHA256 对比：
  - `govio-pi-0.7.2` 24 文件 IDENTICAL、`govio-dsh-0.7.2` 7 文件 IDENTICAL、`govio-dsh-skills` 21 文件 IDENTICAL、`govio-skills.zip` 21 条目内容哈希 IDENTICAL；退出码 `{'harness': 0, 'pi': 0, 'skills': 0}` 前后一致。
  - 唯一文本差异是脚本回显的产物绝对路径（证据目录名 before/after_item3 不同），非产物内容差异；zip 内条目本身含时间戳字段（逐条目内容哈希一致，未比较 zip 容器字节）。
- 验证：`tests/test_package_pi_extension.py` + `tests/test_package_harness_plugin.py` → 16 passed；`uv run package_skills.py`、`uv run scripts/package_pi_extension.py` 退出 0；全量 `482 passed, 18 skipped`。
- 文档同步：`docs/specs/README.md` Packaging 行补充共享模块说明；`docs/install.md` 只描述脚本用法（未描述内部结构），无需改动。
- 提交：`9b7d54d refactor(scripts): 打包排除规则与版本注入收敛到 _packaging_common`。

## 5. `metadata/utility.py` 拆分「导出编排 / 推荐入口」

- 现状：`metadata/utility.py`（264 行）两个公开函数分属两类职责——`make_csv`（TDS 元数据导出编排 + CSV 落盘）与 `data_standard_recommend`（推荐入口，读 CSV 跑 `StandardRecommender`）；spec 里还残留一个实现中并不存在的 `reorder_index` 签名。
- 做法：`metadata/utility.py` → `metadata/export.py`（`make_csv`）+ `metadata/recommend_entry.py`（`data_standard_recommend`），函数体逐字不变；`cli/std_recommend.py` 导入路径同步；`tests/test_meta_export_ids.py` 的 monkeypatch 目标随模块改名（断言与期望值未动）。
- 公开契约：`make_csv` / `data_standard_recommend` 的函数名与签名不变（未改名，因此 skills 文档无需调整）。
- 等价性证据：`git show HEAD:src/govio/metadata/utility.py` 与新模块的函数体逐字 diff → `True / True`（含 docstring 与打印文案）。
- 顺带修正：`docs/specs/metadata.md` 中不存在的 `reorder_index` 签名描述删除（写文档时该函数已不在实现里）。
- 验证：`pytest tests/ -q` → 482 passed, 18 skipped；`ruff check` 全绿。
- 提交：`bbc1c8c refactor(metadata): utility.py 拆分导出编排与推荐入口两类职责`。

## 6. `metadata/gen_networkx.py`（CSV→GML）迁移到 `graph/`

- 现状：CSV→GML 生成器住在 `metadata/`，但它消费的是 CSV 产物、服务的是 NetworkX 图后端，属 T7 审视点名的越界点。
- 做法：`metadata/gen_networkx.py` → `graph/networkx_loader.py`，命名沿用 `graph/` 既有约定（`falkordb_loader.py` / `ladybug_loader.py` 都是「CSV 批量装载到某后端」，本模块即「CSV → GML」）；模块体只改 docstring 首行的模块名。`graph/__init__.py` 不加 re-export，与另两个 loader 一致（避免破坏惰性导入）。
- 调用方同步：`cli/meta`（`_update_graph` 的局部导入）、`tests/regression/conftest.py`（`nx_gml` 夹具）、`metadata/import_report.py`（文件名约定注释）。
- 等价性证据：同一份 `tests/regression/data/graph_csv` 输入，迁移前后 `ontology.gml` SHA256 相同（`c1fbd2d7c7add9c186d82b8fdaf78efff33b2557e4eac2cc811f1468f555fa9b`）、文本 172 行逐字一致（`gml text identical: True`）。
- 文档同步：`docs/specs/graph.md` 新增 NetworkX Loader 小节、`docs/specs/metadata.md` 删除旧章节、`docs/specs/data-model.md`、`docs/specs/README.md` 模块表、`AGENTS.md` 文件树。
- 验证：`pytest tests/ -q` → 482 passed, 18 skipped；`ruff check` 全绿。
- 提交：`7fb842b refactor(graph): CSV→GML 生成器从 metadata 归位到 graph/networkx_loader`。

## 4. `cli/meta.py`（1294 行）按子命令域拆包

- 现状：单文件 1294 行装着 argparse 装配、9 个 `cmd_*`、5 个 `step_*_export`、CSV merge 辅助、图/assets 操作、常量表与 URL→Loader 构造。
- 做法：拆为 `cli/meta/` 包（切点即 spec 指定的 `_build_metadata_loader` 与各 `cmd_*`），依赖方向单向 `parser → commands → steps / graph_ops → merge / common`：

| 模块 | 行数 | 内容 |
|---|---|---|
| `constants.py` | 40 | `DEFAULT_ASSETS_DIR`、`SOURCE_LABELS` / `SCHEMA_HINTS` / `SOURCE_URL_SCHEMES` / `SOURCE_URL_DEFAULTS` |
| `parser.py` | 229 | `meta()` argparse 装配（唯一入口） |
| `commands.py` | 256 | 9 个 `cmd_*`：参数校验、退出码、步骤编排 |
| `steps.py` | 423 | 5 个 `step_*_export`：读源 → 赋 node_id → 幂等写 CSV |
| `merge.py` | 107 | `merge_node_csv` / `merge_edge_csv` / `_load_csv_with_node_ids` |
| `graph_ops.py` | 173 | `_update_graph` / `_clear_graph` / `_generate_assets` / `_write_import_report` |
| `common.py` | 141 | `_build_metadata_loader`、`_describe_duckdb_schemas`、`_resolve_datasource_schemas` |
| `__init__.py` | 63 | 拆分依据 docstring + re-export（`meta` / 9 `cmd_*` / 5 `step_*_export`） |

- 顺手修好一个搬迁期暴露的真 bug：`cmd_recommend` 内的 `from .std_recommend import std_recommend` 在原单文件下指向 `cli/std_recommend.py`，进包后 `.std_recommend` 解析到 `cli/meta/std_recommend`（不存在）→ 改为 `from ..std_recommend import`。`meta recommend` 的 `--help` 与失败路径已复测。
- 加载器 / 配置类不在 `__init__` re-export：留在所属子模块，测试打桩改为 patch 子模块命名空间（`govio.cli.meta.steps.TDSLoader`、`.common.DuckDBLoader` / `MysqlLoader` / `PostgresLoader` / `TrinoLoader`、`.graph_ops.ConfigManager` / `GraphFactory` / `AssetsGenerator`、`.commands.ConfigManager`）。4 个测试文件共改 16 处打桩点 + 4 行 import，断言、期望值、用例本身零改动。
- 等价性证据（`logs/p3/before` vs `logs/p3/after_item4`）：
  - `help identical: True` —— `meta`（无参 / `--help` / 非法参数）与 9 个子命令各自 `--help` 的 stdout、stderr、退出码逐字节一致。
  - AST 对比 `git show main:src/govio/cli/meta.py` 与包内 7 个模块：顶层定义 30 项，missing 0 / added 0，仅 `_update_graph`（第 6 项的 loader 改名）与 `cmd_recommend`（上述相对导入修正）两处 import 行变化。
  - 同轮复跑：打包产物 24/7/21/21 条目 IDENTICAL、GML SHA256 与文本 IDENTICAL、`parse_type identical: True`。
- 验证：`pytest tests/ -q` → 482 passed, 18 skipped；4 个 meta 测试单跑 39 passed；`ruff check src/ tests/ scripts/ package_skills.py` 全绿。
- 文档同步：`docs/specs/cli.md`（`## meta/` 章节 + 模块职责表 + 打桩约定）、`docs/specs/metadata.md`（`step_*_export` 路径）、`AGENTS.md` 文件树、`docs/roadmap.md`。
- 提交：`904ca5f refactor(meta): cli/meta.py 按子命令域拆分为 cli/meta/ 包`。

## 7. observe_data 的 tools/core 分层说明

- 现状：`docs/specs/observe-data.md` 有 `Tool Functions` 与 `CLI Subcommands` 两张表，但没有说明 `core/` 与 `tools/` 的职责边界与调用方向；「失败语义」一节还把 `core/chart.py` 的 `render_chart` 说成工具层函数。
- 做法：文档开头新增「分层结构（core 与 tools）」，逐模块列出真实符号（`ObserveStore` / `DataFrameInfo` / `Manifest`、`DataFrameStore`、`DatabaseManager` / `_resolve_sqlalchemy_target`、`TableComparator`、`RelationExplorer`、`RelationVisualizer`、`render_chart`）与薄层 5 个模块的函数—内核依赖表；补「调用方向」小节（`cli/observe.py` 构造内核对象，`info` / `load` / `release` / `visualize-relations` 走薄层，`compare` / `explore` / `chart` 直连内核）。
- 同时校正：`render_chart` 属 `core/` 但沿用 `{'success', 'error'}` 返回契约；`observe_data.config.load_config` 目前只被测试用作构造 `Config` 的便捷入口，CLI 实际走 `ConfigManager` + `get_db_manager()`；`DataFrameStore` 在 `src/` 与 `tests/` 内已无调用方。
- 记录现状不改动：`tools/list_datasources.py` 直接读 `DatabaseManager._datasources` 私有属性（已在文档标注为现状），收敛为内核公开方法是后续事项。
- 验证：符号与文件树逐项对照 `src/govio/observe_data/`（core 7 + tools 5 + config.py）；文档改动不影响代码，全量测试维持 482 passed / 18 skipped。

## 遗留 / 有意未做

- `datacompy` 依赖漂移（预存在）：代码 `from datacompy.core import Compare`，而 PyPI 最新 1.2.0 已删除 `datacompy.core`。本 worktree 的 `uv.lock`（gitignored）已与主仓对齐锁到 `0.19.5`；建议后续把上限写进 `pyproject.toml`。
- `ruff format --check src/ tests/ scripts/` 仍有 51 个文件待格式化，与主仓 `main` HEAD 的 51 个完全同集（既有漂移）。新增/移动的 8 个 `cli/meta/*` 模块与 `type_parser.py` / `_packaging_common.py` / `export.py` / `recommend_entry.py` / `networkx_loader.py` 均已格式化干净；不做全仓格式化以免污染本次精简的 diff。
- `docs/plans/*`、`docs/tasks/T*/`、`docs/tasks/V*/` 里对 `src/govio/cli/meta.py`、`metadata/utility.py`、`gen_networkx.py` 的引用属历史留档，保持原样不改。
- 未启动任何容器：`tests/test_real_*.py` 保持默认 skip（`GOVIO_REAL_DB` 未设置）。
