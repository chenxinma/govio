# X19-W5a 进展：口径判定升级——维度限定条件的实质比对（govio 侧）

- 任务：X19 crossCutting（W5a，govio 仓）；计划事实源在 map 仓 docs/task-plan.json（本任务不改动）
- 分支：`chenxinma/x19-caliber-govio`
- 记录时间：2026-11-14 12:22

## 1. 目标

按 PD-2 定调（计算表达式 + 维度限定条件为实质判据）把口径判定里的**维度限定条件**从
「剔除词子串包含 + 一律 unknown」升级为「SQL 维度限定三元组（列/算子/值集合）vs 指标侧维度声明」
的结构化三态比对（consistent / conflict / unknown），同批核查 `_compare_time_scope`，
并把 PD-2 补充的 8 档样本固化为回归用例集。

## 2. 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/govio/metadata/caliber.py` | 新增维度限定三元组解析（`DimPredicate`/`_split_predicates`/`_parse_predicate`/`_extract_dim_predicates`/`_group_by_columns`）、指标侧声明对齐（`DimensionDecl`/`DimRequirement`/`_dim_requirements`）、三态实质比对（`_compare_dimensions`）、时间粒度比对（`_compare_time_grain`/`_expected_grains`/`_sql_grains`）；`SqlFacts` 加 3 字段、`CaliberMetric` 加 `dimensions`；图读取沿 `DIMENSION_USED` 取 `Dimension`（读失败降级 + `logger` 告警）；`compare_metric` 接线两条新比对；`_compare_filter` docstring 写明与新路径分工 |
| `tests/test_caliber_dimension_compare.py` | 新增 42 例：三元组解析 11 例、维度声明对齐 6 例、三态矩阵 13 例、判定纪律反例 3 例、8 档样本 3 例、图读取与降级 3 例、输出契约 1 例（+分组/粒度解析若干） |
| `tests/test_caliber_cli.py` | 追加 2 例 CLI 级维度判定（conflict 与 consistent 两径），断言 JSON 契约不变；import 补 `DimensionDecl` |
| `docs/specs/metadata.md` | caliber.py 一节按 AGENTS.md 对应表同步：比对维度表、三元组与声明对齐规则、判定纪律、降噪、输出契约 |
| `docs/specs/cli.md` | `sql caliber` 说明补 `DIMENSION_USED`/`Dimension` 来源与 aspect 语义（参数与 JSON 形状未变） |
| `docs/tasks/X19-caliber-dimension-substance/design.md` | 判定矩阵、三态边界与「为什么 conflict 只有 4 条」、降噪边界、`_compare_time_scope` 核查结论、与方案 C 的界限、遗留问题 8 条、输出契约与合并顺序、8 档逐档前后对照 |

## 3. 验证命令与结果

```
uv run ruff check .            # All checks passed!（退出码 0）
uv run ruff format --check .   # 109 files already formatted（退出码 0）
uv run pytest tests/           # 526 passed / 18 skipped / 0 failed（基线 482 passed，新增 44 例）
```

口径三文件单独跑：`89 passed`（既有 `test_caliber_compare.py` + `test_caliber_cli.py` 45 例零回归）。

环境说明：首轮验证时 worktree 的 `uv sync` 尚未就绪（该会话无外网），改用主 checkout 的
`.venv` + `PYTHONPATH=src` 跑同一套命令；`uv sync` 手工完成后已在 worktree 本地环境复跑，
结果与首轮一致（`uv run ruff --version` = 0.15.12，与 uv.lock 同 minor）：
check=0、format --check=0（109 files already formatted）、526 passed / 18 skipped。

## 4. 8 档样本回归（详见 design.md §9）

s2/t2 正确口径保持 `consistent`（未被新增维度判定稀释）；t4 算术修饰保持 `conflict`；
s1/s3/s4/t1/t3 的 `conflict` 零回归。

## 5. 遗留问题

见 `design.md` §7（8 条）。最关键两条：

1. `values_example` 只是示例码值，且缺「码值 ↔ 展示名/数字码」映射 → 跨语言/跨编码取值只能
   落 unknown（宁缺不误报）；彻底解决需元数据侧显式声明限定取值，本任务不动 schema。
2. `statistical_scope` 纯中文描述（无枚举可对齐）时不产 conflict，仍走既有 `_compare_filter`
   弱证据路径。

## 6. 后续

- 不 push、不建 PR（主会话创建）；改动单 commit 落在本分支。
- map 侧 W5b（`verify-caliber-conflict` 维度三态断言、展示名）由主会话在本改动合并后派发；
  输出契约未变，map 无需先改解析。