# J2 observe 其余子命令失败退出码统一 — 进度

分支：`fix/observe-exit-and-abort`（基于 main @dd223c1 + J1 onAbort 加固）。

退出码契约（与 I1 的 query / observe load 一致）：

- 失败（success:false、参数非法、文件不存在、执行错误、后端异常）→ `❌ ...` 到 stderr + 退出码 1；
- 成功但空结果/无匹配 → 退出码 0；
- stdout 结果 JSON 契约向后兼容：失败时保留结构化 `{"success": false, "error": ...}`（与 load 一致），不破坏既有字段。

## 失败路径现状 → 处置对照表

| 子命令 | 失败路径 | 改造前现状 | 处置 |
|---|---|---|---|
| compare | source / target DataFrame 不存在 | stdout `{"success": false, ...}` 后正常返回 → exit 0 | stdout 保留 success:false JSON + `❌` → stderr + exit 1 |
| compare | 比对执行错误（如 join 列不存在、datacompy 异常） | 裸异常（经 observe() 兜底才 exit 1，直接调用裸奔） | cmd 层捕获 → success:false JSON + `❌` + exit 1 |
| explore | 指定 DataFrame 不存在 | stdout success:false → exit 0 | 同上转 exit 1 |
| explore | 探查执行错误 | 同 compare（仅入口兜底） | cmd 层捕获 → exit 1 |
| explore | 无任何 DataFrame（空结果） | success:true 空 foreign_keys/column_similarities → exit 0 | 保持 exit 0（空结果不是失败） |
| chart | DataFrame 不存在 | stdout success:false → exit 0 | 转 exit 1 |
| chart | 列不存在 / 图表类型非法（render_chart 返回 success:false） | stdout success:false → exit 0 | CLI 层转 exit 1（工具层契约不变） |
| chart | matplotlib 渲染异常 | 仅入口兜底 | cmd 层捕获 → exit 1 |
| chart | 0 行 DataFrame 渲染（空结果） | success:true → exit 0 | 保持 exit 0 |
| info | overview / --datasource / --df 的后端异常 | 仅入口兜底 | 保持入口兜底 exit 1（无 success:false 路径） |
| info | --name 指定 DataFrame 不存在（见 show） | stdout success:false → exit 0 | 转 exit 1 |
| info | 空概览（无数据源、无 DataFrame） | exit 0 | 保持 exit 0（空结果不是失败） |
| show | DataFrame 不存在 | stdout success:false → exit 0 | 转 exit 1 |
| show | `--rows` 负数（参数非法） | success:true + 怪异样本 → exit 0 | `❌` + exit 1 |
| show | 0 行 DataFrame（空结果） | success:true sample:[] → exit 0 | 保持 exit 0 |
| release | `--name` 指定的 DataFrame 不存在 | stdout success:false → exit 0 | 转 exit 1 |
| release | 缺 `--name` 且缺 `--all` | `❌` + exit 1 | 已达标，保持 |
| release | `--name` 与 `--all` 同时给（参数歧义） | 静默按 `--all` 处理 → exit 0 | `❌` + exit 1 |
| release | `--all` 且无 DataFrame（空结果） | success:true released:[] count:0 → exit 0 | 保持 exit 0 |
| visualize-relations | 关系文件不存在 | stdout success:false → exit 0 | 转 exit 1 |
| visualize-relations | JSON 解析失败 | stdout success:false → exit 0 | 转 exit 1 |
| visualize-relations | 关系数据非法（工具层异常） | 仅入口兜底 | cmd 层捕获 → exit 1 |
| visualize-relations | 空关系列表（空结果） | success:true nodes:[] edges:[] → exit 0 | 保持 exit 0 |
| load | （I1 已完成）SQL 错误/连接失败/内存加载失败 | `❌` + exit 1，stdout 保留 result JSON | 不变 |
| 全部子命令 | 未预期异常 | observe() 入口兜底 `❌ 执行失败: ...` + exit 1（J1） | 保持兜底 |

## 实施状态

- [x] 失败路径清单
- [x] observe.py 退出码统一（新增 `_fail` 统一失败出口：stdout 保留 `{"success": false, "error": ...}` + stderr `❌` + exit 1）
- [x] tests/test_exit_codes.py 扩展（compare/explore/chart/info/show/release/visualize 各 成功/失败/空结果，共新增 27 个用例，全部打桩）
- [x] tests/regression 25 用例回归（全量 369 passed, 18 skipped，基线 342+18 只增不减）
- [x] docs/specs/cli.md（「退出码契约（observe 全部子命令）」表）、docs/specs/observe-data.md（「失败语义与退出码」）同步
- [x] 全量测试 + ruff 全绿（`uv run python -m pytest tests/ -q` 369 passed；`uv run ruff check src/ tests/` All checks passed）

## 测试覆盖对照（成功 / 失败 / 空结果）

| 子命令 | 成功 | 失败 | 空结果 |
|---|---|---|---|
| compare | 有差异比对 → 0 | DataFrame 不存在、join 列不存在 → 1 | 无差异 → 0 |
| explore | 双表探查 → 0 | 指定 DataFrame 不存在 → 1 | 无 DataFrame → 0 |
| chart | bar 图生成 → 0 | DataFrame 不存在、列不存在（success:false）→ 1 | 0 行 DataFrame → 0 |
| show / info --name | schema+sample → 0 | DataFrame 不存在、--rows 负数 → 1 | 0 行 DataFrame → 0 |
| info | 概览（数据源+DataFrame）→ 0 | --name DataFrame 不存在 → 1 | 空概览 → 0 |
| release | 释放成功 → 0 | DataFrame 不存在、--name/--all 冲突 → 1 | --all 无 DataFrame → 0 |
| visualize-relations | 单关系 → 0 | 文件不存在、JSON 解析失败 → 1 | 空关系列表 → 0 |
