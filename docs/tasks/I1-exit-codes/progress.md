# I1 query / observe load 查询失败返回非 0 退出码 — progress.md

## 目标

`govio-cli query` 与 `govio-cli observe load` 在查询失败时以非 0 退出码（约定 1）结束进程，
让下游 agent / 脚本能通过退出码识别失败；成功但结果为空仍退出 0（空结果不是失败）。
错误信息保持 `❌` 前缀风格走 stderr，不裸奔 traceback。

## 改动清单

| 文件 | 改动 |
|---|---|
| `src/govio/cli/query.py` | 新增 `_fail(message)`（`❌` 错误到 stderr + `sys.exit(1)`）；`query()` 校验查询语句非空、整体兜底 `except Exception`；`cmd_networkx`/`cmd_falkordb`/`cmd_ladybug` 分别捕获「连接图库失败」「查询执行失败」（Python 语法错、Cypher 错、后端异常），GML 缺失、非 MATCH、配置缺字段统一走 `_fail`；原有 stdout 错误输出全部改走 stderr |
| `src/govio/cli/observe.py` | `cmd_load`：`success: false` 时保留 stdout 结果 JSON（success/error 契约向后兼容），另打印 `❌ {error}` 到 stderr 并 `sys.exit(1)`；`--datasource` 未设的 `raise ValueError` 改为 `❌` + 退出 1；observe 入口分发整体加 `except Exception` 兜底（`❌ 执行失败: ...` + 退出 1）；release 参数校验错误改走 stderr |
| `tests/test_exit_codes.py` | 新建 14 个打桩用例（不连真实库）：query 成功 / 空结果 / 执行失败（语法错、运行时错）/ 连接失败 / 缺 `-c` / 非 MATCH；observe load 成功 / SQL 错误 / 连接失败 / 内存加载失败 / 参数缺失 / 入口兜底。断言 `pytest.raises(SystemExit)` 的 code、错误走 stderr（stdout 不混 `❌`）、成功路径不抛 SystemExit |
| `docs/specs/cli.md` | query.py 段新增「退出码契约（query）」表（成功/空结果/失败各场景）；observe.py 段新增「退出码契约（observe load）」表；函数签名同步 `query(query_text: str | None)`、新增 `_fail` |
| `docs/specs/observe-data.md` | Tool Functions 表补充 `load_dataframe`/`load_from_memory` 失败返回结构；CLI Subcommands 新增「load 失败语义与退出码」小节 |

未改动：`src/govio/observe_data/tools/load_dataframe.py`（现有 `{'success': False, 'error': ...}`
契约已满足失败识别需要，无需新增结构化字段，保持向后兼容）；compare/explore/chart 等其他
observe 子命令（不在本次范围）。

## 退出码约定

- 成功（含 0 行结果、含仅输出提示）：0
- 失败（查询执行失败、连接失败、参数缺失或非法、配置/后端错误、未预期异常兜底）：1
- 错误信息：`❌` 前缀、走 stderr；stdout 不混错误
- `observe load` 失败时 stdout 结果 JSON（`success`/`error`）契约保持不变

## 验证命令与结果

| 命令 | 结果 |
|---|---|
| `uv run python -m pytest tests/test_exit_codes.py -q` | 14 passed |
| `uv run python -m pytest tests/ -q` | 317 passed, 18 skipped（基线 303 passed + 14 新增，18 skipped 不变；integration 默认 skip） |
| `uv run ruff check src/ tests/` | All checks passed! |

## 遗留问题

- 其余 observe 子命令（compare / explore / chart / info）失败时仍输出 `{"success": false}`
  且退出 0，未纳入本次退出码改造（按范围约定不动）；后续如需统一可复用 query 的 `_fail` 模式。
- argparse 本身的用法错误（如缺 `--name`/`--sql`）由 argparse 退出码 2，天然非 0，
  不在本次「约定 1」范围内，属既有行为。
