# query / observe load 回归测试套件

对 `govio-cli query` 与 `govio-cli observe load` 两条命令做端到端回归验证
（经 `govio.cli.main.main()` 真实 CLI 入口执行）。

## 测试数据（data/）

| 数据 | 说明 |
|---|---|
| `seed.sql` | DuckDB 数据源种子：`customers`（5 行，含 NULL/中文/空串/引号/负数）、`orders`（5 行，含 NULL 金额、孤儿外键、日期） |
| `graph_csv/` | FalkorDB 批量导入格式的元数据图 CSV：Datasource/PhysicalTable/Col 节点 + OWNS/HAS_COLUMN 边（中文注释），由夹具分别构建 `ontology.gml`（NetworkX）与 `ontology.lbdb`（Ladybug） |

夹具（conftest.py）自动隔离 HOME（配置文件）与 cwd（`.govio` 输出、ObserveStore），
每条用例在独立临时目录运行，互不干扰。

## 用例清单

### query（test_query_cli.py）

| ID | 用例 | 期望 |
|---|---|---|
| Q01 | NetworkX 查询（exec） | JSONL 结果文件 + 位置键兜底 + 小结果内联 + stdout 全 ASCII |
| Q02 | NetworkX 标量 result | 包装为单行记录 |
| Q03 | NetworkX 60 行结果 | 超内联阈值只出摘要行，文件保留全部行 |
| Q04 | NetworkX 空 result | `Data not found.`，不写结果文件 |
| Q05 | GML 文件不存在 | 报错、退出码 1 |
| Q06 | Ladybug Cypher + RETURN 别名（小写 match） | 列名取别名；中文进 UTF-8 文件、stdout 全 ASCII |
| Q07 | Ladybug 跨节点关系查询 | Table-HAS_COLUMN->Col 联查结果正确 |
| Q08 | Ladybug 查询无匹配 | `Data not found.` |
| Q09 | Ladybug 非 MATCH 语句 | `Please write a MATCH query.`、退出码 1 |
| Q10 | Ladybug 非法 Cypher | 包装为 ValueError（当前契约） |
| Q11 | 配置缺 `ladybug.db_path` | 报错、退出码 1 |
| Q12 | 配置文件不存在 | 提示先 onboard、退出码 1 |
| Q13 | 配置缺 `graph.backend` | 报错、退出码 1 |
| Q14 | 未知 backend | 报错、退出码 1 |

### observe load（test_load_cli.py）

| ID | 用例 | 期望 |
|---|---|---|
| L01 | `--datasource` DuckDB 加载 | success/rows/columns/column_info 契约 + parquet 落盘 + manifest 可回放 |
| L02 | `-o` JSON 输出 | records 格式；中文 UTF-8 原样、NULL→null、空串/引号/负数保留 |
| L03 | 同名覆盖加载 | 只保留最新结果 |
| L04 | 0 行结果 | 成功入库，rows=0，列结构保留 |
| L05 | `--memory` 跨 DataFrame JOIN | 已加载表注册为同名表，source_tables 完整，结果正确 |
| L06 | `--memory` 但 store 为空 | success=false，友好报错 |
| L07 | SQL 执行失败 | success=false，不落盘 |
| L08 | 数据源未配置 | success=false |
| L09 | `--datasource` 与 `--memory` 同用 | 参数互斥，退出码 2 |
| L10 | 缺必填 `--sql` | 退出码 2 |
| L11 | 配置文件不存在 | 报错、退出码 1 |

## 运行

```bash
uv run pytest tests/regression -v
```

## 不在本套件内

- **FalkorDB 后端查询**：需要运行中的 FalkorDB（Redis 协议）服务，属集成测试范畴；
  `cmd_falkordb` 与 `cmd_ladybug` 共享同一输出契约（Q06-Q10 覆盖）。
- **SQLAlchemy 数据源**（mysql 等）：需要真实数据库服务；`--datasource` 路径已由
  L01-L04 用 DuckDB 覆盖，DatabaseManager 对两类数据源走同一 `execute_sql` 返回契约。
