# 元数据查询 Benchmark

衡量 agent 在真实治理元数据上回答查询问题的正确性，对比两种信息层：
**govio 图库**（Cypher 查询）与 **SQLite**（ER 模式）。benchmark 分为两个阶段：

```
执行阶段（execute）                     评估阶段（score）
每题 × 2 条件，pi -p --model deepseek-flash     逐回答 pi 提取 Cypher/SQL 并执行
 ├─ 图库：skills/govio-meta/SKILL.md           ├─ graph：govio-cli query；sqlite：基准库只读
 │        + skills/govio/assets/schema.md      ├─ 按评分标准（checks）对执行结果打分
 │        + 任务提示词 → 文本回答              ├─ 参考答案（oracle SQL 结果）
 └─ SQLite：sqlite-schema.md（库表结构）       └─ artifacts/scores.json + report.md
            + 任务提示词 → 文本回答
        ↓ 全部保存 artifacts/answers/<条件>/<id>.md
```

## 目录结构

```
benchmarks/metadata-query/
├── README.md         # 本文件
├── EXECUTION.md      # 执行方案（可重复执行手册）
├── prompts.json      # 题目集：问题 + oracle_sql + 检查项（评分标准）
├── sqlite-schema.md  # SQLite 库表结构文档（sqlite 条件的材料）
├── eval.py           # 评估程序：build / execute / score
├── data/
│   └── metadata.sqlite   # 基准库（build 产物，oracle_sql 执行库）
└── artifacts/
    ├── answers/graph/<id>.md   # 图库条件文本回答
    ├── answers/sqlite/<id>.md  # SQLite 条件文本回答
    ├── scores.json             # 逐题评分明细
    └── report.md               # 对比评估报告
```

## 执行阶段材料

| 条件 | 材料文档 | 数据访问 |
|---|---|---|
| graph | `skills/govio-meta/SKILL.md` + `skills/govio/assets/schema.md` | `govio-cli query -c '<cypher>'`（Ladybug 图库） |
| sqlite | `sqlite-schema.md`（库表结构 md 文档） | `data/metadata.sqlite` 只读查询 |

两条件均为「材料文档 + 环境说明 + 任务提示词」经 **stdin** 传给 `pi -p`（Windows 下 argv 会被
cmd.exe 吞掉 `<`/`>`/`->`），模型默认 `deepseek-flash`（`--model` 可换），输出文本逐题保存。

## 题目集与评分标准（prompts.json）

23 题，6 个分类，难度 1~3：

| 分类 | 题数 | 考察点 | 示例 |
|---|---|---|---|
| `lookup` | 3 | 单跳固定查询：实体/属性 | T_INVOICE 有哪些字段 |
| `filter` | 3 | 关键词过滤、模糊匹配 | 金额相关字段分布在哪些系统 |
| `aggregate` | 4 | 分组聚合、排序、统计 | 字段数最多的 5 张表 |
| `multi_hop` | 7 | 跨实体多跳路径 | 跨系统同名字段、相似表发现 |
| `open_ended` | 3 | 业务语义 → 表的开放定位 | 排查雇员银行账户该看哪些表 |
| `negative` | 3 | 幻觉与越界边界 | 不存在的表、元数据不含行数 |

每题包含：

- `question`：任务提示词（唯一事实源）；
- `oracle_sql`：参考答案来源（对基准库执行，报告中展示；`null` 表示元数据无法回答）；
- `expected_summary`：标准答案要点（中文）；
- `checks`：**评分标准**——`contains` / `contains_any` / `not_contains` / `regex` 检查项，
  对**执行结果文本**判定，得分 = 通过权重 / 总权重；
  最终得分再按 **answer 输出量**折减：检查得分 × (1 − 0.2 × (1 − 简洁度))，
  简洁度 = min(1, 300 / answer 字符数)，越简单的内容达成目的分越高（详见 EXECUTION.md）。

## 用法

```bash
# 0. 基准库（一次性；CSV 更新后重建）
uv run python benchmarks/metadata-query/eval.py build

# 1. 执行阶段：每题 × 2 条件跑 pi -p，文本回答落盘
uv run python benchmarks/metadata-query/eval.py execute
uv run python benchmarks/metadata-query/eval.py execute --filter lookup-01 --force   # 重跑某题
uv run python benchmarks/metadata-query/eval.py execute --condition sqlite --model deepseek-flash

# 2. 评估阶段：提取查询语句 → 执行 → 按评分标准对执行结果打分并出对比报告
uv run python benchmarks/metadata-query/eval.py score
uv run python benchmarks/metadata-query/eval.py score --filter lookup-01   # 单 case 重评并并入全局报告
```

## 维护

- **加题**：往 `prompts.json` 追加，写清 `question` / `oracle_sql` / `checks` / `expected_summary`；
- **数据更新**：`build` 后 oracle 结果自动刷新，再核对检查项数字；
- **SQL 注意**：SQLite LIKE 的 `%`/`_` 是通配符，含字面下划线要写 `ESCAPE '!'`。
