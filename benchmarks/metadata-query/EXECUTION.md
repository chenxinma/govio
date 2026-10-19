# 元数据查询 Benchmark 执行方案（可重复执行）

benchmark 分为**执行**与**评估**两个阶段：执行阶段用 `pi -p`（指定模型）以
「材料文档 + 任务提示词」方式回答问题并保存文本；评估阶段逐回答提取回答中的
Cypher/SQL 并真实执行、按评分标准对执行结果打分出报告。

## 一、组件

| 文件 | 阶段 | 作用 |
|---|---|---|
| `prompts.json` | 两阶段 | 23 题：`question`（任务提示词）/ `oracle_sql`（参考答案）/ `checks`（评分标准）/ `expected_summary` |
| `sqlite-schema.md` | 执行 | SQLite 库表结构 md 文档（sqlite 条件的材料） |
| `../skills/govio-meta/SKILL.md`、`../skills/govio/assets/schema.md` | 执行 | 图库条件的材料 |
| `eval.py build` | 准备 | CSV → `data/metadata.sqlite`（oracle_sql 的执行库） |
| `eval.py execute` | 执行 | 每题 × 2 条件跑 `pi -p`，保存 `artifacts/answers/<条件>/<id>.md` |
| `eval.py score` | 评估 | 逐回答 pi 提取查询语句 → 执行（graph: `govio-cli query`；sqlite: 基准库只读）→ 按 checks 对执行结果打分 → `artifacts/scores.json` + `artifacts/report.md` |

## 二、前置条件

1. **依赖**：`uv sync`（仓库根目录）。
2. **pi 与模型**：`pi` 在 PATH；执行/提取模型默认 `deepseek-flash`（`pi --list-models deepseek`
   确认；换模型用 `--model`）。
3. **图库条件**：全局 `govio-cli` 可用（`govio-cli query -c` 能查到数据，配置见 `~/.govio/config.yaml`）。
4. **基准库**：`data/metadata.sqlite`（`eval.py build` 构建；`oracle_sql` 依赖它出参考答案）。

## 三、执行阶段

```bash
uv run python benchmarks/metadata-query/eval.py execute            # 23 题 × 2 条件
uv run python benchmarks/metadata-query/eval.py execute --filter lookup-01,agg-01
uv run python benchmarks/metadata-query/eval.py execute --condition graph --force
```

每题的提示词组成（两条件一致的骨架）：

```
材料文档（graph：govio-meta SKILL.md + assets/schema.md；sqlite：sqlite-schema.md）
+ 环境说明（数据访问方式）
+ 任务提示词（prompts.json 的 question）
```

- 提示词经 **stdin** 传入 `pi -p --model deepseek-flash`（Windows 下 argv 会被 cmd.exe
  吞掉 `<`/`>`/`->`，`eval.py` 已内置 stdin 方式）；
- 输出文本保存 `artifacts/answers/graph/<id>.md` 与 `artifacts/answers/sqlite/<id>.md`；
- 已存在的回答默认跳过，`--force` 重新执行。

## 四、评估阶段

```bash
uv run python benchmarks/metadata-query/eval.py score
```

- **单 case 重评**：`score --filter lookup-02`（或 `--category`）只重评指定题，结果
  **并入**历史 `scores.json`/`report.md`：未重评的题保留原结果，汇总与输出量分析按
  合并后全量重算，逐题顺序按 `prompts.json`；

- **提取**：逐回答启动 `pi -p --model <模型> -nt -ne -nc --no-session`，从回答文本中提取
  可执行的 Cypher/SQL 语句（一条一块的代码块）；提取失败时退回解析回答自身的代码块/裸语句；
- **执行**：图库条件的 Cypher 经 `govio-cli query -c '<语句>'` 执行（eval.py 优先解析**全局** govio-cli）；
  SQLite 条件的 SQL 经 python sqlite3 **只读** 打开 `data/metadata.sqlite` 执行；
  多条语句逐条执行、结果合并；
- **评分标准** = `prompts.json` 的 `checks`（contains / contains_any / not_contains / regex），
  作用于**执行结果文本**（单元格用 ` | ` 连接，末尾附 `共 N 行` 便于计数类检查命中），
  题目得分 = 通过权重 / 总权重；两条件各自计分；
- **输出量权重** = 最终得分 = 检查项得分 × (1 − 0.2 × (1 − 简洁度))，简洁度 = min(1, 300 /
  answer 字符数)：越简单的内容达成目的分越高，最多折减 20%（未达成目的不因简洁加分）；
  常量 `BREVITY_BUDGET` / `OUTPUT_WEIGHT` 在 eval.py 顶部可调；
- **参考答案** = `oracle_sql` 在 `data/metadata.sqlite` 上的执行结果（展示在报告中供人工复核）；
- 产出：`artifacts/scores.json`（逐题明细，含提取语句与执行结果）+ `artifacts/report.md`
  （图库 vs SQLite 对比：总览、分类对比、**输出量分析**、汇总表、逐题明细含语句、执行结果与检查项）。
- **已知限制**：`not_contains`（防幻觉）天然适配执行结果；纯文字型检查项（如 negative 题的
  「明确回答不存在」）在查询式回答下无法命中，两条件同等受影响，需人工复核。

## 五、复跑清单

- [ ] `uv run pytest tests/test_metadata_benchmark.py -q`（程序自检）
- [ ] `eval.py build`（CSV 有更新时）
- [ ] `eval.py execute`（或按 `--filter`/`--condition` 局部重跑）
- [ ] `eval.py score`
- [ ] 对照 `report.md` 的 oracle 参考答案人工复核关键题
- [ ] 如需引用结论，把报告关键数字记录进实验记录

## 六、故障排查

| 症状 | 原因 | 处置 |
|---|---|---|
| 提示词里的 `<`/`>`/`->` 丢失 | Windows 下 pi 走 argv 被 cmd.exe 解析 | 已内置 stdin 传题；勿改为 argv |
| `pi: command not found` / 找不到可执行 | Windows PATHEXT | `eval.py` 自动解析完整路径；确认 `pi` 在 PATH |
| 模型解析失败 | 模式名不对 | `pi --list-models deepseek` 核对，`--model provider/id` |
| 回答未更新 | 已存在回答默认跳过 | 加 `--force` |
| 图库条件查不到数据 | govio-cli 配置/图库缺失 | `govio-cli query -c "MATCH (n) RETURN count(n)"` 验证 |
| score 报告参考答案为 `(空)` | 基准库未构建 | 先 `eval.py build` |
| score 提取到 0 条语句 | 回答无代码块且非裸语句，或 pi 超时 | 看回答文件确认；`--timeout` 加大后重跑 score |
| negative 题检查项不通过但执行结果为空 | 空结果集回答不了文字型检查项 | 人工复核报告中的执行结果（两条件同等受影响） |

## 七、有效性注意（oracle 泄漏）

`prompts.json` 中的 `oracle_sql`/`expected_summary`/`checks` 就在仓库内，执行 agent 带工具时
**存在读取泄漏的可能**（历史审计曾发现）。要求严格盲测时：

- 执行后审计回答侧的会话/命令记录，剔除读过 `prompts.json` 的样本重跑；
- 或把执行环境与题库隔离（只给材料文档与数据访问路径）。
