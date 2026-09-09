# Phase 3: 维度聚合与画像

**目标**：对主实体做 GROUP BY 聚合，用 chart 节点展示分布。

## 时机

主实体加载完成后、对比之前。先了解数据全貌，再做精细比对。

## 三种聚合模式

### 模式 1: Treemap 层级统计

适合多层级维度（如：状态 → 主体 → 机构）。

**Step 1: 打标聚合**

```sql
SELECT catalog1, catalog2, catalog3, COUNT(*) AS cnt
FROM (
  SELECT
    CASE
      WHEN status = 'A' THEN '类型A'
      WHEN status = 'B' THEN '类型B'
      ELSE '其他'
    END AS catalog1,
    CASE
      WHEN org = 'X' THEN '机构X'
      ELSE '其他机构'
    END AS catalog2,
    dept AS catalog3
  FROM df_main
)
GROUP BY catalog1, catalog2, catalog3
```

**Step 2: 加载聚合结果**

```bash
govio-cli observe load --name df_cat_stats --memory --sql "<SQL>"
```

**Step 3: 出 treemap**

调用 `govio_show_chart`，config.data 传 treemap trace：
- `type`: "treemap"
- `treeDf`: "df_cat_stats"
- `key`: "cnt"
- `groups`: ["catalog1", "catalog2", "catalog3"]

**CASE WHEN 规则**：
- 必须带 ELSE 兜底
- 层级数 ≤ 3
- 聚合后行数 > 200 时，用 ORDER BY cnt DESC LIMIT N 截断，剩余归入 "其他"

### 模式 2: Bar 排名统计

适合单维度 Top N（如：区域订单数排名）。

```sql
SELECT dimension, COUNT(*) AS cnt
FROM df_main
GROUP BY dimension
ORDER BY cnt DESC
LIMIT 20
```

调用 `govio_show_chart`：
- `type`: "bar"
- `x`: [维度值列表]
- `y`: [计数列表]

**注意**：bar chart 的 x/y 需要 agent 先查询数据再传入内联值。用 `govio-cli observe info --name {df} --rows 50` 取数据。

### 模式 3: 交叉统计表

适合需要同时看两个维度的交叉分布。

```sql
SELECT dim1, dim2, COUNT(*) AS cnt
FROM df_main
GROUP BY dim1, dim2
ORDER BY cnt DESC
```

结果加载为 DataFrame 节点（可预览），不一定要出 chart。

## 产出

- chart 节点（treemap 或 bar）
- 可选的 DataFrame 节点（聚合结果表）
