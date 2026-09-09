# Phase 5: 关联拓展

**目标**：用外部维度数据 enrich 主实体，做更深层的交叉分析。

## 典型场景

- 主实体通过关联实体间接关联到维度实体
- 分析主实体在维度实体各属性上的分布
- 发现跨机构/跨维度的异常模式

## 基本模式

### 加载维度数据（如果尚未加载）

```bash
govio-cli observe load --name df_dimension --datasource {ds} --sql "SELECT ..."
# 或从文件加载
govio-cli observe load --name df_dimension --memory --sql "SELECT * FROM READ_PARQUET('path')"
```

### Enrich JOIN

```sql
SELECT
  a.*,                         -- 主实体全字段
  c.dim_field1,                -- 维度字段
  c.dim_field2
FROM df_main a
LEFT JOIN df_dimension c ON a.key = c.key
```

```bash
govio-cli observe load --name df_enriched --memory --sql "<SQL>"
```

### 维度分布分析

对 enriched 数据做 GROUP BY（复用 Phase 3 模式）：

```sql
SELECT dim_field1, COUNT(*) AS cnt
FROM df_enriched
GROUP BY dim_field1
ORDER BY cnt DESC
```

## 跨机构分析

当同一实体出现在多个组织维度下：

### 识别跨机构实体

```sql
SELECT key, COUNT(DISTINCT org) AS org_cnt,
  LISTAGG(DISTINCT org) AS orgs
FROM df_enriched
GROUP BY key
HAVING COUNT(DISTINCT org) > 1
```

加载为 DataFrame 节点。

### 跨机构明细

```sql
SELECT e.org, e.key, e.name, e.other_fields
FROM df_enriched e
INNER JOIN (
  SELECT key FROM df_enriched
  GROUP BY key HAVING COUNT(DISTINCT org) > 1
) cross_org ON e.key = cross_org.key
ORDER BY e.key, e.org
```

## 关联覆盖率分析

分析主实体在维度实体上的覆盖情况：

```sql
SELECT
  COUNT(*) AS total,
  COUNT(c.key) AS covered,
  COUNT(*) - COUNT(c.key) AS uncovered,
  ROUND(COUNT(c.key) * 100.0 / COUNT(*), 1) AS coverage_pct
FROM df_main a
LEFT JOIN df_dimension c ON a.key = c.key
```

按机构分组看覆盖率：

```sql
SELECT
  a.org,
  COUNT(*) AS total,
  COUNT(c.key) AS covered,
  ROUND(COUNT(c.key) * 100.0 / COUNT(*), 1) AS coverage_pct
FROM df_main a
LEFT JOIN df_dimension c ON a.key = c.key
GROUP BY a.org
ORDER BY coverage_pct
```

用 `govio_show_chart` 出 bar chart 展示各机构覆盖率。

## 产出

- DataFrame 节点（enriched 数据）
- chart 节点（维度分布 / 覆盖率）
- DataFrame 节点（跨机构明细，可选）
