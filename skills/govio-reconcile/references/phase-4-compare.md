# Phase 4: 跨系统对比

**目标**：对有关联的实体对做匹配分析和字段级比对。

## 4.1 匹配概览

### 正向匹配（源 → 目标）

```sql
SELECT
  COUNT(*) AS source_total,
  COUNT(b.join_key) AS matched,
  COUNT(*) - COUNT(b.join_key) AS unmatched,
  ROUND(COUNT(b.join_key) * 100.0 / COUNT(*), 1) AS match_rate_pct
FROM df_source a
LEFT JOIN df_target b ON a.join_key = b.join_key
```

### 反向匹配（目标 → 源）

```sql
SELECT COUNT(*) AS target_only_cnt
FROM df_target b
LEFT JOIN df_source a ON b.join_key = a.join_key
WHERE a.join_key IS NULL
```

### 三域统计（交集 / 仅源 / 仅目标）

```sql
SELECT '仅源表' AS scope, COUNT(DISTINCT a.key) AS cnt
FROM df_source a LEFT JOIN df_target b ON a.key = b.key WHERE b.key IS NULL
UNION ALL
SELECT '仅目标表', COUNT(DISTINCT b.key)
FROM df_target b LEFT JOIN df_source a ON b.key = a.key WHERE a.key IS NULL
UNION ALL
SELECT '交集', COUNT(DISTINCT a.key)
FROM df_source a JOIN df_target b ON a.key = b.key
```

**汇报用户**，询问是否深入看不匹配记录。

## 4.2 不匹配明细

### 仅在源表的记录

```sql
SELECT a.join_key, a.name, a.other_fields
FROM df_source a
LEFT JOIN df_target b ON a.join_key = b.join_key
WHERE b.join_key IS NULL
```

### 仅在目标表的记录

```sql
SELECT b.join_key, b.name, b.other_fields
FROM df_target b
LEFT JOIN df_source a ON b.join_key = a.join_key
WHERE a.join_key IS NULL
```

加载为 DataFrame 节点，方便用户查看和导出。

## 4.3 字段级比对

### 方式 A: datacompy 全量比对

```bash
govio-cli observe compare --source df_source --target df_target --join-columns join_key
```

产出 report 节点（datacompy 报告），包含：
- Schema 差异（列名不同）
- 行匹配率
- 每列的值差异

### 方式 B: 定向字段比对

当只需要比对特定字段时：

```sql
SELECT
  a.join_key,
  a.field1 AS source_field1,
  b.field1 AS target_field1,
  a.field2 AS source_field2,
  b.field2 AS target_field2
FROM df_source a
JOIN df_target b ON a.join_key = b.join_key
WHERE a.field1 != b.field1
   OR a.field2 != b.field2
   OR (a.field1 IS NULL AND b.field1 IS NOT NULL)
   OR (a.field1 IS NOT NULL AND b.field1 IS NULL)
```

加载为 DataFrame 节点。

### 方式 C: NULL 编码检查

常见数据质量问题：一侧有编码、另一侧为 NULL。

```sql
SELECT a.join_key, a.name,
  a.code AS source_code,
  b.code AS target_code
FROM df_source a
JOIN df_target b ON a.join_key = b.join_key
WHERE (a.code IS NULL AND b.code IS NOT NULL)
   OR (a.code IS NOT NULL AND b.code IS NULL)
```

## 产出

- DataFrame 节点（匹配统计）
- report 节点（比对报告）
- DataFrame 节点（差异明细，可选导出）
