# Phase 2: 逐步加载

**目标**：按分析图逐个加载实体数据，每步产出 DataFrame 画布节点。

## 加载顺序

按关联链从左到右：先主实体 → 再关联实体 → 最后维度实体。

每加载完一个实体，暂停询问用户是否继续。

## 标准加载

```bash
govio-cli observe load --name {df_name} --datasource {ds} --sql "{SQL}"
```

**df 命名**：`{实体名}_{来源缩写}`，如 `order_crm`、`customer_erp`

## 分步筛选模式

当主实体需要复杂筛选时，分两步避免 SQL 过长：

```bash
# Step 1: 全量加载
govio-cli observe load --name order_all --datasource crm --sql "
  SELECT o.order_id, o.order_status, o.customer_id,
         c.cust_name, c.region, c.channel
  FROM t_order o
  LEFT JOIN t_customer c ON o.customer_id = c.cust_code
  ...
"

# Step 2: 业务口径筛选
govio-cli observe load --name order --memory --sql "
  SELECT * FROM order_all
  WHERE order_status IN ('active', 'pending')
    AND region = '华东区'
    AND order_type != 'test'
"
```

## 加载后检查

每个 DataFrame 加载后，确认：
- 行数是否合理（太多？太少？）
- 关键字段是否有 NULL（会影响后续 JOIN）
- 数据时间截面是否正确

```bash
govio-cli observe info --name {df_name} --rows 5
```

## 资源管理

加载较多实体后，检查已加载的 DataFrame：

```bash
govio-cli observe info --df
```

不再需要的中间 DataFrame 及时释放：

```bash
govio-cli observe release --name {df_temp}
```
