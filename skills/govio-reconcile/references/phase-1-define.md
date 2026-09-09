# Phase 1: 定义分析图

**目标**：将用户的自然语言分析需求转化为结构化的"实体-关联-口径"模型。

## 交互流程

### Step 1: 理解目标

问用户（一个一个问题来）：
1. "这次分析要解决什么问题？"
2. "涉及哪些数据系统？"
3. "最终报告给谁看？需要什么粒度？"

### Step 2: 识别实体

从用户描述中提取实体。每个实体需要：

| 属性 | 说明 | 示例 |
|------|------|------|
| 名称 | 简短标识 | order, customer, invoice |
| 数据来源 | 数据库+表 或 文件 | crm.t_order |
| 粒度 | 主键/唯一标识 | order_id |
| 业务含义 | 一句话说明 | CRM 系统的有效订单 |

### Step 3: 识别关联

实体间通过什么字段关联：

```
A.[字段X] → B.[字段Y]
```

常见关联模式：
- 同名字段直接 JOIN：`A.customer_id = B.customer_id`
- 编码映射：`A.partner_code = B.cust_code`（不同系统用不同字段名）
- 多字段联合：`A.region + A.channel = B.region + B.channel`

### Step 4: 定义口径

主实体的筛选条件，通常包含：
- 状态筛选（如：有效/已完成）
- 范围筛选（如：某个区域/渠道）
- 时间筛选（如：创建时间早于某个日期）

**口径的关键是"排除什么"**：明确哪些数据不在分析范围内。

### Step 5: 输出分析图

```markdown
## 分析图: {分析名称}

### 实体

| # | 实体 | 数据来源 | 粒度(主键) | 说明 |
|---|------|---------|-----------|------|
| A | order | crm.t_order | order_id | CRM 订单 |
| B | customer | erp.t_customer | cust_code | ERP 客户主数据 |
| C | invoice | finance.t_invoice | invoice_id | 财务系统发票 |

### 关联链
order.[customer_id] → customer.[cust_code] → invoice.[cust_code]

### 分析口径
- **主实体 order**: status IN ('active','pending'), region='华东区'
- **时间截面**: create_time < 2026-01-01
- **排除范围**: 测试订单（order_type = 'test'）
```

### Step 6: 确认

向用户展示分析图，逐项确认：
1. "实体列表是否完整？"
2. "关联键是否正确？"
3. "口径条件是否准确？"

确认后写入 Plan 文件。

## Plan 模板

```markdown
# Reconcile 对账计划: {分析名称}

**目标**: {一句话}
**创建时间**: YYYY-MM-DD

## 分析图
{上面输出的分析图}

## 执行计划
- [ ] Phase 2: 加载实体 A ({数据源})
- [ ] Phase 2: 加载实体 B ({数据源})
- [ ] Phase 3: 维度聚合画像
- [ ] Phase 4: A-B 跨系统对比
- [ ] Phase 5: 关联拓展（可选）
```
