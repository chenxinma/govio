# EDA 技能设计文档

**创建日期**: 2026-07-01
**状态**: 设计阶段
**负责人**: Claude Code

## 1. 概述

### 1.1 目标

创建一个标准化的探索性数据分析（EDA）技能，作为 `govio-observe` 的扩展，提供：
- 标准化的数据探查流程
- 自适应探查深度
- 综合输出（报告 + 可视化 + 数据质量指标）

### 1.2 设计原则

- **流程标准化**: 定义 5 个标准阶段的线性流水线
- **工具复用**: 最大化复用现有 `govio-cli observe` 命令
- **渐进式探查**: 支持断点续查和单独执行某个阶段
- **综合输出**: 生成结构化报告、可视化图表和数据质量指标

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      govio-eda 技能                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  │ 阶段1    │ │ 阶段2    │ │ 阶段3    │ │ 阶段4    │ │ 阶段5    │
│  │ 数据集   │→│ 关系     │→│ 关系     │→│ 一致性   │→│ 报告     │
│  │ 定义     │ │ 推断     │ │ 核查     │ │ 检查     │ │ 生成     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
│       ↓            ↓            ↓            ↓            ↓
│   元信息JSON    关系JSON    验证JSON    一致性JSON   综合报告
└─────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │         govio-cli observe            │
        │  (load / explore / compare / chart)  │
        └──────────────────────────────────────┘
```

### 2.2 技能结构

```
skills/
├── govio-observe/           # 现有主控技能
│   └── SKILL.md
├── observe-dataset-ops/     # 现有子技能
├── observe-explore-relations/
├── observe-compare-dfs/
└── govio-eda/               # 新增 EDA 技能
    └── SKILL.md
```

## 3. 流水线阶段设计

### 3.1 阶段1: 数据集定义 (Profile)

**目标**: 了解数据集的基本结构和特征

**执行步骤**:
1. 加载数据集 (`govio-cli observe load`)
2. 获取数据集元信息:
   - 行数、列数
   - 数据类型分布
   - 空值率
   - 唯一值统计
   - 数值列的基本统计量（均值、中位数、标准差）

**输出格式**:
```json
{
  "dataset": "customers",
  "rows": 10000,
  "columns": 15,
  "column_info": [
    {
      "name": "customer_id",
      "dtype": "int64",
      "null_rate": 0.0,
      "unique_rate": 1.0,
      "min": 1,
      "max": 10000,
      "mean": 5000.5
    }
  ],
  "data_quality": {
    "completeness": 0.95,
    "consistency": 0.98
  }
}
```

### 3.2 阶段2: 关系推断 (Infer)

**目标**: 发现数据集之间的潜在关联

**执行步骤**:
1. 使用 `govio-cli observe explore` 推断关系
2. 分析关系类型:
   - 外键关系（基于列名模式和值重叠）
   - 列名相似关系
   - 语义关联（基于数据字典）

**输出格式**:
```json
{
  "relations": [
    {
      "type": "foreign_key",
      "source": "orders.customer_id",
      "target": "customers.customer_id",
      "confidence": 0.95,
      "overlap_rate": 0.98
    },
    {
      "type": "column_similarity",
      "source": "customers.email",
      "target": "orders.customer_email",
      "similarity": 0.85
    }
  ]
}
```

### 3.3 阶段3: 关系核查 (Verify)

**目标**: 验证推断的关系是否真实有效

**执行步骤**:
1. 对推断的外键关系进行值重叠验证
2. 检查关系的完整性（是否存在孤立记录）
3. 验证关系的基数（1:1, 1:N, M:N）

**输出格式**:
```json
{
  "verified_relations": [
    {
      "relation": "orders.customer_id → customers.customer_id",
      "verified": true,
      "cardinality": "1:N",
      "orphan_records": {
        "source_only": 50,
        "target_only": 100
      },
      "integrity_score": 0.95
    }
  ]
}
```

### 3.4 阶段4: 一致性检查 (Check)

**目标**: 检查关联数据的一致性

**执行步骤**:
1. 使用 `govio-cli observe compare` 比对关联数据
2. 检查数据一致性:
   - 字段值一致性
   - 状态一致性
   - 业务规则一致性

**输出格式**:
```json
{
  "consistency_checks": [
    {
      "source": "customers",
      "target": "orders",
      "join_columns": ["customer_id"],
      "match_rate": 0.985,
      "mismatches": {
        "count": 150,
        "details": [
          {
            "customer_id": 12345,
            "field": "email",
            "source_value": "old@example.com",
            "target_value": "new@example.com"
          }
        ]
      }
    }
  ]
}
```

### 3.5 阶段5: 报告生成 (Report)

**目标**: 生成综合探查报告

**执行步骤**:
1. 汇总前 4 个阶段的结果
2. 生成可视化图表:
   - 数据质量分布图
   - 关系图谱
   - 一致性热力图
3. 输出结构化报告

**输出格式**:
- `report.md`: Markdown 格式的综合报告
- `quality_dist.png`: 数据质量分布图
- `relations.png`: 关系图谱
- `consistency_heatmap.png`: 一致性热力图
- `summary.json`: 结构化摘要

## 4. CLI 命令设计

### 4.1 完整流程命令

```bash
# 执行完整 EDA 流程
govio-cli eda run --datasource <ds> --tables t1,t2,t3 --output ./report

# 参数说明
# --datasource: 数据源名称
# --tables: 要探查的表列表（逗号分隔）
# --output: 输出目录
```

### 4.2 单阶段命令

```bash
# 阶段1: 数据集定义
govio-cli eda profile --name <df_name> [--output profile.json]

# 阶段2: 关系推断
govio-cli eda infer --dataframes df1,df2 [--output relations.json]

# 阶段3: 关系核查
govio-cli eda verify --source <source> --target <target> [--output verified.json]

# 阶段4: 一致性检查
govio-cli eda check --source <source> --target <target> --join-columns col1,col2 [--output consistency.json]

# 阶段5: 报告生成
govio-cli eda report --input ./eda-results --output ./report
```

### 4.3 辅助命令

```bash
# 查看 EDA 结果
govio-cli eda show --input ./eda-results

# 清理 EDA 临时文件
govio-cli eda clean --input ./eda-results
```

## 5. 文件结构设计

### 5.1 工作目录

```
.govio/eda/
├── datasets/                    # 数据集元信息
│   ├── customers_profile.json
│   └── orders_profile.json
├── relations/                   # 关系推断结果
│   └── inferred_relations.json
├── verified/                    # 关系核查结果
│   └── verified_relations.json
├── consistency/                 # 一致性检查结果
│   └── consistency_report.json
└── reports/                     # 最终报告
    ├── report.md
    ├── quality_dist.png
    ├── relations.png
    └── summary.json
```

### 5.2 中间结果格式

所有中间结果使用 JSON 格式，便于：
- 断点续查
- 结果复用
- 程序化处理

## 6. 与现有技能的集成

### 6.1 复用关系

```
govio-eda
    ├── 阶段1: observe-dataset-ops.load_df + 数据分析
    ├── 阶段2: observe-explore-relations
    ├── 阶段3: observe-compare-dfs (部分)
    ├── 阶段4: observe-compare-dfs
    └── 阶段5: observe-chart + 报告生成
```

### 6.2 命令映射

| EDA 阶段 | 现有 CLI 命令 | 说明 |
|---------|--------------|------|
| 数据集定义 | `govio-cli observe load` | 加载数据 |
| 关系推断 | `govio-cli observe explore` | 推断关系 |
| 关系核查 | `govio-cli observe compare` | 验证关系 |
| 一致性检查 | `govio-cli observe compare` | 比对数据 |
| 报告生成 | `govio-cli observe chart` | 生成图表 |

## 7. 错误处理

### 7.1 阶段级错误

- 每个阶段独立处理错误
- 错误信息记录到阶段输出 JSON
- 支持跳过失败阶段继续执行

### 7.2 数据级错误

- 数据加载失败: 提示检查数据源配置
- 关系推断失败: 提示数据量不足或无明显关系
- 一致性检查失败: 提示 join 列不存在或数据类型不匹配

### 7.3 恢复机制

- 支持从任意阶段开始执行
- 自动检测已完成的阶段
- 覆盖已有结果（可配置）

## 8. 测试策略

### 8.1 单元测试

- 测试每个阶段的独立功能
- 测试错误处理逻辑
- 测试输出格式正确性

### 8.2 集成测试

- 测试完整流程
- 测试阶段间的数据传递
- 测试与现有 CLI 命令的集成

### 8.3 端到端测试

- 测试真实数据集的探查
- 验证报告质量
- 性能测试

## 9. 实现计划

### 9.1 第一阶段: 基础框架

- 创建 `govio-eda` 技能目录和 SKILL.md
- 实现阶段1（数据集定义）
- 实现阶段5（报告生成）的基础版本

### 9.2 第二阶段: 核心功能

- 实现阶段2（关系推断）
- 实现阶段3（关系核查）
- 实现阶段4（一致性检查）

### 9.3 第三阶段: 增强功能

- 添加可视化图表
- 优化报告格式
- 添加断点续查功能

### 9.4 第四阶段: 集成测试

- 完善错误处理
- 性能优化
- 文档完善

## 10. 风险与缓解

### 10.1 技术风险

- **风险**: 大数据集性能问题
- **缓解**: 添加数据采样选项，限制单次处理数据量

### 10.2 集成风险

- **风险**: 与现有 CLI 命令不兼容
- **缓解**: 充分测试现有命令，必要时扩展现有命令

### 10.3 用户接受风险

- **风险**: 用户不习惯新的 EDA 流程
- **缓解**: 提供详细的使用文档和示例

## 11. 附录

### 11.1 参考文档

- [govio-observe 技能](../../skills/govio-observe/SKILL.md)
- [observe-dataset-ops 技能](../../skills/observe-dataset-ops/SKILL.md)
- [observe-explore-relations 技能](../../skills/observe-explore-relations/SKILL.md)
- [observe-compare-dfs 技能](../../skills/observe-compare-dfs/SKILL.md)

### 11.2 示例脚本

参考 `/data/home/macx/work/paper_work/docs/销售预测项目/06-数据探查规则.md` 中的探查规则实现。
