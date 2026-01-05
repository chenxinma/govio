# 数据标准推荐器使用指南

## 快速开始

### 1. 基本用法

```python
from govio.metadata.standard import StandardLoader
from govio.metadata.database import DatabseLoader
from govio.metadata import create_recommender

# 初始化加载器
std_loader = StandardLoader(
    db='mysql://user:password@host:port/database',
    workspace_uuid='your-workspace-uuid'
)

db_loader = DatabseLoader(
    db='mysql://user:password@host:port/database',
    workspace_uuid='your-workspace-uuid',
    schema_limits=['schema1', 'schema2']  # 可选：限制查询的schema
)

# 加载数据
std_compliance = std_loader.StdCompliance  # 已贯标列
all_columns = db_loader.Col  # 所有列

# 创建推荐器
recommender = create_recommender(
    std_compliance=std_compliance,
    k_neighbors=5,  # 使用5个最近邻
    top_n=3  # 返回Top 3推荐
)

# 批量推荐
recommendations = recommender.batch_recommend(all_columns)

# 保存结果
recommendations.to_csv('recommendations.csv', index=False)

# 查看推荐结果
print(recommendations.head())
```

### 2. 自定义权重

```python
from govio.metadata import StandardRecommender

# 自定义特征权重
custom_weights = {
    'table': 0.30,    # 表名权重提高到30%（利用表名限定作用范围）
    'name': 0.35,     # 列名权重35%
    'comment': 0.20,  # 列注释权重20%
    'type': 0.10,     # 数据类型权重10%
    'numeric': 0.05   # 数值特征权重降低到5%
}

# 创建自定义推荐器
recommender = StandardRecommender(
    std_compliance=std_compliance,
    weights=custom_weights,
    k_neighbors=10,
    top_n=5
)

# 生成推荐
recommendations = recommender.batch_recommend(all_columns)
```

### 3. 单个列推荐

```python
from govio.metadata import StandardRecommender

# 创建推荐器
recommender = StandardRecommender(std_compliance=std_compliance)

# 定义一个列
column_data = pd.Series({
    'column': 'schema.table.user_id',
    'column_name': 'user_id',
    'full_table_name': 'schema.user_info',
    'name': '用户信息表',
    'table_name': 'user_info',  # 从 full_table_name 提取（schema.table_name）
    'dtype': 'bigint',
    'size': 20,
    'precision': 0,
    'scale': 0
})

# 为单个列推荐
recommendations = recommender.recommend(column_data)

# 打印推荐结果
for rec in recommendations:
    print(f"Rank {rec['rank']}: {rec['standard_id']} - {rec['standard_name']} (Score: {rec['score']})")
```

### 4. 评估推荐器性能

```python
from govio.metadata import StandardRecommender

# 创建推荐器
recommender = StandardRecommender(std_compliance=std_compliance)

# 准备测试数据（使用部分已贯标列作为测试集）
test_columns = std_compliance.sample(frac=0.2, random_state=42)
test_standards = dict(zip(
    test_columns['column'],
    test_columns['standard_id']
))

# 评估性能
metrics = recommender.evaluate(test_columns, test_standards)

print(f"准确率: {metrics['accuracy']:.2%}")
print(f"覆盖率: {metrics['coverage']:.2%}")
print(f"Top-{recommender.top_n}准确率: {metrics['top_n_accuracy']:.2%}")
print(f"测试样本数: {metrics['total_samples']}")
```

### 5. 处理已贯标列

```python
from govio.metadata import StandardRecommender

# 创建推荐器
recommender = StandardRecommender(std_compliance=std_compliance)

# 批量推荐（自动排除已贯标列）
recommendations = recommender.batch_recommend(
    all_columns,
    exclude_compliant=True  # 默认为True，推荐未贯标的列
)

# 或者包含已贯标列（用于验证）
recommendations_with_compliant = recommender.batch_recommend(
    all_columns,
    exclude_compliant=False
)
```

## 输出格式说明

### 推荐结果DataFrame

| 列名 | 类型 | 说明 |
|------|------|------|
| column | str | 完整列名（schema.table.column） |
| column_name | str | 列名 |
| full_table_name | str | 完整表名（schema.table） |
| name | str | 列注释 |
| table_name | str | 表名（从 full_table_name 的 schema.table_name 格式中提取，仅保留 table_name 部分） |
| dtype | str | 原始数据类型 |
| data_type | str | 标准化后的数据类型 |
| recommended_standard_id | str | 推荐的数据标准ID（Top 1） |
| recommended_standard_name | str | 推荐的数据标准名称（Top 1） |
| recommendation_score | float | 推荐分数（Top 1） |
| top_recommendations | str (JSON) | Top N推荐结果（JSON格式） |

### top_recommendations JSON格式

```json
[
    {
        "standard_id": "STD001",
        "standard_name": "用户ID",
        "score": 0.95,
        "rank": 1
    },
    {
        "standard_id": "STD002",
        "standard_name": "客户编号",
        "score": 0.87,
        "rank": 2
    },
    {
        "standard_id": "STD003",
        "standard_name": "人员标识",
        "score": 0.76,
        "rank": 3
    }
]
```

## 高级用法

### 1. 调整相似度阈值

```python
from govio.metadata import StandardRecommender

# 设置更高的相似度阈值（只推荐相似度>0.5的结果）
recommender = StandardRecommender(
    std_compliance=std_compliance,
    min_similarity=0.5
)

recommendations = recommender.batch_recommend(all_columns)
```

### 2. 分析推荐结果

```python
import pandas as pd

# 生成推荐
recommendations = recommender.batch_recommend(all_columns)

# 统计推荐分数分布
print(recommendations['recommendation_score'].describe())

# 查看高置信度推荐
high_confidence = recommendations[recommendations['recommendation_score'] > 0.8]
print(f"高置信度推荐数量: {len(high_confidence)}")

# 查看无推荐结果
no_recommendation = recommendations[recommendations['recommendation_score'] == 0]
print(f"无推荐数量: {len(no_recommendation)}")

# 按数据标准统计推荐数量
if 'recommended_standard_name' in recommendations.columns:
    std_counts = recommendations['recommended_standard_name'].value_counts()
    print("\n推荐最多的数据标准:")
    print(std_counts.head(10))
```

### 3. 参数调优

```python
from sklearn.model_selection import ParameterGrid
from govio.metadata import StandardRecommender

# 定义参数网格
param_grid = {
    'k_neighbors': [3, 5, 7, 10],
    'top_n': [1, 3, 5],
    'min_similarity': [0.3, 0.4, 0.5]
}

# 网格搜索最佳参数
best_score = 0
best_params = None

for params in ParameterGrid(param_grid):
    recommender = StandardRecommender(
        std_compliance=std_compliance,
        **params
    )
    
    # 交叉验证
    metrics = recommender.evaluate(test_columns, test_standards)
    
    if metrics['accuracy'] > best_score:
        best_score = metrics['accuracy']
        best_params = params

print(f"最佳参数: {best_params}")
print(f"最佳准确率: {best_score:.2%}")
```

### 4. 批量处理多个schema

```python
from govio.metadata.database import DatabseLoader
from govio.metadata import create_recommender

schemas = ['schema1', 'schema2', 'schema3']
all_recommendations = []

# 为每个schema生成推荐
for schema in schemas:
    db_loader = DatabseLoader(
        db='mysql://user:password@host:port/database',
        workspace_uuid='your-uuid',
        schema_limits=[schema]
    )
    
    columns = db_loader.Col
    recommendations = recommender.batch_recommend(columns)
    recommendations['schema'] = schema
    
    all_recommendations.append(recommendations)

# 合并所有结果
final_results = pd.concat(all_recommendations, ignore_index=True)
final_results.to_csv('all_recommendations.csv', index=False)
```

## 性能优化建议

### 1. 数据预处理

```python
# 预处理已贯标列数据，提高推荐速度
std_compliance = std_loader.StdCompliance
std_compliance = std_compliance.drop_duplicates(subset=['standard_id', 'column_name'])

# 过滤低质量数据
std_compliance = std_compliance[std_compliance['name'].str.len() > 0]
```

### 2. 分批处理

```python
# 对于大量列，分批处理
batch_size = 1000
all_recommendations = []

for i in range(0, len(all_columns), batch_size):
    batch = all_columns.iloc[i:i+batch_size]
    batch_recs = recommender.batch_recommend(batch)
    all_recommendations.append(batch_recs)
    print(f"已处理 {i+len(batch)}/{len(all_columns)} 列")

final_results = pd.concat(all_recommendations, ignore_index=True)
```

### 3. 缓存推荐结果

```python
import pickle

# 保存推荐器
with open('recommender.pkl', 'wb') as f:
    pickle.dump(recommender, f)

# 加载推荐器
with open('recommender.pkl', 'rb') as f:
    recommender = pickle.load(f)
```

## 常见问题

### Q1: 为什么有些列没有推荐结果？

**A:** 可能的原因：
1. 相似度低于 `min_similarity` 阈值
2. 已贯标列数据不足，无法找到相似列
3. 列特征（列名、注释、类型）与已贯标列差异太大

**解决方案：**
- 降低 `min_similarity` 阈值
- 增加 `k_neighbors` 数量
- 增加已贯标列的训练数据

### Q2: 如何提高推荐准确率？

**A:** 建议方法：
1. 增加高质量的已贯标列数据
2. 调整特征权重，根据业务特点优化
3. 使用交叉验证选择最佳参数
4. 添加更多特征（如业务域、数据质量指标等）
5. 确保表名和表注释的准确性，表名特征能有效限定列的作用范围，避免跨业务域的错误推荐

### Q4: 表名特征在推荐中起什么作用？

**A:** 表名特征的作用包括：
1. **限定作用范围**：相同或相似表中的列更可能使用相同的数据标准
2. **避免跨域错误**：防止将不同业务域的列错误匹配（如财务表的金额列推荐给用户表的年龄列）
3. **提高推荐精度**：表名提供了重要的上下文信息，结合列名和列注释可以更准确地判断列的语义
4. **支持表级推荐**：对于同一表中的多个列，可以基于表名进行批量推荐，提高一致性
5. **跨 schema 匹配**：仅使用 `table_name` 部分（从 `full_table_name` 的 `schema.table_name` 格式中提取），忽略 schema 差异，支持跨环境（如 prod、test）的相似表名匹配

**示例**：
- `user_info.user_id` 和 `user_info.user_name` 更可能使用相同的数据标准
- `order_info.order_id` 和 `order_detail.order_id` 虽然表名不同但相似，也可能使用相同的数据标准
- `finance.amount` 和 `user.age` 虽然都是数值类型，但由于表名差异大，不会互相推荐
- `prod.user_info.user_id` 和 `test.user_info.user_id` 虽然 schema 不同，但 `table_name` 相同，会互相推荐

### Q3: 推荐分数的含义是什么？

**A:** 推荐分数表示推荐的可信度：
- 0.9-1.0: 非常可信，推荐结果与目标列高度相似
- 0.7-0.9: 可信，推荐结果较为匹配
- 0.5-0.7: 一般，需要人工审核
- <0.5: 不推荐，建议人工处理

## 最佳实践

1. **数据质量优先**：确保已贯标列的数据质量，避免噪声影响推荐
2. **定期更新**：定期更新已贯标列数据，保持推荐器的准确性
3. **人工审核**：对推荐结果进行人工审核，特别是低分数的推荐
4. **持续优化**：根据实际使用情况，持续调整参数和权重
5. **监控指标**：定期监控推荐器的性能指标，及时发现和解决问题

## 示例代码

完整示例代码请参考项目中的 `examples/` 目录（如果存在）或联系技术支持。