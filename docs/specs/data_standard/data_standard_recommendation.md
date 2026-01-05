# 数据标准贯标估计系统设计文档

## 1. 概述

### 1.1 背景
在数据治理过程中，需要对数据库中的列进行数据标准贯标。目前已有一部分列已经完成了贯标（存储在 `StdCompliance` 中），需要为未贯标的列推荐合适的数据标准。

### 1.2 目标
基于协同过滤算法，利用已贯标列的数据，为未贯标的列推荐最可能的数据标准，提高数据标准贯标的效率和准确性。

## 2. 数据结构分析

### 2.1 StdCompliance（已贯标列）
来源：`src/govio/metadata/standard.py` 中的 `StandardLoader.StdCompliance`

主要字段：
- `standard_id`: 数据标准ID
- `database_name`: 数据库名
- `full_table_name`: 完整表名
- `column_name`: 列名
- `name`: 列注释
- `data_entity_type`: 数据实体类型
- `dtype`: 原始数据类型
- `size`: 长度
- `precision`: 精度
- `scale`: 标度

### 2.2 Column（所有列）
来源：`src/govio/metadata/database.py` 中的 `DatabseLoader.Col`

主要字段：
- `column`: 完整列名（schema.table.column）
- `column_name`: 列名
- `name`: 列注释
- `full_table_name`: 完整表名
- `data_entity_type`: 数据实体类型
- `dtype`: 原始数据类型
- `size`: 长度
- `precision`: 精度
- `scale`: 标度
- `order_no`: 列序号
- `data_type`: 标准化后的数据类型

## 3. 协同过滤算法设计

### 3.1 算法选择
采用**基于项目的协同过滤（Item-based Collaborative Filtering）**，其中：
- **用户（User）**：未贯标的列
- **项目（Item）**：已贯标的列
- **推荐内容**：数据标准ID

### 3.2 相似度计算

#### 3.2.1 特征向量构建
为每个列构建多维特征向量：

1. **表名特征**：表名的词向量或字符串相似度。从 `full_table_name` 中提取 `table_name` 部分（即 `schema.table_name` 格式中的 `table_name`），使用该部分计算相似度。表名作为重要的上下文信息，能够很好地限定列的作用范围。相同或相似表中的列更可能使用相同的数据标准。
2. **列名特征**：列名的词向量或字符串相似度
3. **列注释特征**：列注释的词向量或字符串相似度
4. **数据类型特征**：数据类型的编码
5. **数值特征**：size、precision、scale 的归一化值

#### 3.2.2 相似度度量方法

采用加权综合相似度：

```
similarity = w1 * table_sim + w2 * name_sim + w3 * comment_sim + w4 * type_sim + w5 * numeric_sim
```

其中：
- `table_sim`: 表名相似度（使用编辑距离或余弦相似度，仅使用从 `full_table_name` 中提取的 `table_name` 部分）
- `name_sim`: 列名相似度（使用编辑距离或余弦相似度）
- `comment_sim`: 列注释相似度（使用余弦相似度）
- `type_sim`: 数据类型相似度（1表示相同，0表示不同）
- `numeric_sim`: 数值特征相似度（欧氏距离或余弦相似度）
- `w1, w2, w3, w4, w5`: 权重系数（可配置）

**表名相似度的重要性**：
表名提供了列的上下文信息，能够有效限定列的作用范围。例如：
- 相同表中的列（如 `user_info.user_id` 和 `user_info.user_name`）更可能使用相同的数据标准
- 相似表名的列（如 `order_info.order_id` 和 `order_detail.order_id`）也可能使用相同的数据标准
- 表名相似度可以避免跨业务域的错误推荐（如避免将财务表的金额列推荐给用户表的年龄列）

**表名提取规则**：
`full_table_name` 的格式为 `schema.table_name`，通过提取 `.` 后的部分得到 `table_name`，用于计算表名相似度。这样可以：
- 聚焦于表本身的语义，忽略 schema 的差异
- 支持跨 schema 的相似表名匹配（如 `prod.user_info` 和 `test.user_info`）
- 提高推荐系统的灵活性和泛化能力

### 3.3 推荐算法流程

```
输入：
  - all_columns: 所有列（DataFrame）
  - std_compliance: 已贯标列（DataFrame）
  
输出：
  - recommendations: 推荐结果（DataFrame）

步骤：
1. 数据预处理
   - 识别已贯标列和未贯标列
   - 构建已贯标列的数据标准索引
   - 从 `full_table_name` 中提取 `table_name` 部分（格式：`schema.table_name`，提取 `.` 后的部分）

2. 特征提取
   - 为每个列提取特征向量
   - 标准化数值特征

3. 相似度计算
   - 对于每个未贯标列，计算与所有已贯标列的相似度

4. K近邻选择
   - 选择相似度最高的K个已贯标列

5. 推荐生成
   - 基于K个近邻的数据标准，计算推荐分数
   - 返回推荐分数最高的N个数据标准
```

### 3.4 推荐分数计算

对于未贯标列 `u`，数据标准 `s` 的推荐分数：

```
score(u, s) = Σ [similarity(u, v) * weight(v, s)] for v in KNN(u) and standard(v) == s
```

其中：
- `KNN(u)`：未贯标列 `u` 的K个最近邻已贯标列
- `similarity(u, v)`：列 `u` 和列 `v` 的相似度
- `weight(v, s)`：权重（可以基于数据标准的权威性、质量等）

## 4. 实现方案

### 4.1 核心类设计

```python
class StandardRecommender:
    """数据标准推荐器"""
    
    def __init__(self, std_compliance: pd.DataFrame,
                 weights: dict[str, float] = None,
                 k_neighbors: int = 5,
                 top_n: int = 3):
        """
        Args:
            std_compliance: 已贯标列数据（需包含 full_table_name, name 等字段）
            weights: 特征权重 {'table': 0.25, 'name': 0.30, 'comment': 0.25, 'type': 0.10, 'numeric': 0.10}
            k_neighbors: K近邻数量
            top_n: 返回推荐数量
        """
        pass
    
    def fit(self, all_columns: pd.DataFrame):
        """训练推荐器，构建特征索引"""
        pass
    
    def recommend(self, column: pd.Series) -> list[dict]:
        """为单个列推荐数据标准"""
        pass
    
    def batch_recommend(self, columns: pd.DataFrame) -> pd.DataFrame:
        """批量推荐"""
        pass
```

### 4.2 相似度计算实现

#### 4.2.1 字符串相似度
使用 `difflib.SequenceMatcher` 或 `Levenshtein` 距离

#### 4.2.2 数值相似度
使用归一化后的欧氏距离

### 4.3 性能优化

1. **特征缓存**：缓存已计算的列特征向量
2. **索引优化**：为已贯标列建立索引，加速相似度搜索
3. **批量处理**：支持批量推荐，减少重复计算
4. **并行计算**：使用多进程加速相似度计算

## 5. 配置参数

### 5.1 默认权重配置
```python
DEFAULT_WEIGHTS = {
    'table': 0.25,     # 表名权重（仅使用从 full_table_name 提取的 table_name）
    'name': 0.30,      # 列名权重
    'comment': 0.25,   # 列注释权重
    'type': 0.10,      # 数据类型权重
    'numeric': 0.10    # 数值特征权重
}
```

**权重说明**：
- `table` (0.25)：表名权重较高，因为表名提供了重要的上下文信息，能有效限定列的作用范围。仅使用 `table_name` 部分（从 `full_table_name` 的 `schema.table_name` 格式中提取）
- `name` (0.30)：列名权重最高，因为列名是最直接的特征
- `comment` (0.25)：列注释权重次之，提供语义信息
- `type` (0.10) 和 `numeric` (0.10)：数据类型和数值特征权重较低，作为辅助特征

### 5.2 默认算法参数
```python
DEFAULT_K_NEIGHBORS = 5   # K近邻数量
DEFAULT_TOP_N = 3         # 返回推荐数量
MIN_SIMILARITY = 0.3      # 最小相似度阈值
```

## 6. 输出格式

### 6.1 推荐结果格式

```python
{
    'column': 'schema.table.column',
    'column_name': 'column_name',
    'name': '列注释',
    'recommendations': [
        {
            'standard_id': 'STD001',
            'standard_name': '数据标准名称',
            'score': 0.95,
            'rank': 1
        },
        {
            'standard_id': 'STD002',
            'standard_name': '数据标准名称',
            'score': 0.87,
            'rank': 2
        }
    ]
}
```

### 6.2 批量推荐结果格式

DataFrame，包含以下列：
- `column`: 完整列名
- `column_name`: 列名
- `name`: 列注释
- `recommended_standard_id`: 推荐的数据标准ID
- `recommended_standard_name`: 推荐的数据标准名称
- `recommendation_score`: 推荐分数
- `top_3_standards`: Top 3推荐（JSON格式）

## 7. 使用示例

```python
from govio.metadata.standard import StandardLoader
from govio.metadata.database import DatabseLoader
from govio.metadata.recommender import StandardRecommender

# 加载数据
std_loader = StandardLoader(db='...', workspace_uuid='...')
db_loader = DatabseLoader(db='...', workspace_uuid='...')

std_compliance = std_loader.StdCompliance
all_columns = db_loader.Col

# 创建推荐器
recommender = StandardRecommender(
    std_compliance=std_compliance,
    k_neighbors=5,
    top_n=3
)

# 训练
recommender.fit(all_columns)

# 批量推荐
recommendations = recommender.batch_recommend(all_columns)

# 保存结果
recommendations.to_csv('recommendations.csv', index=False)
```

## 8. 评估指标

### 8.1 准确率（Accuracy）
推荐的数据标准与实际贯标标准的匹配率

### 8.2 覆盖率（Coverage）
能够给出推荐结果的未贯标列比例

### 8.3 排名准确率（Top-N Accuracy）
正确的数据标准是否出现在Top N推荐中

## 9. 扩展性

### 9.1 特征扩展
已实现的特征：
- ✅ 表名相似度（仅使用从 full_table_name 提取的 table_name 部分）

可以添加更多特征：
- 数据库类型相似度
- 业务域标签
- 数据质量指标
- Schema 相似度
- 表关联关系（外键、主键等）
- 列的使用频率和访问模式

### 9.2 算法扩展
可以支持更多推荐算法：
- 矩阵分解（Matrix Factorization）
- 深度学习推荐
- 混合推荐策略

## 10. 注意事项

1. **数据质量**：确保已贯标列的数据质量，避免噪声影响推荐准确性
2. **冷启动问题**：对于新数据标准，可能需要人工标注一些样本
3. **参数调优**：根据实际数据特点调整权重和K值
4. **性能监控**：监控推荐系统的性能和准确率，持续优化

## 11. 版本历史

- **v1.2** (2026-01-04): 优化表名特征，仅使用从 `full_table_name` 提取的 `table_name` 部分计算相似度，提高跨 schema 匹配能力
- **v1.1** (2026-01-04): 添加表名特征，利用表名限定列的作用范围，提高推荐准确性
- **v1.0** (2026-01-04): 初始设计，实现基于协同过滤的数据标准推荐