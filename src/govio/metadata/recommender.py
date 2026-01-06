"""数据标准推荐器 - 基于协同过滤算法"""

import heapq
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

DEFAULT_WEIGHTS = {
    'table': 0.20,     # 表名权重（仅使用从 full_table_name 提取的 table_name）
    'name': 0.26,      # 列名权重
    'comment': 0.22,   # 列注释权重
    'type': 0.22,      # 数据类型权重
    'numeric': 0.10    # 数值特征权重
}

DEFAULT_K_NEIGHBORS = 5   # K近邻数量
DEFAULT_TOP_N = 3         # 返回推荐数量
MIN_SIMILARITY = 0.7      # 最小相似度阈值
NGRAM_N = 2               # n-gram 的 n 值


class StandardRecommender:
    """数据标准推荐器 - 基于协同过滤算法
    
    利用已贯标列的数据，为未贯标的列推荐最可能的数据标准
    """
    
    def __init__(
        self,
        std_compliance: pd.DataFrame,
        weights: dict[str, float] | None = None,
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        top_n: int = DEFAULT_TOP_N,
        min_similarity: float = MIN_SIMILARITY
    ) -> None:
        """
        Args:
            std_compliance: 已贯标列数据，包含 standard_id, column_name, name, dtype, size, precision, scale
            weights: 特征权重 {'name': 0.35, 'comment': 0.35, 'type': 0.15, 'numeric': 0.15}
            k_neighbors: K近邻数量
            top_n: 返回推荐数量
            min_similarity: 最小相似度阈值，低于此值不推荐
        """
        self.std_compliance = std_compliance.copy()
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.k_neighbors = k_neighbors
        self.top_n = top_n
        self.min_similarity = min_similarity
        
        # 标准化权重
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # 特征缓存和向量化器
        self._std_features_matrix = None
        self._table_vectorizer = TfidfVectorizer(ngram_range=(1, NGRAM_N), lowercase=True, analyzer='char')
        self._name_vectorizer = TfidfVectorizer(ngram_range=(1, NGRAM_N), lowercase=True, analyzer='char')
        self._comment_vectorizer = TfidfVectorizer(ngram_range=(1, NGRAM_N), lowercase=True, analyzer='char')
        self._numeric_scaler = MinMaxScaler()
        self._standard_names: dict[str, str] = {}
        
        # 预处理已贯标列数据
        self._preprocess_std_data()
    
    def _extract_table_name(self, full_table_name: str) -> str:
        """从 full_table_name 中提取 table_name 部分
        
        Args:
            full_table_name: 完整表名，格式为 schema.table_name
        
        Returns:
            table_name 部分
        """
        if not full_table_name:
            return ''
        
        # 提取 . 后的部分
        parts = str(full_table_name).split('.')
        if len(parts) > 1:
            return parts[-1]
        return full_table_name
    
    def _preprocess_std_data(self) -> None:
        """预处理已贯标列数据，构建特征索引"""
        # 填充空值
        self.std_compliance['name'] = self.std_compliance['name'].fillna('')
        self.std_compliance['column_name'] = self.std_compliance['column_name'].fillna('')
        self.std_compliance['dtype'] = self.std_compliance['dtype'].fillna('')
        
        # 提取 table_name
        self.std_compliance['table_name'] = self.std_compliance['full_table_name'].apply(
            self._extract_table_name
        )
        
        # 提取数值特征用于标准化
        numeric_features = self.std_compliance[['size', 'precision', 'scale']].fillna(0)
        self._numeric_scaler.fit(numeric_features)
        
        # 构建数据标准名称映射
        if 'standard_name' in self.std_compliance.columns:
            self._standard_names = dict(zip(
                self.std_compliance['standard_id'],
                self.std_compliance['standard_name']
            ))
        
        # 训练 TF-IDF 向量化器并构建标准列特征矩阵
        self._precompute_std_features()
    
    def _precompute_std_features(self) -> None:
        """预计算所有已贯标列的特征矩阵"""
        # 训练向量化器
        table_texts = self.std_compliance['table_name'].tolist()
        name_texts = self.std_compliance['column_name'].tolist()
        comment_texts = self.std_compliance['name'].tolist()
        
        self._table_vectorizer.fit(table_texts)
        self._name_vectorizer.fit(name_texts)
        self._comment_vectorizer.fit(comment_texts)
        
        # 转换为稀疏向量
        table_features = self._table_vectorizer.transform(table_texts)
        name_features = self._name_vectorizer.transform(name_texts)
        comment_features = self._comment_vectorizer.transform(comment_texts)
        
        # 数值特征标准化
        numeric_features = self.std_compliance[['size', 'precision', 'scale']].fillna(0)
        numeric_features_scaled = self._numeric_scaler.transform(numeric_features)
        
        # 类型特征（转换为数值）
        dtype_values = self.std_compliance['dtype'].fillna('').astype(str)
        dtype_mapping = {dtype: idx for idx, dtype in enumerate(dtype_values.unique())}
        dtype_numeric = np.array([dtype_mapping[dtype] for dtype in dtype_values])
        dtype_features = dtype_numeric.reshape(-1, 1)
        
        # 归一化 dtype 特征
        from sklearn.preprocessing import MinMaxScaler
        dtype_scaler = MinMaxScaler()
        dtype_features_scaled = dtype_scaler.fit_transform(dtype_features)
        
        # 拼接所有特征
        from scipy.sparse import hstack, csr_matrix
        self._std_features_matrix = hstack([
            table_features * self.weights['table'],
            name_features * self.weights['name'],
            comment_features * self.weights['comment'],
            csr_matrix(dtype_features_scaled * self.weights['type']),
            csr_matrix(numeric_features_scaled * self.weights['numeric'])
        ])
        
        # 保存 dtype_scaler 供后续使用
        self._dtype_scaler = dtype_scaler
        self._dtype_mapping = dtype_mapping
    

    def _type_similarity(self, type1: str, type2: str) -> float:
        """计算数据类型相似度
        
        Args:
            type1: 数据类型1
            type2: 数据类型2
        
        Returns:
            相似度分数 [0, 1]
        """
        if not type1 or not type2:
            return 0.0
        
        type1 = str(type1).lower()
        type2 = str(type2).lower()
        
        # 完全匹配
        if type1 == type2:
            return 1.0
        
        # 部分匹配（如 varchar 和 char）
        if type1 in type2 or type2 in type1:
            return 0.8
        
        # 类型族匹配（如 int, bigint, smallint）
        type_families = {
            'int': ['int', 'integer', 'bigint', 'smallint', 'tinyint'],
            'float': ['float', 'double', 'decimal', 'numeric'],
            'str': ['varchar', 'char', 'text', 'string'],
            'date': ['date', 'time', 'datetime', 'timestamp']
        }
        
        for family, types in type_families.items():
            if any(t in type1 for t in types) and any(t in type2 for t in types):
                return 0.6
        
        return 0.0
    
    def find_k_neighbors(
        self,
        column: pd.Series,
        exclude_columns: set[str] | None = None
    ) -> list[tuple[int, float]]:
        """为单个列找到K个最相似的已贯标列（使用 TF-IDF 向量化器和批量相似度计算）
        
        Args:
            column: 目标列
            exclude_columns: 需要排除的列名集合（如已贯标的列）
        
        Returns:
            K个最近邻的索引和相似度 [(index, similarity), ...]
        """
        from scipy.sparse import csr_matrix, hstack
        
        exclude_columns = exclude_columns or set()
        
        # 构建目标列的特征向量
        table_text = [self._extract_table_name(str(column.get('full_table_name', '')))]
        name_text = [str(column.get('column_name', ''))]
        comment_text = [str(column.get('name', ''))]
        
        # 转换为 TF-IDF 向量
        table_vec = self._table_vectorizer.transform(table_text)
        name_vec = self._name_vectorizer.transform(name_text)
        comment_vec = self._comment_vectorizer.transform(comment_text)
        
        # 数值特征标准化
        numeric_scaled = column.get('size_scaled', np.ndarray([0, 0, 0]))
        
        # 类型特征
        dtype = str(column.get('dtype', ''))
        dtype_val = self._dtype_mapping.get(dtype, 0)
        dtype_vec = np.array([[dtype_val]])
        dtype_scaled = self._dtype_scaler.transform(dtype_vec)
        
        # 拼接目标列特征
        target_features = hstack([
            table_vec * self.weights['table'],
            name_vec * self.weights['name'],
            comment_vec * self.weights['comment'],
            csr_matrix(dtype_scaled * self.weights['type']),
            csr_matrix(numeric_scaled * self.weights['numeric'])
        ])
        
        
        # 批量计算相似度
        similarities = cosine_similarity(target_features, self._std_features_matrix).flatten() # type: ignore

        # 过滤掉需要排除的列
        for idx, std_row in self.std_compliance.iterrows():
            col_key = f"{std_row.get('full_table_name', '')}.{std_row.get('column_name', '')}"
            if col_key in exclude_columns:
                similarities[idx] = -1  # type: ignore # 设置为负值以排除
        
        # 使用堆找到最大的 K 个相似度
        neighbors = []
        for idx, sim in enumerate(similarities):
            if sim == -1:  # 跳过被排除的列
                continue
                
            if len(neighbors) < self.k_neighbors:
                heapq.heappush(neighbors, (sim, idx))
            else:
                # 如果当前相似度大于堆中最小的，替换
                if sim > neighbors[0][0]:
                    heapq.heapreplace(neighbors, (sim, idx))
        
        # 转换为降序排列的列表
        neighbors.sort(key=lambda x: x[0], reverse=True)
        
        # 返回 [(index, similarity), ...] 格式
        return [(idx, sim) for sim, idx in neighbors]
    
    def recommend(self, column: pd.Series) -> list[dict[str, Any]]:
        """为单个列推荐数据标准
        
        Args:
            column: 目标列，包含 column, column_name, name, dtype, size, precision, scale
        
        Returns:
            推荐结果列表 [{'standard_id': ..., 'score': ..., 'rank': ...}, ...]
        """
        # 找到K个最近邻
        neighbors = self.find_k_neighbors(column)
        
        # 如果没有足够相似的邻居，返回空列表
        if not neighbors or neighbors[0][1] < self.min_similarity:
            return []
        
        # 统计数据标准的推荐分数
        standard_scores: dict[str, float] = {}
        
        for idx, similarity in neighbors:
            std_row = self.std_compliance.iloc[idx]
            standard_id = std_row['standard_id']
            
            # 累加相似度作为推荐分数
            if standard_id not in standard_scores:
                standard_scores[standard_id] = 0.0
            standard_scores[standard_id] += similarity
        
        # 归一化分数
        max_score = max(standard_scores.values()) if standard_scores else 1.0
        standard_scores = {k: v / max_score for k, v in standard_scores.items()}
        
        # 按分数降序排序
        sorted_standards = sorted(
            standard_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 构建推荐结果
        recommendations = []
        for rank, (standard_id, score) in enumerate(sorted_standards[:self.top_n], 1):
            recommendations.append({
                'standard_id': standard_id,
                'standard_name': self._standard_names.get(standard_id, ''),
                'score': round(score, 4),
                'rank': rank
            })
        
        return recommendations
    
    def batch_recommend(
        self,
        columns: pd.DataFrame,
        exclude_compliant: bool = True
    ) -> pd.DataFrame:
        """批量推荐数据标准

        Args:
            columns: 所有列数据
            exclude_compliant: 是否排除已贯标的列

        Returns:
            推荐结果DataFrame
        """
        # 填充空值
        columns = columns.copy()
        columns['name'] = columns['name'].fillna('')
        columns['column_name'] = columns['column_name'].fillna('')
        columns = columns[~((columns['order_no']==1) & \
                            (columns['column_name'].str.lower() == "id"))] # skip 物理主键ID

        # 数据值标准化
        columns['size_scaled'] = [ s for s  in self._numeric_scaler.transform(columns[['size', 'precision', 'scale']]) ]

        # 构建已贯标列的键集合
        compliant_columns = set()
        if exclude_compliant:
            for _, row in self.std_compliance.iterrows():
                key = f"{row.get('full_table_name', '')}.{row.get('column_name', '')}"
                compliant_columns.add(key)

        # 为每个列生成推荐
        results = []
        for _, column in tqdm(columns.iterrows(), total=columns.shape[0]):
            col_key = f"{column.get('full_table_name', '')}.{column.get('column_name', '')}"

            # 如果是已贯标列，跳过
            if exclude_compliant and col_key in compliant_columns:
                continue

            # 生成推荐
            recommendations = self.recommend(column)

            # 提取 table_name
            full_table_name = column.get('full_table_name', '')
            table_name = self._extract_table_name(full_table_name)

            # 构建结果
            result = {
                'column': column.get('column', ''),
                'column_name': column.get('column_name', ''),
                'full_table_name': full_table_name,
                'name': column.get('name', ''),
                'table_name': table_name,
                'dtype': column.get('dtype', ''),
                'data_type': column.get('data_type', ''),
                'recommended_standard_id': '',
                'recommended_standard_name': '',
                'recommendation_score': 0.0,
                'top_recommendations': json.dumps(recommendations, ensure_ascii=False)
            }

            # 如果有推荐结果，填充推荐信息
            if recommendations:
                top_rec = recommendations[0]
                result['recommended_standard_id'] = top_rec['standard_id']
                result['recommended_standard_name'] = top_rec['standard_name']
                result['recommendation_score'] = top_rec['score']

            results.append(result)

        return pd.DataFrame(results)
    
    def evaluate(
        self,
        test_columns: pd.DataFrame,
        test_standards: dict[str, str]
    ) -> dict[str, float]:
        """评估推荐器的性能

        Args:
            test_columns: 测试列数据
            test_standards: 测试列的真实数据标准映射 {column: standard_id}

        Returns:
            评估指标 {'accuracy': ..., 'coverage': ..., 'top_n_accuracy': ..., 'total_samples': ...}
        """
        # 生成推荐
        recommendations = self.batch_recommend(test_columns, exclude_compliant=False)
        print(recommendations)

        # 计算准确率
        correct = 0
        total = len(recommendations)
        top_n_correct = 0

        for _, rec in recommendations.iterrows():
            col_key = rec['column']
            true_standard = test_standards.get(col_key)

            if true_standard:
                # 检查Top 1推荐是否正确
                if rec['recommended_standard_id'] == true_standard:
                    correct += 1

                # 检查Top N推荐中是否包含正确答案
                top_recs = json.loads(rec['top_recommendations'])
                if any(r['standard_id'] == true_standard for r in top_recs):
                    top_n_correct += 1

        # 计算指标
        accuracy = correct / total if total > 0 else 0.0
        coverage = sum(1 for r in recommendations['recommendation_score'] if r > 0) / total
        top_n_accuracy = top_n_correct / total if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'coverage': coverage,
            'top_n_accuracy': top_n_accuracy,
            'total_samples': total
        }


def create_recommender(
    std_compliance: pd.DataFrame,
    weights: dict[str, float] | None = None,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    top_n: int = DEFAULT_TOP_N
) -> StandardRecommender:
    """创建数据标准推荐器的工厂函数
    
    Args:
        std_compliance: 已贯标列数据
        weights: 特征权重
        k_neighbors: K近邻数量
        top_n: 返回推荐数量
    
    Returns:
        StandardRecommender 实例
    """
    return StandardRecommender(
        std_compliance=std_compliance,
        weights=weights,
        k_neighbors=k_neighbors,
        top_n=top_n
    )
