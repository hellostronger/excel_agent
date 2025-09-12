"""
Embedding Engine for Excel Intelligence Agent System

基于ST-Raptor的多语言embedding模型实现，用于Excel内容的语义分析和匹配。
"""

import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from .utils import setup_logging
from .constants import DEFAULT_TEMPERATURE

# Setup logging
logger = setup_logging()


def find_topk_indices(lst: List[float], k: int) -> List[int]:
    """获取列表中Top-K最大值的索引"""
    import heapq
    topk_with_indices = heapq.nlargest(k, enumerate(lst), key=lambda x: x[1])
    indices = [index for index, value in topk_with_indices]
    return indices


def calculate_topk_similarity(query_vectors: np.ndarray, target_vectors: np.ndarray, topk: int = 6) -> Tuple[List[List[int]], List[List[float]]]:
    """
    计算两个Embedding向量列表之间的相似度，并返回Top-K最相关的数据。
    
    Args:
        query_vectors: 待匹配的向量列表，形状为 (n, embedding_dim)
        target_vectors: 被匹配的向量列表，形状为 (m, embedding_dim)
        topk: 返回的Top-K最相关结果数量
        
    Returns:
        topk_indices: Top-K最相关的索引列表，形状为 (n, topk)
        topk_scores: Top-K最相关的相似度分数列表，形状为 (n, topk)
    """
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(query_vectors, target_vectors)  # 形状 (n, m)

    # 获取Top-K的索引和分数
    topk_indices = np.argsort(similarity_matrix, axis=1)[:, -topk:][:, ::-1]  # 形状 (n, topk)
    topk_scores = np.take_along_axis(similarity_matrix, topk_indices, axis=1)  # 形状 (n, topk)

    return topk_indices.tolist(), topk_scores.tolist()


class EmbeddingEngine:
    """
    Excel智能分析的Embedding引擎
    
    参考ST-Raptor的EmbeddingModelMultilingualE5实现，适配Excel分析场景。
    """
    
    _instance = None  # 单例模式
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available. Please install: pip install sentence-transformers")
            raise ImportError("sentence-transformers is required for embedding functionality")
        
        self.model_name = model_name
        self.model = None
        self.cache_dir = Path("cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialize_model()
    
    def __new__(cls, *args, **kwargs):
        """单例模式确保只有一个embedding模型实例"""
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _initialize_model(self):
        """初始化embedding模型"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # 回退到更小的模型
            try:
                fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                logger.info(f"Trying fallback model: {fallback_model}")
                self.model = SentenceTransformer(fallback_model)
                self.model_name = fallback_model
                logger.info("Fallback embedding model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise
    
    def encode_entities(self, entity_list: List[str]) -> np.ndarray:
        """
        将文本实体编码为embedding向量
        
        Args:
            entity_list: 文本实体列表
            
        Returns:
            embeddings: embedding向量数组
        """
        # 预处理：将空字符串替换为占位符
        processed_entities = ["#EMPTY#" if str(x).strip() == '' else str(x) for x in entity_list]
        
        try:
            embeddings = self.model.encode(processed_entities)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode entities: {e}")
            # 返回零向量作为fallback
            return np.zeros((len(entity_list), 384))  # 假设384维
    
    def get_embedding_dict(self, entity_list: List[str]) -> Dict[str, List[float]]:
        """
        获取实体的embedding字典
        
        Args:
            entity_list: 实体列表
            
        Returns:
            embedding_dict: {实体: embedding向量} 的字典
        """
        embeddings = self.encode_entities(entity_list)
        embedding_dict = {
            str(entity): embedding.tolist()
            for entity, embedding in zip(entity_list, embeddings)
        }
        return embedding_dict
    
    def save_embedding_cache(self, embedding_dict: Dict[str, List[float]], cache_file: str):
        """保存embedding缓存到文件"""
        try:
            cache_path = self.cache_dir / cache_file
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(embedding_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Embedding cache saved to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def load_embedding_cache(self, cache_file: str) -> Optional[Dict[str, np.ndarray]]:
        """从文件加载embedding缓存"""
        try:
            cache_path = self.cache_dir / cache_file
            if not cache_path.exists():
                return None
                
            with open(cache_path, "r", encoding="utf-8") as f:
                loaded_embedding_dict = json.load(f)
            
            # 将列表转换回NumPy数组
            loaded_embedding_dict = {
                k: np.array(v) for k, v in loaded_embedding_dict.items()
            }
            
            logger.info(f"Embedding cache loaded from {cache_path}")
            return loaded_embedding_dict
        except Exception as e:
            logger.error(f"Failed to load embedding cache: {e}")
            return None
    
    def semantic_similarity(self, query: str, candidates: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        计算查询与候选项之间的语义相似度
        
        Args:
            query: 查询文本
            candidates: 候选文本列表
            top_k: 返回top-k最相似的结果
            
        Returns:
            结果列表：[(文本, 相似度分数), ...]
        """
        if not candidates:
            return []
        
        # 编码查询和候选项
        all_texts = [query] + candidates
        embeddings = self.encode_entities(all_texts)
        
        # 计算相似度
        query_embedding = embeddings[0:1]  # 形状：(1, dim)
        candidate_embeddings = embeddings[1:]  # 形状：(n, dim)
        
        # 余弦相似度
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # 获取top-k结果
        top_k = min(top_k, len(candidates))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (candidates[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def batch_similarity_search(self, queries: List[str], candidate_pool: List[str], top_k: int = 5) -> List[List[Tuple[str, float]]]:
        """
        批量相似度搜索
        
        Args:
            queries: 查询列表
            candidate_pool: 候选池
            top_k: 每个查询返回的top-k结果
            
        Returns:
            每个查询的top-k结果列表
        """
        if not queries or not candidate_pool:
            return [[] for _ in queries]
        
        # 编码所有文本
        query_embeddings = self.encode_entities(queries)
        candidate_embeddings = self.encode_entities(candidate_pool)
        
        # 批量计算相似度
        topk_indices, topk_scores = calculate_topk_similarity(
            query_embeddings, candidate_embeddings, top_k
        )
        
        # 构建结果
        results = []
        for i, (indices, scores) in enumerate(zip(topk_indices, topk_scores)):
            query_results = [
                (candidate_pool[idx], score)
                for idx, score in zip(indices, scores)
            ]
            results.append(query_results)
        
        return results
    
    def find_similar_columns(self, target_column: str, column_names: List[str], threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        寻找与目标列相似的列名
        
        Args:
            target_column: 目标列名
            column_names: 所有列名列表
            threshold: 相似度阈值
            
        Returns:
            相似列名及其相似度分数
        """
        results = self.semantic_similarity(target_column, column_names, top_k=len(column_names))
        # 过滤低于阈值的结果
        filtered_results = [
            (col_name, score) for col_name, score in results
            if score >= threshold and col_name != target_column
        ]
        return filtered_results
    
    def categorize_column_content(self, column_values: List[str], categories: List[str]) -> Dict[str, List[str]]:
        """
        基于语义相似度将列内容分类
        
        Args:
            column_values: 列值列表
            categories: 分类标签列表
            
        Returns:
            分类结果：{分类: [值列表]}
        """
        if not column_values or not categories:
            return {}
        
        # 为每个值找到最相似的分类
        categorized = {cat: [] for cat in categories}
        
        for value in column_values:
            if not value or str(value).strip() == '':
                continue
                
            similar_categories = self.semantic_similarity(str(value), categories, top_k=1)
            if similar_categories:
                best_category = similar_categories[0][0]
                categorized[best_category].append(value)
        
        # 移除空分类
        return {k: v for k, v in categorized.items() if v}
    
    def generate_cache_key(self, content: str) -> str:
        """为内容生成缓存键"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()


# 全局实例
embedding_engine = None


def get_embedding_engine() -> EmbeddingEngine:
    """获取embedding引擎的全局实例"""
    global embedding_engine
    if embedding_engine is None:
        embedding_engine = EmbeddingEngine()
    return embedding_engine


# Excel专用的embedding分析函数

async def analyze_column_semantics(column_values: List[str], column_name: str = "") -> Dict[str, Any]:
    """
    分析列的语义特征
    
    Args:
        column_values: 列值列表
        column_name: 列名
        
    Returns:
        语义分析结果
    """
    engine = get_embedding_engine()
    
    # 过滤空值
    valid_values = [str(v) for v in column_values if v is not None and str(v).strip() != '']
    
    if not valid_values:
        return {
            "semantic_categories": [],
            "content_diversity": 0.0,
            "semantic_coherence": 0.0,
            "key_concepts": []
        }
    
    try:
        # 编码所有值
        embeddings = engine.encode_entities(valid_values)
        
        # 计算语义多样性（平均成对距离）
        if len(embeddings) > 1:
            pairwise_similarity = cosine_similarity(embeddings)
            # 排除对角线元素
            mask = np.ones(pairwise_similarity.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_similarity = pairwise_similarity[mask].mean()
            content_diversity = 1.0 - avg_similarity
        else:
            content_diversity = 0.0
        
        # 语义一致性（与列名的相关性）
        semantic_coherence = 0.5  # 默认值
        if column_name.strip():
            name_similarities = engine.semantic_similarity(column_name, valid_values, top_k=len(valid_values))
            if name_similarities:
                semantic_coherence = np.mean([score for _, score in name_similarities])
        
        # 提取关键概念（最具代表性的值）
        if len(valid_values) > 5:
            # 使用聚类方法或简单选取最常见的值
            key_concepts = valid_values[:5]  # 简化实现
        else:
            key_concepts = valid_values
        
        return {
            "semantic_categories": [],  # 可以进一步实现自动分类
            "content_diversity": float(content_diversity),
            "semantic_coherence": float(semantic_coherence),
            "key_concepts": key_concepts,
            "total_values": len(valid_values),
            "embedding_dimension": embeddings.shape[1] if embeddings.size > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Semantic analysis failed: {e}")
        return {
            "semantic_categories": [],
            "content_diversity": 0.0,
            "semantic_coherence": 0.0,
            "key_concepts": valid_values[:5],
            "error": str(e)
        }


async def find_related_columns(target_columns: List[str], all_columns: List[str]) -> Dict[str, List[Tuple[str, float]]]:
    """
    寻找相关的列
    
    Args:
        target_columns: 目标列列表
        all_columns: 所有列列表
        
    Returns:
        每个目标列的相关列及相似度
    """
    engine = get_embedding_engine()
    
    related_columns = {}
    for target_col in target_columns:
        similar_cols = engine.find_similar_columns(target_col, all_columns, threshold=0.3)
        related_columns[target_col] = similar_cols
    
    return related_columns


async def semantic_data_quality_check(column_values: List[str], expected_category: str = "") -> Dict[str, Any]:
    """
    基于语义的数据质量检查
    
    Args:
        column_values: 列值
        expected_category: 期望的数据类别
        
    Returns:
        数据质量检查结果
    """
    engine = get_embedding_engine()
    
    valid_values = [str(v) for v in column_values if v is not None and str(v).strip() != '']
    
    if not valid_values:
        return {
            "consistency_score": 0.0,
            "outliers": [],
            "quality_issues": ["Empty column"],
            "recommendations": ["Column contains no valid data"]
        }
    
    try:
        quality_issues = []
        recommendations = []
        outliers = []
        
        # 编码值
        embeddings = engine.encode_entities(valid_values)
        
        if len(embeddings) > 1:
            # 计算每个值与其他值的平均相似度
            pairwise_similarity = cosine_similarity(embeddings)
            np.fill_diagonal(pairwise_similarity, 0)  # 排除自身
            
            avg_similarities = pairwise_similarity.mean(axis=1)
            
            # 识别异常值（相似度低于阈值）
            outlier_threshold = np.percentile(avg_similarities, 25)  # 底部25%为异常值
            outlier_indices = np.where(avg_similarities < outlier_threshold)[0]
            
            outliers = [valid_values[i] for i in outlier_indices if avg_similarities[i] < 0.3]
            
            # 一致性分数
            consistency_score = float(avg_similarities.mean())
        else:
            consistency_score = 1.0
        
        # 与期望类别的匹配度
        if expected_category.strip():
            category_similarities = engine.semantic_similarity(expected_category, valid_values)
            low_match_count = sum(1 for _, score in category_similarities if score < 0.4)
            
            if low_match_count > len(valid_values) * 0.3:
                quality_issues.append(f"Many values don't match expected category '{expected_category}'")
                recommendations.append("Review data entry standards or column definition")
        
        # 生成建议
        if consistency_score < 0.5:
            quality_issues.append("Low semantic consistency among values")
            recommendations.append("Consider data standardization or column splitting")
        
        if outliers:
            quality_issues.append(f"Found {len(outliers)} potential outlier values")
            recommendations.append("Review outlier values for data entry errors")
        
        return {
            "consistency_score": consistency_score,
            "outliers": outliers,
            "quality_issues": quality_issues,
            "recommendations": recommendations,
            "total_analyzed": len(valid_values)
        }
        
    except Exception as e:
        logger.error(f"Semantic quality check failed: {e}")
        return {
            "consistency_score": 0.0,
            "outliers": [],
            "quality_issues": [f"Analysis failed: {str(e)}"],
            "recommendations": ["Manual review recommended"]
        }