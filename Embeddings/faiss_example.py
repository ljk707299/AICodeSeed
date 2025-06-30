"""
FAISS 向量数据库示例代码
演示如何使用Facebook AI Similarity Search (FAISS)进行高效的向量存储和相似度搜索
FAISS是一个用于高效相似性搜索和密集向量聚类的库
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional
import time


def create_sample_vectors(dimension: int = 128, num_vectors: int = 1000) -> np.ndarray:
    """
    创建示例向量数据用于演示
    
    Args:
        dimension: 向量维度
        num_vectors: 向量数量
    
    Returns:
        np.ndarray: 随机生成的向量数组，形状为 (num_vectors, dimension)
    """
    # 生成随机向量，模拟真实的嵌入向量
    np.random.seed(42)  # 设置随机种子以确保结果可重现
    vectors = np.random.random((num_vectors, dimension)).astype('float32')
    
    # 对向量进行归一化，使其适合余弦相似度计算
    faiss.normalize_L2(vectors)
    
    return vectors


def create_flat_index(dimension: int) -> faiss.Index:
    """
    创建精确搜索的Flat索引
    适用于小规模数据集，搜索精度最高但速度较慢
    
    Args:
        dimension: 向量维度
    
    Returns:
        faiss.Index: FAISS索引对象
    """
    # Flat索引进行精确的暴力搜索
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product，适用于归一化向量的余弦相似度
    return index


def create_ivf_index(dimension: int, num_clusters: int = 100) -> faiss.Index:
    """
    创建倒排文件索引（Inverted File Index）
    适用于中等规模数据集，在精度和速度之间取得平衡
    
    Args:
        dimension: 向量维度
        num_clusters: 聚类中心数量
    
    Returns:
        faiss.Index: FAISS索引对象
    """
    # 使用倒排文件索引，先聚类再搜索
    quantizer = faiss.IndexFlatIP(dimension)  # 量化器
    index = faiss.IndexIVFFlat(quantizer, dimension, num_clusters, faiss.METRIC_INNER_PRODUCT)
    return index


def create_hnsw_index(dimension: int, m: int = 32) -> faiss.Index:
    """
    创建层次可导航小世界图索引（Hierarchical Navigable Small World）
    适用于大规模数据集，搜索速度快，精度较高
    
    Args:
        dimension: 向量维度
        m: 每个节点的最大连接数
    
    Returns:
        faiss.Index: FAISS索引对象
    """
    # HNSW索引，基于图的近似最近邻搜索
    index = faiss.IndexHNSWFlat(dimension, m)
    index.hnsw.efConstruction = 200  # 构建时的搜索深度
    index.hnsw.efSearch = 100        # 查询时的搜索深度
    return index


def train_index(index: faiss.Index, vectors: np.ndarray) -> None:
    """
    训练索引（仅对需要训练的索引类型有效）
    
    Args:
        index: FAISS索引对象
        vectors: 训练向量
    """
    if hasattr(index, 'is_trained') and not index.is_trained:
        print(f"正在训练索引，使用 {len(vectors)} 个向量...")
        index.train(vectors)
        print("索引训练完成")


def add_vectors_to_index(index: faiss.Index, vectors: np.ndarray, ids: Optional[List[int]] = None) -> None:
    """
    将向量添加到索引中
    
    Args:
        index: FAISS索引对象
        vectors: 要添加的向量
        ids: 向量ID列表（可选）
    """
    if ids is not None:
        # 如果提供了ID，使用IDMap包装索引
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
        index.add_with_ids(vectors, np.array(ids))
    else:
        index.add(vectors)
    
    print(f"成功添加 {len(vectors)} 个向量到索引中")


def search_similar_vectors(index: faiss.Index, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    在索引中搜索最相似的向量
    
    Args:
        index: FAISS索引对象
        query_vector: 查询向量
        k: 返回的最相似向量数量
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (距离数组, 索引数组)
    """
    # 确保查询向量是2D数组
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    
    # 执行搜索
    distances, indices = index.search(query_vector, k)
    return distances[0], indices[0]


def benchmark_index_performance(index: faiss.Index, test_vectors: np.ndarray, 
                               num_queries: int = 100) -> dict:
    """
    基准测试索引性能
    
    Args:
        index: FAISS索引对象
        test_vectors: 测试向量
        num_queries: 查询次数
    
    Returns:
        dict: 性能指标字典
    """
    print(f"开始性能测试，执行 {num_queries} 次查询...")
    
    # 随机选择查询向量
    query_indices = np.random.choice(len(test_vectors), num_queries, replace=False)
    query_vectors = test_vectors[query_indices]
    
    # 记录搜索时间
    start_time = time.time()
    for query_vector in query_vectors:
        search_similar_vectors(index, query_vector, k=10)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_queries
    
    return {
        'total_time': total_time,
        'avg_time_per_query': avg_time,
        'queries_per_second': num_queries / total_time
    }


def demonstrate_different_index_types():
    """演示不同类型的FAISS索引"""
    
    print("=== FAISS 向量数据库演示 ===\n")
    
    # 创建示例数据
    dimension = 128
    num_vectors = 10000
    print(f"创建 {num_vectors} 个 {dimension} 维向量...")
    vectors = create_sample_vectors(dimension, num_vectors)
    
    # 1. Flat索引演示（精确搜索）
    print("\n1. Flat索引演示（精确搜索）:")
    flat_index = create_flat_index(dimension)
    add_vectors_to_index(flat_index, vectors)
    
    # 执行搜索
    query_vector = vectors[0]  # 使用第一个向量作为查询
    distances, indices = search_similar_vectors(flat_index, query_vector, k=5)
    
    print("搜索结果:")
    for i, (distance, idx) in enumerate(zip(distances, indices)):
        print(f"  结果{i+1}: 索引={idx}, 相似度={distance:.4f}")
    
    # 性能测试
    perf = benchmark_index_performance(flat_index, vectors[:100])
    print(f"性能: {perf['queries_per_second']:.2f} 查询/秒")
    
    # 2. IVF索引演示（聚类搜索）
    print("\n2. IVF索引演示（聚类搜索）:")
    ivf_index = create_ivf_index(dimension, num_clusters=100)
    train_index(ivf_index, vectors)
    add_vectors_to_index(ivf_index, vectors)
    
    # 执行搜索
    distances, indices = search_similar_vectors(ivf_index, query_vector, k=5)
    
    print("搜索结果:")
    for i, (distance, idx) in enumerate(zip(distances, indices)):
        print(f"  结果{i+1}: 索引={idx}, 相似度={distance:.4f}")
    
    # 性能测试
    perf = benchmark_index_performance(ivf_index, vectors[:100])
    print(f"性能: {perf['queries_per_second']:.2f} 查询/秒")
    
    # 3. HNSW索引演示（图搜索）
    print("\n3. HNSW索引演示（图搜索）:")
    hnsw_index = create_hnsw_index(dimension, m=32)
    add_vectors_to_index(hnsw_index, vectors)
    
    # 执行搜索
    distances, indices = search_similar_vectors(hnsw_index, query_vector, k=5)
    
    print("搜索结果:")
    for i, (distance, idx) in enumerate(zip(distances, indices)):
        print(f"  结果{i+1}: 索引={idx}, 相似度={distance:.4f}")
    
    # 性能测试
    perf = benchmark_index_performance(hnsw_index, vectors[:100])
    print(f"性能: {perf['queries_per_second']:.2f} 查询/秒")


def demonstrate_practical_use_case():
    """演示实际使用场景：文档相似度搜索"""
    
    print("\n=== 实际应用场景：文档相似度搜索 ===\n")
    
    # 模拟文档嵌入向量
    documents = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的重要技术",
        "深度学习是机器学习的一个子领域",
        "自然语言处理是AI的重要应用",
        "计算机视觉处理图像和视频数据",
        "强化学习通过试错来学习策略",
        "神经网络模拟人脑的工作方式",
        "卷积神经网络特别适合图像处理",
        "循环神经网络适合序列数据处理",
        "Transformer模型在NLP领域表现优异"
    ]
    
    # 创建文档的嵌入向量（这里用随机向量模拟）
    doc_vectors = create_sample_vectors(dimension=128, num_vectors=len(documents))
    
    # 创建索引
    index = create_hnsw_index(128)
    add_vectors_to_index(index, doc_vectors)
    
    # 搜索相似文档
    query = "什么是机器学习？"
    print(f"查询: {query}")
    
    # 模拟查询向量（这里用第一个文档的向量）
    query_vector = doc_vectors[0]
    
    distances, indices = search_similar_vectors(index, query_vector, k=3)
    
    print("\n最相似的文档:")
    for i, (distance, idx) in enumerate(zip(distances, indices)):
        print(f"  {i+1}. {documents[idx]} (相似度: {distance:.4f})")


def main():
    """主函数：运行完整的FAISS演示"""
    
    # 演示不同类型的索引
    demonstrate_different_index_types()
    
    # 演示实际应用场景
    demonstrate_practical_use_case()
    
    print("\n=== 演示完成 ===")
    print("\n索引类型总结:")
    print("- Flat索引: 精确搜索，速度慢，适合小数据集")
    print("- IVF索引: 聚类搜索，平衡精度和速度，适合中等数据集")
    print("- HNSW索引: 图搜索，速度快，适合大数据集")


if __name__ == "__main__":
    main() 