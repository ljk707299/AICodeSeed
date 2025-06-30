"""
Pinecone 向量数据库示例代码
演示如何使用Pinecone进行云端向量存储和相似度搜索
注意：此演示使用模拟向量，无需外部API调用
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
import time

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    print("请安装依赖: pip install pinecone-client")
    exit(1)


def setup_pinecone(api_key: str) -> Pinecone:
    """初始化Pinecone客户端"""
    pc = Pinecone(api_key=api_key)
    print("Pinecone客户端初始化完成")
    return pc


def list_existing_indexes(pc: Pinecone):
    """列出现有的索引"""
    indexes = pc.list_indexes()
    print(f"现有索引: {indexes.names()}")
    return indexes.names()


def generate_mock_embeddings(texts: List[str], dimension: int = 1536) -> List[List[float]]:
    """生成模拟的嵌入向量（用于演示）"""
    np.random.seed(42)  # 确保结果可重现
    
    vectors = []
    for i, text in enumerate(texts):
        # 基于文本长度和内容生成一些变化的向量
        base_vector = np.random.normal(0, 1, dimension)
        
        # 为相似主题的文本生成相似的向量
        if "AI" in text or "人工智能" in text:
            base_vector += np.random.normal(0.1, 0.1, dimension)
        elif "机器学习" in text or "ML" in text:
            base_vector += np.random.normal(0.2, 0.1, dimension)
        elif "深度学习" in text or "DL" in text:
            base_vector += np.random.normal(0.15, 0.1, dimension)
        
        # 归一化向量
        base_vector = base_vector / np.linalg.norm(base_vector)
        vectors.append(base_vector.tolist())
    
    return vectors


def demonstrate_embedding_generation():
    """演示嵌入向量生成"""
    print("\n=== 嵌入向量生成演示 ===")
    
    # 准备示例数据
    documents = [
        "人工智能是计算机科学的一个分支，致力于创建智能机器",
        "机器学习是AI的核心技术，通过数据训练模型",
        "深度学习使用神经网络模拟人脑的学习过程",
        "自然语言处理让计算机理解和生成人类语言",
        "计算机视觉使机器能够理解和分析图像"
    ]
    
    print("正在生成模拟嵌入向量...")
    vectors = generate_mock_embeddings(documents)
    
    print(f"成功生成 {len(vectors)} 个嵌入向量")
    print(f"每个向量维度: {len(vectors[0])}")
    print(f"第一个向量的前5个元素: {vectors[0][:5]}")
    
    return vectors, documents


def demonstrate_similarity_calculation(vectors: List[List[float]], documents: List[str]):
    """演示向量相似度计算"""
    print("\n=== 向量相似度计算演示 ===")
    
    # 计算余弦相似度
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    
    # 使用第一个向量作为查询向量
    query_vector = vectors[0]
    query_text = documents[0]
    
    print(f"查询文本: {query_text}")
    print("\n相似度计算结果:")
    
    similarities = []
    for i, (vector, doc) in enumerate(zip(vectors, documents)):
        similarity = cosine_similarity(query_vector, vector)
        similarities.append((i, similarity, doc))
        print(f"  文档{i+1}: {similarity:.4f} - {doc}")
    
    # 找出最相似的文档
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"\n最相似的文档: {similarities[1][2]} (相似度: {similarities[1][1]:.4f})")


def demonstrate_vector_operations():
    """演示向量操作概念"""
    print("\n=== 向量操作演示 ===")
    
    # 模拟向量数据
    vectors = generate_mock_embeddings([
        "文档1: 关于人工智能的介绍",
        "文档2: 机器学习基础知识",
        "文档3: 深度学习技术"
    ])
    
    ids = ["doc_1", "doc_2", "doc_3"]
    metadatas = [
        {"category": "AI", "source": "intro"},
        {"category": "ML", "source": "intro"},
        {"category": "DL", "source": "intro"}
    ]
    
    print("1. 向量插入操作:")
    print("   - 向量ID: doc_1, doc_2, doc_3")
    print("   - 向量维度: 1536")
    print("   - 元数据: 包含分类和来源信息")
    
    print("\n2. 向量搜索操作:")
    query_vector = generate_mock_embeddings(["查询: 什么是机器学习？"])[0]
    print("   - 查询向量维度: 1536")
    print("   - 搜索类型: 余弦相似度")
    print("   - 返回结果: top-k 最相似向量")
    
    print("\n3. 元数据过滤:")
    print("   - 按分类过滤: category = 'AI'")
    print("   - 按来源过滤: source = 'intro'")
    print("   - 组合过滤: 相似度 + 元数据")


def demonstrate_pinecone_concepts():
    """演示Pinecone的核心概念"""
    print("\n=== Pinecone 核心概念演示 ===")
    
    print("1. 索引创建:")
    print("   - 需要指定向量维度")
    print("   - 选择相似度度量方法（cosine, euclidean, dotproduct）")
    print("   - 配置云服务商和区域")
    
    print("\n2. 向量操作:")
    print("   - upsert: 插入或更新向量")
    print("   - query: 相似度搜索")
    print("   - delete: 删除向量")
    
    print("\n3. 元数据过滤:")
    print("   - 支持基于元数据的过滤")
    print("   - 可以结合向量相似度和元数据过滤")
    
    print("\n4. 批量操作:")
    print("   - 支持批量插入向量")
    print("   - 支持批量查询")


def demonstrate_performance_comparison():
    """演示不同向量数据库的性能对比"""
    print("\n=== 向量数据库性能对比 ===")
    
    print("1. Pinecone (托管服务):")
    print("   - 优点: 无需管理基础设施，高可用性")
    print("   - 缺点: 需要付费，依赖网络")
    print("   - 适用场景: 生产环境，大规模应用")
    
    print("\n2. FAISS (本地库):")
    print("   - 优点: 高性能，本地部署，免费")
    print("   - 缺点: 需要自己管理基础设施")
    print("   - 适用场景: 研究，中小规模应用")
    
    print("\n3. ChromaDB (开源):")
    print("   - 优点: 易于使用，支持元数据")
    print("   - 缺点: 性能相对较低")
    print("   - 适用场景: 原型开发，小规模应用")


def main():
    """主函数：演示Pinecone的使用"""
    
    print("=== Pinecone 向量数据库演示 ===\n")
    
    # 检查API密钥
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("未设置 PINECONE_API_KEY 环境变量")
        print("继续演示概念和模拟操作...")
    else:
        # 初始化Pinecone
        pc = setup_pinecone(api_key)
        
        # 列出现有索引
        existing_indexes = list_existing_indexes(pc)
        
        if existing_indexes:
            print(f"\n发现现有索引: {existing_indexes}")
            print("您可以使用现有索引进行向量操作")
        else:
            print("\n没有发现现有索引")
            print("注意：创建新索引需要付费计划")
    
    # 演示嵌入向量生成
    vectors, documents = demonstrate_embedding_generation()
    
    # 演示相似度计算
    demonstrate_similarity_calculation(vectors, documents)
    
    # 演示向量操作
    demonstrate_vector_operations()
    
    # 演示Pinecone概念
    demonstrate_pinecone_concepts()
    
    # 演示性能对比
    demonstrate_performance_comparison()
    
    print("\n=== 演示完成 ===")
    print("\n要完整使用Pinecone，您需要:")
    print("1. 有效的Pinecone API密钥")
    print("2. 付费计划以创建索引")
    print("3. 选择合适的云服务商和区域")
    print("\n或者考虑使用本地解决方案如FAISS或ChromaDB")


if __name__ == "__main__":
    main() 