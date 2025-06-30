"""
ChromaDB 向量数据库示例代码
演示如何使用ChromaDB进行文档存储、向量化和相似度搜索
"""

import chromadb
from chromadb.utils import embedding_functions

def create_chroma_client():
    """
    创建ChromaDB客户端
    
    Returns:
        chromadb.Client: ChromaDB客户端实例
    """
    return chromadb.Client()


def create_collection_with_config():
    """
    创建带有自定义配置的集合
    
    Returns:
        chromadb.Collection: 配置好的集合实例
    """
    client = create_chroma_client()
    
    # 默认情况下，Chroma 使用 DefaultEmbeddingFunction，它是基于 Sentence Transformers 的 MiniLM-L6-v2 模型
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    # 使用 OpenAI 的嵌入模型，默认使用 text-embedding-ada-002 模型
    # openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    #     api_key="YOUR_API_KEY",
    #     model_name="text-embedding-3-small"
    # )
    
    try:
        # 尝试获取已存在的集合
        collection = client.get_collection(name="my_collection")
        print("使用已存在的集合")
    except:
        # 如果集合不存在，创建新集合
        print("创建新的集合")
        collection = client.create_collection(
            name="my_collection",
            configuration={
                # HNSW 索引算法，基于图的近似最近邻搜索算法（Approximate Nearest Neighbor，ANN）
                "hnsw": {
                    "space": "cosine",  # 指定余弦相似度计算
                    "ef_search": 100,   # 搜索时的候选数量
                    "ef_construction": 100,  # 构建索引时的候选数量
                    "max_neighbors": 16,     # 最大邻居数
                    "num_threads": 4         # 线程数
                },
                # 指定向量模型
                "embedding_function": default_ef
            }
        )
    
    return collection


def add_documents_to_collection(collection):
    """
    向集合中添加文档
    
    Args:
        collection: ChromaDB集合实例
    """
    # 方式1：自动生成向量（使用集合指定的嵌入模型）
    documents = [
        "RAG是一种检索增强生成技术，结合了检索和生成两种能力",
        "向量数据库存储文档的嵌入表示，支持高效的相似度搜索",
        "在机器学习领域，智能体（Agent）通常指能够感知环境、做出决策并采取行动以实现特定目标的实体",
        "深度学习是机器学习的一个分支，使用多层神经网络进行特征学习",
        "自然语言处理（NLP）是人工智能的重要分支，专注于计算机理解和生成人类语言"
    ]
    
    metadatas = [
        {"source": "RAG", "category": "技术"},
        {"source": "向量数据库", "category": "技术"},
        {"source": "Agent", "category": "AI"},
        {"source": "深度学习", "category": "AI"},
        {"source": "NLP", "category": "AI"}
    ]
    
    ids = ["id1", "id2", "id3", "id4", "id5"]
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"成功添加 {len(documents)} 个文档到集合中")
    
    # 方式2：手动传入预计算向量（示例代码）
    # collection.add(
    #     embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # 实际的向量值
    #     documents=["文本1", "文本2"],
    #     ids=["id6", "id7"]
    # )


def query_documents(collection):
    """
    查询文档示例
    
    Args:
        collection: ChromaDB集合实例
    """
    print("\n=== 文档查询示例 ===")
    
    # 查询1：基于文本的相似度搜索
    print("1. 基于文本的相似度搜索:")
    results = collection.query(
        query_texts=["什么是RAG技术？"],
        n_results=3,
        # where={"source": "RAG"},  # 按元数据过滤
        # where_document={"$contains": "检索增强生成"}  # 按文档内容过滤
    )
    
    print("查询结果:")
    for i, (doc, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
        print(f"  结果{i+1}: 距离={distance:.4f}, 来源={metadata['source']}")
        print(f"      文档: {doc}")
        print()
    
    # 查询2：基于AI相关内容的搜索
    print("2. 搜索AI相关内容:")
    results = collection.query(
        query_texts=["人工智能和机器学习"],
        n_results=2,
        where={"category": "AI"}  # 按元数据过滤
    )
    
    print("AI相关文档:")
    for i, (doc, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
        print(f"  结果{i+1}: 距离={distance:.4f}, 来源={metadata['source']}")
        print(f"      文档: {doc}")
        print()
    
    # 查询3：基于文档内容包含特定关键词的搜索
    print("3. 搜索包含'技术'关键词的文档:")
    results = collection.query(
        query_texts=["技术"],
        n_results=3,
        where_document={"$contains": "技术"}  # 按文档内容过滤
    )
    
    print("包含'技术'关键词的文档:")
    for i, (doc, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
        print(f"  结果{i+1}: 距离={distance:.4f}, 来源={metadata['source']}")
        print(f"      文档: {doc}")
        print()


def demonstrate_collection_operations(collection):
    """
    演示集合的基本操作
    
    Args:
        collection: ChromaDB集合实例
    """
    print("\n=== 集合操作演示 ===")
    
    # 查看集合基本信息
    print(f"集合名称: {collection.name}")
    print(f"文档数量: {collection.count()}")
    
    # 查看集合中的部分数据
    print("\n集合中的数据预览:")
    peek_data = collection.peek()
    if peek_data['documents']:
        for i, doc in enumerate(peek_data['documents']):
            print(f"  文档{i+1}: {doc}")
    
    # 获取集合的嵌入函数信息
    print(f"\n使用的嵌入函数: {collection._embedding_function.__class__.__name__}")


def main():
    """主函数：演示ChromaDB的完整使用流程"""
    
    print("=== ChromaDB 向量数据库演示 ===\n")
    
    # 1. 创建集合
    print("1. 创建/获取集合...")
    collection = create_collection_with_config()
    
    # 2. 添加文档
    print("\n2. 添加文档到集合...")
    add_documents_to_collection(collection)
    
    # 3. 演示集合操作
    demonstrate_collection_operations(collection)
    
    # 4. 执行查询
    query_documents(collection)
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()


