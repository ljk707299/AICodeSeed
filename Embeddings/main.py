"""
嵌入向量（Embeddings）示例代码
用于演示如何使用阿里百炼API进行文本向量化，并计算文本相似度
"""

import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
from openai import OpenAI

# 需要在系统环境变量中配置好相应的key
# OpenAI key（1 代理方式 2 官网注册购买）

# 阿里百炼配置
# DASHSCOPE_API_KEY sk-xxx
# DASHSCOPE_BASE_URL https://dashscope.aliyuncs.com/compatible-mode/v1
client = OpenAI(
    api_key=os.getenv("ALI_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url=os.getenv("ALI_BASE_URL")  # 百炼服务的base_url
)

def cos_sim(a, b):
    """
    计算余弦相似度
    返回值范围：[-1, 1]，值越大表示越相似
    
    Args:
        a: 向量a
        b: 向量b
    
    Returns:
        float: 余弦相似度值
    """
    return dot(a, b) / (norm(a) * norm(b))

def l2_distance(a, b):
    """
    计算欧氏距离
    返回值范围：[0, +∞)，值越小表示越相似
    
    Args:
        a: 向量a
        b: 向量b
    
    Returns:
        float: 欧氏距离值
    """
    x = np.asarray(a) - np.asarray(b)
    return norm(x)

def get_embeddings(texts, model="text-embedding-v1", dimensions=None):
    """
    获取文本的嵌入向量
    
    Args:
        texts: 文本列表或单个文本
        model: 使用的嵌入模型名称
        dimensions: 向量维度（可选）
    
    Returns:
        list: 嵌入向量列表
    """
    # 确保输入是列表格式
    if isinstance(texts, str):
        texts = [texts]
    
    # 根据模型类型设置参数
    if model == "text-embedding-v1":
        dimensions = None
    
    # 调用API获取嵌入向量
    if dimensions:
        response = client.embeddings.create(
            input=texts, 
            model=model, 
            dimensions=dimensions
        )
    else:
        response = client.embeddings.create(
            input=texts, 
            model=model
        )
    
    return [x.embedding for x in response.data]

def calculate_similarities(query, documents, similarity_type="cosine"):
    """
    计算查询文本与文档集合的相似度
    
    Args:
        query: 查询文本
        documents: 文档列表
        similarity_type: 相似度类型 ("cosine" 或 "euclidean")
    
    Returns:
        list: 相似度分数列表
    """
    # 获取向量表示
    query_vec = get_embeddings([query])[0]
    doc_vecs = get_embeddings(documents)
    
    similarities = []
    for i, doc_vec in enumerate(doc_vecs):
        if similarity_type == "cosine":
            score = cos_sim(query_vec, doc_vec)
        elif similarity_type == "euclidean":
            score = l2_distance(query_vec, doc_vec)
        else:
            raise ValueError("相似度类型必须是 'cosine' 或 'euclidean'")
        
        similarities.append((i, score, documents[i]))
    
    return similarities

def main():
    """主函数：演示嵌入向量的使用"""
    
    # 测试基本功能
    print("=== 基本功能测试 ===")
    test_query = ["用科技力量，构建智能未来！"]
    vec = get_embeddings(test_query)[0]
    print(f"向量维度: {len(vec)}")
    print(f"前10个元素: {vec[:10]}")
    print()
    
    # 相似度计算示例
    print("=== 相似度计算示例 ===")
    query = "国际争端"
    
    # 且能支持跨语言
    # query = "global conflicts"

    # 支持跨语言的文档集合
    documents = [
        "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
        "土耳其、芬兰、瑞典与北约代表将继续就瑞典'入约'问题进行谈判",
        "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
        "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
        "我国首次在空间站开展舱外辐射生物学暴露实验",
    ]
    
    # 计算余弦相似度
    print("余弦相似度结果:")
    cosine_results = calculate_similarities(query, documents, "cosine")
    for idx, score, doc in cosine_results:
        print(f"文档{idx+1}: {score:.4f} - {doc}")
    
    print()
    
    # 计算欧氏距离
    print("欧氏距离结果:")
    euclidean_results = calculate_similarities(query, documents, "euclidean")
    for idx, score, doc in euclidean_results:
        print(f"文档{idx+1}: {score:.4f} - {doc}")
    
    print()
    
    # 找出最相似的文档
    print("=== 最相似文档分析 ===")
    best_cosine = max(cosine_results, key=lambda x: x[1])
    best_euclidean = min(euclidean_results, key=lambda x: x[1])
    
    print(f"余弦相似度最高的文档: {best_cosine[2]}")
    print(f"相似度分数: {best_cosine[1]:.4f}")
    print()
    print(f"欧氏距离最小的文档: {best_euclidean[2]}")
    print(f"距离分数: {best_euclidean[1]:.4f}")

if __name__ == "__main__":
    main()
